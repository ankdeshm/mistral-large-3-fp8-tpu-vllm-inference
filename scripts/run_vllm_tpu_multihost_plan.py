#!/usr/bin/env python3
"""Multi-host vLLM-TorchTPU serving benchmark suite (Ray executor) driven by a JSON plan.

One Indexed Job pod per TPU host. Pod 0 runs a small HTTP coordinator; every experiment is a list of candidate
server configurations tried in order (the first one that boots and completes the sweep wins). A derived experiment
(`derive_from`) starts from an earlier experiment's winning candidate and applies one of its `variants` (tried in
order) - each variant may `override` candidate keys and `extra_args_add` / `extra_args_remove` CLI args.

Per candidate, all hosts: clean leftovers -> barrier -> join a fresh Ray cluster. Host 0 then runs `vllm serve`
(log in /dev/shm, mirrored to the results bucket every 60 s) and, once /v1/models answers, the client sweep against
localhost. Boot/KV-cache facts are written to boot_<exp>_<tag>.json; suite progress to suite_status.json.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PLAN_PATH = Path(os.environ.get("SUITE_PLAN", "/plan/plan.json"))
COORD_PORT = int(os.environ.get("COORD_PORT", "19999"))

_KV: dict[str, str] = {}
_LOCK = threading.Lock()
_STATUS: list[dict] = []


def log(idx: int, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}][vllm-suite][host-{idx}] {msg}", flush=True)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        return

    def _reply(self, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        with _LOCK:
            self._reply({"val": _KV.get(q.get("key", [""])[0], "")})

    def do_POST(self) -> None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query, keep_blank_values=True)
        with _LOCK:
            _KV[q["key"][0]] = q["val"][0]
        self._reply({"ok": True})


def kv_get(url: str, key: str) -> str:
    try:
        with urllib.request.urlopen(f"{url}/?{urllib.parse.urlencode({'key': key})}", timeout=10) as r:
            return json.loads(r.read()).get("val", "")
    except Exception:
        return ""


def kv_set(url: str, key: str, val: str) -> None:
    for _ in range(300):
        try:
            req = urllib.request.Request(f"{url}/?{urllib.parse.urlencode({'key': key, 'val': val})}", data=b"",
                                         method="POST")
            with urllib.request.urlopen(req, timeout=10):
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"coordinator unreachable setting {key}")


def barrier(url: str, name: str, idx: int, hosts: int, timeout_s: float = 3600) -> None:
    kv_set(url, f"{name}/r{idx}", "1")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if all(kv_get(url, f"{name}/r{r}") for r in range(hosts)):
            return
        time.sleep(3)
    raise TimeoutError(f"barrier {name} timed out")


def wait_key(url: str, key: str, timeout_s: float = 8 * 3600) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        v = kv_get(url, key)
        if v:
            return v
        time.sleep(5)
    raise TimeoutError(f"key {key} timed out")


def resolve(fqdn: str, timeout_s: float = 600) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            return socket.gethostbyname(fqdn)
        except socket.gaierror:
            time.sleep(3)
    raise TimeoutError(f"cannot resolve {fqdn}")


def clean_host() -> None:
    subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["bash", "-lc",
         "for p in $(pgrep -f 'vllm serve|VLLM::|EngineCore|ray::|gcs_server|raylet'); do "
         "[ \"$p\" != \"$$\" ] && kill -9 \"$p\" 2>/dev/null || true; done; "
         "rm -rf /dev/shm/vllm_tmp/* /dev/shm/vllm_cache/* /dev/shm/torchinductor/* /tmp/ray/* 2>/dev/null || true; "
         "mkdir -p /dev/shm/hf_cache /dev/shm/vllm_tmp /dev/shm/vllm_cache /dev/shm/torchinductor /dev/shm/logs"],
        check=False,
    )
    time.sleep(3)


def build_env(plan: dict, cand: dict) -> dict[str, str]:
    env = os.environ.copy()
    if plan.get("unset_torch_tpu_topology", True):
        env.pop("TORCH_TPU_TOPOLOGY", None)
    env.update({
        "TPU_MULTIHOST_BACKEND": "ray",
        "TPU_PROCESS_ADDRESSES": "",
        "TPU_SKIP_MDS_QUERY": "1",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
        "VLLM_PLUGINS": "torchtpu",
        "VLLM_TARGET_DEVICE": "tpu",
        "PJRT_DEVICE": "TPU",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_NO_USAGE_STATS": "1",
        "RUNAI_STREAMER_PROCESS_GROUP_SIZE": "8",
        "RUNAI_STREAMER_CONCURRENCY": "8",
        "RUNAI_STREAMER_MEMORY_LIMIT": "8589934592",
        "VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL": "100",
        "HF_HOME": "/dev/shm/hf_cache",
        "TMPDIR": "/dev/shm/vllm_tmp",
        "VLLM_CACHE_ROOT": "/dev/shm/vllm_cache",
        "VLLM_XLA_CACHE_PATH": "/dev/shm/vllm_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torchinductor",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        "TORCH_DIST_TIMEOUT": "1800",
        "VLLM_RPC_TIMEOUT": "1800000",
        "VLLM_SHM_BROADCAST_TIMEOUT_S": "1800",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
        "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": (
            "RUNAI_STREAMER_PROCESS_GROUP_SIZE,RUNAI_STREAMER_CONCURRENCY,RUNAI_STREAMER_MEMORY_LIMIT,"
            "VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL,TORCH_DIST_TIMEOUT,TMPDIR,HF_HOME,TORCHINDUCTOR_CACHE_DIR,"
            "TORCHINDUCTOR_AUTOGRAD_CACHE"
        ),
    })
    if "torch_tpu_topology" in cand:
        env["TORCH_TPU_TOPOLOGY"] = cand["torch_tpu_topology"]
    env.update(plan.get("extra_env", {}))
    env.update(cand.get("extra_env", {}))
    extra_keys = [k for k in {**plan.get("extra_env", {}), **cand.get("extra_env", {})}]
    if extra_keys:
        env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] += "," + ",".join(extra_keys)
    return env


def start_ray(idx: int, head_ip: str, env: dict[str, str], hosts: int, tpu_per_host: int) -> None:
    worker_id = os.environ.get("TPU_WORKER_ID", str(idx)) or str(idx)
    if idx == 0:
        subprocess.run(["ray", "start", "--head", "--port=6379", f"--labels=ray.io/tpu-worker-id={worker_id}"], env=env,
                       check=True)
        probe = (
            "import ray; ray.init(address='auto', logging_level='ERROR'); "
            f"print(sum(1 for n in ray.nodes() if n['Alive'] and n.get('Resources', {{}}).get('TPU', 0) >= {tpu_per_host}))"
        )
        t0 = time.time()
        while time.time() - t0 < 600:
            out = subprocess.run(["python3", "-c", probe], env=env, text=True, capture_output=True).stdout.strip()
            if out == str(hosts):
                log(0, f"all {hosts} hosts joined Ray (TPU>={tpu_per_host} each)")
                return
            time.sleep(4)
        raise TimeoutError("hosts did not join Ray")
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            with socket.create_connection((head_ip, 6379), timeout=3):
                break
        except OSError:
            time.sleep(2)
    subprocess.run(["ray", "start", f"--address={head_ip}:6379", f"--labels=ray.io/tpu-worker-id={worker_id}"], env=env,
                   check=True)


class LogMirror:
    """Copy the local server log to the results bucket periodically (GCS FUSE only uploads on close)."""

    def __init__(self, src: Path, dst: Path, period_s: float = 60.0) -> None:
        self.src, self.dst, self.period_s = src, dst, period_s
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _copy(self) -> None:
        try:
            shutil.copyfile(self.src, self.dst)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self._copy()

    def stop(self) -> None:
        self._stop.set()
        self._copy()


_KV_PAT = re.compile(r"(KV cache|kv cache|num_gpu_blocks|num_tpu_blocks|Maximum concurrency|max_num_seqs|"
                     r"memory|Memory|HBM|compil|Compil|warm|Warm|FP8 weights|requantiz|Loading weights took|"
                     r"Model loading took|init engine|cudagraph|Graph capturing)", re.I)


def extract_boot_facts(log_path: Path) -> list[str]:
    out = []
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if "ActorHandleNotFound" in line:
                continue
            if re.search(r"KV cache size|Maximum concurrency|num_gpu_blocks|GPU KV cache|KV cache memory|"
                         r"Available KV cache|Model loading took|Loading weights took|init engine|"
                         r"FP8 weights transposed|requantiz|Compil(ation|ing).*(took|finished)|warm", line, re.I):
                out.append(line.strip()[:400])
    except Exception:
        pass
    # dedupe preserving order, cap size
    seen, uniq = set(), []
    for l in out:
        k = re.sub(r"\(.*?pid=\d+.*?\)", "", l)
        if k not in seen:
            seen.add(k)
            uniq.append(l)
    return uniq[:300]


def wait_ready(proc: subprocess.Popen, log_path: Path, timeout_s: float) -> tuple[bool, float, str]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, round(time.time() - t0, 1), f"exited rc={proc.returncode}\n" + _tail(log_path)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True, round(time.time() - t0, 1), ""
        except Exception:
            pass
        time.sleep(5)
    return False, round(time.time() - t0, 1), "timeout\n" + _tail(log_path)


def _tail(log_path: Path, n: int = 6000) -> str:
    try:
        lines = [l for l in log_path.read_text(errors="replace").splitlines() if "ActorHandleNotFound" not in l]
        txt = "\n".join(lines)
        # surface the first Traceback block if any
        i = txt.find("Traceback (most recent call last)")
        head = txt[i:i + 6000] if i >= 0 else ""
        return (head + "\n...\n" + txt[-n:]) if head else txt[-n:]
    except Exception as exc:
        return f"<no log: {exc}>"


def vllm_cmd(cand: dict) -> list[str]:
    lf = cand.get("load_format", "runai_streamer")
    return [
        "vllm", "serve", cand["model_path"],
        f"--tokenizer={cand['tokenizer_path']}",
        f"--served-model-name={cand['served_name']}",
        "--host=0.0.0.0", "--port=8000", "--api-server-count=1",
        f"--tensor-parallel-size={cand['tp']}",
        f"--data-parallel-size={cand['dp']}",
        f"--load-format={lf}",
        *(["--safetensors-load-strategy=prefetch",
           '--model-loader-extra-config={"memory_limit":8589934592,"concurrency":8}'] if lf == "runai_streamer" else []),
        f"--kv-cache-dtype={cand.get('kv_dtype', 'auto')}",
        f"--gpu-memory-utilization={cand.get('mem_util', 0.88)}",
        f"--max-model-len={cand.get('max_model_len', 16384)}",
        f"--max-num-seqs={cand.get('max_num_seqs', 64)}",
        f"--max-num-batched-tokens={cand.get('max_batched_tokens', 512)}",
        "--enable-prefix-caching" if cand.get("enable_prefix_caching", True) else "--no-enable-prefix-caching",
        "--async-scheduling" if cand.get("async_scheduling", True) else "--no-async-scheduling",
        "--trust-remote-code",
        *cand.get("extra_args", []),
    ]


def sweep_cmd(plan: dict, cand: dict, tag: str, out_root: Path) -> list[str]:
    s = cand.get("sweep", {})
    cmd = [
        "python3", "/scripts/run_tpu_benchmark_sweep.py", "--host=127.0.0.1", "--port=8000",
        f"--tokenizer={cand.get('bench_tokenizer', cand['tokenizer_path'])}", f"--slice-name={plan['slice_name']}",
        f"--config-tag={tag}", f"--output-dir={out_root}",
        f"--max-input-len={s.get('max_input_len', 65536)}", f"--output-len={s.get('output_len', 1024)}",
        f"--input-lens={s.get('input_lens', '4096')}", f"--concurrencies={s.get('concurrencies', '1,8,32,64')}",
        f"--min-prompts={s.get('min_prompts', 4)}", f"--prompts-per-conc={s.get('prompts_per_conc', 2)}",
        f"--prefix-concurrencies={s.get('prefix_concurrencies', '')}",
        f"--prefix-ratios={s.get('prefix_ratios', '0.8,0.95')}",
        f"--context-lengths={s.get('context_lengths', '')}",
        f"--context-concurrency={s.get('context_concurrency', 4)}",
        f"--context-prompts={s.get('context_prompts', 8)}",
        f"--context-output-len={s.get('context_output_len', 0)}",
        f"--greedy-concurrencies={s.get('greedy_concurrencies', '')}",
        f"--greedy-output-len={s.get('greedy_output_len', 1024)}",
        f"--temperature={s.get('temperature', 0.6)}",
    ]
    if s.get("no_inferencex", True):
        cmd.append("--no-inferencex")
    return cmd


def write_status(out_root: Path, entry: dict) -> None:
    _STATUS.append(entry)
    try:
        (out_root / "suite_status.json").write_text(json.dumps(_STATUS, indent=2))
    except Exception:
        pass


def run_experiment(idx: int, url: str, head_ip: str, plan: dict, exp: dict, cands: list[dict]) -> dict | None:
    out_root = Path(plan["out_root"])
    hosts, tpu_per_host = int(plan["hosts"]), int(plan["tpu_per_host_in_ray"])
    for c_i, cand in enumerate(cands):
        env = build_env(plan, cand)
        key = f"{exp['id']}_c{c_i}"
        barrier(url, f"{key}_clean", idx, hosts)
        clean_host()
        barrier(url, f"{key}_ray", idx, hosts)
        start_ray(idx, head_ip, env, hosts, tpu_per_host)
        if idx != 0:
            if wait_key(url, key) == "OK":
                return cand
            continue
        tag = cand["config_tag"]
        local_log = Path("/dev/shm/logs") / f"server_{exp['id']}_{tag}.log"
        gcs_log = out_root / f"server_{exp['id']}_{tag}.log"
        cmd = vllm_cmd(cand)
        log(0, f"[{exp['id']}] candidate {c_i + 1}/{len(cands)} {tag}: {' '.join(cmd)}")
        t_start = time.time()
        with open(local_log, "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        mirror = LogMirror(local_log, gcs_log)
        ok, boot_s, err = wait_ready(proc, local_log, float(cand.get("ready_timeout_s", 5400)))
        boot = {"exp_id": exp["id"], "config_tag": tag, "boot_s": boot_s, "command": cmd,
                "candidate": cand, "boot_facts": extract_boot_facts(local_log),
                "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))}
        if not ok:
            log(0, f"[{exp['id']}] {tag} failed to boot after {boot_s}s:\n{err[-2500:]}")
            boot.update(status="FAILED_BOOT", error_tail=err[-8000:])
            (out_root / f"diag_{exp['id']}_{tag}.json").write_text(json.dumps(boot, indent=2))
            proc.kill()
            mirror.stop()
            write_status(out_root, {"exp": exp["id"], "tag": tag, "status": "FAILED_BOOT", "boot_s": boot_s})
            kv_set(url, key, "NEXT")
            continue
        log(0, f"[{exp['id']}] {tag} READY in {boot_s}s -> benchmark sweep")
        boot["status"] = "READY"
        (out_root / f"boot_{exp['id']}_{tag}.json").write_text(json.dumps(boot, indent=2))
        write_status(out_root, {"exp": exp["id"], "tag": tag, "status": "READY", "boot_s": boot_s})
        sweep = sweep_cmd(plan, cand, tag, out_root)
        log(0, f"[{exp['id']}] sweep: {' '.join(sweep)}")
        rc = subprocess.run(sweep, env=env, check=False).returncode
        log(0, f"[{exp['id']}] sweep {tag} rc={rc}")
        boot["boot_facts_final"] = extract_boot_facts(local_log)
        (out_root / f"boot_{exp['id']}_{tag}.json").write_text(json.dumps(boot, indent=2))
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        mirror.stop()
        write_status(out_root, {"exp": exp["id"], "tag": tag, "status": "SWEEP_OK" if rc == 0 else f"SWEEP_RC{rc}",
                                "boot_s": boot_s, "total_s": round(time.time() - t_start, 1)})
        kv_set(url, key, "OK" if rc == 0 else "NEXT")
        if rc == 0:
            return cand
    return None


def derive_candidates(exp: dict, base: dict) -> list[dict]:
    variants = exp.get("variants") or [{k: exp[k] for k in ("override", "extra_args_add", "extra_args_remove",
                                                             "config_tag", "tag_suffix") if k in exp}]
    cands = []
    for v in variants:
        cand = json.loads(json.dumps(base))
        ov = dict(v.get("override", {}))
        if "sweep" in ov:
            cand["sweep"] = {**cand.get("sweep", {}), **ov.pop("sweep")}
        if "sweep_replace" in v:
            cand["sweep"] = dict(v["sweep_replace"])
        cand.update(ov)
        cand["config_tag"] = v.get("config_tag") or (base["config_tag"] + v.get("tag_suffix", ""))
        removes = v.get("extra_args_remove", [])
        args = [a for a in base.get("extra_args", []) if not any(a.startswith(r) for r in removes)]
        cand["extra_args"] = args + list(v.get("extra_args_add", []))
        cands.append(cand)
    return cands


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text())
    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    head_fqdn = f"{os.environ['JOB_NAME_PREFIX']}-0.{os.environ['HEADLESS_SVC']}"
    Path(plan["out_root"]).mkdir(parents=True, exist_ok=True)
    if idx == 0:
        srv = ThreadingHTTPServer(("0.0.0.0", COORD_PORT), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    head_ip = resolve(head_fqdn)
    url = f"http://{head_ip}:{COORD_PORT}"
    only = {s.strip() for s in os.environ.get("SUITE_ONLY", "").split(",") if s.strip()}
    winners: dict[str, dict] = {}
    for exp in plan["experiments"]:
        if only and exp["id"] not in only:
            continue
        if "derive_from" in exp:
            base = winners.get(exp["derive_from"])
            if base is None:
                log(idx, f"[{exp['id']}] skipped: base {exp['derive_from']} did not succeed")
                if idx == 0:
                    write_status(Path(plan["out_root"]), {"exp": exp["id"], "status": "SKIPPED_BASE_FAILED"})
                continue
            cands = derive_candidates(exp, base)
        else:
            cands = exp["candidates"]
        won = run_experiment(idx, url, head_ip, plan, exp, cands)
        if won is not None:
            winners[exp["id"]] = won
        log(idx, f"[{exp['id']}] -> {'OK ' + won['config_tag'] if won else 'no candidate succeeded'}")
    barrier(url, "suite_done", idx, int(plan["hosts"]), timeout_s=8 * 3600)
    if idx == 0:
        time.sleep(30)
    clean_host()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
