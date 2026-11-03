#!/usr/bin/env python3
"""TPU serving benchmark sweep against an OpenAI-compatible vLLM endpoint (used by the ml3 v7x-16 suite).

Suites (each row = one closed-loop run with `concurrency` in-flight streams):
  1. concurrency_<in>in : sampled (temperature T, ignore_eos) fixed in/out lengths, for each --input-lens x
     --concurrencies.
  2. cached_<pct>pct_prefix : same, with a shared prompt prefix (prefix-cache hit simulation).
  3. context_sweep : long-context prompts (--context-lengths) at --context-concurrency.
  4. greedy_natural : temperature 0, natural-language chat prompts (no ignore_eos) at --greedy-concurrencies; the
     /metrics spec-decode counters are scraped before/after each row (acceptance rate, mean acceptance length).
Token counts come from the server's `usage` (stream_options.include_usage) so speculative decoding (several tokens per
SSE chunk) is counted correctly. Prompts are built with the model tokenizer to exact token lengths when available.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

_TOK = None


def load_tokenizer(path: str):
    global _TOK
    try:
        from tokenizers import Tokenizer  # type: ignore

        for cand in (Path(path) / "tokenizer.json",):
            if cand.exists():
                _TOK = Tokenizer.from_file(str(cand))
                print(f"[tpu-bench] exact-length prompts via {cand}", flush=True)
                return
    except Exception as exc:  # pragma: no cover
        print(f"[tpu-bench] tokenizer unavailable ({exc}); using word heuristic", flush=True)


def wait_for_server(base_url: str, timeout_s: int = 86400) -> str:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=10) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(15)
    else:
        raise TimeoutError(f"Server at {base_url} did not become healthy within {timeout_s}s")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=15) as resp:
        model_id = json.loads(resp.read().decode("utf-8"))["data"][0]["id"]
    print(f"[tpu-bench] Server READY: {model_id}", flush=True)
    return model_id


def run_sanity_check(base_url: str, model_id: str) -> dict:
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "Explain Mixture-of-Experts routing on TPUs in two concise sentences."}],
        "max_tokens": 128,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or str(msg)
    print(f"[tpu-bench] Sanity check OK in {elapsed:.2f}s (usage={body.get('usage')}): {text[:300]}", flush=True)
    return {"elapsed_s": round(elapsed, 3), "usage": body.get("usage", {}), "sample_output": text[:600]}


_VOCAB = ["tensor", "matrix", "router", "expert", "prefill", "decode", "latency", "bandwidth", "compiler", "kernel",
          "attention", "sequence", "cache", "pipeline", "parallel", "cluster", "topology", "memory", "compute", "token"]


def _build_word_block(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    if _TOK is None:
        n_words = max(16, int(approx_tokens * 0.98))
        return " ".join(rng.choice(_VOCAB) for _ in range(n_words))
    words = " ".join(rng.choice(_VOCAB) for _ in range(int(approx_tokens * 1.3) + 32))
    ids = _TOK.encode(words, add_special_tokens=False).ids[:approx_tokens]
    return _TOK.decode(ids)


_TOPICS = [
    "the history and engineering of suspension bridges", "how TPUs and GPUs differ for transformer inference",
    "the water cycle and climate feedback loops", "designing a fault-tolerant distributed key-value store",
    "the economics of renewable energy storage", "how the human immune system recognizes pathogens",
    "the evolution of programming languages from FORTRAN to Rust", "strategies for teaching mathematics to children",
    "the physics of black holes and Hawking radiation", "best practices for securing a cloud-native application",
    "the cultural impact of the printing press", "how compilers optimize loops", "the biology of coral reefs",
    "the design of modern jet engines", "an introduction to Bayesian statistics with examples",
    "the logistics of global container shipping",
]


def natural_messages(i: int) -> list[dict]:
    topic = _TOPICS[i % len(_TOPICS)]
    return [{"role": "user", "content": (
        f"Write a long, detailed, well-structured essay (at least 1500 words, with numbered sections and concrete "
        f"examples) about {topic}.")}]


async def _stream_one(session, url, payload, sem, max_tokens) -> dict:
    async with sem:
        t_start = time.perf_counter()
        t_first = None
        chunks = 0
        usage = None
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                if resp.status != 200:
                    return {"ok": False, "error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        c0 = choices[0]
                        piece = c0.get("text") or (c0.get("delta") or {}).get("content") or \
                            (c0.get("delta") or {}).get("reasoning_content")
                        if piece:
                            if t_first is None:
                                t_first = time.perf_counter()
                            chunks += 1
            t_end = time.perf_counter()
        except Exception as exc:
            return {"ok": False, "error": repr(exc)[:300]}
        if t_first is None:
            t_first = t_end
        out_toks = int(usage.get("completion_tokens")) if usage and usage.get("completion_tokens") else max(chunks, 1)
        prompt_toks = int(usage.get("prompt_tokens")) if usage and usage.get("prompt_tokens") else None
        ttft_ms = (t_first - t_start) * 1000.0
        e2e_s = t_end - t_start
        decode_s = max(t_end - t_first, 1e-6)
        tpot_ms = decode_s * 1000.0 / max(out_toks - 1, 1)
        return {"ok": True, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "e2e_s": e2e_s, "out_tokens": out_toks,
                "prompt_tokens": prompt_toks, "chunks": chunks,
                "stream_tps": (out_toks - 1) / decode_s if out_toks > 1 else out_toks / max(e2e_s, 1e-6),
                "e2e_tps": out_toks / max(e2e_s, 1e-6)}


def _pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))]


def summarize(rows: list[dict], wall_s: float, meta: dict) -> dict:
    ok = [r for r in rows if r.get("ok")]
    out = dict(meta, successful=len(ok), failed=len(rows) - len(ok))
    if not ok:
        out["errors"] = [r.get("error") for r in rows[:3]]
        return out
    ttfts = [r["ttft_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok]
    stps = [r["stream_tps"] for r in ok]
    total_out = sum(r["out_tokens"] for r in ok)
    ptoks = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    out.update({
        "wall_s": round(wall_s, 2),
        "total_output_tokens": total_out,
        "total_output_tps": round(total_out / max(wall_s, 1e-6), 2),
        "total_tps_incl_prompt": round((total_out + sum(ptoks)) / max(wall_s, 1e-6), 2) if ptoks else None,
        "mean_prompt_tokens": round(sum(ptoks) / len(ptoks), 1) if ptoks else None,
        "mean_output_tokens": round(total_out / len(ok), 1),
        "mean_stream_tps": round(sum(stps) / len(stps), 2),
        "p50_stream_tps": round(_pct(stps, 50), 2),
        "p10_stream_tps": round(_pct(stps, 10), 2),
        "mean_ttft_ms": round(sum(ttfts) / len(ttfts), 1),
        "p50_ttft_ms": round(_pct(ttfts, 50), 1),
        "p90_ttft_ms": round(_pct(ttfts, 90), 1),
        "p99_ttft_ms": round(_pct(ttfts, 99), 1),
        "mean_tpot_ms": round(sum(tpots) / len(tpots), 2),
        "p50_tpot_ms": round(_pct(tpots, 50), 2),
        "p90_tpot_ms": round(_pct(tpots, 90), 2),
        "p99_tpot_ms": round(_pct(tpots, 99), 2),
    })
    print(f"[tpu-bench] RESULT: {json.dumps(out)}", flush=True)
    return out


async def run_sampled(base_url, model_id, input_tokens, output_tokens, concurrency, num_prompts,
                      prefix_hit_ratio=0.0, temperature=0.6) -> dict:
    url = f"{base_url}/v1/completions"
    shared_len = int(input_tokens * prefix_hit_ratio)
    unique_len = max(32, input_tokens - shared_len)
    shared_prefix = _build_word_block(shared_len, seed=42) if shared_len > 0 else ""
    prompts = [(shared_prefix + " " + _build_word_block(unique_len, seed=1000 + i)).strip() for i in range(num_prompts)]
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=max(concurrency * 2, 64))

    def payload(p, mt):
        return {"model": model_id, "prompt": p, "max_tokens": mt, "temperature": temperature, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}

    async with aiohttp.ClientSession(connector=conn) as session:
        if prefix_hit_ratio > 0:
            await _stream_one(session, url, payload(prompts[0], 16), sem, 16)
        t0 = time.perf_counter()
        rows = await asyncio.gather(*[_stream_one(session, url, payload(p, output_tokens), sem, output_tokens)
                                      for p in prompts])
        wall = time.perf_counter() - t0
    return summarize(rows, wall, {"input_tokens": input_tokens, "output_tokens": output_tokens,
                                  "concurrency": concurrency, "num_prompts": num_prompts,
                                  "prefix_hit_ratio": prefix_hit_ratio, "temperature": temperature})


def scrape_spec_metrics(base_url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as r:
            text = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}
    acc: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:spec_decode_[a-z_]+?)(_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name = m.group(1)
        if "per_pos" in name:
            continue
        acc[name] = acc.get(name, 0.0) + float(m.group(4))
    return acc


async def run_greedy(base_url, model_id, output_tokens, concurrency, num_prompts) -> dict:
    url = f"{base_url}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=max(concurrency * 2, 64))
    before = scrape_spec_metrics(base_url)
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        rows = await asyncio.gather(*[_stream_one(session, url, {
            "model": model_id, "messages": natural_messages(i), "max_tokens": output_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}, sem, output_tokens) for i in range(num_prompts)])
        wall = time.perf_counter() - t0
    after = scrape_spec_metrics(base_url)
    res = summarize(rows, wall, {"input_tokens": "natural-chat", "output_tokens": output_tokens,
                                 "concurrency": concurrency, "num_prompts": num_prompts, "prefix_hit_ratio": 0.0,
                                 "temperature": 0.0})
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    draft_toks = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    if draft_toks > 0:
        res["spec_num_drafts"] = drafts
        res["spec_draft_tokens"] = draft_toks
        res["spec_accepted_tokens"] = accepted
        res["spec_acceptance_rate"] = round(accepted / draft_toks, 4)
        if drafts > 0:
            res["spec_mean_acceptance_length"] = round(1.0 + accepted / drafts, 3)
    return res


def maybe_run_inferencex(host, port, model_id, tokenizer, input_len, output_len, concurrency, num_prompts, out_dir):
    script = Path("/workspace/inferencex/utils/bench_serving/benchmark_serving.py")
    if not script.exists():
        return None
    result_file = out_dir / f"inferencex_in{input_len}_out{output_len}_c{concurrency}.json"
    cmd = [sys.executable, str(script), "--backend=vllm", "--request-rate=inf",
           "--percentile-metrics=ttft,tpot,itl,e2el", f"--host={host}", f"--port={port}", f"--model={model_id}",
           f"--tokenizer={tokenizer}", "--dataset-name=random", f"--random-input-len={input_len}",
           f"--random-output-len={output_len}", "--random-range-ratio=0.8", f"--num-prompts={num_prompts}",
           f"--max-concurrency={concurrency}", "--ignore-eos", "--save-result", f"--result-filename={result_file}"]
    print(f"[tpu-bench] Running InferenceX: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        print(f"[tpu-bench] InferenceX warning (rc={proc.returncode}): {proc.stderr[-600:]}", flush=True)
        return None
    return json.loads(result_file.read_text()) if result_file.exists() else None


def render_markdown(meta: dict, sanity: dict, rows: list[dict]) -> str:
    lines = [f"# vLLM-TorchTPU benchmark - `{meta['slice_name']}` / `{meta['config_tag']}`", "",
             f"- Timestamp (UTC): `{meta['timestamp_utc']}`", f"- Served model: `{meta['model_id']}`",
             f"- Sanity check ({sanity['elapsed_s']}s): `{sanity['sample_output'][:200]}`", "",
             "| Suite | In (mean actual) | Out | Prefix | Conc | Total out tok/s | Mean stream tok/s/user | "
             "TTFT p50/p90/p99 (ms) | TPOT p50/p90 (ms) | Accept rate / len | OK/Fail |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("successful", 0) == 0:
            lines.append(f"| {r.get('suite')} | {r.get('input_tokens')} | {r.get('output_tokens')} | "
                         f"{int(r.get('prefix_hit_ratio', 0) * 100)}% | {r.get('concurrency')} | ERR | ERR | - | - | - | "
                         f"0/{r.get('failed')} |")
            continue
        acc = (f"{r['spec_acceptance_rate']:.3f} / {r.get('spec_mean_acceptance_length', '-')}"
               if "spec_acceptance_rate" in r else "-")
        lines.append(
            f"| {r.get('suite')} | {r.get('input_tokens')} ({r.get('mean_prompt_tokens')}) | {r.get('mean_output_tokens')} | "
            f"{int(r.get('prefix_hit_ratio', 0) * 100)}% | {r['concurrency']} | **{r['total_output_tps']}** | "
            f"**{r['mean_stream_tps']}** | {r['p50_ttft_ms']}/{r['p90_ttft_ms']}/{r['p99_ttft_ms']} | "
            f"{r['p50_tpot_ms']}/{r['p90_tpot_ms']} | {acc} | {r['successful']}/{r['failed']} |")
    lines.append("")
    return "\n".join(lines)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--slice-name", default="tpu-slice")
    ap.add_argument("--config-tag", default="baseline")
    ap.add_argument("--max-input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--input-lens", default="4096")
    ap.add_argument("--concurrencies", default="1,8,32,64")
    ap.add_argument("--min-prompts", type=int, default=4)
    ap.add_argument("--prompts-per-conc", type=int, default=2)
    ap.add_argument("--prefix-concurrencies", default="32")
    ap.add_argument("--prefix-ratios", default="0.8,0.95")
    ap.add_argument("--context-lengths", default="2048,8192")
    ap.add_argument("--context-concurrency", type=int, default=4)
    ap.add_argument("--context-prompts", type=int, default=8)
    ap.add_argument("--context-output-len", type=int, default=0)
    ap.add_argument("--greedy-concurrencies", default="")
    ap.add_argument("--greedy-output-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-inferencex", action="store_true")
    ap.add_argument("--output-dir", default="/models/benchmark-results")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    model_id = wait_for_server(base_url)
    if args.tokenizer:
        load_tokenizer(args.tokenizer)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}__{model_id.strip('/').replace('/', '__')}__{args.config_tag}__{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp_utc": ts, "slice_name": args.slice_name, "config_tag": args.config_tag, "model_id": model_id,
            "tokenizer": args.tokenizer, "max_input_len": args.max_input_len, "output_len": args.output_len,
            "args": vars(args)}
    sanity = run_sanity_check(base_url, model_id)
    rows: list[dict] = []

    def save():
        payload = {"meta": meta, "sanity": sanity, "results": rows}
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        (out_dir / "REPORT.md").write_text(render_markdown(meta, sanity, rows))

    concs = _ints(args.concurrencies)
    plan_in: list[tuple[int, list[int]]] = []
    for item in [x.strip() for x in args.input_lens.split(",") if x.strip()]:
        if ":" in item:
            ln, cs = item.split(":", 1)
            plan_in.append((int(ln), [int(c) for c in cs.split("|") if c.strip()]))
        else:
            plan_in.append((int(item), concs))
    for in_len, in_concs in plan_in:
        for c in in_concs:
            n = max(c * args.prompts_per_conc, args.min_prompts)
            r = asyncio.run(run_sampled(base_url, model_id, in_len, args.output_len, c, n, 0.0, args.temperature))
            r["suite"] = f"concurrency_{in_len}in"
            rows.append(r)
            save()
    std_in = min(4096, args.max_input_len)
    for ratio in [float(x) for x in args.prefix_ratios.split(",") if x.strip()]:
        for c in _ints(args.prefix_concurrencies):
            n = max(c * args.prompts_per_conc, args.min_prompts)
            r = asyncio.run(run_sampled(base_url, model_id, std_in, args.output_len, c, n, ratio, args.temperature))
            r["suite"] = f"cached_{int(ratio * 100)}pct_prefix"
            rows.append(r)
            save()
    ctx_out = args.context_output_len or args.output_len
    for ctx in [x for x in _ints(args.context_lengths) if x <= args.max_input_len]:
        r = asyncio.run(run_sampled(base_url, model_id, ctx, ctx_out, args.context_concurrency, args.context_prompts,
                                    0.0, args.temperature))
        r["suite"] = "context_sweep"
        rows.append(r)
        save()
    for c in _ints(args.greedy_concurrencies):
        n = max(c * args.prompts_per_conc, args.min_prompts)
        r = asyncio.run(run_greedy(base_url, model_id, args.greedy_output_len, c, n))
        r["suite"] = "greedy_natural"
        rows.append(r)
        save()
    if not args.no_inferencex and concs:
        meta["inferencex_crosscheck"] = maybe_run_inferencex(args.host, args.port, model_id, args.tokenizer or model_id,
                                                             std_in, args.output_len, min(8, max(concs)), 16, out_dir)
    save()
    print(render_markdown(meta, sanity, rows), flush=True)
    print(f"[tpu-bench] Saved artifacts to {out_dir}", flush=True)
    ok = sum(1 for r in rows if r.get("successful", 0) > 0)
    return 0 if ok > 0 or not rows else 3


if __name__ == "__main__":
    raise SystemExit(main())
