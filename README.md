# `Mistral-Large-3-675B-Instruct-2512` (FP8) Serving on TPU `v7x` with `vllm-torchtpu`

Multi-host deployment manifests, runtime patch script, OpenAI SSE streaming benchmark harness (`c=1..512` concurrency, `8K`–`128K` context, `80%`/`95%` prefix caching, and greedy `natural-chat`), and latency/throughput results for **[`mistralai/Mistral-Large-3-675B-Instruct-2512`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512)** (`675B` total / `41B` active parameters, FP8 checkpoint) across **four `TP × DP × EP=32` parallelism topologies** on a **single TPU `v7x-16` slice (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** with **`vllm 0.32.0` + `vllm-torchtpu`**.

---

## 1. Executive Summary (`v7x-16` Pareto Frontier Across `TP=1..8 × EP=32`)

Sharding all `128` routed experts across all `32` TensorCores (`EP=32` = `4` routed experts per TensorCore) while sweeping Tensor Parallelism (`TP=1, 2, 4, 8`) and Data Parallelism (`DP=32, 16, 8, 4`) maps the full throughput-vs-interactivity Pareto frontier on a single `v7x-16` slice (`0` failed requests across all benchmarks):

| Parallelism Topology (`v7x-16`, `32` TensorCores) | Peak Output TPS (`4K:1K`) | Peak Total TPS (`4K:1K`) | `1K:1K` Peak Output TPS | Single-Stream Decode (`TPOT p50`) | `32K` / `64K` TTFT p50 (`c=4`) | Slice FP8 MLA KV Cache | Manifest & Artifacts |
|---|---:|---:|---:|---:|---:|---:|---|
| **`TP=1, DP=32, EP=32`** (Max Throughput & KV Capacity) | **`5,806.06`** (`c=512`) | **`29,035.98`** (`c=512`) | — | `39.43` tok/s (`25.37 ms`) | `38.10 s` / `110.01 s` | **`37.19M` tokens** | [`yaml`](./k8s/v7x-16-ml3-fp8-tp1-dp32-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp1-dp32-ep32__20261103T054109Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp1-dp32-ep32__20261103T054109Z/summary.json) |
| **`TP=2, DP=16, EP=32`** (Intra-Chip TP + `128K` Context) | **`3,812.96`** (`c=256`) | **`19,068.51`** (`c=256`) | **`5,267.09`** (`c=256`) | `50.09` tok/s (`19.97 ms`) | `25.04 s` / `57.56 s` (`154.47 s` @ `128K`) | **`22.99M` tokens** | [`yaml`](./k8s/v7x-16-ml3-fp8-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp2-dp16-ep32-fp8kv__20261103T044915Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp2-dp16-ep32-fp8kv__20261103T044915Z/summary.json) |
| **`TP=4, DP=8, EP=32`** (`c=64` Balanced Sweet Spot) | **`3,128.35`** (`c=256`) | **`15,644.79`** (`c=256`) | — | `61.94` tok/s (`16.15 ms`, sampled `4K:1K` `c=1`) | — / `40.92 s` | `12.50M` tokens | [`yaml`](./k8s/v7x-16-ml3-fp8-tp4-dp8-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp4-dp8-ep32__20261103T064257Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp4-dp8-ep32__20261103T064257Z/summary.json) |
| **`TP=8, DP=4, EP=32`** (Ultra-Low Latency & Fast Prefill) | **`3,072.43`** (`c=256`) | **`15,365.17`** (`c=256`) | — | **`77.07` tok/s (`12.98 ms`)** | **`8.94 s` / `14.94 s`** | `6.54M` tokens | [`yaml`](./k8s/v7x-16-ml3-fp8-tp8-dp4-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp8-dp4-ep32__20261103T061631Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp8-dp4-ep32__20261103T061631Z/summary.json) |

Conventions: `4K:1K` = `4,096` input / `1,024` output tokens; `c` = concurrent in-flight requests. Output TPS = generated tokens / wall time of the row; Total TPS = (prompt + generated) tokens / wall time of the row. Single-Stream Decode = mean per-request decode rate (`(output tokens − 1) / decode time`) and median TPOT of the greedy `natural-chat` `c=1` row (`TP=4`: sampled `4K:1K` `c=1` row). TTFT is measured client-side from request send to first streamed token and includes queueing. Slice KV cache = KV-cache tokens per DP engine (reported at server start) × number of DP engines. Percentages compare the stated metric at the same workload and concurrency.

### Key Architectural Takeaways

1. **Maximum Slice Throughput (`TP=1, DP=32, EP=32`):** Assigning one independent Data Parallel engine to each of the `32` TensorCores eliminates Tensor Parallel all-reduces on attention/MLA projections while keeping `EP=32` expert all-to-all across the `2x2x4` ICI mesh, reaching **`5,806.06` output tok/s (`29,035.98` total tok/s)** at `4,096` input / `1,024` output (`c=512`) and exposing **`37.19M` FP8 MLA KV-cache tokens** across the slice.
2. **Intra-Chip `TP=2` Balance & `128K` Long Context (`TP=2, DP=16, EP=32`):** Keeping `TP=2` inside the two TensorCores of each `v7x` chip avoids cross-chip TP collectives, delivering **`5,267.09` output tok/s** on `1,024 : 1,024` (`c=256`), **`3,812.96` output tok/s** on `4,096 : 1,024` (`c=256`, **`+20.8%`** higher than the BF16-weight run's `3,156.49` output tok/s with the same `TP=2, DP=16, EP=32` layout), **`50.09` stream tok/s (`19.97 ms` p50 TPOT)** on greedy single-stream decode, and **`131,072` (`128K`)** long-context serving with `22.99M` KV-cache tokens.
3. **Interactive Low-Latency & Long-Prefill Regime (`TP=8, DP=4, EP=32`):** Scaling each engine to `TP=8` (one full 4-chip `v7x` host per DP replica) cuts single-stream median TPOT to **`12.98 ms` (`77.07` stream tok/s)**, leads throughput at `c=1..32` (`501.68` output tok/s at `c=8`, `1,324.28` output tok/s at `c=32`), and accelerates `65,536`-token prefill TTFT (`c=4`) to **`14.94 s` (`9,170.54` total tok/s)** — **`3.85×` faster** than `TP=2` (`57.56 s`) and **`7.36×` faster** than `TP=1` (`110.01 s`); `TP=4` measures `40.92 s`. Each long prompt is prefilled by a single DP engine in `1,024`-token chunks (`--max-num-batched-tokens=1024`), so long-context TTFT falls as the per-engine TP degree grows.
4. **FP8 vs BF16 Weights (`TP=2, DP=16, EP=32`):** FP8 output throughput is `+20.5%` (`c=128`) to `+29.3%` (`c=8`) higher than BF16 across the `4K:1K` concurrency sweep (`3,812.96` vs `3,156.49` output tok/s at `c=256`) and `+10.5%` / `+11.4%` higher at `32K` / `64K` input context (`c=4`). At `131,072` input tokens (`c=4`, `8` requests) the BF16 row reached `+10.9%` higher total tok/s (`2,940.83` vs `2,650.90`); the FP8 row has the lower median TTFT (`154.47 s` vs `163.48 s`) and includes two requests with TTFT ≈ `217.5 s`.

---

## 2. Model, Runtime & Patch Overview

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) (staged via [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py)) |
| **Architecture** | `MistralLarge3ForCausalLM` (`--language-model-only`, `61` Transformer layers: `3` dense + `58` MoE layers, `128` routed experts + `1` shared expert, top-4 active routing, Multi-head Latent Attention `MLA` with `kv_lora_rank=512`, `qk_rope_head_dim=64`, `294,912` max position embeddings) |
| **Runtime Stack** | Measured runs: image `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/vllm-torchtpu-bench:v0.32.0` — `vllm 0.32.0` + `vllm-torchtpu` (source tree at `/workspace/vllm-torchtpu`), `torch 2.13.0+cpu` wheel, `jax 0.11.2`, `ray 2.59.0`; Ray multi-host executor driven by [`scripts/run_vllm_tpu_multihost_plan.py`](./scripts/run_vllm_tpu_multihost_plan.py) from [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml). [`docker/Dockerfile`](./docker/Dockerfile) is a reference build, not used for the measured runs |
| **Runtime Patch ([`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py))** | Idempotent patch applied on every host before `vllm serve`. Default (all results in this repository): registers the compressed-tensors FP8 MoE block scales under the checkpoint name `weight_scale` (instead of the DeepSeek-style `weight_scale_inv`) so `load_weights` maps the expert scale tensors; numerics unchanged (block-128 FP8 weights are re-quantized to per-channel FP8 at load). Eagle draft-model patches are opt-in (`ML3_EAGLE_PATCHES=1`) |

### Boot, Memory & KV Capacity

| Topology | Server Ready (s) | Engine Init / Compilation (s) | Weights per TensorCore (GiB) | KV-Cache Tokens per Engine | Max Concurrency @ `132,096` Tokens |
|---|---:|---:|---:|---:|---:|
| `TP=1, DP=32, EP=32` | `875` | `755.1` / `595.4` | `41.09` | `1,162,240` | `8.80×` |
| `TP=2, DP=16, EP=32` | `980` | `782.6` / `597.7` | `31.11` | `1,436,672` | `10.88×` |
| `TP=4, DP=8, EP=32` | `875` | `762.0` / `587.1` | `26.53` | `1,562,624` | `11.83×` |
| `TP=8, DP=4, EP=32` | `865` | `748.5` / `565.6` | `23.94` | `1,634,304` | `12.37×` |

Server Ready = time from `vllm serve` launch until `/v1/models` responds (weight streaming, compilation, KV-cache allocation). Engine Init / Compilation = vLLM's `init engine (profile, create kv cache, warmup model)` time and its compilation share, maximum over DP engines. Each TensorCore streams only its expert-parallel shard from GCS (`--enable-ep-weight-filter`: `47.9 GB` of the `681.5 GB` checkpoint). The runtime reports `94.74 GiB` of HBM per TensorCore; `--gpu-memory-utilization=0.88` caps usage at `83.37 GiB`.

### Methodology Notes

- **Patch:** every result in this repository was produced with the default path of [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py), which only registers the compressed-tensors FP8 MoE scale names. The Eagle draft-model patches were not applied.
- **KV cache:** `--kv-cache-dtype=fp8` (e4m3) without calibrated KV scales, for both the FP8-weight and the BF16-weight runs.
- **BF16 comparison:** the BF16-weight run ([`mistral-large-3-bf16-tpu-vllm-inference`](https://github.com/ankdeshm/mistral-large-3-bf16-tpu-vllm-inference)) used `--gpu-memory-utilization=0.90`; the FP8 runs used `0.88`. All other server flags are the same.
- **Single-stream decode:** the `TP=4` value comes from the sampled `4K:1K` `c=1` row (no greedy row was run for `TP=4`); the other topologies use the greedy `natural-chat` `c=1` row.
- **Accuracy:** no accuracy evaluation was run; outputs were checked for coherent text only.
- **Prefix caching:** the KV cache uses `1,024`-token pages, so the `80%` and `95%` shared-prefix rows (`4,096`-token prompts) both reuse the same `3` of `4` pages (`3,072` tokens). At `c=32` this lowers TTFT p50 from `3,492.8 ms` (no shared prefix) to `2,703.6 ms` / `2,655.7 ms` and raises output throughput by `+8.8%` / `+11.2%`.
- **`--language-model-only`:** skips the Pixtral vision encoder (`48` layers, ~`2.5B` parameters) and its image-input memory profiling; image inputs were not benchmarked.
- **Workloads:** sampled rows use `/v1/completions` with `T=0.6`, `ignore_eos` and a fixed output length on random-word prompts truncated to the stated token length with the model tokenizer; greedy `natural-chat` rows use `/v1/chat/completions` with `T=0` on natural-language prompts (mean prompt tokens in parentheses). Concurrency rows send `max(2×c, 4)` requests; context rows send `8` requests at `c=4`. Percentiles are nearest-rank over the requests in a row.

### Supported Configurations & Known Limitations

- **Measured on this stack:** FP8 weights with `TP × DP` = `1 × 32`, `2 × 16`, `4 × 8`, `8 × 4` and `EP=32` on one `v7x-16` slice; `--max-model-len=132096`; FP8 KV cache; prefix caching; async scheduling; text-only serving (`--language-model-only`).
- **Eagle speculative decoding** (`mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle`) does not run on this stack yet: the vLLM `0.32.0` Mistral-Large-3 Eagle draft model does not yet initialize and load under `vllm-torchtpu` (draft-model attributes and per-tensor FP8 handling).
- **Eagle patches** in [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) are opt-in (`ML3_EAGLE_PATCHES=1`, default off) and were not used for any result in this repository. The executed plan also lists an Eagle experiment (`ml3_fp8_tp2_dp16_ep32_eagle`), which produces no results on this stack.
- **`131,072`-token row** (`TP=2`, `c=4`): an 8-request sample that includes two slower requests (TTFT ≈ `217.5 s` vs `154.47 s` median).

---

## 3. Detailed Benchmark Results by Topology

### 3.1 `TP=1, DP=32, EP=32` — Maximum Slice Throughput (`ml3-fp8-tp1-dp32-ep32`)

- Manifest: [`k8s/v7x-16-ml3-fp8-tp1-dp32-ep32.yaml`](./k8s/v7x-16-ml3-fp8-tp1-dp32-ep32.yaml) · Artifacts: [`REPORT.md`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp1-dp32-ep32__20261103T054109Z/REPORT.md) · [`summary.json`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp1-dp32-ep32__20261103T054109Z/summary.json)

| Workload Suite (`Script Anchor`) | Input : Output | Temp | Conc | Output TPS | Total TPS | Mean Stream TPS | TTFT p50/p90 (ms) | TPOT p50/p90 (ms) | OK/Fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `1` | `34.78` | `173.92` | `39.05` | `3,242.9 / 3,260.1` | `25.61 / 25.62` | `4/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `8` | `256.72` | `1,283.85` | `37.44` | `4,746.7 / 4,810.3` | `26.55 / 27.34` | `16/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `32` | `989.39` | `4,947.93` | `35.94` | `4,345.9 / 5,125.6` | `27.97 / 28.18` | `64/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `64` | `1,764.58` | `8,824.61` | `30.51` | `3,818.0 / 6,477.0` | `31.99 / 35.19` | `128/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `128` | `2,875.65` | `14,381.07` | `24.83` | `4,984.8 / 7,229.0` | `39.31 / 45.54` | `256/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `256` | **`4,389.15`** | **`21,950.06`** | `19.20` | `5,610.8 / 10,292.3` | `54.61 / 57.70` | `512/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `512` | **`5,806.06`** | **`29,035.98`** | `12.74` | `7,648.6 / 15,716.8` | `76.76 / 99.42` | `1024/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `32,768 : 512` | `0.6` | `4` | `38.88` | `2,526.96` | `34.39` | `38,098.1 / 39,029.4` | `29.80 / 32.86` | `8/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `65,536 : 512` | `0.6` | `4` | `16.48` | `2,125.82` | `33.60` | `110,013.3 / 110,057.7` | `27.90 / 37.36` | `8/0` |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (39.5) : 1,024` | `0.0` | `1` | `39.31` | `40.83` | `39.43` | `86.4 / 157.2` | `25.37 / 25.37` | `4/0` |

### 3.2 `TP=2, DP=16, EP=32` — Intra-Chip TP, Prefix Cache & `128K` Long Context (`ml3-fp8-tp2-dp16-ep32-fp8kv`)

- Manifest: [`k8s/v7x-16-ml3-fp8-tp2-dp16-ep32.yaml`](./k8s/v7x-16-ml3-fp8-tp2-dp16-ep32.yaml) · Artifacts: [`REPORT.md`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp2-dp16-ep32-fp8kv__20261103T044915Z/REPORT.md) · [`summary.json`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp2-dp16-ep32-fp8kv__20261103T044915Z/summary.json)

| Workload Suite (`Script Anchor`) | Input : Output | Prefix Hit | Temp | Conc | Output TPS | Total TPS | Mean Stream TPS | TTFT p50/p90 (ms) | TPOT p50/p90 (ms) | OK/Fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`concurrency_1024in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `1,024 : 1,024` | `0%` | `0.6` | `1` | `48.11` | `96.28` | `49.42` | `588.5 / 637.4` | `20.23 / 20.25` | `4/0` |
| [`concurrency_1024in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `1,024 : 1,024` | `0%` | `0.6` | `32` | `1,246.49` | `2,494.20` | `42.10` | `1,973.5 / 2,502.0` | `23.71 / 24.66` | `64/0` |
| [`concurrency_1024in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `1,024 : 1,024` | `0%` | `0.6` | `128` | `3,279.77` | `6,562.75` | `26.93` | `1,853.2 / 2,864.1` | `37.71 / 37.81` | `256/0` |
| [`concurrency_1024in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `1,024 : 1,024` | `0%` | `0.6` | `256` | **`5,267.09`** | **`10,539.33`** | `22.05` | `2,446.3 / 3,539.0` | `49.61 / 54.59` | `512/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0%` | `0.6` | `1` | `45.92` | `229.65` | `49.52` | `1,642.6 / 1,645.2` | `20.20 / 20.20` | `4/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0%` | `0.6` | `8` | `324.95` | `1,625.06` | `46.17` | `3,106.1 / 3,118.7` | `21.66 / 22.03` | `16/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0%` | `0.6` | `32` | `1,080.69` | `5,404.51` | `37.45` | `3,492.8 / 4,077.3` | `27.04 / 28.43` | `64/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0%` | `0.6` | `64` | `1,741.51` | `8,709.26` | `30.44` | `3,415.1 / 5,317.8` | `34.37 / 40.87` | `128/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0%` | `0.6` | `128` | `2,634.02` | `13,172.68` | `22.72` | `4,778.0 / 7,666.3` | `45.01 / 47.74` | `256/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0%` | `0.6` | `256` | **`3,812.96`** | **`19,068.51`** | `16.99` | `7,670.3 / 12,924.7` | `55.74 / 73.17` | `512/0` |
| [`cached_80pct_prefix` (`#L388`)](./scripts/run_tpu_benchmark_sweep.py#L388-L394) | `4,096 : 1,024` | `80%` | `0.6` | `32` | `1,175.91` | `5,880.25` | `41.08` | `2,703.6 / 4,029.5` | `24.26 / 25.23` | `64/0` |
| [`cached_95pct_prefix` (`#L388`)](./scripts/run_tpu_benchmark_sweep.py#L388-L394) | `4,096 : 1,024` | `95%` | `0.6` | `32` | **`1,201.96`** | **`6,010.50`** | `41.10` | `2,655.7 / 2,679.4` | `24.34 / 25.14` | `64/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `8,192 : 512` | `0%` | `0.6` | `4` | `120.70` | `2,052.13` | `45.28` | `5,652.5 / 6,153.2` | `22.27 / 23.37` | `8/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `32,768 : 512` | `0%` | `0.6` | `4` | `55.97` | `3,638.24` | `42.67` | `25,035.8 / 25,375.8` | `23.50 / 25.26` | `8/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `65,536 : 512` | `0%` | `0.6` | `4` | `27.54` | `3,552.50` | `37.67` | `57,562.8 / 67,898.4` | `22.80 / 40.26` | `8/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | **`131,072 : 512`** | `0%` | `0.6` | `4` | **`10.31`** | **`2,650.90`** | `30.85` | `154,470.0 / 217,482.7` | `24.65 / 143.43` | `8/0` |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (39.5) : 1,024` | `0%` | `0.0` | `1` | `49.89` | `51.82` | **`50.09`** | `65.0 / 214.3` | **`19.97 / 19.97`** | `4/0` |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (38.3) : 1,024` | `0%` | `0.0` | `8` | `367.04` | `380.77` | `46.13` | `117.4 / 176.0` | `21.68 / 21.70` | `16/0` |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (38.3) : 1,024` | `0%` | `0.0` | `32` | **`1,344.33`** | `1,394.63` | `42.43` | `250.9 / 317.5` | `23.56 / 23.72` | `64/0` |

### 3.3 `TP=4, DP=8, EP=32` — Mid-Concurrency Sweet Spot (`ml3-fp8-tp4-dp8-ep32`)

- Manifest: [`k8s/v7x-16-ml3-fp8-tp4-dp8-ep32.yaml`](./k8s/v7x-16-ml3-fp8-tp4-dp8-ep32.yaml) · Artifacts: [`REPORT.md`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp4-dp8-ep32__20261103T064257Z/REPORT.md) · [`summary.json`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp4-dp8-ep32__20261103T064257Z/summary.json)

| Workload Suite (`Script Anchor`) | Input : Output | Temp | Conc | Output TPS | Total TPS | Mean Stream TPS | TTFT p50/p90 (ms) | TPOT p50/p90 (ms) | OK/Fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `1` | `56.17` | `280.88` | **`61.94`** | `1,715.3 / 1,728.7` | **`16.15 / 16.15`** | `4/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `8` | `407.12` | `2,035.99` | `57.86` | `2,476.5 / 2,487.5` | `17.29 / 17.56` | `16/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `32` | `1,119.54` | `5,598.78` | `39.18` | `2,875.1 / 5,311.3` | `24.30 / 29.97` | `64/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | **`64`** | **`1,832.04`** | **`9,161.99`** | `32.42` | `4,059.9 / 6,915.9` | `31.92 / 34.08` | `128/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `128` | `2,405.95` | `12,032.08` | `22.36` | `7,212.1 / 12,595.5` | `41.27 / 61.21` | `256/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `256` | `3,128.35` | `15,644.79` | `14.84` | `12,207.7 / 24,902.1` | `70.08 / 82.96` | `512/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `65,536 : 512` | `0.6` | `4` | `39.29` | `5,067.92` | `49.12` | `40,923.1 / 44,367.6` | `18.57 / 23.52` | `8/0` |

### 3.4 `TP=8, DP=4, EP=32` — Ultra-Low-Latency & Fast Long-Context Prefill (`ml3-fp8-tp8-dp4-ep32`)

- Manifest: [`k8s/v7x-16-ml3-fp8-tp8-dp4-ep32.yaml`](./k8s/v7x-16-ml3-fp8-tp8-dp4-ep32.yaml) · Artifacts: [`REPORT.md`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp8-dp4-ep32__20261103T061631Z/REPORT.md) · [`summary.json`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512__ml3-fp8-tp8-dp4-ep32__20261103T061631Z/summary.json)

| Workload Suite (`Script Anchor`) | Input : Output | Temp | Conc | Output TPS | Total TPS | Mean Stream TPS | TTFT p50/p90 (ms) | TPOT p50/p90 (ms) | OK/Fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `1` | `70.54` | `352.77` | **`75.62`** | `909.9 / 1,230.8` | **`13.24 / 13.24`** | `4/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `8` | **`501.68`** | `2,508.89` | `66.40` | `1,096.4 / 1,235.1` | `15.26 / 15.53` | `16/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | **`32`** | **`1,324.28`** | **`6,622.70`** | `45.84` | `1,954.2 / 3,829.1` | `22.06 / 23.72` | `64/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `64` | `1,789.36` | `8,948.52` | `34.28` | `2,957.6 / 7,983.3` | `34.80 / 44.68` | `128/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `128` | `2,332.60` | `11,665.28` | `21.30` | `6,402.6 / 14,947.5` | `45.75 / 56.47` | `256/0` |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `256` | `3,072.43` | `15,365.17` | `16.49` | `12,117.6 / 31,079.3` | `60.90 / 78.33` | `512/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `32,768 : 512` | `0.6` | `4` | **`120.82`** | **`7,853.84`** | `65.68` | **`8,939.6 / 10,254.6`** | `14.41 / 16.92` | `8/0` |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `65,536 : 512` | `0.6` | `4` | **`71.09`** | **`9,170.54`** | `54.62` | **`14,941.0 / 27,067.0`** | `15.34 / 37.51` | `8/0` |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (39.5) : 1,024` | `0.0` | `1` | **`76.23`** | `79.17` | **`77.07`** | `195.2 / 198.2` | **`12.98 / 12.98`** | `4/0` |

---

## 4. Repository Structure & Reproduction

- [`docker/Dockerfile`](./docker/Dockerfile) — Reference image build (`vllm/vllm-tpu` base + benchmark tooling); not used for the measured runs
- [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) — Idempotent runtime patch: compressed-tensors FP8 MoE scale-name registration (default, used for all results); Eagle draft-model patches opt-in via `ML3_EAGLE_PATCHES=1`
- [`scripts/run_vllm_tpu_multihost_plan.py`](./scripts/run_vllm_tpu_multihost_plan.py) — Multi-host harness used for the measured runs: for each experiment, all hosts join a fresh Ray cluster, host 0 runs `vllm serve` (Ray executor) and, once `/v1/models` responds, the benchmark sweep; writes `boot_*.json` and `suite_status.json` next to the results
- [`scripts/plan_mistral_large3_v7x16.json`](./scripts/plan_mistral_large3_v7x16.json) — Executed plan: server flags and sweep for every configuration (`TP=2` base, `TP=1/4/8` derived from it, BF16 `TP=2`, Eagle experiment); experiments list fallback candidates, and the first candidate of every reported configuration booted
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) — Async OpenAI SSE streaming benchmark (`concurrency_<in>in`, `cached_<ratio>pct_prefix`, `context_sweep`, `greedy_natural`)
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel Hugging Face -> GCS stager
- [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml) — Benchmark Job used for the measured runs: 4-pod Indexed Job (one pod per `v7x` host) + headless Service
- [`k8s/`](./k8s/) `v7x-16-ml3-fp8-tp*-ep32.yaml` — Reference GKE LeaderWorkerSet + Benchmark Job manifests per topology (`TP=1`, `TP=2`, `TP=4`, `TP=8` with `EP=32`; same vLLM server flags as the plan); not used for the measured runs
- [`results/`](./results/) — Raw Markdown (`REPORT.md`) and JSON (`summary.json`) benchmark artifacts across all four topologies

```bash
# 1. Stage mistralai/Mistral-Large-3-675B-Instruct-2512 to GCS
python3 scripts/stage_hf_to_gcs.py --repos mistralai/Mistral-Large-3-675B-Instruct-2512 --bucket YOUR_GCS_BUCKET --prefix models

# 2. Scripts ConfigMap and Hugging Face token Secret for the benchmark Job
kubectl create configmap v7x16-ml3-scripts \
  --from-file=run_vllm_tpu_multihost_plan.py=scripts/run_vllm_tpu_multihost_plan.py \
  --from-file=plan.json=scripts/plan_mistral_large3_v7x16.json \
  --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py \
  --from-file=apply_ml3_patches.py=scripts/apply_ml3_patches.py
kubectl create secret generic hf-token --from-literal=HF_TOKEN=YOUR_HF_TOKEN

# 3. Run the plan on TPU v7x-16 (2x2x4); results are written under /models/benchmark-results/ (GCS FUSE mount)
kubectl apply -f k8s/v7x16-ml3-suite.yaml
```

The plan runs its experiments in sequence (about `15` minutes of server start-up per configuration, plus its sweep). To run a subset, add `SUITE_ONLY=<comma-separated experiment ids>` to the Job environment; the derived experiments (`ml3_fp8_tp1_dp32_ep32`, `ml3_fp8_tp4_dp8_ep32`, `ml3_fp8_tp8_dp4_ep32`) need their base `ml3_fp8_tp2_dp16_ep32` in the same run.

Reference deployment (not used for the measured runs): one LeaderWorkerSet + benchmark Job per topology. These manifests run `/workspace/apply_ml3_patches.py`, which must be present in the image (`docker/Dockerfile` copies it there).

```bash
kubectl create configmap tpu-bench-script \
  --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py \
  --from-file=apply_ml3_patches.py=scripts/apply_ml3_patches.py
kubectl apply -f k8s/v7x-16-ml3-fp8-tp1-dp32-ep32.yaml   # Max throughput (5,806 out tok/s, 37.19M KV tokens)
# OR: kubectl apply -f k8s/v7x-16-ml3-fp8-tp2-dp16-ep32.yaml # Intra-chip TP=2 + 128K context (5,267 out tok/s @ 1K:1K)
# OR: kubectl apply -f k8s/v7x-16-ml3-fp8-tp4-dp8-ep32.yaml  # Mid-concurrency c=64 sweet spot (16.15 ms TPOT)
# OR: kubectl apply -f k8s/v7x-16-ml3-fp8-tp8-dp4-ep32.yaml  # Ultra-low latency (12.98 ms TPOT, 14.94s 64K TTFT)
```

---

## 5. Planned Follow-up Measurements

- `TP=8, DP=4` with larger prefill chunks (`--max-num-batched-tokens=4096`, fallback `2048`) for long-context TTFT, including a `131,072`-token row.
- Eagle speculative decoding (`3` draft tokens) on `TP=8, DP=4` and `TP=2, DP=16` once the draft model runs on `vllm-torchtpu`, measured with the greedy `natural-chat` rows (acceptance rate, tok/s per user).
- n-gram speculative decoding on `TP=8, DP=4`.
- `vllm-torchtpu` fused expert-parallel MoE kernel (`MOE_FUSED_EP_KERNEL_IMPL=v2`) and W8A16 small-batch decode setting (`TPU_FP8_LINEAR_W8A16_MAX_ROWS=64`).
- `TP=1, DP=32` at `c=768` / `1,024` (output throughput was still rising at `c=512`).
- `131,072`-token rows with `16` requests per row.
