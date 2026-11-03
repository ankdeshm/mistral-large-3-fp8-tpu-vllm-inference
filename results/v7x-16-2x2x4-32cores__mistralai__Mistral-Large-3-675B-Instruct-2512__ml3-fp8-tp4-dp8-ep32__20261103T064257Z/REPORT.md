# vLLM-TorchTPU benchmark - `v7x-16-2x2x4-32cores` / `ml3-fp8-tp4-dp8-ep32`

- Timestamp (UTC): `20261103T064257Z`
- Served model: `mistralai/Mistral-Large-3-675B-Instruct-2512`
- Sanity check (1.717s): `Mixture-of-Experts (MoE) routing on TPUs dynamically selects a subset of specialized expert models (via a gating network) to process input tokens, optimizing computational efficiency. TPUs accelerate `

| Suite | In (mean actual) | Out | Prefix | Conc | Total out tok/s | Mean stream tok/s/user | TTFT p50/p90/p99 (ms) | TPOT p50/p90 (ms) | Accept rate / len | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 1 | **56.17** | **61.94** | 1715.3/1728.7/1728.7 | 16.15/16.15 | - | 4/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 8 | **407.12** | **57.86** | 2476.5/2487.5/2487.7 | 17.29/17.56 | - | 16/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 32 | **1119.54** | **39.18** | 2875.1/5311.3/5502.6 | 24.3/29.97 | - | 64/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 64 | **1832.04** | **32.42** | 4059.9/6915.9/8514.0 | 31.92/34.08 | - | 128/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 128 | **2405.95** | **22.36** | 7212.1/12595.5/14265.6 | 41.27/61.21 | - | 256/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 256 | **3128.35** | **14.84** | 12207.7/24902.1/30654.8 | 70.08/82.96 | - | 512/0 |
| context_sweep | 65536 (65537.0) | 512.0 | 0% | 4 | **39.29** | **49.12** | 40923.1/44367.6/44370.0 | 18.57/23.52 | - | 8/0 |
