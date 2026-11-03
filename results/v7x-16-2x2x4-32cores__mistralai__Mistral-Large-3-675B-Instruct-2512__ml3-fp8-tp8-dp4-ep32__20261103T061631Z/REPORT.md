# vLLM-TorchTPU benchmark - `v7x-16-2x2x4-32cores` / `ml3-fp8-tp8-dp4-ep32`

- Timestamp (UTC): `20261103T061631Z`
- Served model: `mistralai/Mistral-Large-3-675B-Instruct-2512`
- Sanity check (1.518s): `Mixture-of-Experts (MoE) routing on TPUs dynamically selects a subset of specialized expert models (via a gating network) for each input, optimizing computational efficiency. TPUs accelerate this proc`

| Suite | In (mean actual) | Out | Prefix | Conc | Total out tok/s | Mean stream tok/s/user | TTFT p50/p90/p99 (ms) | TPOT p50/p90 (ms) | Accept rate / len | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 1 | **70.54** | **75.62** | 909.9/1230.8/1230.8 | 13.24/13.24 | - | 4/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 8 | **501.68** | **66.4** | 1096.4/1235.1/1300.0 | 15.26/15.53 | - | 16/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 32 | **1324.28** | **45.84** | 1954.2/3829.1/5208.0 | 22.06/23.72 | - | 64/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 64 | **1789.36** | **34.28** | 2957.6/7983.3/10582.5 | 34.8/44.68 | - | 128/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 128 | **2332.6** | **21.3** | 6402.6/14947.5/18371.7 | 45.75/56.47 | - | 256/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 256 | **3072.43** | **16.49** | 12117.6/31079.3/37099.4 | 60.9/78.33 | - | 512/0 |
| context_sweep | 32768 (32769.0) | 512.0 | 0% | 4 | **120.82** | **65.68** | 8939.6/10254.6/10255.6 | 14.41/16.92 | - | 8/0 |
| context_sweep | 65536 (65537.0) | 512.0 | 0% | 4 | **71.09** | **54.62** | 14941.0/27067.0/27069.1 | 15.34/37.51 | - | 8/0 |
| greedy_natural | natural-chat (39.5) | 1024.0 | 0% | 1 | **76.23** | **77.07** | 195.2/198.2/198.2 | 12.98/12.98 | - | 4/0 |
