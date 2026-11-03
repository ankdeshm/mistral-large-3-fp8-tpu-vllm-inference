# vLLM-TorchTPU benchmark - `v7x-16-2x2x4-32cores` / `ml3-fp8-tp1-dp32-ep32`

- Timestamp (UTC): `20261103T054109Z`
- Served model: `mistralai/Mistral-Large-3-675B-Instruct-2512`
- Sanity check (2.269s): `Mixture-of-Experts (MoE) routing on TPUs dynamically selects a subset of specialized expert models (via a gating network) to process each input, optimizing computational efficiency. TPUs accelerate th`

| Suite | In (mean actual) | Out | Prefix | Conc | Total out tok/s | Mean stream tok/s/user | TTFT p50/p90/p99 (ms) | TPOT p50/p90 (ms) | Accept rate / len | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 1 | **34.78** | **39.05** | 3242.9/3260.1/3260.1 | 25.61/25.62 | - | 4/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 8 | **256.72** | **37.44** | 4746.7/4810.3/4810.8 | 26.55/27.34 | - | 16/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 32 | **989.39** | **35.94** | 4345.9/5125.6/5127.6 | 27.97/28.18 | - | 64/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 64 | **1764.58** | **30.51** | 3818.0/6477.0/8022.7 | 31.99/35.19 | - | 128/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 128 | **2875.65** | **24.83** | 4984.8/7229.0/9803.5 | 39.31/45.54 | - | 256/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 256 | **4389.15** | **19.2** | 5610.8/10292.3/13317.3 | 54.61/57.7 | - | 512/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 512 | **5806.06** | **12.74** | 7648.6/15716.8/24406.2 | 76.76/99.42 | - | 1024/0 |
| context_sweep | 32768 (32769.0) | 512.0 | 0% | 4 | **38.88** | **34.39** | 38098.1/39029.4/39054.9 | 29.8/32.86 | - | 8/0 |
| context_sweep | 65536 (65537.0) | 512.0 | 0% | 4 | **16.48** | **33.6** | 110013.3/110057.7/110059.0 | 27.9/37.36 | - | 8/0 |
| greedy_natural | natural-chat (39.5) | 1024.0 | 0% | 1 | **39.31** | **39.43** | 86.4/157.2/157.2 | 25.37/25.37 | - | 4/0 |
