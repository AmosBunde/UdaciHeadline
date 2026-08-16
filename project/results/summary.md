| Optimization Technique | Mean Latency (s) | P99 Latency (s) | Throughput (tokens/s) | Peak memory (GB) | ROUGE-1 | ROUGE-L | speedup vs baseline |
|---|---|---|---|---|---|---|---|
| Baseline (No Cache) | 58.388 | 109.534 | 0.204 | 6.200 | 0.154 | 0.145 | 1.00x |
| KV Caching | 6.504 | 9.064 | 1.830 | 6.264 | 0.154 | 0.145 | 8.98x |
| Pruning (30%) | 7.148 | 9.754 | 1.958 | 7.273 | 0.111 | 0.106 | 8.17x |
| Quantization (8-bit) | 18.236 | 29.962 | 0.674 | 3.921 | 0.172 | 0.164 | 3.20x |
| Quantization (4-bit NF4) | 13.415 | 20.170 | 0.939 | 3.913 | 0.123 | 0.123 | 4.35x |
| Tensor Parallelism (simulated) | 6.573 | 9.967 | 1.810 | 7.873 | 0.154 | 0.145 | 8.88x |
| Pipeline Parallelism (simulated) | 6.570 | 9.584 | 1.811 | 7.830 | 0.154 | 0.145 | 8.89x |
| 3B target alone | 49.000 | 54.582 | 0.230 | 9.440 | 0.105 | 0.105 | 1.19x |
| Speculative Decoding (3B + 1B draft) | 67.829 | 78.430 | 0.168 | 11.904 | 0.102 | 0.102 | 0.86x |