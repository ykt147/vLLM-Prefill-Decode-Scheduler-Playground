| Strategy | TTFT P50 (ms) | TTFT P99 (ms) | ITL P50 (ms) | ITL P99 (ms) | Throughput (tok/s) | Timeouts |
|----------|---------------|---------------|--------------|--------------|--------------------|----------|
| vLLM V1 default | 184.8 | 3690.3 | 58.73 | 102.56 | 3368.9 | 0 |
| Strategy A (Prefill-First) | 58.1 | 120.3 | 342.54 | 1143.75 | 668.7 | 682 |
| Strategy B (Pure Decode-First) | 150960.5 | 292833.0 | 11.95 | 13.35 | 644.5 | 86 |
