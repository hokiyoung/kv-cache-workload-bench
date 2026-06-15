# kv-cache-workload-bench
A ShareGPT-based multi-scenario workload generator and benchmark suite for evaluating LLM KV Cache allocation, prefix reuse, long-context paging, cache eviction, concurrent inference, session growth, and CPU/SSD/remote KV offloading.

# Usage
Generate:
```bash
python kv_workload_builder.py --sharegpt-path ../ --tokenizer Qwen/Qwen2.5-7B-Instruct --out ./kv_workloads --per-scenario 200 --max-model-len 16384 --target-long-tokens 4096,8192,12000,16000
```


