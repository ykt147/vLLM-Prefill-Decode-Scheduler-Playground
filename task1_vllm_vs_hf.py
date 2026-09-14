
"""Task 1: Compare throughput of transformers (serial / static batching) vs vLLM offline.

Three schemes are benchmarked on 64 prompts (128-512 input tokens, 128 output tokens):
  1. transformers serial  — one-by-one generate()
  2. transformers static  — padded batch generate()
  3. vLLM offline         — continuous batching via LLM.generate()

Usage:
    pip install -r requirements.txt
    python task1_vllm_vs_hf.py
"""

import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use a free GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Constants
MODEL_PATH = "/public/home/ykt147/model/Qwen3_0.6b"
NUM_PROMPTS = 64
INPUT_LEN_MIN = 128
INPUT_LEN_MAX = 512
MAX_NEW_TOKENS = 128
SEED = 42

# Qwen3-0.6B config (verified from config.json)
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2  # FP16

per_token_KV_bytes = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES  # ≈ 112 KB


# Prompt generation
def generate_prompts(tokenizer, n=NUM_PROMPTS, seed=SEED):
    """Create n synthetic prompts whose tokenized length falls in [INPUT_LEN_MIN, INPUT_LEN_MAX]."""
    rng = random.Random(seed)
    vocab = list(tokenizer.get_vocab().keys())
    # Keep only printable single-token strings to get valid text
    vocab = [t for t in vocab if len(t) > 0 and t.isprintable()][:5000]

    prompts = []
    for _ in range(n):
        target_len = rng.randint(INPUT_LEN_MIN, INPUT_LEN_MAX)
        words = []
        total = 0
        while total < target_len:
            chunk = " ".join(rng.choices(vocab, k=8))
            ids = tokenizer(chunk, add_special_tokens=False).input_ids
            words.append(chunk)
            total += len(ids)
        # Trim to exact target
        prompt = " ".join(words)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids[:target_len]
        prompts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return prompts


# 1. transformers serial benchmark
def benchmark_transformers_serial(model, tokenizer, prompts):
    print("\n[1/3] transformers serial ...")
    device = model.device
    total_in = 0
    total_out = 0
    max_seq_len = 0

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    for i, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt").to(device)
        input_len = ids.input_ids.shape[1]
        total_in += input_len

        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_len = out.shape[1] - input_len
        total_out += gen_len
        max_seq_len = max(max_seq_len, input_len + gen_len)

        if (i + 1) % 16 == 0:
            print(f"  done {i+1}/{len(prompts)}")

    elapsed = time.perf_counter() - t0
    peak_kv = max_seq_len * per_token_KV_bytes
    throughput = total_out / elapsed

    print(f"  time={elapsed:.2f}s  throughput={throughput:.1f} tok/s  peak_kv≈{peak_kv/1024**2:.1f}MB")
    return {
        "name": "transformers 串行",
        "time": elapsed,
        "throughput": throughput,
        "peak_kv_mb": peak_kv / 1024**2,
        "note": f"b=1, max_seq={max_seq_len}",
    }


# 2. transformers static batching benchmark
def benchmark_transformers_static(model, tokenizer, prompts):
    print("\n[2/3] transformers static batching ...")
    device = model.device

    # Encode all prompts
    encoded = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
    sorted_lens = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))

    best_result = None

    for batch_size in [len(prompts), len(prompts)//2, len(prompts)//4, 8, 4, 2]:
        if batch_size < 1:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            batch_ids = [encoded[i] for i in sorted_lens[:batch_size]]
            padded = tokenizer.pad(
                {"input_ids": batch_ids},
                padding=True,
                return_tensors="pt",
            ).to(device)
            L = padded.input_ids.shape[1]  # padded max length

            t0 = time.perf_counter()
            out = model.generate(**padded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            elapsed = time.perf_counter() - t0

            peak_kv = batch_size * L * per_token_KV_bytes
            # out shape: (batch_size, total_seq_len) — all rows padded to same length
            actual_out = sum(out.shape[1] - len(ids) for ids in batch_ids)
            throughput = actual_out / elapsed

            note = f"batch={batch_size}, padded_len={L}"
            print(f"  batch={batch_size}  time={elapsed:.2f}s  throughput={throughput:.1f} tok/s  peak_kv≈{peak_kv/1024**2:.1f}MB")

            result = {
                "name": f"transformers 静态 batching",
                "time": elapsed,
                "throughput": throughput,
                "peak_kv_mb": peak_kv / 1024**2,
                "note": note,
            }

            if batch_size == len(prompts):
                return result
            best_result = result

        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch_size} OOM, trying smaller batch ...")
            torch.cuda.empty_cache()
            continue

    return best_result or {"name": "transformers 静态 batching", "time": float("inf"),
                           "throughput": 0, "peak_kv_mb": 0, "note": "OOM even at batch=2"}


# 3. vLLM offline benchmark
def benchmark_vllm(prompts):
    print("\n[3/3] vLLM offline ...")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.9,
        max_model_len=1024,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    # Total generated output tokens
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_out / elapsed

    # Read KV cache metrics from vLLM's built-in Prometheus metrics
    # The metric vllm:kv_cache_usage_perc gives per-layer usage;
    # we compute peak KV from the scheduler's perspective:
    # total_tokens_in_kv_cache * per_token_KV_bytes
    peak_kv_mb = None
    note = ""
    try:
        metrics = llm.get_metrics()
        for m in metrics:
            name = m.name if hasattr(m, "name") else ""
            if "kv_cache" in name.lower() and hasattr(m, "samples"):
                # Aggregate KV cache usage from vLLM metrics
                for sample in m.samples or []:
                    val = sample.value if hasattr(sample, "value") else 0
                    if "peak" in name.lower():
                        peak_kv_mb = val / 1024  # vLLM reports in MB
                        note = f"peak_computed from metrics"
                        break
    except Exception as e:
        print(f"  (get_metrics unavailable: {e}, falling back to estimate)")

    # Fallback: estimate KV from sum(input+output) tokens per request
    if peak_kv_mb is None:
        tok = llm.get_tokenizer()
        total_tokens = 0
        for req, out in zip(prompts, outputs):
            inp_len = len(tok.encode(req, add_special_tokens=False))
            out_len = len(out.outputs[0].token_ids)
            total_tokens += inp_len + out_len
        peak_kv_mb = total_tokens * per_token_KV_bytes / 1024**2
        note = f"total_tokens_in_cache={total_tokens} (estimated)"

    print(f"  time={elapsed:.2f}s  throughput={throughput:.1f} tok/s  peak_kv≈{peak_kv_mb:.1f}MB")

    return {
        "name": "vLLM 离线",
        "time": elapsed,
        "throughput": throughput,
        "peak_kv_mb": peak_kv_mb,
        "note": note,
    }


# Report helpers
def print_table(results):
    header = f"{'方案':<30} {'总耗时 (s)':>12} {'Throughput (tok/s)':>20} {'峰值 KV (MB)':>14}  备注"
    sep = "-" * 110
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['name']:<30} {r['time']:>12.2f} {r['throughput']:>20.1f} {r['peak_kv_mb']:>14.1f}  {r['note']}")
    print(sep)

    # Speedup vs serial
    serial_t = results[0]["throughput"]
    if serial_t > 0:
        for r in results[1:]:
            if r["throughput"] > 0:
                print(f"  {r['name']} 相对 transformers 串行 吞吐提升: {r['throughput']/serial_t:.2f}x")


def save_results(results, path="task1_results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {path}")


# Main
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Model: {MODEL_PATH}")
    print(f"per_token_KV_bytes = {per_token_KV_bytes}  ({per_token_KV_bytes/1024:.1f} KB)")
    print(f"Prompts: {NUM_PROMPTS}  (input {INPUT_LEN_MIN}-{INPUT_LEN_MAX} tok, output {MAX_NEW_TOKENS} tok)")

    # Load tokenizer first to generate prompts
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"  # causal model needs left padding
    prompts = generate_prompts(tokenizer)
    print(f"Generated {len(prompts)} prompts (avg len={np.mean([len(tokenizer.encode(p)) for p in prompts]):.0f})")

    results = []

    # ── transformers benchmarks ──
    print("\nLoading model (transformers) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    r1 = benchmark_transformers_serial(model, tokenizer, prompts)
    results.append(r1)

    r2 = benchmark_transformers_static(model, tokenizer, prompts)
    results.append(r2)

    del model
    torch.cuda.empty_cache()

    # ── vLLM benchmark ──
    r3 = benchmark_vllm(prompts)
    results.append(r3)

    # ── Summary ──
    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
