#!/usr/bin/env python3
"""Custom benchmark script with per-request timeout support.

Sends requests to vLLM OpenAI API with individual timeouts,
collects TTFT, ITL, and e2e latency metrics, and saves results.
"""

import argparse
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REQUEST_TIMEOUT = 120.0  # seconds - requests exceeding this are counted as timeouts


def generate_random_dataset(num_prompts: int, input_len: int, output_len: int, seed: int):
    """Generate random prompts matching vllm bench serve format."""
    random.seed(seed)
    # Use a pool of random token IDs to generate prompts
    vocab_start = 100
    vocab_end = 50000
    prompts = []
    for _ in range(num_prompts):
        token_ids = [random.randint(vocab_start, vocab_end) for _ in range(input_len)]
        prompts.append(token_ids)
    return prompts


class RequestMetrics:
    """Track timing metrics for a single request."""
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.submit_time: float = 0
        self.ttft: float = 0  # time to first token
        self.itls: list[float] = []  # inter-token latencies
        self.e2e_latency: float = 0
        self.output_tokens: int = 0
        self.timed_out: bool = False
        self.error: str = ""


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    request_id: str,
    scheduled_time: float,
    timeout: float,
) -> RequestMetrics:
    """Send a single completion request with timeout."""
    metrics = RequestMetrics(request_id)

    # Wait until scheduled time (Poisson arrival)
    delay = scheduled_time - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)

    metrics.submit_time = time.monotonic()

    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": prompt_token_ids,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                metrics.error = f"HTTP {response.status}"
                metrics.e2e_latency = time.monotonic() - metrics.submit_time
                return metrics

            first_token = True
            last_token_time = metrics.submit_time

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                if line == "data: [DONE]":
                    break

                try:
                    data = json.loads(line[5:])
                    token = data.get("choices", [{}])[0].get("text", "")
                    if token:
                        now = time.monotonic()
                        if first_token:
                            metrics.ttft = now - metrics.submit_time
                            first_token = False
                        else:
                            metrics.itls.append(now - last_token_time)
                        last_token_time = now
                        metrics.output_tokens += 1
                except json.JSONDecodeError:
                    pass

            metrics.e2e_latency = time.monotonic() - metrics.submit_time

    except asyncio.TimeoutError:
        metrics.timed_out = True
        metrics.error = "timeout"
        metrics.e2e_latency = time.monotonic() - metrics.submit_time
    except Exception as e:
        metrics.error = str(e)
        metrics.e2e_latency = time.monotonic() - metrics.submit_time

    return metrics


def percentile(values: list[float], p: float) -> float:
    """Compute percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


async def run_benchmark(
    base_url: str,
    model_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    request_rate: float,
    seed: int,
    timeout: float,
):
    """Run the benchmark and return metrics."""
    dataset = generate_random_dataset(num_prompts, input_len, output_len, seed)

    # Schedule request arrival times (Poisson process)
    scheduled_times = []
    current_time = time.monotonic()
    for i in range(num_prompts):
        current_time += random.expovariate(request_rate)
        scheduled_times.append(current_time)

    # Add a small jitter to avoid exact synchronization
    random.seed(seed + 1)

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_prompts):
            task = asyncio.create_task(
                send_request(
                    session=session,
                    base_url=base_url,
                    model_name=model_name,
                    prompt_token_ids=dataset[i],
                    max_tokens=output_len,
                    request_id=f"custom-{i}",
                    scheduled_time=scheduled_times[i],
                    timeout=timeout,
                )
            )
            tasks.append(task)

        logger.info(f"Launched {num_prompts} requests, waiting for completion...")
        results = await asyncio.gather(*tasks)

    return results


def compute_and_save_results(
    metrics_list: list[RequestMetrics],
    output_path: Path,
    duration: float,
    request_rate: float,
):
    """Compute summary statistics and save to JSON matching vLLM bench format."""
    ttfts = []
    all_itls = []
    e2e_latencies = []
    total_output_tokens = 0
    total_input_tokens = 0
    timeouts = 0
    errors_list = []

    for m in metrics_list:
        if m.timed_out:
            timeouts += 1
            errors_list.append("timeout")
            e2e_latencies.append(m.e2e_latency)
            continue

        if m.error:
            errors_list.append(m.error)
            e2e_latencies.append(m.e2e_latency)
            continue

        errors_list.append("")
        ttfts.append(m.ttft)
        all_itls.append(m.itls)  # list per request
        e2e_latencies.append(m.e2e_latency)
        total_output_tokens += m.output_tokens
        total_input_tokens += 1024  # fixed input length

    # Flatten ITL for summary
    flat_itls = []
    for req_itls in all_itls:
        flat_itls.extend(req_itls)

    # Compute summary
    completed = len(metrics_list) - timeouts

    # Build result matching vLLM bench serve JSON structure
    result = {
        "date": datetime.now().isoformat(),
        "endpoint_type": "openai",
        "model_id": "custom-benchmark",
        "num_prompts": len(metrics_list),
        "request_rate": request_rate,
        "duration": duration,
        "completed": completed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_token_throughput": total_output_tokens / max(duration, 0.001),
        # TTFT
        "ttfts": ttfts,
        "median_ttft_ms": percentile(ttfts, 50) * 1000 if ttfts else 0,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000 if ttfts else 0,
        "mean_ttft_ms": (sum(ttfts) / len(ttfts) * 1000) if ttfts else 0,
        # ITL (per-step)
        "itls": all_itls,
        "median_itl_ms": percentile(flat_itls, 50) * 1000 if flat_itls else 0,
        "p99_itl_ms": percentile(flat_itls, 99) * 1000 if flat_itls else 0,
        "mean_itl_ms": (sum(flat_itls) / len(flat_itls) * 1000) if flat_itls else 0,
        # E2E
        "e2e_latencies": e2e_latencies,
        "errors": errors_list,
        "timeouts": timeouts,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("=" * 50)
    print(" Custom Benchmark Result")
    print("=" * 50)
    print(f"  Successful requests:   {completed}")
    print(f"  Timed out requests:    {timeouts}")
    print(f"  Duration (s):          {duration:.2f}")
    print(f"  Total output tokens:   {total_output_tokens}")
    print(f"  Token throughput:      {total_output_tokens / max(duration, 0.001):.1f} tok/s")
    if ttfts:
        print(f"  Median TTFT (ms):    {percentile(ttfts, 50) * 1000:.2f}")
        print(f"  P99 TTFT (ms):       {percentile(ttfts, 99) * 1000:.2f}")
    if flat_itls:
        print(f"  Median ITL (ms):     {percentile(flat_itls, 50) * 1000:.2f}")
        print(f"  P99 ITL (ms):        {percentile(flat_itls, 99) * 1000:.2f}")
    print("=" * 50)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name for the API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=256)
    parser.add_argument("--request-rate", type=float, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT,
                        help="Per-request timeout in seconds")
    parser.add_argument("--result-dir", type=str, default=".")
    parser.add_argument("--result-filename", type=str, default="result.json")
    args = parser.parse_args()

    start_time = time.monotonic()

    metrics_list = asyncio.run(run_benchmark(
        base_url=args.base_url,
        model_name=args.model,
        num_prompts=args.num_prompts,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
        request_rate=args.request_rate,
        seed=args.seed,
        timeout=args.timeout,
    ))

    duration = time.monotonic() - start_time

    output_path = Path(args.result_dir) / args.result_filename
    compute_and_save_results(metrics_list, output_path, duration, args.request_rate)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
