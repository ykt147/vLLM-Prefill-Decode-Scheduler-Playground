#!/usr/bin/env python3
"""Task 3: 分析三种调度策略的压测结果，生成对比表和 CDF 图。"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})

# Keys match the JSON file naming
STRATEGY_FILES = {
    "vLLM V1 default": "result_vLLM_V1_default.json",
    "Strategy A (Prefill-First)": "result_Strategy_A_PrefillFirst.json",
    "Strategy B (Pure Decode-First)": "result_Strategy_B_PureDecode.json",
}

STRATEGY_COLORS = {
    "vLLM V1 default": "#1f77b4",
    "Strategy A (Prefill-First)": "#d62728",
    "Strategy B3 (Adaptive Chunk)": "#2ca02c",
}


def compute_cdf(values, num_points=200):
    """Compute CDF from a list of values. Returns (x, y) arrays."""
    if not values:
        return np.array([]), np.array([])
    sorted_vals = np.sort(np.array(values))
    percentiles = np.linspace(0, 100, num_points)
    x = np.percentile(sorted_vals, percentiles)
    y = percentiles / 100.0
    return x, y


def extract_metrics(result_path: Path):
    """Extract summary and detailed metrics from a vLLM bench result JSON."""
    with open(result_path) as f:
        data = json.load(f)

    ttft_p50 = data.get("median_ttft_ms", 0)
    ttft_p99 = data.get("p99_ttft_ms", 0)
    itl_p50 = data.get("median_itl_ms", 0)
    itl_p99 = data.get("p99_itl_ms", 0)
    throughput = data.get("total_token_throughput", 0)

    ttfts_sec = data.get("ttfts", [])
    itls_sec = data.get("itls", [])
    errors = data.get("errors", [])

    all_itls = []
    for req_itls in itls_sec:
        if isinstance(req_itls, list):
            all_itls.extend(req_itls)
        elif isinstance(req_itls, (int, float)):
            all_itls.append(req_itls)

    ttfts_ms = [t * 1000 for t in ttfts_sec]
    itls_ms = [t * 1000 for t in all_itls]

    num_prompts = data.get("num_prompts", 0)
    completed = data.get("completed", 0)
    timeouts = num_prompts - completed
    error_count = sum(1 for e in errors if e and len(str(e).strip()) > 0)
    timeouts = max(timeouts, error_count)

    return {
        "ttft_p50_ms": ttft_p50,
        "ttft_p99_ms": ttft_p99,
        "itl_p50_ms": itl_p50,
        "itl_p99_ms": itl_p99,
        "throughput": throughput,
        "timeouts": timeouts,
        "ttfts_ms": ttfts_ms,
        "itls_ms": itls_ms,
        "num_prompts": num_prompts,
        "completed": completed,
    }


def plot_cdf(all_cdfs, metric_name, output_path, xlim_max=None):
    """Plot CDF curves for multiple strategies."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    for label, (x, y) in all_cdfs.items():
        if len(x) == 0:
            continue
        ax.plot(x, y, label=label, linewidth=2, color=STRATEGY_COLORS.get(label, "#333333"))

    ax.set_xlabel(metric_name + " (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{metric_name} CDF")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    if xlim_max:
        ax.set_xlim(right=xlim_max)
    ax.set_ylim(bottom=0, top=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python task3_plot.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        sys.exit(1)

    print("=" * 50)
    print(" Task 3 Result Analysis")
    print("=" * 50)

    all_metrics = {}
    for label, filename in STRATEGY_FILES.items():
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"  [WARN] file not found: {filepath}")
            continue
        print(f"\n  Loading: {label} <- {filename}")
        all_metrics[label] = extract_metrics(filepath)

    if not all_metrics:
        print("ERROR: no result files found")
        sys.exit(1)

    # ---- Print comparison table ----
    print("\n" + "=" * 50)
    print(" Comparison Table")
    print("=" * 50)
    header = f"{'Strategy':<30} {'TTFT P50(ms)':>12} {'TTFT P99(ms)':>13} {'ITL P50(ms)':>12} {'ITL P99(ms)':>12} {'Throughput':>12} {'Timeouts':>8}"
    print(header)
    print("-" * len(header))
    for label, m in all_metrics.items():
        print(f"{label:<30} {m['ttft_p50_ms']:>12.1f} {m['ttft_p99_ms']:>13.1f} {m['itl_p50_ms']:>12.2f} {m['itl_p99_ms']:>12.2f} {m['throughput']:>12.1f} {m['timeouts']:>8}")

    # ---- Save table to file ----
    table_path = results_dir / "comparison_table.md"
    with open(table_path, "w") as f:
        f.write("| Strategy | TTFT P50 (ms) | TTFT P99 (ms) | ITL P50 (ms) | ITL P99 (ms) | Throughput (tok/s) | Timeouts |\n")
        f.write("|----------|---------------|---------------|--------------|--------------|--------------------|----------|\n")
        for label, m in all_metrics.items():
            f.write(f"| {label} | {m['ttft_p50_ms']:.1f} | {m['ttft_p99_ms']:.1f} | {m['itl_p50_ms']:.2f} | {m['itl_p99_ms']:.2f} | {m['throughput']:.1f} | {m['timeouts']} |\n")
    print(f"\n  Saved table: {table_path}")

    # ---- Plot CDFs ----
    print("\n" + "=" * 50)
    print(" Plotting CDFs")
    print("=" * 50)

    ttft_cdfs = {}
    for label, m in all_metrics.items():
        if m["ttfts_ms"]:
            x, y = compute_cdf(m["ttfts_ms"])
            ttft_cdfs[label] = (x, y)
    # For TTFT, cap x-axis at a reasonable value to see non-timeout behavior
    plot_cdf(ttft_cdfs, "TTFT", results_dir / "figure_3A_ttft_cdf.png", xlim_max=500)

    itl_cdfs = {}
    for label, m in all_metrics.items():
        if m["itls_ms"]:
            x, y = compute_cdf(m["itls_ms"])
            itl_cdfs[label] = (x, y)
    plot_cdf(itl_cdfs, "ITL", results_dir / "figure_3A_itl_cdf.png", xlim_max=100)

    # Combined CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for label, (x, y) in ttft_cdfs.items():
        ax1.plot(x, y, label=label, linewidth=2, color=STRATEGY_COLORS.get(label, "#333333"))
    ax1.set_xlabel("TTFT (ms)")
    ax1.set_ylabel("CDF")
    ax1.set_title("TTFT CDF")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0, right=500)
    ax1.set_ylim(bottom=0, top=1.0)

    for label, (x, y) in itl_cdfs.items():
        ax2.plot(x, y, label=label, linewidth=2, color=STRATEGY_COLORS.get(label, "#333333"))
    ax2.set_xlabel("ITL (ms)")
    ax2.set_ylabel("CDF")
    ax2.set_title("ITL CDF")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0, right=100)
    ax2.set_ylim(bottom=0, top=1.0)

    plt.tight_layout()
    plt.savefig(results_dir / "figure_3A_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {results_dir / 'figure_3A_combined.png'}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print(" Summary")
    print("=" * 50)

    default_m = all_metrics.get("vLLM V1 default")
    a_m = all_metrics.get("Strategy A (Prefill-First)")
    b_m = all_metrics.get("Strategy B (Pure Decode-First)")

    if default_m and a_m:
        ttft_improve = (default_m["ttft_p50_ms"] - a_m["ttft_p50_ms"]) / max(default_m["ttft_p50_ms"], 1) * 100
        itl_worsen = a_m["itl_p50_ms"] / max(default_m["itl_p50_ms"], 0.001)
        print(f"\n  Strategy A vs V1 default:")
        print(f"    TTFT P50 change: {ttft_improve:+.1f}%")
        print(f"    ITL P50: {a_m['itl_p50_ms']:.2f} vs {default_m['itl_p50_ms']:.2f} ms ({itl_worsen:.2f}x)")
        print(f"    Timeouts: {a_m['timeouts']} / {a_m['num_prompts']}")
        print(f"    Throughput: {a_m['throughput']:.1f} vs {default_m['throughput']:.1f} tok/s")

    if default_m and b_m:
        ttft_diff = (b_m["ttft_p50_ms"] - default_m["ttft_p50_ms"]) / max(default_m["ttft_p50_ms"], 1) * 100
        itl_diff = (b_m["itl_p50_ms"] - default_m["itl_p50_ms"]) / max(default_m["itl_p50_ms"], 1) * 100
        print(f"\n  Strategy B vs V1 default:")
        print(f"    TTFT P50 change: {ttft_diff:+.1f}%")
        print(f"    ITL P50 change: {itl_diff:+.1f}%")
        print(f"    Timeouts: {b_m['timeouts']} / {b_m['num_prompts']}")
        print(f"    Throughput: {b_m['throughput']:.1f} vs {default_m['throughput']:.1f} tok/s")


if __name__ == "__main__":
    main()
