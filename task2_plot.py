#!/usr/bin/env python3
"""
Task 2：解析 vllm bench serve 结果 JSON，计算指标、绘制图表。

用法：python task2_plot.py <results_dir>

期望的 JSON 文件命名：run_<max_nbt>_<run_idx>.json
由 task2.sh 生成。
"""

import json
import sys
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --------------- 配置 ---------------
SCAN_VALUES = [512, 1024, 2048, 4096, 8192]
RUNS_PER_CONFIG = 2
WARMUP_DISCARD_RATIO = 0.10  # 丢弃前 10% 的请求数据

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "task2_results"
RESULTS_PATH = Path(RESULTS_DIR)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def find_run_files(max_nbt):
    """查找所有匹配 run_<max_nbt>_<idx>.json 的文件"""
    pattern = re.compile(rf"^run_{max_nbt}_\d+\.json$")
    files = []
    for f in RESULTS_PATH.iterdir():
        if pattern.match(f.name):
            files.append(f)
    return sorted(files)


def parse_result(filepath):
    """
    解析 vllm bench serve 输出的 JSON。
    返回 {ttfts, itls, e2e_latencies, throughput}（均用秒）
    """
    with open(filepath) as f:
        data = json.load(f)

    result = {}

    # ttfts: flat list of floats (one per request)
    if "ttfts" in data:
        result["ttfts"] = np.array(data["ttfts"], dtype=float)
    elif "metric_details" in data and "ttft" in data["metric_details"]:
        td = data["metric_details"]["ttft"]
        if isinstance(td, list):
            result["ttfts"] = np.array(td, dtype=float)

    # itls: list of lists (per-request ITL arrays, varying lengths). Flatten them.
    if "itls" in data:
        itl_raw = data["itls"]
        if isinstance(itl_raw, list) and len(itl_raw) > 0 and isinstance(itl_raw[0], list):
            flat_itls = []
            for sub in itl_raw:
                if isinstance(sub, list):
                    flat_itls.extend([float(v) for v in sub if v is not None])
            if flat_itls:
                result["itls"] = np.array(flat_itls, dtype=float)
        else:
            result["itls"] = np.array(itl_raw, dtype=float)
    elif "metric_details" in data and "itl" in data["metric_details"]:
        td = data["metric_details"]["itl"]
        if isinstance(td, list):
            result["itls"] = np.array(td, dtype=float)

    # e2e latencies
    if "e2e_latencies" in data:
        result["e2e_latencies"] = np.array(data["e2e_latencies"], dtype=float)
    elif "metric_details" in data and "e2e_latency" in data["metric_details"]:
        td = data["metric_details"]["e2e_latency"]
        if isinstance(td, list):
            result["e2e_latencies"] = np.array(td, dtype=float)

    # throughput: use total_token_throughput (output + input token throughput)
    for key in ["total_token_throughput", "throughput", "request_throughput"]:
        if key in data and data[key] is not None:
            result[key] = float(data[key])
            break

    return result


def discard_warmup(arr, ratio=WARMUP_DISCARD_RATIO):
    """丢弃前 ratio 比例的数据点"""
    if arr is None or len(arr) == 0:
        return arr
    n_discard = int(len(arr) * ratio)
    return arr[n_discard:]


def compute_percentile(arr, pct):
    if arr is None or len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, pct))


def compute_metrics_per_run(data):
    """从单次运行的数据中计算指标（丢弃 warmup 后）"""
    metrics = {}

    # TTFT: one value per request — discard first 10% requests
    if "ttfts" in data and len(data["ttfts"]) > 0:
        ttfts = discard_warmup(data["ttfts"])
        metrics["ttft_p50"] = compute_percentile(ttfts, 50)
        metrics["ttft_p99"] = compute_percentile(ttfts, 99)

    # ITL: already flattened from all requests — discard first 10% of values as warmup proxy
    if "itls" in data and len(data["itls"]) > 0:
        itls = discard_warmup(data["itls"])
        metrics["itl_p50"] = compute_percentile(itls, 50)
        metrics["itl_p99"] = compute_percentile(itls, 99)

    # Throughput: use total_token_throughput (the "Total Token throughput" from benchmark output)
    if "total_token_throughput" in data:
        metrics["throughput"] = data["total_token_throughput"]
    elif "throughput" in data:
        metrics["throughput"] = data["throughput"]

    return metrics


def aggregate_runs(run_metrics_list):
    """对同一配置的多次运行取中位数"""
    agg = {}
    keys = ["ttft_p50", "ttft_p99", "itl_p50", "itl_p99", "throughput"]

    for key in keys:
        values = [m[key] for m in run_metrics_list if key in m and not np.isnan(m[key])]
        if values:
            agg[key] = float(np.median(values))
        else:
            agg[key] = float("nan")

    return agg


def collect_all_results():
    """收集所有扫描配置的结果"""
    all_results = {}

    for max_nbt in SCAN_VALUES:
        files = find_run_files(max_nbt)
        if not files:
            print(f"[WARN] 未找到 {max_nbt} 的结果文件")
            continue

        print(f"[{max_nbt}] 找到 {len(files)} 个运行文件: {[f.name for f in files]}")

        run_metrics = []
        for f in files:
            try:
                raw = parse_result(f)
                m = compute_metrics_per_run(raw)
                if m:
                    run_metrics.append(m)
                    print(f"  {f.name}: ttft_p50={m.get('ttft_p50', 'N/A'):.4f}s, "
                          f"itl_p50={m.get('itl_p50', 'N/A'):.4f}s, "
                          f"throughput={m.get('throughput', 'N/A'):.2f}")
            except Exception as e:
                print(f"  [ERROR] 解析 {f.name} 失败: {e}")

        if run_metrics:
            agg = aggregate_runs(run_metrics)
            all_results[max_nbt] = agg
            print(f"  => 中位数: ttft_p50={agg['ttft_p50']*1000:.1f}ms, "
                  f"ttft_p99={agg['ttft_p99']*1000:.1f}ms, "
                  f"itl_p50={agg['itl_p50']*1000:.2f}ms, "
                  f"itl_p99={agg['itl_p99']*1000:.2f}ms, "
                  f"throughput={agg['throughput']:.2f} tok/s")
        else:
            print(f"  [WARN] {max_nbt} 没有有效数据")

    return all_results


def print_table(all_results):
    """打印结果表格"""
    header = f"{'max_num_batched_tokens':>24} | {'TTFT P50 (ms)':>14} | {'TTFT P99 (ms)':>14} | {'ITL P50 (ms)':>12} | {'ITL P99 (ms)':>12} | {'Throughput (tok/s)':>19}"
    print("\n" + "=" * len(header))
    print(" 实验结果表格")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for max_nbt in SCAN_VALUES:
        if max_nbt not in all_results:
            print(f"{max_nbt:>24} | {'N/A':>14} | {'N/A':>14} | {'N/A':>12} | {'N/A':>12} | {'N/A':>19}")
            continue
        r = all_results[max_nbt]
        print(f"{max_nbt:>24} | {r['ttft_p50']*1000:>14.1f} | {r['ttft_p99']*1000:>14.1f} | "
              f"{r['itl_p50']*1000:>12.2f} | {r['itl_p99']*1000:>12.2f} | "
              f"{r['throughput']:>19.2f}")

    print("=" * len(header) + "\n")


def plot_figure_a1(all_results, save_path):
    """图 A1-A1：max_num_batched_tokens vs TTFT P50（独立子图）"""
    x = []
    ttft_p50 = []
    for max_nbt in SCAN_VALUES:
        if max_nbt in all_results:
            r = all_results[max_nbt]
            if not np.isnan(r["ttft_p50"]):
                x.append(max_nbt)
                ttft_p50.append(r["ttft_p50"] * 1000)
    if not x:
        print("[WARN] 无有效数据，跳过图 A1")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(x, ttft_p50, "o-", label="TTFT P50", linewidth=2, markersize=8)
    ax.set_xlabel("max_num_batched_tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("TTFT P50 vs max_num_batched_tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图 A1（TTFT P50）已保存: {save_path}")


def plot_figure_a1_ttft_p99(all_results, save_path):
    """图 A1-B1：max_num_batched_tokens vs TTFT P99"""
    x = []
    ttft_p99 = []
    for max_nbt in SCAN_VALUES:
        if max_nbt in all_results:
            r = all_results[max_nbt]
            if not np.isnan(r["ttft_p99"]):
                x.append(max_nbt)
                ttft_p99.append(r["ttft_p99"] * 1000)
    if not x:
        print("[WARN] 无有效数据，跳过图 A1 TTFT P99")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(x, ttft_p99, "s--", label="TTFT P99", linewidth=2, markersize=8, color="tab:orange")
    ax.set_xlabel("max_num_batched_tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("TTFT P99 vs max_num_batched_tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图 A1（TTFT P99）已保存: {save_path}")


def plot_figure_a1_itl_p50(all_results, save_path):
    """图 A1-C1：max_num_batched_tokens vs ITL P50"""
    x = []
    itl_p50 = []
    for max_nbt in SCAN_VALUES:
        if max_nbt in all_results:
            r = all_results[max_nbt]
            if not np.isnan(r["itl_p50"]):
                x.append(max_nbt)
                itl_p50.append(r["itl_p50"] * 1000)
    if not x:
        print("[WARN] 无有效数据，跳过图 A1 ITL P50")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(x, itl_p50, "o-", label="ITL P50", linewidth=2, markersize=8, color="tab:green")
    ax.set_xlabel("max_num_batched_tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("ITL P50 vs max_num_batched_tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图 A1（ITL P50）已保存: {save_path}")


def plot_figure_a1_itl_p99(all_results, save_path):
    """图 A1-D1：max_num_batched_tokens vs ITL P99"""
    x = []
    itl_p99 = []
    for max_nbt in SCAN_VALUES:
        if max_nbt in all_results:
            r = all_results[max_nbt]
            if not np.isnan(r["itl_p99"]):
                x.append(max_nbt)
                itl_p99.append(r["itl_p99"] * 1000)
    if not x:
        print("[WARN] 无有效数据，跳过图 A1 ITL P99")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(x, itl_p99, "s--", label="ITL P99", linewidth=2, markersize=8, color="tab:red")
    ax.set_xlabel("max_num_batched_tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("ITL P99 vs max_num_batched_tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图 A1（ITL P99）已保存: {save_path}")


def plot_figure_a2(all_results, save_path):
    """图 A2：max_num_batched_tokens vs Throughput"""
    x = []
    y = []

    for max_nbt in SCAN_VALUES:
        if max_nbt in all_results:
            r = all_results[max_nbt]
            if not np.isnan(r["throughput"]):
                x.append(max_nbt)
                y.append(r["throughput"])

    if not x:
        print("[WARN] 无有效数据，跳过图 A2")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogx(x, y, "o-", color="tab:purple", linewidth=2, markersize=10)
    ax.set_xlabel("max_num_batched_tokens")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs max_num_batched_tokens")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.grid(True, alpha=0.3)

    # 标注峰值点
    max_idx = np.argmax(y)
    ax.annotate(f"Peak: {y[max_idx]:.1f} tok/s",
                xy=(x[max_idx], y[max_idx]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=10, color="tab:purple",
                arrowprops=dict(arrowstyle="->", color="tab:purple"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图 A2 已保存: {save_path}")


def generate_table_for_report(all_results):
    """生成可直接粘贴到实验报告的 Markdown 表格"""
    lines = []
    lines.append("| max_num_batched_tokens | TTFT P50 (ms) | TTFT P99 (ms) | ITL P50 (ms) | ITL P99 (ms) | Throughput (tok/s) |")
    lines.append("|------------------------|---------------|---------------|--------------|--------------|--------------------|")

    for max_nbt in SCAN_VALUES:
        if max_nbt in all_results:
            r = all_results[max_nbt]
            lines.append(
                f"| {max_nbt:>22} | {r['ttft_p50']*1000:>13.1f} | {r['ttft_p99']*1000:>13.1f} "
                f"| {r['itl_p50']*1000:>12.2f} | {r['itl_p99']*1000:>12.2f} "
                f"| {r['throughput']:>18.2f} |"
            )
        else:
            lines.append(f"| {max_nbt:>22} | N/A | N/A | N/A | N/A | N/A |")

    return "\n".join(lines)


def main():
    print(f"[INFO] 结果目录: {RESULTS_PATH}")
    if not RESULTS_PATH.exists():
        print(f"[ERROR] 目录不存在: {RESULTS_PATH}")
        sys.exit(1)

    all_results = collect_all_results()

    if not all_results:
        print("[ERROR] 没有收集到任何有效数据，请先运行 task2.sh")
        sys.exit(1)

    # 打印表格
    print_table(all_results)

    # 图表（P50/P99 分开画）
    plot_figure_a1(all_results, os.path.join(RESULTS_DIR, "figure_A1_ttft_p50.png"))
    plot_figure_a1_ttft_p99(all_results, os.path.join(RESULTS_DIR, "figure_A1_ttft_p99.png"))
    plot_figure_a1_itl_p50(all_results, os.path.join(RESULTS_DIR, "figure_A1_itl_p50.png"))
    plot_figure_a1_itl_p99(all_results, os.path.join(RESULTS_DIR, "figure_A1_itl_p99.png"))
    plot_figure_a2(all_results, os.path.join(RESULTS_DIR, "figure_A2_throughput.png"))

    # 生成报告用的 Markdown 表格
    table_md = generate_table_for_report(all_results)
    table_file = os.path.join(RESULTS_DIR, "report_table.md")
    with open(table_file, "w") as f:
        f.write(table_md)
    print(f"[OK] 报告表格已保存: {table_file}")

    # 生成 JSON 汇总
    summary_file = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] 汇总 JSON 已保存: {summary_file}")

    # 生成分析文本
    print_analysis(all_results)


def print_analysis(all_results):
    """输出分析摘要，帮助填充实验报告"""
    keys = list(all_results.keys())
    if len(keys) < 2:
        print("[WARN] 数据点不足，跳过分析")
        return

    sorted_keys = sorted(keys)
    first = all_results[sorted_keys[0]]
    last = all_results[sorted_keys[-1]]

    print("\n" + "=" * 60)
    print(" 分析摘要（供实验报告参考）")
    print("=" * 60)

    # TTFT 趋势
    ttft_p50_first = first["ttft_p50"] * 1000
    ttft_p50_last = last["ttft_p50"] * 1000
    if ttft_p50_last < ttft_p50_first:
        direction = "下降"
        pct = (ttft_p50_first - ttft_p50_last) / ttft_p50_first * 100
    else:
        direction = "上升"
        pct = (ttft_p50_last - ttft_p50_first) / ttft_p50_first * 100
    print(f"\n1. TTFT P50 趋势：从 {sorted_keys[0]} → {sorted_keys[-1]}，TTFT {direction} {pct:.1f}%")

    # ITL 趋势
    itl_p50_first = first["itl_p50"] * 1000
    itl_p50_last = last["itl_p50"] * 1000
    if itl_p50_last > itl_p50_first:
        direction = "上升"
        pct = (itl_p50_last - itl_p50_first) / itl_p50_first * 100
    else:
        direction = "下降"
        pct = (itl_p50_first - itl_p50_last) / itl_p50_first * 100
    print(f"2. ITL P50 趋势：从 {sorted_keys[0]} → {sorted_keys[-1]}，ITL {direction} {pct:.1f}%")

    # Throughput 趋势
    tp_first = first["throughput"]
    tp_last = last["throughput"]
    print(f"3. Throughput 趋势：从 {tp_first:.1f} → {tp_last:.1f} tok/s")

    # 饱和点检测
    tp_values = [all_results[k]["throughput"] for k in sorted_keys]
    tp_diffs = [tp_values[i+1] - tp_values[i] for i in range(len(tp_values)-1)]
    for i, diff in enumerate(tp_diffs):
        pct_change = diff / tp_values[i] * 100
        if pct_change < 2:  # 小于 2% 的增长视为饱和
            print(f"4. Throughput 在 max_num_batched_tokens={sorted_keys[i]} 附近趋于饱和（增幅仅 {pct_change:.1f}%）")
            break
    else:
        print("4. Throughput 在测试范围内持续增长，未观察到明显饱和")

    # 对话服务推荐值
    # 综合考虑 ITL 和 Throughput
    print("\n5. 对话服务推荐：")
    for k in sorted_keys:
        r = all_results[k]
        print(f"   max_num_batched_tokens={k}: ITL P50={r['itl_p50']*1000:.2f}ms, "
              f"Throughput={r['throughput']:.1f} tok/s")

    print("=" * 60)


if __name__ == "__main__":
    main()
