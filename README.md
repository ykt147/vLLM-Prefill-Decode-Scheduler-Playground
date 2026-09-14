# 🚀 vLLM Prefill / Decode Scheduler Playground

> Exploring LLM inference performance from **PagedAttention** and
> **Continuous Batching** to custom **Prefill / Decode scheduling** on vLLM V1.

一个面向 LLM Inference / Serving 学习的 vLLM 调度器开源项目。

本项目从 Hugging Face `transformers` baseline 出发，逐步分析
vLLM 的 PagedAttention、Continuous Batching、Chunked Prefill，
并进一步修改 vLLM V1 Scheduler，实现不同的 Prefill / Decode
调度策略，分析 TTFT、ITL 和 Throughput 之间的权衡。

---

## ✨ Features

- ⚡ Transformers vs vLLM 端到端性能对比
- 🧠 PagedAttention / KV Cache 内存分析
- 🔄 Continuous Batching 性能分析
- 🧩 Chunked Prefill 参数扫描
- 🎛 `max_num_batched_tokens` Benchmark
- 🛠 自定义 vLLM V1 Scheduler
- 🚀 Step-Exclusive Prefill-First
- 🟢 Pure Decode-First
- 📊 TTFT / ITL / Throughput Benchmark
- 📈 P50 / P99 / CDF 长尾延迟分析
- 🔬 一键复现实验脚本

---

# 🧠 Why This Project?

LLM Serving 的性能并不只是：

> “模型 forward 跑得有多快？”

真正影响在线推理性能的还有：

- KV Cache 如何分配
- 请求何时进入 batch
- 已完成请求何时释放
- Prefill 是否切 chunk
- Decode 是否被 Prefill 阻塞
- 每个 scheduling step 分配多少 token
- TTFT 与 ITL 如何取舍

因此，本项目尝试从 **内存管理 + batching + scheduling**
三个层面理解 vLLM。

---

# 🏗 Architecture

```text
Incoming Requests
       │
       ▼
┌─────────────────────┐
│   vLLM Scheduler    │
│                     │
│ waiting   running   │
│ Prefill   Decode    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Continuous Batching │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   PagedAttention    │
│    KV Block Pool    │
└─────────┬───────────┘
          │
          ▼
        GPU

```

# 开始
## 环境依赖

```bash
pip install -r requirements.txt
```

## Task 1: 从 transformers 到 vLLM —— 吞吐量差异

### 运行方式

```bash
python task1_vllm_vs_hf.py
```

脚本会自动完成三个方案的对比：

1. **transformers 串行** — 64 条 prompt 逐条调用 `model.generate()`
2. **transformers 静态 batching** — 64 条 prompt padding 到同一长度后一次性 `generate()`（如 OOM 自动降级到最大可用 batch）
3. **vLLM 离线推理** — 使用 `LLM.generate()` 一次性提交，利用 Continuous Batching + PagedAttention

结果会打印到终端并保存为 `task1_results.json`。

### 测量指标

| 指标 | 说明 |
|------|------|
| 总耗时 (s) | 从第一条请求提交到最后一条请求完成的端到端时间 |
| Throughput (tok/s) | 总输出 token 数 / 总耗时 |
| 峰值 KV cache (MB) | 基于公式 `per_token_KV = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes` 计算 |

### 模型信息

- 模型: Qwen3-0.6B (FP16)
- per_token_KV_bytes = 2 × 28 × 8 × 128 × 2 = **114688 bytes ≈ 112 KB/token**
- 输入长度: 128–512 tokens (随机)
- 输出长度: max_new_tokens=128

## Task 2：max_num_batched_tokens 扫描实验

### 运行方式

```bash
# 确保 vLLM server 未运行
pkill -f "vllm serve" 2>/dev/null || true

# 运行自动化脚本（会自动启动/停止 server、压测、分析、绘图，task2_output.log里记录运行进度）
bash task2.sh 2>&1 | tee task2_output.log
```

脚本功能：
1. 遍历 `max_num_batched_tokens ∈ {512, 1024, 2048, 4096, 8192}`
2. 每个配置重启 vLLM server，先做 warmup（50 prompts @ rate=1），再跑 2 次正式压测
3. 每组取 2 次运行的中位数，丢弃前 10% 的 warmup 数据
4. 自动调用 `task2_plot.py` 解析结果、绘制图表、填充表格

### 可配置参数

编辑 `task2.sh` 头部变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `Qwen/Qwen3-0.6B` | 测试模型 |
| `NUM_PROMPTS` | `1000` | 压测请求数 |
| `REQUEST_RATE` | `16` | Poisson 请求速率（设为 `inf` 即纯吞吐测试） |
| `GPU_MEM_UTIL` | `0.2` | GPU 显存利用率 |
| `SCAN_VALUES` | `512 1024 2048 4096 8192` | 扫描的参数范围 |
| `RUNS_PER_CONFIG` | `2` | 每组配置运行次数 |

### 手动运行单个配置

```bash
# Terminal 1: 启动 vLLM server
vllm serve Qwen/Qwen3-0.6B \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9

# Terminal 2: 压测
vllm bench serve \
  --model Qwen/Qwen3-0.6B \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --seed 42 \
  --num-prompts 1000 \
  --request-rate 16 \
  --save-result --save-detailed
```

### 输出文件

所有结果保存在 `task2_results/` 目录：

| 文件 | 说明 |
|------|------|
| `run_<nbt>_<idx>.json` | vLLM 压测原始 JSON（含详细数组） |
| `figure_A1_-.png` | 图 A1：max_num_batched_tokens vs TTFT/ITL 折线图 |
| `figure_A2_throughput.png` | 图 A2：max_num_batched_tokens vs Throughput 折线图 |
| `report_table.md` | 可直接粘贴到实验报告的 Markdown 表格 |
| `summary.json` | 所有配置的中位数指标汇总 |

## Task 3：自定义调度策略对比

### 运行方式

```bash
# 确保 vLLM server 未运行
pkill -f "vllm serve" 2>/dev/null || true

# 运行自动化脚本（会自动启动/停止 server、压测、分析、绘图）
bash task3.sh 2>&1 | tee task3_output.log
```

脚本功能：
1. 依次测试 3 种调度策略（vLLM 默认 / 策略 A: Step-Exclusive Prefill-First / 策略 B3: Adaptive Chunk-Size）
2. 每种策略重启 vLLM server 后运行压测
3. 自动调用 `task3_plot.py` 解析结果、绘制 CDF 图、生成对比表

### 调度策略说明

| 策略 | 描述 |
|------|------|
| vLLM V1 默认 | 默认混合调度：prefill 与 decode 混合批处理 |
| 策略 A: Step-Exclusive Prefill-First | 每步只调度一种类型（waiting 非空时只 prefill，否则只 decode），降低 TTFT 但恶化 ITL |
| 策略 B3: Adaptive Chunk-Size | 根据 running 队列拥挤程度动态调整 prefill token budget：空闲时放大降 TTFT，拥挤时缩小保 ITL |

### 可配置参数

编辑 `task3.sh` 头部变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `/public/home/ykt147/model/Qwen3_0.6b` | 测试模型 |
| `NUM_PROMPTS` | `1000` | 压测请求数 |
| `REQUEST_RATE` | `32` | Poisson 请求速率 |
| `GPU_MEM_UTIL` | `0.2` | GPU 显存利用率 |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | 最大 batch token 数 |

### 手动运行单个策略

```bash
# Terminal 1: 启动 vLLM server（策略 A 示例）
vllm serve /public/home/ykt147/model/Qwen3_0.6b \
  --scheduler-cls my_scheduler.StepExclusivePrefillFirstScheduler \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.2 \
  --port 8000

# Terminal 2: 压测
vllm bench serve \
  --model /public/home/ykt147/model/Qwen3_0.6b \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --seed 42 \
  --num-prompts 1000 \
  --request-rate 32 \
  --save-result --save-detailed \
  --result-dir task3_results \
  --result-filename result_Strategy_A_PrefillFirst.json
```

### 自定义 Scheduler

自定义调度器在 `my_scheduler.py` 中实现，继承 `vllm.v1.core.sched.scheduler.Scheduler` 并覆写 `schedule()` 方法。通过 `--scheduler-cls` 参数注入。

### 输出文件

所有结果保存在 `task3_results/` 目录：

| 文件 | 说明 |
|------|------|
| `result_<strategy>.json` | vLLM 压测原始 JSON（含详细 ttfts/itls 数组） |
| `figure_3A_ttft_cdf.png` | 图 3A：TTFT CDF 曲线 |
| `figure_3A_itl_cdf.png` | 图 3A：ITL CDF 曲线 |
| `figure_3A_combined.png` | 图 3A：TTFT + ITL CDF 并排对比图 |
| `comparison_table.md` |  Markdown 表格 |


