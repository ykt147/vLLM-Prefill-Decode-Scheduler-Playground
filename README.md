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
### 目标
直观感受 vLLM 的 Continuous Batching + PagedAttention 相对于 Lab 1 中 transformers 朴素 model.generate() 的吞吐优势，并解释差距来源。

### 方法指引
#### Step 1：构造一个并发负载

准备 64 条 prompt（可从 ShareGPT 采样或自己生成），输入长度在 128–512 之间随机，输出 max_new_tokens=128。

#### Step 2：Baseline —— 朴素 transformers（复用 Lab 1 脚本）

方案 A：串行提交 64 条（for 循环调用 generate），记录总耗时。
方案 B：padding 静态 batching，将 64 条填充到同长度，一次性 generate，记录总耗时（batch 过大会 OOM，此时报告你能塞进显存的最大 batch）。
Step 3：vLLM 离线推理

```
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", gpu_memory_utilization=0.9)
sampling = SamplingParams(temperature=0.0, max_tokens=128)
outputs = llm.generate(prompts, sampling)  # 一次性提交 64 条
```

记录端到端耗时与峰值显存。

vLLM —— 通过 scheduler 回调读 metrics：
离线 LLM(...) 模式下没有 /metrics HTTP 端点，可继承 vllm.v1.core.sched.scheduler.Scheduler，在 schedule() 内每步累加 sum(req.num_computed_tokens for req in self.running)，记录最大值后把它写入文件供主进程读取。
在线 vllm serve 模式（Task 2/3 是使用在线vllm）：可以直接 curl http://host:port/metrics | grep vllm:gpu_cache_usage_perc（表示当前实际已使用的 KV Cache 空间占整个 GPU KV Cache 容量的比例），再乘以 GPU KV cache size (启动日志中那个 token 数)表示实际使用的 KV Cache，进一步可用peak_bytes = peak_tokens × per_token_KV_bytes换算成字节表示。

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
### 目标
vLLM V1 调度器默认开启 chunked prefill，max_num_batched_tokens 是其最重要的参数之一（表每step的token预算，决定每步 forward 处理多少 token），直接影响 TTFT / ITL / Throughput 等重要指标。
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
### 目标
真正理解调度器，本 Task 需要替换 vLLM V1 的 Scheduler，实现自设计策略，并与默认策略对比。
### Scheduler 源码（感兴趣可自行阅读了解默认调度器设计）
V1 Scheduler 入口位于 vllm/v1/core/sched/scheduler.py，关键类为 Scheduler，核心方法是 schedule() -> SchedulerOutput，每个调度步被 EngineCore 调用一次。其职责大致是：

遍历 self.running（正在 decode 的请求）与 self.waiting（排队中的请求）
按某种策略选出这一步要跑的请求集合，并决定每个请求处理多少 token（decode 只跑 1 个 token；prefill 可跑 1 到 chunk_size 个 token）
调用 KVCacheManager 为它们分配 block
返回 SchedulerOutput，由 ModelRunner 执行一次 forward
vLLM V1 预留了自定义 scheduler 注入点：SchedulerConfig.scheduler_cls 字段可直接传一个类（或 "mod.submod.MyScheduler" 字符串），所以你不需要 fork vLLM 源码，只需继承 vllm.v1.core.sched.scheduler.Scheduler 并覆写 schedule()来设计你自己的调度器。

### 策略 A：Step-Exclusive Prefill-First（TTFT 优化型）
行为：每个调度步判断：

如果 waiting 队列非空 → 只调度 prefill（不混入 decode），按 token budget 一次塞尽量多的完整 prompt
否则 → 只调度 decode
实现提示：

在 schedule() 入口判断 len(self.waiting) > 0，当有等待队列（waiting 非空）时，临时隐藏 running 队列（设为空列表），调用父类调度器 → 此时父类只能看到 prefill 请求，所以这一整步只做 prefill
一步只产出纯 prefill batch 或 纯 decode batch，没有混合
预期现象：TTFT 低（新请求一来就被处理），但 ITL 严重恶化（decode 被长 prefill 整体阻塞），且 GPU 利用率低（因为decode是访存密集，prefill是计算密集，默认的混合调度通过任务类型互补可填充 pipeline 的气泡）。

### 策略 B
pure-decode · Pure Decode-First：running 不空就只跑 decode，绝不混批 prefill，与策略 A 相反：waiting 让位给 running
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
| 策略 B: Pure Decode-First | running 不空就只跑 decode，绝不混批 prefill，与策略 A 相反：waiting 让位给 running |

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


