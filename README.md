# Cache Chain Benchmark
在绝对路径 `/mnt/beegfs/khr/bench` 下

## 运行环境

```bash
ssh g0033
tmux new-session -s vllm-bench
conda activate vllm_test
cd /mnt/beegfs/khr/bench
```

This package provides three complementary benchmark strategies sharing the
same runner (`scheduled_openai_chat_bench.py`).

| | Strategy A: Synthetic Fill | Strategy B: Conversation Chain | Strategy C: Cold Long-Context Prefill |
|---|---|---|---|
| **用途** | 纯 prefill / cache 性能测试 | cache + **模型精度** 同时验证 | 不 cache hit 的超长 C 段冷 prefill OOM 压力测试 |
| **数据来源** | bench.jsonl 随机采样 + 合成填充文本 | bench.jsonl 长度 ≥100k 字符的长对话 | Strategy A 生成的 C 段 |
| **前置链** | A → A+填充 → A+填充+填充 | A=前50k token → B=前70k token → C=完整对话 | 只发送 C，不发送 A/B |
| **预期输出** | 无（模型自由生成） | 每条带 `expected_response`，可直接对比 | 无；建议 `output_tokens=1` 隔离 prefill |
| **制备脚本** | `prepare_cache_chain_data.sh` | `prepare_conv_chain_data.sh` | `prepare_cold_prefill_data.sh` |
| **schedule 文件** | `bench-70k-ABC-schedule.jsonl` | `bench-conv-chain-schedule.jsonl` | `bench-70k-C-only-cold-schedule.jsonl` |
| **组数** | 可配置（默认 100） | 自动使用全部长对话（~869 组） | 与 Strategy A 的 C 段数量一致，可 `LIMIT` 截断 |
| **Token 范围** | A: ~70k, B: ~100k, C: ~120-142k | A: ~50k, B: ~70k, C: 67k-210k | C-only: ~120-142k |

---

## 共享文件

- `scheduled_openai_chat_bench.py` — 通用 benchmark 执行器（两种策略共用）
- `make_cache_hit_schedule.py` — 把 A/B/C 三个 JSONL 合并为 schedule JSONL
- `generate_initial_distribution_a.py` — 从 bench.jsonl 采样生成 A 分布
- `generate_prefix_chain_datasets.py` — 从 A 生成 B/C（合成填充文本）
- `split_conversation_chain.py` — 从 bench.jsonl 长对话切分出 A/B/C 前置链
- `prepare_cache_chain_data.sh` — 生成 Strategy A 的 A/B/C 数据和 schedule
- `prepare_conv_chain_data.sh` — 生成 Strategy B 的真实对话链 schedule
- `prepare_cold_prefill_data.sh` — 从 A/B/C schedule 抽取 C-only 冷 prefill schedule

---

## 并发控制

`scheduled_openai_chat_bench.py` 提供两层并发控制：

| 参数 | 控制范围 | 释放时机 |
|---|---|---|
| `--max-concurrency N` | **总** in-flight 请求数 | 请求完成 |
| `--max-prefill-concurrency N` | **prefill 阶段** 的请求数 | 首 token 到达 |

**关键区别**：`--max-concurrency` 对 prefill 和 decode 一视同仁——请求完成前都占着槽位。
`--max-prefill-concurrency` 只限制 prefill 队列：首 token 一到立即释放，排队的下一个请求马上进入 prefill。

```
请求 A: [acquire prefill]──prefill──[首token→release]──decode──
请求 B:                  排队......[acquire]──prefill──[release]──decode──
请求 C:                                  排队......[acquire]──prefill──...
```

推荐组合：

| 场景 | 配置 |
|---|---|
| 只限制 prefill，放满 decode | `--max-prefill-concurrency 2` |
| prefill + 总 in-flight 都限制 | `--max-prefill-concurrency 2 --max-concurrency 8` |
| 兼容旧行为（无区分） | `--max-concurrency 4`（prefill 不单独控） |

> 若两者都设，必须 `max-prefill-concurrency ≤ max-concurrency`。

---

## Strategy A: Synthetic Fill — 纯性能测试

### 数据制备

```bash
bash /mnt/beegfs/khr/bench/prepare_cache_chain_data.sh
```

Defaults:

```text
INPUT_DATASET=/mnt/beegfs/dataset/bench.jsonl
MODEL_PATH=/ssd/models/GLM-5.1-FP8
OUT_DIR=/mnt/beegfs/khr/bench
NUM_SAMPLES=100
MIN_TOKENS=20000   MAX_TOKENS=128000
MEAN_TOKENS=70000   STD_TOKENS=18000
OUTPUT_TOKENS=256   SEED=42
B_EXTRA_TOKENS=30000
C_EXTRA_MIN_TOKENS=20000   C_EXTRA_MAX_TOKENS=42000
REQUEST_RATE=0.7   BURSTINESS=1.0   STATIC_T=0
```

Override via environment:

```bash
NUM_SAMPLES=300 REQUEST_RATE=1.5 bash /mnt/beegfs/khr/bench/prepare_cache_chain_data.sh
```

Produces:
- `bench-70k.jsonl` — A 分布（随机采样 + tokenizer 截断）
- `bench-70k-B.jsonl` — B = A + ~30k token 合成填充
- `bench-70k-C.jsonl` — C = B + 20k~42k token 合成填充
- `bench-70k-ABC-schedule.jsonl` — 最终 schedule
- `bench-70k-ABC-vllm.jsonl` — vLLM bench serve 格式

### 运行 Benchmark

```bash
python3 /mnt/beegfs/khr/bench/scheduled_openai_chat_bench.py \
  --schedule-path /mnt/beegfs/khr/bench/bench-70k-ABC-schedule.jsonl \
  --base-url http://g0033:17000 \
  --endpoint /v1/chat/completions \
  --model /ssd/models/GLM-5.1-FP8/ \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --chain-after-complete \
  --increment-interval-min 20 \
  --increment-interval-max 60 \
  --max-prefill-concurrency 2 \
  --seed 42 \
  --save-result /mnt/beegfs/results/cache-chain-result.json
```

`--chain-after-complete` 模式下 B/C 的实际发送时间 = 前一阶段完成时间 + 随机间隔（20~60s）。
`--max-prefill-concurrency 2` 限制 prefill 阶段最多 2 个请求同时排队，decode 不限。

---

## Strategy B: Conversation Chain — 精度 + 性能测试

### 为什么需要这个

Strategy A 的合成填充文本是机械重复的 filler，无法判断模型输出是否正确。
Strategy B 在真实的对话轮次上切分——后续轮次天然引用前文的人设、情节、约束，
你可以直接用 `expected_response`（原始 assistant 回复）对比模型输出，
判断 **cache 开启后模型是否保持了对话一致性**。

### 数据制备

```bash
bash /mnt/beegfs/khr/bench/prepare_conv_chain_data.sh
```

Defaults:

```text
INPUT_DATASET=/mnt/beegfs/dataset/bench.jsonl
OUT_DIR=/mnt/beegfs/khr/bench
MIN_CHARS=100000        # 只取 ≥100k 字符的长对话（~67k token 起）
MIN_TURNS=10            # 最少 10 轮 user+assistant
NUM_GROUPS=0            # 0 = 使用全部符合条件的记录（~869 组）
TARGET_TOKENS_A=50000   # A 段目标 ~50k token
TARGET_TOKENS_B=70000   # B 段目标 ~70k token
CHARS_PER_TOKEN=1.5     # 中文字符→token 估算系数
OUTPUT_TOKENS=256
SEED=42
REQUEST_RATE=0.7
```

Produce:
- `bench-conv-chain-schedule.jsonl` — 最终 schedule

每条 schedule 记录包含 `expected_response` 字段（原始 assistant 回复）。

### 运行 Benchmark

```bash
python3 /mnt/beegfs/khr/bench/scheduled_openai_chat_bench.py \
  --schedule-path /mnt/beegfs/khr/bench/bench-conv-chain-schedule.jsonl \
  --base-url http://g0033:17000 \
  --endpoint /v1/chat/completions \
  --model /ssd/models/GLM-5.1-FP8/ \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --chain-after-complete \
  --increment-interval-min 20 \
  --increment-interval-max 60 \
  --max-prefill-concurrency 2 \
  --seed 42 \
  --save-result /mnt/beegfs/results/conv-chain-result.json
```

### 精度评估

结果 JSON 中每条记录包含：

```json
{
  "request_id": "0:A",
  "group_id": 0,
  "stage": "A",
  "generated_text": "模型的回复...",
  "expected_response": "原始的 assistant 回复..."
}
```

直接对比 `generated_text` 与 `expected_response` 即可评估精度。
也可只对比 C 段（完整对话的最后一轮）——C 段累积了最多的前文依赖，最考验 cache 一致性。

---

## Strategy C: Cold Long-Context Prefill — 冷 cache 超长文本 OOM 压力测试

### 测试目标

这个策略专门用于回答一个问题：**在没有 prefix-cache hit 的情况下，平均 C 段长度的请求直接进入 prefill，会不会让 prefill 实例 OOM。**

它和 Strategy A 共享同一批合成 A/B/C 数据，但运行时只发送 C 段：

- 不发送 A/B，所以不会主动构造 prefix cache hit。
- 不使用 `--chain-after-complete`，因为这里不是链式 cache 测试。
- 建议冷启动服务，或换 `SEED` 重新生成 prompt，避免同一服务里已经存在相同前缀的 KV cache。
- 默认把 C-only 请求的 `output_tokens` 改成 1，尽量隔离 prefill 压力；如需同时观察 decode，可设置 `OUTPUT_TOKENS=256`。

### 数据制备

先生成 Strategy A 的 A/B/C 数据。这里建议保留 C 段默认长度分布，只把输出长度压到 1：

```bash
OUTPUT_TOKENS=1 \
NUM_SAMPLES=100 \
bash /mnt/beegfs/khr/bench/prepare_cache_chain_data.sh
```

再从 A/B/C schedule 中抽出 C-only 冷 prefill schedule：

```bash
OUTPUT_TOKENS=1 \
INTERVAL=0 \
bash /mnt/beegfs/khr/bench/prepare_cold_prefill_data.sh
```

Defaults:

```text
OUT_DIR=/mnt/beegfs/khr/bench
INPUT_SCHEDULE=/mnt/beegfs/khr/bench/bench-70k-ABC-schedule.jsonl
OUTPUT_SCHEDULE=/mnt/beegfs/khr/bench/bench-70k-C-only-cold-schedule.jsonl
STAGE=C
INTERVAL=0              # 所有 C 请求立刻排队，由 --max-prefill-concurrency 控制压力
LIMIT=0                 # 0 = 使用全部 C 请求
OUTPUT_TOKENS=1         # 0 = 保留原 schedule 的 output_tokens
```

Produce:
- `bench-70k-C-only-cold-schedule.jsonl` — 只包含 C 段的冷 prefill schedule

### 单请求 Canary

先用 1 条 C 请求验证模型、context length、服务端参数都能跑通：

```bash
python3 /mnt/beegfs/khr/bench/scheduled_openai_chat_bench.py \
  --schedule-path /mnt/beegfs/khr/bench/bench-70k-C-only-cold-schedule.jsonl \
  --base-url http://g0033:17000 \
  --endpoint /v1/chat/completions \
  --model /ssd/models/GLM-5.1-FP8/ \
  --limit 1 \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --max-prefill-concurrency 1 \
  --max-concurrency 1 \
  --save-result /mnt/beegfs/results/c-cold-canary-1.json
```

### Prefill 并发压力测试

从小并发开始逐步增加，观察在哪个 `max-prefill-concurrency` 下出现 OOM、超时或服务端 worker 崩溃：

```bash
for n in 1 2 3 4; do
  python3 /mnt/beegfs/khr/bench/scheduled_openai_chat_bench.py \
    --schedule-path /mnt/beegfs/khr/bench/bench-70k-C-only-cold-schedule.jsonl \
    --base-url http://g0033:17000 \
    --endpoint /v1/chat/completions \
    --model /ssd/models/GLM-5.1-FP8/ \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 2000 \
    --max-prefill-concurrency "${n}" \
    --max-concurrency "${n}" \
    --save-result "/mnt/beegfs/results/c-cold-prefill-${n}.json"
done
```

判断方式：

- benchmark 输出中的 `Failed requests` 是否大于 0。
- 结果 JSON 中每条失败请求的 `error` 字段。
- vLLM server 日志中是否出现 CUDA OOM、worker exit、request timeout。
- 对这个策略，`TTFT` 和失败率比整体吞吐更重要；`output_tokens=1` 时 `TPOT` 不具备参考意义。

---

## 指标说明（三种策略共用）

| 指标 | 含义 |
|---|---|
| `request_throughput` | 每秒完成的请求数 |
| `output_throughput` | 每秒生成的输出 token 数 |
| `total_token_throughput` | 每秒总 token（输入+输出）吞吐 |
| `max_output_tokens_per_s` | 按秒分桶的峰值输出 token 速率 |
| `max_concurrent_requests` | 峰值并发请求数 |
| `mean/median/pXX_ttft_ms` | 首 token 延迟 |
| `mean/median/pXX_tpot_ms` | 每输出 token 时间（不含首 token） |
| `mean/median/pXX_itl_ms` | token 间延迟 |
| `mean/median/pXX_e2el_ms` | 端到端延迟 |

---

## Optional: vLLM Bench Serve

两种策略的 schedule 都可以转为 vLLM 格式用于确定性排序测试（但不支持 `--chain-after-complete` 间隔时序）：

```bash
# Strategy A 的 vLLM 格式已由 prepare_cache_chain_data.sh 自动生成
vllm bench serve --backend openai-chat \
  --dataset-name custom \
  --dataset-path /mnt/beegfs/khr/bench/bench-70k-ABC-vllm.jsonl \
  --num-prompts 300 \
  --base-url http://g0033:17000 \
  --model /ssd/models/GLM-5.1-FP8/ \
  --request-rate 0.8 \
  --disable-shuffle \
  --custom-output-len -1 \
  --skip-chat-template
```

---

## Efficiency Notes

- `prepare_cache_chain_data.sh` 运行全部预处理步骤后停止，不发 benchmark 流量。
- `prepare_conv_chain_data.sh` 单步完成切分，不需要 tokenizer（用 chars/token 估算）。
- `generate_initial_distribution_a.py` 候选阶段只存文件偏移和 token 长度，选定后再按偏移读取。
- `generate_prefix_chain_datasets.py` 流式读 A 并直接写 B/C，B 后缀只 tokenize/decode 一次。
- `make_cache_hit_schedule.py --disable-shuffle` 流式读 A/B/C 直接写出。
- `split_conversation_chain.py` 纯文本解析，无 tokenizer 依赖，869 组秒级完成。
- `scheduled_openai_chat_bench.py` 复用单个 aiohttp session，增量解析 SSE chunk。

---

## Baseline（Strategy A, 100 groups, max-prefill-concurrency=3）

GLM-5.1-FP8 on g0033:17000, interval 20~60s, 300 requests (100×A/B/C):

```bash
cd /mnt/beegfs/khr/bench && \
python3 /mnt/beegfs/khr/bench/scheduled_openai_chat_bench.py \
  --schedule-path /mnt/beegfs/khr/bench/bench-70k-ABC-schedule.jsonl \
  --base-url http://g0033:17000 \
  --endpoint /v1/chat/completions \
  --model /ssd/models/GLM-5.1-FP8/ \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --chain-after-complete \
  --increment-interval-min 20 \
  --increment-interval-max 60 \
  --seed 42 \
  --max-prefill-concurrency 3

================= Scheduled Benchmark Result =================
Successful requests: 300
Failed requests: 0
Benchmark duration (s): 978.81
Request throughput (req/s): 0.31
Output token throughput (tok/s): 78.39
Peak output token throughput (tok/s): 132.00
Peak concurrent requests: 7
Total token throughput (tok/s): 29307.76
Total input tokens: 28610015
Total output tokens: 76728
Mean TTFT (ms): 9707.15
P50 TTFT (ms): 9869.48
P90 TTFT (ms): 12930.49
P95 TTFT (ms): 13622.94
P99 TTFT (ms): 14775.28
Mean TPOT (ms): 9.74
P50 TPOT (ms): 9.77
P90 TPOT (ms): 15.22
P95 TPOT (ms): 16.37
P99 TPOT (ms): 20.52
Mean ITL (ms): 20.43
P50 ITL (ms): 1.17
P90 ITL (ms): 40.44
P95 ITL (ms): 40.99
P99 ITL (ms): 81.46
Mean E2EL (ms): 12187.61
P50 E2EL (ms): 12357.62
P90 E2EL (ms): 14913.92
P95 E2EL (ms): 15667.90
P99 E2EL (ms): 17038.40
==============================================================
```

| Key metric | Value |
|---|---|
| P50 TTFT | 9.87 s |
| P99 TTFT | 14.78 s |
| Mean TPOT | 9.74 ms |
| Output throughput | 78.39 tok/s |
| Peak concurrent | 7 |
| Failure rate | 0% |

---

## Strategy A2: 20k-200k Incremental Chain

This variant keeps the same synthetic prefix-cache idea, but replaces the fixed
A/B/C chain with a dynamic chain:

- A is sampled from `bench.jsonl` with tokenizer lengths in `[20k, 200k]`:
  the combined A distribution targets 70k average tokens while reserving a
  small forced cold long tail up to 200k tokens.
- Every later request in the same group is the previous prompt plus synthetic
  filler, targeting `ceil(previous_tokens * 1.10)`.
- If the next target would exceed `200000` input tokens, that request is not
  generated. No `>200k` request is kept.
- Each row carries `stage_index`; `scheduled_openai_chat_bench.py` uses it for
  `--chain-after-complete` ordering.

### Data preparation

```bash
bash /mnt/beegfs/khr/bench/prepare_incremental_cache_chain_data.sh
```

Defaults:

```text
INPUT_DATASET=/mnt/beegfs/dataset/bench.jsonl
MODEL_PATH=/ssd/models/GLM-5.1-FP8
OUT_DIR=/mnt/beegfs/khr/bench
NUM_SAMPLES=100
MIN_TOKENS=20000   MAX_TOKENS=200000
MEAN_TOKENS=70000  STD_TOKENS=18000
TAIL_SAMPLES=10    TAIL_MIN_TOKENS=160000  TAIL_MAX_TOKENS=200000
INCREMENT_RATIO=0.10
OUTPUT_TOKENS=256  SEED=42
REQUEST_RATE=0.7   BURSTINESS=1.0   STATIC_T=0
```

Produces:

- `bench-20k-200k-A.jsonl` - A distribution.
- `bench-20k-200k-inc10-schedule.jsonl` - dynamic chain schedule.
- `bench-20k-200k-inc10-vllm.jsonl` - vLLM bench serve format.

### Run benchmark

```bash
python3 /mnt/beegfs/khr/bench/scheduled_openai_chat_bench.py \
  --schedule-path /mnt/beegfs/khr/bench/bench-20k-200k-inc10-schedule.jsonl \
  --base-url http://g0033:17000 \
  --endpoint /v1/chat/completions \
  --model /ssd/models/GLM-5.1-FP8/ \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --chain-after-complete \
  --increment-interval-min 20 \
  --increment-interval-max 60 \
  --max-prefill-concurrency 2 \
  --seed 42 \
  --save-result /mnt/beegfs/results/cache-chain-inc10-200k-result.json
```

For cold prefill on the last stage of each dynamic group:

```bash
INPUT_SCHEDULE=/mnt/beegfs/khr/bench/bench-20k-200k-inc10-schedule.jsonl \
OUTPUT_SCHEDULE=/mnt/beegfs/khr/bench/bench-20k-200k-inc10-last-cold-schedule.jsonl \
STAGE=last \
OUTPUT_TOKENS=1 \
INTERVAL=0 \
bash /mnt/beegfs/khr/bench/prepare_cold_prefill_data.sh
```

vLLM bench serve format:

```bash
vllm bench serve --backend openai-chat \
  --dataset-name custom \
  --dataset-path /mnt/beegfs/khr/bench/bench-20k-200k-inc10-vllm.jsonl \
  --num-prompts -1 \
  --base-url http://g0033:17000 \
  --model /ssd/models/GLM-5.1-FP8/ \
  --request-rate 0.8 \
  --disable-shuffle \
  --custom-output-len -1 \
  --skip-chat-template
```
