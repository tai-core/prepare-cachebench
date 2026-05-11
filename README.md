# Cache Chain Benchmark
在绝对路径 `/mnt/beegfs/khr/bench` 下

This package provides two complementary strategies for prefix-cache chain benchmarking,
sharing the same runner (`scheduled_openai_chat_bench.py`).

| | Strategy A: Synthetic Fill | Strategy B: Conversation Chain |
|---|---|---|
| **用途** | 纯 prefill / cache 性能测试 | cache + **模型精度** 同时验证 |
| **数据来源** | bench.jsonl 随机采样 + 合成填充文本 | bench.jsonl 长度 ≥100k 字符的长对话 |
| **前置链** | A → A+填充 → A+填充+填充 | A=前50k token → B=前70k token → C=完整对话 |
| **预期输出** | 无（模型自由生成） | 每条带 `expected_response`，可直接对比 |
| **制备脚本** | `prepare_cache_chain_data.sh` | `prepare_conv_chain_data.sh` |
| **schedule 文件** | `bench-70k-ABC-schedule.jsonl` | `bench-conv-chain-schedule.jsonl` |
| **组数** | 可配置（默认 100） | 自动使用全部长对话（~869 组） |
| **Token 范围** | A: ~70k, B: ~100k, C: ~120-142k | A: ~50k, B: ~70k, C: 67k-210k |

---

## 共享文件

- `scheduled_openai_chat_bench.py` — 通用 benchmark 执行器（两种策略共用）
- `make_cache_hit_schedule.py` — 把 A/B/C 三个 JSONL 合并为 schedule JSONL
- `generate_initial_distribution_a.py` — 从 bench.jsonl 采样生成 A 分布
- `generate_prefix_chain_datasets.py` — 从 A 生成 B/C（合成填充文本）
- `split_conversation_chain.py` — 从 bench.jsonl 长对话切分出 A/B/C 前置链
- `CACHE_CHAIN_BENCH_README.md` — 本文档

---

## Strategy A: Synthetic Fill — 纯性能测试

### 数据制备

```bash
bash /mnt/beegfs/khr/bench/prepare_cache_chain_data.sh
```

Defaults:

```text
INPUT_DATASET=/mnt/beegfs/dataset/bench.jsonl
MODEL_PATH=/ssd/models/GLM-5-FP8
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
  --model /ssd/models/GLM-5-FP8/ \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --chain-after-complete \
  --increment-interval-min 2 \
  --increment-interval-max 20 \
  --max-concurrency 4 \
  --seed 42 \
  --save-result /mnt/beegfs/results/cache-chain-result.json
```

`--chain-after-complete` 模式下 B/C 的实际发送时间 = 前一阶段完成时间 + 随机间隔（2~20s）。
`--max-concurrency 4` 限制同时 in-flight 的请求数。

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
  --model /ssd/models/GLM-5-FP8/ \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 2000 \
  --chain-after-complete \
  --increment-interval-min 2 \
  --increment-interval-max 20 \
  --max-concurrency 4 \
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

## 指标说明（两种策略共用）

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
  --model /ssd/models/GLM-5-FP8/ \
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
