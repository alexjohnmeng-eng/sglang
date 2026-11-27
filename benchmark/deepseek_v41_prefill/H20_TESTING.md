# H20 三套完整测试：CP8EP8、CP8EP1、DP8EP8

三套统一使用一个 `deep_gemm._C`，并**显式关闭 WQB→RoPE→FP8 融合**：
`SGLANG_EXPERIMENT_DSV41_SM90_WQB_Q8_ROPE=0`。宿主机即使遗留开关 1 也不会启用。
算子实现保留供后续实验。本次按用户要求只整理脚本、做静态检查，**没有重新运行吞吐、
profile 或 GSM8K**。此前 CP8EP8 成绩开启了该融合，不能当作当前关闭配置的成绩。
本次提交是源码/依赖整合，不声明新的性能收益。

## 环境、依赖与公共设置

- 单机 8×H20，模型 `/data1/fjmeng2/models/DeepSeek-V4.1-Flash-1017`。
- 镜像 `lmsysorg/sglang:latest`，校验固定 image ID：
  `sha256:f3a303d4028dc4597e312c96ada28fa19509f46559601a94e89dc630ac1ef152`。
- 服务、算子构建、EvalScope 使用镜像 `/opt/sglang/bin/python3`：Python 3.12.16、
  Torch 2.14.1+cu130。宿主机 Python 只做标准库编排/HTTP。
- 镜像 Humming 0.1.12，FlashInfer Python/cubin 0.7.2、jit-cache 0.7.2+cu130。
  **不使用定制 FlashInfer，也不要叠加 Humming 0.1.18 pylibs**。
- 一个 `deep_gemm._C`，包 `deep-gemm-h20==2.8.0+h20.5`。#473 基础 + dense donor
  + #521 mHC 的完整补丁、pins、构建和算子验证均随提交提供，见
  [dependencies/unified](dependencies/unified/README.md)。
- 三套均设置 `SGLANG_OPT_DSV41_SM90=1`、`SGLANG_OPT_FP8_WO_A_GEMM=1`、
  Engram host `per_rank`、`SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'`。
- `SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1` 跳过 Rust 图像预处理；本轮只测试语言模型。

从仓库根目录设置公共变量，三套按顺序执行，每次结果目录必须不存在：

```bash
cd /data1/fjmeng2/sglang-1108
export PREFILL_REVISION=perf/dsv41-h20
export PREFILL_RUNTIME_DIR=/data1/fjmeng2/dsv41-runtime
export PREFILL_UNIFIED_DEEPGEMM_SITE="$PREFILL_RUNTIME_DIR/repro/deepgemm-unified-20261126/site-v5"
unset PREFILL_DENSE_DEEPGEMM_SITE PREFILL_DEEPGEMM473_SITE PREFILL_DEEPGEMM521_SITE PREFILL_FLASHINFER_SITE
```

其他机器可先构建统一依赖，无需本机历史实验目录：

```bash
bash benchmark/deepseek_v41_prefill/dependencies/unified/build.sh /absolute/new-build-dir
export PREFILL_UNIFIED_DEEPGEMM_SITE=/absolute/new-build-dir/site
```

[run_h20.sh](run_h20.sh) 校验 site，拒绝叠加三个旧 DeepGEMM site 或 FlashInfer overlay。
服务使用 `SGLANG_OPT_DSV41_UNIFIED_DEEPGEMM=1`，
`SGLANG_DG_CACHE_DIR=/repro/cache/deepgemm-h20-unified`；只设 `DG_JIT_CACHE_DIR`
会被 SGLang 覆盖。每次导出指定 revision 的源码，记录实际模块路径、版本、扩展与
JIT 头文件哈希、容器配置及完整启动命令。

## 配置对照与完整运行入口

| 配置 | CP8EP8 | CP8EP1 | DP8EP8 |
| --- | --- | --- | --- |
| 并行 | TP8、prefill CP8 interleave、EP8 | TP8、prefill CP8 interleave、EP1 | TP8、dp_size=1、attn_dp_size=8、EP8、DP lm_head |
| prefill backend | flashmla_sparse_q8 | flashmla_sparse_q8 | 自动；吞吐阶段 fake prefill |
| chunk / max-prefill tokens | 262144 / 262144 | 65536 / 65536 | chunk 16384；max-prefill 为版本默认 |
| mem-fraction-static | 0.74 | 0.84 | 0.85 |
| max-running / decode graph max batch | 64 / 64 | 64 / 64 | 256 / 32 |
| CP8EP8 专用 gate | 开 | 关 | 关 |
| target MoE | 大 CP prefill 使用 DeepGEMM MXFP4 MegaMoE，其余回退 Humming | Humming，paired-W2 40 层 | Humming，原 pull 通信 |
| 专家配置 | dynamic dispatch、64 redundant experts、提交内固定 placement | EP1 | EP8，无 CP 专用 dispatch |
| mHC / dense multicast / Q8 TMA SWA | 开 / 开 / 开 | 开 / 关 / 关 | 关 / 关 / 关 |
| shared expert TP1 / paired-W2 | 开 / 关 | 开 / 开 | 关 / 关 |
| WQB→RoPE→FP8 | **关** | **关** | **关** |

CP8EP8 的 `--moe-runner-backend humming` 是正常配置：专用 gate 在符合条件的大
CP prefill 上调用 DeepGEMM MegaMoE，短批和 DSPARK 其他计算保留 Humming。
MegaMoE 使用 16K CP-local 工作区，32K 本地行分两次调用。不能仅凭 runner 参数
判断活跃算子。CP8EP1 和 DP8EP8 不启用这条 MegaMoE 路径。

**CP8EP8：64K/256K 吞吐、八卡 GPU profile、同实例全量 GSM8K**

```bash
PREFILL_OUTPUT="$PREFILL_RUNTIME_DIR/repro/h20-cp8ep8-wqb-off-$(date -u +%Y%m%dT%H%M%SZ)" \
  bash benchmark/deepseek_v41_prefill/run_h20.sh cp8ep8
```

**CP8EP1：64K/256K 吞吐、八卡 GPU profile、同实例全量 GSM8K**

```bash
PREFILL_OUTPUT="$PREFILL_RUNTIME_DIR/repro/h20-cp8ep1-wqb-off-$(date -u +%Y%m%dT%H%M%SZ)" \
  bash benchmark/deepseek_v41_prefill/run_h20.sh cp8ep1
```

**DP8EP8：fake-prefill 吞吐、八卡 GPU profile，再新启非 PD 服务运行全量 GSM8K**

```bash
PREFILL_OUTPUT="$PREFILL_RUNTIME_DIR/repro/h20-dp8ep8-wqb-off-$(date -u +%Y%m%dT%H%M%SZ)" \
  bash benchmark/deepseek_v41_prefill/run_h20.sh dp8ep8
```

DP8EP8 默认顺序为 `candidate accuracy`。如需合入前后 A/B：

```bash
export DECODE_BASELINE_REVISION=b1ae8c062ddfd809ab5d346795caf8f41aefdaf6
PREFILL_OUTPUT="$PREFILL_RUNTIME_DIR/repro/h20-dp8ep8-ab-$(date -u +%Y%m%dT%H%M%SZ)" \
  bash benchmark/deepseek_v41_prefill/run_h20.sh dp8ep8 --cases control candidate accuracy
```

`control` 使用 baseline 源码与镜像 stock DeepGEMM，`candidate`/`accuracy` 使用
目标源码与统一版；只跑精度可传 `--cases accuracy`。当前 `--dp-size 1 --attn-dp-size 8`
对应 Ant 老版本 `--dp-size 8 --enable-dp-attention`，脚本会核对服务规范化后的参数。

三套完整串行执行，失败立即退出：

```bash
bash benchmark/deepseek_v41_prefill/dependencies/unified/validate.sh \
  "$PREFILL_UNIFIED_DEEPGEMM_SITE" \
  "$PREFILL_RUNTIME_DIR/repro/h20-three-wqb-off-$(date -u +%Y%m%dT%H%M%SZ)" \
  "$PREFILL_REVISION"
```

## 完整启动脚本与评测内容

服务的全部环境变量、参数见 [prefill launch_server.sh](launch_server.sh) 和
[decode launch_server.sh](decode_regression/launch_server.sh)，均包含在本次提交中。
共同保留 DSPARK block5；CP 两套保留 CP decode attention TP 和 bounded replay；
DP8EP8 保留 Ant decode 的 FP8 lm_head、embedding pool 0、tokenizer workers 8、
total_tokens 调度、watchdog 1800 等配置。

Prefill 的完整测试脚本是 [run.sh](run.sh) + [bench.py](bench.py)：

- seed42 精确 input_ids，65536/262144 两档，输出 1 token，关闭 radix cache。
- 每档 flush cache、预热 10 次、测量 5 次，报告 median TTFT 与输入 tok/s。
- 校验服务端 prompt_tokens、cached_tokens=0、completion_tokens=1。
- 两档分别采全部 8 rank GPU-only profile，with_stack=false、record_shapes=false。
- 用同实例运行全量 GSM8K，精度数据不混入吞吐样本。

Decode 的完整脚本是 [run.py](decode_regression/run.py) +
[strict_decode.py](decode_regression/strict_decode.py)：

- 性能服务使用 decode 角色、Mooncake 后端；请求指定 `bootstrap_host=2.2.2.2`，运行时
  选择 fake handoff，不启动 prefill 服务，**不测真实 RDMA 传输性能**。
- 64 并发，seed42 精确 65536 输入、1024 输出、ignore_eos，模拟接受长度 4.5。
- 预热 1 轮、测量 3 轮，统计全部请求均在 decode 的公共区间中间 70% 的真实 token 速率。
  默认自然调度，保存实际 DP 分布，不假定各 rank 必然等长。
- 测时后另采 GPU-only profile，不纳入未 profile 的吞吐中位数。
- 精度另启非 PD 服务、真实 prefill，清除接受率模拟，不发送 fake handoff 字段。

如需对 CP8EP8 单独生成 CPU+GPU attribution trace（不运行吞吐和 GSM8K），使用：

```bash
PREFILL_OUTPUT="$PREFILL_RUNTIME_DIR/repro/h20-cp8ep8-cpu-trace-$(date -u +%Y%m%dT%H%M%SZ)" \
PREFILL_PROFILE_ONLY=1 PREFILL_RUN_GSM8K=0 \
PREFILL_PROFILE_ACTIVITIES=CPU,GPU PREFILL_PROFILE_WITH_STACK=1 \
PREFILL_PROFILE_RECORD_SHAPES=1 PREFILL_PROFILE_TIMEOUT=1200 \
  bash benchmark/deepseek_v41_prefill/run_h20.sh cp8ep8
```

该命令仍使用 CP8EP8 的完整启动参数和精确 64K/256K input IDs，每个长度只发送
一个 profile 请求，profile 请求不计入吞吐。输出位于
`profiles/{65536,262144}/`，包含八个 rank 的 `*.trace.json.gz`；CPU+stack trace
只用于调用来源和 shape 定位，不能与 GPU-only trace 的 kernel 时间直接比较。

`bench.py` 也支持 `--profile-activities CPU,GPU`、`--profile-with-stack`、
`--profile-record-shapes` 和 `--profile-only`，对应的宿主机环境变量由
`run.sh` 传入并记录在 `benchmark/protocol.json`。

三套 GSM8K 均为 EvalScope 1.15.0、全量 test split **1319 题**、默认 4-shot、
temperature0、max_tokens2048、并发64。完整 EvalScope 命令内嵌于上述 host 脚本，
[verify_gsm8k.py](verify_gsm8k.py) 核对逐题分数、1319 个唯一样本、零请求错误、
prompt/target 哈希。默认依赖与数据路径如下：

```text
/data1/fjmeng2/dsv41-runtime/fa3-pylibs          # sglang-kernel0.4.11、sgl-fa3 0.1.0
/data1/fjmeng2/dsv41-runtime/eval-pylibs         # EvalScope1.15.0与客户端依赖
/data1/fjmeng2/dsv41-runtime/datasets/gsm8k/main # train/test用于4-shot和评测
```

## 只检查脚本，不启动服务

这仅导出源码、完整工具与 source manifest，不查询 GPU、不启动模型、不评测：

```bash
for topology in cp8ep8 cp8ep1 dp8ep8; do
  PREFILL_OUTPUT="$PREFILL_RUNTIME_DIR/repro/h20-prepare-$topology-$(date -u +%Y%m%dT%H%M%SZ)" \
    bash benchmark/deepseek_v41_prefill/run_h20.sh "$topology" --prepare-only
done
```

容器内设置 `H20_PRINT_LAUNCH=1` 再执行相应 `launch_server.sh` 可打印最终命令；此模式
不导入 CUDA 模块、不生成运行时 manifest、不启动模型。完整运行则记录
`launch_command.txt`、`runtime_manifest.json`、`container.json`、`source_manifest.json`。

CP 输出：`benchmark/summary.md`、`profiles/{65536,262144}/`、`gsm8k-summary.json`、
`server.log`。DP 输出分 `candidate/`、`accuracy/`（可选 `control/`）：
`candidate/timing/summary.json`、`candidate/profiles/gpu/`、`accuracy/gsm8k-summary.json`。
脚本清理自身容器，不终止已有服务。
