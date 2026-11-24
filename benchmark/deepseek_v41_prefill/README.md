# DeepSeek-V4.1-Flash prefill 完整复现

本目录复现 8 × H20 上的 DeepSeek-V4.1 prefill。默认使用 Ant 对齐配置，包含
TP8 / prefill CP8 interleave / **EP1**、Engram host `per_rank`、Humming 0.1.12、
DSPARK block size 5 和 bounded replay。启动参数见 [launch_server.sh](launch_server.sh)。

## 一键运行（宿主机）

要求 Docker/NVIDIA Container Toolkit、Git、Python 3.10+、`nvidia-smi`，8 卡空闲，
以及完整的 DeepSeek-V4.1-Flash-1017 权重。首次加载/JIT 会耗时数分钟；默认等待上限 1 小时。
不要与其他 GPU 工作负载同时测量。脚本遇到端口占用或已有 GPU 计算进程时会退出。

```bash
cd /data1/fjmeng2/sglang-1108
PREFILL_OUTPUT=/data1/fjmeng2/dsv41-runtime/repro/$(date -u +%Y%m%dT%H%M%SZ) \
  bash benchmark/deepseek_v41_prefill/run.sh
```

脚本准备独立源码、检查依赖、创建专用容器并启动服务，然后依次测试 64K/256K、
采集 GPU-only profile。结束时保存日志并删除自身创建的容器；输出目录与 JIT cache 保留。
设置 `PREFILL_KEEP_SERVER=1` 可保留服务和容器。脚本不会终止已有服务；请先结束之前的测试。

默认配置及可覆盖项：

| 变量 | 默认值/用途 |
| --- | --- |
| `PREFILL_MODEL_DIR` | `/data1/fjmeng2/models/DeepSeek-V4.1-Flash-1017`，只读挂载 |
| `PREFILL_IMAGE` | `lmsysorg/sglang:latest`，严格校验下述 image ID |
| `PREFILL_REVISION` | `HEAD`，导出该已提交 revision 的源码 |
| `PREFILL_OUTPUT` | `/data1/fjmeng2/dsv41-runtime/repro/<UTC时间>`，必须不存在 |
| `PREFILL_DEPS_DIR` | `/data1/fjmeng2/dsv41-runtime/fa3-pylibs` |
| `PREFILL_CACHE_DIR` | `/data1/fjmeng2/dsv41-runtime/sglang-cache-migrated` |
| `PREFILL_PORT` | `30002` |
| `PREFILL_CONTAINER` | `dsv41-prefill-<UTC时间>`，必须不存在 |
| `PREFILL_READY_TIMEOUT` | `3600` 秒 |
| `PREFILL_KEEP_SERVER` | `0`，设为 `1` 会在脚本退出时保留服务，包括失败时 |
| `PREFILL_CHUNK_SIZE` | `16384`，同时设置 `--chunked-prefill-size` 和 `--max-prefill-tokens` |
| `PREFILL_ATTENTION_BACKEND` | `auto`，由服务选择。设为 `flashmla_sparse_q8` 时传入 `--dsv4-prefill-backend flashmla_sparse_q8` |
| `PREFILL_EP_SIZE` | `1`，传给 `--ep-size`。W2 分片要求 EP1 |
| `PREFILL_MEM_FRACTION` | `0.84`，传给 `--mem-fraction-static` |
| `PREFILL_HUMMING_NTAIL` | `0`，对应容器内的 `SGLANG_DSV41_HUMMING_PREFILL_NTAIL`。设为 `1` 会让支持的 Humming 0.1.12 在统一服务的 `mix` 角色选用 `v0_1_12_ntail` overlay |
| `PREFILL_SHARED_TP1` | `0`，对应 `SGLANG_SHARED_EXPERT_TP1`。设为 `1` 为每个 rank 复制 shared expert，CP prefill 在本地行上计算 |
| `PREFILL_W2_PAIR_LAYERS` | `0`，对应 `SGLANG_EXPERIMENT_DSV41_W2_PAIR_LAYERS`。设为 `N` 时，为满足条件且编号小于 `N` 的目标层加载另一份 W2 分片。实测使用 `40` |

容器使用 host network/IPC、无限 memlock、`IPC_LOCK`，不添加 `SYS_NICE`。
不修改宿主机 THP/NUMA 配置。可预先复用现有依赖目录；目录缺少 `sgl_kernel` 或
`sgl_fa3` 时，脚本在镜像中用 `--no-deps --target` 安装 `sglang-kernel==0.4.11`
和 `sgl-fa3==0.1.0`（需要包源网络）。运行时会核验实际版本，依赖不匹配直接失败。
不要把 `humming-pylibs` 0.1.18 加到 `PYTHONPATH` 前面；这会禁用已对齐的 0.1.12 overlay/tuning。

## 复现 dd6de41 的 W2 分片与 FP8 传输

W2 分片适用于本实验的 SM90 TP8 / CP8 interleave / EP1、Humming FP8 输入、
FP32 累加和 TP1 shared expert。运行时要求 CP 对称内存、每 rank 至少 2,048 行、
Top-6 路由，以及使用 `block_m=64` 的 indexed GEMM。短批、CUDA Graph capture
等不满足条件的批次和 decode 继续使用常规 W2。设置 40 层时，额外的 W2 权重
占用约 11.21 GiB/rank，常规 W2 权重也保留在显存中。

执行下方命令可用同一服务完成 64K/256K 吞吐、八卡 GPU-only profile 和全量 GSM8K。
在宿主机的仓库根目录运行。先在评测环境中准备 EvalScope 1.15.0 及 GSM8K 数据集，
把 `GSM8K_DATASET_DIR` 指向本地 GSM8K `main` 数据目录。命令默认使用本项目的数据路径。

```bash
(
  set -euo pipefail
  run_id=$(date -u +%Y%m%dT%H%M%SZ)
  export PREFILL_REVISION=dd6de419d5da276254874c1df7e45bb0a6e10a26
  export PREFILL_OUTPUT="/data1/fjmeng2/dsv41-runtime/repro/w2-fp8-$run_id"
  export PREFILL_PORT=30002 PREFILL_KEEP_SERVER=1
  export PREFILL_CHUNK_SIZE=65536 PREFILL_ATTENTION_BACKEND=flashmla_sparse_q8
  export PREFILL_EP_SIZE=1 PREFILL_MEM_FRACTION=0.84
  export PREFILL_HUMMING_NTAIL=1 PREFILL_SHARED_TP1=1 PREFILL_W2_PAIR_LAYERS=40
  dataset_dir=${GSM8K_DATASET_DIR:-/data1/fjmeng2/dsv41-runtime/datasets/gsm8k/main}
  dataset_args=$(python3 -c '
import json
import sys
print(json.dumps({"gsm8k": {"local_path": sys.argv[1], "subset_list": ["default"], "eval_split": "test"}}))
' "$dataset_dir")

  [[ ! -e "$PREFILL_OUTPUT" ]]
  cleanup() {
    if [[ -s "$PREFILL_OUTPUT/container_id.txt" ]]; then
      local container_id
      container_id=$(cat "$PREFILL_OUTPUT/container_id.txt")
      docker logs "$container_id" > "$PREFILL_OUTPUT/server.log" 2>&1 || true
      docker rm -f "$container_id" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  bash benchmark/deepseek_v41_prefill/run.sh
  evalscope eval \
    --model /models/DeepSeek-V4.1-Flash-1017 --model-id w2-fp8 \
    --api-url "http://127.0.0.1:$PREFILL_PORT/v1" --api-key EMPTY \
    --eval-type openai_api --model-task text_generation --datasets gsm8k \
    --dataset-args "$dataset_args" --eval-batch-size 64 --timeout 1800 \
    --generation-config '{"temperature":0.0,"max_tokens":2048}' \
    --work-dir "$PREFILL_OUTPUT/gsm8k" --no-timestamp \
    2>&1 | tee "$PREFILL_OUTPUT/gsm8k.log"
)
```

GSM8K 使用 EvalScope 1.15.0 默认的 4-shot prompt，评测 test split 全部 1,319 题。
查看 `gsm8k/reports/w2-fp8/gsm8k.json` 的准确率及 `execution_summary`，
并核对 `requested=succeeded=1319`、`errored=0`、`incomplete=false`。
原实验宿主机也可使用
`/data1/fjmeng2/dsv41-runtime/repro/prefill-plus20-20261124/run_prefill_with_gsm8k.sh`。
保留上面的性能配置，指定独立的 `PREFILL_OUTPUT` 和 `PREFILL_CONTAINER`，
以仓库根目录为参数调用该脚本。它会额外生成 `gsm8k-summary.json`。

`runtime_manifest.json` 记录 `SGLANG_EXPERIMENT_DSV41_W2_PAIR_LAYERS=40`、
`SGLANG_SHARED_EXPERT_TP1=1` 和 `SGLANG_DSV41_HUMMING_PREFILL_NTAIL=1`。
服务日志中的 `Prepared paired W2 shard` 表示备用权重已加载。64K profile 中
`_quantize_and_push_pair`、`_sum_colored_slots` 和 Humming W2
`Shape<0u, 2560u, 576u>` 可确认请求实际走了分片路径。当前 `compare_profiles.py`
将这三类 kernel 计入 `Other`，分析分片时可在原始 trace 中按名称查看。
[优化记录](../../OPTIMIZATION_PLAN.md#2026-11-24-cp8-prefill-w2-output-sharding-and-fp8-transport)

## CP8/EP8 的 128K chunk

用 128K chunk 测 256K 时，服务处理两个 131072-token chunk。每个大块
有 16384 个 CP-local 行，调用一次 DeepGEMM MegaMoE。64K 和短批保留
原有的 8192 行工作区，包括 DSPARK 验证图捕获。

本地复现使用 DeepGEMM #473 的隔离 wheel
`8cad9399351e58d070c90546ad98b44900523cfdac8839ed02375ed64e9fb081`。
wheel 与构建记录在实验目录的 `upstream-20261125/`。安装目录保留 wheel 的
`.dist-info`，模块目录命名为 `deep_gemm_mega`，以便和镜像的 dense
`deep_gemm` 同时加载。

`PREFILL_DEEPGEMM473_16K=1` 将
`SGLANG_OPT_DSV41_SM90_MEGAMOE_16K_PREFILL=1` 传给服务。默认值为 0，
适用于原版 DeepGEMM #473 的 8192 行工作区。

```bash
runtime=/data1/fjmeng2/dsv41-runtime
results="$runtime/repro/prefill-110k-20261124"
repo="$runtime/worktrees/prefill-deepgemm473-chunk128-full"
PREFILL_EXPERIMENT_REPO="$repo" \
PREFILL_REVISION="$(git -C "$repo" rev-parse HEAD)" \
PREFILL_RUN_NAME=deepgemm473-chunk128-repro \
PREFILL_CHUNK_SIZE=131072 \
PREFILL_DEEPGEMM473_16K=1 \
PREFILL_DEEPGEMM473_SITE="$results/upstream-20261125/deepgemm-473-prefill-tuning-site" \
bash "$results/run_deepgemm473_serving.sh"
```

脚本使用 8×H20、TP8/CP8 interleave/EP8 与 `per_rank` host Engram，
关闭 radix cache，先测 64K 和 256K 并采集八卡 GPU-only trace，再用同一
实例跑全量 GSM8K。

## 镜像与源码

验证镜像的不可变 image ID：

```text
sha256:f3a303d4028dc4597e312c96ada28fa19509f46559601a94e89dc630ac1ef152
```

本机对应的 registry digest：
`mirror.ccs.tencentyun.com/lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`。
可通过 `PREFILL_IMAGE` 指定该 digest 或本地 image ID。`latest` 会变化，脚本默认拒绝
与已验证 image ID 不同的镜像；`PREFILL_ALLOW_IMAGE_MISMATCH=1` 仅跳过 image ID 检查，
仍检查依赖版本，应作为新环境实验记录。

兼容性修复已直接提交到分支源码中。`run.sh` 默认解析启动时的 `HEAD`，
通过 `git archive` 将该 commit 原样导出到独立的 `source/` 目录后运行。
可用 `PREFILL_REVISION=<完整commit>` 固定复现版本；请选择包含这些兼容修复的版本。
脚本不再应用 patch，也不依赖工作树中未提交的模型代码。
`source_manifest.json` 保存 `source_revision` 和完整源码逐文件 SHA256，
同时记录本次使用的复现脚本哈希。

本页只验证指定的 prefill 配置；fake-prefill decode 和其他并行布局需要单独测试。

仅准备源码、不用 GPU/Docker：

```bash
PREFILL_OUTPUT=/tmp/dsv41-prefill-source \
  bash benchmark/deepseek_v41_prefill/run.sh --prepare-only
```

验证运行时：torch `2.14.1+cu130` / CUDA `13.0`，humming-kernels `0.1.12`，
sglang-kernel `0.4.11`，sgl-fa3 `0.1.0`，sgl-deep-gemm `0.2.3.post2`，
flashinfer-python `0.7.2`，transformers `5.19.0`，triton `3.8.0`，tilelang `0.1.12`。
默认配置的服务器日志应包含 `v0_1_12_bm32` overlay 和 H20-3e tuned tables。
上述 W2 实验配置使用 `v0_1_12_ntail` overlay，启动时会记录实际加载的 overlay 和调优表。

## 严格测试口径

- `random.Random(42)`，ID 闭区间 `[10000,100000]`，直接发 `input_ids`；
  长度精确 **65,536 / 262,144**，64K 是长输入，单请求串行、并发 1。
- 启动时 `--disable-radix-cache`；每档开始 flush cache。
  每条请求必须满足 `prompt_tokens=目标长度`、`cached_tokens=0`、`completion_tokens=1`，
  否则整个实验失败，不能把无效样本计入结果。
- `temperature=0, ignore_eos=true, max_new_tokens=1`，每档预热 10 次，再采样 5 次。
- TTFT 使用 `perf_counter`，从 **JSON 序列化之后** 到首个包含生成 token 的 SSE 事件。
  包括 HTTP 上传、调度、prefill 和首 token 返回开销；不是纯 GPU kernel latency。
- 吞吐为 `输入长度 / median TTFT`，不乘 8；不是多请求服务的饱和吞吐。
  同时保存各次 TTFT/E2E、min/max、输入 SHA256 和完整 server info。
- 单独请求做 profile：`activities=[GPU]`、`with_stack=false`、`record_shapes=false`。
  不将 profile 请求算入性能样本，等待 TP0..7 的 gzip traces 全部完成再成功退出。

仅测已经按同样参数启动的服务器（不启动/关闭服务，客户端仅依赖 Python 标准库）：

```bash
python3 benchmark/deepseek_v41_prefill/bench.py \
  --base-url http://127.0.0.1:30002 \
  --lengths 65536 262144 --warmup 10 --repeat 5 \
  --output /data1/fjmeng2/dsv41-runtime/results/new-prefill-repro \
  --profile-dir /data1/fjmeng2/dsv41-runtime/profiles/new-prefill-repro
```

输出目录必须不存在。profile 目录必须在客户端和服务端都可见；挂载路径不同时，
另传 `--server-profile-dir <服务端的同一目录>`。不需要 profile 时省略 `--profile-dir`。
管理端口默认使用本实验的 `profile` key，也支持 `SGLANG_ADMIN_API_KEY` / `--admin-key`。
客户端校验 CP/TP/EP、cache 和关键服务参数；直接连接已有服务时，镜像/源码/环境变量需
由启动方记录，一键模式则自动保存这些信息。直接测 W2 实验服务时，给客户端追加
`--expected-chunk-size 65536 --expected-attention-backend flashmla_sparse_q8
--expected-ep-size 1 --expected-mem-fraction 0.84`。

## 结果与全部 rank 的 kernel 对比

一键模式输出：

```text
source_manifest.json, runtime_manifest.json, image.json, container.json
launch_command.txt, nvidia-smi.txt, server.log, client.log
benchmark/protocol.json, server_info.json, requests.jsonl
benchmark/measurements.json, summary.json, summary.md, completed.json
benchmark/profile_requests.json
profiles/{65536,262144}/*-TP-{0..7}.trace.json.gz
```

其中 `benchmark/` 那一行列出的所有文件均在 `benchmark/` 子目录。
`requests.jsonl` 包含全部 warmup/sample；`measurements.json` 只含性能样本。
只有生成 `completed.json` 且 `success=true` 才表示客户端完整通过。

Ant 与本地使用同口径生成两组 profile 后，对每档长度分别运行：

```bash
python3 benchmark/deepseek_v41_prefill/compare_profiles.py \
  --reference /data1/fjmeng2/dsv41-runtime/profiles/1123-ant-gpu/65536 \
  --candidate /data1/fjmeng2/dsv41-runtime/repro/<本次目录>/profiles/65536 \
  --output /data1/fjmeng2/dsv41-runtime/results/repro-compare/65536
# 256K 同样运行一次，将路径中的 65536 换为 262144。
```

输出 `comparison.md` / `gpu_summary.json`：kernel family 耗时/调用次数的每 rank 均值，
以及各 rank GPU span、active union、idle、CP barrier 时间和 barrier 顺序明细。
**kernel 累计时间存在重叠，不能相加当作 TTFT**；CPU+stack profile 只适合源码定位，
不能和这些 GPU-only 时间混比。

此前同口径重启确认值（不是所有机器的性能保证）：

| 输入 token | 本地 median TTFT | 本地输入 tok/s | Ant median TTFT |
| ---: | ---: | ---: | ---: |
| 65,536 | 857.078 ms | 76,464.49 | 857.769 ms |
| 262,144 | 3528.397 ms | 74,295.49 | 3533.834 ms |

完整 GSM8K 与性能数据见 [OPTIMIZATION_PLAN.md](../../OPTIMIZATION_PLAN.md)。
本目录只复现 prefill；decode 仍应另用 fake prefill 测试。
