# DeepSeek-V4.1-Flash prefill 完整复现

当前统一 DeepGEMM 的 CP8EP8、CP8EP1、DP8EP8 三套完整测试入口见
[H20_TESTING.md](H20_TESTING.md) 和 [run_h20.sh](run_h20.sh)。三套均关闭
WQB→RoPE→FP8 融合；本轮按要求没有重测，以下历史成绩不能代表关闭后的配置。

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

## SM90 wq_b 与 RoPE/Q8 融合

以下为历史实验说明；当前测试脚本显式设为 0，不再接受宿主机开关重新启用。

在 CP8 和 Q8 sparse prefill 下，对 8192、16384、32768 个 CP-local 行可通过
SGLANG_EXPERIMENT_DSV41_SM90_WQB_Q8_ROPE=1 将 wq_b block32
DeepGEMM、BF16 舍入、局部 64 维 RoPE 和 FP8 Q staging 合并。该开关需要
挂载单独构建的 DeepGEMM 扩展到 /repro/dense_deepgemm，仅用于 SM90
CP8 的 wq_b 形状，其他形状继续走原路径。

完整复现实验保存在
/data1/fjmeng2/dsv41-runtime/repro/prefill-110k-local-20261125/wqb-q8-rope/。
其中 run_serving.sh 0 是关闭开关的同源对照，run_serving.sh 1 是候选，
两者均执行严格 64K/256K 测试、八卡 GPU-only profile 和全量 GSM8K。

```bash
experiment=/data1/fjmeng2/dsv41-runtime/repro/prefill-110k-local-20261125/wqb-q8-rope
bash "$experiment/run_serving.sh" 0
bash "$experiment/run_serving.sh" 1
```

候选运行还要求将 site/ 中的 _C.so 和 Python 包挂载为
PREFILL_DENSE_DEEPGEMM_SITE。当前实测候选为 64K 98,323 tok/s、
256K 94,721 tok/s，同源对照为 64K 97,674 tok/s、256K 94,343 tok/s。
候选 GSM8K 为 1280/1319，零请求错误。首次候选运行的 GSM8K 为 1281/1319，
重复运行采用同一份提示和目标哈希，结果存在题目级波动。

本轮扩展文件为
/data1/fjmeng2/dsv41-runtime/repro/prefill-110k-local-20261125/wqb-q8-rope/site/，
`deep_gemm/_C.so` 的 SHA256 为
d4b637af2991c44d329f4e909add54fd7247cb95dfe1971c30a26935fea04967。
扩展源代码和构建脚本在同一实验目录的 `DeepGEMM/` 与 `build.sh`，运行前需
先构建 `_C.so`，再把 `site/` 传给 `PREFILL_DENSE_DEEPGEMM_SITE`。

## CP8 rank-major KV gather

提交 d03fb60653 在满足等长 CP8 分片条件时，让 SWA KV gather 直接写入
rank-major 槽位，并把对应映射传给 V4 KV 写回。SM90、V4 page layout、8 个
CP rank、无 gather index、各 rank 行数相等的请求才会启用该路径，其他请求
继续使用原有全局顺序 gather。

实验目录
`/data1/fjmeng2/dsv41-runtime/repro/prefill-110k-local-20261125/wqb-q8-rope/`
中的 `run_kv-serving.sh` 和 `run_kv-repeat.sh` 使用相同的 wq_b Q8 epilogue
扩展，分别完成严格吞吐、八卡 profile 和全量 GSM8K。

## CP8/EP8 的 256K 单块 prefill

在 `PREFILL_DEEPGEMM473_16K=1` 与 Q8 sparse prefill 下，单个 262144-token chunk
在每个 CP rank 上有 32768 行。路由、C2 投影池化和 wq_b/RoPE 在整块上处理，MegaMoE
复用 16384 行工作区并调用两次。镜像、扩展与专家 placement 随运行清单记录。

在宿主机从本分支仓库根目录执行。脚本会在同一容器中完成两档严格吞吐、八卡 GPU-only
profile、全量 GSM8K。历史长提示探针保留在原实验目录，不属于当前脚本。可通过 `PREFILL_REVISION` 固定源码提交，
通过 `PREFILL_OUTPUT` 指定独立的结果目录。

```bash
bash benchmark/deepseek_v41_prefill/run_cp8_megamoe.sh 262144
```

| Chunk | 64K tok/s | 256K tok/s | GSM8K |
| ---: | ---: | ---: | ---: |
| 65536 | 98477.12 | 94869.91 | 1279/1319 |
| 131072 | 98355.03 | 95941.48 | 1280/1319 |
| 262144 | 98430.25 | 96216.44 | 1279/1319 |
| 262144 复测 | 98383.98 | 96143.06 | 1278/1319 |

四次运行均零 GSM8K 请求错误，评测提示和目标哈希一致。详细数据在
`/data1/fjmeng2/dsv41-runtime/repro/prefill-110k-local-20261125/` 下的
`wqb-q8-rope/kv-repeat-1`、`rank-major-chunk-131072`、`rank-major-chunk-262144-v2`
与 `rank-major-chunk-262144-repeat`。

## CP8 SM90 Mega mHC

DeepGEMM #521 的 Hopper 算子将 mHC post、使用上一子层 pre 系数的 collapse、
RMSNorm 和下一子层的 mix/Sinkhorn 合并。CP8 eager prefill 的 2048 至 32768
本地行可使用这条 BF16 输出路径。运行入口通过
`PREFILL_DEEPGEMM521_SITE` 挂载隔离扩展并设置
`SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1`。该目录包含
`deep_gemm_mhc`，来源为 DeepGEMM `410ce17fb3cc`，JIT 缓存位于
`/repro/cache/deepgemm521`。

```bash
runtime=/data1/fjmeng2/dsv41-runtime
experiment=$runtime/repro/prefill-110k-local-20261125/mega-mhc-521
PREFILL_DEEPGEMM521_SITE="$experiment/site" \
  bash benchmark/deepseek_v41_prefill/run_cp8_megamoe.sh 262144
```

| 256K chunk 测试 | 64K tok/s | 256K tok/s | GSM8K |
| --- | ---: | ---: | ---: |
| 首轮 | 99598.37 | 97350.52 | 1281/1319 |
| 同源码复测 | 99599.02 | 97296.19 | 1278/1319 |

两次全量评测的提示和答案哈希一致，请求错误均为零。64K 八卡平均 GPU span
从 654.309 ms 降到 646.303 ms，每 rank 调用融合算子 38 次。
算子对照的 BF16 归一化结果相对 L2 差异最高为 2.01×10⁻⁵。
四组长提示中，三组的下一 token 与 256K chunk 对照一致，seed 43 的 256K
提示在多次运行中出现不同的下一 token。
原始报告、模型长提示探针及 wheel 构建材料见上文的实验目录。

## SM90 密集投影 TMA multicast

`PREFILL_DENSE_MULTICAST=1` 选择已验证的密集投影 DeepGEMM
扩展，同时挂载 Mega mHC 扩展。对于 H20 的 block32 GEMM，
`wo_b` 在 8192 和 32768 个本地行使用激活 multicast，
融合 RoPE/Q8 的 `wq_b` 在 32768 行使用权重 multicast。
两组目标形状的算子对照均逐位一致。

```bash
PREFILL_DENSE_MULTICAST=1 bash benchmark/deepseek_v41_prefill/run_cp8_megamoe.sh 262144
```

| 256K chunk 测试 | 64K tok/s | 256K tok/s | GSM8K |
| --- | ---: | ---: | ---: |
| Mega mHC 对照复测 | 99599.02 | 97296.19 | 1278/1319 |
| Dense multicast | 99779.12 | 97470.84 | 1282/1319 |
| Dense multicast 复测 | 99797.03 | 97462.70 | 1280/1319 |

两次候选都完成了全量 GSM8K，零请求错误，提示和目标哈希与对照一致。
两次的四组长提示下一 token 相同。64K 八卡 profile 的 `wo_b`
每 rank 20 次均使用 cluster 2，累计时间由 50.236 ms 降到 49.549 ms。
256K 的 `wo_b` 和 Q8 `wq_b` 每 rank 各 20 次均使用 cluster 2，
累计时间由 199.135 ms 和 221.744 ms 降到 196.411 ms 和 220.784 ms。

扩展源代码、`build.sh` 和两次评测在
`/data1/fjmeng2/dsv41-runtime/repro/prefill-110k-local-20261125/dense-multicast/`。
可在宿主机运行该目录的 `build.sh` 构建 `site/deep_gemm/_C.so`。实测二进制的
SHA256 为 `a225af5048055b3c7acdbee8df19fbab8abb5f83cfa35fff850b176746432a70`。
吞吐、精度、长提示和八卡 profile 明细在 `serving-v2/` 与 `serving-v2-repeat/`。

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

## 合入后的配置隔离与 99.9K 复现

`run.sh` 默认保留 EP1 / 16K chunk，生产参数不因历史 EP8 合入而自动改变。
EP1 paired-W2 配方仍使用前文的 64K chunk、ntail、shared TP1、W2 pair layers 40，
并可挂载 Mega mHC。DP8/EP8 decode 使用原 decode 参数，prefill 专用开关默认关闭。

新增运行时开关 `SGLANG_OPT_DSV41_CP8_EP8_PREFILL=1` 还要求 CP8、EP8、
attention-DP1、attention-TP1，才启用历史 rank-major KV/C2 与 CP-local MegaMoE。
EP1 和 DP8/EP8 即使误开此开关，也不会选中该路径。WQB 融合另需原 WQB 开关。

最高历史记录为 `megamoe-32k/`：64K **99922.39**、256K **97555.53 tok/s**，
GSM8K **1282/1319**；其前一个两次 16K MoE 调用的版本为 99803.78/97538.67。
它们都是 TP8/CP8/EP8、448 个 physical expert slots、256K chunk、memory .74、
host Engram per_rank、cache off、串行请求/output1、10 warmups/5 repeats。
历史最高数值不是新集成提交的测量结果，具体新结果见 OPTIMIZATION_PLAN.md。
注意 `megamoe-32k` 是历史运行目录名；已核对其归档源码及 256K 的 GPU trace，
实际使用 16384-row pool（32K local rows 分两次调用），不是 32768-row pool。
后续 `megamoe-32k-v2/v3` 的实际 32K pool 尝试均因 early-combine bitmap 限制失败。
本入口保留经过验证的 16K pool，不启用失败的 32K pool。

完整启动、两档吞吐、16 份 profile、同一服务的全量 GSM8K 均由仓库内入口执行：

```bash
PREFILL_DENSE_MULTICAST=1 \
PREFILL_OUTPUT=/data1/fjmeng2/dsv41-runtime/repro/choose-new-run-dir \
bash benchmark/deepseek_v41_prefill/run_cp8_megamoe.sh 262144
```

GSM8K 在服务容器内使用 `/opt/sglang/bin/python3` 执行，EvalScope site 与数据集
只读挂载，报告写 `/repro/results/gsm8k`，保证容器删除后结果仍可读取。
`verify_gsm8k.py` 核对 1319 个唯一评测样本、execution_summary 与逐题分数。
默认使用镜像内 FlashInfer 0.7.2；DeepGEMM MegaMoE 不要求 fi075 目录。
如显式提供 `PREFILL_FLASHINFER_SITE`，manifest 单独记录并验证该可选依赖。

所有定制 DeepGEMM 的源码 pin、patch、构建与哈希校验见
[dependencies/README.md](dependencies/README.md)。本入口的外部 site 可以用
`PREFILL_DEEPGEMM473_SITE`、`PREFILL_DENSE_DEEPGEMM_SITE`、`PREFILL_DEEPGEMM521_SITE`
覆盖。三份定制包的修改都随提交保存在本目录，不能用 stock wheel 替代后仍称同一实验。

离线验证时可设置 `PREFILL_REVISION` 为 `git write-tree` 生成的 tree，以便先测试暂存
源码再提交；默认仍从 `HEAD` 导出。源码的逐文件 SHA256 与 source label 均归档，
最终提交必须与经过测试的运行时代码一致。

## DP8/EP8 decode 回归与全量 GSM8K

CP8/EP8 迁移必须另外检查 attention-DP8/EP8。以下入口从 Git revision/tree
导出不可变源码，依次启动 GSM8K、基线 decode、候选 decode；GPU 被占用时拒绝启动：

```bash
python3 benchmark/deepseek_v41_prefill/decode_regression/run.py \
  --revision HEAD --baseline 82bebb1d34 \
  --cases accuracy control candidate \
  --output /data1/fjmeng2/dsv41-runtime/repro/choose-new-dp8ep8-run
```

只补跑精度可用 `--cases accuracy`。使用容器内 Python3.12，统一
TP8/attention-DP8/EP8（当前版本 `dp_size=1, attn_dp_size=8`）、Humming FP8、
Engram host per_rank、DSPARK block5、max_running256、decode graph32。
GSM8K 使用非 PD 模式、真实 prefill、关闭接受率模拟、4-shot、temperature0、
max_tokens2048、并发64、全量1319题。`accuracy/gsm8k-summary.json` 核对逐题分数、
请求成功数和 prompt/target 哈希，`server_info.json` 与 runtime manifest 记录实际配置。

decode 吞吐使用 PD decode 角色和 **fake prefill**：64×65536精确input_ids，
1024输出，模拟接受率4.5，1轮预热/3轮测量；随后另采GPU-only的8-rank profile。
此模拟接受率只用于吞吐测试，不用于GSM8K。`control`/`candidate` 的
`timing/summary.json` 报告全请求都在decode区间中间70%的实测token速率，
`profiles/gpu/` 保存kernel trace。不同配置或profile开销下的数值不能直接混比。

完整迁移验证可一次运行（同一不可变源码，任一步失败立即退出）：

```bash
bash benchmark/deepseek_v41_prefill/validate_migration.sh \
  /data1/fjmeng2/dsv41-runtime/repro/choose-new-migration-run HEAD
```

历史提交到当前配置的映射见 [MIGRATION.md](MIGRATION.md)。

### Single-source H20 DeepGEMM

The consolidated build is documented in [dependencies/unified/README.md](dependencies/unified/README.md). It builds dense, SM90 MXFP4 MegaMoE and SM90 Mega mHC into one `deep_gemm._C`, selected through `PREFILL_UNIFIED_DEEPGEMM_SITE`. The complete deployment recipes are in [H20_TESTING.md](H20_TESTING.md). They disable WQB/RoPE/FP8 fusion and reject legacy DeepGEMM/FlashInfer overlays. This integration has no new performance claim; the user requested no benchmark or GSM8K rerun for the script update. Historical measurements remain in `dependencies/unified/VALIDATION.md` and `OPTIMIZATION_PLAN.md`.
