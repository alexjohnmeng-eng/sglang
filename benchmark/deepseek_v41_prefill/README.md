# DeepSeek-V4.1-Flash prefill 完整复现

本目录复现 `perf/dsv41-ant-prefill-alignment` 的 Ant 对齐实验：8 × H20，
TP8 / prefill CP8 interleave / **EP1**，Engram host `per_rank`，Humming 0.1.12，
DSPARK block size 5，bounded replay。完整启动参数见 [launch_server.sh](launch_server.sh)。

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
| `PREFILL_KEEP_SERVER` | `0`；设为 `1` 则结束后保留服务，包括失败时 |

容器使用 host network/IPC、无限 memlock、`IPC_LOCK`，不添加 `SYS_NICE`。
不修改宿主机 THP/NUMA 配置。可预先复用现有依赖目录；目录缺少 `sgl_kernel` 或
`sgl_fa3` 时，脚本在镜像中用 `--no-deps --target` 安装 `sglang-kernel==0.4.11`
和 `sgl-fa3==0.1.0`（需要包源网络）。运行时会核验实际版本，依赖不匹配直接失败。
不要把 `humming-pylibs` 0.1.18 加到 `PYTHONPATH` 前面；这会禁用已对齐的 0.1.12 overlay/tuning。

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
测试过程中服务器日志应包含 `v0_1_12_bm32` overlay 和 H20-3e tuned tables。

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
由启动方记录，一键模式则自动保存这些信息。

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

完整 GSM8K 与性能优化记录见仓库根目录 `OPTIMIZATION_PLAN.md`。
本目录只复现 prefill；decode 仍应另用 fake prefill 测试。
