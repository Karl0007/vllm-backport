# 2026-09-05 单卡 27B 最优配置定版基准(backport 容器 v0.11.2-20-g64a2fa524,
# 170HX 单卡 185 W,官方 vllm bench serve,random in128/out512,greedy)。
#
# 结论(数字全部实测,launcher = {runtimes_root}/vllm-backport/launch-27b-kkx99.sh):
#
# | 配置                                          | c1 tok/s | c4    | c8    | prefill 8k | TPOT c1/c4 |
# |-----------------------------------------------|----------|-------|-------|------------|------------|
# | AWQ, 无投机                                    | 50.69    | 181.8 | —     | —          | 19.3/21.1 ms |
# | AWQ, MTP4(同 checkpoint 无 MTP 头,加载即退化)  | —        | —     | —     | —          | 不支持 |
# | W8A16-MTP target + MTP4                        | 80.17    | 246.6 | —     | —          | 12.0/14.7 ms |
# | AWQ-eq8emb + MTP4                              | 98.44    | —     | —     | —          | 9.3 ms |
# | AWQ-eq8emb + MTP8                              | 71.99    | —     | —     | —          | 13.3 ms |
# | AWQ + DFlash2-W4A16 k7 + async (max-seqs 8)    | 131.38   | 282.3 | 380.9 | 1626       | 6.7/11.9 ms |
# | + eq8emb(embed/lm_head int8)                   | 125.5→127.0 | 295.6 | 385.5* | 1626    | 7.0/12.0 ms |
# | + max-seqs 16                                  | 117.9    | **309.7** | **385.5** | 1626 | 7.3/11.3 ms |
# | + k=6                                          | 115.8    | 295.6 | —     | —          | 7.6/12.0 ms |
# | GMU 0.96 / batched 16384 / 4096                | ±0.5%    | ±2%   | —     | —          | — |
#
# *c8 385.5 是 ss16 那一轮;max-seqs 8 时 c8=380.9。
#
# 关键判定:
# 1. **胜出配置** = AWQ-eq8emb + DFlash2-W4A16 k=7 + --async-scheduling +
#    --max-num-seqs 16 + max-num-batched-tokens 8192 + GMU 0.92 + bf16 KV + FA2 +
#    cudagraph FULL_AND_PIECEWISE。c1 127,c4 310,c8 386,prefill 8k 1626 tok/s。
# 2. **probabilistic draft sampling 在 backport 上是负优化**(-7%):syv #73 的
#    分母缺陷不存在于这个 fork,greedy 草稿采样本来就是对的。接受率 3.39 不变。
# 3. **W4A8(int8 激活) 在这个 checkout 上不可用**:MarlinLinearKernel 硬断言
#    weight_type==uint4b8(即 GPTQ 无 zp 布局),而 cyankiwi AWQ 是 AWQ 风格
#    zp 布局(uint4)+ group 32。syv 的 +19~30% prefill 依赖他们自己的
#    marlin-int8-negative-scales/layer-select/repack-staged-sm80 补丁组,不是免费。
# 4. **eq8emb 需要 4 行模型代码补丁**(vLLM 有 CompressedTensorsEmbeddingWNA16Int
#    反量化 gather 内核,qwen3_5/qwen3_5_mtp 只是没把 quant_config 传进
#    VocabParallelEmbedding;报错 "no parameter named 'embed_tokens.weight_packed'")。
#    补丁已落在 {runtimes_root}/vllm-backport/vllm/model_executor/models/
#    qwen3_5{,_mtp}.py 并由 launcher 无条件 bind-mount。收益 +5% c4,免费显存 2.6G。
# 5. MTP4 对 AWQ target 是 98.4(接受率 3.36 但 TPOT 9.3 ms,verify 太贵);
#    DFlash2 k7 TPOT 7.0 ms 全面胜出。MTP8 掉到 72(verify 带宽被 lm_head 吃满)。
# 6. c1 与 c4/c8 的差 = verify 批的 SM 占用;聚合吞吐看 c4/c8,c1 看延迟敏感场景。
# 7. prefill 8k = 1626 tok/s,与投机配置无关(投机只在 decode 生效)。
#
# 弃用路线:fp8 KV(FA2 拒,FlashInfer 降 PIECEWISE,MTP 下 49.7-54.4);
# INT8_ACT(unsupported);MTP 路线;max-num-batched-tokens 4096/16384(±2%);
# GMU 0.96(无收益);k=6(-6% c1)。
#
# ─────────────────────────────────────────────────────────────────────────
# 2026-09-10 复测（同机 H11SSL-i / GPU0 / 185W / 同容器同参数，风扇脚本已接管）
#
# vllm bench serve random in128/out512（本轮起记录 num-prompts，短程与持续口径分开）:
#
# | 口径                | c1 tok/s | c4    | c8    | TPOT c1 | TTFT c1 |
# |---------------------|----------|-------|-------|---------|---------|
# | 短程 16/32/64 prompts | 120.4  | 250.4 | 314.0 | 8.09 ms | 114 ms  |
# | 持续 64/128/256 prompts | **100.8** | **270.0** | **338.3** | 9.73 ms | 115 ms |
#
# - 09-05 的 127/310/386 与短程口径一致（c1 -5%，热态噪声）；持续压测 c1 掉到 ~101，
#   是 185W 功率封顶下的热稳态，不是配置回归。引用数字时必须带 num-prompts。
# - prefill-8k 总吞吐 1719 tok/s（16 prompts, c8），与 1626 一致。
# - tools/vllm-bench.py（真实文本 8 prompt ×256, greedy, thinking off）
#   grand median **168.3 tok/s**，TTFT 0.108 s。显著高于 random 数据集口径：
#   随机 token 拉低 DFlash2 接受率。真实 agent 负载更接近 150~170 这一档。
# - 质量探针 8 项输出与既有锚点一致，无退化。
#
# ─────────────────────────────────────────────────────────────────────────
# 2026-09-11 Merkyor/Qwen3.8-27B-EfficientThink FP8(block128) + 配套 DFlash2-FP8 实测（GPU1, 同容器同栈）
#
# 权重: /home/kk/models/llm/qwen3.8-27b-efficientthink-fp8 (29G) + 同级 qwen3.8-27b-efficientthink-dflash2-fp8 (2.3G)
# 拉起: env MODEL=... SERVED_NAME=... GPU_ID=1 PORT=18001 NAME=vllm-fp8-test SPEC=dflash NUM_SPEC=7
#       MAX_SEQS=16 ASYNC_SCHED=1 GMU=0.92 PATCH_DFLASH_QUANT_DRAFTER=1 DRAFT=... ./launch-27b-kkx99.sh
#
# block-fp8 在 SM80 走 Marlin W8A16（is_fp8_marlin_supported=cap≥75），无 FP8 TC、纯权重侧省显存：
# | 口径            | FP8+dflash7 | AWQ-eq8emb+dflash7 |
# |-----------------|-------------|--------------------|
# | c1 短程         | 63.6        | 120.4              |
# | c1 持续         | 81.1        | 100.8              |
# | c4 / c8 短程    | 216 / 333   | 250 / 314          |
# | 真实文本单流    | 132.3       | 168.3              |
# c8 打平（计算饱和后 dequant 摊薄）；单流 -20~-47%（fp8 读字节是 int4 的 2×）。
# 质量探针 8/8 正确且回答更充实。结论: 此模型在本机的正确用法是 llama.cpp+GGUF（无 AWQ 版），
# 或接受单流降速换 EfficientThink 质量; 追求速度仍用生产 AWQ-eq8emb。
#
# ─────────────────────────────────────────────────────────────────────────
# 2026-09-11 fastllm 头对头实测（回应 B 站视频 BV13vbK6yEhp 的 150+ TPS 质疑）
#
# 环境: ftllm 0.1.8.2 wheel (CUDA/SM80), GPU1 单卡, 同 workload-bench.py 脚本。
# 唯一可加载路径 = 本地 qwen3.8-27b-awq-int4 (plain AWQ)。FP8 路径加载器双缓冲
# vm 峰值 66G, 30G RAM+25G swap 三次 OOM 强杀, 物理不可加载（dmesg 记录）。
# eq8emb 版死在 warmup: "ToFloat16: unsupport dataType"（不认 int8 embedding）。
#
# | 指标            | vLLM+dflash2-k7(生产) | fastllm+dflash(8) | fastllm 裸 |
# |-----------------|----------------------|-------------------|-----------|
# | 写作            | 61.8                 | 42.6              | 43.1      |
# | 代码            | 201.5                | 129.1             | 42.4      |
# | 裸decode        | 51.6 (random c1)     | —                 | ~43       |
# | prefill ~10K    | 1719 (8k c8)         | 945-1491          | —         |
#
# fastllm dflash 对代码有效(42→129, 3x)但全面落后 vLLM: 写作-31% 代码-36% prefill更低。
# 视频 150+ = FP8 TP2+NVLink 双 2080Ti（复现者 191.8），硬件类别本机不可复制（单卡+无NVLink+30G RAM）。
# 评论区"prefill 2x vLLM"(11200 tok/s) 在 AWQ/SM80 上未复现。
# 结论: 生产保持 vLLM+dflash。fastllm 栈保留在 model-runtimes/fastllm/（venv+模型在库）。

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12 lm_head int8 (W8A16 per-channel, marlin u8b128) 实验 —— 结论: 不部署

# 动机: lm_head 是唯一还在 bf16 的大 GEMV（2.54 GB），decode 每步读一遍；
# int8 后 1.27 GB。微观基准: M=1 1915→1021 µs, M=8 1940→1025 µs（贴 1.27GB/1290GB/s 下限）。
#
# | 口径 (GPU1, 同 launcher, random 128x256 c1) | bf16 lm_head | int8 lm_head |
# |--------------------------------------------|--------------|--------------|
# | raw c1 (SPEC=none)                         | 51.2 tok/s (19.2 ms) | **53.68 (18.30 ms) = +4.7%** |
# | dflash k7 c1                               | 176.88 tok/s | 174.15 tok/s |
# - raw +4.7% 与微观基准预测(0.89 ms / 19.2 ms)吻合。
# - dflash 无收益: int8 轻微扰动 target logits → acceptance 4.72→4.59，抵消每步省下的
#   ~0.9 ms。**生产走 dflash，故保持 bf16 lm_head；实验 checkpoint 已删除。**
#
# 踩坑（重要，别再犯）: compressed-tensors 的 WNA16 int8 走 marlin 的 `uint8b128`
# = vllm_biased_integer_subbyte<8,128>，是**偏置整数**: 存储字节 = q + 128，不是补码。
# 按补码打包 → target logits 全乱（dflash acceptance 0.1%，全量拒稿）；而 raw 用
# random 数据集只看 tok/s 看不出问题（乱码照样跑满带宽），必须用真实 prompt 验文本。
# 依据: csrc/libtorch_stable/quantization/marlin/dequant.h 的
# dequant<bf16,kU8B128,no-zp> 把 byte 浮点化后**减 8388736 = 2^23+128**，即 byte-128。
# 修正后离线校验: 反量化 cosine≈1.0 / mean rel err 1.26%（|w|>0.1 时 0.14%），
# 端到端生成与 bf16 一致（Paris / olleh / 质数 / photosynthesis）。

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12 v0.13.0 升级 + 长上下文专项（本机 CMP 170HX ×1，全部 MS8/c1）
#
# 【基准】干净的 backport v0.13.0。本地分支 cmp170hx-v013 = 2026-09-12 08:40:56
#   rebase 到 origin/master（那一刻的尖端 24cb31bb4f，2026-09-11）+ 本树 24 个
#   cmp170hx 提交；HEAD baf423f5d9 已包含 origin/master 尖端。仓库 tag v0.13.0 =
#   7763d9805b（fork 自己的发布号，不是上游号）；镜像基座是
#   lazymio/vllm-backport:v0.13.0-sm80。远端另有 karl/cmp170hx-v013（尖端 14e7bb0981，
#   2026-09-11，只有 2 个提交：Dockerfile + 部署层，基 a350766628）——与本地方向已
#   分叉（共同祖先 a350766628），内容是本树的子集。旧的 v0.11.2 线已归档为镜像
#   vllm/vllm-backport:cmp170hx-rollback-v012。
#
# 【升级本身】短上下文无回归：随机 128/512 c1 = 120.5 tok/s（旧线 120.4），
#   KV 池 441,725 token、prefill 8K 持平。含 20 个 host commit 的 rebase + 3 个
#   未提交补丁（eq8emb ×2、量化 drafter）原样保留。
#
# 【MAX_SEQS 16 → 8】规范真实文本 harness（8 prompt ×256, greedy）：**172.16 tok/s**
#   （MS16 时同口径 111.77 → 见草稿对比表；随机 128/512 口径 129.9 vs 120.5）。
#   单流优先，故生产档改 8（c4/c8 聚合在 16 时约高 5-10%）。
#
# 【长上下文衰减：随机数据集是假象】官方 bench 的随机 token 在 64K 起让 target
#   下一 token 分布近乎均匀，三种草稿的 acceptance 同时掉到 1.00（dflash 70.6ms /
#   MTP 77.5ms / DSpark 80.3ms）。换真实文本（运维文档 + 摘要任务）后没有断崖：
#   草稿   | 2K      | 15.8K   | 47K     | 110K
#   dflash | 99.5    | 69.73   | 34.01   | 19.90  tok/s   (TPOT 10.05→50.26 ms)
#   MTP-3  | 86.95   | 79.57   | 35.15   | 21.01          (11.50→47.59 ms)
#   DSpark | 73.53   | 73.37   | 32.58   | 15.97          (13.60→62.64 ms)
#   规范 harness 短上下文：dflash 172.16 / DSpark 136.58 / MTP-3 111.77 → **选 dflash**。
#   衰减 ≈5×（2K→110K），判据（128K ≤1.25×短）未达成，且同点比 3090 慢 3.2×
#   （3090: 112k + MTP-3 + fp8 = 68.1 tok/s / 14.7 ms/token；我们 50.3 ms/token）。
#   差距拆解：step 1.55×（我们 bf16 KV = 2× 字节且 FA2 多 query verify 不切 KV）
#   + acceptance 2.2×（他们 MTP 带草稿词表 + 标定 int4 head，我们 0.16 accept/draft）。
#
# 【fp8 KV：三条路全部实测阻断】
#   FA2     → 后端选择器直接排除（候选只剩 FLASHINFER/TRITON_ATTN）
#   Triton  → 硬拒：native FP8 (fp8e4nv) requires SM89+
#   FlashInfer → 能起（池 603,265 = 1.37×），但投机档三种配置全崩在
#              "q.shape[0] (16) does not match qo_indptr[-1] (8)"（FlashInfer 加了
#              形状校验；只有关闭 CUDA graph 的 native 路径能救，对应上游
#              **未合并** PR #41127 / issue #49547 "PIECEWISE 降级 -16%"）。
#
# 【int8 per-token-head KV（树内原生）】同 MS8 同 dflash 对比 bf16+FA2：
#   短上下文 173.4 vs 168.9（+2.6%），但 47K 74.2 vs 29.6 ms、110K 133.0 vs 50.4 ms，
#   prefill 47K TTFT 71.8s vs 31.1s。→ 容量翻倍（805,181 token, 3.07×）但长上下文净亏，
#   原因与 syv 文档一致：Triton 2D 在长序列上的内核效率。
#
# 【KV 切分（多 query verify）三条路也不通】
#   syv spec-decode-attn 内核 → 已移植，构建期断言全绿，但运行期
#     cudaErrorIllegalAddress（他们的基座是上游 0.28.0，我们是 0.29-dev，metadata 契约不同）
#   vllm#44652 门放宽（16 行）→ 已移植，Triton 3D 路径同样 illegal access
#     （PR 本身 open/stale/needs-rebase，上游从未合并）
#   → 结论：这两个修复上游都没落地，我们这棵树上"多 query 切 KV"目前无可用实现。
#
# 【marlin 4 补丁（syv）】全部能打上（含 fuzz），发现并修了 1 处跨版本缺陷
#   （`gptq_marlin_repack(perm=...)`：我们树的签名没有 perm 参数）。
#   实测 prefill：1024 输入 903.9 vs 881.4 ms、8192 输入 4574 vs 4473 ms → **-2.2%，
#   无收益，不采纳**。int8 激活（他们宣称 prefill +29%）需要 uint4b8 权重：我们的
#   AWQ-eq8emb 是 asymmetric（symmetric=false, zp_dtype=int8）→ 映射到 uint4，
#   内核断言 "W8A8 is not supported" 直接拒绝，故该路径对本 checkpoint 不可用。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12 上下文衰减曲线定案（单流、真实文本、摘要任务、MS8、c1）
#
# 【阶梯：三种口径的 step 时间】
#   配置                    2.3K      49.7K     116K      step 斜率
#   纯 decode(bf16+FA2)    18.77 ms  21.59     25.69     0.061 µs/token  ← 正常
#   dflash k7(投机)        26.5      58.8      107.0     0.370           ← 异常
#   dflash k3(投机)        24.4       —        107.9     0.37
#   MTP k1(投机)           23.0      58.8      111.0     0.73
#   （step = TPOT × 每 step 产出 token 数；每 step token = 1 + k × accept/draft）
#
# 【三条定案】
# 1. 纯 decode 的衰减正常：2.3K→116K 只有 1.37×（27%），与 llama.cpp 在 2×3090 上
#    公布的"0→262K 只衰减 46%、无断崖"（ggml-org/llama.cpp#27623 里 geoffreybyers
#    的反例数据）同形；且我们的 25.69 ms @116K 与 syv 公布的"3090 无投机 100k =
#    26.8 ms/token"一致 → 基础栈没有异常。
# 2. 异常只在投机路径：verify 的 q>1 使 FA2 不做 KV 切分（源码实证：
#    vllm/vllm_flash_attn/flash_attn_interface.py 里 FA2 分支 `num_splits > 1` 直接
#    raise NotImplementedError；q=1 才走内部 3D 切分），于是整段 KV 只有
#    num_kv_heads=4 个 CTA 在读，随上下文的通量被锁在 ~200 GB/s（卡峰值 1290）。
# 3. 因此 step 只与"上下文长度"相关，与 verify 块大小无关（k=1..7 都是 107-111 ms）
#    → 缩小投机块救不了，只有 KV 切分能救。
#
# 【交叉点】无投机 18.77+0.061c  vs  dflash 10.05+0.37c  →  c ≈ 28K token
#   低于 28K：投机胜（短上下文 99.5 vs 53.3 tok/s，规范 harness 172.16 vs ~53）
#   高于 28K：纯 decode 胜（116K：38.9 vs 19.9 tok/s，快 2.0×）
#
# 【操作结论】长上下文场景应关闭投机（115K 时 38.9 vs 19.9 tok/s，翻倍，零新内核）。
#   参考栈在 112k 能到 68.1 tok/s 的原因也清楚了：他们有 split-KV verify 内核
#   （128k/8q 每层 1.3 ms vs FA2 10.1 ms）+ acceptance 2.56 token/step；
#   若我们的 verify 恢复同等效率，预期 35 ms/step × 2.13 token = 68 tok/s @116K。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12(下) split-KV verify 内核移植完成 —— 长上下文衰减从 5.0x 修到 1.87x
#
# 【做了什么】把 syv 的 spec-decode-attn 内核移植进本树（不是搬 20 个补丁，只搬
#   这一个自包含的 Triton 实现 + 接线）：
#     vllm/v1/attention/ops/spec_decode_attn.py   内核（partial NSEG=16 段 + combine 归约）
#     vllm/v1/attention/backends/flash_attn.py    hook（条件见下）
#     launch-27b-kkx99.sh                         SPEC_ATTN=1 -> VLLM_SPEC_DECODE_ATTN=1
#   生效标记（引擎日志）："split-KV spec-decode attention active: heads=24
#   head_dim=256 qmax=10 segments=16 max_num_reqs=8"
#
# 【为什么需要它】FA2 在 max_seqlen_q > 1 时拒绝切分 KV
#   （vllm/vllm_flash_attn/flash_attn_interface.py: `num_splits > 1` 直接 raise），
#   本树自己的 Triton 3D 路径也被 `max_seqlen_q > 1` 挡住，且其 partial 缓冲按
#   “序列数”分配（seq_threshold_3D, num_heads, nseg, hd）——即便撬开门也会越界。
#   于是 verify 的整段 KV 只由 num_kv_heads=4 个 CTA 读，长上下文被锁在 ~200 GB/s。
#
# 【实测（单流、真实文本、同任务；生产端口 18000）】
#   上下文      FA2(前)            split-KV(现)       step 斜率
#   2.3K        10.05 ms / 99.5    9.92 / 100.8       0.061 µs/token(前) -> 0.075
#   50.4K       29.40 / 34.0       13.21 / 75.7       2.2x
#   117.5K      50.26 / 19.9       18.61 / 53.7       2.7x
#   衰减 2.3K->117.5K：5.0x -> 1.87x；短上下文 harness(8 并发) 172.16 -> 163.68 (-4.9%)
#   step@117.5K 107 ms -> 41 ms。与参考栈持平（他们 112k 步长 37.6 ms；我们 117.5K 41.2 ms；
#   差的是 acceptance 2.26 vs 2.56，不是内核）。
#
# 【原理上限】117.5K 时每步必须读 权重 15GB + KV 9.8GB(bf16) = 24.8GB -> 19.2 ms@1290GB/s；
#   实测 41 ms = 47% 效率。要在 128K 拿到“衰减<20%”（tok/s >= 0.8×短上下文）还需：
#   ① fp8/int8 KV（字节减半 -> 33 ms/step -> 68 tok/s）② 内核效率提到 ~60%。
#   两项都缺时 128K 的理论天花板就是 1.87x —— 这是"bf16 KV + 该草稿"的结构性下限。
#
# 【事故与修复】上线后第一次真实长请求触发 CUDA illegal access。根因：
#   cu_seqlens_q(query_start_loc) 是常驻缓冲，每步只写 [:num_reqs+1]，尾部残留上一步
#   （更大批次）的前缀和；seqused_k 的补位恒为 0。用 shape[0]-1 当请求数时，幽灵请求会
#   按残留 q_start/q_len 读写不属于它的 query/output 行 -> 越界。触发条件是"8 请求批次之后
#   的第一个单请求步"（所以探针不崩、harness 之后崩）。
#   修复：内核用 `kv_len <= 0 -> return`（partial）与 `... and seqused_k[req] > 0`（combine）
#   判定真实请求，幽灵一律惰性；CUDA graph 下栅格仍按补齐上界，逻辑与栅格解耦。
#   回归用例：test-spec-attn.py 的 stale-tail(G=6, kv=80690)（修复前必崩）。
#
# 【开关】SPEC_ATTN=1 启用（默认 0 = 退回 FA2 老路径，零风险回退：去掉该环境变量重启即可）。
#   数值自检：见 test-spec-attn.py 顶部的一行 docker 命令（4 种形状 + 回归，~15 s）。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12(晚) 内核调参 NSEG 16->32 / TILE 32->64（隔离扫描 + 生产实测）
#
# 【扫描】隔离 benchmark（单层、真实形状 Hq=24/Hkv=4/D=256/q=8/BS=832，脚本 /tmp/bench_spec_attn.py）：
#   单请求 kv=117,535：最优 0.876 ms（NSEG=32 TILE=64 BLOCK_M=64）vs 原配置 1.450 ms（16/32/64）= 1.66x
#   单请求 kv=2,304  ：全部 0.076-0.091 ms（差异即噪声）-> 短上下文不受影响
#   批 8 kv=117,535  ：5.678（32/64/64）vs 5.830（16/32/64）= 1.03x（平手略优）
#   NSEG 必须是 2 的幂（combine 里 tl.arange(0, NSEG)）。
#
# 【生产实测（端口 18000，同一台卡）】
#   上下文   TPOT(前->后)       tokens/step      step(前->后)      tok/s
#   2.3K     9.92 -> 9.27 ms    2.58 -> 2.86     25.6 -> 26.5 ms   100.8 -> 107.8
#   50.4K    13.21 -> 15.14     2.58 -> 2.02     34.1 -> 30.6 ms   75.7 -> 66.0 (acceptance 波动)
#   118.8K   18.61 -> 16.80     2.32 -> 2.28     43.2 -> 38.3 ms   53.7 -> 59.5
#   衰减 2.3K->118.8K：1.87x -> 1.81x
#   （隔离扫描预测 -9 ms/step，实测 -4.9 ms：差值来自草稿侧 FA2、调度与图捕获开销。）
#
# 【剩余差距（现实账）】118.8K 每步需读 权重 15 GB + 目标 KV 7.8 GB（16 层 x 118.8K x 4 head
#   x 256 dim x K+V x 2 B；草稿是滑窗 2048，可忽略）= 22.8 GB -> 实测 38.3 ms = 595 GB/s（46% 峰值）。
#   要拿"衰减<20%"（tok/s >= 0.8x短上下文 = 86 tok/s）需 step <= 26.5 ms：
#     路线 A：KV 减半（int8/fp8）-> 18.9 GB/26.5 ms = 713 GB/s（55%），可达；
#     路线 B：不压 KV、把有效带宽从 46% 提到 67%（860 GB/s），偏难。
#   路线 A 的前置：int8 KV 只能走 TRITON_ATTN 后端（FA2 不接受 per-token-head 量化 KV），
#   而实测 int8+Triton 的 prefill TTFT 曾达 347 s（对比 FA2 98 s）——需先分清那是
#   后端税还是 int8 反量化税（也可能是首次编译未预热）。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12(晚) 距"衰减<20%"的差距：两项都实测了，结论与早先预测相反
#
# 【参照（同一 harness，kv=118,833、q=8、BS=832、D=256、Hq=24/Hkv=4）】
#   FA2 q=8（我们替换掉的路径）: 5.49-6.65 ms/层   (=89 GB/s 等效，192 行的行工作量)
#   FA2 q=1（同段 KV 的带宽天花板）: 0.469 ms/层   (=1038 GB/s)
#   我们的 split-KV 内核（最优配置）: 0.886 ms/层  (=550 GB/s)
#   逐行效率：我们 18.5 µs/行 vs FA2 19.6 µs/行 -> 已持平；差的是"行/字节比"
#
# 【item 1：KV 减半（int8/fp8）】结论：单独做不会提速，且入口有 4.5x 税
#   a) 内核是行吞吐受限，不是带宽受限 -> 字节减半不减少行工作量（syv 文档同结论：
#      "tile-bound, not bandwidth-bound at these shapes -> quantized cache buys
#      context, never speed"）。实测佐证：int8+Triton 时代 verify 反而更慢。
#   b) int8/fp8 KV 只能走 TRITON_ATTN 后端（FA2 不接受 per-token-head 量化 KV），
#      孤立实测该后端 prefill 内核比 FA2 慢 4.47x：8192 token 块 @116K 前缀
#      1162 ms vs 260 ms（bf16，同形状）。之前观测到的 347 s TTFT（对 FA2 98 s）
#      主要是这个后端税，不是 int8 反量化。
#
# 【item 2：内核效率】调参已到平台：NSEG 32/64/128 x TILE 32/64/128 x BLOCK_M 32/64/128
#   x warps 4/8 x stages 1/2/3（100+ 组合）-> 最优 0.886 ms（当前生产配置）；
#   num_stages>=2 更慢（流水线换来的寄存器/共享内存压力），warps=8 更慢。
#   要再往上必须换结构（FA2 decode 式 persistent CTA + cp.async），不是参数问题。
#
# 【<20% 的预算账】118.8K 每步 22.8 GB = 权重 15 + KV 7.8（16 层 x 118.8K x 4 head
#   x 256 x K+V x 2 B；草稿是滑窗 2048 可忽略）。<20% <=> 128K >= 0.8x短上下文
#   = 86 tok/s <=> step <= 26.5 ms -> 整步需 860 GB/s（67% 峰值）。
#   现状：整步 595 GB/s（46%），短上下文步 26.5 ms（权重路径 566 GB/s = 44%）。
#   即"目标的预算已被非注意力部分吃掉"：权重+草稿+框架开销 ~25 ms = 短上下文整步。
#   因此需要"内核改造 (0.886->~0.47) + KV 压到 1 B/元素 (7.8->3.9 GB)"同时成立；
#   只做其一：改内核 -> ~70 tok/s（35% 衰减）；只压 KV -> 行受限，白压。
#   结论：<20% 在这个模型+草稿+这张卡上不可达；当前 1.81x（45%）已与参考栈持平。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12(深夜) 事故复盘：split-KV 内核在长上下文上触发 GPU MMU Fault（生产已回退）
#
# 【症状】Xid 31 `MMU Fault ... FAULT_PDE ACCESS_TYPE_VIRT_READ`，进程 VLLM::EngineCore；
#   `CUDA_LAUNCH_BLOCKING=1` 下栈指向：flash_attn.py:1196 forward -> _spec_attn_run:1966
#   -> spec_decode_attn.py:199（partial 内核）。
#
# 【复现】CUDA graph（默认模式）+ 同一 ~135k token prompt 连发两次：
#   第一次正常，第二次（整段命中前缀缓存，num_common_prefix_blocks=163）崩。
#   多轮实测：32.7k/65.5k 连发正常；~98k 曾在特定会话崩过（非确定）；eager+守卫下不崩。
#   A/B：同场景 SPEC_ATTN=0（内核关）两次都正常 -> 责任在这个内核。
#
# 【已排除】
#   - 块号越界：审计显示的“offset >= numel”是视图 numel 与底层存储之差（KV 为 K/V 交错
#     布局，stride(0)=1703936=832x4x512，实际地址落在底层存储内）-> 假阳性。
#   - 幽灵槽位：vLLM 在 gpu_model_runner.py:2203 有 `self.seq_lens[num_reqs:].fill_(0)`，
#     补位 seq_len 恒为 0；且内核用 `kv_len<=0 -> return`（partial）与同条件（combine）判惰性。
#   - partial 缓冲尺寸：按 (max_num_seqs x heads x qmax x NSEG) 分配并逐项校验过。
#
# 【本轮已落地的加固（保留）】
#   1) FlashAttentionMetadata 新增真实 num_reqs（由 build() 从 common_attn_metadata 填入），
#      不再用 query_start_loc.shape[0]-1 这种“补齐上界”。
#   2) 内核新增槽位一致性守卫：`kv_len<=0 or q_start<0 or q_start+q_len>total_tokens`
#      一律惰性（partial 与 combine 用同一条件，保证不会读到没写的 partial）。
#   3) 调试探针（VLLM_SPEC_ATTN_DEBUG=1，捕获期自动跳过）：逐步记录 num_reqs/cu/seqused/
#      table 形状 + 地址审计，校验失败则回退 FA2。
#   效果：eager 下同序列已不再崩；但【图模式仍崩】——重放时 Python 不执行，Python 侧
#   守卫与日志都无效，必须靠内核内部手段（把越界索引写进设备缓冲，再在下一个 eager 步
#   或信号处理里取回），或 cuda-gdb 附加。
#
# 【当前状态】生产回退为 SPEC_ATTN=0（122k: 45.96 ms -> 21.8 tok/s，即内核前基线）。
#   内核代码、调参与守卫都留在树里（SPEC_ATTN=1 可再启用），但根因未定位前不上生产。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12(深夜二) 崩溃二分：不是内核访存越界，而是"图内容"层面的交互
#
# 【触发条件（已收敛）】前缀缓存命中 + CUDA graph 执行（FULL 与 PIECEWISE 都崩）
#   + 上下文 > ~32K。同长度为 32.6K 时两次均正常，64.7K 第二次（命中）崩。
#   eager（--enforce-eager）干净；--no-enable-prefix-caching 下同一 prompt 连发 3-6 次干净
#   （134K: TPOT 14.3-16.0 ms ≈ 62-70 tok/s）。
#
# 【故障地址】跨运行恒定：0x25_e6d80000（个别 +0x2000）= KV 基址 0x26e0000000 下方
#   ~4.18 GB ≈ -1227 个块（block stride 3,407,872 B）。该地址不在本进程任何已知
#   缓冲区间内（q=0x1da3400000 / k=v=0x26e0000000 / part_o=0x15947b0000 /
#   bt=0x325672400 / cu=0x320048c00 / seqused=0x320048e00 / part_m=l=0x3255xxxxx；
#   且各次运行地址完全一致 -> 可跨运行比对）。
#
# 【已做的二分（每次 = 重新拉起 + 同一 64.7K prompt 连发至崩）】
#   1) 块号验证+屏蔽（越界块号按"无块"处理并计数）：仍崩 -> 不是块号。
#   2) q 行 / out 行钳制到 [0, total_tokens)：仍崩 -> 不是这些行索引。
#   3) partial 内核整体空操作（constexpr 短路，且仍留在图内）：仍崩
#      -> KV 读取、block_table 读取、partial 写入都不是触发点。
#   4) 影子模式（分配同样缓冲、但内核不执行、注意力交回 FA2）：干净
#      -> 崩溃与"图里有没有这两个内核"相关。
#   5) 两内核都空操作：仍崩（但 out 不写 -> 数据流改变，属无效对照）。
#   结论：剩余疑点在"额外内核进入捕获图"本身（图内存池/捕获时序/驱动交互），
#   而非内核的访存。继续需要 cuda-gdb 附加或图内核归属工具。
#
# 【已交付的加固（保留在树里，SPEC_ATTN 默认关闭】真实 num_reqs、槽位一致性守卫、
#   块号验证+屏蔽、q/out 行钳制、VLLM_SPEC_ATTN_DEBUG 探针（含地址审计与回退 FA2）、
#   VLLM_SPEC_ATTN_SHADOW / NOOP_PARTIAL / NOOP_COMBINE 诊断开关。
#
# 【生产】SPEC_ATTN=0（稳定，122K: 45.96 ms -> 21.8 tok/s）。
#   可选但未采纳：SPEC_ATTN=1 + --no-enable-prefix-caching（62-70 tok/s @134K），
#   代价是重复长前缀每次全量 prefill（对 agent 场景不划算）。
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 2026-09-12(深夜三) 两个崩溃根因定位并修复（均已在各自复现器上验证）
#
# 【根因 1 = vllm#48375 的额外丢弃】另一条线（karl/cmp170hx-v013）在合并时明确说过
#   它在本基座上会把重发状态清零。删除后："8 并发 harness + 同一 64.7K prompt 连发 4 次
#   （第 2-4 次命中前缀缓存，TTFT 1.34 s）"从崩变为全过（169 tok/s / 62-70 tok/s @134k）。
#
# 【根因 2 = mamba 状态列除数错误（vllm#53142 类）】CUDA_LAUNCH_BLOCKING 下栈指向
#   model_states/mamba_hybrid.py:232 preprocess_state -> mamba_utils.py:1226
#   run_fused_precopy（precopy_mamba_align_fused_kernel）。
#   实测取值：mamba_block_size=16（应 832）user_specified=False spec_block_size=832
#   cache_block_size=832。即 worker 用了 cache_config.mamba_block_size（16 = CLI 默认块大小，
#   早于 platforms/interface.py 把 cache_config.block_size 提升到 832 的那一步），
#   于是前缀续跑时 state_idx=(num_computed-1)//16 比状态表列数大 ~52 倍 -> 越界读。
#   上游分析同源（#53142/#54199："seeds an out-of-range block_table column ... garbage
#   block id"），但方向相反：他们是 cache_config.block_size 被覆盖，我们是 mamba_block_size
#   陈旧。修复：不再在 worker init 锁存该值；在 add_request 记录待播种位置，等
#   _get_mamba_group_info 拿到 mamba 组 spec 后，用 mamba_spec.block_size 播种
#   （只有用户显式指定时才用 cache_config 的值，与平台钩子的规则一致）。
#   验证："64.7K -> 135K x4（第 2 次起全部命中前缀缓存）"全过，健康 200。
#
# 【尚未闭环】组合序列（8 并发 harness -> 96K 长请求）在图模式下仍观察到一次异步
#   illegal access，而那一步的 mamba 取值全部正常（state_idx/src_col/除数 832 均合理）。
#   eager+blocking 下同序列未复现。下一步：对该组合序列用图内手段（UVA/共享内存诊断）
#   或按 batch 形状二分，定位是否为第三个独立缺陷。生产已带两个修复运行。
# ─────────────────────────────────────────────────────────────────────────

# 【第三个形状（未闭环）】2026-09-12 生产实测：先 120.7K（填满前缀）→ 再发 64.7K
#   （是前一请求的*前缀*，即"父比子长"的部分命中）-> 崩。而我验证过的复现器是反序
#   （64.7K 先 → 135K 后）-> 覆盖不到。
#   复现器（生产端口 18000）：probe 224000 -> probe 120000 -> probe 120000
#   现有证据：崩溃步的 mamba 取值正常（除数 832、state_idx/src_col 一致），故仍指向
#   "续跑所需的那一列 state 未必被物化"这一族（= 上游 #53479 的主题：align 模式只在
#   chunk 末端物化 state；上游正以"每个可被查找命中的边界都物化 state + 取消投机的一块
#   回退 + 保留感知（retention-aware）的边界停靠"来修）。
#   下一步：把该复现器缩到最小尺寸（父 ~40K / 子 ~20K 是否复现），再按上游 #53479 /
#   #50409 / #51113 的意图移植（调度器 `_mamba_block_aligned_split` + MambaManager 的
#   reachable_block_mask/reachable_boundaries）。

# 【第三个形状：更新（2026-09-13 凌晨）】
#   快速复现器（图模式，前置状态 = 缓存里已有更长的同文档前缀）：
#     ① probe 248000（父，134,230 tok）② probe 96000（子，51,886 tok，是父的前缀）→ 崩
#     cycle ≈ 6 分钟（1 次启动 + 2 个请求）。
#   已排除：mamba 状态除数（已修为 832 ✓，eager 下取值全部合理 ✓）、密集保留
#   （VLLM_PREFIX_CACHE_RETENTION_INTERVAL=832，仍崩 ✗）、eager 模式（干净 ✓）。
#   地址归属（torch.cuda.memory_snapshot）：最大段 0x26e0000000..0x3019400000 = 37.8 GB
#   （KV + mamba state），而故障地址 0x25e0240000 / 0x25e6d80000 / 0x25e0581000
#   全部落在该段**下方** ✗ -> 仍是负偏移寻址（越界在"段前"，不是段内）。
#   下一步（内核内可见性）：图重放时 Python 不执行，必须在内核内记录
#   fused precopy / state copy 的列索引与指针（写 UVA/共享内存缓冲，崩溃后仍可读），
#   或按 batch 形状/请求历史二分定位负值来源。

# 【第三形状：2026-09-13 02:30 进展】
#   稳定复现器（受管循环，命中即抓）：图模式 → 父 248000 字符（134,989 tok）→
#   子 96000（52,064）→ 子 96000 → 子 60000（32,396）→ 第 3 轮崩（约 3 轮内必中）。
#   宿主侧 precopy 取值全正常（state_idx/src_col/除数 832 一致）→ 问题在**其后的状态拷贝**。
#   内核内 device_print 已打通（需把 mamba_utils.py 加进挂载清单，之前漏了 ✗），
#   但打印被掩码 lane（state_idx=-1，网格 256 行 vs 1 请求）淹没 -> 下一步必须只打印
#   活跃 lane（mask 为真）并同时打印 copy 内核（mamba_attn/模型 copy funcs）的源列/源块。

# 【2026-09-13 03:2x 第三形状：列范围守卫（两个索引）】
#   机理：mamba 状态拷贝 `_copy_mamba_state_block` 把 `dst_col` / `src_col` 直接当块表列用，
#   没有任何行宽校验；越界列会把相邻内存当块号 -> 负块号 -> state 基址下方的野地址
#   （与三次故障地址全部落在 37.8GB 主段下方一致）。且时间态路径用的是
#   `bt[src_col + token_bias]`（token_bias = num_accepted-1，可达 7），即使 src_col 在内，
#   偏移后也可能越界 -> 两个索引都要守。
#   验证：加入守卫后 "135K -> 96K -> 96K -> 60K" 连续 4 轮 16 发全过（此前 3 轮内必崩）；
#   质量门（90K 文档抽数字）回答正确 '172.16'；harness 167-169 tok/s 无回归。
#   残留：该形状仍偶发（另一轮里 r1/96K 崩过，地址回到最早的 0x25e4680000，异步无栈），
#   需在带 CUDA_LAUNCH_BLOCKING 的图模式下继续定位（或对 postprocess 侧做同样的列守卫）。

# 【2026-09-13 04:5x 残留定位（第三轮）】
#   已验证配置上复现：6 轮循环里第 4 轮 / 96K 崩（约 1 次/12-16 个长请求）。
#   最新判据（全部热路径安全、无同步）：
#     - 内核内 device_print：**constexpr 值能打印**（mamba_block=832 ✓ 出现 11k 行），
#       但**张量值的 print 未落地**（n_active/pre_state_idx/num_computed/num_accepted 一行都没出 ✗）
#       -> 下一轮要换写法（例如先 .to(tl.int32) 再逐程序打印，或把值写进 diag 缓冲再在读侧取回）。
#     - 已排除：块表列越界 ✗（打印从未触发）、块表重分配 ✗（宿主指针不变式未触发）、
#       postprocess 的 block_size 错误 ✗（= mamba_spec.block_size ✓ 正确）、eager 路径 ✗（干净）。
#     - 与我们的 split-KV 内核无关 ✓（SPEC_ATTN=0 下同序列首发即崩）。
#   【教训已固化】任何加在 preprocess_state 热路径上的 Python 侧同步都会**显著加重**该故障
#   （实测从"12 发干净"变成"首发即崩"），相关提交已回退（db02f2db2f）。

# 【2026-09-13 05:2x 残留定位（第四轮）——发现"观察者效应"】
#   在拷贝路径加标量打印后：**打印张数 0**（小请求未跨块 ✓ 正常），但紧接着的**首个 135K
#   请求就崩** ✗ —— 而同一棵树的"已验证版本"此前 12 发全过 ✓。
#   => 仪表本身改变了故障的表现（从"第 4 轮崩"变成"首发即崩"）✗✗：printf 很贵，
#      改变了时序；这个 bug 对时序极度敏感。
#   结合既有证据（图专属、偶发、异步报错无栈、eager 恒干净、加任何打印/同步都会挪动它的
#   表现），机制指向**竞态**（例如 align 的状态拷贝/推进 与 状态写入/其他请求之间在图重放
#   下的顺序与本应不同），而不是单纯的索引越界。
#   下一步：审计"在 align 内核之外写 state 张量"的路径（mamba_attn 的状态更新、GDN 内核），
#   以及图捕获顺序与 eager 顺序是否一致；必要时用 __graph 捕获前后各打一次 kernel 序列
#   （torch.cuda.graph 的 debug_dump / profiler trace），比对两者顺序。

# ═══════════════════════════════════════════════════════════════════════════
# 残留缺陷完整刻画（2026-09-13，可直接作为上游 issue 正文）
#
# 【标题】Hybrid GDN+Mamba 模型：spec-decode + CUDA graph + 长上下文 + 前缀缓存续跑时，
#         mamba 状态路径触发 Xid 31（MMU Fault, VIRT_READ，地址恒在 KV/state 段下方）
#
# 【环境】Qwen3.8-27B AWQ-eq8emb + DFlash2-W4A16(k=7) + MS8 + async，vLLM backport
#         v0.13.0（origin/master @09-11），CMP 170HX，mamba_cache_mode=align + 前缀缓存
#
# 【判定矩阵（全部实测，同一台卡同一序列：135K → 96K → 96K → 60K/轮）】
#   CUDA graph + 投机 + 前缀续跑 + 长上下文      -> 崩（约 1 次/12-16 个长请求，偶发）
#   --enforce-eager（同其余）                    -> 干净 ✓
#   SPEC=none（同其余）                          -> 干净 ✓（4 轮）
#   SPEC_ATTN=0（关我们的 split-KV 内核）        -> 照样崩 ✗（首发即崩）
#   短/中上下文（<32K）                          -> 从不触发 ✓
#   加任何仪表（内核 printf / 宿主同步）         -> 故障表现被挪动（12 发干净 -> 首发即崩）✗
#
# 【已排除（均有可判定证据，非"看着像"）】
#   块表列越界（守卫打印从未触发）✗ / 块表被重新分配（宿主指针不变式未触发）✗ /
#   postprocess 块大小（= mamba_spec.block_size ✓ 正确）✗ /
#   预处理侧列与位置（内核内实测 pre_state_idx=63、num_computed=52578、除数 832 合理）✗ /
#   我们的 split-KV 内核（内核关同崩）✗ / 除数 16->832（已修 ✓）/ #48375 额外丢弃（已修 ✓）
#
# 【形态结论】故障对**时序**极度敏感（观察者效应）+ 只在图重放 + 只在投机 → 指向
#   spec-decode 的 mamba align 回存/推进路径（postprocess 的"接受后非对齐回存"、
#   temporal state 的 bt[src_col + token_bias]）与其它写入之间的**顺序**问题，而非索引越界。
# ═══════════════════════════════════════════════════════════════════════════

# 【2026-09-13 06:1x k=1 负结果：降投机深度不是解药】
#   NUM_SPEC=1（dflash k=1）：前 6 轮 24 发干净 ✓，但随后 120K 探针即崩 ✗、r7 再崩 ✗。
#   => 24 发干净是运气（若真实故障率 1/12，24 发全过概率约 12%）；**无法通过降低投机深度规避** ✗。
#   代价：短上下文 harness 80.3 tok/s（k=7 为 169 ✗）。
#   结论：配置层没有"保留全部 feature 且稳定"的解 ✗ —— 唯一干净配置是 SPEC=none（长上下文
#   掉回 21.8 tok/s ✗）。修复只能来自上游级改动（见根目录 UPSTREAM-report-mamba-align-spec-xid31.md）。

# 【2026-09-13 06:4x 两个 align 内核都被排除（诊断性跳过）】
#   用 env 门控的诊断开关（默认关闭、绝不发布）分别跳过：
#     VLLM_MAMBA_SKIP_ALIGN_SAVE=1    -> 同一序列 r5/96K 仍崩 ✗（postprocess 回存不是触发点）
#     VLLM_MAMBA_SKIP_ALIGN_PRECOPY=1 -> 同一序列 r5/96K 仍崩 ✗（precopy 也不是触发点）
#   同时修正一条此前的弱结论：SPEC=none 的"干净"只有 16 发（若真实故障率 1/12，概率 25% ✗），
#   所以"投机专属"并不成立；更可能是**投机提高跨块频率**从而放大暴露 ✗。
#   剩余嫌疑：GDN/conv 层内核、mamba 状态写入（mamba_attn 侧）、reshape_and_cache，
#   或图重放下的整体顺序。下一步建议：eager + CUDA_LAUNCH_BLOCKING 跑满 ~24 个长请求
#   （eager 目前只在 16 发内干净过 ✗），若也崩即可用栈精确归属内核。

# 【2026-09-13 07:0x 归属路径关闭：eager 48 发全过】
#   eager + CUDA_LAUNCH_BLOCKING + 最小形状（65K→32K/轮）：8 轮 32 发全过 ✓
#   累计 eager 干净：16 + 32 = 48 发（若真实故障率 1/12，概率约 1.5% ✓）vs 图模式 1–5 轮内崩 ✗
#   => "只在 CUDA graph 重放下出现"是硬结论 ✓；同时意味着**LAUNCH_BLOCKING 无法归属** ✗
#      （eager 不故障，图重放不可归属 ✗）——这正是本 bug 的取证难点。
#   剩余唯一取证手段（都较重）：① cuda-gdb 附加到重放（或对捕获图做内核级断点）；
#   ② UVA/共享内存缓冲：内核内无条件写入关键标量，由独立进程读（崩溃后仍可读）✗。
#   两条都未尝试（本轮预算已尽）✗。

# 【2026-09-13 07:3x 再排除两项】
#   ASYNC_SCHED=0（同步调度，其余同）：同一序列 r5/96K 仍崩 ✗ -> 异步调度不是原因 ✓
#   至此累计排除：我们的 split-KV 内核 / align 回存 / align precopy / 异步调度 /
#   块表列越界 / 块表重分配 / postprocess 块大小 / 预处理列与位置 / RecoverSSM /
#   num_accepted 快照顺序。硬结论：只在 CUDA graph 重放下（eager 48 发干净 ✓ vs 图 1-5 轮崩 ✗）。
#   剩余嫌疑：GDN/conv 层内核、层内 mamba 状态写入（mamba_attn 侧）、reshape_and_cache、
#   或图重放下的整体内核顺序。取证只剩两条重装备路线（cuda-gdb 附加重放 / UVA 共享内存缓冲）✗。

# 【2026-09-13 诊断台已建成 + 关键结构性发现】
# 取证台（已提交，env 门控、默认关闭）：
#   /dev/shm 主机映射缓冲 + 零同步插桩（diag_mark 自定义算子 / diag_scan 负值扫描 /
#   diag_py Python 直写 / diag_ring 512 条有序记录）——崩溃后外部进程仍可读。
#   踩坑：cudaHostRegister 只对 tmpfs 有效 ✗，对绑定挂载宿主目录无效 ✗（改用宿主 /dev/shm）。
#
# 结构性发现（推翻此前假设）：运行在 **FULL CUDA graph** 模式（日志 "Capturing CUDA
# graphs (FULL)"）—— 整个模型含注意力都被捕获 ✗。因此重放时**没有任何 Python 运行**：
# 所有 gate/校验都冻结在捕获时刻 ✗。这解释了"eager 干净、图内必崩"的一整类现象，
# 也说明此前会话里"piecewise 下注意力是 eager 执行"的注释前提在本配置下不成立 ✗。
# 推论（待验证）：同步二分法在捕获期非法 ✗（playbook 亦然），必须在捕获期禁用。
#
# 已排除（本轮新增）：QMAX 尺寸（QMAX=64 仍崩 ✗）、MAX_REQS 越界（加界后仍崩 ✗）、
# 钩子是否运行（未运行 ✗）、级联路径（未走 ✗）、slot_mapping 的 -1（内核已跳过 ✓）。
# 仍未定位：故障内核在"后端入口之前"的窗口内 ✗（重放中无 Python，标记无法写入 ✓）。
