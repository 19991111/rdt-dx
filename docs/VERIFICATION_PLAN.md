# RDT-Dx 项目创新初步验证方案

> 日期: 2026-06-02
> 项目: RDT-Dx (RDT-Fixed-16)
> 基座模型: Qwen3.5-9B-Base

---

## 一、项目核心创新点

| 编号 | 创新点 | 核心主张 |
|------|--------|----------|
| I1 | 循环深度 Core 架构 | 同一组 Core 层重复执行 16 次能在隐藏空间中产生有意义的内部推理过程 |
| I2 | Bridge v2 状态校准器 | 零初始化残差 MLP + step embedding 可控制循环导致的分布漂移 |
| I3 | 分阶段渐进训练 | Progressive K warmup (2→4→8→16) 比直接训练 K=16 更稳定 |
| I4 | 计算深度 > 参数规模 | 对 hard 样本的提升来自计算深度而非模型容量增加 |

---

## 二、验证实验设计

### 实验 1：结构有效性验证（Phase A — 当前阶段）

**待验证**: RDT-Fixed-16 可在 K=16 时正常完成 forward/backward/generate。

| 项目 | 内容 |
|------|------|
| 方法 | 构建 RDT-Fixed-16，K=16 全链路测试 |
| 关键指标 | 显存占用、forward/backward 耗时、生成质量、loss 收敛性 |
| 通过标准 | 无 OOM、无 NaN、生成可读文本、trainable params < 15% |

### 实验 2：Bridge v2 消融（Phase B）

变体: no-bridge, linear-bridge, mlp-bridge, mlp-step-bridge (默认)

**核心指标**: Top-5 overlap vs reference, hidden drift 曲线, last4 variance

### 实验 3：K 步数消融（Phase C — 最关键）

变体: K=1,2,4,8,16 + SFT baseline + "厚 LoRA" baseline (同等 trainable params)

**分层评估**: easy/medium/hard 各自的 CE loss / PPL

### 实验 4：聚合方式消融（Phase D）

变体: last, mean_all, last4_mean, gated_mean

### 实验 5：Progressive K Warmup 验证（Phase D）

变体: direct-K16 vs progressive (2→4→8→16)

---

## 三、评估指标体系

- Layer 1 (语言质量): CE Loss, PPL, Format Rate
- Layer 2 (医疗质量): Diagnosis Acc@1/3, Differential Coverage, Red Flag Recall, Triage Accuracy
- Layer 3 (安全性): Hallucination Rate, Unsafe Advice Rate, Refusal Accuracy
- Layer 4 (RDT 专属): Step-wise KL, Top-k Overlap, Hidden Drift Curve, Last4 Variance, Step Benefit Ratio

---

## 四、最关键判据

> 在 trainable params < 基座 15% 的前提下，RDT-Fixed-16 在 hard 医疗样本上是否显著优于同等参数预算的普通 LoRA SFT，同时 easy 样本不退化？

---

## 五、执行路线图

| Phase | 内容 | 预期时间 |
|-------|------|---------|
| A | K=16 结构可行性 + 显存/延迟 benchmark | 1-2天 |
| B | Bridge 消融 (4 variants) | 3-5天 |
| C | K 步数消融 (K=1,2,4,8,16 + baselines) | 5-7天 |
| D | 聚合 + Warmup 消融 | 3-5天 |
| E | GRPO RL 验证 | 待 Phase C 通过 |

---

*文档状态: Phase A 执行中*
