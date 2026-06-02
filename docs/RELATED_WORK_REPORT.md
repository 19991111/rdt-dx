# RDT-Dx Related Work Report: Prior Art in Looped/Recurrent Depth Transformers

> **日期**: 2026-06-02
> **研究问题**: RDT-Dx 的设计（循环复用 Transformer 中间层 + Bridge 状态校准 + 固定 16 步迭代应用于医疗诊断）是否已有类似研究？

---

## 执行摘要

**简短回答：循环深度 Transformer 的研究方向已经在 2024–2025 年爆发，成为该领域最热门的方向之一。RDT-Dx 的核心思想（"用计算深度替代参数规模"）与多个前沿工作高度一致。但 RDT-Dx 在以下方面具有差异化贡献：(1) Bridge v2 状态校准器的具体设计（零初始化残差 MLP + step embedding）；(2) last4_mean 聚合策略；(3) 将循环深度架构首次应用于医疗诊断场景。**

---

## 一、最直接的相关工作（高度重叠）

### 1.1 Huginn — 开源 Recurrent Depth Transformer（2025 最接近的工作）⭐⭐⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **论文** | [Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer](https://arxiv.org/abs/2502.05171) (ICLR 2025) |
| **作者** | Wenquan Lu, Jonas Geiping 等 (UMD / MPI) |
| **模型** | [huggingface.co/tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) |
| **代码** | [github.com/seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining) |
| **许可** | Apache-2.0 完全开源 |

**架构相似度：极高**

Huginn 的架构与 RDT-Dx 几乎同构：

```
Huginn:  Prelude → [Recurrent Block × R] → Coda
RDT-Dx: Prefix → [Core Layers × 16] + Bridge → Suffix
```

**相同点：**
- 三段式结构：前缀编码 → 循环推理 → 后缀解码
- 前缀/后缀冻结，循环块可训（Huginn 1.5B recurrent params vs RDT-Dx 138M）
- 循环块权重共享
- 固定步数模式（Huginn 支持 16-128 步，RDT-Dx 固定 16 步）
- 支持 per-token adaptive compute（Huginn 已实现，RDT-Dx 计划但未实施）
- 核心主张："计算深度 > 参数规模"（两者完全一致）

**关键区别：**
| 维度 | Huginn | RDT-Dx |
|------|--------|--------|
| **循环块定义** | 4 个独立 transformer 层 (R1→R2→R3→R4) 整体循环 | 连续 16 层 (12-27) 整体循环 |
| **状态校准** | 无专门的 Bridge 模块；依赖 Gaussian noise seed injection | **Bridge v2** (零初始化 MLP + step embedding) 主动校准 |
| **聚合策略** | 最后一步 hidden state | **last4_mean** (最后 4 步平均) |
| **输出层** | 独立的 Coda 参数 | 复用原模型 Suffix 层 (28-31) + LoRA |
| **规模** | 3.5B 参数，800B tokens 训练 | 9.1B 基础模型，138M 可训参数 |
| **应用场景** | 通用语言建模 | **医疗诊断（特定领域）** |
| **稳定性保证** | 无 explicit mechanism | **Bridge v2 专门解决漂移问题** |

> **结论**：Huginn 是 RDT-Dx 最接近的已有工作。但 RDT-Dx 的 Bridge v2 校准器设计是对循环稳定性问题的独特贡献，且医疗诊断场景是差异化应用。

---

### 1.2 Relaxed Recursive Transformers + Mixture-of-Recursions (MoR) — ICLR 2025 / NeurIPS 2025 ⭐⭐⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **论文 1** | [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672) (ICLR 2025) |
| **论文 2** | [Mixture-of-Recursions: Learning Dynamic Recursive Depths](https://arxiv.org/abs/2507.10524) (NeurIPS 2025) |
| **作者** | Sangmin Bae, Adam Fisch 等 (KAIST / Google DeepMind) |
| **代码** | [github.com/raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions) |

**核心贡献：**

1. **Relaxed Recursive Transformers**：将 pretrained LLM 转换为 Recursive Transformer，使用 depth-wise LoRA 模块为每一层循环添加微小差异化适配。与 RDT-Dx 的 Core LoRA 思路高度一致。

2. **Mixture-of-Recursions (MoR)**：引入轻量 Router 动态决定每个 token 需要多少层循环。简单 token 早退出，复杂 token 深层迭代。与 RDT-Dx 计划的 "Adaptive 方案" 一致。

3. **Continuous Depth-wise Batching**：通过 early exit 实现 2-3× 推理吞吐提升。

**与 RDT-Dx 的关系：**
- RDT-Dx 的 "Core LoRA + 固定 16 步" 可以看作 Relaxed Recursive Transformers 的特例（固定步数，不使用早期退出）
- RDT-Dx 的 Bridge v2 提供了 Relaxed Recursive Transformers 未涉及的漂移控制机制
- MoR 的 router-based adaptive depth 是 RDT-Dx 计划中的 Phase 4 方向

---

### 1.3 Parcae — 循环 Transformer 的 Scaling Laws ⭐⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **论文** | Parcae: Scaling Laws For Stable Looped Language Models (UCSD / Together AI, 2025) |
| **核心问题** | 循环 Transformer 的隐藏状态漂移和稳定性 |

**直接针对 RDT-Dx 的稳定性问题**，但解决思路不同：

| 问题 | Parcae 方案 | RDT-Dx 方案 |
|------|------------|------------|
| 隐藏状态漂移 | 约束谱半径 ρ(Ā) < 1（负对角参数化 + ZOH 离散化） | **Bridge v2 残差 MLP 主动校准** |
| 训练稳定性 | Prelude Norm 标准化输入嵌入 | 零初始化 Bridge 输出层（保证 h_t ≈ h_anchor） |
| 循环步数 | Per-sequence 随机采样 | Progressive K Schedule (2→4→8→16) |

**关键发现**：Parcae 证明了循环稳定性是可预测和可控制的。RDT-Dx 的 Bridge v2 提供了另一种解决方案 — 通过额外的可训校准模块而非约束参数化。

---

## 二、基础性相关工作（理论先驱）

### 2.1 Universal Transformer (UT) + ACT — ICLR 2019 ⭐⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **论文** | [Universal Transformers](https://arxiv.org/abs/1807.03819) (Google, ICLR 2019) |
| **关键贡献** | 首次提出在深度方向循环应用同一 Transformer 块，配合 ACT 实现自适应深度 |
| **与 RDT 关系** | RDT-Dx 的 Core 循环复用本质上是 UT 的 "recurrent in depth" 思想，但 RDT-Dx 不共享所有层（仅中间 Core 段），也不使用 ACT 自适应停止 |

**差异点：**
- UT 共享所有 Transformer 层；RDT-Dx 仅共享 Core 段，保留独立的 Prefix 和 Suffix
- UT 的 ACT 使用 halt probability；RDT-Dx 的固定 16 步更简单但缺乏自适应性
- RDT-Dx 的 Bridge v2 是 UT 完全没有的组件

### 2.2 Deep Equilibrium Models (DEQ) — NeurIPS 2019 ⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **论文** | [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) (Bai, Kolter & Koltun, CMU, NeurIPS 2019) |
| **核心思想** | 不显式迭代，直接求解不动点 z* = f(z*; x)，等价于无限深度权重绑定网络 |
| **差异** | RDT-Dx 使用显式迭代（而非隐式求根），且不要求达到不动点（固定 16 步而非收敛） |

DEQ 是"无限深度"的理论框架，RDT-Dx 可以看作其工程化近似（固定步数 + Bridge 校准替代隐式微分）。

### 2.3 Reasoning with Latent Thoughts — ICLR 2025 ⭐⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **论文** | [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416) (Saunshi et al., ICLR 2025) |
| **核心理论** | k 层 transformer 循环 L 次 ≈ kL 层非循环模型的推理能力 |
| **与 RDT 的关系** | 为 RDT-Dx 的核心主张（138M RDT > 2.2B SFT）提供了理论基础 |

---

## 三、相近但不同的方向

### 3.1 Chain-of-Thought (CoT) 替代方案

多个 2025 年工作将 layer recurrence 定位为 CoT 的替代或补充：

- **Test-Time Layer Recurrence** (OpenReview, 2025): 在推理时对已有模型施加层循环，与 CoT 对比发现 layer recurrence 在长序列上更稳定高效
- **Encode-Think-Decode (ETD)** (Koishekenov et al., 2025): 在 mid-training 阶段训练模型迭代推理层
- **URM (Universal Reasoning Model)** (2025): 系统消融证明 recurrence beats depth on ARC-AGI

**与 RDT 的关系**：RDT-Dx 的内部循环本质上也是一种 "latent CoT" — 在隐藏空间中思考，不生成中间 token。

### 3.2 医疗 LLM 专业架构

搜索发现，当前医疗 LLM 研究集中在以下方向，**没有任何工作将循环深度架构应用于医疗诊断**：

| 工作 | 方向 | 与 RDT 重叠 |
|------|------|------------|
| [PRISM](https://arxiv.org/abs/2506.11082) (2025) | 临床序列预测 | 无 |
| [ClinRaGen](https://arxiv.org/abs/2411.07611) (2024) | 知识增强小模型诊断 | 无 |
| Chain-of-Diagnosis (2025) | 诊断链推理 | 仅推理可视化层面 |
| Med-PRM (2025) | Process Reward 医疗推理 | 仅 RL reward 设计层面 |

> **这是 RDT-Dx 的一个明确空白点：循环深度架构 + 医疗诊断的结合是全新的。**

---

## 四、RDT-Dx 的差异化分析

### 4.1 与已有工作高度重叠的部分

| RDT-Dx 设计 | 已有工作 | 重叠度 |
|------------|---------|--------|
| "计算深度替代参数规模" | Huginn, Parcae, URM, MoR, Relaxed Recursive Transformers | 🔴 高度重叠 |
| 三段式架构（前缀/循环/后缀） | Huginn (Prelude/Recurrent/Coda) | 🔴 高度重叠 |
| Core 层循环复用 + LoRA | Relaxed Recursive Transformers (+ depth-wise LoRA) | 🟡 中度重叠 |
| 固定步数迭代 | Huginn, Parcae | 🔴 高度重叠 |
| Progressive K Schedule 训练 | Parcae (per-sequence depth sampling) | 🟡 类似思路 |

### 4.2 RDT-Dx 的差异化创新

| RDT-Dx 设计 | 与已有工作的差异 | 新颖度评估 |
|------------|----------------|----------|
| **Bridge v2 状态校准器** | Huginn 无专门校准模块；Parcae 使用约束参数化而非额外模块 | 🟢 中等新颖 |
| **零初始化残差 MLP + step embedding** | 唯一将 step embedding 与校准器结合的设计 | 🟢 中等新颖 |
| **last4_mean 聚合** | 现有工作多用最后一步或全部平均 | 🟢 低新颖（但验证了效果） |
| **医疗诊断场景应用** | **无先例** | 🟢 场景新颖 |
| **Phase 1 Bridge Warmup（仅 KL loss）** | 独特的渐进式结构稳定化训练策略 | 🟢 中等新颖 |
| **医疗 composite reward (GRPO)** | 与 Med-PRM 思路类似但 reward 设计更细粒度 | 🟡 部分新颖 |

### 4.3 核心主张的原创性评估

> **"用计算深度替代参数规模"并非 RDT-Dx 首创。**

这一主张在以下工作中已被反复验证：
- Universal Transformer (2019): 参数共享 → 更强泛化
- DEQ (2019): 隐式无限深度 → 恒定内存
- Huginn (2025): 3.5B 循环模型匹配 7B 传统模型
- Parcae (2025): 770M 8-loop ≈ 1.3B standard Transformer
- Relaxed Recursive Transformers (2024): 递归 Gemma 1B > TinyLlama 1.1B + Pythia 1B
- URM (2025): 循环模型在 ARC-AGI 上系统性超越堆叠模型

**RDT-Dx 的贡献在于：(1) 为循环结构提供了 Bridge v2 作为新的稳定性解决方案；(2) 首次将这一范式应用于医疗诊断；(3) 在 9B 规模验证了 138M 可训参数超越 2.2B SFT baseline 的强实证结果。**

---

## 五、建议与行动

### 5.1 学术定位建议

RDT-Dx 的论文应定位为：

> **"Looping Transformers for Medical Diagnosis: A Bridge-Calibrated Recurrent Depth Approach"**

创新点应强调：
1. Bridge v2 作为循环稳定性的新方案（与 Parcae 的约束参数化对比）
2. 首次将循环深度架构引入医疗诊断
3. 在 9B 规模验证了 extreme parameter efficiency (1.52%)

**不建议**声称"用计算深度替代参数规模"是首次提出（这已被多个工作证实）。

### 5.2 需要引用的关键文献

```
必须引用（直接相关）:
1. Huginn (Geiping et al., 2025) - 最接近的架构
2. Relaxed Recursive Transformers (Bae et al., 2024) - depth-wise LoRA
3. Parcae (2025) - 循环稳定性 scaling laws
4. Universal Transformer (Dehghani et al., 2019) - 循环深度的先驱
5. DEQ (Bai et al., 2019) - 隐式深度理论基础

推荐引用（相关但稍远）:
6. Reasoning with Latent Thoughts (Saunshi et al., 2025) - 理论支撑
7. Mixture-of-Recursions (Bae et al., 2025) - adaptive depth
8. Encode-Think-Decode (Koishekenov et al., 2025) - test-time compute scaling
9. Two-Scale Latent Dynamics (Pappone et al., 2025) - 循环几何分析
10. Med-PRM (2025) - 医疗 reasoning reward
```

### 5.3 竞争风险

| 风险 | 等级 | 说明 |
|------|------|------|
| Huginn 已经是完全开源的循环深度模型 | 🔴 高 | 若 Huginn 团队进入医疗领域，RDT-Dx 将失去场景优势 |
| Google MoR 可能有后续大模型版本 | 🟡 中 | 但 Google 不太可能专门做医疗诊断优化 |
| Parcae 的约束参数化可能比 Bridge 更优雅 | 🟡 中 | 建议对比 Bridge v2 vs 约束谱半径的稳定性效果 |

---

## 六、总结

| 维度 | 评估 |
|------|------|
| **核心思想（计算深度 > 参数规模）** | 已有充分先例，非 RDT-Dx 首创 |
| **三段式循环架构** | Huginn 已有几乎相同的设计 |
| **Bridge v2 校准器** | 相对新颖，与 Parcae 的约束参数化不同路径 |
| **last4_mean 聚合** | 工程细节，非核心创新 |
| **医疗诊断场景** | 无先例，是最大差异化点 |
| **强实证结果（138M > 2.2B）** | 在 9B 规模上验证了已有理论 |

**一句话总结**：RDT-Dx 站在 2024-2025 循环深度 Transformer 研究浪潮之上，其架构设计与 Huginn 和 Relaxed Recursive Transformers 高度一致，但 Bridge v2 校准器和医疗诊断场景应用构成了有效的差异化贡献。建议在学术写作中明确承认已有工作，将创新点聚焦于 Bridge v2 的独特性和医疗领域应用。

---

*报告由 Claude Code 自动生成，基于 2026-06-02 的网络搜索。建议在正式引用前核实文献的最新状态。*
