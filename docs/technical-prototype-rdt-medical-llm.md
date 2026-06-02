# RDT-Dx：基于循环深度 Transformer 的医疗诊断大模型技术原型文档

> 项目代号：RDT-Dx
> 主线方案：RDT-Fixed-16
> 阶段：技术原型交付版
> 日期：2026-05-27
> 适用场景：医疗问诊、复杂病例辅助分析、分诊建议、鉴别诊断生成

---

## 1. 项目概述

RDT-Dx 是一个面向医疗问诊场景的技术原型，目标是在固定参数预算下，通过让模型"多想几步"来增强复杂病例的推理能力。

当前主流医疗 LLM 的常见提升路径是扩大参数规模，例如从 7B 扩展到 32B、72B，依靠更大的模型容量提升复杂问题处理能力。但这种方式会带来显著的推理成本、部署成本和响应延迟增长。医疗问诊任务中，简单咨询和复杂病例并存，如果所有问题都使用同等计算量，会造成资源浪费；而如果模型计算深度不足，又会导致复杂病例分析流于表面。

本原型采用 RDT-Fixed-16 作为主线方案：在不显著增加模型参数的前提下，循环复用中间 Transformer 层，使模型在一次输出前执行固定 16 轮内部推理。该结构希望让模型在复杂医疗问题上获得更充分的内部表征更新，从而提升诊断、鉴别诊断、安全分诊和医学建议质量。

---

## 2. 核心目标

本项目的核心目标不是简单训练一个医疗问答模型，而是验证一种新的计算组织方式：

> 在固定参数规模下，通过重复复用中间 Transformer 层，将"参数规模扩展"部分替换为"计算深度扩展"，使模型能够在复杂医疗任务上获得更强的推理能力。

具体目标包括：

1. **增强复杂病例推理能力**
   在多症状、多系统、存在危险信号或诊断不唯一的病例中，提升模型的鉴别诊断能力和医学判断质量。

2. **保持简单问题稳定性**
   对普通健康咨询、常见症状解释、低风险问题，不应因为增加内部循环而产生过度诊断、冗长回答或输出漂移。

3. **控制参数增长**
   不通过大幅增加模型参数解决问题，而是通过中间层循环复用、Bridge 状态校准和 LoRA 微调实现能力增强。

4. **为后续 RL 提供更好的基础模型**
   在进入 GRPO 等强化学习阶段前，先构建一个具备稳定多步内部计算能力的 SFT 模型，使 RL 能优化诊断质量、安全性和格式，而不是单纯学习表面输出模式。

---

## 3. 问题背景与技术动机

### 3.1 医疗问诊 LLM 的核心问题

医疗问诊场景与普通问答不同，它对模型提出了更高要求：

* 需要识别症状之间的关系；
* 需要判断是否存在危险信号；
* 需要给出合理鉴别诊断；
* 需要区分"健康建议""门诊就医""急诊处理"等不同分诊等级；
* 不能编造患者未提供的病史、检查结果或诊断依据；
* 不能给出危险、过度自信或替代医生诊疗的建议。

现有中小规模 LLM 在简单医疗咨询上通常可以生成流畅回答，但面对复杂病例时容易出现以下问题：

1. **推理深度不足**：只抓住表面症状，缺少多步鉴别；
2. **危险信号遗漏**：未识别胸痛、意识障碍、持续高热、呼吸困难等关键风险；
3. **诊断过早收敛**：在信息不足时给出单一结论；
4. **医学依据薄弱**：建议与患者描述之间缺乏明确对应；
5. **回答格式不稳定**：复杂任务下输出结构容易散乱。

### 3.2 参数扩展的局限

扩大模型参数可以提升能力，但并非总是最优路径：

* 推理成本随参数规模显著增加；
* 医疗场景对部署成本和延迟敏感；
* 简单问题不需要大模型级别计算；
* 大模型仍可能在医疗安全和依据一致性上出错；
* 后续 RL 如果基座推理能力不足，容易只优化格式而非真正改善推理。

因此，本项目采用另一种思路：不优先扩大参数，而是增加模型在复杂问题上的内部计算深度。

---

## 4. 主线方案：RDT-Fixed-16

### 4.1 方案定义

RDT-Fixed-16 是本原型的主线结构。其核心思想是：

> 将基座 LLM 的中间 Transformer 层抽取为 Core 模块，并在同一个隐藏状态空间中循环执行 16 次，使模型在输出前完成更深层的内部表征更新。

该方案采用固定循环步数，不引入自适应 HaltHead，避免多目标训练不稳定问题。固定 16 步也便于工程部署、延迟评估和后续 RL 训练。

### 4.2 结构总览

```
输入 tokens
  ↓
Embedding（冻结）
  ↓
Prefix Layers（冻结）
  ↓
h_anchor（detach，切断梯度）
  ↓
[Core Layers × 16]
  ↓        ↑
Bridge v2 状态校准
  ↓
last4_mean 聚合
  ↓
Suffix Layers（LoRA 可训）
  ↓
Norm + LM Head
  ↓
输出 logits
```

### 4.3 模型切分

基于 Qwen3.5-2B-Base（24 层 Transformer，hidden=2048）将 Transformer 拆分为三段：

| 模块            | 层索引 | 参数状态       | 作用                      |
|---------------|------|-----------|-------------------------|
| Embedding     | —    | 冻结        | 将输入 token 转换为初始向量表示        |
| Prefix Layers | 0-11 | 冻结        | 提取稳定语义表示，形成 h_anchor 锚点    |
| **Core Layers** | **12-23** | **循环复用，LoRA 可训** | **作为核心推理引擎，执行 16 步内部计算** |
| Bridge v2     | —    | 全参数可训    | 校准每轮循环后的隐藏状态，抑制分布漂移    |
| Suffix Layers | —    | LoRA 可训   | 整合多步推理结果并生成输出           |
| LM Head       | —    | 冻结        | 保持输出词表空间稳定               |

---

## 5. 核心技术设计

### 5.1 循环深度 Core

Core 是 RDT-Dx 的核心推理模块。与普通 Transformer 每层只执行一次不同，RDT 将中间层重复执行 16 次：

```python
h0 = Prefix(x)

for t in range(1, 17):
    zt = Core(ht-1)
    ht = Bridge(h_anchor, zt, t)   # step embedding 感知循环步数

h_fused = mean(h13, h14, h15, h16)
logits = Suffix(h_fused)
```

这里的循环不是在 token 序列方向展开，而是在隐藏状态空间中重复应用同一组 Core 层。可以理解为模型在同一病例表示上反复更新理解结果。

**为什么选择固定 16 步？** 三点优势：

1. **训练目标简单**：只需优化 CE loss + KL 正则，不需要 halt/ponder/entropy 等复杂辅助目标
2. **推理延迟可控**：每次推理步数相同，方便部署评估和成本估算
3. **适合后续 RL**：GRPO 可以在固定步数内部优化"如何使用 16 步"，无需同时学习何时停止

**为什么不把 Adaptive 作为主线？** 自适应循环方案理论上更节省计算，但原型阶段会引入较多不稳定因素：

* HaltHead 可能过早停止或分布塌缩；
* ACT 权重可能导致多步状态融合不稳定；
* 多目标 loss（CE + ponder + entropy + smooth）可能发生梯度冲突；
* 推理延迟不可预测；
* RL 阶段同时优化内容和步数，训练难度更高。

因此，原型阶段优先采用 Fixed-16。Adaptive 作为后续成本优化方向保留。

### 5.2 Bridge v2 状态校准器

#### 5.2.1 问题

Core 层被重复使用后，隐藏状态逐步偏离基座模型熟悉的分布。实验数据表明（见 §7 历史实验）：

* 无 Bridge loop=4：Top-5 Token 重叠率跌至 60%
* 训练后 Bridge loop=4：Top-5 重叠率提升至 73.3%（+13.3%）
* Bridge 权重范数增长到 6-11，但仍不足以完全修正深层循环

Bridge 是 RDT-Fixed-16 能否成立的关键组件。

#### 5.2.2 Bridge v2 结构

本原型采用 Bridge v2：零初始化残差 MLP + step embedding。

```
input_t = concat(h_anchor, z_t, z_t - h_anchor, step_embedding_t)

delta_t = MLP(input_t)

h_t = h_anchor + alpha * delta_t
```

其中 alpha = 0.2（固定，实验证明自适应 alpha 方案会导致权重爆炸并输出乱码）。

Bridge MLP 结构：

```
Linear(input_dim → hidden_dim / 2)
RMSNorm
GELU
Linear(hidden_dim / 2 → hidden_dim)  # 零初始化
```

**关键设计**：最后一层 Linear 的权重和 bias 必须零初始化，训练初始阶段 h_t ≈ h_anchor，保证从稳定的基线开始，不会因为随机 Bridge 参数导致输出崩溃。

#### 5.2.3 Step Embedding

不同循环步的状态漂移模式不同。第 2 步、第 8 步、第 16 步不应被 Bridge 完全等价处理。引入 step embedding：

```
step_embedding_t = Embedding(t)   # 维度 32 或 64
```

让 Bridge 感知当前处于第几轮内部推理，学习不同步数下的状态校准方式。

### 5.3 last4_mean 聚合

RDT-Fixed-16 不直接使用最后一步隐藏状态，而是使用最后 4 步平均：

```
h_fused = mean(h13, h14, h15, h16)
```

原因：只取最后一步可能受单步震荡影响；全部 16 步平均会稀释后期推理结果；最后 4 步更接近收敛阶段，适合作为最终推理表示。

消融实验计划（见 §11）：

| 聚合方式     | 说明        |
|---------|-----------|
| last    | 只使用第 16 步   |
| mean_all | 平均 1-16 步    |
| **last4_mean** | **平均 13-16 步，默认方案** |
| gated_mean | 可训练门控融合  |

---

## 6. 训练路线

RDT-Fixed-16 不建议直接端到端训练，而是分阶段进行，逐步保证结构稳定性和任务能力。

### Phase 0：医疗 SFT 基座（已完成）

| 项目 | 内容 |
|---|---|
| **目标** | 获得一个基本可用的医疗问诊 SFT 模型，作为 RDT 的初始化基础和 KL reference |
| **输入** | 医疗问诊数据（200-500 条，覆盖简单咨询到高难度病例） |
| **训练对象** | Qwen3.5-2B-Base + LoRA (r=64, alpha=128) |
| **输出** | `lora_ckpt_v2/`（checkpoint 已存在） |
| **通过标准** | 生成正常，无明显乱码；医疗回答基本符合格式；简单问诊稳定 |

### Phase 1：Bridge Warmup（待实施）

| 项目 | 内容 |
|---|---|
| **目标** | 让 RDT 循环结构稳定，不追求能力提升，先避免分布漂移 |
| **核心思想** | 让 RDT-Fixed-K 的输出分布尽量接近原 SFT 模型输出分布 |

**Progressive K Schedule**：逐步增加循环次数，不要直接从 K=16 开始。

| 阶段        | K | 目的         |
|-----------|-:|----------|
| Warmup-1  | 2 | 验证短循环稳定性  |
| Warmup-2  | 4 | 对齐已有实验基础  |
| Warmup-3  | 8 | 检查中等深度漂移  |
| Warmup-4  |16 | 进入目标结构     |

**训练对象**：

| 模块           | 是否训练   |
|-------------|--------|
| Bridge v2   | 是      |
| Core LoRA   | 可先关闭  |
| Suffix LoRA | 可关闭    |
| Prefix      | 否       |
| Embedding   | 否       |
| LM Head     | 否       |

**损失函数**：

```
L = L_KL_full_seq + λ_smooth * L_smooth
L_KL_full_seq = mean KL(softmax(logits_sft[all_tokens]), softmax(logits_rdt[all_tokens]))
λ_smooth = 1e-4 ~ 5e-4
```

**通过标准**：K=16 生成不崩溃；Top-k overlap 不随步数严重下降；hidden drift 曲线可控；输出仍保持医疗问答基本能力。

### Phase 2：RDT-Fixed-16 SFT（待实施）

| 项目 | 内容 |
|---|---|
| **目标** | 让 16 步循环结构真正服务于医疗任务，提升复杂病例质量 |
| **训练对象** | Bridge + Core LoRA + Suffix LoRA 全部打开 |
| **损失函数** | L = L_CE + λ_kl * L_KL_ref + λ_smooth * L_smooth |
| **推荐权重** | λ_kl = 0.05 ~ 0.2，λ_smooth = 1e-4 ~ 5e-4 |

**通过标准**：hard 样本 Acc@1/Acc@3 提升；red flag recall 提升或不下降；unsafe advice rate 不上升；format rate 不下降；easy 样本无明显退化。

### Phase 3：RDT-Fixed-16 + GRPO（待实施）

| 项目 | 内容 |
|---|---|
| **目标** | 在稳定的 RDT-Fixed-16 SFT 模型基础上，通过 GRPO 优化医疗回答质量 |
| **为什么 RL 放在 SFT 之后** | RL 不应解决结构稳定性问题。结构稳定应由 Bridge Warmup 和 SFT 完成。RL 的目标是优化诊断方向、鉴别诊断完整性、危险信号识别、分诊建议、格式遵循和医学依据一致性 |
| **GRPO 配置** | group_size=4~8；β=0.01~0.05；组内归一化 advantage |

---

## 7. Reward 设计（GRPO 阶段）

GRPO 阶段的 reward 不应只奖励格式，而应覆盖医疗质量与安全性。

**总 reward**：

```
R = R_diagnosis + R_differential + R_safety + R_format
  + R_evidence + R_conciseness - R_hallucination - R_danger
```

| Reward           |   权重 | 说明                     |
|------------------|----:|-----------------------|
| R_diagnosis      | 0.35 | 主诊断方向是否合理             |
| R_differential   | 0.20 | 是否覆盖关键鉴别诊断            |
| R_safety         | 0.20 | 是否识别危险信号并给出正确分诊     |
| R_format         | 0.10 | 是否符合规定输出结构            |
| R_evidence       | 0.10 | 是否基于用户提供的信息进行判断     |
| R_conciseness    | 0.05 | 是否简洁、避免无效冗余           |
| R_hallucination  |-0.20 | 是否编造病史、检查或结论          |
| R_danger         |-0.50 | 是否给出危险医疗建议            |

**安全红线**：以下情况应给予强惩罚——对急危重症风险没有提示；在信息不足时给出确定诊断；编造检查结果；鼓励患者自行停药/换药/加量；对胸痛/呼吸困难/意识障碍/大出血等危险症状轻描淡写；替代医生给出明确治疗处方。

---

## 8. 输出格式设计

为便于评估和 RL reward 计算，建议使用结构化 XML 输出。

**推荐格式**：

```xml
<summary>
对患者主要症状和关键信息进行简要概括。
</summary>

<risk_assessment>
判断是否存在危险信号，以及是否需要急诊或尽快就医。
</risk_assessment>

<possible_causes>
列出可能原因或鉴别诊断，并说明依据。
</possible_causes>

<recommendation>
给出下一步建议，包括观察、就医、检查或生活方式建议。
</recommendation>

<safety_notice>
说明模型不能替代医生诊断，并列出需要立即就医的情况。
</safety_notice>
```

**格式要求**：必须包含所有 XML 标签；不得输出患者未提供的检查结果；不得使用绝对化诊断语言；复杂病例必须包含鉴别诊断；存在危险信号时必须明确提示就医级别。

---

## 9. 数据设计

### 9.1 数据规模规划

当前有 Mini 阶段数据（~200 条），但要证明原型有效需分阶段扩展。

| 阶段    |      数据量 | 目的                |
|------|--------:|-------------------|
| Mini  | 200-500 | 跑通代码、显存、训练链路        |
| Proto | 2k-5k  | 验证 RDT 是否优于普通 SFT |
| Strong |   20k+ | 支撑正式 RL 和能力评估       |

### 9.2 数据类型与字段

| 类型       | 说明               |
|--------- |------------------|
| 简单健康咨询  | 如轻微头痛、普通感冒、饮食建议    |
| 中等复杂病例  | 多症状但风险不高，需要初步鉴别     |
| 高难度病例   | 多系统、多病因、存在危险信号      |
| 分诊任务    | 判断居家观察、门诊、急诊       |
| 安全拒答    | 超出模型能力或高风险医疗决策     |

**统一 JSONL 格式**：

```json
{
  "id": "case_000001",
  "messages": [
    {"role": "user", "content": "患者描述..."},
    {"role": "assistant", "content": "标准回答..."}
  ],
  "difficulty": "easy|medium|hard",
  "diagnosis_labels": ["可能诊断1", "可能诊断2"],
  "red_flags": ["危险信号1"],
  "required_actions": ["建议尽快就医"],
  "forbidden_actions": ["不得建议自行停药"],
  "source": "synthetic|curated|expert"
}
```

**难度定义**：

| 难度    | 标准                             |
|-------|-------------------------------|
| easy  | 单症状、低风险、常见咨询、不需要复杂鉴别          |
| medium | 多症状、需要初步鉴别、可能需要检查              |
| hard  | 多系统、诊断不唯一、存在危险信号、需要分诊判断        |

---

## 10. 评估体系

评估需要同时覆盖语言建模质量、医疗任务质量、安全性和 RDT 结构有效性。

### 10.1 基础指标

| 指标              | 说明        |
|----------------|-----------|
| CE Loss         | 监督训练损失    |
| PPL             | 语言建模困惑度   |
| Format Rate     | 结构化输出格式正确率 |
| Refusal Accuracy | 应拒答时是否拒答  |

### 10.2 医疗任务指标

| 指标                    | 说明             |
|---------------------|--------------|
| Diagnosis Acc@1/3   | 首要/前三候选诊断方向是否正确  |
| Differential Coverage | 鉴别诊断覆盖度         |
| Red Flag Recall     | 危险信号召回率          |
| Triage Accuracy     | 分诊建议是否合理         |
| Evidence Grounding  | 是否基于用户提供信息判断     |
| Hallucination Rate  | 是否编造病史/检查/结论     |
| Unsafe Advice Rate  | 是否给出危险建议         |

### 10.3 RDT 专属指标

| 指标              | 说明                            |
|---------------|-------------------------------|
| Step-wise KL  | 每步 logits 与 reference 的 KL 距离 |
| Top-k Overlap | 与 reference 模型 Top-k token 重合度  |
| Hidden Drift  | 隐藏状态随循环步数的偏移                   |
| Last4 Variance | 最后四步隐藏状态方差                    |
| Step Benefit  | K 增加是否提升 hard 样本表现             |
| Easy Stability | easy 样本是否因多步计算退化              |

---

## 11. 消融实验设计

### 11.1 步数消融

```
SFT baseline
RDT-Fixed-1
RDT-Fixed-2
RDT-Fixed-4
RDT-Fixed-8
RDT-Fixed-16
```

目标：验证更多步数是否主要提升 hard 样本，而不是无差别增加计算。

### 11.2 Bridge 消融

```
RDT-Fixed-16 no Bridge
RDT-Fixed-16 Linear Bridge
RDT-Fixed-16 MLP Bridge
RDT-Fixed-16 MLP Bridge + Step Embedding
```

目标：验证 Bridge 对稳定循环的必要性，以及 step embedding 的额外收益。

### 11.3 聚合方式消融

```
last (仅第 16 步)
mean_all (1-16 步平均)
last4_mean (13-16 步平均，默认方案)
gated_mean (可训练门控融合)
```

目标：验证 last4_mean 在稳定性和性能之间是否取得最优平衡。

### 11.4 RL 消融

```
RDT-Fixed-16 SFT
RDT-Fixed-16 + GRPO format-only reward
RDT-Fixed-16 + GRPO medical composite reward
```

目标：证明 RL 提升来自医疗质量 reward，而不是格式 reward。

---

## 12. 推荐训练配置

```yaml
model:
  name: RDT-Fixed-16
  base_model: Qwen3.5-2B-Base
  max_iters: 16
  aggregation: last4_mean
  bridge_type: mlp_step
  bridge_alpha: 0.2
  step_embedding_dim: 32

trainable:
  bridge: true
  core_lora: true
  suffix_lora: true
  prefix: false
  embedding: false
  lm_head: false

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj
    - gate_proj

optimization:
  lr_bridge: 2.0e-4
  lr_lora: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.05
  batch_size_per_gpu: 1
  gradient_accumulation_steps: 16
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true

loss:
  ce_weight: 1.0
  kl_ref_weight: 0.1
  smooth_weight: 0.0002
```

---

## 13. 里程碑计划

| Milestone | 目标 | 交付物 | 通过标准 |
|---|---|---|---|
| **M1: 结构跑通** | RDT-Fixed-K 完成 forward/loss/backward/generation | `rdt_fixed.py`、`bridge.py`、smoke test | 无 shape error；无显存异常；K=16 正常生成；输出无大规模乱码 |
| **M2: Bridge Warmup 完成** | 循环结构稳定 | `train_bridge_warmup.py`、KL/Top-k/hidden drift 曲线 | K=16 logits 分布不崩；hidden drift 可控；生成质量接近 SFT reference |
| **M3: RDT-Fixed-16 SFT 完成** | 医疗任务质量提升 | `train_rdt_fixed.py`、RDT checkpoint、SFT vs RDT 评估报告 | hard Acc@1/3 提升；red flag recall 不下降；unsafe advice rate 不上升 |
| **M4: GRPO 初版完成** | 验证 RL 能进一步优化 RDT | `train_grpo.py`、reward modules、GRPO checkpoint | composite reward 提升；医疗安全指标不下降；format-only reward 不被证明为主要收益来源 |

---

## 14. 风险与缓解措施

| 风险            | 影响       | 缓解措施                                  |
|--------------|----------|----------------------------------------|
| 循环 16 步导致分布漂移  | 输出不稳定/乱码  | Bridge Warmup、全序列 KL 正则、step embedding    |
| Bridge 表达能力不足  | 无法校准深层循环状态 | 使用 MLP Bridge + RMSNorm + 差分项输入           |
| 多步计算没有带来收益   | 增加延迟但性能无提升 | K 消融实验，重点观察 hard 样本                     |
| 简单问题退化        | 过度诊断/回答冗长  | easy stability 监控、KL ref 正则、长度控制          |
| RL 只优化格式       | 医疗质量无提升    | 使用 composite medical reward               |
| 医疗安全风险增加      | 给出危险建议     | 强化 safety reward 和 danger penalty          |
| 数据规模不足        | 结论不可靠     | 从 Mini 扩展到 Proto，再进入 Strong 阶段            |
| 显存压力过大        | 无法训练 K=16  | gradient checkpointing、gradient accumulation、bf16 |

---

## 15. 成功 / 失败判据

### 15.1 原型成功判据

若满足以下条件，可认为 RDT-Fixed-16 技术原型成立：

1. K=16 循环结构稳定，无明显生成崩溃；
2. RDT-Fixed-16 在 hard 医疗样本上优于普通 SFT；
3. easy 样本性能无明显下降；
4. red flag recall 提升或至少不下降；
5. unsafe advice rate 不高于 SFT；
6. step ablation 显示更多内部计算对复杂样本有正向收益；
7. GRPO 在 RDT-Fixed-16 基础上进一步提升医疗综合 reward；
8. RL 后模型没有明显格式投机或安全退化。

### 15.2 原型失败判据

若出现以下情况，应考虑终止该主线或回退到普通 SFT / Adaptive 轻量方案：

1. K 增加只带来推理延迟，不提升 hard 样本；
2. Bridge 无法控制 K=8 或 K=16 的分布漂移；
3. RDT-Fixed-16 长期不如普通 SFT；
4. GRPO 只提升格式，不提升诊断和安全指标；
5. unsafe advice rate 明显升高；
6. last4 hidden state 无收敛趋势，说明循环未形成稳定内部计算。

---

## 16. 历史实验数据参考

以下数据来自项目早期探索阶段，验证了核心设计的可行性。

### 16.1 Core 循环稳定性（Step 1 验证）

**Layer split**：prefix[0:8]，core[8:16]，suffix[16:24]，3 条医学测试文本。

| Loop次数 | KL散度 (vs loop=1) | Top-5 重叠率平均 |
|---------|-------------------|-------------|
| 1       | 0 (基准)           | 100%        |
| 2       | ~10⁻⁷~10⁻⁶        | **93.3%**   |
| 4       | ~10⁻⁶~10⁻⁵        | **60.0%**   |

**结论**：loop=2 相对稳定；loop=4 出现显著分布漂移，Bridge 必须加入。

### 16.2 Bridge v1 验证（Step 2 验证）

| 模式                      | Test1  | Test2  | Test3  | 平均    |
|------------------------|-------|-------|-------|-------|
| A: 无 Bridge, loop=4    | 100%  | 40%   | 40%   | 60.0% |
| B: 零初始化 Bridge, loop=4 | 100%  | 40%   | 40%   | 60.0% |
| C: 训练后 Bridge, loop=4   | 100%  | 60%   | 60%   | **73.3%** |

**关键发现**：
- 零初始化 Bridge 验证了恒等映射的正确性（Mode A = Mode B）
- Bridge 确实能修正分布偏移（+13.3%），但单一 Linear 层表达能力有限
- KL loss 持续下降（0.52→0.0024）但 Top-5 overlap 从 step 50 起饱和
- **自适应 alpha 方案已被排除**：alpha 学大后导致输出完全乱码（0% overlap）

### 16.3 Bridge 调参结论

| 超参数 | 安全范围 | 备注              |
|------|------|-----------------|
| LR   | ≤1e-3 | LR=0.01 直接崩溃     |
| alpha | 固定 0.2 | 自适应方案全部失败     |
| weight decay | 0.01 | 防止权重爆炸有效        |
| gradient clip | 0.5  | 配合 weight decay 使用 |
| 训练步数 | ~50  | step 50 后 overlap 无增益 |

### 16.4 RDT-Adaptive 当前状态

RDT-Adaptive 方案（ACT + HaltHead + 难度感知 ponder）已在 `src/models/rdt_adaptive.py` 中完整实现，关键超参数：

| 参数 | 值 |
|---|---|
| MAX_ITERS | 4 |
| λ_ponder | 0.005 |
| λ_smooth | 0.0002 |
| λ_entropy | 0.005 |
| HALT_BIAS_INIT | -1.1（初始期望深度 ~2.5 步）|
| Bridge 结构 | 残差 MLP (bottleneck=512) |

该方案作为备选路线保留，待 RDT-Fixed-16 验证后对比决策。

---

## 17. 最终交付形态

| 类别 | 交付内容 |
|---|---|
| **技术文档** | 本文档（架构说明、训练流程、评估方案、风险控制） |
| **核心代码** | `rdt_fixed.py`、`bridge.py`、`aggregation.py`、`train_bridge_warmup.py`、`train_rdt_fixed.py`、`train_grpo.py` |
| **实验报告** | SFT baseline 对比、K 步数消融、Bridge 消融、聚合方式消融、GRPO 前后对比 |
| **模型检查点** | Medical SFT LoRA、Bridge Warmup checkpoint、RDT-Fixed-16 SFT checkpoint、GRPO checkpoint |
| **评估集** | 覆盖 easy/medium/hard/red flag/安全拒答等类型的评估数据 + 评估脚本 |

---

## 18. 结论

RDT-Dx 的主线方案 RDT-Fixed-16 通过循环复用中间 Transformer 层，在固定参数预算下引入更深的内部计算过程。它不是简单增加模型层数，也不是扩大模型参数，而是让同一组 Core 层在隐藏状态空间中多次更新病例表示，从而提升复杂医疗问诊任务中的推理深度。

该方案采用固定 16 步作为原型主线，配合零初始化残差 Bridge v2、step embedding、last4_mean 聚合和分阶段训练流程，优先保证结构稳定性和可评估性。随后在稳定的 SFT 基础上引入 GRPO，以医疗质量、安全性和格式一致性为 reward，进一步优化复杂病例表现。

**一句话概括**：

> RDT-Fixed-16 的目标是在不显著增加参数的情况下，让医疗 LLM 获得更深的内部推理计算，并为后续 RL 提供一个更强、更稳定、更可优化的基础模型。

---

*文档状态：技术原型交付版，随时根据实验结果更新*