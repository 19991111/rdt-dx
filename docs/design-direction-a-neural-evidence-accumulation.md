# RDT-Dx Direction A：基于神经证据累积的循环深度医疗诊断架构设计文档

> **项目代号**：RDT-Dx
> **方案名称**：Direction A — Neural Evidence Accumulation (RDT-EA)
> **基线方案**：RDT-Fixed-16（详见 [technical-prototype-rdt-medical-llm.md](./technical-prototype-rdt-medical-llm.md)）
> **阶段**：架构设计 / 待实现
> **日期**：2026-06-02
> **适用场景**：需要多步诊断推理的复杂医疗问诊

---

## 1. 动机与认知科学基础

### 1.1 现有原型的成就与局限

RDT-Fixed-16 在初步验证中取得了显著成果（详见 [PRELIMINARY_REPORT.md](./PRELIMINARY_REPORT.md)）：

- **Phase C 核心结论**：138M 可训参数的 RDT-Fixed-16 在 hard 医疗样本上 CE=0.43，远优于 2.2B 参数的 SFT baseline (CE=1.39) —— **1/16 的参数实现 3.2× 的改善**
- **Hard CE 单调递减**：K=4(0.75)→K=8(0.59)→K=16(0.43)，验证了"更多循环步数 → 更强推理能力"的方向正确
- **收益不对称性**：Hard 样本收益（-69%）远超 Easy 样本波动（±0.2），说明循环并非无差别增加计算

然而，[RELATED_WORK_REPORT.md](./RELATED_WORK_REPORT.md) 的分析暴露了一个根本性问题：

> **RDT-Dx 的架构设计与 Huginn (ICLR 2025)、Relaxed Recursive Transformers (ICLR 2025)、Parcae (2025) 高度重叠。** "计算深度 > 参数规模"的核心主张已被充分验证，三段式架构与 Huginn 几乎同构。当前差异化仅在于 Bridge v2 校准器设计（中等新颖）和医疗场景应用（场景新颖）。

具体来说，当前方案存在以下结构性问题：

| 问题 | 表现 | 根源 |
|------|------|------|
| **固定步数无理论依据** | 为何是 16 步而非 8 步或 32 步？如果 reviewer 问"16 步的理论依据是什么"，无法给出认知层面的回答 | 纯工程选型，缺乏理论锚点 |
| **Bridge v2 无跨步记忆** | 每步校准独立执行 `h_t = h_anchor + α · MLP(…)`，Bridge 不知道上一步"决定了什么"，无法建模诊断信息的逐步累积 | 校准器是 stateless 的 |
| **差异化脆弱** | 若 Huginn 团队用其开源模型做一次医疗微调并发表，RDT-Dx 的场景优势即刻消失 | 场景差异不是架构创新 |

### 1.2 医疗诊断的认知科学模型

认知科学对医学诊断推理的建模已有数十年积累，三个经典模型构成了本方案的理论基础：

**Drift-Diffusion Model (DDM, Ratcliff 1978)**：决策者在每次获得新证据时，向决策边界方向累积一小步，当累积证据量超过某个阈值(boundary)，做出决策。DDM 已被大量行为实验和神经成像研究验证，是认知心理学中最稳健的决策模型之一。

**Sequential Probability Ratio Test (SPRT, Wald 1945)**：在统计学上，SPRT 是达到给定置信水平所需样本量最小的序列假设检验方法。医生在诊断中天然在做 SPRT——每问一个问题、每观察一个症状，都在更新对候选诊断的置信度。

**Dual-Process Theory (Kahneman 2011; Evans & Stanovich 2013)**：医生诊断 = System 1（快速模式识别，经验驱动的直觉） + System 2（慢速分析推理，逐条证据评估）。复杂的鉴别诊断本质上是 System 2 主导的序列证据累积过程。

### 1.3 核心洞见：循环深度 Transformer = 神经证据累积

> **循环深度 Transformer 的每一步循环，天然对应认知科学中"累积一条新证据"的过程。这不是比喻，而是可以形式化的结构对应。**

具体来说：

| 认知科学概念 | RDT 架构对应 | 形式化 |
|-------------|-------------|--------|
| 证据累积步 t | 第 t 次 Core 循环 | `t ∈ [1, T]` |
| 当前证据 e_t | Core 输出的新信息 | `e_t = EvidenceExtract(z_t, h_anchor)` |
| 诊断信念 b_t | 累积到第 t 步的诊断状态 | `b_t = Update(b_{t-1}, e_t)` |
| 决策边界 θ | 停止阈值 | `Stop when c(b_t) > θ` |
| 惊喜信号 | 矛盾检测 | `surprise = ||e_t - E[e_t\|b_{t-1}]||` |

当前 RDT-Fixed-16 已经在隐式地做证据累积——Bridge v2 中的 `(z_t - h_anchor)` 项本质上是一个粗粒度的"证据信号"，last4_mean 聚合也暗示了"后期步骤更接近收敛"的直觉。但这一切都是隐式的、未被形式化的。

**Direction A 的核心贡献**：将这一隐式过程**显式化、形式化、可验证化**。

### 1.4 与已有工作的认知科学联系

这是 RDT-Dx 区别于所有已有循环 Transformer 工作的关键：

- **Huginn (2025)**：架构驱动（"Prelude → Recurrent → Coda"），无认知科学框架
- **Parcae (2025)**：工程驱动（"如何让循环稳定" → 谱半径约束），无认知科学框架
- **MoR (2025)**：效率驱动（"不同 token 需要不同深度" → Router），无认知科学框架
- **Universal Transformer (2019)**：计算理论驱动（"Turing-complete if recurrent"），无认知科学框架

> **RDT-EA 是首个将循环深度 Transformer 与认知科学证据累积理论建立形式化连接的工作。** 这种"领域认知理论 → 架构设计"的范式，比"尝试一个 trick → 发现有效"的工程驱动范式在学术价值上高一个量级。

---

## 2. 架构总览

### 2.1 RDT-Fixed-16 → RDT-EA 演进

```
RDT-Fixed-16（当前方案）:
  Input → Prefix → h_anchor
    → [Core + Bridge v2(stateless, step_emb)] × 16
    → last4_mean → Suffix → Output

RDT-EA（本方案）:
  Input → Prefix → h_anchor → b₀ = 0（信念初始化）
    → [Core → 证据提取 → 信念更新 → 置信度评估
       → 矛盾检测 → EABridge(置信度门控校准) ]
    → 循环直到: (置信度 > θ_high 且 无矛盾) 或 t = T_max
    → 信念加权聚合 → Suffix → Output
```

### 2.2 四个核心增强模块

| 模块 | 增强对象 | 认知科学映射 | 核心创新 |
|------|---------|-------------|---------|
| **EABridge** | Bridge v2（`src/bridge.py`） | 证据累积 + 信念追踪 | 显式维护诊断信念状态 b_t，置信度门控校准强度 |
| **自适应停止** | 固定 16 步循环 | DDM 决策边界 | 置信度阈值触发停止，替代固定步数 |
| **矛盾检测** | （新增模块） | 认知失调 / 惊喜信号 | 监测新证据与当前信念的矛盾，强制深化推理 |
| **信念轨迹正则化** | （新增训练目标） | 证据单调性 / 认知规范 | 惩罚无证据支持的信念反转，鼓励"发散→收敛"模式 |

### 2.3 与 RDT-Fixed-16 的关系

RDT-EA 不是替代 RDT-Fixed-16，而是其自然进化。关键降级保证：

- **`belief_dim = 0`** → EABridge 退化为 Bridge v2（完全等价，bitwise identical）
- **`loop_mode = "fixed"`** → 禁用自适应停止，运行固定 `num_iters` 步
- **同时启用两个降级** → RDT-EA 完全等价于 RDT-Fixed-16

这一设计确保：(1) 可以公平对比 RDT-EA vs RDT-Fixed-16 的增量收益；(2) 如果 EABridge 不 work，不会丢失已有成果。

---

## 3. EABridge：证据累积桥接器

### 3.1 设计动机

当前 Bridge v2 的核心操作是：

```
h_t = h_anchor + α · MLP(concat(h_anchor, z_t, z_t - h_anchor, step_emb(t)))
```

这个设计的结构性缺陷在于：

1. **Stateless**：每步校准彼此独立，Bridge 不知道第 t-1 步的校准做了什么
2. **无诊断意识**：校准强度仅依赖于 step embedding（学到的是"第 8 步该用多大力校准"），而非"当前已经累积了多少诊断证据"
3. **α 固定为 0.2**：历史实验证明自适应 α 会导致爆炸，因为缺乏约束 α 的依据——但如果有一个可靠的置信度信号，α 就可以被约束

EABridge 的核心改进：**让 Bridge 拥有跨步记忆（belief state），并根据累积状态动态调节校准行为。**

### 3.2 信念状态定义与初始化

```
b₀ = 0 ∈ R^{n_belief}    # 零初始化，对应"无任何诊断假设"的状态
```

信念维度 `n_belief` 默认 256（对于 hidden=2048 的模型为 hidden/8）。信念状态不是概率分布——它是一个压缩的潜在表征，编码了"模型目前为止认为哪些诊断假设被证据支持"。

### 3.3 证据提取

```
evidence_t = EvidenceExtractor(z_t, h_anchor)
           = Linear_GELU(concat(z_t, z_t - h_anchor))
           ∈ R^{n_belief}
```

从当前 Core 输出中提取"新证据"表征。`(z_t - h_anchor)` 项保留了 Bridge v2 的差分信号——它天然编码了"本步偏离锚点的方向和幅度"，与证据累积的"变化量"语义一致。

**正则化约束**：证据提取器的权重 decay 设为 0.1（高于其他模块的 0.01），防止模型将所有信息都通过证据路径传递而绕开 hidden state。

### 3.4 信念更新（GRU-based）

```
b_t = GRUCell(b_{t-1}, evidence_t)
    = (1 - gate_t) ⊙ b_{t-1} + gate_t ⊙ candidate_t
```

使用单层 GRUCell 而非简单的线性累加。原因：

- **门控机制**允许模型选择性整合证据：当 evidence_t 为空（如 padding 步）时，gate ≈ 0，b_t ≈ b_{t-1}
- **非线性变换**比线性加和（`b_t = b_{t-1} + evidence_t`）表达能力更强，能建模证据间的交互
- **轻量**：GRUCell 参数量 = 3 × n_belief² ≈ 3 × 256² ≈ 197K，可忽略

### 3.5 置信度评估

支持两种模式，通过 `confidence_type` 配置切换：

**方式一：Norm-based（默认）**

```
c_t = ||b_t||₂ / (||b_t||₂ + τ)
```

其中 τ 是可学习标量，初始化为 1.0。物理直觉：信念向量范数越大 → 累积的证据越多 → 置信度越高。优点：**高度可解释**，不需要额外网络；缺点：表达能力有限。

**方式二：Learned**

```
c_t = sigmoid(Linear(n_belief → 1)(b_t))
```

优点：更灵活，能学习到"某些维度对置信度更重要"；缺点：可解释性下降。消融实验中应对比两种方式。

### 3.6 置信度门控校准

EABridge 的核心校准公式：

```
# Bridge v2（现有）:
h_t = h_anchor + α · MLP(concat(h_anchor, z_t, z_t - h_anchor, step_emb))

# EABridge（本方案）:
calib_input = concat(h_anchor, z_t, z_t - h_anchor, step_emb, project(b_t), c_t)
calib_delta = MLP(calib_input)
gate_calib = 1.0 - c_t   # 置信度越高，校准越弱
h_t = h_anchor + α · gate_calib · calib_delta
```

**直觉**：

- 置信度低（c_t → 0）→ gate_calib → 1.0 → 强校准，防止搜索阶段的漂移
- 置信度高（c_t → 1）→ gate_calib → 0.0 → 弱校准，允许 hidden state 表达高置信度的诊断结论
- 这种"先校准后放松"的模式与人类认知的"从探索到收敛"过程一致

**与 Bridge v2 的对比**：

| 维度 | Bridge v2 | EABridge |
|------|----------|----------|
| 输入 | h_anchor + z_t + diff + step_emb | + project(b_t) + c_t |
| 校准强度 | 固定 α = 0.2 | α · (1-c_t)，动态调节 |
| 跨步记忆 | 无 | b_t 跟踪累积诊断状态 |
| 初始化 | 零初始化最后一层 → h_t ≈ h_anchor | 同 Bridge v2，训练初始 c_t ≈ 0 → 行为等价 |

### 3.7 降级兼容性

```python
class EABridge(nn.Module):
    def __init__(self, hidden_dim, n_belief=256, ...):
        if n_belief == 0:
            # 完全退化为 Bridge v2 —— 无信念状态、无证据提取、无置信度门控
            self._degraded = True
            self.bridge_v2 = BridgeV2(hidden_dim, ...)
        else:
            self._degraded = False
            self.evidence_extractor = ...
            self.belief_gru = nn.GRUCell(n_belief, n_belief)
            self.confidence_proj = ...
            self.calib_mlp = ...
```

**关键保证**：当 `n_belief = 0` 且 `loop_mode = "fixed"` 时，RDT-EA 的 forward 输出与 RDT-Fixed-16 **bitwise identical**。这确保任何对比实验不受实现差异的干扰。

### 3.8 参数量分析

| 子模块 | 计算 | 参数量 (hidden=2048, n_belief=256) |
|--------|------|-----------------------------------|
| Evidence Extractor | Linear(4096→256) + Linear(256→256) | ~1.1M |
| Belief GRU | GRUCell(256, 256) | ~197K |
| Confidence Proj | Linear(256→1) | ~257 |
| Calib MLP | Linear(6273→1024) + Linear(1024→2048) | ~8.5M |
| **EABridge 总计** | | **~9.8M** |
| Bridge v2 (对比) | | ~8.4M |

总增量约 **1.4M 参数（+17%）**，在 138M 可训参数的总体预算中可忽略。

---

## 4. 置信度触发的自适应停止

### 4.1 DDM 决策边界的形式化

```
停止条件（三个条件取 AND）:
  1. c_t > θ_high        → 置信度突破高阈值，可以停止
  2. t ≥ T_min           → 达到最小步数，防止过早停止
  3. contradiction_t < γ → 不存在显著证据矛盾

强制继续条件（任一即继续）:
  1. t < T_min           → 未达最小步数
  2. c_t < θ_low         → 置信度严重不足
  3. contradiction_t ≥ γ → 存在证据矛盾需要深化

安全上限:
  t ≥ T_max              → 强制停止，防止无限循环
```

参数默认值：

| 参数 | 默认值 | 认知科学对应 | 说明 |
|------|--------|-------------|------|
| θ_high | 0.85 | DDM 决策边界 | 超过此阈值模型认为"足够确定" |
| θ_low | 0.30 | 最低证据门限 | 低于此阈值说明模型处于茫然状态 |
| T_min | 4 | 最小反应时间 | 即使最简单案例也需要一定处理 |
| T_max | 32 | 最大反应时间 | 安全上限，比固定 16 步翻倍 |
| γ | 0.50 | 矛盾容忍度 | 矛盾信号超过此值强制深化 |

### 4.2 实现：修改 RDT 循环

修改 `RDTFixed16.forward()` 中的核心循环（参考 `src/rdt_fixed.py` 的循环结构）：

```python
def forward_with_adaptive_halting(self, h_anchor, ...):
    b = torch.zeros(B, self.n_belief, device=h_anchor.device)
    loop_outputs = []
    belief_states = [b]
    stop_steps = []

    for t in range(1, self.T_max + 1):
        # 1. Core 推理
        z_t = self._core_step(h_t if t > 1 else h_anchor, ...)

        # 2. EABridge 校准 + 信念更新
        h_t, b, c_t, contradict_t = self.eabridge(
            h_anchor, z_t, t, b
        )

        loop_outputs.append(h_t)
        belief_states.append(b)

        # 3. 停止判断
        if t >= self.T_min:
            if c_t > self.theta_high and contradict_t < self.gamma:
                stop_steps.append(t)
                break

        if t == self.T_max:
            stop_steps.append(t)

    # 4. 聚合（信念加权的 last4）
    h_fused = self._belief_weighted_aggregation(loop_outputs, belief_states)
    return h_fused, loop_outputs, belief_states, stop_steps
```

### 4.3 与 HaltHead 方案的对比

原型文档 §5.1 中明确拒绝了 Adaptive 作为主线，原因是 HaltHead + ACT 引入的不稳定性（早停塌缩、多目标梯度冲突）。Direction A 的停止机制与之有本质区别：

| 维度 | HaltHead + ACT | DDM 自适应停止 |
|------|---------------|---------------|
| 停止信号来源 | 独立学习的 halt probability | 置信度 c_t（与诊断任务耦合） |
| 与主任务的关系 | 竞争（额外 head 与 LM head 无关联） | 协同（c_t 来自信念状态，信念状态由诊断任务驱动） |
| 可解释性 | 不透明（"为什么这一步停了？"无从回答） | 透明（"因为置信度达到 0.87 > 0.85"） |
| 训练稳定性 | 需要平衡 CE + ponder + entropy 多个目标 | 不需要额外训练目标（c_t 自然收敛） |
| 梯度依赖 | halt probability 需要 REINFORCE 或 straight-through | 普通反向传播（c_t 是通过 b_t 可微的） |

### 4.4 配置切换

```yaml
# configs/rdt_config.yaml 扩展
direction_a:
  loop_mode: "fixed"          # "fixed" | "adaptive"
  adaptive_stop:
    theta_high: 0.85
    theta_low: 0.30
    T_min: 4
    T_max: 32
    theta_anneal: false       # 训练时从 0.99 退火到目标值
    theta_anneal_steps: 1000
```

在 `fixed` 模式下，EABridge 正常工作但忽略停止条件，运行完整 `num_iters` 步——这允许在固定步数设定下独立验证 EABridge 的有效性。

---

## 5. 矛盾检测与强制深化

### 5.1 医学诊断中的矛盾处理

在临床推理中，矛盾信号是关键的认知触发因素。例如：
- 年轻患者 + 胸痛 → 不应直接排除心梗（年龄与症状矛盾）
- 已诊断病毒性感冒 + 出现意识障碍 → 应怀疑初始诊断（新证据与当前诊断矛盾）
- 多个症状指向不同系统 → 需要更深的跨系统鉴别

现有 RDT-Fixed-16 无法建模这种"矛盾 → 需要更多思考"的机制——无论第 8 步的证据与第 3 步的信念多么矛盾，循环继续 mechanically 执行到第 16 步。

### 5.2 矛盾检测的形式化

```
# 预测证据（基于当前信念应期望看到什么）
e_pred_t = Linear(b_{t-1})  ∈ R^{n_belief}

# 实际证据（当前 Core 输出实际提供了什么）
e_actual_t = EvidenceExtract(z_t, h_anchor)

# 矛盾度 = 预测与实际的差异
contradiction_t = sigmoid(Linear(concat(e_actual_t, e_pred_t, e_actual_t - e_pred_t)))
                ∈ [0, 1]
```

直觉：如果当前证据与"模型基于累积信念所期望看到的证据"差异很大 → 高矛盾信号。

### 5.3 矛盾触发的行为

```python
if contradiction_t > self.gamma_contradict:
    # 1. 保证至少再走 T_extra 步
    required_steps = max(required_steps, t + self.T_extra)

    # 2. 暂时降低停止阈值（更难停止）
    theta_high_effective = theta_high * 0.8

    # 3. 增强信念更新门控（更开放地接收新证据）
    gate_boost = 1.5
```

这三个行为共同实现了"等等，这不对——我需要重新思考"的认知过程。

### 5.4 与其他组件的关系

- 矛盾检测**不影响 EABridge 的校准行为**——校准继续由置信度门控
- 矛盾检测**只影响停止条件和信念更新门控**——模块职责清晰
- **解耦设计**：关闭矛盾检测（γ = 0），系统退化为纯 DDM 停止

---

## 6. 信念轨迹正则化

### 6.1 设计原则

好医生的诊断推理遵循可预测的认知规范：
1. **先发散后收敛**：早期保持多个假设，后期收敛到最大可能
2. **证据单调累积**：对正确诊断的置信度应大致单调递增
3. **最终高置信**：最终步骤应足够确定

这些规范可以直接编码为训练正则项。

### 6.2 三个正则化损失

```python
def belief_trajectory_loss(belief_states, target_diagnosis_idx, T):
    """
    belief_states: List[Tensor[B, n_belief]], 长度为 T（实际步数）
    target_diagnosis_idx: 正确诊断的索引（用于监控，非直接监督信念方向）
    """
    # 1. 信念平滑性 L_smooth_belief
    # 惩罚相邻步之间信念向量的突变
    L_smooth = sum(||b_t - b_{t-1}||^2 for t in 1..T) / T

    # 2. 早期多样性 L_diversity
    # 前 1/3 步骤中，信念不应过于集中（保持多假设并存）
    early_T = max(T // 3, 1)
    # 测量早期信念的集中度（与均值的 L2 距离）
    early_mean = mean(belief_states[:early_T])
    L_diversity = -mean(||b_t - early_mean||^2 for t in 1..early_T)
    # 注意：负号表示鼓励偏离均值（即鼓励多样性）

    # 3. 最终收敛性 L_convergence
    # 最后一步信念应有足够的范数（表示累积了充分证据）
    b_final_norm = ||belief_states[-1]||_2
    L_convergence = relu(1.0 - b_final_norm)
    # 惩罚范数小于 1.0（说明证据累积不充分）

    return lambda_smooth * L_smooth + lambda_diversity * L_diversity + lambda_convergence * L_convergence
```

### 6.3 与现有 Loss 的整合

现有 Phase 2 损失（参考 `src/train_rdt_fixed.py`）：

```
# 现有:
L = CE + λ_kl · KL_ref + λ_smooth · L_smooth_hidden

# 整合后:
L = CE + λ_kl · KL_ref + λ_smooth · L_smooth_hidden
    + λ_trajectory · L_trajectory
```

推荐 `λ_trajectory = 0.01 ~ 0.05`，足够小以避免支配 CE loss，足够大以提供有意义的正则化信号。

### 6.4 注意事项

- **不直接监督信念方向**：信念状态是潜在表征，不应被强制指向某个"正确诊断"方向。正则化只约束轨迹的形状（平滑、先发散后收敛、最终收敛），不约束轨迹的方向。
- **多样性仅在固定步数模式下有效**：自适应模式下早期步数不确定，多样性约束的意义减弱。
- **与 L_smooth_hidden 互补**：L_smooth_hidden 约束 hidden state 的跨步一致性；L_trajectory 约束信念状态的认知规范性。

---

## 7. 训练路线扩展

### 7.1 现有管线回顾

```
Phase 0: 医疗 SFT 基座 (已完成)
Phase 1: Bridge Warmup — Progressive K, KL loss only (已实现)
Phase 2: RDT-Fixed-16 SFT — CE + KL + Smooth (已实现)
Phase 3: GRPO RL — Medical composite reward (待实施)
```

（详见 [technical-prototype-rdt-medical-llm.md](./technical-prototype-rdt-medical-llm.md) §6）

### 7.2 更新后的管线

```
Phase D: EABridge Warmup (新增, 对应原 Phase 1)
  ├── Progressive K: 2 → 4 → 8 → 16
  ├── 冻结 Core/Suffix/Prefix, 仅训练 EABridge
  ├── 损失: L = KL_ref + λ_smooth · L_smooth_hidden + λ_belief · L_smooth_belief
  ├── belief_dim 渐进: 先 belief_dim=0 稳定, 再逐步开启
  └── 目标: 让信念追踪机制在不对 CE 产生影响的情况下先稳定

Phase E: RDT-EA SFT (替代原 Phase 2)
  ├── EABridge + Core LoRA + Suffix LoRA 全开
  ├── Stage E1: Fixed K=16 → 固定步数下验证 EABridge > Bridge v2
  ├── Stage E2: Adaptive mode → 开启自适应停止, θ_high anneal (0.99 → 0.85)
  └── 损失: L = CE + λ_kl · KL + λ_smooth · L_smooth_hidden + λ_trajectory · L_trajectory

Phase F: RDT-EA + GRPO (替代原 Phase 3)
  ├── 与 Phase 3 相同的 medical composite reward
  ├── 额外 reward: 步数效率 bonus (+0.05 per saved step vs T_max)
  └── 目标: 在自适应模式下优化计算效率与诊断质量的帕累托前沿
```

### 7.3 关键训练技巧

**θ_high 退火**：训练初期将 θ_high 设为 0.99（几乎永不停止），让模型先学会充分的证据累积。然后逐步退火到目标值 0.85。这避免了"模型还没学会累积证据，就被迫提前停止"的冷启动问题。

**信念预热的分阶段开启**：

```
Warmup sub-stage 1: belief_dim = 0   (等价于 Bridge v2, 纯 KL loss)
Warmup sub-stage 2: belief_dim = 64  (开启轻量信念, 仍 KL loss only)
Warmup sub-stage 3: belief_dim = 256 (目标维度, 仍 KL loss only)
```

这确保了信念追踪的每个新增能力都在受控条件下引入。

---

## 8. 差异化分析与学术定位

### 8.1 差异化更新

对比 [RELATED_WORK_REPORT.md](./RELATED_WORK_REPORT.md) 的原始结论，Direction A 在三个维度上显著提升了差异化：

| RDT-EA 设计 | 已有工作 | 差异性 | 新颖度 |
|------------|---------|--------|--------|
| **EABridge 信念追踪** | Huginn 无信念状态；Parcae 使用约束参数化 | 首个在循环 Transformer 中显式建模诊断证据累积的工作 | 🟢 高 |
| **DDM 置信度停止** | MoR 用 Router（token-level, 可学习但不可解释） | 置信度阈值可解释，与诊断任务语义耦合 | 🟢 中高 |
| **矛盾检测与强制深化** | **所有现有工作无此机制** | 首个建模"新证据与当前信念矛盾 → 强制更多推理"的架构 | 🟢 高 |
| **信念轨迹正则化** | 现有工作仅 CE+KL+Smooth | 首个将认知规范编码为训练损失的循环 Transformer 工作 | 🟢 中高 |
| **认知科学框架** | Huginn/Parcae/MoR 均为纯工程驱动 | 首个以认知科学理论（DDM/SPRT/Dual-Process）为架构设计锚点的循环 Transformer | 🟢 高 |

### 8.2 学术定位

原 RELATED_WORK_REPORT 的论文定位建议：

> "Looping Transformers for Medical Diagnosis: A Bridge-Calibrated Recurrent Depth Approach"

Direction A 的新定位：

> **"From Loops to Evidence: Cognitive-Science-Grounded Evidence Accumulation in Recurrent Depth Transformers for Medical Diagnosis"**

核心主张（4条）：

1. **理论贡献**：首次建立循环 Transformer 的迭代过程与认知科学证据累积理论（DDM/SPRT）之间的形式化对应
2. **机制贡献**：EABridge — 带信念追踪和置信度门控的状态校准器；矛盾检测 — 惊喜信号触发的自适应深化
3. **实证贡献**：信念轨迹可视化验证了 DDM 理论预测（简单 case 快收敛、复杂 case 慢收敛、矛盾 case 需要更多步骤）
4. **场景贡献**：在医疗诊断这一天然需要序列证据累积的领域，展示了架构-领域耦合的价值

### 8.3 从"场景差异"到"机制差异"

| 维度 | RDT-Fixed-16 v1 | RDT-EA v2 (Direction A) |
|------|----------------|------------------------|
| 差异化类型 | 场景差异化（医疗） | **机制差异化**（信念追踪 + 认知科学框架） |
| 防御性 | 弱（Huginn 可轻易进入医疗） | 强（认知科学框架 + EABridge 是架构创新，非场景选择） |
| Reviewer 问题 | "这和 Huginn 有什么区别？"(需要大量解释) | "DDM 的决策边界在你们架构中如何体现？"(引导到技术讨论) |
| 复用性 | 仅医疗 | 任何需要序列证据累积的领域（法律、金融、科学） |

---

## 9. 评估体系扩展

### 9.1 现有指标回顾

参考 [technical-prototype-rdt-medical-llm.md](./technical-prototype-rdt-medical-llm.md) §10 和 `src/eval_utils.py` 的实现。

### 9.2 Direction A 新增指标

| 指标 | 计算方式 | 验证的创新点 | 预期结果 |
|------|---------|-------------|---------|
| **Confidence Calibration** | ECE = E[‖c_t - accuracy‖] | I2 (EABridge 置信度有意义) | c_t 与诊断正确率正相关 |
| **Belief Convergence Speed** | argmin_t ‖b_t - b_T‖_2 < ε | I1 (信念确实在收敛) | easy case ~4步, hard case ~12步 |
| **Stop Step Distribution** | easy/medium/hard 各难度的平均停止步数直方图 | I5 (DDM 自适应停止) | easy < medium < hard |
| **Contradiction Frequency** | 各难度触发矛盾信号的比例 | I3 (矛盾检测) | hard case 触发率 > easy |
| **Belief Reversal Rate** | count(‖b_t - b_{t-1}‖ > δ) / T | I4 (信念轨迹正则化) | 训练后反转率显著下降 |
| **Step Efficiency** | (Acc@T - Acc@T-1) / step | I2+I5 | 更多步数确实带来收益，且收益不递减 |
| **Oracle Stop Gap** | Acc(DDM-stop) / Acc(oracle-stop) | I5 | DDM 停止与理想停止的性能差距 |

### 9.3 指标实现

新增评估函数应扩展 `src/eval_utils.py`（或创建新的 `src/eval_direction_a.py`）：

- `compute_confidence_calibration(belief_states, labels)` → ECE score
- `compute_belief_convergence(belief_states, epsilon=0.05)` → convergence step
- `compute_stop_step_distribution(all_stop_steps, difficulties)` → histogram data
- `compute_belief_trajectory_metrics(belief_states)` → smoothness, reversal rate, final norm

---

## 10. 消融实验设计

### 10.1 信念维度消融

```
EABridge n_belief=0    (等价于 Bridge v2, baseline)
EABridge n_belief=64   (轻量信念)
EABridge n_belief=128
EABridge n_belief=256  (默认)
EABridge n_belief=512  (大容量)
```

**假设**：n_belief=64 已能捕获大部分收益（sweet spot），256 提供少量额外增益，512 收益递减甚至过拟合。

### 10.2 停止策略消融

```
fixed K=16                          (baseline)
adaptive + norm confidence          (默认, 可解释)
adaptive + learned confidence       (灵活但不可解释)
adaptive + oracle confidence        (上界: 根据真实标签的理想停止)
adaptive + no T_min                 (测试 T_min 的必要性)
```

### 10.3 矛盾检测消融

```
no contradiction detection (纯 DDM 停止, baseline)
γ = 0.7 (严格, 仅强矛盾触发)
γ = 0.5 (默认)
γ = 0.3 (宽松, 频繁触发)
```

### 10.4 损失项消融

```
CE only (bare minimum)
CE + KL + Smooth                       (当前 Phase 2 baseline)
CE + KL + Smooth + L_smooth_belief     (+信念平滑)
CE + KL + Smooth + L_diversity         (+早期多样性)
CE + KL + Smooth + L_convergence       (+最终收敛)
CE + KL + Smooth + L_trajectory (full) (+全部三个正则项)
```

### 10.5 RDT-EA vs 所有基线（终极消融）

```
SFT baseline (2.2B params, LoRA on all layers)     — 参数规模上限
RDT-Fixed-16 (138M params, Bridge v2, K=16)        — 当前最优原型
RDT-EA Fixed (n_belief=256, fixed K=16)             — EABridge 净收益
RDT-EA Adaptive (自适应停止 + 矛盾检测)               — 完整方案
RDT-EA Adaptive-Only (自适应停止, 无矛盾检测)         — 隔离矛盾检测收益
RDT-EA Contradict-Only (固定K=16, 有矛盾检测无停止)    — 隔离自适应停止收益
```

---

## 11. 风险与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **信念状态无增益** | 中 | 增加复杂度但 Hard CE 不优于 Bridge v2 | n_belief=0 降级验证；如果 EABridge(n_belief=256) ≤ Bridge v2，放弃信念追踪，仅保留置信度门控校准 |
| **置信度校准不良** | 中 | c_t 与诊断正确率不相关，导致停止决策随机 | ECE 监控；使用 norm-based 置信度（比 learned 更稳健）；θ_high 退火 |
| **矛盾检测过敏感** | 中高 | 几乎所有样本都触发，失去区分度 | γ 网格搜索 [0.3, 0.5, 0.7, 0.9]；可视化 contradiction_t 的分布 |
| **信念轨迹正则化过度约束** | 低中 | 惩罚了合理的信念修正（如新证据确实推翻了初始假设） | 设置容差 ε；仅在验证集监控；权重设为极低优先级（0.01） |
| **自适应模式下 Easy 退化** | 中 | 过早停止导致 Easy 样本质量下降 | T_min=4 硬下限；easy 样本停止步数分布监控 |
| **延迟不可预测** | 中 | 自适应停止导致推理延迟不稳定，部署困难 | T_max=32 硬上限；生产环境可回退到 fixed 模式 |
| **与前人工作的区分被质疑** | 低中 | Reviewer 认为认知科学框架只是"换了个说法" | 信念轨迹可视化实验是关键——必须展示具体可验证的认知科学预测 |

---

## 12. 实现路线图

### Phase D1: EABridge 核心实现（2-3 周）

| 任务 | 内容 | 验证方式 |
|------|------|---------|
| D1.1 | 创建 `src/eabridge.py`，实现 `EABridge` 类 | 单元测试 |
| D1.2 | 在 `src/bridge.py` 的 `BridgeRegistry` 注册 `"ea"` 变体 | 工厂方法测试 |
| D1.3 | 修改 `src/rdt_fixed.py` forward 循环以传递信念状态 | smoke test: forward pass 不报错 |
| D1.4 | 降级测试：n_belief=0 时输出与 Bridge v2 bitwise identical | 数值对比测试 |
| D1.5 | 信念初始化验证：b₀=0, 第一次更新后 b₁≠0 | 单元测试 |

### Phase D2: 自适应停止与矛盾检测（2 周）

| 任务 | 内容 | 验证方式 |
|------|------|---------|
| D2.1 | 实现置信度评估（norm + learned 两种模式） | 置信度值范围检查 [0, 1] |
| D2.2 | 修改 forward 循环支持 adaptive halting | 不同难度样本停止步数不同 |
| D2.3 | 实现矛盾检测模块 | contradiction_t 范围检查 [0, 1] |
| D2.4 | 配置切换：fixed/adaptive loop_mode | 两种模式结果一致性测试 |
| D2.5 | 延迟基准：自适应 vs 固定 16 步的平均步数 | 预期 easy~6, medium~12, hard~20 |

### Phase D3: 训练管线集成（2 周）

| 任务 | 内容 | 验证方式 |
|------|------|---------|
| D3.1 | 实现 `belief_trajectory_loss` 函数 | 损失值范围合理 |
| D3.2 | 更新 `src/train_rdt_fixed.py` 支持 Direction A loss | 训练不崩溃 |
| D3.3 | 更新 `configs/rdt_config.yaml` 添加 direction_a section | 配置文件可解析 |
| D3.4 | 实现 Bridge Warmup 兼容模式（belief_dim 渐进） | 渐进开启不导致 loss 跳变 |
| D3.5 | θ_high 退火调度器实现 | 退火曲线可视化 |

### Phase D4: 评估与消融（2 周）

| 任务 | 内容 | 验证方式 |
|------|------|---------|
| D4.1 | 实现 Direction A 评估函数（见 §9.3） | 各指标计算不报错 |
| D4.2 | 执行信念维度消融（§10.1） | 确定最优 n_belief |
| D4.3 | 执行停止策略消融（§10.2） | 确定最优 θ_high 和停止模式 |
| D4.4 | 执行损失项消融（§10.4） | 确定必要正则项 |
| D4.5 | 执行完整对比消融（§10.5） | RDT-EA vs RDT-Fixed-16 净收益量化 |

### Phase D5: 文档与论文（1 周）

| 任务 | 内容 |
|------|------|
| D5.1 | 更新本设计文档为实验报告（含数据） |
| D5.2 | 撰写 Direction A 实验报告 |
| D5.3 | 更新差异化分析确认 |
| D5.4 | 论文大纲准备（如果 results 正面） |

---

## 13. 结论

### 核心洞见

RDT-Fixed-16 的 16 步循环已经在隐式地做证据累积——Bridge v2 的差分项 `(z_t - h_anchor)` 是一个粗粒度的"证据信号"，last4_mean 聚合暗示了"后期步骤更接近收敛"。Direction A 的工作是将这一隐式过程**显式化、形式化、可验证化**。

### 为什么这个方向值得投入

1. **理论差异化**：认知科学框架（DDM/SPRT/Dual-Process）为 RDT-Dx 提供了已有循环 Transformer 工作不具备的理论深度。这使 RDT-Dx 从"又一个循环 Transformer 的医疗应用"升华为"受认知科学启发的诊断推理架构"。

2. **效率收益**：自适应停止使简单 case 用更少步数（~4-6 vs 16），理论上能将平均推理延迟降低 30-50%。

3. **可解释性**：信念状态 b_t 和置信度 c_t 提供了观察模型"思考过程"的窗口——可以可视化哪些步骤信念发生转折、哪些症状触发了矛盾信号。

4. **可证伪性**：DDM 理论给出了明确的可检验预测（置信度应单调递增、矛盾应触发深化、复杂 case 需要更多步骤），这些预测可以被实验证实或证伪——这正是强科学的标志。

### 最小成功判据

> **在相同计算预算（固定 K=16）下，EABridge（n_belief=256）在 hard 医疗样本上的 CE loss 显著低于 Bridge v2。**

如果这一条不成立，Direction A 的核心机制假设不成立，应考虑放弃信念追踪路径，仅保留置信度门控校准（更轻量，与 Bridge v2 合并代价小）。

如果这一条成立，继续验证自适应停止和矛盾检测的增量收益。

### 最坏情况

即使信念轨迹不完全符合 DDM 的理论预测，"部分对齐"仍然是有价值的发现——认知科学理论为架构设计提供了**方向指引**而非**精确约束**。在最坏情况下（比如 confidence calibration 失败、信念状态不可解释），EABridge 仍可降级为改进的 Bridge v2（置信度门控校准本身不需要信念状态可解释就能 work）。

---

*本文档基于 RDT-Dx Phase A/B/C 验证结果和 2026-06-02 的 Direction A 方案讨论撰写。待实现后更新为实验报告。*

## 参考文献

认知科学基础：
- Ratcliff, R. (1978). A theory of memory retrieval. *Psychological Review*, 85(2), 59–108.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117–186.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Evans, J. S. B. T., & Stanovich, K. E. (2013). Dual-process theories of higher cognition: Advancing the debate. *Perspectives on Psychological Science*, 8(3), 223–241.

循环深度 Transformer：
- Dehghani et al. (2019). Universal Transformers. *ICLR 2019*.
- Geiping et al. (2025). Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer. *ICLR 2025*.
- Bae et al. (2024). Relaxed Recursive Transformers. *ICLR 2025*.
- Bae et al. (2025). Mixture-of-Recursions. *NeurIPS 2025*.
- Parcae (2025). Scaling Laws For Stable Looped Language Models.
