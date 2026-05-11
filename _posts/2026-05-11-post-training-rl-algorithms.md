---
layout: post
title: "后训练之 RL 算法篇：从 REINFORCE 到 GRPO"
date: 2026-05-11 11:30:00 +0800
description: "梳理后训练场景下策略梯度算法的演进逻辑，从 REINFORCE 到 GRPO。"
tags: [work, ai, post-training]
giscus_comments: true
stars: 0
---

# 一、前言：从老虎机到万亿参数模型的信用分配

强化学习（Reinforcement Learning, RL）的发展历程，本质上是一部处理**高维状态-动作空间**中信用分配问题（Credit Assignment Problem）的建模演变史。

这一路径的起点可以追溯到最简单的 **多臂老虎机（Multi-Armed Bandits, MAB）** 模型。在 MAB 中，Agent 面临的是一个**无状态（Stateless）**的环境：每一次决策都是孤立的，不存在“当前动作会改变未来状态”的复杂因果。此时，RL 的核心矛盾仅仅在于如何平衡**探索与利用（Exploration vs. Exploitation）**。

然而，当场景演进到**马尔可夫决策过程（MDP）**时，环境变得“有记忆”了。以 **DQN（Deep Q-Network）** 为代表的 **Value-based** 算法曾是这一阶段的霸主，它通过深度网络拟合最优 $Q$ 函数来指导决策。但在 **自然语言处理（NLP）** 领域，传统的 Value-based 范式遭遇了毁灭性的**维度灾难（Curse of Dimensionality）**。在 LLM 动辄数万维的离散词表（Vocabulary）面前，让模型去精确估算每一个 Token 在每一个上下文下的绝对价值，不仅计算量爆炸，且由于语言极度灵活，其收敛速度根本无法满足工程需求。

在这一背景下，**策略梯度（Policy Gradient）** 方法展现了其范式上的优越性。1992 年提出的 **REINFORCE** 算法奠定了参数化策略优化的数学基础：它彻底绕过了价值函数的中间估计，直接在参数空间内对概率分布进行梯度上升。

这种**从“价值估计”向“概率密度建模”的转向**，不仅解决了高维动作空间下的计算可行性问题，更通过 **Log-Derivative Trick** 将不可导的环境奖励转化为可导的参数更新方向。此后，从为了稳定步长而诞生的 **TRPO** 和 **PPO**，到如今 DeepSeek 为了极致显存效率而提出的 **GRPO**，算法的演进始终围绕着三个核心痛点：

1. **方差与偏差的博弈**（如何让更新更稳）；
2. **采样效率的榨取**（如何让数据复用）；
3. **计算成本的平衡**（如何在万亿参数下跑得动）。

# 二、策略梯度定理：NLP 视角下的解析推导

在 NLP 语境下，RL 的目标是优化语言模型（Actor）的参数 $\theta$（即 Transformer 的权重），使生成的序列能获得最高奖励。

## 2.1 变量映射与 $\theta$ 的物理含义

- **参数 $\theta$**：神经网络中所有可学习的权重。它决定了模型在给定上下文时输出 Token 的概率分布。
- **状态 (State) $s_t$**：当前的上下文，即 `Prompt + 已生成的 Token 序列`。
- **动作 (Action) $a_t$**：模型在当前分布下预测出的下一个 `Token`。
- **策略 (Policy) $\pi_\theta(a_t | s_t)$**：在参数 $\theta$ 下，给定上文 $s_t$ 时，输出 $a_t$ 的条件概率。
- **轨迹 (Trajectory) $\tau$**：一个完整的生成序列 $(s_0, a_1, r_1, \dots, a_T, r_T)$。

## 2.2 目标函数的 Log-Derivative Trick 定义

目标函数 $J(\theta)$ 为生成序列的期望奖励：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int \pi_\theta(\tau) R(\tau) d\tau
$$

对参数 $\theta$ 求梯度，并引入 **对数导数技巧**（因为 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$）：

$$
\nabla_\theta J(\theta) = \int \nabla_\theta \pi_\theta(\tau) R(\tau) d\tau
= \int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) R(\tau) d\tau
$$

从而得到期望形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(\tau) R(\tau)]
$$

## 2.3 因果性修正与期望抵消证明

原始公式中，$t$ 时刻的梯度被乘以了总奖励 $R(\tau)$。但在物理逻辑上，$t$ 时刻的动作无法影响过去的奖励 $\sum_{k=1}^{t-1} r_k$。

**数学证明：**  
对于任何与当前动作 $a_t$ 无关的基准 $b(s_t)$，其梯度项的期望恒为 0：

$$
\mathbb{E}_{a_t \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t)]
= \int \pi_\theta(a_t | s_t) \frac{\nabla_\theta \pi_\theta(a_t | s_t)}{\pi_\theta(a_t | s_t)} b(s_t) da_t
$$

$$
= b(s_t) \nabla_\theta \int \pi_\theta(a_t | s_t) da_t
= b(s_t) \nabla_\theta (1) = 0
$$

因此，我们可以剔除过去奖励，将权重替换为 **Reward-to-go**（当前及未来的收益累加 $G_t$），在保持无偏性的同时大幅降低方差。

# 三、REINFORCE (1992)：策略梯度的原点

**历史背景**：由 Ronald Williams 提出。它证明了“黑盒”优化的可能性：只要能观测到最终结果分数，就能通过反向传播调整全局权重 $\theta$。

## 3.1 公式表达

REINFORCE 引入 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 作为权重：

$$
\theta \leftarrow \theta + \alpha \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) G_t
$$

## 3.2 优缺点

- **优点**：实现简单，支持黑盒奖励（无需梯度反馈），是所有策略梯度的奠基石。
- **缺点**：**方差极大**。由于 $G_t$ 依赖单次随机采样，一个偶然的极端样本（Outlier）就会导致模型参数剧烈抖动，训练极不稳定。

# 四、Actor-Critic (2000s)：引入“评论家”进行信用分配

**历史背景**：为了缓解 REINFORCE 的方差问题，研究界借鉴了动态规划的思想，引入了 **Critic** 网络。与其让模型盲目追逐一个随机的 $G_t$，不如让它追逐“超出预期的收益”。

## 4.1 优势函数 (Advantage)

Critic 网络预测状态价值 $V_\phi(s_t)$。我们定义**优势函数**为权重：

$$
\hat{A}_t = G_t - V_\phi(s_t)
$$

## 4.2 优缺点

- **优点**：**降低方差**。通过减去均值基准 $V(s)$，过滤掉了环境带来的随机噪声，使更新方向更纯粹。
- **缺点**：**引入偏差**。如果 Critic 估值不准，会误导 Actor 的更新方向。且需要同时训练两个网络，增加了复杂度。

# 五、TRPO (2015)：置信域更新的数学保障

**历史背景**：由 John Schulman 提出。当时 AC 算法面临“步长难定”的问题：步长太小练不动，步长太大策略崩溃。TRPO 的核心意图是利用二阶优化，强制新旧策略的差异限制在 KL 散度的“置信域”内。

## 5.1 数学约束

$$
\max_\theta \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \hat{A}_t \right]
\quad \text{s.t. } \mathbb{E}[KL(\pi_{old} || \pi_\theta)] \le \delta
$$

## 5.2 优缺点

- **优点**：**极高的稳定性**。几乎能保证每次更新都能单调改进策略，不会出现“学废了”的情况。
- **缺点**：**计算成本爆炸**。需要计算海森矩阵的逆，在参数量巨大的 LLM 时代基本不可行。

# 六、PPO (2017)：工业界的黄金准则

**历史背景**：OpenAI 提出。PPO 的本质是：**用简单的“一阶裁剪”代替 TRPO 复杂的“二阶数学约束”**，在保证稳健性的同时，极大地提升了工程可行性。

## 6.1 重要性采样 (Importance Sampling)

为了提高采样效率，PPO 允许利用旧策略 $\pi_{old}$ 采样的数据更新当前策略 $\pi_\theta$。利用重要性采样比率 $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$，模型可以将数据重复利用多次（Off-policy），大大节省了生成成本。

## 6.2 裁剪机制 (Clipped Objective)

$$
L^{CLIP}(\theta) = \mathbb{E} \left[
\min(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)
\right]
$$

## 6.3 优缺点

- **优点**：**性能与稳定性的完美平衡**。它比 TRPO 容易实现，比 AC 稳健。
- **缺点**：**显存开销高**。在 LLM 中，需要同时加载 Actor、Critic、Ref 三个模型，显存压力极大。

# 七、GRPO (2024)：DeepSeek 的极致优化

**历史背景**：DeepSeek 团队反思了“必须养一个 Critic 网络”的假设，提出通过**组内相对竞争**来消除对价值模型的依赖。

## 7.1 组内标准化

针对同一 Prompt 采样 $G$ 个样本，计算优势：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
$$

## 7.2 优缺点

- **优点**：**显著降低显存**。省去了 Critic 网络的梯度存储。且在逻辑复杂的推理任务（Reasoning）中，组内对比比神经网络估值更精准。
- **缺点**：**依赖大 Batch 采样**。需要每一轮对同一个 Prompt 采样足够多的样本（如 64 个），对推理吞吐量有一定要求。
