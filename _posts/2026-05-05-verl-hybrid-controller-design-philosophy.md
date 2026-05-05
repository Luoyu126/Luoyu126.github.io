---
layout: post
title: "从PnP到系统底层：Verl灵活适配RL的设计哲学"
date: 2026-05-05 17:10:00 +0800
description: "从源码与实验指标出发，梳理 Verl 如何用 Hybrid-Controller 与 one-step-off 流水线连接 rollout 和 update。"
tags: [work, system, ai]
giscus_comments: true
stars: 0
---

# 一、前言

这篇blog的出发点很直接：我在做后训练任务时长期把 `verl` 当成“能跑就行”的训练脚手架，用它拼配置、拉起任务、看 loss 和 reward 曲线，但对它为什么快、快在哪里、为什么它被称为 RL 框架，其实并没有形成系统理解。直到最近在实验中碰到吞吐与时延问题（尤其是 rollout 与 update 的时间关系），我于是产生了去读一读Verl框架的兴趣。在读的过程中顺便补充了一些基础知识，如后训练过程中训推的关键差异、多gpu计算的范式等。有意思的是在整理这些知识的时候发现他们基本都出自我之前学过的讲义，如d2l（动手学深度学习）。如果你也和我一样，是一个在任务中学习更有体验感的人，欢迎你跟随这篇blog的逻辑，以理解Verl框架为出发点去梳理沿途的知识点，最后对Verl框架也有一个PnP之上的理解。

在开始读源码之前，我先补了一轮背景调研，主要想回答一个问题：**为什么这些年 RL 后训练框架很多，但 `verl` 会这么快成为主流选项之一？**  

要理解 `verl` ，必须先拆解分布式系统中的“控制权”逻辑。在 LLM 这种超大规模分布式 Workload 中，如何管理成百上千个 GPU 进程，决定了框架的上限。

- **Single-Controller (MPMD)**：这是一种典型的“主从模式”。一个中心化的 Python 进程（如主控制脚本）通过 RPC 直接对接集群里的**每一张显卡**。虽然这让开发者能像写单机脚本一样方便，但当卡数增加，主进程的分发压力和 RPC 通信开销会成为系统瓶颈，导致严重的“指挥官迟钝”。
- **Multi-Controller (SPMD)**：为了追求极致效率，许多框架走向了另一个极端——让每张卡都运行完全相同的程序副本。在这种模式下，vLLM 推理或 FSDP 训练后端能发挥出硬件的巅峰性能，但代价是代码变得极其零散且难以编写，就像让成千上万名士兵在没有对讲机的情况下全靠默契协同，开发门槛极高。
- **verl 的 Hybrid-Controller（混合控制）**：`verl` 的高明之处在于它选择“全都要”。它在宏观上保留了 **Single-Controller** 的编排逻辑，让用户只需写 `actor.generate()` 这样直观的代码；但在微观执行上，它将任务下发给由 **Ray** 启动的一组组“子节点群”——这些群组内部以 **Multi-Controller** 模式高速运行 vLLM 或 FSDP 等专业引擎。

这种“**主进程管引擎群，引擎群管显卡**”的层级结构，既利用了 vLLM 和 FSDP 这种顶级“工人”的垂直能力，又通过异步 RPC 解决了大规模集群的调度延迟。这也正是 `verl` 能够在同一个 PPO 循环中，丝滑切换推理与训练两种截然不同负载的底层奥秘。

因此，这篇文章不再停留在“怎么用 verl”，而是尝试回答几个更底层的问题：  
1）`verl` 到底把什么模块解耦了，为什么要解耦？  
2）它如何同时接上 vLLM（rollout）与 FSDP（update）这两套计算范式？  
3）one-step-off 异步流水线到底改变了什么，是否真的带来可量化的收益？  

为回答这些问题，我采用了“**源码 + 实验指标**”双线方法：一方面沿 `main_ppo / ray_trainer / engine_workers / checkpoint_engine` 追调用链，另一方面结合一次真实 run 的 W&B timing 指标做反事实估算（异步 vs 同步），尽量把“框架设计意图”与“实际运行效果”对齐起来。

从结果看，`verl` 的价值不只在于接口齐全，而在于它把 RL 后训练中最核心、最昂贵的两段流水线（gen 与 update）做成了可编排、可异构、可重叠的系统。

# 二、基础知识

## 2.1 推理 vs 训练

在 RL 训练的一个循环中，模型需在“生成采样”（rollout）与“参数更新”（update）两个阶段之间快速切换。

### 2.1.1 推理阶段（Rollout）：利用固定参数进行增量生成

*   **计算本质**：此时模型参数（$W_Q, W_K, W_V$ 等权重矩阵）是**固定的**，此阶段要做的是在固定参数的前提下通过自回归采样生成最终的序列并拿到奖励。
*   **Token 绑定**：
    *   模型将每个新输入的 Token 通过 $W_K$ 和 $W_V$ 矩阵映射为对应的向量，并将其存入 **KV Cache** 中。
    *   每一步生成仅需产出一个新的 $Query$，并去访问（搬运）显存中已有的 $KV$ 序列。
*   **显存负载**：由于不涉及反向传播，系统无需保留中间激活值，显存压力主要源于随序列增长而累积的 **KV Cache**。
*   **瓶颈**：由于需要频繁从显存中搬运大量的 $KV$ 数据，系统性能受限于**显存带宽**。

### 2.1.2 训练阶段（Update）：根据反馈更新参数矩阵

*   **计算本质**：此时模型参数是**待优化的变量**，需要根据拿到的奖励/损失进行更新学习。
*   **句子绑定**：
    *   系统将 Rollout 产生的完整序列一次性输入重新进行前向传播。通过矩阵乘法（$X \times W$），全序列的 $Q, K, V$ 被一次性并行计算出来。而后加入注意力掩码机制将矩阵右上角置为负无穷模拟序列化生成过程，并根据loss对前序参数的梯度更新参数。
*   **核心负载——激活值驻留**：
    *   为了根据 Loss 计算 $W$ 矩阵的梯度，系统必须在显存中“锁死”前向传播产生的所有中间激活值。
    *   **更新动作**：根据奖励信号计算 Loss，通过反向传播算出梯度，最终修正 $W$ 矩阵的数值，更新模型参数。
*   **瓶颈**：由于涉及到密集矩阵运算及全量激活值、梯度、优化器状态的存储，系统性能受限于 **GPU 算力峰值**及**显存容量**。

## 2.2 多GPU计算范式

在利用多个GPU进行计算时，目标是利用它们的**计算能力**（算得更快）和**显存空间**（塞下更大的模型）。针对一个小批量（Mini-batch）的训练任务，主要有以下三种拆解方式：

1.网络并行（**P**ipeline **P**arallelism）

- **做法：** 把网络“横着切”。GPU 0 算前几层，GPU 1 算中间几层，以此类推。
- **优点：** 能处理单块 GPU 塞不下的**大模型**（控制每个 GPU 的显存占用）。
- **痛点：**
    - **同步困难**：各层计算量不均会导致有的 GPU 闲死，有的忙死。
    - **通信带宽瓶颈**：层与层之间需要传输大量的中间激活值和梯度，容易卡在总线带宽上。    

2.分层并行（**T**ensor **P**arallelism）

- **做法：** 把每一层内部的计算“拆开”。比如一个有 512 个通道的卷积层，由 2 个 GPU 各负责计算 256 个通道。
- **优点：** 显存随 GPU 数量线性扩展，能支持**更宽**的网络。
- **痛点：** 每一层计算完都需要进行全量同步，通信开销极大，甚至超过了层间拆分。

3.数据并行（**D**ata **P**arallelism）

- **做法：** 模型不动，把数据“切开”。每个 GPU 都拥有一份完整的模型副本，但处理不同的样本。训练完一个 Batch 后，大家交换并聚合梯度。
- **优点：**
    - **最简单、普适性最强**：几乎适用于所有网络架构。
    - **效率高**：同步只发生在 Batch 结束时，且计算和通信可以并行（Overlap）。
- **痛点：** 无法通过增加 GPU 来训练“更大的模型”（因为每个 GPU 都要装下整个模型）。

![多 GPU 计算范式示意图](/assets/img/blog/blog2/multi-gpu-paradigms.png)

## 2.3 vLLM（推理）与 FSDP（训练）

为了在 RL 的不同阶段榨干 GPU 性能，**verl** 引入了两个底层逻辑截然不同的引擎后端vLLM和FSDP，这两个引擎分别针对2.1节中提到的推理和训练进行优化。

### 2.3.1 vLLM：推理时的“带宽魔术师”

vLLM 是专为提高采样吞吐量设计的纯推理引擎，核心目标是解决推理阶段的访存瓶颈。

1.**PagedAttention**：打破显存的“围墙”

传统的推理框架（如 HuggingFace Transformers）在存储 KV Cache 时，需要为每个序列预先分配一块**连续**的显存空间。

- **痛点**：由于生成的文本长度不可预测，系统必须按“最大长度”预留空间。这导致了严重的**内部碎片**（预留了没用上）和**外部碎片**（剩下的零散空间塞不下新序列），显存利用率往往不足 50%。
- **PagedAttention 的解法**：它借鉴了操作系统的虚拟内存逻辑，将 KV Cache 划分为固定大小的“页”（Blocks）。
    - **非连续存储**：物理上的 KV 向量可以散落在显存的任何角落，通过一张“映射表”与逻辑序列对应。
    - **按需动态分配**：只有当新 Token 产生时，才会分配新的物理块。这彻底消除了碎片化，使显存利用率提升至 96% 以上，从而在同等显存下支持极高并发的采样（High Throughput）。

2.非驻留特性：极致的“阅后即焚”

在训练（Update）中，为了计算梯度，模型必须像“录像”一样保留前向传播的所有中间状态。而 vLLM 采样时则采用“直播”模式：

- **即时销毁**：在处理当前层的算子（矩阵乘法、激活函数等）时，产生的临时张量在完成计算、传递给下一层后，会**立即被显存管理器标记为释放**。   
- **空间腾挪**：通过这种非驻留策略，显存中除了模型权重（固定占用）外，几乎所有的剩余空间都被动态分配给了 **KV Cache**。
- **工程意义**：这意味着在 Rollout 阶段，我们可以使用极大的 Batch Size，一次性并行采样成百上千个回复，极大提升了强化学习的数据生产效率。

3.并行策略：**TP**实现低延迟

虽然 PagedAttention 提升了吞吐（一次跑很多个），但为了让每个 Token 出来的速度、更快，vLLM 通常配合 **TP** 执行。

- **原理**：TP 将每一层内部的权重矩阵（如 $W_Q, W_K, W_V$）横向切分到多个 GPU 上。
- **协同作业**：当计算一个 Token 时，所有 GPU 同时启动，各自负责计算结果的一部分，最后通过一次快速的同步（All-Reduce）合并结果。
- **结果**：这种策略避免了网络并行（PP）带来的气泡等待，确保了自回归生成时，单次前向传播的延迟被压制到毫秒级，这对于需要大量采样的 RL 任务至关重要。

### 2.3.2 FSDP：训练时的“空间管家”

FSDP（Fully Sharded Data Parallelism）是 PyTorch 官方对 **ZeRO-3** 协议的高效实现。它的核心目标是打破单卡显存对模型规模的限制，解决训练阶段的“显存容量受限”瓶颈。FSDP本质上是DP的一种进阶，优化点主要体现在借用TP的思想讲训练状态分布存储在N个GPU上，按需合并调度。

1.状态全分片（Sharding）：消除显存冗余

在传统的分布式数据并行（DDP）中，每张 GPU 都要完整存储一份模型副本，这造成了极大的空间浪费。FSDP 通过“分而治之”的思想，将三类数据进行了彻底切分：
- **模型参数（Parameters）**：将原本巨大的权重矩阵切成 $N$ 份（$N$ 为 GPU 数量），每张卡只负责维护其中的 $1/N$。
- **梯度（Gradients）**：反向传播计算出的梯度在同步后不再保留全局副本，仅保留本卡对应分片。
- **优化器状态（Optimizer States）**：这是显存消耗的“隐形杀手”。以 Adam 优化器为例，它需要为每个参数存储一阶和二阶动量。通过分片存储，显存占用直接缩减为原来的 $1/N$。

2.通信换空间：动态的“拼图”计算

FSDP 并不是静态的切分，而是一个高度动态的计算流。它通过增加通信开销，换取了训练超大规模模型的能力：
- **前向传播（Forward）**：当 GPU 0 需要计算第 5 层网络时，它会发起一个 **All-Gather** 操作，瞬间从其他 GPU 那里把第 5 层的参数分片拉过来，拼成一个临时完整的矩阵进行计算。计算一结束，为了给接下来的激活值腾位置，这个临时矩阵会被**立即从显存中抹除**。
- **反向传播（Backward）**：同样的逻辑。系统再次动态拉取参数分片以计算梯度。在梯度产出后，通过 **Reduce-Scatter** 操作将梯度同步并切碎分发到各卡上，随后原始梯度也被销毁。

3.核心意义：为“激活值”腾出战场

在 2.1.2 中我们提到，训练阶段最沉重的负担是**激活值驻留**。FSDP 的精髓就在于此：

- **空间让位**：由于模型参数、梯度、优化器状态都被切碎到了整个集群中，单张 GPU 的基础显存占用（Static Memory）被压到了极低。
- **激活值容纳**：腾出来的这些巨量空间，被用来存放前向传播中产生的所有中间变量。这使得我们可以在不 OOM 的前提下，使用更长的 Sequence Length（如 8k, 32k）或更大的 Batch Size。
- **verl 的必然选择**：在 RL 的 Update 阶段，由于 PPO 等算法需要多次迭代更新，对显存的压榨到了极致。FSDP 这种“通信换空间”的特性，是 verl 能够承载 70B 甚至更大模型进行 RL 训练的底气。

### 2.3.3 verl 的 Hybrid Engine：权重的“变形金刚”

**verl** 的核心创新在于 **Hybrid Engine**，它实现了模型权重的动态转换逻辑：

1.  **在 Rollout 阶段**：模型切换为 **vLLM** 模式，权重被聚合为 **TP 格式**，利用 PagedAttention 进行高速采样。
2.  **在 Update 阶段**：KV Cache 清空，权重立刻重新切碎为**FSDP 格式**。显存全力支持激活值驻留，完成参数更新。

| 维度        | vLLM (Rollout)      | FSDP (Update)                 |
| :-------- | :------------------ | :---------------------------- |
| **主要目标**  | 提高采样吞吐，降低延迟         | 承载大规模参数与激活值训练                 |
| **分布式策略** | 通常为 **TP**          | **FSDP / ZeRO-3**             |
| **显存杀手**  | **KV Cache**        | **Activations**               |
| **中间状态**  | 立即释放 (Forward Only) | 严格保留、按需调度(Forward + Backward) |

## 2.4 分布式异步计算编排：Ray

在 verl 的系统架构中，RL中多种算法如PPO涉及多个角色（Actor, Critic, Reward, Reference）的频繁交互，且每个角色对计算资源（如算力与显存）及并行模式（如 TP 与 FSDP）的需求高度异构。**Ray** 作为底层分布式执行框架，提供了高效的资源抽象与跨节点协调能力。

### 2.4.1 异构资源解耦与 Placement Group：打破“显存互踢”

在复杂的 RL 系统中，最棘手的挑战在于如何处理**计算异构性**：Actor 往往需要低延迟的推理引擎（vLLM + TP 并行），而 Critic 或 Reward Model 则需要高吞吐的训练后端（FSDP/ZeRO-3 分片）。如果简单地将它们堆叠在物理显卡上，由于不同引擎的显存管理策略（如 vLLM 的预分配机制与 FSDP 的动态申请）互不感知，极易引发“显存互踢”导致的 OOM。

- **逻辑孤岛（Placement Group）**：Ray 引入了 `Placement Group` 机制，允许 Verl 在物理集群中划定一组**原子化的逻辑资源池**。它像是在繁忙的机房里圈出了几个“包间”，确保 Actor 的 TP 组和 Critic 的 FSDP 组拥有各自独立的算力与显存配额。
- **策略隔离与独占**：通过这种机制，Verl 可以精确地指定：Actor 独占前 $N$ 张卡的计算核心用于生成，而 Critic 在剩余的 $M$ 张卡上运行状态分片。这种**空间上的物理隔离**，使得 Actor 可以在不被 Critic 抢占显存的前提下，将剩余空间全部腾给 KV Cache。
- **切换的物理前提**：这种稳定的资源边界，是实现 **Hybrid Engine** 动态切换的物理基石。正因为每个 Worker Group 的资源是确定的，Verl 才能丝滑地执行“清空 KV Cache -> 重新分片权重 -> 启动训练”这一系列精密的显存体操，而不用担心受到其他组件的随机干扰。
### 2.4.2 基于分布式多进程 Actor 的异步 RPC 通信

在底层实现上，Ray 的 Actor 机制可以被视为一种高度工程化的**分布式多进程范式**。这种设计彻底解决了传统 Python 并发在 RL 大规模计算中的效率瓶颈。

1. **多进程绕过 GIL**

    - **真并行执行**：Ray 通过将模型实例（如 Actor, Critic）封装为 `Remote Actor`，本质上是在集群中开启了多个**完全独立的 Python 进程**。由于每个 Actor 运行在独立的进程空间中，拥有各自的 GIL。这使得 `verl` 可以让推理进程（vLLM）和训练进程（FSDP）在同一物理节点或跨节点同时运行，互不干扰，从而实现了物理层面的“真并行”。

> 关于线程、协程、进程的讨论欢迎参见 [这篇 GIL 博客]({% post_url 2026-04-29-gil-cpu-parallelism %})

2. **状态化进程驻留**
    - **持久化内存**：不同于普通的临时子进程，Ray Actor 进程在创建后会**持续驻留**在内存中。它长期持有模型权重矩阵（$W$），避免了在 PPO 循环中反复创建进程、重复加载模型带来的巨大初始化开销。
    - **微服务化抽象**：这些驻留进程被提升为“分布式微服务”，Trainer 仅需通过进程标识符即可发送指令。

3. **异步非阻塞 RPC：流水线式进程协同**
    - **异步语义**：在 `ray_trainer.py` 中，主控进程调用 `remote()` 方法后会立即获得一个 `ObjectRef`（类似 Future/Promise 凭据），而无需等待目标进程执行完毕。
    - **利用率最大化**：这种异步 RPC 机制允许 Trainer 在**采样进程**（Rollout）进行密集矩阵运算的同时，同步触发**奖励进程**（Reward Model）的打分预处理，或者**评估进程**的并发校验。
    - **工程意义**：通过这种多进程间的异步协同，`verl` 构建起了一套极其高效的计算流水线，最大化了集群硬件的整体利用率，避免了 CPU 调度死锁或单进程阻塞导致的 GPU 闲置。
    
### 2.4.3 分布式对象存储与零拷贝传输

RL 训练伴随着海量的中间数据交换（如Rollout阶段产生的输出移交给Update阶段作为输入）

- **Plasma Object Store**：Ray 维护了一个全局分布式的共享内存存储系统。
- **零拷贝（Zero-copy）效率**：当采样数据在不同的分布式节点（如从 Actor 传输到 Critic）流动时，Ray 利用**共享内存技术**避免了耗时的序列化与反向序列化操作，极大地降低了 PPO 循环中由于数据搬运产生的 IO 延迟。

# 三、Verl源码解析

如第二节中提到，`verl` 的系统设计很完整，但这篇文章我主要聚焦两点：第一点是 `verl` 最鲜明、也是官方文档中反复强调的核心特征——Hybrid-Controller；第二点是我在实际训练里直接遇到的关键机制——one-step-off-policy。下面的源码解析就围绕这两条主线展开。至于其余实现细节，这里不做展开，仅在第二节保留方法论层面的介绍。

## 3.1 Hybrid-Controller 在 `verl` 里的真实落地

> 问题：前言里提到的 Hybrid-Controller 在 `verl` 里到底怎么实现？这里的 controller 是不是 vLLM/FSDP 这样的推理/训练引擎？

先给结论：**不是。**  
在 `verl` 里，controller 是控制编排层，vLLM/FSDP/Megatron 是执行引擎层。两者是上下层关系，不是同一个概念。

### 3.1.1 核心概念

1. **Single-Controller（上层控制）**  
   可以理解为高层单控制（MPMD）：一个单进程 `RLTrainer` 管全局计算图编排，负责 rollout 触发、reward 评分触发、分布式训练任务下发等宏观调度。这个层面关注的是“**先后顺序与数据流**”。

2. **Multi-Controller（下层执行）**  
   可以理解为内部多控制（SPMD）：内部 worker 执行同构程序，用 FSDP/Megatron/VeOmni 等 trainer backend，或 vLLM/SGLang/TensorRT-LLM 等 rollout executor 做重计算，并通过 collectives 同步。这个层面关注的是“**怎么并行算得快**”。

3. **Hybrid-Controller（`verl` 方案）**  
   上层保留 single-controller 的灵活编排，下层交给 multi-process/multi-worker 的执行体系。即**主控负责调度 DAG，子节点负责吞吐。**

![Verl Hybrid-Controller 源码路径示意图](/assets/img/blog/blog2/verl-hybrid-controller-codepath.png)

### 3.1.2 源码里谁是“主控”

这里我用一句最直白的话概括：**主控调 group，就是 single-controller。**

在代码里，这个“主控”对应 `main_ppo.py` + `ray_trainer.py` 这一层：

- `run_ppo` 启动 `TaskRunner`；
- `TaskRunner.run(...)` 创建 `RayPPOTrainer`；
- `RayPPOTrainer.fit()` 执行主训练循环。

所以主控视角下看到的是这种“单进程风格”调用：

- `self.async_rollout_manager.generate_sequences(...)`
- `self.actor_rollout_wg.update_actor(...)`
- `self.checkpoint_manager.update_weights(...)`

主控并不直接操作每个 worker，它只表达“这一步该谁算”。

### 3.1.3 主控如何通过 RPC 驱动子节点

这一层也可以用一句话概括：**group 自动把任务分发到各个 worker，就是 multi-controller。**

具体过程是：

1. 每个 worker 先定义自己的方法（如 `update_actor`）。
2. 这些方法通过 `@register` 声明分发/收集规则。
3. `RayWorkerGroup` 初始化时，把这些 worker 方法绑定成 group 的同名代理方法。
4. 当主控调用 `group.update_actor(batch)` 时，group 内部自动完成：
   - 数据切分（dispatch）
   - 对每个 worker 的 `method.remote(...)` 调用
   - 结果聚合（collect）

这就是前面我们讨论的核心：主控不手写 `for worker in workers`，而是通过 group 一次调用完成整组 worker 的 RPC 执行。

### 3.1.4 子节点里才是 vLLM / FSDP 等引擎

再强调一次边界：

- 训练侧通过 `EngineRegistry.new(... backend=strategy ...)` 选择 FSDP / Megatron 等；
- rollout 侧通过 `get_rollout_class(...)` 选择 vLLM / SGLang / HF rollout。

这说明：

- **controller 决定“调用谁、何时调用、数据怎么分发收集”**；
- **engine 决定“这个节点内部用什么并行与 kernel 去算”**。

它们是控制层与执行层的解耦关系，这正是 Hybrid-Controller 的核心含义。

## 3.2 PPO中Actor和vLLM的权重同步

> 问题：在 `verl` 的 PPO 训练中，Actor 更新完参数后，权重是如何同步到 vLLM 的？同步频率是不是每个 step 一次？推理参数是否会“落后一个 step”？

这部分可以按两种执行范式来理解：**同步（sync）**与 **异步 one-step-off（async）**。两者的核心差别不是“是否同步权重”，而是**rollout 数据和 update 步骤的时间对齐关系**，即训练是on-policy或off-policy。

### 3.2.1 同步模式（RayPPOTrainer 主线）

同步模式下，一个 step 的临界路径基本是串行：

1. `rollout` 生成样本；
2. `actor/critic` 用这批样本更新；
3. `update_weights` 把新参数同步到 rollout 端；
4. 下一步 rollout 才用新参数。

可以抽象成：
$$
\text{step}_t \approx \text{gen}_t + \text{update}_t + \text{sync}_t
$$
对应主线 `ray_trainer.py` 中的顺序（简化）：

```python
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
...
actor_output = self._update_actor(batch)
...
self.checkpoint_manager.update_weights(self.global_steps)
```

因此，同步模式里“本步采样、下步生效”是正常现象，不是延迟故障。
### 3.2.2 异步模式（One-Step-Off）

![One-Step-Off 异步流水线示意图](/assets/img/blog/blog2/one-step-off-pipeline.png)
上面一条（$r$，rollout）和下面一条（$u$，update）**错位一个 step**。

- rollout 在时间 $t$ 用的是 $\omega_t$；
- update 在时间 $t+1$ 消费 rollout 在 $t$ 产生的数据；
- update 完成后参数变成 $\omega_{t+1}$，再同步给下一轮 rollout。
也就是：
$$
\text{update}_{t+1} \leftarrow \text{rollout}_{t}
$$

这正是 one-step-off 的定义：**训练数据相对当前策略滞后一步**。

在 `OneStepOffRayTrainer` 中，关键时序是：

1. `await batch_data_future`：取上一轮异步 rollout 结果（`gen`）；
2. `sync_rollout_weights`：把当前 actor 权重推到 rollout 端；
3. `create_task(_async_gen_next_batch(...))`：立刻启动下一轮异步生成；
4. 前台继续当前 batch 的 `compute_* + update_*`。

对应代码（简化）：
```python
with marked_timer("gen", timing_raw):
    _, _, _, batch, _ = await batch_data_future   # 拿到上一轮结果

with marked_timer("sync_rollout_weights", timing_raw):
    self._fit_update_weights()

batch_data_future = asyncio.create_task(self._async_gen_next_batch(...))  # 启动下一轮异步生成

batch = self._fit_compute_log_prob(batch)
batch = self._fit_compute_advantage(batch)
batch = self._fit_update_actor(batch)
```
因此在异步模式下：
$$
\text{step}_t \not\approx \text{gen}_t + \text{update}_t
$$

而更接近（存在重叠）：
$$
\text{step}_t \approx \max(\text{generate\_async}_t,\ \text{training\_only}_t) + \text{sync\_rollout\_weights}_t + \epsilon
$$
### 3.2.3 “权重不一致”现象存在，但可被校正

在 one-step-off 里，同一条轨迹在 rollout 和 update 时对应的策略参数一般不一致，也就是 rollout 用 $\pi_b$（旧），update 用 $\pi_\theta$（新），这会带来 off-policy 偏差，因此需要补偿（importance sampling / rollout correction）。

#### (1) 目标与采样分布不一致

我们希望优化目标函数：

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[f(\tau)]
$$

但数据来自 $p_b(\tau)$。用测度变换可写为：

$$
\mathbb{E}_{\tau\sim p_\theta}[f(\tau)]
=
\mathbb{E}_{\tau\sim p_b}\left[\frac{p_\theta(\tau)}{p_b(\tau)}f(\tau)\right]
$$

其中，重要性权重定义为：

$$
w(\tau)=\frac{p_\theta(\tau)}{p_b(\tau)}
$$

这里的直觉是：从旧策略迁移到新策略后，一条轨迹是否还“值得被学习”，取决于它在新策略下出现的相对概率。比如新策略下采到该轨迹的概率趋近于 0，就不应继续根据这条轨迹更新参数。

#### (2) 在 Transformer 语境下体现

自回归模型下（省略与 $\theta$ 无关项）：

$$
p_\theta(\tau)\propto \prod_t \pi_\theta(a_t|s_t),\quad
p_b(\tau)\propto \prod_t \pi_b(a_t|s_t)
$$

所以有：

$$
\log w(\tau)=\sum_t\left[\log\pi_\theta(a_t|s_t)-\log\pi_b(a_t|s_t)\right]
$$

进一步可得：

$$
w(\tau)=\exp\left(\sum_t(\log\pi_\theta-\log\pi_b)\right)
$$

这就是“通过 logprob 比率补偿”的数学来源。

当新旧策略偏差过大时，通常会配合 **PPO 的 Clip 机制**来防止更新步幅过大导致训练崩溃。

### 3.2.4 实验量化结果：异步比同步快多少，硬等待有多少

基于一次总计699step的训练数据，可以做一个反事实估算：

1. **如果改成同步实现（不再 overlap）**，每步近似为：
$$
\text{step}_{\text{sync-est}} \approx \text{training\_only} + \text{generate\_async} + \text{sync\_rollout\_weights}
$$
统计结果：
- 异步实测总时长：$62976.392\ \text{s}$
- 同步估算总时长：$117517.538\ \text{s}$
- 比值（同步/异步）：$1.866\times$

这意味着在当前配置下（gen和update的gpu数量1：1），one-step-off 的异步重叠把总时长压到了同步实现的大约 $53.6\%$。

2. **异步中难以消除的等待窗口**：
$$
\text{wait\_hard}=\sum_t\left(\text{wait\_prev\_gen}_t+\text{sync\_rollout\_weights}_t\right)
$$

其中用指标映射：

- $\text{wait\_prev\_gen}_t \leftrightarrow \texttt{第t步等待t-1步rollout}$
- $\text{sync\_rollout\_weights}_t \leftrightarrow \texttt{第t-1步rollout结束后把权重同步给vLLM}$

统计结果：
- $\sum \texttt{timing\_s/gen}=357.007\ \text{s}$（占总时长 $0.57\%$）
- $\sum \texttt{timing\_s/sync\_rollout\_weights}=3042.036\ \text{s}$（占总时长 $4.83\%$）
- 合计硬等待：$3399.043\ \text{s}$，占总时长 **$5.40\%$**

结论：这 $5.40\%$ 可以视为当前系统在“边界等待 + 权重切换”上的硬成本窗口；它未被完全消除，但相对于同步实现的反事实时长，异步 one-step-off 仍然带来了显著优化。

# 四、Verl 框架核心设计总结

Verl 的核心竞争力并非仅在于其功能接口的完备性，而在于它通过对 RL 后训练（Post-training）计算负载的深度拆解，在系统架构上实现了 **灵活编排与极致吞吐的平衡**。其创新设计主要体现在以下三个维度：

### 1. 混合控制架构（Hybrid-Controller）

Verl 创新性地提出了 **Hybrid-Controller** 方案，打破了传统分布式框架在开发灵活性（Single-Controller）与执行性能（Multi-Controller）之间的平衡难题：

* **上层解耦编排**：保留了类似单机脚本的 Single-Controller 逻辑，由主控进程（如 `RayPPOTrainer`）负责定义任务的先后顺序与数据流向，降低了算法开发门槛。
* **下层高效执行**：在微观执行层面，Verl 将任务下发给由 Ray 管理的“子节点群”，各组内部以 Multi-Controller 模式驱动 **vLLM** 或 **FSDP** 等专业引擎，确保了硬件性能的达峰。

### 2. 异构引擎的切换（Hybrid Engine）

针对 RL 训练中推理（Rollout）与更新（Update）截然不同的显存与计算瓶颈，Verl 实现了模型权重的动态转换：

* **Rollout 阶段（带宽优化）**：切换为 vLLM 模式，权重聚合为 TP 格式，利用 **PagedAttention** 机制消除显存碎片，将空间留给 KV Cache 以实现高并发采样。
* **Update 阶段（算力优化）**：权重立刻重切为 FSDP 格式，通过分片存储（Sharding）压低基础显存占用，全力支持**激活值驻留**，从而承载超大规模参数的训练。

### 3. 异步流水线设计（One-Step-Off Policy）

Verl 通过 Ray 提供的异步 RPC 能力，重构了 PPO 的迭代逻辑，显著提升了集群利用率：
* **计算与生成的 Overlap**：通过 One-Step-Off 机制，Verl 允许当前步的**参数更新**与下一步的**数据采样**在时间轴上重叠（错位执行），极大地减少了 GPU 的空转等待。


# 五、延伸阅读与致谢

在整理这篇 Blog 的过程中，我受益于以下优秀的开源项目与教程：

- **[verl 官方文档](https://verl.readthedocs.io/)**：了解框架设计哲学的第一站，关于 **Hybrid-Controller** 的定义非常精辟。
- **[动手学深度学习 (D2L)](https://zh.d2l.ai/)**：李沐老师的经典之作，我关于多 GPU 并行范式的理解大多源自于此。

同时，特别感谢童雨轩学长在调研过程中提供的宝贵技术资料与思路启发~
