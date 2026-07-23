---
layout: post
title: "膜拜 & 瑞平 GDPO"
date: 2026-07-24 01:30:00 +0800
description: "读 GDPO 论文笔记：多奖励设定下从 GRPO 到按奖励各自归一化再聚合的动机、分析与实验。"
tags: [work, ai, post-training]
giscus_comments: true
stars: 0
---

# 一、Background

指路原文：[GDPO (arXiv:2601.05242)](https://arxiv.org/pdf/2601.05242)

随着模型在各个单领域任务上的表现逐渐饱和，新出现的赛道“agent”背后体现的模型能力是更长线、更综合的模型能力。在以tool call、math reasoning为代表的长链条任务中，由于SFT的teacher-forcing带来的off-policy本质问题，RL依然是解决问题的关键手段。

> 对于非第一梯队的模型，各种各样的蒸馏或许是收益最快的方法，但第一梯队的模型或许只能通过RL来探索自身更强的能力

正如[前文]({% post_url 2026-05-11-post-training-rl-algorithms %})所说，长链条任务对于RL天然提出了更高的要求，单一的奖励设计不足以约束模型行为，于是在这类任务上大家普遍采用多奖励叠加的方式。

![多奖励叠加示意](/assets/img/blog/blog3/multi-reward.png)

简而言之，一个rollout的全部reward指标先求和再作为总的reward当作这条rollout的奖励，进行减均值除以标准差的操作得到Advantage。

# 二、Method

本文提出的GDPO，本质上就是在上述算法的基础上，对于一个case的rollout，在不同的奖励上先各自归一化，之后再求和作为一个rollout的reward。

![GDPO 公式示意一](/assets/img/blog/blog3/fun1.png)

![GDPO 公式示意二](/assets/img/blog/blog3/fun2.png)

> 这里有一个细节是作者提到使用了Batch-wise的归一化，使得训练更稳定

# 三、Analysis

作者通过离线的分析，将不同的reward decouple开显然会带来更多的adv group，曲线如下：

![离线 adv group 曲线](/assets/img/blog/blog3/offline-curve.png)

# 四、Experiment

读到这里停下来回顾一下，经过数学的分析和离线的测试，GDPO预期可以带来的收益是什么呢？我认为可以概括为两点，第一点是GDPO解耦了不同的奖励，缓解了GRPO对奖励梯度的削弱作用，因而模型可以得到更精确的反馈进而获得更快的收敛速度以及最佳性能；第二点是GDPO增加了batch-wise归一化操作，在任务类型比较单一的训练setting上预计会获得更稳定的效果。

> 当然这是我自己读到这里的想法，如果大家在看实验之前对这个操作产生了不同的预期欢迎一起讨论！

继续往后读，论文在3个setting上验证（or 试图验证）GDPO的有效性。

## 4.1 Tool Call

我不是很熟悉的Agentic RL领域，但相对来说训练曲线和评测数据看起来还是很漂亮的，比较有力的证明了我的第一点猜想，收敛速度及最佳性能的提升。

> 这里如果是我自己做实验，或许会补充一下GDPO和GRPO两个setting模型实际训练过程中的adv group曲线，评估一下模型是否真正是由于reward的精细拿到了更高的性能。毕竟离线的分析只是adv group的理论上限值。

![Tool Call 训练曲线](/assets/img/blog/blog3/tool-call-1.png)

![Tool Call 评测结果](/assets/img/blog/blog3/tool-call-2.png)

## 4.2 Math Reasoning

在这个setting上，reward的设置较为稀疏，具体分为格式奖励和内容奖励。格式奖励只惩罚过长输出，内容奖励根据数学推理最终答案的正误给一个binary的0、1分数。

论文中列出来的训练曲线如图：

![Math Reasoning 训练曲线](/assets/img/blog/blog3/math-reasoning.png)

这里我觉得ablagation不是特别充分的点在于我认为这个setting作者的核心观点是GDPO更加稳定，我认为增加一个GRPO + batch_wise归一化的对比实验会更明确一些。

> 当然因为我读的论文还是太少，也没自己写过论文，不知道真实做过的实验有多少会expose出来。论文在提到batch-wise的操作的时候提到了这会很大程度上提升训练的stability。我猜他们先训发现奖励信号可能太稀疏太容易训崩溃了之后想到了这个操作实验才稳定下来。 /doge

## 4.3 Coding Reasoning

懒得看了，大家有兴趣去看看吧。

# 五、Conclusion

看到这里，问问自己为什么这篇论文能拿到spotlight？我觉得论文中反复提到的，大家都会defaulted to use GRPO with multi reward，而这确实是从PPO跳到GRPO时代大家的commensense。而在Background中提到的，当GRPO迎来了更长链条更综合的任务从而不得不增加reward设计的时候，第一个提出GRPO或许不适合直接trivial的多reward叠加的人也许是第一个吃了螃蟹的人。

换句话说，GDPO某种程度上为后续RL在multi reward的发展指明了方向。

膜拜。
