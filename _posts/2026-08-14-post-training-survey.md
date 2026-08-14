---
layout: post
title: "Post-training: Motivation, Self-Improvement, and Open Problems"
date: 2026-08-14 13:31:00 -0400
description: "A survey of why post-training matters, how self-improving post-training systems work, and the central open problems in feedback and policy optimization."
article_type: Survey
tags: [ai, post-training, reinforcement-learning]
thumbnail: /assets/img/blog/post-training-survey/post-training-hero.webp
giscus_comments: true
stars: 0
---

# Why Post-training？

**Pre-training primarily gives capability and knowledge; post-training shapes behavior.**

![Abstract illustration of post-training shaping broad model capabilities into reliable behavior](/assets/img/blog/post-training-survey/post-training-hero.webp)

## Alignment

### 1. Objective mismatch

While the classic mode for pre-train is next-token-prediction, pre-trained models have sufficient knowledge learnt from human-write online data. However, when asked basic question like "what's the weather like today in the US?", those models performed worse than extremely simple model. For example, it may give a response like "what's the weather like today in China?" because the mass distribution of following data in pre-training dataset: "Q: question 1 Q: question 2 Q: question 3".

Therefore, the first and most important reason for post-training is to convert a general next-token predictor into a model that reliably follows human instructions and intentions.

### 2. Preference underdetermination

A model with basic competence to chat with human doesn't make us satisfied. There are lots of questions with countless answers. Here comes the second reason for post-training: to **meet human's Preference**.

For example, there might be 3 answers to a question with similar probability that model will offer. But for human, the answer A may be too long, the second is short but with fake news and only the answer C is preferred. Then the process of training model to increase the probability of answer C.

## Capability Improvement

Post-training, especially RL, can significantly improve a model's effective problem-solving capability. However, **where this improvement comes from remains an important open question**.

### 1. Capability elicitation

The pretrained model may already contain useful reasoning strategies and successful trajectories, but assigns them relatively low probability. Post-training increases the probability of these successful behaviors and makes latent capabilities more reliably accessible.

As the conclusion in the paper [_Does RL Really Incentivize Reasoning Beyond the Base Model?_](https://openreview.net/pdf/5796cee2ae1c6db88604ce9da20316481e66739d.pdf) ,**RL may improve performance primarily by shifting probability mass toward reasoning patterns already present in the base model.**

Btw, some may question that the article above has proved that improvement in post-training was to clicitate capability, then why is this still an open question? My answer is: what the work has proved is the upper bound of RL under current pre-trained model, RL method, benchmark metrics (like pass@k). These constrains might lead to conclusion above. But the theoretical frontier might be far far far away.

> Besides, there might be some training virance on that single work : )

### 2. Genuine capability expansion

The stronger hypothesis is that RL does more than reweight existing behaviors: through exploration and feedback, it may discover and reinforce new reasoning strategies, potentially expanding the model's capability frontier.

I would like to share an exciting work here, which is exactly why I am devoted to post-training: [_The technical report of Deepseek-R1_](https://arxiv.org/abs/2501.12948).

**DeepSeek-R1-Zero provides a particularly clean experiment for this hypothesis.** Instead of first teaching the model how to reason through supervised cot demonstrations, DeepSeek directly applies RL to a base model. The supervision is surprisingly simple: the model explores different reasoning trajectories, while rule-based rewards mainly tell it whether the final answer is correct and whether the output follows the required format.

Despite such sparse supervision, increasingly sophisticated reasoning behaviors emerge during training. The model learns to **think for longer, reflect on intermediate results, backtrack when a solution appears incorrect, and explore alternative approaches**. None of these reasoning strategies are explicitly demonstrated or individually rewarded.

> Doesn't it seem like a baby first saying "mama" !!

This suggests a fundamentally different learning paradigm:

**Traditional SFT:** Humans demonstrate _how to solve the problem_ → the model imitates.

**R1-Zero-style RL:** Humans define _what counts as success_ → the model explores how to achieve it.

![Traditional SFT compared with R1-Zero-style reinforcement learning](/assets/img/blog/post-training-survey/sft-vs-r1-zero.svg)

Sadly, this does not prove genuine capability expansion. These reasoning behaviors may already exist with very low probability in the base model and simply be amplified by RL. Whether RL truly discovers new reasoning capabilities or primarily elicits latent ones therefore remains an open question.

## Ultimate goal : Self-improvement

Post-training, especially **on-policy learning**, provides a fundamentally different learning paradigm: instead of merely imitating static human-generated data, the model **explores its own capability frontier, generates new experience, receives feedback, and learns from the consequences of its own behavior**.

This creates the possibility of a self-improving loop:

**Explore → Generate new experience → Receive feedback → Learn → Become stronger → Explore further**

![The self-improving post-training loop](/assets/img/blog/post-training-survey/self-improvement-loop.svg)

The long-term significance is that the model is no longer strictly bounded by human demonstrations. As long as reliable feedback or an environment is available, it may discover solutions and strategies that humans never explicitly demonstrated.

Therefore, scalable on-policy post-training could be one of the key ingredients for AI systems to **move beyond imitation of human intelligence and potentially surpass human-level performance in domains where success can be reliably evaluated.**

# How to post-train?

Here, we focus specifically on **capability-oriented post-training that has the potential to go beyond static human-generated data**. Conventional SFT and imitation-based post-training are therefore not our main focus, since their learning signal is fundamentally bounded by existing demonstrations.

The paradigm of interest is an **iterative, on-policy learning loop** in which the model actively explores, generates its own experience, receives external feedback, and learns from the consequences of its behavior. In principle, such a paradigm allows the model to discover solutions or strategies that humans never explicitly demonstrated.

A modern self-improving post-training system can be roughly decomposed into six components:

1. **Task & Curriculum** — Determine _what the model should explore_. To sustain self-improvement, tasks should evolve with the model rather than remain a fixed set of human-written problems.
2. **Rollout & Exploration** — The current policy generates new trajectories through sampling or interaction. The crucial goal is not merely to generate more responses, but to discover useful behaviors beyond the policy's dominant modes and potentially push its current capability frontier.
3. **Environment, Verifier & Reward** — Determine _what counts as progress_. Feedback can come from rule-based verifiers, executable environments, learned reward models, or process-level signals. Importantly, the evaluator need not provide an expert solution—it only needs to reliably distinguish better behavior.
4. **Experience & Data Management** — Transform self-generated trajectories into useful learning experience through selection, filtering, weighting, replay, and difficulty control.
5. **Policy Optimization** — Convert experience and feedback into an improved policy. Algorithms such as PPO, GRPO, DAPO, and GSPO provide different mechanisms for making this update efficiently and stably.
6. **Systems & Scaling** — Make the entire loop computationally feasible at scale, including massive rollout generation, distributed training, environment execution, policy synchronization, and asynchronous training.

The resulting loop is:

**Task → Exploration → Feedback → Experience → Policy Update → Stronger Policy → New Exploration → ...**

The long-term goal is therefore a system in which the policy, generated experience, curriculum, and potentially even the feedback mechanism **co-evolve**, reducing dependence on static human-generated data and enabling sustained self-improvement beyond the capabilities explicitly demonstrated by humans.

# Where are the bottlenecks?

Among all the steps above, I suppose the following 2 are what matter: environment and policy optimization. Besides that, the system is of high importance but beyond my competence, so it isn't considered here.

## Environment, Verifier & Reward

The fundamental challenge of reward design is a **trade-off between reliability and coverage**.

**Verifiable rewards** evaluate model behavior through objective external signals, such as exact-answer checking, code execution, unit tests, theorem provers, or environment outcomes. They are highly reliable, scalable, and difficult to manipulate, making them particularly suitable for large-scale RL. However, their **coverage is limited**: many important tasks, such as scientific reasoning, open-ended research, writing, or complex real-world decision making, do not have easily verifiable ground truth.

**Learned reward models**, in contrast, can evaluate much broader and more subjective behaviors by approximating human or expert judgment. However, this broader coverage comes at the cost of **lower reliability**: reward models can make mistakes, fail to generalize as the policy improves, and eventually become targets for reward hacking.

Therefore, a central challenge for self-improving post-training is:

> **How can we expand the coverage of feedback without sacrificing its reliability?**

In short:

**Verifiable Reward = High Reliability + Low Coverage**

**Learned Reward = High Coverage + Lower Reliability**

![The reliability and coverage trade-off between verifiable and learned rewards](/assets/img/blog/post-training-survey/reward-tradeoff.svg)

For sustained self-improvement, the long-term goal is to build feedback mechanisms that are both **broad enough to evaluate increasingly complex behaviors and reliable enough to remain trustworthy as the policy becomes stronger.**

## Policy Optimization & Credit Assignment

Even with a perfect reward signal, another fundamental question remains: **how can the model efficiently convert feedback into capability improvement?**

From the classical policy-gradient perspective, RL fundamentally performs **action-level credit assignment**. For an LLM, each token can be viewed as an action conditioned on the current prefix (state). Ideally, we want to estimate how much choosing a particular action improves the expected future return:

**State + Action → Future Return → Advantage**

![Sequence-level reward and token-level credit assignment](/assets/img/blog/post-training-survey/credit-assignment.svg)

Importantly, this is not simply asking whether an individual token is "correct", but rather **how much causal credit that action deserves for the eventual success or failure of the trajectory**.

However, modern RLVR usually provides only a sparse outcome reward for the entire response. GRPO further removes the critic used in PPO for temporal credit estimation, so the same sequence-level advantage is often broadcast to all tokens. This creates a potentially severe information bottleneck: successful and harmful intermediate actions within the same trajectory may receive essentially the same learning signal.

Recent methods such as GSPO approach the problem from the opposite direction. Since current rewards and advantages are largely sequence-level, GSPO moves importance weighting and clipping toward the sequence level as well, improving optimization stability. This suggests that sequence-level optimization is not merely an approximation—it represents a different choice of optimization granularity that may be better matched to today's sequence-level feedback.

This leads to a deeper open question:

**What is the appropriate granularity of reward, credit assignment, and policy optimization for LLM reinforcement learning?**

# Open Problems and Outlook

Despite rapid progress in post-training, scalable self-improvement remains far from solved. Several fundamental questions remain open: How can feedback achieve broader coverage without sacrificing reliability? What is the appropriate granularity for reward, credit assignment, and policy optimization? How can RL remain stable and sample-efficient as training scales? And ultimately, can post-training continuously expand a model's capability frontier rather than merely amplify behaviors already present in the base model?

These questions suggest that the challenge of self-improving post-training is not a single algorithmic problem, but a systems-level learning problem spanning **exploration, feedback, credit assignment, optimization, and scaling**.

I'm on my way!!!

# References

1. Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. [_Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?_](https://openreview.net/pdf/5796cee2ae1c6db88604ce9da20316481e66739d.pdf). NeurIPS, 2025.
2. DeepSeek-AI. [_DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning_](https://arxiv.org/abs/2501.12948). 2025.
3. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. [_Proximal Policy Optimization Algorithms_](https://arxiv.org/abs/1707.06347). 2017.
4. Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. [_DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_](https://arxiv.org/abs/2402.03300). 2024.
5. Qiying Yu et al. [_DAPO: An Open-Source LLM Reinforcement Learning System at Scale_](https://arxiv.org/abs/2503.14476). 2025.
6. Chujie Zheng et al. [_Group Sequence Policy Optimization_](https://arxiv.org/abs/2507.18071). 2025.
