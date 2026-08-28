［#1］
Published in Transactions on Machine Learning Research (07/2025)

# Sparsity-Driven Plasticity in Multi-Task
［#2］
Reinforcement Learning

［#3］
Aleksandar Todorov$^\dagger$$^*$
a.todorov.4@student.rug.nl

［#4］
Juan Cardenas-Cartagena$^\dagger$
j.d.cardenas.cartagena@rug.nl

［#5］
Rafael F. Cunha$^\dagger$
r.f.cunha@rug.nl

［#6］
Marco Zullich$^\dagger$
m.zullich@rug.nl

［#7］
Matthia Sabatelli$^\dagger$
m.sabatelli@rug.nl

［#7］
$^\dagger$University of Groningen, Groningen, The Netherlands

［#8］
Reviewed on OpenReview: https://openreview.net/forum?id=9L4Z23EJE9

## Abstract
［#9］
Plasticity loss, a diminishing capacity to adapt as training progresses, is a critical challenge in deep reinforcement learning. We examine this issue in multi-task reinforcement learning (MTRL), where higher representational flexibility is crucial for managing diverse and potentially conflicting task demands. We systematically explore how sparsification methods, particularly Gradual Magnitude Pruning (GMP) and Sparse Evolutionary Training (SET), enhance plasticity and consequently improve performance in MTRL agents. We evaluate these approaches across distinct MTRL architectures (shared backbone, Mixture of Experts, Mixture of Orthogonal Experts) on standardized MTRL benchmarks, comparing against dense baselines, and a comprehensive range of alternative plasticity-inducing or regularization methods. Our results demonstrate that both GMP and SET effectively mitigate key indicators of plasticity degradation, such as neuron dormancy and representational collapse. These plasticity improvements often correlate with enhanced multi-task performance, with sparse agents frequently outperforming dense counterparts and achieving competitive results against explicit plasticity interventions. Our findings offer insights into the interplay between plasticity, network sparsity, and MTRL designs, highlighting dynamic sparsification as a robust but context-sensitive tool for developing more adaptable MTRL systems.

## 1 Introduction
［#10］
Although deep reinforcement learning (DRL) agents have demonstrated impressive results in various applications (Levine et al., 2016; Silver et al., 2017; Bellemare et al., 2020; Mathieu et al., 2023), these achievements come with notable trade-offs. Attaining state-of-the-art performance often relies on large-scale computational resources and heavily overparameterized models (Botvinick et al., 2019; Glanois et al., 2022; Thompson et al., 2022), which may lead to agents that either generalize poorly (Kirk et al., 2023) or struggle to adapt to new tasks or data over time. The former issue is a topic of interest within the transfer learning literature (Farebrother et al., 2020; Sabatelli & Geurts, 2021; Sasso et al., 2023; Zhu et al., 2023), while the latter is commonly referred to as plasticity loss (Nikishin et al., 2022; Lyle et al., 2023; Dohare et al., 2024). Plasticity loss manifests through several interconnected optimization pathologies: gradient interference leading to premature convergence (Lyle et al., 2024a), representational collapse, limiting the diversity of learned features

［#10］
$^*$Corresponding author


［#11］
(Moalla et al., 2024), and neuronal saturation or dormancy that reduces effective network capacity (Bjorck et al., 2021; Sokar et al., 2023). While these challenges have been primarily investigated within single-task RL, (Nikishin et al., 2023; Abbas et al., 2023; Klein et al., 2024; Nauman et al., 2024a; Dohare et al., 2024), in this paper, we study them under the lens of multi-task reinforcement learning (MTRL), where maintaining representational flexibility across diverse tasks with potentially conflicting demands is even more crucial (Teh et al., 2017; Sodhani et al., 2021; D'Eramo et al., 2024). It naturally follows that this increased need for dynamic adaptation can make MTRL agents especially vulnerable to plasticity loss, as networks must simultaneously accommodate varied objectives without experiencing negative task interference (Liu et al., 2023). The necessity of determining which knowledge to share across tasks, and how to share it without harmful interference, further complicates the learning process (Devin et al., 2016; Sasso et al., 2023). Moreover, this challenge can be even further exacerbated by inefficient use of network capacity (Kumar et al., 2021), with significant portions of large networks becoming underutilized during training, ultimately hindering the acquisition of a universal policy capable of addressing multiple tasks concurrently. Recent work in neural network pruning offers a promising direction beyond mere compression, showing that sparse agents can match or even exceed dense counterparts in single-task RL (Livne & Cohen, 2020; Graesser et al., 2022; Obando-Ceron et al., 2024). Notably, methods like Gradual Magnitude Pruning (GMP) have shown positive effects on both single-task performance and plasticity (Obando-Ceron et al., 2024). Similarly, dynamic sparse training methods such as Sparse Evolutionary Training (SET) (Mocanu et al., 2018) have also proven effective in single-task RL, offering an alternative way to maintain and adapt sparsity throughout training (Graesser et al., 2022). These successes suggest that such sparsification approaches could address the optimization pathologies that undermine effective multi-task learning. Nonetheless, a systematic investigation of their impact on MTRL, where representational flexibility demands are significantly higher (Devin et al., 2016), remains largely unexplored.

［#12］
This paper investigates whether sparsification methods, specifically GMP and SET (Mocanu et al., 2018), can enhance plasticity in MTRL agents, thereby improving performance across multiple tasks simultaneously. Our choice to explore these methods is motivated by their demonstrated success in single-task settings and the need to understand their efficacy in the MTRL domain. We evaluate this across various multi-task architectures, including shared backbones with task-specific heads (MTPPO), Mixture of Experts (MoE) (Ceron et al., 2024), and Mixture of Orthogonal Experts (MOORE) (Hendawy et al., 2024), using common MTRL benchmarks that range from partially observable environments with sparse reward to high-dimensional and continuous state and action spaces. Our central aim is to understand if the benefits of pruning can be primarily attributed to the mitigation of key plasticity loss indicators. Our experiments compare sparse agents against dense baselines, less adaptive sparsification techniques, and a suite of alternative plasticity-inducing or regularization methods, including Layer Normalization (Ba et al., 2016; Lyle et al., 2024a), ReDo (Sokar et al., 2023), Reset (Ash & Adams, 2020; Nikishin et al., 2022), and Weight Decay.

［#13］
Our main contributions are, therefore, threefold:

［#14］
- We establish that sparsification methods, particularly Gradual Magnitude Pruning (GMP) and Sparse Evolutionary Training (SET), serve as effective mechanisms for mitigating key indicators of plasticity degradation in MTRL, such as neuron dormancy and representational collapse. While the extent of these benefits varies with network architecture, sparse agents, especially in MTPPO and MoE configurations, consistently exhibit improved plasticity profiles compared to their dense counterparts.
- We empirically demonstrate that these plasticity improvements induced by sparsification often correlate with enhanced multi-task performance. Sparse agents frequently outperform dense baselines and demonstrate competitive performance against alternative, specialized methods explicitly designed to induce plasticity, as well as common regularization techniques.
- We show that the impact of sparsification on both plasticity and performance is architecture-dependent, offering insights into the interplay between network design, sparsity, and learning dynamics. This highlights sparsification as a valuable but context-sensitive tool in the MTRL toolkit. Furthermore, we highlight that beyond performance, sparsification offers inherent advantages such as potential for computational efficiency and a distinct form of implicit regularization not fully replicated by common regularization methods.


## 2 Background

［#15］
This section provides the necessary context for our approach. We begin by outlining the mathematical preliminaries underlying our framework, including key concepts and notations from reinforcement learning. Subsequently, we review related work, focusing on recent advances in sparsity in deep reinforcement learning, plasticity loss, and multi-task learning.

### 2.1 Preliminaries

［#16］
We consider the Partially Observable Markov Decision Process (POMDP), defined by a tuple $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\Omega,\mathcal{O},\gamma)$, consisting of a state space $\mathcal{S}$, an action space $\mathcal{A}$, transition dynamics $\mathcal{P}:\mathcal{S}\times\mathcal{A}\to\Delta(\mathcal{S})$, a reward function $\mathcal{R}:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$, observation space $\Omega$, observation probability function $\mathcal{O}:\mathcal{S}\times\mathcal{A}\to\Delta(\Omega)$, and discount factor $\gamma\in[0,1)$. At each timestep $t$, the agent is situated in the true state $s_t\in\mathcal{S}$ and performs an action $a_t\in\mathcal{A}$. This causes the agent to transition to a new state $s_{t+1}\in\mathcal{S}$, receiving an observation $o_{t+1}\in\Omega$, and a reward $r_{t+1}=\mathcal{R}(s_t,a_t)$. The objective is to learn a policy $\pi_\theta(a_t|o_t)$ with parameters $\theta$ that maximizes the expected sum of discounted future rewards $J(\theta)$. In MTRL, the agent must learn a policy for a distribution of tasks $\mathcal{T}$. We adopt the Block Contextual POMDP framework (Sodhani et al., 2021; Hendawy et al., 2024), defined as $(\mathcal{C},\mathcal{S},\mathcal{A},\mathcal{M}')$, where $\mathcal{C}$ represents the contextual space such that $c\in\mathcal{C}$ identifies a specific task $\tau\sim\mathcal{T}$. The mapping $\mathcal{M}'(c)$ provides the task-specific POMDP components $\{\mathcal{R}^c,\mathcal{P}^c,\mathcal{S}^c,\Omega^c,\mathcal{O}^c,\gamma^c\}$. The policy is now conditioned on the current observation $o\in\Omega^c$ and task context $c\in\mathcal{C}$. The objective is to maximize the expected return across all tasks $\mathbb{E}_{\tau\sim\mathcal{T}}[J_\tau(\theta)]$.

### 2.2 Related Work

［#17］
**Sparsity in Reinforcement Learning** While deep reinforcement learning has traditionally relied on overparameterized networks, recent research challenges the necessity of such scale, suggesting that sparse networks can match or even exceed the performance of dense models (Livne & Cohen, 2020; Graesser et al., 2022). This trend highlights that DRL agents often underutilize their capacity (Kumar et al., 2021) or overfit to early experiences (Nikishin et al., 2022), making them particularly amenable to the regularizing effects of sparsity. Pruning neural connections reduces model complexity and noise, offering a form of structural regularization that can improve robustness and generalization (Jin et al., 2022). Our work investigates how sparsity-based methods can influence learning dynamics and plasticity in multi-task RL.

［#18］
**Plasticity Loss in Reinforcement Learning** It is well known that Reinforcement learning systems face a unique form of non-stationarity stemming from evolving policies, shifting data distributions, and the bootstrapping nature of value updates. This can culminate in the form of plasticity loss, a reduced ability of the network to learn from new experiences, even within familiar data distributions (Lyle et al., 2022; Dohare et al., 2024). Plasticity loss often manifests as premature performance plateaus, training instability, and heightened sensitivity to hyperparameter settings (Igl et al., 2021; Berariu et al., 2023; Klein et al., 2024). Current understanding attributes plasticity loss primarily to unstable learning targets that create challenging optimization landscapes, with associated symptoms like collinear gradients (Lyle et al., 2024a), representational collapse (Moalla et al., 2024), and volatile gradient norms under adaptive optimizers (Lyle et al., 2024b). Internally, networks may suffer from shifting activation distributions, neuron saturation, or increasing dormancy over time (Sokar et al., 2023; Bjorck et al., 2021). Several interventions have been proposed to mitigate these effects. These include resetting techniques, such as periodic last-layer reinitialization (Ash & Adams, 2020; Nikishin et al., 2022; 2023), parameter update modulation strategies like Hare and Tortoise networks (Lee et al., 2024), and various architectural or optimization-based approaches, including weight decay (Sokar et al., 2023), deep Fourier features (Lewandowski et al., 2024a), and classification-based value learning (Farebrother et al., 2024). Normalization layers have also demonstrated benefits in this regard (Bhatt et al., 2023). Notably, Obando-Ceron et al. (2024) showed that gradual pruning can outperform many methods specifically designed to promote plasticity. This supports the broader notion that general-purpose regularization might offer a more robust and simpler solution to plasticity loss than domain-specific mechanisms, reinforcing the broader lesson that simplicity often outperforms specialized interventions (Klein et al., 2024; Nauman et al., 2024a).


［#19］
Multi-Task Reinforcement Learning MTRL seeks to train a single agent across multiple tasks, balancing knowledge sharing for transfer against the risk of negative interference. Common MTRL techniques include shared encoders with task-specific heads (Teh et al., 2017), modular network designs (Yang et al., 2020), reward normalization (Hessel et al., 2018), compositional policy learning (Sun et al., 2022), and gradient projection or masking strategies (Yu et al., 2020; Hendawy et al., 2024). Mixture-of-Experts models have also gained traction, often enhanced with attention or orthogonality constraints for better task separation (Ceron et al., 2024; Cheng et al., 2023). Similarly, maintaining weight matrix orthogonality through regularization has been explored to enhance plasticity in continual learning settings, which face similar challenges to MTRL (Chung et al., 2024). A key observation in MTRL is that, unlike trends in supervised learning, simply scaling model capacity does not inherently guarantee performance improvements (Hansen et al., 2023; Ceron et al., 2024; Nauman et al., 2024b). Supervised approaches like SimBa, for instance, suggest that gains from scaling require careful inductive biases (Lee et al., 2025a). In contrast, network sparsity has demonstrated improvements in both generalization and plasticity in RL without necessarily relying on increased model scale (Graesser et al., 2022; Obando-Ceron et al., 2024). Despite the promise of sparsity, its implications in MTRL have remained largely unexplored. To the best of our knowledge, this work is the first to systematically examine how different sparsity-inducing techniques affect performance and plasticity in the multi-task regime. We aim to fill this gap by investigating how pruning and sparse connectivity can mitigate plasticity loss and promote stable, generalizable learning in complex task environments.

## 3 Experimental Setup

［#20］
Our experiments systematically compare the effects of different sparsification approaches against dense baselines and two families of plasticity-enhancing interventions. The first family comprises explicit plasticity-restoring techniques that directly intervene on the agent's parameters to counteract plasticity loss. These include ReDo (Sokar et al., 2023), which periodically resets dormant neurons based on activity thresholds, and Reset (Ash & Adams, 2020; Nikishin et al., 2022), which reinitializes specific network layers at fixed intervals to combat primacy bias. The second family involves more implicit regularization-based mechanisms, which do not directly manipulate network dynamics but are known to stabilize training and encourage generalization. Specifically, we evaluate standard Weight Decay (WD) applied to dense agents, and Layer Normalization (LayerNorm) (Ba et al., 2016), which has recently been linked to mitigating plasticity loss by reducing covariate shift and promoting balanced neuron activations (Lyle et al., 2024a).

［#21］
Unless otherwise specified, we benchmark these methods across three representative multi-task reinforcement learning algorithms: MTPPO, a shared-policy baseline with task-specific heads; a Mixture-of-Experts (MoE) model (Ceron et al., 2024); and MOORE (Hendawy et al., 2024), which incorporates orthogonal submodules for each task. This comprehensive evaluation allows us to disentangle the relative contributions of sparsity, explicit resets, and architectural regularization to plasticity preservation and multi-task performance. We report the normalized interquantile mean (IQM) with shaded regions indicating 95% stratified bootstrap confidence intervals, calculated using the `rliable` library (Agarwal et al., 2021).

［#22］
Environment and Benchmarks We mostly consider the three multi-task MiniGrid (Chevalier-Boisvert et al., 2023) benchmarks proposed by Hendawy et al. (2024) – MT3, MT5, and MT7, with the exception being made for the results presented in Section 4.3, which use the MetaWorld MT10 benchmark (Yu et al., 2021). All environment details are outlined in Appendix B. We note that to ensure fair comparison across tasks with inherently different reward scales, for MiniGrid, raw episodic returns are normalized with respect to the maximum achievable reward in each environment (see Appendix B.3).

［#23］
Implementation and Training For MiniGrid, we use the Proximal Policy Optimization (PPO) algorithm (Schulman et al., 2017) via the `mushroom_rl` library (D'Eramo et al., 2021) and the code provided by Hendawy et al. (2024) for multi-task architectures. Performance is measured by the episodic return across all tasks within the respective benchmark. Tasks are sampled randomly with replacement at the beginning of each episode during training. For MetaWorld, we use the Multi-Task Multi-Headed Soft Actor-Critic (MTMH SAC) (Haarnoja et al., 2018; Yu et al., 2021) and track the mean success rate across all tasks. We outline full training details and hyperparameters in Appendix A.


［#24］
Sparse Methods To better characterize the role of sparsification in the MTRL setting, we began with a series of preliminary experiments comparing various sparsification strategies. Our goal was to identify methods that balance learning stability, generalization, and simplicity, while remaining compatible with the dynamic nature of multi-task settings. We evaluated several sparsification techniques, including: the Gradual Magnitude Pruning (GMP) schedule proposed by Zhu & Gupta (2017), Sparse Evolutionary Training (SET) (Mocanu et al., 2018), and Lottery Ticket Hypothesis (LTH) style rewinding (Frankle & Carbin, 2019). These initial experiments were conducted on the MT5 benchmark, selected as a practical compromise: it is more challenging than MT3, allowing us to meaningfully stress-test pruning strategies, yet significantly more computationally efficient than MT7, enabling extensive ablations at a reasonable cost. The results of this comparison are presented in Figure 1.

［#25］
Among the three tested approaches, both GMP and SET resulted in more stable learning dynamics and improved generalization performance. As shown in Figure 1, sparse models trained with GMP and SET not only surpass their dense counterparts but also outperform single-task base- lines. Conversely, LTH-based models fail to yield signif- icant improvements over single-task training, highlighting their limited capacity to adapt in multi-task settings. We note that these results are consistent with findings in single-task reinforcement learning (Graesser et al., 2022; Obando-Ceron et al., 2024), and further underscore the ad- vantages of sparsity mechanisms that adapt progressively throughout training. Given these insights, the remainder of our experimental study focuses on the two approaches that consistently demonstrated better performance: GMP and SET. The first, GMP, incrementally increases the net- work's sparsity level during training by gradually remov- ing low-magnitude weights over a predefined time window. This allows the network to adapt to the increasing sparsity and mitigates the risk of destabilizing learning dynamics. For further information about the pruning schedule, we refer the reader to Appendix C.4. The second, SET, takes inspiration from evolutionary algorithms and maintains a fixed overall sparsity throughout training by continuously rewiring the network's connectivity. Unlike gradual prun- ing, which increases sparsity over time, SET preserves a constant sparsity level but introduces dynamic plasticity through periodic topological updates. More details about the SET algorithm are presented in Appendix C.5. We note that both strategies are well aligned with the core objective of this work, which is to investigate the learning dynamics of sparse agents rather than to optimize inference-time performance. As both SET and GMP rely on unstructured pruning, they provide greater representational flexibility than structured approaches, making them particularly suitable for analyzing adaptation and interference in multi-task reinforcement learning (Hoefler et al., 2021).

［#26］
![](./images/1162088558115160127_1.jpg)

［#27］
Figure 1: Comparative performance of vari- ous sparsification strategies (GMP, SET, LTH) against a dense multi-task baseline and an aggre- gation of single-task agents trained individually on each task on the MT5 benchmark. Both GMP and SET outperform the dense multi-task base- line and the LTH-based models, while LTH-based models show limited improvement over single-task performance and struggle to adapt effectively in the multi-task setting.

［#28］
Plasticity Measures We monitor three metrics during training, interpreted as correlative indicators of plasticity based on recent surveys and analyses (Berariu et al., 2023; Lyle et al., 2023; Klein et al., 2024; Falzari & Sabatelli, 2025), namely dormant neuron percentage, effective rank, and the trace of the Fisher Information Matrix. The computation of these metrics is detailed in Appendix C. Our analysis focuses on observing consistent patterns between pruning interventions, changes in these metrics, and MTRL performance, rather than claiming direct causality.

## 4 Core Effects of Sparse Methods

［#29］
This section details our first set of empirical findings, beginning with the effect of sparsity on multi-task performance, followed by an analysis of its effects on plasticity indicators. All performance comparisons refer


［#29］
to the aggregated final outcomes presented in Table 1. Plasticity metric analyses are primarily illustrated using the MT5 benchmark data shown in Figure 2, Figure 4, and Figure 5, as trends were generally consistent across other benchmarks unless otherwise stated. Detailed learning curves and plasticity metrics for all benchmarks are available in Appendix F and Appendix G, respectively.

［#30］
Table 1: Final aggregate performance at epoch 200 across architectures and agent treatments on MT3, MT5, and MT7 benchmarks. Gold marks the best performance within the respective multi-task architecture and benchmark, while blue marks the second-best performance within each architecture and benchmark. Full learning curves illustrating training progression are available in Appendix F.

［#31］
<table>
<thead>
<tr>
<th rowspan="2">Agent Treatment</th>
<th colspan="2">MT3</th>
<th colspan="2">MT5</th>
<th colspan="2">MT7</th>
</tr>
<tr>
<th>IQM (↑)</th>
<th>95% CI</th>
<th>IQM (↑)</th>
<th>95% CI</th>
<th>IQM (↑)</th>
<th>95% CI</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7">Multi-Task PPO (MTPPO)</td>
</tr>
<tr>
<td>Dense</td>
<td>0.70</td>
<td>(0.60, 0.76)</td>
<td>0.65</td>
<td>(0.61, 0.71)</td>
<td>0.72</td>
<td>(0.69, 0.75)</td>
</tr>
<tr>
<td>Gradual Pruning</td>
<td>0.77</td>
<td>(0.73, 0.80)</td>
<td>0.81</td>
<td>(0.75, 0.86)</td>
<td>0.76</td>
<td>(0.70, 0.80)</td>
</tr>
<tr>
<td>SET</td>
<td>0.76</td>
<td>(0.74, 0.77)</td>
<td>0.80</td>
<td>(0.75, 0.84)</td>
<td>0.80</td>
<td>(0.77, 0.84)</td>
</tr>
<tr>
<td>ReDo</td>
<td>0.74</td>
<td>(0.68, 0.77)</td>
<td>0.83</td>
<td>(0.78, 0.84)</td>
<td>0.80</td>
<td>(0.76, 0.83)</td>
</tr>
<tr>
<td>Reset</td>
<td>0.70</td>
<td>(0.64, 0.72)</td>
<td>0.80</td>
<td>(0.74, 0.84)</td>
<td>0.80</td>
<td>(0.76, 0.83)</td>
</tr>
<tr>
<td>Weight Decay</td>
<td>0.74</td>
<td>(0.70, 0.78)</td>
<td>0.75</td>
<td>(0.66, 0.82)</td>
<td>0.74</td>
<td>(0.70, 0.77)</td>
</tr>
<tr>
<td>LayerNorm</td>
<td>0.28</td>
<td>(0.20, 0.37)</td>
<td>0.33</td>
<td>(0.25, 0.40)</td>
<td>0.38</td>
<td>(0.33, 0.45)</td>
</tr>
<tr>
<td colspan="7">Mixture of Experts (MoE)</td>
</tr>
<tr>
<td>Dense</td>
<td>0.74</td>
<td>(0.71, 0.76)</td>
<td>0.77</td>
<td>(0.70, 0.82)</td>
<td>0.80</td>
<td>(0.75, 0.84)</td>
</tr>
<tr>
<td>Gradual Pruning</td>
<td>0.77</td>
<td>(0.74, 0.79)</td>
<td>0.84</td>
<td>(0.78, 0.86)</td>
<td>0.87</td>
<td>(0.83, 0.88)</td>
</tr>
<tr>
<td>SET</td>
<td>0.76</td>
<td>(0.74, 0.78)</td>
<td>0.79</td>
<td>(0.72, 0.85)</td>
<td>0.82</td>
<td>(0.78, 0.85)</td>
</tr>
<tr>
<td>ReDo</td>
<td>0.77</td>
<td>(0.76, 0.80)</td>
<td>0.82</td>
<td>(0.81, 0.85)</td>
<td>0.85</td>
<td>(0.82, 0.88)</td>
</tr>
<tr>
<td>Reset</td>
<td>0.64</td>
<td>(0.54, 0.73)</td>
<td>0.78</td>
<td>(0.73, 0.83)</td>
<td>0.84</td>
<td>(0.80, 0.87)</td>
</tr>
<tr>
<td>Weight Decay</td>
<td>0.75</td>
<td>(0.71, 0.76)</td>
<td>0.78</td>
<td>(0.70, 0.85)</td>
<td>0.77</td>
<td>(0.71, 0.82)</td>
</tr>
<tr>
<td>LayerNorm</td>
<td>0.44</td>
<td>(0.38, 0.46)</td>
<td>0.34</td>
<td>(0.29, 0.40)</td>
<td>0.39</td>
<td>(0.32, 0.43)</td>
</tr>
<tr>
<td colspan="7">Mixture of Orthogonal Experts (MOORE)</td>
</tr>
<tr>
<td>Dense</td>
<td>0.78</td>
<td>(0.71, 0.80)</td>
<td>0.84</td>
<td>(0.81, 0.85)</td>
<td>0.87</td>
<td>(0.84, 0.88)</td>
</tr>
<tr>
<td>Gradual Pruning</td>
<td>0.80</td>
<td>(0.77, 0.81)</td>
<td>0.85</td>
<td>(0.80, 0.87)</td>
<td>0.88</td>
<td>(0.86, 0.89)</td>
</tr>
<tr>
<td>SET</td>
<td>0.69</td>
<td>(0.60, 0.74)</td>
<td>0.78</td>
<td>(0.72, 0.83)</td>
<td>0.82</td>
<td>(0.79, 0.85)</td>
</tr>
<tr>
<td>ReDo</td>
<td>0.72</td>
<td>(0.68, 0.75)</td>
<td>0.82</td>
<td>(0.78, 0.84)</td>
<td>0.85</td>
<td>(0.84, 0.87)</td>
</tr>
<tr>
<td>Reset</td>
<td>0.67</td>
<td>(0.62, 0.72)</td>
<td>0.75</td>
<td>(0.71, 0.79)</td>
<td>0.84</td>
<td>(0.80, 0.87)</td>
</tr>
<tr>
<td>WD</td>
<td>0.75</td>
<td>(0.73, 0.76)</td>
<td>0.82</td>
<td>(0.76, 0.87)</td>
<td>0.87</td>
<td>(0.84, 0.88)</td>
</tr>
<tr>
<td>LayerNorm</td>
<td>0.54</td>
<td>(0.49, 0.62)</td>
<td>0.60</td>
<td>(0.53, 0.65)</td>
<td>0.66</td>
<td>(0.61, 0.70)</td>
</tr>
</tbody>
</table>

## 4.1 Sparse Methods Improve Task Performance

［#32］
Our findings indicate that both GMP and SET generally lead to improvements in multi-task performance, an observation consistent with a significant body of research in supervised learning, where appropriately pruned sparse networks have been shown to match, outperform, and generalize better than their dense counterparts (Guo et al., 2019; Morcos et al., 2019; Sabatelli et al., 2020; Hoefler et al., 2021). However, in our multi-task settings, the extent of these benefits from pruning varies with the underlying agent architecture and desired sparsity level. For MTPPO and MoE architectures, both Gradual Pruning and SET consistently resulted in improved final aggregate returns compared to their respective dense baselines across all tested benchmarks (MT3, MT5, MT7), as shown in Table 1. This suggests that these common MTRL architectures frequently contain considerable overparameterization that sparse methods can effectively address, hinting at a direct link between sparse intervention and improved MTRL outcomes. In contrast, the impact of sparse methods on MOORE was more nuanced. While in general, the effect of GMP on MOORE on performance was close to that of the dense baseline, SET led to a slight decline across all benchmarks. While substantial gains were not observed for MOORE with sparsification, the ability to prune to high levels of sparsity (up to

［#33］
95%) without significant performance degradation still indicates that even sophisticated architectures can be overparameterized. Nonetheless, we note that very aggressive pruning (e.g., 99% sparsity with GMP) could lead to issues such as rank collapse or performance drops in MOORE (see Appendix H, Figure 23 and Figure 24).

### 4.2 Sparse Methods Mitigate Plasticity Loss

［#34］
![](./images/1162088558115160127_2.jpg)

［#35］
Figure 2: Evolution of plasticity indicators for Dense, Gradual Magnitude Pruning (GMP), and Sparse Evolutionary Training (SET) across different MTRL architectures (MTPPO, MoE, MOORE) on the MT5 benchmark. Subplots display Fisher Trace, Effective Rank, and percentages of Actor and Critic Dormant neurons, illustrating the distinct effects of each sparsification strategy compared to the dense baseline.

［#36］
The observed performance improvements, particularly within the MTPPO and MoE architectures, strongly correlate with the ability of sparsification methods to mitigate common indicators of plasticity loss, while displaying distinct learning dynamics for GMP and SET. Notably, the plasticity profiles (Figure 2) show that agents employing either GMP or SET generally exhibit more favorable plasticity metrics compared to their dense counterparts. Specifically, sparse agents typically maintain lower percentages of dormant neurons and a higher or more stable mean effective rank in their representations. Furthermore, the trace of the Fisher Information Matrix (FIM) in sparse agents typically stabilizes at lower values post-initial learning, suggesting convergence to less sensitive parameter configurations, in contrast to the often persistently high values in dense networks. While both GMP and SET contribute to these general improvements over dense networks, they induce individually different plasticity dynamics. SET, with its continuous rewiring, proved particularly effective at minimizing neuron dormancy to very low levels in both actor and critic components throughout training, while also maintaining a higher effective rank. In contrast, GMP's impact on dormancy was more pronounced in the actor network, with both actor and critic dormant percentages remaining higher than those under SET, though still an improvement over dense networks. The FIM trace also differed: GMP often displayed a characteristic peak-and-decline pattern, whereas SET maintained a low and stable FIM trace throughout training, suggesting continuous adaptation within a less volatile optimization regime. Collectively, these observations support the hypothesis that sparsification methods enhance the learning capability of MTRL agents, plausibly through the mitigation of processes associated with plasticity degradation in dense networks. Nevertheless, the influence of sparsification on MOORE's plasticity did not mirror the benefits seen in MTPPO and MoE agents, aligning with the more varied overall performance outcomes discussed

［#36］
above. For MOORE, SET did reduce neuron dormancy, and its Fisher Trace showed a slow growth and stabilization pattern. However, the effective rank for both SET and GMP remained similar to the dense baseline. GMP, in contrast to its effect in other architectures, sometimes even slightly increased dormancy compared to dense MOORE on certain metrics. Importantly, these specific plasticity modulations, such as SET's reduced dormancy in MOORE, generally did not translate into performance improvements for this architecture, with SET often resulting in a slight performance decline. This suggests that MOORE's inherent design, particularly its emphasis on representation orthogonalization (Hendawy et al., 2024), may interact with sparsification in various ways. Its sophisticated structure might be less responsive to the typical benefits derived from these plasticity changes, as it already exhibits relatively stable plasticity characteristics.

## 4.3 Generalization to Continuous Control

［#37］
To evaluate whether our plasticity-related findings in MiniGrid generalize to continuous control, we extended our analysis to the MetaWorld MT10 benchmark (Yu et al., 2021). Our experimental setup was guided by two insights from McLean et al. (2025): increasing the critic's capacity tends to yield greater benefits than increasing the actor's, making the critic the more capacity-sensitive component; and while overall plasticity loss (e.g., neuron dormancy) is relatively low in dense agents, it tends to be more pronounced in the actor than in the critic. These findings led us to hypothesize that pruning only the actor, while preserving the full capacity of the critic, could enhance performance by improving network efficiency without compromising representational power. We evaluated this hypothesis by comparing three conditions: a dense MTMH-SAC baseline, GMP applied to both actor and critic, and GMP applied to the actor only. The actor-only pruning approach achieved the highest final success rate at 81% (95% CI: 0.77, 0.83), outperforming both the dense baseline (73%; 95% CI: 0.67, 0.75) and the global pruning condition (75%; 95% CI: 0.73, 0.78); see Appendix F, Figure 13 for exact learning curves. This performance improvement was accompanied by a sustained reduction in actor neuron dormancy, as depicted in Figure 3, suggesting a more adaptive and efficient use of network capacity. Overall, these results extend our MiniGrid-based plasticity findings to the more complex MetaWorld benchmark and offer a complementary perspective to McLean et al. (2025): while they emphasize scaling the critic, we show that selectively pruning the actor can be equally beneficial. Together, these insights highlight the asymmetry in actor-critic dynamics and suggest that the benefits of sparsity are both role and context-dependent.

［#38］
![](./images/1162088558115160127_3.jpg)

［#39］
Figure 3: Reduction in actor neuron dormancy on MetaWorld MT10. Selective pruning of the actor network (GMP-Actor) leads to a sustained decrease in dormant neurons compared to both the dense MTMH-SAC baseline and the globally pruned model (GMP-Both).

## 5 Interactions with Alternative Mechanisms

［#40］
This section shifts focus to a comparative analysis between sparsification and other alternative strategies for plasticity and multi-task learning. We first examine pruning in relation to explicit interventions such as ReDo and Reset, which directly manipulate network parameters to counteract plasticity loss. We then consider implicit mechanisms such as standard regularization techniques (Weight Decay) and architectural choices (LayerNorm) that influence plasticity without explicit intervention. All comparisons are conducted under the same multi-task training setup and are summarized in Table 1. Additionally, we present a final ablation study exploring potential synergies of combining GMP with other optimization techniques (Weight Decay and PCGrad (Yu et al., 2020)).


### 5.1 Sparsification versus Explicit Plasticity-Inducing Mechanisms

［#41］
We compared GMP and SET against interventions that explicitly target symptoms of plasticity loss: ReDo (reinitializing dormant neurons) and Reset (layer reinitialization), using their best-performing configurations derived after hyperparameter tuning (see Figure 11 of Appendix E). In terms of final task performance (Table 1), sparsification methods generally achieved returns competitive with, and occasionally better than, ReDo or Reset, especially for MTPPO and MoE architectures. While statistical significance for outperformance was not always established due to overlapping confidence intervals, sparse methods consistently presented a strong alternative without directly targeting specific plasticity symptoms. For MOORE agents, performance differences between ReDo and the sparse approaches were minimal, while Reset introduced substantial variability. Examining the plasticity profiles (Figure 4), SET was particularly effective for MTPPO and MoE, often maintaining a lower percentage of actor dormant neurons than even ReDo and consistently achieving the highest effective rank. In contrast, within MOORE, ReDo was more effective in reducing dormancy, while SET's effective rank advantage was less apparent. ReDo's impact on the Fisher Trace and mean effective rank often mirrored that of the dense baseline, indicating it primarily addressed dormancy without broadly altering other representational characteristics. The Reset intervention, due to its periodic reinitializations, frequently induced abrupt shifts and instability in markers like the FIM and effective rank, especially post-reset, consistent with prior work (Falzari & Sabatelli, 2025). Performance-wise, Reset rarely outperformed sparse agents, whereas ReDo was more competitive; however, achieving a lower percentage of dormant neurons via ReDo did not always guarantee superior task performance (e.g., in MOORE), and SET sometimes achieved lower actor dormancy without this direct targeting.

［#42］
![](./images/1162088558115160127_4.jpg)

［#43］
Figure 4: Comparative plasticity dynamics of sparse methods (GMP, SET) versus explicit plasticity-inducing interventions (ReDo, Reset) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the MT5 benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.

### 5.2 Sparsification versus Implicit Plasticity-Inducing Mechanisms

［#44］
To further characterize the role of sparsification in fostering plasticity, we contrast its effects with more implicit plasticity-inducing mechanisms: weight regularization (Weight Decay, WD) and architectural normalization (LayerNorm). These techniques have been explored for mitigating plasticity loss by promoting parameter

［#45］
![](./images/1162088558115160127_5.jpg)

［#46］
Figure 5: Comparative plasticity dynamics of sparse methods (GMP, SET) versus regularization techniques (Weight Decay, LayerNorm) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the MT5 benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.

［#47］
stability or scale-invariant updates (Lyle et al., 2023). These experiments evaluate whether sparsification methods like GMP and SET offer distinct advantages over, or complementary benefits to, these common regularization approaches. In terms of task performance, both GMP and SET generally outperform agents trained with only Weight Decay or LayerNorm across all architectures and benchmarks. In Figure 5, Weight Decay exhibits plasticity dynamics remarkably similar to the dense baseline across all metrics and architectures. This suggests that while WD is a common regularizer, in these MTRL contexts, it does not substantially alter plasticity characteristics beyond a standard dense network, nor does it typically lead to performance surpassing well-configured sparse agents. LayerNorm, conversely, induces more pronounced changes to plasticity. Even though it can lead to very low levels of dormant neurons, this is accompanied by a severe and persistent reduction in the effective rank. This drop, visible across all architectures, signifies limited representational diversity, which also correlates with its lowered performance. We hypothesize that in the multi-task learning setting, LayerNorm, by normalizing activations across features within each layer and sample, might inadvertently introduce strong correlations in the gradients from different tasks or smooth out task-specific feature distinctions excessively. This could lead to a less expressive representation space, hindering the network's ability to learn diverse task solutions despite the apparent reduction in neuron dormancy. The low effective rank, coupled with often the atypically low Fisher Traces for a dense agent, likely contributes to LayerNorm's consistently poor task performance. In contrast, sparsification methods like GMP and SET generally achieve a better balance: they effectively mitigate dormancy (SET often being most effective) and maintain or improve effective rank compared to the dense baseline (especially GMP initially), without the severe representational collapse seen with LayerNorm. This suggests that sparsification offers a more nuanced approach to capacity control and plasticity preservation than these standard implicit regularization techniques in the studied MTRL scenarios, leading to superior overall learning outcomes.

### 5.3 Interaction with Optimizers

［#48］
We finally explored the potential synergies of combining GMP with other optimization techniques, specifically weight decay and PCGrad (Yu et al., 2020), a popular method designed to mitigate

［#48］
gradient interference in multi-task settings, on the MTPPO architecture using the MT5 benchmark. The performance results are shown in Figure 6 and plasticity dynamics in Figure 7. In terms of final task performance, GMP in isolation achieved the highest returns (Figure 6). While PCGrad alone improved performance over the dense baseline, its combination with GMP did not yield further gains beyond GMP alone. Interestingly, the plasticity profiles also reveal that GMP alone maintained the most favorable characteristics (Figure 7). The GMP+WD combination showed similar plasticity dynamics to GMP alone, although with slightly worse values in some metrics, correlating with its slightly lower performance. This lack of synergy with Weight Decay might be anticipated, as WD encourages smaller weight magnitudes overall, potentially increasing the pool of weights that magnitude-based pruning would target, which could lead to a less discerning pruning process or even premature removal of weights that might have otherwise become important. PCGrad, when applied to a dense network, did not demonstrably improve plasticity indicators such as Fisher Trace or actor dormancy compared to the dense baseline, despite its performance uplift. This suggests that the primary benefits of GMP in this context may stem from its inherent regularization effects and capacity optimization, which are not necessarily enhanced by, or may even be slightly counteracted by, the addition of these particular optimizers.

［#49］
![](./images/1162088558115160127_6.jpg)

［#50］
Figure 6: Performance comparison for Gradual Magnitude Pruning (GMP) interactions with Weight Decay (WD) and PCGrad on the MTPPO architecture with the MT5 benchmark.

［#51］
![](./images/1162088558115160127_7.jpg)

［#52］
Figure 7: Comparative plasticity dynamics for Gradual Magnitude Pruning (GMP) and its interactions with Weight Decay (GMP+WD) and PCGrad (GMP+PCGrad), contrasted with PCGrad on a dense network and a dense baseline. All experiments are on the MTPPO architecture with the MT5 benchmark. Subplots display (from left to right): Fisher Trace, Effective Rank, Actor Dormant %, and Critic Dormant %.

## 6 Considerations of Sparse Methods

［#53］
While this study is primarily empirical, the observed benefits of sparsification methods in MTRL can be interpreted through several established concepts from the sparsity, optimization, and multi-task learning literature. However, we also want to acknowledge their limitations and considerations for practical application.

［#54］
Optimization, Sparsity, and Generalization The iterative removal (GMP) or rewiring of connections (SET) effectively guides the network towards sparse solutions. This mechanism can generally be viewed as $L_0$ regularization, encouraging sparsity by penalizing the number of non-zero parameters (Louizos et al., 2018) and reducing the model's degrees of freedom. While this might confine optimization to lower-dimensional subspaces (Gao & Jojic, 2016; Hoefler et al., 2021), such sparse solutions are often associated with "flatter" minima in the loss landscape (Peste, 2023; Shah et al., 2024). Flatter minima are highly desirable due to lower sensitivity to parameter perturbations (Foret et al., 2021; Lee et al., 2025b), widely believed to result in better generalization and robustness under distribution shifts (Hochreiter & Schmidhuber, 1997; Jiang et al., 2019; Kaddour et al., 2023; Li et al., 2024), the primary drivers of plasticity loss. The convergence to such local flat minima is often indicated by specific dynamics in the curvature of the loss for instance, the

［#54］
maximal Hessian eigenvalue typically grows, peaks, and then declines during training (Fort & Ganguli, 2019). Our empirical results for MTPPO and MoE, where the Fisher Trace (a proxy for curvature (Lewandowski et al., 2024b)) exhibited a peak-and-decline pattern with GMP (Figure 2), align with convergence to such flatter minima. SET, with its continuous rewiring, maintained a low, stable Fisher trace, suggesting a robust optimization process.

［#55］
This aligns with findings in supervised learning where sparse networks are recognized for reduced overfitting and better generalization than dense counterparts (Gopalakrishnan et al., 2018; Cosentino et al., 2019; Guo et al., 2019; Liu et al., 2019; Liu, 2020). The iterative nature of GMP and the dynamic regrowing of SET are generally thought to help models evade suboptimal local minima (Jin et al., 2016; Gale et al., 2019; Hoefler et al., 2021; Jin et al., 2022; Graesser et al., 2022), a principle consistent with our findings. Furthermore, the success of pruning in MTRL models with shared backbones and task-specific heads mirrors similar effectiveness in multi-task supervised learning Xiang et al. (2024). Nonetheless, an excessive reduction in degrees of freedom via aggressive pruning can hinder the satisfaction of specific architectural demands, such as maintaining expert orthogonality in MOORE (see Appendix H, Figure 23 and Figure 24), especially if capacity becomes overly constrained.

［#56］
Limitations and Practical Considerations Despite their benefits, these sparsification techniques have considerations. GMP, while conceptually simple, typically operates on dense weight matrices internally during training, applying masks to simulate sparsity. This means it does not inherently reduce the memory footprint or computational cost during training compared to dense models; true benefits often require specialized hardware or software for sparse operations at inference. While inference can be efficient, the training phase still bears the overhead of the original dense model, with the additional overhead of pruning at specified timesteps. SET, on the other hand, can maintain true sparsity throughout training and inference if implemented with sparse data structures. However, current widely available implementations are often optimized for fully connected layers, and extending their dynamic rewiring efficiently to convolutional or recurrent architectures can be more complex. Additionally, the random nature of SET's regrowth phase, while promoting exploration, might not always lead to the most optimal connectivity patterns without more guided heuristics. Both methods also require careful tuning of their own hyperparameters to achieve optimal results. While potentially less sensitive than some explicit plasticity interventions, this still constitutes a tuning effort. The architecture-dependent nature of the benefits, as shown by our experiments with MOORE, also indicates that these are not one-size-fits-all solutions.

## 7 Conclusion

［#57］
In this work, we examined dynamic sparsification, specifically Gradual Magnitude Pruning (GMP) and Sparse Evolutionary Training (SET), as a means to mitigate plasticity loss and improve performance in multi-task reinforcement learning (MTRL). Our results show that both methods can enhance adaptability and generalization across several architectures, with consistent gains observed in MTPPO and MoE agents. Similar benefits were also found in Multi-Headed SAC agents evaluated on the MetaWorld MT10 benchmark, suggesting that the effectiveness of sparsification extends to continuous control tasks. While performance on MOORE was more variable, likely due to its built-in mechanisms for managing interference, our findings highlight the importance of aligning sparsification strategies with architectural design. In addition to performance improvements, sparsity can offer benefits such as reduced hyperparameter sensitivity, computational efficiency, and implicit regularization through structured parameter removal. These results support the view that general-purpose mechanisms that shape learning dynamics rather than task-specific interventions can yield robust benefits in MTRL. As future work, we plan on investigating the theoretical underpinnings and potential interpretability advantages of sparse MTRL models.

## References




































































































# A Training Details and Hyperparameters

［#58］
This appendix details the hyperparameters used for the experimental evaluations presented in this study. For MiniGrid, we report the IQM and CIs of the episodic return over 30 random seeds, whereas for MetaWorld, we report the IQM and CIs of the mean success rate across 10 seeds. Table 2 provides a list covering the general experimental settings, architecture of the used networks, and specific hyperparameters used for MoE and MOORE. All architectures are multi-headed, with task-specific heads. Hyperparameters were largely adopted from Hendawy et al. (2024), with modifications in the number of training epochs, number of evaluation episodes, and evaluation frequency. Table 3 outlines the training details and parameters for MetaWorld MT10. The full implementation is available at https://github.com/atodorov284/sparsity_driven_plasticity.

# B Environment Details

［#59］
This appendix provides details on the MiniGrid (Chevalier-Boisvert et al., 2023) environments used in our multi-task benchmarks. We use standard environments from the MiniGrid suite, which are designed to test various capabilities such as navigation, memory, and problem-solving in partially observable grid-world settings with sparse reward. For environmental details on MetaWorld, we refer the reader to Yu et al. (2021).

## B.1 Composition

［#60］
Our experiments use three multi-task benchmarks – MT3, MT5, and MT7, as proposed by Hendawy et al. (2024), composed as follows:

［#61］
- MT3: LavaGapS7-v0 + RedBlueDoors-6x6-v0 + MemoryS11-v0
- MT5: MT3 + DoorKey-6x6-v0 + DistShift1-v0
- MT7: MT5 + SimpleCrossingS9N2 + MultiRoom-N2-S4

## B.2 Descriptions

［#62］
Below are descriptions for each unique environment used in the benchmarks, adapted from Chevalier-Boisvert et al. (2023). In all environments S specifies the size of the map SxS.

［#63］
- DoorKey-6x6-v0: The agent must pick up a key, navigate to a locked door, and open it to reach a goal square.
- DistShift1-v0: The agent starts in the top-left corner and must reach the goal, which is in the top-right corner, but has to avoid stepping into lava on its way. Stepping into lava terminates the episode.
- RedBlueDoors-6x6-v0: The agent is in a room with two doors, one red and one blue. The agent has to open the red door and then open the blue door, in that order.
- MemoryS11-v0: The agent starts in a small room where it sees an object. It then has to go through a narrow hallway, which ends in a split. At each end of the split, there is an object, one of which is the same as the object in the starting room. The agent has to remember the initial object and go to the matching object at the split.
- SimpleCrossingS9N2-v0: The agent has to reach the green goal square on the other corner of the room while avoiding walls. Walls run across the room either horizontally or vertically, and have N crossing points which can be safely used; the path to the goal is guaranteed to exist.
- MultiRoom-N2-S4-v0: This environment has a series of connected rooms with doors that must be opened to get to the next room. The final room has the green goal square that the agent must get to. N specifies the number of rooms.


［#64］
Table 2: Core experimental setup, agent architecture, and algorithm hyperparameters on MiniGrid. The choice for hyperparameters is largely borrowed from Hendawy et al. (2024), while following their exact training configuration (except number of evaluation episodes).

［#65］
<table>
<thead>
  <tr>
    <th>Hyperparameter</th>
    <th>Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><i>General:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Number of environments</td>
    <td>[3, 5, 7]</td>
  </tr>
  <tr>
    <td>Steps per environment</td>
    <td>1 step per environment</td>
  </tr>
  <tr>
    <td>Number of epochs</td>
    <td>200</td>
  </tr>
  <tr>
    <td>Steps per epoch</td>
    <td>2000</td>
  </tr>
  <tr>
    <td>Total number of timesteps</td>
    <td>400000</td>
  </tr>
  <tr>
    <td>Train frequency</td>
    <td>2000 timesteps</td>
  </tr>
  <tr>
    <td>Evaluation episodes</td>
    <td>25 per task</td>
  </tr>
  <tr>
    <td>Evaluation frequency</td>
    <td>10000 timesteps</td>
  </tr>
  <tr>
    <td><i>Shared Feature Extractor:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Type</td>
    <td>Conv2D</td>
  </tr>
  <tr>
    <td>Channels per Layer</td>
    <td>[16, 32, 64]</td>
  </tr>
  <tr>
    <td>Kernel Size</td>
    <td>[(2,2), (2,2), (2,2)]</td>
  </tr>
  <tr>
    <td>Activations</td>
    <td>[ReLU, ReLU, Tanh]</td>
  </tr>
  <tr>
    <td><i>PPO:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Optimizer</td>
    <td>Adam (Kingma & Ba, 2017)</td>
  </tr>
  <tr>
    <td>Critic Loss</td>
    <td>MSE</td>
  </tr>
  <tr>
    <td>Actor Learning Rate</td>
    <td>$1 \times 10^{-3}$</td>
  </tr>
  <tr>
    <td>Critic Learning Rate</td>
    <td>$1 \times 10^{-3}$</td>
  </tr>
  <tr>
    <td>Critic Network Hidden Size</td>
    <td>128</td>
  </tr>
  <tr>
    <td>Actor Network Hidden Size</td>
    <td>128</td>
  </tr>
  <tr>
    <td>Number of Linear Layers</td>
    <td>$2 \times |\mathcal{T}|$ (number of tasks)</td>
  </tr>
  <tr>
    <td>Number of Output Units</td>
    <td>$|\mathcal{A}|$ for actor, 1 for critic</td>
  </tr>
  <tr>
    <td>Output Activations</td>
    <td>[Tanh, Linear]</td>
  </tr>
  <tr>
    <td>GAE $\lambda$</td>
    <td>0.95</td>
  </tr>
  <tr>
    <td>Entropy Term Coefficient</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td>Clipping $\varepsilon$</td>
    <td>0.2</td>
  </tr>
  <tr>
    <td>Epochs for Policy</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Epochs for Critic</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Batch Size for Policy</td>
    <td>256</td>
  </tr>
  <tr>
    <td>Batch Size for Critic</td>
    <td>2000</td>
  </tr>
  <tr>
    <td>Discount Factor ($\gamma$)</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td><i>Task Encoder (for MoE/MOORE):</i></td>
    <td></td>
  </tr>
  <tr>
    <td>$k$ Experts</td>
    <td>[2, 3, 4]</td>
  </tr>
  <tr>
    <td>Encoder Linear Layers</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Encoder Output Units</td>
    <td>$k$ (number of experts)</td>
  </tr>
  <tr>
    <td>Encoder Use Bias</td>
    <td>False</td>
  </tr>
  <tr>
    <td>Encoder Activation</td>
    <td>Linear</td>
  </tr>
</tbody>
</table>

［#66］
Table 3: The hyperparameters and training setup used for MTMH SAC on MetaWorld MT10.

［#67］
<table>
<thead>
  <tr>
    <th>Hyperparameter</th>
    <th>Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><i>General Training:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Total Timesteps</td>
    <td>20000000</td>
  </tr>
  <tr>
    <td>Batch Size</td>
    <td>1280</td>
  </tr>
  <tr>
    <td>Replay Buffer Size</td>
    <td>1000000</td>
  </tr>
  <tr>
    <td>Warmstart Steps</td>
    <td>40000</td>
  </tr>
  <tr>
    <td>Evaluation Frequency</td>
    <td>200000 steps</td>
  </tr>
  <tr>
    <td>Number of Epochs</td>
    <td>200</td>
  </tr>
  <tr>
    <td>Number of Updates</td>
    <td>2000000</td>
  </tr>
  <tr>
    <td>Number of Tasks</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Evaluation Episodes</td>
    <td>50 per task</td>
  </tr>
  <tr>
    <td>Max Episode Steps</td>
    <td>500</td>
  </tr>
  <tr>
    <td><i>SAC:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Discount Factor ($\gamma$)</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td>Target Smoothing Coeff. ($\tau$)</td>
    <td>0.005</td>
  </tr>
  <tr>
    <td>Number of Critics</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Initial Temperature ($\alpha$)</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>Target Q-Value Clip</td>
    <td>5000</td>
  </tr>
  <tr>
    <td><i>Actor and Critic</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Optimizer</td>
    <td>Adam</td>
  </tr>
  <tr>
    <td>Layer Type</td>
    <td>Linear</td>
  </tr>
  <tr>
    <td>Learning Rate</td>
    <td>$3 \times 10^{-4}$</td>
  </tr>
  <tr>
    <td>Max Gradient Norm</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>Network Depth</td>
    <td>3</td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>400</td>
  </tr>
  <tr>
    <td>Activation</td>
    <td>ReLU</td>
  </tr>
  <tr>
    <td>Log-Std Bounds</td>
    <td>$[-20, 2]$</td>
  </tr>
  <tr>
    <td><i>Temperature Optimizer:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Optimizer</td>
    <td>Adam</td>
  </tr>
  <tr>
    <td>Learning Rate</td>
    <td>$3 \times 10^{-4}$</td>
  </tr>
  <tr>
    <td>Max Gradient Norm</td>
    <td>None</td>
  </tr>
  <tr>
    <td><i>Gradual Magnitude Pruning:</i></td>
    <td></td>
  </tr>
  <tr>
    <td>Desired Sparsity $\rho_F$</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>Pruning Frequency $f_p$</td>
    <td>5000 timesteps</td>
  </tr>
  <tr>
    <td>Pruning start interval $t_{\text{start}}$</td>
    <td>$0.05 \times$ number of timesteps</td>
  </tr>
  <tr>
    <td>Pruning end interval $t_{\text{end}}$</td>
    <td>$0.80 \times$ number of timesteps</td>
  </tr>
  <tr>
    <td>Sparsity $\rho_t$ at timestep $t$</td>
    <td>$\rho_F \left[1 - \left(1 - \frac{t-t_{\text{start}}}{t_{\text{end}}-t_{\text{start}}}\right)^3\right]$</td>
  </tr>
</tbody>
</table>

［#68］
- LavaGapS7-v0: The agent has to reach the green goal square at the opposite corner of the room, and must pass through a narrow gap in a vertical strip of deadly lava. Touching the lava terminates the episode with a zero reward.

## B.3 Reward Normalization

［#69］
To ensure fair comparison across tasks with inherently different reward scales, raw episodic returns are normalized with respect to the maximum achievable reward in each environment. The standard MiniGrid reward for successful task completion is calculated as
［#69］
$$1 - 0.9 \times (\text{steps\_taken}/\text{max\_episode\_steps}),$$
［#69］
while failure results in a score of 0. We further normalize this score by performing a Min-Max scaling with respect to the maximum performance obtainable in each environment. We use the default maximum timesteps of each environment and estimate how many steps an optimal agent can solve the environment. This normalization procedure scales the performance such that a score of 1.0 represents achieving the optimal (shortest path) solution, facilitating comparisons of learning efficacy across environments with varying complexities and step horizons and addressing reward scales. Table 4 presents the optimal steps, maximum allowed steps, and the maximum achievable reward used for the score normalization of each environment.

［#70］
Table 4: Environment-specific parameters for reward normalization. This table lists the optimal number of steps to solve each task, the maximum default permissible steps per episode, and the resulting maximum achievable raw reward score (used as the max score for normalization).

［#71］
<table>
<thead>
<tr>
<th>Environment Name</th>
<th>Optimal Steps</th>
<th>Max Steps</th>
<th>Achievable Reward</th>
</tr>
</thead>
<tbody>
<tr>
<td>DoorKey-6x6-v0</td>
<td>11</td>
<td>360</td>
<td>0.9725</td>
</tr>
<tr>
<td>DistShift1-v0</td>
<td>11</td>
<td>252</td>
<td>0.9607</td>
</tr>
<tr>
<td>RedBlueDoors-6x6-v0</td>
<td>8</td>
<td>720</td>
<td>0.9900</td>
</tr>
<tr>
<td>LavaGapS7-v0</td>
<td>8</td>
<td>196</td>
<td>0.9633</td>
</tr>
<tr>
<td>MemoryS11-v0</td>
<td>15</td>
<td>605</td>
<td>0.9777</td>
</tr>
<tr>
<td>SimpleCrossingS9N2-v0</td>
<td>15</td>
<td>324</td>
<td>0.9583</td>
</tr>
<tr>
<td>MultiRoom-N2-S4-v0</td>
<td>5</td>
<td>40</td>
<td>0.8875</td>
</tr>
</tbody>
</table>

# C Implementation Details

［#72］
This appendix details the methodology and interpretation of the plasticity metrics, pruning schedule, and the sparse evolutionary training used in this work. The plasticity measures serve as correlative indicators of an agent's learning capacity and adaptability. The computation of activations and gradients for these metrics relies on sampling from a plasticity replay buffer of training observations to approximate expected values via sample means. Hyperparameters specific to these calculations are detailed in Table 5.

## C.1 Neuron Dormancy

［#73］
We adapt the dormant neuron formalization from Sokar et al. (2023). A neuron's activity is assessed relative to other (non-masked) neurons in the same layer. Given an input distribution $D$ (approximated by the plasticity replay buffer) and an activation $h_i^l(x)$ of a neuron $i$ in layer $l$ with $H^l$ neurons under input $x \in D$, the normalized activation is
［#73］
$$
s_i^l = \frac{\mathbb{E}_{x \in D} |h_i^l(x)|}{\frac{1}{H^l} \sum_{k=1}^{H^l} \mathbb{E}_{x \in D} |h_k^l(x)|}.
$$

［#74］
Neuron $i$ is called $\tau$-dormant for some threshold $\tau > 0$ if $s_i^l \leq \tau$. If $H_\tau^l$ denotes the number of dormant neurons per layer, then the dormancy ratio $\beta_\tau$ is the ratio of dormant neurons and all neurons across all


［#74］
layers in the network $L_{\text{all}}$ except the final $L_{\text{out}}$

［#74］
$$
\beta_{\tau}=\frac{\sum_{l \in L_{\text{all}} \backslash\left\{L_{\text{out}}\right\}} H_{\tau}^{l}}{\sum_{l \in L_{\text{all}} \backslash\left\{L_{\text{out}}\right\}} H^{l}}.
$$

［#75］
A high percentage of dormant neurons suggests significant underutilization of the network's capacity, potentially hindering its ability to learn complex functions or adapt to new data, a key aspect of plasticity. For the ReDo method, dormant neurons were reinitialized at specific time intervals $f_{d}$ and a threshold $\tau$, determined through the hyperparameter sweeps in Appendix E

## C.2 Trace of the Fisher Information Matrix

［#76］
The Fisher Information Matrix (FIM) $F$ quantifies the sensitivity of a model's output (e.g., the policy) to changes in the parameters $\theta$ (Klein et al., 2024; Falzari & Sabatelli, 2025; Ven, 2025). For a policy $\pi$, its Fisher trace is given by

［#76］
$$
\operatorname{Tr}(F)=\mathbb{E}_{s, a \sim \pi}\left[\left\|\nabla_{\theta} \log \pi(a \mid s)\right\|_{2}^{2}\right].
$$

［#77］
The trace of the FIM can be viewed as a measure of the policy's sensitivity to parameter perturbations. A very high or persistently increasing trace might indicate that the policy is in a "sharp" region of the loss landscape, making it brittle to small changes and potentially indicative of overfitting or optimization instability. Conversely, a lower, stabilized trace, as observed in our pruned agents (see Section 4.2), can suggest convergence to "flatter" minima, implying a more robust policy that is less sensitive to parameter variations and more capable of sustained learning or adaptation.

## C.3 Effective Rank

［#78］
For a feature matrix $\Phi$ (e.g., a shared feature extractor) with $d$ singular values $\sigma_{i}$ sorted descendingly, the effective rank at tolerance $\delta$ is

［#78］
$$
\operatorname{srank}_{\delta}(\Phi)=\min _{k}\left\{\frac{\sum_{i=1}^{k} \sigma_{i}}{\sum_{i=1}^{d} \sigma_{i}} \geq 1-\delta\right\}.
$$

［#79］
The effective rank measures the dimensionality of the space spanned by the features. A low effective rank suggests a representation collapse, where learned features are highly correlated and less diverse, limiting a network's ability to represent various information. Conversely, a high effective rank implies a richer, more diverse set of feature representations, implying a greater capacity to learn and distinguish between inputs.

## C.4 Gradual Magnitude Pruning (GMP)

［#80］
We implement the pruning schedule proposed by Zhu & Gupta (2017), which progressively increases network sparsity during training. At regular pruning intervals (defined by a pruning frequency $f_{p}$), connections (weights) with the smallest absolute magnitudes are masked (set to zero). This process continues until a target sparsity level $\rho_{t}$ is achieved for the current training step $t$. The sparsity level $\rho_{t}$ follows a cubic growth schedule, gradually increasing from an initial sparsity at $t_{\text{start}}$ to a final target sparsity $\rho_{F}$ at $t_{\text{end}}$:

［#80］
$$
\rho_{t}=
\begin{cases}
0 & \text { if } t<t_{\text{start}}, \\
\rho_{F}\left[1-\left(1-\frac{t-t_{\text{start}}}{t_{\text{end}}-t_{\text{start}}}\right)^{3}\right] & \text { if } t_{\text{start}} \leq t \leq t_{\text{end}}, \\
\rho_{F} & \text { if } t>t_{\text{end}}.
\end{cases}
$$

［#81］
This schedule allows the network to adapt to increasing levels of sparsity rather than undergoing abrupt structural changes. The specific values for $\rho_{F}$, pruning frequency, $t_{\text{start}}$, and $t_{\text{end}}$ used in our experiments were determined through ablation studies (see Appendix D).


### C.5 Sparse Evolutionary Training (SET)

［#82］
The Sparse Evolutionary Training (SET) mechanism, inspired by Mocanu et al. (2018), maintains a constant overall network sparsity $\rho = 1 - (\|\mathbf{W}\|_0/N_{\text{total}})$ throughout training, where $\|\mathbf{W}\|_0$ is the total number of non-zero weights and $N_{\text{total}}$ is the total number of parameters in the sparsified layers. At predefined evolution intervals, a fraction $\varepsilon$ of the existing connections with the smallest absolute magnitudes $|w_{ij}|$ are pruned. To preserve the sparsity level $s$, an equivalent number of new connections, $N_{\text{new}} = \varepsilon \cdot \|\mathbf{W}\|_0^{\text{current}}$, are simultaneously regrown. These new connections are typically introduced randomly at locations within the network that currently have zero weight, allowing exploration of novel sparse topologies. The initial sparse connectivity for SET is established using an Erdős-Rényi-Kernel (ERK) scheme, controlled by a parameter $\zeta$. SET was applied to all linear layers in our models, with $\varepsilon$, $\zeta$, and evolution frequency determined via ablations (Appendix D), while the fixed sparsity was kept at 95% for consistency with the GMP sparsity.

［#83］
Table 5: Configuration details for gradual magnitude pruning, sparse evolutionary training, and the alternative plasticity-enhancing methods (ReDo, Reset, and Weight Decay) evaluated.

［#84］
<table>
<thead>
  <tr>
    <th>Hyperparameter</th>
    <th>Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="2"><i>Gradual Magnitude Pruning:</i></td>
  </tr>
  <tr>
    <td>Desired Sparsity $\rho_F$</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>Pruning Frequency $f_p$</td>
    <td>500 timesteps</td>
  </tr>
  <tr>
    <td>Pruning start interval $t_{\text{start}}$</td>
    <td>$0.05 \times$ number of timesteps</td>
  </tr>
  <tr>
    <td>Pruning end interval $t_{\text{end}}$</td>
    <td>$0.80 \times$ number of timesteps</td>
  </tr>
  <tr>
    <td>Sparsity $\rho_t$ at timestep $t$</td>
    <td>$\rho_F \left[1 - \left(1 - \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}\right)^3\right]$</td>
  </tr>
  <tr>
    <td>Prune Bias</td>
    <td>False</td>
  </tr>
  <tr>
    <td>Pruned Layers</td>
    <td>[Conv2D, Linear]</td>
  </tr>
  <tr>
    <td colspan="2"><i>Sparse Evolutionary Training:</i></td>
  </tr>
  <tr>
    <td>Sparsity</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>$\varepsilon$ density</td>
    <td>11</td>
  </tr>
  <tr>
    <td>Sparsity Distribution $\zeta$</td>
    <td>0.3</td>
  </tr>
  <tr>
    <td>Evolution Frequency</td>
    <td>2000 timesteps</td>
  </tr>
  <tr>
    <td colspan="2"><i>Plasticity:</i></td>
  </tr>
  <tr>
    <td>Plasticity Buffer Max Size</td>
    <td>100000</td>
  </tr>
  <tr>
    <td>ReDo (Sokar et al., 2023) Frequency $f_d$</td>
    <td>5000 timesteps</td>
  </tr>
  <tr>
    <td>Dormant Neuron Threshold $\tau$</td>
    <td>0.001</td>
  </tr>
  <tr>
    <td>Dormant Activation Batch Size</td>
    <td>1024</td>
  </tr>
  <tr>
    <td>Fisher Trace Batch Size</td>
    <td>1024</td>
  </tr>
  <tr>
    <td>Effective Rank Batch Size</td>
    <td>1024</td>
  </tr>
  <tr>
    <td>Effective Rank Target</td>
    <td>Shared Feature Extractor</td>
  </tr>
  <tr>
    <td>Reset Frequency $f_r$</td>
    <td>100000</td>
  </tr>
  <tr>
    <td>Number of Resets $m$</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Reset Target Layers</td>
    <td>Output</td>
  </tr>
  <tr>
    <td>Weight Decay Coefficient $\lambda$</td>
    <td>$1 \times 10^{-6}$</td>
  </tr>
</tbody>
</table>

## D Sparse Methods Hyperparameters

［#85］
This appendix details the hyperparameter selection for GMP, with tuning experiments conducted on MTPPO with the MT5 benchmark. Figure 8 illustrates GMP ablations: the left subplot shows that initiating pruning early (e.g., at 5% of training) and concluding by 80% is favorable for the schedule window $[t_{\text{start}}, t_{\text{end}}]$; the center subplot indicates stable performance across moderate pruning frequencies $f_p$; and the right subplot explores various final target sparsities $\rho$. Figure 9 presents SET tuning: the left subplot suggests moderate evolution connection $\varepsilon$ (e.g., 11-15) perform well; the center explores sparsity distribution parameters $\zeta$ and the right shows less frequent evolution can be beneficial. These ablations informed the hyperparameter choices used in our main experimental evaluations. All tuning experiments were run with the original training configuration, outlined in Table 2 for 30 seeds and 200 epochs. For visual purposes, we omit the confidence intervals of the less successful runs to avoid cluttering and only include them for the best hyperparameter run.

［#86］
![](./images/1162088558115160127_8.jpg)

［#87］
Figure 8: Hyperparameter ablation for Gradual Magnitude Pruning (GMP) on MTPPO with the MT5 benchmark. (Left) Varying pruning schedule window $[t_{\text{start}}, t_{\text{end}}]$. (Center) Different pruning frequencies $f_p$. (Right) Various final target sparsity levels $\rho$.

［#88］
![](./images/1162088558115160127_9.jpg)

［#89］
Figure 9: Hyperparameter ablation for Sparse Evolutionary Training (SET) on MTPPO with the MT5 benchmark. (Left) Varying evolution connection percentage $\varepsilon$. (Center) Different sparsity distribution parameters $\zeta$. (Right) Different evolution frequencies (episodes).


# E Comparison Methods Hyperparameters

［#90］
This appendix outlines hyperparameter considerations for the comparison methods evaluated in this work. All tuning experiments were primarily conducted on the MTPPO architecture using the MT5 benchmark to establish robust configurations. All tuning experiments were run with the original training configuration, outlined in Table 2 for 30 seeds and 200 epochs. For visual purposes, we omit the confidence intervals of the less successful runs to avoid cluttering and only include them for the best hyperparameter run.

［#91］
For ReDo, tuning (Figure 10) highlighted the dormancy threshold $\tau$ as the most critical hyperparameter. Lower values, specifically $\tau = 0.001$ and $\tau = 0.0001$, demonstrated significantly better performance compared to higher thresholds or omitting the threshold entirely. The ReDo application frequency $f_d$ exhibited less sensitivity, with a moderate frequency (e.g., $f_d = 5000$ steps) performing well.

［#92］
For the Reset mechanism, we investigated the impact of the reset frequency $f_r$ (how often resets occur) and the maximum number of times $m$ specific layers are reset throughout training, as illustrated in Figure 11. Our ablations indicated that a reset frequency of $f_r = 100k$ timesteps with a maximum of $m = 2$ resets per targeted layer (black line in both subplots) often provided a good balance between promoting plasticity and avoiding excessive training instability.

［#93］
The Weight Decay (WD) coefficient $\lambda$ was selected from a standard range. As shown in Figure 12, very small coefficients (e.g., $\lambda = 10^{-6}$, black line) resulted overall in the best performance.

［#94］
![](./images/1162088558115160127_10.jpg)

［#95］
Figure 10: Hyperparameter ablation for ReDo on MTPPO with the MT5 benchmark. (Left) Varying dormancy threshold $\tau$. (Right) Different ReDo application frequencies $f_d$. The black line ($\tau = 0.001, f_d = 5000$) shows the best performance.


［#96］
![](./images/1162088558115160127_11.jpg)

［#97］
Figure 11: Hyperparameter ablations for the Reset mechanism on MTPPO with the MT5 benchmark. (Left) Varying reset frequency $f_r$ (with $m=2$). (Right) Varying maximum number of resets $m$. The black line ($f_r=100k, m=2$) indicates the best performance.

［#98］
![](./images/1162088558115160127_12.jpg)

［#99］
Figure 12: Hyperparameter ablation for Weight Decay (WD) on MTPPO with the MT5 benchmark, showing performance for different decay coefficients $\lambda$. The black line ($\lambda=10^{-6}$) shows the best performance.

## F Detailed Learning Curves

［#100］
This appendix provides the complete learning curves for various agent configurations, complementing the aggregated final performance data presented in Table 1. These plots illustrate the training progression across all evaluated MTRL architectures (MTPPO, MoE, MOORE) on the MT3, MT5, and MT7 benchmarks. Additionally, Figure 13 presents the results obtained on the continuous control MetaWorld MT10 benchmark that complement the findings described in Section 4.1.

［#101］
Figure 14 displays learning curves comparing our primary sparsification methods (Gradual Magnitude Pruning, GMP, at 95% sparsity, and Sparse Evolutionary Training, SET, at 95% sparsity) against a dense baseline and explicit plasticity-inducing interventions (ReDo and Reset).

［#102］
Figure 15 similarly presents learning curves, in this case contrasting the same sparsification methods (GMP 95% and SET 95%) against a dense baseline and common implicit regularization techniques (Weight Decay - WD, and LayerNorm).

［#103］
In all subplots within Figure 14 and Figure 15, the horizontal dashed line indicates the aggregated performance of single-task PPO (ST PPO) agents. Each ST PPO agent was trained separately on a single environment from the respective benchmark for the full 400,000 timesteps (the same total duration as the multi-task agents) across 30 runs. Consequently, this ST PPO performance should be viewed as a potentially **near-maximal** reference point from a single-task perspective, as multi-task agents faced the more challenging scenario of learning all tasks within a benchmark concurrently using the same total number of timesteps.

［#104］
![](./images/1162088558115160127_13.jpg)

［#105］
Figure 13: Final success rates on MetaWorld MT10 for different pruning configurations. Selective pruning of the actor network (GMP-actor) outperforms both the dense MTMH-SAC baseline and the globally pruned model (GMP-all), achieving a final success rate of 81%. This supports the hypothesis that actor-only pruning enhances performance by improving efficiency while preserving critical representational capacity in the critic.

［#106］
![](./images/1162088558115160127_14.jpg)

［#107］
Figure 14: Learning curves (Normalized IQM) comparing sparsification methods (GMP 95%, SET 95%) with explicit plasticity-inducing interventions (ReDo, Reset) and a dense baseline. Results are shown for MTPPO, MoE, and MOORE architectures across MT3, MT5, and MT7 benchmarks. The dashed line represents single-task PPO performance.

［#108］
![](./images/1162088558115160127_15.jpg)

［#109］
Figure 15: Learning curves (Normalized IQM) comparing sparsification methods (GMP 95%, SET 95%) with implicit regularization techniques (Weight Decay, LayerNorm) and a dense baseline. Results are shown for MTPPO, MoE, and MOORE architectures across MT3, MT5, and MT7 benchmarks. The dashed line represents single-task PPO performance.

## G Detailed Plasticity Metrics

［#110］
This appendix provides a comprehensive view of the plasticity metric evolutions across different benchmarks, complementing the primary analysis presented in Section 4.2 (which predominantly features results from the MT5 benchmark, also shown here for completeness as Figure 17 and Figure 20). The following figures illustrate the dynamics of Fisher Trace, Effective Rank, Actor Dormant percentage, and Critic Dormant percentage for all evaluated MTRL architectures (MTPPO, MoE, MOORE).

［#111］
Figures 16, 17, and 18 compare the effects of sparsification methods (GMP and SET) against explicit plasticity-inducing interventions (ReDo and Reset) and a dense baseline, on the MT3, MT5, and MT7 benchmarks, respectively. While specific magnitudes vary, general trends such as SET's strong effect on reducing dormancy and GMP's characteristic Fisher Trace dynamics are often observable across benchmarks, though interactions with architecture (especially MOORE) can modulate these effects.

［#112］
Similarly, Figures 19, 20, and 21 present a comparison of the same sparsification methods (GMP and SET) against implicit regularization techniques (Weight Decay - WD, and LayerNorm) and a dense baseline, for the MT3, MT5, and MT7 benchmarks, respectively.

［#113］
Figure 22 presents the plasticity profiles on MetaWorld MT10, using the MTMH SAC architecture. The Critic MER and Critic Dormant neuron percentages are nearly identical across all three configurations, with dormancy already close to zero even for the dense baseline, leaving little room for improvement. The actor metrics, however, show clear distinctions. Selectively pruning the actor resulted in maintaining the lowest percentage of actor dormant neurons and an increased mean effective rank compared to both the dense baseline and the globally pruned agent. The Fisher Trace remained highly volatile for all methods and showed no discernible pattern.

［#114］
![](./images/1162088558115160127_16.jpg)

［#115］
Figure 16: Comparative plasticity dynamics of sparse methods (GMP, SET) versus explicit plasticity-inducing interventions (ReDo, Reset) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the **MT3** benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.

［#116］
![](./images/1162088558115160127_17.jpg)

［#117］
Figure 17: Comparative plasticity dynamics of sparse methods (GMP, SET) versus explicit plasticity-inducing interventions (ReDo, Reset) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the **MT5** benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.

［#118］
![](./images/1162088558115160127_18.jpg)

［#119］
Figure 18: Comparative plasticity dynamics of sparse methods (GMP, SET) versus explicit plasticity-inducing interventions (ReDo, Reset) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the **MT7** benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.


［#120］
![](./images/1162088558115160127_19.jpg)

［#121］
Figure 19: Comparative plasticity dynamics of sparse methods (GMP, SET) versus regularization techniques (Weight Decay, LayerNorm) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the **MT3** benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.

［#122］
![](./images/1162088558115160127_20.jpg)

［#123］
Figure 20: Comparative plasticity dynamics of sparse methods (GMP, SET) versus regularization techniques (Weight Decay, LayerNorm) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the **MT5** benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.


［#124］
![](./images/1162088558115160127_21.jpg)

［#125］
Figure 21: Comparative plasticity dynamics of sparse methods (GMP, SET) versus regularization techniques (Weight Decay, LayerNorm) and a dense baseline, across MTPPO, MoE, and MOORE architectures on the MT7 benchmark. Metrics include Fisher Trace, Effective Rank, and percentage of Actor and Critic Dormant Neurons.

［#126］
![](./images/1162088558115160127_22.jpg)

［#127］
Figure 22: Plasticity dynamics on MetaWorld MT10. Selective pruning of the actor (GMP-Actor) leads to a decrease in dormant neurons and an increase in the mean effective rank compared to both the dense MTMH-SAC baseline and the globally pruned model (GMP-Both). In contrast, the critic metrics and Fisher Trace show minimal or no clear patterns.


## H Gradual Magnitude Pruning Performance Across Sparsity Levels

［#128］
This appendix presents learning curves for Gradual Magnitude Pruning (GMP) across various target sparsity levels (Dense, 80%, 95%, 99%) for all architectures and benchmarks (Figure 23). These results show that while 80% and 95% sparsity generally yield strong performance, often matching or exceeding dense baselines especially for MTPPO and MoE, the optimal sparsity level is architecture and benchmark-specific. Notably, for the MOORE architecture, 99% sparsity leads to performance degradation later in training (visible in Figure 23, bottom row). This performance drop correlates with a significant representational rank collapse, as illustrated by the sharp decline in effective rank for MOORE (Figure 24), indicating a loss of representational diversity under extreme pruning in this specific architecture.

［#129］
![](./images/1162088558115160127_23.jpg)

［#130］
Figure 23: Normalized aggregate returns for agents under different GMP sparsity levels (Dense, 80%, 95%, 99%) across the MT3, MT5, and MT7 benchmarks for MTPPO (top row), MoE (middle row), and MOORE (bottom row) architectures. The dashed line represents single-task PPO performance.


［#131］
![](./images/1162088558115160127_24.jpg)

［#132］
Figure 24: Plasticity metrics for MOORE on the MT5 benchmark under different GMP sparsity levels, illustrating the rank collapse at 99% sparsity, characterized by a sudden increase in the Fisher Trace and neuron dormancy, and a sharp drop in the effective rank.