# The Impact of Quantization and Pruning on Deep Reinforcement Learning Models

［#1］
Heng Lu¹, Mehdi Alemi²,³, and Reza Rawassizadeh¹

［#2］
¹Department of Computer Science at Metropolitan College, Boston University, Boston, MA, USA
²Department of Orthopaedic Surgery, Harvard Medical School, Boston, MA, USA.
³Training Services, MathWorks, Natick, MA, USA.

## Abstract

［#3］
Deep reinforcement learning (DRL) has achieved remarkable success across various domains, such as video games, robotics, and, recently, large language models. However, the computational costs and memory requirements of DRL models often limit their deployment in resource-constrained environments. The challenge underscores the urgent need to explore neural network compression methods to make RDL models more practical and broadly applicable. Our study investigates the impact of two prominent compression methods, *quantization* and *pruning* on DRL models. We examine how these techniques influence four performance factors: average return, memory, inference time, and battery utilization across various DRL algorithms and environments. Despite the decrease in model size, we identify that these compression techniques generally do not improve the energy efficiency of DRL models, but the model size decreases. We provide insights into the trade-offs between model compression and DRL performance, offering guidelines for deploying efficient DRL models in resource-constrained settings.

## Introduction and Background

［#4］
Reinforcement learning has been applied in many fields, including robotics, video games, and recently Reinforcement Learning with Human Feedback (RLHF) [1, 2, 3, 4, 5], has become common in large language models. RLHF methods mitigate biases inherent in language models themselves [4, 6, 7]. Reinforcement learning models that address real-world problems predominantly utilize continuous models based on neural network architecture, known as Deep Reinforcement Learning (DRL).

［#5］
DRL methods typically involve a world model, agents interacting with the world, and a reward function that evaluates the effectiveness of actions based on the agent's policy towards predefined objectives [8]. Depending on whether the algorithm learns a specific world model, DRL algorithms are categorized into model-based DRL algorithms and model-free DRL algorithms [9]. Model-free DRL methods generally fall into three main categories: deep Q-learning methods [10, 11, 12], policy gradient methods [13, 14, 15], and actor-critic methods [16, 17, 18, 19]. Unlike model-based algorithms, model-free approaches circumvent model bias and offer greater generalizability, which contributes to their popularity in RLHF applications [4, 20].

［#6］
Neural networks, which are the backbone of DRL methods, are associated with high computational costs and, therefore, resource intensive. Recently, there has been a significant increase in the energy and water consumption of artificial intelligence (AI) data centers ¹²³⁴. This trend has led to several research studies [21, 22, 23] investigating the resource costs of recent advances in AI. Reducing the energy utilization of DRL has become a crucial need.

［#7］
Additionally, many systems that benefit from reinforcement learning operate on battery-powered devices, such as extended reality devices and mobile robots. As the size of these devices decreases, their computational capabilities also diminish [24]. One common approach to reducing the computational costs of neural network models is compressing them via pruning and quantization[25, 26]. Network compression methods have been widely applied in computer vision [27, 28] and large language models [29, 30, 31] to improve inference times and reduce memory requirement with minimal compromise to accuracy. Compression enables advanced DRL models to be deployed in robots with low latency and high energy efficiency under constrained resources. Despite its promise, neural network compression in DRL models has received less research attention [32, 33] compared to other fields, such as computer vision and natural language processing.

---
［#1］
¹https://www.theatlantic.com/technology/archive/2024/03/ai-water-climate-microsoft/677602
［#2］
²https://www.oregonlive.com/silicon-forest/2022/12/googles-water-use-is-soaring-in-the-dalles-records-show-with-two-more-data-centers-to-come.html
［#2］
³https://www.bloomberg.com/news/articles/2023-07-26/thames-water-considers-restricting-flow-to-london-data-centers
［#8］
⁴https://www.washingtonpost.com/business/2024/03/07/ai-data-centers-power

［#9］
Several general approaches have been proposed for neural network compression method quantization and pruning [34, 35]. As for quantization, DRL researchers might be more familiar with vector quantization, which aims to discretize continuous space into a discrete vector set to reduce dimensions [36, 37, 38, 39]. However, in this work, we specifically apply neural network quantization, which focuses on converting float32 format weights and biases into smaller-scale numbers such as int8 or 4-bits rather than vector quantization. Pruning methods, on the other hand, involve removing neurons deemed least important based on criteria such as weight or activation value [40, 41, 42, 43, 44].

［#10］
In addition to conventional compression approaches [35, 25, 26], there are promising DRL-specific compression approaches [32, 36, 40, 43]. AQuaDem [39] discretizes the action space and learns a discrete set of actions for each state $s$ using behavior cloning from expert demonstrations [45]. The adaptive state aggregation algorithm[36] adaptively discretizes the state space based on Bregman divergence, enabling distinct partitions of the state space. Another group of methods focuses on scalar quantization that reduces the numeric precision of values [46, 40, 33, 32]. NNC-DRL[46] accelerates the DRL training process by speeding up prediction based on GA3C[47] and employs policy distillation to compress the behavior policy network prediction with minimal performance degradation. FIXAR [33] proposes quantization-aware training in fixed points to reduce model size without significant accuracy loss. ActorQ [32] introduces an int8 quantized policy block for rollouts within a traditional distributed RL training loop. PoPS[41] accelerates model speed by initially training a sparse network from a large-scale teacher network through iterative policy pruning, then compacting it into a dense network with minimal performance loss. UVNQ[40] integrates sparse variational dropout[48] with quantization, adjusting the dropout rate to enhance quantization awareness. Dynamic Structured Pruning[43] enhances DRL training by applying a neuron-importance group sparse regularizer and dynamically pruning insignificant neurons based on a threshold. Double Sparse Deep Reinforcement Learning [44] uses multilayer sparse-coding structural network with a nonconvex log regularizer to enforce sparsity while maintaining performance.

［#11］
In this work, we apply common neural network compression methods, including common quantization and pruning, to five popular deep reinforcement learning models (TRPO[14], PPO[15], DDPG[17], TD3[18], and SAC[19]). We then measure the performance of these algorithms post-compression using metrics such as average return, inference time, and energy usage. In particular, we apply $L_1$ and $L_2$ pruning techniques to these models. For quantization, we utilize int8 quantization and apply (i) post-training dynamic quantization, (ii) post-training static quantization, and (iii) quantization aware training across the listed models.

［#12］
To our knowledge, this study represents the first comprehensive evaluation of the effects of pruning and quantization across a range of deep reinforcement learning models. Our experiments and findings offer valuable insights to researchers and developers, assisting them in making informed decisions when choosing between quantization or pruning methods for DRL models. Another prevalent approach for compressing neural network is knowledge distillation [49], but due to its model or application-specific nature (e.g., image classification), we did not include knowledge distillation in our experimental setup.

## 2 Methods
［#13］
We have applied two types of neural network compression techniques —pruning and quantization— across five prominent DRL models: TRPO[14], PPO[15], DDPG[17], TD3[18], and SAC[19]. This section outlines our quantization methods, followed by our pruning methods.

### 2.1 Quantization
［#14］
We applied linear quantization across all models, where the relationship between the original input $r$ and its quantized version $q$ is defined as $r = S(q+Z)$. Here, $Z$ represents the zero point in the quantization space, and the scaling factor $S$ maps floating-point numbers to the quantization space. For Post-Training Dynamic Quantization (PTDQ) and Post-Training Static Quantization (PTSQ), we computed $S$ and $Z$ for activations exclusively. In PTSQ, first, baseline models go through a calibration process to compute these quantization parameters and then the models make inferences based on the fixed quantization parameters. In PTDQ, the quantization parameters are computed dynamically. In Quantization-Aware Training (QAT), baseline models are pseudo-quantized during training, meaning computations are conducted in floating-point precision but rounded to integer values to simulate quantization. Subsequently, the original models are converted into quantized versions, and the quantization parameters are stabilized.

### 2.2 Pruning
［#15］
Neural network pruning typically involves removing neurons within layers, and dependencies can exist where pruning in one layer affects subsequent related layers. The DepGraph approach we employed [50], addresses these dependencies by grouping layers based on their inter-dependencies rather than manually resolving dependencies.

［#16］
Conceptually, one might consider constructing a grouping matrix $G \in R^{L \times L}$, where $G_{ij}=1$ signifies a dependency between layer $i$ and layer $j$. However, due to the complexity arising from non-local relations, $G$ can not be easily constructed. Thus, dependency graph $D$ is proposed, which only contains the local dependency between adjacent layers and from which the grouping matrix can be reduced. These dependencies are categorized into two types: inter-layer dependencies, where the output of one layer $i$ connects to the input of another layer $j$, and intra-layer dependencies, such as within BatchNorm layers, where inputs and outputs share the same pruning scheme.

［#17］
After constructing the dependency graph and determining grouping parameters based on this graph, we utilized a norm-based importance score. However, directly summing importance scores across different layers can lead to meaningless results and potential divergence. Therefore, for a parameter $w$ in group $g$ with $K$ prune-able dimensions, a regularization term $R(g,k)$ is used in sparse training to select the optimal input variables, $R(g,k) = \sum_{k=1}^{K} \gamma_k \cdot I_{g,k}$, where $I_{g,k} = \sum_{w \in g} ||w[k]||_2^2$ is the importance for dimension $k$ in $L_2$ pruning and $\gamma_k = 2^{\alpha(I_g^{max}-I_{g,k})/(I_g^{max}-I_g^{min})}$.

## 3 Experiments
### 3.1 Experimental Settings

［#18］
Our experiments are structured into two main components: quantization and pruning of DRL algorithms. We evaluated the performance of TRPO[14], PPO[15], DDPG[17], TD3[18], and SAC[19] across five Gymnasium[51] (formerly OpenAI Gym[52]) Mujoco environments including: HalfCheetah, HumanoidStandup, Ant, Humanoid, and Hopper. These models are trained using Gymnasium (formerly OpenAI Gym) environments.

［#19］
To ensure consistency among our reported results, each experiment has been repeated at least 10 times in the same configuration.

［#20］
**Quantization and Pruning Libraries:** The implementations of quantization and pruning in neural network libraries are not as mature as other functionalities. For instance, in pyTorch pruning does not remove neurons but merely masks them. To ensure the reliability of our experiments, we evaluated various quantization and pruning libraries and selected those that offer the highest accuracy and resource efficiency.

［#21］
Therefore, to implement pruning, we explored PyTorch$^5$ and Torch-pruning. In our experiment, Torch-pruning$^6$, integrated with DepGraph [50], performed exceptionally well, and thus, we utilized it for pruning purposes. Regarding quantization, we experimented with Pytroch, TensorFlow, and ONNX Runtime$^7$. Ultimately, we chose PyTorch for QAT, and ONNX Runtime for PTDQ and PTSQ.

［#22］
**Hardware Settings:** Our hardware infrastructure included two NVidia RTX 4090 GPUs with 24GB of VRAM, 256GB of RAM, and an Intel Core i9 CPU running at 3.30 GHz. The operating system was Ubuntu 20.04 LTS, and we used CUDA Version 12.0 for GPU operations.

### 3.2 Quantization

［#23］
To implement quantization, we experimented with three approaches: PTDQ, PTSQ and QAT. Quantization-aware training (QAT) involved initially training quantized models with an equivalent dataset size as the baseline models, followed by exporting them into ONNX runtime for comparative analysis.

#### 3.2.1 Average Return

［#24］
The impact of quantization on average return is reported in Table 1. The table underscores the variability of quantization outcomes across different environments and DRL models. For instance, QAT demonstrates its highest efficacy in HumanoidStandup environments, resulting in improved average returns across models except for PPO. The SAC algorithm generally benefits more from QAT, except in the Hopper environment, where its effectiveness is limited. Overall, PTDQ exhibits superior performance, while PTSQ consistently shows the lowest results. The observed performance discrepancies may stem from distribution shifts between data used for optimal path calculations and that utilized during the calibration phase, which are challenging to rectify due to the stochastic nature of the environment.

---

［#21］
$^5$https://pyTorch.org
［#21］
$^6$https://github.com/VainF/Torch-Pruning
［#21］
$^7$https://onnxruntime.ai

［#25］
<table>
<tbody>
<tr>
<td>
</td>
<td>
</td>
<td>
Baseline
</td>
<td>
PTDQ
</td>
<td>
PTSQ
</td>
<td>
QAT
</td>
</tr>
<tr>
<td rowspan="5">
TRPO
</td>
<td>
HalfCheetah
</td>
<td>
978.48
</td>
<td>
1004.65
</td>
<td>
909.52
</td>
<td>
287.94
</td>
</tr>
<tr>
<td>
HumanoidStandup
</td>
<td>
39444.51
</td>
<td>
37002.02
</td>
<td>
35779.53
</td>
<td>
52252.48
</td>
</tr>
<tr>
<td>
Ant
</td>
<td>
851.33
</td>
<td>
799.46
</td>
<td>
1055.39
</td>
<td>
970.59
</td>
</tr>
<tr>
<td>
Humanoid
</td>
<td>
74.49
</td>
<td>
75.09
</td>
<td>
74.98
</td>
<td>
178.36
</td>
</tr>
<tr>
<td>
Hopper
</td>
<td>
164.06
</td>
<td>
163.84
</td>
<td>
162.93
</td>
<td>
7.49
</td>
</tr>
<tr>
<td rowspan="5">
PPO
</td>
<td>
HalfCheetah
</td>
<td>
1542.98
</td>
<td>
1508.82
</td>
<td>
1482.4
</td>
<td>
432.21
</td>
</tr>
<tr>
<td>
HumanoidStandup
</td>
<td>
117138.8
</td>
<td>
126509.58
</td>
<td>
128155.96
</td>
<td>
28270.98
</td>
</tr>
<tr>
<td>
Ant
</td>
<td>
1335.55
</td>
<td>
1493.93
</td>
<td>
1528.36
</td>
<td>
963.21
</td>
</tr>
<tr>
<td>
Humanoid
</td>
<td>
453.56
</td>
<td>
430.5
</td>
<td>
495.0
</td>
<td>
295.75
</td>
</tr>
<tr>
<td>
Hopper
</td>
<td>
8.33
</td>
<td>
9.41
</td>
<td>
19.28
</td>
<td>
92.32
</td>
</tr>
<tr>
<td rowspan="5">
DDPG
</td>
<td>
HalfCheetah
</td>
<td>
4475.62
</td>
<td>
4656.76
</td>
<td>
3801.11
</td>
<td>
937.53
</td>
</tr>
<tr>
<td>
HumanoidStandup
</td>
<td>
82747.99
</td>
<td>
82747.99
</td>
<td>
87346.75
</td>
<td>
115325.59
</td>
</tr>
<tr>
<td>
Ant
</td>
<td>
589.46
</td>
<td>
630.37
</td>
<td>
896.44
</td>
<td>
540.7
</td>
</tr>
<tr>
<td>
Humanoid
</td>
<td>
1544.76
</td>
<td>
314.51
</td>
<td>
433.15
</td>
<td>
411.92
</td>
</tr>
<tr>
<td>
Hopper
</td>
<td>
1419.69
</td>
<td>
1260.39
</td>
<td>
349.31
</td>
<td>
996.61
</td>
</tr>
<tr>
<td rowspan="5">
TD3
</td>
<td>
HalfCheetah
</td>
<td>
8333.37
</td>
<td>
5204.59
</td>
<td>
3915.44
</td>
<td>
4169.07
</td>
</tr>
<tr>
<td>
HumanoidStandup
</td>
<td>
77140.94
</td>
<td>
77140.94
</td>
<td>
78492.56
</td>
<td>
82211.28
</td>
</tr>
<tr>
<td>
Ant
</td>
<td>
3423.51
</td>
<td>
2789.51
</td>
<td>
2728.76
</td>
<td>
1833.74
</td>
</tr>
<tr>
<td>
Humanoid
</td>
<td>
5035.79
</td>
<td>
441.34
</td>
<td>
287.35
</td>
<td>
80.67
</td>
</tr>
<tr>
<td>
Hopper
</td>
<td>
3596.54
</td>
<td>
3532.45
</td>
<td>
2842.55
</td>
<td>
1826.21
</td>
</tr>
<tr>
<td rowspan="5">
SAC
</td>
<td>
HalfCheetah
</td>
<td>
10460.06
</td>
<td>
3104.33
</td>
<td>
1805.41
</td>
<td>
6163.8
</td>
</tr>
<tr>
<td>
HumanoidStandup
</td>
<td>
151015.38
</td>
<td>
109714.88
</td>
<td>
83898.23
</td>
<td>
151213.82
</td>
</tr>
<tr>
<td>
Ant
</td>
<td>
4021.96
</td>
<td>
2474.88
</td>
<td>
1144.59
</td>
<td>
3119.85
</td>
</tr>
<tr>
<td>
Humanoid
</td>
<td>
4287.29
</td>
<td>
-84.51
</td>
<td>
23.4
</td>
<td>
311.68
</td>
</tr>
<tr>
<td>
Hopper
</td>
<td>
2539.05
</td>
<td>
3621.34
</td>
<td>
2525.01
</td>
<td>
2998.79
</td>
</tr>
</tbody>
</table>

［#26］
Table 1: Average returns of quantization for TRPO, PPO, DDPG, TD3, and SAC. The best quantized version for each DRL model on the specific environments is shown in bold.

### 3.2.2 Resource Utilization

［#27］
To assess the impact of quantization on resource utilization, we conducted measurements and comparisons of memory usage, inference time, and energy consumption between baseline models and their quantized counterparts. Figure 1 illustrates the differences observed in inference time and energy usage between baseline and quantized models.

［#28］
![](./images/1229506077859512414_1.jpg)

［#29］
Figure 1: Inference time (in seconds), energy usage (in Joules) and memory utilization (in MegaByte) of quantization models.

### 3.3 Pruning

［#30］
To implement pruning, we utilized the torch-pruning package $^8$ for all our experiments. Each baseline model underwent $L_1$ and $L_2$ pruning, with various pruning percentages ranging from 5% to 70%. In particular, experimented pruning percentages are as follows: {5%,10%,15%,20%,25%,30%,35%,40%,45%,50%,55%,60%,65%,70%}.

［#31］
<table>
  <thead>
    <tr>
      <th colspan="2"></th>
      <th>Baseline</th>
      <th>Pruned</th>
      <th>$L_1$ Pruning Percentage</th>
      <th>$L_2$ Pruning Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">TRPO</td>
      <td>HalfCheetah</td>
      <td>1003.37</td>
      <td>905.94</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>HumanoidStandup</td>
      <td>35281.42</td>
      <td>42314.62</td>
      <td>0.55</td>
      <td>0.6</td>
    </tr>
    <tr>
      <td>Ant</td>
      <td>768.97</td>
      <td>979.42</td>
      <td>0.7</td>
      <td>0.7</td>
    </tr>
    <tr>
      <td>Humanoid</td>
      <td>74.49</td>
      <td>79.81</td>
      <td>0.25</td>
      <td>0.35</td>
    </tr>
    <tr>
      <td>Hopper</td>
      <td>164.06</td>
      <td>163.33</td>
      <td>0.15</td>
      <td>0.45</td>
    </tr>
    <tr>
      <td rowspan="5">PPO</td>
      <td>HalfCheetah</td>
      <td>1529.18</td>
      <td>1453.9</td>
      <td>0.1</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>HumanoidStandup</td>
      <td>128722.71</td>
      <td>117239.65</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>Ant</td>
      <td>1563.54</td>
      <td>263.37</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>Humanoid</td>
      <td>474.86</td>
      <td>417.99</td>
      <td>0.1</td>
      <td>0.15</td>
    </tr>
    <tr>
      <td>Hopper</td>
      <td>21.06</td>
      <td>7.6</td>
      <td>0.3</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td rowspan="5">DDPG</td>
      <td>HalfCheetah</td>
      <td>5104.06</td>
      <td>4026.46</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>HumanoidStandup</td>
      <td>82747.99</td>
      <td>83513.05</td>
      <td>0.7</td>
      <td>0.7</td>
    </tr>
    <tr>
      <td>Ant</td>
      <td>935.72</td>
      <td>761.55</td>
      <td>0.1</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>Humanoid</td>
      <td>1659.59</td>
      <td>1059.68</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>Hopper</td>
      <td>1574.66</td>
      <td>1423.7</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td rowspan="5">TD3</td>
      <td>HalfCheetah</td>
      <td>8298.95</td>
      <td>6535.42</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>HumanoidStandup</td>
      <td>77140.94</td>
      <td>97061.37</td>
      <td>0.7</td>
      <td>0.7</td>
    </tr>
    <tr>
      <td>Ant</td>
      <td>3381.72</td>
      <td>2564.85</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>Humanoid</td>
      <td>5040.01</td>
      <td>5046.25</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>Hopper</td>
      <td>3593.0</td>
      <td>3589.47</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td rowspan="5">SAC</td>
      <td>HalfCheetah</td>
      <td>10467.97</td>
      <td>10531.88</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>HumanoidStandup</td>
      <td>136900.13</td>
      <td>137574.71</td>
      <td>0.7</td>
      <td>0.7</td>
    </tr>
    <tr>
      <td>Ant</td>
      <td>3499.52</td>
      <td>282.28</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>Humanoid</td>
      <td>4251.2</td>
      <td>3549.46</td>
      <td>0.25</td>
      <td>0.35</td>
    </tr>
    <tr>
      <td>Hopper</td>
      <td>2549.96</td>
      <td>2412.92</td>
      <td>0.05</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>

［#32］
Table 2: Average returns of pruning results for TRPO, PPO, DDPG, TD3, and SAC

［#33］
The optimal pruning method for each baseline model was determined based on earning at least 90% average return of the corresponding baseline model while achieving the highest possible pruning percentage. The results of pruning experiments are presented in Table 2 and summarized comprehensively in Table 3. In Figure 2 and Figure 3, we present the effect of $L_1$ and $L_2$ pruning on the inference speed, energy usage, and memory usage. In these figures, we scaled the data according to the baseline.

［#34］
<table>
  <thead>
    <tr>
      <th></th>
      <th>TRPO</th>
      <th>PPO</th>
      <th>DDPG</th>
      <th>TD3</th>
      <th>SAC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HalfCheetah</td>
      <td>$L_1$ 5%</td>
      <td>$L_1$ 10%</td>
      <td>$L_2$ 5%</td>
      <td>$L_2$ 5%</td>
      <td>$L_1$ 5%</td>
    </tr>
    <tr>
      <td>HumanoidStandup</td>
      <td>$L_2$ 55%</td>
      <td>$L_1$ 5%</td>
      <td>$L_1$ 70%</td>
      <td>$L_1$ 70%</td>
      <td>$L_2$ 70%</td>
    </tr>
    <tr>
      <td>Ant</td>
      <td>$L_2$ 70%</td>
      <td>$L2$ 5%</td>
      <td>$L_1$ 10%</td>
      <td>$L_2$ 5%</td>
      <td>$L_2$ 25%</td>
    </tr>
    <tr>
      <td>Humanoid</td>
      <td>$L_2$ 30%</td>
      <td>$L_1$ 5%</td>
      <td>$L_2$ 5%</td>
      <td>$L_1$ 10%</td>
      <td>$L_2$ 25%</td>
    </tr>
    <tr>
      <td>Hopper</td>
      <td>$L_2$ 40%</td>
      <td>$L_2$ 30%</td>
      <td>$L_1$ 5%</td>
      <td>$L_2$ 5%</td>
      <td>$L_2$ 20%</td>
    </tr>
  </tbody>
</table>

［#35］
Table 3: Best pruning method for each environment and each model.

［#36］
$^8$https://github.com/VainF/Torch-Pruning

［#37］
![](./images/1229506077859512414_2.jpg)

［#38］
Figure 2: Inference time, energy usage and RAM of $L_1$ models, scaled by baseline models

［#39］
![](./images/1229506077859512414_3.jpg)

［#40］
Figure 3: Inference time (in seconds), energy usage (in Joules) and memory utilization (in Megabytes) of $L_2$ models, scaled by baseline models.

## 4 Discussions and Findings

［#41］
In this work, we studied two pruning approaches and three quantization approaches on five platforms (HalfCheetah-v4, HumanoidStandup-v4, Ant-v4, Hopper-v4, and Humanoid-v4) used for experimenting reinforcement learning methods and five common DRL methods (TRPO, PPO, DDPG, TD3). To our knowledge, this is the largest study performed on compressing DRL methods, and we listed our findings in this section. These findings could be used as a guideline for further studies that try to compress DRL methods.

［#42］
Pruning and quantization do not improve the energy efficiency and memory usage of DRL models. While pruning and quantization reduce model size (see Table 3), they do not necessarily enhance the energy efficiency of DRL models due to the maintained or increased average return. Energy consumption tends to decrease only when there is a significant drop in average return, prompting the agent to terminate early and requiring less computation.

［#43］
Despite reducing model size, quantization does not improve memory usage, and pruning yields only a negligible 1% decrease in memory usage. Results in Figure 1 present no changes in memory utilization in any platforms while applying quantization. Even PTDQ and PTSQ cause more memory utilization than the baseline method. This might be due to the overhead of the quantization library, and the way it is implemented is not optimized.

［#44］
$L_2$ pruning is favored over $L_1$ pruning for most of DRL models. Table 2-3 illustrates that the optimal pruning method varies based on the DRL algorithm and environmental complexity. Most environments, except for those trained with SAC on HalfCheetah, allow for substantial pruning without a notable decline in average return, while PPO models exhibit lower pruning thresholds. In instances where $L_1$ pruning outperforms $L_2$, the average return values remain closely aligned. Generally, a 10% reduction in DRL model size through $L_2$ pruning is beneficial, although exceptions include PPO models applied to HalfCheetah environments.

［#45］
PTDQ emerges as the superior quantization method for DRL algorithms, whereas PTSQ is not recommended. As shown in Table 1, 40% of our quantized models benefit from PTDQ, 36% from QAT, and only 24% from PTSQ. Our findings reveal that post-training dynamic quantization statistically outperforms the other methods, while post-training static quantization performs the worst, likely due to distribution shifts between the calibration data and the randomness that existed in RL environments.

［#46］
The Lottery ticket hypothesis [53] does not hold for DRL models. The Lottery Ticket Hypothesis (LTH) in the context of neural networks suggests that within a large, randomly initialized network, there exists a smaller sub-network, typically around 10-20% of the original size, that, when trained in isolation, can achieve performance comparable to the original large network. This idea has significant implications for model quantization and pruning, two techniques used to reduce the size and computational requirements of neural networks. However, based on the results demonstrated in Table 2 demonstrate significant performance drops in most models after 50% pruning, contradicting the hypothesis's assertion that original network performance can persist even when pruned to less than 10%-20% of its original size. In particular, around 40% of the models don't survive after more than 5% pruning, but 80% of the models don't survive after 50%.

［#47］
Our work has two limitations. First, by focusing on classical Mujoco environments with continuous action spaces, our work excludes discrete action spaces, which are common in video games or some decision-making scenarios. However, in these situations, task-specific methods might be employed for a satisfying performance, which adds additional complexities, and we might explore it in our future work. Moreover, we are limited to six simulated environments, this leaves an important aspect of real-world applicability unexplored. An ideal scenario is to experiment with this compression approach on a robot or drone in a real-world task and measure the differences in their performance.

## 5 Conclusion

［#48］
In this paper, we examined the effect of quantization methods and pruning methods on deep reinforcement learning algorithms. While the effect depended on the specific DRL algorithm used and the environment in which the agent is trained, the results shared some common patterns. Quantization converts the floating point with 32 bits into an integer with 8 bits model and effectively shrinks the model size while maintaining acceptable performance. We found that PTDQ models generally had the best average return, while PTSQ models might suffer from distribution shifts and had poorer results. DepGraph pruned baseline models by constructing a dependency graph, and we experimented with $L_1$ and $L_2$ pruning. Experiments outlined that $L_2$ pruning was preferred for DRL algorithms on continuous action spaces, and in general, models benefited from 10% $L_2$ pruning with some exceptions. However, while pruning actually removed some neurons, it did not always result in inference speedup or energy saving.

## References





















































