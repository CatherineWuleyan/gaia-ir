［#1］
Published as a conference paper at ICLR 2023

# ON THE SOFT-SUBNETWORK FOR FEW-SHOT CLASS INCREMENTAL LEARNING

［#2］
Haeyong Kang, Jaehong Yoon, Sultan Rizky Madjid, Sung Ju Hwang, and Chang D. Yoo*
Korea Advanced Institute of Science and Technology (KAIST)
291 Daehak-ro, Yuseong-gu, Daejeon
{haeyong.kang, jaehong.yoon, suulkyy, sjhwang82, cd_yoo}@kaist.ac.k r

## ABSTRACT

［#3］
Inspired by *Regularized Lottery Ticket Hypothesis*, which states that competitive smooth (non-binary) subnetworks exist within a dense network, we propose a few-shot class-incremental learning method referred to as *Soft-SubNetworks (SoftNet)*.
Our objective is to learn a sequence of sessions incrementally, where each session only includes a few training instances per class while preserving the knowledge of the previously learned ones. SoftNet jointly learns the model weights and adaptive non-binary soft masks at a base training session in which each mask consists of the major and minor subnetwork; the former aims to minimize catastrophic forgetting during training, and the latter aims to avoid overfitting to a few samples in each new training session. We provide comprehensive empirical validations demonstrating that our SoftNet effectively tackles the few-shot incremental learning problem by surpassing the performance of state-of-the-art baselines over benchmark datasets. The public code is available at https://github.com/ihaeyong/SoftNet-FSCIL.

## 1 INTRODUCTION

［#4］
Lifelong Learning, or Continual Learning, is a learning paradigm to expand knowledge and skills through sequential training of multiple tasks (Thrun, 1995). According to the accessibility of task identity during training and inference, the community often categorizes the field into specific problems, such as task-incremental (Pfülb and Gepperth, 2019; Delange et al., 2021; Yoon et al., 2020; Kang et al., 2022), class-incremental (Chaudhry et al., 2018; Kuzborskij et al., 2013; Li and Hoiem, 2017; Rebuffi et al., 2017; Kemker and Kanan, 2017; Castro et al., 2018; Hou et al., 2019; Wu et al., 2019), and task-free continual learning (Aljundi et al., 2019; Jin et al., 2021; Pham et al., 2022; Harrison et al., 2020). While the standard scenarios for continual learning assume a sufficiently large number of instances per task, a lifelong learner for real-world applications often suffers from insufficient training instances for each problem to solve. This paper aims to tackle the issue of limited training instances for practical Class-Incremental Learning (CIL), referred to as Few-Shot CIL (FSCIL) (Ren et al., 2019; Chen and Lee, 2020; Tao et al., 2020; Zhang et al., 2021; Cheraghian et al., 2021; Shi et al., 2021).

［#5］
However, there are two critical challenges in solving FSCIL problems: *catastrophic forgetting* and *overfitting*. Catastrophic forgetting (Goodfellow et al., 2013; Kirkpatrick et al., 2017) or Catastrophic Interference McCloskey and Cohen (1989) is a phenomenon in which a continual learner loses the previously learned task knowledge by updating the weights to adapt to new tasks, resulting in significant performance degeneration on previous tasks. Such undesired knowledge drift is irreversible since the scenario does not allow the model to revisit past task data. Recent works propose to mitigate catastrophic forgetting for class-incremental learning, often categorized in multiple directions, such *constraint-based* (Rebuffi et al., 2017; Castro et al., 2018; Hou et al., 2018; 2019; Wu et al., 2019), *memory-based* (Rebuffi et al., 2017; Chen and Lee, 2020; Mazumder et al., 2021; Shi et al., 2021), and *architecture-based methods* (Mazumder et al., 2021; Serra et al., 2018; Mallya and Lazebnik, 2018; Kang et al., 2022). However, we note that catastrophic forgetting becomes further challenging

［#6］
*Corresponding Author.


［#7］
in FSCIL. Due to the small amount of training data for new tasks, the model tends to severely **overfit to new classes** and quickly forget old classes, deteriorating the model performance.

［#8］
Meanwhile, several works address overfitting issues for continual learning from various perspectives. NCM (Hou et al., 2019) and BiC (Wu et al., 2019) highlight the prediction bias problem during sequential training that the models are prone to predict the data to classes in recently trained tasks. OCS (Yoon et al., 2022) tackles the class imbalance problems for rehearsal-based continual learning, where the number of instances at each class varies per task so that the model would perform biased training on dominant classes. Nevertheless, these works do not consider the overfitting issues caused by training a sequence of few-shot tasks. FSLL (Mazumder et al., 2021) tackles overfitting for few-shot CIL by partially-splitting model parameters for different sessions through multiple substeps of iterative reidentification and weight selection. However, it led to computationally inefficient.

［#9］
To deploy a practical few-shot CIL model, we propose a simple yet efficient method named *SoftNet*, effectively alleviating catastrophic forgetting and overfitting. Motivated by *Lottery Ticket Hypothesis* (Frankle and Carbin, 2019), which hypothesizes the existence of competitive subnetworks (winning tickets) within the randomly initialized dense neural network, we suggest a new paradigm for few-shot CIL, named *Regularized Lottery Ticket Hypothesis*:

［#10］
**Regularized Lottery Ticket Hypothesis (RLTH).** A randomly-initialized dense neural network contains a regularized subnetwork that can retain the prior class knowledge while providing room to learn the new class knowledge through isolated training of the subnetwork.

［#11］
Based on RLTH, we propose a method, referred to as **Soft-SubNetworks (SoftNet)**, illustrated in Figure 1. First, SoftNet jointly learns the randomly initialized dense model (Figure 1 (a)) and soft mask $\boldsymbol{m} \in [0,1]^{|\boldsymbol{\theta}|}$ pertaining to Soft-subnetwork (Figure 1 (b)) on the base session training; the soft mask consists of the major part of the model parameters $m=1$ and the minor ones $m<1$ where $m=1$ is obtained by the top-$c\%$ of model parameters and $m<1$ is obtained by the remaining ones $(100-\text{top-}c\%)$ sampled from the uniform distribution. Then, we freeze the major part of pre-trained subnetwork weights for maintaining prior class knowledge and update the only minor part of weights for the novel class knowledge (Figure 1 (c)).

［#12］
We summarize our key contributions as follows:
- This paper presents a new masking-based method, *Soft-SubNetwork (SoftNet)*, that tackles two critical challenges in the few-shot class incremental learning (FSCIL), known as catastrophic forgetting and overfitting.
- Our SoftNet trains two different types of non-binary masks (subnetworks) for solving FSCIL, preventing the continual learner from forgetting previous sessions and overfitting simultaneously.
- We conduct a comprehensive empirical study on SoftNet with multiple class incremental learning methods. Our method significantly outperforms strong baselines on benchmark tasks for FSCIL problems.

## 2 RELATED WORK
［#13］
**Catastrophic Forgetting.** Many recent works have made remarkable progress in tackling the challenges of catastrophic forgetting in lifelong learning. To be specific, Architecture-based approaches (Mallya et al., 2018; Serrà et al., 2018; Li et al., 2019) utilize an additional capacity to expand (Xu and Zhu, 2018; Yoon et al., 2018) or isolate (Rusu et al., 2016) model parameters, thereby avoiding knowledge interference during continual learning; SupSup (Wortsman et al., 2020) allocates model parameters dedicated to different tasks. Very recently, Chen et al. (2021); Kang et al. (2022) shows the existence of a sparse subnetwork, called winning tickets, that performs well on all tasks during continual learning. However, many subnetwork-based approaches are incompatible with the FSCIL setting since performing task inference under data imbalances is challenging. FSLL (Mazumder et al., 2021) aims to search session-specific subnetworks while preserving weights for previous sessions for incremental few-shot learning. However, the expansion process comprises another series of retraining and pruning steps, requiring excessive training time and computational costs. On the contrary, our proposed method, SoftNet, jointly learns the model and task-adaptive

［#14］
![](./images/867754338234139070_1.jpg)
![](./images/867754338234139070_2.jpg)

［#15］
![](./images/867754338234139070_3.jpg)
![](./images/867754338234139070_4.jpg)

［#16］
![](./images/867754338234139070_5.jpg)
![](./images/867754338234139070_6.jpg)

［#17］
Figure 1: Incremental Soft-Subnetwork (SoftNet): (a) Dense Neural Network is randomly initialized for the base session (S-1) training (b) SoftNet is trained by major subnetwork $\boldsymbol{m}_{major}=1$ (thick solid line) and minor $\boldsymbol{m}_{minor} \sim U(0,1)$, and (c) SoftNet updates only a few minor weights (thin solid line) for new sessions (S).

［#18］
smooth (i.e., non-binary) masks of the subnetwork associated with the base session while selecting an essential subset of the model weights for the upcoming session. Furthermore, smooth masks behave like regularizers that prevent overfitting when learning new classes.

［#19］
Soft-subnetwork. Recent works with context-dependent gating of sub-spaces (He and Jaeger, 2018), parameters (Mallya and Lazebnik, 2018; He et al., 2019; Mazumder et al., 2021), or layers (Serra et al., 2018) of a single deep neural network demonstrated its effectiveness in addressing catastrophic forgetting during continual learning. Masse et al. (2018) combines context-dependent gating with the constraints preventing significant changes in model weights, such as SI (Zenke et al., 2017) and EWC (Kirkpatrick et al., 2017), achieving further performance increases than using them alone. A flat minima could also be considered as acquiring sub-spaces. Previous works have shown that a flat minimizer is more robust to random perturbations (Hinton and Van Camp, 1993; Hochreiter and Schmidhuber, 1994; Jiang et al., 2019). Recently, Shi et al. (2021) showed that obtaining flat loss minima in the base session, which stands for the first task session with sufficient training instances, is necessary to alleviate catastrophic forgetting in FSCIL. To minimize forgetting, they updated the model weights on the obtained flat loss contour. In our work, by selecting sub-networks (Frankle and Carbin, 2019; Zhou et al., 2019; Wortsman et al., 2019; Ramanujan et al., 2020; Kang et al., 2022; Chijiwa et al., 2022) and optimizing the sub-network parameters in a sub-space, we propose a new method to preserve learned knowledge from a base session on a major subnetwork and learn new sessions through regularized minor subnetworks.

## 3 SOFT-SUBNETWORK FOR FEW-SHOT CLASS INCREMENTAL LEARNING

### 3.1 PROBLEM STATEMENTS

［#20］
Various works have tried to mitigate catastrophic forgetting problems in class incremental learning using knowledge distillation, revisiting a subset of prior samples, or isolating essential model param-eters to retain prior class knowledge even after the model loses accessibility to them. However, as a few-shot class incremental learning scenario regards following tasks/sessions containing a small amount of training data, the model tends to severely overfit to new classes, making it difficult to fine-tune the previously trained model on a few samples. In addition, the fine-tuning process often leads to the catastrophic forgetting of base class knowledge. As a result, regularization is indispensable in the models to avoid forgetting and prevent the model from overfitting to new class samples by updating only the selected parameters for learning in the new session.

［#21］
Few-shot Class Incremental Learning (FSCIL) aims to learn new sessions with only a few examples continually. A FSCIL model learns a sequence of $T$ training sessions $\{\mathcal{D}^{1},\cdots,\mathcal{D}^{T}\}$, where $\mathcal{D}^{t}=$ $\{z_{i}^{t}=(\boldsymbol{x}_{i}^{t},y_{i}^{t})\}_{i}^{n_{t}}$ is the training data of session $t$ and $\boldsymbol{x}_{i}^{t}$ is an example of class $y_{i}^{t} \in \mathcal{O}^{t}$. In FSCIL, the base session $\mathcal{D}^{1}$ usually contains a large number of classes with sufficient training data for each class. In contrast, the subsequent sessions $(t \geq 2)$ will only contain a small number of classes with a few training samples per class, e.g., the $t^{\text{th}}$ session $\mathcal{D}^{t}$ is often presented as a $N$-way $K$-shot task. In each training session $t$, the model can access only the training data $\mathcal{D}^{t}$ and a few examples stored

［#21］
in the previous session. When the training of session $t$ is completed, we evaluate the model on test samples from all classes $\mathcal{O}=\bigcup_{i=1}^{t} \mathcal{O}^{i}$, where $\mathcal{O}^{i} \bigcap \mathcal{O}^{j \neq i}=\emptyset$ for $\forall i, j \leq T$.

［#22］
Consider a supervised learning setup where the $T$ sessions arrive in a lifelong learner $f(\cdot ; \boldsymbol{\theta})$ parameterized by the model weights $\boldsymbol{\theta}$ in sequential order. A few-shot class incremental learning scenario aims to learn the classes in a sequence of sessions without catastrophic forgetting. In the training session $t$, the model solves the following optimization procedure:

［#22］
$$
\boldsymbol{\theta}^{*}=\underset{\boldsymbol{\theta}}{\operatorname{minimize}} \frac{1}{n_{t}} \sum_{i=1}^{n_{t}} \mathcal{L}_{t}\left(f\left(\boldsymbol{x}_{i}^{t} ; \boldsymbol{\theta}\right), y_{i}^{t}\right),
\tag{1}
$$

［#22］
where $\mathcal{L}_{t}$ is a classification loss like cross-entropy, and $n_{t}$ is the number of instances for session $t$.

### 3.2 SUBNETWORK-BASED TRAINING FOR FEW-SHOT CLASS INCREMENTAL LEARNING

［#23］
As lifelong learners often adopt over-parameterized dense neural networks to allow resource freedom for future classes or tasks, updating entire weights in neural networks for few-shot tasks is often not preferable and often yields the overfitting problem. To overcome the limitations in FSCIL, we focus on updating partial weights in neural networks when a new task arrives. The desired set of partial weights, named subnetwork, can achieve on-par or even better performance with the following motivations: (1) Lottery Ticket Hypothesis (Frankle and Carbin, 2019) shows the existence of a subnetwork that performs well as the dense network, and (2) The subnetwork significantly downsized from the dense network reduces the size of the expansion of the solver while providing extra capacity to learn new sessions or tasks.

［#24］
We first suggest the objective referred to as HardNet as follows: given dense neural network parameters $\boldsymbol{\theta}$, the binary attention mask $\boldsymbol{m}_{t}^{*}$ describes the optimal subnetwork for session $t$ such that $|\boldsymbol{m}_{t}^{*}|$ is less than the dense model capacity $|\theta|$. However, such binarized subnetworks $\boldsymbol{m}_{t} \in\{0,1\}^{|\boldsymbol{\theta}|}$ cannot adjust the remaining parameters in a dense network for future sessions while solving past task problems cost- and memory efficiently. In FSCIL, the test accuracy of the base session drops significantly when it proceeds to learn sequential sessions since the subnetwork of $m=1$ plays a crucial role in maintaining the base class knowledge. To this end, we propose a soft-subnetwork $\boldsymbol{m}_{t} \in[0,1]^{|\boldsymbol{\theta}|}$ instead of the binarized subnetwork. It gives more flexibility to fine-tune a small part of the soft-subnetwork while fixing the rest to retain base class knowledge for FSCIL. As such, we find the soft-subnetwork through the following objective:

［#24］
$$
\begin{aligned}
\boldsymbol{m}_{t}^{*}=\underset{\boldsymbol{m}_{t} \in[0,1]^{|\boldsymbol{\theta}|}}{\operatorname{minimize}} & \frac{1}{n_{t}} \sum_{i=1}^{n_{t}} \mathcal{L}_{t}\left(f\left(\boldsymbol{x}_{i}^{t} ; \boldsymbol{\theta} \odot \boldsymbol{m}_{t}\right), y_{i}^{t}\right)-\mathcal{J} \\
& \text { subject to }\left|\boldsymbol{m}_{t}\right| \leq c.
\end{aligned}
\tag{2}
$$

［#25］
where session loss $\mathcal{J}=\mathcal{L}\left(f\left(\boldsymbol{x}_{i}^{t} ; \boldsymbol{\theta}\right), y_{i}^{t}\right)$, the subnetwork sparsity $c \ll|\boldsymbol{\theta}|$ (used as the selected proportion $\%$ of model parameters in the following section), and $\odot$ represents an element-wise product. In the following section, we describe how to obtain the soft-subnetwork $\boldsymbol{m}_{t}^{*}$ using the magnitude-based criterion (RLTH) while minimizing session loss jointly.

### 3.3 OBTAINING SOFT-SUBNETWORKS VIA COMPLEMENTARY WINNING TICKETS

［#26］
Let each weight be associated with a learnable parameter we call weight score $\boldsymbol{s}$, which numerically determines the importance of the associated weight. In other words, we declare a weight with a higher score as more important. At first, we find a subnetwork $\boldsymbol{\theta}^{*}=\boldsymbol{\theta} \odot \boldsymbol{m}_{t}^{*}$ of the dense neural network and then assign it as a solver of the current session $t$. The subnetworks associated with each session jointly learn the model weight $\boldsymbol{\theta}$ and binary mask $\boldsymbol{m}_{t}$. Given an objective $\mathcal{L}_{t}$, we optimize $\boldsymbol{\theta}$ as follows:

［#26］
$$
\boldsymbol{\theta}^{*}, \boldsymbol{m}_{t}^{*}=\underset{\boldsymbol{\theta}, \boldsymbol{s}}{\operatorname{minimize}} \mathcal{L}_{t}\left(\boldsymbol{\theta} \odot \boldsymbol{m}_{t} ; \mathcal{D}_{t}\right).
\tag{3}
$$

［#27］
where $\boldsymbol{m}_{t}$ is obtained by applying an indicator function $\mathbb{1}_{c}$ on weight scores $\boldsymbol{s}$. Note $\mathbb{1}_{c}(s)=1$ if $s$ belongs to top-c$\%$ scores and 0 otherwise.

［#28］
In the optimization process for FSCIL, however, we consider two main problems: (1) Catastrophic forgetting: updating all $\boldsymbol{\theta} \odot \boldsymbol{m}_{t-1}$ when training for new sessions will cause interference with

［#28］
the weights allocated for previous tasks; thus, we need to freeze all previously learned parameters $\boldsymbol{\theta} \odot \boldsymbol{m}_{t-1}$; (2) Overfitting: the subnetwork also encounters overfitting issues when training an incremental task on a few samples, as such, we need to update a few parameters irrelevant to previous task knowledge., i.e., $\boldsymbol{\theta} \odot (1 - \boldsymbol{m}_{t-1})$.

［#29］
To acquire the optimal subnetworks that alleviate the two issues, we define a soft-subnetwork by dividing the dense neural network into two parts-one is the major subnetwork $\boldsymbol{m}_{\text{major}}$, and another is the minor subnetwork $\boldsymbol{m}_{\text{minor}}$. The defined soft-subnetwork follows as:
［#29］
$$
\boldsymbol{m}_{\text{soft}} = \boldsymbol{m}_{\text{major}} \oplus \boldsymbol{m}_{\text{minor}},
\tag{4}
$$
［#29］
where $\boldsymbol{m}_{\text{major}}$ is a binary mask and $\boldsymbol{m}_{\text{minor}} \sim U(0, 1)$ and $\oplus$ represents an element-wise summation.
As such, a soft-mask is given as $\boldsymbol{m}_t^* \in [0, 1]^{|\boldsymbol{\theta}|}$ in Eq.3. In the all-experimental FSCIL setting, $\boldsymbol{m}_{\text{major}}$ maintains the base task knowledge $t=1$ while $\boldsymbol{m}_{\text{minor}}$ acquires the novel task knowledge $t \geq 2$. Then, with base session learning rate $\alpha$, the $\boldsymbol{\theta}$ is updated as follows: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} \odot \boldsymbol{m}_{\text{soft}}\right)$ effectively regularize the weights of the subnetworks for incremental learning. The subnetworks are obtained by the indicator function that always has a gradient value of 0; therefore, updating the weight scores $\boldsymbol{s}$ with its loss gradient is impossible. To update the weight scores, we use Straight-through Estimator (Hinton, 2012; Bengio et al., 2013; Ramanujan et al., 2020) in the backward pass. Specifically, we ignore the derivatives of the indicator function and update the weight score $\boldsymbol{s} \leftarrow \boldsymbol{s} - \alpha\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{s}} \odot \boldsymbol{m}_{\text{soft}}\right)$, where $\boldsymbol{m}_{\text{soft}}=1$ for exploring the optimal subnetwork for base session training. Our Soft-subnetwork optimizing procedure is summarized in Algorithm 1. Once a single soft-subnetwork $\boldsymbol{m}_{\text{soft}}$ is obtained in the base session, then we use the soft-subnetwork for the entire new sessions without updating.

［#30］
<table style="border-collapse:collapse; width:100%">
<thead>
<tr><th colspan="2">Algorithm 1 Soft-Subnetworks (SoftNet)</th></tr>
</thead>
<tbody>
<tr><td colspan="2"><b>input</b> $\{\mathcal{D}^t\}_{t=1}^{\mathcal{T}}$, model weights $\boldsymbol{\theta}$, and score weights $\boldsymbol{s}$, layer-wise capacity $c$</td></tr>
<tr><td>1:</td><td>// Training over base classes $t=1$</td></tr>
<tr><td>2:</td><td>Randomly initialize $\boldsymbol{\theta}$ and $\boldsymbol{s}$.</td></tr>
<tr><td>3:</td><td>for epoch $e=1,2,\cdots$ <b>do</b></td></tr>
<tr><td>4:</td><td>$\quad$ Obtain softmask $\boldsymbol{m}_{\text{soft}}$ of $\boldsymbol{m}_{major}$ and $\boldsymbol{m}_{minor} \sim U(0,1)$ at each layer</td></tr>
<tr><td>5:</td><td>$\quad$ for batch $\boldsymbol{b}_t \sim \mathcal{D}^t$ <b>do</b></td></tr>
<tr><td>6:</td><td>$\quad$$\quad$ Compute $\mathcal{L}_{base}\left(\boldsymbol{\theta} \odot \boldsymbol{m}_{\text{soft}}; \boldsymbol{b}_t\right)$ by Eq. 3</td></tr>
<tr><td>7:</td><td>$\quad$$\quad$ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} \odot \boldsymbol{m}_{\text{soft}}\right)$</td></tr>
<tr><td>8:</td><td>$\quad$$\quad$ $\boldsymbol{s} \leftarrow \boldsymbol{s} - \alpha\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{s}} \odot \boldsymbol{m}_{\text{soft}}\right)$</td></tr>
<tr><td>9:</td><td>$\quad$ <b>end for</b></td></tr>
<tr><td>10:</td><td><b>end for</b></td></tr>
<tr><td>11:</td><td>// Incremental learning $t \geq 2$</td></tr>
<tr><td>12:</td><td>Combine the training data $\mathcal{D}^t$ and the exemplars saved in previous few-shot sessions</td></tr>
<tr><td>13:</td><td>for epoch $e=1,2,\cdots$ <b>do</b></td></tr>
<tr><td>14:</td><td>$\quad$ for batch $\boldsymbol{b}_t \sim \mathcal{D}^t$ <b>do</b></td></tr>
<tr><td>15:</td><td>$\quad$$\quad$ Compute $\mathcal{L}_{m}\left(\boldsymbol{\theta} \odot \boldsymbol{m}_{\text{soft}}; \boldsymbol{b}_t\right)$ by Eq. 5</td></tr>
<tr><td>16:</td><td>$\quad$$\quad$ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \beta\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} \odot \boldsymbol{m}_{minor}\right)$</td></tr>
<tr><td>17:</td><td>$\quad$ <b>end for</b></td></tr>
<tr><td>18:</td><td><b>end for</b></td></tr>
<tr><td colspan="2"><b>output</b> model parameters $\boldsymbol{\theta}$, $\boldsymbol{s}$, and $\boldsymbol{m}_{\text{soft}}$.</td></tr>
</tbody>
</table>

## 4 INCREMENTAL LEARNING FOR SOFT-SUBNETWORK

［#31］
We now describe the overall procedure of our soft-pruning-based incremental learning/inference method, including the training phase with a normalized informative measurement in Section 4.1, as followed by the prior work (Shi et al., 2021), and the inference phase in Section 4.2.

### 4.1 INCREMENTAL SOFT-SUBNETWORK TRAINING

［#32］
**Base Training** ($t=1$). In the base learning session, we optimize the soft-subnetwork parameter $\boldsymbol{\theta}$ (including a fully-connected layer as a classifier) and weight score $\boldsymbol{s}$ with cross-entropy loss jointly using the training examples of $\mathcal{D}^1$.

［#33］
**Incremental Training** ($t \geq 2$). In the incremental few-shot learning sessions ($t \geq 2$), leveraged by $\boldsymbol{\theta} \odot \boldsymbol{m}_{\text{soft}}$, we fine-tune few minor parameters $\boldsymbol{\theta} \odot \boldsymbol{m}_{\text{minor}}$ of the soft-subnetwork to learn new classes.


［#34］
<table>
<caption>Table 1: Classification accuracy of ResNet18 on CIFAR-100 for 5-way 5-shot incremental learning. Underbar denotes the comparable results with FSLL (Mazumder et al., 2021). * denotes the results reported from Shi et al. (2021).</caption>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="9">sessions</th>
<th rowspan="2">The gap with cRT</th>
</tr>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
</tr>
</thead>
<tbody>
<tr>
<td>cRT (Shi et al., 2021)</td>
<td>65.18</td>
<td>63.89</td>
<td>60.20</td>
<td>57.23</td>
<td>53.71</td>
<td>50.39</td>
<td>48.77</td>
<td>47.29</td>
<td>45.28</td>
<td>-</td>
</tr>
<tr>
<td>iCaRL (Rebuffi et al., 2017)*</td>
<td>66.52</td>
<td>57.26</td>
<td>54.27</td>
<td>50.62</td>
<td>47.33</td>
<td>44.99</td>
<td>43.14</td>
<td>41.16</td>
<td>39.49</td>
<td>-5.79</td>
</tr>
<tr>
<td>Rebalance (Hou et al., 2019)*</td>
<td>66.66</td>
<td>61.42</td>
<td>57.29</td>
<td>53.02</td>
<td>48.85</td>
<td>45.68</td>
<td>43.06</td>
<td>40.56</td>
<td>38.35</td>
<td>-6.93</td>
</tr>
<tr>
<td>FSLL (Mazumder et al., 2021)*</td>
<td>65.18</td>
<td>56.24</td>
<td>54.55</td>
<td>51.61</td>
<td>49.11</td>
<td>47.27</td>
<td>45.35</td>
<td>43.95</td>
<td>42.22</td>
<td>-3.08</td>
</tr>
<tr>
<td>iCaRL (Rebuffi et al., 2017)</td>
<td>64.10</td>
<td>53.28</td>
<td>41.69</td>
<td>34.13</td>
<td>27.93</td>
<td>25.06</td>
<td>20.41</td>
<td>15.48</td>
<td>13.73</td>
<td>-31.55</td>
</tr>
<tr>
<td>Rebalance (Hou et al., 2019)</td>
<td>64.10</td>
<td>53.05</td>
<td>43.96</td>
<td>36.97</td>
<td>31.61</td>
<td>26.73</td>
<td>21.23</td>
<td>16.78</td>
<td>13.54</td>
<td>-31.74</td>
</tr>
<tr>
<td>TOPIC (Cheraghian et al., 2021)</td>
<td>64.10</td>
<td>55.88</td>
<td>47.07</td>
<td>45.16</td>
<td>40.11</td>
<td>36.38</td>
<td>33.96</td>
<td>31.55</td>
<td>29.37</td>
<td>-15.91</td>
</tr>
<tr>
<td>F2M (Shi et al., 2021)</td>
<td>64.71</td>
<td>62.05</td>
<td>59.01</td>
<td>55.58</td>
<td>52.55</td>
<td>49.96</td>
<td>48.08</td>
<td>46.28</td>
<td>44.67</td>
<td>-0.61</td>
</tr>
<tr>
<td>FSLL (Mazumder et al., 2021)</td>
<td>64.10</td>
<td>55.85</td>
<td>51.71</td>
<td>48.59</td>
<td>45.34</td>
<td>43.25</td>
<td>41.52</td>
<td>39.81</td>
<td>38.16</td>
<td>-7.12</td>
</tr>
<tr>
<td>HardNet, $c=50\%$</td>
<td>64.80</td>
<td>60.77</td>
<td>56.95</td>
<td>53.53</td>
<td>50.40</td>
<td>47.82</td>
<td>45.93</td>
<td>43.95</td>
<td>41.91</td>
<td>-3.37</td>
</tr>
<tr>
<td>HardNet, $c=80\%$</td>
<td>69.65</td>
<td>64.60</td>
<td>60.59</td>
<td>56.93</td>
<td>53.60</td>
<td>50.80</td>
<td>48.69</td>
<td>46.69</td>
<td>44.63</td>
<td>-0.65</td>
</tr>
<tr>
<td>HardNet, $c=99\%$</td>
<td>71.95</td>
<td>66.83</td>
<td>62.75</td>
<td>59.09</td>
<td>55.92</td>
<td>53.03</td>
<td>50.78</td>
<td>48.52</td>
<td>46.31</td>
<td>+1.03</td>
</tr>
<tr>
<td>SoftNet, $c=50\%$</td>
<td>69.20</td>
<td>64.18</td>
<td>60.01</td>
<td>56.43</td>
<td>53.11</td>
<td>50.62</td>
<td>48.60</td>
<td>46.51</td>
<td>44.61</td>
<td>-0.67</td>
</tr>
<tr>
<td>SoftNet, $c=80\%$</td>
<td>70.38</td>
<td>65.04</td>
<td>60.94</td>
<td>57.26</td>
<td>54.13</td>
<td>51.58</td>
<td>49.52</td>
<td>47.36</td>
<td>45.16</td>
<td>-0.12</td>
</tr>
<tr>
<td>SoftNet, $c=99\%$</td>
<td>72.62</td>
<td>67.31</td>
<td>63.05</td>
<td>59.39</td>
<td>56.00</td>
<td>53.23</td>
<td>51.06</td>
<td>48.83</td>
<td>46.63</td>
<td>+1.35</td>
</tr>
</tbody>
</table>

［#35］
Since $m_{minor} < 1$, the soft-subnetwork alleviates the overfitting of a few samples. Furthermore, instead of Euclidean distance (Shi et al., 2021), we employ a metric-based classification algorithm with cosine distance to finetune the few selected parameters. In some cases, Euclidean distance fails to give the real distances between representations, especially when two points with the same distance from prototypes do not fall in the same class. In contrast, representations with a low cosine distance are located in the same direction from the origin, providing a normalized informative measurement.
We define the loss function as:

［#35］
$$
\mathcal{L}_{m}(z ; \boldsymbol{\theta} \odot \boldsymbol{m}_{soft})=-\sum_{z \in \mathcal{D}} \sum_{o \in \mathcal{O}} \mathbb{1}(y=o) \log \left(\frac{e^{-d\left(\boldsymbol{p}_{o}, f\left(\boldsymbol{x} ; \boldsymbol{\theta} \odot \boldsymbol{m}_{soft}\right)\right)}}{\sum_{o_{k} \in \mathcal{O}} e^{-d\left(\boldsymbol{p}_{o_{k}}, f\left(\boldsymbol{x} ; \boldsymbol{\theta} \odot \boldsymbol{m}_{soft}\right)\right)}}\right)\qquad(5)
$$

［#35］
where $d(\cdot, \cdot)$ denotes cosine distance, $\boldsymbol{p}_{o}$ is the prototype of class $o$, $\mathcal{O}=\bigcup_{i=1}^{t} \mathcal{O}^{i}$ refers to all encountered classes, and $\mathcal{D}=\mathcal{D}^{t} \bigcup \mathcal{P}$ denotes the union of the current training data $\mathcal{D}^{t}$ and the exemplar set $\mathcal{P}=\{\boldsymbol{p}_{2} \cdots, \boldsymbol{p}_{t-1}\}$, where $\mathcal{P}_{t_{e}}\ (2 \leq t_{e}<t)$ is the set of saved exemplars in session $t_{e}$. Note that the prototypes of new classes are computed by $\boldsymbol{p}_{o}=\frac{1}{N_{o}} \sum_{i} \mathbb{1}(y_{i}=o) f(\boldsymbol{x}_{i} ; \boldsymbol{\theta} \odot \boldsymbol{m}_{soft})$ and those of base classes are saved in the base session, and $N_{o}$ denotes the number of the training images of class $o$. We also save the prototypes of all classes in $\mathcal{O}^{t}$ for later evaluation.

## 4.2 INFERENCE FOR INCREMENTAL SOFT-SUBNETWORK

［#36］
In each session, the inference is also conducted by a simple nearest class mean (NCM) classification algorithm (Mensink et al., 2013; Shi et al., 2021) for fair comparisons. Specifically, all the training and test samples are mapped to the embedding space of the feature extractor $f$, and Euclidean distance $d_{u}(\cdot, \cdot)$ is used to measure the similarity between them. The classifier gives the $k$th prototype index $o_{k}^{*}=\arg \min _{o \in \mathcal{O}} d_{u}(f(\boldsymbol{x} ; \boldsymbol{\theta} \odot \boldsymbol{m}_{soft}), \boldsymbol{p}_{o})$ as output.

## 5 EXPERIMENTS

［#37］
We introduce experimental setups in Section 5.1. Then, we empirically evaluate our soft-subnetworks for incremental few-shot learning and demonstrate its effectiveness through comparison with state-of-the-art methods and vanilla subnetworks in the following subsections.

### 5.1 EXPERIMENTAL SETUP

［#38］
Datasets. To validate the effectiveness of the soft-subnetwork, we follow the standard FSCIL experimental setting. We randomly select 60 classes as the base class and the remaining 40 as new classes for CIFAR-100 and miniImageNet. In each incremental learning session, we construct 5-way 5-shot tasks by randomly picking five classes and sampling five training examples for each class.

［#39］
Table 2: Classification accuracy of ResNet18 on miniImageNet for 5-way 5-shot incremental learning. Underbar denotes the comparable results with FSLL Mazumder et al. (2021). * denotes the results reported from Shi et al. (2021).

［#40］
<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="9">sessions</th>
    <th rowspan="2">The gap with cRT</th>
  </tr>
  <tr>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
    <th>6</th>
    <th>7</th>
    <th>8</th>
    <th>9</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>cRT (Shi et al., 2021)</td>
    <td>67.30</td>
    <td>64.15</td>
    <td>60.59</td>
    <td>57.32</td>
    <td>54.22</td>
    <td>51.43</td>
    <td>48.92</td>
    <td>46.78</td>
    <td>44.85</td>
    <td>-</td>
  </tr>
  <tr>
    <td>iCaRL (Rebuffi et al., 2017)*</td>
    <td>67.35</td>
    <td>59.91</td>
    <td>55.64</td>
    <td>52.60</td>
    <td>49.43</td>
    <td>46.73</td>
    <td>44.13</td>
    <td>42.17</td>
    <td>40.29</td>
    <td>-4.56</td>
  </tr>
  <tr>
    <td>Rebalance (Hou et al., 2019)*</td>
    <td>67.91</td>
    <td>63.11</td>
    <td>58.75</td>
    <td>54.83</td>
    <td>50.68</td>
    <td>47.11</td>
    <td>43.88</td>
    <td>41.19</td>
    <td>38.72</td>
    <td>-6.13</td>
  </tr>
  <tr>
    <td>FSLL (Mazumder et al., 2021)*</td>
    <td>67.30</td>
    <td>59.81</td>
    <td>57.26</td>
    <td>54.57</td>
    <td>52.05</td>
    <td>49.42</td>
    <td>46.95</td>
    <td>44.94</td>
    <td>42.87</td>
    <td>-1.11</td>
  </tr>
  <tr>
    <td>iCaRL (Rebuffi et al., 2017)</td>
    <td>61.31</td>
    <td>46.32</td>
    <td>42.94</td>
    <td>37.63</td>
    <td>30.49</td>
    <td>24.00</td>
    <td>20.89</td>
    <td>18.80</td>
    <td>17.21</td>
    <td>-27.64</td>
  </tr>
  <tr>
    <td>Rebalance (Hou et al., 2019)</td>
    <td>61.31</td>
    <td>47.80</td>
    <td>39.31</td>
    <td>31.91</td>
    <td>25.68</td>
    <td>21.35</td>
    <td>18.67</td>
    <td>17.24</td>
    <td>14.17</td>
    <td>-30.68</td>
  </tr>
  <tr>
    <td>TOPIC (Cheraghian et al., 2021)</td>
    <td>61.31</td>
    <td>50.09</td>
    <td>45.17</td>
    <td>41.16</td>
    <td>37.48</td>
    <td>35.52</td>
    <td>32.19</td>
    <td>29.46</td>
    <td>24.42</td>
    <td>-20.43</td>
  </tr>
  <tr>
    <td>IDLVQ-C (Chen and Lee, 2020)</td>
    <td>64.77</td>
    <td>59.87</td>
    <td>55.93</td>
    <td>52.62</td>
    <td>49.88</td>
    <td>47.55</td>
    <td>44.83</td>
    <td>43.14</td>
    <td>41.84</td>
    <td>-3.01</td>
  </tr>
  <tr>
    <td>F2M (Shi et al., 2021)</td>
    <td>67.28</td>
    <td>63.80</td>
    <td>60.38</td>
    <td>57.06</td>
    <td>54.08</td>
    <td>51.39</td>
    <td>48.82</td>
    <td>46.58</td>
    <td>44.65</td>
    <td>-0.20</td>
  </tr>
  <tr>
    <td>FSLL (Mazumder et al., 2021)</td>
    <td>66.48</td>
    <td>61.75</td>
    <td>58.16</td>
    <td>54.16</td>
    <td>51.10</td>
    <td>48.53</td>
    <td>46.54</td>
    <td>44.20</td>
    <td>42.28</td>
    <td>-2.57</td>
  </tr>
  <tr>
    <td>HardNet, $c=50\%$</td>
    <td>65.13</td>
    <td>60.37</td>
    <td>56.12</td>
    <td>53.17</td>
    <td>50.17</td>
    <td>47.74</td>
    <td>45.34</td>
    <td>43.35</td>
    <td>42.13</td>
    <td>-2.72</td>
  </tr>
  <tr>
    <td>HardNet, $c=80\%$</td>
    <td>69.73</td>
    <td>64.46</td>
    <td>60.42</td>
    <td>57.09</td>
    <td>54.09</td>
    <td>51.18</td>
    <td>48.76</td>
    <td>46.81</td>
    <td>45.66</td>
    <td>+0.81</td>
  </tr>
  <tr>
    <td>HardNet, $c=90\%$</td>
    <td>64.68</td>
    <td>59.80</td>
    <td>55.70</td>
    <td>52.82</td>
    <td>50.01</td>
    <td>47.30</td>
    <td>45.17</td>
    <td>43.34</td>
    <td>42.09</td>
    <td>-2.76</td>
  </tr>
  <tr>
    <td>SoftNet, $c=50\%$</td>
    <td>72.83</td>
    <td>67.23</td>
    <td>62.82</td>
    <td>59.41</td>
    <td>56.44</td>
    <td>53.55</td>
    <td>50.92</td>
    <td>48.99</td>
    <td>47.60</td>
    <td>+2.75</td>
  </tr>
  <tr>
    <td>SoftNet, $c=80\%$</td>
    <td>76.63</td>
    <td>70.13</td>
    <td>65.92</td>
    <td>62.52</td>
    <td>59.49</td>
    <td>56.56</td>
    <td>53.71</td>
    <td>51.72</td>
    <td>50.48</td>
    <td>+5.63</td>
  </tr>
  <tr>
    <td>SoftNet, $c=90\%$</td>
    <td>77.00</td>
    <td>70.38</td>
    <td>65.94</td>
    <td>62.45</td>
    <td>59.32</td>
    <td>56.25</td>
    <td>53.76</td>
    <td>51.75</td>
    <td>50.39</td>
    <td>+5.54</td>
  </tr>
  <tr>
    <td>SoftNet, $c=97\%$</td>
    <td>77.17</td>
    <td>70.32</td>
    <td>66.15</td>
    <td>62.55</td>
    <td>59.48</td>
    <td>56.46</td>
    <td>53.71</td>
    <td>51.68</td>
    <td>50.24</td>
    <td>+5.39</td>
  </tr>
</tbody>
</table>

［#41］
Baselines. We mainly compare our SoftNet with architecture-based methods for FSCIL: FSLL (Mazumder et al., 2021) that selects important parameters for each session, and HardNet, representing a binary subnetwork. Furthermore, we compare other FSCIL methods such as iCaRL (Rebuffi et al., 2017), Rebalance (Hou et al., 2019), TOPIC (Tao et al., 2020), IDLVQ-C (Chen and Lee, 2020), and F2M (Shi et al., 2021). We also include a joint training method (Shi et al., 2021) that uses all previously seen data, including the base and the following few-shot tasks for training as a reference. Furthermore, we fix the classifier re-training method (cRT) (Kang et al., 2019) for long-tailed classification trained with all encountered data as the approximated upper bound.

［#42］
Experimental details. The experiments are conducted with NVIDIA GPU RTX8000 on CUDA 11.0. We also randomly split each dataset into multiple sessions. We run each algorithm ten times for each dataset and report their mean accuracy. We adopt ResNet18 (He et al., 2016) as the backbone network. For data augmentation, we use standard random crop and horizontal flips. In the base session training stage, we select top-$c\%$ weights at each layer and acquire the optimal soft-subnetworks with the best validation accuracy. In each incremental few-shot learning session, the total number of training epochs is 6, and the learning rate is 0.02. We train new class session samples using a few minor weights of the soft-subnetwork (Conv4x layer of ResNet18 and Conv3x layer of ResNet20) obtained by the base session learning. We specify further experiment details in Appendix A.

### 5.2 RESULTS AND COMPARISONS

［#43］
We compared SoftNet with the architecture-based methods - FSLL and HardNet. We pick FSLL as an architecture-based baseline since it selects important parameters for acquiring old/new class knowledge. The architecture-based results on CIFAR-100 and miniImageNet are presented in Table 1 and Table 2 respectively. The performances of HardNet show the effectiveness of the subnetworks that go with less model capacity compared to dense networks. To emphasize our point, we found that ResNet18, with approximately $50\%$ parameters, achieves comparable performances with FSLL on CIFAR-100 and miniImageNet. In addition, the performances of ResNet20 with $30\%$ parameters (HardNet) are comparable with those of FSLL on CIFAR-100, as denoted in Appendix of Table 9 and Table 11, including performances (Figure 4 and Figure 5) and smoothness in t-SNE plots (Figure 6).

［#44］
Experimental results are prepared to analyze the overall performances of SoftNet according to the sparsity and dataset as shown in Figure 2. As we increase the number of parameters employed by SoftNet, we achieve performance gain on both benchmark datasets. The performance variance of SoftNet's sparsity seems to be depending on datasets from the fact that the performance variance on CIFAR-100 is less than that on miniImageNet. In addition, SoftNet retains prior session knowledge successfully in both experiments as described in the dashed line, and the performances of SoftNet ($c=60.0\%$) on the new class session (8, 9) of CIFAR-100 than those of SoftNet ($c=80.0\%$) as depicted in the dashed-dot line. From these results, we could expect that the best performances

［#44］
depend on the number of parameters and properties of datasets. We further result on comparisons of HardNet and SoftNet in Appendix B.

［#45］
![](./images/867754338234139070_7.jpg)

［#46］
![](./images/867754338234139070_8.jpg)

［#47］
**Figure 2: Classification accuracy of SoftNet on CIFAR-100 and miniImageNet for 5-way 5-shot FSCIL:** the overall performance depends on capacity $c$ and the softness of subnetwork. Note that solid(–––), dashed(- - - ), and dashed-dot(–·–) lines denote overall, base, and novel class performances respectively.

［#48］
Our SoftNet outperforms the state-of-the-art methods and cRT, which is used as the approximate upper bound of FSCIL (Shi et al., 2021) as shown in Table 1 and Table 2. Moreover, Figure 3 represents the outstanding performances of SoftNet on CIFAR-100 and miniImageNet. SoftNet provides a new upper bound on each dataset, outperforming cRT, while HardNet provides new baselines among pruning-based methods.

［#49］
![](./images/867754338234139070_9.jpg)

［#50］
**Figure 3: Comparision of subnetworks (HardNet and SoftNet) with state-of-the-art methods.**

### 5.3 LAYER-WISE ACCURACY

［#51］
In incremental few-shot learning sessions, we train new class session samples using a few minor weights $m_{\text{minor}}$ of the specific layer. At the same, we entirely fix the remaining weights to investigate the best performances as shown in Table 3. The best performances involve fine-tuning at the Conv5x layer with $c=97\%$. It means features computed by the lower layer are general and reusable in different classes. On the other hand, features from the higher layer are specific and highly dependent on the dataset.

### 5.4 ARCHITECTURE-WISE ACCURACY

［#52］
Depending on architectures, the performances of subnetworks vary, and the sparsity is also one another: ResNet18 tends to use dense parameters, whereas ResNet20 tends to use sparse parameters on CIFAR-100 for 5-way 5-shot as shown in Table 4. We observed that the SoftNet with ResNet20 has a more sparse solution as $c=90\%$ than ResNet18 on this CIFAR-100 FSCIL setting. From these

［#53］
Table 3: Classification accuracy of ResNet18 on miniImageNet for 5-way 5-shot incremental learning.
The layer-wise inspection with fixed $c=97\%$. all denotes that all minor weights $m_{\text{minor}}$ of the entire
layers were trained while only $conv{*}x$ trained.

［#54］
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="9">sessions</th>
      <th rowspan="2">The gap with cRT</th>
    </tr>
    <tr>
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cRT (Shi et al., 2021)</td>
      <td>67.30</td>
      <td>64.15</td>
      <td>60.59</td>
      <td>57.32</td>
      <td>54.22</td>
      <td>51.43</td>
      <td>48.92</td>
      <td>46.78</td>
      <td>44.85</td>
      <td>-</td>
    </tr>
    <tr>
      <td>SoftNet, Conv2x</td>
      <td>77.17</td>
      <td>70.29</td>
      <td>66.09</td>
      <td>62.54</td>
      <td>59.44</td>
      <td>56.43</td>
      <td>53.68</td>
      <td>51.60</td>
      <td>50.19</td>
      <td>+5.34</td>
    </tr>
    <tr>
      <td>SoftNet, Conv3x</td>
      <td>77.17</td>
      <td>70.30</td>
      <td>66.05</td>
      <td>62.51</td>
      <td>59.42</td>
      <td>56.44</td>
      <td>53.70</td>
      <td>51.59</td>
      <td>50.15</td>
      <td>+5.30</td>
    </tr>
    <tr>
      <td>SoftNet, Conv4x</td>
      <td>77.17</td>
      <td>70.30</td>
      <td>66.08</td>
      <td>62.54</td>
      <td>59.45</td>
      <td>56.46</td>
      <td>53.70</td>
      <td>51.59</td>
      <td>50.17</td>
      <td>+5.32</td>
    </tr>
    <tr>
      <td>SoftNet, Conv5x</td>
      <td>77.17</td>
      <td>70.32</td>
      <td>66.15</td>
      <td>62.55</td>
      <td>59.48</td>
      <td>56.46</td>
      <td>53.71</td>
      <td>51.68</td>
      <td>50.24</td>
      <td>+5.39</td>
    </tr>
    <tr>
      <td>SoftNet, All</td>
      <td>77.17</td>
      <td>65.09</td>
      <td>55.25</td>
      <td>45.92</td>
      <td>38.20</td>
      <td>33.37</td>
      <td>29.52</td>
      <td>27.66</td>
      <td>25.64</td>
      <td>-19.21</td>
    </tr>
  </tbody>
</table>

［#55］
observations, our SoftNet could significantly impact deep neural network architecture search - it helps to search sparse and task-specific architecture.

［#56］
Table 4: Classification accuracy of ResNet18, 20, 32, and 50 on CIFAR-100 for 5-way 5-shot FSCIL.

［#57］
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="9">sessions</th>
      <th rowspan="2">The gap with cRT</th>
    </tr>
    <tr>
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet18, cRT (Shi et al., 2021)</td>
      <td>65.18</td>
      <td>63.89</td>
      <td>60.20</td>
      <td>57.23</td>
      <td>53.71</td>
      <td>50.39</td>
      <td>48.77</td>
      <td>47.29</td>
      <td>45.28</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ResNet18, SoftNet, $c=70\%$</td>
      <td>70.92</td>
      <td>65.16</td>
      <td>61.00</td>
      <td>57.25</td>
      <td>54.09</td>
      <td>51.37</td>
      <td>49.29</td>
      <td>47.03</td>
      <td>44.90</td>
      <td>-0.38</td>
    </tr>
    <tr>
      <td>SoftNet, $c=90\%$</td>
      <td>72.25</td>
      <td>66.82</td>
      <td>62.63</td>
      <td>58.98</td>
      <td>55.64</td>
      <td>52.77</td>
      <td>50.71</td>
      <td>48.42</td>
      <td>46.15</td>
      <td>+0.87</td>
    </tr>
    <tr>
      <td>SoftNet, $c=99\%$</td>
      <td>72.62</td>
      <td>67.31</td>
      <td>63.05</td>
      <td>59.39</td>
      <td>56.00</td>
      <td>53.23</td>
      <td>51.06</td>
      <td>48.83</td>
      <td>46.63</td>
      <td>+1.35</td>
    </tr>
    <tr>
      <td>ResNet20, SoftNet, $c=70\%$</td>
      <td>70.38</td>
      <td>66.16</td>
      <td>62.63</td>
      <td>58.93</td>
      <td>55.81</td>
      <td>53.11</td>
      <td>51.38</td>
      <td>49.29</td>
      <td>47.08</td>
      <td>+1.80</td>
    </tr>
    <tr>
      <td>SoftNet, $c=90\%$</td>
      <td>72.63</td>
      <td>68.60</td>
      <td>64.96</td>
      <td>61.25</td>
      <td>57.98</td>
      <td>55.32</td>
      <td>53.48</td>
      <td>51.46</td>
      <td>49.20</td>
      <td>+3.92</td>
    </tr>
    <tr>
      <td>SoftNet, $c=99\%$</td>
      <td>71.78</td>
      <td>67.79</td>
      <td>63.86</td>
      <td>60.07</td>
      <td>57.05</td>
      <td>54.32</td>
      <td>52.34</td>
      <td>50.28</td>
      <td>48.11</td>
      <td>+2.83</td>
    </tr>
    <tr>
      <td>ResNet32, SoftNet, $c=90\%$</td>
      <td>75.47</td>
      <td>70.84</td>
      <td>66.84</td>
      <td>63.01</td>
      <td>59.69</td>
      <td>56.86</td>
      <td>54.75</td>
      <td>52.70</td>
      <td>50.36</td>
      <td>+5.08</td>
    </tr>
    <tr>
      <td>SoftNet, $c=93\%$</td>
      <td>75.35</td>
      <td>71.22</td>
      <td>67.25</td>
      <td>63.25</td>
      <td>60.05</td>
      <td>57.24</td>
      <td>55.16</td>
      <td>53.01</td>
      <td>50.76</td>
      <td>+5.48</td>
    </tr>
    <tr>
      <td>SoftNet, $c=95\%$</td>
      <td>74.67</td>
      <td>70.32</td>
      <td>66.31</td>
      <td>62.61</td>
      <td>59.29</td>
      <td>56.54</td>
      <td>54.52</td>
      <td>52.30</td>
      <td>50.05</td>
      <td>+4.77</td>
    </tr>
    <tr>
      <td>ResNet50, SoftNet, $c=70\%$</td>
      <td>76.20</td>
      <td>71.82</td>
      <td>67.90</td>
      <td>64.17</td>
      <td>60.91</td>
      <td>57.89</td>
      <td>55.72</td>
      <td>53.20</td>
      <td>50.87</td>
      <td>+5.59</td>
    </tr>
    <tr>
      <td>SoftNet, $c=80\%$</td>
      <td>78.20</td>
      <td>73.32</td>
      <td>69.22</td>
      <td>65.43</td>
      <td>62.09</td>
      <td>59.08</td>
      <td>56.80</td>
      <td>54.45</td>
      <td>52.18</td>
      <td>+6.90</td>
    </tr>
    <tr>
      <td>SoftNet, $c=90\%$</td>
      <td>77.67</td>
      <td>72.73</td>
      <td>68.50</td>
      <td>64.57</td>
      <td>61.08</td>
      <td>58.06</td>
      <td>55.70</td>
      <td>53.37</td>
      <td>51.20</td>
      <td>+5.92</td>
    </tr>
  </tbody>
</table>

## 5.5 DISCUSSIONS

［#58］
Based on our thorough empirical study, we uncover the following facts: (1) Depending on architec- tures, the performances of subnetworks vary, and the sparsity is also one another: ResNet18 tends to use dense parameters, while ResNet20 tends to use sparse parameters on CIFAR-100 FSCIL settings. This result provides the general pruning-based model with a hidden clue. (2) In general, fine-tuning strategies are essential in retaining prior knowledge and learning new knowledge. We found that performance varies depending on fine-tuning a Conv layer through the layer-wise inspection. Lastly, (3) from overall experimental results, the base session learning is significant for lifelong learners to acquire generalized performances in FSCIL.

## 6 CONCLUSION

［#59］
Inspired by Regularized Lottery Ticket Hypothesis (RLTH), which hypothesizes that smooth subnet- works exist within a dense network, we propose Soft-SubNetworks (SoftNet); an incremental learning strategy that preserves the learned class knowledge and learns the newer ones. More specifically, SoftNet jointly learned the model weights and adaptive soft masks to minimize catastrophic forgetting and to avoid overfitting novel few samples in FSCIL. Finally, we compared a comprehensive empiri- cal study on SoftNet with multiple class incremental learning methods. Extensive experiments on benchmark tasks demonstrate how our method achieves superior performance over the state-of-the-art class incremental learning methodologies. We also discovered how subnetworks perform differently under specified architectures and datasets through ablation studies. In addition, we emphasized the importance of fine-tuning and base session learning in achieving optimum performance for FSCIL. We believe that our findings could bring a monumental on deep neural network architecture search, both on task-specific architectures and utilization of sparse models.


［#60］
Acknowledgement.This work was supported by Institute for Information & communications Tech-
nology Promotion (IITP) grant funded by the Korea government (MSIT) (No. 2021-0-01381, Develop-
ment of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to
Real Environments) and partly supported by Institute of Information & communications Technology
Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00184,
Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).






































































# A EXPERIMENTAL DETAILS

［#61］
We validate the effectiveness of the soft-subnetwork in our method on several benchmark datasets against various architecture-based methods for Few-Shot Class Incremental Learning (FSCIL). To proceed with the details of our experiments, we first explain the datasets and how we involve them in our experiments. Later, we detail experiment setups, including architecture details, preprocessing, and training budget.

## A.1 DATASETS

［#62］
The following datasets are utilized for comparisons:

［#63］
**CIFAR-100** In CIFAR-100, each class contains 500 images for training and 100 images for testing. Each image has a size of $32 \times 32$. Here, we follow an identical FSCIL procedure as in (Shi et al., 2021), where we divide the dataset into a base session with 60 base classes and eight novel sessions with a 5-way 5-shot problem on each session.

［#64］
**miniImageNet** miniImageNet consists of RGB images from 100 different classes, where each class contains 500 training images and 100 test images of size $84 \times 84$. Originally proposed for few-shot learning problems, miniImageNet is part of a much larger ImageNet dataset. Compared with CIFAR-100, the miniImageNet dataset is more complex and suitable for prototyping. The setup of miniImageNet is similar to that of CIFAR-100. To proceed with our evaluation, we follow the procedure described in (Shi et al., 2021), where we incorporate 60 base classes and eight novel sessions through 5-way 5-shot problems.

［#65］
**CUB-200-2011** CUB-200-2011 contains 200 fine-grained bird species with 11,788 images with varying images for each class. To proceed with experiments, we split the dataset into 6,000 training images and 6,000 test images as in (Tao et al., 2020). During training, We randomly crop each image to be of size $224 \times 224$. We fix the first 100 classes as base classes, where we utilize all samples in these respective classes to train the model. On the other hand, we treat the remaining 100 classes as novel categories split into ten novel sessions with a 10-way 5-shot problem in each session.

## A.2 EXPERIMENT SETUPS

［#66］
We begin this section by describing the setups used for experiments in CIFAR-100 and miniImageNet. After that, we proceed with a follow-up discussion on the configuration we employ for experiments involving the CUB-200-2011 dataset.

［#67］
**CIFAR-100 and miniImageNet.** For experiments in these two datasets, we are using NVIDIA GPU RTX8000 on CUDA 11.0. We randomly split these two datasets into multiple sessions, as described in the previous sub-section. We run each algorithm ten times for experiments on both datasets with a fixed split and report their mean accuracy. We adopt ResNet18 (He et al., 2016) as the backbone network. For data augmentation, we use standard random crop and horizontal flips. During the training stage in the base session, we select top-$c\%$ weights at each layer and acquire the optimal soft-subnetworks with the best validation accuracy. For each incremental few-shot learning session, we train our model for six epochs with a learning rate is 0.02. We train new class session samples using a few minor weights of the soft-subnetwork (conv4x layer of ResNet18 and conv3x layer of ResNet20) obtained by learning at the base session.

［#68］
**CUB-200-2011.** Besides experiments in the previous two datasets, we conducted an additional experiment on this dataset. We prepare this dataset following the split procedure described in the previous sub-section. We run each algorithm ten times and report their mean accuracy. We also adopt ResNet18 (He et al., 2016) as the backbone network and follow the same data augmentation as in the previous two datasets. We follow the same base-session training procedure as in the other two datasets. In each incremental few-shot learning session $t > 1$, the total number of training epochs is 10, and the learning rate is 0.1. We train new class session samples using a few minor weights of the soft-subnetwork (conv4x layer of ResNet18) obtained at the base session.

# B RESULTS AND CONCLUSIONS

［#69］
To expand upon the results of our paper, we conduct more experiments on various datasets mentioned in the previous section. We first display the full performance table with more capacity values $c$ employed towards our method in Table 9 and Table 11. Next, we identify how choosing a different architecture would impact the performance of our algorithm in Table 10. Furthermore, we analyze the performance of our method on the CUB-200-2011 dataset in Table 7.

［#70］
Through extensive experiments, we deduce the following three conclusions for incorporating our method in the few-shot class incremental learning:

［#71］
**Structure.** We identified a SubNetwork of ResNet18 and ResNet20 with varying capacities on CIFAR-100 for the 5-way 5-shot FSCIL setting as shown in Table 9 and Table 10. First, according to both tables, our method performs better as we use more parameters within our network. In addition, as denoted in our paper, we see how effective subnetwork is by observing how HardNet, with only 50% of its dense capacity, achieves comparable performance to methods utilizing dense networks, while SoftNet can do the same with only 30% of its dense capacity. Furthermore, we argue that our method is architecture-dependent. Our observation from Table 10 shows that at ResNet18, our architecture performs the best at the maximum capacity of $c=99\%$, while at ResNet20, we achieve the optimum performance at $c=90\%$.

［#72］
**Comparisions of Hard and SoftNet.** Furthermore, increasing the number of network parameters leads to better overall performance in both subnetworks types, as shown in Figure 4 and Figure 5. Subnetworks, in the form of HardNet and SoftNet, tend to retain prior (base) session knowledge denoted in dashed (---) line, and HardNet seems to be able to classify new session class samples without continuous updates stated in dashed-dot (-·-) line. From this, we could expect how much previous knowledge HardNet learned at the base session to help learn new incoming tasks (Forward Transfer). The overall performances of SoftNet are better than HardNet since SoftNet improves both base/new session knowledge by updating minor subnetworks. Subnetworks have a broader spectrum of performances on miniImageNet (Figure 5) than on CIFAR-100 (Figure 4). This could be an observation caused by the dataset complexity - i.e., if the miniImagenet dataset is more complex or harder to learn for a subnetwork or a deep model as such subnetworks need more parameters to learn miniImageNet than the CIFAR-100 dataset.

［#73］
![](./images/867754338234139070_10.jpg)
［#74］
*(a) $c=10.0\%$*

［#75］
![](./images/867754338234139070_11.jpg)
［#76］
*(b) $c=30.0\%$*

［#77］
![](./images/867754338234139070_12.jpg)
［#78］
*(c) $c=80.0\%$*

［#79］
**Figure 4: Performances of HardNet v.s. SoftNet on CIFAR-100 for 5-way 5-shot FSCIL:** the overall performance depends on capacity $c$ and the softness of subnetwork. Note that solid(---), dashed(---), and dashed-dot(-·-) lines denote overall, base, and novel class performances respectively.

［#80］
**Smoothness of SoftNet.** As emphasized in Table 11, SoftNet has a broader spectrum of performances than HardNet on miniImageNet. 20% of minor subnet might provide a smoother representation than HardNet because the performance of SoftNet was the best approximately at $c=80\%$. From these results, we could expect that model parameter smoothness guarantees quite competitive results. To support the claim, we prepared the loss landscapes of a dense neural network, HardNet, and SoftNet on two Hessian eigenvectors (Yao et al., 2020) as shown in Fig. 7. We observed the following points through simple experiments:

［#81］
From these results, we can expect how much knowledge the specified subnetworks can retain and acquire on each dataset.

---


［#82］
![](./images/867754338234139070_13.jpg)
［#83］
(a) $c = 10.0\%$

［#84］
![](./images/867754338234139070_14.jpg)
［#85］
(b) $c = 30.0\%$

［#86］
![](./images/867754338234139070_15.jpg)
［#87］
(c) $c = 80.0\%$

［#88］
Figure 5: Performances of HardNet v.s. SoftNet on miniImageNet for 5-way 5-shot FSCIL: the overall performance depends on capacity $c$ and the softness of subnetwork. Note that solid(——), dashed(- - - ), and dashed-dot(-·—) lines denote overall, base, and novel class performances respectively.

［#89］
- The loss landscapes of Subnetworks (HardNet and SoftNet) were flatter than those of dense neural networks.
- The minor subnet of SoftNet helped find a flat global minimum despite random scaling weights in the training process.

［#90］
Moreover, we compared the embeddings using t-SNE plots as shown in Figure 6. In t-SNE's 2D embedding spaces, the overall discriminative of SoftNet is better than that of HardNet in terms of base class set and novel class set. This $70\%$ of minor subnet affects SoftNet positively in base session training and offers good initialized weights in novel session training.

［#91］
![](./images/867754338234139070_16.jpg)
［#92］
(a) HardNet with $c=30.0\%$ @Session2

［#93］
![](./images/867754338234139070_17.jpg)
［#94］
(b) SoftNet with $c=30.0\%$ @Session2

［#95］
Figure 6: t-SNE Plots of HardNet v.s. SoftNet on miniImageNet for 5-way 5-shot FSCIL: t-SNE plots represent the embeddings of the even-numbered test class samples and compare one another. Note Session1 Class Set:$\{0,\cdots,59\}$ and Session2 Novel Class Set:$\{60,\cdots,64\}$.

［#96］
Preciseness. Regarding fine-grained and small-sized CUB200-2011 FSCIL settings, HardNet also shows comparable results with the baselines, and SoftNet outperforms others as denoted in Table 7. In this FSCIL setting, we acquired the best performances of SoftNet through the specific parameter selections. As of now, our SoftNet achieves state-of-the-art results on the three datasets.

## C CONVERGENCE OF SUBNETWORKS

［#97］
Convergences of HardNet and SoftNet. To interpret the convergence of SoftNet, we follow the Lipschitz-continuous objective gradients (Bottou et al., 2018): the objective function of dense networks $R : \mathbb{R}^d \to \mathbb{R}$ is continuously differentiable and the gradient function of $R$, namely, $\nabla R: \mathbb{R}^d \to \mathbb{R}^d$, Lipschitz continuous with Lipschitz constant $L > 0$, i.e.,

［#97］
$$
||\nabla R(\boldsymbol{\theta}) - \nabla R(\boldsymbol{\theta}')||_2 \leq L||\boldsymbol{\theta} - \boldsymbol{\theta}'|| \quad \text{for all } \{\boldsymbol{\theta}, \boldsymbol{\theta}'\} \subset \mathbb{R}^d. \tag{6}
$$

［#98］
![](./images/867754338234139070_18.jpg)

［#99］
(a) epoch=70, $c=10.0\%$

［#100］
![](./images/867754338234139070_19.jpg)

［#101］
(b) epoch=100, $c=10.0\%$

［#102］
Figure 7: **Loss landscapes of** DenseNet, HardNet, and SoftNet: Subnetworks provide a more flat global minimum than dense neural networks. To demonstrate the loss landscapes, we trained a simple three-layered, fully connected model (fc-4-25-30-3) on the Iris Flower dataset (which is three classification problem) for 100 epochs.

［#103］
Following the same formula, we define the Lipschitz-continuous objective gradients of subnetworks as follows:

［#103］
$$\|\nabla R(\boldsymbol{\theta} \odot \boldsymbol{m}) - \nabla R(\boldsymbol{\theta}' \odot \boldsymbol{m})\|_2 \leq L\|(\boldsymbol{\theta} - \boldsymbol{\theta}') \odot \boldsymbol{m}\| \quad \text{for all } \{\boldsymbol{\theta}, \boldsymbol{\theta}'\} \subset \mathbb{R}^d. \tag{7}$$

［#104］
where $\boldsymbol{m}$ is a binary mask. In comparison of Eq. 6 and 7, we use the theoretical analysis (Ye et al., 2020) where subnetwork achieve a faster rate of $R(\boldsymbol{\theta} \odot \boldsymbol{m}) = \mathcal{O}(1/\|\boldsymbol{m}\|_1^2)$ at most. The comparison is as follows:

［#104］
$$\frac{\|\nabla R(\boldsymbol{\theta} \odot \boldsymbol{m}) - \nabla R(\boldsymbol{\theta}' \odot \boldsymbol{m})\|_2}{\|(\boldsymbol{\theta} - \boldsymbol{\theta}') \odot \boldsymbol{m}\|} < \frac{\|\nabla R(\boldsymbol{\theta}) - \nabla R(\boldsymbol{\theta}')\|_2}{\|\boldsymbol{\theta} - \boldsymbol{\theta}'\|} \leq L \tag{8}$$

［#105］
The smaller the value is, the flatter the solution (loss landscape) has. The equation is established from the relationship $R(\boldsymbol{\theta} \odot \boldsymbol{m}) \ll R^*(\boldsymbol{\theta})$, where $R^*(\boldsymbol{\theta}$ denotes the best possible loss achievable by convex combinations of all parameters despite $\|(\boldsymbol{\theta} - \boldsymbol{\theta}') \odot \boldsymbol{m}\| < \|\boldsymbol{\theta} - \boldsymbol{\theta}'\|$. Furthermore, we have the following inequality if $\|R(\boldsymbol{\theta} \odot \boldsymbol{m}_{hard}) - R(\boldsymbol{\theta} \odot \boldsymbol{m}_{soft})\| \simeq 0$ and $\|\boldsymbol{m}_{hard}\| < \|\boldsymbol{m}_{soft}\|$:

［#105］
$$\frac{\|\nabla R(\boldsymbol{\theta} \odot \boldsymbol{m}_{hard}) - \nabla R(\boldsymbol{\theta}' \odot \boldsymbol{m}_{hard})\|_2}{\|(\boldsymbol{\theta} - \boldsymbol{\theta}') \odot \boldsymbol{m}_{hard}\|} \geq \frac{\|\nabla R(\boldsymbol{\theta} \odot \boldsymbol{m}_{soft}) - \nabla R(\boldsymbol{\theta}' \odot \boldsymbol{m}_{soft})\|_2}{\|(\boldsymbol{\theta} - \boldsymbol{\theta}') \odot \boldsymbol{m}_{soft}\|} \tag{9}$$

［#105］
where the equality holds iff $\|\boldsymbol{m}_{hard}\| = \|\boldsymbol{m}_{soft}\|$. We prepare the loss landscapes of Dense Network, Hard-WSN, and Soft-WSN as shown in Figure 7 as an example to support the inequality.

## D ADDITIONAL COMPARISONS WITH CURRENT WORKS

［#106］
**Comparisons with SOTA.** We compare SoftNet with the following state-of-art-methods on TOPIC class split (Tao et al., 2020) of three benchmark datasets - CIFAR100 (Table 5), miniImageNet (Table 6), and CUB-200-2011 (Table 7). We summarize the current FSCIL methods as follows:

［#107］
- **CEC** Zhang et al. (2021): The authors proposed a Continually Evolved Classifier (CEC) that employs a graph model to propagate context information between classifiers for adaptation.
- **LIMIT** Zhou et al. (2022): The authors proposed a new paradigm for FSCIL based on meta-learning by LearnIng Multi-phase Incremental Tasks (LIMIT), which synthesizes fake FSCIL tasks from the base dataset. Besides, LIMIT also constructs a calibration module based on a transformer,

［#107］
which calibrates the old class classifiers and new class prototypes into the same scale and fills in the semantic gap.

［#108］
- **MetaFSCIL** Chi et al. (2022): The authors proposed a bilevel optimization based on meta-learning to directly optimize the network to learn how to learn incrementally in the setting of FSCIL. Concretely, They proposed to sample sequences of incremental tasks from base classes for training to simulate the evaluation protocol. For each task, the model is learned using a meta-objective to perform fast adaptation without forgetting. Furthermore, they proposed a bi-directional guided modulation to modulate activations and reduce catastrophic forgetting.
- **C-FSCIL** Hersche et al. (2022): The authors proposed C-FSCIL, which is architecturally composed of a frozen meta-learned feature extractor, a trainable fixed-size fully connected layer, and a rewritable dynamically growing memory that stores as many vectors as the number of encountered classes.
- **Subspace Reg.** Akyürek et al. (2021): The authors presented a straightforward approach that enables using logistic regression classifiers for few-shot incremental learning. The key to this approach is a new family of subspace regularization schemes that encourage weight vectors for new classes to lie close to the subspace spanned by the weights of existing classes.
- **Entropy-Reg** Liu et al. (2022): The authors alternatively proposed using data-free replay to synthesize data by a generator without accessing real data.
- **ALICE** Peng et al. (2022): The authors proposed a method - Augmented Angular Loss Incremental Classification or ALICE - inspired by the similarity of the goals for FSCIL and modern face recognition systems. Instead of the commonly used cross-entropy loss, they proposed using the angular penalty loss to obtain well-clustered features in ALICE.

［#109］
Table 5: Classification accuracy of ResNet18 on CIFAR-100 for 5-way 5-shot incremental learning with the same class split as in TOPIC (Cheraghian et al., 2021). * denotes the results reported from Shi et al. (2021). $^\dagger$ represents our reproduced results.

［#110］
<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="9">sessions</th>
<th rowspan="2">The gap with cRT</th>
</tr>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
</tr>
</thead>
<tbody>
<tr>
<td>cRT Shi et al. (2021)*</td>
<td>72.28</td>
<td>69.58</td>
<td>65.16</td>
<td>61.41</td>
<td>58.83</td>
<td>55.87</td>
<td>53.28</td>
<td>51.38</td>
<td>49.51</td>
<td></td>
</tr>
<tr>
<td>Joint-training Shi et al. (2021)*</td>
<td>72.28</td>
<td>68.40</td>
<td>63.31</td>
<td>59.16</td>
<td>55.73</td>
<td>52.81</td>
<td>49.01</td>
<td>46.74</td>
<td>44.34</td>
<td>-5.17</td>
</tr>
<tr>
<td>Baseline Shi et al. (2021)</td>
<td>72.28</td>
<td>68.01</td>
<td>64.18</td>
<td>60.56</td>
<td>57.44</td>
<td>54.69</td>
<td>52.98</td>
<td>50.80</td>
<td>48.70</td>
<td>-0.81</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)*</td>
<td>72.05</td>
<td>65.35</td>
<td>61.55</td>
<td>57.83</td>
<td>54.61</td>
<td>51.74</td>
<td>49.71</td>
<td>47.49</td>
<td>45.03</td>
<td>-4.48</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)*</td>
<td>74.45</td>
<td>67.74</td>
<td>62.72</td>
<td>57.14</td>
<td>52.78</td>
<td>48.62</td>
<td>45.56</td>
<td>42.43</td>
<td>39.22</td>
<td>-10.29</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)*</td>
<td>72.28</td>
<td>63.84</td>
<td>59.64</td>
<td>55.49</td>
<td>53.21</td>
<td>51.77</td>
<td>50.93</td>
<td>48.94</td>
<td>46.96</td>
<td>-2.55</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)</td>
<td>64.10</td>
<td>53.28</td>
<td>41.69</td>
<td>34.13</td>
<td>27.93</td>
<td>25.06</td>
<td>20.41</td>
<td>15.48</td>
<td>13.73</td>
<td>-35.78</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)</td>
<td>64.10</td>
<td>53.05</td>
<td>43.96</td>
<td>36.97</td>
<td>31.61</td>
<td>26.73</td>
<td>21.23</td>
<td>16.78</td>
<td>13.54</td>
<td>-35.97</td>
</tr>
<tr>
<td>TOPIC Cheraghian et al. (2021)</td>
<td>64.10</td>
<td>55.88</td>
<td>47.07</td>
<td>45.16</td>
<td>40.11</td>
<td>36.38</td>
<td>33.96</td>
<td>31.55</td>
<td>29.37</td>
<td>-20.14</td>
</tr>
<tr>
<td>CEC Zhang et al. (2021)</td>
<td>73.07</td>
<td>68.88</td>
<td>65.26</td>
<td>61.19</td>
<td>58.09</td>
<td>55.57</td>
<td>53.22</td>
<td>51.34</td>
<td>49.14</td>
<td>-0.37</td>
</tr>
<tr>
<td>F2M Shi et al. (2021)</td>
<td>71.45</td>
<td>68.10</td>
<td>64.43</td>
<td>60.80</td>
<td>57.76</td>
<td>55.26</td>
<td>53.53</td>
<td>51.57</td>
<td>49.35</td>
<td>-0.16</td>
</tr>
<tr>
<td>LIMIT Zhou et al. (2022)</td>
<td>73.81</td>
<td>72.09</td>
<td>67.87</td>
<td>63.89</td>
<td>60.70</td>
<td>57.77</td>
<td>55.67</td>
<td>53.52</td>
<td>51.23</td>
<td>+1.72</td>
</tr>
<tr>
<td>MetaFSCIL Chi et al. (2022)</td>
<td>74.50</td>
<td>70.10</td>
<td>66.84</td>
<td>62.77</td>
<td>59.48</td>
<td>56.52</td>
<td>54.36</td>
<td>52.56</td>
<td>49.97</td>
<td>+0.46</td>
</tr>
<tr>
<td>ALICE Peng et al. (2022)</td>
<td>79.00</td>
<td>70.50</td>
<td>67.10</td>
<td>63.40</td>
<td>61.20</td>
<td>59.20</td>
<td>58.10</td>
<td>56.30</td>
<td>54.10</td>
<td>+4.59</td>
</tr>
<tr>
<td>Entropy-Reg Liu et al. (2022)</td>
<td>74.40</td>
<td>70.20</td>
<td>66.54</td>
<td>62.51</td>
<td>59.71</td>
<td>56.58</td>
<td>54.52</td>
<td>52.39</td>
<td>50.14</td>
<td>+0.63</td>
</tr>
<tr>
<td>C-FSCIL Hersche et al. (2022)</td>
<td>77.50</td>
<td>72.45</td>
<td>67.94</td>
<td>63.80</td>
<td>60.24</td>
<td>57.34</td>
<td>54.61</td>
<td>52.41</td>
<td>50.23</td>
<td>+0.72</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)</td>
<td>64.10</td>
<td>55.85</td>
<td>51.71</td>
<td>48.59</td>
<td>45.34</td>
<td>43.25</td>
<td>41.52</td>
<td>39.81</td>
<td>38.16</td>
<td>-11.35</td>
</tr>
<tr>
<td>FSLL+SS Mazumder et al. (2021)</td>
<td>66.76</td>
<td>55.52</td>
<td>52.20</td>
<td>49.17</td>
<td>46.23</td>
<td>44.64</td>
<td>43.07</td>
<td>41.20</td>
<td>39.57</td>
<td>-9.94</td>
</tr>
<tr>
<td>HardNet, $c=50\%$</td>
<td>78.35</td>
<td>74.12</td>
<td>70.13</td>
<td>65.88</td>
<td>62.74</td>
<td>59.56</td>
<td>57.98</td>
<td>56.31</td>
<td>54.32</td>
<td>+4.81</td>
</tr>
<tr>
<td>HardNet, $c=80\%$</td>
<td>79.27</td>
<td>75.38</td>
<td>71.11</td>
<td>66.68</td>
<td>63.32</td>
<td>60.06</td>
<td>58.16</td>
<td>56.40</td>
<td>54.31</td>
<td>+4.80</td>
</tr>
<tr>
<td>HardNet, $c=90\%$</td>
<td>79.22</td>
<td>74.77</td>
<td>70.89</td>
<td>66.41</td>
<td>62.90</td>
<td>59.48</td>
<td>58.10</td>
<td>56.13</td>
<td>53.92</td>
<td>+4.41</td>
</tr>
<tr>
<td>SoftNet, $c=50\%$</td>
<td>79.88</td>
<td>75.54</td>
<td>71.64</td>
<td>67.47</td>
<td>64.45</td>
<td>61.09</td>
<td>59.07</td>
<td>57.29</td>
<td>55.33</td>
<td>+5.82</td>
</tr>
<tr>
<td>SoftNet, $c=80\%$</td>
<td>80.33</td>
<td>76.23</td>
<td>72.19</td>
<td>67.83</td>
<td>64.64</td>
<td>61.39</td>
<td>59.32</td>
<td>57.37</td>
<td>54.94</td>
<td>+5.43</td>
</tr>
<tr>
<td>SoftNet, $c=90\%$</td>
<td>79.97</td>
<td>75.75</td>
<td>71.76</td>
<td>67.36</td>
<td>64.09</td>
<td>60.91</td>
<td>59.07</td>
<td>56.94</td>
<td>54.76</td>
<td>+5.25</td>
</tr>
</tbody>
</table>

［#111］
Leveraged by regularized backbone ResNet, SoftNet outperformed all existing current works on CIFAR100 as shown in Table 5. On miniImageNet Table 6 and CUB-200-201 Table 7, the performances of SoftNet were comparable with those of ALICE and LIMIT, considering that ALICE used class/data augmentations and LIMIT added an extra multi-head attention layer.

［#112］
Comparisions of SoftNet and AANet. Our SoftNet and AANet Liu et al. (2021) have proposed alleviating catastrophic forgetting in FSCIL and CIL, respectively. AANet consists of multi-ResNets: one residual block learns new knowledge while another fine-tunes to maintain the previously learned knowledge. Through the learnable scaling parameter for the linear combination of the multi-ResNet

［#113］
Published as a conference paper at ICLR 2023

［#114］
Table 6: Classification accuracy of ResNet18 on miniImageNet for 5-way 5-shot incremental learning with the same class split as in TOPIC (Cheraghian et al., 2021). * denotes results reported from Shi et al. (2021). $^\dagger$ represents our reproduced results.

［#115］
<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="9">sessions</th>
<th rowspan="2">The gap with cRT</th>
</tr>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
</tr>
</thead>
<tbody>
<tr>
<td>cRT Shi et al. (2021)*</td>
<td>72.08</td>
<td>68.15</td>
<td>63.06</td>
<td>61.12</td>
<td>56.57</td>
<td>54.47</td>
<td>51.81</td>
<td>49.86</td>
<td>48.31</td>
<td>-</td>
</tr>
<tr>
<td>Joint-training Shi et al. (2021)*</td>
<td>72.08</td>
<td>67.31</td>
<td>62.04</td>
<td>58.51</td>
<td>54.41</td>
<td>51.53</td>
<td>48.70</td>
<td>45.49</td>
<td>43.88</td>
<td>-4.43</td>
</tr>
<tr>
<td>Baseline Shi et al. (2021)</td>
<td>72.08</td>
<td>66.29</td>
<td>61.99</td>
<td>58.71</td>
<td>55.73</td>
<td>53.04</td>
<td>50.40</td>
<td>48.59</td>
<td>47.31</td>
<td>-1.0</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)*</td>
<td>71.77</td>
<td>61.85</td>
<td>58.12</td>
<td>54.60</td>
<td>51.49</td>
<td>48.47</td>
<td>45.90</td>
<td>44.19</td>
<td>42.71</td>
<td>-5.6</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)*</td>
<td>72.30</td>
<td>66.37</td>
<td>61.00</td>
<td>56.93</td>
<td>53.31</td>
<td>49.93</td>
<td>46.47</td>
<td>44.13</td>
<td>42.19</td>
<td>-6.12</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)*</td>
<td>72.08</td>
<td>59.04</td>
<td>53.75</td>
<td>51.17</td>
<td>49.11</td>
<td>47.21</td>
<td>45.35</td>
<td>44.06</td>
<td>43.65</td>
<td>-4.66</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)</td>
<td>61.31</td>
<td>46.32</td>
<td>42.94</td>
<td>37.63</td>
<td>30.49</td>
<td>24.00</td>
<td>20.89</td>
<td>18.80</td>
<td>17.21</td>
<td>-31.10</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)</td>
<td>61.31</td>
<td>47.80</td>
<td>39.31</td>
<td>31.91</td>
<td>25.68</td>
<td>21.35</td>
<td>18.67</td>
<td>17.24</td>
<td>14.17</td>
<td>-34.14</td>
</tr>
<tr>
<td>TOPIC Cheraghian et al. (2021)</td>
<td>61.31</td>
<td>50.09</td>
<td>45.17</td>
<td>41.16</td>
<td>37.48</td>
<td>35.52</td>
<td>32.19</td>
<td>29.46</td>
<td>24.42</td>
<td>-23.89</td>
</tr>
<tr>
<td>IDLVQ-C Chen and Lee (2020)</td>
<td>64.77</td>
<td>59.87</td>
<td>55.93</td>
<td>52.62</td>
<td>49.88</td>
<td>47.55</td>
<td>44.83</td>
<td>43.14</td>
<td>41.84</td>
<td>-6.47</td>
</tr>
<tr>
<td>CEC Zhang et al. (2021)</td>
<td>72.00</td>
<td>66.83</td>
<td>62.97</td>
<td>59.43</td>
<td>56.70</td>
<td>53.73</td>
<td>51.19</td>
<td>49.24</td>
<td>47.63</td>
<td>-0.68</td>
</tr>
<tr>
<td>F2M Shi et al. (2021)</td>
<td>72.05</td>
<td>67.47</td>
<td>63.16</td>
<td>59.70</td>
<td>56.71</td>
<td>53.77</td>
<td>51.11</td>
<td>49.21</td>
<td>47.84</td>
<td>-0.43</td>
</tr>
<tr>
<td>LIMIT Zhou et al. (2022)</td>
<td>73.81</td>
<td>72.09</td>
<td>67.87</td>
<td>63.89</td>
<td>60.70</td>
<td>57.77</td>
<td>55.67</td>
<td>53.52</td>
<td>51.23</td>
<td>+2.92</td>
</tr>
<tr>
<td>MetaFSCIL Chi et al. (2022)</td>
<td>72.04</td>
<td>67.94</td>
<td>63.77</td>
<td>60.29</td>
<td>57.58</td>
<td>55.16</td>
<td>52.90</td>
<td>50.79</td>
<td>49.19</td>
<td>+0.88</td>
</tr>
<tr>
<td>ALICE Peng et al. (2022)</td>
<td>80.60</td>
<td>70.60</td>
<td>67.40</td>
<td>64.50</td>
<td>62.50</td>
<td>60.00</td>
<td>57.80</td>
<td>56.80</td>
<td>55.70</td>
<td>+7.39</td>
</tr>
<tr>
<td>C-FSCIL Hersche et al. (2022)</td>
<td>76.40</td>
<td>71.14</td>
<td>66.46</td>
<td>63.29</td>
<td>60.42</td>
<td>57.46</td>
<td>54.78</td>
<td>53.11</td>
<td>51.41</td>
<td>+3.10</td>
</tr>
<tr>
<td>Entropy-Reg Liu et al. (2022)</td>
<td>71.84</td>
<td>67.12</td>
<td>63.21</td>
<td>59.77</td>
<td>57.01</td>
<td>53.95</td>
<td>51.55</td>
<td>49.52</td>
<td>48.21</td>
<td>-0.10</td>
</tr>
<tr>
<td>Subspace Reg. Akyürek et al. (2021)</td>
<td>80.37</td>
<td>71.69</td>
<td>66.94</td>
<td>62.53</td>
<td>58.90</td>
<td>55.00</td>
<td>51.94</td>
<td>49.76</td>
<td>46.79</td>
<td>-1.52</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)</td>
<td>66.48</td>
<td>61.75</td>
<td>58.16</td>
<td>54.16</td>
<td>51.10</td>
<td>48.53</td>
<td>46.54</td>
<td>44.20</td>
<td>42.28</td>
<td>-6.03</td>
</tr>
<tr>
<td>FSLL+SS Mazumder et al. (2021)</td>
<td>68.85</td>
<td>63.14</td>
<td>59.24</td>
<td>55.23</td>
<td>52.24</td>
<td>49.65</td>
<td>47.74</td>
<td>45.23</td>
<td>43.92</td>
<td>-4.39</td>
</tr>
<tr>
<td>HardNet, $c=80\%$</td>
<td>78.70</td>
<td>72.55</td>
<td>68.26</td>
<td>64.45</td>
<td>61.74</td>
<td>58.93</td>
<td>55.99</td>
<td>54.09</td>
<td>52.74</td>
<td>+4.43</td>
</tr>
<tr>
<td>HardNet, $c=87\%$</td>
<td>79.17</td>
<td>73.05</td>
<td>69.16</td>
<td>65.43</td>
<td>62.61</td>
<td>59.31</td>
<td>56.73</td>
<td>54.69</td>
<td>53.47</td>
<td>+5.16</td>
</tr>
<tr>
<td>HardNet, $c=90\%$</td>
<td>79.15</td>
<td>72.03</td>
<td>68.76</td>
<td>65.32</td>
<td>62.00</td>
<td>58.21</td>
<td>56.52</td>
<td>53.66</td>
<td>53.07</td>
<td>+4.76</td>
</tr>
<tr>
<td>SoftNet, $c=80\%$</td>
<td>79.37</td>
<td>74.31</td>
<td>69.89</td>
<td>66.16</td>
<td>63.40</td>
<td>60.75</td>
<td>57.62</td>
<td>55.67</td>
<td>54.34</td>
<td>+6.03</td>
</tr>
<tr>
<td>SoftNet, $c=87\%$</td>
<td>79.77</td>
<td>75.08</td>
<td>70.59</td>
<td>66.93</td>
<td>64.00</td>
<td>61.00</td>
<td>57.81</td>
<td>55.81</td>
<td>54.68</td>
<td>+6.37</td>
</tr>
<tr>
<td>SoftNet, $c=90\%$</td>
<td>79.72</td>
<td>74.25</td>
<td>70.00</td>
<td>66.35</td>
<td>63.19</td>
<td>60.04</td>
<td>57.36</td>
<td>55.38</td>
<td>54.14</td>
<td>+5.83</td>
</tr>
</tbody>
</table>

［#116］
Table 7: Classification accuracy of ResNet18 on CUB-200-2011 for 10-way 5-shot incremental learning (TOPIC class split Tao et al. (2020)). * denotes results reported from Shi et al. (2021). $^\dagger$ represents our reproduced results.

［#117］
<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="11">sessions</th>
<th rowspan="2">The gap with cRT</th>
</tr>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
<th>10</th>
<th>11</th>
</tr>
</thead>
<tbody>
<tr>
<td>cRT Shi et al. (2021)*</td>
<td>77.16</td>
<td>74.41</td>
<td>71.31</td>
<td>68.08</td>
<td>65.57</td>
<td>63.08</td>
<td>62.44</td>
<td>61.29</td>
<td>60.12</td>
<td>59.85</td>
<td>59.30</td>
<td>-</td>
</tr>
<tr>
<td>Joint-training Shi et al. (2021)</td>
<td>77.16</td>
<td>74.39</td>
<td>69.83</td>
<td>67.17</td>
<td>64.72</td>
<td>62.25</td>
<td>59.77</td>
<td>59.05</td>
<td>57.99</td>
<td>57.81</td>
<td>56.82</td>
<td>-2.48</td>
</tr>
<tr>
<td>Baseline Shi et al. (2021)</td>
<td>77.16</td>
<td>74.00</td>
<td>70.21</td>
<td>66.07</td>
<td>63.90</td>
<td>61.35</td>
<td>60.01</td>
<td>58.66</td>
<td>56.33</td>
<td>56.12</td>
<td>55.07</td>
<td>-4.23</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)*</td>
<td>75.95</td>
<td>60.90</td>
<td>57.65</td>
<td>54.51</td>
<td>50.83</td>
<td>48.21</td>
<td>46.95</td>
<td>45.74</td>
<td>43.21</td>
<td>43.01</td>
<td>41.27</td>
<td>-18.03</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)*</td>
<td>77.44</td>
<td>58.10</td>
<td>50.15</td>
<td>44.80</td>
<td>39.12</td>
<td>34.44</td>
<td>31.73</td>
<td>29.75</td>
<td>27.56</td>
<td>26.93</td>
<td>25.30</td>
<td>-34.00</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)*</td>
<td>77.16</td>
<td>71.85</td>
<td>66.53</td>
<td>59.95</td>
<td>58.01</td>
<td>57.00</td>
<td>56.06</td>
<td>54.78</td>
<td>52.24</td>
<td>52.01</td>
<td>51.47</td>
<td>-7.83</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)</td>
<td>68.68</td>
<td>52.65</td>
<td>48.61</td>
<td>44.16</td>
<td>36.62</td>
<td>29.52</td>
<td>27.83</td>
<td>26.26</td>
<td>24.01</td>
<td>23.89</td>
<td>21.16</td>
<td>-39.92</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)</td>
<td>68.68</td>
<td>57.12</td>
<td>44.21</td>
<td>28.78</td>
<td>26.71</td>
<td>25.66</td>
<td>24.62</td>
<td>21.52</td>
<td>20.12</td>
<td>20.06</td>
<td>19.87</td>
<td>-41.21</td>
</tr>
<tr>
<td>TOPIC Cheraghian et al. (2021)</td>
<td>68.68</td>
<td>62.49</td>
<td>54.81</td>
<td>49.99</td>
<td>45.25</td>
<td>41.40</td>
<td>38.35</td>
<td>35.36</td>
<td>32.22</td>
<td>28.31</td>
<td>26.28</td>
<td>-34.80</td>
</tr>
<tr>
<td>SPPR Zhu et al. (2021)</td>
<td>68.68</td>
<td>61.85</td>
<td>57.43</td>
<td>52.68</td>
<td>50.19</td>
<td>46.88</td>
<td>44.65</td>
<td>43.07</td>
<td>40.17</td>
<td>39.63</td>
<td>37.33</td>
<td>-21.97</td>
</tr>
<tr>
<td>CEC Zhang et al. (2021)</td>
<td>75.85</td>
<td>71.94</td>
<td>68.50</td>
<td>63.50</td>
<td>62.43</td>
<td>58.27</td>
<td>57.73</td>
<td>55.81</td>
<td>54.83</td>
<td>53.52</td>
<td>52.28</td>
<td>-7.02</td>
</tr>
<tr>
<td>F2M Shi et al. (2021)</td>
<td>77.13</td>
<td>73.92</td>
<td>70.27</td>
<td>66.37</td>
<td>64.34</td>
<td>61.69</td>
<td>60.52</td>
<td>59.38</td>
<td>57.15</td>
<td>56.94</td>
<td>55.89</td>
<td>-3.41</td>
</tr>
<tr>
<td>LIMIT Zhou et al. (2022)</td>
<td>75.89</td>
<td>73.55</td>
<td>71.99</td>
<td>68.14</td>
<td>67.42</td>
<td>63.61</td>
<td>62.40</td>
<td>61.35</td>
<td>59.91</td>
<td>58.66</td>
<td>57.41</td>
<td>-1.89</td>
</tr>
<tr>
<td>MetaFSCIL Chi et al. (2022)</td>
<td>75.90</td>
<td>72.41</td>
<td>68.78</td>
<td>64.78</td>
<td>62.96</td>
<td>59.99</td>
<td>58.30</td>
<td>56.85</td>
<td>54.78</td>
<td>53.82</td>
<td>52.64</td>
<td>-6.66</td>
</tr>
<tr>
<td>ALICE Peng et al. (2022)</td>
<td>77.40</td>
<td>72.70</td>
<td>70.60</td>
<td>67.20</td>
<td>65.90</td>
<td>63.40</td>
<td>62.90</td>
<td>61.90</td>
<td>60.50</td>
<td>60.60</td>
<td>60.10</td>
<td>-0.02</td>
</tr>
<tr>
<td>Entropy-Reg Liu et al. (2022)</td>
<td>75.90</td>
<td>72.14</td>
<td>68.64</td>
<td>63.76</td>
<td>62.58</td>
<td>59.11</td>
<td>57.82</td>
<td>55.89</td>
<td>54.92</td>
<td>53.58</td>
<td>52.39</td>
<td>-6.91</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)</td><td>72.77</td><td>69.33</td><td>65.51</td><td>62.66</td><td>61.10</td><td>58.65</td><td>57.78</td><td>57.26</td><td>55.59</td><td>55.39</td><td>54.21</td>
<td>-6.87</td>
</tr>
<tr>
<td>FSLL+SS Mazumder et al. (2021)</td>
<td>75.63</td>
<td>71.81</td>
<td>68.16</td>
<td>64.32</td>
<td>62.61</td>
<td>60.10</td>
<td>58.82</td>
<td>58.70</td>
<td>56.56</td>
<td>56.41</td>
<td>55.82</td>
<td>-5.26</td>
</tr>
<tr>
<td>HardNet, $c=88\%$</td>
<td>76.89</td>
<td>73.40</td>
<td>69.77</td>
<td>66.15</td>
<td>64.00</td>
<td>60.98</td>
<td>59.56</td>
<td>58.05</td>
<td>56.05</td>
<td>55.84</td>
<td>55.20</td>
<td>-4.10</td>
</tr>
<tr>
<td>HardNet, $c=90\%$</td>
<td>77.23</td>
<td>73.62</td>
<td>70.20</td>
<td>66.36</td>
<td>64.32</td>
<td>61.40</td>
<td>59.86</td>
<td>58.28</td>
<td>56.36</td>
<td>55.88</td>
<td>55.30</td>
<td>-4.00</td>
</tr>
<tr>
<td>HardNet, $c=93\%$</td>
<td>77.76</td>
<td>73.97</td>
<td>70.41</td>
<td>66.60</td>
<td>64.47</td>
<td>61.35</td>
<td>59.80</td>
<td>58.18</td>
<td>56.17</td>
<td>55.73</td>
<td>55.18</td>
<td>-4.12</td>
</tr>
<tr>
<td>SoftNet, $c=88\%$</td>
<td>78.14</td>
<td>74.61</td>
<td>71.28</td>
<td>67.46</td>
<td>65.14</td>
<td>62.39</td>
<td>60.84</td>
<td>59.17</td>
<td>57.41</td>
<td>57.12</td>
<td>56.64</td>
<td>-2.66</td>
</tr>
<tr>
<td>SoftNet, $c=90\%$</td>
<td>78.07</td>
<td>74.58</td>
<td>71.37</td>
<td>67.54</td>
<td>65.37</td>
<td>62.60</td>
<td>61.07</td>
<td>59.37</td>
<td>57.53</td>
<td>57.21</td>
<td>56.75</td>
<td>-2.55</td>
</tr>
<tr>
<td>SoftNet, $c=93\%$</td>
<td>78.11</td>
<td>74.51</td>
<td>71.14</td>
<td>62.27</td>
<td>65.14</td>
<td>62.27</td>
<td>60.77</td>
<td>59.03</td>
<td>57.13</td>
<td>56.77</td>
<td>56.28</td>
<td>-3.02</td>
</tr>
</tbody>
</table>

［#118］
features, AANet showed outstanding performances in the CSIL setting. However, AANet tends to overfit since the ResNet block's parameters are fully used to update a few new class data in FSCIL. This point makes it difficult to train AANet on a few samples even though the performance at session 1 is comparable with SoftNet as shown in Table 8.


［#119］
Table 8: Classification accuracy of ResNet18 on CIFAR-100 for 5-way 5-shot incremental learning with the same class split as in TOPIC (Cheraghian et al., 2021). * denotes the results reported from Shi et al. (2021). $^\dagger$ represents our reproduced results.

［#120］
<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="9">sessions</th>
<th rowspan="2">The gap with cRT</th>
</tr>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
</tr>
</thead>
<tbody>
<tr>
<td>cRT Shi et al. (2021)*</td>
<td>72.28</td>
<td>69.58</td>
<td>65.16</td>
<td>61.41</td>
<td>58.83</td>
<td>55.87</td>
<td>53.28</td>
<td>51.38</td>
<td>49.51</td>
<td></td>
</tr>
<tr>
<td>AANet Liu et al. (2021)†</td>
<td>79.05</td>
<td>67.52</td>
<td>62.33</td>
<td>56.10</td>
<td>51.92</td>
<td>45.92</td>
<td>45.92</td>
<td>48.38</td>
<td>47.21</td>
<td>-2.30</td>
</tr>
<tr>
<td>HardNet, $c=50\%$</td>
<td>78.35</td>
<td>74.12</td>
<td>70.13</td>
<td>65.88</td>
<td>62.74</td>
<td>59.56</td>
<td>57.98</td>
<td>56.31</td>
<td>54.32</td>
<td>+4.81</td>
</tr>
<tr>
<td>HardNet, $c=80\%$</td>
<td>79.27</td>
<td>75.38</td>
<td>71.11</td>
<td>66.68</td>
<td>63.32</td>
<td>60.06</td>
<td>58.16</td>
<td>56.40</td>
<td>54.31</td>
<td>+4.80</td>
</tr>
<tr>
<td>HardNet, $c=90\%$</td>
<td>79.22</td>
<td>74.77</td>
<td>70.89</td>
<td>66.41</td>
<td>62.90</td>
<td>59.48</td>
<td>58.10</td>
<td>56.13</td>
<td>53.92</td>
<td>+4.41</td>
</tr>
<tr>
<td>SoftNet, $c=50\%$</td>
<td>79.88</td>
<td>75.54</td>
<td>71.64</td>
<td>67.47</td>
<td>64.45</td>
<td>61.09</td>
<td>59.07</td>
<td>57.29</td>
<td>55.33</td>
<td>+5.82</td>
</tr>
<tr>
<td>SoftNet, $c=80\%$</td>
<td>80.33</td>
<td>76.23</td>
<td>72.19</td>
<td>67.83</td>
<td>64.64</td>
<td>61.39</td>
<td>59.32</td>
<td>57.37</td>
<td>54.94</td>
<td>+5.43</td>
</tr>
<tr>
<td>SoftNet, $c=90\%$</td>
<td>79.97</td>
<td>75.75</td>
<td>71.76</td>
<td>67.36</td>
<td>64.09</td>
<td>60.91</td>
<td>59.07</td>
<td>56.94</td>
<td>54.76</td>
<td>+5.25</td>
</tr>
</tbody>
</table>

## E LIMITATIONS AND FUTURE WORKS

［#121］
Our method employs two sets of subnetworks. One is the major subnetworks, whereas the other is minor subnets. Since the former serve their duty to retain the base session knowledge, once the major subnetwork is tuned, there could be a potential loss of previously acquired knowledge. Furthermore, we explicitly divide SoftNet by the magnitude criterion. As a result, when SoftNet parameters are exposed, the essential parameters will be vulnerable to intentional attacks. It could result in the leak of knowledge maintained by SoftNet. First, to avoid tunning the major subnetwork issue, the new session learner should know the model sparsity for maintaining base session knowledge. Second, to address the leaking information issue, the binary mask should be encoded by a compression method to reduce model capacity and protect the privacy of task knowledge. Moreover, in FSCIL tasks, SoftNet alleviates overfitting issues while effectively maintaining base-session performance. In future work, we consider expanding the model parameters to acquire a long sequence of incoming new class knowledge depending on the data or task size, i.e., CIL tasks.


［#122］
Table 9: Classification accuracy of ResNet18 on CIFAR-100 for 5-way 5-shot incremental learning.
Underbar denotes the comparable results with baseline. * denotes the results reported from Shi et al.
(2021).

［#123］
<table>
 <thead>
  <tr>
   <th rowspan="2">Method</th>
   <th colspan="9">sessions</th>
   <th rowspan="2">The gap<br/>with cRT</th>
  </tr>
  <tr>
   <th>1</th>
   <th>2</th>
   <th>3</th>
   <th>4</th>
   <th>5</th>
   <th>6</th>
   <th>7</th>
   <th>8</th>
   <th>9</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>cRT Shi et al. (2021)</td>
   <td>65.18</td>
   <td>63.89</td>
   <td>60.20</td>
   <td>57.23</td>
   <td>53.71</td>
   <td>50.39</td>
   <td>48.77</td>
   <td>47.29</td>
   <td>45.28</td>
   <td>-</td>
  </tr>
  <tr>
   <td>Joint-training Shi et al. (2021)</td>
   <td>65.18</td>
   <td>61.45</td>
   <td>57.36</td>
   <td>53.68</td>
   <td>50.84</td>
   <td>47.33</td>
   <td>44.79</td>
   <td>42.62</td>
   <td>40.08</td>
   <td>-5.20</td>
  </tr>
  <tr>
   <td>Baseline Shi et al. (2021)</td>
   <td>65.18</td>
   <td>61.67</td>
   <td>58.61</td>
   <td>55.11</td>
   <td>51.86</td>
   <td>49.43</td>
   <td>47.60</td>
   <td>45.64</td>
   <td>43.83</td>
   <td>-1.45</td>
  </tr>
  <tr>
   <td>iCaRL Rebuffi et al. (2017)*</td>
   <td>66.52</td>
   <td>57.26</td>
   <td>54.27</td>
   <td>50.62</td>
   <td>47.33</td>
   <td>44.99</td>
   <td>43.14</td>
   <td>41.16</td>
   <td>39.49</td>
   <td>-5.79</td>
  </tr>
  <tr>
   <td>Rebalance Hou et al. (2019)*</td>
   <td>66.66</td>
   <td>61.42</td>
   <td>57.29</td>
   <td>53.02</td>
   <td>48.85</td>
   <td>45.68</td>
   <td>43.06</td>
   <td>40.56</td>
   <td>38.35</td>
   <td>-6.93</td>
  </tr>
  <tr>
   <td>FSLL Mazumder et al. (2021)*</td>
   <td>65.18</td>
   <td>56.24</td>
   <td>54.55</td>
   <td>51.61</td>
   <td>49.11</td>
   <td>47.27</td>
   <td>45.35</td>
   <td>43.95</td>
   <td>42.22</td>
   <td>-3.08</td>
  </tr>
  <tr>
   <td>iCaRL Rebuffi et al. (2017)</td>
   <td>64.10</td>
   <td>53.28</td>
   <td>41.69</td>
   <td>34.13</td>
   <td>27.93</td>
   <td>25.06</td>
   <td>20.41</td>
   <td>15.48</td>
   <td>13.73</td>
   <td>-31.55</td>
  </tr>
  <tr>
   <td>Rebalance Hou et al. (2019)</td>
   <td>64.10</td>
   <td>53.05</td>
   <td>43.96</td>
   <td>36.97</td>
   <td>31.61</td>
   <td>26.73</td>
   <td>21.23</td>
   <td>16.78</td>
   <td>13.54</td>
   <td>-31.74</td>
  </tr>
  <tr>
   <td>TOPIC Cheraghian et al. (2021)</td>
   <td>64.10</td>
   <td>55.88</td>
   <td>47.07</td>
   <td>45.16</td>
   <td>40.11</td>
   <td>36.38</td>
   <td>33.96</td>
   <td>31.55</td>
   <td>29.37</td>
   <td>-15.91</td>
  </tr>
  <tr>
   <td>F2M Shi et al. (2021)</td>
   <td>64.71</td>
   <td>62.05</td>
   <td>59.01</td>
   <td>55.58</td>
   <td>52.55</td>
   <td>49.96</td>
   <td>48.08</td>
   <td>46.28</td>
   <td>44.67</td>
   <td>-0.61</td>
  </tr>
  <tr>
   <td>FSLL Mazumder et al. (2021)</td>
   <td>64.10</td>
   <td>55.85</td>
   <td>51.71</td>
   <td>48.59</td>
   <td>45.34</td>
   <td>43.25</td>
   <td>41.52</td>
   <td>39.81</td>
   <td>38.16</td>
   <td>-7.12</td>
  </tr>
  <tr>
   <td>HardNet, $c = 10\%$</td>
   <td>28.97</td>
   <td>27.71</td>
   <td>26.08</td>
   <td>24.68</td>
   <td>23.34</td>
   <td>22.02</td>
   <td>21.12</td>
   <td>20.48</td>
   <td>19.50</td>
   <td>-25.78</td>
  </tr>
  <tr>
   <td>HardNet, $c = 20\%$</td>
   <td>37.42</td>
   <td>35.29</td>
   <td>33.22</td>
   <td>31.32</td>
   <td>29.51</td>
   <td>27.80</td>
   <td>26.54</td>
   <td>25.28</td>
   <td>24.16</td>
   <td>-21.12</td>
  </tr>
  <tr>
   <td>HardNet, $c = 30\%$</td>
   <td>55.47</td>
   <td>52.37</td>
   <td>49.38</td>
   <td>46.53</td>
   <td>43.88</td>
   <td>41.50</td>
   <td>39.58</td>
   <td>37.82</td>
   <td>36.06</td>
   <td>-9.22</td>
  </tr>
  <tr>
   <td>HardNet, $c = 40\%$</td>
   <td>57.52</td>
   <td>53.85</td>
   <td>50.62</td>
   <td>47.74</td>
   <td>44.90</td>
   <td>42.64</td>
   <td>40.76</td>
   <td>38.95</td>
   <td>37.07</td>
   <td>-8.21</td>
  </tr>
  <tr>
   <td>HardNet, $c = 50\%$</td>
   <td>64.80</td>
   <td>60.77</td>
   <td>56.95</td>
   <td>53.53</td>
   <td>50.40</td>
   <td>47.82</td>
   <td>45.93</td>
   <td>43.95</td>
   <td>41.91</td>
   <td>-3.37</td>
  </tr>
  <tr>
   <td>HardNet, $c = 60\%$</td>
   <td>66.72</td>
   <td>62.21</td>
   <td>58.14</td>
   <td>54.60</td>
   <td>51.47</td>
   <td>48.86</td>
   <td>46.67</td>
   <td>44.67</td>
   <td>42.66</td>
   <td>-2.62</td>
  </tr>
  <tr>
   <td>HardNet, $c = 70\%$</td>
   <td>68.27</td>
   <td>63.52</td>
   <td>59.45</td>
   <td>55.89</td>
   <td>52.91</td>
   <td>50.30</td>
   <td>48.27</td>
   <td>46.25</td>
   <td>44.22</td>
   <td>-1.06</td>
  </tr>
  <tr>
   <td>HardNet, $c = 80\%$</td>
   <td>69.65</td>
   <td>64.60</td>
   <td>60.59</td>
   <td>56.93</td>
   <td>53.60</td>
   <td>50.80</td>
   <td>48.69</td>
   <td>46.69</td>
   <td>44.63</td>
   <td>-0.65</td>
  </tr>
  <tr>
   <td>HardNet, $c = 90\%$</td>
   <td>70.85</td>
   <td>65.84</td>
   <td>61.59</td>
   <td>57.92</td>
   <td>54.65</td>
   <td>51.90</td>
   <td>49.79</td>
   <td>47.66</td>
   <td>45.47</td>
   <td>+0.19</td>
  </tr>
  <tr>
   <td>HardNet, $c = 93\%$</td>
   <td>71.22</td>
   <td>66.20</td>
   <td>62.00</td>
   <td>58.34</td>
   <td>55.04</td>
   <td>52.34</td>
   <td>50.22</td>
   <td>48.07</td>
   <td>46.04</td>
   <td>+0.76</td>
  </tr>
  <tr>
   <td>HardNet, $c = 95\%$</td>
   <td>71.73</td>
   <td>66.31</td>
   <td>62.17</td>
   <td>58.44</td>
   <td>54.98</td>
   <td>52.20</td>
   <td>50.17</td>
   <td>47.97</td>
   <td>45.87</td>
   <td>+0.59</td>
  </tr>
  <tr>
   <td>HardNet, $c = 97\%$</td>
   <td>71.85</td>
   <td>66.48</td>
   <td>62.29</td>
   <td>58.62</td>
   <td>55.36</td>
   <td>52.55</td>
   <td>50.60</td>
   <td>48.43</td>
   <td>46.22</td>
   <td>+0.94</td>
  </tr>
  <tr>
   <td>HardNet, $c = 99\%$</td>
   <td>71.95</td>
   <td>66.83</td>
   <td>62.75</td>
   <td>59.09</td>
   <td>55.92</td>
   <td>53.03</td>
   <td>50.78</td>
   <td>48.52</td>
   <td>46.31</td>
   <td>+1.03</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 10\%$</td>
   <td>60.77</td>
   <td>57.02</td>
   <td>53.62</td>
   <td>50.51</td>
   <td>47.67</td>
   <td>45.14</td>
   <td>43.32</td>
   <td>41.6</td>
   <td>39.58</td>
   <td>-5.70</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 20\%$</td>
   <td>64.67</td>
   <td>60.69</td>
   <td>57.15</td>
   <td>53.77</td>
   <td>50.76</td>
   <td>48.28</td>
   <td>46.24</td>
   <td>44.23</td>
   <td>42.31</td>
   <td>-2.97</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 30\%$</td>
   <td>67.00</td>
   <td>62.18</td>
   <td>58.22</td>
   <td>54.69</td>
   <td>51.82</td>
   <td>49.12</td>
   <td>47.13</td>
   <td>44.98</td>
   <td>42.44</td>
   <td>-2.84</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 40\%$</td>
   <td>67.50</td>
   <td>63.11</td>
   <td>59.29</td>
   <td>55.61</td>
   <td>52.53</td>
   <td>49.85</td>
   <td>47.85</td>
   <td>45.84</td>
   <td>43.85</td>
   <td>-1.43</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 50\%$</td>
   <td>69.20</td>
   <td>64.18</td>
   <td>60.01</td>
   <td>56.43</td>
   <td>53.11</td>
   <td>50.62</td>
   <td>48.60</td>
   <td>46.51</td>
   <td>44.61</td>
   <td>-0.67</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 60\%$</td>
   <td>69.15</td>
   <td>63.68</td>
   <td>59.54</td>
   <td>56.05</td>
   <td>52.72</td>
   <td>50.10</td>
   <td>48.20</td>
   <td>46.18</td>
   <td>44.15</td>
   <td>-1.13</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 70\%$</td>
   <td>70.92</td>
   <td>65.16</td>
   <td>61.00</td>
   <td>57.25</td>
   <td>54.09</td>
   <td>51.37</td>
   <td>49.29</td>
   <td>47.03</td>
   <td>44.90</td>
   <td>-0.38</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 80\%$</td>
   <td>70.38</td>
   <td>65.04</td>
   <td>60.94</td>
   <td>57.26</td>
   <td>54.13</td>
   <td>51.58</td>
   <td>49.52</td>
   <td>47.36</td>
   <td>45.16</td>
   <td>-0.12</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 90\%$</td>
   <td>72.25</td>
   <td>66.82</td>
   <td>62.63</td>
   <td>58.98</td>
   <td>55.64</td>
   <td>52.77</td>
   <td>50.71</td>
   <td>48.42</td>
   <td>46.15</td>
   <td>+0.87</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 93\%$</td>
   <td>71.38</td>
   <td>65.93</td>
   <td>61.89</td>
   <td>58.20</td>
   <td>54.87</td>
   <td>51.83</td>
   <td>49.82</td>
   <td>47.57</td>
   <td>45.47</td>
   <td>+0.19</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 95\%$</td>
   <td>72.23</td>
   <td>66.94</td>
   <td>62.56</td>
   <td>58.84</td>
   <td>55.65</td>
   <td>52.74</td>
   <td>50.61</td>
   <td>48.47</td>
   <td>46.27</td>
   <td>+0.99</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 97\%$</td>
   <td>70.88</td>
   <td>65.72</td>
   <td>61.38</td>
   <td>57.88</td>
   <td>54.63</td>
   <td>51.82</td>
   <td>49.57</td>
   <td>47.30</td>
   <td>45.19</td>
   <td>-0.09</td>
  </tr>
  <tr>
   <td>SoftNet, $c = 99\%$</td>
   <td>72.62</td>
   <td>67.31</td>
   <td>63.05</td>
   <td>59.39</td>
   <td>56.00</td>
   <td>53.23</td>
   <td>51.06</td>
   <td>48.83</td>
   <td>46.63</td>
   <td>+1.35</td>
  </tr>
 </tbody>
</table>

［#124］
Table 10: Classification accuracy of ResNet18 v.s. ResNet20 on CIFAR-100 for 5-way 5-shot FSCIL
with varying capacity $c$. Underbar denotes the comparable results with baseline. $^*$ denotes results
reported from Shi et al. (2021).

［#125］
<table>
 <thead>
  <tr>
   <th rowspan="2">Method</th>
   <th colspan="9">sessions</th>
   <th rowspan="2">The gap<br>with cRT</th>
  </tr>
  <tr>
   <th>1</th>
   <th>2</th>
   <th>3</th>
   <th>4</th>
   <th>5</th>
   <th>6</th>
   <th>7</th>
   <th>8</th>
   <th>9</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>ResNet18, cRT Shi et al. (2021)</td>
   <td>65.18</td>
   <td>63.89</td>
   <td>60.20</td>
   <td>57.23</td>
   <td>53.71</td>
   <td>50.39</td>
   <td>48.77</td>
   <td>47.29</td>
   <td>45.28</td>
   <td>-</td>
  </tr>
  <tr>
   <td>ResNet18, Joint-training Shi et al. (2021)</td>
   <td>65.18</td>
   <td>61.45</td>
   <td>57.36</td>
   <td>53.68</td>
   <td>50.84</td>
   <td>47.33</td>
   <td>44.79</td>
   <td>42.62</td>
   <td>40.08</td>
   <td>-5.20</td>
  </tr>
  <tr>
   <td>ResNet18, Baseline Shi et al. (2021)</td>
   <td>65.18</td>
   <td>61.67</td>
   <td>58.61</td>
   <td>55.11</td>
   <td>51.86</td>
   <td>49.43</td>
   <td>47.60</td>
   <td>45.64</td>
   <td>43.83</td>
   <td>-1.45</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {10\%}$</td>
   <td>28.97</td>
   <td>27.71</td>
   <td>26.08</td>
   <td>24.68</td>
   <td>23.34</td>
   <td>22.02</td>
   <td>21.12</td>
   <td>20.48</td>
   <td>19.50</td>
   <td>-25.78</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {20\%}$</td>
   <td>37.42</td>
   <td>35.29</td>
   <td>33.22</td>
   <td>31.32</td>
   <td>29.51</td>
   <td>27.80</td>
   <td>26.54</td>
   <td>25.28</td>
   <td>24.16</td>
   <td>-21.12</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {30\%}$</td>
   <td>55.47</td>
   <td>52.37</td>
   <td>49.38</td>
   <td>46.53</td>
   <td>43.88</td>
   <td>41.50</td>
   <td>39.58</td>
   <td>37.82</td>
   <td>36.06</td>
   <td>-9.22</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {40\%}$</td>
   <td>57.52</td>
   <td>53.85</td>
   <td>50.62</td>
   <td>47.74</td>
   <td>44.90</td>
   <td>42.64</td>
   <td>40.76</td>
   <td>38.95</td>
   <td>37.07</td>
   <td>-8.21</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {50\%}$</td>
   <td>64.80</td>
   <td>60.77</td>
   <td>56.95</td>
   <td>53.53</td>
   <td>50.40</td>
   <td>47.82</td>
   <td>45.93</td>
   <td>43.95</td>
   <td>41.91</td>
   <td>-3.37</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {60\%}$</td>
   <td>66.72</td>
   <td>62.21</td>
   <td>58.14</td>
   <td>54.60</td>
   <td>51.47</td>
   <td>48.86</td>
   <td>46.67</td>
   <td>44.67</td>
   <td>42.66</td>
   <td>-2.62</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {70\%}$</td>
   <td>68.27</td>
   <td>63.52</td>
   <td>59.45</td>
   <td>55.89</td>
   <td>52.91</td>
   <td>50.30</td>
   <td>48.27</td>
   <td>46.25</td>
   <td>44.22</td>
   <td>-1.06</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {80\%}$</td>
   <td>69.65</td>
   <td>64.60</td>
   <td>60.59</td>
   <td>56.93</td>
   <td>53.60</td>
   <td>50.80</td>
   <td>48.69</td>
   <td>46.69</td>
   <td>44.63</td>
   <td>-0.65</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {90\%}$</td>
   <td>70.85</td>
   <td>65.84</td>
   <td>61.59</td>
   <td>57.92</td>
   <td>54.65</td>
   <td>51.90</td>
   <td>49.79</td>
   <td>47.66</td>
   <td>45.47</td>
   <td>+0.19</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {93\%}$</td>
   <td>71.22</td>
   <td>66.20</td>
   <td>62.00</td>
   <td>58.34</td>
   <td>55.04</td>
   <td>52.34</td>
   <td>50.22</td>
   <td>48.07</td>
   <td>46.04</td>
   <td>+0.76</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {95\%}$</td>
   <td>71.73</td>
   <td>66.31</td>
   <td>62.17</td>
   <td>58.44</td>
   <td>54.98</td>
   <td>52.20</td>
   <td>50.17</td>
   <td>47.97</td>
   <td>45.87</td>
   <td>+0.59</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {97\%}$</td>
   <td>71.85</td>
   <td>66.48</td>
   <td>62.29</td>
   <td>58.62</td>
   <td>55.36</td>
   <td>52.55</td>
   <td>50.60</td>
   <td>48.43</td>
   <td>46.22</td>
   <td>+0.94</td>
  </tr>
  <tr>
   <td>ResNet18, HardNet, $c = {99\%}$</td>
   <td>71.95</td>
   <td>66.83</td>
   <td>62.75</td>
   <td>59.09</td>
   <td>55.92</td>
   <td>53.03</td>
   <td>50.78</td>
   <td>48.52</td>
   <td>46.31</td>
   <td>+1.03</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {10\%}$</td>
   <td>60.77</td>
   <td>57.02</td>
   <td>53.62</td>
   <td>50.51</td>
   <td>47.67</td>
   <td>45.14</td>
   <td>43.32</td>
   <td>41.6</td>
   <td>39.58</td>
   <td>-5.7</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {20\%}$</td>
   <td>64.67</td>
   <td>60.69</td>
   <td>57.15</td>
   <td>53.77</td>
   <td>50.76</td>
   <td>48.28</td>
   <td>46.24</td>
   <td>44.23</td>
   <td>42.31</td>
   <td>-2.97</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {30\%}$</td>
   <td>67.00</td>
   <td>62.18</td>
   <td>58.22</td>
   <td>54.69</td>
   <td>51.82</td>
   <td>49.12</td>
   <td>47.13</td>
   <td>44.98</td>
   <td>42.44</td>
   <td>-2.84</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {40\%}$</td>
   <td>67.50</td>
   <td>63.11</td>
   <td>59.29</td>
   <td>55.61</td>
   <td>52.53</td>
   <td>49.85</td>
   <td>47.85</td>
   <td>45.84</td>
   <td>43.85</td>
   <td>-1.43</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {50\%}$</td>
   <td>69.20</td>
   <td>64.18</td>
   <td>60.01</td>
   <td>56.43</td>
   <td>53.11</td>
   <td>50.62</td>
   <td>48.60</td>
   <td>46.51</td>
   <td>44.61</td>
   <td>-0.67</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {60\%}$</td>
   <td>69.15</td>
   <td>63.68</td>
   <td>59.54</td>
   <td>56.05</td>
   <td>52.72</td>
   <td>50.10</td>
   <td>48.20</td>
   <td>46.18</td>
   <td>44.15</td>
   <td>-1.13</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {70\%}$</td>
   <td>70.92</td>
   <td>65.16</td>
   <td>61.00</td>
   <td>57.25</td>
   <td>54.09</td>
   <td>51.37</td>
   <td>49.29</td>
   <td>47.03</td>
   <td>44.90</td>
   <td>-0.38</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {80\%}$</td>
   <td>70.38</td>
   <td>65.04</td>
   <td>60.94</td>
   <td>57.26</td>
   <td>54.13</td>
   <td>51.58</td>
   <td>49.52</td>
   <td>47.36</td>
   <td>45.16</td>
   <td>-0.12</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {90\%}$</td>
   <td>72.25</td>
   <td>66.82</td>
   <td>62.63</td>
   <td>58.98</td>
   <td>55.64</td>
   <td>52.77</td>
   <td>50.71</td>
   <td>48.42</td>
   <td>46.15</td>
   <td>+0.87</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {93\%}$</td>
   <td>71.38</td>
   <td>65.93</td>
   <td>61.89</td>
   <td>58.20</td>
   <td>54.87</td>
   <td>51.83</td>
   <td>49.82</td>
   <td>47.57</td>
   <td>45.47</td>
   <td>+0.19</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {95\%}$</td>
   <td>72.23</td>
   <td>66.94</td>
   <td>62.56</td>
   <td>58.84</td>
   <td>55.65</td>
   <td>52.74</td>
   <td>50.61</td>
   <td>48.47</td>
   <td>46.27</td>
   <td>+0.99</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {97\%}$</td>
   <td>70.88</td>
   <td>65.72</td>
   <td>61.38</td>
   <td>57.88</td>
   <td>54.63</td>
   <td>51.82</td>
   <td>49.57</td>
   <td>47.30</td>
   <td>45.19</td>
   <td>-0.09</td>
  </tr>
  <tr>
   <td>ResNet18, SoftNet, $c = {99\%}$</td>
   <td>72.62</td>
   <td>67.31</td>
   <td>63.05</td>
   <td>59.39</td>
   <td>56.00</td>
   <td>53.23</td>
   <td>51.06</td>
   <td>48.83</td>
   <td>46.63</td>
   <td>+1.35</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {10\%}$</td>
   <td>53.50</td>
   <td>50.52</td>
   <td>47.79</td>
   <td>45.11</td>
   <td>42.57</td>
   <td>40.38</td>
   <td>38.88</td>
   <td>37.43</td>
   <td>35.69</td>
   <td>-9.59</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {20\%}$</td>
   <td>61.83</td>
   <td>58.24</td>
   <td>55.17</td>
   <td>51.92</td>
   <td>49.07</td>
   <td>46.68</td>
   <td>45.02</td>
   <td>43.31</td>
   <td>41.40</td>
   <td>-3.88</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {30\%}$</td>
   <td>66.68</td>
   <td>62.81</td>
   <td>59.54</td>
   <td>56.12</td>
   <td>53.21</td>
   <td>50.52</td>
   <td>48.85</td>
   <td>46.93</td>
   <td>44.84</td>
   <td>-0.44</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {40\%}$</td>
   <td>68.31</td>
   <td>64.17</td>
   <td>60.72</td>
   <td>57.19</td>
   <td>54.26</td>
   <td>51.79</td>
   <td>50.13</td>
   <td>48.01</td>
   <td>45.93</td>
   <td>+0.65</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {50\%}$</td>
   <td>70.73</td>
   <td>66.56</td>
   <td>63.10</td>
   <td>59.59</td>
   <td>56.58</td>
   <td>53.87</td>
   <td>51.86</td>
   <td>49.72</td>
   <td>47.46</td>
   <td>+2.18</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {60\%}$</td>
   <td>71.90</td>
   <td>67.64</td>
   <td>63.84</td>
   <td>60.22</td>
   <td>57.06</td>
   <td>54.30</td>
   <td>52.53</td>
   <td>50.53</td>
   <td>48.34</td>
   <td>+3.06</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {70\%}$</td>
   <td>71.41</td>
   <td>67.44</td>
   <td>63.76</td>
   <td>60.02</td>
   <td>56.84</td>
   <td>54.14</td>
   <td>52.54</td>
   <td>50.42</td>
   <td>48.24</td>
   <td>+2.96</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {80\%}$</td>
   <td>71.97</td>
   <td>67.65</td>
   <td>63.93</td>
   <td>60.14</td>
   <td>57.12</td>
   <td>54.44</td>
   <td>52.72</td>
   <td>50.67</td>
   <td>48.43</td>
   <td>+3.15</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {90\%}$</td>
   <td>71.82</td>
   <td>67.73</td>
   <td>64.21</td>
   <td>60.44</td>
   <td>57.44</td>
   <td>54.93</td>
   <td>53.14</td>
   <td>51.03</td>
   <td>48.86</td>
   <td>+3.58</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {93\%}$</td>
   <td>72.28</td>
   <td>67.99</td>
   <td>64.48</td>
   <td>60.83</td>
   <td>57.64</td>
   <td>55.16</td>
   <td>53.27</td>
   <td>51.11</td>
   <td>48.93</td>
   <td>+3.65</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {95\%}$</td>
   <td>72.13</td>
   <td>68.14</td>
   <td>64.50</td>
   <td>60.74</td>
   <td>57.68</td>
   <td>55.12</td>
   <td>53.17</td>
   <td>51.23</td>
   <td>48.97</td>
   <td>+3.69</td>
  </tr>
  <tr>
   <td>ResNet20, HardNet, $c = {97\%}$</td>
   <td>71.90</td>
   <td>67.81</td>
   <td>64.11</td>
   <td>60.24</td>
   <td>57.14</td>
   <td>54.41</td>
   <td>52.74</td>
   <td>50.71</td>
   <td>48.67</td>
   <td>+3.39</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {10\%}$</td>
   <td>53.13</td>
   <td>49.73</td>
   <td>46.85</td>
   <td>44.01</td>
   <td>41.54</td>
   <td>39.45</td>
   <td>37.84</td>
   <td>36.30</td>
   <td>34.62</td>
   <td>-10.66</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {20\%}$</td>
   <td>60.15</td>
   <td>56.25</td>
   <td>53.26</td>
   <td>50.16</td>
   <td>47.46</td>
   <td>45.23</td>
   <td>43.75</td>
   <td>41.84</td>
   <td>39.90</td>
   <td>-5.38</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {30\%}$</td>
   <td>64.65</td>
   <td>59.99</td>
   <td>56.65</td>
   <td>53.45</td>
   <td>50.42</td>
   <td>48.02</td>
   <td>46.48</td>
   <td>44.61</td>
   <td>42.43</td>
   <td>-2.85</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {40\%}$</td>
   <td>66.77</td>
   <td>62.55</td>
   <td>59.35</td>
   <td>55.63</td>
   <td>52.70</td>
   <td>50.21</td>
   <td>48.54</td>
   <td>46.60</td>
   <td>44.50</td>
   <td>-0.78</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {50\%}$</td>
   <td>68.20</td>
   <td>64.21</td>
   <td>60.78</td>
   <td>57.15</td>
   <td>54.20</td>
   <td>51.53</td>
   <td>49.84</td>
   <td>47.92</td>
   <td>45.84</td>
   <td>+0.56</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {60\%}$</td>
   <td>68.63</td>
   <td>64.76</td>
   <td>61.00</td>
   <td>57.53</td>
   <td>54.56</td>
   <td>52.11</td>
   <td>50.34</td>
   <td>48.41</td>
   <td>46.30</td>
   <td>+1.02</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {70\%}$</td>
   <td>70.38</td>
   <td>66.16</td>
   <td>62.63</td>
   <td>58.93</td>
   <td>55.81</td>
   <td>53.11</td>
   <td>51.38</td>
   <td>49.29</td>
   <td>47.08</td>
   <td>+1.80</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {80\%}$</td>
   <td>70.87</td>
   <td>66.47</td>
   <td>62.85</td>
   <td>59.28</td>
   <td>56.27</td>
   <td>53.84</td>
   <td>52.01</td>
   <td>50.16</td>
   <td>48.01</td>
   <td>+2.73</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {90\%}$</td>
   <td>72.63</td>
   <td>68.60</td>
   <td>64.96</td>
   <td>61.25</td>
   <td>57.98</td>
   <td>55.32</td>
   <td>53.48</td>
   <td>51.46</td>
   <td>49.20</td>
   <td>+3.92</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {93\%}$</td>
   <td>72.53</td>
   <td>68.45</td>
   <td>64.59</td>
   <td>60.80</td>
   <td>57.48</td>
   <td>54.91</td>
   <td>53.19</td>
   <td>51.03</td>
   <td>48.83</td>
   <td>+3.55</td>
  </tr>
  <tr>
   <td>ResNet20, SoftNet, $c = {95\%}$</td>

Table 11: Classification accuracy of ResNet18 on miniImageNet for 5-way 5-shot incremental learning. Underbar denotes the comparable results with baseline. * denotes results reported from Shi et al. (2021).

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="9">sessions</th>
<th rowspan="2">The gap with cRT</th>
</tr>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
</tr>
</thead>
<tbody>
<tr>
<td>cRT Shi et al. (2021)</td>
<td>67.30</td>
<td>64.15</td>
<td>60.59</td>
<td>57.32</td>
<td>54.22</td>
<td>51.43</td>
<td>48.92</td>
<td>46.78</td>
<td>44.85</td>
<td>-</td>
</tr>
<tr>
<td>Joint-training Shi et al. (2021)</td>
<td>67.30</td>
<td>62.34</td>
<td>57.79</td>
<td>54.08</td>
<td>50.93</td>
<td>47.65</td>
<td>44.64</td>
<td>42.61</td>
<td>40.29</td>
<td>-4.56</td>
</tr>
<tr>
<td>Baseline Shi et al. (2021)</td>
<td>67.30</td>
<td>63.18</td>
<td>59.62</td>
<td>56.33</td>
<td>53.28</td>
<td>50.50</td>
<td>47.96</td>
<td>45.85</td>
<td>43.88</td>
<td>-0.97</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)*</td>
<td>67.35</td>
<td>59.91</td>
<td>55.64</td>
<td>52.60</td>
<td>49.43</td>
<td>46.73</td>
<td>44.13</td>
<td>42.17</td>
<td>40.29</td>
<td>-4.56</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)*</td>
<td>67.91</td>
<td>63.11</td>
<td>58.75</td>
<td>54.83</td>
<td>50.68</td>
<td>47.11</td>
<td>43.88</td>
<td>41.19</td>
<td>38.72</td>
<td>-6.13</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)*</td>
<td>67.30</td>
<td>59.81</td>
<td>57.26</td>
<td>54.57</td>
<td>52.05</td>
<td>49.42</td>
<td>46.95</td>
<td>44.94</td>
<td>42.87</td>
<td>-1.11</td>
</tr>
<tr>
<td>iCaRL Rebuffi et al. (2017)</td>
<td>61.31</td>
<td>46.32</td>
<td>42.94</td>
<td>37.63</td>
<td>30.49</td>
<td>24.00</td>
<td>20.89</td>
<td>18.80</td>
<td>17.21</td>
<td>-27.64</td>
</tr>
<tr>
<td>Rebalance Hou et al. (2019)</td>
<td>61.31</td>
<td>47.80</td>
<td>39.31</td>
<td>31.91</td>
<td>25.68</td>
<td>21.35</td>
<td>18.67</td>
<td>17.24</td>
<td>14.17</td>
<td>-30.68</td>
</tr>
<tr>
<td>TOPIC Cheraghian et al. (2021)</td>
<td>61.31</td>
<td>50.09</td>
<td>45.17</td>
<td>41.16</td>
<td>37.48</td>
<td>35.52</td>
<td>32.19</td>
<td>29.46</td>
<td>24.42</td>
<td>-20.43</td>
</tr>
<tr>
<td>IDLVQ-C Chen and Lee (2020)</td>
<td>64.77</td>
<td>59.87</td>
<td>55.93</td>
<td>52.62</td>
<td>49.88</td>
<td>47.55</td>
<td>44.83</td>
<td>43.14</td>
<td>41.84</td>
<td>-3.01</td>
</tr>
<tr>
<td>F2M Shi et al. (2021)</td>
<td>67.28</td>
<td>63.80</td>
<td>60.38</td>
<td>57.06</td>
<td>54.08</td>
<td>51.39</td>
<td>48.82</td>
<td>46.58</td>
<td>44.65</td>
<td>-0.20</td>
</tr>
<tr>
<td>FSLL Mazumder et al. (2021)</td>
<td>66.48</td>
<td>61.75</td>
<td>58.16</td>
<td>54.16</td>
<td>51.10</td>
<td>48.53</td>
<td>46.54</td>
<td>44.20</td>
<td>42.28</td>
<td>-2.57</td>
</tr>
<tr>
<td>HardNet, $c=10\%$</td>
<td>9.70</td>
<td>8.46</td>
<td>8.13</td>
<td>7.67</td>
<td>7.35</td>
<td>6.78</td>
<td>6.55</td>
<td>6.3</td>
<td>5.8</td>
<td>-39.05</td>
</tr>
<tr>
<td>HardNet, $c=20\%$</td>
<td>24.95</td>
<td>22.20</td>
<td>20.66</td>
<td>19.39</td>
<td>18.32</td>
<td>17.41</td>
<td>16.64</td>
<td>15.79</td>
<td>15.28</td>
<td>-29.57</td>
</tr>
<tr>
<td>HardNet, $c=30\%$</td>
<td>44.08</td>
<td>40.66</td>
<td>38.24</td>
<td>36.06</td>
<td>33.65</td>
<td>31.84</td>
<td>30.25</td>
<td>29.01</td>
<td>28.02</td>
<td>-16.83</td>
</tr>
<tr>
<td>HardNet, $c=40\%$</td>
<td>37.05</td>
<td>34.94</td>
<td>32.25</td>
<td>30.72</td>
<td>29.09</td>
<td>27.53</td>
<td>25.84</td>
<td>24.76</td>
<td>23.71</td>
<td>-21.14</td>
</tr>
<tr>
<td>HardNet, $c=50\%$</td>
<td>65.13</td>
<td>60.37</td>
<td>56.12</td>
<td>53.17</td>
<td>50.17</td>
<td>47.74</td>
<td>45.34</td>
<td>43.35</td>
<td>42.13</td>
<td>-2.72</td>
</tr>
<tr>
<td>HardNet, $c=60\%$</td>
<td>73.32</td>
<td>67.77</td>
<td>63.46</td>
<td>60.13</td>
<td>57.13</td>
<td>50.36</td>
<td>47.88</td>
<td>46.05</td>
<td>44.59</td>
<td>-0.26</td>
</tr>
<tr>
<td>HardNet, $c=70\%$</td>
<td>71.75</td>
<td>66.66</td>
<td>62.19</td>
<td>58.85</td>
<td>55.74</td>
<td>52.82</td>
<td>50.14</td>
<td>48.45</td>
<td>47.01</td>
<td>+2.16</td>
</tr>
<tr>
<td>HardNet, $c=80\%$</td>
<td>69.73</td>
<td>64.46</td>
<td>60.42</td>
<td>57.09</td>
<td>54.09</td>
<td>51.18</td>
<td>48.76</td>
<td>46.81</td>
<td>45.66</td>
<td>+0.81</td>
</tr>
<tr>
<td>HardNet, $c=90\%$</td>
<td>64.68</td>
<td>59.80</td>
<td>55.70</td>
<td>52.82</td>
<td>50.01</td>
<td>47.30</td>
<td>45.17</td>
<td>43.34</td>
<td>42.09</td>
<td>-2.76</td>
</tr>
<tr>
<td>HardNet, $c=93\%$</td>
<td>67.17</td>
<td>61.74</td>
<td>57.53</td>
<td>54.43</td>
<td>51.52</td>
<td>48.86</td>
<td>46.42</td>
<td>44.68</td>
<td>43.43</td>
<td>-1.42</td>
</tr>
<tr>
<td>HardNet, $c=95\%$</td>
<td>64.72</td>
<td>60.13</td>
<td>56.05</td>
<td>53.25</td>
<td>50.20</td>
<td>47.62</td>
<td>45.11</td>
<td>43.40</td>
<td>42.33</td>
<td>-2.52</td>
</tr>
<tr>
<td>HardNet, $c=97\%$</td>
<td>63.92</td>
<td>58.85</td>
<td>55.12</td>
<td>52.16</td>
<td>49.44</td>
<td>46.78</td>
<td>44.48</td>
<td>42.68</td>
<td>41.52</td>
<td>-3.33</td>
</tr>
<tr>
<td>HardNet, $c=99\%$</td>
<td>67.28</td>
<td>62.17</td>
<td>58.06</td>
<td>55.05</td>
<td>52.01</td>
<td>49.12</td>
<td>46.92</td>
<td>45.07</td>
<td>43.90</td>
<td>-0.95</td>
</tr>
<tr>
<td>SoftNet, $c=10\%$</td>
<td>62.75</td>
<td>58.26</td>
<td>53.80</td>
<td>50.82</td>
<td>47.68</td>
<td>44.86</td>
<td>42.05</td>
<td>39.86</td>
<td>38.15</td>
<td>-6.70</td>
</tr>
<tr>
<td>SoftNet, $c=20\%$</td>
<td>68.33</td>
<td>62.76</td>
<td>58.60</td>
<td>55.25</td>
<td>52.07</td>
<td>49.36</td>
<td>46.48</td>
<td>44.21</td>
<td>42.52</td>
<td>-2.33</td>
</tr>
<tr>
<td>SoftNet, $c=30\%$</td>
<td>72.20</td>
<td>66.92</td>
<td>62.56</td>
<td>59.16</td>
<td>56.07</td>
<td>53.10</td>
<td>50.59</td>
<td>48.46</td>
<td>47.03</td>
<td>+2.18</td>
</tr>
<tr>
<td>SoftNet, $c=40\%$</td>
<td>72.58</td>
<td>67.34</td>
<td>62.91</td>
<td>59.65</td>
<td>56.72</td>
<td>54.00</td>
<td>51.46</td>
<td>49.39</td>
<td>48.11</td>
<td>+3.26</td>
</tr>
<tr>
<td>SoftNet, $c=50\%$</td>
<td>72.83</td>
<td>67.23</td>
<td>62.82</td>
<td>59.41</td>
<td>56.44</td>
<td>53.55</td>
<td>50.92</td>
<td>48.99</td>
<td>47.60</td>
<td>+2.75</td>
</tr>
<tr>
<td>SoftNet, $c=60\%$</td>
<td>73.83</td>
<td>67.78</td>
<td>63.46</td>
<td>60.21</td>
<td>57.27</td>
<td>54.42</td>
<td>51.74</td>
<td>49.94</td>
<td>48.57</td>
<td>+3.72</td>
</tr>
<tr>
<td>SoftNet, $c=70\%$</td>
<td>75.15</td>
<td>69.06</td>
<td>64.79</td>
<td>61.40</td>
<td>58.38</td>
<td>55.49</td>
<td>52.87</td>
<td>50.89</td>
<td>49.69</td>
<td>+4.84</td>
</tr>
<tr>
<td>SoftNet, $c=80\%$</td>
<td>76.63</td>
<td>70.13</td>
<td>65.92</td>
<td>62.52</td>
<td>59.49</td>
<td>56.56</td>
<td>53.71</td>
<td>51.72</td>
<td>50.48</td>
<td>+5.63</td>
</tr>
<tr>
<td>SoftNet, $c=90\%$</td>
<td>77.00</td>
<td>70.38</td>
<td>65.94</td>
<td>62.45</td>
<td>59.32</td>
<td>56.25</td>
<td>53.76</td>
<td>51.75</td>
<td>50.39</td>
<td>+5.54</td>
</tr>
<tr>
<td>SoftNet, $c=93\%$</td>
<td>73.97</td>
<td>67.39</td>
<td>63.35</td>
<td>59.90</td>
<td>56.89</td>
<td>54.18</td>
<td>51.61</td>
<td>49.71</td>
<td>48.45</td>
<td>+3.60</td>
</tr>
<tr>
<td>SoftNet, $c=95\%$</td>
<td>76.22</td>
<td>69.64</td>
<td>65.22</td>
<td>61.91</td>
<td>58.84</td>
<td>55.75</td>
<td>53.07</td>
<td>51.18</td>
<td>49.84</td>
<td>+4.99</td>
</tr>
<tr>
<td>SoftNet, $c=97\%$</td>
<td>77.17</td>
<td>70.32</td>
<td>66.15</td>
<td>62.55</td>
<td>59.48</td>
<td>56.46</td>
<td>53.71</td>
<td>51.68</td>
<td>50.24</td>
<td>+5.39</td>
</tr>
<tr>
<td>SoftNet, $c=99\%$</td>
<td>76.80</td>
<td>69.79</td>
<td>65.44</td>
<td>62.01</td>
<td>58.87</td>
<td>55.94</td>
<td>53.21</td>
<td>51.25</td>
<td>50.04</td>
<td>+5.19</td>
</tr>
</tbody>
</table>
