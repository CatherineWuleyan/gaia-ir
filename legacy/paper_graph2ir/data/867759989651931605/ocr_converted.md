# Robust Lottery Tickets for Pre-trained Language Models

［#1］
Rui Zheng$^{\star \star}$, Rong Bao$^{\star \star}$, Yuhao Zhou$^{\star}$, Di Liang$^\spadesuit$, Sirui Wang$^\spadesuit$,
Wei Wu$^\spadesuit$, Tao Gui$^\lozenge \dagger$, Qi Zhang$^{\star \clubsuit}$, Xuanjing Huang$^\star$

［#1］
$^\star$ School of Computer Science, Fudan University, Shanghai, China
［#1］
$^\lozenge$ Institute of Modern Languages and Linguistics, Fudan University, Shanghai, China
［#1］
$^\clubsuit$ Shanghai Collaborative Innovation Center of Intelligent Visual Computing, Fudan University
［#1］
$^\spadesuit$ Meituan Inc., Beijing, China

［#2］
{rzheng20,rbao18,tgui,qz,xjhuang}@fudan.edu.cn
zhouyh21@m.fudan.edu.cn

## Abstract
［#3］
Recent works on *Lottery Ticket Hypothesis* have shown that pre-trained language models (PLMs) contain smaller matching subnetworks (winning tickets) which are capable of reaching accuracy comparable to the original models. However, these tickets are proved to be not robust to adversarial examples, and even worse than their PLM counterparts. To address this problem, we propose a novel method based on learning binary weight masks to identify robust tickets hidden in the original PLMs. Since the loss is not differentiable for the binary mask, we assign the hard concrete distribution to the masks and encourage their sparsity using a smoothing approximation of $L_0$ regularization. Furthermore, we design an adversarial loss objective to guide the search for robust tickets and ensure that the tickets perform well both in accuracy and robustness. Experimental results show the significant improvement of the proposed method over previous work on adversarial robustness evaluation.

## 1 Introduction
［#4］
Large-scale pre-trained language models (PLMs), such as BERT (Devlin et al., 2019), Roberta (Liu et al., 2019) and T5 (Raffel et al., 2019) have achieved great success in the field of natural language processing. As more transformer layers are stacked with larger self-attention blocks, the complexity of PLMs increases rapidly. Due to the over-parametrization of PLMs, some Transformer heads and even layers can be pruned without significant losses in performance (Michel et al., 2019; Kovaleva et al., 2019; Rogers et al., 2020).

［#5］
The Lottery Ticket Hypothesis suggests an over-parameterized network contains certain subnetworks (i.e., winning tickets) that can match the performance of the original model when trained in isolation (Frankle and Carbin, 2019). Chen et al. (2020); Prasanna et al. (2020) also find these winning tickets exist in PLMs. Chen et al. (2020) prune BERT in an unstructured fashion and obtain winning tickets at sparsity from 40% to 90%. Prasanna et al. (2020) aim at finding structurally sparse tickets for BERT by pruning entire attention heads and MLP. Previous works mainly focused on using winning tickets to reduce model size and speed up training time (Chen et al., 2021), while little work has been done to explore more benefits, such as better adversarial robustness than the original model.

［#6］
As we all know, PLMs are vulnerable to adversarial examples that are legitimately crafted by imposing imperceptible perturbations on normal examples (Jin et al., 2020; Garg and Ramakrishnan, 2020; Wang et al., 2021). Recent studies have shown that pruned subnetworks of PLMs are even less robust than their PLM counterparts (Xu et al., 2021; Du et al., 2021). Xu et al. (2021) observe that when fine-tuning the pruned model again, the model yields a lower robustness. Du et al. (2021) clarify the above phenomenon further: the compressed models overfit on shortcut samples and thus perform consistently less robust than the uncompressed large model on adversarial test sets.

［#7］
In this work, our goal is to find robust PLM tickets that, when fine-tuned on downstream tasks, achieve matching test performance but are more robust than the original PLMs. In order to make the topology structure of tickets learnable, we assign binary masks to pre-trained weights to determine which connections need to be removed. To solve discrete optimization problem of binary masks, we assume the masks follow a hard concrete distribution (a soft version of the Bernoulli distribution), which can be solved using Gumbel-Softmax trick (Louizos et al., 2018). We then use an adversarial loss objective to guide the search for robust tickets and an approximate $L_0$

---
［#7］
$^\ast$ Equal contribution.
［#8］
$^\dagger$ Corresponding authors.

［#9］
regularization is used to encourage the sparsity of robust tickets. Robust tickets can be used as a robust substitute of original PLMs to fine-tune downstream tasks. Experimental results show that robust tickets achieve a significant improvement in adversarial robustness on various tasks and maintain a matching accuracy. Our codes are publicly available at $Github^1$.

［#10］
The main contributions of our work are summarized as follows:

［#11］
- We demonstrate that PLMs contain robust tickets with matching accuracy but better robustness than the original network.
- We propose a novel and effective technique to find the robust tickets based on learnable binary masks rather than the traditional iterative magnitude-based pruning.
- We provide a new perspective to explain the vulnerability of PLMs on adversarial examples: some weights of PLMs do not contribute to the accuracy but may harm the robustness.

## 2 Related Work

### 2.1 Textual Adversarial Attack and Defense
［#12］
Textual attacks typically generate explicit adversarial examples by replacing the components of sentences with their counterparts and maintaining a high similarity in semantics (Ren et al., 2019) or embedding space (Li et al., 2020). These adversarial attackers can be divided into character-level (Gao et al., 2018), word-level (Ren et al., 2019; Zang et al., 2020; Jin et al., 2020; Li et al., 2020) and multi-level (Li et al., 2018). In response to adversarial attackers, various adversarial defense methods are proposed to improve model robustness. Adversarial training solves a min-max robust optimization and is generally considered as one of the strongest defense methods (Madry et al., 2018; Zhu et al., 2020; Li and Qiu, 2020). Adversarial data augmentation (ADA) has been widely adopted to improve robustness by adding textual adversarial examples during training (Jin et al., 2020; Si et al., 2021). However, ADA is not sufficient to cover the entire perturbed search space, which grows exponentially with the length of the input text. Some regularization methods, such as smoothness-inducing regularization (Jiang et al., 2020) and information bottleneck regularization (Wang et al., 2020), are also beneficial for robustness. Different from the above methods, we dig robust tickets from original BERT, and the subnetworks we find have better robustness after fine-tuning.

### 2.2 Lottery Ticket Hypothesis
［#13］
Lottery Ticket Hypothesis (LTH) suggests the existence of certain sparse subnetworks (i.e., winning tickets) at initialization that can achieve almost the same test performance compared to the original model (Frankle and Carbin, 2019). In the field of NLP, previous works find that the winning tickets also exist in Transformers and LSTM (Yu et al., 2020; Renda et al., 2020). Evci et al. (2020) propose a method to optimize the topology of the sparse network during training without sacrificing accuracy relative to existing dense-to-sparse training methods. Chen et al. (2020) find that PLMs such as BERT contain winning tickets with a sparsity of 40% to 90%, and the winning tickets found in the mask language modeling task can universally be transfered to other downstream tasks. Prasanna et al. (2020) find structurally sparse winning tickets for BERT, and they notice that all subnetworks (winning tickets and randomly pruned subnetworks) have comparable performance when fine-tuned on downstream tasks. Chen et al. (2021) propose an efficient BERT training method using Early-bird lottery tickets to reduce the training time and inference time. Some recent studies have tried to dig out more features of winning tickets. Zhang et al. (2021) demonstrate that even in biased models (which focus on spurious correlations) there still exist unbiased winning tickets. Liang et al. (2021) observe that at a certain sparsity, the generalization performance of the winning tickets can not only match but also exceed that of the full model. (Du et al., 2021; Xu et al., 2021) show that the winning tickets that only consider accuracy are over-fitting on easy samples and generalize poorly on adversarial examples. Our work makes the first attempt to find the robust winning tickets for PLMs.

### 2.3 Robustness in Model Pruning
［#14］
Learning to identify a subnetwork with high adversarial robustness is widely discussed in the field of computer vision. Post-train pruning approaches require a pre-trained model with adversarial robustness before pruning (Sehwag et al., 2019; Gui et al., 2019). In-train pruning methods integrate the pruning process into the robust

---
［#15］
$^1$https://github.com/ruizheng20/robust_ticket

［#16］
learning process, which jointly optimize the model parameters and pruning connections (Vemparala et al., 2021; Ye et al., 2019). Sehwag et al. (2020) integrate the robust training objective into the pruning process and remove the connections based on importance scores. In our work, we focus on finding robust tickets hidden in original PLMs rather than pruning subnetworks from a robust model.

## 3 The Robust Ticket Framework

［#17］
In this section, we propose a novel pruning method to extract robust tickets of PLMs by learning binary weights masks with an adversarial loss objective. Furthermore, we articulate the Robust Lottery Ticket Hypothesis: the full PLM contains subnetworks (robust tickets) that can achieve better adversarial robustness and comparable accuracy.

### 3.1 Revisiting Lottery Ticket Hypothesis

［#18］
Denote $f(\theta)$ as a PLM with parameters $\theta$ that has been fine-tuned on a downstream task. A subnetwork of $f(\theta)$ can be denoted as $f(m \odot \theta)$, where $m$ are binary masks with the same dimension as $\theta$ and $\odot$ is the Hadamard product operator. LTH suggests that, for a network initialized with $\theta_0$, the Iterative Magnitude Pruning (IMP) can identify a mask $m$, such that the subnetwork $f(x; m \odot \theta_0)$ can be trained to almost the same performance to the full model $f(\theta_0)$ in a comparable number of iterations. Such a subnetwork $f(x; m \odot \theta_0)$ is called as *winning tickets*, including both the structure mask $m$ and initialization $\theta_0$. IMP iteratively removes the weights with the smallest magnitudes from $m \odot \theta$ until a certain sparsity is reached. However, the magnitude-based pruning is not suitable for robustness-aware techniques (Vemparala et al., 2021; Sehwag et al., 2020).

### 3.2 Discovering Robust Tickets

［#19］
Our goal is to learn the sparse subnetwork, however, the training loss is not differentiable for the binary masks. A simple choice is to adopt a straight-through estimator to approximate the derivative (Bengio et al., 2013). Unfortunately, this approach ignores the Heaviside function in the likelihood and results in biased gradients. Thus, we resort to a practical method to learn sparse neural networks (Louizos et al., 2018).

［#20］
In our method, we assume each mask $m_i$ to be a independent random variable that follows a hard concrete distribution $\text{HardConcrete}(\log \alpha_i, \beta_i)$ with temperature $\beta_i$ and location $\alpha_i$ (Louizos et al., 2018):

［#20］
$$
\mu_i \sim \mathcal{U}(0,1), \tag{1}
$$

［#20］
$$
s_i = \sigma\left(\frac{1}{\beta_i}\left(\log \frac{\mu_i}{1-\mu_i} + \log \alpha_i\right)\right), \tag{2}
$$

［#20］
$$
m_i = \min\left(1, \max\left(0, s_i(\zeta - \gamma) + \gamma\right)\right), \tag{3}
$$

［#20］
where $\sigma$ denotes the sigmoid function, $\gamma=-0.1$, $\zeta=1.1$ are constants, and $u_i$ is the sample drawn from uniform distribution $\mathcal{U}(0,1)$. The random variable $s_i$ follows a binary concrete (or Gumbel-Softmax) distribution, which is a smoothing approximation of the discrete Bernoulli distribution (Maddison et al., 2017; Jang et al., 2017). Samples from the binary concrete distribution are identical to samples from a Bernoulli distribution with probability $\alpha_i$ as $\beta_i \to 0$. The location $\alpha_i$ in (2) allows for gradient-based optimization through reparametrization tricks. Using (3), the $s_i$ larger than $\frac{1-\gamma}{\zeta-\gamma}$ is rounded to 1, whereas the value smaller than $\frac{-\gamma}{\zeta-\gamma}$ is rounded to 0. To encourage the sparsity, we penalize the $L_0$ complexity of masks based on the probability which are non-zero:

［#20］
$$
\mathcal{R}(m) = \frac{1}{|m|} \sum_{i=1}^{|m|} \sigma\left(\log \alpha_i - \beta_i \log \frac{-\gamma}{\zeta}\right). \tag{4}
$$

［#21］
During the inference stage, the mask $\hat{m}_i$ can be estimated through a hard concrete gate:

［#21］
$$
\min\left(1, \max\left(0, \sigma\left(\log \alpha_i\right)(\zeta - \gamma) + \gamma\right)\right). \tag{5}
$$

#### 3.2.1 Adversarial Loss Objective

［#22］
To find the connections responsible for adversarial robustness, we incorporate the adversarial loss into the mask learning objective:

［#22］
$$
\underbrace{\min_{m} \mathbb{E}_{(x,y) \sim \mathcal{D}} \max_{\|\delta\| \leq \epsilon} \mathcal{L}\left(f(x+\delta; m \odot \theta), y\right)}_{\mathcal{L}_{adv}(m)}, \tag{6}
$$

［#22］
where $(x,y)$ is a data point from dataset $\mathcal{D}$, $\delta$ is the perturbation that constrained within the $\epsilon$ ball. The inner maximization problem in (6) is to find the worst-case adversarial examples to maximize the classification loss, while the outer minimization problem in (6) aims at optimizing the masks to minimize the loss of adversarial examples, i.e., $\mathcal{L}_{adv}(m)$.

［#23］
Adversarial attack method, typically with PGD, can be used to solve the inner maximization

［#23］
problem. PGD applies the $K$-step stochastic gradient descent to search for the perturbation $\delta$ (Madry et al., 2018):

［#23］
$$
\delta_{k+1}=\prod_{\|\delta\| \leq \epsilon}\left(\delta_{k}+\eta \frac{g\left(\delta_{k}\right)}{\left\|g\left(\delta_{k}\right)\right\|}\right), \quad(7)
$$

［#23］
where $g\left(\delta_{k}\right)=\nabla_{x} \mathcal{L}\left(f\left(x+\delta_{k} ; m \odot \theta\right), y\right), \delta_{k}$ is the perturbation in $k$-th step and $\prod_{\|\delta\| \leq \epsilon}(\cdot)$ projects the perturbation back onto the Frobenius normalization ball. Then robust training optimizes the network on adversarially perturbed input $x+\delta_{K}$. Through the above process, we can conveniently obtain a large number of adversarial examples for training.

［#24］
By integrating the $L_0$ complexity regularizer into the training process of masks, our adversarial loss objective becomes:

［#24］
$$
\min _{m} \mathcal{L}_{a d v}(m)+\lambda \mathcal{R}(m), \quad(8)
$$

［#24］
where $\lambda$ denotes regularization strength.

### 3.2.2 Effect of Regularization Strength
［#25］
The selection of the regularization strength $\lambda$ decides the quality of robust tickets. Results carried on SST-2 in Fig.1 show that eventually more than 90% of the masks will be very close to 0 or 1, and the $L_0$ complexity regularizer $\mathcal{R}(m)$ will converge to a fixed value. As $\lambda$ increases, $\mathcal{R}(m)$ decreases (the sparsity of the subnetwork increases). The training of the adversarial loss objective in (8) is insensitive to the $\lambda$, and in all experiments, $\lambda$ is chosen in the range $[0.1,1]$. In the Appendix A, we show more details about mask learning process.

### 3.3 Drawing and Retraining Winning Tickets
［#26］
After training the masks $m$, we use the location parameters $\log \alpha$ of masks to extract robust tickets. For the Gumbel-Softmax distribution in (2), $\alpha_i$ is the expectation (confidence) of random variable $s_i$, i.e, $\mathbb{E}\{s_i\}=\alpha_i$. Thus, we prune the weights whose masks have the smallest expectation. We prune all attention heads and intermediate neurons in an unstructured manner, which empirically has better performance than structured pruning. Unlike the Lottery Ticket Hypothesis that requires iterative magnitude pruning, the proposed method is a one-shot pruning method that can obtain subnetworks of any sparsity. Then we retrain (i.e., fine-tune) the robust tickets $f(m \odot \theta_0)$ on downstream tasks.

［#27］
![](./images/867759989651931605_1.jpg)

［#28］
Figure 1: Effect of regularization strength $\lambda$ on regularizer $\mathcal{R}(m)$, and the percentage of masks that exact 0 and 1.

### 3.4 Robust Lottery Tickets Hypothesis
［#29］
In the context of adversarial robustness, we seek winning tickets that balance accuracy and robustness, and then we state and demonstrate Robust Lottery Tickets Hypothesis.

［#30］
Robust Lottery Tickets Hypothesis: A pre-trained language model, such as BERT, contains some subnetworks (robust tickets) initialized by pre-trained weights, and when these subnetworks are trained in isolation, they can achieve better adversarial robustness and comparable accuracy. In addition, robust tickets retain an important characteristic of traditional lottery tickets —the ability to speed up the training process.

［#31］
The practical merits of Robust Lottery Ticket Hypothesis: 1) It provides an effective pruning method that can reduce memory constraints during inference time by identifying well-performing smaller networks which can fit in memory. 2) Our proposed robust ticket is more robust than the existing defense methods, so it can be used as a defense method.

## 4 Experiments
［#32］
We conduct several experiments to demonstrate the effectiveness of our method. We first compare the proposed method with baseline methods in terms of clean accuracy and robust evaluation. Then, we perform an ablation study to illustrate the role of sparse mask learning and adversarial loss objective in our method. In addition, we try to further flesh out our method with several additional analysis experiments. Following the official BERT implementation (Devlin et al., 2019; Wolf et al., 2020), we use $\text{BERT}_{\text{BASE}}$ as our backbone model for all experiments.

［#33］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Clean%</th>
      <th colspan="3">BERT-Attack</th>
      <th colspan="3">TextFooler</th>
      <th colspan="3">TextBugger</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>Aua%</th>
      <th>Suc%</th>
      <th>#Query</th>
      <th>Aua%</th>
      <th>Suc%</th>
      <th>#Query</th>
      <th>Aua%</th>
      <th>Suc%</th>
      <th>#Query</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">IMDB</td>
      <td>Fine-tune</td>
      <td>94.1</td>
      <td>7.8</td>
      <td>91.7</td>
      <td>1572.2</td>
      <td>12.2</td>
      <td>87.0</td>
      <td>1209.8</td>
      <td>25.8</td>
      <td>72.5</td>
      <td>783.2</td>
    </tr>
    <tr>
      <td>LTH₂₀%</td>
      <td>94.0</td>
      <td>3.6</td>
      <td>96.2</td>
      <td>1074.44</td>
      <td>7.2</td>
      <td>92.3</td>
      <td>894.1</td>
      <td>16.0</td>
      <td>83.0</td>
      <td>574.0</td>
    </tr>
    <tr>
      <td>FreeLB</td>
      <td>94.8</td>
      <td>22.6</td>
      <td>76.2</td>
      <td>1954.7</td>
      <td>27.2</td>
      <td>71.3</td>
      <td>1479.1</td>
      <td>36.0</td>
      <td>62.0</td>
      <td>907.3</td>
    </tr>
    <tr>
      <td>InfoBERT</td>
      <td>95.2</td>
      <td>26.0</td>
      <td>72.7</td>
      <td>2326.0</td>
      <td>32.4</td>
      <td>66.0</td>
      <td>1572.2</td>
      <td>43.6</td>
      <td>54.2</td>
      <td>969.8</td>
    </tr>
    <tr>
      <td>Rand₂₀%</td>
      <td>93.1</td>
      <td>6.8</td>
      <td>92.8</td>
      <td>731.5</td>
      <td>7.4</td>
      <td>92.1</td>
      <td>598.7</td>
      <td>8.4</td>
      <td>91.9</td>
      <td>464.3</td>
    </tr>
    <tr>
      <td>RobustT₂₀%</td>
      <td>93.8</td>
      <td>55.2</td>
      <td>41.2</td>
      <td>3128.0</td>
      <td>55.6</td>
      <td>40.7</td>
      <td>1988.4</td>
      <td>57.6</td>
      <td>38.6</td>
      <td>1149.1</td>
    </tr>
    <tr>
      <td rowspan="6">AGNEWS</td>
      <td>Fine-tune</td>
      <td>94.7</td>
      <td>3.8</td>
      <td>96.0</td>
      <td>436.7</td>
      <td>14.9</td>
      <td>84.2</td>
      <td>333.2</td>
      <td>41.5</td>
      <td>56.1</td>
      <td>178.3</td>
    </tr>
    <tr>
      <td>LTH₄₀%</td>
      <td>93.7</td>
      <td>2.5</td>
      <td>97.3</td>
      <td>394.4</td>
      <td>11.0</td>
      <td>88.3</td>
      <td>295.2</td>
      <td>36.8</td>
      <td>60.7</td>
      <td>179.7</td>
    </tr>
    <tr>
      <td>FreeLB</td>
      <td>95.2</td>
      <td>10.8</td>
      <td>88.6</td>
      <td>563.9</td>
      <td>24.3</td>
      <td>74.4</td>
      <td>394.6</td>
      <td>51.7</td>
      <td>45.5</td>
      <td>190.4</td>
    </tr>
    <tr>
      <td>InfoBERT</td>
      <td>94.4</td>
      <td>11.1</td>
      <td>88.3</td>
      <td>517.0</td>
      <td>25.1</td>
      <td>73.4</td>
      <td>374.7</td>
      <td>47.9</td>
      <td>49.3</td>
      <td>193.1</td>
    </tr>
    <tr>
      <td>Rand₄₀%</td>
      <td>94.0</td>
      <td>1.3</td>
      <td>98.6</td>
      <td>357.2</td>
      <td>6.3</td>
      <td>93.2</td>
      <td>275.1</td>
      <td>27.5</td>
      <td>70.1</td>
      <td>148.7</td>
    </tr>
    <tr>
      <td>RobustT₄₀%</td>
      <td>94.9</td>
      <td>12.1</td>
      <td>87.2</td>
      <td>607.7</td>
      <td>28.5</td>
      <td>70.0</td>
      <td>442.1</td>
      <td>53.4</td>
      <td>43.7</td>
      <td>207.8</td>
    </tr>
    <tr>
      <td rowspan="6">SST-2</td>
      <td>Fine-tune</td>
      <td>92.0</td>
      <td>2.9</td>
      <td>96.8</td>
      <td>114.2</td>
      <td>5.0</td>
      <td>94.6</td>
      <td>98.4</td>
      <td>29.4</td>
      <td>68.3</td>
      <td>49.7</td>
    </tr>
    <tr>
      <td>LTH₃₀%</td>
      <td>92.1</td>
      <td>2.2</td>
      <td>97.6</td>
      <td>98.9</td>
      <td>4.1</td>
      <td>95.5</td>
      <td>90.5</td>
      <td>29.1</td>
      <td>68.4</td>
      <td>49.6</td>
    </tr>
    <tr>
      <td>FreeLB</td>
      <td>91.6</td>
      <td>10.2</td>
      <td>88.9</td>
      <td>154.6</td>
      <td>14.4</td>
      <td>84.2</td>
      <td>123.8</td>
      <td>42.4</td>
      <td>53.7</td>
      <td>54.9</td>
    </tr>
    <tr>
      <td>InfoBERT</td>
      <td>92.1</td>
      <td>14.4</td>
      <td>84.4</td>
      <td>162.3</td>
      <td>18.3</td>
      <td>80.1</td>
      <td>121.4</td>
      <td>40.3</td>
      <td>56.3</td>
      <td>51.2</td>
    </tr>
    <tr>
      <td>Rand₃₀%</td>
      <td>83.2</td>
      <td>2.1</td>
      <td>97.5</td>
      <td>89.4</td>
      <td>2.4</td>
      <td>97.1</td>
      <td>75.6</td>
      <td>16.5</td>
      <td>80.2</td>
      <td>44.2</td>
    </tr>
    <tr>
      <td>RobustT₃₀%</td>
      <td>90.9</td>
      <td>17.9</td>
      <td>80.3</td>
      <td>164.9</td>
      <td>26.7</td>
      <td>70.6</td>
      <td>149.8</td>
      <td>42.1</td>
      <td>53.7</td>
      <td>53.9</td>
    </tr>
  </tbody>
</table>

［#34］
Table 1: Main results on adversarial robustness evaluation. Fine-tuning RobustT for downstream tasks achieves a significant improvement of robustness. The percentage on the subscript denotes the sparsity of the subnetworks. The best performance is marked in bold. Suc% lower is better.

［#35］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Clean%</th>
      <th colspan="2">Aua%</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>TextFooler</th>
      <th>TextBugger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">QNLI</td>
      <td>Fine-tune</td>
      <td>91.6</td>
      <td>4.7</td>
      <td>10.5</td>
    </tr>
    <tr>
      <td>FreeLB</td>
      <td>90.5</td>
      <td>12.8</td>
      <td>12.0</td>
    </tr>
    <tr>
      <td>InfoBERT</td>
      <td>91.5</td>
      <td>16.4</td>
      <td>20.9</td>
    </tr>
    <tr>
      <td>RobustT₃₀%</td>
      <td>91.5</td>
      <td>17.0</td>
      <td>25.9</td>
    </tr>
    <tr>
      <td rowspan="4">MNLI</td>
      <td>Fine-tune</td>
      <td>84.4</td>
      <td>7.7</td>
      <td>4.3</td>
    </tr>
    <tr>
      <td>FreeLB</td>
      <td>82.9</td>
      <td>11.0</td>
      <td>8.4</td>
    </tr>
    <tr>
      <td>InfoBERT</td>
      <td>84.1</td>
      <td>10.8</td>
      <td>8.4</td>
    </tr>
    <tr>
      <td>RobustT₃₀%</td>
      <td>84.0</td>
      <td>18.4</td>
      <td>22.6</td>
    </tr>
    <tr>
      <td rowspan="4">QQP</td>
      <td>Fine-tune</td>
      <td>91.3</td>
      <td>24.8</td>
      <td>27.8</td>
    </tr>
    <tr>
      <td>FreeLB</td>
      <td>91.2</td>
      <td>27.4</td>
      <td>28.1</td>
    </tr>
    <tr>
      <td>InfoBERT</td>
      <td>91.9</td>
      <td>34.4</td>
      <td>35.9</td>
    </tr>
    <tr>
      <td>RobustT₃₀%</td>
      <td>91.5</td>
      <td>47.2</td>
      <td>46.0</td>
    </tr>
  </tbody>
</table>

［#36］
Table 2: Adversarial robustness evaluation of RobustT on QNLI, MNLI and QQP datasets. Compare with the original BERT, fine-tuning on robust tickets improves the adversarial robustness.

［#37］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Clean%</th>
      <th>Aua%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">IMDB</td>
      <td>RobustT₂₀%</td>
      <td>93.8</td>
      <td>55.6</td>
    </tr>
    <tr>
      <td>w/o Mask Leaning</td>
      <td>94.0</td>
      <td>15.1</td>
    </tr>
    <tr>
      <td>w/o Adv</td>
      <td>93.4</td>
      <td>5.4</td>
    </tr>
    <tr>
      <td rowspan="3">AGNEWS</td>
      <td>RobustT₄₀%</td>
      <td>94.9</td>
      <td>28.5</td>
    </tr>
    <tr>
      <td>w/o Mask Learning</td>
      <td>94.2</td>
      <td>16.1</td>
    </tr>
    <tr>
      <td>w/o Adv</td>
      <td>94.5</td>
      <td>8.8</td>
    </tr>
    <tr>
      <td rowspan="3">SST-2</td>
      <td>RobustT₃₀%</td>
      <td>90.9</td>
      <td>26.7</td>
    </tr>
    <tr>
      <td>w/o Mask Learning</td>
      <td>92.2</td>
      <td>6.2</td>
    </tr>
    <tr>
      <td>w/o Adv</td>
      <td>91.2</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>

［#38］
Table 3: Ablation study on text classification datasets. Aua% is obtained after using TextFooler attack.

［#39］
both AGNEWS and IMDB, the randomly pruned subnetwork loses only about 1 performance point in test accuracy, but performs poorly in adversarial robustness. This suggests that robust tickets are more difficult to discovered than traditional lottery tickets. 4) Robust tickets sacrifice accuracy performance in SST-2 and IMDB. We speculate that this may be due to the trade-off between accuracy and robustness (Tsipras et al., 2019).

［#40］
We also evaluate the performance of our proposed method on more tasks. From Table 2, we can see that our proposed method yields significant improvements of robustness over the original BERT on QNLI, MNLI and QQP datasets. There is a significant improvement even compared with InfoBERT and FreeLB.

## 4.6 Ablation Study
［#41］
To better illustrate the contribution of each component of our method, we perform the ablation study by removing the following components: sparse mask learning (but with IMP instead) and adversarial loss objective (Adv). The test results are shown in Table 3. We can observe that: 1)

［#42］
![](./images/867759989651931605_2.jpg)

［#43］
Figure 2: Fine-tuning evaluation results of the robust ticket, the traditional lottery ticket, FreeLB and the original BERT fine-tuning under various sparsity levels. The adversarial robustness improves as the compression ratio grows until a certain threshold, then the robustness deteriorates. Aua% is obtained after using TextFooler attack.

［#44］
Mask learning is important for performance and IMP does not identify robust subnetworks well (Vemparala et al., 2021). 2) Without adversarial loss objective, the proposed method identifies subnetworks that perform well in terms of clean accuracy, but does not provide any improvement in terms of robustness.

## 5 Discussion
［#45］
In this section, we study how the implementation of robust tickets affects the model's robustness.

### 5.1 Impact of Sparsity on Robust Tickets
［#46］
The proposed method can prune out a subnetwork with arbitrary sparsity based on the confidence of masks. In Fig.2, we compare the robust tickets and traditional lottery tickets across all sparsities. When the sparsity increases to a certain level, the robustness decreases faster than the accuracy, which indicates that the robustness is more likely to be affected by the model structure than the accuracy. Therefore, it is more difficult to find a robust ticket from BERT. The accuracy of the subnetwork is slowly decreasing with increasing sparsity, but the robustness shows a different trend. The change in robustness can be roughly divided into three phases: The robustness improves as the sparsity grows until a certain threshold; beyond this threshold, the robustness deteriorates but is still better than that of the lottery tickets. In the end, when being highly compressed, the robust network collapses into a lottery network. A similar phenomenon is also be observed (Liang et al., 2021). The robustness performance curve is not as smooth as the accuracy, this may be due to the gap between the adversarial loss objective and the real textual attacks.

### 5.2 Sparsity Pattern
［#47］
Fig.3 shows the sparsity patterns of robust tickets on all six datasets. We can clearly find that the pruning rate increases from bottom to top on the text classification tasks (IMDB, SST2, AGNEWS), while it is more uniform in the natural language inference tasks (MNLI and QNLI) and Quora question pairs (QQP). Recent works show that BERT encodes a rich hierarchy of linguistic information. Taking the advantage of the probing task, Jawahar et al. (2019) indicate that the surface information features are encoded at the bottom, syntactic information features are in the middle network, and semantic information features in the top. Therefore, we speculate that the sparsity pattern of robust tickets is task-dependent.

### 5.3 Speedup Training Process
［#48］
An important property of winning tickets is to accelerate the convergence of the training process (Chen et al., 2021; You et al., 2020). The training curve in Fig.4 shows that the convergence speed of robust tickets is much faster compared with the default fine-tuning and FreeLB. Moreover, the convergence rate of both accuracy and robustness

［#49］
![](./images/867759989651931605_3.jpg)

［#50］
(a) IMDB

［#51］
![](./images/867759989651931605_4.jpg)

［#52］
(b) SST-2

［#53］
![](./images/867759989651931605_5.jpg)

［#54］
(c) AGNEWS

［#55］
![](./images/867759989651931605_6.jpg)

［#56］
(d) MNLI

［#57］
![](./images/867759989651931605_7.jpg)

［#58］
(e) QNLI

［#59］
![](./images/867759989651931605_8.jpg)

［#60］
(f) QQP

［#61］
Figure 3: Heatmaps of sparsity patterns found on different tasks, each cell gives the percentage of surviving weights in self-attention heads and MLPs. The sparsity patterns on IMDB and SST-2 are similar, which may be due to the fact that they are both text classification datasets based on movie reviews.

［#62］
is accelerating. The traditional lottery tickets converge faster than our method, which may be due to the fact that robust tickets require maintaining a trade-off between robustness and accuracy.

### 5.4 The Importance of Robust Tickets Initialization and Structure

［#63］
To better understand which factor, initialization or structure, has a greater impact on the robust ticket, we conduct corresponding analysis studies. We avoid the effect of initializations by re-initializing the weights of robust tickets. To avoid the effect of structures and preserve the effect of initializations, we use the full BERT and re-initialize the weights that are not contained in the robust tickets. $\text{Aua}\%$ is obtained after using TextFooler attack. The results are shown in Table 4.

#### 5.4.1 Importance of initialization

［#64］
LTH suggests that the winning tickets can not be learned effectively without its original initialization. For our robust BERT tickets, their initializations are pre-trained weights. Table 4 shows the failure of robust tickets when the random re-initialization is performed.

［#65］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Clean$\%$</th>
      <th>Aua$\%$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">IMDB</td>
      <td>RobustT$_{20\%}$</td>
      <td>93.7</td>
      <td>55.6</td>
    </tr>
    <tr>
      <td>w/o Initialization</td>
      <td>87.9</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>w/o Structure</td>
      <td>93.7</td>
      <td>13.4</td>
    </tr>
    <tr>
      <td>w/o Structure+Longer</td>
      <td>93.6</td>
      <td>18.6</td>
    </tr>
    <tr>
      <td rowspan="4">AGNEWS</td>
      <td>RobustT$_{40\%}$</td>
      <td>94.9</td>
      <td>28.5</td>
    </tr>
    <tr>
      <td>w/o Initialization</td>
      <td>92.4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>w/o Structure</td>
      <td>94.9</td>
      <td>21.8</td>
    </tr>
    <tr>
      <td>w/o Structure+Longer</td>
      <td>94.8</td>
      <td>24.6</td>
    </tr>
    <tr>
      <td rowspan="4">SST-2</td>
      <td>RobustT$_{30\%}$</td>
      <td>90.9</td>
      <td>26.7</td>
    </tr>
    <tr>
      <td>w/o Initialization</td>
      <td>83.1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <td>w/o Structure</td>
      <td>92.0</td>
      <td>15.7</td>
    </tr>
    <tr>
      <td>w/o Structure+Longer</td>
      <td>91.9</td>
      <td>27.5</td>
    </tr>
  </tbody>
</table>

［#66］
Table 4: Importance of robust ticket initialization and structure. Our results show that the initialization of robust tickets seems to be more important than the structure, although both of them play a role.

#### 5.4.2 Importance of structure

［#67］
Frankle and Carbin (2019) hypothesize that the structure of winning tickets encodes an inductive bias customized for the learning task at hand. Although removing this inductive bias reduces performance compared to the robust tickets, it still outperforms the original BERT, and its

［#68］
![](./images/867759989651931605_9.jpg)

［#69］
Figure 4: Clean accuracy and accuracy under attack as training proceeds. Robust tickets accelerate both accuracy and robustness. Aua% is obtained after using TextFooler attack.

［#70］
performance improves further with longer training time (3 epochs $\rightarrow$ 10 epochs). It can be seen that the initializations of some pre-training weights may lead to a decrease in the robustness of the model.

## 6 Conclusion
［#71］
In this paper, we articulate and demonstrate the Robust Lottery Ticket Hypothesis for PLMs: the full PLM contains subnetworks (robust tickets) that can achieve a better robustness performance. We propose an effective method to solve the ticket selection problem by encouraging weights that are not responsible for robustness to become exactly zero. Experiments on various tasks corroborate the effectiveness of our method. We also find that pre-trained weights may be a key factor affecting the robustness on downstream tasks.

## Acknowledgements
［#72］
The authors wish to thank the anonymous reviewers for their helpful comments. This work was partially funded by National Natural Science Foundation of China (No. 62076069, 61976056). This research was supported by Meituan, Beijing Academy of Artificial Intelligence(BAAI), and CAAI-Huawei MindSpore Open Fund.

## References




















































## A The Effect of Regularization Strength during Mask Learning

［#73］
In section 3.2.2, we show the mask learning curves for various regularization strengths $\lambda$ in SST-2 dataset. The results on more datasets are shown in the Fig.5, where we can observe that the mask learning process is insensitive to the regularization strength, and the convergence of masks is eventually achieved.

## B Implementation Details

### B.1 Details for Fine-tuning Models

［#74］
We report the hyperparameters used for fine-tuning the BERT-base and retraining the winning tickets in table 5.

［#75］
<table>
  <thead>
    <tr>
      <th>Hyperparameters</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Optimizer</td>
      <td>Adamw(Loshchilov and Hutter, 2019)</td>
    </tr>
    <tr>
      <td>Learning rate</td>
      <td>$2 \times 10^{-5}$</td>
    </tr>
    <tr>
      <td>Dropout</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>Weight decay</td>
      <td>$1 \times 10^{-2}$</td>
    </tr>
    <tr>
      <td>Batch size</td>
      <td>16 or 32</td>
    </tr>
    <tr>
      <td>Gradient clip</td>
      <td>$(-1, 1)$</td>
    </tr>
    <tr>
      <td>Epochs</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Bias-correction</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

［#76］
Table 5: Hyperparameters used for fine-tuning the BERT-base and retraining the winning tickets.

### B.2 Details for Adversarial Attack

［#77］
We use textattack (Morris et al., 2020) to implement the adversarial attack methods. For all attack methods, we use the default parameters of third-party libraries. Adversarial robustness evaluation metrics (e.g., Aua% and #Query) are evaluated on the all 872 test samples for SST-2, 500 randomly selected test samples for IMDB, and 1000 randomly selected test samples for other datasets.

### B.3 Hyperparameters

［#78］
Adversarial loss objective introduces four widely used hyperparameters: the perturbation step size $\eta$, the initial magnitude of perturbations $\epsilon_0$, the number of adversarial steps $s$, and we do not constrain the bound of perturbations. In addition, we also report two important hyperparameters during mask learning. They are mask learning rate $\gamma$ and regularization penalty coefficient $\lambda$. The weight decay $wd$ in the optimizer are also changed compared with default settings to make the mask sparsity rate converge better. We list the hyperparameters used for each tasks in Table 6.

［#79］
<table>
  <thead>
    <tr>
      <th>Datasets</th>
      <th>$\eta$</th>
      <th>$\gamma$</th>
      <th>$\lambda$</th>
      <th>$\epsilon_0$</th>
      <th>$s$</th>
      <th>$wd$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SST2</td>
      <td>0.03</td>
      <td>0.1</td>
      <td>0.5</td>
      <td>0.05</td>
      <td>5</td>
      <td>$1e-6$</td>
    </tr>
    <tr>
      <td>AGNEWS</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>0.5</td>
      <td>0.05</td>
      <td>5</td>
      <td>$1e-6$</td>
    </tr>
    <tr>
      <td>IMDB</td>
      <td>0.03</td>
      <td>0.1</td>
      <td>0.5</td>
      <td>0.05</td>
      <td>5</td>
      <td>$1e-6$</td>
    </tr>
    <tr>
      <td>QQP</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.1</td>
      <td>0.05</td>
      <td>3</td>
      <td>$1e-6$</td>
    </tr>
    <tr>
      <td>QNLI</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.1</td>
      <td>0.05</td>
      <td>3</td>
      <td>$1e-6$</td>
    </tr>
    <tr>
      <td>MNLI</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.05</td>
      <td>2</td>
      <td>$1e-6$</td>
    </tr>
  </tbody>
</table>

［#80］
Table 6: Hyperparameters used during mask learning.

［#81］
![](./images/867759989651931605_10.jpg)

［#82］
Figure 5: Effect of regularization strength during mask learning.