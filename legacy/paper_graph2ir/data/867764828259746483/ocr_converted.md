# A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models

［#1］
Yuanxin Liu$^{1,2,3,*}$,Fandong Meng$^5$, Zheng Lin$^{1,4,\dagger}$, Jiangnan Li$^{1,4}$, Peng Fu$^1$, Yanan Cao$^{1,4}$, Weiping Wang$^1$, Jie Zhou$^5$

［#1］
$^1$Institute of Information Engineering, Chinese Academy of Sciences
［#1］
$^2$MOE Key Laboratory of Computational Linguistics, Peking University
［#1］
$^3$School of Computer Science, Peking University
［#1］
$^4$School of Cyber Security, University of Chinese Academy of Sciences
［#1］
$^5$Pattern Recognition Center, WeChat AI, Tencent Inc, China$^\ddagger$

［#1］
liuyuanxin@stu.pku.edu.cn, {fandongmeng,withtomzhou}@tencent.com
{linzheng,lijiangnan,fupeng,caoyanan,wangweiping}@iie.ac.cn

## Abstract

［#2］
Despite the remarkable success of pre-trained language models (PLMs), they still face two challenges: First, large-scale PLMs are inefficient in terms of memory footprint and computation. Second, on the downstream tasks, PLMs tend to rely on the dataset bias and struggle to generalize to out-of-distribution (OOD) data. In response to the efficiency problem, recent studies show that dense PLMs can be replaced with sparse subnetworks without hurting the performance. Such subnetworks can be found in three scenarios: 1) the fine-tuned PLMs, 2) the raw PLMs and then fine-tuned in isolation, and even inside 3) PLMs without any parameter fine-tuning. However, these results are only obtained in the in-distribution (ID) setting. In this paper, we extend the study on PLMs subnetworks to the OOD setting, investigating whether sparsity and robustness to dataset bias can be achieved simultaneously. To this end, we conduct extensive experiments with the pre-trained BERT model on three natural language understanding (NLU) tasks. Our results demonstrate that **sparse and robust subnetworks (SRNets) can consistently be found in BERT**, across the aforementioned three scenarios, using different training and compression methods. Furthermore, we explore the upper bound of SRNets using the OOD information and show that **there exist sparse and almost unbiased BERT subnetworks**. Finally, we present 1) an analytical study that provides insights on how to promote the efficiency of SRNets searching process and 2) a solution to improve subnetworks' performance at high sparsity. The code is available at https://github.com/11yx97/sparse-and-robust-PLM.

## 1 Introduction

［#3］
Pre-trained language models (PLMs) have enjoyed impressive success in natural language processing (NLP) tasks. However, they still face two major problems. On the one hand, the prohibitive model size of PLMs leads to poor efficiency in terms of memory footprint and computational cost [12, 49]. On the other hand, despite being pre-trained on large-scale corpus, PLMs still tend to rely on *dataset bias* [18, 37, 65, 46], i.e., the spurious features of input examples that strongly correlate with the

---
［#4］
$^*$Work was done when Yuanxin Liu was a graduate student of IIE, CAS.
［#5］
$^\dagger$Corresponding author: Zheng Lin.
［#6］
$^\ddagger$Joint work with Pattern Recognition Center, WeChat AI, Tencent Inc, China.

［#7］
36th Conference on Neural Information Processing Systems (NeurIPS 2022).

［#8］
![](./images/867764828259746483_1.jpg)

［#9］
Figure 1: Three kinds of PLM subnetworks obtained from different pruning and fine-tuning paradigms.
(a) Pruning a fine-tuned PLM. (b) Pruning the PLM and then fine-tuning the subnetwork. (c) Pruning
the PLM without fine-tuning model parameters. The obtained subnetworks are used for testing.

［#10］
label, during downstream fine-tuning. These two problems pose great challenge to the real-world
deployment of PLMs, and they have triggered two separate lines of works.

［#11］
In terms of the efficiency problem, some recent studies resort to sparse subnetworks as alternatives to
the dense PLMs. [27, 38, 30] compress the fine-tuned PLMs in a post-hoc fashion. [4, 40, 32, 28]
extend the *Lottery Ticket Hypothesis* (LTH) [9] to search PLMs subnetworks that can be fine-tuned
in isolation. Taking one step further, [66] propose to learn task-specific subnetwork structures via
mask training [23, 35], without fine-tuning any pre-trained parameter. Fig. 1 illustrates these three
paradigms. Encouragingly, the empirical evidences suggest that PLMs can indeed be replaced with
sparse subnetworks without compromising the in-distribution (ID) performance.

［#12］
To address the dataset bias problem, numerous debiasing methods have been proposed. A prevailing
category of debiasing methods [5, 54, 25, 20, 46, 13, 55] adjust the importance of training examples,
in terms of training loss, according to their bias degree, so as to reduce the impact of biased examples
(examples that can be correctly classified based on the spurious features). As a result, the model is
forced to rely less on the dataset bias during training and generalizes better to OOD situations.

［#13］
Although progress has been made in both directions, most existing work tackle the two problems
independently. To facilitate real-world application of PLMs, the problems of robustness and efficiency
should be addressed simultaneously. Motivated by this, we extend the study on PLM subnetwork
to the OOD scenario, investigating **whether there exist PLM subnetworks that are both sparse
and robust against dataset bias?** To answer this question, we conduct large-scale experiments
with the pre-trained BERT model [6] on three natural language understanding (NLU) tasks that
are widely-studied in the question of dataset bias. We consider a variety of setups including the
three pruning and fine-tuning paradigms, standard and debiasing training objectives, different model
pruning methods, and different variants of PLMs from the BERT family. Our results show that **BERT
does contain sparse and robust subnetworks (SRNets)** within certain sparsity constraint (e.g., less
than 70%), giving affirmative answer to the above question. Compared with a standard fine-tuned
BERT, SRNets exhibit comparable ID performance and remarkable OOD improvement. When it
comes to BERT model fine-tuned with debiasing method, SRNets can preserve the full model's ID
and OOD performance with much fewer parameters. On this basis, we further explore the upper
bound of SRNets by making use of the OOD information, which reveals that **there exist sparse and
almost unbiased subnetworks, even in a standard fine-tuned BERT that is biased**.

［#14］
Regardless of the intriguing properties of SRNets, we find that the subnetwork searching process
still have room for improvement, based on some observations from the above experiments. First,
we study the timing to start searching SRNets during full BERT fine-tuning, and find that the entire
training and searching cost can be reduced from this perspective. Second, we refine the mask training
method with gradual sparsity increase, which is quite effective in identifying SRNets at high sparsity.

［#15］
Our main contributions are summarized as follows:
- We extend the study on PLMs subnetworks to the OOD scenario. To our knowledge, this paper presents the first systematic study on sparsity and dataset bias robustness for PLMs.
- We conduct extensive experiments to demonstrate the existence of sparse and robust BERT subnetworks, across different pruning and fine-tuning setups. By using the OOD information, we further reveal that there exist sparse and almost unbiased BERT subnetworks.
- We present analytical studies and solutions that can help further refine the SRNets searching process in terms of efficiency and the performance of subnetworks at high sparsity.

## 2 Related Work

### 2.1 BERT Compression
［#16］
Studies on BERT compression can be divided into two classes. The first one focuses on the design of model compression techniques, which include pruning [15, 38, 11], knowledge distillation [44, 50, 24, 31], parameter sharing [26], quantization [61, 64], and combining multiple techniques [51, 36, 30]. The second one, which is based on the lottery ticket hypothesis [9], investigates the compressibility of BERT on different phases of the pre-training and fine-tuning paradigm. It has been shown that BERT can be pruned to a sparse subnetwork after [11] and before fine-tuning [4, 40, 28, 32, 15], without hurting the accuracy. Moreover, [66] show that directly learning subnetwork structures on the pre-trained weights can match fine-tuning the full BERT. In this paper, we follow the second branch of works, and extend the evaluation of BERT subnetworks to the OOD scenario.

### 2.2 Dataset Bias in NLP Tasks
［#17］
To facilitate the development of NLP systems that truly learn the intended task solution, instead of relying on dataset bias, many efforts have been made recently. On the one hand, challenging OOD test sets are constructed [18, 37, 65, 46, 1] by eliminating the spurious correlations in the training sets, in order to establish more strict evaluation. On the other hand, numerous debiasing methods [5, 54, 25, 20, 46, 13, 55] are proposed to discourage the model from learning dataset bias during training. However, few attention has been paid to the influence of pruning on the OOD generalization ability of PLMs. This work presents a systematic study on this question.

### 2.3 Model Compression and Robustness
［#18］
Some pioneer attempts have also been made to obtain models that are both compact and robust to adversarial attacks [16, 60, 48, 10, 59] and spurious correlations [62, 8]. Specially, [59, 8] study the compression and robustness question on PLM. Different from [59], which is based on adversarial robustness, we focus on the spurious correlations, which is more common than the worst-case adversarial attack. Compared with [8], which focus on post-hoc pruning of the standard fine-tuned BERT, we thoroughly investigate different fine-tuning methods (standard and debiasing) and subnetworks obtained from the three pruning and fine-tuning paradigms. A more detailed discussion of the relation and difference between our work and previous studies on model compression and robustness is provided in Appendix D.

## 3 Preliminaries

### 3.1 BERT Architecture and Subnetworks
［#19］
BERT is composed of an embedding layer, a stack of Transformer layers [56] and a task-specific classifier. Each Transformer layer has a multi-head self-attention (MHAtt) module and a feed-forward network (FFN). MHAtt has four kinds of weight matrices, i.e., the query, key and value matrices $\mathbf{W}_{Q,K,V} \in \mathbb{R}^{d_{model} \times d_{model}}$, and the output matrix $\mathbf{W}_{AO} \in \mathbb{R}^{d_{model} \times d_{model}}$. FFN consits of two linear layers $\mathbf{W}_{in} \in \mathbb{R}^{d_{model} \times d_{FFN}}$, $\mathbf{W}_{out} \in \mathbb{R}^{d_{FFN} \times d_{model}}$, where $d_{FFN}$ is the hidden dimension of FFN.

［#20］
To obtain the subnetwork of a model $f(\boldsymbol{\theta})$ parameterized by $\boldsymbol{\theta}$, we apply a binary pruning mask $\mathbf{m} \in \{0,1\}^{|\boldsymbol{\theta}|}$ to its weight matrices, which produces $f(\mathbf{m} \odot \boldsymbol{\theta})$, where $\odot$ is the Hadamard product.


［#21］
For BERT, we focus on the $L$ Transformer layers and the classifier. The parameters to be pruned are $\boldsymbol{\theta}_{pr} = \{\mathbf{W}_{\text{cls}}\} \cup \{\mathbf{W}_Q^l, \mathbf{W}_K^l, \mathbf{W}_V^l, \mathbf{W}_{AO}^l, \mathbf{W}_{\text{in}}^l, \mathbf{W}_{\text{out}}^l\}_{l=1}^L$, where $\mathbf{W}_{\text{cls}}$ is the classifier weights.

### 3.2 Pruning Methods
#### 3.2.1 Magnitude-based Pruning
［#22］
Magnitude-based pruning [19, 9] zeros-out parameters with low absolute values. It is usually realized in an iterative manner, namely, iterative magnitude pruning (IMP). IMP alternates between pruning and training and gradually increases the sparsity of subnetworks. Specifically, a typical IMP algorithm consists of four steps: (i) Training the full model to convergence. (ii) Pruning a fraction of parameters with the smallest magnitude. (iii) Re-training the pruned subnetwork. (iv) Repeat (ii)-(iii) until reaching the target sparsity. To obtain subnetworks from the pre-trained BERT, i.e., (b) and (c) in Fig. 1, the subnetwork parameters are rewound to the pre-trained values after (iii), and (i) can be abandoned. More details about our IMP implementations can be found in Appendix A.1.1.

#### 3.2.2 Mask Training
［#23］
Mask training treats the pruning mask $\mathbf{m}$ as trainable parameters. Following [35, 66, 42, 32], we achieve this through binarization in forward pass and gradient estimation in backward pass.

［#24］
Each weight matrix $\mathbf{W} \in \mathbb{R}^{d_1 \times d_2}$, which is frozen during mask training, is associated with a bianry mask $\mathbf{m} \in \{0,1\}^{d_1 \times d_2}$, and a real-valued mask $\hat{\mathbf{m}} \in \mathbb{R}^{d_1 \times d_2}$. In the forward pass, $\mathbf{W}$ is replaced with $\mathbf{m} \odot \mathbf{W}$, where $\mathbf{m}$ is derived from $\hat{\mathbf{m}}$ through binarization:
［#24］
$$
\mathbf{m}_{i,j} =
\begin{cases}
1 & \text{if } \hat{\mathbf{m}}_{i,j} \geq \phi \\
0 & \text{otherwise}
\end{cases} \tag{1}
$$
［#24］
where $\phi$ is the threshold. In the backward pass, since the binarization operation is not differentiable, we use the *straight-through estimator* [3] to compute the gradients for $\hat{\mathbf{m}}$ using the gradients of $\mathbf{m}$, i.e., $\frac{\partial \mathcal{L}}{\partial \mathbf{m}}$, where $\mathcal{L}$ is the loss. Then, $\hat{\mathbf{m}}$ is updated as $\hat{\mathbf{m}} \leftarrow \hat{\mathbf{m}} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{m}}$, where $\eta$ is the learning rate.

［#25］
Following [42, 32], we initialize the real-valued masks according to the magnitude of the original weights. The complete mask training algorithm is summarized in Appendix A.1.2.

### 3.3 Debiasing Methods
［#26］
As described in the Introduction, the debiasing methods measure the bias degree of training examples. This is achieved by training a *bias model*. The inputs to the bias model are hand-crafted spurious features based on our prior knowledge of the dataset bias (Section 4.1.3 describes the details). In this way, the bias model mainly relies on the spurious features to make predictions, which can then serve as a measurement of the bias degree. Specifically, given the bias model prediction $\mathbf{p}_b = (\mathbf{p}_b^1, \cdots, \mathbf{p}_b^K)$ over the $K$ classes, the bias degree $\beta = \mathbf{p}_b^c$, i.e., the the probability of the ground-truth class $c$.

［#27］
Then, $\beta$ can be used to adjust the training loss in several ways, including *product-of-experts* (PoE) [5, 20, 25], *example reweighting* [46, 13] and *confidence regularization* [54]. Here we describe the standard cross-entropy and PoE, and the other two methods are introduced in Appendix A.2.

［#28］
Standard Cross-Entropy computes the cross-entropy between the predicted distribution $\mathbf{p}_m$ and the ground-truth one-hot distribution $\mathbf{y}$ as $\mathcal{L}_{\text{std}} = -\mathbf{y} \cdot \log \mathbf{p}_m$.

［#29］
Product-of-Experts combines the predictions of main model and bias model, i.e., $\mathbf{p}_b$ and $\mathbf{p}_m$, and then computes the training loss as $\mathcal{L}_{\text{poe}} = -\mathbf{y} \cdot \log \text{softmax}\left(\log \mathbf{p}_m + \log \mathbf{p}_b\right)$.

### 3.4 Notations
［#30］
Here we define some notations, which will be used in the following sections.
- $\mathcal{A}_{\mathcal{L}}^t(f(\boldsymbol{\theta}))$: Training $f(\boldsymbol{\theta})$ with loss $\mathcal{L}$ for $t$ steps, where $t$ can be omitted for simplicity.
- $\mathcal{P}_{\mathcal{L}}^p(f(\boldsymbol{\theta}))$: Pruning $f(\boldsymbol{\theta})$ using pruning method $p$ and training loss $\mathcal{L}$.
- $\mathcal{M}(f(\mathbf{m}\boldsymbol{\theta}))$: Extracting the pruning mask of $f(\mathbf{m}\boldsymbol{\theta})$, i.e., $\mathcal{M}(f(\mathbf{m}\boldsymbol{\theta})) = \mathbf{m}$.


［#31］
- $\mathcal{L} \in \{\mathcal{L}_{\text{std}}, \mathcal{L}_{\text{poe}}, \mathcal{L}_{\text{reweight}}, \mathcal{L}_{\text{confreg}}\}$ and $p \in \{\text{imp}, \text{imp-rw}, \text{mask}\}$, where "imp" and "imp-rw"denote the standard IMP and IMP with weight rewinding, as described in Section 3.2.1. "mask" stands for mask training.
- $\mathcal{E}_d(f(\boldsymbol{\theta}))$: Evaluating $f(\boldsymbol{\theta})$ on the test data with distribution $d \in \{\text{ID}, \text{OOD}\}$.

## 4 Sparse and Robust BERT Subnetworks

### 4.1 Experimental Setups

#### 4.1.1 Datasets and Evaluation
［#32］
Natural Language Inference We use MNLI [57] as the ID dataset for NLI. MNLI is comprised of premise-hypothesis pairs, whose relationship may be entailment, contradiction, or neutral. In MNLI the word overlap between premise and hypothesis is strongly correlated with the entailment class. To solve this problem, the OOD HANS dataset [37] is built so that such correlation does not hold.

［#33］
Paraphrase Identification The ID dataset for paraphrase identification is QQP $^4$, which contains question pairs that are labelled as either duplicate or non-duplicate. In QQP, high lexical overlap is also strongly associated with the duplicate class. The OOD datasets PAWS-qqp and PAWS-wiki [65] are built from sentences in Quora and Wikipedia respectively. In PAWS sentence pairs with high word overlap have a balanced distribution over duplicate and non-duplicate.

［#34］
Fact Verification FEVER $^5$ [52] is adopted as the ID dataset of fact verification, where the task is to assess whether a given evidence supports or refutes the claim, or whether there is not-enough-info to reach a conclusion. The OOD dataset Fever-Symmetric (v1 and v2) [46] is proposed to evaluate the influence of the claim-only bias (the label can be predicted correctly without the evidence).

［#35］
For NLI and fact verification, we use Accuracy as the evaluation metric. For paraphrase identification, we evaluate using the F1 score. More details of datasets and evaluation are shown in Appendix B.1.

#### 4.1.2 PLM Backbone
［#36］
We mainly experiment with the BERT-base-uncased model [6]. It has roughly 110M parameters in total, and 84M parameters in the Transformer layers. As described in Section 3.1, we derive the subnetworks from the Transformer layers and report sparsity levels relative to the 84M parameters. To generalize our conclusions to other PLMs, we also consider two variants of the BERT family, namely RoBERTa-base and BERT-large, the results of which can be found in Appendix C.5.

#### 4.1.3 Training Details
［#37］
Following [5], we use a simple linear classifier as the bias model. For HANS and PAWS, the spurious features are based on the the word overlapping information between the two input text sequences. For Fever-Symmetric, the spurious features are max-pooled word embeddings of the claim sentence. More details about the bias model and the spurious features are presented in Appendix B.3.1.

［#38］
Mask training and IMP basically use the same hyper-parameters (adopting from [55]) as full BERT. An exception is longer training, because we find that good subnetworks at high sparsity levels require more training to be found. Unless otherwise specified, we select the best checkpoints based on the ID dev performance, without using OOD information. All the reported results are averaged over 4 runs. We defer training details about each dataset, and each training and pruning setup, to Appendix B.3.

### 4.2 Subnetworks from Fine-tuned BERT

#### 4.2.1 Problem Formulation and Experimental Setups
［#39］
Given the fine-tuned full BERT $f(\boldsymbol{\theta}_{ft}) = \mathcal{A}_{\mathcal{L}_1}(f(\boldsymbol{\theta}_{pt}))$, where $\boldsymbol{\theta}_{pt}$ and $\boldsymbol{\theta}_{ft}$ are the pre-trained and fine-tuned parameters respectively, the goal is to find a subnetwork $f(\mathbf{m} \odot \boldsymbol{\theta}_{ft}') = \mathcal{P}_{\mathcal{L}_2}^p(f(\boldsymbol{\theta}_{ft}))$ that

［#39］
$^4$https://www.kaggle.com/c/quora-question-pairs
［#39］
$^5$See the licence information at https://fever.ai/download/fever/license.html


［#40］
![](./images/867764828259746483_2.jpg)

［#41］
Figure 2: Results of subnetworks pruned from the CE fine-tuned BERT. "std" means standard, and the shadowed areas denote standard deviations, which also apply to the other figures of this paper.

［#42］
![](./images/867764828259746483_3.jpg)

［#43］
Figure 3: Results of subnetworks pruned from the PoE fine-tuned BERT. Results of the "mask train (poe)" subnetworks from Fig. 2 (the orange line) are also reported for reference.

［#44］
satisfies a target sparsity level $s$ and maximize the ID and OOD performance.

［#45］
$$
\max _{\mathbf{m}, \boldsymbol{\theta}_{f t}^{\prime}}\left(\mathcal{E}_{\mathrm{ID}}\left(f\left(\mathbf{m} \odot \boldsymbol{\theta}_{f t}^{\prime}\right)\right)+\mathcal{E}_{\mathrm{OOD}}\left(f\left(\mathbf{m} \odot \boldsymbol{\theta}_{f t}^{\prime}\right)\right)\right), \text { s.t. } \frac{\|\mathbf{m}\|_{0}}{\left|\boldsymbol{\theta}_{p r}\right|}=(1-s)
\tag{2}
$$

［#45］
where $\|\cdot\|_{0}$ is the $L_{0}$ norm and $|\boldsymbol{\theta}_{p r}|$ is the total number of parameters to be pruned. In practice, the above optimization problem is achieved via $\mathcal{P}_{\mathcal{L}_{2}}^{p}()$, which minimizes the loss $\mathcal{L}_{2}$ on the ID training set. When the pruning method is IMP, the subnetwork parameters will be further fine-tuned and $\boldsymbol{\theta}_{f t}^{\prime} \neq \boldsymbol{\theta}_{f t}$. For mask training, only the subnetwork structure is updated and $\boldsymbol{\theta}_{f t}^{\prime}=\boldsymbol{\theta}_{f t}$.

［#46］
We consider two kinds of fine-tuned full BERT, which utilize the standard CE loss and PoE loss respectively (i.e., $\mathcal{L}_{1} \in\{\mathcal{L}_{\text {std}}, \mathcal{L}_{\text {poe}}\}$). IMP and mask training are used as the pruning methods (i.e., $p \in\{$imp, mask$\}$). For the standard fine-tuned BERT, both $\mathcal{L}_{\text {std}}$ and $\mathcal{L}_{\text {poe}}$ are examined in the pruning process. For the PoE fine-tuned BERT, we only use $\mathcal{L}_{\text {poe}}$ during pruning. Note that in this work, we mainly experiment with $\mathcal{L}_{\text {std}}$ and $\mathcal{L}_{\text {poe}}$. $\mathcal{L}_{\text {reweight}}$ and $\mathcal{L}_{\text {confreg}}$ are also examined for subnetworks from fine-tuned BERT, the results of which can be found in Appendix C.1.

### 4.2.2 Results

［#47］
Subnetworks from Standard Fine-tuned BERT The results are shown in Fig. 2 (In this paper, we present most results in figures for clear comparisons. Actual values of the results can be found in the code link.). We discuss them from three perspectives. For the full BERT, we can see that standard

［#48］
![](./images/867764828259746483_4.jpg)

［#49］
Figure 4: Results of BERT subnetworks fine-tuned in isolation. "ft" is short for fine-tuning.

［#50］
CE fine-tuning, which achieves good results on the ID dev sets, performs significantly worse on the OOD test sets. This demonstrates that the ID performance of BERT depends, to a large extent, on memorizing the dataset bias.

［#51］
In terms of the subnetworks, we can derive the following observations: (1) Using any of the four pruning methods, we can compress a large proportion of the BERT parameters (up to 70% sparsity) and still preserve 95% of the full model's ID performance. (2) With standard pruning, i.e., "mask train (std)" or "imp (std)", we can observe small but perceivable improvement over the full BERT on the HANS and PAWS datasets. This suggests that pruning may remove some parameters related to the bias features. (3) The OOD performance of "mask train (poe)" and "imp (poe)" subnetworks is even better, and the ID performance degrades slightly but is still above 95% of the full BERT. This shows that introducing the debiasing objective in the pruning process is beneficial. Specially, as mask training does not change the model parameters, the results of "mask train (poe)" implicates that the biased "full bert (std)" contains sparse and robust subnetworks (SRNets) that already encode a less biased solution to the task. (4) SRNets can be identified across a wide range of sparsity levels (from 20% ~ 70%). However at higher sparsity of 90%, the performance of the subnetworks is not desirable. (5) We also find that there is an abnormal increase of the PAWS F1 score at 70% ~ 90% sparsity for some pruning methods, when the corresponding ID performance drops sharply. This is because the class distribution of PAWS is imbalanced (see Appendix B.1), and thus even a naive random-guessing model can outperform the biased full model on PAWS. Therefore, the OOD improvement should only be acceptable when there is no large ID performance decline.

［#52］
Comparing IMP and mask training, the latter performs better in general, except for "mask train (poe)" at 90% sparsity on QQP and FEVER. This suggests that directly optimizing the subnetwork structure is a better choice than using the magnitude heuristic as the pruning metric.

［#53］
Subnetworks from PoE Fine-tuned BERT Fig. 3 presents the results. We can find that: (1) For the full BERT, the OOD performance is obviously promoted with the PoE debiasing method, while the ID performance is sacrificed slightly. (2) Unlike the subnetworks from the standard fine-tuned BERT, the subnetworks of PoE fine-tuned BERT (the green and blue lines) cannot outperform the full model. However, these subnetworks maintain comparable performance at up to 70% sparsity, on both the ID and OOD settings, making them desirable alternatives to the full model in resource-constraint scenarios. Moreover, this phenomenon suggests that there is a great redundancy of BERT parameters, even when OOD generalization is taken into account. (3) With PoE-based pruning, subnetworks from the standard fine-tuned BERT (the orange line) is comparable with subnetworks from the PoE fine-tuned BERT (the blue line). This means we do not have to fine-tune a debiased BERT before searching for the SRNets. (4) IMP, again, slightly underperforms mask training at moderate sparsity levels, while it is better at 90% sparsity on the fact verification task.

### 4.3 BERT Subnetworks Fine-tuned in Isolation

#### 4.3.1 Problem Formulation and Experimental Setups
［#54］
Given the pre-trained BERT $f(\boldsymbol{\theta}_{pt})$, a subnetwork $f(\mathbf{m} \odot \boldsymbol{\theta}_{pt})$ is obtained before downstream fine-tuning. The goal is to maximize the performance of the fine-tuned subnetwork $\mathcal{A}_{\mathcal{L}_1}(f(\mathbf{m} \odot \boldsymbol{\theta}_{pt}))$:
［#54］
$$
\max _{\mathbf{m}}\left(\mathcal{E}_{\mathrm{ID}}\left(\mathcal{A}_{\mathcal{L}_{1}}\left(f\left(\mathbf{m} \odot \boldsymbol{\theta}_{p t}\right)\right)\right)+\mathcal{E}_{\mathrm{OOD}}\left(\mathcal{A}_{\mathcal{L}_{1}}\left(f\left(\mathbf{m} \odot \boldsymbol{\theta}_{p t}\right)\right)\right)\right), \text { s.t. } \frac{\|\mathbf{m}\|_{0}}{\left|\boldsymbol{\theta}_{p r}\right|}=(1-s) \tag{3}
$$

［#55］
Following the LTH [9], we solve this problem using the train-prune-rewind pipeline. For IMP, the procedure is described in Section 3.2.1 and $\mathbf{m} = \mathcal{M}(\mathcal{P}_{\mathcal{L}_{2}}^{\text{imp-rw}}(f(\boldsymbol{\theta}_{pt})))$. For mask training, the subnetwork structure is learned from $f(\boldsymbol{\theta}_{ft})$ (same as the previous section) and $\mathbf{m} = \mathcal{M}(\mathcal{P}_{\mathcal{L}_{2}}^{\text{mask}}(f(\boldsymbol{\theta}_{ft})))$.

［#56］
We employ CE and PoE loss for model fine-tuning (i.e., $\mathcal{L}_1 \in \{\mathcal{L}_{\text{std}}, \mathcal{L}_{\text{poe}}\}$). Since we have shown that using the debiasing loss in pruning is conducive, the CE loss is not considered (i.e., $\mathcal{L}_2 = \mathcal{L}_{\text{poe}}$).

#### 4.3.2 Results
［#57］
The results of subnetworks fine-tuned in isolation are presented in Fig. 4. It can be found that: (1) For standard CE fine-tuning, the "mask train (poe)" subnetworks are superior to "full bert (std)" on the OOD test data, i.e., the subnetworks are less susceptible to the dataset bias during training. (2) In terms of the PoE-based fine-tuning, the "imp (poe)" and "mask train (poe)" subnetworks are generally comparable to "full bert (poe)". (3) For most of the subnetworks, "poe ft" clearly outperforms "std ft" in the OOD setting, which suggests that it is important to use the debiasing method in fine-tuning, even if the BERT subnetwork structure has already encoded some unbiased information.

［#58］
Moreover, based on (1) and (2), we can extend the LTH on BERT [4, 40, 28, 32]: **The pre-trained BERT contains SRNets that can be fine-tuned in isolation, using either standard or debiasing method, and match or even outperform the full model in both the ID and OOD evaluations.**

### 4.4 BERT Subnetworks Without Fine-tuning

#### 4.4.1 Problem Formulation and Experimental Setups
［#59］
This setup aims at finding a subnetwork $f(\mathbf{m} \odot \boldsymbol{\theta}_{pt})$ inside the pre-trained BERT, which can be directly employed to a task. The problem is formulated as:
［#59］
$$
\max _{\mathbf{m}}\left(\mathcal{E}_{\mathrm{ID}}\left(f\left(\mathbf{m} \odot \boldsymbol{\theta}_{p t}\right)\right)+\mathcal{E}_{\mathrm{OOD}}\left(f\left(\mathbf{m} \odot \boldsymbol{\theta}_{p t}\right)\right)\right), \text { s.t. } \frac{\|\mathbf{m}\|_{0}}{\left|\boldsymbol{\theta}_{p r}\right|}=(1-s) \tag{4}
$$

［#60］
Following [66], we fix the pre-trained parameters $\boldsymbol{\theta}_{pt}$ and optimize the mask variables $\mathbf{m}$. This process can be represented as $\mathcal{P}_{\mathcal{L}}^{\text{mask}}(f(\boldsymbol{\theta}_{pt}))$, where $\mathcal{L} \in \{\mathcal{L}_{\text{std}}, \mathcal{L}_{\text{poe}}\}$.

#### 4.4.2 Results
［#61］
As we can see in Fig. 5: (1) With CE-based mask training, the identified subnetworks (under 50% sparsity) in pre-trained BERT are competitive with the CE fine-tuned full BERT. (2) Similarly, using PoE-based mask training, the subnetworks under 50% sparsity are comparable to the PoE fine-tuned full BERT, which demonstrates that SRNets for a particular downstream task already exist in the pre-trained BERT. (3) "mask train (poe)" subnetworks in pre-trained BERT can even match the subnetworks found in the fine-tuned BERT (the orange lines) in some cases (e.g., on PAWS and on FEVER under 50% sparsity). Nonetheless, the latter exhibits a better overall performance.

### 4.5 Sparse and Unbiased BERT Subnetworks

#### 4.5.1 Problem Formulation and Experimental Setups
［#62］
To explore the upper bound of BERT subnetworks in terms of OOD generalization, we include the OOD training data in mask training, and use the OOD test sets for evaluation. Like the previous sections, we investigate three pruning and fine-tuning paradigms, as formulated by Eq. 2, 3 and 4 respectively. We only consider the standard CE for subnetwork and full BERT fine-tuning, which is more vulnerable to the dataset bias. Appendix B.3.3 summarizes the detailed experimental setups.


［#63］
![](./images/867764828259746483_5.jpg)

［#64］
Figure 5: Results of BERT subnetworks without fine-tuning. Results of the "mask train (poe)" subnetworks from Fig. 2 (the orange line) are also reported for reference.

［#65］
![](./images/867764828259746483_6.jpg)

［#66］
Figure 6: NLI results of BERT subnetworks found using the OOD information. Results of the other two tasks can be found in Appendix C.2.

［#67］
![](./images/867764828259746483_7.jpg)

［#68］
Figure 7: NLI mask training curves (70% sparse), starting from BERT fine-tuned for varied steps. Appendix C.3 shows results of the other two tasks.

### 4.5.2 Results
［#69］
From Fig. 6 we can observe that: (1) The subnetworks from fine-tuned BERT ("bert-ft subnet") at $20\% \sim 70\%$ sparsity achieve nearly $100\%$ accuracy on HANS, and their ID performance is also close to the full BERT. (2) The subnetworks in the pre-trained BERT ("bert-pt subnet") also have very high OOD accuracy, while they perform worse than "bert-ft subnet" in the ID setting. (3) "bert-pt subnet + ft" subnetworks, which are fine-tuned in isolation with CE loss, exhibits the best ID performance, and the poorest OOD performance. However, compared to the full BERT, these subnetworks still rely much less on the dataset bias, reaching nearly $90\%$ HANS accuracy at $50\%$ sparsity. Jointly, these results show that there consistently exist BERT subnetworks that are almost unbiased towards the MNLI training set bias, under the three kinds of pruning and fine-tuning paradigms.

## 5 Refining the SRNets Searching Process
［#70］
In this section, we study how to further improve the SRNets searching process based on mask training, which generally performs better than IMP, as shown in Section 4.2 and Section 4.3.

### 5.1 The Timing to Start Searching SRNets
［#71］
Compared with searching subnetworks from the fine-tuned BERT, directly searching from the pre- trained BERT is more efficient in that it dispenses with fine-tuning the full model. However, the former has a better overall performance, as we have shown in Section 4.4. This induces a question: **At which point of the BERT fine-tuning process, can we find subnetworks comparable to those found after the end of fine-tuning using mask training?** To answer this question, we perform mask training on the model checkpoints $f(\boldsymbol{\theta}_t) = \mathcal{A}_{\mathcal{L}_{std}}^t(f(\boldsymbol{\theta}_{pt}))$ from different steps $t$ of BERT fine-tuning.

［#72］
![](./images/867764828259746483_8.jpg)

［#73］
Figure 8: Comparison between fixed sparsity and gradual sparsity increase for mask training with the standard fine-tuned full BERT. The subnetworks are at $90\%$ sparsity.

［#74］
Fig. 7 shows the mask training curves, which start from different $f(\boldsymbol{\theta}_t)$. We can see that "ft step=0" converges slower and to a worse final accuracy, as compared with "ft to end", especially on the HANS dataset. However, with 20,000 steps of full BERT fine-tuning, which is roughly $55\%$ of the "ft to end", the mask training performance is very competitive. This suggests that the total training cost of SRNet searching can be reduced, by a large amount, in the full model training stage.

［#75］
To actually reduce the training cost, we need to predict the exact timing to start mask training. This is intractable without information of all the training curves in Fig. 7. A feasible solution is adopting the idea of early-stopping (see Appendix E.1 for detailed discussions). However, accurately predicting the optimal timing (with the least amount of fine-tuning and comparable subnetwork performance to fully fine-tuning) is indeed difficult and we invite follow-up studies to investigate this question.

### 5.2 SRNets at High Sparsity
［#76］
As the results of Section 4 demonstrate, there is a sharp decline of the subnetworks' performance from $70\% \sim 90\%$ sparsity. We conjecture that this is because directly initializing mask training to $90\%$ reduces the model's capacity too drastically, and thus causes some difficulties in optimization. Therefore, we gradually increase the sparsity from $70\% \sim 90\%$ during mask training, using the cubic sparsity schedule [67] (see Appendix C.4 for ablation studies). Fig. 8 compares the fixed sparsity used in the previous sections and the gradual sparsity increase, across varied mask training epochs. We find that while simply extending the training process is conducive, gradual sparsity increase achieves better results. In particular, "gradual" outperforms "fixed" with lower training cost on all the three tasks, except for the PAWS dataset, A similar phenomenon is explained in Section 4.2.2.

## 6 Conclusions and Limitations
［#77］
In this paper, we investigate whether sparsity and robustness to dataset bias can be achieved simulta- neously for PLM subnetworks. Through extensive experiments, we demonstrate that BERT indeed contains sparse and robust subnetworks (SRNets) across a variety of NLU tasks and training and pruning setups. We further use the OOD information to reveal that there exist sparse and almost unbiased BERT subnetworks. Finally, we present analysis and solutions to refine the SRNet searching process in terms of subnetwork performance and searching efficiency.

［#78］
The limitations of this work is twofold. First, we focus on BERT-like PLMs and NLU tasks, while dataset biases are also common in other scenarios. For example, gender and racial biases exist in dialogue generation systems [7] and PLMs [17]. In the future work, we would like to extend our exploration to other types of PLMs and NLP tasks (see Appendix E.2 for a discussion). Second, as we discussed in Section 5.1, our analysis on "the timing to start searching SRNets" mainly serves as a proof-of-concept, and actually reducing the training cost requires predicting the exact timing.

## Acknowledgments and Disclosure of Funding
［#79］
This work was supported by National Natural Science Foundation of China (61976207 and 61906187).

### References







































































# Checklist
［#80］
1. For all authors...
    (a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]
    (b) Did you describe the limitations of your work? [Yes] See Section 6.
    (c) Did you discuss any potential negative societal impacts of your work? [No] Currently, we think there are no apparent negative societal impacts related to our work.
    (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]
2. If you are including theoretical results...
    (a) Did you state the full set of assumptions of all theoretical results? [N/A]
    (b) Did you include complete proofs of all theoretical results? [N/A]
3. If you ran experiments...

［#81］
(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [No] We will release the codes and reproduction instructions upon publication.

［#82］
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] See Section 4.1 and Appendix B.

［#83］
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] See all the figures of our experiments.

［#84］
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] See Appendix B.

［#85］
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

［#86］
(a) If your work uses existing assets, did you cite the creators? [Yes] See Section 4.1.

［#87］
(b) Did you mention the license of the assets? [Yes] Licenses of some dataset we used are mentioned in Section 4.1. However, for the other datasets, we were unable to find the licenses.

［#88］
(c) Did you include any new assets either in the supplemental material or as a URL? [No]

［#89］
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]

［#90］
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

［#91］
5. If you used crowdsourcing or conducted research with human subjects...

［#92］
(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]

［#93］
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]

［#94］
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# A More Information of Pruning and Debiasing Methods

## A.1 Pruning Methods

### A.1.1 Iterative Magnitude Pruning

［#95］
Algo. 1 summarizes our implementation of IMP and IMP with weight rewinding. In practice, we set the per time pruning ratio $\Delta s = 10\%$ and the pruning interval $\Delta t = 0.1 \cdot t_{\text{max}}$.

### A.1.2 Mask Training

［#96］
As we described in Section 3.2.2 of the main paper, we realize mask training via binarization in forward pass and gradient estimation in backward pass. Following [42, 32], we adopt a magnitude-based strategy to initialize the real-valued masks. Specially, we consider two variants: The first one (hard variant) identifies the weights in matrix $\mathbf{W}$ with the smallest magnitudes, and sets the corresponding elements in $\hat{\mathbf{m}}$ to zero, and the remaining elements to a fixed value:

［#96］
$$
\hat{\mathbf{m}}_{i,j} =
\begin{cases}
0 & \text{if } \mathbf{W}_{i,j} \in \text{Min}_s(\text{abs}(\mathbf{W})) \\
\alpha \times \phi & \text{otherwise}
\end{cases}
\tag{5}
$$

［#96］
where $\text{Min}_s(\text{abs}(\mathbf{W}))$ extracts the weights with the lowest absolute value, according to sparsity level $s$. $\alpha \geq 1$ is a hyper-parameter. The second one (soft variant) directly utilizes the absolute values of the weights for mask initialization:

［#96］
$$
\hat{\mathbf{m}}_{i,j} = \text{abs}(\mathbf{W}_{i,j})
\tag{6}
$$

［#97］
To control the sparsity of the model, the threshold $\phi$ is adjusted dynamically at a frequency of $\Delta t_{\phi}$ training steps. In practice, we control the sparsity in a local way, i.e., all the weight matrices $\mathbf{W} \in \boldsymbol{\theta}_{pr}$ should satisfy the same sparsity constraint $s$. Algo. 2 summarizes the entire process of mask training.

［#98］
```
Algorithm 1: Iterative Magnitude Pruning (+ weight rewinding)
Input: PLM $f(\boldsymbol{\theta}_0)$ w. $\boldsymbol{\theta}_0 = \boldsymbol{\theta}_{ft}$, maximum training steps $t_{\text{max}}$, pruning interval $\Delta t$, per time pruning ratio $\Delta s$, target sparsity level $s = k \cdot \Delta s$ ($k \in \{1,2,\cdots\}$), pruning method $p \in \{\text{imp}, \text{imp-rw}\}$
Output: Pruned subentwork $f(\mathfrak{m} \odot \boldsymbol{\theta}_{ft}^\prime)$
1 Initialize the pruning mask $\mathfrak{m} = 1^{|\boldsymbol{\theta}_0|}$ and the number of pruning $n = 0$
2 while $t < t_{max}$ do
3    if $(t \bmod \Delta t) == 0$ then
4        # For imp, return the subnetwork after some further training
5        if $n \cdot \Delta s == s$ and $p$==imp then
6            return $f(\mathfrak{m} \odot \boldsymbol{\theta}_t)$
7        end
8        Prune $\Delta s \cdot |\boldsymbol{\theta}_0|$ from the remaining parameters $\mathfrak{m} \odot \boldsymbol{\theta}_t$ based on the magnitudes, and update $\mathfrak{m}$ accordingly
9        $n \leftarrow n + 1$
10       # For imp-rw, return the subnetwork directly after pruning
11       if $n \cdot \Delta s == s$ and $p$==imp-rw then
12           return $f(\mathfrak{m} \odot \boldsymbol{\theta}_0)$
13       end
14   end
15   Update the remaining model parameters $\mathfrak{m} \odot \boldsymbol{\theta}_t$ via AdamW [33];
16 end
```

### A.2 Debiasing Methods
［#99］
We have introduced the PoE method in Section 3.3. Here we provide descriptions of the other two debiasing methods, i.e., example reweighting and confidence regularization.

［#100］
Example Reweighting directly assigns an importance weight to the standard CE training loss, according to the bias degree $\beta$:
［#100］
$$
\mathcal{L}_{\text{reweight}} = -(1 - \beta) \mathbf{y} \cdot \log \mathbf{p}_m \tag{7}
$$

［#101］
Confidence Regularization is based on knowledge distillation [22]. It involves a teacher model trained with the standard CE loss. The teacher model's prediction $\mathbf{p}_t$ is used as a supervision signal to train the main model. To account for the bias degree of training examples, $\mathbf{p}_t$ is smoothed using a scaling function $\mathrm{S}(\mathbf{p}_t, \beta)$, and the final loss is computed as:
［#101］
$$
\begin{aligned}
\mathcal{L}_{\text{confreg}} &= -\mathrm{S}\left(\mathbf{p}_t, \beta\right) \cdot \log \mathbf{p}_m \\
\mathrm{S}\left(\mathbf{p}_t, \beta\right) &= \frac{\left(\mathbf{p}_t^j\right)^{(1-\beta)}}{\sum_{k=1}^{K}\left(\mathbf{p}_t^k\right)^{(1-\beta)}}
\end{aligned} \tag{8}
$$

## B More Experimental Setups
### B.1 Datasets and Evaluations
［#102］
We utilize eight datasets from three NLU tasks. The statistics of different dataset splits are summarized in Tab. 1. If one dataset has a test set, we use it for evaluation, and otherwise we report results on the dev set. For MNLI and QQP, since the official test server $^6$ only allows two submissions a day, we instead evaluate on the dev sets, following [4, 32, 45]. For FEVER, we use the training and evaluation data processed by [46] $^7$.

［#103］
Tab. 2 shows the distribution of examples over classes. We can see that the distributions of the QQP and $\text{PAWS}_{qqp}$ evaluation sets are imbalanced. Specially, in the OOD $\text{PAWS}_{qqp}$, where a biased model

［#103］
$^6$https://gluebenchmark.com/
［#103］
$^7$https://github.com/TalSchuster/FeverSymmetric
16

［#104］
```
Algorithm 2: Mask Training
Input: PLM $f(\boldsymbol{\theta}_0)$ w. $\boldsymbol{\theta}_0 \in \{\boldsymbol{\theta}_{pt}, \boldsymbol{\theta}_{ft}\}$, maximum training steps $t_{\text{max}}$, frequency $\Delta t_{\phi}$, target
        sparsity level $s$, threshold $\phi$, hyper-parameter $\alpha$, initialization method $init \in \{\text{hard}, \text{soft}\}$
Output: Pruned subentwork $f(\mathbf{m} \odot \boldsymbol{\theta}_0)$
1 if $init == \text{hard}$ then
2    Initialize the real-valued mask $\hat{\mathbf{m}}$ according to Eq. 5
3    Set threshold $\phi = 0.01$
4 else
5    Initialize the real-valued mask $\hat{\mathbf{m}}$ according to Eq. 6
6    Set threshold $\phi$ according to the sparsity constraint
7 end
8 while $t < t_{\text{max}}$ do
9    Get a mini-batch of $B$ examples $\{(\mathbf{x}_b, y_b)\}_{b=1}^B$
10   Forward pass through binarization:
11        $\mathcal{L}(f(\mathbf{x}_b, \mathbf{m} \odot \boldsymbol{\theta}_0), y_b)$, $\quad$ where $\mathbf{m}_{i,j} = \begin{cases} 1 & \text{if } \hat{\mathbf{m}}_{i,j} \geq \phi \\ 0 & \text{otherwise} \end{cases}$
12   Backward pass through gradient estimation:
13        $\hat{\mathbf{m}} \leftarrow \hat{\mathbf{m}} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{m}}$
14   if $(t \bmod \Delta t_{\phi}) == 0$ then
15       Update the threshold $\phi$ to satisfy the sparsity constraint
16   end
17 end
18 return $f(\mathbf{m} \odot \boldsymbol{\theta}_0)$
```

［#105］
Table 1: The number of examples in different dataset splits. The splits used for evaluation are highlighted with red color. The dev set for MNLI is MNLI-m.

［#106］
<table>
<thead>
  <tr>
    <th></th>
    <th colspan="2">NLI</th>
    <th colspan="3">Paraphrase Identification</th>
    <th colspan="3">Fact Verification</th>
  </tr>
  <tr>
    <th></th>
    <th>MNLI</th>
    <th>HANS</th>
    <th>QQP</th>
    <th>PAWS-qqp</th>
    <th>PAWS-wiki</th>
    <th>FEVER</th>
    <th>FEVER-Symm1</th>
    <th>FEVER-Symm2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Train</td>
    <td>392,702</td>
    <td>30,000</td>
    <td>363,849</td>
    <td>11,988</td>
    <td>49,401</td>
    <td>242,911</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Dev</td>
    <td>9,815</td>
    <td>30,000</td>
    <td>40,432</td>
    <td>677</td>
    <td>8,000</td>
    <td>16,664</td>
    <td>-</td>
    <td>708</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>8,000</td>
    <td>-</td>
    <td>717</td>
    <td>712</td>
  </tr>
</tbody>
</table>

［#107］
tends to predict most examples to the duplicate class, simply classifying all examples as non-duplicate can achieve substantial improvement in accuracy (from 28.2% to 71.8%). To account for this, we use the F1 score to evaluate the performance on the three paraphrase identification datasets. Specifically, we calculate the weighted average of the F1 score of each class. However, the class imbalance may still affect the evaluation on PAWS (as we discussed in Section 4.2.2) and therefore the OOD improvement should be assessed by also considering the ID performance.

### B.2 Software and Computational Resources

［#108］
We use two types of GPU, i.e., Nvidia V100 and TITAN RTX. All the experiments are run on a single GPU. Our codes are based on the Pytorch⁸ and the huggingface transformers library⁹ [58].

### B.3 Training Details

#### B.3.1 Bias Model

［#109］
As mentioned in Section 4.1.3, we train the bias model with spurious features. For MNLI and QQP, we adopt the hand-crafted word overlapping features proposed by [5], which includes:

［#110］
- Whether all the hypothesis words also belong to the premise.

［#111］
⁸https://pytorch.org/
⁹https://github.com/huggingface/transformers


［#112］
Table 2: Data distribution over classes. The meaning of the abbreviations are: ent (entailment), cont (contradiction), dulp (duplicate), supp (support), not-info (not-enough-info). "Eval" represents the dataset split used for evaluation, as described in Tab. 1

［#113］
<table>
  <tr>
    <td></td>
    <td></td>
    <td>MNLI</td>
    <td>HANS</td>
    <td></td>
    <td></td>
    <td>QQP</td>
    <td>PAWS<sub>qqp</sub></td>
    <td>PAWS<sub>wiki</sub></td>
    <td></td>
    <td></td>
    <td></td>
    <td>FEVER</td>
    <td>Symm1</td>
    <td>Symm2</td>
  </tr>
  <tr>
    <td rowspan="3">Train</td>
    <td>ent</td>
    <td>33.3%</td>
    <td>50%</td>
    <td rowspan="3">Train</td>
    <td>dulp</td>
    <td>36.9%</td>
    <td>31.5%</td>
    <td>44.2%</td>
    <td rowspan="3">Train</td>
    <td>supp</td>
    <td>41.4%</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cont</td>
    <td>33.3%</td>
    <td>50%</td>
    <td>non-dulp</td>
    <td>63.1%</td>
    <td>68.5%</td>
    <td>55.8%</td>
    <td>refute</td>
    <td>17.2%</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>neutral</td>
    <td>33.3%</td>
    <td>0%</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>not-info</td>
    <td>41.4%</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="3">Eval</td>
    <td>ent</td>
    <td>35.4%</td>
    <td>50%</td>
    <td rowspan="3">Eval</td>
    <td>dulp</td>
    <td>36.8%</td>
    <td>28.2%</td>
    <td>44.2%</td>
    <td rowspan="3">Eval</td>
    <td>supp</td>
    <td>47.9%</td>
    <td>52.9%</td>
    <td>50%</td>
  </tr>
  <tr>
    <td>cont</td>
    <td>32.7%</td>
    <td>50%</td>
    <td>non-dulp</td>
    <td>63.2%</td>
    <td>71.8%</td>
    <td>55.8%</td>
    <td>refute</td>
    <td>52.1%</td>
    <td>47.1%</td>
    <td>50%</td>
  </tr>
  <tr>
    <td>neutral</td>
    <td>31.8%</td>
    <td>0%</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>not-info</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
</table>

［#114］
Table 3: Basic training hyper-parameters.

［#115］
<table>
  <tr>
    <td></td>
    <td>#Epoch</td>
    <td>Learning Rate</td>
    <td>Batch Size</td>
    <td>Max Length</td>
    <td>Eval Interval</td>
    <td>Eval Metric</td>
    <td>Optimizer</td>
  </tr>
  <tr>
    <td>MNLI</td>
    <td>3 or 5</td>
    <td>5e-5</td>
    <td>32</td>
    <td>128</td>
    <td>1,000</td>
    <td>Acc</td>
    <td>AdamW</td>
  </tr>
  <tr>
    <td>QQP</td>
    <td>3</td>
    <td>2e-5</td>
    <td>32</td>
    <td>128</td>
    <td>1,000</td>
    <td>F1</td>
    <td>AdamW</td>
  </tr>
  <tr>
    <td>FEVER</td>
    <td>3</td>
    <td>2e-5</td>
    <td>32</td>
    <td>128</td>
    <td>500</td>
    <td>Acc</td>
    <td>AdamW</td>
  </tr>
</table>

［#116］
- Whether the hypothesis appears as a continuous subsequence in the premise.
- The percentage of the hypothesis words $\mathbf{w}^h = \{\mathbf{w}_1^h, \mathbf{w}_2^h, \cdots, \mathbf{w}_{|\mathbf{w}^h|}^h\}$ that appear in the premise $\mathbf{w}^p = \{\mathbf{w}_1^p, \mathbf{w}_2^p, \cdots, \mathbf{w}_{|\mathbf{w}^p|}^p\}$. Formally $\frac{|\mathbf{w}^h \cap \mathbf{w}^p|}{|\mathbf{w}^h|}$.
- The average of the maximum similarity between each hypothesis word and all the premise words: $\frac{1}{|\mathbf{w}^h|} \text{sum}(\{\text{max}(\{\text{sim}(\mathbf{w}_i^p, \mathbf{w}_j^h)|\forall \mathbf{w}_j^p \in \mathbf{w}^p\})|\forall \mathbf{w}_i^h \in \mathbf{w}^h\})$, where the similarity is computed based on the fastText word vectors [39] and the cosine distance.
- The minimum of the same similarities above: $\text{min}(\{\text{max}(\{\text{sim}(\mathbf{w}_i^p, \mathbf{w}_j^h)|\forall \mathbf{w}_j^p \in \mathbf{w}^p\})|\forall \mathbf{w}_i^h \in \mathbf{w}^h\})$.

［#117］
For FEVER, we use the max-pooled word embeddings of the claim sentence, which are also based on the fastText word vectors.

### B.3.2 Full BERT
［#118］
The main training hyper-parameters are shown in Tab. 3, which basically follow [55]. Most of the hyper-parameters are the same for different training strategies, except for the number of training epochs (#Epoch) on MNLI. For the standard CE loss and example reweighting, the model is trained for 3 epochs. For PoE and confidence regularization, the model is trained for 5 epochs.

### B.3.3 Mask Training and IMP
［#119］
Mask training and IMP basically use the same set of hyper-parameters as full BERT, except for longer training. The number of training epochs for mask training and IMP is 5 on MNLI, and 7 on QQP and FEVER. The hyper-parameters specific to mask training or IMP are summarized in Tab. 4. Unless otherwise specified, we adopt the hard-variant of mask initialization (Eq. 5) and fix the subnetwork sparsity to target sparsity $s$ throughout the process of mask training. Some special experimental setups are described as follows:

［#120］
Subnetworks from Fine-tuned BERT When we search for subnetworks at low sparsity (e.g., 20%) from a fine-tuned BERT, we find that mask training (with debiasing loss) stably improves the OOD performance, while the ID performance peaks at an early point of training and then slightly drops and recovers later. Therefore, the ID performance favors the early checkpoints, which are not good at the OOD generalization. To address this problem, we select the best checkpoint after $0.7 \cdot t_{\text{max}}$ of training, but still according to the performance on the ID dev set. This strategy is only adopted for mask training on fine-tuned BERT (for all sparsity levels), and in other cases we select the best checkpoint across training based on ID performance.


［#121］
Table 4: Basic hyper-parameters related to pruning methods. $t_{\text{max}}$ is the number of optimization steps by training #Epoch epochs.

［#122］
<table>
  <thead>
    <tr>
      <th colspan="5">Mask Training</th>
      <th colspan="2">IMP</th>
    </tr>
    <tr>
      <th>Mask Init</th>
      <th>Sparsity Schedule</th>
      <th>$\phi$</th>
      <th>$\alpha$</th>
      <th>$\Delta t_{\phi}$</th>
      <th>$\Delta s$</th>
      <th>$\Delta t$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>magnitude (hard)</td>
      <td>fixed to $s$</td>
      <td>0.01</td>
      <td>2</td>
      <td>equal to Eval Interval</td>
      <td>10%</td>
      <td>$0.1 \cdot t_{\text{max}}$</td>
    </tr>
  </tbody>
</table>

［#123］
![](./images/867764828259746483_9.jpg)

［#124］
Figure 9: Results of subnetworks pruned from the CE fine-tuned BERT, with different debiasing methods in pruning.

［#125］
BERT Subnetworks Fine-tuned in Isolation When fine-tuning the searched subnetworks (with their weights rewound to pre-trained values) in isolation, we use the same set of hyper-parameters as full BERT fine-tuning.

［#126］
Sparse and Unbiased BERT Subnetworks The OOD data is used in this setup. Specifically, we utilize the training data of HANS and PAWS for NLI and paraphrase identification respectively. In terms of the FEVER-Symmetric dataset, which does not provide a training set (see Tab. 1), we use the dev set of FEVER-Symm2 and copy the data 10 times to construct the OOD training data. The OOD and ID training data are then combined to form the final training set. Note that the evaluation sets are the same as the other setups, and NO test data is used in mask training.

［#127］
Gradual Sparsity Increase We mainly experiment with the gradual sparsity increase schedule for subnetworks at 90% sparsity. Concretely, we increase the sparsity from 70% to 90% during the process of mask training. The real-valued mask is initialized using the soft-variant (Eq. 6). This is because we find that the hard-variant is difficult to optimize with sparsity increase.

## C More Results and Analysis

### C.1 More Debiasing Methods

［#128］
In Section 4, we mainly experiment with the PoE debiasing method. Here, we combine mask training with the other two debiasing methods, namely example reweighting and confidence regularization, and search for SRNets from the CE fine-tuned BERT. Fig. 9 presents the results. As we can see: (1) Pruning with different debiasing methods almost consistently improves the OOD performance over the CE fine-tuned BERT. (2) The confidence regularization method (the grey lines) only achieves mild OOD improvement over the full BERT, while it preserves more ID performance compared with the other two methods. This phenomenon is in accordance with the results from [54], which propose the confidence regularization method to achieve a better trade-off between the ID and OOD performance.

［#129］
![](./images/867764828259746483_10.jpg)

［#130］
Figure 10: Results of subnetworks found using the OOD information.

［#131］
![](./images/867764828259746483_11.jpg)

［#132］
Figure 11: Mask training curves starting from full BERT checkpoints fine-tuned for varied steps. The sparsity levels are 70%, 70% and 90% for MNLI, QQP and FEVER respectively. At these sparsity levels, the gap between "ft step=0" and "ft to end" is the largest, according to Fig. 5.

### C.2 Sparse and Unbiased Subnetworks

［#133］
Fig. 10 shows the results of mask training with the OOD training data. We can see that the general patterns in paraphrase identification and fact verification datasets are basically the same as the NLI datasets. Although the identified subnetworks cannot achieve 100% accuracy on PAWS and FEVER-Symmetric as on HANS, they substantially narrow the gap between OOD and ID performance, as compared with the full BERT. An exception is on the Symm2, where the upper bound of SRNets seems not very high. This is probably because we do not have enough examples (708 in total) to represent the data distribution of the FEVER-Symmetric dataset. Therefore, we conjecture that the existence of sparse and unbiased subnetworks might be ubiquitous.

### C.3 The Timing to Start Searching SRNets

［#134］
Fig. 11 shows the mask training curves on all the 8 datasets. Similar to the NLI datasets, mask training on the other two tasks can achieve comparable results as "ft to end" by starting from an intermediate checkpoint of BERT fine-tuning. For QQP, we can start from 15,000 steps of full BERT fine-tuning (44% of $t_{max}$). For FEVER, we can start from 10,000 steps (44% of $t_{max}$).


［#135］
Table 5: Ablation studies of the gradual sparsity increase schedule. The number of training epochs are 3, 5 and 5 for MNLI, QQP and FEVER respectively. The subnetworks are at 90% sparsity. The numbers in the subscripts are standard deviations.

［#136］
<table>
<tbody>
<tr>
<td rowspan="3">fixed</td>
<td colspan="2"></td>
<td>MNLI</td>
<td>HANS</td>
<td rowspan="3">fixed</td>
<td colspan="2"></td>
<td>QQP</td>
<td>PAWS<sub>qqp</sub></td>
<td>PAWS<sub>qap</sub></td>
<td rowspan="3">fixed</td>
<td colspan="2"></td>
<td>FEVER</td>
<td>Symm1</td>
<td>Symm2</td>
</tr>
<tr>
<td colspan="2">hard</td>
<td>72.09<sub>0.92</sub></td>
<td>52.56<sub>0.92</sub></td>
<td colspan="2">hard</td>
<td>71.64<sub>1.85</sub></td>
<td>55.70<sub>1.92</sub></td>
<td>49.59<sub>1.84</sub></td>
<td colspan="2">hard</td>
<td>49.56<sub>5.09</sub></td>
<td>27.45<sub>2.94</sub></td>
<td>29.75<sub>4.40</sub></td>
</tr>
<tr>
<td colspan="2">soft</td>
<td>72.63<sub>0.31</sub></td>
<td>52.82<sub>0.47</sub></td>
<td colspan="2">soft</td>
<td>77.08<sub>0.66</sub></td>
<td>46.48<sub>3.55</sub></td>
<td>49.38<sub>0.98</sub></td>
<td colspan="2">soft</td>
<td>72.80<sub>0.95</sub></td>
<td>46.67<sub>0.73</sub></td>
<td>52.33<sub>0.75</sub></td>
</tr>
<tr>
<td rowspan="3">gradual</td>
<td colspan="2">0.2~0.9</td>
<td>73.61<sub>0.28</sub></td>
<td>53.90<sub>0.87</sub></td>
<td rowspan="3">gradual</td>
<td colspan="2">0.2~0.9</td>
<td>75.79<sub>0.39</sub></td>
<td>51.57<sub>0.69</sub></td>
<td>47.94<sub>0.98</sub></td>
<td rowspan="3">gradual</td>
<td colspan="2">0.2~0.9</td>
<td>73.53<sub>1.36</sub></td>
<td>46.47<sub>1.66</sub></td>
<td>52.42<sub>1.39</sub></td>
</tr>
<tr>
<td colspan="2">0.5~0.9</td>
<td>75.06<sub>0.31</sub></td>
<td>54.99<sub>1.28</sub></td>
<td colspan="2">0.5~0.9</td>
<td>77.54<sub>0.47</sub></td>
<td>50.92<sub>0.97</sub></td>
<td>48.86<sub>0.89</sub></td>
<td colspan="2">0.5~0.9</td>
<td>77.01<sub>0.43</sub></td>
<td>49.87<sub>0.95</sub></td>
<td>56.57<sub>0.22</sub></td>
</tr>
<tr>
<td colspan="2">0.7~0.9</td>
<td>76.84<sub>0.46</sub></td>
<td>56.72<sub>0.75</sub></td>
<td colspan="2">0.7~0.9</td>
<td>79.49<sub>0.58</sub></td>
<td>46.59<sub>1.81</sub></td>
<td>51.15<sub>0.73</sub></td>
<td colspan="2">0.7~0.9</td>
<td>79.01<sub>0.68</sub></td>
<td>51.74<sub>0.71</sub></td>
<td>58.17<sub>0.33</sub></td>
</tr>
</tbody>
</table>

［#137］
Table 6: Results of RoBERTa-base and BERT-large on the NLI task. We conduct mask training with PoE loss on the standard fine-tuned PLMs. "0.5~0.7" denotes gradual sparsity increase. The numbers in the subscripts are standard deviations.

［#138］
<table>
<tbody>
<tr>
<td colspan="2">RoBERTa-base</td>
<td>MNLI</td>
<td>HANS</td>
<td colspan="2">BERT-large</td>
<td>MNLI</td>
<td>HANS</td>
</tr>
<tr>
<td rowspan="2">full model</td>
<td>std</td>
<td>87.14<sub>0.21</sub></td>
<td>68.33<sub>0.88</sub></td>
<td rowspan="2">full model</td>
<td>std</td>
<td>86.84<sub>0.13</sub></td>
<td>69.44<sub>2.39</sub></td>
</tr>
<tr>
<td>poe</td>
<td>86.56<sub>0.18</sub></td>
<td>76.15<sub>1.35</sub></td>
<td>poe</td>
<td>86.25<sub>0.17</sub></td>
<td>76.27<sub>1.55</sub></td>
</tr>
<tr>
<td rowspan="3">mask train</td>
<td>0.5</td>
<td>85.40<sub>0.14</sub></td>
<td>75.17<sub>0.55</sub></td>
<td rowspan="3">mask train</td>
<td>0.5</td>
<td>85.47<sub>0.28</sub></td>
<td>75.40<sub>0.64</sub></td>
</tr>
<tr>
<td>0.7</td>
<td>83.48<sub>0.29</sub></td>
<td>68.63<sub>1.33</sub></td>
<td>0.7</td>
<td>77.54<sub>6.10</sub></td>
<td>60.19<sub>7.56</sub></td>
</tr>
<tr>
<td>0.5~0.7</td>
<td>84.41<sub>0.15</sub></td>
<td>71.95<sub>1.23</sub></td>
<td>0.5~0.7</td>
<td>84.83<sub>0.26</sub></td>
<td>70.18<sub>2.24</sub></td>
</tr>
</tbody>
</table>

### C.4 Ablation Studies on Gradual Sparsity Increase

［#139］
As we mentioned in Appendix B.3.3, we increase the sparsity from 70% to 90% and adopt the soft variant of mask initialization. To explain the reason for using this specific strategy, we present the ablation study results in Tab. 5. We can observe that: (1) Replacing the hard variant of mask initialization with the soft variant is beneficial, which leads to obvious improvements on the QQP, FEVER, Symm1 and Symm2 datasets. (2) Gradually increasing the sparsity further promotes the performance, with the 0.7~0.9 strategy achieving the best results on 7 out of the 8 datasets.

### C.5 Results on RoBERTa-base and BERT-large

［#140］
It has been shown by [21, 53] that pre-trained model RoBERTa [29] have better OOD generalization than BERT. [53] also shows that larger PLMs, which are more computationally expensive, are more robust. To examine whether our conclusions can generalize to RoBERTa and larger versions of BERT, we conduct mask training on the standard fine-tuned RoBERTa-base and BERT-large models and use the PoE debiasing loss in the mask training process.

［#141］
The results are shown in Tab. 6. We can see that, for RoBERTa-base: (1) At 50% sparsity, the searched subnetworks outperform the full RoBERTa (std) by 6.84 points on HANS, with a relative small drop of 1.74 on MNLI, validating that SRNets can be found in RoBERTa. (2) At 70% sparsity, the vanilla mask training produces subnetworks with undesirable ID performance and OOD performance comparable to full model (std). In comparison, when we gradually increase the sparsity level from 50% to 70%, the ID and OOD performance are improved simultaneously, demonstrating that gradual sparsity increase is also effective for RoBERTa.

［#142］
When it comes to BERT-large, the conclusions are basically the same as BERT-base and RoBERTa-base: (1) We can find 50% sparse SRNets from BERT-large using the original mask training. (2) Gradual sparsity increase is also effective for BERT-large. Additionally, we find that the original mask training exhibits high variance at 70% sparsity because the training fails for some random seeds. In comparison, with gradual sparsity increase, the searched subnetworks have better performance and low variance.

## D Related Work on Model Compression and Robustness

［#143］
Some prior attempts have also been made to obtain compact and robust deep neural networks. We discuss the relationship and difference between these works and our paper from three perspectives:


［#144］
Robustness Types There are various types of model robustness, including generalization to in-distribution unseen examples, robustness towards dataset bias [2, 37, 65, 46] and adversarial attacks [14], etc. Among the researches on model compression and robustness, adversarial robustness [16, 60, 48, 10, 59] and dataset bias robustness [62, 8] are the most widely studied. In this paper, we focus on the dataset bias problem, which is more common than the worst-case adversarial attack, in terms of real-world application.

［#145］
Compression Methods A major direction in robust model compression is about the design of compression methods. [47] investigate the effect of magnitude-based pruning on adversarially trained models. [16, 60] treat sparsity and adversarial robustness as a constrained optimization problem, and solve it using the alternating direction method of multipliers (ADMM) framework [63]. [48, 62, 34] combine learnable weight mask (i.e., mask training) and robust training objectives. Our study investigates the use of magnitude-based pruning and mask training, which are also widely employed in the literature of BERT compression.

［#146］
Application Fields Despite the topic of model compression and robustness has been proposed for years, it is mostly studied in the context of computer vision (CV) tasks and models, and few attention has been paid to the NLP field. Considering the real-world application potential of PLMs, it is critical to study the questions of PLM compression and robustness jointly. To this end, some recent studies extend the evaluation of compressed PLMs to consider adversarial robustness [59] and dataset bias robustness [8].

［#147］
Although our work shares the same topic with [8], we differ in several aspects. First, the scope and focus of our research questions are different. They aim at analyzing the impact of different compression methods (pruning and knowledge distillation [22]) on the OOD robustness of standard fine-tuned BERT. By contrast, we focus on subnetworks obtained from different pruning and fine-tuning paradigms and consider both standard fine-tuning and debiasing fine-tuning. Second, our conclusions are different. The results of [8] suggest that pruning generally has a negative impact on the robustness of BERT. In comparison, we revel the consistent existence of sparse BERT subnetworks that are more robust to dataset bias than the full model.

## E More Discussions

### E.1 How to Predict the Timing to Start Searching SRNets?

［#148］
A feasible way of solution is to stop full BERT fine-tuning when there is no significant improvement across several consecutive evaluation steps. The patience of early-stopping can be determined based on the computational budget. If our resource is limited, we can at least directly training the mask on $\theta_{pt}$, which can still produce SRNets at 50% sparsity (as shown by Section 4.4.2).

### E.2 How to Generalize to Other Scenarios?

［#149］
In this work, we focus on NLU tasks and PLMs from the BERT family. However, the methodology we utilize is agnostic to the type of bias, task and backbone model. Theoretically, it can be flexibly adapted to other scenarios by simply change the spurious features to train the bias model (for the three debiasing methods considered in this paper) or combine the pruning method with another kind of debiasing method that also involves model training. In the future work, we would like to extend our exploration to other types of PLMs (e.g., language generation models like GPT [41] and T5 [43]) and other types of NLP tasks (e.g., dialogue generation).