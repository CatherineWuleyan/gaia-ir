# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#1］
Ning Liu $^{1\ast}$ Geng Yuan $^{2\ast}$ Zhengping Che $^{3}$ Xuan Shen $^{2}$ Xiaolong Ma $^{2}$ Qing Jin $^{2}$ Jian Ren $^{4}$ Jian Tang $^{1\dagger}$
Sijia Liu $^{5}$ Yanzhi Wang $^{2\dagger}$

## Abstract
［#2］
In deep model compression, the recent finding "Lottery Ticket Hypothesis" (LTH) (Frankle & Carbin, 2018) pointed out that there could exist a winning ticket (i.e., a properly pruned subnetwork together with original weight initialization) that can achieve competitive performance than the original dense network. However, it is not easy to observe such winning property in many scenarios, where for example, a relatively large learning rate is used even if it benefits training the original dense model. In this work, we investigate the underlying condition and rationale behind the winning property, and find that the underlying reason is largely attributed to the correlation between initialized weights and final-trained weights when the learning rate is not sufficiently large. Thus, the existence of winning property is correlated with an insufficient DNN pretraining, and is unlikely to occur for a well-trained DNN. To overcome this limitation, we propose the "pruning & fine-tuning" method that consistently outperforms lottery ticket sparse training under the same pruning algorithm and the same total training epochs. Extensive experiments over multiple deep models (VGG, ResNet, MobileNet-v2) on different datasets have been conducted to justify our proposals.

## 1. Introduction
［#3］
Weight pruning has been widely studied and utilized to effectively remove the redundant weights in the over-parameterized deep neural networks (DNNs) while maintaining the accuracy performance (Han et al., 2015a; 2016; Wen et al., 2016; He et al., 2017; Min et al., 2018; He et al., 2019; Dai et al., 2019; Lin et al., 2020; He et al., 2020). The typical pruning pipeline has three main stages: 1) train an over-parameterized DNN, 2) prune the unimportant weights in the original DNN, and 3) fine-tune the pruned DNN to restore accuracy.

［#4］
Many works have been proposed to investigate the behaviors on weight pruning (Tanaka et al., 2020; Ye et al., 2020; Renda et al., 2020; Malach et al., 2020). The Lottery Ticket Hypothesis (LTH) (Frankle & Carbin, 2018) reveals that, inside a dense network with randomly initialized weights, a small sparse subnetwork, when trained in isolation using the identical initial weights, can reach a similar accuracy as the dense network. Such a sparse subnetwork with the initial weights is called the winning ticket.

［#5］
For a more rigorous definition, let $f(x;\theta_0)$ be a given network initialization, where $\theta_0 \sim D_\theta$ denotes the initial weights. We then formalize pre-training, pruning and sparse training below. Pre-training: The network is trained for $T$ epochs arriving at weights $\theta_T$ and network function $f(x;\theta_T)$. Pruning: Based on the pretrained weights $\theta_T$, adopt certain pruning algorithm to generate a sparse mask $m \in \{0,1\}^{|\theta|}$. Sparse Training: The LTH paper considers two cases of sparse training. The first ("winning ticket") is the direct application of mask $m$ to initial weights $\theta_0$, resulting in weights $\theta_0 \odot m$ and network function $f(x;\theta_0 \odot m)$. The second (random reinitialization) is the application of mask $m$ to a random initialization of weights $\theta_0' \sim D_\theta$, resulting in weights $\theta_0'\odot m$ and network function $f(x;\theta_0'\odot m)$. The winning property has two aspects ①-② for identification: ① Training $f(x;\theta_0 \odot m)$ for $T$ epochs (or fewer) will result in similar accuracy as that of the dense pre-trained network $f(x;\theta_T)$. ② There should be a notable accuracy gap between training $f(x;\theta_0\odot m)$ for $T$ epochs and training $f(x;\theta_0' \odot m)$, and the former shall be higher.

［#6］
In the standard LTH setup (Frankle & Carbin, 2018), the winning property can be observed in the case of low learning rate via the simple iterative magnitude pruning algorithm, but fails to occur at higher initial learning rates especially in deeper neural networks. For instance, the LTH work identifies the winning tickets on the CIFAR-10 dataset for the CONV-2/4/6 architectures (the down-scaled variants of VGG (Simonyan & Zisserman, 2014)), with the initial learning rate as low as 0.0001. For deeper networks such as

［#6］
$^{\ast}$Equal contribution.
［#7］
$^{1}$Midea Group, Beijing, China
［#7］
$^{2}$Northeastern University, Boston, MA, USA $^{3}$Didi Chuxing, Beijing, China $^{4}$Snap Inc., CA, USA $^{5}$Michigan State University, MI, USA. $^{\dagger}$Corresponding author.

［#8］
Proceedings of the $38^{th}$ International Conference on Machine Learning, PMLR 139, 2021. Copyright 2021 by the author(s).

［#9］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#10］
ResNet-20 and VGG-19 on CIFAR-10, the winning tickets can be identified only in the case of low learning rate. At higher learning rates, additional warm up is needed to find the winning tickets. In Liu et al. (2018) (the latest ArXiv version), it revisits LTH and finds out that with a widely-adopted learning rate, the winning ticket has no accuracy advantage over the random reinitialization. This questions the second aspect of winning property on the accuracy gap between training $f(x; \theta_0 \odot m)$ and training $f(x; \theta_0' \odot m)$.

［#11］
Further, the following work Frankle et al. (2019) proposes the iterative pruning with rewinding to stabilize identifying the winning tickets. Specifically, it resets the weights to $\theta_k$ in each pruning iteration, where $\theta_k$ denotes the weights trained from $\theta_0$ for a small number of $k$ epochs.

［#12］
In this paper, we investigate the underlying condition and rationale behind winning property. We ask if such a property is a natural characteristic of DNNs across their architectures and/or applications. We revisit LTH via extensive experiments built upon various representative DNN models and datasets, and have confirmed that the winning property only exists at a low learning rate. In fact, such a "low learning rate" (e.g., 0.01 for ResNet-20 and 0.0001 for CONV-2/4/6 architectures on CIFAR-10) is already significantly deviated from the standard learning rate, and results in notable accuracy degradation in the pretrained DNN. Besides, training from the "winning ticket" at such a low learning rate can only restore the accuracy of the pretrained DNN under the same insufficient learning rate, instead of that under the desirable learning rate. By introducing a correlation indicator for quantitative analysis, we found that the underlying reason is largely attributed to the correlation between initialized weights and final-trained weights when the learning rate is not sufficiently large. We draw the following conclusions:

［#13］
- As a result of low learning rate, such weight correlation results in low accuracy in DNN pretraining.
- Such weight correlation is also a key condition of winning property, concluded through a detailed analysis of the cause of winning property.
- Thus, the existence of winning property is correlated with an insufficient DNN pretraining, i.e., it is unlikely to occur for a well-trained DNN.

［#14］
Different from sparse training under lottery ticket setting, we propose the "pruning & fine-tuning" method, i.e., apply mask $m$ to pre-trained weights $\theta_T$ and perform fine-tuning for $T$ epochs. The generated sparse subnetwork can largely achieve the accuracy of the pretrained dense DNN. Through comprehensive experiments and analysis we draw the following conclusions:

［#15］
- "Pruning & fine-tuning" consistently outperforms lottery ticket setting under the same pruning algorithm for mask generation, and the same total training epochs.
- The pruning algorithm responsible for mask generation plays an important role in the quality of generated sparse subnetwork.
- Thus, if one wants to optimize the accuracy of sparse subnetwork and restore the accuracy of the pretrained dense DNN, we suggest adopting the pruning & fine-tuning method instead of lottery ticket setting.

## 2. Related Work

### 2.1. DNN Weight Pruning

［#16］
DNN weight pruning as a model compression technique can effectively remove the redundant weights in DNN models and hence reduce both storage and computation costs. The general flow of weight pruning consists of three steps: (1) train the neural network first; (2) derive a sub-network structure (i.e., removing unimportant weights) using a certain pruning algorithm; and (3) fine-tune the remaining weights in the sub-network to restore accuracy. Different pruning algorithms will deliver different capabilities to search for the best-suited sparse sub-network and lead to different final accuracies.

［#17］
The most straightforward method is the magnitude-based one-shot pruning, which will directly zero-out a given percentage of trained weights with the smallest magnitude. However, this method usually leads to a severe accuracy drop under a relatively high pruning rate. Iterative magnitude pruning is proposed in (Han et al., 2015b), which removes the weights with the smallest magnitude in an iterative manner. It repeats step (1) and step (2) multiple times until reaching the target pruning rate. In (Frankle & Carbin, 2018), iterative pruning is adopted to find the sparse subnetwork (i.e., winning ticket). The iterative pruning process is still a greedy search, and has been extended in (Zhu & Gupta, 2017; Tan & Motani, 2020; Liu et al., 2020) to derive better sub-network structures.

［#18］
To overcome the greedy nature in the heuristic pruning methods, the more mathematics-oriented regularization-based algorithm (Wen et al., 2016; He et al., 2017) has been proposed, to generate sparsity by incorporating $\ell_1$ or $\ell_2$ structured regularization in the loss function. However, this method directly applies fixed regularization terms that penalize all weights equally and will lead to a potential accuracy drop. Later work (Zhang et al., 2018; Ren et al., 2019) incorporate Alternating Direction Methods of Multipliers (ADMM) (Boyd et al., 2011; Ouyang et al., 2013) to solve the pruning problem as an optimization problem, which adopts dynamic regularization penalties and maintains high accuracy.

# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

## 2.2. Lottery Ticket Hypothesis

### 2.2.1. THE ORIGIN AND CONTROVERSY OF LOTTERY TICKET HYPOTHESIS

［#19］
The recent work (Frankle & Carbin, 2018) reveals that, inside a dense network with randomly initialized weights, a small sparse subnetwork can reach a similar test accuracy when trained in isolation using the identical initial weights as training the dense network. Such sparse subnetwork is called the winning ticket and can be found by pruning the pre-trained dense network under a non-trivial pruning ratio.

［#20］
As demonstrated in (Frankle & Carbin, 2018), winning tickets can be found in small networks and small dataset when using relatively low learning rates (e.g., 0.01 for SGD). The work from the same period (Liu et al., 2018) finds that, when using a relatively large learning rate (e.g., 0.1 for SGD), training a "winning ticket" with identical initialized weights will not provide any unique advantage in accuracy compared to training with randomly initialized weights. The following work (Frankle et al., 2019; Renda et al., 2020) also confirms that, for deeper networks and using relatively large learning rates, the winning property can hardly be observed. They propose a weight rewinding technique to identify small subnetworks, which can be trained in isolation to achieve competitive accuracy as the dense pretrained network.

### 2.2.2. OTHER ASPECTS AND APPLICATIONS

［#21］
Later work (Chen et al., 2020) further extends the lottery ticket hypothesis to a pre-trained BERT model to evaluate the transferability of the sparse subnetworks among different downstream NLP tasks. Recent works (Morcos et al., 2019; Chen et al., 2020) have studied the lottery ticket hypothesis in computer vision tasks and in unsupervised learning.

［#22］
The potential of sparse training suggested by the lottery ticket hypothesis has motivated the study of deriving the "winning tickets" at an early stage of training, thereby accelerating training process. There is a number of work in this direction (Frankle et al., 2020a; You et al., 2019; Frankle et al., 2020b), which are orthogonal to the discussions in this paper.

## 3. Notations in this Paper

［#23］
In this paper, we follow the notations from (Frankle & Carbin, 2018) and generalize to the "pruning & fine-tuning" setup. Detailed notations (as shown in Figure 1) are illustrated as follows:

［#24］
- **Initialization**: Given a network $f(x;\theta_0)$, where $\theta_0 \sim D_\theta$ denotes the initial weights.
- **Pre-training**: Train the network for $T$ epochs arriving at weights $\theta_T$ and network function $f(x;\theta_T)$.
- **Pruning**: Based on the trained weights $\theta_T$, adopt certain algorithm to generate a pruning mask $m \in \{0,1\}^{|\theta|}$. The LTH paper (Frankle & Carbin, 2018) uses the iterative pruning algorithm. We start from this algorithm for a fair evaluation, but are not restricted to it. Other algorithms, e.g., one-shot pruning and ADMM-based pruning are also employed to evaluate the impact on sparse training and pruning & fine-tuning, as shown in Section 5.
- **Sparse Training (Lottery Ticket Setting)**: The LTH paper considers two cases in the sparse training setup. The first is the direct application of mask $m$ to initial weights $\theta_0$, resulting in weights $\theta_0 \odot m$ and network function $f(x;\theta_0 \odot m)$. The LTH paper termed this case the "winning tickets"$^1$. The second is the application of mask $m$ to a random initialization of weights $\theta_0' \sim D_\theta$, resulting in weights $\theta_0' \odot m$ (network function $f(x;\theta_0' \odot m)$). This case is termed "random reinitialization" in the LTH paper. The weights after training $f(x;\theta_0 \odot m)$ for $T$ epochs are denoted by $(\theta_0 \odot m)_T$, while the weights after training $f(x;\theta_0' \odot m)$ for $T$ epochs are denoted by $(\theta_0' \odot m)_T$. Please note that the mask $m$ is kept through this training process.
- **Pruning & fine-tuning**: After generating the mask $m$, we directly apply it to the trained weights $\theta_T$, resulting in weights $\theta_T \odot m$, and perform fine-tuning (retraining) for another $T'$ epochs. The final weights are denoted by $(\theta_T \odot m)_{T'}$. To maintain the same number of total epochs as the lottery ticket setting, we set $T' = T$. Please note that the mask $m$ is kept through this fine-tuning process.

［#25］
The winning property has twofold meaning: First, training $f(x;\theta_0 \odot m)$ for $T$ epochs (or fewer) will result in similar accuracy as $f(x;\theta_T)$ (pre-training result of the dense network), under a non-trivial pruning rate. Second, there should be a notable accuracy gap between training $f(x;\theta_0 \odot m)$ for $T$ epochs and training $f(x;\theta_0' \odot m)$, and the former shall be higher.

## 4. Why Lottery Ticket Exists? An Analysis from the Weight Correlation Perspective

### 4.1. Revisiting Lottery Ticket: When does this winning property exist?

［#26］
We revisit the lottery ticket experiments on various DNN architectures including VGG-11, ResNet-20, and MobileNet-V2 on the CIFAR-10 and CIFAR-100 datasets. Our goal is

---
［#24］
$^1$We inherit this terminology, although it does not result in the winning property in many of our testing results.

# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#27］
![](./images/867769419944689875_1.jpg)

［#28］
Figure 1. Illustration of notations: "pre-training", "pruning" (mask generation), "sparse training", and "pruning & fine-tuning".

［#29］
to investigate the precise condition when winning property exists. We explore two different initial learning rates. The pruning approach for deriving masks follows the iterative pruning in Frankle & Carbin (2018). Namely, iteratively remove a percentage of the weights with the least magni- tudes in each layer. In each iterative pruning round, reset the weights to the initial weight $\theta_0$. We use the uniform per-layer pruning rate. Note the first convolutional layer is not pruned for all DNNs in this work.

［#30］
![](./images/867769419944689875_2.jpg)

［#31］
(a) Iterative pruning at learning rate of 0.01.
(b) Iterative pruning at learning rate of 0.1.

［#32］
Figure 2. Illustration of random reinitialization and "winning tick- ets" for ResNet-20 on CIFAR-10 at learning rates 0.01 and 0.1.

［#33］
Figure 2 illustrates the experiments of accuracy comparison between random reinitialization and "winning ticket" (both sparse training) for ResNet-20 on CIFAR-10 at learning rates 0.01 and 0.1 over a range of different sparsity ratios (Frankle & Carbin (2018) uses the low learning rate 0.01). We conduct each experiment five times (result variation shown in the figure). We set the same training epochs 150 rounds for training the original DNN with initial weights $f(x;\theta_0)$ (i.e., pretraining), training from randomly reinitial- ized weights with the mask $f(x;\theta'_0 \odot m)$ (random reinitial- ization), and training from the initial weights with the mask $f(x;\theta_0 \odot m)$ ("winning ticket"). The hyperparameters used are the same as (Frankle & Carbin, 2018): SGD with mo- mentum (0.9), and the learning rates decrease by a factor of 10 after 80 and 120 epochs. The batch size is 128. No additional training tricks are utilized throughout the paper for fairness in comparison.

［#34］
In the case of the initial learning rate of 0.01, the pre-trained DNN's accuracy is 89.62%. The "winning tickets" consis- tently outperform the random reinitialization over different sparsity ratios. It achieves the highest accuracy 90.04% (higher than the pre-trained DNN) at sparsity ratio of 62%. This is similar to the observations found in (Frankle & Carbin, 2018) on the same network and dataset. On the other hand, in the case of the initial learning rate of 0.1, the pre-trained DNN's accuracy is 91.7%. In this case, the "winning ticket" has a similar accuracy performance as the random reinitialization, and cannot achieve the accuracy close to the pre-trained DNN with a reasonable sparsity ratio (say 50% or beyond). Thus no winning property is satisfied. Similar results can be found in the experiments of ResNet-20, VGG-11 and MobileNet-v2 on both CIFAR-10 and CIFAR-100, while the results of the rest of the experi- ments are detailed in Appendix A.

［#35］
From these experiments, the winning property exists at a low learning rate but does not exist at a relatively high learning rate, which is also observed in (Liu et al., 2018). However, we would like to point out that the relatively high learning rate 0.1 (which is in fact the standard learning rate on these datasets) results in a notably higher accuracy in the pre- trained DNN (91.7% vs. 89.6%) than the low learning rate². The associated sparse training results ("winning ticket", ran- dom reinitialization) in the lottery ticket setting are also higher with the learning rate 0.1. This point is largely miss- ing in the previous discussions. Now the key question is: Are the above two observations correlated? If the answer is yes, it means that the winning property is not universal to DNNs, nor is it a natural characteristic of DNN architecture or application. Rather, it indicates that the learning rate is not sufficiently large, and the original, pretrained DNN is not well-trained.

［#36］
Our hypothesis is that the above observations are correlated, and this is largely attributed to the correlation between ini- tialized weights and final-trained weights when the learning rate is not sufficiently large. Before validating our hypothe- sis, we will introduce a correlation indicator (CI) for quan- titative analysis.

## 4.2. Weight Correlation Indicator
［#37］
Consider a DNN with two collections of weights $\theta$ and $\theta'$. Note that this is a general definition that applies to both the original DNN and sparse DNN (when the mask is

［#38］
²As CIFAR-10 is a relatively small dataset, 2% accuracy is a notable accuracy difference that should not be ignored.

# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#39］
![](./images/867769419944689875_3.jpg)

［#40］
Figure 3. Scenarios for quantitative analysis of the weight corre-
lation with an example of sparsity ratio $=50\%$ and $p=0.5$.
This example is one DNN layer, while our actual computation is
on the whole DNN.

［#41］
applied and a portion of weights eliminated). We define the
correlation indicator to quantify the amount of overlapped
indices of large-magnitude weights between $\theta$ and $\theta'$. More
specifically, given a DNN with $L$ layers, where the $l$-th layer
has $N_l$ weights, the weight index set $T_p((\theta)^l)$ ($p \in [0,1]$) is
the top-$p \cdot 100\%$ largest-magnitude weights in the $l$-layer.
Similarly, we define $T_p((\theta')^l)$. Please note that for a sparse
DNN, the portion $p$ is defined with respect to the number
of remaining weights in the sparse (sub-)network$^3$. The
intersection of these two sets includes those weights that
are large (top-$p \cdot 100\%$) in magnitude in both $\theta$ and $\theta'$, and
［#41］
$\boldsymbol{\mathrm{card}}\left(T_p((\theta)^l) \cap T_p((\theta')^l)\right)$ denotes the number of such
weights in layer $l$. The correlation indicator (overlap ratio)
between $\theta$ and $\theta'$ is finally defined as:

［#41］
$$
R_p(\theta, \theta') = \frac{\sum_l \boldsymbol{\mathrm{card}}\left(T_p((\theta)^l) \cap T_p((\theta')^l)\right)}{p \cdot \sum_l N_l} \tag{1}
$$

［#42］
When $R_p(\theta, \theta') \approx p$, the top-$p \cdot 100\%$ largest-magnitude
weights in $\theta$ and $\theta'$ are largely independent. In this case
the correlation is relatively weak$^4$. On the other hand, if
there is a large deviation of $R_p(\theta, \theta')$ from $p$, then there is
a strong correlation. Especially when $R_p(\theta, \theta') > p$, the
weights that are large in magnitude in $\theta$ are likely to also
be large in $\theta'$, indicating a positive correlation. Otherwise,
when $R_p(\theta, \theta') < p$, it implies a negative correlation.

［#43］
As shown in Figure 3, the above correlation indicator will
be utilized to quantify the correlation between a dense DNN
and a dense DNN, i.e., $R_p(\theta_0, \theta_T)$ for DNN pre-training,
and between a sparse DNN and a sparse DNN, i.e., $R_p(\theta_0 \odot
［#43］
m, (\theta_T \odot m)_T)$ and $R_p(\theta_0' \odot m, (\theta_T \odot m)_T)$ for the cases
［#43］
of "winning ticket" and random reinitialization under lottery
ticket setting. Next, we will use the former to demonstrate
the effect of different learning rates in DNN pre-training
and the latter to demonstrate the rationale and condition of
winning property.

## 4.3. Weight Correlation in DNN Pre-Training

［#44］
Intuitively, the weight correlation means that if a weight is
large in magnitude at initialization, it is likely to be large
after training. The reason for such correlation is that the
learning rate is too low and weight updating is slow. Such
weight correlation is not desirable for DNN training and
typically results in lower accuracy, as weights in a well-
trained DNN should depend more on the location of those
weights instead of initialization (Liu et al., 2018). Thus
when such weight correlation is strong, the DNN accuracy
will be lower, i.e., not well-trained.

［#45］
To validate the above statement, we have performed ex-
periments to derive $R_p(\theta_0, \theta_T)$ on DNN pretraining with
different initial learning rates. Using ResNet-20 on CIFAR-
10 dataset as an illustrative example. Figure 4 illustrates the
correlation indicator between the initial weights $\theta_0$ and the
trained weights $\theta_T$ from DNN pretraining at learning rates
of 0.01 and 0.1, respectively. We use $T=150$ the same
as Section 4.1, also the same other hyperparameters and no
additional training tricks. We can observe that $R_p(\theta_0, \theta_T)$
at learning rate 0.01 has a notably higher correlation com-
pared to the case of learning rate 0.1. This observation
indicates that the large-magnitude weights of $\theta_0$ are not
fully updated at a low learning rate 0.01, indicating that the
pre-trained DNN is not well-trained. In the case of learning
rate 0.1, the weights are sufficiently updated thus largely in-
dependent from the initial weights ($R_p(\theta_0, \theta_T) \approx p$, where
［#45］
$p=10\%,20\%,30\%,40\%,50\%$), indicating a well-trained
DNN. Results on other DNN models and datasets are pro-
vided in Appendix B, and a similar conclusion can be drawn.

［#46］
![](./images/867769419944689875_4.jpg)

［#47］
Figure 4. The overlap ratios (when $p=10\%,20\%,30\%,40\%$ and
［#47］
$50\%$) between the initial weights $\theta_0$ and the pretrained weights $\theta_T$
at learning rate of 0.01 and 0.1.

［#48］
As shown in the result discussions, learning rates 0.1 and
0.01 (for ResNet-20) are not merely two candidate hyperpa-
rameter values. Rather, they result in a well-trained DNN

［#48］
$^3$In this way the formula can be unified for dense and sparse
DNNs.
［#49］
$^4$We cannot say that there is no correlation here because
［#49］
$R_p(\theta, \theta') \approx p$ is only a necessary condition.

［#50］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#51］
(so a desirable learning rate) and a not well-trained DNN (so a not-so-good learning rate), respectively. We shall not rely on the conclusion drawn from the latter that results in an insufficient DNN pre-training.

### 4.4. Cause and Condition of the Winning Property
［#52］
Weight Correlation under Lottery Ticket Setting: In this subsection, our goal is to understand the different accuracy from training $f(x; \theta_0 \odot m)$ ("winning ticket") and training $f(x; \theta_0' \odot m)$ (random reinitialization) when the learning rate is low, thereby revealing the cause and condition of winning property. We will achieve this goal by studying the weight correlation.

［#53］
Consider the "pruning & fine-tuning" case formally defined in Section 3, in which we apply mask $m$ on the trained weights $\theta_T$ from DNN pretraining, and then perform fine-tuning for another $T$ epochs. The final weights are denoted by $(\theta_T \odot m)_T$. Using ResNet-20 on CIFAR-10 as an illustrative example. Figure 5(a) and 5(b) show the accuracy of the "pruning & fine-tuning" result $f(x; (\theta_T \odot m)_T)$ at different sparsity ratios, with learning rates 0.01 and 0.1, respectively. Again we use $T=150$ epochs and the same other hyperparameters. The accuracies of the pretrained DNN with corresponding learning rates are also provided. One can observe that $f(x; (\theta_T \odot m)_T)$ achieves relatively high accuracy, close to or higher than the accuracy of the pretrained DNN at the same learning rate (even at the desirable learning rate 0.1$)^5$. Results on other DNN models and datasets are provided in Appendix C, and a similar conclusion can be drawn.

［#54］
We study the correlation between $\theta_0 \odot m$ ($\theta_0' \odot m$) and $(\theta_T \odot m)_T$ to shed some light on the cause of winning property. Again use ResNet-20 on CIFAR-10 as an illustrative example, while the results on other DNN models and datasets are provided in Appendix D with similar conclusion. Figure 5(c) shows the correlation indicator between $\theta_0 \odot m$ ("winning ticket") and $(\theta_T \odot m)_T$, and between $\theta_0' \odot m$ (random reinitialization) and $(\theta_T \odot m)_T$, under the insufficient learning rate 0.01. While Figure 5(d) shows the correlation indicator results under the desirable learning rate 0.1. One can observe the positive correlation between $\theta_0 \odot m$ and $(\theta_T \odot m)_T$ at the low learning rate, when the winning property exists. Such correlation is minor in the other cases.

［#55］
Analysis of Weight Correlation and Condition of Winning Property: Let us investigate the cause of correlation between $\theta_0 \odot m$ and $(\theta_T \odot m)_T$ at low learning rate. As shown in Section 4.3, there is a correlation between $\theta_0$ and $\theta_T$ at the insufficient learning rate. Then there is also a correlation between $\theta_0 \odot m$ and $\theta_T \odot m$ (both applied the same mask). As $\theta_T \odot m$ includes the pretrained weights and $(\theta_T \odot m)_T$ only applies additional fine-tuning, there will be positive correlation between $\theta_T \odot m$ and $(\theta_T \odot m)_T$. Combining the above two statements will yield the correlation between $\theta_0 \odot m$ and $(\theta_T \odot m)_T$. When we consider random reinitialization, there is no correlation between $\theta_0'$ and $\theta_T$ as a reinitialization is applied. So there is no correlation between $\theta_0' \odot m$ and $\theta_T \odot m$, or between $\theta_0' \odot m$ and $(\theta_T \odot m)_T$.

［#56］
At a desirable learning rate 0.1, there is a minor (or no) correlation between $\theta_0$ and $\theta_T$ as shown in Section 4.3. As a result, there is minor (or no) correlation between $\theta_0 \odot m$ and $\theta_T \odot m$, or between $\theta_0 \odot m$ and $(\theta_T \odot m)_T$. From the above analysis, one can observe that the correlation between $\theta_0$ and $\theta_T$ is the key condition in weight correlation analysis.

［#57］
The positive correlation between $\theta_0 \odot m$ and $(\theta_T \odot m)_T$ helps to explain the winning property at low learning rate. Compared with random reinitialization $\theta_0' \odot m$, the "winning ticket" $\theta_0 \odot m$ is "closer" to a reasonably accurate solution. As the weight upscaling is slow (learning rate insufficient),

［#58］
![](./images/867769419944689875_5.jpg)
［#59］
(a) Pruning&fine-tuning at learning rate of 0.01.

［#60］
![](./images/867769419944689875_6.jpg)
［#61］
(b) Pruning&fine-tuning at learning rate of 0.1.

［#62］
![](./images/867769419944689875_7.jpg)
［#63］
(c) The overlap ratio comparison at learning rate 0.01.

［#64］
![](./images/867769419944689875_8.jpg)
［#65］
(d) The overlap ratio comparison at learning rate 0.1.

［#66］
Figure 5. (a), (b): Accuracies of $f(x; (\theta_T \odot m)_T)$ ("pruning & fine-tuning") at different sparsity ratios with masks generated by iterative pruning. (c), (d): The weight correlation (overlap ratio) comparison at $p=0.2$, between $\theta_0 \odot m$ ("winning ticket") and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights), and between $\theta_0'\odot m$ (random reinitialization) and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights) under 0.3, 0.5, 0.7 sparsity ratios.

［#67］
$^5$In fact, the relatively high accuracy of $f(x; (\theta_T \odot m)_T)$ is one major reason for us to explore the correlation between $\theta_0 \odot m$ ($\theta_0' \odot m$) and $(\theta_T \odot m)_T$. In Section 5 we will generalize to the conclusion that "pruning & fine-tuning" results in higher accuracy in general than sparse training (the lottery ticket setting).

［#68］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#69］
it takes less effort to reach a higher accuracy starting from $\theta_0 \odot m$ compared with starting from $\theta_0' \odot m$. Besides, as pointed out in Section 4.3, the insufficient learning rate (and correlation between $\theta_0$ and $\theta_T$) results in a low accuracy in the pre-trained DNN, which makes it easier for sparse training to reach its accuracy. On the other hand, at a sufficient learning rate, such correlations do not exist (or are very minor), and then the winning property does not exist.

［#70］
Remarks: From the above analysis, we conclude that a key condition of winning property is the correlation between $\theta_0$ and $\theta_T$. However, as already demonstrated in Section 4.3, under the same condition the pretrained network will not be well-trained, as weights in a well-trained DNN should depend more on the location of those weights instead of initialization. In fact, as shown in Figure 2 and Appendix A, the "winning ticket" can only restore the accuracy of the pretrained DNN under the same insufficient learning rate, instead of reaching the pretrained DNN accuracy at a desirable learning rate. This makes the value of the winning property questionable.

### 4.5. Takeaway
［#71］
As discussed above, the existence of winning property is correlated with an insufficient DNN pretraining. It seems that winning property is not a natural characteristic of DNN architecture or application, and is unlikely to occur for a well-trained DNN (with a desirable learning rate). As a result, we do not suggest investigating the winning property under an insufficient learning rate.

## 5. Pruning & Fine-tuning – A Better Way to Restore Accuracy under Sparsity
［#72］
As concluded from the above discussions, it is difficult for sparse training to restore the accuracy of the pre-trained dense DNN, when a desirable learning rate is applied. On the other hand, as already hinted in Section 4.4, the "pruning \& fine-tuning" (i.e., fine-tuning from $(\theta_T \odot m)_T$) exhibits a higher capability in achieving the accuracy of a pre-trained DNN, no matter what the learning rate is. Compared with the lottery ticket setting, the only difference in "pruning & fine-tuning" is that the mask $m$ is applied to the pretrained weights $\theta_T$. Is this the key reason for the high accuracy? Is this a universal property for different DNN architectures and applications? If the answer is yes, what is the underlying reason? We aim to answer these questions.

［#73］
In this section, we only consider the **desirable learning rate** (e.g., 0.1 for ResNet-20 on CIFAR-10 dataset) and sufficient DNN pre-training, as the associated conclusions will be more meaningful.

［#74］
Fair Comparison with Lottery Ticket Setting: We claim that under the same $T$ epochs for fine-tuning and sparse training, it is a fair comparison. The generation of mask $m$ is the same. The only difference is that pruning & fine- tuning applies mask $m$ to the pre-trained weights $\theta_T$, while sparse training applies $m$ to $\theta_0$. Note that $\theta_T$ is available before $m$ as the latter is derived based on $\theta_T$ using pruning algorithm. Thus pruning & fine-tuning will have **no additional training epochs** compared with sparse training.

［#75］
Comparison between Pruning & Fine-tuning and Sparse Training: We use ResNet-20 on CIFAR-10 dataset as an illustrative example, and the rest of results are provided in Appendix E (with the similar conclusion). We use the desirable learning rate 0.1, $T=150$ epochs, and the same as Section 4.1 for the rest of hyperparameters. Figure 6(a) shows the accuracy comparison between pruning & fine-tuning (i.e., training (fine-tuning) from $\theta_T \odot m$) and the two sparse training scenarios "winning ticket" (i.e., training from $\theta_0 \odot m$) and random reinitialization (i.e., training from $\theta_0' \odot m$) at different sparsity ratios. Iterative pruning algorithm is used to derive mask $m$ here. One can clearly observe the accuracy gap between pruning & fine-tuning and the two sparse training cases (lottery ticket setting). In fact, the pruning & fine-tuning scheme can consistently outperform the pretrained dense DNN up to sparsity ratio 70%. Again, there is no accuracy difference between the two sparse training cases.

［#76］
Furthermore, we consider other two candidate pruning algorithms, ADMM-based pruning (Zhang et al., 2018) and one-shot pruning, for pruning mask generation. Figure 6(b) and Figure 6(c) demonstrate the corresponding accuracy comparison results between pruning & fine-tuning and the two sparsity training scenarios. Again one can observe the notable advantage of pruning & fine-tuning over the lottery ticket setting, even with a weak one-shot pruning algorithm for mask generation. In fact, pruning & fine-tuning under ADMM-based pruning can restore the accuracy of pretrained DNN with 80% sparsity. The property is not found under any of these pruning algorithms. Clearly, the consistent advantage of pruning & fine-tuning is attributed to the fact that mask $m$ is applied to pretrained weights $\theta_T$ instead of the initialized weights $\theta_0$. In fact, information in $\theta_T$ is important for the sparse subnetwork to maintain accuracy of the pretrained dense network. Or in other words, weights in the desirable sparse subnetwork should have correlation with $\theta_T$ instead of $\theta_0$.

［#77］
Effect of Different Pruning Algorithms – Towards a Better Mask Generation: We have tested three pruning algorithms for mask generation. How to evaluate their relative performance? Figure 7 combines the above results and demonstrates the accuracy performances of pruning & fine- tuning and sparse training ("winning ticket" case), under all three pruning algorithms. The rest of results are in Appendix E. One can observe the order in the accuracy perfor-

［#78］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#79］
![](./images/867769419944689875_9.jpg)
![](./images/867769419944689875_10.jpg)
![](./images/867769419944689875_11.jpg)

［#80］
(a) The iterative pruning at learning rate of 0.1.
(b) The ADMM-based pruning at learning rate of 0.1.
(c) The least magnitude one-shot pruning at learning rate of 0.1.

［#81］
Figure 6. Accuracy performance of pruning & fine-tuning vs. two sparse training cases ("winning ticket" and random reinitialization). Three pruning algorithms are utilized for mask generation: iterative pruning (a), ADMM-based pruning (Zhang et al., 2018) (b), and one-shot pruning (c).

［#82］
![](./images/867769419944689875_12.jpg)

［#83］
Figure 7. Accuracy performances of pruning & fine-tuning and sparse training ("winning ticket" case), under all three pruning algorithms (iterative pruning, ADMM-based pruning, and one-shot pruning) for mask generation.

［#84］
mance: ADMM-based pruning on top, iterative pruning in the middle, and one-shot pruning the lowest. This order is the same for pruning & fine-tuning and sparse training. Note that the pruning algorithm is utilized to generate mask $m$, while the other conditions are the same (i.e., $\theta_T$, fine-tuning $T$ epochs on $\theta_T \odot m$, or sparse training on $\theta_0 \odot m$). Hence, the relative performance is attributed to the quality in mask generation. One can conclude that the selection of pruning algorithm is critical in generating the sparse subnetwork as the quality of mask generation plays a key role here.

［#85］
An Analysis from Weight Correlation Perspective: We conduct a weight correlation analysis of pruning & fine-tuning results that can largely restore the accuracy of pre-trained, dense DNN, between the final weights $(\theta_T \odot m)_T$ and the initialization $\theta_0$. Detailed results and discussions are provided in Appendix F. The major conclusion is that there is a lack of correlation between $(\theta_T \odot m)_T$ and $\theta_0$, but there is a correlation between $(\theta_T \odot m)_T$ and $\theta_T$. It further strengthens the conclusion that it is not desirable to have the weight correlation between final-trained weights and weight initialization.

［#86］
Comparison with Frankle et al. (2019): The work Frankle et al. (2019) suggests applying mask $m$ to $\theta_k$ and then apply sparse training, where $\theta_k$ denotes the weights trained from $\theta_0$ for a small number of $k$ epochs. This technique is training from $\theta_k \odot m$, and is in between sparse training (training from $\theta_0 \odot m$) and pruning & fine-tuning (training from $\theta_T \odot m$). We point out that these three cases require the same number of total epochs under the same pruning algorithm, as mask $m$ is generated later than $\theta_k$ or $\theta_T$. We conduct a comprehensive comparison on the relative performance, with detailed results and discussions in Appendix G. The major conclusion is that pruning & fine-tuning consistently outperforms the method Frankle et al. (2019) over different sparsity ratios, DNN models, and datasets. As they exhibit the same number of training epochs, we suggest directly applying the mask $m$ to $\theta_T$ and perform fine-tuning, instead of applying to $\theta_k$.

［#87］
Remarks: If one wants to optimize the accuracy of sparse subnetwork and restore the accuracy of the pretrained dense DNN, we suggest adopting the pruning & fine-tuning method instead of lottery ticket setting.

### 6. Conclusion
［#88］
In this work, we investigate the underlying condition and rationale behind lottery ticket property. We introduce a correlation indicator for quantitative analysis. Extensive experiments over multiple deep models on different datasets have been conducted to justify that the existence of winning property is correlated with an insufficient DNN pretraining, and is unlikely to occur for a well-trained DNN. Meanwhile, the sparse training of lottery ticket setting is difficult to restore the accuracy of the pre-trained dense DNN. To overcome this limitation, we propose the "pruning & fine-tuning" method that consistently outperforms lottery ticket sparse training under the same pruning algorithm and total training epochs over various DNNs on different datasets.

# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

## Acknowledgements
［#89］
This research was supported by the National Science Foundation under awards CNS-1704662 and CCF-1937500. Any opinions, findings, and conclusions or recommendations in this material are those of the authors and do not necessarily reflect the views of NSF.

## References































# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

## Appendix

［#90］
We explore a variety of deep neural networks (DNNs) to support our results. All experiments on CIFAR-10 and CIFAR-100 use the basic data augmentation including random cropping and random horizontal flipping. No other tricks are used. We use 150 training epochs ($T=150$) and decay the initial learning rate after 80, 120 epochs by a factor of 10 (with step learning rate scheduler). The optimizer is stochastic gradient descent (SGD) and the momentum is 0.9. This training setup is used for pre-training ($f(x; \theta_T)$), pruning & fine-tuning $((\theta_T \odot m)_T)$, and sparse training $((\theta_0 \odot m)_T)$ as introduced in Section 3.

## A. Revisit Lottery Tickets

［#91］
We show the experiment results of MobileNet-V2 on CIFAR-10, and ResNet-20, VGG-11, and MobileNet-V2 on CIFAR-100 over a range of different sparsity ratios with the masks generated from iterative pruning (Frankle & Carbin, 2018) at learning rate 0.01 and 0.1, respectively. We conduct each experiment *five* times (result variation shown in figures). We set the same training epochs (i.e., 150 epochs) for training the original DNNs with initial weights $f(x; \theta_0)$ (i.e., pretraining), training from randomly reinitialized weights with the mask $f(x; \theta'_0 \odot m)$ (random reinitialization), and training from the initial weights with the mask $f(x; \theta_0 \odot m)$ ("winning ticket").

［#92］
**CIFAR-10 Results:** Figure 1(a) and 1(b) illustrate the result on MobileNet-V2 using CIFAR-10. The pre-trained MobileNet-V2's accuracy on CIFAR-10 is 92.20% at initial learning rate 0.01, and 93.86% at initial learning rate 0.1.

［#93］
**CIFAR-100 Results:** Figure 1(c) and 1(d) show the result on MobileNet-V2 using CIFAR-100. The pre-trained MobileNet-V2's accuracy on CIFAR-100 is 73.10% at initial learning rate 0.01, and 74.76% at initial learning rate 0.1. Figure 1(e) and 1(f) show the result on ResNet-20 for CIFAR-100. The pre-trained ResNet-20's accuracy on CIFAR-100 is 63.10% at initial learning rate 0.01, and 67.15% at initial learning rate 0.1 (see the significant gap here). Figure 1(g) and 1(h) show the result on VGG-11 for CIFAR-100. The pre-trained VGG-11's accuracy on CIFAR-100 is 67.74% at initial learning rate 0.01, and 69.83% at initial learning rate 0.1. In the case of MobileNet-V2 on CIFAR-100 at low learning rate, we observe that the "winning ticket" can outperform the random reinitialization but failed to restore the baseline accuracy (73.10%). This indicates the low learning rate is not desirable. For all illustrated cases, the "winning ticket"'s accuracy is close to the random reinitialization at the initial learning rate 0.1. While in the case of learning rate 0.01, the "winning ticket" can outperform the random reinitialization over different sparsity ratios. Note there is a clearly accuracy gap between the pretrained DNNs with the initial learning rate 0.1 and with the initial learning rate 0.01.

［#94］
From these experiments, the winning property exists at a low learning rate but does not exist at a relatively high learning rate. However, we would like to point out that the relatively high learning rate of 0.1 (which is, in fact, the standard learning rate on these datasets) results in *notably higher accuracy* in the pretrained DNNs than the low learning rate (MobiletNet-V2 on CIFAR-10 93.86% vs. 92.20%, MobiletNet-V2 on CIFAR-100 74.76% vs. 73.10%, VGG-11 on CIFAR-100 69.83% vs. 67.74%, ResNet-20 on CIFAR-100 67.15% vs. 63.10%). We should not draw conclusion basd on the low (insufficient) learning rate in general.

## B. Weight Correlation in DNN Pre-Training

［#95］
We investigate the *correlation indicator* between the initial weights $\theta_0$ and the trained weights $\theta_T$ from DNN pretraining on VGG-11, ResNet-20, and MobileNet-V2 on CIFAR-10 and CIFAR-100 under learning rates of 0.01 and 0.1, respectively.

［#96］
We have performed experiments to derive $R_p(\theta_0, \theta_T)$ on different DNN pretraining with different initial learning rates. Figure 2 illustrates the correlation indicator between the initial weights $\theta_0$ and the trained weights $\theta_T$ from DNN pretraining at learning rates of 0.01 and 0.1 on VGG-11 and MobileNet-V2 using CIFAR-10/100, respectively. We use the same hyperparameters mentioned in the setup without additional training tricks. Figure 2(a) and 2(b) illustrate the result on VGG-11 for CIFAR-10/100. Figure 2(c) and 2(d) illustrate the result on MobileNet-V2 for CIFAR-10/100.

［#97］
We can observe that $R_p(\theta_0, \theta_T)$ at a learning rate of 0.01 has a notably higher correlation compared to the case of learning rate 0.1. This observation indicates that the large-magnitude weights of $\theta_0$ are not fully updated at a low learning rate of 0.01, indicating that the pre-trained DNN is not well-trained. In the case of learning rate 0.1, the weights are sufficiently updated thus largely independent from the initial weights ($R_p(\theta_0, \theta_T) \approx p$, where $p=10\%, 20\%, 30\%, 40\%, 50\%$), indicating a well-trained DNN.

## C. Pruning & Fine-tuning

［#98］
Consider the "pruning & fine-tuning" case formally defined in Section 3, in which we apply mask $m$ on the trained weights $\theta_T$ from DNN pretraining, and then perform fine-tuning for another $T$ epochs. The final weights are denoted by $(\theta_T \odot m)_T$. We study accuracy of the "pruning & fine-tuning" result $f(x; (\theta_T \odot m)_T)$ at different sparsity ratios, with learning rates of 0.01 and 0.1 on different DNNs using CIFAR-10 and CIFAR-100. We use the same hyperparame-

［#99］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#100］
![](./images/867769419944689875_13.jpg)
［#101］
(a) Iterative pruning at learning rate of 0.01 on MobileNet-V2 using CIFAR-10.

［#102］
![](./images/867769419944689875_14.jpg)
［#103］
(b) Iterative pruning at learning rate of 0.1 on Mobilenet-V2 using CIFAR-10.

［#104］
![](./images/867769419944689875_15.jpg)
［#105］
(c) Iterative pruning at learning rate of 0.01 on MobileNet-V2 using CIFAR-100.

［#106］
![](./images/867769419944689875_16.jpg)
［#107］
(d) Iterative pruning at learning rate of 0.1 on MobileNet-V2 using CIFAR-100.

［#108］
![](./images/867769419944689875_17.jpg)
［#109］
(e) Iterative pruning at learning rate of 0.01 on ResNet-20 using CIFAR-100.

［#110］
![](./images/867769419944689875_18.jpg)
［#111］
(f) Iterative pruning at learning rate of 0.1 on ResNet-20 using CIFAR-100.

［#112］
![](./images/867769419944689875_19.jpg)
［#113］
(g) Iterative pruning at learning rate of 0.01 on VGG-11 using CIFAR-100.

［#114］
![](./images/867769419944689875_20.jpg)
［#115］
(h) Iterative pruning at learning rate of 0.1 on VGG-11 using CIFAR-100.

［#116］
Figure 1. Accuracy illustration of random reinitialization and "winning tickets" for MobileNet-V2 on CIFAR-10, and MobileNet-V2, ResNet-20 and VGG-11 on CIFAR-100 at learning rates 0.01 and 0.1.

［#117］
ters as mentioned in the setup ($T=150$). The accuracies of the pretrained DNNs with corresponding learning rates are also provided. Figure 3(a) and 3(b) illustrate the "pruning & fine-tuning" result on MobileNet-V2 for CIFAR-10 using learning rates of 0.01 and 0.1, respectively. Figure 3(c) and 3(d) illustrate the "pruning & fine-tuning" result on MobileNet-V2 for CIFAR-100 with learning rates of 0.01 and 0.1, respectively. In the case of MobileNet-V2 for CIFAR-100 with the initial learning rate 0.1, the "pruning & fine-tuning" scheme consistently perform better than the pretrained dense DNN (74.76%).

［#118］
We can observe that $f\left(x ;\left(\theta_{T} \odot m\right)_{T}\right)$ achieves relatively high accuracy, close to or higher than the accuracy of the pretrained DNN at the same learning rate (even at the desirable learning rate 0.1).

### D. Sparse Correlation
［#119］
We study the correlation between $\theta_{0} \odot m\left(\theta_{0}^{\prime} \odot m\right)$ and $(\theta_{T} \odot m)_{T}$ to shed some light on the cause of winning property. We illustrate the correlation on ResNet-20, VGG-11 and MobileNet-V2 for CIFAR-10/100 at learning rate 0.01, 0.1, respectively. We show the correlation indicator between $\theta_{0} \odot m$ ("winning ticket") and $(\theta_{T} \odot m)_{T}$, and between $\theta_{0}^{\prime} \odot m$ (random reinitialization) and $(\theta_{T} \odot m)_{T}$ at learning rate 0.01, 0.1. Figure 4 illustrates the result of ResNet-20 for CIFAR-100 at the learning rate 0.01 and 0.1. Figure 5 shows the result of VGG-11 for CIFAR-10/100 and Figure 6 shows the result of MobileNet-V2 for CIFAR-10/100 at learning rates 0.01, 0.1, respectively. In the case of high learning rate 0.1, the weight correlation between $\theta_{0} \odot m$ ("winning ticket") and $(\theta_{T} \odot m)_{T}$ (pruned&fine-tuned weights), and between $\theta_{0}^{\prime} \odot m$ (random reinitialization) and $(\theta_{T} \odot m)_{T}$ (pruned&fine-tuned weights) are similar (and minor) under different sparsity ratios.

［#120］
From these results we can observe the positive correlation between $\theta_{0} \odot m$ and $(\theta_{T} \odot m)_{T}$ at the low learning rate, when the winning property exists. Such correlation is minor in the other cases.

### E. Different Pruning Algorithms
［#121］
We explore the different pruning algorithms on ResNet-20, MobileNet-V2 and VGG-11 using CIFAR-10/100. We use the desirable learning rate $0.1, T=150$ epochs, and the same hyperparameters introduced in Section 4.1. We compare accuracy between pruning & fine-tuning (i.e., training (fine-tuning) from $\theta_{T} \odot m$) and the two sparse training scenarios "winning ticket" (i.e., training from $\theta_{0} \odot m$) and random reinitialization (i.e., training from $\theta_{0}^{\prime} \odot m$) at different sparsity ratios. We investigate three pruning algorithms to derive mask $m$: Iterative pruning algorithm, ADMM-based pruning (Zhang et al., 2018) and one-shot pruning algorithm. We explore accuracy comparison results between pruning & fine-tuning and the two sparsity training scenarios. Figure 7 and 8 illustrate the accuracy comparison on MobileNet-V2 using CIFAR-10 and CIFAR-100, respectively. Figure 9 shows the result on ResNet-20 using CIFAR-100. Figure 10 illustrates the result on VGG-11 for CIFAR-100.

［#122］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#123］
![](./images/867769419944689875_21.jpg)
［#124］
(a) VGG-11 for CIFAR-10

［#125］
![](./images/867769419944689875_22.jpg)
［#126］
(b) VGG-11 for CIFAR-100

［#127］
![](./images/867769419944689875_23.jpg)
［#128］
(c) MobileNet-V2 for CIFAR-10

［#129］
![](./images/867769419944689875_24.jpg)
［#130］
(d) MobileNet-V2 for CIFAR-100

［#131］
Figure 2. The overlap ratios (when $p = 10\%$, $20\%$, $30\%$, $40\%$ and $50\%$) between the initial weights $\theta_0$ and the pretrained weights $\theta_T$ at learning rate of 0.01 and 0.1 on VGG-11 and MobileNet-V2 using CIFAR-10/100.

［#132］
![](./images/867769419944689875_25.jpg)
［#133］
(a) MobileNet-V2 for CIFAR-10 with learning rate 0.01.

［#134］
![](./images/867769419944689875_26.jpg)
［#135］
(b) MobileNet-V2 for CIFAR-10 with learning rate 0.1.

［#136］
![](./images/867769419944689875_27.jpg)
［#137］
(c) MobileNet-V2 for CIFAR-100 with learning rate 0.01.

［#138］
![](./images/867769419944689875_28.jpg)
［#139］
(d) MobileNet-V2 for CIFAR-100 with learning rate 0.1.

［#140］
Figure 3. Accuracy of $f\left(x;(\theta_T \odot m)_T\right)$ (“pruning & fine-tuning”) at different sparsity ratios with masks generated by iterative pruning on MobileNet-V2 using CIFAR-10/100.

［#141］
From these results, we can clearly observe the accuracy gap between pruning & fine-tuning and the two sparse training cases (lottery ticket setting). For MobiletNet-V2 on CIFAR-100, with the masks generated from iterative pruning and ADMM-based pruning, the pruning & fine-tuning scheme can consistently outperform the pretrained dense DNN up to sparsity ratio 85%. Similarly results can be observed on VGG-11 using CIFAR-100. Meanwhile, at sparsity ratio 0.39 (39%), the pruning & fine-tuning scheme with mask

［#142］
![](./images/867769419944689875_29.jpg)
［#143］
(a) ResNet-20 for CIFAR-100 with learning rate 0.01.

［#144］
![](./images/867769419944689875_30.jpg)
［#145］
(b) ResNet-20 for CIFAR-100 with learning rate 0.1.

［#146］
Figure 4. The weight correlation (overlap ratio) comparison at $p=0.2$, between $\theta_0 \odot m$ (“winning ticket”) and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights), and between $\theta_0' \odot m$ (random reinitialization) and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights) under 0.3, 0.5, 0.7 sparsity ratios on ResNet-20 using CIFAR-100.

# Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#147］
![](./images/867769419944689875_31.jpg)

［#148］
(a) VGG-11 for CIFAR-10 with learning rate 0.01.

［#149］
![](./images/867769419944689875_32.jpg)

［#150］
(b) VGG-11 for CIFAR-10 with learning rate 0.1.

［#151］
![](./images/867769419944689875_33.jpg)

［#152］
(c) VGG-11 for CIFAR-100 with learning rate 0.01.

［#153］
![](./images/867769419944689875_34.jpg)

［#154］
(d) VGG-11 for CIFAR-100 with learning rate 0.1.

［#155］
Figure 5. The weight correlation (overlap ratio) comparison at $p=0.2$, between $\theta_0 \odot m$ ("winning ticket") and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights), and between $\theta_0' \odot m$ (random reinitialization) and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights) under 0.3, 0.5, 0.7 sparsity ratios on VGG-11 using CIFAR-10/100.

［#156］
![](./images/867769419944689875_35.jpg)

［#157］
(a) MobileNet-V2 for CIFAR-10 with learning rate 0.01.

［#158］
![](./images/867769419944689875_36.jpg)

［#159］
(b) MobileNet-V2 for CIFAR-10 with learning rate 0.1.

［#160］
![](./images/867769419944689875_37.jpg)

［#161］
(c) MobileNet-V2 for CIFAR-100 with learning rate 0.01.

［#162］
![](./images/867769419944689875_38.jpg)

［#163］
(d) MobileNet-V2 for CIFAR-100 with learning rate 0.1.

［#164］
Figure 6. The weight correlation (overlap ratio) comparison at $p=0.2$, between $\theta_0 \odot m$ ("winning ticket") and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights), and between $\theta_0' \odot m$ (random reinitialization) and $(\theta_T \odot m)_T$ (pruned&fine-tuned weights) under 0.3, 0.5, 0.7 sparsity ratios on MobileNet-V2 using CIFAR-10/100.

［#165］
generated from ADMM-based pruning can achieve accuracy 76.04% while the pretrained DNN's accuracy is only 74.76% (under the desirable learning rate 0.1).

［#166］
We observe the notable advantage of pruning & fine-tuning over the lottery ticket setting, even with a weak one-shot pruning algorithm for mask generation. Note there is no accuracy difference between the two sparse training cases. Pruning & fine-tuning under ADMM-based pruning can restore the accuracy of pretrained DNN with the highest sparsity ratio compared to the other two pruning algorithms. Clearly, the consistent advantage of pruning & fine-tuning is attributed to the fact that mask $m$ is applied to pretrained weights $\theta_T$ instead of the initialized weights $\theta_0$. In fact, information in $\theta_T$ is important for the sparse subnetwork to maintain accuracy of the pretrained dense network. Or in other words, weights in the desirable sparse subnetwork should have correlation with $\theta_T$ instead of $\theta_0$.

［#167］
Further we evaluate the relative performance (accuracy) of these three pruning algorithms. We combine the above results and demonstrate the accuracy performances of pruning & fine-tuning and sparse training ("winning ticket" case), under all three pruning algorithms. Figure 11(a) and 11(b) show the overall accuracy performance comparison on MobileNet-V2 using CIFAR-10 and CIFAR-100, respectively. Figure 12(a) shows the result on ResNet-20 using CIFAR-100 and Figure 12(b) shows the result on VGG-11 using CIFAR-100.

［#168］
We observe the order in the accuracy performance: ADMM-based pruning on top, iterative pruning in the middle, and one-shot pruning the lowest. This order is the same for pruning & fine-tuning and sparse training. Note that the pruning algorithm is utilized to generate mask $m$, while the other conditions are the same (i.e., $\theta_T$, fine-tuning $T$ epochs on $\theta_T \odot m$, or sparse training on $\theta_0 \odot m$). Hence, the relative performance is attributed to the quality in mask generation. We can conclude that the selection of pruning algorithm is critical in generating the sparse subnetwork as the quality of mask generation plays a key role in the context of pruning scenario.

## F. An Analysis from Weight Correlation Perspective

［#169］
We provide the correlation between $(\theta_T \odot m)_T$ and $\theta_0$, and between $(\theta_T \odot m)_T$ and $\theta_T$ for VGG-11, ResNet-20 and MobileNet-V2 using CIFAR-10/100. The training epoch $T=150$ and the initial learning rate is 0.1. The masks are generated by ADMM-based pruning algorithm. Note that $\theta_0$ and $\theta_T$ are dense models, while $(\theta_T \odot m)_T$ is a sparse model. To utilize the *correlation indicator*, we extend the correlation scenario of dense DNNs vs. dense DNNs to sparse DNNs vs. dense DNNs by restricting $p$ less than

［#169］
$(1-\text{sparsity ratio})$ of sparse DNNs. In this experiment, we consider weight correlation at $p=0.2$ and the sparsity ratio is $0.50\ (50\%)$ for the DNNs. The results are illustrated in Table 1. The results indicate that there is a lack of correlation between $(\theta_T \odot m)_T$ and $\theta_0$, but there is a correlation between $(\theta_T \odot m)_T$ and $\theta_T$. It further strengthens the conclusion that it is not desirable to have the weight correlation between final-trained weights and weight initialization.

［#170］
Table 1. Weight correlation analysis at $p=0.2$ between $(\theta_T \odot m)_T$ and $\theta_0$, and between $(\theta_T \odot m)_T$ and $\theta_T$ for VGG-11, ResNet-20, MobileNet-V2 using CIFAR-10/100 at learning rate 0.1 under sparsity ratio 50% and the masks $m$ are generated by ADMM-based pruning algorithm.

［#171］
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>$R_p((\theta_T \odot m)_T, \theta_0)$</th>
      <th>$R_p((\theta_T \odot m)_T, \theta_T)$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet-20</td>
      <td>CIFAR-100</td>
      <td>20.36%</td>
      <td>63.97%</td>
    </tr>
    <tr>
      <td>MobileNet-V2</td>
      <td>CIFAR-100</td>
      <td>20.11%</td>
      <td>64.71%</td>
    </tr>
    <tr>
      <td>VGG-11</td>
      <td>CIFAR-100</td>
      <td>20.41%</td>
      <td>49.32%</td>
    </tr>
    <tr>
      <td>MobileNet-V2</td>
      <td>CIFAR-10</td>
      <td>20.26%</td>
      <td>49.36%</td>
    </tr>
    <tr>
      <td>VGG-11</td>
      <td>CIFAR-10</td>
      <td>20.21%</td>
      <td>48.08%</td>
    </tr>
  </tbody>
</table>

## G. Comparison with (Frankle et al., 2019)

［#172］
The work Frankle et al. (2019) suggests applying mask $m$ to $\theta_k$ and then apply sparse training, where $\theta_k$ denotes the weights trained from $\theta_0$ for a small number of $k$ epochs. This technique is training from $\theta_k \odot m$, and is in between sparse training (training from $\theta_0 \odot m$) and pruning & fine-tuning (training from $\theta_T \odot m$). We evaluate the relative sparse training performance among $(\theta_0 \odot m)_T$ ("winning ticket"), $(\theta_T \odot m)_T$ (pruned&fine-tuned) and $(\theta_k \odot m)_T$ ("rewind") under a desirable learning rate. We set $T=150$, $k=10$ and the initial learning rate is 0.1. The same hyperparameters are adopted as introduced in Section 4.1. We study the accuracy performance comparison on MobileNet-V2, ResNet-20 and VGG-11 on CIFAR-100. We use the masks generated from the ADMM-based pruning algorithm. Figure 13 illustrates the accuracy comparison results of MobileNet-V2, ResNet-20 and VGG-11 on CIFAR-100. We can observe the order in the accuracy performance: $(\theta_T \odot m)_T$ (pruned&fine-tuned) on top, $(\theta_k \odot m)_T$ ("rewind") in the middle, and $(\theta_0 \odot m)_T$ ("winning ticket") the lowest. As they exhibit the same number of training epochs (please note that $m$ is generated later than $\theta_k$ or $\theta_T$), we suggest directly applying the mask $m$ to $\theta_T$ and perform fine-tuning, instead of applying to $\theta_k$.

［#173］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#174］
![](./images/867769419944689875_39.jpg)

［#175］
Figure 7. Accuracy of pruning & fine-tuning vs. two sparse training cases ("winning ticket" and random reinitialization) on MobileNet-V2 using CIFAR-10.

［#176］
![](./images/867769419944689875_40.jpg)

［#177］
Figure 8. Accuracy of pruning & fine-tuning vs. two sparse training cases ("winning ticket" and random reinitialization) on MobileNet-V2 using CIFAR-100.

［#178］
![](./images/867769419944689875_41.jpg)

［#179］
Figure 9. Accuracy of pruning & fine-tuning vs. two sparse training cases ("winning ticket" and random reinitialization) on ResNet-20 using CIFAR-100.

［#180］
![](./images/867769419944689875_42.jpg)

［#181］
Figure 10. Accuracy of pruning & fine-tuning vs. two sparse training cases ("winning ticket" and random reinitialization) on VGG-11 using CIFAR-100.

［#182］
Lottery Ticket Preserves Weight Correlation: Is it Desirable or Not?

［#183］
![](./images/867769419944689875_43.jpg)

［#184］
(a) MobileNet-V2 on CIFAR-10 at learning rate 0.1.

［#185］
![](./images/867769419944689875_44.jpg)

［#186］
(b) MobileNet-V2 on CIFAR-100 at learning rate 0.1.

［#187］
Figure 11. Accuracy of pruning & fine-tuning and sparse training ("winning ticket" case), under all three pruning algorithms (iterative pruning, ADMM-based pruning, and one-shot pruning) for mask generation on MobileNet-V2 using CIFAR-10/100.

［#188］
![](./images/867769419944689875_45.jpg)

［#189］
(a) ResNet-20 on CIFAR-100 at learning rate 0.1.

［#190］
![](./images/867769419944689875_46.jpg)

［#191］
(b) VGG-11 on CIFAR-100 at learning rate 0.1.

［#192］
Figure 12. Accuracy of pruning & fine-tuning and sparse training ("winning ticket" case), under all three pruning algorithms (iterative pruning, ADMM-based pruning, and one-shot pruning) for mask generation on ResNet-20 and VGG-11 using CIFAR-100.

［#193］
![](./images/867769419944689875_47.jpg)

［#194］
(a) MobileNet-V2 for CIFAR-100 at learning rate of 0.1.

［#195］
![](./images/867769419944689875_48.jpg)

［#196］
(b) ResNet-20 for CIFAR-100 at learning rate of 0.1.

［#197］
![](./images/867769419944689875_49.jpg)

［#198］
(c) VGG-11 for CIFAR-100 at learning rate of 0.1.

［#199］
Figure 13. Accuracy performance of $(\theta_T \odot m)_T$ (pruned&fine-tuned), $(\theta_k \odot m)_T$ ("rewind") and $(\theta_0 \odot m)_T$ ("winning ticket") on MobileNet-V2, ResNet-20 and VGG-11 for CIFAR-100 over a range of different sparsity ratios. The masks are generated by ADMM-based pruning algorithm and the initial learning rate is 0.1.