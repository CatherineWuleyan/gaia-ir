# Efficient Lottery Ticket Finding: Less Data is More

［#1］
Zhenyu Zhang $^{*1}$ Xuxi Chen $^{*2}$ Tianlong Chen $^{*2}$ Zhangyang Wang $^{2}$

## Abstract
［#2］
The lottery ticket hypothesis (LTH) (Frankle & Carbin, 2018) reveals the existence of winning tickets (sparse but critical subnetworks) for dense networks, that can be trained in isolation from random initialization to match the latter's accuracies. However, finding winning tickets requires burdensome computations in the train-prune-retrain process, especially on large-scale datasets (e.g., ImageNet), restricting their practical benefits. This paper explores a new perspective on finding lottery tickets more efficiently, by doing so only with a specially selected subset of data, called Pruning-Aware Critical set (PrAC set), rather than using the full training set. The concept of PrAC set was inspired by the recent observation, that deep networks have samples that are either hard to memorize during training, or easy to forget during pruning. A PrAC set is thus hypothesized to capture those most challenging and informative examples for the dense model. We observe that a high-quality winning ticket can be found with training and pruning the dense network on the very compact PrAC set, which can substantially save training iterations for the ticket finding process. Extensive experiments validate our proposal across diverse datasets and network architectures. Specifically, on CIFAR-10, CIFAR-100, and Tiny ImageNet, we locate effective PrAC sets at $35.32\% \sim 78.19\%$ of their training set sizes. On top of them, we can obtain the same competitive winning tickets for the corresponding dense networks, yet saving up to $82.85\% \sim 92.77\%$, $63.54\% \sim 74.92\%$, and $76.14\% \sim 86.56\%$ training iterations, respectively. Crucially, we show that a PrAC set found is reusable across different network architectures, which can amortize the extra cost of finding PrAC sets, yielding a practical regime for efficient lottery ticket finding.

［#3］
![](./images/867766605889667803_1.jpg)

［#4］
Figure 1. Test accuracy of found subnetworks from ResNets at different sparsity levels on CIFAR-10 and CIFAR-100. Black dots (●) represent the performance of unpruned baselines; blue dots (●) indicate the performance of vanilla lottery tickets found with full data (Frankle & Carbin, 2018), and red star (★) are established by our PrAC lottery tickets. Red numbers are the number of samples in the PrAC set. The lottery tickets found on the PrAC sets could perform on par with the vanilla ones at moderate sparsity levels, and even outperform the latter at the highest sparsity of 96.48%.

## 1. Introduction
［#5］
Deep neural networks (DNNs) have revolutionized the performance bar of various tasks, yet suffer from substantial over-parameterization (Voulodimos et al., 2018). Parameter counts are frequently measured in billions rather than millions, with the time and financial outlay necessary to train these models growing in concert. Once trained, they can be pruned of excessive capacity (Han et al., 2015; Tang et al., 2020). However, conventional approaches first train dense DNNs, and then prune the trained them to high levels of sparsity. Those methods significantly reduce the inference complexity yet cost even greater computational resources and memory footprints at training.

［#6］
An emerging subfield has explored the prospect of directly training smaller, sparse subnetworks in place of the full models without sacrificing performance. The key idea is to reuse the sparsity pattern found through pruning and train a sparse network from scratch. The seminal work (Frankle & Carbin, 2018) hypothesized that standard DNNs contain sparse matching subnetworks, often called winning ticket, capable of training in isolation to full accuracy. In other words, we could have trained smaller networks from the start if only we had known which subnetworks to choose. In larger-scale real-world settings, current methods often

［#6］
$^{*}$Equal contribution $^{1}$University of Science and Technology of China $^{2}$University of Texas at Austin. Correspondence to: Zhangyang Wang <atlaswang@utexas.edu>.

［#7］
Proceedings of the $38^{th}$ International Conference on Machine Learning, PMLR 139, 2021. Copyright 2021 by the author(s).

# Efficient Lottery Ticket Finding: Less Data is More

［#8］
empirically choose winning tickets by *Iterative Magnitude Pruning* (IMP), sometimes at an early training point called "rewinding" (Frankle et al., 2019b; 2020a). Other works also showed sparsity might emerge at the initialization (Lee et al., 2018; Wang et al., 2020), or at the early training stage (You et al., 2020). However, it was observed in (Frankle et al., 2020b) that IMP still outperforms those carefully designed alternatives by clear margins, and remain as the most effective lottery ticket finding approach. However, the cumbersome train-prune-train cycle required by IMP makes it extremely expensive to find lottery tickets from large models and datasets, and also questioning the practical efficiency benefits of finding lottery tickets.

［#9］
In parallel to seeking *model sparsity* during training, another complementary and promising line of ideas exploits *data sparsity*, i.e, reducing training costs by the informed selection of training samples (Tsang et al., 2005; Har-Peled & Kushal, 2007). Such techniques often select a small but critical *core set* from a large dataset, by which way a significant fraction of examples can be omitted from training while still maintaining the trained models' generalization (Zhao & Zhang, 2015; Katharopoulos & Fleuret, 2018; Toneva et al., 2019; Mirzasoleiman et al., 2020). Also related to the core set approach is the dataset distillation (Wang et al., 2018) that aims to summarize training images into a handful of synthetic images, ensuring that DNNs trained on the latter generalize almost as well as trained on the former.

## 1.1. Research Questions & Our Contributions

［#10］
However, the questions below are not yet clear:

［#11］
($Q_1$) *How will the "model sparsity" (e.g. LTH) and "data sparsity" (e.g., core set) interplay? Can one help the other? Can they possibly be jointly utilized to push training efficiency to the next level?*

［#12］
To answer the above question ($Q_1$), we first formulate and address a prerequisite question ($Q_0$):

［#13］
($Q_0$) *What samples are considered as "core" for finding a lottery ticket (trainable sparse DNN)?*

［#14］
A typical "coreset" (Mirzasoleiman et al., 2020) aims to guarantee that models fitting the coreset also provide a good fit for the original data, and finding it is treated as an approximation problem such as sampling or clustering. To find a sparse subnetwork that can *match* the performance of the full model, the challenge level is escalated higher since sparse DNNs are way tougher to train (Evci et al., 2019), and the core samples have also to identify the trainable sparse connectivity patterns. In other words, the new core set needs to encode not only the full dataset's knowledge, but also the trainability.

［#15］
In this paper, we first attempt to address ($Q_0$) by investigating a new concept called **Pruning-Aware Critical set (PrAC set)**, that targets to characterize important samples for finding lottery tickets that are both same *generalizable* and *trainable*. Considering that the lottery ticket iterates between two steps: (re-)training, and pruning. Conceptually, we hope a PrAC set to capture two types of samples:

［#16］
- Samples that are *hard to memorize*, during (re-)training of the (original or pruned) DNN. Recent observations by (Toneva et al., 2019; Yao et al., 2020; Xia et al., 2021; Han et al., 2020) reveal that certain examples are memorized easily during training, but some others are repeatedly forgotten. Such (un)forgettable examples generalize across different architectures in the same dataset. The forgetting dynamics also suggest one can train a DNN on a dataset with a large fraction of the least forgotten examples removed.
- Samples that are *easy to forget*, during pruning the dense DNN into a sparse DNN. Pruning steps are essential to the final (trainable) sparsity, yet hampering both memorization and generalization. Moreover, it has been observed by (Hooker et al., 2020a) that pruning disproportionately impacts the model performance when run on a narrow subset of the dataset, e.g., the atypical, semantically ambiguous or underrepresented images.

［#17］
During lottery ticket finding, by calculating the forgotten dynamics for each sample within training and the prediction differences after each pruning, we can effectively collect those most informative samples and build a PrAC set. In fact, our approach is a *co-design between data and model sparsity*, which we feel essential due to the hardness of ($Q_0$).

［#18］
Equipped with PrAC sets, we then examine ($Q_1$) and present a comprehensive set of experiments, integrating the PrAC set with an efficient lottery ticket finding and training framework. In general, we find PrAC sets to help find comparable winning tickets with much higher training efficiency, compared to the vanilla IMP scheme using the full set, with little performance drop (sometimes even with performance gains)$^1$. We summarize our main findings as follows:

［#19］
- We identify winning tickets and PrAC sets broadly across different datasets (CIFAR-10, CIFAR-100, Tiny ImageNet) and architectures (ResNet-20, ResNet-56, and VGG-16). High-quality winning tickets can be found on the PrAC sets while saving training time and costs. Specifically, we save $82.85\% \sim 92.77\%$ on CIFAR-10, $63.54\% \sim 74.92\%$ on CIFAR-100, and $76.14\% \sim 86.56\%$ on Tiny ImageNet in training iterations, while maintaining or even boosting their achievable accuracies.
- PrAC sets show great transferability across architectures on the same dataset, which can amortize the cost of finding PrAC sets in practice. Taking ResNet-20 as the source architecture, the PrAC set found in CIFAR-10 and CIFAR-100 can locate winning tickets in ResNet-56 and VGG-19

---
［#18］
$^1$Our implementations are available at: https://github.com/VITA-Group/PrAC-LTH


［#20］
with almost no performance degradation. We further visualize the PrAC set samples, conclude their patterns, and compare them with multiple sample selection methods.

［#21］
- On CIFAR-10, the PrAC winning ticket (79.03%) are sparser than tickets from random pruning (48.80%). Our ticket finding also outperforms other efficient network pruning methods. For example, at 93.13% sparsity, our PrAC lottery tickets can outperform SynFlow (Tanaka et al., 2020) by 1.51%, SNIP (Lee et al., 2018) by 5.47%, and GraSP (Wang et al., 2020) by 18.73%.

## 2. Related Work

### Lottery Ticket Hypothesis (LTH).
［#22］
LTH (Frankle & Carbin, 2018) has drawn lots of attention. Later on, (Frankle et al., 2019a; Renda et al., 2020) scaled up LTH to larger models by early weight rewinding that relaxes the use of original random initialization. Another intriguing property of lottery tickets, the transferability, has also been thoroughly examined (Mehta, 2019; Morcos et al., 2019; Desai et al., 2019; Chen et al., 2020b;a). Zhou et al. (2019) investigated different components in LTH and observed supermasks in winning tickets. LTH has also been extended to various applications (Gale et al., 2019; Chen et al., 2020b; Yu et al., 2020; Chen et al., 2021c; Kalibhat et al., 2020; Chen et al., 2021a; Ma et al., 2021; Gan et al., 2021; Chen et al., 2021b) beyond image classification.

［#23］
Unstructured IMP (Han et al., 2015; Frankle & Carbin, 2018) serves as an effective method to find these winning tickets, and Dynamic Sparse Training (Mostafa & Wang, 2019; Mocanu et al., 2018; Evci et al., 2020) is also capable of identifying subnetworks with promising performance. However their computational expensiveness motivates many efficient alternatives that hope to locate sparse trainable subnetworks at random initialization or early training stage, with less or no training (Lee et al., 2018; You et al., 2020; Wang et al., 2020; Tanaka et al., 2020; Frankle et al., 2020b). Unfortunately, those sparse subnetworks found at beginning usually have clearly *inferior performance* to the IMP-found winning tickets, leaving IMP still the mainstream LTH scheme. This paper explores a complementary new perspective on finding lottery tickets more efficiently by co-designing a specially crafted subset. Our method secures winning tickets of *fully comparable performance* to the full IMP scheme, and it can also be straightforwardly combined with those efficient pruning methods if needed.

### Active Learning and Core-Set Approaches.
［#24］
Another closely related literature is the problem of active learning (Settles, 2009; 2012) and core-set selection (Tsang et al., 2005; Har-Peled & Kushal, 2007; Bachem et al., 2017; Sener & Savarese, 2017). Specifically, Zhao & Zhang (2015); Katharopoulos & Fleuret (2018); Toneva et al. (2019); Wang et al. (2018); Mirzasoleiman et al. (2020); Hooker et al. (2020a;b) select core-sets by the importance sampling. Zhao & Zhang (2015); Katharopoulos & Fleuret (2018) sort the samples according to the magnitude of its loss gradient with respect to parameters of the network. Toneva et al. (2019) samples the examples based on the forgetting dynamics during the course of learning. Mirzasoleiman et al. (2020) constructs core-set that provides an approximately low-rank Jacobian matrix. Wang et al. (2018) generates synthetic examples to distill the knowledge from the entire dataset, and Hooker et al. (2020a;b) find pruning can cause disproportionately high errors on a small subset. We draw inspirations from several of those ideas, and extend the idea of core-set to be co-optimized with LTH.

## 3. Methodology

［#25］
In this section, we present our framework to co-design model and data sparsity, which works in an iterative fashion of two alternative steps: i) constructing the Pruning-Aware Critical (PrAC) set with pruned models, which selects the most challenging and informative examples; ii) utilizing PrAC sets to identify critical subnetworks, (i.e., lottery tickets), which takes much less training iterations. In this way, the burdensome computations of the train-prune-retrain process in tickets finding, can be substantially reduced. The overall pipeline is summarized in Algorithm 1.

［#26］
```
Algorithm 1 Data and Model Sparsity Co-Design
Input: Full training data $\mathcal{D}_0$, a threshold for the number of
    forgets $\mathcal{E}_\text{F}$, a network $f(\boldsymbol{\theta}_0, \cdot)$ with initialization weights
［#26］
    $\boldsymbol{\theta}_0$, pruning ratios $\rho$, and the desired sparsity level $s$.
Output: Sparse mask $\boldsymbol{m}$ ($\|\boldsymbol{m}\|_0 \ll \|\boldsymbol{\theta}_0\|$), pruning-aware
    critical (PrAC) set $\mathcal{P}$ ($|\mathcal{P}| \ll |\mathcal{D}_0|$)
 1: Set $\boldsymbol{m} = \boldsymbol{1} \in \mathbb{R}^{\|\boldsymbol{\theta}_0\|_0}$, and $\mathcal{D} = \mathcal{D}_0$
 2: while $(1 - \frac{\|\boldsymbol{m}\|_0}{\|\boldsymbol{\theta}_0\|_0} \leq s)$ do
 3:   # Data slimming to construct PrAC sets
 4:   Set $\mathcal{P} = \varnothing$
 5:   Train $f(\boldsymbol{m} \odot \boldsymbol{\theta}_0, \cdot)$ on $\mathcal{D}$ for T epochs and update
    the forgetting statistics for all training samples in $\mathcal{D}$
 6:   Select samples from $\mathcal{D}$ with forgetting statistics
    greater than $\mathcal{E}_\text{F}$, and add them into $\mathcal{P}$
 7:   # Model slimming to locate critical subnetworks
 8:   Prune $\rho = 20\%$ remaining weights of subnetworks
［#26］
    $f(\boldsymbol{m} \odot \boldsymbol{\theta}_\text{T}, \cdot)$, and update $\boldsymbol{m}$ accordingly
 9:   # Data slimming to construct PrAC sets
10:   Select samples from $\mathcal{D}_0$ that full model and subnetworks
    *disagree with*, and add them into into $P$
11:   Set $\mathcal{D} = \mathcal{P}$
12: end while
```

### 3.1. Identifying the Pruning-Aware Critical (PrAC) Set
［#27］
This section shows the details about how to shrink the training set to proposed Pruning-Aware Critical set, which illustrates the process in lines 3-6, 10 of Algorithm 1.

#### Rationale I: Critical Examples for Training.
［#28］
In the network training, each batch of data has its own and likely different statistics. Therefore, they can be regarded as differ-


［#28］
ent "tasks". Catastrophic forgetting happens (Toneva et al., 2019) during the training process so that certain examples are memorized easily during training while some others are repeatedly forgotten. Different behaviors on samples reveal the difficulty of them, providing a natural way to select critical examples, *i.e.*, the degree of difficult-to-forget of each training sample. As pointed out by (Toneva et al., 2019), training models on a dataset with a large fraction of the least forgotten examples removed can yield extremely competitive performance as training on the full data.

［#29］
Approach I: Calculating the Forgetting Statistics. To measure how easy for a model to forget a sample, we use *forgetting statistics* (Toneva et al., 2019) as the metric. Specifically, the forgetting statistics for a sample is the number of transition from a correctly to incorrectly classified sample. We sort the number of forgetting statistics of all training data, and select those have statistics greater than a pre-defined threshold into the PrAC set, in lines 3-6 of Alg. 1.

［#30］
Rationale II: Critical Examples for Pruning. Although the performance of located sparse lottery tickets can match the performance of the full model, the increased number of zero weights might have hampered the memorization and generalization ability of models. Such conjecture has been supported by recent observation (Hooker et al., 2020a), which demonstrates there exists pruning-aware examples that have different prediction between the full and pruned model. These examples are semantically ambiguous and hard for the pruned model to memorize.As a consequence, we merge these easy-to-forget samples into the PrAC set we construct to remedy such capacity loss.

［#31］
Approach II: Utilizing the Disagreement between Full and Pruned Models. For each sample in the training set, we calculate the predicted class of $x$ by full dense models and pruned subnetworks, *i.e.*, $f(\boldsymbol{\theta}, x)$ and $f(\boldsymbol{m} \odot \boldsymbol{\theta}, x)$, where $f(\boldsymbol{\theta}, \cdot)$ is a model with parameters $\boldsymbol{\theta}$, and $\boldsymbol{m}$ is a sparse mask. If two predictions are different, then we include this sample to the PrAC set (line 10 of Algorithm 1).

### 3.2. Efficient Lottery Tickets Finding

［#32］
Matching Subnetworks and Lottery Ticket. A subnetwork within a dense network $f(\boldsymbol{\theta}, \cdot)$ is defined as $f(\boldsymbol{m} \odot \boldsymbol{\theta}, \cdot)$, where $\boldsymbol{m} \in \{0,1\}^{\|\boldsymbol{\theta}\|_0}$ is a binary mask indicating the sparsity levels, and $\odot$ is the element-wise product. Let $\boldsymbol{\theta}_0$ be the initial weights, and $\boldsymbol{\theta}_i$ be the weights after $i$ training steps. Following Frankle & Carbin (2018), we define the *matching network* as a subnetwork $f(\cdot, \boldsymbol{m} \odot \boldsymbol{\theta})$, with $\boldsymbol{\theta}_t$ being the initialization of $\boldsymbol{\theta}$, that can reach the comparable performance to the full network within a similar training iterations; a *winning ticket* is defined as a matching subnetwork where $\boldsymbol{\theta}_0$ as the initial weights.

［#33］
Identifying Subnetworks. To identify subnetworks, we adopt an iterative magnitude pruning method (Han et al., ![](./images/867766605889667803_2.jpg)

［#34］
Figure 2. Results of the pairwise hamming distance between identified subnetworks on CIFAR-10 with ResNet-20

［#35］
2015). We follow a conventional iterative train-prune-retrain process in Frankle & Carbin (2018), yet with our PrAC set: We train the model $f(\boldsymbol{m} \odot \boldsymbol{\theta}, \cdot)$ on our PrAC set, prune a certain percent of the weights, reset and retrain the model, and repeat the process until we meet the sparsity requirement.

［#36］
Turning PrAC set into actual training efficiency. Using PrAC sets for training can save training cost, firstly because less training data directly lead to fewer training iterations per training epoch. However, **the gains are way beyond linear** - since less training data could also imply easier fitting and faster convergence, e.g., less number of epochs. To fully leverage the potential of PrAC sets for efficient ticket finding, we introduce two training strategies for PrAC:
i) **Dynamic training iterations**. After constructing the PrAC set, we will tune the training iterations according to the size of the PrAC set. We linearly scale down the number of iterations using the following formula to decide a new number of training iterations: $N = \frac{|\mathcal{P}|}{|\mathcal{D}_0|}N_0$, where $\mathcal{D}_0$ and $\mathcal{P}$ are the full training set and the PrAC set respectively, and $N_0$ is the original training iterations.

［#37］
In practice, we also tune the learning rate scheduler using the above adjustment formula to re-calculate the decay schedule for learning rates. By scaling down the required training iterations, we can gain training efficiency in a simple but meaningful way.
ii) **Early stopping**. We build an early stopping mechanism upon the dynamic training iterations technique by introducing the Early Bird Ticket (You et al., 2020). It was originally designed for one-shot pruning; however, we reformulate and extend it to our iterative pruning context. As shown by You et al. (2020), winning tickets will emerge at the early period of the training process, which provides empirical support for using the early stopping technique. In our work, we calculate sparsity masks for the model after every epoch of training and monitor the distance between masks as a criterion for early stopping.

［#38］
The distance metric for matrices we use is the Hamming distance, *i.e.*, the number of different elements in two masks. Once the distance becomes smaller than a threshold, we interrupt the training, prune the network and update the sparsity mask, and use it for further retraining. The Hamming distances between masks at different sparsities on different architectures are shown in Figure 2. The graph validates the convergence of Hamming distance between sparsity masks at about half of training.

# Efficient Lottery Ticket Finding: Less Data is More

［#39］
![](./images/867766605889667803_3.jpg)

［#40］
Figure 3. Testing accuracy of subnetworks at a range of sparsity levels from 0% to 99.85% (the first and third rows) and the training iterations for finding each subnetwork (the second and fourth rows) on CIFAR-10, CIFAR-100, and Tiny-ImageNet with ResNet-18, ResNet-20, ResNet-56, and VGG-16. Blue, Green, Orange and Black curves represent our PrAC lottery tickets, vanilla lottery tickets, random pruning, and dense network, respectively. The solid line and shading are the mean and standard deviation of testing accuracy. The numbers within figures are the iterations used to find subnetworks with **the same sparsity and comparable performance**, which indicate our achieved training resources saving. We consider PrAC lottery tickets to achieve a matched performance as vanilla lottery tickets when the performance of PrAC lottery tickets is within one standard deviation of the performance of vanilla lottery tickets.

［#41］
Integrating the above two techniques with the PrAC set, we build our data-model sparsity co-design framework to efficiently find matching subnetworks, termed as *PrAC lottery ticket*, with much less training resources.

## 4. Experiments

### General Setup.
［#42］
We summarize the key setups and hyperparameters of our implementation in Table 1, and refer readers to Appendix A1 for more details. Our experiments use two popular architectures, ResNet (He et al., 2016) and VGG (Simonyan & Zisserman, 2014), on three representative datasets, *i.e.*, CIFAR-10 (Krizhevsky et al.,

［#43］
Table 1. Implementation Details. For ResNet-20 and ResNet-56, we adopt three different training settings: standard, *low* and *warmup* (Frankle et al., 2019a). The low variant means a lower learning rate, and the warmup variant adopts a warm-up method that linearly increases the learning rate from zero.

［#44］
| Network    | Variant  | Dataset               | Batch Size | Learning Rate | Warmup     |
|------------|----------|-----------------------|------------|---------------|------------|
| ResNet-20  | Standard | CIFAR10 & CIFAR100    | 128        | 0.1           | 0          |
|            | Low      |                       |            | 0.01          | 0          |
|            | Warmup   |                       |            | 0.03          | 15 epochs  |
| ResNet-56  | Standard | CIFAR10 & CIFAR100    | 128        | 0.1           | 0          |
|            | Low      |                       |            | 0.01          | 0          |
|            | Warmup   |                       |            |               | 15 epochs  |
| VGG-16     | -        | CIFAR10 & CIFAR100    | 128        | 0.1           | 0          |
|            | -        | Tiny-ImageNet         | 512        | 0.1           |            |
| ResNet-18  | -        | Tiny-ImageNet         | 512        | 0.1           | 0          |


［#45］
Efficient Lottery Ticket Finding: Less Data is More

［#46］
2009), CIFAR-100 (Krizhevsky et al., 2009) and Tiny-ImageNet (Wu et al., 2017). Specifically, we train networks for 182 epochs with a multi-step learning rate schedule, which decays the learning rate to its one-tenth at epoch 91 and 136, respectively. We evaluate the quality of obtained subnetworks, *i.e.*, lottery tickets, by testing accuracy after independently trained from the same random initialization or early rewound weights (Frankle et al., 2019a). All reported results are averaged over three independent runs.

### 4.1. Identifying Winning Tickets with PrAC Sets
［#47］
We evaluate our data and model sparsity co-design framework across diverse datasets and architectures with a total of eight combinations, specifically, CIFAR-10 with {ResNet-20, ResNet-56, VGG-16}, CIFAR-100 with {ResNet-20, ResNet-56, VGG-16}, and Tiny-ImageNet with {ResNet-18, VGG-16}. We consider vanilla lottery tickets (LT) method (Frankle & Carbin, 2018) and random pruning for comparisons. Figure 3 collects the achieved performance of subnetworks with different sparsity and their training effort for identifying each subnetwork, in terms of training iterations. Several observations can be drawn as follows:

［#48］
- Our PrAC lottery tickets match the performance as vanilla lottery tickets in all combinations while notably less training costs, specifically achieving training iteration saving of 83.93% and 69.53% for ResNet-20, 82.85% and 63.54% for ResNet-56, 92.77% and 74.92% for VGG-16 on CIFAR-10 and CIFAR-100, respectively; 76.14% for ResNet-18, and 86.56% for VGG-16 on Tiny-ImageNet. As shown in Figure 3, we record the number of training iterations for the PrAC lottery tickets and the vanilla LT before reaching the highest sparsity that the former can match. And we color the area of the graph according to the number of training iterations for better demonstration.
- Somehow surprisingly, PrAC lottery tickets can even outperform the vanilla lottery tickets at some very high sparsity levels. This intriguing phenomenon implies that utilizing the data-level sparsity by PrAC sets, in addition to efficiency purpose, may even have additional regularization effects on improving the found model's generalization. We will leave further investigation for future work.
- The numbers of examples in PrAC sets across different datasets are adaptively varying. On CIFAR-10, the percentage of the number of the PrAC sets ranges from 35.32% to 37.07%, from 69.55% to 78.19% on CIFAR-100, and from 68.23% to 75.10% on Tiny ImageNet. The ratios of training iterations saved also vary between datasets. On CIFAR-10, we can save training iterations more than 80% but no more than 75% on CIFAR-100, which means that it requires more training effort to find PrAC lottery ticket on CIFAR-100 than CIFAR-10.
- Different architectures show the different percentage of training iteration saving and indicates the speed of matching subnetworks emerge. On VGG-16, our method can save the highest percentage of training iterations, indicating the highest speed to find lottery tickets. On ResNet-56 and ResNet-18, the speed to find lottery ticket is slower; On CIFAR-10 our framework can save 82.85% of training iterations on ResNet-56 while 92.77% on VGG-16; On Tiny ImageNet our framework can save 76.14% on ResNet-18 while 86.56% on VGG-16.

［#49］
![](./images/867766605889667803_4.jpg)

［#50］
Figure 4. The transferability study of PrAC sets on CIFAR-10 and CIFAR-100. Blue, Red, Orange and Black curves represent our PrAC lottery tickets, PrAC tickets found with transferred PrAC sets, random pruning and full network. Each curve contains the mean and standard deviation of test accuracy of subnetworks.

### 4.2. PrAC Sets Are Transferable Across Models
［#51］
The construction of PrAC sets seems model-dependent, relying on a given full dense network and pruned subnetworks. It motives us to investigate to what extent the PrAC sets depend on those factors. As shown in Figure 4, we conduct transferability studies of PrAC sets across network architectures. Specifically, taking ResNet-20 as the source architecture to build PrAC sets on CIFAR-10 and CIFAR-100, and then finding PrAC lottery tickets in ResNet-56 and VGG-16 (target architectures) with transferred PrAC sets.

［#52］
Results in Figure 4 demonstrate that *Transfer PrAC Lottery Tickets* present competitive performance to *PrAC Lottery Tickets*. They show similar accuracies at most sparsity levels, and both surpass randomly pruned subnetworks by a significant performance margin. It demonstrates that PrAC sets are surprisingly transferable for identifying lottery tickets across diverse architectures, which opens up promising avenues of efficiently finding winning tickets in huge models with compact PrAC sets constructed by tiny networks.

### 4.3. Comparisons with Strong Baselines.
［#53］
**Core-set and active learning.** Natural comparative baselines, *i.e.*, core-set and active learning approaches, are considered to assess the quality of PrAC sets further. In spe-


［#54］
![](./images/867766605889667803_5.jpg)

［#55］
Figure 5. Comparison results of our PrAC lottery tickets with subnetworks identified with subsets of random sampling across different architectures and datasets. More results can be found at Figure A10.

［#56］
cific, we adopt two representative methods, active learning via maximum entropy sampling (Lewis & Gale, 1994; Set- tles, 2012) termed as "Entropy", and core-set selection via proxy (Coleman et al., 2020) named as "SVP". Meanwhile, random sampling is designed for a sanity check. For fair comparisons, we keep the training iterations and the number of data in baselines consistent with our sparsity co-design ap- proach. Figure A10 collects the achieved performance of in- dependently trained subnetworks from different approaches on CIFAR-10 with ResNet-20 and Figure 5 further provides a comprehensive comparison with random sampling across different datasets and architectures. Results demonstrate that utilizing PrAC sets is capable of finding consistently better subnetworks with higher accuracies across diverse sparsity. It suggests that our co-design of data and model sparsity produces more informative pruning-aware subsets, which benefits to locate high-quality winning tickets.

［#57］
![](./images/867766605889667803_6.jpg)

［#58］
Figure 6. Comparison results with strong baselines. Left: Compar- ison of our PrAC lottery tickets with other pruning methods. Right: Comparison of our methods with random pruning or initialization.

［#59］
Other efficient network pruning approaches. Recent proposed SNIP (Lee et al., 2018), GraSP (Wang et al.,2020), and SynFlow (Tanaka et al., 2020) aim to prune networks at initialization, thereby saving resources at train- ing stages. They usually only require a single batch of training examples with certain effective pruning criterion to find subnetworks in one-shot, which can be enhanced with more data and training budgets (Wang et al., 2020; Tanaka et al., 2020). For fair comparisons, we implement these methods in an iterative manner (usually better than one-shot (Han et al., 2015; Frankle & Carbin, 2018)) with the training iterations and the number of training data con- sistent with our approaches. As shown in Figure 6 (Left), only our approach is able to identify winning tickets with matched performance to full unpruned models (i.e., Base- line), and obtain a consistent performance margin compared to other pruning methods. Specifically, PrAC lottery tickets with 93.13% sparsity surpass SynFlow, SNIP, and GraSP by 1.51%, 4.22% and 4.36% test accuracy. With the computa- tion consumption remains constant for all algorithms, this achieved significant performance gap verifies the superiority of our proposal.

［#60］
Random tickets with random re-initialization. To ex- clude the possibility of trivial solutions, we consider the commonly adopted baseline, random tickets trained from randomly re-initialized weights, from the LTH literature (Frankle & Carbin, 2018). From Figure 6 (Right), we ob- serve that PrAC lottery tickets hold overwhelming advan- tages. For example, with a 1% accuracy gap against the full model, our identified matching subnetworks with a spar- sity of 79.03%, which are much sparser than both random pruning (48.80%) and random tickets (48.80%).

### 4.4. Ablation Study

［#61］
The Two Components in the PrAC set. To investigate the individual effect of critical examples for training (CET) and pruning (CEP), we only utilize CET to identify match- ing subnetworks, as presented in Figure A12. Results show that without the assistance of CEP, the found subnetworks consistently incur $\sim1\%$ performance drop. Table A3 col- lects the number of samples in CET and CEP. We observe that as the sparsity grows, the number of CEP keeps increas- ing; meanwhile, CEP shares fewer overlap images with CET, which indicates gradually detached distributions of critical samples during training and pruning.

# Efficient Lottery Ticket Finding: Less Data is More

［#62］
![](./images/867766605889667803_7.jpg)

［#63］
Figure 7. Testing accuracy of subnetworks at a range of sparsity levels from 0% to 96.48% (the first row) and the training iterations for finding each subnetwork (the second row) on CIFAR-10 with ResNet-20 under different lottery ticket settings. The numbers within figures are the iterations used to find the subnetworks with the same sparsity and comparable performance. More results can be found at Figure A11.

［#64］
With or without early stopping. We adopt the early stopping (You et al., 2020) technique in our framework to find PrAC lottery tickets more efficiently. To understand its effect, we implement the variant, PrAC w.o. Early Stop, that disables the early stopping in our methods. As shown in Figure 8, we observe that PrAC w.o. Early Stop finds subnetworks with the same sparsity level and similar performance as vanilla lottery tickets, achieving 40.63% training resources saving. Equipped with the early stopping, PrAC lottery tickets at the same sparsity, obtain 83.93% training resources saving at the cost of $\leq 0.50\%$ accuracy loss.

［#65］
![](./images/867766605889667803_8.jpg)

［#66］
Figure 8. Ablation studies of vanilla lottery tickets and our PrAC lottery tickets w/w.o. early stopping on CIFAR-10 with ResNet-20. Left: Testing accuracy of subnetworks with different sparsity. Right: Training iterations for finding each subnetwork. And the numbers within the figure are the iterations used to find the corresponding subnetworks with the same sparsity and comparable performance. (72k, 266k, 448k represent PrAC lottery tickets, PrAC w.o. Early Stop and vanilla lottery tickets, respectively.)

［#67］
Validating across diverse lottery ticket settings. Here we further evaluate our framework under two additional lottery ticket settings proposed by Frankle & Carbin (2018), i.e. low and warmup, with ResNet-20 and ResNet-56 on CIFAR-10 and CIFAR-100, respectively. Detailed hyperparameters are listed in Table 1. Figure 7 and A11 shows that to find subnetworks with similar performance under the low and warmup settings, our methods only cost 18.91% $\sim$ 22.22% and 34.33% $\sim$ 38.01% training resources on CIFAR-10 and CIFAR-100, compared to vanilla lottery tickets. These consistently achieved training savings further verify the efficiency of PrAC lottery tickets, and the effectiveness of our sparsity co-design framework.

［#68］
![](./images/867766605889667803_9.jpg)

［#69］
Figure 9. Visualization of examples out of (upper) and within (bottom) the final PrAC set on Tiny-ImageNet.

## 4.5. Visualization and Analyses of PrAC Sets

［#70］
Visualization of examples out of and within PrAC sets on Tiny-ImageNet is provided in Figure 9, and the class-wise ratios of images in PrAC sets can be found in Figure A15. As shown in Figure 9, the images out of the PrAC set show less complexity where objects are centered and easily distinguishable, such as identifying an orange from white backgrounds. In contrast, the images within PrAC sets contain multiple ambiguous elements, including lower quality, depict multiple objects, complicated backgrounds and resulting in a challenging recognition even for a human. In addition, the distribution of PrAC set's classes in Figure A15 is quite balanced, where the number of images is in the same order. Such observations may provide possible insights on why PrAC sets are capable of locating critical subnetworks, i.e., PrAC tickets, with satisfying performance.

## 5. Conclusion

［#71］
In this paper, we explore a new perspective to finding lottery tickets more efficiently by doing so on small pruning-aware critical (PrAC) subsets, which is constructed via data and model sparsity co-design. Extensive experiments verify the effectiveness of our proposals with diverse network architectures on multiple common datasets, including CIFAR-10, CIFAR-100, and Tiny ImageNet. High-quality winning tickets, can be identified efficiently on such compact PrAC sets and enjoys significant training cost reduction.

［#72］
Efficient Lottery Ticket Finding: Less Data is More

## References
























































# Efficient Lottery Ticket Finding: Less Data is More

## A1. More Implementation Details

［#73］
Training and evaluation details. We use an SGD optimizer with a momentum of 0.9 and a weight decay of $10^{-4}$ in our experiments. And we choose the model with the best validation accuracy during the training process. Besides, we use an early weight rewinding (Frankle et al., 2019a) method (rewind to the third epoch) to help scale up the lottery ticket hypothesis in these models, except for the warmup and low variant of ResNet-20 and ResNet-56, in which the weight will be rewound to the same random initialization. For the variant of warmup, we replace the original 85 epochs (Frankle & Carbin, 2018) with 15 epochs, which does not affect the performance. The threshold of the number of forgets is set to 0 and we default to use 0.07 as the threshold for the distance between masks. Note that our baseline results are aligned with (Frankle & Carbin, 2018).

［#74］
Dataset. We consider three datasets in our implementation, which can be download at `https://www.cs.toronto.edu/~kriz/cifar.html` for CIFAR-10 and CIFAR-100, and `http://cs231n.stanford.edu/tiny-imagenet-200.zip` for Tiny-ImageNet. For all three datasets, 10 percent of data from the training set are randomly split up as validation set. And we utilize random cropping and random horizontal flipping for data augmentation.

［#75］
Computing infrastructures. All our experiments are conducted on Quadro RTX 6000 and Tesla V100 GPUs.

## A2. More Experiment Results

### A2.1. More Results of Sampling Strategy

［#76］
As shown in Figure A10, our PrAC sets achieve consistent improvement compare with other sampling strategies. It indicates that our approach produces more informative pruning-aware subsets and contribute for finding high-quality winning tickets.

［#77］
![](./images/867766605889667803_10.jpg)

［#78］
Figure A10. Comparison of our PrAC sets with other core-sets or active learning approaches.

［#79］
![](./images/867766605889667803_11.jpg)

［#80］
Figure A11. Testing accuracy of subnetworks at a range of sparsity levels from 0% to 96.48% (the first row) and the training iterations for finding each subnetwork (the second row) on CIFAR-100 with ResNet-56 under different lottery ticket settings. Blue, Green, Orange, and Black curves represent our PrAC lottery tickets, vanilla lottery tickets, random pruning and full network, respectively. The numbers within figures are the iterations used to find the subnetworks with the same sparsity and comparable performance.

### A2.2. More Results of Different Lottery Ticket Settings

［#81］
Figure A11 reports the performance on CIFAR-100 with ResNet-56 under two additinal lottery tickets settings, low and warmup. We can observe that our methods cost 34.33% $\sim$ 38.01% training sources and achieve comparable performance, which suggests the efficiency of our PrAC lottery tickets.

### A2.3. More Statistics of PrAC Sets

［#82］
Table A2 contains the size of PrAC sets across different datasets and networks. On CIFAR-10 (10 classes), we locate PrAC sets with the size range from 35.32% to 37.07% of the training set, while 69.55% to 78.19% on CIFAR-100 (100 classes) and 68.23% to 75.10% on Tiny-ImageNet (200 classes). The result suggests that more data are needed to find high-quality PrAC lottery tickets for the image recognition with more classes.

### A2.4. More Results of Ablation Study

［#83］
The two components in the PrAC set. We conduct our data and model co-design framework with only critical examples for training (CET), named as CET lottery tickets. As

# Efficient Lottery Ticket Finding: Less Data is More

［#84］
Table A2. Proportion of PrAC sets to their training set sizes of CIFAR-10, CIFAR-100 and Tiny-ImageNet

［#85］
| Dataset         | Network   | Proportion of PrAC sets |
|-----------------|-----------|-------------------------|
|                 | ResNet-20 | 36.66%                  |
| CIFAR-10        | ResNet-56 | 37.07%                  |
|                 | VGG-16    | 35.32%                  |
|                 | ResNet-20 | 78.19%                  |
| CIFAR-100       | ResNet-56 | 74.94%                  |
|                 | VGG-16    | 69.55%                  |
| Tiny-ImageNet   | ResNet-18 | 75.10%                  |
|                 | VGG-16    | 68.23%                  |

［#84］
shown in Figure A12, without the assistance of critical examples for pruning (CEP), there is a consistent performance gap between PrAC lottery tickets and CET lottery tickets. Besides, we collect the number of CET, CEP and PrAC sets in Table A3. The overlapping rate means the percentage of the overlap images between CET and CEP sets in CEP sets, ($i.e.$, $\frac{|CEP| \cap |CET|}{|CEP|}$). We observe that as the sparsity grows, the number of CEP sets increases while the overlapping rate decreases, which indicates gradually detached distributions of critical samples during training and pruning.

［#86］
Table A3. Results of the number of the identified CET, CEP and PrAC sets, as well as the overlapping rate of CEP sets during the process of our co-design framework on CIFAR-10 with ResNet-20.

［#87］
| Sparsity of Subnetworks | CEP  | CET  | PrAC | Overlapping Rate |
|--------------------------|------|------|------|------------------|
| 20.00%                   | 1501 | 24159| 24168| 99.40%           |
| 36.00%                   | 1481 | 21708| 21728| 98.65%           |
| 48.80%                   | 3935 | 19542| 19838| 92.48%           |
| 59.04%                   | 3782 | 17674| 18161| 87.12%           |
| 67.23%                   | 4723 | 16091| 16712| 86.85%           |
| 73.79%                   | 5514 | 14771| 16026| 77.24%           |
| 79.03%                   | 4420 | 14357| 15202| 80.88%           |
| 83.22%                   | 4602 | 13909| 14880| 78.90%           |
| 86.58%                   | 5391 | 13741| 14980| 77.02%           |
| 89.26%                   | 5360 | 14168| 15376| 77.46%           |
| 91.41%                   | 5098 | 14365| 15247| 82.70%           |
| 93.13%                   | 5840 | 14553| 15804| 78.58%           |
| 94.50%                   | 5728 | 14959| 16062| 80.74%           |
| 95.60%                   | 6370 | 15290| 16360| 83.20%           |
| 96.48%                   | 6616 | 15369| 16499| 82.92%           |

［#88］
Relative similarity between PrAC LT and LT. We evaluate the overlap degree in sparsity patterns with relative similarity ($i.e.$, $\frac{m_i \cap m_j}{m_i \cup m_j}$), where $m_i$ and $m_j$ are the sparsity masks of identified subnetworks. We keep the same random initialization for PrAC lottery tickets and two independent runs of vanilla lottery tickets. Figure A13 shows that as the sparsity grows, subnetworks share fewer sparsity patterns. And the relative similarity between PrAC lottery tickets and lottery tickets are slightly smaller than between two different runs of lottery tickets, which indicates the non-trivial difference between sparse masks of PrAC LT and LT.

［#89］
![](./images/867766605889667803_12.jpg)

［#90］
Figure A12. Comparison of PrAC lottery tickets with CET lottery tickets on CIFAR-10 with ResNet-20. Each curve contains the mean and standard deviation of testing accuracy of subnetworks at different sparsity levels.

［#91］
![](./images/867766605889667803_13.jpg)

［#92］
Figure A13. Results of the relative mask similarity on CIFAR-10 with ResNet-20. Green and Orange represents the relative similarity between two independent runs of vanilla lottery tickets, and the one between PrAC lottery tickets and vanilla lottery tickets. We adopt the same random initialization for identifying these three groups of subnetworks.

［#93］
Lottery tickets with subsets of random sampling To investigate that how many examples of random sampling can match the performance of our PrAC subsets in terms of locating subnetworks, we conduct an ablation study on CIFAR-10 with ResNet-20 and record the results in Figure A14. We can observe that nearly 70% data are needed for random subsets to match the performance of our PrAC sets, which only contain 37% $\sim$ 54% data.

［#94］
Table A4. Results of test accuracy of identified subnetworks with respect to the threshold for the number of forgets on CIFAR-10 with ResNet-20. We select subnetworks with the same sparsity of 16.78%, which is the maximum sparsity of subnetworks identified by PrAC subsets ($\mathcal{E}_F = 0$) have comparable performance.

［#95］
| $\mathcal{E}_F$ | 0     | 2     | 4     | 6     | 8     | 10    |
|------------------|-------|-------|-------|-------|-------|-------|
| Accuracy (%)     | 91.05 | 90.43 | 90.35 | 89.32 | 89.08 | 88.79 |
| PrAC             | 19748 | 14992 | 12536 | 11141 | 11152 | 10338 |

［#96］
The threshold for the number of forgets Table A4 records the test accuracy and the size of PrAC subsets un-


［#97］
![](./images/867766605889667803_14.jpg)

［#98］
Figure A14. Comparison of the quality of subnetworks identified by our PrAC subsets and subsets from random sampling on CIFAR-10 with ResNet-20. We keep the size of random subsets consistent during the whole IMP process, ranging from $60\% \sim 80\%$.

［#99］
der different threshold for the number of forgets. We can observe that both the test accuracy and the size of PrAC decrease as the threshold rises. Thus we choose $\mathcal{E}_\text{F} = 0$ in our implementation.

### A2.5. More Visualization and Analyses

［#100］
![](./images/867766605889667803_15.jpg)

［#101］
Figure A15. The class-wise ratios of images in PrAC sets on CIFAR-10/100 and Tiny-ImageNet, respectively. Red and Orange represent the classes with maximum and minimum images.

［#102］
Figure A15 demonstrates the class-wise ratios of images in PrAC set, from which we can find that the number of images from different classes are in the same order. This balanced distribution of PrAC set's classes may provide possible insights on the effectiveness of PrAC sets, with respect to locating critical subnetworks, i.e., PrAC tickets, with satisfying performance.

### A2.6. Additional Results of Forgetting Statistics in LT

［#103］
Figure A16 shows the distribution of training data's forgetting times at different sparsity from $0\%$ to $96.48\%$ on CIFAR-10 with ResNet-20. We consider three pruning methods: Basic iterative magnitude pruning (IMP) (Han et al., 2015), vanilla lottery tickets (LT) (Frankle & Carbin, 2018), and random tickets (RT). IMP fine-tune the subnetworks directly after pruning while LT rewinds the weight to the same initialization and RT reinitializes the subnetworks before fine-tuning. We can observe that as the sparsity increases, for IMP and LT, the number of unforgettable images first increases and then decreases, while the one for RT consistently decreases. Besides, the maximum number of forgetting times grows as the sparsity becomes larger.

［#104］
![](./images/867766605889667803_16.jpg)

［#105］
Figure A16. Visualization of the forgetting statistics of subnetworks at different sparsity from $0\%$ to $96.48\%$ on CIFAR-10 with ResNet-20 when training with full data. Top: Basic iterative magnitude pruning (fine-tune after pruning). Middle: vanilla lottery tickets. Bottom: random tickets.