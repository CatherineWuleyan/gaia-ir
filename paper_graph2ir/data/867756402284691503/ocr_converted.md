# When Layers Play the Lottery, all Tickets Win at Initialization

［#1］
Artur Jordao, George Corrêa de Araújo, Helena de Almeida Maia and Helio Pedrini

［#2］
Institute of Computing, University of Campinas, Campinas

## Abstract

［#3］
Pruning is a standard technique for reducing the computational cost of deep networks. Many advances in pruning leverage concepts from the Lottery Ticket Hypothesis (LTH). LTH reveals that inside a trained dense network exists sparse subnetworks (tickets) able to achieve similar accuracy (i.e., win the lottery – winning tickets). Pruning at initialization focuses on finding winning tickets without training a dense network. Studies on these concepts share the trend that subnetworks come from weight or filter pruning. In this work, we investigate LTH and pruning at initialization from the lens of layer pruning. First, we confirm the existence of winning tickets when the pruning process removes layers. Leveraged by this observation, we propose to discover these winning tickets at initialization, eliminating the requirement of heavy computational resources for training the initial (over-parameterized) dense network. Extensive experiments show that our winning tickets notably speed up the training phase and reduce up to 51% of carbon emission, an important step towards democratization and green Artificial Intelligence. Beyond computational benefits, our winning tickets exhibit robustness against adversarial and out-of-distribution examples. Finally, we show that our subnetworks easily win the lottery at initialization while tickets from filter removal (the standard structured LTH) hardly become winning tickets.

［#4］
![](./images/867756402284691503_1.jpg)

［#5］
Figure 1: Lottery Ticket Hypothesis (LTH) views according to the structure (weights, neurons/filters or layers) the pruning is eliminating (transparent regions). Top-left. Original unstructured LTH: the pruning removes weights and yields unstructured tickets; thereby, the tickets only provide practical benefits on specialized frameworks for sparse computations. Top-right. Structured LTH: the pruning eliminates neurons/filters. In this setting, the tickets are structured and promote computational advantages to standard deep learning frameworks. Bottom-left. Ours structured LTH: the pruning eliminates entire layers, encouraging additional performance gains since it decreases the sequential processing (latency). Bottom-right. The highest gain (the higher, the better) obtained by a winning ticket regarding its dense counterpart. Our winning tickets successfully emerge at initialization, which means we can discover efficient subnetworks without training a dense network. In this direction, we can considerably speed up the learning phase by replacing a dense network with its sparse version before training begins. Our winning tickets also exhibit robustness against adversarial attacks.

## 1 Introduction

［#6］
The Lottery Ticket Hypothesis (LTH) conjectures that (pre-trained) dense networks contain sparse subnetworks capable of obtaining the same accuracy when trained from the original initialization of their dense counterpart [Frankle and Carbin, 2019]. Subnetworks (tickets) that satisfy this property are referred to as winning tickets. Many advances emerge from the LTH, for example, we can reduce the computational cost of learning a dense network by replacing it with a sparse subnetwork during [You et al., 2020; Chen et al., 2022] or before the course of training [Lee et al., 2019; Wang et al., 2020; Tanaka et al., 2020]. In the context of adversarial attacks, we can increase the robustness of dense networks by considering their sparse versions [Diffenderfer et al., 2021; Liu et al., 2022; T et al., 2022].

［#7］
The traditional LTH seeks winning tickets by removing (i.e., pruning) weights of a dense network and assigning the survival weights to their original initialization [Frankle and Carbin, 2019] – weight rewinding. Variants of this mechanism propose relaxing the weight-rewinding constraint and enabling subnetworks to inherit weights from different training epochs [Frankle et al., 2020; Renda et al., 2020] and even

［#7］
from random (re) initialization [Liu et al., 2019]. Another variant in the traditional LTH is the replacement of weight pruning (unstructured) by filter pruning (structured) [Liu et al., 2019; Renda et al., 2020; You et al., 2020]. For example, Prasanna et al. [2020] studied the LHN on structured pruning by removing self-attention modules (i.e., heads from multi-head attention layers [Vaswani et al., 2017]). One practical drawback of unstructured pruning is that existing deep learning frameworks (e.g., TensorFlow and PyTorch) do not support sparse tensor computations. Hence, in order to obtain performance gains, this family of pruning requires specialized frameworks or hardware (Nvidia A100 GPU) for optimizing sparse computations [Han et al., 2016; Niu et al., 2020; Zhou et al., 2021].

［#8］
Regardless of the structure, we can categorize LTH according to the phase in which the pruning algorithm picks the subnetworks (tickets): after, during, or before training. The first class trains a dense network and, then, extracts a subnetwork according to an importance criterion (pruning criterion), such steps compose the original LTH [Frankle and Carbin, 2019]. The second prunes a dense network during the course of training, obtaining a subnetwork when the learning phase is over [You et al., 2020; Chen et al., 2022]. Finally, pruning before training constitutes a relatively recent form of pruning named pruning at initialization [Lee et al., 2019; Tanaka et al., 2020; Wang et al., 2020]. The idea behind this category is to find subnetworks prior to training, which means picking subnetworks using the randomly initialized parameters (without any update) to guide the pruning algorithm. Many efforts have been put into pruning at initialization as it provides all benefits of sparse subnetworks without spending computational resources to train a dense network (a process required by the other categories).

［#9］
Importantly, all pruning categories in LTH share the same characteristic: the pruning eliminates small structures such as weights or filters. Outside the context of LTH, previous works have demonstrated notable benefits of removing entire layers instead of small structures. In particular, this type of structured pruning promotes additional performance gains since it reduces the sequential processing (latency) of a network [Li et al., 2021; Zhou et al., 2022]. Hence, many studies have focused on removing layers instead of other structures [Veit and Belongie, 2020; Fan et al., 2020; Zhou et al., 2022; Zhang and Liu, 2022]. Unfortunately, none of these efforts have been done in the direction of LTH and pruning at initialization. To bridge this gap, we take a step towards understanding the behavior of LTH when the pruning process removes layers. In this setting, we first verify the existence of winning tickets. Then, we propose a systematic strategy for discovering winning tickets before training, which means finding sparse subnetworks (from layer removal) at initialization that match the predictive ability of their dense equivalent.
Contributions. We list the following key contributions. First, we demonstrate the existence of winning tickets – sparse subnetworks that obtain the same accuracy as their dense equivalent – when the pruning process takes into account the removal of layers. From a practical perspective, such a contribution enables winning tickets to be more efficient in terms of memory consumption, inference time (latency) and carbon emission, as removing layers provides superior computational benefits than standard forms of pruning employed in LTH (see Figure 1). Second, we successfully find winning tickets from layer removal at initialization without any training. This contribution plays a role in low-cost and energy-efficient training, as we can replace the learning of a dense network with its sparse version. In contrast to previous LTH studies [Frankle et al., 2020; Renda et al., 2020], this contribution eliminates the requirement of heavy computational resources for learning the initial (over-parameterized) dense network. Finally, we show that this novel family of subnetworks shares some properties of the standard structured LTH (filter pruning), for example, they emerge early at training. On the other hand, we observe that subnetworks from filter pruning exhibit poor performance when applied to shallow (underparametrized) networks and before training, which is aligned with previous evidence [Liu et al., 2022]. In other words, tickets from filter pruning hardly become winning tickets. However, our tickets (layer removal) easily win the lottery, raising the question of whether the eliminated structure plays a role in the LTH and pruning at initialization.

［#10］
According to our analysis, winning tickets from layer pruning are accurate and robust to many settings such as the pruning criteria and density, and weight rewinding. When found at initialization, these winning tickets speed up the training time by up to $2\times$ and save notable FLOPs and memory consumption. On a specific pruning criterion, all tickets outperform the predictive ability of their dense counterpart – all tickets become winning tickets. Additionally, our winning tickets reduce up to $51\%$ of carbon emission during the training phase, an important step towards democratization and sustainability of Artificial Intelligence (AI) – green AI. As suggested by previous works [Liu et al., 2022; T et al., 2022; Chen et al., 2022], we also evaluate our winning tickets on adversarial images and out-of-distribution examples using CIFAR-C [Hendrycks and Dietterich, 2019] and CIFAR-10.2 [Lu et al., 2020] datasets, respectively. On the standard clean training (i.e., without any defense mechanism), tickets from layer pruning achieve higher robustness than their dense equivalent; thus, confirming their suitability for safety-critical applications. For reproducibility purposes, our code is publicly available: https://github.com/arturjordao/LayerLottery.

## 2 Related Work

［#11］
Frankle et al. [2019] introduced the Lottery Ticket Hypothesis and observed that subnetworks from a dense network can obtain similar accuracy since trained from the same original initialization. Later, Frankle et al. [2020] and Renda et al. [2020] confirmed that rewinding the weights to other epochs of the training, instead of the original initialization, enables subnetworks to match the accuracy of their dense counterpart.

［#12］
The work by Achille et al. [2019] investigated the sensitivity of networks during training. The authors observed that perturbations, such as blur and random label, affect less the training dynamic when applied in early epochs. By considering pruning as a form of perturbation (e.g., due to the removal of structures), such evidence reinforces the superior performance of subnetworks rewound to early training epochs, as

［#12］
argued by subsequent studies [You *et al.*, 2020]. While weight rewinding lies at the heart of LTH, Liu et al. [2019] demonstrated that the winning tickets exist even on a random initialization regime. Throughout this study, we show that our winning tickets (resulting from layer pruning) emerge from different rewinding epochs, random initialization and before training, with the latter achieving the best gains in terms of accuracy and computational performance.

［#13］
Liu et al. [2021] suggested that the existence of winning tickets correlates with the transition from the initial and final parameters of the dense network. From a different perspective, Paul et al. [2022] showed that data quantity and quality play a role in LTH. More concretely, the authors observed that training on easy or on a small fraction of randomly chosen data promotes a fine initialization to dense networks such that their subnetworks become winning tickets. In line with Liu et al. [2021], You et al. [2020] proposed drawing sparse subnetworks early at training by identifying when the magnitude of the weights becomes stable (i.e., suffer small changes) during the course of training. After locating a subnetwork, named early bird, the authors replaced the training of the dense with the subnetwork; thus leading to faster and lower-cost training. Built upon this idea, Chen et al. [2022] proposed to find early birds to reduce the cost of adversarial training. However, the authors identify subnetworks by pruning weights (unstructured) while You et al. [2020] focus on removing filters (structured pruning). Interestingly, the results by Chen et al. [2022] corroborate the observation of previous works [Diffenderfer *et al.*, 2021; Liu *et al.*, 2022], which state that pruning not only reduces the computational cost but also promotes robustness against adversarial images. Additionally, T et al. [2022] confirmed that winning tickets generalize better than dense networks in limited-data regimes and out-of-distribution scenarios.

［#14］
Beyond computational benefits, You et al [2020] and Chen et al. [2022] showed that winning tickets drawn early at training obtain higher accuracy. From the LTH rewinding perspective, this means that resetting the weights of subnetworks to early epochs of the dense network allow subnetworks to achieve higher predictive ability, hence, they are more probable to become a winning ticket. Our results are aligned with these findings: subnetworks from layer pruning achieve superior accuracy when rewound to early epochs. In contrast to these works, we also propose to identify winning tickets at initialization (prior to training). Informally speaking, our subnetworks are the earliest as possible.

［#15］
Closely related to LTH, several efforts have been put into pruning at initialization. In this strategy, the pruning algorithm removes unimportant structures before any training [Lee *et al.*, 2019; Wang *et al.*, 2020; Tanaka *et al.*, 2020]. The study by Frankle et al. [2021] pointed out the difficulties and inherent properties of many pruning at initialization strategies. The authors showed that this family of strategies is less effective than standard pruning (i.e., LTH) in terms of both accuracy and sparsity. Jorge et al. [2021] corroborate such findings; further, they observed that pruning at initialization is no better than random pruning on high sparsity regimes. Liu et al. [2022] suggested an opposite behavior: pruning at initialization works well on deep and wide networks, even randomly selecting the unimportant structures. Additionally, the authors showed that employing a layer-wise pruning density yields more accurate winning tickets. We observe that the evidence by Liu et al. [2022] holds when pruning takes into account filters; however, we successfully find winning tickets at initialization on both shallow (i.e., ResNet32) and deep architectures (i.e., ResNet56), which suggest an architecture-agnostic form of discovering winning tickets prior to training. Though it looks like a counterintuitive phenomenon, there is a body of studies confirming the benefits of removing layers instead of filters [Zhang and Liu, 2022; Zhou *et al.*, 2022]. In summary, our work differs from previous studies on LTH and pruning at the initialization in terms of the structure we focus on removing – layers.

［#16］
To the best of our knowledge, the idea behind pruning layers dates back to 2016, when Veit et al. [2016] and Huang et al. [2016] demonstrated that residual-based architectures exhibit no degradation when we remove some of their layers. Later, Dong et al. [2021] confirmed that self-attention architectures also share similar behavior. Since these pioneer studies, many works have proposed either removing layers statically or dynamically. It is important to mention the difference between these categories of layer pruning. In the former, the pruning algorithm eliminates layers in the same way as weight and filter pruning does, yielding a permanently shallower network [Zhang and Liu, 2022; Zhou *et al.*, 2022]. Differently, the rationale behind the dynamic layer pruning consists of skipping (i.e., deactivating) some layers according to the input sample the network receives [Han *et al.*, 2022]. Our work belongs to the first category of layer pruning. In particular, due to its dynamic nature, we believe it is impractical to study LTH and pruning at the initialization in the second group.

## 3 Preliminaries and Problem Statement

［#17］
Definitions. Let $\mathcal{F}$ and $\mathcal{F}'$ be a dense and sparse (subnetwork) network, respectively. The latter is a version of $\mathcal{F}$ without some structures (neurons, filters or layers), meaning $\mathcal{F}$ after some pruning process. Assume $\theta_i$ the parameters from $\mathcal{F}$, in which the subscript $i$ indicates the weights at the training epoch $i$. We indicate the randomly initialized weights and the ones after the training stage as $\theta_0$ and $\theta_n$, in this order.

［#18］
Pruning Algorithm. A pruning algorithm identifies and removes unimportant structures composing a network. For this purpose, it measures the importance of each structure according to an importance criterion $c$. Let $S$ be a set of (sorted) scores that indicates the importance of each structure of $\mathcal{F}$. Given $S$, the pruning algorithm removes the least important structures in order to satisfy a pruning density $p$ (i.e., the percentage or number of structures removed.). In the literature on pruning, it is common to mask the eliminated structures with zeros values. In contrast, we indeed remove the structures (layers) to achieve practical performance gains without needing specialized frameworks or hardware for sparse computations. We refer interested readers to Appendix A.1 for additional details about this technical process.

［#19］
Lottery Ticket Hypothesis. Following the previous definitions, the original Lottery Ticket Hypothesis (LTH) [Frankle

［#20］
```
Algorithm 1 LTH Removing Layers of Deep Networks
Input: Convolutional Network $\mathcal{F}$ trained on $n$ epochs
Input: Weight Rewind $\theta_{i}$
Input: Pruning Criterion $c$
Input: Pruning Density $p$
Output: Subnetwork (Ticket) $\mathcal{F}'$
［#20］
$S \leftarrow c(\mathcal{F}, \theta_{n}) \triangleright$ Assigns importance for each layer
［#20］
$I \leftarrow p$ unimportant layers based on $S$
［#20］
$\mathcal{F}' \leftarrow \mathcal{F} \setminus I \triangleright$ Removes the layers indexed by $I$
Set the weights of $\mathcal{F}'$ as $\theta_{i} \triangleright$ Weight rewinding
Train $\mathcal{F}'$ via standard training for $n-i$ epochs
```

［#21］
```
Algorithm 2 Winning Tickets at Initialization
Input: Untrained Convolutional Network $\mathcal{F}$
Input: Pruning Criterion $c \triangleright$ SNIP or GraSP
Input: Pruning Density $p$
Output: Subnetwork (Ticket) $\mathcal{F}'$
［#21］
$S \leftarrow c(\mathcal{F}, \theta_{0}) \triangleright$ Assigns importance for each layer
［#21］
$I \leftarrow p$ unimportant layers based on $S$
［#21］
$\mathcal{F}' \leftarrow \mathcal{F} \setminus I \triangleright$ Removes the layers indexed by $I$
Train $\mathcal{F}'$ via standard training for $n$ epochs
```

［#21］
and Carbin, 2019] states that inside $\mathcal{F}$ exists sparse subnetworks $\mathcal{F}'$ able to achieve similar (or, ideally, superior) accuracy since trained from the identical initialization $\theta_{0}$. In this definition, the subnetworks are named tickets and the ones that satisfy such property are named winning tickets. In other words, a winning ticket is a subnetwork with the same predictive ability (i.e., accuracy) as its dense equivalent. In practice, we can define the existence of winning tickets in terms of

［#21］
$$Accuracy(\mathcal{F}') + \xi \geq Accuracy(\mathcal{F}), \xi \geq 0. \tag{1}$$

［#22］
As we mentioned before, variants of LTH enable $\mathcal{F}'$ subnetworks to inherit weights from different training epochs (i.e., $\theta_{i}$ with $i>0$) [Frankle et al., 2020; Frankle et al., 2021]. This step composes the weight rewinding process. Algorithm 1 summarizes the steps of our LTH (layer pruning).

［#23］
From the above description, we highlight two observations. (i) In Equation 1, $\xi$ enables controlling (relaxing) how much the accuracy of a ticket can differ from its dense network and still be considered a winning ticket, where common values are one percentage point or one standard deviation [Chen et al., 2020; Frankle et al., 2021]. (ii) Before extracting a potential winning ticket, we need to train $\mathcal{F}$ to completion (i.e., training for $n$ epochs to obtain $\theta_{n}$) – see the first input in Algorithm 1.
**Pruning at Initialization**. This relatively recent category of pruning estimates which structures to remove before training. Technically speaking, it focuses on discovering subnetworks using the random initialization $(\theta_{0})$ of $\mathcal{F}$ to guide the pruning algorithm. From the lens of LTH, pruning at initialization aims at yielding winning tickets without training $\mathcal{F}$.

［#24］
Algorithm 2 summarizes the steps of our pruning at initialization. In Algorithm 2, it is important to observe that it receives an untrained network, whereas in our standard LTH (Algorithm 1) the input is a trained network.
**Research Questions.** From the above definitions, our research questions are the following. (i) We ask if there are winning tickets when $\mathcal{F}'$ derives from a layer-pruning process. In other words, we investigate if Equation 1 holds when the pruning removes layers. We confirm that the answer is positive; thus, raising our second question: (ii) Is it possible to discover such winning tickets (from layer pruning) at initialization? Formally, is it possible to extract an $\mathcal{F}'$ that satisfies Equation 1 without training $\mathcal{F}$? Answering this enables us to avoid the computationally expensive training of a dense network by replacing it directly with its sparse version before training begins. Overall, both questions focus on analyzing Equation 1 after performing Algorithm 1 and Algorithm 2.

## 4 Experiments
［#25］
**Experimental Setup.** We conduct our experiments on CIFAR-10 [Krizhevsky et al., 2009] using the ResNet32 and ResNet56 networks [He et al., 2016]. Such settings are common choices for LTH and general pruning evaluation [Blalock et al., 2020]. In our LTH, we employ the learning rate rewinding scheme suggested by Renda et al. [2020], as it leads to better results than the one proposed in the original LTH [Frankle and Carbin, 2019]. In contrast to most LTH studies, we evaluate the LTH on multiple criteria instead of employing only the $\ell_{1}$-norm criterion (the iterative magnitude pruning - IMP - originally proposed by Frankle et al. [2019]). The reason for this choice is to assess the existence of winning tickets on multiple pruning settings. Additionally, layers at different depths exhibit different magnitudes, hence, it is unfeasible to compare the $\ell_{1}$-norm from multiple layers (see Appendix A.2). Specifically, we employ the pruning criteria proposed by Lin et al. [2020] (HRank), Tan and Motani [2020] (expectABS), Jordao et al. [2020] (PLS) and Luo and Wu [2020] (KL). In addition, we use the commonly used random criterion.

［#26］
We report (in percentage points - pp) the improvement and decline of the subnetworks (tickets) with respect to their dense counterpart using the symbols (+) and (-), in this order. Thereby, the symbol (+) stands for winning tickets since these networks match or outperform the accuracy of their dense equivalent. Since our pruning process considers removing layers, it cannot be evaluated on VGG-like architectures (plain and non-residual networks) due to technical and theoretical details. We refer interested readers to Appendix A.1 for additional information.

［#27］
Unless stated otherwise, the term subnetworks indicate networks yielded by the process of removing layers, which is the scope of our work. In this setting, the pruning density $p$ stands for the number of layers removed, e.g., $p=1$ indicates that the pruning removed 1 residual block. (In ResNet32-56 as well as its shallower and deeper variations, one residual block (layer) corresponds to two convolutional layers.)
**Existence of Winning Tickets and Weight Rewinding.** Our first experiment verifies if subnetworks become winning tickets when pruning removes layers. One of the most important facts to a subnetwork becoming a winning ticket is the epoch we rewind its weights (weight rewinding), which means the $\theta_{i}$ that the subnetwork inherits from its dense version.

［#28］
Previous studies on LTH confirmed that inheriting weights from early epochs enables subnetworks successfully become winning tickets [You et al., 2020; Frankle et al., 2021]. From
```

［#29］
Table 1: Predictive ability of the tickets when rewinding their weights to $i$-th training epoch ($\theta_i$) of the dense network (ResNet32). For each criterion, we highlight in bold and underline the rewind epoch that leads to the top-1 and top-2 best results, respectively.

［#30］
|          | $\theta_0$ | $\theta_{25}$ | $\theta_{50}$ | $\theta_{75}$ |
| :------- | :--------- | :------------ | :------------ | :------------ |
| HRank    | (-) 0.01   | (+) $\mathbf{0.21}$ | $\underline{(+) 0.10}$ | (+) 0.12 |
| expectABS| (+) $\mathbf{0.33}$ | (+) 0.13 | (-) 0.48 | (-) 0.62 |
| PLS      | (+) 0.06   | (+) $\mathbf{0.31}$ | $\underline{(+) 0.10}$ | (+) 0.10 |
| Random   | (+) 0.38   | (+) 0.15 | $\underline{(+) 0.26}$ | (-) 0.07 |
| KL       | (+) $\mathbf{0.74}$ | $\underline{(+) 0.46}$ | (-) 0.03 | (-) 0.17 |

［#31］
a practical perspective, resetting the weights to these epochs requires more training iterations to completion (see the last step in Algorithm 1). In this experiment, we evaluate the effect of setting the weights of subnetworks from layer pruning to different epochs. In other words, we study the behavior of subnetworks when they inherit different $\theta_i$.

［#32］
Table 1 summarizes the results. From this table, we highlight the following observations. First, rewinding subnetworks to the same (random) initialization of the dense network, $\theta_0$, enables all subnetworks to become winning tickets (except HRank which exhibited a negligible drop). This means that, rewinding to $\theta_0$, all subnetworks satisfy Equation 1. Second, most subnetworks achieve the lowest accuracy rewinding after 50 epochs. This aligns with the findings of Achille et al. [2019] and You et al. [2020]: rewinding to late epochs, the subnetworks obtain lower accuracy, as they have a short period (epochs) to recover from damage in their structure. Overall, the weight rewinding of LTH when the pruning removes filters and layers shares similar behavior.

［#33］
The previous discussion takes into account the original setup by Frankle et al. [2019] and its variations [Frankle et al., 2020; Renda et al., 2020]. Liu et al. [2019] suggested an alternative to recover the weights for some initialization composing the training trajectory $(\theta_i)$ of the dense network. The authors demonstrated that randomly re-initializing subnetworks allows them to become winning tickets. From the lens of layer pruning, we observe that subnetworks share the same trends. More specifically, on the rewinding by Liu et al. [2019], our subnetworks outperform the dense network by up to 0.85 percentage points.

［#34］
Altogether, the previous results confirm the existence of winning tickets when the pruning process removes entire layers; thus answering our first research question.

［#35］
Picking Winning Tickets at Initialization. So far, our experiments confirm the existence of winning tickets with the weight rewinding strategy, as in the traditional LTH conjecture. Additionally, we observe that relaxing the weight rewinding and re-initializing the subnetworks enable them to become winning tickets. Both settings assume that winning tickets emerge from a well-trained dense network. In this experiment, we remove this constraint and propose to discover winning tickets without any training. In other words, we propose to find winning tickets before training begins. Such a setting eliminates the requirement of heavy computational resources for training the initial (over-parameterized) dense network, an important step towards democratization and green machine learning. As we discussed before, early works have proposed to pick winning tickets at initialization [Lee et al., 2019; Wang et al., 2020; Tanaka et al., 2020; Liu et al., 2022]. However, these works focus on removing weights while we target removing layers, which is a more efficient and hardware-agnostic form of pruning.

［#36］
Table 2: Predictive ability of tickets from ResNet32 when our LTH discovers them at initialization.

［#37］
| Pruning Density | SNIP | GraSP |
| :-------------- | :--- | :---- |
| 1               | (+) 0.08 | (+) 0.40 |
| 2               | (+) 0.00 | (+) 0.72 |
| 3               | (-) 0.50 | (+) 0.34 |
| 4               | (-) 0.53 | (+) 0.41 |
| 5               | (-) 1.00 | (+) 0.12 |

［#38］
One relevant question to find subnetworks from initialization is the choice of the criterion for estimating structure importance. It turns out that most criteria fail to measure importance without any training as the weights change drastically (hence the importance score). In summary, the criteria evaluated in Table 1 are unsuitable¹ to prune at initialization.

［#39］
To face the above issue, we employ two criteria specifically designed for estimating importance at initialization. The first criterion, named SNIP, computes the importance by multiplying the weight by its gradient (at the initial – randomly – configuration); then, it takes the absolute value of the resulting operation [Lee et al., 2019]. The second criterion, termed GraSP, is a variant of SNIP, in which the importance considers the signal of the weight [Wang et al., 2020]. Importantly, SNIP and GraSP are data-driven criteria since both forward data (and labels) through the network to compute the gradient during the importance estimation.

［#40］
Table 2 summarizes the results. From this table, we highlight the following observations. (i) SNIP is an inefficient criterion for discovering very sparse subnetworks (tickets) since its tickets are no longer winning tickets after removing three layers. (ii) All tickets from GraSP win the lottery. Particularly, GraSP provides subnetworks with predictive ability superior to the dense network even on the highest pruning density we consider (5 layers – 66% of layers of ResNet32).

［#41］
In order to confirm the superiority of Grasp over SNIP, we compare the subnetworks yielded by each criterion with the dense network through the lens of representation similarity among models. For this purpose, we employ the method by Kornblith et al. [2019] (named CKA) that measures the representation similarity of two (architecturally equal or distinct) models. The idea is to verify if this similarity correlates with the results in Table 2. On the lowest pruning severity ($p=1$), the difference in CKA similarity between the dense network and the subnetworks from GraSP and SNIP is less than one percentage point, with the subnetwork from GRASP obtaining the higher similarity. However, on the highest pruning

［#42］
¹In fact, we can always employ the random criterion; however, we are interested in a systemic process for selecting winning tickets at initialization.

［#43］
severity ($p=5$), the subnetwork from SNIP exhibited a difference of almost two percentage points. On the one hand, the representation similarity between the subnetworks yielded by SNIP and the dense network decreases as a function of the pruning severity. On the other hand, according to the CKA similarity, the internal representation of subnetworks from GraSP remains (partially) undamaged compared to the dense network. This observation reinforces the positive results of GraSP in Table 2 and indicates that we can effectively extract winning tickets at initialization using GraSP. Unless stated otherwise, we employ GraSP to discover winning tickets at initialization in the subsequent experiments.

［#44］
Comparison with standard (structured) LTH. This experiment compares our winning tickets (layer pruning) with the standard structured LTH (filter pruning). During this evaluation, we extract winning tickets of layer and filter pruning at initialization. Here, a critical aspect is how to produce subnetworks with the same sparsity (in terms of filters) when the pruning removes different structures. It turns out that when it eliminates layers, the sparsity becomes inflexible. For example, by removing 1 and 2 layers, the number of remaining filters is 1200 and 1168 and we cannot obtain a filter sparsity among these values (i.e., 1190). For fairness of comparison, we adopt the following procedure. First, we remove layers from a dense network and calculate the number of filters in the obtained subnetwork (i.e., the number of kept filters). Then, to produce comparable subnetworks removing only filters, we run the pruning process forcing it to eliminate the closest number of filters of subnetworks from layer pruning. Thereby, subnetworks yielded from a dense network without layers and filters are as close as possible in terms of filter sparsity. We highlight that the opposite process – first removing filters and then layers – impairs a comparable sparsity.

［#45］
Table 3 shows the results in the lowest sparsity regime (one layer removed – $p=1$). On both SNIP and GraSP criteria, our winning tickets at initialization yielded early winning tickets with better improvements. In particular, the subnetworks from the standard structured LTH become winning tickets only if we relax Equation 1 by increasing the value $\xi$.

［#46］
Previous observations revealed the difficulty of extracting winning tickets at initialization and confirmed that it is no better than the standard *training and prune* paradigm [Frankle et al., 2021; de Jorge et al., 2021]; thus, corroborating the poor results of filter pruning in Table 3. Our winning tickets (layer pruning) pose a different perspective for such an issue: filters cannot be the most effective structure in structured LTH. Technically speaking, removing a layer consists of eliminating a group of filters (in particular, all filters) from a specific location, see Appendix A.1. Hence, it sounds like a counter-intuitive behavior that our winning tickets obtain accuracy superior to the standard LTH. Our results, however, are aligned with a body of studies that confirm the benefits of removing layers over other structures [Zhang and Liu, 2022; Zhou et al., 2022; Han et al., 2022]. To reinforce these claims, we measure the representation similarity between the models of Table 3 and the dense network, similarly as we performed in previous experiments. On this metric, our subnetworks exhibited a higher similarity than subnetworks from filter pruning, regardless of the importance criterion (SNIP or GraSP). Such values suggest that our winning tickets kept an internal representation more similar to the original dense network than the ones found by the standard structured LTH, which supports their highest accuracies in Table 3.

［#47］
Overall, we believe the findings above open a new direction for research in LTH: the influence of the structure taken into account during the pruning process.

［#48］
Computational Cost of Winning Tickets at Initialization. The aforementioned discussion confirms the existence of winning tickets at initialization when pruning removes layers (recall that our subnetworks come from layer pruning – see Algorithms 1 and 2). From a practical perspective, identifying winning tickets at initialization means that we can speed up the training stage and save computation in terms of different performance metrics, as we can replace the learning of a dense network with its sparse version. Such achievements become possible since we discover subnetworks that (i) are more efficient than the heavy and over-parametrized dense network; (ii) match the predictive ability of their dense equivalent (i.e., the subnetworks are winning tickets) and (iii) spend no additional cost since we discover them before any training epoch.

［#49］
Table 4 reports the performance gains of our winning tickets at initialization on standard pruning metrics such as floating point operations (FLOPs), memory consumption and training speed up. Following a modern trend [Lacoste et al., 2019; Strubell et al., 2019; You et al., 2020], we also report the $\text{CO}_2$ emission during the training stage. According to Table 4, it is evident the advantages of discovering winning tickets at initialization. For example, our winning tickets speed up the training time of ResNet32 and ResNet56 by up to $1.57\times$ and $2.03\times$, meaning that we can significantly reduce the costs involved during the training step such as energy consumption and $\text{CO}_2$ emission. Importantly, all these improvements come at no additional cost.

［#50］
Previous attempts have proposed to find winning tickets early in the training phase [You et al., 2020; Chen et al., 2022]. Our work differs from these works, as we discover winning tickets prior to any training (roughly speaking, our subnetworks are the earliest as possible), which is a more efficient strategy since the gains in Table 4 occur before the training starts.

［#51］
Robustness of Winning Tickets at Initialization. There is a growing body of work evaluating pruning methods in adversarial images since the technique emerges as a powerful and efficient defense mechanism against adversarial attacks and out-of-distribution examples [Diffenderfer et al., 2021;

［#52］
Table 3: Comparison between tickets from the standard structured LTH (filter pruning) and our LTH (layer pruning). On both criteria for pruning at initialization, our subnetworks outperform their dense (ResNet32) counterpart. In other words, our tickets become winning tickets (they win the lottery). However, tickets from the standard structured LTH hardly become winning tickets.

［#53］
<table>
  <thead>
    <tr>
      <th></th>
      <th>SNIP</th>
      <th>GraSP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Filter (Standard LTH)</td>
      <td>(-) 0.79</td>
      <td>(-) 0.11</td>
    </tr>
    <tr>
      <td>Layers (Ours LTH)</td>
      <td>(+) 0.08</td>
      <td>(+) 0.40</td>
    </tr>
  </tbody>
</table>

［#54］
Table 4: Performance gains when the pruning discovers winning tickets (layer pruning) at initialization. The values indicate the improvements (in percentage – the higher, the better) of tickets regarding the dense network.

［#55］
<table>
<thead>
<tr>
<th></th>
<th>Pruning Density (p)</th>
<th>FLOPs</th>
<th>Memory Consumption</th>
<th>CO₂ Emission</th>
<th>Training Speed up</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Winning Tickets at Initialization (ResNet32)</td>
<td>1</td>
<td>6.82</td>
<td>10.13</td>
<td>11.11</td>
<td>1.10</td>
</tr>
<tr>
<td>3</td>
<td>20.47</td>
<td>25.29</td>
<td>25.92</td>
<td>1.35</td>
</tr>
<tr>
<td>4</td>
<td>34.13</td>
<td>30.11</td>
<td>33.33</td>
<td>1.57</td>
</tr>
<tr>
<td rowspan="4">Winning Tickets at Initialization (ResNet56)</td>
<td>3</td>
<td>11.25</td>
<td>15.15</td>
<td>13.33</td>
<td>1.16</td>
</tr>
<tr>
<td>4</td>
<td>15.00</td>
<td>19.42</td>
<td>20.00</td>
<td>1.24</td>
</tr>
<tr>
<td>5</td>
<td>18.76</td>
<td>23.69</td>
<td>24.44</td>
<td>1.34</td>
</tr>
<tr>
<td>12</td>
<td>45.00</td>
<td>45.82</td>
<td>51.11</td>
<td>2.03</td>
</tr>
</tbody>
</table>

［#56］
Liu et al., 2022; T et al., 2022; Chen et al., 2022]. In this experiment, we demonstrate if there exists some sensibility of our winning tickets to these scenarios. For this purpose, we employ the CIFAR-C [Hendrycks and Dietterich, 2019] and CIFAR-10.2 [Lu et al., 2020] datasets. It is important to mention that our training stage employs the standard clean training on CIFAR-10, which means that we do not employ any defense mechanism (i.e., adversarial training).

［#57］
Table 5 (third column) shows the improvements in robustness of winning tickets at initialization (the ones in Table 4) over the deep network used to discover them. According to the results, our winning tickets effectively increase robustness to adversarial images. Specifically, our winning tickets outperformed their respective dense network by up to 1.61 and 3.73 pp for ResNet32 and ResNet56, respectively. Intriguingly, for each architecture, one winning ticket exhibited low robustness, in which the highest decline is only 0.16 pp.

［#58］
Regarding the out-of-distribution generalization (last column in Table 5), our winning ticket increase the generalization of ResNet56 in up to 1.75. On the ResNet32, however, we observe that the results become negative as a function of the pruning severity; thus raising the question of whether our tickets generalize well in out-of-distribution examples when extracted from shallow architectures. The analysis of such behavior is an interesting direction for future work.

［#59］
For some settings (5 out of 14 configurations), our winning tickets underperform their dense equivalent. We did not observe any pattern in these low-performance tickets and left an in-depth investigation of this issue for future research. Overall, the achievements in robustness against adversarial images and out-of-distribution examples suggest that the benefits of our winning tickets are beyond reducing computational performance, which aligns with findings of previous works: removing structures from networks increases robustness and out-of-distribution generation [Diffenderfer et al., 2021; Chen et al., 2022; T et al., 2022].

## 5 Conclusions
［#60］
In this work, we explore the concepts of the Lottery Ticket Hypothesis (LTH) and pruning at initialization from the lens of layer pruning. From this form of pruning, we bring both theoretical and practical benefits. First, we verify the behavior of LTH when the pruning yields subnetworks (tickets) by removing layers from a dense network, in which we confirm that the tickets successfully become winning tickets. Built upon this, we propose to systematically discover winning tickets at initialization, which means identifying sparse subnetworks (from layer pruning) without the need of training a dense network. According to extensive experiments, our winning tickets at initialization speed up the learning phase by up to $2\times$, reducing the carbon emission by up to $51\%$. Such achievements come at no additional price, as we extract winning tickets before training the dense network. Additionally, our winning tickets become more robust against adversarial images and generalize better in out-of-distribution examples than their dense counterpart. We hope these benefits attract more research on LTH from the lens of layer pruning.

［#61］
Limitations. Our strategy that discovers winning tickets at initialization employs well-established criteria for assigning the importance of each layer before training (i.e., using the randomly initialized parameters to guide the pruning algorithm). Hence, it inherits the limitation of being unable to run on an iterative pruning scheme, which has been demonstrated to be more effective than one-shot pruning. Due to theoretical and technical details involving layer pruning, our structured LTH is confined to the realm of residual-like architectures. Thus, our LTH is infeasible to plain networks (e.g., VGG-16), as well as other works of this category of pruning. Fortunately, most modern architectures employ some type of residual connection.

［#62］
Table 5: Robustness against adversarial and out-of-distribution examples of our winning tickets at initialization. On both CIFAR-C and CIFAR-10.2 datasets, we report the improvement (in percentage points – the higher, the better) of subnetworks regarding their dense counterpart.

［#63］
<table>
<thead>
<tr>
<th></th>
<th>p</th>
<th>CIFAR-C</th>
<th>CIFAR-10.2</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Winning Tickets at Initialization (ResNet32)</td>
<td>1</td>
<td>(+) 1.13</td>
<td>(+) 0.00</td>
</tr>
<tr>
<td>3</td>
<td>(-) 0.05</td>
<td>(-) 1.15</td>
</tr>
<tr>
<td>4</td>
<td>(+) 1.61</td>
<td>(-) 0.65</td>
</tr>
<tr>
<td rowspan="4">Winning Tickets at Initialization (ResNet56)</td>
<td>3</td>
<td>(+) 0.59</td>
<td>(+) 0.70</td>
</tr>
<tr>
<td>4</td>
<td>(+) 3.73</td>
<td>(+) 1.75</td>
</tr>
<tr>
<td>5</td>
<td>(-) 0.16</td>
<td>(+) 0.70</td>
</tr>
<tr>
<td>12</td>
<td>(+) 1.98</td>
<td>(-) 1.95</td>
</tr>
</tbody>
</table>

［#64］
[Li *et al.*, 2021] Zhengang Li, Geng Yuan, Wei Niu, Pu Zhao, Yanyu Li, Yuxuan Cai, Xuan Shen, Zheng Zhan, Zhenglun Kong, Qing Jin, Zhiyu Chen, Sijia Liu, Kaiyuan Yang, Bin Ren, Yanzhi Wang, and Xue Lin. NPAS: A compiler-aware framework of unified network pruning and architecture search for beyond real-time mobile acceleration. In *Conference on Computer Vision and Pattern Recognition* (CVPR), 2021.

［#65］
[Lin *et al.*, 2020] Mingbao Lin, Rongrong Ji, Yan Wang, Yichen Zhang, Baochang Zhang, Yonghong Tian, and Ling Shao. Hrank: Filter pruning using high-rank feature map. In *Conference on Computer Vision and Pattern Recognition* (CVPR), 2020.

［#66］
[Liu *et al.*, 2019] Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In *International Conference on Learning Representations* (ICLR), 2019.

［#67］
[Liu *et al.*, 2021] Ning Liu, Geng Yuan, Zhengping Che, Xuan Shen, Xiaolong Ma, Qing Jin, Jian Ren, Jian Tang, Sijia Liu, and Yanzhi Wang. Lottery ticket preserves weight correlation: Is it desirable or not? In *International Conference on Machine Learning* (ICML), 2021.

［#68］
[Liu *et al.*, 2022] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Li Shen, Decebal Constantin Mocanu, Zhangyang Wang, and Mykola Pechenizkiy. The unreasonable effectiveness of random pruning: Return of the most naive baseline for sparse training. In *International Conference on Learning Representations* (ICLR), 2022.

［#69］
[Lu *et al.*, 2020] Shangyun Lu, Bradley Nott, Aaron Olson, Alberto Todeschini, Hossein Vahabi, Yair Carmon, and Ludwig Schmidt. Harder or different? a closer look at distribution shift in dataset reproduction. In *International Conference on Machine Learning* (ICML) Workshop on Uncertainty and Robustness in Deep Learning, 2020.

［#70］
[Luo and Wu, 2020] Jian-Hao Luo and Jianxin Wu. Neural network pruning with residual-connections and limited data. In *Conference on Computer Vision and Pattern Recognition* (CVPR), 2020.

［#71］
[Niu *et al.*, 2020] Wei Niu, Xiaolong Ma, Sheng Lin, Shihao Wang, Xuehai Qian, Xue Lin, Yanzhi Wang, and Bin Ren. Patdnn: Achieving real-time DNN execution on mobile devices with pattern-based weight pruning. In *Architectural Support for Programming Languages and Operating Systems* (ASPLOS), 2020.

［#72］
[Paul *et al.*, 2022] Mansheej Paul, Brett W. Larsen, Surya Ganguli, Jonathan Frankle, and Gintare Karolina Dziugaite. Lottery tickets on a data diet: Finding initializations with sparse trainable networks. In *Neural Information Processing Systems* (NeurIPS), 2022.

［#73］
[Prasanna *et al.*, 2020] Sai Prasanna, Anna Rogers, and Anna Rumshisky. When BERT plays the lottery, all tickets are winning. In *Conference on Empirical Methods in Natural Language Processing* (EMNLP), 2020.

［#74］
[Renda *et al.*, 2020] Alex Renda, Jonathan Frankle, and Michael Carbin. Comparing rewinding and fine-tuning in neural network pruning. In *International Conference on Learning Representations* (ICLR), 2020.

［#75］
[Strubell *et al.*, 2019] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP. In *ACL*, 2019.

［#76］
[T *et al.*, 2022] Mukund Varma T, Xuxi Chen, Zhenyu Zhang, Tianlong Chen, Subhashini Venugopalan, and Zhangyang Wang. Sparse winning tickets are data-efficient image recognizers. In *Neural Information Processing Systems* (NeurIPS), 2022.

［#77］
[Tan and Motani, 2020] Chong Min John Tan and Mehul Motani. Dropnet: Reducing neural network complexity via iterative pruning. In *International Conference on Machine Learning* (ICML). 2020.

［#78］
[Tanaka *et al.*, 2020] Hidenori Tanaka, Daniel Kunin, Daniel L. Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. In *Neural Information Processing Systems* (NeurIPS), 2020.

［#79］
[Vaswani *et al.*, 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Neural Information Processing Systems* (NeurIPS), 2017.

［#80］
[Veit and Belongie, 2020] Andreas Veit and Serge J. Belongie. Convolutional networks with adaptive inference graphs. *International Journal of Computer Vision* (IJCV), 128:730–741, 2020.

［#81］
[Veit *et al.*, 2016] Andreas Veit, Michael J. Wilber, and Serge J. Belongie. Residual networks behave like ensembles of relatively shallow networks. In *Neural Information Processing Systems* (NeurIPS), 2016.

［#82］
[Wang *et al.*, 2020] Chaoqi Wang, Guodong Zhang, and Roger B. Grosse. Picking winning tickets before training by preserving gradient flow. In *International Conference on Learning Representations* (ICLR), 2020.

［#83］
[You *et al.*, 2020] Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Yingyan Lin, Zhangyang Wang, and Richard G. Baraniuk. Drawing early-bird tickets: Toward more efficient training of deep networks. In *International Conference on Learning Representations* (ICLR), 2020.

［#84］
[Zhang and Liu, 2022] Ke Zhang and Guangzhe Liu. Layer pruning for obtaining shallower resnets. *IEEE Signal Processing Letters*, 2022.

［#85］
[Zhou *et al.*, 2021] Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, and Hongsheng Li. Learning N: M fine-grained structured sparse neural networks from scratch. In *International Conference on Learning Representations* (ICLR), 2021.

［#86］
[Zhou *et al.*, 2022] Yao Zhou, Gary G. Yen, and Zhang Yi. Evolutionary shallowing deep neural networks at block levels. *IEEE Transactions on Neural Networks and Learning Systems*, 2022.

## A Appendix
### A.1 Layer Pruning
［#87］
**Theoretical Issues.** Assume a network $\mathcal{F}$ of $L$ layers as a set of $L$ transformations $f_i(.)$. For the sake of simplicity, $f_i$ consists of a series of convolution, batch normalization, and activation operations. In this definition, we obtain the network output $(y)$ by forwarding the input data through the sequential layers $f$, where the input of layer $i$ is the output of the previous layer $i-1$; therefore, $y = f(x) \ f_L(...f_2(f_1(.)))$. This composes the idea behind plain networks (i.e., VGG).

［#88］
In residual-like networks, the output of layer $i$, $y_i$, consists of the transformation $f_i$ plus the input it receives $y_{i-1}$ (see Figure 4). Formally, we can define the output of the $i$-th layer as

［#88］
$$
y_i = f_i(y-1) + y_{i-1}. \tag{2}
$$

［#89］
Equation 2 composes a residual module, where the rightmost part is named *identity-mapping shortcut* (or identity for short). It is important to observe in Equation 2 that if we disable $f(i)$ (a layer) then $y_i = y_{i-1}$.

［#90］
Veit et al. [2016] showed that the identity enables the information to take different paths in the network, in the sense that, we can disable some $f_i$ without degrading (or with negligible damage) the expected representation of the subsequent layers (i.e., $f_{i+1}$). In other words, some layers $f_i$ do not depend strongly on each other; hence, we can eliminate them. For example, in Figure 4, we could remove layer $i$ with no loss in the predictive ability of the network. On the other hand, due to the absence of identity, plain networks meet collapse in the representation if we remove only one of their layers. We refer interested readers to Figure 3 of the study by Veit et al. [2016] for a comparison of accuracy drop between residual and plain networks.

［#91］
**Technical Issues.** We can disable layer $i$ by setting its weighs to zero (the widely employed zeroed-out scheme). This way, the output of layer $i-1$ is directly connected to layer $i+1$ (see Figure 4 middle). However, such a process does not achieve performance gains without specialized frameworks or hardware for sparse computation. Instead of zeroing weights, we can perform the following process. After identifying which layers remove (i.e., a victim), we create a new network without layer $i$ and transfer the weights of the kept layers to the new network. For example, if we have a network with $L$ layers and want to remove $k$ layers, then, we create a novel network with $L - k$ layers. In summary, the pruned network (bottom in Figure 2) inherits the weights of the kept layers of the original network (top in Figure 2).

［#92］
We highlight that the pruning cannot remove some layers due to incompatible dimensions of (input/output) tensors. Such an incompatibility comes from the spatial resolution layer (downsampling layers). More specifically, we cannot remove layers before and after the downsampling layers. Importantly, filter pruning also suffers from this issue.

［#93］
![](./images/867756402284691503_2.jpg)

［#94］
Figure 2: Overall process to remove layers (residual models) from a residual network. After identifying a victim layer (dashed rectangle), we create a novel network (bottom) without it. Finally, we transfer the weights (red arrows) of the kept layers from the original unpruned network (top) to the new network.

### A.2 $\ell_1$-norm
［#95］
Figure 3 illustrates the $\ell_1$-norm scores of layers (the ones that the pruning could remove) of ResNet32. From this figure, we see that the magnitude of scores of layers correlates with the stage (groups of layers operating on the same resolution of feature maps) to which they belong. Since our pruning strategy takes into account all layers (i.e., all scores) at once, this criterion is infeasible. Specifically, there is a bias to layers of early stages (i.e., they will always be selected as victims).

［#96］
![](./images/867756402284691503_3.jpg)

［#97］
Figure 3: $\ell_1$-norm score of layers of ResNet32. Layers within a stage operate on the same input/output spatial resolution (i.e., the size of the feature map – values in parentheses).

［#98］
![](./images/867756402284691503_4.jpg)

［#99］
Figure 4: Architecture of a residual-like network. The rationale behind this architecture is that the output of a layer takes into account the transformation performed by it ($f$) plus ($\oplus$) the input ($y$) it receives. Due to this essence, when we disable layer $i$ (its transformation – dashed lines), the output (representation) of layer $i-1$ is propagated to layer $i+1$, which means that the output $y_i$ belongs $y_{i-1}$. For the sake of simplicity, we omit the batch normalization and activation layers.