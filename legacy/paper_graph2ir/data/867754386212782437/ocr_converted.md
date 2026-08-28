# Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#1］
Tianlong Chen $^1$ Xuxi Chen $^1$ Xiaolong Ma $^2$ Yanzhi Wang $^2$ Zhangyang Wang $^1$

## Abstract
［#2］
The lottery ticket hypothesis (LTH) has shown that dense models contain highly sparse subnetworks (i.e., winning tickets) that can be trained in isolation to match full accuracy. Despite many exciting efforts being made, there is one "commonsense" rarely challenged: a winning ticket is found by iterative magnitude pruning (IMP) and hence the resultant pruned subnetworks have only unstructured sparsity. That gap limits the appeal of winning tickets in practice, since the highly irregular sparse patterns are challenging to accelerate on hardware. Meanwhile, directly substituting structured pruning for unstructured pruning in IMP damages performance more severely and is usually unable to locate winning tickets. In this paper, we demonstrate the first positive result that a structurally sparse winning ticket can be effectively found in general. The core idea is to append "post-processing techniques" after each round of (unstructured) IMP, to enforce the formation of structural sparsity. Specifically, we first "re-fill" pruned elements back in some channels deemed to be important, and then "re-group" non-zero elements to create flexible group-wise structural patterns. Both our identified channel- and group-wise structural subnetworks win the lottery, with substantial inference speedups readily supported by existing hardware. Extensive experiments, conducted on diverse datasets across multiple network backbones, consistently validate our proposal, showing that the hardware acceleration roadblock of LTH is now removed. Specifically, the structural winning tickets obtain up to $\{64.93\%, 64.84\%, 60.23\%\}$ running time savings at $\{36\% \sim 80\%, 74\%, 58\%\}$ sparsity on {CIFAR, Tiny-ImageNet, ImageNet}, while maintaining comparable accuracy. Code is at https://github.com/VITA-Group/Structure-LTH.

［#3］
![](./images/867754386212782437_1.jpg)

［#4］
Figure 1. Achieved test accuracy over different sparsity levels of diverse unstructured and structural subnetworks. Sparse models from classical channel-wise structural pruning algorithms (He et al., 2017; Liu et al., 2017; Bartoldson et al., 2019; Molchanov et al., 2019) can not match the full accuracy of the dense model.

## 1. Introduction
［#5］
Recently, the machine learning research community has devoted considerable efforts and financial outlay to scaling deep neural networks (DNNs) to enormous sizes (175 billion parameter-counts in GPT-3 (Brown et al., 2020)). Although such overparameterization simplifies the training of DNNs and dramatically improves their generalization (Bartlett et al., 2021; Du et al., 2018; Kaplan et al., 2020), it may severely obstruct the practical usage on resource-limited platforms like mobile devices, due to its large memory footprint and inference time (Hoefler et al., 2021). Pruning is one of the effective remedies that can be dated back to LeCun et al. (1990): it can eliminate substantial redundant model parameters and boost the computational and storage efficiency of DNNs.

［#6］
Such benefits drive numerous interests in designing model pruning algorithms (Han et al., 2015a;b; Ren et al., 2018; He et al., 2017; Liu et al., 2017). Among this huge family, an emerging representative studies the prospect of training sparse subnetworks in lieu of the full dense models without impacting performance (Frankle & Carbin, 2019; Chen et al., 2020b). For instance, Frankle & Carbin (2019) demonstrates that dense models contain sparse, matching subnetworks (Frankle et al., 2020a) (a.k.a. winning tickets) capable of training in isolation from the original initialization to

---
［#1］
$^1$University of Texas at Austin $^2$Northeastern University. Correspondence to: Zhangyang Wang <atlaswang@utexas.edu>.

［#7］
Proceedings of the $39^{th}$ International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

［#8］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#8］
match or even surpass the full accuracy. This phenomenon is referred to as the lottery tickets hypothesis (LTH), which indicates several impressive observations: (i) usually extreme sparsity levels (e.g., 90%, 95%) can be achieved without sacrificing the test accuracy; (ii) the located winning ticket maintains undamaged expressive power as its dense counterpart, and can be easily trained from scratch or early-epoch weights (Renda et al., 2020; Frankle et al., 2020a) to recover the full performance. These advances are positive signs of the substantial potential of sparse DNNs.

［#9］
However, almost all LTH literature investigates unstructured sparsity only. In practical scenarios, it brings little hardware efficiency benefits due to the poor data locality and low parallelism (He et al., 2017; Mao et al., 2017; Wen et al., 2016) caused by highly irregular sparse patterns. Meanwhile, most of the accelerators are optimized for dense matrix operations (Han et al., 2016), which means there is limited speedup for unstructured pruned subnetworks even if the sparsity level exceeds 95% (Wen et al., 2016). Structural pruning (He et al., 2017; Liu et al., 2017) as an alternative to exploring sparse subnetworks, removes the entire filter or channel in DNNs to gain more computational efficiency at the cost of (more) accuracy degradation. As shown in Fig. 1, traditional channel-wise structural pruning approaches (He et al., 2017; Bartoldson et al., 2019; Molchanov et al., 2019) quickly degrade performance and cannot lead to winning tickets, which was also echoed in You et al. (2020).

［#10］
In our paper, we present the first study into the structural lottery tickets, which explores hardware-friendly structural sparsity (including channel-wise and group-wise patterns) in order to find lottery tickets. Specifically, we start from unstructured sparse subnetworks, and then adopt proposed refilling techniques to create channel-wise structural sparsity by growing back the pruned elements within the most important channels and abandoning the rest. Our results (Section 4) show such refined channel-wise structural subnetworks win the lottery at a moderate sparsity level with $\sim50\%$ running time savings on an Nvidia 2080 TI GPU. In order to push the compression ratio higher, we introduce a regrouping algorithm based on hypergraph partitioning (Rumi et al., 2020) to establish group-wise structural patterns which are more amenable to pruning due to the shape flexibility of grouped dense blocks. These group-wise structural winning tickets achieve $\sim60\%$ running time savings at $50\% \sim 80\%$ sparsity without any performance degradation compared to the dense models.

［#11］
Note that this paper focuses on general structural sparse patterns capable of acceleration, including conventional channel-wise sparsity and other fine-grained structural sparsity. The latter actually becomes prevailing recently since it achieves superior performance and maintains satisfied speedup, sparking great interest in industries such as NVIDIA (N:M) (Zhou et al., 2021) and Google (Block-wise) (Shangguan et al., 2019). Meanwhile, unlike Zhou et al. (2021), our group-wise sparse patterns do NOT need any specific hardware accelerators and are generally applicable to common GPU devices. Lastly, although we mainly investigate inference efficiency, our proposals can also enable efficient training in transfer learning paradigms as demonstrated in Appendix A2. Our main contributions lie in the following aspects:

［#12］
- To our best knowledge, we are the first to demonstrate the existence of structurally sparse winning tickets at non-trivial sparsity levels (i.e., $>30\%$), and with both channel-wise and group-wise sparse patterns.
- We propose the refilling technique and introduce the regrouping algorithm to form channel-wise and group-wise structural sparsity. Such refined structural subnetworks match the trainability and expressiveness of dense networks, while enabling the inference speedup on practical hardware platforms like GPU machines (general and not tied to particular hardware).
- Extensive experiments validate our proposal on diverse datasets (i.e., CIFAR-10/100, Tiny-ImageNet, and ImageNet) across multiple network architectures, including ResNets, VGG, and MobileNet. Specifically, our structural winning tickets achieve $53.75\% \sim 64.93\%$ GPU running time savings at $45\% \sim 80\%$ channel and group-wise sparsity.

## 2. Related Work

［#13］
Pruning. Network pruning is a technique that aims at eliminating the unnecessary model parameters (Blalock et al., 2020), which can effectively shrink models for the deployment on resource-constrained devices (LeCun et al., 1990; Hanson & Pratt, 1988). Pruning algorithms are roughly categorized into two groups: (1) unstructured pruning (LeCun et al., 1990; Han et al., 2015a;b; Ren et al., 2018; Zhang et al., 2018) with irregular sparse patterns; (2) structural pruning (He et al., 2017; Liu et al., 2017; Li et al., 2016; Hu et al., 2016; Wen et al., 2016; Hong et al., 2018) with structural sparse patterns such as layer-wise, channel-wise, block-wise, column-wise, etc..

［#14］
Within the group of unstructured pruning methods, Han et al. (2015a;b) remove insignificant connections of models in the post-training stage, with respect to certain heuristics like weight/gradient magnitudes; during training sparsification is also another popular trend for pruning by leveraging $\ell_0$ regularization (Louizos et al., 2017) or alternating direction method of multipliers (ADMM) (Ren et al., 2018; Zhang et al., 2018). Recently, several pruning-at-initialization methods (Wang et al., 2020; Lee et al., 2019b; Tanaka et al., 2020) are proposed to identify critical unstructured connections for gradient-flow preserving, without

［#15］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#16］
![](./images/867754386212782437_2.jpg)

［#17］
Figure 2. Overview of our proposals including refilling, refilling+, and regrouping, which turn unstructured sparse mask into channel-wise and group-wise structured sparse masks.

［#18］
any training. Although the unstructured sparse model has superior performance, it usually suffers from poor data lo-cality and low parallelism (He et al., 2017; Mao et al., 2017; Wen et al., 2016), which make it hard to be accelerated in real-world hardware platforms.

［#19］
On the contrary, structural pruning is more hardware-friendly at the cost of notable accuracy loss when the com-pression ratio increases. He et al. (2017); Liu et al. (2017) slim the network channels via $\ell_1$ regularization, and Bar-toldson et al. (2019) selects important channels according to heuristics of feature maps. To combine the benefits of structural and unstructured pruning, hybrid pruning strate-gies have been introduced to pursue more general struc-tural spares patterns which are also capable of accelera-tion. For example, convolution kernels with half regular sparsity (Chen et al., 2018) or pattern-based structural spar-sity (Ma et al., 2020) or vector-wise (Zhu et al., 2019) and group-wise (Rumi et al., 2020) regular sparsity.

［#20］
The lottery tickets hypothesis (LTH). The lottery ticket hypothesis (LTH) (Frankle & Carbin, 2019) conjectures that there exists a sparse subnetwork called winning ticket within a dense network, whose performance can match with the dense network when training from the same initialization. With the assistance of weight rewinding techniques (Renda et al., 2020; Frankle et al., 2020a), the original LTH can be scaled up to larger networks and datasets. The existence of winning tickets are broadly verified under diverse contexts, such as image classification (Frankle & Carbin, 2019; Zhang et al., 2021; Chen et al., 2020a; Ma et al., 2021a; Gan et al., 2021; Chen et al., 2021c), natural language processing (Gale et al., 2019; Chen et al., 2020b), generative adversarial net-works (Chen et al., 2021d;a), graph neural networks (Chen et al., 2021b), and reinforcement learning (Yu et al., 2020).

［#21］
However, all of the above LTH literature only locate un-structured sparse winning tickets, which can hardly bring hardware efficiency boost to real-world applications.

［#22］
As the most related work, You et al. (2020) finds structural winning tickets at only low sparsity levels around 30% in a few cases. It again reveals the complication and difficulty of identifying computation-friendly sparse patterns. Another concurrent work (Alabdulmohsin et al., 2021) investigates a generalized LTH with weight space factorization, which is orthogonal to our work.

［#23］
Sparse convolutional neural network acceleration on GPU. Previous works have explored the acceleration of sparse convolution operations in two different directions. One direction is to design efficient implementation of un-structured pruned networks for improved data locality and utilization of hardware (Chen, 2018; Park et al., 2016). For example, Dong et al. (2019) proposes "Acorns" to accel-erate the sparse computations of convolution kernels with an input sparsity. Peng et al. (2017) has proposed a matrix splitting algorithm for efficient inference of convolutional neural networks (CNN). Nvidia's cuSPARSE¹ library con-tains various efficient sparse matrix computation algorithms like SpMM on GPUs, drawing great attention to efficient sci-entific computing. Furthermore, advanced approaches are developed based on SpMM, such as Adaptive Sparse Tiling (ASpT) (Hong et al., 2019). ASpT significantly improves the data usage of SpMM and achieves the current state-of-the-art performance among SpMM implementation vari-ants. Another direction focuses on more hardware-friendly pruning methods (Chen et al., 2018; Ma et al., 2020; Niu et al., 2020). During the model pruning, these works aim

［#24］
¹https://docs.nvidia.com/cuda/archive/10.2/cusparse/index.html

# Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#25］
<table>
  <caption>Table 1. Implementation details which follow the standard settings in Ma et al. (2021b).</caption>
  <thead>
    <tr>
      <th rowspan="2">Settings</th>
      <th colspan="4">CIFAR-10</th>
      <th colspan="4">CIFAR-100</th>
      <th>Tiny-ImageNet</th>
      <th>ImageNet</th>
    </tr>
    <tr>
      <td>WRN-32-2</td>
      <td>RN-18</td>
      <td>MBNet-v1</td>
      <td>VGG-16</td>
      <td>WRN-32-2</td>
      <td>RN-18</td>
      <td>MBNet-v1</td>
      <td>VGG-16</td>
      <td>RN-50</td>
      <td>RN-50</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Batch Size</td>
      <td>128</td>
      <td>128</td>
      <td>128</td>
      <td>128</td>
      <td>-</td>
      <td>-</td>
      <td>64</td>
      <td>-</td>
      <td>32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Weight Decay</td>
      <td>$1 \times 10^{-4}$</td>
      <td>$1 \times 10^{-4}$</td>
      <td>$1 \times 10^{-4}$</td>
      <td>$2 \times 10^{-4}$</td>
      <td>$2 \times 10^{-4}$</td>
      <td>$2 \times 10^{-4}$</td>
      <td>$2 \times 10^{-4}$</td>
      <td>$5 \times 10^{-4}$</td>
      <td>$5 \times 10^{-4}$</td>
      <td>$1 \times 10^{-4}$</td>
    </tr>
    <tr>
      <td>Learning Rate</td>
      <td colspan="8">0.1;$\times$0.1 at 80,120 epoch of total 160 epochs</td>
      <td colspan="2">0.1;$\times$0.1 at 30,60 epoch of total 95 epochs</td>
    </tr>
    <tr>
      <td>Optimizer</td>
      <td colspan="10">SGD (Ruder, 2016) with a momentum of 0.9</td>
    </tr>
    <tr>
      <td>Model Size</td>
      <td>1.86 M</td>
      <td>11.22 M</td>
      <td>3.21 M</td>
      <td>14.72 M</td>
      <td>1.86 M</td>
      <td>11.22 M</td>
      <td>3.21 M</td>
      <td>14.72 M</td>
      <td>25.56 M</td>
      <td>25.56 M</td>
    </tr>
  </tbody>
</table>

［#26］
to maintain certain regular sparse patterns, which benefit the hardware processing/computing of corresponding sparse matrices. However, Chen et al. (2018) achieves unsatisfactory compression ratio, while the pruning methods used in Ma et al. (2020) and Niu et al. (2020) require dedicated compiler optimization to accelerate network execution.

## 3. Methodology
### 3.1. Notations and Preliminaries

［#27］
Sparse subnetworks and pruning methods. In this paper, we mainly follow the routine notations in (Frankle & Carbin, 2019; Renda et al., 2020). For a network $f(x; \theta)$ with input samples $x$ and model parameters $\theta$, a sparse subnetwork is a network $f(x; m \odot \theta)$ with a binary pruning mask $m \in \{0, 1\}^{|\theta|}$, where $\odot$ is the element-wise product. In other words, it is a copy of dense network $f(x; \theta)$ with some weights fixed to 0. If the non-fixed remaining weights are distributed irregularly, we call it unstructured sparse patterns (e.g., the left of Figure 2); if they are clustered into channels or groups, we name it structural sparse patterns (e.g., the right of Figure 2).

［#28］
To obtain the desired sparse subnetworks, we consider and benchmark multiple classical pruning algorithms: (1) random pruning (RP) which usually works as a necessary baseline for the sanctity check (Frankle & Carbin, 2019); (2) one-shot magnitude pruning (OMP) by eliminating a part of model parameters with the globally smallest magnitudes (Han et al., 2015a); (3) the lottery ticket hypothesis (Frankle & Carbin, 2019) with iterative weight magnitude pruning (LTH-IMP or IMP for simplicity) (Han et al., 2015a). As adopted in LTH literature (Frankle & Carbin, 2019), we identify the sparse lottery tickets by iteratively removing the 20% of remaining weight with the globally smallest magnitudes, and rewinding model weights to the original random initialization (Frankle & Carbin, 2019) or early training epochs (Frankle et al., 2020b; Chen et al., 2020a). In this paper, the model weights are rewound to the eighth epoch (i.e., the 5% of the entire training process) for all CIFAR, Tiny-ImageNet, and ImageNet experiments. (4) pruning at initialization mechanisms. We choose several representative approaches such as SNIP (Lee et al., 2019a), GraSP (Wang et al., 2020), and SynFlow (Tanaka et al., 2020), which explore sparse patterns at random initialization with some gradient flow-based criterion. (5) Alternating Direction Method of Multipliers (ADMM) for punning. It is a well-known optimization-based pruning method (Niu et al., 2020; Zhang et al., 2018), which can obtain superior compression ratios with little performance degradation for deep neural networks. Note that all pruning approaches are mainly conducted over networks without counting their classification heads (Frankle & Carbin, 2019).

［#29］
Structural winning tickets. We begin by extending the original lottery tickets hypothesis to the context of structural sparse patterns. A subnetwork $f(x; m \odot \theta)$ is a structural winning ticket for an algorithm $\mathcal{A}_t^{\mathcal{T}}$ if it satisfies: ① training subnetworks $f(x; m \odot \theta)$ with algorithm $\mathcal{A}_t$ results in performance measurement on task $\mathcal{T}$ no lower than training dense networks $f(x; \theta)$ with algorithm $\mathcal{A}_t^{\mathcal{T}}$, where $\theta$ is the original random initialization $\theta_0$ or early rewound weights like $\theta_{5\%}$, and $t$ is the training iterations; ② the non-zero elements in pruning mask $m$ are clustered as channels, groups or other hardware-friendly structural patterns.

［#30］
Implementation details. We conduct experiments on diverse combinations of network architectures and datasets. Specifically, we adopt Wide-ResNet-32-2 (Zagoruyko & Komodakis, 2016) (or WRN-32-2), ResNet-18 (He et al., 2016) (or RN-18), MobileNet-v1 (or MBNet-v1) (Howard et al., 2017), and VGG-16 (Simonyan & Zisserman, 2014) on both CIFAR-10 (Krizhevsky et al., 2009) and CIFAR-100 datasets. ResNet-50 (or RN-50) is evaluated on both Tiny-ImageNet (Le & Yang, 2015) and ImageNet (Deng et al., 2009) datasets. Table 1 includes more training and evaluation details of our experiments.

### 3.2. Refilling for Structural Patterns

［#31］
It is well-known that the irregular sparsity patterns from unstructured magnitude pruning block the acceleration on practical hardware devices. To overcome the limitation, we propose a simple refilling strategy to reorganize the unstructured sparse patterns and make them more hardware friendly. Specifically, we first select important channels from the unstructured subnetwork according to certain criteria. The number of picked channels are depended on the desired sparsity level. Then, the pruned elements are grown back to be trainable (i.e., unpruned) and are reset to the same random initialization or early rewound weights. Lastly, the rest parameters in the remaining insignificant channels will be removed. In this way, we refill important channels and


［#31］
empty the rest to create a channel-wise structural sparse pattern that essentially brings computational reductions. Note that the picking criterion can be the number of remaining weights in the channel, or the channel's weight statistics or feature statistics or salience scores, which are comprehensively investigated in the ablation (Section A2). The complete pipeline and illustration are summarized in Algorithm 2 and Figure 2, respectively.

［#32］
```
Algorithm 1 IMP with rewinding step i
Input: $f(x;\theta_0)$, unstructured sparsity $s$
Output: $f(x;m \odot \theta_i)$
 1: Set the pruning mask $m = \mathbf{1} \in \mathbb{R}^{|\theta|}$
 2: Train $f(x;\theta_0)$ for $i$ steps: $f(x;\theta_i) = \mathcal{A}_i^{\mathcal{T}}(f(x;\theta_0))$
 3: while not reach sparsity $s$ do
 4:   Train $f(x;m \odot \theta_i)$ for $t-i$ steps: $f(x;m \odot \theta_t) = \mathcal{A}_{t-i}^{\mathcal{T}}(f(x;m \odot \theta_i))$
 5:   Pruning 20% of remaining weight of $m \odot \theta_t$, and
      update $m$
 6: end while
```

［#33］
```
Algorithm 2 IMP-Refill (+)
Input: $f(x;m \odot \theta_i)$ with unstructured sparsity $s$ (Algo. 1)
Output: $f(x;m \odot \theta_i)$ with channel-wise structural mask
［#33］
      $m$ at sparsity $\tilde{s}$
 1: Calculate importance scores of each channel according
    to certain criterion
 2: Pick top-$k$ channels in $m$, refill back their 0 (pruned)
    elements with 1 (trainable) and update $m$, maintaining
［#33］
    $\tilde{s} \sim s$
［#34］
 3: Pick and refill back extra channels in $m$ with $\tilde{s}^+ < s$
    # Optional for Refill+
```

［#35］
```
Algorithm 3 IMP-Regroup
Input: $f(x;m \odot \theta_i)$ with unstructured sparsity $s$ from Al-
      gorithm 1, hyperparameters $t_1, t_2, b_1$, and $b_2$
Output: $f(x;m \odot \theta_i)$ with group-wise structural mask $m$
      at sparsity $s^*$
 1: while dense block can be found do
 2:   Divide the rows of the sparse pruning mask $m$ into
［#35］
      $t_1$ groups using hypergraph partitioning (hMETIS)$^a$
 3:   for group $c_i \in \{c_1, c_2, ..., c_{t_1}\}$ do
 4:     if $c_i$ has $\geq b_1$ rows then
 5:       Select columns in $c_i$ that has no less than $t_2$
          non-zero items
 6:       if $\geq b_2$ columns are selected then
 7:         Group and Refill the selected columns as well
            as rows to a dense block, and update $m$
 8:       end if
 9:     end if
10:   end for
11: end while
12: Set other elements out of dense blocks to 0
```

［#35］
$^a$http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview

［#36］
Here we provide a detailed description of how many and which channels we choose to refill. Our main experiments adopt the $\ell_1$ norm of channel weights as the picking criterion to score the channel importance due to its superior performance. Let $\theta^l \in \mathbb{R}^{c_{\text{out}} \times n}$ denotes the parameters of the convolutional layer $l$, where $c_{\text{out}}$ is the number of output channel and $n$ is the continued product of the number of input channel, channel height and weight, as shown in Figure 2. $\theta_i^l \in \mathbb{R}^n$ represents the weights in the $i$th kernel and $m_i^l \in \{0,1\}^{|\theta_i^l|}$ is the corresponding mask. We first calculate the $\ell_1$ norm of $m_i^l \odot \theta_i^l$, which is a summation of the absolute value of remaining weights in the kernel $i$. Then we use it to pick the top-$k$ scored kernels, which will be fully refilled. $k = \lceil s^l \times c_{\text{out}} \times n \rceil$, where $s^l$ is the original layerwise sparsity and $c_{\text{out}} \times n$ is the total number of weights in kernel $i$. Meanwhile, the rest $c_{\text{out}} - k$ kernels are dropped for efficiency gains.

［#37］
Furthermore, we propose a soft version, refilling+, to make a redemption for the aggressive nature of wiping out all remaining channels. It picks and re-actives an extra proportion of channels to slow down the network capacity reduction, as indicated by shallow blue blocks in Figure 2.

### 3.3. Regrouping for Structural Patterns

［#38］
Although proposed refilling+ reorganizes the unstructured mask and produces useful channel-wise structural subnetworks, it is rigid and inelastic since the smallest manageable unit is a kernel. In other words, the dense matrices in identified structural patterns have a restricted shape where one dimension must align with the kernel size $n$, i.e., the continued product of the number of input channels, channel height, and weight. Motivated by Rumi et al. (2020), we introduce a regrouping strategy (Figure 2) to create more fine-grained group-wise structural patterns with flexible shapes for remaining dense matrices.

［#39］
▷ How to perform regrouping? Regrouping aims to find and extract dense blocks of non-pruned elements in the sparse weight matrix. These blocks have diverse shapes, as demonstrated in Figure 2, which are usually smaller in size compared to the original sparse matrix. Note that a channel/kernel can be regarded as a special case of the dense block.

［#40］
As described in Algorithm 3, to achieve the goal, we first need to find similar rows and columns, and then bring them together. Specifically, We adopt the Jaccard similarity (Rumi et al., 2020; Jiang et al., 2020) among non-zero columns as the similarity between two rows in the sparse matrix, which is calculated as a cardinality ratio of the intersections to the union of non-zero columns. For instance, kernel 1 and kernel 2 in Figure 2 (upper left) share three columns in eight non-zero distinct columns, and their similarity is $\frac{3}{8}$. Then, if two rows have a larger similarity, they

# Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#41］
![](./images/867754386212782437_3.jpg)

［#42］
Figure 3. (Curve plots) Testing accuracy (%) over network sparsity (%) on Tiny-ImageNet and ImageNet datasets with ResNet-50 (25.56 M). (Radar plots) The end-to-end inference time saving of extreme structural winning tickets. Unstructured subnetworks or dense models do not have structural sparsity, and thus they are plotted as dots in the axes of accuracy in the corresponding radar plot. The rightmost plot includes three extreme regroup tickets with accuracy drop $<1\%$, where "RG S: $x\%$" indicates unstructured sparsity before regrouping.

［#43］
can form a denser block when we group them together. Take Figure 2 as an example. We can group kernel 1,2,3's non-zero columns 1,3,6,11 with at least two elements together, which leads to the first orange dense block.

［#44］
More precisely, we take the hypergraph partitioning in the regrouping algorithm to generate dense blocks. It treats each row and column from the sparse matrix as a node and hyperedge in the hypergraph, where hyperedge (i.e., column) connects the corresponding nodes (i.e., row). Then, the pair-wise similarity is leveraged to locate an optimal partitioning, which can be achieved with hMETIS$^2$. More details are referred to Rumi et al. (2020). After obtaining the desired dense blocks, we enable all their parameters to be trainable by refilling the corresponding pruned elements. Note that refilling these pruned weights does not cause any efficiency loss since the size of the blocks is fixed, while it potentially maximizes the usage of these blocks and brings accuracy gains. Meanwhile, the rest parameters not included in the dense blocks will be discarded, i.e., setting the corresponding position in binary mask $m$ to zero, for reducing the computational overhead as illustrated in Figure 2. It is because any parameters outside the dense blocks require extra weights loading and have little data reuse (Rumi et al., 2020), which harms the trade-off of accuracy and efficiency.

［#45］
$\triangleright$ How refilled / regrouped dense blocks be beneficial?
We notice that the common tools like cuDNN (Chetlur et al., 2014) have a significant drawback that the inference time does not linearly change with the number of kernels, since they are only optimized for kernel matrices with a multiple of 32 rows (Radu et al., 2019). For example, as stated in Rumi et al. (2020), a convolutional layer with 10 kernels might have a similar inference time with a convolutional layer with 32 kernels. However, the number of kernels in these dense blocks is almost arbitrary, so a more sophisticated GEMM-based efficient implementation (Rumi et al., 2020) is needed to accelerate better our refilled / regrouped structural patterns. Following Rumi et al. (2020), we split a kernel with $r$ rows into two parts: one has $\lceil r/32 \rceil \times 32$ rows and the other one has $r \bmod 32$ rows. First, we directly apply the standard GEMM-based convolution algorithm with shared memory to cache the input and output matrix. For the second part, due to the poor data reuse of input matrices, we choose to cache the kernel and output matrices for an improved cache hit rate and overall performance. More details are referred to Rumi et al. (2020).

## 4. The Existence of Structural Winning Ticket
［#46］
Tiny-ImageNet and ImageNet. In this section, we reveal the existence of our proposed structural winning tickets on ImageNet and Tiny-ImageNet with ResNet-50 backbone. Results of unstructured IMP, channel-wise structural `IMP-Refill(+)`, and group-wise structural `IMP-Regroup` are collected in the Figure 3. The end-to-end inference time$^3$ of obtained structural winning tickets with extreme sparsity levels are presented, which is calculated on a single 2080 TI GPU with a batch size of 64. Extreme sparsity is defined as maximum sparsity when the subnetwork has superior accuracy to its dense counterpart.

［#47］
From Tiny-ImageNet results in Figure 3 (left), several positive observations can be drawn: ❶ Structural winning tickets with $60\%$ channel-wise structural sparsity and $74\%$ group-wise structural sparsity are located by `IMP-Refill` and `IMP-Regroup` respectively, which validate the effectiveness of our proposals. ❷ Although at the high sparsity levels (i.e., $>50\%$), `IMP-Refill+` outperforms `IMP-Refill` if they are from the same unstructured IMP subnetworks. Considering the overall trade-off between channel-wise structural sparsity and accuracy, `IMP-Refill` appears a clear advantage. A possible explanation is that *refilling+* seems to bring undesired channels which potentially result in a degraded performance trade-off. ❸ `IMP-Regroup` performs better at high sparsities. It is within expecta-

［#47］
$^2$http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview
［#47］
$^3$TorchPerf (https://github.com/awwong1/torchprof) is adopted as our tool to benchmark both the end-to-end and layer-wise running time on GPU devices.

［#48］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#49］
![](./images/867754386212782437_4.jpg)

［#50］
Figure 4. Testing accuracy (%) over sparsity (%) on CIFAR-10/100 with Wide-ResNet-32-2 (1.86 M) and MobileNet-v1 (3.21 M).

［#51］
![](./images/867754386212782437_5.jpg)

［#52］
Figure 5. (Curve plots) Testing accuracy (%) over sparsity (%) on CIFAR-10/100 with large models VGG-16 (14.72 M) and RN-18 (11.22 M). (Radar plots) The end-to-end inference time saving of extreme structural winning tickets. Note that unstructured subnetworks or dense models do not have structural sparsity, and thus they are plotted as dots in the axes of accuracy in the corresponding radar plot.

［#53］
tion since fine-grained group-wise structural patterns tend to make the networks be more amenable to pruning. ❹ Extreme channel- / group-wise structural winning tickets with $45\% \sim 50\% / 74\%$ sparsity from $\text{IMP-Refill}(+)$ / $\text{IMP-Regroup}$ achieve $57.53\% \sim 61.79\% / 64.84\%$ GPU running time savings, without sacrificing accuracies.

［#54］
As for large-scale ImageNet experiments, the conclusion are slightly different: ❶ There is almost no difference between the performance of $\text{IMP-Refill}$ and $\text{IMP-Refill+}$, and both can not find channel-wise structural winning tickets. But it seems to suggest our picking rule (i.e., channel weights' $\ell_1$ norm) provides a great estimation for channel importance, although it is too aggressive for ImageNet experiments. ❷ The group-wise structural winning ticket at $31\%$ sparsity still exist in (RN-50, ImageNet), while the low sparsity brings limited $1\%$ time savings. For a better efficiency and performance trade-off, $\text{IMP-Regroup}$ is capable of locating structural subnetworks at $51\% / 58\%$ sparsity with $53.75\% / 60.23\%$ time savings and $0.33\% / 0.95\%$ accuracy drop.

［#55］
CIFAR with diverse network architectures. We then validate our approaches on CIFAR-10/100 (C10/100) with diverse network backbones including Wide-ResNet-32-2, MobileNet-v1, VGG-16, and ResNet-18. Based on the extensive results in Figure 4 and 5, we find: ❶ On {(WRN-32-2,C10), (WRN-32-2,C100), (MBNet-v1,C10), (MBNet-v1,C100), (VGG-16,C10), (VGG-16,C100), (RN-18,C10), (RN-18,C100)} schemes, we consistently disclose the existence of structural winning tickets with $\{53\%, 28\%, 67\%, 0\%, 60\%, 40\%, 50\%, 0\%\}$ channel-wise sparsity and $\{66\%, 36\%, 72\%, 56\%, 80\%, 80\%, 78\%, 78\%\}$ group-wise sparsity from $\text{IMP-Refill}(+)$ and $\text{IMP-Regroup}$, respectively. ❷ With the same network, pursuing channel-wise sparse patterns on CIFAR-100 is more challenging than it on CIFAR-10, possibly due to the larger dataset complexity. On the same dataset, larger networks tend to have larger extreme sparsities for both channel- and group-wise structural winning tickets, with the exception of $\text{IMP-Refill}(+)$ on (RN-18, C100). ❸ At the mid-dle sparsity levels (i.e., $< 50\%$), $\text{IMP-Regroup}$ behaves closely to $\text{IMP-Refill}(+)$, while $\text{IMP-Regroup}$ has

［#56］
![](./images/867754386212782437_6.jpg)

［#57］
Figure 6. (Left) Performance of structural tickets grouped from diverse initial unstructured masks. (Middle) Performance of group-wise structural tickets with different weight rewinding. (Right) Performance comparisons between $\text{IMP-Regroup}$ and group-aware $\text{IMP}$ as described in Algorithm 4. Testing accuracies (%) over network sparsity levels (%) are reported on (RN-18,C10).

［#58］
a superior performance at high sparsity levels. ④ Up to $\{57.75\%,\ 60.60\%,\ 55.45\%,\ 64.93\%\}$ GPU running time savings are obtained by group-wise structural winning tickets with undamaged performance on $\{\text{(VGG-16,C10)},\ \text{(VGG-16,C100)},\ \text{(RN-18,C10)},\ \text{(RN-18,C100)}\}$, which surpass $\text{IMP}$, $\text{IMP-Refill}(+)$, and dense models by a significant efficiency margin. A exception is that $\text{IMP-Refill}$ on (VGG-16,C10) achieves the best time savings, i.e., $63.11\%$.

［#59］
Layer-wise speedups. Figure 7 and A10 shows the layer-wise speedup performance of convolution operations in VGG-16's extreme structured winning tickets from different algorithms.$\text{IMP-Regroup}$ presents impressive layer-wise speedups up to 6.67x compared to others, especially on the last a few layers (e.g., conv. 12). The possible reasons lie in two aspects: (i) the latter layers reach a larger compression ratio and have greater potentials for acceleration; (ii) the regrouping algorithm prefers convolutional layers (i.e., latter layers in VGG-16) with a larger number of kernels which benefits to group appropriate dense blocks, as also suggested by Rumi et al. (2020).

［#60］
![](./images/867754386212782437_7.jpg)

［#61］
Figure 7. The layer-wise performance of convolution operations in extreme structural winning tickets of (VGG-16, C10). The first six conv. operations are omitted since there is no meaningful speedup, coincided with Rumi et al. (2020). Marks like “C: 2.77” indicate the layer-wise compression ratio of $\text{IMP-Regroup}$.

## 5. Ablation Study and Visualization
［#62］
Different sources of unstructured masks. Intuitively, the initial unstructured sparse mask should plays an essential role in the achievable performance of our proposed “post-processing techniques”. We therefore conduct a comprehensive ablation study about the various sources of the initial sparse masks in Figure 6, including $\text{IMP}$, $\text{OMP}$, $\text{RP}$, $\text{SNIP}$, $\text{GraSP}$, $\text{SynFlow}$, and $\text{ADMM}$. The details of comparison methods are in Section 3.1. We observe that $\text{IMP}$ and $\text{OMP}$ provide initial unstructured masks with the top-2 highest quality for our regrouping algorithm, in terms of the train-from-scratch accuracy of grouped structural subnetworks.

［#63］
Different initialization for the re-training. Initialization (Frankle & Carbin, 2019; Renda et al., 2020) as another key factor in LTH, also contributes significantly to the existence of winning tickets. To exhaustively investigate the effect from different initialization (e.g., rewound weights), we launch experiments started from diverse rewound weights $(\{5\%,10\%,20\%,50\%,100\%\}$ of total training epochs) as well as a random re-initialization. In Figure 6, using 50% rewound weight reaches the overall best performance; other weight rewinding setups perform similarly and clearly surpass random re-initializing at sparsity levels $>30\%$.

［#64］
Group-aware $\text{IMP}$. This work mainly focuses on the post-processing of unstructured sparse masks. Another possibility is integrating regrouping into $\text{IMP}$ by alternatively performing unstructured magnitude pruning and regrouping, which we term as group-aware $\text{IMP}$. From Fig. 6, it has a worse performance due to the stricter constraint on sparse patterns, compared to $\text{IMP-Regroup}$.

［#65］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#66］
Extra study. More investigations about (1) transfer tickets and training efficiency; (2) comparison with random tickets; (3) ablation on different training settings; (4) FLOPs saving; (5) visualization of sparse masks are in Appendix A2.

## 6. Conclusion
［#67］
In this paper, we challenge the "common sense" that an identified IMP winning ticket can only have unstructured sparsity, which severely limits its practical usage due to the irregular patterns. We for the first time demonstrate the existence of structural winning tickets by leveraging post-processing techniques, i.e., refilling(+) and regrouping. The located channel- and group-wise structural subnetworks achieve significant inference speedups up to 6.67x on hardware platforms. In this sense, our positive results bridge the gap between the lottery ticket hypothesis and practical accelerations in real-world scenarios. We would be interested in examining LTH with more effective structural sparsity for real-time mobile computing in future work.

## Acknowledgment
［#68］
Z.W. is in part supported by an NSF EPCN project (#2053272). Y.W. is in part supported by an NSF CMMI project (#2013067).

## References


















### Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#69］
Frankle, J., Dziugaite, G. K., Roy, D., and Carbin, M. Linear mode connectivity and the lottery ticket hypothesis. In *International Conference on Machine Learning*, pp. 3259–3269. PMLR, 2020a.

［#70］
Frankle, J., Schwab, D. J., and Morcos, A. S. The early phase of neural network training. In *International Conference on Learning Representations*, 2020b. URL https://openreview.net/forum?id=Hk11iRNFwS.

［#71］
Gale, T., Elsen, E., and Hooker, S. The state of sparsity in deep neural networks. *arXiv preprint* arXiv:1902.09574, 2019.

［#72］
Gan, Z., Chen, Y.-C., Li, L., Chen, T., Cheng, Y., Wang, S., and Liu, J. Playing lottery tickets with vision and language. *arXiv preprint* arXiv:2104.11832, 2021.

［#73］
Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *arXiv preprint* arXiv:1510.00149, 2015a.

［#74］
Han, S., Pool, J., Tran, J., and Dally, W. J. Learning both weights and connections for efficient neural networks. *arXiv preprint* arXiv:1506.02626, 2015b.

［#75］
Han, S., Liu, X., Mao, H., Pu, J., Pedram, A., Horowitz, M. A., and Dally, W. J. Eie: efficient inference engine on compressed deep neural network. In *ISCA*, 2016.

［#76］
Hanson, S. and Pratt, L. Comparing biases for minimal network construction with back-propagation. *Advances in neural information processing systems*, 1:177–185, 1988.

［#77］
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

［#78］
He, Y., Zhang, X., and Sun, J. Channel pruning for accelerating very deep neural networks. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 1389–1397, 2017.

［#79］
Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., and Peste, A. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *arXiv preprint* arXiv:2102.00554, 2021.

［#80］
Hong, C., Sukumaran-Rajam, A., Bandyopadhyay, B., Kim, J., Kurt, S. E., Nisa, I., Sabhlok, S., Çatalyürek, U. V., Parthasarathy, S., and Sadayappan, P. Efficient sparse-matrix multi-vector product on gpus. Association for Computing Machinery, 2018.

［#81］
Hong, C., Sukumaran-Rajam, A., Nisa, I., Singh, K., and Sadayappan, P. Adaptive sparse tiling for sparse matrix multiplication. In *Proceedings of the 24th Symposium on Principles and Practice of Parallel Programming*, pp. 300–314, 2019.

［#82］
Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint* arXiv:1704.04861, 2017.

［#83］
Hu, H., Peng, R., Tai, Y.-W., and Tang, C.-K. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. *arXiv preprint* arXiv:1607.03250, 2016.

［#84］
Jiang, P., Hong, C., and Agrawal, G. A novel data transformation and execution strategy for accelerating sparse matrix multiplication on gpus. PPoPP ’20. Association for Computing Machinery, 2020.

［#85］
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. *arXiv preprint* arXiv:2001.08361, 2020.

［#86］
Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

［#87］
Le, Y. and Yang, X. Tiny imagenet visual recognition challenge. *CS 231N*, 7:7, 2015.

［#88］
LeCun, Y., Denker, J. S., and Solla, S. A. Optimal brain damage. In *Advances in neural information processing systems*, pp. 598–605, 1990.

［#89］
Lee, N., Ajanthan, T., and Torr, P. Snip: Single-shot network pruning based on connection sensitivity. In *International Conference on Learning Representations (ICLR)*, 2019a.

［#90］
Lee, N., Ajanthan, T., and Torr, P. Snip: Single-shot network pruning based on connection sensitivity. In *International Conference on Learning Representations*, 2019b. URL https://openreview.net/forum?id=B1VZqjAcYX.

［#91］
Li, H., Kadav, A., Durdanovic, I., Samet, H., and Graf, H. P. Pruning filters for efficient convnets. *arXiv preprint* arXiv:1608.08710, 2016.

［#92］
Lin, M., Ji, R., Zhang, Y., Zhang, B., Wu, Y., and Tian, Y. Channel pruning via automatic structure search. *arXiv preprint* arXiv:2001.08565, 2020.

［#93］
Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., and Zhang, C. Learning efficient convolutional networks through network slimming. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 2736–2744, 2017.

# Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#94］
Louizos, C., Welling, M., and Kingma, D. P. Learning sparse neural networks through $l_0$ regularization. *arXiv preprint* arXiv:1712.01312, 2017.

［#95］
Ma, H., Chen, T., Hu, T.-K., You, C., Xie, X., and Wang, Z. Good students play big lottery better. *arXiv preprint* arXiv:2101.03255, 2021a.

［#96］
Ma, X., Guo, F.-M., Niu, W., Lin, X., Tang, J., Ma, K., Ren, B., and Wang, Y. Pconv: The missing but desirable sparsity in dnn weight pruning for real-time execution on mobile devices. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pp. 5117–5124, 2020.

［#97］
Ma, X., Yuan, G., Shen, X., Chen, T., Chen, X., Chen, X., Liu, N., Qin, M., Liu, S., Wang, Z., et al. Sanity checks for lottery tickets: Does your winning ticket really win the jackpot? *arXiv preprint* arXiv:2107.00166, 2021b.

［#98］
Mao, H., Han, S., Pool, J., Li, W., Liu, X., Wang, Y., and Dally, W. J. Exploring the regularity of sparse structure in convolutional neural networks. *arXiv preprint* arXiv:1705.08922, 2017.

［#99］
Molchanov, P., Mallya, A., Tyree, S., Frosio, I., and Kautz, J. Importance estimation for neural network pruning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 11264–11272, 2019.

［#100］
Niu, W., Ma, X., Lin, S., Wang, S., Qian, X., Lin, X., Wang, Y., and Ren, B. Patdnn: Achieving real-time dnn execution on mobile devices with pattern-based weight pruning. *arXiv preprint* arXiv:2001.00138, 2020.

［#101］
Park, J., Li, S., Wen, W., Tang, P. T. P., Li, H., Chen, Y., and Dubey, P. Faster cnns with direct sparse convolutions and guided pruning. In *International Conference on Learning Representations*, 2016.

［#102］
Peng, K.-Y., Fu, S.-Y., Liu, Y.-P., and Hsu, W.-C. Adaptive runtime exploiting sparsity in tensor of deep learning neural network on heterogeneous systems. In *2017 International Conference on Embedded Computer Systems: Architectures, Modeling, and Simulation (SAMOS)*, pp. 105–112. IEEE, 2017.

［#103］
Radu, V., Kaszyk, K., Wen, Y., Turner, J., Cano, J., Crowley, E. J., Franke, B., Storkey, A., and O'Boyle, M. Performance aware convolutional neural network channel pruning for embedded gpus. In *2019 IEEE International Symposium on Workload Characterization (IISWC)*, pp. 24–34. IEEE, 2019.

［#104］
Ren, A., Zhang, T., Ye, S., Li, J., Xu, W., Qian, X., Lin, X., and Wang, Y. Admm-nn: An algorithm-hardware co-design framework of dnns using alternating direction method of multipliers, 2018.

［#105］
Renda, A., Frankle, J., and Carbin, M. Comparing rewinding and fine-tuning in neural network pruning. In *8th International Conference on Learning Representations*, 2020.

［#106］
Ruder, S. An overview of gradient descent optimization algorithms. *arXiv preprint* arXiv:1609.04747, 2016.

［#107］
Rumi, M. A., Ma, X., Wang, Y., and Jiang, P. Accelerating sparse cnn inference on gpus with performance-aware weight pruning. In *Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques*, pp. 267–278, 2020.

［#108］
Shangguan, Y., Li, J., Liang, Q., Alvarez, R., and McGraw, I. Optimizing speech recognition for the edge. *arXiv preprint* arXiv:1909.12408, 2019.

［#109］
Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. *arXiv preprint* arXiv:1409.1556, 2014.

［#110］
Tanaka, H., Kunin, D., Yamins, D. L., and Ganguli, S. Pruning neural networks without any data by iteratively conserving synaptic flow. In *Advances in Neural Information Processing Systems 33 pre-proceedings*, 2020.

［#111］
Tang, Y., Wang, Y., Xu, Y., Tao, D., Xu, C., Xu, C., and Xu, C. Scop: Scientific control for reliable neural network pruning. *Advances in Neural Information Processing Systems*, 33:10936–10947, 2020.

［#112］
Wang, C., Zhang, G., and Grosse, R. Picking winning tickets before training by preserving gradient flow. In *International Conference on Learning Representations*, 2020. URL `https://openreview.net/forum?id=SkgsACVKPH`.

［#113］
Wen, W., Wu, C., Wang, Y., Chen, Y., and Li, H. Learning structured sparsity in deep neural networks. In *Advances in neural information processing systems (NeurIPS)*, pp. 2074–2082, 2016.

［#114］
You, H., Li, C., Xu, P., Fu, Y., Wang, Y., Chen, X., Baraniuk, R. G., Wang, Z., and Lin, Y. Drawing early-bird tickets: Toward more efficient training of deep networks. In *International Conference on Learning Representations*, 2020. URL `https://openreview.net/forum?id=BJxsrgStvr`.

［#115］
Yu, H., Edunov, S., Tian, Y., and Morcos, A. S. Playing the lottery with rewards and multiple languages: lottery tickets in rl and nlp. In *8th International Conference on Learning Representations*, 2020.

［#116］
Zagoruyko, S. and Komodakis, N. Wide residual networks. *arXiv preprint* arXiv:1605.07146, 2016.

［#117］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#118］
Zhang, T., Ye, S., Zhang, K., Tang, J., Wen, W., Fardad, M., and Wang, Y. A systematic DNN weight pruning framework using alternating direction method of multipliers. In *ECCV*, 2018.

［#119］
Zhang, Z., Chen, X., Chen, T., and Wang, Z. Efficient lottery ticket finding: Less data is more. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 12380–12390. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/zhang21c.html.

［#120］
Zhou, A., Ma, Y., Zhu, J., Liu, J., Zhang, Z., Yuan, K., Sun, W., and Li, H. Learning n: M fine-grained structured sparse neural networks from scratch. *arXiv preprint arXiv:2102.04010*, 2021.

［#121］
Zhu, M., Zhang, T., Gu, Z., and Xie, Y. Sparse tensor core: Algorithm and hardware co-design for vector-wise sparse neural networks on modern gpus. In *Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture*, pp. 359–371, 2019.

［#122］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

## A1. More Implementation Details

［#123］
Group-aware IMP. Here we provides the detailed procedures of group-aware IMP in Algorithm 4. Intuitively, it embeds regrouping (Algorithm 3) into IMP (Algorithm 1) by performing regrouping on the unstructured mask $m$ from each IMP round.

［#124］
```
Algorithm 4 Group-aware IMP
Input: $f(x;\theta_0)$, group-wise structural sparsity $s$
Output: $f(x;m \odot \theta_i)$ with group-wise structural sparse
    mask $s$
 1: Set the pruning mask $m=1 \in \mathbb{R}^{|\theta|}$
    Train $f(x;\theta_0)$ to rewinding step $i$: $f(x;\theta_i) = \mathcal{A}_i^{\mathcal{T}}(f(x;\theta_0))$
 2: while not reach sparsity $s$ do
 3:   Train $f(x;m \odot \theta_i)$ to step $t$: $f(x;m \odot \theta_t) = \mathcal{A}_{t-i}^{\mathcal{T}}(f(x;m \odot \theta_i))$
 4:   Pruning 20% of remaining weight of $m \odot \theta_t$, and
        update $m$
 5:   Refining the unstructured mask $m$ by performing
        regrouping, as shown in Algorithm 3
 6: end while
```

［#125］
Profiling. To compute the GPU running time of regrouped convolution layers, we adopt their CUDA C/C++ implementation. Our results do not include the running time of normalization and activation layers, following the standard in Rumi et al. (2020). For a fair calculation, we feed the same input features to convolution layers that belong to the same model. For ResNet-18 and VGG-16, the size of the input features is $(64, 64, 127, 127)$. For ResNet-50, the size of input features is $(64, 64, 64, 64)$. The GPU we use for profiling is NVIDIA RTX 2080 TI, with a CUDA version of 10.2 and a cuDNN (Chetlur et al., 2014) version of 7.6.5.

## A2. More Experiment Results

［#126］
Different channel picking criterion for refilling. We ablation the channel picking criterion for IMP-Refill(+), including 1) the $\ell_1$ norm of channel's remaining weight, 2) the $\ell_1$ or $\ell_2$ norms of channel's feature map, 3) the number of remaining weights in the channel, 4) the channel's saliency score (Molchanov et al., 2019). Experiment results are collected in Figure A8, which demonstrate the superior performance of IMP-Refill w. $\ell_1$ of channel weights (yellow curve in Figure A8).

［#127］
![](./images/867754386212782437_8.jpg)

［#128］
Figure A8. Performance of structural tickets refilled by diverse channel picking criterion. Testing accuracies (%) over network sparsity levels (%) are reported on (RN-18,C10).

［#129］
Transfer tickets and training efficiency. We investigate the transferability of our found (fine-grained) structural winning tickets, which grants the extra bonus of training efficiency to our proposals. Specifically, following the setups in Chen et al. (2020a), we first identify refilled and regrouped structural winning tickets in ResNet-18 on ImageNet, and then transfer them to the downstream CIFAR-10 task. Transfer results are presented in Table A2. Compared to the dense network baseline (95.37%), IMP-Refill locates channel-wise structural (transfer) winning tickets at the sparsity around 36%, and IMP-Regroup locates group-wise structural (transfer) winning tickets at a higher sparsity (more than 56.00%). Such an encouraging transfer study means that we can even replace the full model with a much smaller subnetwork while maintaining an undamaged downstream performance. And this is also why our IMP-Regroup and IMP-Refill obtain 7.14% ~ 34.67% and 34.53% training time savings during downstream training with matched or even improved generalization. Note that this efficient training is an extra benefits of our proposal, in addition to impressive inference efficiency.

［#130］
Table A2. Transfer accuracy (%), time saving (%) and remaining weights (%) on ResNet-50 with CIFAR-10. IMP-Refill and IMP-Regroup are evaluated. The baseline accuracy of dense network is 95.37%.

### IMP-Refill
［#131］
| Remaining Weight | Transfer Accuracy | Time Savings |
|------------------|-------------------|--------------|
| 64.14%           | 95.81%            | 34.53%       |
| 51.37%           | 95.14%            | 48.10%       |
| 41.01%           | 94.51%            | 60.67%       |
| 32.76%           | 94.38%            | 65.98%       |
| 26.17%           | 94.19%            | 69.04%       |
| 20.97%           | 94.11%            | 71.08%       |

### IMP-Regroup
［#132］
| Remaining Weight | Transfer Accuracy | Time Savings |
|------------------|-------------------|--------------|
| 59.43%           | 95.65%            | 7.14%        |
| 51.84%           | 95.39%            | 21.85%       |
| 43.99%           | 95.51%            | 34.67%       |


［#133］
Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#134］
Comparison with random tickets. As a sanity check, we conduct a comparison with random tickets (Frankle & Carbin, 2019) which are trained from random re-initialization. Experiments results on (RN-18, C10) are collected in Table A3 and A4. We show that random tickets have obviously inferior performance, which suggests that our identified refilled and regrouped subnetworks are highly non-trivial (fine-grained) structural winning tickets.

［#135］
Table A3. Testing accuracy (%) of IMP-Refill with rewinding weights (Ours) and random re-initialization (Random Tickets).
［#136］
<table>
  <thead>
    <tr>
      <th>IMP Round</th>
      <th>Remaining Weight</th>
      <th>Accuracy (Ours)</th>
      <th>Accuracy (Random Tickets)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>80.29%</td>
      <td>94.14%</td>
      <td>94.04%</td>
    </tr>
    <tr>
      <td>2</td>
      <td>64.49%</td>
      <td>94.24%</td>
      <td>93.96%</td>
    </tr>
    <tr>
      <td>3</td>
      <td>51.43%</td>
      <td>94.45%</td>
      <td>94.20%</td>
    </tr>
    <tr>
      <td>4</td>
      <td>41.24%</td>
      <td>94.16%</td>
      <td>93.98%</td>
    </tr>
    <tr>
      <td>5</td>
      <td>32.97%</td>
      <td>93.86%</td>
      <td>93.53%</td>
    </tr>
  </tbody>
</table>

［#137］
Table A4. Testing accuracy (%) of IMP-Regroup with rewinding weights (Ours) and random re-initialization (Random Tickets).
［#138］
<table>
  <thead>
    <tr>
      <th>IMP Round</th>
      <th>Remaining Weight</th>
      <th>Accuracy (Ours)</th>
      <th>Accuracy (Random Tickets)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>80.00%</td>
      <td>94.48%</td>
      <td>94.19%</td>
    </tr>
    <tr>
      <td>2</td>
      <td>72.95%</td>
      <td>94.75%</td>
      <td>93.42%</td>
    </tr>
    <tr>
      <td>3</td>
      <td>69.37%</td>
      <td>94.29%</td>
      <td>93.58%</td>
    </tr>
    <tr>
      <td>4</td>
      <td>67.12%</td>
      <td>94.62%</td>
      <td>93.84%</td>
    </tr>
    <tr>
      <td>5</td>
      <td>58.54%</td>
      <td>94.32%</td>
      <td>93.76%</td>
    </tr>
  </tbody>
</table>

［#139］
Different training settings. To validate our algorithm's effectiveness under different training configurations, we perform extra experiments with VGG-16(+), WRN-32-2(+), and RN-50(+). The changes of training settings are summarized as below:

［#140］
① For VGG-16(+), we increase the number of training epochs to 240, and decay the learning rate at 150-th, 180-th, and 210-th epoch.

［#141］
② For WRN-32-2(+), we do not split the official training set into the a training and a validation set as our other experiments did. We also report the best validation accuracy instead of the best test accuracy. The number of training epochs is increased to 240 and the learning rate is decayed at 150-th, 180-th, and 210-th epoch.

［#142］
③ For RN-50(+), we replace the first convolution layer to be of kernel size 3, padding size 1, and strides 1.

［#143］
▷ VGG-16(+) on C100. As shown in Table A5, we demonstrated that our conclusions are still hold: IMP-Regroup can locate structural winning tickets at very high sparsity levels (e.g., $>75\%$).

［#144］
Table A5. Testing accuracy (%) and remaining weights (%) on CIFAR-100 with VGG-16(+). IMP, IMP-Refill, and IMP-Regroup are evaluated. The baseline accuracy of dense network is $73.43\%$.
［#145］
<table>
  <thead>
    <tr>
      <th rowspan="2">Round</th>
      <th colspan="2">IMP</th>
      <th colspan="2">IMP-Refill</th>
      <th colspan="2">IMP-Regroup</th>
    </tr>
    <tr>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>80.00%</td>
      <td>73.64</td>
      <td>80.17%</td>
      <td>73.43</td>
      <td>82.36%</td>
      <td>73.63</td>
    </tr>
    <tr>
      <td>2</td>
      <td>64.00%</td>
      <td>73.80</td>
      <td>64.06%</td>
      <td>72.87</td>
      <td>80.00%</td>
      <td>73.81</td>
    </tr>
    <tr>
      <td>3</td>
      <td>51.20%</td>
      <td>73.67</td>
      <td>51.31%</td>
      <td>72.67</td>
      <td>69.46%</td>
      <td>74.31</td>
    </tr>
    <tr>
      <td>4</td>
      <td>40.96%</td>
      <td>74.01</td>
      <td>41.08%</td>
      <td>71.37</td>
      <td>62.61%</td>
      <td>73.94</td>
    </tr>
    <tr>
      <td>5</td>
      <td>32.77%</td>
      <td>74.27</td>
      <td>32.85%</td>
      <td>70.79</td>
      <td>56.09%</td>
      <td>75.05</td>
    </tr>
    <tr>
      <td>6</td>
      <td>26.21%</td>
      <td>74.56</td>
      <td>26.33%</td>
      <td>71.07</td>
      <td>46.53%</td>
      <td>74.98</td>
    </tr>
    <tr>
      <td>7</td>
      <td>20.97%</td>
      <td>74.58</td>
      <td>21.03%</td>
      <td>69.42</td>
      <td>38.18%</td>
      <td>75.24</td>
    </tr>
    <tr>
      <td>8</td>
      <td>16.78%</td>
      <td>74.52</td>
      <td>16.94%</td>
      <td>68.75</td>
      <td>30.98%</td>
      <td>74.68</td>
    </tr>
    <tr>
      <td>9</td>
      <td>13.42%</td>
      <td>74.42</td>
      <td>13.42%</td>
      <td>67.25</td>
      <td>25.27%</td>
      <td>75.25</td>
    </tr>
  </tbody>
</table>

［#146］
▷ WRN-32-2(+) on C100. As shown in Table A6, we find consistent observations: IMP-Regroup locates structural winning tickets at about 75% sparsity, and IMP-Refill identifies structural winning tickets at 20% sparsity.

［#147］
Table A6. Testing accuracy (%) and remaining weights (%) on CIFAR-100 with WideResNet-32-2(+). IMP, IMP-Refill, and IMP-Regroup are evaluated. The baseline accuracy of dense network is $75.53\%$.
［#148］
<table>
  <thead>
    <tr>
      <th rowspan="2">Round</th>
      <th colspan="2">IMP</th>
      <th colspan="2">IMP-Refill</th>
      <th colspan="2">IMP-Regroup</th>
    </tr>
    <tr>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>80.00%</td>
      <td>76.21</td>
      <td>80.00%</td>
      <td>75.46</td>
      <td>80.00%</td>
      <td>75.98</td>
    </tr>
    <tr>
      <td>2</td>
      <td>64.00%</td>
      <td>75.78</td>
      <td>64.06%</td>
      <td>74.59</td>
      <td>64.00%</td>
      <td>76.19</td>
    </tr>
    <tr>
      <td>3</td>
      <td>51.20%</td>
      <td>76.02</td>
      <td>51.51%</td>
      <td>73.53</td>
      <td>51.20%</td>
      <td>76.13</td>
    </tr>
    <tr>
      <td>4</td>
      <td>40.96%</td>
      <td>75.92</td>
      <td>41.51%</td>
      <td>72.95</td>
      <td>40.96%</td>
      <td>75.88</td>
    </tr>
    <tr>
      <td>6</td>
      <td>26.21%</td>
      <td>75.74</td>
      <td>26.53%</td>
      <td>70.91</td>
      <td>26.27%</td>
      <td>75.78</td>
    </tr>
    <tr>
      <td>7</td>
      <td>20.97%</td>
      <td>75.92</td>
      <td>21.11%</td>
      <td>69.55</td>
      <td>21.76%</td>
      <td>74.74</td>
    </tr>
    <tr>
      <td>8</td>
      <td>16.78%</td>
      <td>75.87</td>
      <td>17.11%</td>
      <td>67.74</td>
      <td>18.14%</td>
      <td>73.85</td>
    </tr>
    <tr>
      <td>9</td>
      <td>13.42%</td>
      <td>75.41</td>
      <td>13.67%</td>
      <td>65.73</td>
      <td>14.85%</td>
      <td>72.99</td>
    </tr>
  </tbody>
</table>

［#149］
![](./images/867754386212782437_9.jpg)

［#150］
Figure A9. Sparse mask visualizations of the extreme winning tickets from IMP (unstructured), IMP-Refill (+) (channel-wise structural), and IMP-Regroup (group-wise structural) on (VGG-16,C10). The darker color indicates the remaining unpruned elements. (a,b,c) are the last three conv. layers.

［#151］
▷ RN-50(+) on Tiny-ImageNet. Experimental results in Table A7 suggest that: IMP-Regroup locates structural winning tickets at about 42% sparsity, and IMP-Refill discovers structural winning tickets at 20% sparsity, which echo our findings in the main text.

# Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

［#152］
Table A7. Testing accuracy (%) and remaining weights (%) on Tiny-ImageNet with ResNet-50(+). IMP, IMP-Refill, and IMP-Regroup are evaluated. The baseline accuracy of dense network is 65.33%.

［#153］
<table>
  <thead>
    <tr>
      <th rowspan="2">Round</th>
      <th colspan="2">IMP</th>
      <th colspan="2">IMP-Refill</th>
      <th colspan="2">IMP-Regroup</th>
    </tr>
    <tr>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
      <th>Remaining Weight</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>80.00%</td>
      <td>65.44</td>
      <td>80.30%</td>
      <td>65.27</td>
      <td>80.15%</td>
      <td>65.51</td>
    </tr>
    <tr>
      <td>2</td>
      <td>64.00%</td>
      <td>65.69</td>
      <td>64.16%</td>
      <td>63.40</td>
      <td>68.25%</td>
      <td>65.16</td>
    </tr>
    <tr>
      <td>3</td>
      <td>51.20%</td>
      <td>65.50</td>
      <td>51.42%</td>
      <td>61.89</td>
      <td>58.19%</td>
      <td>65.21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>40.96%</td>
      <td>65.73</td>
      <td>41.08%</td>
      <td>60.43</td>
      <td>54.19%</td>
      <td>64.42</td>
    </tr>
    <tr>
      <td>5</td>
      <td>32.77%</td>
      <td>65.23</td>
      <td>32.85%</td>
      <td>59.64</td>
      <td>51.75%</td>
      <td>64.52</td>
    </tr>
  </tbody>
</table>

［#154］
FLOPs saving. For a sufficient evaluation, we calculate the FLOPs of diverse subnetworks from VGG-16 on CIFAR-10 dataset. The FLOPs of a dense VGG-16 is about 0.314G. We select sparsity levels across different methods as similar as possible for a better comparison. Subnetworks from IMP-Refill, IMP-Refill+, and IMP-Regroup at sparsity levels of $\{32.84\%, 46.41\%, 20.12\%\}$ have $\{0.089G, 0.122G, 0.093G\}$ FLOPs, respectively. It is noteworthy that Refill and Refill+ trim down the input and output channels of a convolution layer while Regroup cannot. Thus, Refill and Refill+ can save more FLOPs under a similar sparsity level.

［#155］
Visualization of sparse masks. Figure A9 visualizes different types of obtained sparse masks from (VGG-16,C10). Sub-figures (a,b,c) plot the mask matrices of size $c_{out} \times n$ for certain layers. Similar to the illustration in Figure 2, IMP-Refill (+) masks show clear kernel-wise sparse patterns across the rows, and IMP-Regroup masks present fine-grained structural sparse patterns capable of forming neat dense blocks after regrouping.

［#156］
More results of layer-wise speedups. Figure A10 presents extra layer-wise speedup results of VGG-16 on CIFAR-100. Similar observations to Figure 7 can be obtain.

［#157］
![](./images/867754386212782437_10.jpg)

［#158］
Figure A10. The layer-wise performance of convolution operations in extreme structural winning tickets of (VGG-16, C10). The first six conv. operations are omitted since there is no meaningful speedup, coincided with Rumi et al. (2020). Marks like "C: 2.77" indicate the layer-wise compression ratio of IMP-Regroup.

［#159］
Different visualizations of the radar plots. We offer an alternative histogram visualization (Figure A11) for radar plots in the main text. In each histogram, four approaches are reported: Dense, IMP-Refill, IMP-Refill+, and IMP-Regroup. Dense as the compared baseline with zero time saving, so the corresponding bars are always unseen from the charts.

［#160］
![](./images/867754386212782437_11.jpg)

［#161］
Figure A11. Time saving (%), sparsity level (%), and test accuracy (%) of various models {ResNet-18, VGG-19, ResNet-50} on different datasets {CIFAR-10/100, Tiny ImageNet}.

［#162］
Comparison to NAS and SOTA structure pruning methods. We conduct extra experiments with a neural architecture search (NAS) based approach, i.e., ABC pruner (Lin et al., 2020). Specifically, it searches the channel numbers per layer, and then the derived structure will be trained from the same initialization. At the same sparsity level 77%, our IMP-Refill surpasses the ABC pruner by 1.5% accuracy, which demonstrates the superiority of located refill tickets. More sparse levels are presented in Fig. A12 (C1).

［#163］
Under the same setup of training from scratch, we further compare our proposal with SCOP (Tang et al., 2020) and observe that ours $(94.16 \sim 94.62\%)$ outperform SCOP $(93.48\%)$ by up to 1.14% accuracy at $\sim 41\%$ sparsity. More sparse levels are collected in Fig. A12 (C2).

［#164］
![](./images/867754386212782437_12.jpg)

［#165］
Figure A12. Comparisons with NAS and SOTA structural pruning.