# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#1］
Grzegorz Stefański $^{1}$ Alberto Presta $^{1}$ Michał Byra $^{12}$

## Abstract
［#2］
In pruning, the Lottery Ticket Hypothesis posits that large networks contain sparse subnetworks, or winning tickets, that can be trained in isolation to match the performance of their dense counterparts. However, most existing approaches assume a single universal winning ticket shared across all inputs, ignoring the inherent heterogeneity of real-world data. In this work, we propose Routing the Lottery (RTL), an adaptive pruning framework that discovers multiple specialized subnetworks, called adaptive tickets, each tailored to a class, semantic cluster, or environmental condition. Across diverse datasets and tasks, RTL consistently outperforms single- and multi-model baselines in balanced accuracy and recall, while using up to 10 times fewer parameters than independent models and exhibiting semantically aligned. Furthermore, we identify subnetwork collapse, a performance drop under aggressive pruning, and introduce a subnetwork similarity score that enables label-free diagnosis of over-sparsification. Overall, our results recast pruning as a mechanism for aligning model structure with data heterogeneity, paving the way toward more modular and context-aware deep learning.

## 1. Introduction
［#3］
Despite the remarkable progress deep neural networks have achieved over the past decades, modern models often require billions of parameters and hundreds of gigaflops per inference, making them impractical for deployment in resource-constrained or real-time settings.

［#4］
This inefficiency stands in stark contrast to biological intelligence, which achieves high performance with extreme parsimony. While specialized hardware continues to advance, the growth in model scale consistently outpaces gains in computational efficiency. Consequently, reducing model complexity is not only crucial for practical deployment but also for understanding the fundamental principles of generalization in deep learning.

［#5］
Pruning has been a cornerstone of model efficiency research (Cheng et al., 2024). By removing redundant weights or structures, it aims to uncover smaller subnetworks that retain the performance of their dense counterparts. The Lottery Ticket Hypothesis (LTH) (Frankle & Carbin, 2019) revitalized this line of work by positing that "winning tickets", i.e. sparse subnetworks within large, randomly initialized networks, can be trained in isolation to match full-model accuracy. This reframed pruning as a discovery process aiming to reveal the minimal structures responsible for learning.

［#6］
However, nearly all LTH-inspired methods assume a universal subnetwork - a single sparse mask applied uniformly across all inputs, overlooking real-world data heterogeneity. Different classes, clusters, or environmental conditions often rely on distinct feature representations. A one-size-fits-all mask may therefore sacrifice performance by forcing diverse patterns through a shared, rigid architecture. Bridging this gap requires moving beyond global sparsity toward adaptive, data-aware pruning, a shift that aligns model structure with the intrinsic organization of the data itself.

［#7］
In this work, we introduce *Routing the Lottery* (RTL), a novel adaptive pruning framework that rethinks LTH by discovering multiple specialized subnetworks, dubbed *adaptive tickets*, each tailored to a distinct data subset (e.g., a class or semantic cluster). This shift enables the model to allocate representational capacity heterogeneously across the input space, aligning sparsity with data structure rather than enforcing uniform compression. Our framework achieves specialization through pruning alone, without auxiliary routing networks or additional parameters. It discovers multiple stable, sparse subnetworks, each tied to a class or cluster, and selects them via simple context-based routing (e.g., label or environment), offering a lightweight and interpretable alternative to Mixture-of-Experts (MoE) architectures that prioritizes structural efficiency over dynamic routing flexibility.

［#8］
This work serves as a precursor, aiming to guide pruning from a static tool into a dynamic mechanism for building modular and semantically grounded models. The major

［#8］
$^{1}$Samsung AI Center Warsaw, Poland $^{2}$Institute of Fundamental Technological Research, Polish Academy of Sciences, Poland. Correspondence to: Grzegorz Stefański <g.stefanski@samsung.com>.

［#9］
Preprint. January 30, 2026.


# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#10］
contributions of this paper are:

［#11］
- RTL jointly learns multiple sparse subnetworks from a shared dense initialization, with masks adapted to data subsets while preserving parameter sharing, maintaining a single compact backbone and employing a mask-based routing to enable context-aware inference.
- We show that class-specific subnetworks on CIFAR-10 outperform both single-mask and multi-model pruning baselines while using up to an order of magnitude fewer parameters than independent models, and that on CIFAR-100, RTL scales naturally to enable effective specialization across a larger number of classes.
- We validate RTL on a real-world speech enhancement task, where subnetworks specialized for acoustic environments achieve higher SI-SNRi than universal or independent baselines.

## 2. Related Work

［#12］
Pruning has a rich history, beginning with sensitivity-based methods like Optimal Brain Damage (LeCun et al., 1989) and Optimal Brain Surgeon (Hassibi et al., 1993). In particular, unstructured pruning removes non-relevant weights without imposing structural constraints, i.e., without removing entire layers. (Han et al., 2015) performed pruning by learning meaningful connections, while (Yang et al., 2017; 2018) exploit energy consumption as a metric for removing elements. (Sreenivasan et al., 2022; Tartaglione et al., 2022) introduced regularization terms to constrain the magnitude of non-relevant parameters during iterative training, while (Benbaki et al., 2023) considered the combined effect of pruning and updating multiple weights under a sparsity constraint. Furthermore, (Tartaglione et al., 2020) compared and analyze two different pruning approaches, i.e. one-shot and gradual, highlighting that the latter allows for better generalization. (Zhang et al., 2024) analyzed fundamental aspects of pruning, identifying two key factors that determine the pruning ratio limit, i.e., weight magnitude and network sharpness, while (Liao et al., 2023; Hur & Kang, 2019; Luo & Wu, 2017; Min et al., 2018) proposed entropy-based approaches to guide pruning. In general, owing to the conceptual simplicity of pruning, a wide range of methods have been proposed for different scenarios, such as LLM pruning (Frantar & Alistarh, 2023; Sun et al., 2023; Lu et al., 2024; Wei et al., 2024; Tan et al., 2024), convolutional pruning (Zhao et al., 2023), and even spiking neural networks (Shi et al., 2024). However, none of these have considered whether finding a single mask applicable to all types of data might be a suboptimal solution.

［#13］
Iterative Magnitude Pruning (IMP) (Frankle & Carbin, 2019) emerged as a practical algorithm for identifying winning tickets, though it remains computationally intensive due to repeated training cycles. Furthermore, (Paul et al., 2022) attempted to demystify the IMP method, investigating what kind of information the obtained mask encodes and how SGD allows the network to extract such information. Despite numerous extensions covering initialization schemes (Frankle et al., 2021), learning rate rewinding (Renda et al., 2020), theoretical analysis (Tartaglione, 2022; Burkholz et al., 2021; Sakamoto & Sato, 2022; Paul et al., 2022), and algorithmic variants (Wang et al., 2023; Lin et al., 2023), the LTH paradigm has largely focused on global masks shared across the entire dataset.

［#14］
Dynamic sparse training methods relax the fixed-mask assumption by evolving sparsity patterns during training. Approaches such as SET (Mocanu et al., 2017), SNFS (Dettmers & Zettlemoyer, 2020), and RigL (Evci et al., 2020) prune and regrow connections online, implicitly acknowledging that different inputs may activate different network pathways. Furthermore, (Molchanov et al., 2017) extended variational dropout in order to sparsify deep neural networks, while (Tartaglione et al., 2021) introduced sensitivity-based regularization of neurons to learn structured sparse topologies, exploiting neural sensitivity as a regularizer. Nevertheless, these methods still maintain a single evolving subnetwork rather than explicitly specializing distinct structures for different data regimes.

［#15］
Closer in spirit are conditional computation techniques like MoE (Shazeer et al., 2017; Fedus et al., 2022) and conditional convolutions (Yang et al., 2019), which activate input-dependent subnetworks to improve scalability. However, these methods typically require complex routing mechanisms, large auxiliary parameter sets, and substantial compute budgets.

## 3. Method

［#16］
We propose an adaptive pruning framework that extends the LTH to support multiple specialized subnetworks. An overview of the method is shown in Fig. 1. Rather than identifying a single "winning ticket", our approach discovers distinct subnetworks, each tailored to a specific data cluster.

［#17］
Let $f(x; \theta)$ denote a neural network parameterized by $\theta \in \mathbb{R}^d$, $\mathcal{T}$ be a given task, and $\mathcal{D}$ a general dataset. In standard pruning, a binary mask $m \in \{0, 1\}^d$ selects a subnetwork $f_m$ via element-wise multiplication:
［#17］
$$
f_m = f(x; m \odot \theta), \tag{1}
$$
［#17］
where $\odot$ denotes the Hadamard product. The LTH posits that there exists a mask $m^*$ such that the corresponding subnetwork, when trained in isolation from random initialization, matches the performance of the dense model, i.e.:
［#17］
$$
\mathcal{L}_{f_m}(\mathcal{D}, \mathcal{T}) = \mathcal{L}_f(\mathcal{D}, \mathcal{T}), \tag{2}
$$

# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#18］
![](./images/1223972462526464001_1.jpg)

［#19］
Figure 1. Adaptive pruning pipeline. First, the dataset is divided into subsets via predefined clustering. Then we extracts adaptive tickets, i.e. subnetworks, optimized to specific data cluster, and finally we performed network joint retraining.

［#20］
where $\mathcal{L}_{\mathcal{.}}$ denotes the model's performance on task $\mathcal{T}$ and dataset $\mathcal{D}$.

［#21］
While prior work seeks a single universal mask $m^*$, we hypothesize that different data subsets $\mathcal{D}_k$ benefit from distinct, pruned subnetworks. The remainder of this section is organized as follows: Sec. 3.1 formalizes the adaptive pruning objective, Sec. 3.2 describes adaptive mask extraction, and Sec. 3.3 presents the joint retraining procedure.

## 3.1. Formulation of the adaptive pruning algorithm

［#22］
Given dataset $\mathcal{D}$ and task $\mathcal{T}$, we partition $\mathcal{D}$ into $K$ subsets $\{\mathcal{D}_1, \dots, \mathcal{D}_K\}$ corresponding to classes or clusters. Such labels can be obtained from supervised labeling systems, such as class annotations, or derived through unsupervised clustering, enabling flexible adaptation across settings.

［#23］
For each subset $\mathcal{D}_k$, our method involves learning a dedicated mask $m_k$, yielding a subnetwork:
［#23］
$$
\varphi_{m_k} = f(x; m_k \odot \theta), \quad \text{for } x \in \mathcal{D}_k. \tag{3}
$$

［#24］
Given this setup, our objective is to jointly learn the set of masks $\mathcal{M}^* = \{m_1^*, \dots, m_K^*\}$ that maximize predictive performance on task $\mathcal{T}$ under a sparsity constraint:
［#24］
$$
\mathcal{L}_{\mathcal{M}^*} = \min_{m_k \in \mathcal{M}^*} \sum_{k=1}^K \mathbb{E}_{(x,y) \sim \mathcal{D}_k} \left[\ell\left(y, f\left(x; m_k \odot \theta\right)\right)\right] \tag{4}
$$
［#24］
$$
\text{subject to } \quad \|m_k\|_0 \leq s \quad \forall k, \tag{5}
$$
［#24］
where $\ell$ denotes the loss function and $s$ controls the maximum number of non-zero parameters per subnetwork.

## 3.2. Adaptive tickets extraction

［#25］
We now describe how adaptive tickets are extracted for each data subset (see Alg. 1). Starting from a random initialization $f(x; \theta_0)$, we create an equal-size binary mask $m_k$ for each of the $K$ target subsets, initialized to ones. The dataset is then partitioned into $K$ disjoint subsets $\mathcal{D}_K = \{d_1, ..., d_K\}$ using a predefined rule, like manual labeling or automatic clustering. Importantly, RTL is fully agnostic with respect to how these subsets are defined, allowing flexibility across diverse tasks and data modalities.

［#26］
```plaintext
Algorithm 1 Adaptive ticket extraction
Input: Network $f(x; \theta_0)$, train data $\mathcal{D}$, number of subsets $K$,
target sparsity $s$, pruning factor $p$, steps $T$, training epoch $N$.
Output: Set of $K$ mask $\mathcal{M}$.

［#27］
Random initialization of $f(x; \theta_0)$.
［#28］
$m_k \leftarrow \mathbf{1}^{|\theta_0|}$ for $k = 1, ..., K$.
［#29］
$\mathcal{D}_K = \{d_1, ..., d_K\} \leftarrow$ partition of $\mathcal{D}$ into $K$ subsets.
［#30］
$\theta \leftarrow \theta_0$
［#30］
while density$(m_K) < s$ do
  for $k = 1$ to $K$ do
［#30］
    $d_k \leftarrow$ $k$-th subset from $\mathcal{D}_K$
［#30］
    $f(\cdot, \theta_T^{(k)}) \leftarrow$ optimize $f$ for $T$ steps on $d_k$
［#30］
    $m_k \leftarrow$ pruning $f$ with factor $p$
［#30］
    $\theta \leftarrow \theta_0$.
  end for
end while
```

［#31］
Each pruning iteration proceeds sequentially over all subsets. For a given subset $d_k$, the network $f(x; \theta_0)$ is trained for $T$ steps to obtain temporary parameters $\theta_T^{(k)}$. We then perform pruning on $f$, removing the lowest-magnitude weights from $\theta_T^{(k)}$ by a fraction $p$. From the set of removed parameters, we yield a sparse subnetwork defined by the mask $m_k$. The remaining weights are subsequently reset to their initial values $\theta_0$. This process repeats until each mask reaches the desired sparsity level $s$, producing the final set of $K$ adaptive masks $\mathcal{M} = \{m_1, ..., m_K\}$. Each mask defines a specialized subnetwork $f(x; m_k \odot \theta_0)$.

## 3.3. Joint Retraining of adaptive tickets

［#32］
After obtaining the sparse masks $\mathcal{M} = \{m_1, ..., m_K\}$, we perform a lightweight joint retraining phase to refine subnetwork performance and reinforce specialization. This step, outlined in Alg. 2, is crucial in our proposed methodology, since it preserves the sparsity structure discovered during mask extraction (alg. 1) and fine-tunes only the active weights without altering mask topology.

［#33］
The process begins with balanced dataset creation. As before, the training set $\mathcal{D}$ is partitioned into $K$ subsets $\mathcal{D}_K = \{d_1, ..., d_K\}$, each corresponding to a target class or cluster. Each subset $d_k$ is divided into mini-batches $\mathcal{B}^k = \{B_1^k, ..., B_{M_k}^k\}$. To ensure synchronized updates across subnetworks, we repeat batches from smaller subsets cyclically until all subsets contain the same number of batches $M = \max(M_1, ..., M_K)$. This balancing step guarantees that each subnetwork receives the same number of gradient updates per epoch, even if subsets differ in size.


### Algorithm 2 Joint retraining of the adaptive tickets
［#34］
**Input:** Network $f(x;\theta)$, train data $\mathcal{D}$, number of subsets $K$, training epochs $N$, learning rate $\eta$, mask set $\mathcal{M}$.
**Output:** $K$ optimized adaptive tickets $f_k(\cdot;\theta_k)$.

［#35］
**Balanced dataset creation**
［#35］
$\mathcal{D}_K = \{d_1, ..., d_K\} \leftarrow$ partition of $\mathcal{D}$ into $K$ subsets.
for $k = 1$ to $K$ do
［#35］
  $d_k \leftarrow k$-th subset from $\mathcal{D}_K$
［#35］
  $\mathcal{B}^k \leftarrow \{B_1^k, ..., B_{M_k}^k \}$ {Set of batches from $d_k$}
［#35］
  $M_k \leftarrow |\mathcal{B}^k|$
［#35］
end for
［#35］
$M \leftarrow \max(M_1, ..., M_K)$.
for $k = 1$ to $K$ do
  Repeat batches in $\mathcal{B}^k$ cyclically until $|\mathcal{B}^k| = M$.
end for

［#36］
**Model training**
for $j = 1$ to $N$ do {Training epochs}
  for $m = 1$ to $M$ do {Batch indices}
    for $k = 1$ to $K$ do {Subnetwork indices}
［#36］
      $m_k \leftarrow \mathcal{M}[k]$.
［#37］
      $f_k(\cdot;\theta_k) \leftarrow f(\cdot; m_k \odot \theta)$ {$k$-th subnetwork}
［#37］
      $(x_m^k, y_m^k) \leftarrow B_m^k$. {Input and label }
［#37］
      $\theta_k \leftarrow \theta_k - \eta\left(\nabla_\theta \mathcal{L}(f_k(x_m^k;\theta_k), y_m^k) \odot m_k\right)$.
    end for
  end for
end for

［#38］
During joint retraining, each subnetwork $f(x; m_k \odot \theta)$ is trained exclusively on its corresponding data subset $d_k$. We interleave mini-batches from different subsets and apply gradient updates to the shared dense parameter tensor $\theta$, masking out gradients for pruned weights. Specifically, for subnetwork $k$, the parameter update is given by:

［#38］
$$
f_k(\cdot, \theta_k) \leftarrow f(\cdot, m_k \odot \theta) \tag{6}
$$

［#38］
$$
\theta_k \leftarrow \theta-\eta\left(\nabla_\theta \mathcal{L}(f_k(x^k;\theta_k), y^k) \odot m_k\right) \tag{7}
$$

［#38］
where $\eta$ is the learning rate, $\mathcal{L}$ denotes the empirical loss, while $(x^k, y^k)$ represents the input data with its corresponding label coming from the $k$-th subset. By masking the gradient as in Eq. 7, only the weights retained by mask $m_k$ are updated, preventing interference between subnetworks and avoiding both catastrophic forgetting and collapse. Maintaining limited overlap between subnetworks is crucial for ensuring optimal performance within each cluster without mutual interference or cancellation.

## 4. Experiments

［#39］
We present a sequence of experiments that progressively validate our adaptive pruning framework, moving from controlled settings to real-world applications: (i) class-specific subnetworks on CIFAR-10 (Krizhevsky, 2009), (ii) cluster-aware pruning on CIFAR-100, (iii) implicit neural representations (INRs) with within-image semantic specialization, (iv) speech enhancement in heterogeneous acoustic environments, and (v) subnetwork overlap and semantic alignment analysis.

［#40］
Across all tasks, we control for architecture, sparsity, and training budget to isolate the effects of specialization. Quantitative results and discussion are reported in Sec. 5, while full implementation details are provided in the appendix.

### 4.1. CIFAR-10 subnetwork specialization

［#41］
We begin with a controlled CIFAR-10 experiment, where each of the 10 classes defines a distinct data subset and is assigned its own subnetwork. This idealized setting allows us to directly test the core premise of our approach: whether subnetworks specialized to disjoint subsets outperform a single universal pruning mask under identical constraints.

［#42］
We compare RTL against two baselines: (i) IMP (single model), which produces a single shared subnetwork across all classes, and (ii) IMP (multiple models), which independently prunes one model per class without weight sharing.

［#43］
All methods use the same backbone architecture, sparsity targets, and pruning budget. Performance is evaluated using balanced accuracy, precision, recall, and parameter count at 25%, 50%, and 75% sparsity. This experiment establishes an upper bound on specialization benefits and serves as a reference point for more challenging scenarios. Training protocols and architectural details are provided in Appendix A.

### 4.2. Cluster-aware pruning on CIFAR-100

［#44］
To assess robustness under less ideal conditions, we evaluate RTL on CIFAR-100, where class boundaries are fine-grained and semantically overlapping. Instead of assigning subnetworks to individual classes, we group the 100 classes into 8 coarse semantic clusters using an unsupervised text-based clustering procedure.

［#45］
The resulting clusters are semantically coherent but imperfectly aligned with visual features, introducing ambiguity that better reflects real-world data partitioning. RTL is compared against the same two IMP baselines under matched conditions, using the same metrics and sparsity levels as in the CIFAR-10 experiment. Details of the clustering pipeline are provided in Appendix B.

### 4.3. Implicit Neural Representations

［#46］
We evaluate RTL in the context of INRs, where specialization is defined over semantic regions within a single image. The task consists of reconstructing an image by mapping continuous pixel coordinates to RGB values using a coordinate-based neural network.

［#47］
Experiments are conducted on 10 images from the ADE20K dataset (Zhou et al., 2019) selected to ensure diversity in


［#47］
scene category and luminosity/color characteristics. Semantic segmentation masks define region-level classes, and RTL learns specialized subnetworks for these regions, while a standard baseline conditions a single network on region identity via class embeddings.

［#48］
Reconstruction quality is evaluated using peak signal-to-noise ratio (PSNR), averaged over all pixels and all images. Architectural choices, positional encoding, training protocol, and per-image results are reported in Appendix C.

### 4.4. Speech enhancement in realistic environments

［#49］
We evaluate RTL on a real-world speech enhancement task characterized by heterogeneous acoustic conditions. Clean speech is mixed with noise from three distinct acoustic scenes (indoor, outdoor, and transportation), each defining a specialization subset.

［#50］
RTL learns one subnetwork per acoustic scene and is compared against (i) a single IMP-pruned model shared across all environments and (ii) independently pruned IMP models without weight sharing. All methods operate under identical sparsity and computational constraints. Performance is measured using scale-invariant signal-to-noise ratio improvement (SI-SNRi). Dataset construction, model architecture, and signal processing details are provided in Appendix D.

### 4.5. Subnetwork collapse and semantic alignment

［#51］
Beyond task-level performance, we analyze relationships among learned subnetworks in the CIFAR-10 and CIFAR-100 experiments. Specifically, we study how subnetwork overlap evolves with increasing sparsity and how excessive overlap relates to performance degradation, a phenomenon we refer to as subnetwork collapse.

［#52］
We quantify pairwise mask similarity at multiple sparsity levels and examine its correlation with balanced accuracy. Finally, using CIFAR-10, we compare structural similarity with semantic distances between class labels derived from WordNet, assessing whether pruning structures encode high-level conceptual relationships. Formal definitions and analysis procedures are provided in Appendix E.

## 5. Results

### 5.1. CIFAR-10 subnetwork specialization

［#53］
Table 1 summarizes CIFAR-10 results under class-specific pruning. RTL consistently achieves the highest balanced accuracy and recall across all sparsity levels. At 25% sparsity, RTL attains a balanced accuracy of 0.781, significantly outperforming both baselines (0.711 for single-model IMP and 0.712 for multi-model IMP). This advantage persists at 50% sparsity (0.778 vs. 0.711) and remains competitive even at 75% sparsity (0.772 vs. 0.760 for multi-model IMP), despite using an order of magnitude fewer parameters.

［#54］
RTL also achieves the highest recall at all sparsity levels (0.821 / 0.810 / 0.816), exceeding the single-model baseline and matching or surpassing the multi-model alternative. This demonstrates that RTL subnetworks effectively preserve class-relevant signals under aggressive pruning.

［#55］
The lower precision of RTL (0.257-0.282) compared to the single-model IMP baseline (0.478-0.515) reflects a design trade-off: by prioritizing recall, RTL subnetworks favor sensitivity to true positives over strict discrimination, which is well-suited for class-specialized inference. In downstream applications, precision can be recovered via thresholding or ensemble methods if needed.

［#56］
Table 1. Results on CIFAR-10 with class-specific subnetworks and CIFAR-100 with cluster-specialized subnetworks. We report balanced accuracy, precision, recall, and the number of remaining parameters at three sparsity levels. RTL is compared against (a) IMP with a single shared subnetwork and (b) IMP trained independently per class (multiple models). Best values per metric and sparsity level are bolded. Clusters are derived from semantic embeddings (see Section 4).

［#57］
<table>
  <thead>
    <tr>
      <th colspan="2" rowspan="2">Method</th>
      <th colspan="3">Balanced accuracy $\uparrow$</th>
      <th colspan="3">Precision $\uparrow$</th>
      <th colspan="3">Recall $\uparrow$</th>
      <th colspan="3">#Params $\downarrow$</th>
    </tr>
    <tr>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">CIFAR-10</td>
      <td>RTL (ours)</td>
      <td><b>0.781</b></td>
      <td><b>0.778</b></td>
      <td><b>0.772</b></td>
      <td>0.282</td>
      <td>0.276</td>
      <td>0.257</td>
      <td><b>0.821</b></td>
      <td><b>0.810</b></td>
      <td><b>0.816</b></td>
      <td>103K</td>
      <td>72K</td>
      <td>38K</td>
    </tr>
    <tr>
      <td>IMP (single model)</td>
      <td>0.711</td>
      <td>0.711</td>
      <td>0.732</td>
      <td><b>0.479</b></td>
      <td><b>0.478</b></td>
      <td><b>0.515</b></td>
      <td>0.480</td>
      <td>0.480</td>
      <td>0.518</td>
      <td><b>94K</b></td>
      <td><b>63K</b></td>
      <td><b>31K</b></td>
    </tr>
    <tr>
      <td>IMP (multiple models)</td>
      <td>0.712</td>
      <td>0.710</td>
      <td>0.760</td>
      <td>0.233</td>
      <td>0.222</td>
      <td>0.262</td>
      <td>0.701</td>
      <td>0.702</td>
      <td>0.766</td>
      <td>944K</td>
      <td>629K</td>
      <td>314K</td>
    </tr>
    <tr>
      <td rowspan="3">CIFAR-100</td>
      <td>RTL (ours)</td>
      <td><b>0.765</b></td>
      <td><b>0.751</b></td>
      <td><b>0.759</b></td>
      <td>0.298</td>
      <td>0.290</td>
      <td>0.289</td>
      <td><b>0.764</b></td>
      <td><b>0.729</b></td>
      <td><b>0.754</b></td>
      <td>108K</td>
      <td>76K</td>
      <td>40K</td>
    </tr>
    <tr>
      <td>IMP (single model)</td>
      <td>0.722</td>
      <td>0.707</td>
      <td>0.742</td>
      <td><b>0.45</b></td>
      <td><b>0.463</b></td>
      <td><b>0.522</b></td>
      <td>0.42</td>
      <td>0.421</td>
      <td>0.463</td>
      <td><b>94K</b></td>
      <td><b>63K</b></td>
      <td><b>31K</b></td>
    </tr>
    <tr>
      <td>IMP (multiple models)</td>
      <td>0.712</td>
      <td>0.700</td>
      <td>0.744</td>
      <td>0.276</td>
      <td>0.271</td>
      <td>0.286</td>
      <td>0.660</td>
      <td>0.637</td>
      <td>0.732</td>
      <td>944K</td>
      <td>629K</td>
      <td>314K</td>
    </tr>
  </tbody>
</table>

# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#58］
Table 2. INR results on ADE20k dataset. Methods are evaluated at 25%, 50%, and 75% sparsity using PSNR, reporting the number of trainable parameters.

［#59］
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">PSNR $\uparrow$</th>
      <th colspan="3">#Params $\downarrow$</th>
    </tr>
    <tr>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RTL (ours)</td>
      <td><b>18.86</b></td>
      <td><b>17.25</b></td>
      <td><b>14.87</b></td>
      <td>48.0K</td>
      <td>36.5K</td>
      <td>22.0K</td>
    </tr>
    <tr>
      <td>IMP<br>(single model)</td>
      <td>15.94</td>
      <td>14.72</td>
      <td>12.69</td>
      <td><b>40.5K</b></td>
      <td><b>27.0K</b></td>
      <td><b>13.5K</b></td>
    </tr>
  </tbody>
</table>

［#60］
Critically, these gains come with exceptional parameter efficiency. RTL uses only 103K / 72K / 38K parameters at 25% / 50% / 75% sparsity, compared to 944K / 629K / 314K for multi-model IMP (nearly 10× more) to achieve comparable accuracy. This underscores RTL's ability to deliver high performance without sacrificing compactness.

## 5.2. Cluster-aware pruning on CIFAR-100

［#61］
Table 1 also reports CIFAR-100 results, where subnetworks are specialized to semantically derived clusters. Despite the inherent ambiguity in this partitioning, RTL again achieves the highest balanced accuracy across all sparsity levels: 0.765 (25%), 0.751 (50%), and 0.759 (75%). Notably, RTL outperforms both baselines even at 75% sparsity, where model capacity is most constrained, demonstrating robustness to imperfect data grouping.

［#62］
Recall follows the same trend: RTL achieves 0.764 / 0.729 / 0.754, substantially exceeding the single-model IMP baseline (0.420-0.463) and consistently outperforming the multi-model alternative. This confirms that RTL subnetworks retain cluster-discriminative features more effectively under pruning, even when cluster boundaries are noisy.

［#63］
As in CIFAR-10, RTL exhibits lower precision than the single-model baseline, reflecting its emphasis on sensitivity over selectivity. Given the coarse and overlapping nature of the clusters, high recall is particularly valuable for ensuring coverage of relevant visual concepts.

［#64］
In terms of efficiency, RTL uses only 108K / 76K / 40K parameters, comparable to the single-model model baseline and drastically fewer than the multi-model baseline (944K / 629K / 314K). This reaffirms that RTL achieves strong specialization without requiring redundant, over-parameterized subnetworks, making it especially suitable for scenarios with limited memory or compute.

## 5.3. Implicit Neural Representations

［#65］
Table 2 reports reconstruction performance for the INR experiment on ADE20K images at varying sparsity levels.

［#66］
Across all sparsity regimes, RTL consistently outperforms the standard IMP baseline. At 25% sparsity, RTL achieves a PSNR of 18.58, exceeding the single-model IMP baseline by almost 3 dB. This performance gap remains substantial as sparsity increases, with RTL maintaining advantages of 2.53 dB and 2.18 dB at 50% and 75% sparsity, respectively.

［#67］
Notably, these gains are achieved despite RTL retaining a larger number of parameters than the single-model IMP baseline. This behavior mirrors observations from the classification and speech enhancement experiments: enforcing a single global pruning mask across heterogeneous subsets (in this case, semantically distinct image regions) leads to suboptimal allocation of model capacity. By contrast, RTL enables region-specific subnetworks that preserve functionally important parameters, resulting in higher reconstruction fidelity under identical computational budgets.

［#68］
As sparsity increases, performance degrades for both methods, but RTL degrades more gracefully, indicating that specialization is especially beneficial in high-sparsity regimes, where competition for shared parameters intensifies. The results demonstrate that adaptive pruning can effectively capture semantic structure even in coordinate-based representations, extending the benefits of specialization beyond dataset-level tasks to within-image semantic decomposition.

［#69］
Per-image PSNR values are reported in Appendix C, confirming that the observed trends are consistent across images with diverse scene content and appearance characteristics.

## 5.4. Speech Enhancement in realistic environments

［#70］
Table 3 reports speech enhancement performance across three acoustic environments (indoor, outdoor, and transportation). RTL achieves the highest SI-SNRi at all sparsity levels: 7.248, 7.178, and 6.992 at 25%, 50%, and 75% sparsity, respectively, consistently outperforming both IMP baselines. This demonstrates that environment-specific subnetworks better capture each noise type's distinct spectro-temporal characteristics, yielding superior waveform reconstruction.

［#71］
The single-mask IMP model performs reasonably well, likely by learning a general-purpose denoising strategy. However, it lags behind RTL, confirming that specialization

［#72］
Table 3. Speech enhancement results on DNS Challenge 2020 and TAU Urban Acoustic Scenes 2020 datasets. Methods are evaluated at 25%, 50%, and 75% sparsity using SI-SNRi, reporting the number of trainable parameters.

［#73］
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">SI-SNRi $\uparrow$</th>
      <th colspan="3">#Params $\downarrow$</th>
    </tr>
    <tr>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RTL (ours)</td>
      <td><b>7.248</b></td>
      <td><b>7.178</b></td>
      <td><b>6.992</b></td>
      <td>32.0K</td>
      <td>22.8K</td>
      <td>12.3K</td>
    </tr>
    <tr>
      <td>IMP<br>(single model)</td>
      <td>6.885</td>
      <td>6.97</td>
      <td>6.967</td>
      <td><b>28.0K</b></td>
      <td><b>18.6K</b></td>
      <td><b>9.3K</b></td>
    </tr>
    <tr>
      <td>IMP<br>(multiple model)</td>
      <td>5.295</td>
      <td>5.775</td>
      <td>5.899</td>
      <td>84.1K</td>
      <td>56.1K</td>
      <td>28.0K</td>
    </tr>
  </tbody>
</table>

［#74］
Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#74］
yields tangible gains even in a non-classification task. In contrast, the multi-model IMP baseline underperforms despite a larger parameter budget (84.1K vs. 32.0K at 25% sparsity), suggesting that independent pruning without shared initialization or joint retraining yields suboptimal results.

［#75］
Critically, RTL achieves these gains with only modest overhead in model size: 32.0K / 22.8K / 12.3K parameters, just slightly larger than the single-mask model and less than half the size of the multi-model alternative. Notably, even the most aggressively pruned RTL subnetwork (12.3K parameters) surpasses both baselines in SI-SNRi, underscoring the efficiency of our approach.

［#76］
These results confirm that RTL generalizes beyond controlled vision benchmarks: when subnetworks align with meaningful real-world structure (in this case, acoustic environments), they deliver significantly better performance while remaining compact and deployable.

### 5.5. Subnetworks collapse and semantic alignment

［#77］
Subnetworks collapse Fig. 2 illustrates the relationship between performance and mask similarity across sparsity levels for CIFAR-10 and CIFAR-100. For each class or cluster, we plot (i) balanced accuracy and (ii) average Jaccard similarity (IoU) to all other subnetworks. These curves reveal the onset of subnetwork collapse - a failure mode in which excessive pruning forces subnetworks to converge to overlapping weight sets, eroding their specialization.

［#78］
In CIFAR-10 (Fig. 2A-J), where classes are perfectly separated, most subnetworks maintain high accuracy and low mask similarity up to 70-80% sparsity. Beyond this point, a sharp increase in IoU coincides with a precipitous drop in accuracy, confirming that performance degradation is directly linked to loss of structural distinctiveness. Crucially, the IoU spike reliably precedes or coincides with the accuracy drop, suggesting that mask similarity can serve as a label-free early-warning signal for oversparsification.

［#79］
On CIFAR-100 (Fig. 2K-S), where classes are grouped into eight semantically derived clusters, the same collapse signature emerges: accuracy remains stable as long as mask similarity is bounded, but deteriorates sharply once pruning forces subnetworks into excessive overlap. Collapse occurs at slightly higher sparsity thresholds than in CIFAR-10, likely because clusters share more visual features, allowing for greater weight reuse before specialization is lost.

［#80］
Across both datasets, the evidence is consistent: subnetwork specialization is essential for RTL's performance, and mask similarity is a robust, label-free predictor of collapse. Moreover, the abruptness of the accuracy drop suggests that once specialized weights are pruned beyond a critical point, recovery is unlikely without full retraining, highlighting the importance of stopping pruning before collapse occurs.

［#81］
Semantic alignment Fig. 3 summarizes how semantic alignment relates to subnetwork structure throughout training and across depth. Panels A-D report Spearman's rank-order correlations between semantic proximity (from WordNet) and mask similarity (Jaccard index), while panel E shows the corresponding WordNet and mask similarity matrices for shallow and deep layers.

［#82］
Panel A shows pruning correlations across layer depths: shallow and middle layers increase correlation up to 20-35% sparsity, then decline, indicating that moderate pruning enhances semantic organization in early layers, while excessive sparsity degrades it. Deep layers start with low correlations but rise steadily to around 55% sparsity, reflecting stronger semantic alignment as higher-level representations develop.

［#83］
Panel B tracks training evolution: early optimization shows weak correlations decreasing with depth, but later epochs reveal consistent increases, particularly in middle and deep layers, indicating a gradual emergence of semantic structure as subnetworks refine specialized connectivity.

［#84］
Panels C and D highlight four representative classes (*air-

［#85］
![](./images/1223972462526464001_2.jpg)

［#86］
Figure 2. Mask collapse analysis on CIFAR-10 and CIFAR-100. Each subplot corresponds to one class and shows balanced accuracy (solid line) and mask similarity to other subnetworks (dashed line). Plots A-J corresponds to CIFAR-10 network and plots K-S to CIFAR-100.

# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#87］
![](./images/1223972462526464001_3.jpg)

［#88］
Figure 3. Semantic and structural correlation analysis. (A) Spearman's rank-order correlation between semantic similarity and mask similarity versus pruning ratio across shallow, middle, and deep layers. (B) Correlation across depth for early, middle, and late training stages. (C) Correlation versus pruning ratio for four representative classes: airplane, cat, deer, and truck. (D) Correlation across depth for the same four classes. (E) WordNet path similarity (top) and RTL mask similarity matrices for shallow (middle) and deep (bottom) layers.

［#89］
plane, cat, deer, truck). Subnetworks for semantically related classes (cat, deer) show stronger correlations (0.6-0.7) that grow with pruning and depth, while unrelated ones (airplane, truck) remain weaker (0.2-0.3). Thus, RTL subnetworks for related categories preserve overlapping structures, whereas distant ones evolve independently - a pattern consistent with overall mask stability across dissimilar classes.

［#90］
Panel E provides complementary heatmaps. The top matrix shows WordNet path similarities, followed by mask similarities for shallow and deep layers. Shallow masks are uniformly similar (mean $\approx 0.86$), reflecting shared early filters for low-level, class-agnostic features such as edges and textures. Slight local increases for related pairs, for example cat-dog (0.89), automobile-truck (0.87), and deer-horse (0.88), suggest broad semantic grouping even at early stages. In deep layers, mask similarity drops (mean $\approx 0.31$), and a clear block-diagonal structure emerges that aligns with WordNet relations. Related animal classes (cat, dog, horse, deer) show higher overlap (0.33-0.36) than unrelated ones (0.27-0.31). RTL forms weaker specialization among vehicle classes (airplane, automobile, ship, truck), likely due to limited visual similarity beyond the automobile-truck pair.

［#91］
Overall, RTL subnetworks gradually organize according to semantic structure in the data. Early layers are shared and class-agnostic, while deeper layers increasingly reflect conceptual hierarchies. The growing correlation with both depth and training indicates that RTL pruning not only enforces sparsity but also promotes semantically meaningful specialization within a shared model architecture.

## 6. Conclusion

［#92］
We introduced Routing the Lottery (RTL), an adaptive pruning framework that discovers multiple specialized subnetworks (adaptive tickets) rather than a single universal winning ticket. Across multiple settings RTL consistently achieves superior or competitive performance while maintaining a compact parameter footprint. These results show that specialization naturally emerges when subnetworks are allowed to diverge in response to data heterogeneity.

［#93］
Our analysis reveals two critical insights. First, subnetwork distinctiveness is essential: mask similarity serves as a reliable, label-free indicator of subnetwork collapse, showing that the bottleneck is not raw capacity, but the preservation of structural diversity. Second, RTL is robust to imperfect data partitions - it excels with clean class boundaries (CIFAR-10), remains effective under noisy semantic clustering (CIFAR-100), and generalizes to real-world applications where subnetworks align with meaningful environmental factors.

［#94］
We also observe that RTL subnetworks tend to favor recall over precision, increasing sensitivity to weak class- or environment-specific signals. While beneficial for routing-based inference, this suggests that downstream calibration or selective filtering could further improve precision without compromising specialization.

［#95］
This work positions adaptive pruning as a pathway toward more modular, efficient, and interpretable deep models, ones that dynamically allocate representational capacity in alignment with the intrinsic structure of the data they process.

［#96］
Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

## Impact Statement
［#97］
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References



















































# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

## A. Vision Model and Training Setup

### A.1. Model architecture and pruning scope

［#98］
For all computer vision experiments on CIFAR-10 and CIFAR-100, we use GhostNet (Han et al., 2019) as the backbone. To control model capacity and isolate the effects of adaptive pruning, we retain the convolutional stem followed by the first nine Ghost bottleneck blocks, discarding deeper stages of the original network.

［#99］
All convolutional layers are implemented as masked convolutions and constitute the pruning scope. Batch normalization layers, squeeze-and-excitation modules, and the final classifier remain dense. Under this configuration, the model contains approximately 126K prunable parameters.

［#100］
The retained GhostNet variant follows the standard Ghost bottleneck design, combining inexpensive depthwise convolutions with pointwise expansions and optional downsampling and squeeze-and-excitation. Spatial resolution is progressively reduced via strided depthwise convolutions in selected bottlenecks. A detailed summary of the architecture, including channel dimensions and downsampling stages, is provided in Table 4 to ensure reproducibility.

［#101］
Table 4. GhostNet backbone used for CIFAR experiments. Only convolutional layers are pruned. All convolutions are followed by batch normalization and ReLU unless stated otherwise.

［#102］
<table>
  <tr>
    <th>Stage</th>
    <th>Block type</th>
    <th>Input → Output</th>
    <th>Kernel / Stride</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>Stem</td>
    <td>Conv2D</td>
    <td>3 → 16</td>
    <td>3 × 3 / 2</td>
    <td>Initial downsampling</td>
  </tr>
  <tr>
    <td>1</td>
    <td>GhostBottleneck</td>
    <td>16 → 16</td>
    <td>1 × 1, 3 × 3 / 1</td>
    <td>No shortcut</td>
  </tr>
  <tr>
    <td>2</td>
    <td>GhostBottleneck</td>
    <td>16 → 24</td>
    <td>3 × 3 / 2</td>
    <td>Downsampling + shortcut</td>
  </tr>
  <tr>
    <td>3</td>
    <td>GhostBottleneck</td>
    <td>24 → 24</td>
    <td>1 × 1, 3 × 3 / 1</td>
    <td>Shortcut</td>
  </tr>
  <tr>
    <td>4</td>
    <td>GhostBottleneck</td>
    <td>24 → 40</td>
    <td>5 × 5 / 2</td>
    <td>SE + shortcut</td>
  </tr>
  <tr>
    <td>5</td>
    <td>GhostBottleneck</td>
    <td>40 → 40</td>
    <td>1 × 1, 3 × 3 / 1</td>
    <td>SE</td>
  </tr>
  <tr>
    <td>6</td>
    <td>GhostBottleneck</td>
    <td>40 → 80</td>
    <td>3 × 3 / 2</td>
    <td>Downsampling + shortcut</td>
  </tr>
  <tr>
    <td>7</td>
    <td>GhostBottleneck ×3</td>
    <td>80 → 80</td>
    <td>1 × 1, 3 × 3 / 1</td>
    <td>Repeated blocks</td>
  </tr>
  <tr>
    <td>8</td>
    <td>Conv1×1</td>
    <td>80 → 184</td>
    <td>1 × 1 / 1</td>
    <td>Channel expansion</td>
  </tr>
  <tr>
    <td>Head</td>
    <td>Conv + Pool</td>
    <td>184 → 128</td>
    <td>1 × 1</td>
    <td>Global avg pooling</td>
  </tr>
  <tr>
    <td>Classifier</td>
    <td>Linear</td>
    <td>128 → $C$</td>
    <td>–</td>
    <td>$C = 10$ or 100</td>
  </tr>
</table>

### A.2. Optimization and training protocol

［#103］
All models are trained using the Adam optimizer (Kingma & Ba, 2015) with a learning rate of $1\mathrm{e}{-4}$ and no weight decay. Each pruning iteration consists of two phases: (i) pruning and (ii) joint retraining, both run for 10 epochs. Batch sizes are set to 320 for CIFAR-10 and 256 for CIFAR-100.

［#104］
To ensure fair comparison, all methods (RTL, single-model IMP, and multi-model IMP) share the same random Kaiming initialization (He et al., 2015). RTL and multi-model IMP subnetworks are trained as binary classifiers with balanced mini-batches, effectively doubling the number of sample presentations per epoch. To match total data exposure, the single-model IMP baseline is trained for 20 epochs.

### A.3. Pruning schedule and compute

［#105］
Pruning follows a fixed schedule, removing 4,096 weights per epoch for each subnetwork until the target sparsity is reached. All experiments are executed on a single NVIDIA H100 GPU. Full RTL and multi-model IMP runs require approximately 6 hours, while the single-model IMP baseline completes in roughly 45 minutes.

## B. Semantic Clustering Procedure

［#106］
To construct coarse semantic subsets for CIFAR-100, we apply an unsupervised clustering pipeline based on textual class descriptions rather than visual features.


［#107］
Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#108］
First, we extract text embeddings for each of the 100 CIFAR-100 class names using the CLIP text encoder. These embeddings are then reduced in dimensionality using UMAP to facilitate clustering in a lower-dimensional space. Finally, we apply HDBSCAN to the reduced embeddings to obtain cluster assignments.

［#109］
This procedure yields 8 semantic clusters. While the resulting clusters are semantically coherent, they are not perfectly aligned with visual similarity, intentionally introducing noise and overlap. This imperfect alignment better reflects real-world scenarios where data partitions are ambiguous or only approximately defined. The full cluster composition is reported in Table 5.

［#110］
Table 5. Semantic clusters used in the CIFAR-100 experiments. Each cluster defines a data subset to which a specialized subnetwork is assigned.

［#111］
<table>
  <tr>
    <th>Cluster</th>
    <th>CIFAR-100 Classes</th>
  </tr>
  <tr>
    <td>1 (Aquatic animals)</td>
    <td>aquarium fish, dinosaur, dolphin, flatfish, ray, seal, shark, trout, whale</td>
  </tr>
  <tr>
    <td>2 (People & objects)</td>
    <td>baby, bed, bicycle, bottle, bowl, boy, bridge, can, castle, chair, clock, couch, cup, girl, house, keyboard, lamp, man, plain, plate, road, rocket, skyscraper, table, telephone, television, wardrobe, woman</td>
  </tr>
  <tr>
    <td>3 (Mammals)</td>
    <td>bear, beaver, camel, cattle, chimpanzee, elephant, fox, hamster, kangaroo, leopard, lion, mouse, otter, porcupine, possum, rabbit, raccoon, shrew, skunk, squirrel, tiger, wolf</td>
  </tr>
  <tr>
    <td>4 (Reptiles & insects)</td>
    <td>bee, beetle, butterfly, caterpillar, cockroach, crab, crocodile, lizard, lobster, mushroom, snail, snake, spider, turtle, worm</td>
  </tr>
  <tr>
    <td>5 (Vehicles)</td>
    <td>bus, lawn mower, motorcycle, pickup truck, streetcar, tank, tractor, train</td>
  </tr>
  <tr>
    <td>6 (Natural scenes)</td>
    <td>cloud, forest, maple tree, mountain, oak tree, palm tree, pine tree, sea, willow tree</td>
  </tr>
  <tr>
    <td>7 (Fruits)</td>
    <td>apple, orange, pear, sweet pepper</td>
  </tr>
  <tr>
    <td>8 (Flowers)</td>
    <td>orchid, poppy, rose, sunflower, tulip</td>
  </tr>
</table>

［#112］
All clustering hyperparameters are fixed across experiments and are not tuned to downstream task performance. As shown in Sec. 5, RTL remains effective under this setting, demonstrating robustness to non-ideal specialization boundaries and highlighting the benefits of adaptive pruning beyond strictly class-aligned scenarios.

## C. Implicit Neural Representation

### C.1. Task setup

［#113］
We consider the standard INR formulation of mapping continuous pixel coordinates $(x, y)$ to RGB values. The full image is reconstructed through point-wise evaluation of the network over all pixel locations.

［#114］
Experiments are conducted on 10 images selected from the training split of the ADE20K dataset to ensure diversity in scene category and luminosity/color characteristics. Specifically, we use images with indices 0, 3733, 7304, 8399, 11708, 12963, 13783, 14236, 18813, and 23790. Each image is treated independently, and a separate model is trained per image. The selected images and their corresponding preprocessed semantic segmentation masks are shown in Fig. 4.

［#115］
Semantic segmentation annotations are used to define classes: all masks corresponding to the same semantic object category within an image are merged into a single class (e.g., multiple instances of tree are treated as one mask).

［#116］
The dataset contains small unassigned or noisy regions, typically at object boundaries. These regions are reassigned to a neighboring semantic class, which was empirically found to stabilize training for both RTL and baseline models.

### C.2. Positional encoding

［#117］
Input coordinates $(x, y)$ are encoded using Fourier features prior to being passed to the network, following standard INR practice. This encoded representation forms the sole input to the RTL model.


［#118］
![](./images/1223972462526464001_4.jpg)

［#119］
Figure 4. Data samples from the ADE20K dataset used in the INR experiments, shown together with their corresponding preprocessed semantic segmentation masks.

### C.3. Model architectures
［#120］
All models are multilayer perceptrons (MLPs) with ReLU activations. For RTL, the network consists of five layers with the following input-output dimensions: [[128, 34], [128, 128], [128, 128], [128, 128], [3, 128]]

［#121］
For the standard baseline, the input is augmented with a learned class embedding corresponding to the semantic mask of each pixel, increasing the first-layer dimensionality to [128, 50]. All subsequent layers are identical to those used in RTL.

### C.4. Training and pruning protocol
［#122］
All models are trained using the Adam optimizer with a learning rate of 0.01. Training proceeds for 10,000 optimization steps, after which pruning is applied by removing 4,096 weights per pruning iteration.

［#123］
Following each pruning step, all remaining (non-pruned) weights are rewound to their initial values, consistent with the IMP-style pruning protocol used throughout the paper. This prune-and-rewind process is repeated until the target sparsity is reached.

### C.5. Initialization and fairness considerations
［#124］
Because the RTL and baseline models differ in input dimensionality, care is taken to ensure fair initialization. We initialize the larger baseline network (including the class embedding input) and manually remove the weights corresponding to the class embedding dimensions to obtain the RTL initialization. This ensures that all shared parameters are initialized identically across methods.


［#125］
The same initialization seed is used across all images.

### C.6. Evaluation protocol
［#126］
Reconstruction quality is evaluated using peak signal-to-noise ratio (PSNR). In the main text, PSNR is reported as a single scalar value averaged over all pixels and all images. For completeness, we additionally report per-image PSNR values in the appendix to illustrate variability across samples.

### C.7. Per-Image INR Results
［#127］
Table 6 reports per-image reconstruction performance for the INR experiment on 10 ADE20K images. Results are shown for RTL and the single-model IMP baseline at 25%, 50%, and 75% sparsity, measured using PSNR.

［#128］
Table 6. Per-image PSNR for the INR experiment on 10 ADE20K images. Results are reported for RTL and the single-model IMP baseline at 25%, 50%, and 75% sparsity.

［#129］
<table>
  <thead>
    <tr>
      <th rowspan="2">Sample</th>
      <th colspan="6">PSNR $\uparrow$</th>
    </tr>
    <tr>
      <th colspan="3">RTL</th>
      <th colspan="3">IMP (single model)</th>
    </tr>
    <tr>
      <th></th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>13.73</td>
      <td>12.08</td>
      <td>9.68</td>
      <td>11.51</td>
      <td>10.14</td>
      <td>8.39</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10.40</td>
      <td>9.27</td>
      <td>7.94</td>
      <td>8.60</td>
      <td>7.95</td>
      <td>6.83</td>
    </tr>
    <tr>
      <td>3</td>
      <td>17.59</td>
      <td>16.92</td>
      <td>15.66</td>
      <td>16.32</td>
      <td>15.45</td>
      <td>13.96</td>
    </tr>
    <tr>
      <td>4</td>
      <td>24.46</td>
      <td>22.58</td>
      <td>20.56</td>
      <td>21.58</td>
      <td>20.19</td>
      <td>17.79</td>
    </tr>
    <tr>
      <td>5</td>
      <td>18.52</td>
      <td>16.88</td>
      <td>13.99</td>
      <td>14.63</td>
      <td>12.99</td>
      <td>10.96</td>
    </tr>
    <tr>
      <td>6</td>
      <td>19.01</td>
      <td>17.23</td>
      <td>14.50</td>
      <td>16.63</td>
      <td>15.44</td>
      <td>13.24</td>
    </tr>
    <tr>
      <td>7</td>
      <td>25.73</td>
      <td>23.14</td>
      <td>19.30</td>
      <td>19.81</td>
      <td>18.62</td>
      <td>15.96</td>
    </tr>
    <tr>
      <td>8</td>
      <td>15.55</td>
      <td>14.08</td>
      <td>11.97</td>
      <td>12.47</td>
      <td>11.45</td>
      <td>9.79</td>
    </tr>
    <tr>
      <td>9</td>
      <td>19.34</td>
      <td>17.73</td>
      <td>15.41</td>
      <td>16.81</td>
      <td>15.43</td>
      <td>13.17</td>
    </tr>
    <tr>
      <td>10</td>
      <td>24.31</td>
      <td>22.63</td>
      <td>19.68</td>
      <td>21.07</td>
      <td>19.53</td>
      <td>16.78</td>
    </tr>
  </tbody>
</table>

［#130］
Across all samples and sparsity levels, RTL consistently achieves higher PSNR than the single-model IMP baseline. The performance gap is observed for images with diverse scene content and appearance characteristics, confirming that the aggregate improvements reported in the main text are not driven by a small subset of favorable samples.

［#131］
The advantage of RTL is particularly pronounced at higher sparsity levels, where enforcing a single global pruning mask across semantically heterogeneous regions leads to larger reconstruction errors. In contrast, RTL preserves region-specific parameters through specialized subnetworks, resulting in more stable degradation as sparsity increases.

［#132］
These per-image results complement the averaged PSNR values reported in the main text by demonstrating that the benefits of adaptive pruning in INR settings are consistent across images, rather than arising from outliers or dataset bias. Figure 5 shows the relationship between reconstruction quality and pruning mask similarity for each of the 10 ADE20K images (A–J) used in the INR experiment. For each image, we report PSNR (orange solid line, left axis) and average mask similarity measured by the Jaccard index (blue dashed line, right axis) as a function of sparsity.

［#133］
Across all images, PSNR decreases gradually as sparsity increases, followed by a sharper drop at high sparsity levels. A similar trend is observed for mask similarity: pruning masks remain relatively stable at low to moderate sparsity, but their similarity rapidly decreases beyond a critical sparsity threshold. This transition is consistent across images, despite substantial differences in scene content and appearance.

［#134］
Notably, the sharp decline in PSNR closely coincides with the point at which mask similarity collapses. This indicates that reconstruction quality degrades most significantly once subnetworks corresponding to different semantic regions begin to overlap or interfere, rather than as a direct consequence of parameter removal alone. At very high sparsity, mask similarity approaches zero, reflecting near-disjoint or unstable subnetworks and resulting in poor reconstruction quality.

［#135］
![](./images/1223972462526464001_5.jpg)

［#136］
Figure 5. Per-image relationship between reconstruction quality and pruning mask similarity in the INR experiment. For each ADE20K image (A-J), PSNR (orange, left axis) and average mask similarity measured by the Jaccard index (blue dashed, right axis) are shown as a function of sparsity.

［#137］
These per-image trends mirror the averaged results reported in the main text and further support the interpretation that maintaining distinct, region-specific pruning masks is critical for preserving reconstruction performance in high-sparsity INR settings.

### C.8. INR subnetwork similarity vs. reconstruction quality

［#138］
We further analyze the relationship between subnetwork specialization and reconstruction quality in the INR setting by examining how per-region PSNR correlates with mask similarity.

［#139］
Fig. 6 reports results for a single ADE20K image containing four semantic regions. For each region-specific subnetwork, we plot PSNR together with the average Jaccard similarity of its pruning mask to all other region masks across sparsity levels. PSNR remains relatively stable while mask similarity is low, but degrades sharply once similarity increases, indicating the onset of subnetwork collapse.

［#140］
Fig. 7 extends this analysis to a second ADE20K image with fifteen semantic regions. Despite the larger number of classes, the same trend holds across all regions (A-O): reconstruction quality deteriorates rapidly once subnetworks lose structural distinctiveness under aggressive pruning.

［#141］
To isolate correlation effects independent of absolute scale, Fig. 8 re-plots the 15-region case with both PSNR and mask similarity mean-centered (panels A-O). Across regions, fluctuations in reconstruction quality closely track changes in mask similarity, revealing an inverse specialization effect: regions that maintain higher PSNR tend to exhibit greater mask overlap, while regions with lower PSNR show increased structural specialization. This suggests that well-reconstructed regions rely

［#142］
![](./images/1223972462526464001_6.jpg)

［#143］
Figure 6. Per-class PSNR and subnetwork similarity for a single ADE20K image with four semantic regions. PSNR (solid) and average mask similarity to all other region-specific subnetworks (dashed) are shown as functions of sparsity. Performance degradation coincides with a rapid increase in mask similarity, indicating subnetwork collapse at high pruning ratios.

# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#144］
![](./images/1223972462526464001_7.jpg)

［#145］
Figure 7. Per-class PSNR and subnetwork similarity for a second ADE20K image with fifteen semantic regions. Each subplot (A-O) corresponds to one semantic region. As sparsity increases, PSNR declines gradually until a sharp drop aligns with increasing similarity between region-specific pruning masks, reflecting loss of structural specialization.

［#146］
on more shared parameters, whereas harder regions benefit from stronger subnetwork differentiation.

［#147］
Panel P summarizes this relationship at 50% sparsity by plotting PSNR against mask similarity for all regions and fitting a linear regression. We observe a strong monotonic relationship, with an overall Spearman rank correlation of $\rho \approx 0.982$. This confirms that subnetwork similarity is a strong predictor of reconstruction quality in INRs, even within a single image.

［#148］
Together, these results reinforce the central claim of RTL: preserving subnetwork distinctiveness is critical not only across datasets or tasks, but also for fine-grained, within-image semantic specialization. Mask similarity thus provides a reliable, label-free diagnostic for identifying oversparsification and impending performance collapse in coordinate-based representations.

## C.9. Qualitative INR Reconstructions under High Sparsity

［#149］
To complement the quantitative PSNR analysis, we provide qualitative visualizations of INR reconstructions under increasing sparsity levels. Figures 9, 10, and 11 show reconstructed images at 50%, 75%, and 90% sparsity, respectively.

［#150］
![](./images/1223972462526464001_8.jpg)

［#151］
Figure 8. Correlation between reconstruction quality and subnetwork similarity in INRs. Panels A-O show mean-centered PSNR and mean-centered mask similarity for the fifteen-region image. Panel P reports linear regression between the two quantities at 50% sparsity across regions, revealing a strong monotonic relationship (Spearman $\rho \approx 0.982$).

# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#152］
Each figure contains reconstructions for all 10 ADE20K images used in our experiments (rows). For each image, we display three columns: the ground-truth target image, the reconstruction produced by RTL, and the reconstruction produced by a single-mask IMP baseline. All models are trained under identical architectural, optimization, and sparsity constraints.

［#153］
At 50% sparsity (Fig. 9), both methods recover the overall scene structure, but RTL consistently preserves finer details and sharper object boundaries. Differences are particularly noticeable in textured regions and along semantic edges, where IMP reconstructions exhibit mild blurring and loss of contrast.

［#154］
At 75% sparsity (Fig. 10), the qualitative gap widens. IMP reconstructions show clear degradation in high-frequency content, with washed-out colors and incomplete reconstruction of small structures. In contrast, RTL maintains more faithful geometry and color consistency across most images, indicating that region-specialized subnetworks better preserve semantically important parameters under aggressive pruning.

［#155］
At the extreme 90% sparsity regime (Fig. 11), the difference becomes pronounced. IMP often collapses to coarse, low-detail approximations, with significant artifacts and loss of semantic coherence. RTL reconstructions, while degraded relative to lower sparsity levels, retain recognizable object shapes, clearer region boundaries, and more stable color distributions. This qualitative evidence aligns with the PSNR trends reported in the main paper, confirming that adaptive, region-specific pruning enables more graceful degradation as sparsity increases.

［#156］
Overall, these visual results demonstrate that RTL not only improves average reconstruction metrics but also yields perceptually superior outputs, especially in the high-sparsity regime where competition for shared parameters is most severe.



［#157］
![](./images/1223972462526464001_9.jpg)

［#158］
Figure 9. Qualitative INR reconstructions at 50% sparsity. Rows correspond to 10 ADE20K images. Columns show the ground-truth target, RTL reconstruction, and IMP reconstruction. RTL preserves sharper edges and finer details compared to IMP, particularly in textured and semantically complex regions.

# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#159］
![](./images/1223972462526464001_10.jpg)

［#160］
Figure 10. Qualitative INR reconstructions at 75% sparsity. RTL maintains more faithful structure and color consistency, while IMP exhibits increased blurring and loss of high-frequency details. The gap between methods becomes more visible as sparsity increases.

［#161］
Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#162］
![](./images/1223972462526464001_11.jpg)

［#163］
Figure 11. Qualitative INR reconstructions at 90% sparsity. Under extreme pruning, IMP reconstructions often collapse to coarse approximations with severe artifacts. RTL degrades more gracefully, preserving recognizable object shapes and semantic boundaries across images.

［#164］
Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

## D. Speech Enhancement

### D.1. Dataset construction

［#165］
We use clean speech samples from the DNS Challenge 2020 dataset (Reddy et al., 2020) and mix them with environmental noise from the TAU Urban Acoustic Scenes 2020 dataset (Heittola et al., 2020). Noise samples are grouped into three acoustic scenes: *indoor*, *outdoor*, and *transportation*. Each scene defines a distinct subset used to train a specialized subnetwork.

### D.2. Model architecture

［#166］
For the speech enhancement task, we use a lightweight U-Net–style architecture (Ronneberger et al., 2015) operating on complex STFT representations. The network takes a 2-channel input (real and imaginary components) and predicts a 2-channel complex ratio mask.

［#167］
The model consists of five encoder and five decoder stages. Each encoder stage applies a masked 2D convolution followed by ELU activation and batch normalization. Downsampling is performed only along the time axis using a stride of (1,2), preserving frequency resolution. The encoder channel dimensions are [12, 12, 12, 24, 48]. The decoder mirrors the encoder using masked transposed convolutions with symmetric kernel sizes and strides. Skip connections concatenate encoder features with decoder inputs, doubling the channel dimensionality prior to decoding. The final decoder layer outputs a 2-channel complex mask.

［#168］
All convolutions use $3 \times 3$ kernels, except for the first and last layers, which use $3 \times 5$ kernels to capture a wider temporal context. This compact design intentionally limits model capacity, allowing improvements from adaptive pruning to be isolated from architectural scaling effects. A detailed layer-by-layer specification of the network is provided in Table 7 for reproducibility.

［#169］
Table 7. Speech enhancement U-Net architecture. All layers use ELU activation and batch normalization. Stride (1,2) downsamples only along the time axis.

［#170］
<table>
  <tr>
    <th>Stage</th>
    <th>Type</th>
    <th>Channels (in $\rightarrow$ out)</th>
    <th>Kernel / Stride</th>
  </tr>
  <tr>
    <td>Enc-1</td>
    <td>Conv2D</td>
    <td>$2 \rightarrow 12$</td>
    <td>$3 \times 5 / (1,2)$</td>
  </tr>
  <tr>
    <td>Enc-2</td>
    <td>Conv2D</td>
    <td>$12 \rightarrow 12$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Enc-3</td>
    <td>Conv2D</td>
    <td>$12 \rightarrow 12$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Enc-4</td>
    <td>Conv2D</td>
    <td>$12 \rightarrow 24$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Enc-5</td>
    <td>Conv2D</td>
    <td>$24 \rightarrow 48$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Dec-1</td>
    <td>TConv2D</td>
    <td>$48 \rightarrow 24$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Dec-2</td>
    <td>TConv2D</td>
    <td>$48 \rightarrow 12$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Dec-3</td>
    <td>TConv2D</td>
    <td>$24 \rightarrow 12$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Dec-4</td>
    <td>TConv2D</td>
    <td>$24 \rightarrow 12$</td>
    <td>$3 \times 3 / (1,2)$</td>
  </tr>
  <tr>
    <td>Dec-5</td>
    <td>TConv2D</td>
    <td>$24 \rightarrow 2$</td>
    <td>$3 \times 5 / (1,2)$</td>
  </tr>
</table>

### D.3. Signal processing and loss

［#171］
Input audio consists of 10-second, 16-bit waveforms. We compute STFTs using a 1024-sample window and a 256-sample hop size, producing spectrograms of shape [2, 626, 513]. The model predicts a mask of identical shape, which is applied to the noisy spectrogram.

［#172］
Training uses the weighted source-to-distortion ratio (wSDR) loss. Unlike the vision experiments, this task does not involve negative samples; consequently, all methods are trained for the same number of epochs.

### D.4. Optimization and runtime

［#173］
We use the Adam optimizer with a learning rate of $1\mathrm{e}{-4}$ and no weight decay. Both pruning and retraining phases run for 10 epochs. All models share identical initial weights. Experiments are conducted on a single NVIDIA H100 GPU. Full RTL

［#174］
Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

［#175］
![](./images/1223972462526464001_12.jpg)

［#176］
Figure 12. Qualitative speech enhancement results at 50% sparsity. Each row corresponds to a different acoustic environment, and columns show the noisy input, clean target, RTL output, and IMP output. RTL more effectively suppresses noise and preserves harmonic speech structure across all environments.

［#177］
and multi-model IMP runs take approximately 10 hours, while the single-model IMP baseline completes in about 8 hours.

### D.5. Qualitative Analysis of Speech Enhancement

［#178］
Fig. 12 provides a qualitative comparison of speech enhancement results across three acoustic environments at 50% weight sparsity. Each row corresponds to a distinct environment class, while columns show the noisy input spectrogram, the clean target, and the enhanced outputs produced by RTL and the IMP baseline.

［#179］
Across all environments, RTL more faithfully reconstructs the time-frequency structure of clean speech. Harmonic components are clearer, transient events are better preserved, and noise-dominated regions are more effectively suppressed. In contrast, the IMP baseline exhibits residual noise, smeared harmonics, and reduced contrast between speech and background, particularly in mid- and high-frequency bands.

［#180］
The qualitative gap becomes especially apparent in challenging conditions, where environment-specific noise patterns dominate the input. RTL subnetworks, specialized to each acoustic scene, recover speech structure with higher temporal co-herence and sharper spectral detail, whereas the single-mask IMP model struggles to balance denoising across heterogeneous conditions.

［#181］
These visual results are consistent with the quantitative SI-SNRi improvements reported in Sec. 5, and further support the claim that adaptive, environment-aligned pruning yields more effective representations than a single global sparse model.

## E. Subnetwork Similarity Analysis

### E.1. Mask similarity

［#182］
To analyze relationships between learned subnetworks, we compute pairwise similarity between binary pruning masks at multiple sparsity levels. Given two masks $M_i$ and $M_j$, similarity is measured using the Jaccard coefficient:

［#182］
$$
J(M_i, M_j) = \frac{|M_i \cap M_j|}{|M_i \cup M_j|}, \tag{8}
$$

［#182］
where intersection and union are computed element-wise over all prunable parameters.

［#183］
Similarity is evaluated both globally and on a per-layer basis to study how overlap varies across network depth.

### E.2. Collapse analysis

［#184］
For each subnetwork, we record balanced accuracy at multiple sparsity levels and compute its average pairwise mask similarity to other subnetworks. Correlating these quantities allows us to assess how excessive overlap (i.e., subnetwork collapse) relates to performance degradation.

### E.3. Semantic alignment

［#185］
To assess whether structural similarity reflects semantic similarity, we use CIFAR-10 as a case study. Pairwise semantic distances between class labels are computed using WordNet path similarity (Miller, 1995). These distances are compared against corresponding mask similarity values, yielding aligned similarity matrices that reveal whether conceptually related classes share pruning structure.