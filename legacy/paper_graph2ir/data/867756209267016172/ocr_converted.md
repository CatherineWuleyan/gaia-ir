［#1］
**RESEARCH ARTICLE**
International Journal of Intelligent Systems

# Exploring Lottery Ticket Hypothesis in Media Recommender Systems

［#2］
Yanfang Wang¹ | Yongduo Sui¹ | Xiang Wang² |
Zhenguang Liu³ | Xiangnan He¹*

［#3］
¹School of Information Science and
Technology, University of Science and
Technology of China, Hefei, Anhui, 230022,
China
²School of Computing, National University
of Singapore, Kent Ridge, 119077,
Singapore
³College of Computer Science and
Technology, Zhejiang University, Hangzhou,
Zhejiang, 310007, China

## Correspondence
［#4］
Xiangnan He, School of Information Science and
Technology, University of Science and
Technology of China, Hefei, Anhui, 230022,
China
Email: hexn@ustc.edu.cn

## Funding information
［#5］
National Key Research and Development
Program of China, Grant/Award Number:
2020YFB1406703; National Natural
Science Foundation of China, Grant/Award
Number: U19A2079, 61972372 and
62121002

## Conflict of interest statement
［#6］
The authors declared that they have no
conflicts of interest to this work.

## Authorship contributions
［#7］
All persons who meet authorship criteria
are listed as authors, and all authors certify
that they have participated sufficiently in
the work to take public responsibility for
the content.

［#8］
Media recommender systems aim to capture users' prefer-
ences and provide precise personalized recommendation
of media content. There are two critical components in
the common paradigm of modern recommender models: (1)
representation learning, which generates an embedding for
each user and item; and (2) interaction modeling, which fits
user preferences towards items based on their representa-
tions. In spite of great success, when a great amount of
users and items exist, it usually needs to create, store, and
optimize a huge embedding table, where the scale of model
parameters easily reach millions or even larger. Hence, it
naturally raises questions about the heavy recommender
models: Do we really need such large-scale parameters?
We get inspirations from the recently proposed lottery ticket
hypothesis (LTH), which argues that the dense and over-
parameterized model contains a much smaller and sparser
sub-model that can reach comparable performance to the
full model. In this paper, we extend LTH to media recom-
mender systems, aiming to find the winning tickets in deep
recommender models. To the best of our knowledge, this
is the first work to study LTH in media recommender sys-
tems. With MF and LightGCN as the backbone models,
we found that there widely exist winning tickets in recom-
mender models. On three media convergence datasets —


［#9］
Yelp2018, TikTok and Kwai, the winning tickets can achieve comparable recommendation performance with only 29% ~48%, 7% ~10% and 3% ~17% of parameters, respectively.

［#10］
KEYWORDS
media recommender system, lottery ticket hypothesis, lightweight embedding, model pruning, iterative magnitude-based pruning

## 1 | INTRODUCTION

［#11］
In the era of information explosion, media recommendation is becoming the core of many online platforms, to accurately yield the information of interest that meets user needs. Media recommender systems aim to provide precise personalized recommendation for users to find their preferences from the deluge of items. Towards better media recommendation, extensive deep recommender models $^{1;2;3;4}$ have been proposed to capture user preference from behavioral data. In general, the common paradigm of these models can be systematized as two critical components: (1) representation learning, which creates a representation vector (i.e., embedding) for each single user and item; and (2) interaction modeling, which fits the historical user-item interactions and predicts the preference of a user to an item based on their embeddings $^{5;6}$. Obviously, the capacity of recommender model depends heavily on the representation ability of user and item embeddings, which are influenced by the scale of parameters — an embedding with the larger size not only delineates the characteristics of users and items better $^{7}$, but also predicts the user-item interaction more precisely $^{8}$. Although the large embedding table brings obvious accuracy improvements, it suffers from two main limitations:

［#12］
(1) The massive parameters become the major obstacle for production deployment and real-time prediction of deep recommender models. Specifically, in real-world scenarios, the number of users and items is usually up to tens of millions or even hundreds of millions. If each user (or item) is associated with a certain dense embedding, it will bring extremely expensive costs of memory and inference time to maintain the huge embedding table for all users and items. A good case is the famous audio-visual media convergence recommender system YouTube Recommendation $^{9}$, which projects a million videos into a 256-dimensional vector space, producing an embedding table with 256 million model parameters.

［#13］
(2) Interaction histories of different users usually contain varying item numbers and sets, thus having different informativeness. Nevertheless, current recommender models almost assign all users and items with the same embedding size. Hence, unifying the embedding size might limit the representation ability of user and item embeddings. Especially in a large-scale industrial media recommendation scenario, it may cause serious over-parameterization and over-fitting issues.

［#14］
One prevalent solution is to adopt unstructured pruning (a.k.a. sparse pruning) $^{10}$ on the user-item embedding table. As a branch of neural network pruning technology, sparse pruning aims to refine sparse sub-networks with similar performance but fewer parameters by removing redundant parameters of dense networks. In traditional sparse pruning, a large over-parameterized dense network is required to be trained first, followed by pruning and fine-tuning. Obviously, training a large dense network is expensive. Therefore, some researchers $^{11}$ suggested that, rather than training a large dense network and then pruning it into a small sparse sub-network, it is better to train a small sparse sub-network from the scratch. This view is normally considered infeasible in previous studies, because large dense networks have wide parameter search spaces to find optimum solution through independent training, while small

［#14］
sparse networks merely have limited parameter search spaces.

［#15］
More recently, Frankle and Carbin $^{11}$ proposed Lottery Ticket Hypothesis (LTH): a large and dense deep neural network contains a small sparse sub-network, which can match the test accuracy of the original large network for at most the same number of iterations when trained independently. This trainable sparse sub-network is called a Winning Ticket. At present, LTH has been verified in the fields of Computer Vision (CV) $^{11}$ and Natural Language Processing (NLP) $^{12}$. However, to the best of our knowledge, there is no relevant research in the field of recommendation. If we can verify the existence of winning tickets and find them in deep recommender models, then the model parameters can be effectively reduced while preserving the test performance. Benefiting from the technique of sparse matrix storage $^{13}$, we can not only reduce the memory usage, but also speed up the inference.

［#16］
In this work, we explore LTH in media recommender systems. Specifically, we focus on the widely-used model parameters – the user-item embedding table. Following the previous studies $^{11}$, we speculate that a large dense user-item embedding table contains a smaller sparser sub-matrix, which can be independently trained to achieve a similar performance as the full matrix. Technically, we exploit the iterative magnitude-based pruning (IMP) $^{11:14}$, which gradually creates the embedding's binary masks based on their weight magnitudes and eventually reaches a powerful sub-matrix. The sub-matrix is referred to as a winning ticket in the recommender model. We conduct the extensive experiments on three benchmark datasets, where the empirical results consistently show that: (1) winning tickets exist widely in deep recommender models; (2) the IMP algorithm can stably find the winning tickets. Meanwhile, the winning tickets we found significantly outperform the original large matrix in terms of time consumption, memory usage and test performance.

［#17］
Our main contributions are as follows:

［#18］
- We explore lottery ticket hypothesis in media recommender systems, aiming to reduce the memory usage and speed up the inference while preserving the recommendation accuracy. To the best of our knowledge, this is the first work to study lottery ticket hypothesis in the field of recommendation.
- With two deep recommender models Matrix Factorization (MF) $^{15}$ and Light Graph Convolution Networks (Light-GCN) $^{4}$ as the backbone models being pruned, we conduct a lot of experiments on three real-world media convergence recommendation datasets, demonstrating that winning tickets widely exist in deep recommender models and the IMP algorithm we used can stably find the winning tickets.
- The experimental results show that the winning tickets are significantly superior over the original large embedding table in terms of time consumption, memory usage, and test performance. Specifically, on three datasets Yelp2018, TikTok and Kwai, our found winning tickets achieve the same performance as the complete models with only 29% ~48%, 7% ~10% and 3% ~17% of the parameters, respectively.

## 2 | RELATED WORK

### 2.1 | Lightweight Embedding

［#19］
As the most basic parameters in the deep recommender models, the huge user-item embedding table dominates both the parameter scale and the inductive bias of the model. In order to choose an appropriate embedding size and reduce the parameter scale of the embedding table, there are two promising lightweight techniques:

［#20］
Auto Machine Learning (AutoML). $^{7}$ Joglekar et al. $^{16}$ designed two neural search approaches: Single-Size embedding (NIS-SE) and Multi-Size embedding (NIS-ME). They defined the embedding table search space by blocks and adjusted the controller with the validation set, making the model automatically determine the embedding size by

［#20］
maximizing the accuracy under the constraint of the embedding table memory. On top of that, some other neural search approaches such as Neural Architecture Search (NAS)¹⁷, Efficient Neural Architecture Search (ENAS)¹⁸, and Differentiable Architecture Search (DAS)¹⁹ are also utilized to automatically determine the size of embeddings and some of them have been deployed in the industry media recommender systems and achieved certain benefits.

［#21］
Model Compression.²⁰⁻²¹ Common model compression technologies mainly includes pruning²², quantization²³ and distillation²⁴. For example, Liu et al.¹⁰ compressed the embedding table by pruning the embedded vectors of each feature domain of the data. Sun et al.²⁵ proposed a general sequential model compression framework, which decomposes the embedding table into multiple low-rank matrices. Wu et al.²⁶ clustered the features for different domains of users and items by similarity calculation, effectively reducing the total feature number. Zhang et al.²⁷ shared the parameters among similar features so as to reduce the parameters of embedding tables.

## 2.2 | Lottery Ticket Hypothesis

［#22］
For preserving accuracy, conventional sparse pruning methods have to train a large dense neural network first, and then prune it into a small sparse sub-network. Obviously, this process is computationally expensive. To address this problem, Frankle and Carbin¹¹ proposed lottery ticket hypothesis, which states that there exist some small sparse sub-networks (winning tickets), which can be directly trained from the scratch to achieve performance comparable to the complete model with a similar or even faster training speed. To find the winning tickets in deep neural networks, they further proposed the IMP algorithm.

［#23］
Recently, many efforts have been devoted to improving or extending lottery ticket hypothesis. Brix et al.²⁸ applied LTH to Transformer models, by rewinding the pruned sub-network weights to the values at iteration k instead of 0. Renda et al.²⁹ compared three different retraining techniques: fine-tuning, weight rewinding and learning rate rewinding. In their experiments, the IMP approach with weight rewinding achieved the best performance in terms of accuracy, compression ratio and search efficiency. You et al.³⁰ modified the IMP algorithm to improve the search efficiency for winning tickets. Morcos et al.³¹ generalized LTH across different datasets and optimizers, while Yu et al.¹² extended it to the field of NLP and reinforcement learning, confirming the generality of winning tickets.

# 3 | METHODOLOGY

［#24］
This section begins with notations and definitions of the recommendation task (3.1), followed by an introduction of the two widely-used deep recommender models MF and LightGCN (3.2). Then we formally define LTH-MRS (3.3), and detail our used method for searching winning tickets in recommender models (3.4). Finally, we will provide some complexity analyses to show the superiority of the winning ticket (3.5).

## 3.1 | Notations and Definitions

［#25］
For a media recommender system containing $M$ users and $N$ items, let $\mathcal{U} = \{u_1, \cdots, u_M\}$ denotes the user set and $\mathcal{I} = \{i_1, \cdots, i_N\}$ denotes the item set. Let the node set $\mathcal{V} = \{v_1, \cdots, v_{|\mathcal{V}|}\}$ denotes all the users and items, where $\mathcal{V} = \mathcal{U} \cup \mathcal{I}$ and $|\mathcal{V}| = M + N$. Let the edge set $\mathcal{E} = \{(u, i) | u \in \mathcal{U}, i \in \mathcal{I}, \text{rec}(u, i) = 1\}$ denotes the interactions between users and items, where $\text{rec}(u, i) = 1$ if there exists an interaction between the user $u$ and the item $i$, and 0 otherwise. Then we can define the user-item interaction records as a bipartite graph $\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$. Learning with $\mathcal{G}$, the purpose of deep recommender systems is to predict the item preference ranking of each user.

［#26］
For each node $v_t$ of $\mathcal{G}$, we represent it as an $F$-dimentional vector $\mathbf{x}_t \in \mathbb{R}^F$. Then the node embedding table of the whole graph $\mathcal{G}$ can be represented as follows:

［#26］
$$
\boldsymbol{X} = \{\underbrace{\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_M}_{\text{users}}; \underbrace{\mathbf{x}_{M+1}, \mathbf{x}_{M+2}, \cdots, \mathbf{x}_{M+N}}_{\text{items}}\} \in \mathbb{R}^{|\mathcal{V}| \times F}. \tag{1}
$$

［#27］
We define the user-item interaction matrix as $\mathbf{R} \in \mathbb{R}^{M \times N}$, where

［#27］
$$
\mathbf{R}[i,j] =
\begin{cases}
1 & \text{if } (v_i, v_j) \in \mathcal{E}, \\
0 & \text{otherwise}.
\end{cases} \tag{2}
$$

［#28］
Then, we can obtain the adjacency matrix of $\mathcal{G}$ as follows:

［#28］
$$
\mathbf{A} = \begin{pmatrix}
0 & \mathbf{R} \\
\mathbf{R}^\top & 0
\end{pmatrix} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}. \tag{3}
$$

### 3.2 | Backbone Model
［#29］
Taking the classical $\text{MF}^{15}$ and the advanced $\text{LightGCN}^4$ as the backbone models, we attempt to explore lottery ticket hypothesis in media recommender systems.

#### 3.2.1 | MF
［#30］
$\text{MF}^{15}$ is a classic recommender model, which aims to optimize the user-item embedding table to fit the user-item interaction records. We define the objective function of MF with bayesian personalized ranking $\text{(BPR)}^{15}$. For a pair of interacting user $u$ and item $i$, we can obtain an item $j$ not interacting with $u$ by negative sampling. We define the score of the positive instance $(u, i)$ as $\tilde{y}_{ui} = \mathbf{x}_u^\top \mathbf{x}_i$, and the score of the negative instance $(u, j)$ as $\tilde{y}_{uj} = \mathbf{x}_u^\top \mathbf{x}_j$. Then the loss function of MF is derived as follows:

［#30］
$$
L(\boldsymbol{X}) = -\sum_{u=1}^{|M|} \sum_{i \in \mathcal{N}_u} \sum_{j \notin \mathcal{N}_u} ln\sigma(\tilde{y}_{ui} - \tilde{y}_{uj}) + \lambda \|\boldsymbol{X}\|^2. \tag{4}
$$

［#31］
where $\mathcal{N}_u$ is the set of all neighbors of the node $u$. With the loss function in Eq. (4), MF tends to maximize the scores of positive instances while minimize that of negative instances, so as to integrate the historical interaction information into the user-item embedding table.

［#32］
At the stage of inference, MF predicts the score of each pair of user and item, and then generates an item preference ranking list for each user relying on the scores.

#### 3.2.2 | LightGCN
［#33］
$\text{LightGCN}^4$, which removes the unnecessary feature transformation and nonlinear activation operations inherited from graph convolution networks, achieves efficient and effective recommendation with its neighborhood aggregation and layer combination.

［#34］
At the $k$-th layer, LightGCN obtains node embeddings $\boldsymbol{X}^{(k)}$ by neighborhood aggregation mechanism as follows:

［#34］
$$
\boldsymbol{X}^{(k)}=\left(\boldsymbol{D}^{-\frac{1}{2}} \boldsymbol{A} \boldsymbol{D}^{-\frac{1}{2}}\right) \boldsymbol{X}^{(k-1)}. \tag{5}
$$

［#35］
where $\boldsymbol{D}$ is a diagonal matrix of $|\mathcal{V}| \times|\mathcal{V}|, \mathbf{D}_{i i}$ indicates the number of non-zero elements in $i$-th row of $\boldsymbol{A}$, and $\boldsymbol{X}^{(0)}=\boldsymbol{X}$ is the user-item embedding table of the model. Finally, the output vectors can be achieved by combining the node representations of all layers as follows:

［#35］
$$
\boldsymbol{O}=\alpha_{0} \boldsymbol{X}+\alpha_{1} \boldsymbol{X}^{(1)}+\alpha_{2} \boldsymbol{X}^{(2)}+\cdots+\alpha_{K} \boldsymbol{X}^{(K)}. \tag{6}
$$

［#36］
where the coefficients $\alpha_{0}, \alpha_{1}, \cdots, \alpha_{K}$ are conventionally 1, and $K$ is the total layer number of LightGCN.

［#37］
Following the previous studies⁴, we use BPR loss function to optimize LightGCN. Let $\boldsymbol{O}=\left\{\boldsymbol{o}_{1}, \boldsymbol{o}_{2}, \ldots, \boldsymbol{o}_{M+N}\right\}$ denotes the output node vectors, we define the score of the positive instance $(u, i)$ as $\hat{y}_{u i}=\boldsymbol{o}_{u}^{\mathrm{T}} \boldsymbol{o}_{i}$, and the score of the negative instance $(u, j)$ as $\hat{y}_{u j}=\boldsymbol{o}_{u}^{\mathrm{T}} \boldsymbol{o}_{j}$. Then the loss function of LightGCN is defined as follows:

［#37］
$$
L(\boldsymbol{X}, \boldsymbol{A})=-\sum_{u=1}^{|M|} \sum_{i \in N_{u}} \sum_{j \notin N_{u}} \ln \sigma\left(\hat{y}_{u i}-\hat{y}_{u j}\right)+\lambda\|\boldsymbol{X}\|^{2}. \tag{7}
$$

### 3.3 | Lottery Ticket Hypothesis in Media Recommender Systems

［#38］
In order to extend lottery ticket hypothesis to the field of recommendation, we explore lottery ticket hypothesis in media recommender systems (LTH-MRS) to find winning tickets in the most basic parameter of recommender models—user-item embedding tables. For a deep media recommender model, we assume its large dense user-item embedding table contains a small sparse sub-matrix, and when independently training the model with this sub-matrix, it can match or even outperform that with the original large matrix. This sub-matrix is called a winning ticket in media recommender systems.

［#39］
Assume that a media recommender model $f(\mathcal{G} ; \boldsymbol{X})$ achieves the best test result of $a$ after $j$ epochs of training. Let $\mathbf{M}=\{0,1\} \in \mathbb{R}^{\|\boldsymbol{X}\|_{0}}$ denotes a binary mask, then $\mathbf{M} \odot \boldsymbol{X}$ denotes a sparse sub-matrix of $\boldsymbol{X}$. Assume that $f(\mathcal{G} ; \mathbf{M} \odot \boldsymbol{X})$ achieves a test result of $a'$ after $j'$ epochs of training. Formally, LTH-MRS can be formulated as: $\exists \mathbf{M}$, s.t. $\mathrm{j}^{\prime} \leqslant \mathrm{j}, \mathrm{a}^{\prime} \geqslant$ a and $\|\mathbf{M}\|_{0} \ll\|\boldsymbol{X}\|_{0}$. The sub-matrix satisfying LTH-MRS is called a winning ticket in media recommender systems.

［#40］
Once LTH-MRS is verified and the winning tickets in media recommender systems are found, the performance of a dense model can be achieved merely with far fewer parameters. The costs of memory, training and inference can be reduced effectively, providing more possibilities for the production deployment of large scale of media recommender systems.

### 3.4 | Identifying Winning Tickets in Media Recommender Systems

［#41］
To verify LTH-MRS, we leverage the IMP algorithm to find the winning tickets of user-item embedding tables for media recommender models. The IMP algorithm process is shown in Figure 1. We use an untrainable matrix $\mathbf{M}$, where each element is a binary flag indicating whether the corresponding parameter has been pruned or not, to prevent the pruned parameters from participating in computing. Let $\boldsymbol{X}^{0}$ denotes a randomly initialized dense user-item embedding table, and $\mathbf{M}^{0}=1 \in \mathbb{R}^{\left\|\boldsymbol{X}^{0}\right\|_{0}}$ denotes the initial mask which is an all-ones matrix. Then the original dense model $f\left(\mathcal{G} ; \boldsymbol{X}^{0}\right)$ is equivalent to $f\left(\mathcal{G} ; \mathbf{M}^{0} \odot \boldsymbol{X}^{0}\right)$. We perform magnitude-based sparse pruning on the embedding table $\mathbf{M}^{0} \odot \boldsymbol{X}^{0}$ by

［#42］
![](./images/867756209267016172_1.jpg)

［#43］
FIGURE 1 Overview of our used IMP algorithm to identify the winning tickets in media recommender systems.

［#44］
```plaintext
Algorithm 1: An IMP algorithm for the user-item embedding table.
    Input: A user-item bipartite graph $\mathcal{G}$, a recommender model $f(\mathcal{G} ; \mathbf{X}^{0})$, the iterative pruning rate $pr\%$, the
           total pruning iteration $I$ and the training epoch $J$.
    Output: The lottery ticket set $S$.
  1 Initialize a mask $\mathbf{M}^{0} \leftarrow 1 \in \mathbb{R}^{\left\|\mathbf{X}^{0}\right\|_{0}}$.
  2 Initialize a lottery ticket set $S \leftarrow \oslash$.
  3 for $i \leftarrow 0$ to $I-1$ do
  4     Training: train $f(\mathcal{G} ; \mathbf{M}^{i} \odot \mathbf{X}^{0})$ with $\mathcal{G}$ for $J$ epochs to achieve $f(\mathcal{G} ; \mathbf{M}^{i} \odot \mathbf{X}^{J})$.
  5     Pruning: select $pr\%$ of non-zero lowest magnitude values in $\mathbf{M}^{i} \odot \mathbf{X}^{J}$, and set the values at the
           corresponding positions in $\mathbf{M}^{i}$ to 0 to generate $\mathbf{M}^{i+1}$.
  6     Rewinding: reset $\mathbf{X}^{J}$ to $\mathbf{X}^{0}$ to generate a new lottery ticket $\mathbf{M}^{i+1} \odot \mathbf{X}^{0}$.
  7     $S \leftarrow S \cup \{\mathbf{M}^{i+1} \odot \mathbf{X}^{0}\}$
  8 end
```

［#44］
setting the values at the corresponding positions in $\mathbf{M}^{0}$ to 0 . We define the percentage of parameters to be pruned at each iteration as the iterative pruning rate $pr\%$, and the proportion of 0 in $\mathbf{M}^{i}$ (i.e. the mask after $i$ pruning iterations) as the sparsity $\frac{\left\|\mathbf{M}^{i}\right\|_{0}}{\left\|\mathbf{M}^{0}\right\|_{0}}$. Then, an embedding sub-matrix $\mathbf{M}^{I} \odot \mathbf{X}^{0}$ with the sparsity of $1-(1-pr\% )^{I}$ can be achieved after $I$ pruning iterations.

［#45］
To find the winning tickets, we run the following operations at $i$-th iteration:
(1) Training: train the model $f(\mathcal{G} ; \mathbf{M}^{i} \odot \mathbf{X}^{0})$ for $J$ epochs to achieve $f(\mathcal{G} ; \mathbf{M}^{i} \odot \mathbf{X}^{J})$.
(2) Pruning: prune $pr\%$ of non-zero lowest magnitude values in $f(\mathcal{G} ; \mathbf{M}^{i} \odot \mathbf{X}^{J})$ to generate a updated mask $\mathbf{M}^{i+1}$.
(3) Rewinding: reset the updated dense matrix $\mathbf{X}^{J}$ to $\mathbf{X}^{0}$, creating a new lottery ticket $\mathbf{M}^{i+1} \odot \mathbf{X}^{0}$.

［#46］
Note that the operation (3) is called rewinding mechanism, which is proposed by Frankle and Carbin $^{11}$. They believed that only when the small sparse sub-networks obtained the same initialization of the original large dense networks, can they be trained independently and efficiently.

［#47］
The specifical process is detailed in Algorithm 1. When Algorithm 1 is completed (i.e., after $I$ iterations of pruning), we can achieve a lottery ticket set $S=\{\mathbf{M}^{1} \odot \mathbf{X}^{0}, \mathbf{M}^{2} \odot \mathbf{X}^{0}, \cdots, \mathbf{M}^{I} \odot \mathbf{X}^{0}\}$, where each ticket is a sparse embedding sub-matrix with the sparsity of $1-(1-pr\% ), 1-(1-pr\% )^{2}, \cdots, 1-(1-pr\% )^{I}$, respectively. Finally, we independently train and test each lottery ticket in $S$ to verify the existence of winning tickets.

### 3.5 | Complexity Analysis

［#48］
Here, we provide the complexity analyses for MF and LightGCN to show the superiority of the winning ticket.

［#49］
MF. The inference time complexity of the original dense MF is $O(M \times N \times F)$ and the memory complexity is $O(M \times F + N \times F)$, where $M$ and $N$ are the number of users and items in the recommender system, and $F$ is the size of user-item embedding vector. For a winning ticket $\mathbf{M} \odot \mathbf{X}$ of MF, we denote $\mathbf{M}_u$ as the mask of the user $u \in \mathcal{U}$, and $\mathbf{M}_i$ as the mask of the item $i \in \mathcal{I}$, where $\mathcal{U}$ and $\mathcal{I}$ are the sets of user and item, respectively. Then the inference time complexity of the winning ticket is $O\left(M \times N \times \min \left(F_u^*, F_i^*\right)\right)$, and the memory complexity is $O\left(M \times F_u^* + N \times F_i^*\right)$, where $F_u^* = \max_{u \in \mathcal{U}} \|\mathbf{M}_u\|_0$ indicates the maximum size of sparse user embeddings, and $F_i^* = \max_{i \in \mathcal{I}} \|\mathbf{M}_i\|_0$ indicates the maximum size of sparse item embeddings.

［#50］
LightGCN. For a $K$-layer LightGCN, the time complexity of neighborhood aggregation is $O\left(K \times \|\mathbf{A}\|_0 \times F\right)$, where $\mathbf{A}$ is the adjacency matrix, the time complexity of layer combination is $O(M \times F + N \times F)$ and the time complexity of similarity computation is $O(M \times N \times F)$. Considering $M + N \ll M \times N$, the inference time complexity of the original dense LightGCN is $O\left(K \times \|\mathbf{A}\|_0 \times F + M \times N \times F\right)$, and the memory complexity is $O(M \times F + N \times F)$. For a winning ticket of LightGCN, the inference time complexity is $O\left(K \times \|\mathbf{A}\|_0 \times \min \left(F_u^*, F_i^*\right) + M \times N \times \min \left(F_u^*, F_i^*\right)\right)$, and the memory complexity is $O\left(M \times F_u^* + N \times F_i^*\right)$.

［#51］
As compared with the large dense models, the winning tickets represent users and items with much smaller sparse vectors (i.e., $F_u^*, F_i^* \ll F$). Therefore, the inference time complexity and the memory complexity of the winning tickets are far lower than that of the original dense models.

---

## 4 | EXPERIMENTS

［#52］
Beginning with an introduction of the experiment settings (4.1), this section mainly attempts to explore the following questions through a series of comparison experiments:
- Q1: Does the winning ticket in media recommender systems exists? (4.2)
- Q2: Does the IMP algorithm reliably find winning tickets? (4.3)
- Q3: Does the found winning tickets outperform the other compression models? (4.4)

［#53］
To further explore the effects of different implementation details of IMP, we conduct ablation studies (4.5). On top of that, we also compare the training speed of the winning tickets and the original embedding table (4.6). Finally, we visualize the winning tickets for interpretable analysis (4.7).

---

### 4.1 | Experiment Settings

［#54］
Here, we will detail our experiment settings, including the datasets, the evaluation metrics, the backbone models and the hyperparameters settings.

［#55］
Datasets. We conducted extensive experiments on three public benchmark datasets:
- Yelp2018 $^1$ is a dataset for recommending restaurants and bars, presented in the 2018 Yelp Dataset Challenge.
- TikTok $^2$ is a dataset for recommending the TikTok videos, presented in the 2019 ICME Challenge.
- Kwai $^3$ is a dataset for recommending the Kwai videos, presented in the China MM 2018.

［#56］
The TikTok and Kwai datasets are both the real-world user-video interaction datasets under the media convergence environment. Following the previous works $^{3,32}$, we split the datasets into training set, validation set and test

［#57］
<table><thead><tr><th>Datasets</th><th>User Number</th><th>Items Number</th><th>Interaction Number</th><th>Density</th></tr></thead><tbody><tr><td>Yelp2018</td><td>31,668</td><td>38,048</td><td>1,561,406</td><td>0.00130</td></tr><tr><td>TikTok</td><td>36,638</td><td>97,117</td><td>746,546</td><td>0.00021</td></tr><tr><td>Kwai</td><td>7,010</td><td>80,631</td><td>292,042</td><td>0.00052</td></tr></tbody></table>

［#56］
set with a ratio of 7:1:2. Table 1 shows overall statistics for the three datasets. Note that the original interaction data of TikTok is too huge, so we merely use the videos within a period, and filter the data without complete modalities.

［#58］
Evaluation Metrics. We adopt Recall and Normalized Discounted Cumulative Gain (NDCG) as the main experimental evaluation metrics. For users in the test set, we follow the all-ranking protocol to evaluate the top-$K$ recommendation performance and report the average Recall@K and NDCG@K, where we set $K=20$.

［#59］
Backbone Models. We adopt MF and LightGCN as our backbone models. MF and LightGCN are representative because they are both merely user-item interaction-based deep recommender models without any external information, and the user-item embedding table is the only model parameter. (See 3.2 for more details)

［#60］
Hyper-parameters Settings. We apply the Adam $^{33}$ as the optimizer with the learning rate of 0.001, the L2-regularization weight $\lambda$ of 0.0001 and the batch size of 2048. In main experiments, the training epoch number $J$ is 1000, the iterative pruning rate $pr\%$ is 0.1 and the pruning iteration number $I$ is 30.

### 4.2 | Existence of the Winning Ticket
［#61］
We apply the IMP algorithm on Yelp2018, TikTok and Kwai datasets to obtain a series of sparse sub-networks (a.k.a. lottery tickets) of MF and LightGCN. Then each lottery ticket will be trained and tested independently to determine whether it wins or not. We present the experimental results of the 32-dimensional original embedding size in Table 2. As shown in these results, we have found a lot of winning tickets, which can achieve significantly better performance with far fewer parameters than the original models. Taking the results on Kwai dataset for example, as compared with the original MF, the winning ticket improves Recall@20 and NDCG@20 by 28.7% and 35.5%, respectively, while reducing 95.29% parameters; as compared with the original LightGCN, the winning ticket improves Recall@20 and NDCG@20 by 10.0% and 7.99%, respectively, while reducing 65.13% parameters. The above observations show that

［#62］
<table><thead><tr><th colspan="2">Evaluation Metrics</th><th colspan="4">Recall@20</th><th colspan="4">NDCG@20</th></tr><tr><th colspan="2">Sparsity (%)</th><th>0</th><th>27.1</th><th>65.13</th><th>95.29</th><th>0</th><th>27.1</th><th>65.13</th><th>95.29</th></tr></thead><tbody><tr><td rowspan="2">Yelp2018</td><td>MF</td><td>0.0459</td><td>0.0487</td><td>0.0468</td><td>0.0395</td><td>0.0372</td><td>0.0396</td><td>0.0378</td><td>0.0320</td></tr><tr><td>LightGCN</td><td>0.0600</td><td>0.0602</td><td>0.0584</td><td>0.0510</td><td>0.0488</td><td>0.0491</td><td>0.0476</td><td>0.0418</td></tr><tr><td rowspan="2">TikTok</td><td>MF</td><td>0.0851</td><td>0.0955</td><td>0.0961</td><td>0.0638</td><td>0.0500</td><td>0.0552</td><td>0.0557</td><td>0.0354</td></tr><tr><td>LightGCN</td><td>0.1423</td><td>0.1539</td><td>0.1577</td><td>0.1379</td><td>0.0828</td><td>0.0898</td><td>0.0918</td><td>0.0813</td></tr><tr><td rowspan="2">Kwai</td><td>MF</td><td>0.0411</td><td>0.0464</td><td>0.0525</td><td>0.0529</td><td>0.0318</td><td>0.0353</td><td>0.0393</td><td>0.0431</td></tr><tr><td>LightGCN</td><td>0.0779</td><td>0.0847</td><td>0.0857</td><td>0.0751</td><td>0.0638</td><td>0.0679</td><td>0.0689</td><td>0.0614</td></tr></tbody></table>

［#63］
![](./images/867756209267016172_2.jpg)

［#64］
FIGURE 2 The Recall@20 results of the lottery tickets with different sparsity of MF and LightGCN obtained by IMP, RP and OMP approaches on Yelp2018, TikTok and Kwai datasets. The positions of red stars (★) indicate the highest sparsity that winning tickets can achieve while preserving comparable performance.

［#65］
LTH-MRS is valid, and the winning tickets exist widely in media recommender systems.

## 4.3 | Effectiveness of the IMP Algorithm

［#66］
To explore whether the IMP algorithm can find the winning tickets stably and effectively, we select two representative pruning approaches¹¹ for comparison: Random Pruning (RP) randomly removes elements from a trained embedding table, and One-shot Magnitude-based Pruning (OMP) directly reduce a trained embedding table to the target sparsity based on the element magnitude without iterations. We compare the performance of the lottery tickets obtained by IMP, RP and OMP, with the setting of 32-dimensional original embedding size, to demonstrate the effectiveness of the IMP algorithm for finding winning tickets. Notably, the *Baselines* with black dotted lines in Figure 2-5 indicate the corresponding large-dense recommender models.

［#67］
From the results shown in Figure 2, we can observe that:
(1) RP can not stably find the winning tickets. In our experiments, RP merely found winning tickets of LightGCN on TikTok dataset. Moreover, the winning tickets found by RP achieved 40.7% and 27.3% declines in the highest sparsity, and achieved 4.8% and 0.7% declines in highest Recall@20, compared with that by IMP and OMP, respectively.

［#68］
<table>
<thead>
<tr><th colspan="2" rowspan="3">Evaluation Metrics</th><th colspan="4">Recall@20</th><th colspan="4">NDCG@20</th></tr>
<tr><th colspan="2" rowspan="2">64</th><th colspan="2" rowspan="2">128</th><th colspan="2" rowspan="2">64</th><th colspan="2" rowspan="2">128</th></tr>
<tr></tr>
<tr><th colspan="2">Sparsity (%)</th><th>50</th><th>75</th><th>50</th><th>75</th><th>50</th><th>75</th><th>50</th><th>75</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Yelp2018</td><td>LCM</td><td>0.0558</td><td>0.0459</td><td>0.0610</td><td>0.0537</td><td>0.0460</td><td>0.0374</td><td>0.0493</td><td>0.0438</td></tr>
<tr><td>PEP</td><td>0.0584</td><td>0.0570</td><td>0.0627</td><td>0.0601</td><td>0.0473</td><td>0.0459</td><td>0.0507</td><td>0.0483</td></tr>
<tr><td>Winning Ticket</td><td>0.0635</td><td>0.0609</td><td>0.0666</td><td>0.0651</td><td>0.0525</td><td>0.0500</td><td>0.0545</td><td>0.0530</td></tr>
<tr><td rowspan="3">TikTok</td><td>LCM</td><td>0.1508</td><td>0.1341</td><td>0.1594</td><td>0.1500</td><td>0.0895</td><td>0.0791</td><td>0.0941</td><td>0.0885</td></tr>
<tr><td>PEP</td><td>0.1640</td><td>0.1567</td><td>0.1814</td><td>0.1801</td><td>0.0956</td><td>0.0901</td><td>0.1068</td><td>0.1047</td></tr>
<tr><td>Winning Ticket</td><td>0.1851</td><td>0.1829</td><td>0.1911</td><td>0.1913</td><td>0.1085</td><td>0.1064</td><td>0.1128</td><td>0.1130</td></tr>
<tr><td rowspan="3">Kwai</td><td>LCM</td><td>0.0690</td><td>0.0640</td><td>0.0742</td><td>0.0652</td><td>0.0585</td><td>0.0542</td><td>0.0544</td><td>0.0538</td></tr>
<tr><td>PEP</td><td>0.0840</td><td>0.0836</td><td>0.0862</td><td>0.0860</td><td>0.0684</td><td>0.0673</td><td>0.0696</td><td>0.0695</td></tr>
<tr><td>Winning Ticket</td><td>0.0867</td><td>0.0856</td><td>0.0888</td><td>0.0882</td><td>0.0696</td><td>0.0693</td><td>0.0706</td><td>0.0705</td></tr>
</tbody>
</table>

［#69］
(2) IMP and OMP can both stably find the winning tickets, but the winning tickets found by IMP significantly and consistently outperform that by OMP. Specifically, on Yelp2018, TikTok and Kwai datasets, as compared with OMP, the winning tickets of MF found by IMP achieve 30.8%, 25.2% and 2.5% improvements in the highest sparsity, and achieve 18.8%, 8.8% and 1.1% improvements in the highest Recall@20; the winning tickets of LightGCN found by IMP achieve 22.0%, 15.6% and 8.9% improvements in the highest sparsity, and achieve 6.6%, 4.0% and 3.4% improvements in the highest Recall@20.

［#70］
The above observations show that our used IMP algorithm can reliably find the winning tickets, and the winning tickets found by IMP always outperform that by the compared approaches.

### 4.4 | Performance of the Winning Ticket

［#71］
To further demonstrate the performance of the found winning tickets, we compare against two competitive baselines: Linear Compression Model (LCM) $^4$, which compresses the large embedding table into a small dense matrix with trainable feature transformation; and Plug-in Embedding Pruning model (PEP)$^{10}$, which prunes the embedding table into a small sparse matrix with a trainable dynamic pruning threshold.

［#72］
We present the comparison results of winning tickets, LCM and PEP for LightGCN in Table 3, demonstrating that our found winning tickets remarkably outperform LCM and PEP with equivalent quantities of parameters. Taking the results on Yelp2018 dataset as examples, for a 128-dimensional dense embedding table, as compared with LCM, the winning tickets improve Recall@20 and NDCG@20 by 9.18% and 10.54% when the sparsity is 50%, and improves that by 21.23% and 21.00% when the sparsity is 25%.

［#73］
As aforementioned, it is generally considered that dense models are easier to train for better performance than sparse models. However, the experimental results show that the sparse compressed models—winning tickets and PEP, can consistently outperform the dense compressed model—LCM, which seems a little counter-intuitive. The reason we believed is that, as a dense model, LCM represents different users and items with a uniform size, neglecting the diversity and the specificity among users and items. Hence, it may be hard for LCM to handle the heterogeneity of users and items with different popularities. By contrast, our used IMP algorithm automatically determines an

［#74］
![](./images/867756209267016172_3.jpg)

［#75］
FIGURE 3 The Recall@20 results of the lottery tickets of MF and LightGCN with different original embedding sizes on Yelp2018, TikTok and Kwai datasets, where the IMP-id indicates the IMP with original embedding size of i.

［#76］
appropriate embedding size for each user or item by pruning unimportant parameters during searching for winning tickets, and PEP also achieves this property with its dynamic threshold mechanism. As compared with LCM, winning tickets and PEP can express the heterogeneity of users and items better by assigning different embedding sizes for them. The various sparsity of embedding vectors describes the intrinsic diversity among different users and items, and thus improves the performance of winning tickets.

［#77］
Moreover, as shown in Table 3, although the dynamic threshold mechanism in PEP remarkably speeds up the process of pruning, the performance of final sparse models is worse than that of winning tickets. In addition, the performance of PEP highly depends on hyperparameters for the trainable pruning threshold, so it is required to consume expensive time for tuning those hyperparameters carefully. In terms of performance, simplicity and stability, the winning tickets has shown great superiority as compared with PEP.

## 4.5 | Ablation Study

［#78］
To further explore the effects and contributions of different implementation details of IMP, including the original embedding size, the iterative pruning rate $pr\%$ and the rewinding mechanism, we conduct a series of ablation studies.

［#79］
Effect of Original Embedding Size. We present the performance of winning tickets with different original embed-

［#80］
![](./images/867756209267016172_4.jpg)

［#81］
FIGURE 4 The performance comparison of IMP algorithm with different iterative pruning rates, where the IMP-ipr indicates the IMP with iterative pruning rate of i.

［#82］
![](./images/867756209267016172_5.jpg)

［#83］
FIGURE 5 The ablation study about rewinding mechanism, where the IMP w/o rewind indicates the IMP algorithm without rewinding mechanism.

［#84］
ding sizes in Figure 3, from which it can be observed that: (1) for the same deep recommender models with different sizes, with the increase of sparsity, the performance trends of lottery tickets are almost consistent; (2) for deep rec-ommender models with different original embedding sizes, the winning tickets widely exist and always can be found by our used IMP algorithm.

［#85］
Effect of Iterative Pruning Rate. We utilize IMP algorithm with different $pr\%$ (5%, 10% and 20%) to find winning tickets of LightGCN on Yelp2018 dataset, and present the results in Figure 4, from which it can be observed that: (1) the highest sparsity achieved with different $pr\%$ is almost the same; (2) lower $pr\%$ achieves higher best Recall@20; (3) higher $pr\%$ takes fewer steps to reach the sparsest winning ticket. These observations demonstrate that a high iterative pruning rate can effectively improve the search efficiency, while it may miss the optimal winning ticket. Therefore, to trade off effectiveness and efficiency, it is necessary to set a reasonable $pr\%$ when applying the IMP algorithm to find winning tickets.

［#86］
Effect of Rewinding Mechanism. To explore the importance of rewinding mechanism in our IMP algorithm, we conduct ablation studies for MF and LightGCN on Yelp2018 dataset. It can be observed in Figure 5 that compared with IMP w/o rewind, (1) the IMP with rewinding mechanism can find winning tickets more stably and reliably; (2) the IMP with rewinding mechanism improves the performance in Recall@20 and sparsity significantly and consistently.

［#87］
Overall, in IMP algorithm, the iterative pruning is a search technique for winning tickets, while the rewinding mechanism ensures the stability of the search and the effectiveness of the winning tickets.

［#88］
![](./images/867756209267016172_6.jpg)

［#89］
FIGURE 6 Relative training epochs of winning tickets and original embeddings.

## 4.6 | Comparison of Training Speed

［#90］
We compare the training speed of winning tickets and original embeddings to show that during independently training, the winning tickets can achieve comparable performance to the original dense embedding table with faster learning speed. We take the consumed training epochs of winning tickets as one unit, and then present the relative training epochs of original embeddings in Figure 6, from which we can observe that the training speed of winning tickets is far higher than that of original embeddings. For example, the winning ticket of MF-32d reached comparable performance with a learning speed 8 times the original embedding table. These experimental results demonstrate the huge potential of the winning ticket for reducing expensive training cost of large-scale media recommender systems.

## 4.7 | Case Study

［#91］
We visualize the original embedding table and the winning ticket on Yelp2018 dataset for case study. Due to the limited space, we merely show the first 10 users and items. As shown in Figure 7, the highly sparse winning ticket achieved by IMP tends to preserve more critical features (of high magnitude) for different users and items. During the iterations of identifying winning tickets, the IMP algorithm can reduce the parameter scale of redundancy features while preserving the significance of those meaningful features. By denoising those unimportant features, we can obtain a sparse embedding table (winning ticket) that can achieve comparable test performance of the full embedding with much fewer parameters.

# 5 | CONCLUSION AND FUTUTE WORK

［#92］
To address the problem of parameter redundancy in media recommender systems, we extended the lottery ticket hypothesis to the field of recommendation. We studied lottery ticket hypothesis in media recommender systems, exploiting IMP algorithm to find winning tickets of the user-item embedding table in two representative deep recommender models — MF and LightGCN. Empirical results on three real-world datasets showed the winning tickets can achieve better performance with much fewer parameters and faster training speed than the full user-item embeddings.

［#93］
In the future, we plan to explore: (1) **efficient schemes for identifying winning tickets**. The iterative identification of IMP requires a costly train-prune-retrain process, which limits the practical benefits; (2) **winning tickets for**

［#94］
![](./images/867756209267016172_7.jpg)

［#95］
**FIGURE 7** Visualization of winning ticket and original embedding.

［#96］
implicit feedback interaction records. There exists a lot of noisy data in the implicit feedback interaction records of recommender systems. For example, a noisy interaction will be produced when a user accidentally clicks on an un-interested item. Huge number of interaction records will lead to an expensive training cost, while the serious noisy interactions will mislead the learning of recommender models. (3)inductive mode for new-coming users and items. The proposed method of this paper is limited to the transductive setting. In each iteration, we performed pruning on the whole embedding matrix, which allowed us to optimize the networks globally, but also limited the generalization to new-coming users/items. In the future, we will consider extending our method to the inductive setting.

## ACKNOWLEDGEMENTS
［#97］
This work is supported by the National Natural Science Foundation of China (U19A2079, 61972372, 62121002) and National Key Research and Development Program of China (2020YFB1406703).

## ENDNOTES
［#55］
1 https://www.kaggle.com/yelp-dataset/yelp-dataset
［#55］
2 https://www.biendata.xyz/competition/icmechallenge2019
［#55］
3 https://www.kuaishou.com/activity/uimc
［#71］
4 https://github.com/gusye1234/KD_on_Ranking

## REFERENCES































