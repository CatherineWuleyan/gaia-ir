# Graph Neural Networks at a Fraction

［#1］
Rucha Bhalchandra Joshi¹⋆, Sagar Prakash Barad²³, Nidhi Tiwari⁴, and
Subhankar Mishra²³

［#2］
¹ The Cyprus Institute, Nicosia, Cyprus
² National Institute of Science Education and Research, Bhubaneswar, India
³ Homi Bhabha National Institute, Mumbai, India
⁴ Microsoft Ltd., India

［#2］
r.joshi@cyi.ac.cy, sagar.barad@niser.ac.in, nidhitiwari@microsoft.com,
smishra@niser.ac.in

［#3］
**Abstract.** Graph Neural Networks (GNNs) have emerged as powerful tools for learning representations of graph-structured data. In addition to real-valued GNNs, quaternion GNNs also perform well on tasks on graph-structured data. With the aim of reducing the energy footprint, we reduce the model size while maintaining accuracy comparable to that of the original-sized GNNs. This paper introduces Quaternion Message Passing Neural Networks (QMPNNs), a framework that leverages quaternion space to compute node representations. Our approach offers a generalizable method for incorporating quaternion representations into GNN architectures at one-fourth of the original parameter count. Furthermore, we present a novel perspective on Graph Lottery Tickets, redefining their applicability within the context of GNNs and QMPNNs. We specifically aim to find the initialization lottery from the subnetwork of the GNNs that can achieve comparable performance to the original GNN upon training. Thereby reducing the trainable model parameters even further. To validate the effectiveness of our proposed QMPNN framework and LTH for both GNNs and QMPNNs, we evaluate their performance on real-world datasets across three fundamental graph-based tasks: node classification, link prediction, and graph classification. Our code is available at project's GitHub repository.

［#4］
**Keywords:** Graph Neural Networks · Pruning · Quaternion Graph Neural Networks · Lottery Ticket Hypothesis.

## 1 Introduction

［#5］
Quaternions, a hypercomplex number system, offer several advantages over traditional real and complex numbers, making them well-suited for various tasks in deep learning. Incorporating quaternions into deep neural networks holds significant promise for enhancing their expressive power and versatility. When used in a quaternion setting, Quaternion deep neural networks perform better than

［#6］
⋆ This research was done while R. Joshi was a student at NISER, Bhubaneswar¹ and HBNI, Mumbai².

［#7］
their natural counterparts [9]. The added advantage of the quaternions is that they reduce the degree of freedom of parameters by one-fourth, which effectively means that only one-fourth of the parameters must be tuned. The inherent mul- tidimensional nature of quaternions, coupled with their capacity to diminish parameter counts significantly, suggests that hyper-complex numbers are more suitable than real numbers for crafting more efficient models in multidimensional spaces with lesser trainable parameter counts.

［#8］
Most GNNs learn the embeddings in the Euclidean space. A recent approach [8] proposes learning graph embeddings in Quaternion space. However, it is the quaternion version of one model GCN [5], and it cannot be generalized further. The GNN variants exploit different interesting properties of the graphs to learn the representations efficiently. A quaternion framework should be able to main- tain these characteristics of the GNN model and embrace them in the quaternion space. With this realization, in this work, we present a generalizable quaternion framework that can adapt to any GNN variant.

［#9］
Additionally, unlike in the case of the non-GNN deep neural networks [4,6,7], the existing QGNNs do not leverage the quaternion space to reduce the number of parameters. In these non-GNN methods, the input features are usually divided in such a way that overall the four components of the quaternion vector can aggregate them and make a meaningful transformation out of it. The previous work [8], however, uses each quaternion component to work with all the features. This adds four times the computational overhead, making the model expensive in terms of training and inference time. Additionally, it makes the outputs very sensitive to even a tiny change in the input, as this type of quaternion GNN amplifies the tiny change. To this end, we restrict our quaternion model from quadrupling the number of parameters, thereby reducing the trainable parameter count to one-fourth of that of real, and this also prohibits the output from being extremely sensitive to the tiny change in the input features.

［#10］
In this work, our goal is to attain Quaternion Graph Neural Networks that are expressive and have fewer parameters. To do this, we first propose a generic framework to get the Quaternion model corresponding to any GNN model with real representations. Secondly, we present a framework to sparsify the Quater- nion GNNs so that we reduce the number of parameters by finding a lottery ticket. We generalize the Lottery Ticket Hypothesis [2] to Quaternion Graph Neural Networks. For graphs, the existing unified graph lottery ticket, which prunes the input graph as well as the GNN to obtain the winning graph lottery ticket, has some drawbacks. Primarily, it is not feasible to prune the input graph along with the GNNs in graph-level tasks. UGS lacks effectiveness for transfer learning purposes because of its dependency on the input graph. We tackle this by redefining the lottery ticket for graphs. Figure 1 presents an overview of our approach.

［#11］
To summarize, this paper makes contributions as follows: (1) Quaternion Message Passing Neural Networks (QMPNNs): We present a generalizable frame- work for computing quaternion space representations using any GNN, reducing trainable parameters by 75%. (2) We redefine the graph lottery tickets from a

［#12］
![](./images/1229689055977930822_1.jpg)

［#13］
Fig. 1: Pruning a quaternion message passing neural network. Graph's features are first transformed into quaternion features. We train QMPNN, which has quaternion weights, for the given task. Furthermore, to find our proposed winning lottery ticket, we prune and train the QMPNN until we get a model with pruned quaternion weights that gives a comparable accuracy. The pruned quaternion network can be trained with only a small fraction of the parameters in the real GNN.

［#14］
different perspective for GNNs and QMPNNs. (3) To our knowledge, we are the first to empirically demonstrate the existence of graph lottery tickets in QMPNNs. (4) We provide the performance evaluation of the proposed LTH on both GNNs and QMPNNs on real-world datasets for three graph-based tasks - node classification, link prediction, and graph classification.

## 2 Background
### 2.1 Notations and Definitions
［#15］
We use the following notations in this paper. A graph $\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$ has a set of vertices $\mathcal{V}$ and a set of edges $\mathcal{E}$. The feature vector corresponding to a node $v \in \mathcal{V}$ is given as $\mathrm{x}_v$ in $\mathbb{R}^F$. The feature matrix for the graph is given as $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times F}$. The edge $e_{ij} = (v_i, v_j) \in \mathcal{E}$ if and only if there is an edge between two nodes $v_i$ and $v_j$. In the adjacency matrix representation $\mathbf{A} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ of the graph, the $\mathbf{A}_{ij} = 1$ if $e_{ij} \in \mathcal{E}$, and 0 otherwise.

［#16］
The intermediate representations in layer $l$ corresponding to a node $v$ are given by $h_v^{(l)}$. Similarly, we denote representations in layer $l$ for node $v$ by $h_v^{(l), Q}$, using the additional superscript $Q$ to represent quaternion.

### 2.2 Quaternions
［#17］
The set of quaternions is denoted by $\mathbb{H}$. A quaternion $q$ is denoted as $q = q_r + q_i \mathrm{i} + q_j \mathrm{j} + q_k \mathrm{k}$, where $q_r, q_i, q_j, q_k \in \mathbb{R}$ and $\mathrm{i}, \mathrm{j}, \mathrm{k}$ are imaginary units. They follow the relation $\mathrm{i}^2 = \mathrm{j}^2 = \mathrm{k}^2 = \mathrm{ijk} = -1$.

［#18］
Addition: The quaternion addition of two quaternoons $q$ and $p$ is component-wise: $(q_r + q_i \mathrm{i} + q_j \mathrm{j} + q_k \mathrm{k}) + (p_r + p_i \mathrm{i} + p_j \mathrm{j} + p_k \mathrm{k}) = (q_r + p_r) + (q_i + p_i)\mathrm{i} + (q_j + p_j)\mathrm{j} + (q_k + p_k)\mathrm{k}$.

［#19］
Multiplication: Quaternions can be multiplied by each other. The multiplication is associative, i.e., $(pq)r = p(qr)$ for $p,q,r \in \mathbb{H}$. It is also distributive, i.e., $(p+q)r = pr + qr$. However, the multiplication is not commutative, i.e., $pq \neq qp$. When multiplied with the scalar $\lambda$, it gives $\lambda q = \lambda q_r + \lambda q_i \mathrm{i} + \lambda q_j \mathrm{j} + \lambda q_k \mathrm{k}$. Multiplication of two quaternions $q$ and $p$ is defined by the Hamilton product $q \otimes p$. This is given in the matrix form as follows:

［#19］
$$
q \otimes p = \begin{bmatrix} 1 \\ \mathrm{i} \\ \mathrm{j} \\ \mathrm{k} \end{bmatrix}^\top \begin{bmatrix} q_r & -q_i & -q_j & -q_k \\ q_i & q_r & -q_k & q_j \\ q_j & q_k & q_r & -q_i \\ q_k & -q_j & q_i & q_r \end{bmatrix} \begin{bmatrix} p_r \\ p_i \\ p_j \\ p_k \end{bmatrix}
\tag{1}
$$

［#20］
Conjugation: $\bar{q} = q_r - q_i \mathrm{i} - q_j \mathrm{j} - q_k \mathrm{k}$ is the conjugation of the quaternion $q$ given above.

［#21］
Norm: The norm of quaternion $q$ is given as $\| q \| = \sqrt{q_r^2 + q_i^2 + q_j^2 + q_k^2}$.

## 3 Quaternion Message Passing Neural Networks

［#22］
Graph Neural Networks (GNNs) analyze graph-structured data by updating node representations through message passing. At each layer, messages from neighboring nodes are aggregated and used to update node features. The process is formalized as:

［#22］
$$
m_{uv} = \operatorname{MESSAGE}(h_u^{(l)}, h_v^{(l)}, e_{uv})
\tag{2}
$$

［#22］
$$
h_u' = \operatorname{AGGREGATE}(m_{uv}), \forall v \in \mathcal{N}(u)
\tag{3}
$$

［#22］
$$
h^{(l+1)} = \operatorname{UPDATE}(h_u^{(l)}, h_u')
\tag{4}
$$

［#23］
Here, MESSAGE computes messages based on node features $h_u^{(l)}, h_v^{(l)}$ and edge features $e_{uv}$; AGGREGATE collects messages from neighbors $\mathcal{N}(u)$; and UPDATE refines the node representation. These functions, often learnable and differentiable, enable GNNs to be trained via backpropagation.

［#24］
Quaternion Graph Neural Networks (QGNNs) refer to neural networks that utilize quaternion weights to perform computations within the quaternion vector space. Notably, QGNN, as developed by [8] is a quaternion variant of the Graph Convolutional Network (GCN) layer only. This is the first limitation of QGNN, as it cannot be generalized to other GNN variants. The second limitation of QGNN is by construction, number of trainable parameters remain the same as that of real-valued GCN. While leveraging the quaternion space to generate expressive representations for nodes, we also seek to take advantage of the quaternions by reducing the number of trainable parameters. To this end, we propose a generalized framework to give the quaternion equivalent of any graph neural network method.

［#25］
In the quaternion variant of any GNNs, in the equations 2, 3, 4, we ensure that the MESSAGE, AGGREGATE and UPDATE functions follow the quaternion operations as given in section 2.2. We describe this in more detail below:

［#26］
To begin, we have the features represented using the quaternions. $F$ is the number of features corresponding to every node. To use these features with quaternions, we make sure that $F$ is divisible by four so that we can associate $F/4$ real-valued features with a real and three imaginary components of quaternion. Consequently, the feature vector $h_{v}^{(l),Q}$ of node $v$ in quaternion form is represented as $h_{v}^{(l),Q}=h_{r}^{(l)}+h_{i}^{(l)}\mathrm{i}+h_{j}^{(l)}\mathrm{j}+h_{k}^{(l)}\mathrm{k}$.

［#27］
In every layer $l$ of GNN, the learnable functions have associated parameters, which we represent using $W^{(l),Q}$. This weight matrix $W^{(l),Q}$ has consists of four real-valued components $W_{r}^{(l)}$, $W_{i}^{(l)}$, $W_{j}^{(l)}$, $W_{k}^{(l)}$ of the quaternion weight matrix. Specifically, $W^{(l),Q}=W_{r}^{(l)}+W_{i}^{(l)}\mathrm{i}+W_{j}^{(l)}\mathrm{j}+W_{k}^{(l)}\mathrm{k}$.

［#28］
The weights are multiplied with feature maps using the quaternion multiplication rule given in equation 1. Following this, it is done as follows:
［#28］
$$
W^{(l+1),Q}=W^{(l),Q}\otimes h_{v}^{(l),Q}\tag{5}
$$

［#28］
$$
\begin{bmatrix}
\mathcal{R}(W^{(l),Q})\\
\mathcal{I}(W^{(l),Q})\\
\mathcal{J}(W^{(l),Q})\\
\mathcal{K}(W^{(l),Q})
\end{bmatrix}
=
\begin{bmatrix}
W_{r}^{(l)} & -W_{i}^{(l)} & -W_{j}^{(l)} & -W_{k}^{(l)}\\
W_{i}^{(l)} & W_{r}^{(l)} & -W_{k}^{(l)} & W_{j}^{(l)}\\
W_{j}^{(l)} & W_{k}^{(l)} & W_{r}^{(l)} & -W_{i}^{(l)}\\
W_{k}^{(l)} & -W_{j}^{(l)} & W_{i}^{(l)} & W_{r}^{(l)}
\end{bmatrix}
\times
\begin{bmatrix}
h_{r}^{(l)}\\
h_{i}^{(l)}\\
h_{j}^{(l)}\\
h_{k}^{(l)}
\end{bmatrix}\tag{6}
$$

［#28］
where $\mathcal{R},\mathcal{I},\mathcal{J},\mathcal{K}$ denote the four imaginary components of the product of weights with the node representations.

［#29］
Due to quaternion weights being multiplied with quaternion features, the degrees of freedom are reduced by one-fourth. The product in equation 5 combines components $(r,i,j,k)$, capturing complex interdependencies. Each quaternion weight, as shown in equation 6, encodes spatial correlations across features, resulting in more expressive networks compared to real-valued counterparts [11]. Additionally, the trainable parameters are reduced to one-fourth of those in an equivalent GNN with real weights, aiding in model size reduction.

### 3.1 Differentiability
［#30］
For the QMPNNs to learn, the quaternion versions of learnable functions from MESSAGE, AGGREGATE, and UPDATE should be differentiable. For our framework, the differentiability follows from the generalized complex chain rule for a real-valued loss function, which is provided in much detail in Deep Complex Networks [11] and Deep Quaternion Networks [3].

### 3.2 Computational Complexity
［#31］
Four feature values are clubbed together to form a single quaternion in QMPNNs, hence $F/4$ quaternions are necessary to transform the feature vector. The transformed feature space is of dimension $F'$. While using quaternion weights, these

［#31］
are considered as the combination of $F'/4$ quaternions. The number of parameters of the quaternion weight matrix, hence, is $F/4 \times F'/4$, with each of them having four quaternion components. Since the components $W_{r}^{(l)}, W_{i}^{(l)}, W_{j}^{(l)}$, and $W_{k}^{(l)}$ of the quaternion weight $W^{(l),Q}$ are shared during the Hamilton product, the degree of freedom is reduced by one-fourth of that of the real weight matrix. For quaternion matrix multiplications has same time complexity as that of the real ones, we reduce the number of trainable parameters without compromising on the time complexity due to use of quaternion parameters.

## 4 Lottery Ticket Hypothesis on QMPNNs

［#32］
The lottery ticket hypothesis (LTH) [2] states that a randomly initialized, dense neural network contains a sub-network that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.

［#33］
Different from the lottery tickets defined previously [1,12,13,14] - that consider a subgraph of a graph with or without the subnetwork of the GNN as a ticket, we define the LTH for Graph Neural Networks and Quaternion GNNs. In a graph neural network, $f(x; W)$, with initial parameters $W^{(0)}$, using Stochastic Gradient Descent (SGD), we train on the training set, minimizes to a minimum validation loss $l$ in $j$ number of iterations achieving accuracy $a$. The lottery ticket hypothesis is that there exists $m$ when optimizing with SGD on the same training set, $f$ attains the test accuracy $a' \geq a$ with a validation loss $l' \leq l$ in $j' \leq j$ iterations, such that $\|m\| \ll \|W\|$, thereby reducing the number of parameters. The sub-network with parameters $m \odot W^Q$ is the winning lottery ticket, with the mask $m$. Algorithm 1 sparsifies the Graph Neural Network which is required to find the winning lottery ticket. The iterative algorithm that finds it is described in algorithm 2.

［#34］
```plaintext
Algorithm 1 GNN Sparsification
Require: Input graph $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$, GNN's initialization $W^{(0)}$, initial mask $m^0 = 1 \in \mathbb{R}^{\|W\|}$, step size $\eta, \lambda$
Ensure: Sparsified weight mask $m$
1: for $i \leftarrow 0$ to $N-1$ do
2:    Forward $f(\cdot, m^i \odot W)$ with $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$ to compute the loss $\mathcal{L}_{\text{task}}$
3:    Back-propagate to update $W^{(i+1)} \leftarrow W^{(i)} - \eta \nabla_W \mathcal{L}_{\text{task}}$
4:    Update $m^{i+1} \leftarrow m^i - \lambda \nabla_{m^i} \mathcal{L}_{\text{task}}$
5: end for
6: Set $p=20\%$ of the lowest magnitude values in $m^N$ to 0 and others to 1, get mask $m$
```

［#35］
We primarily differ from the definition of GLT in the previous works, as we observe that this is not suitable for graph-level tasks [10,8]. Popular methods

［#35］
such as UGS [1], AdaGLT [14] consider only the node-classification and the link prediction task. In graph-level tasks, sparsifying the input graph does not have any significance. For graph-level tasks, the structure and the features in the entire graph are considered, unlike in the case of node- or edge-level tasks where only the nodes in the local neighborhood and their features are considered by a graph neural network. Also, in the case of transfer learning, where the pruned sub-network is to be further finetuned for a different but related task, the dependency of the winning ticket on the input graph restricts us from fine-tuning it.

### 4.1 Quaternion Graph Lottery Tickets
［#36］
Similar to finding a winning lottery ticket in GNNs, given a QMPNN $f(\cdot, W^Q)$ and a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$, the subnetwork, i.e., winning lottery ticket, of the QMPNN is defined as $f(\cdot, m \odot W^Q)$, where $m$ is the binary mask on parameters. To find the winning lottery ticket in a QMPNN, we use the algorithms 1 and 2 with the quaternion-values model weights and input features.

［#37］
```
Algorithm 2 Iterative algorithm to find Graph Lottery Ticket
Require: Input graph $\mathcal{G}=(\mathcal{V}, \mathcal{E}, \mathbf{X})$, GNN's initialization $W^{(0)}$, initial mask $m^0 = 1 \in \mathbb{R}^{\|W\|}$, sparsity level $s$
Ensure: Lottery Ticket $f((\mathcal{V}, \mathcal{E}, \mathbf{X}), m \odot W^0)$
 1: while $1 - \frac{\|m\|}{\|W\|}$ do
 2:   Sparsify GNN $f(\cdot, m \odot W^0)$ using algorithm 1
 3:   Update $m$ according to step 2
 4:   Reset GNN's weight to $W^0$
 5: end while
```

## 5 Experiments
［#38］
The effectiveness of QMPNNs for different GNN variants is validated with extensive experimentation on different real-world standard datasets. We also verify the existence of the GLTs as per our definition. We evaluated their performance on three tasks, node classification, link prediction, and graph classification.

### 5.1 Datasets
［#39］
Tables 1 and 2 summarize the datasets and their statistics. For quaternion models, dataset features and classes were adjusted to be divisible by 4 by padding features with the average value and adding dummy classes. For instance, Cora's feature size was padded from 1,433 to 1,436, and the number of classes was increased from 7 to 8.

［#40］
<table>
<caption>Table 1: Node classification and link prediction datasets statistics</caption>
<thead>
<tr>
<th>Dataset</th>
<th>#Nodes</th>
<th>#Edges</th>
<th>#Features</th>
<th>#Classes</th>
<th>Metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cora</td>
<td>2,708</td>
<td>5,429</td>
<td>1,433</td>
<td>7</td>
<td>Accuracy, ROC-AUC</td>
</tr>
<tr>
<td>Citeseer</td>
<td>3,327</td>
<td>4,732</td>
<td>3,703</td>
<td>6</td>
<td>Accuracy, ROC-AUC</td>
</tr>
<tr>
<td>PubMed</td>
<td>19,717</td>
<td>44,338</td>
<td>500</td>
<td>3</td>
<td>Accuracy, ROC-AUC</td>
</tr>
<tr>
<td>ogbn-arxiv</td>
<td>169,343</td>
<td>1,166,243</td>
<td>128</td>
<td>40</td>
<td>Accuracy, ROC-AUC</td>
</tr>
<tr>
<td>ogbbl-collab</td>
<td>235,868</td>
<td>2,358,104</td>
<td>128</td>
<td>0</td>
<td>Hits@50</td>
</tr>
</tbody>
</table>

［#41］
<table>
<caption>Table 2: Graph classification datasets statistics</caption>
<thead>
<tr>
<th>Dataset</th>
<th>#Graphs</th>
<th>Avg. #Edges</th>
<th>#Classes</th>
<th>Metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>MUTAG</td>
<td>188</td>
<td>744</td>
<td>2</td>
<td>Accuracy</td>
</tr>
<tr>
<td>ENZYMES</td>
<td>600</td>
<td>1,686</td>
<td>6</td>
<td>Accuracy</td>
</tr>
<tr>
<td>PROTEINS</td>
<td>1,113</td>
<td>3,666</td>
<td>2</td>
<td>Accuracy</td>
</tr>
<tr>
<td>ogbg-molhiv</td>
<td>41,127</td>
<td>40</td>
<td>2</td>
<td>Accuracy</td>
</tr>
</tbody>
</table>

### 5.2 Experimental Setup
［#42］
- **GNN Architecture:** Two-layer GCN, GAT, and GraphSAGE models with 128 hidden units were used for node classification and link prediction on Cora, Citeseer, PubMed, ogbn-arxiv, and ogbn-collab. For graph classification on MUTAG, PROTEINS, ENZYMES, and ogbg-molhiv, three-layer models with 128 hidden units per layer were employed.
- **Training and Hyperparameters:** Node classification datasets followed an 80-10-10 split, while link prediction and graph classification used an 85-5-10 split. Models were trained with a learning rate of 0.01, weight decay of $5 \times 10^{-4}$, dropout rate of 0.6, and the Adam optimizer, for up to 1000 epochs with early stopping after 200 epochs of no improvement. A prune fraction of 0.3 was applied.
- **Evaluation and Infrastructure:** Evaluation metrics are provided in Tables 1 and 2. Experiments were conducted on NVIDIA RTX 3090 GPUs with 24GB VRAM.

### 5.3 Training and Inference Details
［#43］
Our evaluation metrics, detailed in Tables 1 and 2 and following [5], include both accuracy and ROC-AUC. Accuracy measures the proportion of correctly classified instances among the total instances, providing a straightforward metric for model performance. The ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) score, representing the degree of separability, is particularly valuable for assessing performance on imbalanced datasets, such as in link prediction tasks with a substantial class imbalance between positive and negative edges. Summarizing the results across these diverse tasks, tables 3 , 4,

［#44］
![](./images/1229689055977930822_2.jpg)

［#45］
Fig. 2: Performance After Pruning: The plots show GCN, GAT, and Graph-
SAGE performance on OGBN-ARXIV (node classification), OGBL-COLLAB
(link prediction), and OGBG-MOHLIV (graph classification) at a pruning weight
fraction of 0.3. GLTs are marked by red $(\star)$ and green $(\star)$ stars, indicating com-
parable performance despite sparsity, while dashed lines represent pre-pruning
baselines. The plot sections correspond to node classification (top), link predic-
tion (middle), and graph classification (bottom).

［#46］
and 5 present the inference outcomes for all models, facilitating comprehensive
analysis.

### 5.4 Results
［#47］
GCN, GAT, and GraphSAGE results on Cora, Citesser, PubMed, and ogbn-
arxiv for node classification are collected in table 3 and 4 shows results for link
prediction on Cora, Citesser, PubMed, and ogbl-collab. The results pertaining
to graph classification tasks on MUTAG, PROTEINS, ENZYMES, and ogbg-
molhiv datasets are presented in table 5. These tables display the accuracy and
parameter count (Params) in millions for each model and dataset, with better-
performing models highlighted in bold for easy identification. Pruning results for
the same models on ogbn-arxiv for node classification are shown in the top section
of figure 2. The middle section of figure 2 presents results for link prediction on

［#47］
the ogbl-collab dataset. The bottom section of figure 2 illustrates the results for graph classification tasks on the ogbg-molhiv dataset. An extensive list of results is available at project's GitHub repository.

［#48］
Table 3: Node classification results for GCN, GAT, and GraphSAGE mod- els on semi-supervised graph datasets, including Cora, Citeseer, PubMed, and OGBN-ARXIV. Accuracy values are reported with their standard deviations, along with the corresponding GFLOPs (denoted as GF) and parameter counts (in K units, denoted as GF).

［#49］
<table>
<thead>
<tr>
<th>Model</th>
<th colspan="3">Cora</th>
<th colspan="3">Citeseer</th>
<th colspan="3">PubMed</th>
<th colspan="3">OGBN-Arxiv</th>
</tr>
<tr>
<th></th>
<th>Accuracy</th>
<th>Par (K)</th>
<th>GF</th>
<th>Accuracy</th>
<th>Par (K)</th>
<th>GF</th>
<th>Accuracy</th>
<th>Par (K)</th>
<th>GF</th>
<th>Accuracy</th>
<th>Par (K)</th>
<th>GF</th>
</tr>
</thead>
<tbody>
<tr>
<td>GCN</td>
<td>85.2 ± 1.3</td>
<td>140.5</td>
<td>0.33</td>
<td>73.5 ± 1.8</td>
<td>125.7</td>
<td>0.29</td>
<td>79.1 ± 1.2</td>
<td>131.9</td>
<td>0.31</td>
<td>72.8 ± 1.5</td>
<td>145.1</td>
<td>0.42</td>
</tr>
<tr>
<td>QGCN</td>
<td>84.1 ± 1.5</td>
<td>42.1</td>
<td>0.11</td>
<td>72.6 ± 1.3</td>
<td>39.2</td>
<td>0.09</td>
<td>78.4 ± 1.4</td>
<td>41.7</td>
<td>0.10</td>
<td>72.3 ± 1.6</td>
<td>43.5</td>
<td>0.12</td>
</tr>
<tr>
<td>GAT</td>
<td>87.3 ± 1.0</td>
<td>143.2</td>
<td>0.37</td>
<td>75.4 ± 1.9</td>
<td>130.8</td>
<td>0.34</td>
<td>80.7 ± 1.6</td>
<td>136.3</td>
<td>0.36</td>
<td>72.5 ± 1.2</td>
<td>149.5</td>
<td>0.45</td>
</tr>
<tr>
<td>QGAT</td>
<td>86.4 ± 1.7</td>
<td>43.7</td>
<td>0.14</td>
<td>74.8 ± 1.1</td>
<td>41.6</td>
<td>0.12</td>
<td>79.8 ± 1.5</td>
<td>42.9</td>
<td>0.13</td>
<td>72.9 ± 1.8</td>
<td>44.5</td>
<td>0.15</td>
</tr>
<tr>
<td>SAGE</td>
<td>85.5 ± 1.2</td>
<td>72.6</td>
<td>0.23</td>
<td>74.9 ± 1.4</td>
<td>66.1</td>
<td>0.19</td>
<td>80.2 ± 1.9</td>
<td>68.7</td>
<td>0.20</td>
<td>73.0 ± 1.0</td>
<td>75.3</td>
<td>0.26</td>
</tr>
<tr>
<td>QSAGE</td>
<td>84.9 ± 0.8</td>
<td>22.1</td>
<td>0.08</td>
<td>74.3 ± 1.5</td>
<td>20.5</td>
<td>0.07</td>
<td>80.5 ± 1.3</td>
<td>21.3</td>
<td>0.08</td>
<td>72.8 ± 1.4</td>
<td>23.6</td>
<td>0.10</td>
</tr>
</tbody>
</table>

［#50］
The results indicate that quaternion models (QGCN, QGAT, and QSAGE) perform equally well or better than their real counterparts (GCN, GAT, and GraphSAGE) under the same hyperparameters and training parameters. For in- stance, QGCN shows slightly improved accuracy on the PubMed dataset, and QGAT performs nearly as well as GAT on Cora and Citeseer for node classi- fication tasks. Moreover, quaternion models consistently outperform their real counterparts in other tasks, including link prediction and graph classification, demonstrating superior performance in almost all cases. Furthermore, it is con- sistently shown that even when the real models perform better than QMPNNs, they are mostly ahead by only marginal values.

［#51］
Table 4: Link prediction on various datasets including CORA, CITESEER, PUBMED, and OBGL-COLLAB. ROC-AUC and Hits@50 values are reported with their standard deviations, along with the corresponding GFLOPs (denoted as GF) and parameter counts (in K units, denoted as GF).

［#52］
<table>
<thead>
<tr>
<td>Model Name</td>
<th colspan="3">CORA</th>
<th colspan="3">CITESEER</th>
<th colspan="3">PUBMED</th>
<th colspan="3">OGBL-COLLAB</th>
</tr>
<tr>
<td></td>
<th>ROC-AUC</th>
<th>Par (K)</th>
<th>GF</th>
<th>ROC-AUC</th>
<th>Par (K)</th>
<th>GF</th>
<th>ROC-AUC</th>
<th>Par (K)</th>
<th>GF</th>
<th>Hits@50</th>
<th>Par (K)</th>
<th>GF</th>
</tr>
</thead>
<tbody>
<tr>
<td>GCN</td>
<td>79.27 ± 0.47</td>
<td>96.0</td>
<td>0.23</td>
<td>78.53 ± 0.45</td>
<td>241.4</td>
<td>0.58</td>
<td>87.60 ± 0.043</td>
<td>36.3</td>
<td>0.09</td>
<td>44.02 ± 1.57</td>
<td>12.5</td>
<td>0.03</td>
</tr>
<tr>
<td>QGCN</td>
<td>83.55 ± 0.03</td>
<td>27.2</td>
<td>0.07</td>
<td>78.36 ± 4.13</td>
<td>63.6</td>
<td>0.15</td>
<td>87.66 ± 0.32</td>
<td>12.3</td>
<td>0.03</td>
<td>45.80 ± 1.35</td>
<td>6.4</td>
<td>0.02</td>
</tr>
<tr>
<td>GAT</td>
<td>93.89 ± 0.06</td>
<td>96.3</td>
<td>0.24</td>
<td>85.56 ± 0.38</td>
<td>241.6</td>
<td>0.60</td>
<td>89.32 ± 0.38</td>
<td>36.6</td>
<td>0.09</td>
<td>45.51 ± 1.61</td>
<td>12.8</td>
<td>0.03</td>
</tr>
<tr>
<td>QGAT</td>
<td>94.99 ± 0.19</td>
<td>27.5</td>
<td>0.07</td>
<td>86.59 ± 0.10</td>
<td>63.8</td>
<td>0.16</td>
<td>84.04 ± 0.54</td>
<td>12.6</td>
<td>0.03</td>
<td>32.03 ± 1.92</td>
<td>65.6</td>
<td>0.16</td>
</tr>
<tr>
<td>SAGE</td>
<td>89.00 ± 0.19</td>
<td>48.1</td>
<td>0.12</td>
<td>81.65 ± 0.13</td>
<td>120.5</td>
<td>0.30</td>
<td>83.36 ± 0.0189</td>
<td>16.1</td>
<td>0.04</td>
<td>48.50 ± 0.88</td>
<td>6.4</td>
<td>0.02</td>
</tr>
<tr>
<td>QSAGE</td>
<td>89.35 ± 0.26</td>
<td>13.6</td>
<td>0.03</td>
<td>81.93 ± 0.04</td>
<td>31.8</td>
<td>0.08</td>
<td>79.05 ± 0.32</td>
<td>6.1</td>
<td>0.02</td>
<td>48.07 ± 0.56</td>
<td>3.2</td>
<td>0.01</td>
</tr>
</tbody>
</table>

［#53］
The results presented in the tables comparing real and quaternion models, and are trained and evaluated using five different seeds. For each data point,

［#53］
we calculated the mean and standard deviation based on these evaluations. The better-performing models are determined by performing paired t-tests. We chose the model with the better performance when the difference between p-values of the models was found to be statistically significant.

［#54］
Table 5: Graph Classification on various datasets including MUTAG, PRO- TEINS, ENZYMES, and OGBG-MOLHIV. Accuracy values are reported with their standard deviations, along with the corresponding GFLOPs (denoted as GF) and parameter counts (in K units, denoted as GF).

［#55］
<table>
<thead>
<tr>
<th rowspan="2">Model Name</th>
<th colspan="3">MUTAG</th>
<th colspan="3">PROTEINS</th>
<th colspan="3">ENZYMES</th>
<th colspan="3">OGBG-MOLHIV</th>
</tr>
<tr>
<td>Accuracy</td>
<td>Par (K)</td>
<td>GF</td>
<td>Accuracy</td>
<td>Par (K)</td>
<td>GF</td>
<td>Accuracy</td>
<td>Par (K)</td>
<td>GF</td>
<td>Accuracy</td>
<td>Par (K)</td>
<td>GF</td>
</tr>
</thead>
<tbody>
<tr>
<td>GCN</td>
<td>83.54 ± 0.40</td>
<td>4.8</td>
<td>0.01</td>
<td>73.57 ± 0.45</td>
<td>9.2</td>
<td>0.02</td>
<td>72.96 ± 0.15</td>
<td>5.6</td>
<td>0.01</td>
<td>76.24 ± 0.89</td>
<td>5.1</td>
<td>0.01</td>
</tr>
<tr>
<td>QGCN</td>
<td>88.61 ± 0.66</td>
<td>4.4</td>
<td>0.01</td>
<td>71.55 ± 0.42</td>
<td>5.5</td>
<td>0.01</td>
<td>71.92 ± 1.16</td>
<td>4.6</td>
<td>0.01</td>
<td>75.85 ± 0.93</td>
<td>4.5</td>
<td>0.01</td>
</tr>
<tr>
<td>GAT</td>
<td>85.26 ± 0.02</td>
<td>5.4</td>
<td>0.01</td>
<td>72.68 ± 0.93</td>
<td>9.4</td>
<td>0.02</td>
<td>74.61 ± 0.05</td>
<td>5.8</td>
<td>0.01</td>
<td>76.24 ± 0.32</td>
<td>5.8</td>
<td>0.01</td>
</tr>
<tr>
<td>QGAT</td>
<td>87.25 ± 0.18</td>
<td>4.7</td>
<td>0.01</td>
<td>73.22 ± 0.02</td>
<td>5.8</td>
<td>0.01</td>
<td>73.35 ± 0.02</td>
<td>4.9</td>
<td>0.01</td>
<td>78.16 ± 0.78</td>
<td>4.8</td>
<td>0.01</td>
</tr>
<tr>
<td>SAGE</td>
<td>82.56 ± 2.67</td>
<td>2.5</td>
<td>0.01</td>
<td>72.62 ± 0.22</td>
<td>4.7</td>
<td>0.01</td>
<td>79.65 ± 3.06</td>
<td>2.8</td>
<td>0.01</td>
<td>71.77 ± 0.97</td>
<td>2.6</td>
<td>0.01</td>
</tr>
<tr>
<td>QSAGE</td>
<td>84.26 ± 0.22</td>
<td>2.2</td>
<td>0.01</td>
<td>73.08 ± 1.36</td>
<td>2.7</td>
<td>0.01</td>
<td>78.59 ± 0.73</td>
<td>2.3</td>
<td>0.01</td>
<td>72.98 ± 0.12</td>
<td>2.2</td>
<td>0.01</td>
</tr>
</tbody>
</table>

［#56］
It is worth noting that the performance of real-valued GNN models, on datasets with and without the dataset augmentation mentioned in the section 5.1 is the same. Hence, it serves as a fair baseline to compare the quaternion-valued GNN performance with the real-valued GNN performance on the augmented dataset. We provide the code in the repository linked in the paper.

### 5.5 Key Findings
［#57］
We summarize the effectiveness of magnitude pruning and quaternion-based models in GNNs:
1. **QMPNNs operate at 1/4th parameters of real-valued models:** QMPNNs achieve comparable or better performance than real-valued models with just 1/4th of the parameters, showcasing their efficiency and expressiveness, as shown in Tables 3, 4, and 5.
2. **GLTs exist at 1/5th or smaller of original size:** Magnitude pruning identifies lottery tickets at 1–20% of the original model size without performance loss across QGCN, QGAT, and QGraphSAGE tasks, as observed in Tables 3 and 4.
3. **GAT and GraphSAGE sparsify well; Cora is pruning-sensitive:** GATs and GraphSAGE produce sparse GLTs due to attention and sampling techniques, while Cora exhibits significant sensitivity to pruning, with performance dropping at half model size, as detailed in the comprehensive experiments available on the project’s GitHub repository.
4. **Less sparsity in graph classification GLTs:** Graph classification tasks require less sparse GLTs to capture complex global features, with smaller parameter counts due to the datasets’ limited graph size, as shown in Table 5 and Figure 2.

［#58］
5. Large graphs maintain performance: QMPNNs scale well to large graphs, as seen with ogbn-arxiv, ogbn-collab, and ogbn-molhiv datasets in Tables 3, 4, and 5, delivering comparable or better performance relative to trainable parameters.

## 6 Conclusion and Future Work

［#59］
This paper introduces Quaternion Message Passing Neural Networks (QMPNNs) as a versatile framework for graph representation learning, leveraging quaternion representations to capture intricate relationships in graph-structured data. The framework generalizes easily to existing GNN architectures with minimal adjust- ments, enhancing flexibility and performance across tasks like node classification, link prediction, and graph classification. By redefining graph lottery tickets, we identified key subnetworks that enable efficient training and inference, demon- strating the scalability and effectiveness of QMPNNs on real-world datasets. Future work could explore dynamic graphs, hybrid modalities, and improving interpretability to further expand QMPNN capabilities.

## References













