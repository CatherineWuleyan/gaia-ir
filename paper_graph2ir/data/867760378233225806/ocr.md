# Permutation Equivariant Neural Functionals

Allan Zhou¹ Kaien Yang¹ Kaylee Burns¹ Adriano Cardace² Yiding Jiang³
Samuel Sokota³ J. Zico Kolter³ Chelsea Finn¹
¹Stanford University ²University of Bologna ³Carnegie Mellon University
ayz@cs.stanford.edu

## Abstract
This work studies the design of neural networks that can process the weights or gradients of other neural networks, which we refer to as *neural functional networks* (NFNs). Despite a wide range of potential applications, including learned optimization, processing implicit neural representations, network editing, and policy evaluation, there are few unifying principles for designing effective architectures that process the weights of other networks. We approach the design of neural functionals through the lens of symmetry, in particular by focusing on the permutation symmetries that arise in the weights of deep feedforward networks because hidden layer neurons have no inherent order. We introduce a framework for building *permutation equivariant* neural functionals, whose architectures encode these symmetries as an inductive bias. The key building blocks of this framework are *NF-Layers* (neural functional layers) that we constrain to be permutation equivariant through an appropriate parameter sharing scheme. In our experiments, we find that permutation equivariant neural functionals are effective on a diverse set of tasks that require processing the weights of MLPs and CNNs, such as predicting classifier generalization, producing "winning ticket" sparsity masks for initializations, and classifying or editing implicit neural representations (INRs). In addition, we provide code for our models and experiments¹.

## 1 Introduction
As deep neural networks have become increasingly prevalent across various domains, there has been a growing interest in techniques for processing their weights and gradients as data. Example applications include learnable optimizers for neural network training [3, 53, 2, 42], extracting information from implicit neural representations of data [58, 43, 55], corrective editing of network weights [54, 11, 44], policy evaluation [23], and Bayesian inference given networks as evidence [57]. We refer to functions of a neural network's weight space (such as weights, gradients, or sparsity masks) as *neural functionals*; when these functions are themselves neural networks, we call them *neural functional networks* (NFNs).

In this work, we design neural functional networks by incorporating relevant symmetries directly into the architecture, following a general line of work in "geometric deep learning" [8, 51, 33, 5]. For neural functionals, the symmetries of interest are transformations of a network's weights that preserve the network's behavior. In particular, we focus on *neuron permutation symmetries*, which are those that arise from the fact that the neurons of hidden layers have no inherent order.

Neuron permutation symmetries are simplest in feedforward networks, such as multilayer perceptrons (MLPs) and basic convolutional neural networks (CNNs). These symmetries are induced by the fact that the neurons in each hidden layer of a feedforward network can be arbitrarily permuted without changing its behavior [26]. In MLPs, permuting the neurons in hidden layer $i$ corresponds to

¹https://github.com/AllanYangZhou/nfn

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

![](./images/867760378233225806_1.jpg)

Figure 1: The internal operation of our permutation equivariant neural functionals (NFNs). The NFN processes the input weights through a series of equivariant NF-Layers, with each one producing weight-space features with varying numbers of channels. In this example, a neuron permutation symmetry simultaneously permutes the rows of $W^{(2)}$ and the columns of $W^{(3)}$. This permutation propagates through the NFN in an equivariant manner.

permuting the rows of the weight matrix $W^{(i)}$, and the columns of the next weight matrix $W^{(i+1)}$ as shown on the left-hand side of Figure 1. Note that the same permutation must be applied to the rows $W^{(i)}$ and columns of $W^{(i+1)}$, since applying different permutations generally changes network behavior and hence does not constitute a neuron permutation symmetry.

We introduce a new framework for constructing neural functional networks that are invariant or equivariant to neuron permutation symmetries. Our framework extends a long line of work on permutation equivariant architectures [49, 65, 24, 60, 39] that design equivariant layers for a particular permutation symmetry of interest. Specifically, we introduce neural functional layers (NF-Layers) that operate on weight-space features (see Figure 1) while being equivariant to neuron permutation symmetries. Composing these NF-Layers with pointwise non-linearities produces equivariant neural functionals.

We propose different NF-Layers depending on the assumed symmetries of the input weight space: either only the hidden neurons of the feedforward network can be permuted (hidden neuron permutation, HNP), or all neurons, including inputs and outputs, can be permuted (neuron permutation, NP). Although the HNP assumption is typically more appropriate, the corresponding NF-Layers can be parameter inefficient and computationally infeasible in some settings. In contrast, NF-Layers derived under NP assumptions often lead to much more efficient architectures, and, when combined with a positional encoding scheme we design, can even be effective on tasks that require breaking input and output symmetry. For situations where invariance is required, we also define invariant NF-Layers that can be applied on top of equivariant weight-space features.

Finally, we investigate the applications of permutation equivariant neural functionals on tasks involving both feedforward MLPs and CNNs. Our first two tasks require (1) predicting the test accuracy of CNN image classifiers and (2) classifying implicit neural representations (INRs) of images and 3D shapes. We then evaluate NFNs on their ability to (3) predict good sparsity masks for initializations (also called winning tickets [19]), and on (4) a weight-space "style-editing" task where the goal is to modify the content an INR encodes by directly editing its weights. In multiple experiments across these diverse settings, we find that permutation equivariant neural functionals consistently outperform non-equivariant methods and are effective for solving weight space tasks.

Relation to DWSNets. The recent work of Navon et al. [45] recognized the potential of leveraging weight space symmetries to build equivariant architectures on deep weight spaces; they characterize a weight-space layer which is mathematically equivalent to our NF-Layer in the HNP setting. Their work additionally studies interesting universality properties of the resulting equivariant architectures, and demonstrates strong empirical results for a suite of tasks that require processing the weights of MLPs. Our framework additionally introduces the NP setting, where we make stronger symmetry assumptions to develop equivariant layers with improved parameter efficiency and practical scalability. We also extend our NFN variants to process convolutional neural networks (CNNs) as input, leading to applications such as predicting the generalization of CNN classifiers (Section 3.1).

Table 1: Permutation symmetries of $L$-layer feedforward networks with $n_0, \dots, n_L$ neurons at each layer. All feedforward networks are invariant under hidden neuron permutations (HNP), while NP assumes that input and output neurons can also be permuted. We show the corresponding equivariant NF-Layers which process weight-space features from $\mathcal{U}$, with $c_i$ input channels and $c_o$ output channels.

| Group | Abbry | Permutable layers | Equivariant NF-Layer |
| :---: | :---: | :---: | :---: |
| | | | Signature | Parameter count |
| $\mathcal{S} = \prod_{i=0}^L S_{n_i}$ | NP | All layers | $H: \mathcal{U}^{c_i} \to \mathcal{U}^{c_o}$ | $O(c_i c_o L^2)$ |
| $\tilde{\mathcal{S}} = \prod_{i=1}^{L-1} S_{n_i}$ | HNP | Hidden layers | $\tilde{H}: \mathcal{U}^{c_i} \to \mathcal{U}^{c_o}$ | $O\left(c_i c_o(L + n_0 + n_L)^2\right)$ |
| --- | --- | None | $T: \mathcal{U}^{c_i} \to \mathcal{U}^{c_o}$ | $c_i c_o \dim(\mathcal{U})^2$ |

## 2 Equivariant neural functionals

We begin by setting up basic concepts related to (hidden) neuron permutation symmetries, before defining the equivariant NF-Layers in Sec. 2.2 and invariant NF-Layers in Sec. 2.3.

### 2.1 Preliminaries

Consider an $L$-layer feedforward network having $n_i$ neurons at layer $i$, with $n_0$ and $n_L$ being the input and output dimensions, respectively. The network is parameterized by weights $W = \left\{ W^{(i)} \in \mathbb{R}^{n_i \times n_{i-1}} \mid i \in [\![1..L]\!] \right\}$ and biases $v = \left\{ v^{(i)} \in \mathbb{R}^{n_i} \mid i \in [\![1..L]\!] \right\}$. We denote the combined collection $U := (W, v)$ belonging to weight space, $\mathcal{U} := \mathcal{W} \times \mathcal{V}$.

Since the neurons in a hidden layer $i \in \{1, \cdots, L-1\}$ have no inherent ordering, the network is invariant to the symmetric group $S_{n_i}$ of permutations of the neurons in layer $i$. This reasoning applies to every hidden layer, so the network is invariant to $\tilde{\mathcal{S}} := S_{n_1} \times \cdots \times S_{n_{L-1}}$, which we refer to as the **hidden neuron permutation (HNP)** group. Under the stronger assumption that the input and output neurons are also unordered, the network is invariant to $\mathcal{S} := S_0 \times \cdots \times S_{n_L}$, which we refer to as the **neuron permutation (NP)** group. We focus on the NP setting throughout the main text, and treat the HNP case in Appendix B. See Table 1 for a concise summary of the relevant notation for each symmetry group we consider.

Consider an MLP and a permutation $\sigma = (\sigma_0, \cdots, \sigma_L) \in \mathcal{S}$. The action of the neuron permutation group is to permute the rows of each weight matrix $W^{(i)}$ by $\sigma_i$, and the columns by $\sigma_{i-1}$. Each bias vector $v^{(i)}$ is also permuted by $\sigma_i$. So the action is $\sigma U := (\sigma W, \sigma v)$, where:
$$
[\sigma W]_{jk}^i = W_{\sigma_i^{-1}(j), \sigma_{i-1}^{-1}(k)}^{(i)}, \quad [\sigma v]_j^i = v_{\sigma_i^{-1}(j)}^{(i)}. \tag{1}
$$

Until now we have used $U = (W, v)$ to denote actual weights and biases, but the inputs to a neural functional layer could be any weight-space **feature** such as a gradient, sparsity mask, or the output of a previous NF-Layer (Figure 1). Moreover, we may consider inputs with $c \geq 1$ feature channels, belonging to $\mathcal{U}^c = \bigoplus_{i=1}^c \mathcal{U}$, the direct sum of $c$ copies of $\mathcal{U}$. Concretely, each $U \in \mathcal{U}^c$ consists of weights $W = \left\{ W^{(i)} \in \mathbb{R}^{n_i \times n_{i-1} \times c} \mid i \in [\![1..L]\!] \right\}$ and biases $v = \left\{ v^{(i)} \in \mathbb{R}^{n_i \times c} \mid i \in [\![1..L]\!] \right\}$, with the channels in the final dimension. The action defined in Eq. 1 extends to the multiple channel case if we define $W_{jk}^{(i)} := W_{j,k,:}^{(i)} \in \mathbb{R}^c$ and $v_j^{(i)} := v_{j,:}^{(i)} \in \mathbb{R}^c$.

The focus of this work is on making neural functionals that are equivariant (or invariant) to neuron permutation symmetries. Letting $c_i$ and $c_o$ be the number of input and output channels, we refer to a function $f: \mathcal{U}^{c_i} \to \mathcal{U}^{c_o}$ as **$\mathcal{S}$-equivariant** if $\sigma f(U) = f(\sigma U)$ for all $\sigma \in \mathcal{S}$ and $U \in \mathcal{U}^{c_i}$, where the action of $\mathcal{S}$ on the input and output spaces is defined by Eq. 1. Similarly, a function $f: \mathcal{U}^c \to \mathbb{R}$ is **$\mathcal{S}$-invariant** if $f(\sigma U) = f(U)$ for all $\sigma$ and $U$.

If $f, g$ are equivariant, then their composition $f \circ g$ is also equivariant; if $g$ is equivariant and $f$ is invariant, then $f \circ g$ is invariant. Since pointwise nonlinearities are already permutation equivariant, our remaining task is to design a **linear** NF-Layer that is $\mathcal{S}$-equivariant. We can then construct equivariant neural functionals by stacking these NF-Layers with pointwise nonlinearities.

![](./images/867760378233225806_2.jpg)

$$
H(W)_{j k}^{(i)}=\left(\sum_{s} a^{i, s} W_{*, *}^{(s)}\right)+c^{i, i} W_{j, *}^{(i)}+c^{i, i+1} W_{*, j}^{(i+1)}+b^{i, i} W_{*, k}^{(i)}+b^{i, i-1} W_{k, *}^{(i-1)}+d^{i} W_{j k}^{(i)}
$$

Figure 2: A permutation equivariant NF-Layer takes in weight-space features as input (bottom) and outputs transformed features (top), while respecting the neuron permutation symmetries of feedforward networks. This illustrates the computation of a single output element $H(W)_{j k}^{i}$, defined in Eq. 2. Each output is a weighted combination of rows or column sums of the input weights, which preserves permutation symmetry. The first term contributes a weighted combination of row-and-column sums from every input weight, though this is omitted for visual clarity.

## 2.2 Equivariant NF-Layers

We now construct a linear $\mathcal{S}$-equivariant layer that serves as a key building block for neural functional networks. In the single channel case, we begin with generic linear layers $T(\cdot ; \theta): \operatorname{vec}(U) \mapsto \theta \operatorname{vec}(U)$, where $\operatorname{vec}(U) \in \mathbb{R}^{\operatorname{dim}(U)}$ is $U$ flattened as a vector and $\theta \in \mathbb{R}^{\operatorname{dim}(\mathcal{U}) \times \operatorname{dim}(\mathcal{U})}$ is a matrix of parameters. We show in Appendix B.3 that any $\mathcal{S}$-equivariant $T(\cdot ; \theta)$ must satisfy a system of constraints on $\theta$ known as equivariant parameter sharing. We derive this parameter sharing by partitioning the entries of $\theta$ by the orbits of their indices under the action of $\mathcal{S}$, with parameters shared in each orbit [51]. Table 7 of the appendix describes the parameter sharing in detail.

Equivariant parameter sharing reduces the matrix-vector product $\theta \operatorname{vec}(U)$ to the NF-Layer we now present. For simplicity we ignore $\mathcal{V}$ and assume here that $\mathcal{U}=\mathcal{W}$ and defer the full form to Eq. 3 in the appendix. Then $H: \mathcal{W}^{c_{i}} \rightarrow \mathcal{W}^{c_{o}}$ maps input $\left(W^{(1)}, \cdots, W^{(L)}\right)$ to $\left(H(W)^{(1)}, \cdots, H(W)^{(L)}\right)$. Recall that the inputs are not necessarily weights, but could be arbitrary weight-space features including the output of a previous NF-Layer. For $W^{(i)} \in \mathbb{R}^{n_{i} \times n_{i-1} \times c_{i}}$, the corresponding output is $H(W)^{(i)} \in \mathbb{R}^{n_{i} \times n_{i-1} \times c_{o}}$ with entries computed:

$$
H(W)_{j k}^{(i)}=\left(\sum_{s} a^{i, s} W_{\star, \star}^{(s)}\right)+b^{i, i} W_{\star, k}^{(i)}+b^{i, i-1} W_{k, \star}^{(i-1)}+c^{i, i} W_{j, \star}^{(i)}+c^{i, i+1} W_{\star, j}^{(i+1)}+d^{i} W_{j k}^{(i)}. \quad (2)
$$

Note that the terms involving $W^{(i-1)}$ or $W^{(i+1)}$ should be omitted for $i=0$ and $i=L$, respectively, and $\star$ denotes summation or averaging over either the rows or columns. Recall that in the multi-channel case, each $W_{j k}^{(i)}$ is a vector in $\mathbb{R}^{c_{i}}$ so each parameter is a $c_{o} \times c_{i}$ matrix. We also provide a concrete pseudocode description of $H$ in Appendix A. Figure 2 visually illustrates the NF-Layer in the single-channel case, showing how the row or column sums from each input contribute to each output. To gain intuition for the operation of $H$, it is straightforward to check $\mathcal{S}$-equivariance:

Proposition 1. The NF-Layer $H: \mathcal{U}^{c_{i}} \rightarrow \mathcal{U}^{c_{o}}$ (Eq. 2 and Eq. 3) is $\mathcal{S}$-equivariant, where the group's action on input and output spaces is defined by Eq. 1. Moreover, any linear $\mathcal{S}$-equivariant map $T: \mathcal{U}^{c_{i}} \rightarrow \mathcal{U}^{c_{o}}$ is equivalent to $H$ for some choice of parameters a, b, c, d.

Proof (sketch). We can verify that $H$ satisfies the equivariance condition $[\sigma H(W)]_{j k}^{(i)}=H(\sigma W)_{j k}^{(i)}$ for any $i, j, k$ by expanding each side of the equation using the definitions of the layer and action

(Eq. 1). Moreover, Appendix B.3 shows that any $\mathcal{S}$-equivariant linear map $T(\cdot, \theta)$ must have the same equivariant parameter sharing as $H$, meaning that it must be equivalent to $H$ for some choice of parameter $a, b, c, d$. See Appendix B for the full proof. $\square$

Informally, the above proposition tells us that $H$ can express any linear $\mathcal{S}$-equivariant function of a weight space. Since $\tilde{\mathcal{S}}$ is a subgroup of $\mathcal{S}$, $H$ is also $\tilde{\mathcal{S}}$-equivariant. However, it does not express every possible linear $\tilde{\mathcal{S}}$-equivariant function. We derive the full $\tilde{\mathcal{S}}$-equivariant NF-Layer $\tilde{H}: \mathcal{U} \rightarrow \mathcal{U}$ in Appendix C.

Table 1 summarizes the number of parameters (after parameter sharing) under different symmetry assumptions. While in general a linear layer $T(\cdot ; \theta): \mathcal{U}^{c_{i}} \rightarrow \mathcal{U}^{c_{o}}$ has $c_{i} c_{o} \dim(\mathcal{U})^{2}$ parameters, the equivariant NF-Layers have significantly fewer free parameters due to parameter sharing. The $\mathcal{S}$-equivariant layer $H$ has $O\left(c_{i} c_{o} L^{2}\right)$, while the $\tilde{\mathcal{S}}$-equivariant layer $\tilde{H}$ has $O\left(c_{i} c_{o}\left(L+n_{0}+n_{L}\right)^{2}\right)$ parameters. The latter's quadratic dependence on input and output dimensions can be prohibitive in some settings, such as in classification where the number of outputs can be tens of thousands.

Extension to convolutional weight spaces. In convolution layers, since neurons correspond to spatial channels, we let $n_{i}$ denote the number of channels at the $i^{\text {th }}$ layer. Each bias $v^{(i)} \in \mathbb{R}^{n_{i}}$ has the same dimensions as in the fully connected case, so only the convolution filter needs to be treated differently since it has additional spatial dimension(s) that cannot be permuted. For example, consider a 1D CNN with filters $W=\left\{W^{(i)} \in \mathbb{R}^{n_{i} \times n_{i-1} \times w} \mid i \in \llbracket 1 . . L \rrbracket\right\}$, where $n_{i} \times n_{i-1}$ are the output and input channel dimensions and $w$ is the filter width. We let $W_{j k}^{(i)}:=W_{j, k,:}^{(i)} \in \mathbb{R}^{w}$ denote the $k^{\text {th }}$ filter in the $j^{\text {th }}$ output channel, then define the same way as in Eq. 1.

We immediately observe the similarities to multi-channel features: both add dimensions that are not permuted by the group action. In fact, suppose we have $c$-channel features $U \in \mathcal{U}^{c}$ where $\mathcal{U}$ is the weight-space of a 1D CNN. Then we combine the filter and channel dimensions of the weights, with $W^{(i)} \in \mathbb{R}^{n_{i} \times n_{i-1} \times(c w)}$. This allows us to use the multi-channel NF-Layer $H: \mathcal{U}^{w c_{i}} \rightarrow \mathcal{U}^{w c_{o}}$. Any further channel dimensions, such as those for 2D convolutions, can also be folded into the channel dimension.

It is common for CNNs in image classification to follow convolutional layers with pooling and fully connected (FC) layers, which opens the question of defining the $\mathcal{S}$-action when layer $\ell$ is FC and layer $\ell-1$ is convolutional. If global spatial pooling removes all spatial dimensions from the output of $\ell-1$ (as in e.g., ResNets [25] and the Small CNN Zoo [61]), then we can verify that the existing action definitions work without modification. We leave more complicated situations (e.g., when nontrivial spatial dimensions are flattened as input to FC layers) to future work.

IO-encoding. The $\mathcal{S}$-equivariant layer $H$ is more parameter efficient than $\tilde{H}$ (Table 1), but its NP assumptions are typically too strong. To resolve this problem, we can add either learned or fixed (sinusoidal) position embeddings to the columns of $W^{(1)}$ and the rows of $W^{(L)}$ and $v^{(L)}$; this breaks the symmetry at input and output neurons even when using $\mathcal{S}$-equivariant layers. In our experiments, we find that IO-encoding makes $H$ competitive or superior to $\tilde{H}$, while using a fraction of the parameters.

### 2.3 Invariant NF-Layers
Invariant neural functionals can be designed by composing multiple equivariant NF-Layers with an invariant NF-Layer, which can then be followed by an MLP. We define an $\mathcal{S}$-invariant layer $P: \mathcal{U} \rightarrow \mathbb{R}^{2 L}$ by simply summing or averaging the weight matrices and bias vectors across any axis that has permutation symmetry, i.e., $P(U)=\left(W_{\star, \star}^{(1)}, \cdots, W_{\star, \star}^{(L)}, v_{\star}^{(1)}, \cdots, v_{\star}^{(L)}\right)$. We define the analogous $\tilde{\mathcal{S}}$-invariant layer $\tilde{P}$ in Eq. 17 of the appendix.

## 3 Experiments
Our experiments evaluate permutation equivariant neural functionals on a variety of tasks that require either invariance (predicting CNN generalization and extracting information from INRs) or equivariance (predicting "winning ticket" sparsity masks and weight-space editing of INR content).

Table 2: Test $\tau$ of generalization prediction methods on the Small CNN Zoo [61], which contains the weights and test accuracies of many small CNNs trained on different datasets, such as CIFAR-10-GS or SVHN-GS. $\mathrm{NFN}_{\mathrm{HNP}}$ outperforms other methods on both datasets. Uncertainties indicate max and min over two runs.

<table>
<thead>
<tr>
<th></th>
<th>$\mathrm{NFN}_{\mathrm{HNP}}$</th>
<th>$\mathrm{NFN}_{\mathrm{NP}}$</th>
<th>$\mathrm{STATNN}$</th>
</tr>
</thead>
<tbody>
<tr>
<td>CIFAR-10-GS</td>
<td>$0.934 \pm 0.001$</td>
<td>$0.922 \pm 0.001$</td>
<td>$0.915 \pm 0.002$</td>
</tr>
<tr>
<td>SVHN-GS</td>
<td>$0.931 \pm 0.005$</td>
<td>$0.856 \pm 0.001$</td>
<td>$0.843 \pm 0.000$</td>
</tr>
</tbody>
</table>

Throughout the experiments, we construct neural functional networks (NFNs) using the NF-Layers described in the previous section. Although the specific design varies depending on the task, we will broadly refer to our permutation equivariant NFNs as $\mathrm{NFN}_{\mathrm{NP}}$ and $\mathrm{NFN}_{\mathrm{HNP}}$, depending on which NF-Layer variant they use (see Table 1). We also evaluate a "pointwise" ablation of our equivariant NF-Layer that ignores interactions between weights by only using the last term of Eq. 2, computing $H(W)_{j k}^{i}:=d^{i} W_{j k}^{(i)}$. We refer to NFNs that use this pointwise NF-Layer as $\mathrm{NFN}_{\mathrm{PT}}$.

Where feasible we also compare against neural functionals with standard FC layers, instead of equivariant NF-Layers. We optionally augment the training data with permutations (using Eq. 1) to encourage permutation symmetry. We refer to these methods as $\mathrm{MLP}$ and $\mathrm{MLP}_{\text {Aug }}$.

### 3.1 Predicting CNN generalization from weights

Why deep neural networks generalize despite being heavily overparameterized is a longstanding research problem in deep learning. One recent line of work has investigated the possibility of directly predicting the test accuracy of the models from the weights [61, 16]. The goal is to study generalization in a data-driven fashion and ultimately identify useful patterns from the weights.

Prior methods develop various strategies for extracting potentially useful features from the weights before using them to predict the test accuracy [28, 64, 61, 29, 40]. However, using hand-crafted features could fail to capture intricate correlations between the weights and test accuracy. Instead, we explore using neural functionals to predict test accuracy from the raw weights of feedforward convolutional neural networks (CNN) from the Small CNN Zoo dataset [61], which contains thousands of CNN weights trained on several datasets with varied hyperparameters. We compare the predictive power of $\mathrm{NFN}_{\mathrm{HNP}}$ and $\mathrm{NFN}_{\mathrm{NP}}$ against a method of Unterthiner et al. [61] that trains predictors on statistical features extracted from each weight and bias, and refer to it as $\mathrm{STATNN}$. To measure the predictive performance of each method, we use Kendall's $\tau$ [30], a popular rank correlation metric with values in $[-1,1]$.

In Table 2, we show the results on two challenging subsets of Small CNN Zoo corresponding to CNNs trained on CIFAR-10-GS and SVHN-GS (GS stands for grayscaled). We see that $\mathrm{NFN}_{\mathrm{HNP}}$ consistently performs the best on both datasets by a significant margin, showing that having access to the full weights can increase predictive power over hand-designed features as in $\mathrm{STATNN}$. Because the input and output dimensionalities are small on these datasets, $\mathrm{NFN}_{\mathrm{HNP}}$ only uses moderately more $(\sim 1.4 \times)$ parameters than $\mathrm{NFN}_{\mathrm{NP}}$ with equivalent depth and channel dimensions, while having significantly better performance.

### 3.2 Classifying implicit neural representations of images and 3D shapes

Given the rise of implicit neural representations (INRs) that encode data such as images and 3D- scenes [58, 41, 7, 46, 55, 43, 14, 15], it is natural to wonder how to extract information about the original data directly from the weights.

In this task, our goal is to classify the contents of INRs given only the weights as input. We consider datasets of SIRENs [55] that encode images (MNIST [37], FashionMNIST [63], and CIFAR [34]) and 3D shapes (ShapeNet-10 and ScanNet-10 [50]). For image datasets each SIREN network represents the mapping from pixel coordinate to RGB (or grayscale) value for a single image, while for 3D shapes each network is a signed (or unsigned) distance function encoding a single shape. Each dataset of SIREN weights is split into training, validation, and testing sets.

Table 3: Classification train and test accuracies (%) for implicit neural representations of MNIST, FashionMNIST, and CIFAR-10. Our equivariant NFNs outperform the MLP baselines, even when the MLP has permutation augmentations to encourage invariance. Uncertainties indicate standard error over three runs.

|             | $\text{NFN}_{\text{HNP}}$ | $\text{NFN}_{\text{NP}}$ | MLP          | $\text{MLP}_{\text{Aug}}$ |
|-------------|---------------------------|---------------------------|--------------|---------------------------|
| CIFAR-10    | $44.1\pm0.471$            | $\mathbf{46.6\pm0.072}$   | $16.9\pm0.250$| $18.9\pm0.432$            |
| MNIST-10    | $92.5\pm0.071$            | $\mathbf{92.9\pm0.218}$   | $14.5\pm0.035$| $21.0\pm0.172$            |
| FashionMNIST| $72.7\pm1.53$             | $\mathbf{75.6\pm1.07}$    | $12.5\pm0.111$| $15.9\pm0.181$            |

Table 4: Classification test accuracies (%) for datasets of implicit neural representations (INRs) of either ShapeNet-10 [6] or ScanNet-10 [10] Our equivariant NFNs outperform the MLP baselines and recent non-equivariant methods such as inr2vec [12]. Uncertainties indicate standard error over three runs.

|             | $\text{NFN}_{\text{HNP}}$ | $\text{NFN}_{\text{NP}}$ | MLP          | $\text{MLP}_{\text{Aug}}$ | inr2vec[12]  |
|-------------|---------------------------|---------------------------|--------------|---------------------------|--------------|
| ShapeNet-10 | $86.9\pm0.860$            | $\mathbf{88.7\pm0.461}$   | $25.4\pm0.121$| $33.8\pm0.126$            | $39.1\pm0.385$|
| ScanNet-10  | $64.1\pm0.572$            | $\mathbf{65.9\pm1.10}$    | $32.9\pm0.351$| $45.5\pm0.126$            | $38.2\pm0.409$|

We construct and train invariant neural functionals to classify the INRs, and compare their performance against the MLP and $\text{MLP}_{\text{Aug}}$ baselines, which are three-layer MLPs with ReLU activations and 1,000 hidden units per layer. For the 3D-shape datasets we also report the performance of inr2vec [12], a recent non-equivariant method with results on classifying 3D shapes from INR weights. Note that inr2vec's original setting assumes that all INRs in a dataset are trained from the same shared initialization, whereas our problem setting makes no such assumption and allows INRs to be trained from random and independent initializations.

The results in Table 3 and Table 4 show that $\text{NFN}_{\text{HNP}}$ and $\text{NFN}_{\text{NP}}$ consistently achieve higher test accuracies than the baseline methods on both datasets. In addition to superior generalization, Tables 17-18 in the appendix show that NFNs are also usually better at fitting the training data (higher train accuracy). The MLPs struggle to even fit the training data, especially under permutations augmentations, even with the same number of parameters as the NFNs. Interestingly, $\text{NFN}_{\text{NP}}$ matches or exceeds $\text{NFN}_{\text{HNP}}$ performance on both CIFAR-10 and the 3D-shape datasets while using fewer parameters (e.g., 35% as many parameters on CIFAR-10).

### 3.3 Predicting "winning ticket" masks from initialization

The Lottery Ticket Hypothesis [19, 20, LTH] conjectures the existence of winning tickets, or sparse initializations that train to the same final performance as dense networks, and showed their existence in some settings through iterative magnitude pruning (IMP). IMP retroactively finds a winning ticket by pruning trained models by magnitude; however, finding the winning ticket from only the initialization without training remains challenging.

We demonstrate that permutation equivariant neural functionals are a promising approach for finding winning tickets at initialization by learning over datasets of initializations and their winning tickets. Let $U_0 \in \mathcal{U}$ be an initialization and let the sparsity mask $M \in \{0,1\}^{\text{dim}(\mathcal{U})}$ be a winning ticket for

Table 5: Test accuracy (%) of training with winning tickets (95% sparsity masks) produced either by running IMP or predicted by an NFN. We also show the performance of Random ticket (random mask of equivalent sparsity level), and Dense training (no sparsity). We show results for MLPs (trained on MNIST) and CNNs (trained on CIFAR-10). Uncertainties show standard error over initializations.

|             | Dense        | IMP          | Random       | $\text{NFN}_{\text{NP}}$ | $\text{NFN}_{\text{PT}}$ |
|-------------|--------------|--------------|--------------|---------------------------|---------------------------|
| CIFAR-10    | $63.1\pm0.06$| $44.0\pm0.06$| $21.1\pm0.26$| $\mathbf{41.4\pm0.08}$    | $\mathbf{42.6\pm0.07}$    |
| MNIST       | $97.8\pm0.0$ | $96.2\pm0.04$| $89.6\pm0.36$| $\mathbf{94.8\pm0.01}$    | $\mathbf{95.0\pm0.01}$    |

![](./images/867760378233225806_3.jpg)

Figure 3: In weight-space style editing, an NFN directly edits the weights of an INR to alter the content it encodes. In this example, the NFN edits the weights to dilate the encoded image.

<table>
<thead>
  <tr>
    <th>Method</th>
    <th>Contrast (CIFAR-10)</th>
    <th>Dilate (MNIST)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MLP</td>
    <td>0.031</td>
    <td>0.306</td>
  </tr>
  <tr>
    <td>MLP<sub>Aug</sub></td>
    <td>0.029</td>
    <td>0.307</td>
  </tr>
  <tr>
    <td>NFN<sub>PT</sub></td>
    <td>0.029</td>
    <td>0.197</td>
  </tr>
  <tr>
    <td>NFN<sub>HNP</sub></td>
    <td>0.021</td>
    <td>0.070</td>
  </tr>
  <tr>
    <td>NFN<sub>NP</sub></td>
    <td>0.020</td>
    <td>0.068</td>
  </tr>
</tbody>
</table>

Table 6: Test mean squared error (lower is better) between weight-space editing methods and ground-truth image-space transformations.

![](./images/867760378233225806_4.jpg)

Figure 4: Random qualitative samples of INR editing behavior on the Dilate (MNIST) and Contrast (CIFAR-10) editing tasks. The first row shows the image produced by the original INR, while the rows below show the result of editing the INR weights with an NFN. The difference between MLP neural functionals and equivariant neural functionals is especially pronounced on the more challenging Dilate tasks, which require modifying the geometry of the image. In the Contrast tasks, the MLP baseline produces dimmer images compared to the ground truth, which is especially evident in the second and third columns.

the initialization, with zeros indicating that the corresponding entries of $U_0$ should be pruned. The goal is to predict a winning ticket $\hat{M}$ given a held out initialization $U_0$, such that the MLP initialized with $U_0$ and sparsity pattern $\hat{M}$ will achieve a high test accuracy after training.

We construct a conditional variational autoencoder [31, 56, cVAE] that learns a generative model of the winning tickets conditioned on initialization and train on datasets of (initialization, ticket) pairs found by one step of IMP with a sparsity level of $P_m = 0.95$ for both MLPs trained on MNIST and CNNs trained on CIFAR-10. Table 5 compares the performance of tickets predicted by equivariant neural functionals against IMP tickets and random tickets. We generate random tickets by randomly sampling sparsity mask entries from $\text{Bernoulli}(1 - P_m)$. In this setting, NFN$_{\text{HNP}}$ is prohibitively parameter inefficient, but NFN$_{\text{NP}}$ is able to recover test accuracies that are close to that of IMP pruned networks in CIFAR-10 and MNIST, respectively. Somewhat surprisingly, NFN$_{\text{PT}}$ performs just as well as the other NFNs, indicating that one can approach IMP performance in these settings without considering interactions between weights or layers. Appendix E.1 further analyzes how NFN$_{\text{PT}}$ learns to prune.

### 3.4 Weight space style editing

Another potentially useful application of neural functionals is to edit (i.e., transform) the weights of a given INR to alter the content that it encodes. In particular, the goal of this task is to edit the weights of a trained SIREN to alter its encoded image (Figure 3). We evaluate two editing tasks: (1) making MNIST digits thicker via image dilation (**Dilate**), and (2) increasing image contrast on

CIFAR-10 (**Contrast**). Both of these tasks require neural functionals to process *the relationships between different pixels* to successfully solve the task.

To produce training data for this task, we use standard image processing libraries [27, OpenCV] to dilate or increase the contrast of the MNIST and CIFAR-10 images, respectively. The training objective is to minimize the mean squared error between the image generated by the NFN-edited INR and the image produced by image processing. We construct equivariant neural functionals to edit the INR weights, and compare them against MLP-based neural functionals with and without permutation augmentation.

Table 6 shows that permutation equivariant neural functionals (NFN$_{\text{HNP}}$ and NFN$_{\text{NP}}$) achieve significantly better test MSE when editing held out INRs compared to other methods, on both the Dilate (MNIST) and Contrast (CIFAR-10) tasks. In other words, they produce results that are closest to the "ground truth" image-space processing operations for each task. The pointwise ablation NFN$_{\text{PT}}$ performs significantly worse, indicating that accounting for interactions between weights and layers is important to accomplishing these tasks. Figure 4 shows random qualitative samples of editing by different methods below the original (pre-edit) INR. We observe that NFNs are more effective than $\text{MLP}_{\text{Aug}}$ at dilating MNIST digits and increasing the contrast in CIFAR-10 images.

## 4 Related work

The permutation symmetries of neurons have been a topic of interest in the context of loss landscapes and model merging [21, 4, 59, 17, 1]. Other works have analyzed the degree of learned permutation symmetry in networks that process weights [61] and studied ways of accounting for symmetries when measuring or encouraging diversity in the weight space [13]. However, these symmetries have not been a key consideration in architecture design for processing weight space objects [2, 38, 22, 35, 66, 13, 32]. Instead, existing approaches try to encourage permutation equivariance through data augmentation [48, 42]. In contrast, this work directly encodes the equivariance of the weight space into our architecture design, which can result in much higher data and computational efficiency, as evidenced by the success of convolutional neural networks [36].

Our work follows a long line of literature that incorporates structure and symmetry into neural network architectures [36, 8, 51, 33, 9, 18], including works that design equivariant layers for various permutation symmetries [49, 65, 24, 60, 39]. Our key contribution is applying the framework of Ravanbakhsh et al. [51] to the particular neuron permutation symmetries found in the weights of deep neural networks [26], leading to the characterization of our equivariant NF-Layers. As discussed in Section 1, Navon et al. [45] recently developed an equivariant weight-space layer that is equivalent to our NF-Layer in the HNP setting. Our work introduces the NP setting to improve parameter efficiency and scalability over the HNP setting, and extends beyond the fully connected case to handle convolutional weight space inputs.

## 5 Conclusion

This paper proposes a novel symmetry-inspired framework for the design of neural functional networks (NFNs), which process weight-space features such as weights, gradients, and sparsity masks. Our framework focuses on the permutation symmetries that arise in weight spaces due to the particular structure of neural networks. We introduce two equivariant NF-Layers as building blocks for NFNs, which differ in their underlying symmetry assumptions and parameter efficiency, then use them to construct a variety of permutation equivariant neural functionals. Experimental results across diverse settings demonstrate that permutation equivariant neural functionals outperform prior methods and are effective for solving weight-space tasks.

**Limitations and future work.** Although we believe this framework is a step toward the principled design of effective neural functionals, there remain multiple directions for improvement. One such direction would involve reducing the activation sizes produced by NF-Layers, which could be useful to scaling neural functionals to process the weights of very large networks. Another such direction would concern extending the NF-Layers to process weight inputs of more complex architectures such as ResNet [25] and Transformer [62] weights, which would enable larger-scale applications.

# References

[1] S. K. Ainsworth, J. Hayase, and S. Srinivasa. Git re-basin: Merging models modulo permutation symmetries. *arXiv preprint arXiv:2209.04836*, 2022.

[2] M. Andrychowicz, M. Denil, S. Gomez, M. W. Hoffman, D. Pfau, T. Schaul, B. Shillingford, and N. De Freitas. Learning to learn by gradient descent by gradient descent. *Advances in neural information processing systems*, 29, 2016.

[3] S. Bengio, Y. Bengio, J. Cloutier, and J. Gescei. On the optimization of a synaptic learning rule. In *Optimality in Biological and Artificial Networks?*, pages 281–303. Routledge, 2013.

[4] J. Brea, B. Simsek, B. Illing, and W. Gerstner. Weight-space symmetry in deep networks gives rise to permutation saddles, connected by equal-loss valleys across the loss landscape. *arXiv preprint arXiv:1907.02911*, 2019.

[5] M. M. Bronstein, J. Bruna, T. Cohen, and P. Veličković. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*, 2021.

[6] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi, and F. Yu. ShapeNet: An Information-Rich 3D Model Repository. Technical Report arXiv:1512.03012 [cs.GR], Stanford University — Princeton University — Toyota Technological Institute at Chicago, 2015.

[7] Z. Chen and H. Zhang. Learning implicit fields for generative shape modeling. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5939–5948, 2019.

[8] T. Cohen and M. Welling. Group equivariant convolutional networks. In *International conference on machine learning*, pages 2990–2999. PMLR, 2016.

[9] T. S. Cohen, M. Geiger, J. Köhler, and M. Welling. Spherical CNNs. *arXiv preprint arXiv:1801.10130*, 2018.

[10] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 5828–5839, 2017.

[11] N. De Cao, W. Aziz, and I. Titov. Editing factual knowledge in language models. *arXiv preprint arXiv:2104.08164*, 2021.

[12] L. De Luigi, A. Cardace, R. Spezialetti, P. Zama Ramirez, S. Salti, and L. Di Stefano. Deep learning on implicit neural representations of shapes. In *International Conference on Learning Representations (ICLR)*, 2023.

[13] L. Deutsch, E. Nijkamp, and Y. Yang. A generative model for sampling high-performance and diverse weights for neural networks. *arXiv preprint arXiv:1905.02898*, 2019.

[14] E. Dupont, Y. W. Teh, and A. Doucet. Generative models as distributions of functions. *arXiv preprint arXiv:2102.04776*, 2021.

[15] E. Dupont, H. Kim, S. Eslami, D. Rezende, and D. Rosenbaum. From data to functa: Your data point is a function and you should treat it like one. *arXiv preprint arXiv:2201.12204*, 2022.

[16] G. Eilertsen, D. Jönsson, T. Ropinski, J. Unger, and A. Ynnerman. Classifying the classifier: dissecting the weight space of neural networks. *arXiv preprint arXiv:2002.05688*, 2020.

[17] R. Entezari, H. Sedghi, O. Saukh, and B. Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. *arXiv preprint arXiv:2110.06296*, 2021.

[18] M. Finzi, M. Welling, and A. G. Wilson. A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups. In *International Conference on Machine Learning*, pages 3318–3328. PMLR, 2021.


[19] J. Frankle and M. Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.

[20] J. Frankle, G. K. Dziugaite, D. M. Roy, and M. Carbin. Stabilizing the lottery ticket hypothesis. arXiv preprint arXiv:1903.01611, 2019.

[21] T. Garipov, P. Izmailov, D. Podoprikhin, D. P. Vetrov, and A. G. Wilson. Loss surfaces, mode connectivity, and fast ensembling of DNNs. Advances in neural information processing systems, 31, 2018.

[22] D. Ha, A. Dai, and Q. V. Le. Hypernetworks. arXiv preprint arXiv:1609.09106, 2016.

[23] J. Harb, T. Schaul, D. Precup, and P. Bacon. Policy evaluation networks. CoRR, abs/2002.11833, 2020. URL https://arxiv.org/abs/2002.11833.

[24] J. Hartford, D. Graham, K. Leyton-Brown, and S. Ravanbakhsh. Deep models of interactions across sets. In International Conference on Machine Learning, pages 1909–1918. PMLR, 2018.

[25] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CoRR, abs/1512, 3385:2, 2015.

[26] R. Hecht-Nielsen. On the algebraic structure of feedforward network weight spaces. In Advanced Neural Computers, pages 129–135. Elsevier, 1990.

[27] Itseez. Open source computer vision library. https://github.com/itseez/opencv, 2015.

[28] Y. Jiang, D. Krishnan, H. Mobahi, and S. Bengio. Predicting the generalization gap in deep networks with margin distributions. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=HJ1QfnCqKX.

[29] Y. Jiang, P. Natekar, M. Sharma, S. K. Aithal, D. Kashyap, N. Subramanyam, C. Lassance, D. M. Roy, G. K. Dziugaite, S. Gunasekar, et al. Methods and analysis of the first competition in predicting generalization of deep learning. In NeurIPS 2020 Competition and Demonstration Track, pages 170–190. PMLR, 2021.

[30] M. G. Kendall. A new measure of rank correlation. Biometrika, 30(1/2):81–93, 1938.

[31] D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[32] B. Knyazev, M. Drozdal, G. W. Taylor, and A. Romero Soriano. Parameter prediction for unseen deep architectures. Advances in Neural Information Processing Systems, 34:29433–29448, 2021.

[33] R. Kondor and S. Trivedi. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In International Conference on Machine Learning, pages 2747–2755. PMLR, 2018.

[34] A. Krizhevsky, G. Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[35] D. Krueger, C.-W. Huang, R. Islam, R. Turner, A. Lacoste, and A. Courville. Bayesian hypernetworks. arXiv preprint arXiv:1710.04759, 2017.

[36] Y. LeCun, Y. Bengio, et al. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995.

[37] Y. LeCun, C. Cortes, and C. Burges. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

[38] K. Li and J. Malik. Learning to optimize. arXiv preprint arXiv:1606.01885, 2016.

[39] H. Maron, O. Litany, G. Chechik, and E. Fetaya. On learning sets of symmetric elements. In International conference on machine learning, pages 6734–6744. PMLR, 2020.

11

[40] C. H. Martin and M. W. Mahoney. Implicit self-regularization in deep neural networks: Evidence from random matrix theory and implications for learning. *The Journal of Machine Learning Research*, 22(1):7479–7551, 2021.

[41] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy networks: Learning 3d reconstruction in function space. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4460–4470, 2019.

[42] L. Metz, J. Harrison, C. D. Freeman, A. Merchant, L. Beyer, J. Bradbury, N. Agrawal, B. Poole, I. Mordatch, A. Roberts, et al. Velo: Training versatile learned optimizers by scaling up. *arXiv preprint arXiv:2211.09760*, 2022.

[43] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: representing scenes as neural radiance fields for view synthesis (2020). *arXiv preprint arXiv:2003.08934*, 2020.

[44] E. Mitchell, C. Lin, A. Bosselut, C. Finn, and C. D. Manning. Fast model editing at scale. *arXiv preprint arXiv:2110.11309*, 2021.

[45] A. Navon, A. Shamsian, I. Achituve, E. Fetaya, G. Chechik, and H. Maron. Equivariant architectures for learning in deep weight spaces. *arXiv preprint arXiv:2301.12780*, 2023.

[46] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 165–174, 2019.

[47] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.

[48] W. Peebles, I. Radosavovic, T. Brooks, A. A. Efros, and J. Malik. Learning to learn with generative models of neural network checkpoints. *arXiv preprint arXiv:2209.12892*, 2022.

[49] C. Qi, H. Su, K. Mo, and L. Guibas. Pointnet: deep learning on point sets for 3d classification and segmentation. cvpr (2017). *arXiv preprint arXiv:1612.00593*, 2016.

[50] C. Qin, H. You, L. Wang, C.-C. J. Kuo, and Y. Fu. Pointdan: A multi-scale 3d domain adaption network for point cloud representation. *Advances in Neural Information Processing Systems*, 32, 2019.

[51] S. Ravanbakhsh, J. Schneider, and B. Poczos. Equivariance through parameter-sharing. In *International conference on machine learning*, pages 2892–2901. PMLR, 2017.

[52] A. Rogozhnikov. Einops: Clear and reliable tensor manipulations with einstein-like notation. In *International Conference on Learning Representations*, 2022.

[53] T. P. Runarsson and M. T. Jonsson. Evolution and design of distributed learning rules. In *2000 IEEE Symposium on Combinations of Evolutionary Computation and Neural Networks. Proceedings of the First IEEE Symposium on Combinations of Evolutionary Computation and Neural Networks (Cat. No. 00*, pages 59–63. IEEE, 2000.

[54] A. Sinitsin, V. Plokhotnyuk, D. Pyrkin, S. Popov, and A. Babenko. Editable neural networks. *arXiv preprint arXiv:2004.00345*, 2020.

[55] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein. Implicit neural representations with periodic activation functions. *Advances in Neural Information Processing Systems*, 33: 7462–7473, 2020.

[56] K. Sohn, H. Lee, and X. Yan. Learning structured output representation using deep conditional generative models. *Advances in neural information processing systems*, 28, 2015.

[57] S. Sokota, H. Hu, D. J. Wu, J. Z. Kolter, J. N. Foerster, and N. Brown. A fine-tuning approach to belief state modeling. In *International Conference on Learning Representations*, 2022. URL https://openreview.net/forum?id=ckZY7DG a7FQ.

[58] K. O. Stanley. Compositional pattern producing networks: A novel abstraction of development. Genetic programming and evolvable machines, 8:131–162, 2007.

[59] N. Tatro, P.-Y. Chen, P. Das, I. Melnyk, P. Sattigeri, and R. Lai. Optimizing mode connectivity via neuron alignment. Advances in Neural Information Processing Systems, 33:15300–15311, 2020.

[60] E. H. Thiede, T. S. Hy, and R. Kondor. The general theory of permutation equivarant neural networks and higher order graph variational encoders. arXiv preprint arXiv:2004.03990, 2020.

[61] T. Unterthiner, D. Keysers, S. Gelly, O. Bousquet, and I. Tolstikhin. Predicting neural network accuracy from weights. arXiv preprint arXiv:2002.11448, 2020.

[62] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[63] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.

[64] S. Yak, J. Gonzalvo, and H. Mazzawi. Towards task and architecture-independent generalization gap predictors. arXiv preprint arXiv:1906.01550, 2019.

[65] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola. Deep sets. doi: 10.48550. arXiv preprint ARXIV.1703.06114, 2017.

[66] C. Zhang, M. Ren, and R. Urtasun. Graph hypernetworks for neural architecture search. arXiv preprint arXiv:1810.05749, 2018.

Appendix

A Equivariant NF-Layer pseudocode 14

B $S$-equivariant NF-Layer 15
B.1 Full definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
B.2 General NF-Layers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.3 Equivariance and parameter sharing . . . . . . . . . . . . . . . . . . . . . . . 17
B.4 $S$-equivariant parameter sharing . . . . . . . . . . . . . . . . . . . . . . . . 17
B.5 Equivalence to equivariant NF-Layer definition . . . . . . . . . . . . . . . . . 18

C NF-Layers for the HNP setting 18
C.1 Equivariant NF-Layer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
C.2 Invariant NF-Layer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

D Additional experimental details 20
D.1 Predicting generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
D.2 Predicting "winning ticket" masks from initialization . . . . . . . . . . . . . . 21
D.3 Classifying INRs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
D.4 Weight space style editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

E Additional experiments and analysis 23
E.1 Interpreting learned lottery ticket masks . . . . . . . . . . . . . . . . . . . . . 23
E.2 Predicting MLP generalization from weights . . . . . . . . . . . . . . . . . . . 23

## A Equivariant NF-Layer pseudocode

Here we present a multi-channel implementation of the $S$-equivariant NF-Layer presented in Eq. 2 (which ignores biases), using PyTorch [47] and Einops-like [52] pseudocode. That is, it implements a linear layer $H: \mathcal{W}^{c_i} \to \mathcal{W}^{c_o}$, where $c_i$ and $c_o$ are the number of input and output channels.

Note that our actual implementation differs from this pseudocode in a few ways: (1) it supports the full weight space $\mathcal{U}$ which includes biases, (2) it supports convolution weights as well as fully connected weights, and (3) it initializes parameters based on the fan-in of the NF-Layer, instead of from $\mathcal{N}(0,1)$.

```
1 class NPLayer(nn.Module):
2   def __init__(self, L, co, ci):
3       super().__init__()
4       # initialize weights. co=output channels, ci=input channels.
5       self.A = nn.Parameter(torch.randn(L, L, co, ci))
6       self.B = nn.Parameter(torch.randn(L, co, ci))
7       self.B_prev = nn.Parameter(torch.randn(L, co, ci))
8       self.C = nn.Parameter(torch.randn(L, co, ci))
9       self.C_next = nn.Parameter(torch.randn(L, co, ci))
10      self.D = nn.Parameter(torch.randn(L, co, ci))
11
12  def forward(self, W):
13      # Input W is a list of L weight-space tensors with shapes:
```
14

Table 7: $\mathcal{S}$-equivariant parameter sharing for linear maps $\operatorname{vec}(U) \mapsto \theta \operatorname{vec}(U)$. Parameter sharing is a system of constraints on the entries of $(\vartheta, \phi, \varphi, \psi)=\theta$. Each table is organized by the layer indices $(i, s)$. For example, the first table says that for any $(i, s)$ where $s=i-1$, we constrain $\vartheta_{s p q}^{i j k}=\vartheta_{s, p^{\prime}, q^{\prime}}^{i, j^{\prime}, k^{\prime}}=a_{\vartheta}^{i, i-1}$ for any $j, k, p, q$ and $j^{\prime}, k^{\prime}, p^{\prime}, q^{\prime}$ where $k \neq p$ and $k^{\prime} \neq p^{\prime}$. After parameter sharing, we observe that there are only a constant number of free parameters for each $(i, s)$ pair, adding up to $O\left(L^{2}\right)$ parameters total.

$$
\begin{array}{c|llll} 
& s=i-1 & s=i & s=i+1 & \text { other } s \\
\hline \vartheta_{s p q}^{i j k} & \begin{cases}a_{\vartheta}^{i, i-1} & k \neq p \\
b_{\vartheta}^{i, i-1} & k=p\end{cases} & \begin{cases}a_{\vartheta}^{i, i} & j \neq p, k \neq q \\
b_{\vartheta}^{i, i} & j=p, k \neq q \\
c_{\vartheta}^{i, i} & j \neq p, k=q \\
d_{\vartheta}^{i} & j=p, k=q\end{cases} & \begin{cases}a_{\vartheta}^{i, i+1} & j \neq q \\
c_{\vartheta}^{i, i+1} & j=q\end{cases} & a_{\vartheta}^{i, s}
\end{array}
$$

$$
\begin{array}{c|lll} 
& s=i-1 & s=i & \text { other } s \\
\hline \phi_{s p}^{i j k} & \begin{cases}a_{\phi}^{i, i-1} & k \neq p \\
b_{\phi}^{i, i-1} & k=p\end{cases} & \begin{cases}a_{\phi}^{i, i} & j \neq p \\
b_{\phi}^{i, i} & j=p\end{cases} & a_{\phi}^{i, s}
\end{array}
$$

$$
\begin{array}{c|lll} 
& s=i & s=i+1 & \text { other } s \\
\hline \varphi_{s p q}^{i j} & \begin{cases}a_{\varphi}^{i, i-1} & j \neq p \\
b_{\varphi}^{i, i-1} & j=p\end{cases} & \begin{cases}a_{\varphi}^{i, i} & j \neq q \\
b_{\varphi}^{i, i} & j=q\end{cases} & a_{\varphi}^{i, s}
\end{array}
$$

$$
\begin{array}{c|ll} 
& s=i & \text { other } s \\
\hline \psi_{s p}^{i j} & \begin{cases}a_{\psi}^{i, i} & j \neq p \\
b_{\psi}^{i} & j=p\end{cases} & a_{\psi}^{i, s}
\end{array}
$$

where $\star$ denotes summation or averaging over a dimension.

## B.2 General NF-Layers

To arrive at Eq. 3, we begin by considering linear NF-Layers $T(\cdot ; \theta): \mathcal{U} \rightarrow \mathcal{U}$ parameterized by $\theta \in \Theta$. If we flatten the input $U=(W, v)$ into a vector $\operatorname{vec}(U) \in \mathbb{R}^{\operatorname{dim}(\mathcal{U})}$, then the NF-Layer would be a matrix-vector product $T(\cdot, \theta): \operatorname{vec}(U) \mapsto \theta \operatorname{vec}(U)$ for square matrix $\theta \in \mathbb{R}^{\operatorname{dim}(\mathcal{U}) \times \operatorname{dim}(\mathcal{U})}$.

For our purposes, it is sometimes convenient to distinguish layer, row, and column indices of entries in $U$ without any flattening, so we split the parameters $\theta=(\vartheta, \phi, \varphi, \psi)$ and write $T: \mathcal{U} \rightarrow \mathcal{U}$ in the form:

$$
T(U ; \theta):=(Y(U), z(U)) \quad \in \mathcal{W} \times \mathcal{V}=\mathcal{U}
\tag{4}
$$

$$
Y(U)_{j k}^{(i)}:=\sum_{s=1}^{L} \sum_{p=1}^{n_{s}} \sum_{q=1}^{n_{s-1}} \vartheta_{s p q}^{i j k} W_{p q}^{(s)}+\sum_{s=1}^{L} \sum_{p=1}^{n_{s}} \phi_{s p}^{i j k} v_{p}^{(s)}
\tag{5}
$$

$$
z(U)_{j}^{(i)}:=\sum_{s=1}^{L} \sum_{p=1}^{n_{s}} \sum_{q=1}^{n_{s-1}} \varphi_{s p q}^{i j} W_{p q}^{(s)}+\sum_{s=1}^{L} \sum_{p=1}^{n_{s}} \psi_{s p}^{i j} v_{p}^{(s)}.
\tag{6}
$$

Since we can equivalently flatten this operation into the matrix-vector product $\operatorname{vec}(U) \mapsto \theta \operatorname{vec}(U)$, we introduce the notation $U_{\alpha}$ to identify individual entries of $U$. Here $\alpha$ is a tuple of length two or three, for indexing into either a weight or bias. We denote the space of valid index tuples of $\mathcal{W}$ and $\mathcal{V}$ by $\mathbb{W}$ and $\mathbb{V}$, respectively, and define $\mathbb{U}:=\mathbb{W} \cup \mathbb{V}$ as the combined index space of $\mathcal{U}$. For example, if $\alpha=(i, j, k) \in \mathbb{W}$, then $U_{\alpha}=W_{j k}^{(i)}$.

We can then define the index space $\mathbb{I}:=\mathbb{U} \times \mathbb{U}$ for parameters $\theta \in \Theta$. We use $\theta_{\beta}^{\alpha}$ to index an entry of $\theta$ with upper and lower indices $[\alpha, \beta] \in \mathbb{I}$. For example, if $\alpha=(i, j, k) \in \mathbb{W}$ and $\beta=(s, p, q) \in \mathbb{W}$, we have $\theta_{\beta}^{\alpha}=\vartheta_{s p q}^{i j k}$.

The indices $\alpha$ and $\beta$ correspond to rows and columns of the matrix $\theta$, respectively. Eq. 4 can be rewritten in the flattened form:

$$
T(U ; \theta)_{\alpha}=\sum_{\beta} \theta_{\beta}^{\alpha} U_{\beta}. \tag{7}
$$

Finally, we can re-express the action of $\mathcal{S}$ on $\mathcal{U}$ (Eq. 1) as an action on the index space $\mathbb{U}$:

$$
\sigma(i, j, k)=\left(i, \sigma_{i}(j), \sigma_{i-1}(k)\right) \quad(i, j, k) \in \mathbb{W} \tag{8}
$$

$$
\sigma(i, j)=\left(i, \sigma_{i}(j)\right) \quad(i, j) \in \mathbb{V}, \tag{9}
$$

for any $\sigma \in \mathcal{S}$. We extend this definition into an action of $\mathcal{S}$ on $\mathbb{I}$:

$$
\sigma[\alpha, \beta]:=[\sigma \alpha, \sigma \beta], \quad[\alpha, \beta] \in \mathbb{U} \times \mathbb{U}. \tag{10}
$$

### B.3 Equivariance and parameter sharing

We would like to find the constraints on $\theta$ that make the linear map $T(\cdot ; \theta): \operatorname{vec}(U) \mapsto \theta \operatorname{vec}(U)$ equivariant to $\mathcal{S}$.

We can represent the action of $\sigma \in \mathcal{S}$ on $\operatorname{vec}(U)$ by a matrix $P_{\sigma} \in\{0,1\}^{\operatorname{dim}(U) \times \operatorname{dim}(U)}$. Equivariance requires that $P_{\sigma} \theta \operatorname{vec}(U)=\theta P_{\sigma} \operatorname{vec}(U)$ for any $\sigma \in \mathcal{S}$. Since the input $U$ can be anything, we get the following constraint on $\theta$:

$$
P_{\sigma} \theta=\theta P_{\sigma}, \quad \forall \sigma \in \mathcal{S}. \tag{11}
$$

When written out using indices $\alpha, \beta$, the constraint requires that for any $\sigma \in \mathcal{S}$:

$$
\left[P_{\sigma} \theta\right]_{\beta}^{\alpha}=\theta_{\beta}^{\sigma^{-1}(\alpha)}=\theta_{\sigma(\beta)}^{\alpha}=\left[\theta P_{\sigma}\right]_{\beta}^{\alpha}. \tag{12}
$$

By relabeling $\alpha \leftarrow \sigma^{-1}(\alpha)$, we can rewrite this condition $\theta_{\beta}^{\alpha}=\theta_{\sigma(\beta)}^{\sigma(\alpha)}$. Hence for any linear $\mathcal{S}$-equivariant map $T(\cdot, \theta): \mathcal{U} \rightarrow \mathcal{U}, \theta$ must share parameters within orbits under the action of $\mathcal{S}$ on its indices $\alpha, \beta$ (Eq. 10). In fact, this strategy was first proposed as a way of constructing equivariant layers by Ravanbakhsh et al. [51, Prop 3.1].

### B.4 $\mathcal{S}$-equivariant parameter sharing

We now derive the required parameter sharing conditions on $\theta$ to make $T(\cdot ; \theta): \mathcal{U} \rightarrow \mathcal{U}$ equivariant to $\mathcal{S}$. Our approach is to partition the parameters of $\theta$ into orbits under the $\mathcal{S}$-action on its index space (Eq. 10), and share parameters within an orbit.

The index space of $\theta$ is $\mathbb{I}=\mathbb{U} \times \mathbb{U}$. There are four subsets of $\mathbb{I}$:
1.  $\mathbb{I}^{W W}:=\mathbb{W} \times \mathbb{W}$: Contains $[\alpha, \beta]=[(i, j, k),(s, p, q)]$, indexing parameters $\vartheta_{s p q}^{i j k}$.
2.  $\mathbb{I}^{W V}:=\mathbb{W} \times \mathbb{V}$: Contains $[\alpha, \beta]=[(i, j, k),(s, p)]$, indexing parameters $\phi_{s p}^{i j k}$.
3.  $\mathbb{I}^{V W}:=\mathbb{V} \times \mathbb{W}$: Contains $[\alpha, \beta]=[(i, j),(s, p, q)]$, indexing parameters $\varphi_{s p q}^{i j}$.
4.  $\mathbb{I}^{V V}:=\mathbb{V} \times \mathbb{V}$: Contains $[\alpha, \beta]=[(i, j),(s, p)]$, indexing parameters $\psi_{s p}^{i j}$.

Equivariant parameter sharing then amounts to partitioning $\mathbb{I}$ into orbits under $\mathcal{S}$, and then sharing the corresponding parameters within each orbit.

Consider the block of indices $\mathbb{I}^{W W}=\mathbb{W} \times \mathbb{W}$, containing $[\alpha, \beta]=[(i, j, k),(s, p, q)]$ indexing parameters $\vartheta_{\beta}^{\alpha}$. Since the $\mathcal{S}$-action never changes the layer indices $(i, s)$, we can independently consider orbits within sub-blocks of indices $\mathbb{I}_{i, s}^{W W}=\{[(i, j, k),(s, p, q)] \mid \forall j, k, p, q\}$. The number of orbits within each sub-block $\mathbb{I}_{i, s}^{W W}$ depends on the relationship between the layer indices $i$ and $s$: they are either the same layer $(s=i)$, they are adjacent $(s=i-1$ or $s=i+1)$, or they are non-adjacent $(s \notin\{i-1, i, i+1\})$. We now analyze the orbits of sub-blocks for a few cases.

If $s=i-1$, then choose any two indices $\left[\alpha^{(1)}, \beta^{(1)}\right],\left[\alpha^{(2)}, \beta^{(2)}\right] \in \mathbb{I}_{i, s}^{W W}$ where the first satisfies $p \neq k$ and the second satisfies $p=k$. Then the orbits of each index are:

$$
\operatorname{Orbit}\left(\left[\alpha^{(1)}, \beta^{(1)}\right]\right)=\{[(i, j, k),(s, p, q)] \mid \forall j, k, p, q: p \neq k\} \tag{13}
$$

$$
\operatorname{Orbit}\left(\left[\alpha^{(2)}, \beta^{(2)}\right]\right)=\{[(i, j, k),(s, p, q)] \mid \forall j, k, p, q: p=k\}. \tag{14}
$$

We see that these two orbits actually partition the entire sub-block of indices $\mathbb{I}_{i,s}^{WW}$, with each orbit characterized by whether or not $p = k$. We introduce the parameters $a_{\vartheta}^{i,i-1}$ (for the first orbit) and $b_{\vartheta}^{i,i-1}$ (for the second orbit). Under equivariant parameter sharing, all parameters of $\vartheta$ corresponding $\mathbb{I}_{i,s}^{WW}$ are equal to either $a_{\vartheta}^{i,i-1}$ or $b_{\vartheta}^{i,i-1}$, depending on whether $p = k$ or $p \neq k$.

If $s = i+1$, we instead choose any two indices where the first satisfies $j \neq q$ and the second satisfies $j = q$. Then the sub-block of indices $\mathbb{I}_{i,s}^{WW}$ is again partitioned into two orbits:

$$\{[(i,j,k),(s,p,q)] \mid \forall j,k,p,q: j \neq q \} , \text{ and } \{[(i,j,k),(s,p,q)] \mid \forall j,k,p,q: j = q \} \tag{15}$$

depending on the condition $j = q$. We name two parameters $a_{\vartheta}^{i,i+1}$ and $c_{\vartheta}^{i,i+1}$ for this sub-block, with one for each orbit.

We can repeat this process for sub-blocks of $\mathbb{I}^{WW}$ where $i = s$ and $s \notin \{i-1,i,i+1\}$, as well as for the other three blocks of $\mathbb{I}$. Table 7 shows the complete parameter sharing constraints on $\theta$ resulting from partitioning all possible sub-blocks into orbits.

**Number of parameters.** We also note that every layer pair $(i,s)$ introduces only a constant number of parameters: the number of parameters in each cell of Table 7 has no dependence on the input, output, or hidden dimensions of $\mathcal{U}$. Hence the number of distinct parameters after parameter sharing simply grows with the number of layer pairs, i.e. $O\left(L^2\right)$.

### B.5 Equivalence to equivariant NF-Layer definition

All that remains is to show that the map $T(\cdot ; \theta): \mathcal{U} \to \mathcal{U}$ with $\mathcal{S}$-equivariant parameter sharing (Table 7) is equivalent to the NF-Layer $H$ we defined in Eq. 3.

Consider a single term from Eq. 4 where $s = i-1$. Substituting using the constraints of Table 7, we simplify:

$$
\begin{aligned}
\sum_{p,q} \vartheta_{i-1,p,q}^{i,j,k} W_{pq}^{(i-1)} &= a^{i,i-1} \sum_{q} \sum_{k \neq p} W_{p,q}^{(i-1)} + b^{i,i-1} \sum_{q} W_{k,q}^{(i-1)} \\
&= a^{i,i-1} W_{\star,\star}^{(i-1)} + \left(b^{i,i-1} - a^{i,i-1}\right) W_{k,\star}^{(i-1)}.
\end{aligned} \tag{16}
$$

We can then reparameterize $b^{i,i-1} \leftarrow b^{i,i-1} - a^{i,i-1}$, resulting in two terms that appear in Eq. 3. We can simplify every term of Eq. 4 in a similar manner using the parameter sharing of Table 7, reducing the general layer to the $\mathcal{S}$-equivariant NF-Layer.

## C NF-Layers for the HNP setting

### C.1 Equivariant NF-Layer

Because an expression for the $\tilde{\mathcal{S}}$-equivariant NF-Layer analogous to Eq. 3 would be unwieldy, we instead define the layer in terms of its parameter sharing (Tables 8-11) on $\theta$.

We can derive HNP-equivariant parameter sharing of $\theta$ using a similar strategy to Sec. B.4: we partition the index spaces $\mathbb{I}^{WW}, \mathbb{I}^{WV}, \mathbb{I}^{VW}, \mathbb{I}^{VV}$ into orbits under the action of $\tilde{\mathcal{S}}$, and share parameters within each corresponding orbit of $\vartheta, \phi, \varphi, \psi$. The resulting parameter sharing is different from the NP-setting because while the action of $\mathcal{S}$ on $\mathcal{U}$ could permute the rows and columns of every weight and bias, the action of $\tilde{\mathcal{S}}$, on $\mathcal{U}$ does not affect the columns of $W^{(1)}$ or the rows of $W^{(L)}, v^{(L)}$, which correspond to input and output dimensions (respectively).

The orbits are again analyzed within sub-blocks defined by the values of the layer indices $(i,s)$. As with the NP setting, there are broadly four types of sub-blocks based on whether $i = s, s = i-1$, $s = i+1$, or $s \notin \{i-1,i,i+1\}$. However, there are now additional considerations based on whether $i$ or $s$ is an input or output layer. For example, consider the sub-block of $\mathbb{I}^{WW}$ where $i = s = 1$, which we denote $\mathbb{I}_{1,1}^{WW}$. The action on the indices in this sub-block can be written $\sigma [\alpha,\beta] = [(1,\sigma_1(j),k),(1,\sigma_1(p),q)]$. Importantly, the column indices $k,q$ are never permuted since they correspond to the input layer. We see that $\mathbb{I}_{1,1}^{WW}$ contains two orbits *for each* $k \in [\![1..n_0]\!]$ and $q \in [\![1..n_0]\!]$, with the two orbits characterized by whether or not $j = p$. Hence we have $2n_0^2$

18

Table 8: HNP-equivariant parameter sharing on $\vartheta \subseteq \theta$, corresponding to the NF-Layer $\tilde{H}: \mathcal{U} \rightarrow \mathcal{U}$.

$\vartheta_{s p q}^{i j k}$

<table>
<tbody>
<tr>
<td rowspan="2">$s = i - 1$</td>
<td>$i = 2$</td>
<td>$2 < i < L$</td>
<td>$i = L$</td>
</tr>
<tr>
<td>$\begin{cases} a_{\vartheta}^{2,1,q} & k \neq p \\ b_{\vartheta}^{2,1,q} & k = p \end{cases}$</td>
<td>$\begin{cases} a_{\vartheta}^{i,i-1} & k \neq p \\ b_{\vartheta}^{i,i-1} & k = p \end{cases}$</td>
<td>$\begin{cases} a_{\vartheta}^{L,L-1,j} & k \neq p \\ b_{\vartheta}^{L,L-1,j} & k = p \end{cases}$</td>
</tr>
<tr>
<td rowspan="2">$s = i$</td>
<td>$i = 1$</td>
<td>$1 < i < L$</td>
<td>$i = L$</td>
</tr>
<tr>
<td>$\begin{cases} a_{\vartheta}^{1,1,k,q} & j \neq p \\ b_{\vartheta}^{1,1,k,q} & j = p \end{cases}$</td>
<td>$\begin{cases} a_{\vartheta}^{i,i} & j \neq p, k \neq q \\ b_{\vartheta}^{i,i} & j = p, k \neq q \\ c_{\vartheta}^{i,i} & j \neq p, k = q \\ d_{\vartheta}^{i,i} & j = p, k = q \end{cases}$</td>
<td>$\begin{cases} a_{\vartheta}^{L,L,j,p} & k \neq q \\ c_{\vartheta}^{L,L,j,p} & k = q \end{cases}$</td>
</tr>
<tr>
<td rowspan="2">$s = i + 1$</td>
<td>$i = 1$</td>
<td>$1 < i < L - 1$</td>
<td>$i = L - 1$</td>
</tr>
<tr>
<td>$\begin{cases} a_{\vartheta}^{1,2,k} & j \neq q \\ c_{\vartheta}^{1,2,k} & j = q \end{cases}$</td>
<td>$\begin{cases} a_{\vartheta}^{1,2} & j \neq q \\ c_{\vartheta}^{1,2} & j = q \end{cases}$</td>
<td>$\begin{cases} a_{\vartheta}^{L-1,L,p} & j \neq q \\ c_{\vartheta}^{L-1,L,p} & j = q \end{cases}$</td>
</tr>
<tr>
<td rowspan="4">other $s$</td>
<td>$i = 1, 1 < s < L$</td>
<td>$i = 1, s = L$</td>
<td>$1 < i < L, s = L$</td>
</tr>
<tr>
<td>$a_{\vartheta}^{1,s,k}$</td>
<td>$a_{\vartheta}^{1,L,k,p}$</td>
<td>$a_{\vartheta}^{i,L,p}$</td>
</tr>
<tr>
<td>$1 < i < L, s = 1$</td>
<td>$i = L, s = 1$</td>
<td>$i = L, 1 < s < L$</td>
</tr>
<tr>
<td>$a_{\vartheta}^{i,1,q}$</td>
<td>$a_{\vartheta}^{L,1,j,q}$</td>
<td>$a_{\vartheta}^{L,s,j}$</td>
</tr>
<tr>
<td></td>
<td>$1 < i < L, 1 < s < L$</td>
<td></td>
<td></td>
</tr>
<tr>
<td></td>
<td>$a_{\vartheta}^{i,s}$</td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

Table 9: HNP-equivariant parameter sharing on $\phi \subset \theta$, corresponding to the NF-Layer $\tilde{H}: \mathcal{U} \rightarrow \mathcal{U}$.

$\phi_{s p}^{i j k}$

<table>
<tbody>
<tr>
<td rowspan="2">$s = i - 1$</td>
<td></td>
<td>$1 < i < L$</td>
<td>$i = L$</td>
</tr>
<tr>
<td></td>
<td>$\begin{cases} a_{\phi}^{i,i-1} & k \neq p \\ b_{\phi}^{i,i-1} & k = p \end{cases}$</td>
<td>$\begin{cases} a_{\phi}^{L,L-1,j} & k \neq p \\ b_{\phi}^{L,L-1,j} & k = p \end{cases}$</td>
</tr>
<tr>
<td rowspan="2">$s = i$</td>
<td>$i = 1$</td>
<td>$1 < i < L$</td>
<td>$i = L$</td>
</tr>
<tr>
<td>$\begin{cases} a_{\phi}^{1,1,k} & j \neq p \\ b_{\phi}^{1,1,k} & j = p \end{cases}$</td>
<td>$\begin{cases} a_{\phi}^{i,i} & j \neq p \\ b_{\phi}^{i,i} & j = p \end{cases}$</td>
<td>$b_{\phi}^{L,L,j,p}$</td>
</tr>
<tr>
<td rowspan="3">other $s$</td>
<td>$i = 1, 1 < s < L$</td>
<td>$i = 1, s = L$</td>
<td>$1 < i < L, s = L$</td>
</tr>
<tr>
<td>$a_{\phi}^{1,s,k}$</td>
<td>$a_{\phi}^{1,L,k,p}$</td>
<td>$a_{\phi}^{i,L,p}$</td>
</tr>
<tr>
<td>$1 < i < L, 1 \leq s < L$</td>
<td>$i = L, 1 \leq s < L$</td>
<td rowspan="2">$a_{\phi}^{L,s,j}$</td>
</tr>
<tr>
<td></td>
<td>$a_{\phi}^{i,s}$</td>
<td></td>
</tr>
</tbody>
</table>

orbits and Table 8 introduces $2n_0^2$ parameters $\left\{ a_{\vartheta}^{1,1,k,q}, b_{\vartheta}^{1,1,k,q} \mid k, q \in [1..n_0] \right\}$ for this sub-block of parameters.

Now consider another sub-block of $\mathbb{I}^{WW}$ where $1 < i = s < L$. Now the action of $\tilde{\mathcal{S}}$ on indices in this sub-block can be written $\sigma [\alpha, \beta] = [(i, \sigma_i(j), \sigma_{i-1}(k)), (i, \sigma_i(p), \sigma_{i-1}(q))]$. Then we have a total of two orbits characterized by whether or not $k = p$, rather than $2n_0^2$ orbits for the $i = 1$ case. Tables 8-11 present the complete parameter sharing for each of $\vartheta, \phi, \varphi, \psi$, resulting from analyzing every possible orbit within any sub-block of $\mathbb{I}^{WW}, \mathbb{I}^{WV}, \mathbb{I}^{VW}, \mathbb{I}^{VV}$.

19

Table 10: HNP-equivariant parameter sharing on $\varphi \subset \theta$, corresponding to the NF-Layer $\tilde{H}: \mathcal{U} \to \mathcal{U}$.

<table>
<thead>
  <tr>
    <th colspan="2"></th>
    <th colspan="3">$\varphi_{spq}^{ij}$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">$s = i$</td>
    <td></td>
    <td>$i=1$</td>
    <td>$1 < i < L$</td>
    <td>$i=L$</td>
  </tr>
  <tr>
    <td></td>
    <td>$\begin{cases} a_{\varphi}^{1,1,q} & j \neq p \\ b_{\varphi}^{1,1,k} & j = p \end{cases}$</td>
    <td>$\begin{cases} a_{\varphi}^{i,i} & j \neq p \\ b_{\varphi}^{i,i} & j = p \end{cases}$</td>
    <td>$b_{\varphi}^{L,L,j,p}$</td>
  </tr>
  <tr>
    <td rowspan="2">$s = i+1$</td>
    <td></td>
    <td></td>
    <td>$1 \leq i < L - 1$</td>
    <td>$i = L - 1$</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>$\begin{cases} a_{\varphi}^{i,i+1} & j \neq p \\ b_{\varphi}^{i,i+1} & j = p \end{cases}$</td>
    <td>$\begin{cases} a_{\varphi}^{L-1,L,p} & j \neq p \\ b_{\varphi}^{L-1,L,p} & j = p \end{cases}$</td>
  </tr>
  <tr>
    <td rowspan="4">other $s$</td>
    <td></td>
    <td>$1 \leq i < L, s=1$</td>
    <td>$1 \leq i < L, 1 < s < L$</td>
    <td>$1 \leq i < L, s=L$</td>
  </tr>
  <tr>
    <td></td>
    <td>$a_{\varphi}^{i,s,q}$</td>
    <td>$a_{\varphi}^{i,s}$</td>
    <td>$a_{\varphi}^{i,L,p}$</td>
  </tr>
  <tr>
    <td></td>
    <td>$i = L, s=1$</td>
    <td>$i = L, 1 < s < L$</td>
    <td rowspan="2"></td>
  </tr>
  <tr>
    <td></td>
    <td>$a_{\varphi}^{L,1,j,q}$</td>
    <td>$a_{\varphi}^{L,s,j}$</td>
  </tr>
</tbody>
</table>

Table 11: HNP-equivariant parameter sharing on $\psi \subset \theta$, corresponding to the NF-Layer $\tilde{H}: \mathcal{U} \to \mathcal{U}$.

<table>
<thead>
  <tr>
    <th colspan="2"></th>
    <th colspan="3">$\psi_{sp}^{ij}$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">$s = i$</td>
    <td></td>
    <td></td>
    <td>$1 \leq i < L$</td>
    <td>$i=L$</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>$\begin{cases} a_{\psi}^{i,i} & j \neq p \\ b_{\psi}^{i,i} & j = p \end{cases}$</td>
    <td>$b_{\psi}^{L,L,j,p}$</td>
  </tr>
  <tr>
    <td rowspan="2">other $s$</td>
    <td></td>
    <td>$1 \leq i < L, s=L$</td>
    <td>$i = L, 1 \leq s < L$</td>
    <td>$1 \leq i < L, 1 \leq s < L$</td>
  </tr>
  <tr>
    <td></td>
    <td>$a_{\psi}^{i,L,p}$</td>
    <td>$a_{\psi}^{L,s,j}$</td>
    <td>$a_{\psi}^{i,s}$</td>
  </tr>
</tbody>
</table>

## C.2 Invariant NF-Layer

While the NP-invariant NF-Layer sums over the rows and columns of every weight and bias, under HNP assumptions there is no need to sum over the columns of $W^{(1)}$ (inputs) or the rows of $W^{(L)}, v^{(L)}$ (outputs). So the HNP invariant NF-Layer $\tilde{P}: \mathcal{U} \to \mathbb{R}^{2L + n_0 + 2n_L}$ is defined:

$$
\tilde{P}(U)=\left(P(U), W_{\star,:}^{(1)}, W_{:,\star}^{(L)}, v^{(L)}\right), \tag{17}
$$

where $W_{\star,:}^{(1)}$ and $W_{:,\star}^{(L)}$ denote summing over only the rows or only the columns of the matrix, respectively. Note that $\tilde{P}$ satisfies $\tilde{\mathcal{S}}$-invariance without satifying $\mathcal{S}$-invariance.

## D Additional experimental details

### D.1 Predicting generalization

The model we use consists of three equivariant NF-Layers with 16, 16, and 5 channels respectively. We apply ReLU activations after each linear NF-Layer. The resulting weight space features are passed into an invariant NF-Layer with mean pooling. The output of the invariant NF-Layer is flattened and projected to $\mathbb{R}^{1,000}$. The resulting vector is then passed through an MLP with two hidden layers, each with 1,000 units and ReLU activations. The output is linearly projected to a scalar and passed through a sigmoid function. Since the output of the model can be interpreted as a probability, we train the model with binary cross-entropy with hyperparameters outlined in Table 12. The model is trained for 50 epochs with early stopping based on $\tau$ on the validation set, which takes 1 hour on a Titan RTX GPU.

20

Table 12: Hyperparameters for predicting generalization on Small CNN Zoo.

<table>
<thead>
  <tr>
    <th>Name</th>
    <th>Values</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Optimizer</td>
    <td>Adam</td>
  </tr>
  <tr>
    <td>Learning rate</td>
    <td>0.001</td>
  </tr>
  <tr>
    <td>Batch size</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Loss</td>
    <td>Binary cross-entropy</td>
  </tr>
  <tr>
    <td>Epoch</td>
    <td>50</td>
  </tr>
</tbody>
</table>

## D.2 Predicting "winning ticket" masks from initialization

Concretely, the encoder learns the posterior distribution $q_\theta(Z \mid U_0, M)$ where $Z \in \mathbb{R}^{\text{dim}(\mathcal{U}) \times C}$ is the latent variable for the winning tickets and $C$ is the number of latent channels. The decoder learns $p_\theta(M \mid U_0, Z)$, and both encoder and decoder are implemented using our equivariant NF-Layers. For the prior $p(Z)$ we choose the isometric Gaussian distribution, and train using the evidence lower bound (ELBO):

$$
\mathcal{L}_\theta(M, U_0) = \mathbb{E}_{z \sim q_\theta(\cdot \mid U_0, M)} \left[ \ln p_\theta(M \mid U_0, z) \right] - \mathrm{D}_{\mathrm{KL}} \left( q_\theta(\cdot \mid U_0, M) \parallel p(\cdot) \right).
$$

The initialization and sparsity mask are concatenated so the input to the encoder $q_\theta$ is $(U, M) \in \mathbb{R}^{\text{dim}(\mathcal{U}) \times 2}$. After the bottleneck, we concatenate the latent variables and the original mask along the channels, i.e. the decoder input is $(U_0, Z) \in \mathbb{R}^{\text{dim}(\mathcal{U}) \times (C+1)}$.

The first dataset uses three-layer MLPs with 128 hidden units trained on MNIST and the second uses CNNs with three convolution layers (128 channels) and 2 fully-connected layers trained on CIFAR-10. In each dataset, we include 400 pairs for training and hold out 50 for evaluation. The hyperparameter details are in Table 13. The encoder and decoder models contain 4 equivariant NF-Layers with 64 hidden channels within each layer. The latent variable is 5 dimensions. Training takes 5H on a Titan RTX GPU.

Table 13: Hyperparameters for predicting LTH on MNIST and CIFAR-10.

<table>
<thead>
  <tr>
    <th>Name</th>
    <th>Values</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Optimizer</td>
    <td>Adam</td>
  </tr>
  <tr>
    <td>Learning rate</td>
    <td>$1 \times 10^{-3}$</td>
  </tr>
  <tr>
    <td>Batch size</td>
    <td>[4, 8]</td>
  </tr>
  <tr>
    <td>Epoch</td>
    <td>200</td>
  </tr>
</tbody>
</table>

## D.3 Classifying INRs

We use SIREN [55] for our INRs of CIFAR, FashionMNIST, and MNIST. For the SIREN models, we used a three-layer architecture with 32 hidden neurons in each layer. We trained the SIRENs for 5,000 steps using Adam optimizer with a learning rate of $5 \times 10^{-5}$. Datasets were split into 45,000 training images, 5,000 validation images, and 10,000 (MNIST, CIFAR) or 20,000 (FashionMNIST) test images. We trained 10 copies (MNIST, FashionMNIST) or 20 copies (CIFAR-10) of SIRENs on each training image with different initializations, and a single SIREN on each validation and test image. No additional data augmentation was applied. For 3D shape classification, we adopt the same protocol introduced in [12], and we train each SIREN to fit the Unsigned Distance Function (UDF) value of points sampled around a shape. Each SIREN is composed of a single hidden layer with 128 neurons. We use Adam as an optimizer and we train for 1,000 steps.

We also trained neural functionals with three equivariant NF-Layers + ReLU activations, each with 512 channels, followed by invariant NF-Layers (mean pooling) and a three-layer MLP head with 1,000 hidden units and ReLU activation. Dropout was applied to the MLP head only. For the NFN IO-encoding, we used sinusoidal position encoding with a maximum frequency of 10 and 6 frequency bands (dimension 13). The training hyperparameters are shown in Table 14, and training took $\sim 4$H on a Titan RTX GPU.

21

Table 14: Hyperparameters for classifying INRs on MNIST and CIFAR-10 using neural functionals.

| Name             | Values       |
|------------------|--------------|
| Optimizer        | Adam         |
| Learning rate    | $1 \times 10^{-4}$ |
| Batch size       | 32           |
| Training steps   | $2 \times 10^5$ |
| MLP dropout      | 0.5          |

Table 15: Classification train and test accuracies (%) for datasets of implicit neural representations (INRs) of either MNIST or CIFAR-10. Although permutation augmentations slightly increase performance by reducing overfitting, even the larger MLPs are unable to robustly classify INRs. Uncertainties indicate standard error over three runs.

|              |       | MLP-4000       | MLP-4000$_{\text{Aug}}$ | MLP-8000       | MLP-8000$_{\text{Aug}}$ |
|--------------|-------|----------------|-------------------------|----------------|-------------------------|
| CIFAR-10     | Train | $30.4 \pm 0.521$ | $20.5 \pm 0.333$        | $35.9 \pm 0.868$ | $18.1 \pm 0.347$        |
|              | Test  | $17.1 \pm 0.120$ | $19.3 \pm 0.325$        | $17.3 \pm 0.280$ | $19.6 \pm 0.060$        |
| MNIST        | Train | $72.6 \pm 1.39$  | $19.4 \pm 1.39$         | $77.8 \pm 1.74$  | $20.3 \pm 3.30$         |
|              | Test  | $15.5 \pm 0.090$ | $21.1 \pm 0.010$        | $15.8 \pm 0.014$ | $21.3 \pm 0.075$        |

We also experimented with larger MLPs (4,000 and 8,000 hidden units per layer) that have parameter counts comparable to those of the NFNs, but found that it did not significantly increase test accuracy, as shown in Table 15.

### D.4 Weight space style editing
For weight space editing, we use the same INRs as the ones used for classification but we do not augment the dataset with additional INRs. Let $U_i$ be the INR weights for the $i^{\text{th}}$ image and $\text{SIREN}(x, y; U)$ be the output of the INR parameterized by $U$ at coordinates $(x, y)$. We edit the INR weights $U_i' = U_i + \gamma \cdot \text{NFN}(U_i)$, and $\gamma$ is a learned scalar initialized to 0.01. Letting $f_i(x, y)$ be the pixel values of the ground truth edited image (obtained from image-space processing), the objective is to minimize mean squared error:

$$
\mathcal{L}(\mathrm{NFN}) = \frac{1}{N \cdot d^2} \sum_{i=1}^N \sum_{x,y}^d \left\| \mathrm{SIREN}\left(x, y; U'\right) - f_i(x, y) \right\|_2^2. \tag{18}
$$

Note that since the SIREN itself is differentiable, the loss can be directly backpropagated through $U'$ to the parameters of the NFN.

The neural functionals contain 3 equivariant NF-Layers with 128 channels, one invariant NF-Layer (mean pooling) followed by 4 linear layers with 1,000 hidden neurons. Every layer uses ReLU activation. The training hyperparameters can be found in Table 16, and training takes $\sim 1$ hour on a Titan RTX GPU.

Table 16: Hyperparameters for weight space style editing using neural functionals.

| Name             | Values       |
|------------------|--------------|
| Optimizer        | Adam         |
| Learning rate    | $1 \times 10^{-3}$ |
| Batch size       | 32           |
| Training steps   | $5 \times 10^4$ |

Table 17: Classification train and test accuracies (%) for implicit neural representations of MNIST, FashionMNIST, and CIFAR-10. Our equivariant NFNs outperform the MLP baselines, even when the MLP has permutation augmentations to encourage invariance. Uncertainties indicate standard error over three runs.

<table>
<thead>
<tr>
<th rowspan="2" colspan="2"></th>
<th colspan="2">NFN<sub>HNP</sub></th>
<th colspan="2">NFN<sub>NP</sub></th>
<th colspan="2">MLP</th>
<th colspan="2">MLP<sub>Aug</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">CIFAR-10</td>
<td>Train</td>
<td>75.5 $\pm$ 0.810</td>
<td>66.0 $\pm$ 0.694</td>
<td>23.7 $\pm$ 2.39</td>
<td>19.1 $\pm$ 1.75</td>
</tr>
<tr>
<td>Test</td>
<td>44.1 $\pm$ 0.471</td>
<td><b>46.6 $\pm$ 0.072</b></td>
<td>16.9 $\pm$ 0.250</td>
<td>18.9 $\pm$ 0.432</td>
</tr>
<tr>
<td rowspan="2">MNIST</td>
<td>Train</td>
<td>94.9 $\pm$ 0.579</td>
<td>95.0 $\pm$ 0.115</td>
<td>42.4 $\pm$ 2.44</td>
<td>20.5 $\pm$ 0.401</td>
</tr>
<tr>
<td>Test</td>
<td>92.5 $\pm$ 0.071</td>
<td><b>92.9 $\pm$ 0.218</b></td>
<td>14.5 $\pm$ 0.035</td>
<td>21.0 $\pm$ 0.172</td>
</tr>
<tr>
<td rowspan="2">FashionMNIST</td>
<td>Train</td>
<td>82.3 $\pm$ 2.78</td>
<td>81.8 $\pm$ 0.868</td>
<td>44.5 $\pm$ 2.17</td>
<td>14.9 $\pm$ 1.45</td>
</tr>
<tr>
<td>Test</td>
<td>72.7 $\pm$ 1.53</td>
<td><b>75.6 $\pm$ 1.07</b></td>
<td>12.5 $\pm$ 0.111</td>
<td>15.9 $\pm$ 0.181</td>
</tr>
</tbody>
</table>

Table 18: Classification train and test accuracies (%) for datasets of implicit neural representations (INRs) of either ShapeNet-10 [6] or ScanNet-10 [10] Our equivariant NFNs outperform the MLP baselines and recent non-equivariant methods such as inr2vec [12]. Uncertainties indicate standard error over three runs.

<table>
<thead>
<tr>
<th rowspan="2" colspan="2"></th>
<th colspan="2">NFN<sub>HNP</sub></th>
<th colspan="2">NFN<sub>NP</sub></th>
<th colspan="2">MLP</th>
<th colspan="2">MLP<sub>Aug</sub></th>
<th colspan="2">inr2vec[12]</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">ShapeNet-10</td>
<td>Train</td>
<td>100 $\pm$ 0.0</td>
<td>100.0 $\pm$ 0.0</td>
<td>100.0 $\pm$ 0.0</td>
<td>34.0 $\pm$ 0.0</td>
<td>99.0 $\pm$ 0.0</td>
</tr>
<tr>
<td>Test</td>
<td>86.9 $\pm$ 0.860</td>
<td><b>88.7 $\pm$ 0.461</b></td>
<td>25.4 $\pm$ 0.121</td>
<td>33.8 $\pm$ 0.126</td>
<td>39.1 $\pm$ 0.385</td>
</tr>
<tr>
<td rowspan="2">ScanNet-10</td>
<td>Train</td>
<td>100.0 $\pm$ 0.0</td>
<td>100.0 $\pm$ 0.0</td>
<td>100.0 $\pm$ 0.0</td>
<td>42.7 $\pm$ 0.012</td>
<td>93.8 $\pm$ 0.090</td>
</tr>
<tr>
<td>Test</td>
<td>64.1 $\pm$ 0.572</td>
<td><b>65.9 $\pm$ 1.10</b></td>
<td>32.9 $\pm$ 0.351</td>
<td>45.5 $\pm$ 0.126</td>
<td>38.2 $\pm$ 0.409</td>
</tr>
</tbody>
</table>

## E Additional experiments and analysis

### E.1 Interpreting learned lottery ticket masks

We further analyze the behavior of NFN<sub>PT</sub> on lottery ticket mask prediction by plotting the mask score predicted for a given initialization value at each layer. To make the visualization clear we train NFN<sub>PT</sub> on MLP mask prediction without layer norm, which can be viewed as a scalar function of the initialization $f^{(i)}: \mathbb{R} \rightarrow \mathbb{R}$ for each layer $i$. Figure 5 plots, for a fixed latent value, the predicted mask score as a function of the initialization value (low mask scores are pruned, while high mask scores are not). These plots suggest that, in the MLP setting, neural functionals are learning something similar to magnitude pruning of the initialization. In our setting, this turns out to be a strong baseline for lottery ticket mask prediction: the test accuracy of models pruned with the modified network is 95.0%.

### E.2 Predicting MLP generalization from weights

In addition to predicting generalization on the Small CNN Zoo benchmark (Section 3.1), we also construct our own datasets to evaluate predicting generalization on MLPs. Specifically, we study three- and five-layer MLPs with 128 units in each hidden layer. For each of the two

Table 19: Kendall’s $\tau$ coefficient and $R^2$ between predicted and actual test accuracies of three- and five-layer MLPs trained on MNIST. Our equivariant neural functionals outperform the baseline from [61] which predicts generalization using only simple weight statistics as features. Uncertainties indicate standard error over five runs.

<table>
<thead>
<tr>
<th rowspan="2" colspan="2"></th>
<th colspan="2">NFN<sub>HNP</sub></th>
<th colspan="2">NFN<sub>NP</sub></th>
<th colspan="2">StatNN</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">$\tau$</td>
<td>3-Layer</td>
<td><b>0.876 $\pm$ 0.003</b></td>
<td>0.859 $\pm$ 0.002</td>
<td>0.854 $\pm$ 0.002</td>
</tr>
<tr>
<td>5-Layer</td>
<td><b>0.871 $\pm$ 0.001</b></td>
<td>0.855 $\pm$ 0.001</td>
<td>0.860 $\pm$ 0.001</td>
</tr>
<tr>
<td rowspan="2">$R^2$</td>
<td>3-Layer</td>
<td><b>0.957 $\pm$ 0.003</b></td>
<td>0.9424 $\pm$ 0.003</td>
<td>0.937 $\pm$ 0.002</td>
</tr>
<tr>
<td>5-Layer</td>
<td><b>0.956 $\pm$ 0.002</b></td>
<td>0.947 $\pm$ 0.001</td>
<td>0.950 $\pm$ 0.001</td>
</tr>
</tbody>
</table>

23

![](./images/867760378233225806_5.jpg)

Figure 5: Mask scores vs weight magnitude for modified NFN$_{\text{PT}}$.

architectures, we train 2,000 MLPs on MNIST with varying optimization hyperparameters, and save 10 randomly-selected checkpoints from each run to construct a dataset of 20,000 (weight, test accuracy) pairs. Runs are partitioned according to a 90% / 10% split for training and testing.

We evaluate $\text{NFN}_{\text{HNP}}$ and $\text{NFN}_{\text{NP}}$ on this task and compare them to the $\text{STATNN}$ baseline [61] which predicts test accuracy from hand-crafted features extracted from the weights. Table 19 shows that $\text{NFN}_{\text{NP}}$ and $\text{STATNN}$ are broadly comparable, while $\text{NFN}_{\text{HNP}}$ consistently outperform other methods across both datasets in two measures of correlation: Kendall's tau and $R^2$. These results confirm that processing the raw weights with permutation equivariant neural functionals can lead to greater predictive power when assessing generalization from weights.