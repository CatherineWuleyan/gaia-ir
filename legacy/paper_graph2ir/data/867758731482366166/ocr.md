# Towards Accurate Quantization and Pruning via Data-free
Knowledge Transfer

Chen Zhu$^\dagger$ &emsp; Zheng Xu$^\dagger$ &emsp; Ali Shafahi$^\dagger$ &emsp; Manli Shu$^\dagger$ &emsp; Amin Ghiasi$^\dagger$
Tom Goldstein$^\dagger$

## Abstract
When large scale training data is available, one can obtain compact and accurate networks to be deployed in resource-constrained environments effectively through quantization and pruning. However, training data are often protected due to privacy concerns and it is challenging to obtain compact networks without data. We study data-free quantization and pruning by transferring knowledge from trained large networks to compact networks. Auxiliary generators are simultaneously and adversarially trained with the targeted compact networks to generate synthetic inputs that maximize the discrepancy between the given large network and its quantized or pruned version. We show theoretically that the alternating optimization for the underlying minimax problem converges under mild conditions for pruning and quantization. Our data-free compact networks achieve competitive accuracy to networks trained and fine-tuned *with* training data. Our quantized and pruned networks achieve good performance while being more compact and lightweight. Further, we demonstrate that the compact structure and corresponding initialization from the Lottery Ticket Hypothesis can also help in data-free training.

## 1 Introduction
Deep neural networks (DNNs) have been applied to a wide range of tasks and applications in computer vision and sequence modeling. DNNs with impressive performance are often huge models with a large number of parameters and high computational cost, which limits their deployment on resource constrained devices with limited memory and processing power. With the emergence of edge devices and wide-ranging applications of deep neural networks, the demand for lightweight neural networks has increased.

To address this demand, many methods have been proposed to obtain lightweight models with modest computational/memory costs without a great sacrifice in performance compared to the full model. Some common techniques include knowledge distillation, quantization, and network pruning. Knowledge distillation works by enforcing the smaller network named the student to generate outputs similar to those of the trained larger network named the teacher [HVD15]. Quantization refers to reducing the number of bits for representing network parameters or their activations [CBD15]. Network pruning corresponds to keeping a minimal set of network parameters [HMD15]. All these methods, in their conventional setting, require some kind of access to the training set to achieve their best performance. While the availability of training data is a viable assumption for public datasets, there exists many critical cases where the training data is

$^\dagger$University of Maryland, College Park. {chenzhu,xuzh,ashafahi,manlis,ghiasi,tomg}@umd.edu

inaccessible due to concerns about protecting privacy of the users or the intellectual properties of the corporations [TYRW14, WSC⁺¹⁶, MS19]. These practical limitations motivate us to seek solutions for compressing deep models without accessing training data.

## Contributions
Given a pre-trained large scale model with high performance on practical applications, we study data-free methods for training compact models that can run on resource-limited devices. Our contributions are:

- We train compact networks with fast inference capacity and low memory footprint by combining knowledge distillation, quantization, and pruning under an adversarial training framework, where an auxiliary network is adversarially trained to find the worst case synthetic data that differentiates between the given larger network and the target compact network.
- Our method can quantize networks to use extreme low-bit, i.e. binary, representations for weights without noticeable performance degradation, which was not possible by previous data-free methods.
- We compress large networks by pruning the weights of the original network to compression ratios previously only possible with fine-tuning on a large number training data.
- We analyze the convergence of the alternating optimization used for solving the minimax problem of our proposed method. For quantization, we prove an $O(1/\sqrt{k})$ convergence rate for the error bound of convex-concave objectives and bounded gradient variance assumptions. For pruning, we prove linear convergence rate of the nonconvex-nonconcave objective to stationary points under a mild smoothness assumption and a two-sided Polyak-Łojasiewicz condition.
- We find that compared with random initialization, the winning lottery ticket found in the supervised setting also achieves higher accuracy in the data-free setting, indicating the Lottery Ticket Hypothesis may transfer across learning methods and has data-dependent benefits to generalization.

The proposed method can be widely applied to different network architectures, applications, and datasets.

## 2 Related Works
Data-free Knowledge Transfer Overall, image synthesis is a common technique used in many recent methods for accomplishing tasks such as distillation, network compression, quantization and model inversion in the data-free setting. [YML⁺¹⁹] proposes adaptive model inversion to tackle tasks such as pruning, distillation, and continual learning without training data. They use a squared error penalty to enforce the batch-statistics of the synthetic images to be similar to those of the training data to generate the synthetic images. Apart from the batch-statistics penalty, the image generation/inversion step follows principles of inceptionism [MOT15]. Adaptive model inversion is an enhanced version of DeepInversion [YML⁺¹⁹] which aims at increasing diversity by incorporating a loss term in model inversion which maximizes the Jensen-Shannon divergence between the logits of the teacher and student networks. [NMS⁺¹⁹] samples class labels from a Dirichlet distribution

and finds synthetic inputs that minimize the KL Divergence between their outputs in the teacher model and the sampled class labels. It then uses such synthetic data for the downstream tasks.

Given a teacher network trained on an unknown dataset, [CWX⁺19] use a generator to synthesis images that maximize certain responses of the teacher network, so that it can approximate the original training data. Then they use the synthesized images to distill the knowledge of the teacher network onto the student network. [MS19] also use a generator, which is trained to generate pseudo data that maximize the output discrepancy between the student and teacher network. This allows the student network to be trained on data spreading over the input space. These methods require full access to the weights and architecture of the teacher network and are not easily applicable to cases where we only have black-box access to the teacher or only know its architecture. [FSS⁺19] trains a generator to generate inputs that maximize the discrepancy between the teacher and student models, while training the student to minimize such discrepancy. Despite the similarity in adversarial framework, we train more compact models with quantization and pruning, and provide convergence analysis of such minimax optimization under reasonable assumptions.

Data-free Quantization [HHHS19] illustrate that by enforcing a KL penalty on the batch-normalization statistics for image synthesis, one can produce synthetic images which can be used for quantization. Similarly, [CYD⁺20] perform calibration and fine-tuning for quantization by generating synthetic data based on the batch-norm statistics. These batch-statistic-based inversion methods have the limitation that they are targeted for models which are trained with batch-normalization layers. In addition to the methods which do quantization by image synthesis, there does exist data-free quantization methods which are post-training. [NBBW19] propose weight equalization and bias correction for data-free quantization. Their proposed method results in minimal loss of ImageNet top-1 accuracy for MobileNetV2 for quantization up to 8-bits ($\approx 0.8\%$ drop). To the best of our knowledge, none of the previous data-free methods have been able to efficiently train compact netowrks with binary weights.

Data-free Compression Model compression by pruning, in the conventional setting where we have access to at least a portion of training data, has greatly progressed during recent years. Early works in reducing redundancies in network parameters illustrated that it is possible to reduce the network complexity by removing redundant neurons [SB15, ZQ10] and weights [LDS90]. Most pruning methods result in smaller subnetworks with higher accuracy than training the same subnetwork from scratch. However, most of the progress has been made under the assumption of data availability, and very few works focus on the data-free setting. Some recent works [YML⁺19, HHHS19] proposed data-free compression by utilizing the batch normalization (BN) statistics [IS15] which store first- and second-order statistics of the training data. These data-free methods use gradient methods to generate synthetic images which have similar batch statistics to those of the training data by minimizing the distance between the batch statistics of the synthetic images and the stored BN statistics in the trained model, and then directly use the synthetic data for model pruning.

Lottery Ticket Hypothesis Recently, [FC18] proposed the lottery-ticket hypothesis which shows that randomly-initialized dense neural networks contain a much smaller sub-network with proper initialization that have comparable performance to the larger network when trained using the same

number of iterations.¹ This smaller subnetwork when initialized with the original initialized values used for training the larger network, achieves comparable accuracy to that of the larger network even when trained, in isolation, from scratch. This sub-network is said to have won the initialization lottery and thus is called the winning ticket. Unlike the the original lottery ticket hypothesis that relies on the availability of training data, our focus is on evaluating the transferability of lottery ticket from the supervised setting to the data-free setting.

## 3 Data-free Quantization and Pruning

### 3.1 Data-free via Adversarial Training

Inspired by [MS19], we exploit adversarial training in a knowledge distillation setting [HVD15] for data-free quantization and pruning. We use the pre-trained large network as the teacher network $T(x;\theta_0)$, and train the compact student network $S(x;\theta_s)$ with quantization or pruning, together with an auxiliary generator $G(z;\theta_g)$. The inputs of the generator $G(z;\theta_g)$ are samples from a Guassian distribution $z \sim \mathcal{N}(0,I)$. The compact network $S(x;\theta_s)$ is trained to match the output of given network $T(x;\theta_0)$ for any input $x$, while generator $G(z;\theta_g)$ is trained to generate samples that maximize the discrepancy between $S(x;\theta_s)$ and $T(x;\theta_0)$. The minimax objective is written as

$$
\min_{\theta_s} \max_{\theta_g} \mathbb{E}_{z\sim\mathcal{N}(0,I)} D\left(T(G(z))||S(G(z));\theta_g,\theta_s\right), \tag{1}
$$

where $D$ is a function that measures the divergence between the predicted class probabilities of the two networks. We use $D_{KL}(x||y) = \sum_i x^{(i)} \log(x^{(i)}/y^{(i)})$ for quantization follow [MS19]. For pruning, we empirically find that the symmetric Jensen-Shannon Divergence $D_{JS}(x||y) = \frac{1}{2}D_{KL}(x||y) + \frac{1}{2}D_{KL}(y||x)$ improves the stability. Notice this objective is different from [YML⁺19, CWX⁺19], where $S(x;\theta_s)$ and $T(x;\theta_0)$ are trained in two separate stages.

In addition, we find the spatial attention regularizations used in [MS19, ZK16a] is also beneficial for data-free quantization and pruning:

$$
\mathcal{R}_a(z;\theta_s) = \beta \sum_{l\in\mathcal{S}_a} \left\| \frac{f(s_l)}{\|f(s_l)\|} - \frac{f(t_l)}{\|f(t_l)\|} \right\|, \tag{2}
$$

where $\mathcal{S}_a$ is a selected subset of layers, such as the layers before spatial down-sampling operations. $f(x)=1/N_c\sum_c(x^{(c)})^2$ computes the spatial attention map as the mean of the squared features over the channel dimension, and $s_l,t_l$ are the feature maps of the student and teacher networks at layer $l$.

For notational convenience, we denote the divergence term as

$$
\mathcal{D}(z;\theta_g,\theta_s) = D\left(T(G(z))||S(G(z));\theta_g,\theta_s\right), \tag{3}
$$

and the objective function as

$$
\mathcal{L}(\theta_s,\theta_g) = \mathbb{E}_{z\sim\mathcal{N}(0,I)} \left[ \mathcal{D}(z;\theta_s,\theta_g) + \mathcal{R}_a(z;\theta_s) \right]. \tag{4}
$$

The minimax problem can be optimized by alternating gradient steps. Note that extra constraints are introduced for quantization (Eq. 12) and pruning (Eq. 15). We initialize the to-be-quantized

---
¹In their experiments, the smaller subnetwork only contained 1.5% of the #params of VGG-19, and 11.8% of ResNet-18

compact network by quantizing the full-precision pre-trained weights, and initialize the to-be-pruned network with pre-trained weights. Note that when the compact network is initialized to be exactly the same as the given teacher network, both $\mathcal{D}(z; \theta_g, \theta_s)$ and $\mathcal{R}_a(z; \theta_s)$ would be zero, which could make initial training steps challenging. However, interestingly, the pruning process introduces data-independent regularizations on weights that are not zero (unless all weights are zero), which drives the initial stage of training.

### 3.2 Quantization via BinaryConnect

We slightly modify BinaryConnect (BC) [CBD15] as the quantization method in the gradient descent steps to update the compact student network. The weights of the network are quantized into binary values $\{-\delta, \delta\}$ during the optimization process following [CBD15, LDX$^{+}$17], where $\delta$ is a full-precision scale factor fixed as a constant across all layers.

We accumulate gradients with a full-precision buffer $\theta_b$, and quantize it to get the binary weights. We project the scale of $\theta_b$ to be between $-\delta$ and $\delta$ so that the full precision buffer and the binary weights will not diverge. In summary, each descent step for the compact network with updated generator $\theta_g^k$ proceeds as following

1  Compute the gradients from the binary weights by taking the sign of the buffer $\theta_b$ as

$$
g_{k}=\nabla_{\theta_{s}} \mathcal{L}\left(\delta \operatorname{sign}\left(\theta_{b}^{k}\right), \theta_{g}^{k}\right), \tag{5}
$$

2  Accumulate the weight updates into the buffer as

$$
\hat{\theta}_{b}^{k+1}=\theta_{b}^{k}-\alpha_{k} g_{k}, \tag{6}
$$

3  Clip the weights so that it does not exceed the maximum magnitude specified by $\delta$

$$
\theta_{b}^{k+1}=\Pi_{\left\|\theta_{b}\right\|_{\infty} \leq \delta}\left(\hat{\theta}_{b}^{k+1}\right), \tag{7}
$$

where $\alpha_t$ is the learning rate, $\Pi_{\|\theta_b\|_\infty \leq \delta}(\cdot)$ is a projection operator on the buffer $\theta_b$ such that its magnitude does not exceed $\delta$. Note that we have to keep track of a full precision buffer to quantize to extremely low precision (binary) weights. However, the extra RAM consumption during training is small as the major consumption of RAM comes from the gradient computation. After training for $K$ steps, the weights of the binary network is set to

$$
\theta_{s}=\delta \operatorname{sign}\left(\theta_{b}^{K}\right). \tag{8}
$$

### 3.3 Pruning via Sparse Regularization

We prune the filters of convolutional layers so that the pruned network can achieve acceleration on any platform without requiring the hardware to support accelerated sparse operations. Specifically, let $W \in \mathbb{R}^{n \times m k^{2}}$ be the (flattened) weight matrix of any convolutional layer, with $n$ output channels, $m$ input channels, and a kernel size of $k$.

Inspired by [LLS⁺¹⁷], we introduce a train-
able scaling factor scaling factor $s \in \mathbb{R}^n$ for each
convolutional filter, i.e., each of the $n$ filters
of $W$ and the $n$ entries of the bias (if any) is
multiplied by $s \in \mathbb{R}^n$. This is equivalent to mul-
tiplying each channel of the output feature map
of the convolution operation by $s$. For layers
with Batch Normalization (BN) [IS15], we can
use the trainable scaling factor introduced by
BN, and change BN into the following equivalent
form:

$$
y=s\left(\frac{x-\mu}{\sqrt{\sigma^{2}+\varepsilon}}+b\right), \tag{9}
$$

where $x$ is the input batch of features, $\mu, \sigma^2$ are
the mean and variance of the batch, $\varepsilon > 0$ is a
constant which prevents division by zero, and
$s, b$ are trainable parameters in BN.

We enforce sparsity of these scaling factors $s$
by adding an $\ell_1$-norm regularization on $s$, assum-
ing that the number of necessary filters are less
than the pre-defined redundant structure of the
large network. Together with the regularization
from weight decay, redundant filters for the task
will be guided to have small weights, and can
be identified by the corresponding scaling factor
$s$ since removing filters with small magnitudes
will not have much effect on the final feature representations. After the training process, we set a
threshold $t_s$, and convolutional filters with small trainable scaling factors $s < t_s$ will be pruned.

![](./images/867758731482366166_1.jpg)

Figure 1: Sharing the scaling factor $s$ between resid-
ual blocks with same number of output channels for
the pre-activation residual connections used by the
networks in this paper. We group the residual blocks
according to the number of output channels. Inside
the dashed rectangles are two types of residual blocks,
where the first one containing up-sampling operation in
the residual connections is the first block of each group.
Its first BN is followed by a scaling factor $s_{i-1}'$ shared
with the last group. For the other types of residual
blocks, their first BN is followed by $s_i'$ shared inside
the group.

More specifically, we add the following sparse regularization to the original loss function $\mathcal{L}(\theta_s, \theta_g)$
for pruning:

$$
\mathcal{R}_p(\theta_s)=\sum_{l=1}^{L} \gamma_{l}\left\|s_{l}\right\|_{1}+\lambda\left\|W_{l}\right\|_{F}^{2}+\lambda\left\|b_{l}\right\|^{2}, \tag{10}
$$

where $l$ is the index of the layer, $\gamma_l$ and $\lambda$ are constants. The values of $\gamma_l$ are decided by the size
of the feature map. In multi-layer convolutional networks, feature maps with larger spatial sizes
typically have fewer number of channels and each feature map potentially carries more information.
Hence we set $\gamma_l = \gamma/w_l$, where $w_l$ is the width of the feature map in layer $l$ and $\gamma$ is a constant.

For residual blocks in modern convolutional networks, pruning is more efficient when the
corresponding pruned features maps are aligned for layers connected by the residual connection. We
apply shared scaling factors for the entire residual block to avoid potential inconsistency between
convolutional layers within the residual blocks.

## 4 Convergence analysis

In this section, we analyze the convergence of the alternating optimization for solving the minimax
problem under quantization and pruning constraints. This fills in the blank of theoretical analysis for

previous data-free/zero-shot knowledge transfer methods which utilize a generator to generate the synthetic data [FSS⁺19, MS19]. To make the conclusions applicable to a broader class of problems, by an abuse of notation, we use $\mathcal{F}$ to denote the objective function satisfying certain properties, instead of the loss functions $\mathcal{L}$ for the specific problems.

### 4.1 Data-free Quantization

In data-free quantization, we are solving the following minimax problem,
$$
\min _{x} \max _{y} \mathcal{F}(x, y)
\tag{11}
$$
by the stochastic update rule
$$
\begin{aligned}
\hat{x}_{k+1} & =\hat{x}_{k}-\alpha_{k} g_{x}\left(x_{k}, y_{k}\right) \\
x_{k+1} & =\mathcal{Q}\left(\hat{x}_{k+1}\right) \\
y_{k+1} & =y_{k}+\beta_{k} g_{y}\left(x_{k+1}, y_{k}\right)
\end{aligned}
\tag{12}
$$
where $\mathbb{E}g_{x}(x, y)=\nabla_{x} \mathcal{F}(x, y), \mathbb{E}g_{y}(x, y)=\nabla_{y} \mathcal{F}(x, y), \alpha_{k}, \beta_{k}$ are stepsizes, and $\mathcal{Q}$ is the quantization function $\mathcal{Q}=\delta \operatorname{sign}(x)$.

Assume the optimal solution $(x^{\star}, y^{\star})$ exists, then $\nabla_{x} \mathcal{F}(x^{\star}, y)=\nabla_{y} \mathcal{F}(x, y^{\star})=0$. The following theorem illustrates the convergence of this method by stating that the duality gap, $P(x_{k}, y_{k})=\mathcal{F}(x_{k}, y^{\star})-\mathcal{F}(x^{\star}, y_{k})$, vanishes.

**Theorem 4.1.** Suppose the function $\mathcal{F}(x, y)$ is convex in $x$, concave in $y$, and Lipschitz (i.e., $\left\|\mathcal{F}(x_{1}, y)-\mathcal{F}(x_{2}, y)\right\| \leq L\left\|x_{1}-x_{2}\right\|$); and that the partial gradients are uniformly Lipschitz smooth in $x$, (i.e., $\left\|\nabla_{x} \mathcal{F}(x_{1}, y)-\nabla_{x} \mathcal{F}(x_{2}, y)\right\| \leq L_{x}\left\|x_{1}-x_{2}\right\|$, $\left\|\nabla_{y} \mathcal{F}(x_{1}, y)-\nabla_{y} \mathcal{F}(x_{2}, y)\right\| \leq L_{y}\left\|x_{1}-x_{2}\right\|$). Suppose further that the stochastic gradient approximations satisfy $\mathbb{E}\left\|g_{x}(x, y)\right\|^{2} \leq G_{x}^{2}, \mathbb{E}\left\|g_{y}(x, y)\right\|^{2} \leq G_{y}^{2}$ for scalars $G_{x}$ and $G_{y}$, and that $\mathbb{E}\left\|x^{k}-x^{\star}\right\|^{2} \leq D_{x}^{2}$, and $\mathbb{E}\left\|y^{k}-y^{\star}\right\|^{2} \leq D_{y}^{2}$ for scalars $D_{x}$ and $D_{y}$.

If we choose decreasing learning rate parameters of the form $\alpha_{k}=\frac{C_{\alpha}}{\sqrt{k}}$ and $\beta_{k}=\frac{C_{\beta}}{\sqrt{k}}$, then the alternating optimization has the error bound,
$$
\begin{aligned}
& \mathbb{E}\left[P\left(\bar{x}^{l}, \bar{y}^{l}\right)\right] \\
\leq & \frac{1}{2 \sqrt{l}}\left(\frac{D_{x}^{2}}{C_{\alpha}}+\frac{D_{y}^{2}}{C_{\beta}}\right)+\frac{\sqrt{l+1}}{2 l}\left(\right. \\
& \left.C_{\alpha} G_{x}^{2}+C_{\alpha} L_{y} G_{x}^{2}+C_{\alpha} L_{y} D_{y}^{2}+C_{\beta} G_{y}^{2}\right) \\
& +\left(L_{x} D_{x}+L D_{x}+2 L_{y} D_{y}\right) \sqrt{d} \Delta
\end{aligned}
\tag{13}
$$
where $\bar{x}^{l}=\frac{1}{l} \sum_{k=1}^{l} x^{k}, \bar{y}^{l}=\frac{1}{l} \sum_{k=1}^{l} y^{k}$.

From Theorem 4.1, we can see that the error bound decreases with a standard $O(1/\sqrt{k})$ convergence rate with the convex-concave assumption for stochastic alternating optimization, and eventually converges to a region characterized by the quantization grain $\Delta$. Our error bound also suggests that though we only quantize the compact network $x$, the smoothness of the partial gradients of the compact network $\nabla_{x} \mathcal{F}$ and the teacher network $\nabla_{y} \mathcal{F}$ will reflect in the quantization error. Although the convex-concave assumption is a widely used assumption cannot be satisfied by neural networks in practice, our result provide useful insights and fills in the blank of such analysis in data-free quantization.

### 4.2 Data-free pruning

Formally, in the case of data-free pruning, we are solving the following minimax problem with sparse rank-reduced regularization,
$$
\min _{x} \max _{y} \mathcal{F}(x, y),\tag{14}
$$
where $x=\theta_{s}, y=\theta_{g}$, and $\mathcal{F}(x, y)=\mathbb{E}_{z \sim \mathcal{N}(0, I)}\left[\mathcal{D}(z ; x, y)+\mathcal{R}_{a}(z ; x)+\mathcal{R}_{p}(x)\right]$. Note that this analysis is general and can be directly applied to previous data-free methods without quantization or pruning, e.g., [MS19]. We are unaware of a previous theoretical analysis for such data-free methods. We assume the gradient $\nabla \mathcal{F}$ can be obtained directly, and use the following updates
$$
\begin{aligned}
&x_{k+1}=x_{k}-\alpha_{k} \nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right) \\
&y_{k+1}=y_{k}+\beta_{k} \nabla_{y} \mathcal{F}\left(x_{k+1}, y_{k}\right).
\end{aligned}\tag{15}
$$

For the above scheme, we can prove convergence for a class of nonconvex-nonconcave functions $\mathcal{F}$ in the sense that the gradients vanish, and the method approaches as stationary point. Note, this is stronger than the duality gap notion of convergence used for Theorem 1. Such class of functions should satisfy the following three assumptions.

**Assumption 4.2** (L-Lipschitz gradient/L-Smooth). We say $\mathcal{F}(x, y)$ has $L$-Lipschitz gradient, or equivalently $L$-smooth, if there exists a positive constant $L>0$ such that
$$
\begin{aligned}
&\left\|\nabla_{x} \mathcal{F}\left(x_{1}, y_{1}\right)-\nabla_{x} \mathcal{F}\left(x_{2}, y_{2}\right)\right\| \leq L\left[\left\|x_{1}-x_{2}\right\|+\left\|y_{1}-y_{2}\right\|\right], \\
&\left\|\nabla_{y} \mathcal{F}\left(x_{1}, y_{1}\right)-\nabla_{y} \mathcal{F}\left(x_{2}, y_{2}\right)\right\| \leq L\left[\left\|x_{1}-x_{2}\right\|+\left\|y_{1}-y_{2}\right\|\right].
\end{aligned}
$$

**Assumption 4.3** (Existence of Stationary Point). The objective function $\mathcal{F}$ has at least one stationary point $\left(x^{\star}, y^{\star}\right)$ where $\left\|\nabla_{x} \mathcal{F}\left(x^{\star}, y^{\star}\right)\right\|=\left\|\nabla_{y} \mathcal{F}\left(x^{\star}, y^{\star}\right)\right\|=0$. Also, assume for any fixed $y$, $\arg \min _{x} \mathcal{F}(x, y)$ is a non-empty set with finite optimal values, and $\arg \max _{y} \mathcal{F}(x, y)$ is a non-empty set with finite optimal values.

**Assumption 4.4** (Two-sided PL condition [YKH20]). The objective function $\mathcal{F}(x, y)$ satisfies the two-sided PL condition if there exists constants $\mu_{1}, \mu_{2}>0$ such that
$$
\begin{aligned}
& \frac{1}{2}\left\|\nabla_{x} \mathcal{F}(x, y)\right\|^{2} \geq \mu_{1}\left[\mathcal{F}(x, y)-\min _{x} \mathcal{F}(x, y)\right], \forall x, y, \\
& \frac{1}{2}\left\|\nabla_{y} \mathcal{F}(x, y)\right\|^{2} \geq \mu_{2}\left[\max _{y} \mathcal{F}(x, y)-\mathcal{F}(x, y)\right], \forall x, y.
\end{aligned}
$$

Notice that two-sided PL condition does not imply convexity-concavity. The objective function $\mathcal{F}$ can still be nonconvex-nonconcave, as is the case for neural networks.

Also, define the following potential function to measure the inaccuracy of $\left(x_{k}, y_{k}\right)$
$$
P_{k}:=a_{k}+\lambda b_{k},\tag{16}
$$
where $a_{k}=h\left(x_{k}\right)-h^{*}, b_{k}=h\left(x_{k}\right)-\mathcal{F}\left(x_{k}, y_{k}\right), h(x)=\max _{y} \mathcal{F}(x, y)$, and $h^{*}=\min _{x} h(x)$. Notice both $a_{k}$ and $b_{k}$ are non-negative.

With these assumptions, we prove linear convergence of the objective function to its stationary point with the update rules in Eq. 15. We give the proof in the supplementary material. The proof technique follows [YKH20].

8

both supervised and data-free setting. We demonstrate the transferability of winning tickets and provide additional evidence that the winning tickets are intrinsic properties of neural networks.

## 6 Experiments

We follow the experimental setting in data-free knowledge distillation [MS19], where WRN-40-2 and WRN-16-2 networks [ZK16b] are pre-trained on the CIFAR-10 dataset [KNH09] as teacher networks. We train compact networks by the proposed data-free quantization and pruning methods, and report the accuracy on the test set of CIFAR10. We also present the size of the networks, and compare with baselines of data-free methods and fine-tuning methods with data. We perform ablation study on hyperparameters of the proposed methods.

All experiments run on a single GPU (2080 Ti). Unless otherwise specified, we use the default settings following [MS19]. We use Adam optimizer for both the compact network and the auxiliary generator, with learning rates of $2 \times 10^{-3}$ and $10^{-3}$, respectively, and a batch size of 128. The dimension of the generator input $z$ is 100. The generator takes 1 gradient ascent step to increase $\mathcal{L}(\theta_s, \theta_g)$, followed by the student taking 10 gradient descent steps (followed by clipping $\theta_b$ for quantization) on the synthetic batch generated by the generator to decrease $\mathcal{L}(\theta_s, \theta_g)$ (plus regularization terms $\mathcal{R}_p$ for pruning).

### 6.1 Data-free Quantization

Following the practice of [RORF16], we leave the first convolutional layer and the final linear layer's weights as full-precision, and quantize the weights of all intermediate layers into binary. We fix the scaling factor across all quantized layers as a constant $\delta$. We do a grid search for $\delta$ by setting the teacher to WRN-40-2 and the student to WRN-16-2, and present the results in Figure 2. The accuracy of our best result (88.14%) is only 1.57% lower then that of the full-precision network in the data-free setting [MS19], even though most of the networks weights are binarized. Further, by using WRN-16-2 as the teacher network, and initializing the weights of the binary student as $\delta \text{sign}(\theta_0)$, we can achieve a higher accuracy of 88.98%. As a comparison, training a WRN-16-2 with BC on CIFAR10 achieves 92.97% in the presence of data with data augmentation. This indicates our data-free framework is able to recover most of the capabilities of augmented data for training binary networks.

![](./images/867758731482366166_2.jpg)

Figure 2: Accuracies of the quantized WRN-16-2 under different scales $\delta$, using WRN-40-2 as the teacher network. The solid line is the result of training a full-precision WRN-16-2 under the same hyperparameters in the data-free setting [MS19]. The highest accuracy after quantization is 88.14%, while the full precision one has an average accuracy of 89.71%.

<table>
  <thead>
    <tr>
      <th>$\lambda$</th>
      <th>1e-5</th>
      <th>2e-5</th>
      <th>4e-5</th>
      <th>5e-5</th>
      <th>6e-5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>#Params</td>
      <td>570K</td>
      <td>408K</td>
      <td>348K</td>
      <td>337K</td>
      <td>323K</td>
    </tr>
    <tr>
      <td>#FLOPs</td>
      <td>82.7M</td>
      <td>68.5M</td>
      <td>54.9M</td>
      <td>50.3M</td>
      <td>47.7M</td>
    </tr>
    <tr>
      <td>Acc (%)</td>
      <td>92.77</td>
      <td>92.19</td>
      <td>90.79</td>
      <td>89.92</td>
      <td>89.17</td>
    </tr>
  </tbody>
</table>

Table 1: Performance of the pruned network (WRN-16-2) under different weight decay ($\lambda$) when $\gamma$=2e-3 and the pruning threshold $t_s = 0.1$.

## 6.2 Data-free Pruning

For pruning, we have introduced additional hyper-parameters $\gamma, \lambda$ as defined in Eq. 10, and we use Jensen-Shannon divergence $D_{JS}$. Following a grid search, we set the learning rate to $10^{-3}$. We prune a WRN-16-2 network. Table 1 shows the number of parameters (#Params), floating point operations (#FLOPs) and the accuracy (Acc) of the pruned model under different weight decays ($\lambda$), from which we can see that weight decay has significant impact on the size of the pruned model. Increasing the weight decay penalty by a factor of 6, results in a network with 43% fewer parameters.

By comparison, the impact of $\gamma$ is less significant in the observed range. If we fix $\lambda$ =4e-3, #Params are 372K, 348K and 333K for $\gamma$ =1e-3, 2e-3 and 4e-3, respectively. However, higher weight decay can lead to instability of the training process.

The value of weight decay $\lambda$ not only affects the compression ratio, but also affects the quality of the generator. We plot the images generated by the generator at the end of the optimization process in Figure 3. As $\lambda$ goes higher, the compression ratio is higher and the generated images become sharper.

![](./images/867758731482366166_3.jpg)

Figure 3: Random samples from the generator. Every three columns correspond to the generator from a different setting, corresponding to the settings for $\lambda$ =1e-5, 2e-5, 4e-5, 5e-5 in Table 1. As $\lambda$ becomes larger, the network is pruned further and the generated images look sharper.

**Comparing two divergence metrics:** We find using the symmetric Jensen-Shannon divergence (JSD) as the objective results in better compression ratios and improves the stability. To analyze what contributes to such an improvement, we look at the average entropy of the teacher network's predictions throughout the training process. The lower the entropy is, the teacher network's output probabilities for the input pseudo batches generated by $G(z;\theta_g)$ are more concentrated, which indicates that the pseudo batches are closer to the teacher network's training data. In fact, using JSD does reduce such entropy under the same setting, as shown in Table 2. With JSD,

<table>
    <thead>
        <tr>
            <th></th>
            <th>#Params</th>
            <th>Acc (%)</th>
            <th>#Params</th>
            <th>Acc (%)</th>
            <th>Entropy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>KLD($\gamma$=1e-3, $\lambda$=2e-5)</td>
            <td>462K</td>
            <td>92.89</td>
            <td>450K</td>
            <td>92.88</td>
            <td>0.93</td>
        </tr>
        <tr>
            <td>KLD($\gamma$=2e-3, $\lambda$=2e-5)</td>
            <td>478K</td>
            <td>92.75</td>
            <td>451K</td>
            <td>73.48</td>
            <td>1.01</td>
        </tr>
        <tr>
            <td>JSD($\gamma$=1e-3, $\lambda$=2e-5)</td>
            <td>453K</td>
            <td>92.49</td>
            <td>445K</td>
            <td>92.49</td>
            <td>0.87</td>
        </tr>
        <tr>
            <td>JSD($\gamma$=2e-3, $\lambda$=2e-5)</td>
            <td>429K</td>
            <td>92.19</td>
            <td>408K</td>
            <td>92.19</td>
            <td>0.87</td>
        </tr>
    </tbody>
</table>

Table 2: Comparing the KL divergence (KLD) and symmetric Jensen-Shannon divergences (JSD) for pruning the WRN-16-2 model under similar settings. The first two columns of the results are obtained when setting the pruning threshold $t_s = 0.01$, while the following two are setting $t_s = 0.1$. Larger $\lambda$ and $\gamma$ can lead to higher sparsity and compression ratio, but the accuracy of using KL breaks down to 73.5% when $\gamma$ is increased from 1e-3 to 2e-3. The compression ratio with SKL also tends to be higher.

the generator generates pseudo batches that are closer to the data distribution for the following compression.

**Comparison to supervised setting:** In the supervised setting, various data augmentation techniques can be applied to improve the generalization of the model. Counter-intuitively, such data augmentations can also improve the compression ratio of the WRN's for our compression approach. Our data-free pruning does not perform as good as the supervised setting with data augmentation, but is quite close to the supervised setting without data augmentation. For instance, we can prune the network down to $\approx 250K$ parameters while maintaining $\approx 90\%$ accuracy. The results are shown in Table 3. This inspires us to further investigate enhancing the variety of the generated pseudo batches.

<table>
    <thead>
        <tr>
            <th></th>
            <th>$\lambda$</th>
            <th>$\gamma$</th>
            <th>#Params</th>
            <th>#FLOPs</th>
            <th>Acc</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Supervised</td>
            <td>5e-4</td>
            <td>1e-3</td>
            <td>243K</td>
            <td>36.8M</td>
            <td>88.84</td>
        </tr>
        <tr>
            <td>Supervised + Aug.</td>
            <td>5e-4</td>
            <td>1e-3</td>
            <td>236K</td>
            <td>43.3M</td>
            <td>92.71</td>
        </tr>
        <tr>
            <td>Data-free</td>
            <td>5e-5</td>
            <td>3e-3</td>
            <td>339K</td>
            <td>52.3M</td>
            <td>90.57</td>
        </tr>
        <tr>
            <td>Data-free + Warm up</td>
            <td>3e-4</td>
            <td>1e-2</td>
            <td>254K</td>
            <td>48.2M</td>
            <td>89.19</td>
        </tr>
    </tbody>
</table>

Table 3: Comparing the data free approach with supervised setting, where in the supervised setting the training data is available. For "Data-free + Warm up", we increase the value of $\lambda$ and $\gamma$ linearly from 0 to the values reported in the table. With data augmentation, even more parameters can be pruned in the supervised setting, despite having more computations for a higher test accuracy. The data free setting preserves slightly more parameters than the supervised setting without data augmentations, but the test accuracy is higher. Note that the unpruned network in the same setting has an accuracy of 89.71% [MS19].

<table>
    <thead>
        <tr>
            <th>Student</th>
            <th>#Params (ticket)</th>
            <th>Acc. (ticket)</th>
            <th>Acc. (random)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>WRN-16-2</td>
            <td>53.0K</td>
            <td>77.03</td>
            <td>75.48</td>
        </tr>
        <tr>
            <td>WRN-16-2</td>
            <td>45.5K</td>
            <td>74.66</td>
            <td>73.56</td>
        </tr>
    </tbody>
</table>

Table 4: Comparing the test accuracy of training the winning ticket found in the supervised setting, and the network with the same structure but a different random initialization. We prune 20% weights of the convolutional layers, and run 13 and 14 rounds to find the two tickets.


### 6.3 Finding Winning Tickets for Data-free Setting

To find the winning lottery ticket, we use the procedure as described in the previous section, finding the lottery ticket on supervised setting and evaluate the winning ticket in the data-free setting by training only the weights from the winning ticket in our framework. Following the same setup as [FC18], we only prune the parameters of the weights of convolutional layers. In each round, we prune 20% of the remaining weights. We use a batch size of 128, a learning rate of 0.03 with SGD (momentum 0.9) and train the network for 30000 iterations in the supervised settings. For the data-free setting, the is set to WRN-40-2. The results are in Table 4. From the results of [FC18], we have already know that such winning tickets are beneficial for supervised learning. Despite using a different optimizer (Adam), a different learning rate (2e-3), and a different learning approach (data-free), such winning tickets is still beneficial for the network to generalize well on the same dataset, indicating that the winning ticket has some inductive bias which benefits generalization and transfers across optimizers and learning approaches.

## 7 Conclusion

We study data-free quantization and pruning for training compact networks with strong performance, and provide empirical and theoretical analysis for the proposed adversarial training method. To the best of our knowledge, this paper presents the first method that can train compact network with extreme low bit precision, i.e., binary quantization, without having access to training data. Empirically, we show that weight decay has a great effect on accuracy and the compression ratio. We also illustrate that a symmetric divergence such as Jensen-Shannon enhances the quality of the synthetic input examples. We provide convergence guarantees under mild conditions for the general minimax problem underlying the data-free adversarial training framework, with and without extra constraints from quantization and pruning. Finally, we demonstrate the transferability of Lottery Tickets by showing that winning tickets from the standard supervised setting can benefit the data-free training, shedding some light on the connections in optimziation landscapes between supervised and the proposed data-free learning.

## References

[CBD15] Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural networks with binary weights during propagations. In *Advances in neural information processing systems*, pages 3123–3131, 2015.

[CWX+19] Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, and Qi Tian. Data-free learning of student networks. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 3514–3522, 2019.

[CYD+20] Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Zeroq: A novel zero shot quantization framework. *arXiv preprint arXiv:2001.00281*, 2020.

13

[DZGW18] Bin Dai, Chen Zhu, Baining Guo, and David Wipf. Compressing neural networks using the variational information bottleneck. In *International Conference on Machine Learning*, pages 1135–1144. PMLR, 2018.

[FC18] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv preprint arXiv:1803.03635*, 2018.

[FSS⁺19] Gongfan Fang, Jie Song, Chengchao Shen, Xinchao Wang, Da Chen, and Mingli Song. Data-free adversarial distillation. *arXiv preprint arXiv:1912.11006*, 2019.

[HG16] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016.

[HHHS19] Matan Haroush, Itay Hubara, Elad Hoffer, and Daniel Soudry. The knowledge within: Methods for data-free model compression. *arXiv preprint arXiv:1912.01274*, 2019.

[HMD15] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *arXiv preprint arXiv:1510.00149*, 2015.

[HVD15] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.

[HZRS15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In *Proceedings of the IEEE international conference on computer vision*, pages 1026–1034, 2015.

[IS15] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*, 2015.

[KNH09] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10. (*Canadian Institute for Advanced Research*), 2009.

[KNS16] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, pages 795–811. Springer, 2016.

[LDS90] Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. In *Advances in neural information processing systems*, pages 598–605, 1990.

[LDX⁺17] Hao Li, Soham De, Zheng Xu, Christoph Studer, Hanan Samet, and Tom Goldstein. Training quantized nets: A deeper understanding. In *Advances in Neural Information Processing Systems*, pages 5811–5821, 2017.

[LLS⁺17] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In *Pro- ceedings of the IEEE International Conference on Computer Vision*, pages 2736–2744, 2017.


[MOT15] Alexander Mordvintsev, Christopher Olah, and Mike Tyka. Inceptionism: Going deeper into neural networks, 2015.

[MS19] Paul Micaelli and Amos J Storkey. Zero-shot knowledge transfer via adversarial belief matching. In *Advances in Neural Information Processing Systems*, pages 9547–9557, 2019.

[NBBW19] Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. Data-free quantization through weight equalization and bias correction. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 1325–1334, 2019.

[NMS+19] GK Nayak, KR Mopuri, V Shaj, R Venkatesh Babu, and A Chakraborty. Zero-shot knowledge distillation in deep networks. In *36th International Conference on Machine Learning, ICML 2019*, volume 2019, pages 8317–8325. International Machine Learning Society (IMLS), 2019.

[NSH+19] Maher Nouiehed, Maziar Sanjabi, Tianjian Huang, Jason D Lee, and Meisam Razaviyayn. Solving a class of non-convex min-max games using iterative first order methods. In *Advances in Neural Information Processing Systems*, pages 14905–14916, 2019.

[RORF16] Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. In *European conference on computer vision*, pages 525–542. Springer, 2016.

[SB15] Suraj Srinivas and R Venkatesh Babu. Data-free parameter pruning for deep neural networks. *arXiv preprint arXiv:1507.06149*, 2015.

[TYRW14] Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, and Lior Wolf. Deepface: Closing the gap to human-level performance in face verification. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1701–1708, 2014.

[WSC+16] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*, 2016.

[YKH20] Junchi Yang, Negar Kiyavash, and Niao He. Global convergence and variance-reduced optimization for a class of nonconvex-nonconcave minimax problems. *arXiv preprint arXiv:2002.09621*, 2020.

[YML+19] Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M Alvarez, Arun Mallya, Derek Hoiem, Niraj K Jha, and Jan Kautz. Dreaming to distill: Data-free knowledge transfer via deepinversion. *arXiv preprint arXiv:1912.08795*, 2019.

[ZK16a] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. *arXiv preprint arXiv:1612.03928*, 2016.

[ZK16b] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. *arXiv preprint arXiv:1605.07146*, 2016.

[ZQ10] Zhaozhao Zhang and Junfei Qiao. A node pruning algorithm for feedforward neural network based on neural complexity. In *2010 international conference on intelligent control and information processing*, pages 406–410. IEEE, 2010.

## A Proofs for Data-free Quantization

Assume the optimal solution $(x^\star,y^\star)$ exists, then $\nabla_x \mathcal{F}(x^\star,y) = \nabla_y \mathcal{F}(x,y^\star) = 0$. We show the convergence for the primal-dual gap $P(x_k,y_k) = \mathcal{F}(x_k,y^\star) - \mathcal{F}(x^\star,y_k)$. We prove the $O(1/\sqrt{k})$ convergence rate in Theorem 4.1 by using Lemma A.2 and Lemma A.3, which present the contraction of primal and dual updates, respectively.

Lemma A.1. The quantization error is bounded by
$$
\|\mathcal{Q}(x)-x\| \leq \sqrt{d} \Delta
\tag{18}
$$

Lemma A.2. Suppose $\mathcal{F}(x,y)$ is convex in $x$ and Lipschitz $\|\mathcal{F}(x_1,y)-\mathcal{F}(x_2,y)\| \leq L\|x_1-x_2\|$; and has Lipschitz gradients $\|\nabla_x \mathcal{F}(x_1,y) - \nabla_x \mathcal{F}(x_2,y)\| \leq L_x\|x_1-x_2\|$; and bounded variance $\mathbb{E}[\|g_x(x,y)\|^2] \leq G_x^2$; and $\mathbb{E}[\|x_k-x^\*\|^2] \leq D_x^2$ , we have
$$
\begin{aligned}
\mathbb{E}[\mathcal{F}(x_k,y_k)] - \mathbb{E}[\mathcal{F}(x^\star,y_k)] &\leq \frac{1}{2\alpha_k} \left(\mathbb{E}[\|\hat{x}_k-x^\star\|^2] - \mathbb{E}[\|\hat{x}_{k+1}-x^\star\|^2]\right) \\
&+ \frac{\alpha_k}{2}G_x^2 + (L_x+L)D_x\sqrt{d}\Delta
\end{aligned}
\tag{19}
$$

Proof. From gradient descent step , we have
$$
\begin{aligned}
&\|\hat{x}_{k+1}-x^\star\|^2 \\
=&\|\hat{x}_k - \alpha_k g_x(x_k,y_k) - x^\star\|^2 \\
=&\|\hat{x}_k - x^\star\|^2 - 2\alpha_k \langle g_x(x_k,y_k), \hat{x}_k - x^\star \rangle + \alpha_k^2 \|g_x(x_k,y_k)\|^2 \\
=&\|\hat{x}_k - x^\star\|^2 - 2\alpha_k \langle g_x(\hat{x}_k,y_k) - g_x(\hat{x}_k,y_k) + g_x(x_k,y_k), \hat{x}_k - x^\star \rangle + \alpha_k^2 \|g_x(x_k,y_k)\|^2 \\
=&\|\hat{x}_k - x^\star\|^2 - 2\alpha_k \langle g_x(\hat{x}_k,y_k), \hat{x}_k - x^\star \rangle + 2\alpha_k \langle g_x(\hat{x}_k,y_k) - g_x(x_k,y_k), \hat{x}_k - x^\star \rangle \\
&+ \alpha_k^2 \|g_x(x_k,y_k)\|^2
\end{aligned}
\tag{20}
$$

Take expectation on both side of the equation, $\langle g_x(\hat{x}_k,y_k) - g_x(x_k,y_k), \hat{x}_k - x^\star \rangle$ on the right hand side can be written as
$$
\mathbb{E}[\langle g_x(\hat{x}_k,y_k) - g_x(x_k,y_k), \hat{x}_k - x^\star \rangle]
\tag{21}
$$
$$
=\mathbb{E}[\langle \nabla_x \mathcal{F}(\hat{x}_k,y_k) - \nabla_x \mathcal{F}(x_k,y_k), \hat{x}_k - x^\star \rangle]
\tag{22}
$$
$$
\leq \mathbb{E}[\|\nabla_x \mathcal{F}(\hat{x}_k,y_k) - \nabla_x \mathcal{F}(x_k,y_k)\| \|\hat{x}_k - x^\star\|]
\tag{23}
$$
$$
\leq \mathbb{E}[L_x\|\hat{x}_k - x_k\| \|\hat{x}_k - x^\star\|]
\tag{24}
$$
$$
\leq L_x\sqrt{d}\Delta \mathbb{E}[\|\hat{x}_k - x^\star\|]
\tag{25}
$$

Substitute with $\mathbb{E}[g_x(x,y)] = \nabla_x \mathcal{F}(x,y)$, apply $\mathbb{E}[\|g_x^2(x,y)\|] \leq G_x^2$ and $\mathbb{E}[\|\hat{x}_k-x^\star\|] \leq \sqrt{\mathbb{E}[\|\hat{x}_-x^\star\|^2]} = D_x$ to get
$$
\begin{aligned}
\mathbb{E}[\|x_{k+1}-x^\star\|^2] \leq& \mathbb{E}[\|x_k-x^\star\|^2] - 2\alpha_k \mathbb{E}[\langle \nabla_x \mathcal{F}(x_k,y_k), x_k - x^\star \rangle] \\
&+ \alpha_k^2 G_x^2 + 2\alpha_k L_x D_x \sqrt{d}\Delta.
\end{aligned}
\tag{26}
$$

Since $\mathcal{F}(x,y)$ is convex in $x$, we have
$$
\langle \nabla_x \mathcal{F}(x_k,y_k), x_k - x^\star \rangle \geq \mathcal{F}(x_k,y_k) - \mathcal{F}(x^\star,y_k).
\tag{27}
$$

17

Combining Eq. 26 and Eq. 27, we have

$$
\begin{aligned}
\mathbb{E}[\mathcal{F}(\hat{x}_{k}, y_{k})] - \mathbb{E}[\mathcal{F}(x^{\star}, y_{k})] \leq & \frac{1}{2\alpha_{k}} \left(\mathbb{E}[\|\hat{x}_{k} - x^{\star}\|^{2}] - \mathbb{E}[\|\hat{x}_{k+1} - x^{\star}\|^{2}]\right) \\
& + \frac{\alpha_{k}}{2} G_{x}^{2} + L_{x} D_{x} \sqrt{d} \Delta.
\end{aligned}
\tag{28}
$$

We further have

$$
\mathbb{E}[\mathcal{F}(x_{k}, y_{k})] - \mathbb{E}[\mathcal{F}(x^{\star}, y_{k})]
\tag{29}
$$

$$
=\mathbb{E}[\mathcal{F}(x_{k}, y_{k})] - \mathbb{E}[\mathcal{F}(\hat{x}_{k}, y_{k})] + \mathbb{E}[\mathcal{F}(\hat{x}_{k}, y_{k})] - \mathbb{E}[\mathcal{F}(x^{\star}, y_{k})]
\tag{30}
$$

$$
\leq\mathbb{E}[\|\mathcal{F}(x_{k}, y_{k}) - \mathcal{F}(\hat{x}_{k}, y_{k})\|] + \mathbb{E}[\mathcal{F}(\hat{x}_{k}, y_{k})] - \mathbb{E}[\mathcal{F}(x^{\star}, y_{k})]
\tag{31}
$$

$$
\leq L D_{x} \sqrt{d} \Delta + \mathbb{E}[\mathcal{F}(\hat{x}_{k}, y_{k})] - \mathbb{E}[\mathcal{F}(x^{\star}, y_{k})]
\tag{32}
$$

where Eq. 32 can be proved by applying qunatization error, function Lipschitz, and diameter bound.
Combine Eq. 28 and Eq. 32 to get Eq. 19 in the lemma.
$\square$

Lemma A.3. Suppose $\mathcal{F}(x, y)$ is concave in $y$ and has Lipschitz gradients, i.e., $\|\nabla_{y}\mathcal{F}(x_{1}, y) - \nabla_{y}\mathcal{F}(x_{2}, y)\| \leq L_{y}\|x_{1} - x_{2}\|$; and bounded variance, $\mathbb{E}[\|g_{x}(x, y)\|^{2}] \leq G_{x}^{2}$, $\mathbb{E}[\|g_{y}(x, y)\|^{2}] \leq G_{y}^{2}$; and $\mathbb{E}[\|y_{k} - y^{\star}\|^{2}] \leq D_{y}^{2}$, we have

$$
\begin{aligned}
\mathbb{E}[\mathcal{F}(x_{k}, y^{\star})] - \mathbb{E}[\mathcal{F}(x_{k}, y_{k})] \leq & \frac{1}{2\beta_{k}} \left(\mathbb{E}[\|y_{k} - y^{\star}\|^{2}] - \mathbb{E}[\|y_{k+1} - y^{\star}\|^{2}]\right) \\
& + \frac{\beta_{k}}{2} G_{y}^{2} + 2 L_{y} D_{y} \sqrt{d} \Delta + \frac{L_{y} \alpha_{k}}{2} \left(G_{x}^{2} + D_{y}^{2}\right).
\end{aligned}
\tag{33}
$$

Proof. From the gradient ascent step, we have

$$
\begin{aligned}
\|y_{k+1} - y^{\star}\|^{2} & = \|y_{k} + \beta_{k} g_{y}(x_{k+1}, y_{k}) - y^{\star}\|^{2} \\
& = \|y_{k} - y^{\star}\|^{2} + 2 \beta_{k} \langle g_{y}(x_{k+1}, y_{k}), y_{k} - y^{\star} \rangle + \beta_{k}^{2} \|g_{y}(x_{k+1}, y_{k})\|^{2}.
\end{aligned}
\tag{34}
$$

Take expectation on both sides of the equation, substitute $\mathbb{E}[g_{y}(x, y)] = \nabla_{y}\mathcal{F}(x, y)$, and apply $\mathbb{E}[\|g_{y}^{2}(x, y)\|] \leq G_{y}^{2}$ to get

$$
\mathbb{E}[\|y_{k+1} - y^{\star}\|^{2}] \leq \mathbb{E}[\|y_{k} - y^{\star}\|^{2}] + 2 \beta_{k} \mathbb{E}[\langle \nabla_{y}\mathcal{F}(x_{k+1}, y_{k}), y_{k} - y^{\star} \rangle] + \beta_{k}^{2} G_{y}^{2}.
\tag{36}
$$

Reorganize Eq. 36 to get

$$
\mathbb{E}[\|y_{k+1} - y^{\star}\|^{2}] - \mathbb{E}[\|y_{k} - y^{\star}\|^{2}] - \beta_{k}^{2} G_{y}^{2} \leq 2 \beta_{k} \mathbb{E}[\langle \nabla_{y}\mathcal{F}(x_{k+1}, y_{k}), y_{k} - y^{\star} \rangle].
\tag{37}
$$

The right hand side of Eq. 37 can be represented as

$$
\mathbb{E}[\langle \nabla_{y}\mathcal{F}(x_{k+1}, y_{k}), y_{k} - y^{\star} \rangle]
\tag{38}
$$

$$
=\mathbb{E}[\langle \nabla_{y}\mathcal{F}(x_{k+1}, y_{k}) - \nabla_{y}\mathcal{F}(x_{k}, y_{k}) + \nabla_{y}\mathcal{F}(x_{k}, y_{k}), y_{k} - y^{\star} \rangle]
\tag{39}
$$

$$
=\mathbb{E}[\langle \nabla_{y}\mathcal{F}(x_{k+1}, y_{k}) - \nabla_{y}\mathcal{F}(x_{k}, y_{k}), y_{k} - y^{\star} \rangle] + \mathbb{E}[\langle \nabla_{y}\mathcal{F}(x_{k}, y_{k}), y_{k} - y^{\star} \rangle],
\tag{40}
$$

18

where

$$
\mathbb{E}[\langle\nabla_{y} \mathcal{F}(x_{k+1}, y_{k})-\nabla_{y} \mathcal{F}(x_{k}, y_{k}), y_{k}-y^{\star}\rangle] \tag{41}
$$

$$
\leq \mathbb{E}[\|\nabla_{y} \mathcal{F}(x_{k+1}, y_{k})-\nabla_{y} \mathcal{F}(x_{k}, y_{k})\| \|y_{k}-y^{\star}\|] \tag{42}
$$

$$
\leq \mathbb{E}[L_{y}\|x_{k+1}-x_{k}\| \|y_{k}-y^{\star}\|] \tag{43}
$$

$$
=L_{y} \mathbb{E}[\|\mathcal{Q}(\hat{x}_{k+1})-\mathcal{Q}(\hat{x}_{k})\| \|y_{k}-y^{\star}\|] \tag{44}
$$

$$
=L_{y} \mathbb{E}[\|(\mathcal{Q}(\hat{x}_{k+1})-\hat{x}_{k+1})-(\mathcal{Q}(\hat{x}_{k})-\hat{x}_{k})+(\hat{x}_{k+1}-\hat{x}_{k})\| \|y_{k}-y^{\star}\|] \tag{45}
$$

$$
\leq L_{y} \mathbb{E}[(\|\mathcal{Q}(\hat{x}_{k+1}-\hat{x}_{k+1}\|+\|\mathcal{Q}(\hat{x}_{k})-\hat{x}_{k}\|+\|\alpha_{k} g_{x}(x_{k}, y_{k})\|)\|y_{k}-y^{\star}\|] \tag{46}
$$

$$
\leq L_{y} \mathbb{E}[(\sqrt{d} \Delta+\sqrt{d} \Delta+\|\alpha_{k} g_{x}(x_{k}, y_{k})\|)\|y_{k}-y^{\star}\|] \tag{47}
$$

$$
\leq 2 L_{y} \sqrt{d} \Delta \mathbb{E}[\|y_{k}-y^{\star}\|]+\mathbb{E}[\|\alpha_{k} g_{x}(x_{k}, y_{k})\| \|y_{k}-y^{\star}\|] \tag{48}
$$

$$
\leq 2 L_{y} \sqrt{d} \Delta \mathbb{E}[\|y_{k}-y^{\star}\|]+\frac{L_{y} \alpha_{k}}{2} \mathbb{E}[\|g_{x}(x_{k}, y_{k})\|^{2}+\|y_{k}-y^{\star}\|^{2}] \tag{49}
$$

$$
\leq 2 L_{y} \sqrt{d} \Delta D_{y}+\frac{L_{y} \alpha_{k}}{2}\left(G_{x}^{2}+D_{y}^{2}\right). \tag{50}
$$

Lipschitz smoothness is used for Eq. 43; quanitzation error bound is used in Eq. 47, which is independent of the stochasticity. From the convexity of quadratic function, we have $\mathbb{E}[\|y_{k}-y^{\star}\|] \leq \sqrt{\mathbb{E}[\|y_{k}-y^{\star}\|^{2}]} \leq D_{y}$ to get Eq. 50 Since $\mathcal{F}(x, y)$ is concave in $y$, we have

$$
\langle\nabla_{y} \mathcal{F}(x_{k}, y_{k}), y_{k}-y^{\star}\rangle \leq \mathcal{F}(x_{k}, y_{k})-\mathcal{F}(x_{k}, y^{\star}). \tag{51}
$$

Combine equations (37, 40, 50 to 51)

$$
\begin{aligned}
& \frac{1}{2 \beta_{k}}\left(\mathbb{E}[\|y_{k+1}-y^{\star}\|^{2}]-\mathbb{E}[\|y_{k}-y^{\star}\|^{2}]\right)-\frac{\beta_{k}}{2} G_{y}^{2} \\
& \leq 2 L_{y} \sqrt{d} \Delta D_{y}+\frac{L_{y} \alpha_{k}}{2}\left(G_{x}^{2}+D_{y}^{2}\right)+\mathbb{E}[\mathcal{F}(x_{k}, y_{k})-\mathcal{F}(x_{k}, y^{\star})].
\end{aligned} \tag{52}
$$

Rearrange the order of Eq. 52 to achieve Eq. 33.

---

### Proof for Theorem 4.1

*Proof.* Combining Eq. 19 and Eq. 33 in the Lemmas, the primal-dual gap $P(x_{k}, y_{k})=\mathcal{F}(x_{k}, y^{\star})-\mathcal{F}(x^{\star}, y_{k})$ satisfies,

$$
\begin{aligned}
\mathbb{E}[P(x_{k}, y_{k})] \leq & \frac{1}{2 \alpha_{k}}\left(\mathbb{E}[\|\hat{x}_{k}-x^{\star}\|^{2}]-\mathbb{E}[\|\hat{x}_{k+1}-x^{\star}\|^{2}]\right)+\frac{\alpha_{k}}{2} G_{x}^{2}+(L_{x}+L) D_{x} \sqrt{d} \Delta \\
& +\frac{1}{2 \beta_{k}}\left(\mathbb{E}[\|y_{k}-y^{\star}\|^{2}]-\mathbb{E}[\|y_{k+1}-y^{\star}\|^{2}]\right)+\frac{\beta_{k}}{2} G_{y}^{2}+2 L_{y} \sqrt{d} D_{y} \Delta \\
& +\frac{L_{y} \alpha_{k}}{2}\left(G_{x}^{2}+D_{y}^{2}\right).
\end{aligned} \tag{53}
$$

Accumulate Eq. 53 from $k=1,\dots,l$ to obtain

$$
\begin{aligned}
\sum_{k=1}^{l} \mathbb{E}\left[P\left(x_{k}, y_{k}\right)\right] \leq & \\
\frac{1}{2 \alpha_{1}} \mathbb{E}\left[\left\|x^{1}-x^{\star}\right\|^{2}\right]+\sum_{k=2}^{l}( & \left.\frac{1}{2 \alpha_{k}}-\frac{1}{2 \alpha_{k-1}}\right) \mathbb{E}\left[\left\|x_{k}-x^{\star}\right\|^{2}\right]+\sum_{k=1}^{l} \frac{\alpha_{k}}{2}\left(G_{x}^{2}+L_{y} G_{x}^{2}+L_{y} D_{y}^{2}\right) \\
+\frac{1}{2 \beta_{1}} \mathbb{E}\left[\left\|y^{1}-y^{\star}\right\|^{2}\right]+\sum_{k=2}^{l}( & \left.\frac{1}{2 \beta_{k}}-\frac{1}{2 \beta_{k-1}}\right) \mathbb{E}\left[\left\|y_{k}-y^{\star}\right\|^{2}\right]+\sum_{k=1}^{l} \frac{\beta_{k}}{2} G_{y}^{2} \\
+ & l\left(L_{x} D_{x}+L D_{x}+2 L_{y} D_{y}\right) \sqrt{d}.
\end{aligned}
\tag{54}
$$

Assume $\mathbb{E}[\left\|x_{k}-x^{\star}\right\|^{2}] \leq D_{u}^{2}, \mathbb{E}[\left\|y_{k}-y^{\star}\right\|^{2}] \leq D_{y}^{2}$ are bounded, we have

$$
\begin{aligned}
\sum_{k=1}^{l} \mathbb{E}\left[P\left(x_{k}, y_{k}\right)\right] \leq & \frac{1}{2 \alpha_{1}} D_{x}^{2}+\sum_{k=2}^{l}\left(\frac{1}{2 \alpha_{k}}-\frac{1}{2 \alpha_{k-1}}\right) D_{u}^{2}+\sum_{k=1}^{l} \frac{\alpha_{k}}{2}\left(G x^{2}+L_{y} G_{x}^{2}+L_{y} D_{y}^{2}\right) \\
& +\frac{1}{2 \beta_{1}} D_{y}^{2}+\sum_{k=2}^{l}\left(\frac{1}{2 \beta_{k}}-\frac{1}{2 \beta_{k-1}}\right) D_{y}^{2}+\sum_{k=1}^{l} \frac{\beta_{k}}{2} G_{y}^{2} \\
& +l\left(L_{x} D_{x}+L D_{x}+2 L_{y} D_{y}\right) \sqrt{d}.
\end{aligned}
\tag{55}
$$

Since $\alpha_{k}, \beta_{k}$ are decreasing and $\sum_{k=1}^{l} \alpha_{k} \leq C_{\alpha} \sqrt{l+1}, \sum_{k=1}^{l} \beta_{k} \leq C_{\beta} \sqrt{l+1}$, we have

$$
\begin{aligned}
\sum_{k=1}^{l} \mathbb{E}\left[P\left(x_{k}, y_{k}\right)\right] \leq \frac{\sqrt{l}}{2}\left(\frac{D_{x}^{2}}{C_{\alpha}}+\frac{D_{y}^{2}}{C_{\beta}}\right) & +\frac{\sqrt{l+1}}{2}\left(C_{\alpha} G_{x}^{2}+C_{\alpha} L_{y} G_{x}^{2}+C_{\alpha} L_{y} D_{y}^{2}+C_{\beta} G_{y}^{2}\right) \\
& +l\left(L_{x} D_{x}+L D_{x}+2 L_{y} D_{y}\right) \sqrt{d} \Delta
\end{aligned}
\tag{56}
$$

For $\bar{x}^{l}=\frac{1}{l} \sum_{k=1}^{l} x_{k}, \bar{y}^{l}=\frac{1}{l} \sum_{k=1}^{l} y_{k}$, because $\mathcal{F}(x, y)$ is convex-concave, we have

$$
\mathbb{E}\left[P\left(\bar{x}^{l}, \bar{y}^{l}\right)\right]=\mathbb{E}\left[\mathcal{F}\left(\bar{x}^{l}, y^{\star}\right)-\mathcal{F}\left(y^{\star}, \bar{y}^{l}\right)\right]
\tag{57}
$$

$$
\leq \mathbb{E}\left[\frac{1}{l} \sum_{k=1}^{l}\left(\mathcal{F}\left(x_{k}, y^{\star}\right)-\mathcal{F}\left(x^{\star}, y_{k}\right)\right)\right]
\tag{58}
$$

$$
=\frac{1}{l} \sum_{k=1}^{l} \mathbb{E}\left[\mathcal{F}\left(x_{k}, y^{\star}\right)-\mathcal{F}\left(x^{\star}, y_{k}\right)\right]
\tag{59}
$$

$$
=\frac{1}{l} \sum_{k=1}^{l} \mathbb{E}\left[P\left(x_{k}, y_{k}\right)\right].
\tag{60}
$$

Combine Eq. 56 and Eq. 60 to prove

$$
\begin{aligned}
\mathbb{E}\left[P\left(\bar{x}^{l}, \bar{y}^{l}\right)\right] \leq \frac{1}{2 \sqrt{l}}\left(\frac{D_{x}^{2}}{C_{\alpha}}+\frac{D_{y}^{2}}{C_{\beta}}\right) & +\frac{\sqrt{l+1}}{2 l}\left(C_{\alpha} G_{x}^{2}+C_{\alpha} L_{y} G_{x}^{2}+C_{\alpha} L_{y} D_{y}^{2}+C_{\beta} G_{y}^{2}\right) \\
& +\left(L_{x} D_{x}+L D_{x}+2 L_{y} D_{y}\right) \sqrt{d} \Delta.
\end{aligned}
\tag{61}
$$

$\square$

20

## B Proofs for Data-free Pruning

The proof is a simplification of [YKH20] assuming a gradient oracle, i.e., $\nabla \mathcal{F}(x,y)$ can be obtained at any $(x,y)$.

### B.1 Key Lemmas

The following lemmas will be used in the main proofs.

Lemma B.1 (PL indicates EB and QG[KNS16]). Any $l$-smooth function $f(\cdot)$ satisfying PL with constant $\mu$ also satisfies Error Bound (EB) condition with $\mu$, i.e.,
$$\|\nabla f(x)\| \geq \mu\|x^\star - x\|, \forall x,$$
where $x^\star$ is the projection of $x$ onto the optimal set.

Such $f(\cdot)$ also satisfies Quadratic Growth (QG) condition with $\mu$, i.e.,
$$f(x) - f^\star \geq \frac{\mu}{2}\|x^\star - x\|^2, \forall x.$$

It is easy to derive from the EB condition that $l \geq \mu$.

Lemma B.2 (Smoothness and gradient of $h$ [NSH$^+$19]). In the original minimax problem, if $-\mathcal{F}(x,\cdot)$ satisfies PL condition with constant $\mu_2$ for any $x$, and $\mathcal{F}$ is $L$-smooth (Assumption 1), then the function $h(x) := \max_y \mathcal{F}(x,y)$ is $L_h$-smooth with $L_h = L + \frac{L^2}{2\mu_2}$, and $\nabla h(x) = \nabla_x \mathcal{F}(x,y^\star(x))$ for any $y^\star(x) \in \arg\max_y \mathcal{F}(x,y)$.

Also, $h(x)$ satisfies PL condition.

Lemma B.3 ($h$ is $\mu_1$-PL [YKH20]). If $\mathcal{F}(x,y)$ satisfies Assumption 1 and Assumption 3, then function $h(x) := \max_y \mathcal{F}(x,y)$ satisfies the PL condition with $\mu_1$.

### B.2 Main Proofs

We first prove a contraction theorem for each iteration in the noiseless setting.

Theorem B.4 (Contraction of Potential Function). Assume Assumptions 1,2,3 hold for $\mathcal{F}(x,y)$. If we run one iteration of updates in Eq. 15 with $\alpha_k = \alpha \leq 1/L_h$ ($L_h = L + \frac{L^2}{2\mu_2}$ as specified in Lemma B.2) and $\beta_k = \beta \leq 1/L$, then
$$a_{k+1} + \lambda b_{k+1} \leq \max\{\gamma_1, \gamma_2\}(a_k + \lambda b_k), \tag{62}$$
where
$$
\begin{aligned}
\gamma_1 &= 1 - \mu_1\alpha - \lambda\mu_1(1 - \mu_2\beta)\left[\alpha - \left(2\alpha + \alpha^2 L\right)\left(1 + \frac{1}{\varepsilon}\right)\right], \\
\gamma_2 &= 1 - \mu_2\beta + \frac{\alpha L^2}{\lambda\mu_2} + (1 - \mu_2\beta)\frac{L^2}{\mu_2}\left[\left(2\alpha + \alpha^2 L\right)(1 + \varepsilon) + \alpha\right],
\end{aligned} \tag{63}
$$
and $\lambda > 0, \varepsilon > 0$ are constants satisfying
$$\frac{\alpha}{2} + \lambda(1 - \mu_2\beta)\left[\frac{\alpha}{2} - \left(\alpha + \frac{\alpha^2 L}{2}\right)\left(1 + \frac{1}{\varepsilon}\right)\right] \geq 0.$$

21

Proof. We look at $a_{k+1}$ and $b_{k+1}$ separately to derive bounds for the potential function. Since $h(x)$ is $L_h$-smooth by Lemma B.2, we have

$$
\begin{aligned}
a_{k+1}=h\left(x_{k+1}\right)-h^{\star} & \leq h\left(x_{k}\right)-h^{\star}+\left\langle\nabla h\left(x_{k}\right), x_{k+1}-x_{k}\right\rangle+\frac{L_{h}}{2}\left\|x_{k+1}-x_{k}\right\|^{2} \\
& =a_{k}-\alpha\left\langle\nabla h\left(x_{k}\right), \nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)\right\rangle+\frac{L_{h} \alpha^{2}}{2}\left\|\mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2} \\
& \leq a_{k}+\frac{\alpha}{2}\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2}-\frac{\alpha}{2}\left\|\nabla h\left(x_{k}\right)\right\|^{2},
\end{aligned}
\tag{64}
$$

where the second inequality uses the assumption that $\alpha \leq 1 / L_{h}$.

The values of $\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2}$ and $\left\|\nabla h\left(x_{k}\right)\right\|^{2}$ can be bounded by $a_{k}, b_{k}$. With Lemma B.2 and Assumption 1, we have

$$
\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2} \leq\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla_{x} \mathcal{F}\left(x_{k}, y^{\star}\left(x_{k}\right)\right)\right\|^{2} \leq L^{2}\left\|y^{\star}\left(x_{k}\right)-y_{k}\right\|^{2},
$$

for $\forall y^{\star}\left(x_{k}\right) \in \arg \max _{y} \mathcal{F}\left(x_{k}, y\right)$. Because $-\mathcal{F}\left(x_{k}, y\right)$ is $\mu_{2}$-PL in $y$, it has Quadratic Growth as defined in Lemma B.1, so

$$
\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2} \leq L^{2}\left\|y^{\star}\left(x_{k}\right)-y_{k}\right\|^{2} \leq \frac{2 L^{2}}{\mu_{2}}\left[h\left(x_{k}\right)-\mathcal{F}\left(x_{k}, y_{k}\right)\right]=\frac{2 L^{2}}{\mu_{2}} b_{k}.
\tag{65}
$$

For $\left\|\nabla h\left(x_{k}\right)\right\|^{2}$, we know $g(x)$ is $\mu_{1}$-PL from Lemma B.3, so

$$
\left\|\nabla h\left(x_{k}\right)\right\|^{2} \geq 2 \mu_{1}\left[h\left(x_{k}\right)-h^{\star}\right]=2 \mu_{1} a_{k}.
\tag{66}
$$

For $b_{k+1}$, we first prove it is a contraction with respect to $y_{k}$. Specifically,

$$
\begin{aligned}
b_{k+1} & =h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k+1}\right) \\
& \leq h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)-\left\langle\nabla_{y} \mathcal{F}\left(x_{k+1}, y_{k}\right), y_{k+1}-y_{k}\right\rangle+\frac{L}{2}\left\|y_{k+1}-y_{k}\right\|^{2} \\
& =h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)+\left(\frac{L \beta^{2}}{2}-\beta\right)\left\|\nabla_{y} \mathcal{F}\left(x_{k+1}, y_{k}\right)\right\|^{2} \\
& \leq h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)-\mu_{2} \beta\left[h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)\right] \\
& =\left(1-\mu_{2} \beta\right)\left[h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)\right],
\end{aligned}
\tag{67}
$$

where the first inequality comes from the assumption that $\mathcal{F}(x, y)$ is $L$-smooth in $y$, and the second inequality uses the assumptions that $\beta \leq 1 / L$ and $\mathcal{F}(x, y)$ is $\mu_{2}$-PL in $y$. Further, observe that

$$
h\left(x_{k+1}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)=b_{k}+\mathcal{F}\left(x_{k}, y_{k}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right)+h\left(x_{k+1}\right)-h\left(x_{k}\right).
\tag{68}
$$

Because $\mathcal{F}(x, y)$ is $L$-smooth by Assumption 1, we have

$$
\begin{aligned}
\mathcal{F}\left(x_{k}, y_{k}\right)-\mathcal{F}\left(x_{k+1}, y_{k}\right) & \leq-\left\langle\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right), x_{k+1}-x_{k}\right\rangle+\frac{L}{2}\left\|x_{k+1}-x_{k}\right\|^{2} \\
& =\left(\alpha+\frac{\alpha^{2} L}{2}\right)\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2} \\
& \leq\left(\alpha+\frac{\alpha^{2} L}{2}\right)\left[(1+\varepsilon)\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2}\right. \\
& \left.\quad+\left(1+\frac{1}{\varepsilon}\right)\left\|\nabla h\left(x_{k}\right)\right\|^{2}\right],
\end{aligned}
\tag{69}
$$

22

where the second inequality holds according to Young's inequality for any $\varepsilon > 0$. From 64, we know that
$$
h\left(x_{k+1}\right)-h\left(x_{k}\right)=a_{k+1}-a_{k} \leq \frac{\alpha}{2}\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2}-\frac{\alpha}{2}\left\|\nabla h\left(x_{k}\right)\right\|^{2}.\qquad(70)
$$

Combining Eq. 67, 68, 69 and 70 together,
$$
\begin{aligned}
b_{k+1} \leq & \left(1-\mu_{2} \beta\right)\left\{b_{k}+\left[\left(\alpha+\frac{\alpha^{2} L}{2}\right)(1+\varepsilon)+\frac{\alpha}{2}\right]\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2}\right. \\
& \left.-\left[\frac{\alpha}{2}-\left(\alpha+\frac{\alpha^{2} L}{2}\right)\left(1+\frac{1}{\varepsilon}\right)\right]\left\|\nabla h\left(x_{k}\right)\right\|^{2}\right\}.
\end{aligned}
$$

Together with Eq. 65 and 66, we know that
$$
\begin{aligned}
a_{k+1}+\lambda b_{k+1} \leq & a_{k}+\lambda\left(1-\mu_{2} \beta\right) b_{k} \\
+ & \left\{\frac{\alpha}{2}+\lambda\left(1-\mu_{2} \beta\right)\left[\left(\alpha+\frac{\alpha^{2} L}{2}\right)(1+\varepsilon)+\frac{\alpha}{2}\right]\right\}\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla h\left(x_{k}\right)\right\|^{2} \\
- & \left\{\frac{\alpha}{2}+\lambda\left(1-\mu_{2} \beta\right)\left[\frac{\alpha}{2}-\left(\alpha+\frac{\alpha^{2} L}{2}\right)\left(1+\frac{1}{\varepsilon}\right)\right]\right\}\left\|\nabla h\left(x_{k}\right)\right\|^{2} \\
\leq & \left\{1-\mu_{1} \alpha-\lambda \mu_{1}\left(1-\mu_{2} \beta\right)\left[\alpha-\left(2 \alpha+\alpha^{2} L\right)\left(1+\frac{1}{\varepsilon}\right)\right]\right\} a_{k} \\
& +\lambda\left\{1-\mu_{2} \beta+\frac{\alpha L^{2}}{\lambda \mu_{2}}+\left(1-\mu_{2} \beta\right) \frac{L^{2}}{\mu_{2}}\left[\left(2 \alpha+\alpha^{2} L\right)(1+\varepsilon)+\alpha\right]\right\} b_{k}, \\
\leq & \max \left\{\gamma_{1}, \gamma_{2}\right\}\left(a_{k}+\lambda b_{k}\right)
\end{aligned}
$$
where we have defined
$$
\gamma_{1}=1-\mu_{1} \alpha-\lambda \mu_{1}\left(1-\mu_{2} \beta\right)\left[\alpha-\left(2 \alpha+\alpha^{2} L\right)\left(1+\frac{1}{\varepsilon}\right)\right],
$$

$$
\gamma_{2}=1-\mu_{2} \beta+\frac{\alpha L^{2}}{\lambda \mu_{2}}+\left(1-\mu_{2} \beta\right) \frac{L^{2}}{\mu_{2}}\left[\left(2 \alpha+\alpha^{2} L\right)(1+\varepsilon)+\alpha\right],
$$
and the second inequality requires
$$
\frac{\alpha}{2}+\lambda\left(1-\mu_{2} \beta\right)\left[\frac{\alpha}{2}-\left(\alpha+\frac{\alpha^{2} L}{2}\right)\left(1+\frac{1}{\varepsilon}\right)\right] \geq 0.
$$

In addition, the contraction requires both $\gamma_{1}<1$ and $\gamma_{2}<1$.
$\square$

With the results from Theorem B.4, we prove the linear convergence to stationary points for a class of nonconvex-nonconcave objective functions under proper choice of learning rates.

### Proof of Theorem 4.5
Proof. We first prove that with $\alpha=\frac{\mu_{2}^{2}}{18 L^{3}}$ and $\beta=\frac{1}{L}$, the potential function converges as
$$
P_{k} \leq\left(1-\frac{\mu_{1} \mu_{2}^{2}}{36 L^{3}}\right)^{k} P_{0}.\qquad(72)
$$

23

Recall that Theorem B.4 requires $\alpha \leq \frac{1}{L_h} \leq \frac{2}{3L}$ (using the corollary that $L \geq \mu_2$ from the Error Bound of Lemma B.1), and $\beta \leq \frac{1}{L}$. Let $\lambda = \frac{1}{10}$ and $\varepsilon = 1$ in Theorem B.4. We have

$$
\begin{aligned}
\gamma_{1} & =1-\mu_{1} \alpha\left\{1+\lambda\left(1-\mu_{2} \beta\right)\left[1-(2+\alpha L)\left(1+\frac{1}{\varepsilon}\right)\right]\right\} \\
& \leq 1-\mu_{1} \alpha\left[1+\frac{1}{10}\left(1-\mu_{2} \beta\right)(1-6)\right] \\
& \leq 1-\frac{1}{2} \mu_{1} \alpha,
\end{aligned}
\tag{73}
$$

where the first inequality plugs in the values of $\lambda, \varepsilon$ and uses the fact that $\alpha \leq \frac{2}{3 L} \leq \frac{1}{L}$. With an additional assumption that $\frac{\mu_{2}^{2} \beta}{\alpha L^{2}} \geq \frac{52}{3}$ (which is satisfied when $\alpha=\frac{\mu_{2}^{2}}{18 L^{3}}$ and $\beta=\frac{1}{L}$),

$$
\begin{aligned}
\gamma_{2} & =1-\frac{\alpha L^{2}}{\mu_{2}}\left\{\frac{\mu_{2}^{2} \beta}{\alpha L^{2}}-\frac{1}{\lambda}-\left(1-\mu_{2} \beta\right)[(2+\alpha L)(1+\varepsilon)+1]\right\} \\
& \leq 1-\frac{\alpha L^{2}}{\mu_{2}}\left[\frac{\mu_{2}^{2} \beta}{\alpha L^{2}}-10-\frac{19}{3}\left(1-\mu_{2} \beta\right)\right] \\
& \leq 1-\frac{\alpha L^{2}}{\mu_{2}},
\end{aligned}
\tag{74}
$$

where the first inequality plugs in the values of $\lambda$ and $\varepsilon$, and uses the fact that $\alpha \leq \frac{2}{3 L}$. Again, using the corollary from EB of Lemma B.1, we know that $\frac{\mu_{1} \mu_{2}}{2 L^{2}}<1$, therefore $\frac{1}{2} \mu_{1} \alpha<\frac{\alpha L^{2}}{\mu_{2}}$ and $\gamma_{1}>\gamma_{2}$. Plug in the value of $\alpha$ and $\gamma_{1}$, we reach the conclusion of Eq. 72.

Finally, we prove the convergence rate of $\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2}+\left\|\nabla_{y} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2}$ by upper bounding it with the potential function, which is similar to the proof of [YKH20].

First,

$$
\begin{aligned}
\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2} & \leq\left\|\nabla h\left(x_{k}\right)\right\|^{2}+\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla g\left(x_{k}\right)\right\|^{2} \\
& =\left\|\nabla h\left(x_{k}\right)-\nabla h\left(x^{\star}\right)\right\|^{2}+\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla g\left(x_{k}\right)\right\|^{2} \\
& \leq L_{h}^{2}\left\|x_{k}-x^{\star}\right\|^{2}+L^{2}\left\|y^{\star}\left(x_{k}\right)-y_{k}\right\|^{2} \\
& \leq \frac{2 L_{h}^{2}}{\mu_{1}} a_{k}+\frac{2 L^{2}}{\mu_{2}} b_{k},
\end{aligned}
\tag{75}
$$

where the the second inequality are based on Lemma B.2, and the last inequality is based on Lemma B.3 and the Quadratic Growth property in B.1.

Second,

$$
\begin{aligned}
\left\|\nabla_{y} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2} & \leq\left\|\nabla_{y} \mathcal{F}\left(x_{k}, y_{k}\right)-\nabla_{y} \mathcal{F}\left(x_{k}, y^{\star}\left(x_{k}\right)\right)\right\|^{2} \\
& \leq L^{2}\left\|y_{k}-y^{\star}\left(x_{k}\right)\right\|^{2} \\
& \leq \frac{2 L^{2}}{\mu_{2}} b_{k},
\end{aligned}
\tag{76}
$$


where the last inequality comes from the Quadratic Growth property for $\mathcal{F}(x_k, \cdot)$. As a result,

$$
\begin{aligned}
\left\|\nabla_{x} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2}+\left\|\nabla_{y} \mathcal{F}\left(x_{k}, y_{k}\right)\right\|^{2} & \leq \frac{2 L_{h}^{2}}{\mu_{1}} a_{k}+\frac{4 L^{2}}{\mu_{2}} b_{k} \\
& \leq \max \left\{\frac{2 L_{h}^{2}}{\mu_{1}}, \frac{40 L^{2}}{\mu_{2}}\right\}\left(a_{k}+\frac{1}{10} b_{k}\right) \\
& \leq \max \left\{\frac{2 L_{h}^{2}}{\mu_{1}}, \frac{40 L^{2}}{\mu_{2}}\right\}\left(1-\frac{\mu_{1} \mu_{2}^{2}}{36 L^{3}}\right)^{k} P_{0}.
\end{aligned} \tag{77}
$$