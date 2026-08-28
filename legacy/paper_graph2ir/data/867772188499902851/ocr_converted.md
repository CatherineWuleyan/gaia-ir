# Can Pruning Improve Certified Robustness of Neural Networks?

［#1］
Zhangheng Li
UT Austin
Austin, Texas, USA
zoharli@utexas.edu

［#2］
Tianlong Chen
UT Austin
Austin, Texas, USA
tianlong.chen@utexas.edu

［#3］
Linyi Li
UIUC
Urbana, Illinois, USA
linyi2@illinois.edu

［#4］
Bo Li
UIUC
Urbana, Illinois, USA
lbo@illinois.edu

［#5］
Zhangyang Wang
UT Austin
Austin, Texas, USA
atlaswang@utexas.edu

［#6］
**Abstract**—With the rapid development of deep learning, the sizes of neural networks become larger and larger so that the training and inference often overwhelm the hardware resources. Given the fact that neural networks are often over-parameterized, one effective way to reduce such computational overhead is neural network pruning, by removing redundant parameters from trained neural networks. It has been recently observed that pruning can not only reduce computational overhead but also can improve empirical robustness of deep neural networks (NNs), potentially owing to removing spurious correlations while preserving the predictive accuracies. This paper for the first time demonstrates that pruning can generally improve certified robustness for ReLU-based NNs under the *complete verification* setting. Using the popular Branch-and-Bound (BaB) framework, we find that pruning can enhance the estimated bound tightness of certified robustness verification, by alleviating linear relaxation and sub-domain split problems. We empirically verify our findings with off-the-shelf pruning methods and further present a new stability-based pruning method tailored for reducing neuron instability, that outperforms existing pruning methods in enhancing certified robustness. Our experiments show that by appropriately pruning an NN, its certified accuracy can be boosted up to 8.2% under standard training, and up to 24.5% under adversarial training on the CIFAR10 dataset. We additionally observe the existence of *certified lottery tickets* that can match both standard and certified robust accuracies of the original dense models across different datasets. Our findings offer a new angle to study the intriguing interaction between sparsity and robustness, i.e. interpreting the interaction of sparsity and certified robustness via neuron stability. Codes are available at: github.com/VITA-Group/CertifiedPruning.

## I. INTRODUCTION
［#7］
Neural Network (NN)-based framework is a strong general solution to many problems, yet many of these solutions remain impractical for real-world applications of low fault tolerance. A slight perturbation in the raw input sensory data could completely change the predicting behaviors of the networks. Furthermore, researchers have shown that various kinds of targeted adversarial attacks can easily fool the neural networks [32, 10], which poses threat to many deep learning applications. Fortunately, researchers introduced formal methods to verify neural network behaviors, which made it possible to mathematically derive the prediction bound of a neural network w.r.t. a certain input, and thus evaluate the certified robustness of the neural network. For example, in the image classification task, given an input image with some perturbation, if the lower bound of the output probability of the correct label is higher than the upper bound of probabilities of other incorrect labels, we say the model is certifiably robust w.r.t. this image sample. The goal of neural network verification is to estimate the ground-truth bound as close as possible.

［#8］
In this paper, we are concerned about the certified robustness under the *complete verification* setting, where the verifier should output the *exact* bounds given the input domain $\mathcal{C}$, rather than some relaxation of $\mathcal{C}$, given sufficient time. Despite its theoretical appeal, the complete verification of neural networks is known to be a challenging NP-hard problem [16, 37], mainly due to the non-linear activation functions in neural networks, such as Sigmoid and ReLU. The popular *Branch-and-Bound (BaB)* framework [4] utilized the feature of ReLU activations and adopted the classical divide-and-conquer method to solve the complete verification problem. It branches the bound computation into multiple sub-domains recursively on ReLU nodes and computes the bounds on each sub-domain respectively. The time complexity of this framework is exponential, and typically has pre-set time limit for each sample verification.

［#9］
Several verifiers [43, 35] based on the BaB framework were later proposed for efficient complete verification. The core problem addressed in these methods is how to estimate the pre-activation bound (i.e. propagated input bounds of the non-linear activation layers) as tight as possible given limited time. To approach this, they use Linear-Relaxation based Perturbation Analysis (LiRPA) to relax non-linear bound propagation with linear ones and use GPU-accelerated BaB methods to further tighten the estimated bounds as much as possible. However, the estimated bound is still loose, mainly because (1) an efficient linear relaxation of multiple non-linear activation layers is loose both empirically [28] and theoretically [16, 37], where tightening the relaxation requires an exponential number of linear constraints which is inefficient [33]; and (2) the BaB framework requires solving an exceedingly large number of sub-domains (which is exponential in the worst case [16]) to provide a tight bound, so in practice we often solve only a part of the sub-domains which yields a loose bound.

［#10］
Recent efforts [9, 11, 13, 44, 15, 41] reveal that proper network pruning can empirically enhance neural network robustness to adversarial attacks. We take one step further to argue that pruning can also be utilized to improve the estimated bound tightness of certified robustness verification, by alleviating linear relaxation and sub-domain split problems. Improving empirical robustness (as a surrogate of "ground-truth" robustness) and verification tightness can together lead to overall measurable certified robustness, and we find that existing pruning schemes

［#10］
can already co-achieve both. Moreover, inspired by that sparsity can eliminate unstable neurons and improve non-linear neuron stability for verification [41], we present a new stability-based pruning method, that even outperforms existing pruning methods on improving certified robustness. As one last "hidden gem" finding, we demonstrate the existence of certified lottery tickets, that generalizes the lottery ticket hypothesis [8] to the certified robustness field for the first time: it is defined as those sparse subnetworks found by pruning that can match both the standard and certified accuracies of the original dense models.
Our contributions are outlined below:

［#11］
- For the first time, we demonstrate that pruning can generally improve certified robustness. We analyze pruning effects from the perspectives of both improving ground-truth robustness of the model and the verification bound tightness, and empirically validate it with extensive pruning methods and training schemes.
- As pruning can be utilized to improve non-linear neuron stability, we propose a novel regularizer for pruning called NRSLoss (see Figure 2) that effectively regularizes the neuron stability and outperforms existing pruning methods in enhancing certified robustness.
- Our experiments validate the above proposals by presenting verification results under various settings. For example on the CIFAR10 dataset, under certified training, existing pruning methods as well as our proposed NRSLoss-based pruning boost the certified accuracy for 1.6-7.1% and 8.2% respectively; under adversarial training, they boost the certified accuracy for 12.5-24.5%.
- We additionally observe the existence of certified lottery tickets that can match both standard and certified robust accuracies of the original dense models, using either the classical iterative magnitude pruning (IMP) [8] or NRSLoss-based pruning.

## II. RELATED WORK

### A. Incomplete and complete NN verification

［#12］
Neural Network verification is a critical issue for developing trustworthy and safe AI. Existing verifiers can be divided into either incomplete or complete verifiers. Complete verifiers can typically produce tighter bound than incomplete verifiers, but consume much larger computational resources than incomplete verifiers. Typical incomplete verifiers are based on duality [5, 26] and linear approximations [37, 38, 46], whereas existing complete verifiers are based on satisfiability modulo theories (SMT) [16, 6], mixed integer programming (MIP) [34], convex hull approximatio [24], or Branch-and-Bound (BaB) [4, 43, 35].

［#13］
Traditional complete verifiers such as SMT and MIP are computationally expensive and hard to parallelize. To this end, a series of verifiers based on the BaB framework were recently proposed for efficient and parallelizable complete verification. Auto-LiPRA [42] was an early incomplete verifier and certified trainer, that relaxes ReLU non-linearity with a tight linear relaxation. Following that, Fast-and-Complete algorithm [43] proposed to combine auto-LiRPA with BaB for tighter bound estimation and use LP solver for completeness check, which is a GPU-parallelizable complete verification method. Beta-CROWN [35] further extended Fast-and-Complete algorithm by replacing the LP completeness check with optimizable constraints based on the Lagrange function, and improved the verification efficiency.

### B. Neural network pruning

［#14］
Pruning removes the redundant structures in NNs and reduces the size of parameter numbers from the computation graph of NNs. It not only constitutes an important class of NN model compression methods but also can act as a regularizer for NN training which can improve the performance w.r.t. original unpruned networks. The pruning process can be conducted at different levels or granularities, such as parameter-level [8, 47, 23], filter-level [22, 20, 27] and layer-wise [36, 39, 45]. Especially, Lottery Ticket Hypothesis (LTH) [8] claims the existence of independently trainable sparse subnetworks that can match or even surpass the performance of dense networks. Such sparse subnetworks called "winning tickets", can be obtained by simple iterative parameter-level pruning.

### C. Pruning and robustness

［#15］
Recently, several works [44, 11, 15] have revealed that proper network pruning can empirically improve the robustness of a trained NN, potentially due to removing spurious correlations while preserving the predictive accuracies. [9] found that randomly initialized robust subnetworks with better adversarial accuracy than dense model counterparts can be found by IMP. [41] was the first work to inject sparsity during NN training, with the primary goal to speed up certified verification. It considered only weight magnitude pruning, and did not generally demonstrate sparsity to improve the achievable certified robustness. HYDRA [29] incorporated the robustness loss as a pruning objective, and showed such a robustness-aware pruning scheme can lead to high NN sparsity without much robust accuracy loss. Besides studying incomplete certified verification, HYDRA did not specifically analyze what benefits pruning brings for verification, while our NRSLoss-based pruning is explicitly motivated by reducing unstable neurons and tightening the estimated bound in the complete verification. [12] proposed that superficial neurons that contribute significantly to the feature map values in shallow layers were highly localized and are more prune toadversarial patches. Hence they used pruning to remove superficial neurons and improved certified defense against adversarial patches - an orthogonal goal to our work.

## III. METHODOLOGY

［#16］
In this paper, following prior works on certified robustness, we focus on non-linear feed-forward neural networks with ReLU activations. In this section, we first analyze what benefits would network pruning brings to certified robustness and verification, and then introduce the specific pruning methods we test in our experiments.

### A. Preliminary

［#17］
1) Unstructured and Structured Pruning: Network pruning is one of the most effective model compression paradigms for deep neural network, by removing redundant parameters or

［#18］
![](./images/867772188499902851_1.jpg)

［#19］
Fig. 1: (Figure source from [18].) Demonstration of typical unstructured pruning and structured pruning paradigms for a simple Multi-Layer-Perceptron (MLP). Unstructured pruning removes individual weight parameters, while structured pruning removes all input and output weights associated with certain channels.

［#20］
neurons from over-parameterized neural networks, and many of them focus on pruning learnable weight parameters. Existing weight pruning methods can be divided to unstructured pruning and structured pruning, depending on whether weights are pruned individually or by group. An example of unstructured pruning v.s. structured pruning for a simple network composed of multiple fully-connected layers is shown in Figure 1. For unstructured pruning, individual weights that connect two channels (neurons) of adjacent layers are removed; for structured pruning, all input and output weights associated with certain channels are removed. Another typical case is pruning for convolutional neural networks, unstructured pruning usually removes individual weight elements of the convolutional kernels, whereas structured pruning removes all input and output kernels associated with certain channels.

［#21］
Each of these two pruning paradigms has its advantages and disadvantages over the other. Unstructured pruning can better preserve the performance of the original dense networks due to pruning flexibility on individual weights, but is hard to realize real hardware acceleration during inference. In contrast, the hardware compression during inference for structured pruning can be easily implemented due to the removal of entire channels, but has less pruning flexibility compared to unstructured pruning, which would generally lead to worse performance that unstructured pruning. In this paper, we investigate the influence of both unstructured and structured pruning on certified robustness and are interested in both the performance gain and the reduction of computational overhead brought by pruning.

［#22］
2) Lottery Ticket Hypothesis(LTH): The lottery ticket hy- pothesis [8] states that a randomly initialized dense neural network contains at least one subnetwork (i.e. by pruning the parameters of the dense network) that has the same initialization of the unpruned parameters and can match the test performance of the dense network after training for at most the same iterations as the dense network, and such subnetworks are called the winning tickets of the dense network. To find these winning tickets, they propose **Iterative Magnitude Pruning(IMP)** algorithm: Firstly, start from a dense initialization $W_0$, and then train the network until convergence to weight $W_t$. Then we determine the $\rho$ percent smallest magnitude weights in $|W_t|$ and create a binary mask $m_0$ that prunes these. Then retrain the pruned network from the same initialization weight $W_0 \odot m_0$ to convergence. Iterating this procedure will produce subnetworks with different sparsity, among which certain subnetworks can match the test performance of the original dense network, i.e. the winning tickets.

［#23］
3) ReLU Neuron Stability: The illustration of ReLU neuron stability is demonstrated in Figure 3. The ReLU activation function is zero when input value is less than 0, and an identity function when input value is no less than 0. As shown in the Figure 3, $\mathbf{h}_j^{(i)}$ means the pre-activation value of $j$th ReLU neuron at $i$th layer of the network. $\mathbf{g}_j^{(i)}$ means the corresponding value after passing the ReLU neuron. $\mathbf{l}_j^{(i)}$ and $\mathbf{u}_j^{(i)}$ refers to the lower bound and upper bound of the pre-activation $\mathbf{u}_j^{(i)}$ w.r.t. certain input perturbation. Fig. 3(a) and Fig. 3(d) are unstable neurons where $\mathbf{l}_j^{(i)}$ and $\mathbf{u}_j^{(i)}$ has different signs, while Fig. 3(b) and Fig. 3(c) are stable neurons where $\mathbf{l}_j^{(i)}$ and $\mathbf{u}_j^{(i)}$ has the same sign. The yellow areas in Fig. 3(a) and Fig. 3(d) refer to the bounded area of "triangle" relaxation and linear relaxation, respectively.

### B. What factors influence certified robustness?

［#24］
Up to now, with state-of-the-art robust training method and certified verifier, the measurable bound of a trained non-linear neural network is influenced by two major factors:

［#25］
1) The ground-truth bound of the network: This is mainly decided by the training method. For example, certified train- ing usually provides much higher certified robustness than adversarial training, which is demonstrated in our experiments. In this paper, we ideally hope pruning has positive or no influences on the normal training process, e.g. the influences of pruning-related regularizer to normal gradient back-propagation. Besides, the model size, i.e. the parameter number also matters, which is important in this paper since we can use pruning to reduce parameter number. Intuitively, with more parameter numbers, the bound of the network output tends to be looser.

［#26］
![](./images/867772188499902851_2.jpg)
［#27］
(a) RSLoss

［#28］
![](./images/867772188499902851_3.jpg)
［#29］
(b) NRSLoss

［#30］
![](./images/867772188499902851_4.jpg)
［#31］
(c) NRSLoss($\gamma=0.5$)

［#32］
![](./images/867772188499902851_5.jpg)
［#33］
(d) NRSLoss($\gamma=1$) (i.e. RSLoss)

［#34］
![](./images/867772188499902851_6.jpg)
［#35］
(e) NRSLoss($\gamma=2$)

［#36］
![](./images/867772188499902851_7.jpg)
［#37］
(f) NRSLoss($\gamma=4$)

［#38］
Fig. 2: (a)(b) The landscape of RSLoss and NRSLoss with varied stability and BN channel weight $\gamma$. *Stability* means the stability of a ReLU neuron, i.e. the pre-activation lower bound times upper bound. $\gamma$ is the corresponding channel weight of Batch-Normalization layer, whose magnitude denotes the importance of each channel. The NRSLoss is high when the neuron is unstable and the corresponding channel has low importance. (c)-(f) The sample landscape of NRSLoss with varied lower and upper pre-activation bounds given different fixed $\gamma$. With the growth of $\gamma$ that implies channel(neuron) importance, the NRSLoss gets increasingly suppressed.

［#39］
![](./images/867772188499902851_8.jpg)

［#40］
Fig. 3: (Figure source from [43])) Illustration of the stability of ReLU neuron and linear relaxation.

［#41］
With network pruning, we can heuristically tighten the bound of the network since the network has fewer parameters.

［#42］
2) *Estimated bound tightness of the verifier*: This refers to the closeness of the estimated bound of the verification to the ground truth bound of the network, and it reflects the performance of the verifier. Since existing certified verifiers usually have a very large computational overhead to verify even a single sample, in practice, we are concerned about the bound tightness a verifier can reach given a limited verification time.

［#43］
Although efficient certified verification for non-linear neural networks has made rapid progress in recent years, the bound tightness, running speed and affordable network capacity on limited hardware resources still have huge space awaiting to be improved. Specifically, we identify two major problems that influence the bound tightness:

［#44］
- **Neuron Stability and Verification Speed**. As mentioned in Section 1, BaB is the main-stream framework of existing certified verification methods. The bound tightness is largely influenced by how many unstable neurons have been branched given a limited time. (Please refer to Section III-A3 for the explanation of neuron stability). The verifier would need to visit more sub-domains by branching on unstable neurons within

［#44］
limited time, and thus the verification speed matters.

［#45］
- **Linear Relaxation.** To facilitate bound computation, many recent verifiers [35, 43, 38, 46, 31], utilize the linear relaxation method for unstable ReLU neurons, as we illustrated in Section III-A3. Normally, if an unstable neuron has not been branched by BaB, then this neuron gets linearly relaxed during bound propagation, which will also loosen the bound tightness of the verifier.

### C. What benefits for certified robustness can we expect from pruning?
［#46］
Network pruning can bring many benefits to problems as mentioned in Section III-B, outlined as follows:

［#47］
- **Reducing parameter number.** It can tighten bound propagation directly. Both structured and unstructured pruning can reduce parameter number and the bound propagation becomes tighter after pruning. Take the widely used bound propagation method for linear layers(e.g. convolution and full-connected layer)——Interval Bound Propagation (IBP) as an example, its computation can be formulated as follows:

［#47］
$$
\begin{aligned}
\mu_{i-1} &=\frac{\bar{z}_{i-1}+\underline{z}_{i-1}}{2} \\
r_{k-1} &=\frac{\bar{z}_{i-1}-\underline{z}_{i-1}}{2} \\
\mu_{k} &=\mathbf{W} \mu_{k-1}+b \\
r_{k} &=|\mathbf{W}| r_{k-1} \\
\underline{z}_{i} &=\mu_{k}-r_{k} \\
\bar{z}_{i} &=\mu_{k}+r_{k}
\end{aligned} \quad(1)
$$

［#48］
Eq. 1 computes the linear bound propagation for $i$th linear layer of the network, where $\bar{z}_{i-1}$ and $\underline{z}_{i-1}$ are the input lower and upper bounds of $i$th linear layer, and $\bar{z}_{i}$ and $\underline{z}_{i}$ the corresponding output lower and upper bounds, respectively. $\mathbf{W}$ and $b$ denote the weight and bias of the linear layer. The difference between output upper and lower bound equals $2|\mathbf{W}| r_{k-1}$. By network pruning, the weight matrix $\mathbf{W}$ becomes sparser, and thus the difference between output upper and lower bound tends to become smaller, and thus the overall output bound of the network would be tightened.

［#49］
- **Reducing unstable neurons.** By reducing the number of unstable neurons, we can reduce the number of linear relaxations and sub-domain splits needed by the verification process, which can also directly tighten the bound and accelerate the verification process as well. However, for most existing pruning methods, reducing unstable neurons is not an explicitly designated goal, but rather a possible side effect.

［#50］
- **Real hardware acceleration with structural sparsity.** If adopting structured pruning, we can eliminate channels which will concretely reduce the network width on the hardware implementation level. This can accelerate verification and even make resource-intensive verification possible on larger models.

［#51］
Those possible benefits are further entangled with each other. For empirical evaluation of these benefits, we simply follow the classical criteria: to evaluate the certified accuracy, time and memory consumption, and network width/depth that can be verified after pruning.

### D. Pruning methods
［#52］
In this section, we test a range of off-the-shelf pruning methods for improving certified robustness. For each pruning method, unless otherwise mentioned, we combine it with iterative pruning with weight rewinding [8], as we find iterative pruning with weight rewinding generally enhances performance compared to finetune-based pruning or one-shot pruning in our experiments.

#### 1) Existing pruning methods:
［#53］
In unstructured pruning, for simplicity, we only prune the weights of convolutional layers and ignore linear layers. We pick several representative methods including:
1) **Random Pruning**: pruning weights randomly, which is used for sanity check in our experiments.
2) Lottery ticket hypothesis (LTH), or denoted as **IMP** [8]: pruning weights with the smallest magnitudes, the most standard pruning scheme.
3) **SNIP** [19]: pruning weights with least loss sensitivity w.r.t. percentile magnitude change.
4) **TaylorPruning (TP)** [7]: saliency-based pruning via a first-order Taylor approximation.
5) **HYDRA** [29]: learnable mask-based pruning that minimizes the robustness loss empirically.

［#54］
For structured pruning, we choose two methods:
1) **StructLTH** [2]: a structured variant of LTH recently proposed, by using IMP first then ranking channels by their total magnitudes in remaining weights from high-to-low. Then we prune lowest-ranked channels and refill the IMP-pruned weights in the remaining channels.
2) **Network Slimming** [21]: for batch normalization (BN) [14] layers, we have:

［#54］
$$
y=\frac{x-E[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta \tag{2}
$$

［#55］
Network slimming enforces the L1-norm regularizer on $\gamma$ and prune channels with the smallest $\gamma$ magnitudes.

#### 2) Stability-based Pruning:
［#56］
In the context of certified robustness, we wish to eliminate unstable neurons as much as possible, such that the estimated bound tightness of the verification can be improved. Pruning is a natural choice to accomplish this goal. Next, we first introduce a criterion that measures the degree of neuron stability of a given network, and then introduce an effective stability-based regularizer and corresponding unstructured and structured pruning methods.

［#57］
Formally, we denote $[\mathbf{l}_{j}^{(i)}, \mathbf{u}_{j}^{(i)}]$ as **bound interval** of the pre-activation value of $j$-th ReLU neuron at $i$-th layer. We want to measure not only the number of unstable ReLU neurons with bound intervals crossing the zero point but also to what extent the instability is. To address this problem, we propose to use $-\mathbf{l}_{j}^{(i)} \cdot \mathbf{u}_{j}^{(i)}$ to measure the degree of instability of this neuron. It is easy to know that if we keep the bound interval width $\mathbf{u}_{j}^{(i)}-\mathbf{l}_{j}^{(i)}$ unchanged, then $-\mathbf{l}_{j}^{(i)} \cdot \mathbf{u}_{j}^{(i)}$ reaches maximum when $\mathbf{l}_{j}^{(i)}=-\mathbf{u}_{j}^{(i)}$, which means maximal instability given the same bound interval width. And thus, we simply use the average of this criterion to denote the total degree of instability of the network:


［#57］
$$
instability = \sum_{i} \sum_{j} -\mathbf{l}_{j}^{(i)} \cdot \mathbf{u}_{j}^{(i)} \tag{3}
$$

［#58］
Similar to this criterion, [41] proposed a regularizer named RS Loss to regularize ReLU stability and improved certified robustness. The RS Loss is defined as:

［#58］
$$
l_{j}^{r s}=-\tanh \left(1+\mathbf{l}_{j} \cdot \mathbf{u}_{j}\right) \tag{4}
$$

［#59］
This loss can be naturally used to optimize the instability criterion, as the tanh wrapper provides smooth gradients. However, we empirically find that for deep networks, batch-normalization (BN) [14] layers, which are placed before ReLU layers, are necessary for the training convergence of pruned subnetworks. In this way, the performance of the RS Loss regularizer is insignificant, because it passes gradients to $\gamma$ and affects the training process, and the optimization space is relatively small due to the BN constraint. However, we find the pre-BN bounds (i.e. the input bounds of BN layers) to be very flexible. Thus, instead of regularizing the pre-activation bounds using RS Loss, we propose an alternative of Normalized RS Loss (NRSLoss) to directly regularize the pre-BN bounds, normalized from pre-activation bounds ($\gamma$ is the BN weight of the corresponding channel):

［#59］
$$
l_{j}^{n r s}=-\tanh \left(1+\frac{\mathbf{l}_{j} \cdot \mathbf{u}_{j}}{\gamma^{2}}\right). \tag{5}
$$

［#60］
Since the magnitudes of $\mathbf{l}_{j}$ and $\mathbf{u}_{j}$ are scaled from pre-BN bounds by the factor of $\gamma$, NRSLoss essentially computes RS Loss on the pre-BN bounds $\mathbf{l}_{j}/\gamma$ and $\mathbf{u}_{j}/\gamma$. During training, the NRSLoss is combined with the original loss with an empirical coefficient. Note that we stop the gradients back-propagated from NRSLoss to $\gamma$, to ensure stable training, especially for pruned subnetworks.

［#61］
The loss landscape of NRSLoss is shown in Figure 2. Note that the stability term is essentially $0-instability$. From the loss landscape, we can interpret NRSLoss from another perspective: it penalizes neurons with high instability and low channel importance; when the instability increases, it takes larger channel importance to suppress NRSLoss.

［#62］
We empirically find that combining NRSLoss as a training regularizer with pruning weights based on the least weight magnitude criterion is most effective. We call this pruning scheme IMP+NRSLoss, which we interpret as follows: the model is trained based on a weighted combination of robustness loss and NRSLoss, and the magnitude of each trained individual weight reflects its saliency w.r.t. both robustness loss and NRSLoss, i.e. saliency w.r.t. robustness and stability, and by pruning weights with minimal magnitude-based saliency of robustness and stability, we can minimize the negative effects on robustness and stability brought by pruning, and benefit from positive effects of pruning to certified robustness as introduced above.

［#63］
We stress the two-fold novelty of NRSLoss as follows:
- It takes into account both neuron importance and stability (see Figure 2). In NRSLoss, $\gamma$ is the corresponding channel weight of the Batch-Normalization layer, whose magnitude denotes the importance of each channel. The NRSLoss is high when the neuron is unstable and the corresponding channel has low importance. In contrast, the RSLoss is irrelevant to channel importance, which might lead to imposing too much regularization on important neurons.
- It also disentangles the influence of stability regulariza- tion with the BN layers, via eliminating the magnitude scaling effect to the pre-activation bounds brought by the channel weight $\gamma$. In this way, BN layers can still be learned normally to control the gradients.

## IV. EXPERIMENTS

［#64］
In this section, we evaluate all introduced pruning methods with different training schemes and perturbation scales, and try to address three major questions: (1) Can existing pruning methods improve certified robustness generally? (2) How can NRSLoss-based pruning improve certified robustness? (3) Can we find certified lottery tickets, i.e., sparse subnetworks after pruning that can restore not only the original performance but also certified robustness?

［#65］
Based on our experiment findings, we also provide more ablation studies to further rationalize our claims. Finally, we briefly summarize our experimental findings with several interesting takeaways.

### A. Experiment Setup
［#66］
1) Dataset and Network architecture: Across our experiments, we use FashionMNIST [40], SVHN [25], and CIFAR10 [17] as the benchmark datasets. We introduce them as follows:
- **FashionMNIST**: FashionMNIST is an MNIST-like greyscale image classification dataset by replacing hand-written digits with fashion items, which are more difficult to classify. It has a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 greyscale image, associated with a label from 10 classes. We use the first 200 samples from the testing dataset for verification.
- **SVHN**: SVHN is a dataset consisting of Street View House Number images, with each image consisting of a single cropped digit labeled from 0 to 9. Each example is a 32x32 RGB image. We use the first 200 samples from the testing dataset for verification.
- **CIFAR10**: CIFAR10 is a dataset consisting of 10 object classes in the wild, and each class has 6000 samples. This dataset is commonly used in prior works in complete verification, following [35], we choose the ERAN test set [31] which consists of 1000 images from the CIFAR10 test set. Note that we only use the first 200 samples in the ERAN test set for verification efficiency.

［#67］
We use a 7-layer feed-forward convolutional neural network as the benchmark model, whose architecture is shown in Table III and Figure 5. The design follows the cifar10-model-deep setting in [35], but is wider, deeper, and has BN layers. This model is the largest network that can be fitted in a GPU with 24GB memory for complete verification.


［#68］
<table>
  <thead>
    <tr>
      <th rowspan="3">Dataset</th>
      <th colspan="2">Training Method</th>
      <th colspan="4">FGSM</th>
      <th></th>
      <th colspan="3">AUTO-LiRPA</th>
    </tr>
    <tr>
      <th>Pruning
Method</th>
      <th>Remain
Ratio</th>
      <th>std</th>
      <th>adv</th>
      <th>ver</th>
      <th>t</th>
      <th>Remain
Ratio</th>
      <th>std</th>
      <th>ver</th>
      <th>t</th>
    </tr>
    <tr>
      <th>Dense</th>
      <th>1</th>
      <th>85.2</th>
      <th>81.2</th>
      <th>1.5</th>
      <th>298.3</th>
      <th>1</th>
      <th>77.2</th>
      <th>68.8</th>
      <th>7.23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9">FashionMNIST
$\epsilon = 0.1$</th>
      <th>IMP</th>
      <th>0.03</th>
      <td>80.2</td>
      <td>75.3</td>
      <td>39.0(+37.5)</td>
      <td>85.9</td>
      <td>0.07</td>
      <td>78.1(+0.9)</td>
      <td>73.5(+4.7)</td>
      <td>7.17</td>
    </tr>
    <tr>
      <th>SNIP</th>
      <th>0.03</th>
      <td>81.2</td>
      <td>77.3</td>
      <td>36.5(+35.0)</td>
      <td>96.3</td>
      <td>0.80</td>
      <td>77.5(+0.3)</td>
      <td>71.3(+2.5)</td>
      <td>4.82</td>
    </tr>
    <tr>
      <th>TP</th>
      <th>0.03</th>
      <td>80.3</td>
      <td>76.1</td>
      <td>35.3(+33.8)</td>
      <td>92.3</td>
      <td>0.32</td>
      <td>79.4(+2.2)</td>
      <td>73.5(+4.7)</td>
      <td>6.60</td>
    </tr>
    <tr>
      <th>HYDRA</th>
      <th>0.03</th>
      <td>81.5</td>
      <td>77.2</td>
      <td>33.5(+32.0)</td>
      <td>105.9</td>
      <td>0.51</td>
      <td>79.1(+1.9)</td>
      <td>73.3(+4.5)</td>
      <td>6.60</td>
    </tr>
    <tr>
      <th>HYDRA+NRSLoss</th>
      <th>0.03</th>
      <td>80.3</td>
      <td>76.7</td>
      <td>36.5(+39.5)</td>
      <td>105.3</td>
      <td>0.03</td>
      <td>81.5(+4.3)</td>
      <td>74.2(+5.4)</td>
      <td>8.90</td>
    </tr>
    <tr>
      <th>IMP+RSLoss</th>
      <th>0.03</th>
      <td>79.3</td>
      <td>72.2</td>
      <td>37.0(+35.5)</td>
      <td>94.1</td>
      <td>0.41</td>
      <td>80.5(+3.3)</td>
      <td>72.5(+3.7)</td>
      <td>6.21</td>
    </tr>
    <tr>
      <th>IMP+NRSLoss</th>
      <th>0.03</th>
      <td>81.2</td>
      <td>74.9</td>
      <td>41.5(+40.0)</td>
      <td>78.4</td>
      <td>0.05</td>
      <td>80.5(+3.3)</td>
      <td>74.0(+5.2)</td>
      <td>6.06</td>
    </tr>
    <tr>
      <th>StructLTH[1]</th>
      <th>0.35</th>
      <td>80.5</td>
      <td>78.3</td>
      <td>22.5(+21.0)</td>
      <td>143.0</td>
      <td>0.80</td>
      <td>80.0(+2.8)</td>
      <td>72.4(+3.6)</td>
      <td>6.14</td>
    </tr>
    <tr>
      <th>Slim</th>
      <th>0.35</th>
      <td>80.7</td>
      <td>78.3</td>
      <td>31.0(+29.5)</td>
      <td>105.0</td>
      <td>0.32</td>
      <td>78.5(+1.3)</td>
      <td>71.8(+3.0)</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th rowspan="9">SVHN
$\epsilon = 2/255$</th>
      <th>Dense</th>
      <th>1</th>
      <td>94.8</td>
      <td>89.2</td>
      <td>2.0</td>
      <td>294.2</td>
      <td>1</td>
      <td>76.3</td>
      <td>62.0</td>
      <td>6.68</td>
    </tr>
    <tr>
      <th>IMP</th>
      <th>0.03</th>
      <td>92.3</td>
      <td>84.4</td>
      <td>31.0(+29.0)</td>
      <td>188.3</td>
      <td>0.16</td>
      <td>82.0(+5.7)</td>
      <td>67.3(+5.3)</td>
      <td>6.65</td>
    </tr>
    <tr>
      <th>SNIP</th>
      <th>0.03</th>
      <td>92.4</td>
      <td>84.3</td>
      <td>29.1(+27.1)</td>
      <td>204.4</td>
      <td>0.32</td>
      <td>82.5(+6.2)</td>
      <td>66.5(+4.5)</td>
      <td>5.93</td>
    </tr>
    <tr>
      <th>TP</th>
      <th>0.03</th>
      <td>92.4</td>
      <td>84.2</td>
      <td>28.5(+26.5)</td>
      <td>197.3</td>
      <td>0.21</td>
      <td>83.0(+6.7)</td>
      <td>67.3(+5.3)</td>
      <td>5.12</td>
    </tr>
    <tr>
      <th>HYDRA</th>
      <th>0.03</th>
      <td>91.2</td>
      <td>82.7</td>
      <td>41.5(+39.5)</td>
      <td>146.3</td>
      <td>0.64</td>
      <td>79.5(+3.2)</td>
      <td>66.3(+4.3)</td>
      <td>8.75</td>
    </tr>
    <tr>
      <th>HYDRA+NRSLoss</th>
      <th>0.03</th>
      <td>89.8</td>
      <td>81.1</td>
      <td>40.5(+38.5)</td>
      <td>144.2</td>
      <td>0.55</td>
      <td>82.0(+5.7)</td>
      <td>66.0(+4.0)</td>
      <td>4.02</td>
    </tr>
    <tr>
      <th>IMP+RSLoss</th>
      <th>0.03</th>
      <td>85.6</td>
      <td>75.9</td>
      <td>25.2(+23.2)</td>
      <td>179.2</td>
      <td>0.03</td>
      <td>83.5(+7.2)</td>
      <td>65.7(+3.7)</td>
      <td>6.52</td>
    </tr>
    <tr>
      <th>IMP+NRSLoss</th>
      <th>0.03</th>
      <td>92.1</td>
      <td>83.2</td>
      <td>33.5(+31.5)</td>
      <td>153.2</td>
      <td>0.08</td>
      <td>86.0(+9.7)</td>
      <td>68.3(+6.3)</td>
      <td>6.06</td>
    </tr>
    <tr>
      <th>StructLTH[1]</th>
      <th>0.27</th>
      <td>87.7</td>
      <td>78.0</td>
      <td>33.0(+31.0)</td>
      <td>159.1</td>
      <td>0.55</td>
      <td>82.2(+5.9)</td>
      <td>65.9(+3.9)</td>
      <td>3.71</td>
    </tr>
    <tr>
      <th></th>
      <th>Slim</th>
      <th>0.35</th>
      <td>89.7</td>
      <td>78.7</td>
      <td>35.5(+33.5)</td>
      <td>151.8</td>
      <td>0.63</td>
      <td>84.0(+7.7)</td>
      <td>65.3(+3.3)</td>
      <td>7.77</td>
    </tr>
    <tr>
      <th rowspan="9">CIFAR10
$\epsilon = 2/255$</th>
      <th>Dense</th>
      <th>1</th>
      <td>82.4</td>
      <td>68.6</td>
      <td>1.5</td>
      <td>278.9</td>
      <td>1</td>
      <td>54.1</td>
      <td>43.0</td>
      <td>6.68</td>
    </tr>
    <tr>
      <th>IMP</th>
      <th>0.03</th>
      <td>62.2</td>
      <td>55.4</td>
      <td>23.5(+22.0)</td>
      <td>135.3</td>
      <td>0.13</td>
      <td>61.0(+6.9)</td>
      <td>50.1(+7.1)</td>
      <td>6.65</td>
    </tr>
    <tr>
      <th>SNIP</th>
      <th>0.03</th>
      <td>61.5</td>
      <td>55.1</td>
      <td>22.5(+21.0)</td>
      <td>128.4</td>
      <td>0.04</td>
      <td>59.8(+5.7)</td>
      <td>48.4(+5.4)</td>
      <td>6.67</td>
    </tr>
    <tr>
      <th>TP</th>
      <th>0.03</th>
      <td>59.7</td>
      <td>55.4</td>
      <td>24.0(+22.5)</td>
      <td>132.4</td>
      <td>0.05</td>
      <td>59.9(+5.8)</td>
      <td>47.6(+4.6)</td>
      <td>6.02</td>
    </tr>
    <tr>
      <th>HYDRA</th>
      <th>0.03</th>
      <td>60.4</td>
      <td>55.4</td>
      <td>23.5(+22.0)</td>
      <td>132.2</td>
      <td>0.11</td>
      <td>60.5(+6.4)</td>
      <td>48.3(+5.3)</td>
      <td>8.75</td>
    </tr>
    <tr>
      <th>HYDRA+NRSLoss</th>
      <th>0.03</th>
      <td>54.9</td>
      <td>48.4</td>
      <td>25.0(+22.0)</td>
      <td>132.2</td>
      <td>0.05</td>
      <td>58.0(+3.9)</td>
      <td>49.0(+6.0)</td>
      <td>8.75</td>
    </tr>
    <tr>
      <th>IMP+RSLoss</th>
      <th>0.03</th>
      <td>60.2</td>
      <td>54.2</td>
      <td>23.5(+22.0)</td>
      <td>134.4</td>
      <td>0.13</td>
      <td>58.6(+4.5)</td>
      <td>46.3(+3.3)</td>
      <td>6.52</td>
    </tr>
    <tr>
      <th>IMP+NRSLoss</th>
      <th>0.03</th>
      <td>60.7</td>
      <td>51.0</td>
      <td>25.0(+23.0)</td>
      <td>131.2</td>
      <td>0.21</td>
      <td>62.2(+8.1)</td>
      <td>51.2(+8.2)</td>
      <td>6.06</td>
    </tr>
    <tr>
      <th>StructLTH[1]</th>
      <th>0.35</th>
      <td>55.6</td>
      <td>48.3</td>
      <td>14.0(+12.5)</td>
      <td>143.7</td>
      <td>0.55</td>
      <td>57.5(+3.4)</td>
      <td>44.6(+1.6)</td>
      <td>3.71</td>
    </tr>
    <tr>
      <th></th>
      <th>Slim</th>
      <th>0.35</th>
      <td>56.9</td>
      <td>49.6</td>
      <td>26.0(+24.5)</td>
      <td>72.9</td>
      <td>0.79</td>
      <td>59.2(+5.1)</td>
      <td>47.5(+4.5)</td>
      <td>5.65</td>
    </tr>
  </tbody>
</table>

［#69］
2) Pruning methods: We try different setups of hyperparameters for unstructured and structured pruning. For unstructured pruning methods, we follow the default setting in [8] and set the iterative weight pruning rate to 0.2, and prune 16 times with re-training; for structured pruning methods, we keep a similar pruning speed. $^1$

［#70］
We set the NRSLoss weight to 0.01 and set the L1-norm regularizer weight to 0.0001 following [21]. For RS Loss and NRS Loss-based unstructured pruning, we train with these losses and pruning with IMP, as Section 3 introduced, denoted as IMP+RSLoss and IMP+NRSLoss. We also test training with NRSLoss and pruning with HYDRA, denoted as HYDRA+NRSLoss, to demonstrate the effectiveness of NRSLoss as a regularizer.

［#71］
3) Training methods: To demonstrate the general effectiveness of network pruning, we choose SOTA adversarial and certified training methods:

［#72］
Adversarial training: we choose the advanced FGSM+GradAlign [1] as the adversarial training method (we denote it as FGSM for conciseness hereinafter). The learning rate is set to 0.01, and we use Stochastic Gradient Descent with 0.9 momentum and 0.0005 weight decay as optimizer. All GradAlign-related hyperparameters follow [1].

［#73］
Certified training: we choose auto-LiRPA [42] under CROWN-IBP + Loss Fusion setting. The learning rate is set to 0.001, and we use Adam with a weight decay of 0 for RSLoss and NRSLoss-based pruning and 0.00001 for other pruning methods. For the bound computation of NRSLoss and RSLoss, we use the bound produced by auto-LiRPA during certified training and use IBP during adversarial training. auto-LiRPA essentially uses IBP to compute bounds when input perturbation reaches a maximum during training and uses IBP constantly during testing. As we find the results to be unstable for certified training, we use five different random seeds to initialize the training process, and then average the results of the same iterations. This is different from adversarial training where we only run one experiment.

［#74］
FGSM and auto-LiRPA share some common hyperparameters. The batch size is set to 128, and we clip the norm of

［#74］
$^1$Note that for HYDRA pruning, the semi-supervised training scheme which exploits extra unlabeled data as in [29] is NOT used in our experiments, for a fair comparison.


［#75］
<table>
  <caption>TABLE II: Verified accuracies of different pruning and robust training methods and perturbation scales $\epsilon$ under auto-LiRPA setting.</caption>
  <thead>
    <tr>
      <th colspan="3">$\epsilon$</th>
      <th colspan="3">2/255</th>
      <th colspan="4">8/255</th>
    </tr>
    <tr>
      <th>Pruning Type</th>
      <th>Pruning Method</th>
      <th>Remain Ratio</th>
      <th>$std$</th>
      <th>$ver$</th>
      <th>$t$</th>
      <th>Remain Ratio</th>
      <th>$std$</th>
      <th>$ver$</th>
      <th>$t$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">Dense</td>
      <td>1</td>
      <td>54.1</td>
      <td>43.0</td>
      <td>6.68</td>
      <td>1</td>
      <td>36.1</td>
      <td>29.3</td>
      <td>5.83</td>
    </tr>
    <tr>
      <td rowspan="4"></td>
      <td>IMP</td>
      <td>0.13</td>
      <td>61.0</td>
      <td>50.1</td>
      <td>6.65</td>
      <td>0.13</td>
      <td>35.0</td>
      <td>29.8</td>
      <td>5.11</td>
    </tr>
    <tr>
      <td>HYDRA</td>
      <td>0.11</td>
      <td>60.4</td>
      <td>48.3</td>
      <td>8.75</td>
      <td>0.11</td>
      <td>34.5</td>
      <td>29.5</td>
      <td>6.36</td>
    </tr>
    <tr>
      <td>IMP+RSLoss</td>
      <td>0.13</td>
      <td>58.6</td>
      <td>46.3</td>
      <td>6.52</td>
      <td>0.64</td>
      <td>35.0</td>
      <td>28.0</td>
      <td>4.39</td>
    </tr>
    <tr>
      <td>IMP+NRSLoss</td>
      <td>0.21</td>
      <td>62.2</td>
      <td>51.2</td>
      <td>6.06</td>
      <td>0.51</td>
      <td>39.0</td>
      <td>31.5</td>
      <td>4.32</td>
    </tr>
    <tr>
      <td>Structured</td>
      <td>SLIM</td>
      <td>0.79</td>
      <td>59.2</td>
      <td>47.5</td>
      <td>5.65</td>
      <td>0.7</td>
      <td>35.9</td>
      <td>29.8</td>
      <td>3.86</td>
    </tr>
  </tbody>
</table>

［#76］
![](./images/867772188499902851_9.jpg)

［#77］
Fig. 4: The ratio of unstable neurons v.s. pruning times of different pruning methods under FGSM setting.

［#78］
gradients to a maximum of 8. We train 200 epochs in one pruning iteration for each experiment, with a learning rate decay factor of 0.1 at 140 and 170 epochs. We set the input perturbation $\epsilon$ as $L_{inf}$-norm ball to 0.1 for the FashionMNIST dataset and 2/255 for SVHN and CIFAR10 datasets, and gradually increase $\epsilon$ from 0 to 2/255 starting from 11th epoch and until 80th epoch. We also scale the perturbation to 8/255 to validate the effectiveness of pruning under bigger perturbations. The training epochs under 8/255 perturbation is set to 300. After each pruning iteration, we rewind the remaining weights to initial states and reset the optimizer with the initial learning rate and $\epsilon$.

［#79］
We replace the standard IMP with weight rewinding [8] with each training method and each pruning method, and output the pruned subnetworks with different sparsity during iterative pruning and re-training.

［#80］
4) Verifier and Evaluation Criterion: For each training method and each pruning method, we use the SOTA certified verifier Beta-CROWN [35] to obtain the final accuracy of the subnetworks. Beta-CROWN is a highly GPU-parallelized verification framework and has SOTA performance in terms of bound tightness and verification speed. We choose ERAN benchmark [3] which contains 1000 test images, and test on the first 200 images to reduce the time budget. We set the timeout of each test image to 300 seconds. We test the standard, adversarial, and verified accuracies, as well as time and GPU memory consumption of each model. We run the verifications using one NVIDIA RTX A6000 GPU card.

［#81］
<table>
  <caption>TABLE III: The feedforward model architecture in our experiments. ConvBlock($in$,$out$,$k$,$s$) refers to the composition of (convolution layer, BN layer, ReLU layer) where the convolution layer has $in$ input channels, $out$ output channels, $k \times k$ kernel size and $s$ strides. Note that for the FashionMNIST task which takes greyscale images instead of RGM images as input, we modify the input channel number of the first convolutional layer from 3 to 1, and modify the input units of the first FC layer from 2048 to 1152 accordingly.</caption>
  <tbody>
    <tr>
      <td>Input</td>
    </tr>
    <tr>
      <td>ConvBlock(3,32,3,1)</td>
    </tr>
    <tr>
      <td>ConvBlock(32,64,4,2)</td>
    </tr>
    <tr>
      <td>ConvBlock(64,64,3,1)</td>
    </tr>
    <tr>
      <td>ConvBlock(64,128,4,2)</td>
    </tr>
    <tr>
      <td>ConvBlock(128,128,4,2)</td>
    </tr>
    <tr>
      <td>FC(2048,100)</td>
    </tr>
    <tr>
      <td>ReLU</td>
    </tr>
    <tr>
      <td>FC(100,10)</td>
    </tr>
    <tr>
      <td>Output</td>
    </tr>
  </tbody>
</table>

### B. Experiment Results and Analysis

［#82］
For the 3 benchmark datasets, the comprehensive experiment results under $\epsilon=2/255$ are shown in Table I, and the sample curves of Verified Accuracy v.s. iterative pruning times with seed 100 are shown in Figure 6. For each model, we verify all subnetworks produced by iterative pruning and report the best-verified accuracy and other corresponding evaluation metrics. We then choose several representative pruning methods that either have good performance or distinctive motivations or are important baselines, and evaluate them under different perturbation scales on the CIFAR10 dataset, as shown in Table II. For a fair comparison with HYDRA which is SOTA robustness-based pruning, we also reproduce a similar CROWN-

［#83］
![](./images/867772188499902851_10.jpg)

［#84］
Fig. 5: The illustration of the feedforward model architecture in our experiments.

［#85］
IBP based experiment from [29] as shown in Section IV-C5 to demonstrate that IMP can indeed outperform HYDRA. We also conduct a experiment to validate that weight-rewinding [8] is better than finetuning [29] for improving certified robustness, as shown in Section IV-C4. The results of Random Pruning are omitted from Table I and II since its standard accuracies are very poor and non-competitive (some can be found in Figure 6 (a) for illustration purpose). We next present the result analysis.

［#86］
1) Can existing pruning methods improve certified robustness?: From Table I and Table II, we observe general improvements in certified robustness brought by pruning, both under FGSM and auto-LiRPA settings. Specifically, Table I shows that under the auto-LiRPA setting, existing pruning methods can improve verified accuracies for $2.5-5.2\%$ on FashionMNIST, $3.3-6.3\%$ on SVHN, $1.6-7.1\%$ on CIFAR10, respectively and improve standard accuracies for $0.3-3.3\%$ on FashionMNIST, $3.2-9.7\%$ on SVHN, $3.4-8.1\%$ on CIFAR10 respectively, among which IMP consistently outperforms other existing pruning methods, with highest improvements of both standard and verified accuracies. This demonstrates that pruning can generally improve certified robustness. Moreover, under certified training, this improvement comes with no extra trade-off such as standard accuracy. We also observe that on more realistic datasets (SVHN, CIFAR10), the improvements under certified training are significantly bigger than that of the synthetic dataset (FashionMNIST).

［#87］
Under the FGSM setting, there are great improvements in verified accuracies with different pruning methods ranging from $21.0-40.0\%$ on FashionMNIST, $23.2-39.5\%$ on SVHN, and $12.5-24.5\%$ on CIFAR10, respectively, among which IMP+NRSLoss, HYDRA and Network Slimming (Slim) obtain highest verified accuracy on FashionMNIST, SVHN, and CIFAR10, respectively. We also observe an obvious trade-off of standard/adversarial accuracy v.s. verified accuracy, i.e. with the big increase of verified accuracy after pruning, the standard and adversarial accuracies drop significantly. To explain this trade-off, we visualize the ratio of unstable neurons of different pruning methods, as shown in Figure 4. We find that the ratio of unstable neurons generally decreases as the sparsity gets higher, this is compliant with that neuron stability is important for certified robustness. However, if all neurons become stable, the whole network will become a linear function, which in turn withholds the standard accuracy. Hence the standard/verified accuracy trade-off is essentially the stability/expressiveness trade-off of the network. Nevertheless, this trade-off is not obvious under the auto-LiRPA setting since the training objective of auto-LiRPA incorporates standard accuracy.

［#88］
Across different datasets, we observe general improvement brought by pruning for certified robustness, which consolidates our conclusion that pruning can generally improve certified robustness. In particular, NRSLoss-based pruning can outperform other pruning methods consistently under certified training and achieves competitive performance under adversarial training, which demonstrates the effectiveness of NRSLoss regularizer and the pruning scheme of IMP+NRSLoss, as we explained in the methodology.

［#89］
Resource Consumption: It can be observed from Table I and II that unstructured pruning tends to produce better performance than structured pruning. However, structured pruning has the advantage over unstructured pruning that it brings real hardware acceleration for certified verification, especially given that the computational overhead is a significant bottleneck for verifying large neural networks even with highly GPU-parallelized verifiers such as Beta-CROWN. We show an overview of the time and peak GPU memory consumption of structured pruning under different pruning stages as in Figure 7 in Section. We can see that with every 3 pruning, which increases about 30% channel sparsity, the time consumption for models trained with auto-LiRPA can reduce by about 50%, whereas for models trained with FGSM can reduce by about $60-80\%$. We also observe that GPU memory consumption can be greatly reduced at high channel sparsity. The reduction in GPU memory consumption is even more important given the GPU memory bottleneck for complete verification of large neural networks. Furthermore, for these pruning methods, we observe similar high performing sparsity under different random seeds as shown in Figure 10 in Section, which means we do not need to verify every sparsity one by one to pick out the best sparsity, and it is crucial for accelerating the verification process in practice.

［#90］
2) How does NRSLoss-based pruning outperform other pruning methods?: From Table I and II, we observe that IMP+NRSLoss outperform other pruning methods under auto-LiRPA setting. Take the CIFAR10 dataset as an example, with 2/255 perturbation, IMP+NRSLoss improves certified accuracy for 8.2% and standard accuracy for 8.1%; with 8/255 perturbation, IMP+NRSLoss improves certified accuracy for 2.9% and standard accuracy for 2.3%. Notably, IMP+NRSLoss achieves both the highest standard and verified accuracies, since the training objective of auto-LiRPA incorporates standard accuracy. By comparing HYDRA setting and HYDRA+NRSLoss setting, we observe NRSLoss can improve the verified accuracy for HYDRA pruning in FashionMNIST and CIFAR10 dataset, and has better standard/verified accuracy trade-off under certified training for SVHN dataset. We thus conclude that NRSLoss regularizer is effective for HYDRA pruning in most cases and is effective for IMP pruning for all cases we have tested. To

［#91］
![](./images/867772188499902851_11.jpg)
［#92］
*(a) auto-LiRPA unstructured*

［#93］
![](./images/867772188499902851_12.jpg)
［#94］
*(b) auto-LiRPA structured*

［#95］
![](./images/867772188499902851_13.jpg)
［#96］
*(c) FGSM unstructured*

［#97］
Fig. 6: Verified Accuracy v.s. iterative pruning times on CIFAR10 dataset. (a) is unstructured pruning under auto-LiRPA training, (b) is structured pruning under auto-LiRPA training, (c) is unstructured pruning under FGSM. We omit structured pruning under FGSM due to page limit. Note that (c) is plotted using CROWN verifier instead of Beta-CROWN due to long verification time on Beta-CROWN.

［#98］
demonstrate that the performance improvements of NRSLoss indeed come from stability-based regularization as discussed in Section 3, we visualize the pre-activation *network instability* (as proposed in Section 3) in Figure 8. We observe that the RS Loss and NRS Loss-based pruning have significantly lower instability compared to IMP, and the instability decreases as the sparsity gets higher, which proves that pruning with NRSLoss and RSLoss regularizer can decrease network instability, hence improving the certified robustness. It can also be observed that the RSLoss has lower instability than NRSLoss, however, since NRSLoss eliminates the gradients from BN layer, the RSLoss actually gets lower instability by influencing BN layers, which in turn would hurt normal training, and thus hurt overall performance. The advantage of NRSLoss can also be interpreted using the NRSLoss landscape as shown in Figure 2. Compared to RSLoss, NRSLoss takes account of the channel importance, so that using NRSLoss can avoid regularizing neurons that are in the important channels. From these results, we can again conclude that neuron stability is important for certified robustness, in particular, IMP+NRSLoss motivated by improving neuron stability is effective for improving certified robustness.

［#99］
We note that the literature results as in [41] are much better than the results reported in Table I and II. In fact, this is because they use much larger networks, whereas we focus on complete verification which causes Out-Of-Memory error on large networks, so we only choose small networks for testing, and note that our designed network, though small, is still the largest network we can perform complete verification on a GPU card with 24GB memory.

［#100］
3) *Existence of certified lottery tickets.:* As one last "hidden gem" finding, we demonstrate the existence of *certified lottery tickets*, that generalizes the lottery ticket hypothesis [8] to certified robustness. Specifically, from Table I, we observe that **all** pruning methods under certified training across all 3 datasets can find certified lottery tickets that can match *both* standard and verified accuracies to the original dense models, and most of the pruning methods can produce certified lottery tickets that significantly outperform original dense networks. From Table II, we see that certified lottery tickets can be found on most pruning methods with a bigger perturbation scale except for random pruning and IMP+RSLoss. From Figure 6(a), we observe that except for unstructured pruning (except for random pruning), certified lottery tickets occur almost in every sparsity. The above findings hence validate the existence of certified lottery tickets.

［#101］
4) *How should we choose pruning methods for certified robustness?:* Generally, we would recommend IMP and IMP+NRSLoss for performance concerns because they have the best verified accuracy under certified training across different datasets, and we recommend Network Slimming for efficiency concerns because its structured pruning nature can essentially reduce the computational overhead of the complete verification. We empirically find that the relative performance of our tested pruning methods under certified training is similar to that under standard training. We conjecture that this is because an important goal of most pruning methods is causing a minimal negative influence on the training objective function, and this objective function is benign accuracy under standard training and verified accuracy under certified training, respectively. These pruning methods also implicitly regularize the network stability and bound tightness as stated in Section III-C, which leads to general improvement compared to dense baselines. However, our proposed NRSLoss-based pruning explicitly regularizes network stability which makes it outperform other pruning methods.

### C. Ablation
［#102］
In this section, we conduct several ablation studies mainly on the CIFAR10 dataset to further consolidate our claims.

［#103］
1) *Why focusing on complete verification:* In this subsection, we show the reasons why we focus on complete verification instead of both complete verification and incomplete verification.


［#104］
![](./images/867772188499902851_14.jpg)

［#105］
![](./images/867772188499902851_15.jpg)

［#106］
![](./images/867772188499902851_16.jpg)

［#107］
![](./images/867772188499902851_17.jpg)

［#108］
Fig. 7: The mean verification time and average GPU memory consumption for structured pruning methods on CIFAR10 dataset under CROWN mode. X-axis means the number of pruning iterations with 0.11 channel pruning rate. Note that we don't do such test on complete verification of Beta-CROWN mode which is time consuming but has highly similar trend to the results under CROWN mode.

［#109］
Firstly, we show that the bound produced by complete verifi- cation methods is always tighter than incomplete verification methods such as IBP or CROWN. Secondly, complete verifica- tion is further needed because neuron stability is important for the sub-domain split problem of complete verification, which is one motivation for proposing NRSLoss. To verify the first point, we test certified robustness under certified training about the comparison of incomplete/complete verification and suggest comparing Table V and Figure 6(c) for adversarial training setting, these results demonstrate that complete verification ($\beta$- CROWN) is always tighter than incomplete verification (IBP, CROWN).

［#110］
2) Results under different evaluation criterion: In Section IV-A3, we mentioned that the final numerical results are the averaged result of 5 different random seeds of the same training and pruning iteration. We here present another evaluation criterion, i.e. by first picking the iteration with the best verified accuracy, and then averaging the results of the picked iterations of the 5 different random seeds. The comparison of these 2 criteria is shown in Table V, where the "AVG." column denotes the first criterion, while the "BEST" column denotes the second criterion. We observe slightly higher results under the second criterion, but the relative performance among different pruning methods is similar of these two criteria.

［#111］
3) Comparison of pruning under different certified training methods: In our main experiments, we choose the auto-LiRPA as the certified training method. The reason we choose this method is based on its training efficiency and competitive performance, and its training efficiency mainly comes from the loss fusion technique as proposed in [42]. The training efficiency is important in our experiments because we use iterative training and pruning, which boosts the overall training time to 16 times longer. We notice that the certified training method[30] (denoted as FastIBP in the following context) with

［#112］
![](./images/867772188499902851_18.jpg)

［#113］
Fig. 8: Network instability v.s. iterative pruning times of pre- activation and pre-BN forward pass values. The bounds are computed under 2/255 input perturbation using auto-LiRPA and the whole test set of CIFAR10.

［#114］
SOTA performance (i.e. SOTA verified accuracy) claims that loss-fusion has a negative influence on the performance and thus doesn't adopt it in their method. We empirically find that without loss-fusion, the training speed of FastIBP is 4 times slower than auto-LiRPA. We thus choose auto-LiRPA as the certified training method in our main experiments. However, we here present a comparison of results (see Table IV) of these 2 certified training methods on the CIFAR10 dataset and pruned with several pruning methods, to demonstrate that the improvement of certified robustness brought by pruning is consistent with different certified training methods. The hyperparameter settings are the same as mentioned in our main experiments. From Table IV, we observe better performance can be obtained with FastIBP, and standard/verified accuracies are consistently improved with different pruning methods, among which IMP+NRSLoss still achieves the best performance.

［#115］
4) Comparison of finetuning and weight-rewinding for pruning: We empirically find that after each pruning, rewinding the network parameters to their initial states as in [8] produces better performance than finetuning the parameters as in [29]. Specifically, we follow the experiment setup as in Section IV-A, except that for finetuning mode we don't re-initialize the learning rate after pruning. The results are demonstrated in Figure 9. We observe that the finetuning-based pruning always produces worse performance than weight rewinding- based pruning, and its accuracy tends to collapse at 3rd pruning iteration. Therefore we conclude that weight-rewinding-based pruning is more effective than finetuning-based pruning for certified robustness.

［#116］
5) The performance of HYDRA in original paper: We reproduce the performance of HYDRA using the released code from [29]. Specifically, we run their original experiments of HYDRA pruning and Least Weight Magnitude(LWM) pruning under the CROWN-IBP setting and SVHN dataset. The pruning is applied only once followed by finetuning. The results under different pruning rates $k$ are shown in Table VI. We observe that LWM pruning produces better performance than HYDRA pruning. Since the pruning process of LWM is essentially the same as IMP in our experiments (except that IMP prunes multiple times instead of pruning only once), the result that LWM is better than HYDRA is consistent with our experiments in Table I where IMP is relatively better than HYDRA in many cases.

### D. Summary of Findings
［#117］
In our experiments, we find that pruning can generally improve certified robustness for neural networks trained with different robust training methods and observe the existence of certified lottery tickets. Under adversarial training, we observe a significant trade-off between standard and verified accuracies with different pruning methods, but under certified training, pruning can improve both standard and verified accuracies. From our experiments, we know that RSLoss and NRSLoss are both effective at regularizing network stability but NRSLoss is better for imposing less regularization on more important neurons and removing the negative influence of stability regularization for BN layers. From Figure 7, we observe that structured pruning can considerably reduce the computational overhead of complete verification for neural networks. Among existing pruning methods that we tested, we empirically find that IMP can generally achieve relatively good performance, while it is outperformed by IMP+NRSLoss which incorporates stability regularization.

### V. CONCLUSION
［#118］
In this paper, we demonstrate that pruning can generally improve certified robustness, both for adversarial and certified training. We analyze some important factors that influence certified robustness, and offer a new angle to study the intriguing interaction between sparsity and robustness, i.e. interpreting the interaction of sparsity and certified robustness via neuron stability. In particular, we find neuron stability to be crucial for improving certified robustness, on which motivation we propose the novel NRSLoss-based pruning that outperforms existing pruning methods. We also observe the existence of certified lottery tickets. We believe our work has revealed the relationships between pruning and certified robustness, which can shed light on future research to design better sparse networks with certified robustness.

### REFERENCES






















































