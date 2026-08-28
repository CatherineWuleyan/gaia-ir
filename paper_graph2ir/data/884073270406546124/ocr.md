# Probabilistic Modeling: Proving the Lottery Ticket Hypothesis in Spiking Neural Network

Man Yao$^{1,2,3,*}$,Yuhong Chou$^{4,2*}$, Guangshe Zhao$^{1}$, Xiawu Zheng$^{3}$,
Yonghong Tian$^{5,3}$, Bo Xu$^{2}$, Guoqi Li$^{2\dagger}$

$^{1}$School of Automation Science and Engineering, Xi'an Jiaotong University, Xi'an, Shaanxi, China
$^{2}$Institute of Automation, Chinese Academy of Sciences, Beijing, China
$^{3}$Peng Cheng Laboratory, Shenzhen, Guangzhou, China
$^{4}$College of Artificial Intelligence, Xi'an Jiaotong University, Xi'an, Shaanxi, China
$^{5}$Institute for Artificial Intelligence, Peking University, Beijing, China

## Abstract
The Lottery Ticket Hypothesis (LTH) states that a randomly-initialized large neural network contains a small sub-network (i.e., winning tickets) which, when trained in isolation, can achieve comparable performance to the large network. LTH opens up a new path for network pruning. Existing proofs of LTH in Artificial Neural Networks (ANNs) are based on continuous activation functions, such as ReLU, which satisfying the Lipschitz condition. However, these theoretical methods are not applicable in Spiking Neural Networks (SNNs) due to the discontinuous of spiking function. We argue that it is possible to extend the scope of LTH by elimi- nating Lipschitz condition. Specifically, we propose a novel probabilistic modeling approach for spiking neurons with complicated spatio-temporal dynamics. Then we theoretically and experimentally prove that LTH holds in SNNs. According to our theorem, we conclude that pruning directly in accordance with the weight size in existing SNNs is clearly not optimal. We further design a new criterion for pruning based on our theory, which achieves better pruning results than baseline.

## 1 Introduction
By mimicking the spatio-temporal dynamics behaviors of biological neural circuits, Spiking Neural Networks (SNNs) [1-4] provide a low-power alternative to traditional Artificial Neural Networks (ANNs). The binary spiking communication enables SNNs to be deployed on neuromorphic chips [5-7] to perform sparse synaptic accumulation for low energy consumption. Given the memory storage limitations of such devices, neural pruning methods are well recognized as one of the crucial methods for implementing SNNs in real-world applications. Pruning redundant weights from an over-parameterized model is a mature and efficient way of obtaining significant compression [8, 9].

Recently, Lottery Ticket Hypothesis (LTH), a mile stone is proposed in the literature of network pruning, which asserts that an over-parameterized neural network contains sub-networks that can achieve a similar or even better accuracy than the fully-trained original dense networks by *training only once* [10]. A Stronger version of the LTH (SLTH) was then proposed: there is a high probability that a network with random weights contains sub-networks that can approximate any given sufficiently- smaller neural network, *without any training* [11]. SLTH claims to find the target sub-network without training, thus it is a kind of complement to original LTH that requires training [11]. Meanwhile, SLTH is considered to be "stronger" than LTH because it claims to require no training [12].

---
$^*$Equal contribution. manyao@stu.xjtu.edu.cn; yhjldl@stu.xjtu.edu.cn
$^\dagger$Corresponding author, guoqi.li@ia.ac.cn

Preprint. Under review.

The effectiveness of LTH in ANNs have been verified by a series of experiments [10, 13, 14, 11]. Furthermore, it has been theoretically proved by several works with various assumptions [12, 15-17], due to its attractive properties in statement. A line of work is dedicated to designing efficient pruning algorithms based on LTH [18-20]. But the role of LTH in SNN is almost blank. There is only one work that experimentally verifies that LTH can be applied to SNNs [21], whether it can also be theoretically established remains unknown.

In this work, we theoretically and experimentally prove that LTH (Strictly speaking, $SLTH^3$ ) holds in SNNs. There are two main hurdles. First and foremost, the binary spike signals are fired when the membrane potentials of the spiking neurons exceed the firing threshold, thus the activation function of a spiking neuron is discrete. But all current works [12, 15-17] proving LTH in ANNs relies on the Lipschitz condition. Only when the Lipschitz condition is satisfied, the error between the two layers of the neural network will be bounded when they are approximated. Second, brain-inspired SNNs have complex dynamics in the temporal dimension, which is not considered in the existing proofs and increases the difficulty of proving LTH in SNNs.

To bypass the Lipschitz condition, we design a novel probabilistic modeling approach. The approach can theoretically provide the probability that two SNNs behave identically, which is not limited by the complex spatio-temporal dynamics of SNNs. In a nutshell, we establish the following result:

Informal version of Theorem 4.3 For any given target SNN $\hat{G}$, there is a sufficiently large SNN $G$ with a sub-network (equivalent SNN) $\tilde{G}$ that, with a high probability, can achieve the same output as $\hat{G}$ for the same input,

$$
\sup_{\boldsymbol{S} \in S} \frac{1}{T} \sum_{t=1}^{T}\left\|\tilde{G}^{t}(\boldsymbol{S})-\hat{G}^{t}(\boldsymbol{S})\right\|_{2}=0,
$$

where $t=1,2,\cdots,T$ is the timestep, $\boldsymbol{S}$ is the input spiking tensor containing only 0 or 1.

As a complement to the theoretical proof part, we show experimentally that there are also sub-networks in SNNs that can work without training. Subsequently, it is clear that ANNs and SNNs operate on fundamentally distinct principles about how weight influences the output of the activation layer. Thus, simply using the weight size as the criterion for pruning like ANNs must not be the optimal strategy. Based on this understanding, we design a new weight pruning criterion for SNNs. We evaluate how likely weights are to affect the firing of spiking neurons, and prune according to the estimated probabilities. In summary, our contributions are as follows:

- We propose a novel probabilistic modeling method for SNNs that for the first time theoretically establishes the link between spiking firing and pruning (Section 3). The probabilistic modeling method is a general theoretical analysis tool for SNNs, and it can also be used to analyze the robustness and other compression methods of SNNs (Section 6).
- With the probabilistic modeling method, we theoretically prove that LTH also holds in SNNs with binary spiking activations and complex spatio-temporal dynamics (Section 4).
- We experimentally find the good sub-networks without weight training in random initialized SNNs, which is consistent with our theoretical results. (Section 5)
- We apply LTH for pruning in SNNs and design a new probability-based pruning criterion for it (Section 6). The proposed pruning criterion can achieve better performance in LTH-based pruning methods and can also be exploited in non-LTH traditional pruning methods.

## 2 Related Work

Spiking neural networks. The spike-based communication paradigm and event-driven nature of SNNs are key to their energy-efficiency advantages [3, 22, 23, 4]. Spike-based communication makes cheap synaptic Accumulation (AC) the basic operation unit, and the event-driven nature means that only a small portion of the entire network is active at any given time while the rest is idle. In contrast, neurons in ANNs communicate information using continuous values, so Multiply-and-Accumulate (MAC) is the major operation, and generally all MAC must be performed even if all inputs or activations are zeros. SNNs can be deployed on neuromorphic chips for low energy consumption

---
$^3$In theoretical proofs, SLTH is considered a stronger version of LTH[12, 17]

![](./images/884073270406546124_1.jpg)

Figure 1: **Overview of this work.** (a), The goal of spiking neuron approximate. (b), The firing behavior in the approximation of spiking neurons can be described by the proposed probabilistic modeling approach (Section 3). Spikes are fired when the membrane potential exceeds the firing threshold $u_{th}$. As long as a weight change induces $u$ to fall into the crisis neighborhood, there is a certain probability that the spiking firing will be changed (0 to 1, or 1 to 0). We hope that spiking firing will not change after the redundant weights are pruned. (c), Equivalent structure modeling. The error between the target and equivalent SNN is related to the network width (Section 4). (d), Proof of LTH in SNN (Section 4). (e), We provide a pruning technique for SNNs (Section 6). New pruning criterion: compute the probability that the firing of a spiking neuron changes when weights are pruned, and pruning according to the rank of the probability.

[6, 5, 24]. Thus, spike-based neuromorphic computing has broad application prospects in battery constrained edge computing platforms [25], e.g., internet of things, smart phones, etc.

Pruning in spiking neural networks. Recent studies on SNN pruning have mostly taken two approaches: 1) Gaining knowledge from ANN pruning's successful experience 2) incorporating the unique biological properties of SNNs. The former technical route is popular and effective. Some typical methods include pruning according to predefined threshold value [26-28], soft-pruning that training weights and pruning thresholds concurrently [29], etc. The temporal dynamics of SNN are often also taken into consideration in the design of pruning algorithms [30, 21]. Meanwhile, There have been attempts to develop pruning algorithms based on the similarities between SNNs and neural systems, e.g., regrowth process [31], spine motility [32, 33], gradient rewiring [34], state transition of dendritic spines [35], etc. None of these studies, however, took into account the crucial factor that affects network performance: the link between weights and spiking firing. In this work, we use probabilistic modeling to analyze the impact of pruning on spiking firing, and accordingly design a pruning criterion for SNNs.

## 3 Probabilistic Modeling for Spiking Neurons

No matter how intricate a spiking neuron's internal spatio-temporal dynamics are, ultimately the neuron decide whether to fire a spike or not based on its membrane potential at a specific moment. Thus, the essential question of the proposed probabilistic modeling is how likely is the firing of a spiking neuron to change after changing a weight.

Leaky Integrate-and-Fire (LIF) is one of the most commonly used spiking neuron models [1, 2], because it is a trade-off between the complex spatio-temporal dynamic characteristics of biological neurons and the simplified mathematical form. It can be described by a differential function

$$
\tau \frac{d u(t)}{d t}=-u(t)+I(t), \tag{1}
$$

where $\tau$ is a time constant, and $u(t)$ and $I(t)$ are the membrane potential of the postsynaptic neuron and the input collected from presynaptic neurons, respectively. Solving Eq. (1), a simple iterative

representation of the LIF neuron [36, 37] for easy inference and training can be obtained as follow

$$
u_{i}^{t, l}=h_{i}^{t-1, l}+x_{i}^{t, l}, \tag{2}
$$

$$
s_{i}^{t, l}=\operatorname{Hea}\left(u_{i}^{t, l}-u_{t h}\right), \tag{3}
$$

$$
h_{i}^{t, l}=V_{r e s e t} s_{i}^{t, l}+\beta u_{i}^{t, l}\left(1-s_{i}^{t, l}\right), \tag{4}
$$

$$
x_{i}^{t, l}=\sum_{j=1}^{N} w_{i j}^{l} s_{j}^{t, l-1}, \tag{5}
$$

where $u_{i}^{t, l}$ means the membrane potential of the $i$-th neuron in $l$-th layer at timestep $t$, which is produced by coupling the spatial input feature $x_{i}^{t, l}$ and temporal input $h_{i}^{t-1, l}$, $u_{t h}$ is the threshold to determine whether the output spike $s_{i}^{t, l} \in\{0,1\}$ should be given or stay as zero, $\operatorname{Hea}(\cdot)$ is a Heaviside step function that satisfies $\operatorname{Hea}(x)=1$ when $x \geq 0$, otherwise $\operatorname{Hea}(x)=0$, $V_{reset }$ denotes the reset potential which is set after activating the output spiking, and $\beta=e^{-\frac{d t}{\tau}}<1$ reflects the decay factor. In Eq. (2), spatial feature $x_{i}^{t, l}$ can be extracted from the spike $s_{j}^{t, l-1}$ from the spatial output of the previous layer through a linear or convolution operation (i.e., Eq. (5)), where the latter can also be regraded as a linear operation [38]. $w_{i j}^{l}$ denotes the weight connect from the $j$-th neuron in $(l-1)$-th layer to the $i$-th neuron in $l$-th layer, $N$ indicates the width of the $(l-1)$-th layer.

The spatio-temporal dynamics of LIF can be described as: the LIF neuron integrates the spatial input feature $x_{i}^{t, l}$ and the temporal input $h_{i}^{t-1, l}$ into membrane potential $u_{i}^{t, l}$, then, the fire and leak mechanism is exploited to generate spatial output $s_{i}^{t, l}$ and the new neuron state $h_{i}^{t, l}$ for the next timestep. Specifically, When $u_{i}^{t, l}$ is greater than the threshold $u_{t h}$, a spike is fired and the neuron state $h_{i}^{t, l}$ is reset to $V_{reset }$. Otherwise, no spike is fired and the neuron state is decayed to $\beta u_{i}^{t, l}$. Richer neuronal dynamics [2] can be obtained by adjusting information fusion [39], threshold [40], decay [41], or reset [42] mechanisms, etc. Note, notations used in this work are summarized in Appendix A.

Probabilistic modeling of spiking neurons. Our method is applicable to various spiking neuron models as long as Eq. (3) is satisfied. In this section, superscript $l$ and $i$ will be omitted when location is not discussed, and superscript $t$ will be omitted when the specific timestep is not important. The crux of probabilistic modeling is how errors introduced by pruning alter the firing of spiking neurons.

We start by discussing only spatial input features. For a spiking neuron, assuming that the temporal input of a certain timestep is fixed, for two different spatial input features $x$ and $x^{\prime} \in[x-\epsilon, x+\epsilon]$, the corresponding membrane potentials are also different, i.e., $u$ and $u^{\prime} \in[u-\epsilon, u+\epsilon]$. Once $u$ and $u^{\prime}$ are located in different sides of the threshold $u_{t h}$, the output will be different (see Fig. 1a). This situation can only happen when $u$ is located in crisis neighborhood $[u_{t h}-\epsilon, u_{t h}+\epsilon]$ (see Fig. 1b). That is, suppose membrane potential $u \notin[u_{t h}-\epsilon, u_{t h}+\epsilon]$, if it changes from $u$ to $u^{\prime}$, and $u^{\prime} \in[u-\epsilon, u+\epsilon]$, the output of the spiking neuron must not change. Consequently, the probability upperbound of a spiking neuron output change (crisis probability) is:

$$
\int_{u_{t h}-\epsilon}^{u_{t h}+\epsilon} p(u) \mathrm{d} u, \tag{6}
$$

where $p(\cdot)$ is the probability density function of the membrane potential distribution. For the case of two independent spiking neurons, if the input is the same, then $\epsilon$ is controlled by the weights of the two neurons (Fig. 1a).

It is reasonable to consider that the membrane potential follows a certain probability distribution. The membrane potential is accumulated by temporal input and spatial input feature, the former can be regarded as a random variable, and the latter is determined by the input spikes and weights. The input spike is a binary random variable according to a firing rate, and usually the weights are also assumed to satisfy a certain distribution. Moreover, some existing works directly assume that the membrane potential satisfies the Gaussian distribution [43, 44]. In this work, our assumptions about the membrane potential distribution are rather relaxed. Specifically, we give out a class of distributions for membrane potential, and we suppose all membrane potential should satisfy:

Definition 3.1. m-Neighborhood-Finite Distribution. For a probability density function $p(\cdot)$ and a value $m$, if there exists $\epsilon>0$ for the neighborhood $[m-\epsilon, m+\epsilon]$, in this interval, the max value

of function $p(\cdot)$ is finite, we call the distribute function $p(\cdot)$ as m-Neighborhood-Finite Distribution.
The symbolic language is as follows
$$
\exists \epsilon>0, x \in[m-\epsilon, m+\epsilon], \sup _{x} p(x)<+\infty. \tag{7}
$$

The upperbound of probability is controllable by controlling the input error $\epsilon$:

Lemma 3.2. For the probability density function $p(\cdot)$, if it is m-Neighborhood-Finite Distribution,
for any $\delta>0$, there exists a constant $\epsilon>0$ so that $\int_{m-\epsilon}^{m+\epsilon} p(x) \mathrm{dx} \leq \delta$.

Now, we add consideration of the temporal dynamics of spiking neurons. As shown in Eq. (2) and
Eq. (4), spiking neurons have a memory function that the membrane potential retains the neuronal
state information from all previous timesteps. The spatial input feature $x$ is only related to the current
input, and the temporal input $h$ is related to the spatial input features at all previous timesteps. For
the convenience of mathematical expression, if there is no special mention to consider the temporal
dynamics inside the neuron, we directly write the spiking neuron as $\sigma$. But it should be noted that $\sigma$
actually has complex internal dynamics. In its entirety, a spiking neuron in Eq. (3) can be denoted
as $\sigma_{\tilde{x}^{1: t-1}}^{t}(x^{t})$, where the temporal input $h^{t-1}$ in the membrane potential $u^{t}$ depends on $x^{1: t-1}$, i.e.,
spatial input features from 1 to $t-1$.

Using Definition 3.1, Lemma 3.2, and the math above, we can determine the probability of dif-
fering outputs from the two neurons due to errors in their spatial input features, under varying
constraints.Specifically, in Lemma 3.3, it is assumed that the timestep is fixed and the temporal inputs
to the two neurons are the same; in Lemma 3.4, we loosen the constraint on the timestep of the two
neurons; finally, in Theorem 3.5, we generalize our results to arbitrary timesteps for two spiking
layers ( $N$ neurons each layer).

Lemma 3.3. At a certain temestep $T$, if the spiking neurons are $u_{th}$-Neighborhood-Finite Distribution
and the spatial input features of two neuron got an error upperbound $\epsilon$, and they got the same temporal
input $h^{t-1}$, the probability upperbound of different outputs is proportional to $\epsilon$. Formally:

For two spiking neurons $\hat{\sigma}^{T}$ and $\tilde{\sigma}^{T}$, when $\tilde{h}^{T-1}=\hat{h}^{T-1}$ and $\hat{u}^{T}=\hat{h}^{T-1}+\hat{x}^{T}$ is a
random variable follows the $u_{th}$-Neighborhood-Finite Distribution, if $\left\|\tilde{x}^{T}-\hat{x}^{T}\right\| \leq \epsilon$, then:
$P\left[\hat{\sigma}^{T}(\hat{x}^{T}) \neq \tilde{\sigma}^{T}(\tilde{x}^{T})\right] \propto \epsilon$

Lemma 3.4. Suppose the spiking neurons are $u_{th}$-Neighborhood-Finite Distribution at timestep $T$
and the spatial input features of two corresponding spiking neurons got an error upperbound $\epsilon$ at any
timestep, and they got the same temporal input $h^{0}$, if both spiking neurons have the same output at
the first $T-1$ timesteps, then the probability upperbound is proportional to $\frac{\epsilon}{1-\beta}$. Formally:

For two spiking neurons $\hat{\sigma}^{T}$ and $\tilde{\sigma}^{T}$, when $\tilde{h}^{0}=\hat{h}^{0}$ and $\hat{u}^{T}=\hat{h}^{T-1}+\hat{x}^{T}$ is a random variable
follows the $u_{th}$-Neighborhood-Finite Distribution, if $\left\|\tilde{x}^{t}-\hat{x}^{t}\right\| \leq \epsilon$ and $\hat{\sigma}^{t}(\hat{x}^{t})=\hat{\sigma}^{t}(\tilde{x}^{t})$ for
$t=1,2, \cdots, T-1$, then: $P\left[\hat{\sigma}^{T}(\hat{x}^{T}) \neq \tilde{\sigma}^{T}(\tilde{x}^{T})\right] \propto \frac{\epsilon}{1-\beta}$.

Theorem 3.5. Suppose the spiking layers are $u_{th}$-Neighborhood-Finite Distribution at timestep $t$
and the inputs of two corresponding spiking layers with a width $N$ got an error upperbound $\epsilon$ for
each element of spatial input feature vector at any timestep, and they got the same initial temporal
input vector $h^{0}$, if there is no different spiking output at the first $T-1$ timesteps, then the probability
upperbound is proportional to $N \frac{\epsilon}{1-\beta}$. Formally:

For two spiking layers $\hat{\sigma}^{T}$ and $\tilde{\sigma}^{T}$, when $\tilde{h}^{0}=\hat{h}^{0}$ and $\hat{u}^{t}=\hat{h}^{t-1}+\hat{x}^{t}$ is a random variable follows
the $u_{th}$-Neighborhood-Finite Distribution, if $\left\|\tilde{x}_{k}^{t}-\hat{x}_{k}^{t}\right\| \leq \epsilon(k=1,2, \cdots, N ; i=1,2, \cdots, T)$ and
$\hat{\sigma}^{t}(\hat{x}^{t})=\hat{\sigma}^{t}(\tilde{x}^{t})$ for $t=1,2, \cdots, T-1$, then: $P\left[\hat{\sigma}^{T}(\hat{x}^{T}) \neq \tilde{\sigma}^{T}(\tilde{x}^{T})\right] \propto N \frac{\epsilon}{1-\beta}$.

## 4 Proving Lottery Ticket Hypothesis for Spiking Neural Networks

SLTH states that a large randomly initialized neural network has a sub-network that is equivalent to any
well-trained network. In this work, the large initialized network, sub-network, well-trained network
are named large network $G$, equivalent (pruned) network $\tilde{G}$, and target network $\hat{G}$, respectively. We
use the same method to differentiate the weights and other notations in these networks.

We generally follow the theoretical proof approach in [12]. Note, the only unique additional
assumption we made (considering the characteristics of SNN) is that the membrane potentials

follow a class of distribution defined in Definition 3.1, which is easy to satisfy. Specifically, the whole proof in this work is divided into two parts: approximation of linear transformation (Fig. 1c) and approximation of spatio-temporal nonlinear transformation (Fig. 1d), each of which contains three steps. The first part is roughly similar to the proof of LTH in the existing ANN [12]. The second part requires our spatio-temporal probabilistic modeling approach in Section 3, i.e., Theorem 3.5.

Approximation of linear transformation. The existing methods for proving LTH in ANNs all first approach a single weight between two neurons by adding a virtual layer (introducing redundant weights), i.e., Step 1. Different virtual layer modeling methods [12, 15, 16] will affect the width of the network in the final conclusion. All previous proofs introduce two neurons in the virtual layer for approximation. In this work, we exploit only one spiking neuron in the virtual layer for the approximation, given the nature of the binary firing of spiking neurons. We then approximate the weights between one neuron and one layer of neurons (Step 2), and the weights between two layers of neurons (Step 3). Step 2 and Step 3 are the same as previous works.

Step 1: Approximate single weight by a spiking neuron. As shown in Fig. 2, a connection weight in SNNs can be approximated via an additional spiking neuron (i.e, a new virtual layer with only one neuron) with two weights, one connect in and one connect out. We set the two weights of the virtual neuron as $v$ and $\tilde{w}$, and the target connection weight is $\hat{w}$. Consequently, the equivalent function for the target weight can be written as $g(s) = \tilde{w}\tilde{\sigma}(vs)$, where the temporal input of the virtual neuron does not need to be considered. Once $v \geq u_{th}$, the virtual neuron will fire no matter how the temporal input is if spatial input $s = 1$, while not fire if $s = 0$. Thus, the output of the virtual neuron is independent of the temporal input, which is $V_{reset}$ or the decayed membrane potential (neither of which is likely to affect the firing of the virtual neuron). We have: 1) target weight connection: $\hat{g}(s) = \hat{w}s$; 2) equivalent structure: $\tilde{g}(s) = \tilde{w}\tilde{\sigma}(vs)$. If the weight $v$ satisfy $v \geq u_{th}$, the error between the target weight connection and the equivalent structure is

$$
\|\tilde{g}(s)-\hat{w}s\|=\|\tilde{w}s-\hat{w}s\| \leq\|\tilde{w}-\hat{w}\|.\tag{8}
$$

Once the number of initialized weights in the large network $G$ is large enough, it can choose the weights $\tilde{w}$ and $v$ to make the error approximate to 0 by probability convergence. Thus, only one spiking neuron is needed for the virtual layer. Formally, we have Lemma C.1.

Step 2: Layer weights approximation. Based on Lemma C.1, we can extend the conclusion to the case where there is one neuron in layer $l$ and several neurons in layer $l-1$. The lemma is detailed in Lemma C.2.

Step 3: Layer to layer approximation. Finally, we get an approximation between the two layers of spiking neurons. See Lemma C.3 for proof.

![](./images/884073270406546124_2.jpg)

Figure 2: The target weight $\hat{w}$ is simulated in the pruned network $\tilde{G}$ by only one intermediate spiking neuron (this work), due to the spiking neuron only output 0 or 1. In contrast, previous ANN works required two intermediate artificial neurons to simulate the target weights.

Lemma 4.1. Layer to Layer Approximation.
Fix the weight matrix $\hat{\boldsymbol{W}} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}$ and $v_{i j}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B} \in\{0,1\}^{N \times k}$ is the mask for matrix $\tilde{\boldsymbol{W}}, \sum_{i, j}\left\|B_{i j}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}\right\|_{0} \leq N$. Then, let the function of equivalent structure be $\tilde{g}(\boldsymbol{s})=(\tilde{\boldsymbol{W}} \odot \boldsymbol{B}) \tilde{\sigma}(\boldsymbol{V} \boldsymbol{s})$, where input spiking $\boldsymbol{s}$ is a vector that $\boldsymbol{s} \in\{0,1\}^{N}$. Then, $\forall 0<\delta \leq 1, \exists \epsilon>0$ when $k \geq N\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N}{\delta}\right\rceil$ and $C_{t h}=\frac{1-u_{t h}}{2}$, there exists a mask matrix $\boldsymbol{B}$ that $\left\|[\tilde{g}(\boldsymbol{s})]_{i}-[\hat{W} \boldsymbol{s}]_{i}\right\| \leq \epsilon$ w.p at least $1-\delta$.

Approximation of spatio-temporal nonlinear transformation. Since the activation functions in ANNs satisfy the Lipschitz condition, the input error can control the output error of the activation function. In contrast, the output error is not governed by the input error when the membrane potential of a spiking neuron is at a step position. To break this limitation, in Step 4, we combine the probabilistic modeling method in Theorem 3.5 to analyze the probability of consistent output


(only 0 or 1) of spiking layers. Then, in Step 5, we generalize the conclusions in Step 4 to the entire network regardless of the temporal input. Finally, we consider the dynamics of SNNs in the temporal dimension in Step 6. Specifically, we can denote a SNN as follow:

$$
G^{t}(\boldsymbol{S})=G^{t, L} \circ G^{t, L-1} \circ \cdots \circ G^{t, 2} \circ G^{t, 1}(\boldsymbol{S}),
\tag{9}
$$

where $\boldsymbol{S} \in\{0,1\}^{N \times T}$ is the spatial input of the entire network at all timesteps. $G^{t, l}(\cdot)$ represents the function of network $G$ at $t$-th timestep and $l$-th layer, when considering the specific temporal input, it can be written as:

$$
G^{t, l}(\boldsymbol{S})=\sigma_{\boldsymbol{W}^{l}}^{t, l} \boldsymbol{S}_{1: t-1, l}\left(\boldsymbol{W}^{l} \boldsymbol{S}^{t, l-1}\right),
\tag{10}
$$

where spatial input at $t$-th timestep is $\boldsymbol{S}^{t, l-1} \in\{0,1\}^{N}$, $\sigma$ denotes the activation function of spiking layer, its subscript $\boldsymbol{S}^{1: t-1, l}$ indicates that the membrane potential at $t$-th timestep is related to the spatial inputs at all previous timesteps. We are not concerned with these details in probabilistic modeling, thus they do not cause confusion and are omitted in the remaining chapters.

Step 4: Layer spiking activation approximation. Compared to the Step 3, we here include spiking activations to evaluate the probability that two different spiking layers (same input, different weights) have the same output. See Lemma C.4 for proof

Lemma 4.2. Layer Spiking Activation Approximation. Fix the weight matrix $\hat{\boldsymbol{W}} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}$ and $v_{i j}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B} \in\{0,1\}^{N \times k}$ is the mask for matrix $\boldsymbol{V}$, $\sum_{i, j}\left\|B_{i j}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}\right\|_{0} \leq N$. Then, let the function of equivalent structure be $\tilde{g}(\boldsymbol{s})=\tilde{\sigma}((\tilde{\boldsymbol{W}} \odot \boldsymbol{B}) \tilde{\sigma}(\boldsymbol{V} \boldsymbol{s}))$, where input spiking $\boldsymbol{s}$ is a vector that $\boldsymbol{s} \in\{0,1\}^{N}$. $C$ is the constant depending on the supremum probability density of the dataset of the network. Then, $\forall \delta, \exists \epsilon$ when $k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2}}{\delta-N C \epsilon}\right\rceil$, there exists a mask matrix $\boldsymbol{B}$ that $\|\tilde{g}(\boldsymbol{s})-\hat{\sigma}(\hat{\boldsymbol{W}} \boldsymbol{s})\|=0$ w.p at least $1-\delta$.

Step 5: Network approximation $(T=1)$. The lemma is generalized to the whole SNN in Lemma C.5.

Step 6: Network approximation $(T>1)$. If the output of two SNNs in the first $T-1$ timestep is consistent, and we assume that the error of spatial input features at all timesteps has an upperbound, there is a certain probability that the output of the two SNNs in timestep $T$ is also the same. See detail proof in Theorem C.6 and detail theorem as follow Theorem 4.3:

Theorem 4.3. All Steps Approximation. Fix the weight matrix $\hat{\boldsymbol{W}}^{l} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V}^{l} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}}^{l} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}^{l}$ and $v_{i j}^{l}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B}^{l} \in\{0,1\}^{k \times N}$ is the mask for matrix $\boldsymbol{V}^{l}$, $\sum_{i, j}\left\|B_{i j}^{l}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}^{l}\right\|_{0} \leq N$. Then, let the function of equivalent network at timestep $t$ be $\tilde{G}^{t}(\boldsymbol{S})=\tilde{G}^{t, L} \circ \tilde{G}^{t, L-1} \circ \cdots \circ \tilde{G}^{t, 1}(\boldsymbol{S})$ and $\tilde{G}^{t, l}=\tilde{\sigma}^{t}\left(\left(\tilde{\boldsymbol{W}}^{l} \odot \boldsymbol{B}^{l}\right) \tilde{\sigma}^{t}\left(\boldsymbol{V}^{l} \boldsymbol{S}^{t}\right)\right)$, where input spiking $\boldsymbol{S}$ is a tensor that $\boldsymbol{S} \in\{0,1\}^{N \times T}$. And the target network at timestep $t$ is $\hat{G}^{t}(\boldsymbol{S})=\hat{G}^{t, L} \circ \hat{G}^{t, L-1} \circ \cdots \circ \hat{G}^{t, 1}(\boldsymbol{S})$, where $\hat{G}^{t, l}(\boldsymbol{S})=\hat{\sigma}^{t}\left(\hat{\boldsymbol{W}}^{l} \boldsymbol{S}^{t}\right)$. $l=1,2, \cdots, L, t=1,2, \cdots, T$. $C$ is the constant depending on the supremum probability density of the dataset of the network. Then, $\forall 0<\delta \leq 1, \exists \epsilon>0$ when $k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2} L}{\delta-N C L T \epsilon}\right\rceil$, there exists a mask matrix $\boldsymbol{B}$ that $\|\tilde{G}(\boldsymbol{S})-\hat{G}(\boldsymbol{S})\|=0$, w.p at least $1-\delta$.

## 5 Searching Winning Tickets from SNNs without Weight Training

By applying the top-$k\%$ sub-network searching (edge-popup) algorithm[11], we empirically verified SLTH as the complement of the theoretical proof in Section 4. Spiking neurons do not satisfy the Lipschitz condition, but the ANN technique can still be employed since the surrogate gradient of the SNN during BP training [36]. We first simply introduce the search algorithm, then show the results.

The sub-network search algorithm first proposed by [11] provides scores for every weight as a criterion for sub-network topology choosing. In each step of forward propagation, the top-$k\%$ score weights are chosen while others are masked by 0. It means the network used for classification is actually a sub-network of the original random initialized network. During the BP training process, the scores are randomly initialized as weights at the beginning, and then the gradient of the score is updated while the gradient of weights is not recorded (weights are not changed). The updating rule of scores can be formally expressed in mathematical notations as follow:

$$
s_{u v} \leftarrow s_{u v}+\alpha \frac{\partial \mathcal{L}}{\partial \mathcal{I}_{v}} S_{u} w_{u v}, \tag{11}
$$

Where $s_{uv}$ and $w_{uv}$ denote the score and corresponding weight that connect from spiking neuron $u$ to $v$, respectively, $S_{u} \in\{0,1\}$ indicates the output of neuron $u$, $\mathcal{L}$ and $\mathcal{I}_{v}$ are loss function value of the network and the input value of the neuron $v$, and $\alpha$ is the learning rate.

We apply the edge-popup in SNN-VGG-16 and Res-SNN-19. We set maskable weights for all layers (the first layer and the last layer are included). The hyper-parameter sparsity $k\%$ is set every 0.2 from 0.1 to 0.9. The datasets used to verify are CIFAR10/100. The network structures and other experiment details are shown in Appendix E. As shown in Fig. 3, the existence of good sub-networks is empirically verified, which is highly conformed with our theoretical analysis. The additional timesteps and discontinuous activation complicate the sub-network search problem in SNNs, but good sub-networks still exist without any weight training and can be found by specific algorithms.

## 6 Pruning Criterion Based on Probabilistic Modeling

Discussion. For two spiking neurons, layers, or networks that have the same input but different weights, the proposed probabilistic modeling method can give the probability that the outputs (that is, the spiking firing) of both are the same. We convert the perturbation brought by pruning under the same input into the error of spatial input features, and then analyze its impact on spiking firing. Then, we prove that the lottery ticket hypothesis also holds in SNNs.

The potential of probabilistic modeling goes far beyond that. Probabilistic modeling prompts us to rethink whether the existing pruning methods in SNNs are reasonable, since most methods directly continue the idea and methods of ANN pruning. Firstly, from the view of linear transformation, how to metrics the importance of weights in SNNs? The relationship between the inner state of the artificial neuron before activation and the input is usually not considered in ANN pruning. But we have to take this into account in binary-activated SNN pruning because the membrane potential is related to both weights and inputs. For instance, when the input spike frequency is 0, no matter how large the weight is, it will not affect the result of the linear transformation (Fig. 4). Secondly, from the view of nonlinear transformation, how does pruning affect the output of SNNs? As shown in Fig. 1b, when the membrane potential is in different intervals, the probability that the output of the spiking neuron is changed is also different. Thus, we study this important question theoretically using probabilistic modeling.

Moreover, we can assume that the weights of the network remain unchanged, and add disturbance to the input, then exploit probabilistic modeling to analyze the impact of input perturbations on

![](./images/884073270406546124_3.jpg)

Figure 3: The horizontal and vertical coordinates represent the sparsity and accuracy. The solid line is the untrained sub-network. The dashed line is the trained large network (sparsity=$0\%$).

![](./images/884073270406546124_4.jpg)

the output, i.e., analyze the robustness [45] of the SNNs in a new way. Similarly, probabilistic modeling can also analyze other compression methods in SNNs, such as quantization [46], tensor decomposition [47], etc. The perturbations brought about by these compression methods can be converted into errors in the membrane potential, which in turn affects the firing of spiking neurons.

Pruning criterion design. We design a new criterion to prune weights according to their probabilities of affecting the firing of spiking neurons. Based on Theorem 3.5, for each weight, its influence on the output of the spiking neuron has a probability upperbound $P$, which can be estimated as:

$$
P \approx E\left(\left|u^{\prime}-u\right|\right) \mathcal{N}\left(0 | \mu-u_{t h}, v a r\right), \tag{12}
$$

where $E(|u' - u|)$ is the expectation of the error in the membrane potential brought about by pruning. In this work, it is written as:

$$
E\left(\left|u^{\prime}-u\right|\right)=\frac{E_{a c t}[|w|]|\gamma|}{\sqrt{\sigma_{B}^{2}+\epsilon}}, \tag{13}
$$

where $E_{act}[w]$ is the expectation that a weight is activated (the pre-synaptic spiking neuron outputs a spike). $\gamma$ and $\sigma_{B}$ are a pair of hyper-parameters in Batch Normalization [48], which can also affect the linear transformation of weights, thus we incorporate them into Eq. (13). The detailed derivation of Eq. (13) can be found in Appendix D. Note, in keeping with the notations in classic BN, there is some notation mixed here, but only for Eq. (12) and Eq. (13). Next, following [43, 44], here we suppose the membrane potential follows Gaussian distribution $\mathcal{N}(\mu, var)$. Consequently, $\mathcal{N}(0|\mu - u_{th}, var)$ represents the probability density when the membrane potential is at the threshold $u_{th}$. Finally, for each weight, we have a probability $P$, and we prune according to the size of $P$.

Experiments. Since our subject is to theoretically prove LTH in SNNs, we test the proposed new criteria using methods related to LTH. Note, Eq. (12) is a general pruning criterion and can also be incorporated in traditional non-LTH SNN pruning methods. Specifically, in this work we employ the LTH-based Iterative Magnitude Pruning (IMP) method for prune [10], whose criterion is to prune weights with small absolute values (algorithm details are given in Appendix F). Then, we change the criterion to pruning according to the size of $P$ in Eq. (12).

We re-implement the pipeline network (Res-SNN-19 proposed in [43]) and IMP pruning method on the CIFAR-10/100 [49] datasets. We use the official code provided by authors in [21] for the whole pruning process. Then, we perform rigorous ablation experiments without bells and whistles, i.e., all experiment settings are same, the only change is to regulate the pruning criterion to Eq. (12). This is a IMP pruning based on Spiking Firing, thus we name it SF-IMP-1 (Appendix F). Moreover, the encoding layer (the first convolutional layer) in SNNs has a greater impact on task performance, and we can also allow the weights pruning in the encoding layer to be reactivated [50], called SF-IMP-2 (Appendix F).

![](./images/884073270406546124_5.jpg)

Figure 5: Observations from Iterative Magnitude Pruning (IMP) with the proposed pruning criterion $P$ in eq. (12).

The experimental results are shown in Fig. 5. First, these two approaches perform marginally differently on CIFAR-10/100, with SF-IMP-2 outperforming on CIFAR-10, and SF-IMP-1 performing better on CIFAR-100. Especially, compared with the vanilla IMP algorithm on CIFAR-100, SF-IMP-1 performs very well at all pruning sparsities. But on CIFAR-10, when the sparsity is high, the vanilla IMP looks better. One of the possible reasons is that the design of Eq. (12) still has room for improvement. These preliminary results demonstrate the effectiveness of the pruning criteria designed based on probabilistic modeling, and also lay the foundation for the subsequent design of more advanced pruning algorithms.

## 7 Conclusions

In this work, we aim to theoretically prove that the Lottery Ticket Hypothesis (LTH) also holds in SNNs, where the LTH theory has had a wide influence in traditional ANNs and is often used for pruning. In particular, spiking neurons employ binary discrete activation functions and have complex spatio-temporal dynamics, which are different from traditional ANNs. To address these challenges, we propose a new probabilistic modeling method for SNNs, modeling the impact of pruning on spiking firing, and then prove both theoretically and experimentally that $\text{LTH}^4$ holds in SNNs. We design a new pruning criterion from the probabilistic modeling view and test it in an LTH-based pruning method with promising results. Both the probabilistic modeling and the pruning criterion are general. The former can also be used to analyze the robustness and other network compression methods of SNNs, and the latter can also be exploited in non-LTH pruning methods. In conclusion, we have for the first time theoretically established the link between membrane potential perturbations and spiking firing through the proposed novel probabilistic modeling approach, and we believe that this work will provide new inspiration for the field of SNNs.

## References

[1] Wolfgang Maass. Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9):1659–1671, 1997.

[2] Wulfram Gerstner, Werner M Kistler, Richard Naud, and Liam Paninski. *Neuronal dynamics: From single neurons to networks and models of cognition*. Cambridge University Press, 2014.

[3] Kaushik Roy, Akhilesh Jaiswal, and Priyadarshini Panda. Towards spike-based machine intelligence with neuromorphic computing. *Nature*, 575(7784):607–617, 2019.

[4] Man Yao, Guangshe Zhao, Hengyu Zhang, Yifan Hu, Lei Deng, Yonghong Tian, Bo Xu, and Guoqi Li. Attention spiking neural networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, pages 1–18, 2023.

[5] Mike Davies, Narayan Srinivasa, Tsung-Han Lin, Gautham Chinya, Yongqiang Cao, Sri Harsha Choday, Georgios Dimou, Prasad Joshi, Nabil Imam, Shweta Jain, et al. Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1):82–99, 2018.

[6] Jing Pei, Lei Deng, et al. Towards artificial general intelligence with hybrid tianjic chip architecture. *Nature*, 572(7767):106–111, 2019.

[7] Catherine D Schuman, Shruti R Kulkarni, Maryam Parsa, J Parker Mitchell, Bill Kay, et al. Opportunities for neuromorphic computing algorithms and applications. *Nature Computational Science*, 2(1):10–19, 2022.

[8] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. *Advances in neural information processing systems*, 28, 2015.

[9] Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *J. Mach. Learn. Res.*, 22(241):1–124, 2021.

[10] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2019.

[11] Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, and Mohammad Rastegari. What's hidden in a randomly weighted neural network? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11893–11902, 2020.

[12] Eran Malach, Gilad Yehudai, Shai Shalev-Schwartz, and Ohad Shamir. Proving the lottery ticket hypothesis: Pruning is all you need. In *International Conference on Machine Learning*, pages 6682–6691. PMLR, 2020.

$^4$Specifically and strictly, strong LTH. We get a sub-network that performs well without any training.


[13] Hattie Zhou, Janice Lan, Rosanne Liu, and Jason Yosinski. Deconstructing lottery tickets: Zeros, signs, and the supermask. Advances in neural information processing systems, 32, 2019.

[14] Yulong Wang, Xiaolu Zhang, Lingxi Xie, Jun Zhou, Hang Su, Bo Zhang, and Xiaolin Hu. Pruning from scratch. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12273-12280, 2020.

[15] Laurent Orseau, Marcus Hutter, and Omar Rivasplata. Logarithmic pruning is all you need. Advances in Neural Information Processing Systems, 33:2925-2934, 2020.

[16] Ankit Pensia, Shashank Rajput, Alliot Nagle, Harit Vishwakarma, and Dimitris Papailiopou- los. Optimal lottery tickets via subset sum: Logarithmic over-parameterization is sufficient. Advances in Neural Information Processing Systems, 33:2599-2610, 2020.

[17] Arthur da Cunha, Emanuele Natale, and Laurent Viennot. Proving the strong lottery ticket hypothesis for convolutional neural networks. In International Conference on Learning Repre- sentations, 2022.

[18] Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G Baraniuk, Zhangyang Wang, and Yingyan Lin. Drawing early-bird tickets: Towards more efficient training of deep networks. arXiv preprint arXiv:1909.11957, 2019.

[19] Sharath Girish, Shishira R Maiya, Kamal Gupta, Hao Chen, Larry S Davis, and Abhinav Shrivastava. The lottery ticket hypothesis for object recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 762-771, 2021.

[20] Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, and Michael Carbin. The lottery ticket hypothesis for pre-trained bert networks. Advances in neural information processing systems, 33:15834-15846, 2020.

[21] Youngeun Kim, Yuhang Li, Hyoungseob Park, Yeshwanth Venkatesha, Ruokai Yin, and Priyadarshini Panda. Exploring lottery ticket hypothesis in spiking neural networks. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022, pages 102-120, Cham, 2022. Springer Nature Switzerland.

[22] Lei Deng, Yujie Wu, Xing Hu, Ling Liang, Yufei Ding, Guoqi Li, and et al. Rethinking the performance comparison between snns and anns. Neural Networks, 121:294-307, 2020.

[23] Man Yao, Huanhuan Gao, Guangshe Zhao, Dingheng Wang, Yihan Lin, Zhaoxu Yang, and Guoqi Li. Temporal-wise attention spiking neural networks for event streams classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10221-10230, October 2021.

[24] Arjun Rao, Philipp Plank, Andreas Wild, and Wolfgang Maass. A long short-term memory for ai applications in spike-based neuromorphic hardware. Nature Machine Intelligence, 4(5):467-479, 2022.

[25] Mike Davies, Andreas Wild, Garrick Orchard, Yulia Sandamirskaya, Gabriel A Fonseca Guerra, Prasad Joshi, Philipp Plank, and Sumedh R Risbud. Advancing neuromorphic computing with loihi: A survey of results and outlook. Proceedings of the IEEE, 109(5):911-934, 2021.

[26] Emre O Neftci, Bruno U Pedroni, Siddharth Joshi, Maruan Al-Shedivat, and Gert Cauwenberghs. Stochastic synapses enable efficient brain-inspired learning machines. Frontiers in neuroscience, 10:241, 2016.

[27] Nitin Rathi, Priyadarshini Panda, and Kaushik Roy. Stdp-based pruning of connections and weight quantization in spiking neural networks for energy-efficient recognition. IEEE Transac- tions on Computer-Aided Design of Integrated Circuits and Systems, 38(4):668-677, 2018.

[28] Thao NN Nguyen, Bharadwaj Veeravalli, and Xuanyao Fong. Connection pruning for deep spiking neural networks with on-chip learning. In International Conference on Neuromorphic Systems 2021, pages 1-8, 2021.

[29] Yuhan Shi, Leon Nguyen, Sangheon Oh, Xin Liu, and Duygu Kuzum. A soft-pruning method applied during training of spiking neural networks for in-memory computing applications. Frontiers in neuroscience, 13:405, 2019.

[30] Wenzhe Guo, Mohammed E Fouda, Hasan Erdem Yantir, Ahmed M Eltawil, and Khaled Nabil Salama. Unsupervised adaptive weight pruning for energy-efficient neuromorphic systems. Frontiers in Neuroscience, 14:598876, 2020.

[31] Souvik Kundu, Gourav Datta, Massoud Pedram, and Peter A Beerel. Spike-thrift: Towards energy-efficient deep spiking neural networks by limiting spiking activity via attention-guided compression. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3953-3962, 2021.

[32] David Kappel, Stefan Habenschuss, Robert Legenstein, and Wolfgang Maass. Network plasticity as bayesian inference. PLoS computational biology, 11(11):e1004485, 2015.

[33] Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, and Wolfgang Maass. Long short-term memory and learning-to-learn in networks of spiking neurons. Advances in neural information processing systems, 31, 2018.

[34] Yanqi Chen, Zhaofei Yu, Wei Fang, Tiejun Huang, and Yonghong Tian. Pruning of deep spiking neural networks through gradient rewiring. In IJCAI, 2021.

[35] Yanqi Chen, Zhaofei Yu, Wei Fang, Zhengyu Ma, Tiejun Huang, and Yonghong Tian. State tran- sition of dendritic spines improves learning of sparse spiking neural networks. In International Conference on Machine Learning, pages 3701-3715. PMLR, 2022.

[36] Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi. Spatio-temporal backpropagation for training high-performance spiking neural networks. Frontiers in neuroscience, 12:331, 2018.

[37] Emre O Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36(6):51-63, 2019.

[38] Zhaodong Chen, Lei Deng, Bangyan Wang, Guoqi Li, and Yuan Xie. A comprehensive and modularized statistical framework for gradient norm equality in deep neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.

[39] Xingting Yao, Fanrong Li, Zitao Mo, and Jian Cheng. Glif: A unified gated leaky integrate-and- fire neuron for spiking neural networks. In Advances in Neural Information Processing Systems, 2022.

[40] Ahmed Shaban, Sai Sukruth Bezugam, and Manan Suri. An adaptive threshold neuron for recurrent spiking neural networks with nanodevice hardware implementation. Nature Communi- cations, 12(1):1-11, 2021.

[41] Wei Fang, Zhaofei Yu, Yanqi Chen, Timothée Masquelier, Tiejun Huang, and Yonghong Tian. Incorporating learnable membrane time constant to enhance learning of spiking neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2661-2671, 2021.

[42] Peter U Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing. In 2015 International joint conference on neural networks (IJCNN), pages 1-8. ieee, 2015.

[43] Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li. Going deeper with directly-trained larger spiking neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 11062-11070, 2021.

[44] Yufei Guo, Xinyi Tong, Yuanpei Chen, Liwen Zhang, Xiaode Liu, Zhe Ma, and Xuhui Huang. Recdis-snn: Rectifying membrane potential distribution for directly training spiking neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 326-335, 2022.

12

[45] Souvik Kundu, Massoud Pedram, and Peter A Beerel. Hire-snn: Harnessing the inherent robustness of energy-efficient deep spiking neural networks by training with crafted input noise. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5209–5218, 2021.

[46] Lei Deng, Yujie Wu, Yifan Hu, Ling Liang, Guoqi Li, Xing Hu, Yufei Ding, Peng Li, and Yuan Xie. Comprehensive snn compression using admm optimization and activity regularization. *IEEE transactions on neural networks and learning systems*, 2021.

[47] Dingheng Wang, Bijiao Wu, Guangshe Zhao, Man Yao, Hengnu Chen, Lei Deng, Tianyi Yan, and Guoqi Li. Kronecker cp decomposition with fast multiplication for compressing rnns. *IEEE Transactions on Neural Networks and Learning Systems*, pages 1–15, 2021.

[48] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pages 448–456. PMLR, 2015.

[49] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[50] Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. *Advances in neural information processing systems*, 29, 2016.

## A Notations

Table 1: Important Notations.

<table>
  <thead>
    <tr>
      <th>Symbol</th>
      <th>Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$t$</td>
      <td>Timestep index</td>
    </tr>
    <tr>
      <td>$l$</td>
      <td>Layer index</td>
    </tr>
    <tr>
      <td>$i$</td>
      <td>Neuron index</td>
    </tr>
    <tr>
      <td>$u_{i}^{t,l}$</td>
      <td>Membrane potential of spiking neuron</td>
    </tr>
    <tr>
      <td>$h_{i}^{t,l}$</td>
      <td>Temporal input of spiking neuron</td>
    </tr>
    <tr>
      <td>$s_{i}^{t,l}$</td>
      <td>Spatial input of spiking neuron</td>
    </tr>
    <tr>
      <td>$x_{i}^{t,l}$</td>
      <td>Spatial input feature of spiking neuron</td>
    </tr>
    <tr>
      <td>$u_{th}$</td>
      <td>Firing threshold</td>
    </tr>
    <tr>
      <td>$\text{Hea}(\cdot)$</td>
      <td>Heaviside step function</td>
    </tr>
    <tr>
      <td>$V_{reset}$</td>
      <td>Reset membrane potential after firing a spike</td>
    </tr>
    <tr>
      <td>$\beta$</td>
      <td>Decay factor</td>
    </tr>
    <tr>
      <td>$w_{ij}^{l}$</td>
      <td>Weight connect from two spiking neuron</td>
    </tr>
    <tr>
      <td>$u_{th}$</td>
      <td>Firing threshold</td>
    </tr>
    <tr>
      <td>$x$</td>
      <td>Spatial input feature of a spiking neuron without indexes</td>
    </tr>
    <tr>
      <td>$x'$</td>
      <td>Spatial input feature after adding perturbation</td>
    </tr>
    <tr>
      <td>$u$</td>
      <td>Membrane potential of a spiking neuron without indexes</td>
    </tr>
    <tr>
      <td>$u'$</td>
      <td>Membrane potential after adding perturbation</td>
    </tr>
    <tr>
      <td>$\sigma_{x^{1:t-1}}^{t}$</td>
      <td>Membrane potential depends on all previous spatial input features</td>
    </tr>
    <tr>
      <td>$\sigma$</td>
      <td>Shorthand for activation function $\sigma_{x^{1:t-1}}^{t}$</td>
    </tr>
    <tr>
      <td>$N$</td>
      <td>Width of network</td>
    </tr>
    <tr>
      <td>$T$</td>
      <td>Total timestep</td>
    </tr>
    <tr>
      <td>$G$</td>
      <td>Large network</td>
    </tr>
    <tr>
      <td>$G^{l}$</td>
      <td>$l$-th layer of large network</td>
    </tr>
    <tr>
      <td>$\hat{G}$</td>
      <td>Target network</td>
    </tr>
    <tr>
      <td>$\tilde{G}$</td>
      <td>Equivalent network</td>
    </tr>
    <tr>
      <td>$\boldsymbol{s}^{t}$</td>
      <td>Spatial input vector at timestep $t$</td>
    </tr>
    <tr>
      <td>$\boldsymbol{s}$</td>
      <td>Spatial input vector without timestep index</td>
    </tr>
    <tr>
      <td>$\boldsymbol{S}^{t}$</td>
      <td>Spatial input tensor at timestep $t$</td>
    </tr>
    <tr>
      <td>$\boldsymbol{S}^{1:t}$</td>
      <td>All spatial input tensors before timestep $t$</td>
    </tr>
    <tr>
      <td>$\boldsymbol{S}$</td>
      <td>Shorthand for $\boldsymbol{S}^{1:t}$</td>
    </tr>
    <tr>
      <td>$\mathcal{S}$</td>
      <td>Dataset</td>
    </tr>
    <tr>
      <td>$\boldsymbol{v}$</td>
      <td>Virtual layer weight</td>
    </tr>
    <tr>
      <td>$\boldsymbol{V}$</td>
      <td>Weight matrix of Virtual layer</td>
    </tr>
    <tr>
      <td>$\boldsymbol{b}$</td>
      <td>Mask of weight vector</td>
    </tr>
    <tr>
      <td>$\boldsymbol{B}$</td>
      <td>Mask of weight matrix</td>
    </tr>
    <tr>
      <td>$\boldsymbol{W}^{l}$</td>
      <td>Weight matrix of $l$-th layer</td>
    </tr>
    <tr>
      <td>$C_{th}$</td>
      <td>A constant related to the hyper-parameter $u_{th}$</td>
    </tr>
    <tr>
      <td>$C$</td>
      <td>A constant related to the dataset $\mathcal{S}$ and threshold $u_{th}$</td>
    </tr>
    <tr>
      <td>$\mathbb{E}$</td>
      <td>Event</td>
    </tr>
    <tr>
      <td>$k$</td>
      <td>Network width required for LTH establishment</td>
    </tr>
  </tbody>
</table>

## B Proofs of Probabilistic Modeling

**Lemma B.1.** For the probability density function $p(\cdot)$, if it is v-Neighborhood-Finite Distribution, for any $\delta>0$, there exists $\epsilon>0$ that $\int_{v-\epsilon}^{v+\epsilon} p(x)\mathrm{d}x \leq \delta$.

*Proof.* Since distribution $p$ is v-Neighborhood-Finite Distribution, for any $\delta>0$,there exists a constant $\epsilon>0$so that $x \in (v-\epsilon, v+\epsilon)$

$$p_{sup} = \sup_{x} p(x) \geq p(x), \tag{S.1}$$

thus:

$$\int_{v-\epsilon}^{v+\epsilon} p(x)\mathrm{d}x \leq 2\epsilon p_{sup}, \tag{S.2}$$


when $\epsilon \leq \frac{p_{sup}}{2\delta}$, the lemma holds.

$\square$

Lemma B.2. Suppose the spiking neurons are $u_{th}$-Neighborhood-Finite Distribution and the inputs of two corresponding spiking neuron got an error upperbound $\epsilon$, and they got the same inner state $h^{T-1}$, the probability upperbound of different outputs is proportional to $\epsilon$.

For norm, we use $\|\cdot\|_0$ and $\|\cdot\|_2$. The $\|\cdot\|_0$ is used to count the number of non-zero weights while the $\|\cdot\|_2$ is used to measure the distance of outputs.

Formally:

For two spiking neurons $\hat{\sigma}^T$ and $\tilde{\sigma}^T$, when $\tilde{h}^{T-1} = \hat{h}^{T-1}$ and $\hat{u}^T = \hat{h}^{T-1} + \hat{x}^T$ is a random variable follows the $u_{th}$-Neighborhood-Finite Distribution, if $\|\tilde{x}^T - \hat{x}^T\| \leq \epsilon$, then:
$P\left[\hat{\sigma}^T(\hat{x}^T) \neq \tilde{\sigma}^T(\tilde{x}^T)\right] \propto \epsilon$

Proof. Since distribution of $\hat{u}^T = \hat{h}^{T-1} + \hat{x}^T$, denotes as $p$, is $u_{th}$-Neighborhood-Finite Distribution, for any $\delta$, exists $\epsilon$ that $\hat{u} \in (u_{th} - \epsilon, u_{th} + \epsilon)$

$$
p_{sup} = \sup_{\hat{u}^T} p(\hat{u}^T) \geq p(\hat{u}^T), \tag{S.3}
$$

since $\tilde{u}^T = \tilde{h}^{T-1} + \tilde{x}^T = \hat{u}^T + \tilde{x}^T - \hat{x}^T$, $\|\hat{u}^T - \tilde{u}^T\| \leq \epsilon$, then:

$$
\begin{aligned}
&P\left[\hat{\sigma}^T(\hat{x}^T) \neq \tilde{\sigma}^T(\tilde{x}^T)\right] \\
=& P\left[sign(\hat{u}^T - u_{th}) sign(\tilde{u}^T - u_{th}) = -1\right] \\
\leq& \int_{u_{th}-\epsilon}^{u_{th}+\epsilon} p(\hat{u}^T) \mathrm{d}u \leq 2p_{sup}\epsilon,
\end{aligned} \tag{S.4}
$$

thus the lemma holds.

$\square$

Note, for timestep $T$, the membrane potential will follow $p(u^T; S^{1:T}, W)$, where $S^{1:T} \in \{0,1\}^{N \times T}$ is the input from timesteps 1 to $T$, $W \in \mathbf{R}^N$ is the weight vector for the linear combination that shared at each timestep. When we do not emphasize the specific timestep and input data, we use $p(u)$ for brevity.

Lemma B.3. Suppose the spiking neurons are $u_{th}$-Neighborhood-Finite Distribution at timestep $T$ and the inputs of two corresponding spiking neurons got an error upperbound $\epsilon$ at any timestep, and they got the same inner state $h^0$, if there is no different output at the first $T-1$ timesteps, then the probability upperbound is proportional to $\frac{\epsilon}{1-\beta}$.

Formally:

For two spiking neurons $\hat{\sigma}^T$ and $\tilde{\sigma}^T$, when $\tilde{h}^0 = \hat{h}^0$ and $\hat{u}^T = \hat{h}^{T-1} + \hat{x}^T$ is a random variable follows the $u_{th}$-Neighborhood-Finite Distribution, if $\|\tilde{x}^t - \hat{x}^t\| \leq \epsilon$ and $\hat{\sigma}^t(\hat{x}^t) = \tilde{\sigma}^t(\tilde{x}^t)$ for $t=1,2,\cdots,T-1$, then: $P\left[\hat{\sigma}^T(\hat{x}^T) \neq \tilde{\sigma}^T(\tilde{x}^T)\right] \propto \frac{\epsilon}{1-\beta}$.

Proof. For each timestep $i$, we got:

$$
\hat{u}^t = \hat{h}^{t-1} + \hat{x}^t \tag{S.5}
$$

$$
\tilde{u}^t = \tilde{h}^{t-1} + \tilde{x}^t, \tag{S.6}
$$

at timestep $t=1$, we have:

$$
\|\hat{u}^1 - \tilde{u}^1\| \leq \epsilon, \tag{S.7}
$$

if the same output of the former timestep $t-1$ is 1, there will be no error for inner state thus $\hat{h}^{t-1} = \tilde{h}^{t-1}$, then $\|\hat{u}^t - \tilde{u}^t\| \leq \epsilon$, otherwise, the error will be $\|\hat{u}^t - \tilde{u}^t\| \leq \epsilon + \beta\|\hat{u}^{t-1} - \tilde{u}^{t-1}\|$.


Thus, by iterating, here is the upperbound error for membrane potential in the timestep $T$:

$$\left\|\hat{u}^{T}-\tilde{u}^{T}\right\| \leq \sum_{n=0}^{T-1} \beta^{n} \epsilon \leq \sum_{n=0}^{+\infty} \beta^{n} \epsilon \leq \frac{\epsilon}{1-\beta} \tag{S.8}$$

then:

$$P\left[\hat{\sigma}^{T}\left(\hat{x}^{T}\right) \neq \tilde{\sigma}^{T}\left(\tilde{x}^{T}\right)\right] \leq 2 p_{s u p} \frac{\epsilon}{1-\beta}, \tag{S.9}$$

where $p_{sup} = \sup_{\hat{u}^{T}} p(\hat{u}^{T}) \geq p(\hat{u}^{T})$.

Here the lemma holds.
$\square$

Theorem B.4. Suppose the spiking layers are $u_{th}$-Neighborhood-Finite Distribution at timestep $T$ and the inputs of two corresponding spiking layers with a width $N$ got an error upperbound $\epsilon$ for each element of input vectors at any timestep, and they got the same inner state vector $\boldsymbol{h^0}$, if there is no different output at the first $T-1$ timesteps, then the probability upperbound is proportional to $N\frac{\epsilon}{1-\beta}$.

Formally:

For two spiking layers $\hat{\sigma}^{T}$ and $\tilde{\sigma}^{T}$, when $\tilde{\boldsymbol{h^0}} = \hat{\boldsymbol{h^0}}$ and $\hat{\boldsymbol{x}}^{T} = \hat{\boldsymbol{h}}^{T-1} + \hat{\boldsymbol{x}}^{T}$ is a random variable follows the $u_{th}$-Neighborhood-Finite Distribution, if $\|\tilde{x}_{k}^{t}-\hat{x}_{k}^{t}\| \leq \epsilon$ ($k=1,2,\cdots,N; t=1,2,\cdots,T$) and $\hat{\sigma}^{t}(\hat{\boldsymbol{x}}^{\boldsymbol{t}}) = \hat{\sigma}^{t}(\tilde{\boldsymbol{x}}^{\boldsymbol{t}})$ for $t=1,2,\cdots,T-1$, then: $P\left[\hat{\sigma}^{T}(\hat{\boldsymbol{x}}^{T}) \neq \tilde{\sigma}^{T}(\tilde{\boldsymbol{x}}^{T})\right] \propto N\frac{\epsilon}{1-\beta}$.

Proof. Here, we define $p_{sup}$ as the upperbound probability density of all entries at timestep $T$. Thus:

$$p_{sup} = \sup_{\hat{u}_{k}^{T}} p(\hat{u}_{k}^{T}), \tag{S.10}$$

then, according to Lemma 3.4, for any single entry we have:

$$P\left[\hat{\sigma}_{k}^{T}\left(\hat{x}_{k}^{T}\right) \neq \tilde{\sigma}_{k}^{T}\left(\tilde{x}_{k}^{T}\right)\right] \leq \frac{2}{1-\beta} p_{sup} \epsilon \tag{S.11}$$

then, according to the union bound inequality:

$$P\left[\exists k, \hat{\sigma}_{k}^{T}\left(\hat{x}_{k}^{T}\right) \neq \tilde{\sigma}_{k}^{T}\left(\tilde{x}_{k}^{T}\right)\right] \leq N \frac{2}{1-\beta} p_{sup} \epsilon \tag{S.12}$$

$\square$

## C Proofs of Lottery Ticket Hypothesis in SNNs

Lemma C.1. Single Weight Approximation. Fix the weight scalar $\hat{w} \in [-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}]$ which is the connection in target network between two neurons. The equivalent structure is $k$ spiking neurons with $k$ weights $\boldsymbol{v} = [v_1, v_2, \cdots, v_k]^\top$ connect the input and $\tilde{\boldsymbol{w}} = [\tilde{w}_1, \tilde{w}_2, \cdots, \tilde{w}_k]^\top$ connect out. All the weights $\tilde{w}_i$ and $v_i$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{b} = [b_1, b_2, \cdots, b_k]$ is the mask for weight vector $\boldsymbol{v}$, and $\boldsymbol{b} \in \{0,1\}^k$, $\|\boldsymbol{b}\|_0 \leq 1$. Then, let the function of equivalent structure be $\tilde{g}(s) = (\tilde{\boldsymbol{w}} \odot \boldsymbol{b})^\top \tilde{\sigma}(v s)$, where input spiking $s$ is a scalar that $s \in \{0,1\}$. Then, $\forall 0 < \delta \leq 1, \exists \epsilon > 0$ when

$$k \geq \frac{1}{C_{th} \epsilon} \log \frac{1}{\delta}$$

and $C_{th} = \frac{1-u_{th}}{2}$, there exists a mask vector $\boldsymbol{b}$ that

$$\|\tilde{g}(s)-\hat{w}s\| \leq \epsilon$$

w.p at least $1-\delta$.

Proof. For $k$ spiking neurons in equivalent structure, since $\|\boldsymbol{b}\|_0 \leq 1$, only one spiking neuron is active while others are pruned out with their weights. For the chosen active neuron, the weights $\tilde{w}_i$ and $v_i$ should satisfy the following sufficient conditions:

- $\|\tilde{w}_i - \hat{w}_i\| \leq \epsilon$
- $v_i \geq v_{th}$

Since $\tilde{w}, v \sim \mathrm{U}[-1,1]$,
$$
\mathrm{P}\left[\left\|\tilde{w}_{i}-\hat{w}_{i}\right\| \leq \epsilon\right]=\frac{2 \epsilon}{2}=\epsilon \tag{S.13}
$$

$$
\mathrm{P}\left[v_{i} \geq v_{t h}\right]=\frac{1-v_{t h}}{2}=C_{t h} \tag{S.14}
$$

$C_{t h}$ is the constant that $C_{t h}=\frac{1-v_{t h}}{2}$.

The probability for a single spiking neuron that satisfies the condition is:
$$
\begin{aligned}
& \mathrm{P}\left[\left\|\tilde{w}_{i}-\hat{w}_{i}\right\| \leq \epsilon \wedge v_{i} \geq v_{t h}\right] \\
= & \mathrm{P}\left[\left\|\tilde{w}_{i}-\hat{w}_{i}\right\| \leq \epsilon\right] \mathrm{P}\left[v_{i} \geq v_{t h}\right] \\
= & C_{t h} \epsilon
\end{aligned} \tag{S.15}
$$

The probability of not satisfying the condition for all $k$ spiking neurons is:
$$
\begin{aligned}
& \mathrm{P}\left[\forall i \in[k], \neg\left(\left\|\tilde{w}_{i}-\hat{w}_{i}\right\| \leq \epsilon \wedge v_{i} \geq v_{t h}\right)\right] \\
= & \left(1-C_{t h} \epsilon\right)^{k} \leq \exp \left(-k C_{t h} \epsilon\right) \leq \delta
\end{aligned} \tag{S.16}
$$

Thus, when $k \geq \frac{1}{C_{t h} \epsilon} \log \frac{1}{\delta}$, the active spiking neuron satisfies the condition with probability at least $1-\delta$.
$\square$

Lemma C.2. Layer Weights Approximation. Fix the weights vector $\hat{\boldsymbol{w}} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N}$ which is the connection in target network between a layer of spiking inputs and a neuron. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{w}}=\left[\tilde{w}_{1}, \tilde{w}_{2}, \cdots, \tilde{w}_{k}\right]^{\top}$ connect out. All the weights $\tilde{w}_{i}$ and $v_{i j}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{b} \in\{0,1\}^{k}$ is the mask for matrix $\tilde{\boldsymbol{w}},\|\boldsymbol{b}\|_{0} \leq N$. Then, let the function of equivalent structure be $\tilde{g}(\boldsymbol{s})=(\tilde{\boldsymbol{w}} \odot \boldsymbol{b})^{\top} \tilde{\sigma}(V \boldsymbol{s})$, where input spiking $\boldsymbol{s}$ is a vector that $\boldsymbol{s} \in\{0,1\}^{N}$. Then, $\forall 0<\delta \leq 1, \exists \epsilon>0$ when
$$
k \geq N\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N}{\delta}\right\rceil
$$
and $C_{t h}=\frac{1-v_{t h}}{2}$, there exists a mask vector $\boldsymbol{b}$ that
$$
\left\|\tilde{g}(\boldsymbol{s})-\hat{\boldsymbol{w}}^{\top} \boldsymbol{s}\right\| \leq \epsilon
$$
w.p at least $1-\delta$.

Proof. Define the event:
$$
\mathbb{E}_{i, k^{\prime}, \epsilon}=\left\{\forall a \in\left[k^{\prime}\right], \neg\left(\left\|\tilde{w}_{(i-1) k^{\prime}+a}-\hat{w}_{i}\right\| \leq \epsilon \wedge v_{(i-1) k^{\prime}+a, i} \geq v_{t h}\right)\right\}, \tag{S.17}
$$
this event means in a block of approximation structures with $k'$ structures to approximate the connecting weight from the $i$-th input to the output, but no one structure satisfies the $\epsilon$ approximation error condition.

Thus
$$
P\left(\mathbb{E}_{i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right), \tag{S.18}
$$

We totally have $N$ blocks thus $k=N k'$ and for each block, using union bound inequality, we have:
$$
P\left(\bigcup_{i} \mathbb{E}_{i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq \sum_{i} P\left(\mathbb{E}_{i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq N \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right) \leq \delta, \tag{S.19}
$$
thus:
$$
P\left(\left(\bigcup_{i} \mathbb{E}_{i, k^{\prime}, \frac{\epsilon}{N}}\right)^{C}\right) \geq 1-\delta, \tag{S.20}
$$
where
$$
k \geq N\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N}{\delta}\right\rceil \tag{S.21}
$$
then the lemma holds.
$\square$

Lemma C.3. Layer to Layer Approximation. Fix the weight matrix $\hat{\boldsymbol{W}} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}$ and $v_{i j}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B} \in\{0,1\}^{N \times k}$ is the mask for matrix $\tilde{\boldsymbol{W}}, \sum_{i, j}\left\|B_{i j}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}\right\|_{0} \leq N$. Then, let the function of equivalent structure be $\tilde{g}(\boldsymbol{s})=(\tilde{\boldsymbol{W}} \odot \boldsymbol{B}) \tilde{\sigma}(\boldsymbol{V} \boldsymbol{s})$, where input spiking $\boldsymbol{s}$ is a vector that $\boldsymbol{s} \in\{0,1\}^{N}$. Then, $\forall 0<\delta \leq 1, \exists \epsilon>0$ when

$$
k \geq N\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2}}{\delta}\right\rceil
$$

and $C_{t h}=\frac{1-u_{t h}}{2}$, there exists a mask matrix $\boldsymbol{B}$ that

$$
\left\|[\tilde{g}(\boldsymbol{s})]_{i}-[\hat{W} \boldsymbol{s}]_{i}\right\| \leq \epsilon
$$

w.p at least $1-\delta$.

Proof. Define the event:

$$
\mathbb{E}_{j, i, k^{\prime}, \epsilon}=\left\{\forall a \in\left[k^{\prime}\right], \neg\left(\left\|\tilde{w}_{j,(i-1) k^{\prime}+a}-\hat{w}_{j i}\right\| \leq \epsilon \wedge v_{(i-1) k^{\prime}+a, i} \geq u_{t h}\right)\right\}, \tag{S.22}
$$

this event means in a block of approximation structures with $k'$ structures to approximate the connecting weight from the $i$-th input to the $j$-th output, but no one structure satisfies the $\epsilon$ approximation error condition.

Thus

$$
P\left(\mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right), \tag{S.23}
$$

We totally have $N$ blocks thus $k=N k'$ and for each block, using union bound inequality, we have:

$$
P\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq \sum_{j, i} P\left(\mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq N^{2} \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right) \leq \delta, \tag{S.24}
$$

thus:

$$
P\left(\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right)^{C}\right) \geq 1-\delta, \tag{S.25}
$$

where

$$
k \geq N\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2}}{\delta}\right\rceil \tag{S.26}
$$

then the lemma holds.
$\square$

Lemma C.4. Layer Spiking Activation Approximation. Fix the weight matrix $\hat{\boldsymbol{W}} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}$ and $v_{i j}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B} \in\{0,1\}^{N \times k}$ is the mask for matrix $\boldsymbol{V}, \sum_{i, j}\left\|B_{i j}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}\right\|_{0} \leq N$. Then, let the function of equivalent structure be $\tilde{g}(\boldsymbol{s})=\tilde{\sigma}((\tilde{\boldsymbol{W}} \odot \boldsymbol{B}) \tilde{\sigma}(\boldsymbol{V} \boldsymbol{s}))$, where input spiking $\boldsymbol{s}$ is a vector that $\boldsymbol{s} \in\{0,1\}^{N}$. $C$ is the constant depending on the supremum probability density of the dataset of the network. Then, $\forall 0<\delta \leq 1, \exists \epsilon>0$ when

$$
k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2}}{\delta-N C \epsilon}\right\rceil
$$

, there exists a mask matrix $\boldsymbol{B}$ that

$$
\|\tilde{g}(\boldsymbol{s})-\hat{\sigma}(\hat{\boldsymbol{W}} \boldsymbol{s})\|=0
$$

w.p at least $1-\delta$.

Proof. The definition of $\mathbb{E}_{j, i, k^{\prime}, \epsilon}$ follows the proof of LemmaLemma C.3, thus we have:
$$
P\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq \sum_{j, i} P\left(\mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \leq N^{2} \exp \left(k^{\prime} C_{t h} \frac{\epsilon}{N}\right)
\tag{S.27}
$$

And the event $\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right)^{C}$ implies that the error of each channel smaller than $\epsilon$.

According to the proof of theoremTheorem 3.5, the probability upperbound of different output of spiking neurons with the same temporal state at $t=0$ is $2 N p_{s u p} \frac{\epsilon}{1-\beta}$. We define $\mathbb{E}_{fire}$ as the event of different output of corresponding spiking layer.

Then, according to union bound, we have:
$$
\begin{aligned}
&P\left(\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{fire}\right) \\
\leq & \sum_{j, i} P\left(\mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right)+P\left(\mathbb{E}_{fire}\right) \leq N^{2} \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right)+N p_{s u p} \frac{2 \epsilon}{1-\beta} \leq \delta,
\end{aligned}
\tag{S.28}
$$

Let $C=2 p_{sup} \frac{1}{1-\beta}$, then, we have:
$$
k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2}}{\delta-N C \epsilon}\right\rceil
\tag{S.29}
$$

Here, the event $\left(\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{fire}\right)^{C}$ implies that the error of each entry is smaller than $\epsilon$ while the output have no difference.
$$
P\left(\left(\left(\bigcup_{j, i} \mathbb{E}_{j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{fire}\right)^{C}\right) \geq 1-\delta,
\tag{S.30}
$$
the lemma holds.
$\square$

Lemma C.5. All Layers Approximation. Fix the weight matrix $\hat{\boldsymbol{W}}^{l} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V}^{l} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}}^{l} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}^{l}$ and $v_{i j}^{l}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B}^{l} \in\{0,1\}^{k \times N}$ is the mask for matrix $\boldsymbol{V}, \sum_{i, j}\left\|B_{i j}^{l}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}^{l}\right\|_{0} \leq N$.

Then, let the function of equivalent network be $\tilde{G}(\boldsymbol{s})=\tilde{G}^{L} \circ \tilde{G}^{L-1} \circ \cdots \circ \tilde{G}^{1}(\boldsymbol{s})$ and $\tilde{G}^{l}=\tilde{\sigma}\left(\left(\tilde{\boldsymbol{W}}^{l} \odot\right.\right.$ $\left.\left.\boldsymbol{B}^{l}\right) \tilde{\sigma}\left(\boldsymbol{V}^{l} \boldsymbol{s}\right)\right)$, where input spiking $\boldsymbol{s}$ is a vector that $\boldsymbol{s} \in\{0,1\}^{N}$. And the target network is $\hat{G}(\boldsymbol{s})=$ $\hat{G}^{L} \circ \hat{G}^{L-1} \circ \cdots \circ \hat{G}^{1}(\boldsymbol{s})$, where $\hat{G}^{l}(\boldsymbol{s})=\hat{\sigma}\left(\hat{\boldsymbol{W}}^{l} \boldsymbol{s}\right) . l=1,2, \cdots, L$. $C$ is the constant depending on the supremum probability density of the dataset of the network. Then, $\forall 0<delta \leq 1, \exists \epsilon>0$ when
$$
k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2} L}{\delta-N C L \epsilon}\right\rceil,
$$
there exists a mask matrix $\boldsymbol{B}$ that
$$
\|\tilde{G}(\boldsymbol{s})-\hat{G}(\boldsymbol{s})\|=0
$$
w.p at least $1-\delta$.

Proof. We inherit the event expression $\mathbb{E}_{j, i, k^{\prime}, \epsilon}$ and $\mathbb{E}_{fire}$ from the proof of LemmaLemma C.4 with a delight modify. Here we add subscript $l$ to denote the layer of the target network. Thus we have:
$$
\begin{aligned}
&P\left(\left(\bigcup_{l, j, i} \mathbb{E}_{l, j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{fire, l}\right) \\
\leq & \sum_{l, j, i} P\left(\mathbb{E}_{l, j, i, k^{\prime}, \frac{\epsilon}{N}}\right)+\sum_{l} P\left(\mathbb{E}_{fire, l}\right) \leq L N^{2} \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right)+L N C \epsilon \\
\leq & \delta,
\end{aligned}
\tag{S.31}
$$

Thus,
$$
k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2} L}{\delta-N C L \epsilon}\right\rceil, \tag{S.32}
$$
and:
$$
P\left(\left(\left(\bigcup_{l, j, i} \mathbb{E}_{t, j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{f i r e, l}\right)^{C}\right) \geq 1-\delta \tag{S.33}
$$
the lemma holds.
$\square$

Theorem C.6. All Steps Approximation. Fix the weight matrix $\hat{\boldsymbol{W}}^{l} \in\left[-\frac{1}{\sqrt{N}}, \frac{1}{\sqrt{N}}\right]^{N \times N}$ which is the connection in target network between a layer of spiking inputs and the next layer of neurons. The equivalent structure is $k$ spiking neurons with $k \times N$ weights $\boldsymbol{V}^{l} \in \mathbf{R}^{k \times N}$ connect the input and $\tilde{\boldsymbol{W}}^{l} \in \mathbf{R}^{N \times k}$ connect out. All the weights $\tilde{w}_{i j}^{l}$ and $v_{i j}^{l}$ random initialized with uniform distribution $\mathrm{U}[-1,1]$ and i.i.d. $\boldsymbol{B}^{l} \in\{0,1\}^{k \times N}$ is the mask for matrix $\boldsymbol{V}^{l}, \sum_{i, j}\left\|B_{i j}^{l}\right\|_{0} \leq N^{2}, \sum_{j}\left\|B_{i j}^{l}\right\|_{0} \leq N$. Then, let the function of equivalent network at timestep $t$ be $\tilde{G}^{t}(\boldsymbol{S})=\tilde{G}^{t, L} \circ \tilde{G}^{t, L-1} \circ \cdots \circ \tilde{G}^{t, 1}(\boldsymbol{S})$ and $\tilde{G}^{t, l}=\tilde{\sigma}^{t}\left(\left(\tilde{\boldsymbol{W}}^{l} \odot \boldsymbol{B}^{l}\right) \tilde{\sigma}^{t}\left(\boldsymbol{V}^{l} \boldsymbol{S}^{t}\right)\right)$, where input spiking $\boldsymbol{S}$ is a tensor that $\boldsymbol{S} \in\{0,1\}^{N \times T}$. And the target network at timestep $t$ is $\hat{G}^{t}(\boldsymbol{S})=\hat{G}^{t, L} \circ \hat{G}^{t, L-1} \circ \cdots \circ \hat{G}^{t, 1}(\boldsymbol{S})$, where $\hat{G}^{t, l}=\hat{\sigma}^{t}\left(\hat{W}^{l} \boldsymbol{S}^{t}\right)$. $l=1,2, \cdots, L, t=1,2, \cdots, T$. $C$ is the constant depending on the supremum probability density of the dataset of the network. Then, $\forall 0<\delta \leq 1, \exists \epsilon>0$ when
$$
k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2} L}{\delta-N C L T \epsilon}\right\rceil,
$$
there exists a mask matrix $\boldsymbol{B}$ that
$$
\|\tilde{G}(\boldsymbol{S})-\hat{G}(\boldsymbol{S})\|=0
$$
w.p at least $1-\delta$.

Proof. We inherit the event expression $\mathbb{E}_{l, j, i, k^{\prime}, \epsilon}$ and $\mathbb{E}_{f i r e, l}$ from the proof of LemmaLemma C.5 with a delight modify. Here we add subscript $t$ to $\mathbb{E}_{f i r e, l}$ to denote the layer of the target network. Thus we have:
$$
\begin{aligned}
& P\left(\left(\bigcup_{t, l, j, i} \mathbb{E}_{t, l, j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{f i r e, t, l}\right) \\
\leq & \sum_{t, l, j, i} P\left(\mathbb{E}_{t, l, j, i, k^{\prime}, \frac{\epsilon}{N}}\right)+\sum_{t, l} P\left(\mathbb{E}_{f i r e, t, l}\right) \\
\leq & L N^{2} \exp \left(-k^{\prime} C_{t h} \frac{\epsilon}{N}\right)+T L N C \epsilon \\
\leq & \delta,
\end{aligned} \tag{S.34}
$$

Thus,
$$
k \geq N^{2}\left\lceil\frac{N}{C_{t h} \epsilon} \log \frac{N^{2} L}{\delta-N C T L \epsilon}\right\rceil, \tag{S.35}
$$
and:
$$
P\left(\left(\left(\bigcup_{l, j, i} \mathbb{E}_{t, l, j, i, k^{\prime}, \frac{\epsilon}{N}}\right) \bigcup \mathbb{E}_{f i r e, t, l}\right)^{C}\right) \geq 1-\delta \tag{S.36}
$$
the lemma holds.
$\square$

## D Derivation of Equation (13)

We design a new criterion to prune weights according to their probabilities of affecting the output of spiking neurons. Based on the probabilistic modeling (Theorem 3.5), for each weight, its influence on the output of the spiking neuron has a probability upperbound $P$, which can be estimated as:
$$
P \approx E\left(\left|u^{\prime}-u\right|\right) \mathcal{N}\left(0 | \mu-u_{t h}, v a r\right), \tag{S.37}
$$

20

where $E(|u' - u|)$ is the expectation of the error in the membrane potential brought about by pruning (i.e., effect of weights on linear transformations). In our method, it is written as:

$$
E(|u' - u|) = \frac{E_{act}[w]\gamma}{\sqrt{\sigma_{B}^{2} + \epsilon}}, \tag{S.38}
$$

where $E_{act}[w]$ is the expectation that a weight is activated (the pre-synaptic spiking neuron outputs a spike), $\gamma$ and $\sigma_{B}$ are a pair of hyper-parameters in Batch Normalization [48]. The detailed derivation of Eq. (S.38) can be found in Appendix D. Next, following [43, 44], here we suppose the membrane potential follows Gaussian distribution $\mathcal{N}(\mu, var)$. Consequently, $\mathcal{N}(0|\mu - u_{th}, var)$ represents the probability density when the membrane potential is at the threshold $u_{th}$ (i.e., effect of weights on binary nonlinear transformations). Finally, for each weight, we have a probability $P$, and we prune according to the size of $P$ (from small to large).

Now we explain that how we get the Eq. (S.38), i.e., Eq. (13) in the main text. The expression of Batch Normalization (BN) can be written as:

$$
\widehat{x}_{i} \leftarrow \frac{x_{i} - \mu_{B}}{\sqrt{\sigma_{B}^{2} + \epsilon}}, \tag{S.39}
$$

$$
y_{i} \leftarrow \gamma \widehat{x}_{i} + \beta \equiv \mathrm{BN}_{\gamma,\beta}(x_{i}). \tag{S.40}
$$

When a spike $s$ passes through a weight in convolution layer and the BN layer, the output is:

$$
\frac{sw\gamma}{\sqrt{\sigma_{B}^{2} + \epsilon}} + \left( \frac{-\mu_{B}\gamma}{\sqrt{\sigma_{B}^{2} + \epsilon}} + \beta \right). \tag{S.41}
$$

We know that these two operations can be regard as linear transformation, where the input spike $s$ multiplies the scaling weight $\frac{w\gamma}{\sqrt{\sigma_{B}^{2} + \epsilon}}$ and add a bias ( $\frac{-\mu_{B}\gamma}{\sqrt{\sigma_{B}^{2} + \epsilon}} + \beta$ ). Since the bias will not change after changing the weight, the error of membrane potential only related to the scaling weight $\frac{w\gamma}{\sqrt{\sigma_{B}^{2} + \epsilon}}$. Thus, we design $E(|u' - u|)$ as follows:

$$
E(|u' - u|) = E\left( \frac{|sw\gamma|}{\sqrt{\sigma_{B}^{2} + \epsilon}} \right) = \frac{|\gamma|}{\sqrt{\sigma_{B}^{2} + \epsilon}}E(|sw|) = \frac{E_{act}[|w|]|\gamma|}{\sqrt{\sigma_{B}^{2} + \epsilon}}. \tag{S.42}
$$

Note, in keeping with the notations in classic BN [48], there is some notation mixed here, but only for the derivation of Eq. (13).

**Algorithm 1 Rewind Iterative Magnitude Pruning (IMP).**
Input: pruning rate $p$, iterations $K$, the rewind epoch $R$, max training epoch $N$
Train network for $R$ epochs.
Save the parameters as rewind parameters $\theta_{rewind}$
Train network for $n$ epochs.
for $k=1$ to $K$ do
  Prune network by remove $p\%$ of the lowest magnitude nonzero weights of the network base on a metrics, get the mask $m_{k}$
  Reload the rewind parameters and mask them. $(\theta_{rewind} \odot m_{k})$
  Train the network for $N$ epochs.
end for

## E Implementation Details of Sub-Network Search
All training methods strictly follow experiments in [21], and the sub-network search module follows [11]. For the two datasets of CIFAR10/100, we use cosine learning scheduling and SGD optimizer with momentum 0.9 and weight decay 5e-4, and the total number of training epochs is 300. The learning rate is set to 0.3. batch size is set to 128. The timestep $T$ of SNN is 5. We simply replace all the weight modules in network with the corresponding sub-network search modules in hidden-networks/blob/master/simple_mnist_example.py


## F Iterative Magnitude Pruning (IMP) and the proposed SF-IMP Algorithms

Among the LTH-based pruning algorithms, the Iterative Magnitude Pruning (IMP) method has good performance. In IMP, the parameter $\theta \in \mathbf{R}^n$ of network $f(x; \theta)$ is pruned iteratively. (For the sake of briefness and convention of the symbols, unless otherwise specified, the meanings of the symbols in this chapter have nothing to do with the previous chapters.) We set $K$ iterations. In the $k$-th iteration, we first train the network till convergence, then prune $p\%$ of nonzero parameters of $\theta_{trained} \odot m_{k-1}$ by mask $m_k \in \{0,1\}^n$. Then reinitialize the network with parameter $\theta_{init} \odot m_k$ and repeat the operations until the $K$-th iteration ends.

For SF-IMP-1, we change the criterion from considering magnitude to:

$$
\frac{E_{act}[||w_{ij}||\left|\gamma_{j}\right|}{\sqrt{\sigma_{\mathcal{B} j}^{2}+\epsilon}} \mathcal{N}(0|\mu_{j}-u_{t h}, var_{j}), \tag{S.43}
$$

Here, $i$ is the index of the input channel while $j$ is the index of the output channel. We statistic the firing frequency for each input channel to evaluate $E_{act}[||w_{ij}||]$, and other parameters $\gamma_{j}, \sigma_{\mathcal{B} j}, \mu_{j}, var_{j}$ are statistic by every output channel. However, the encoding layer and the Fully Connected (FC) layer are not suited for this algorithm, thus we keep the magnitude criterion for these two layers.

For SF-IMP-2, since the sparsity of the encoding layer and the FC layer have a great influence on pruning, we apply a dynamic strategy for these layers [50]. Specifically, we only mask the weight when computing loss, while when updating weight and re-initialize weight, the mask is not used.