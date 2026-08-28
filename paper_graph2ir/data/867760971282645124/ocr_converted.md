# Exploring Lottery Ticket Hypothesis in Spiking Neural Networks

［#1］
Youngeun Kim, Yuhang Li, Hyoungseob Park, Yeshwanth Venkatesha, Ruokai Yin, and Priyadarshini Panda

［#2］
Department of Electrical Engineering
Yale University
New Haven, CT, USA
{youngeun.kim, yuhang.li, hyoungseob.park, yeshwanth.venkatesha, ruokai.yin, priya.panda}@yale.edu

［#3］
Abstract. Spiking Neural Networks (SNNs) have recently emerged as a new generation of low-power deep neural networks, which is suitable to be implemented on low-power mobile/edge devices. As such devices have limited memory storage, neural pruning on SNNs has been widely explored in recent years. Most existing SNN pruning works focus on shallow SNNs (2~6 layers), however, deeper SNNs ($\geq$16 layers) are proposed by state-of-the-art SNN works, which is difficult to be compatible with the current SNN pruning work. To scale up a pruning technique towards deep SNNs, we investigate Lottery Ticket Hypothesis (LTH) which states that dense networks contain smaller subnetworks (i.e., winning tickets) that achieve comparable performance to the dense networks. Our studies on LTH reveal that the winning tickets consistently exist in deep SNNs across various datasets and architectures, providing up to 97% sparsity without huge performance degradation. However, the iterative searching process of LTH brings a huge training computational cost when combined with the multiple timesteps of SNNs. To alleviate such heavy searching cost, we propose Early-Time (ET) ticket where we find the important weight connectivity from a smaller number of timesteps. The proposed ET ticket can be seamlessly combined with a common pruning techniques for finding winning tickets, such as Iterative Magnitude Pruning (IMP) and Early-Bird (EB) tickets. Our experiment results show that the proposed ET ticket reduces search time by up to 38% compared to IMP or EB methods. Code is available at Github.

［#4］
Keywords: Spiking Neural Networks, Neural Network Pruning, Lottery Ticket Hypothesis, Neuromorphic Computing

## 1 Introduction

［#5］
Spiking Neural Networks (SNNs) [63,11,74,75,37,20,76,34] have gained significant attention as promising low-power alternative to Artificial Neural Networks (ANNs). Inspired by the biological neuron, SNNs process visual information through discrete spikes over multiple timesteps. This event-driven behavior of

［#6］
![](./images/867760971282645124_1.jpg)

［#7］
Fig. 1. Accuracy and Search time comparison of various pruning methods on SNNs including Iterative Magnitude Pruning (IMP) [21], Early-Bird (EB) ticket [80], Early-Time (ET) ticket (ours), Transferred winning ticket from ANN (TT), Random pruning, and SNIP [41]. We use VGG16 on the CIFAR10 dataset and show mean/standard deviation from 5 random runs.

［#8］
SNNs brings huge energy-efficiency, therefore they are suitable to be implemented on low-power neuromorphic chips [1,13,23,57] which compute spikes in an asynchronous manner. However, as such devices have limited memory storage, neural pruning can be one of the essential techniques by reducing memory usage for weight parameters, thus promoting the practical deployment.

［#9］
Accordingly, researchers have made certain progress on the pruning technique for SNNs. Neftci et al. [55] and Rathi et al. [60] prune weight connections of SNNs using a predefined threshold value. Guo et al. [25] propose an unsupervised online adaptive weight pruning algorithm that dynamically removes non-critical weights over time. Moreover, Shi et al. [65] present a soft-pruning method where both weight connections and a pruning mask are trained during training. Recently, Deng et al. [14] adapt ADMM optimization tool with sparsity regularization to compress SNNs. Chen et al. [9] propose a gradient-based rewiring method for pruning, where weight values and connections are jointly optimized. However, although existing SNN pruning works significantly increase weight sparsity, they focus on a shallow architecture such as 2-layer MLP [55,60,25] or 6∼7 convolutional layers [14,9]. Such pruning techniques are difficult to scale up to the recent state-of-the-art deep SNN architectures where the number of parameters and network depth is scaled up [43,44,20,61,83].

［#10］
In this paper, we explore sparse deep SNNs based on the recently proposed Lottery Ticket Hypothesis (LTH) [21]. They assert that an over-parameterized neural network contains sparse sub-networks that achieve a similar or even better accuracy than the original dense networks. The discovered sub-networks and their corresponding initialization parameters are referred as winning tickets. Based on LTH, a line of works successfully have shown the existence of winning tickets across various tasks such as standard recognition task [80,22,24], reinforcement learning [69,81], natural language processing [4,7,53], and generative model [31]. Along the same line, our primary research objective is to investigate the existence of winning tickets in deep SNNs which has a different type of neuronal dynamics from the common ANNs.

［#11］
Furthermore, applying LTH on SNNs poses a practical challenge. In gen-eral, finding winning tickets requires **Iterative Magnitude Pruning (IMP)** where a network becomes sparse by repetitive initialization-training-pruning op-erations [21]. Such iterative training process goes much slower with SNNs where multiple feedforward steps (*i.e.*, timesteps) are required. To make LTH with SNNs more practical, we explore several techniques for reducing search costs. We first investigate **Early-Bird (EB) ticket** phenomenon [80] that states the sub-networks can be discovered in the early-training phase. We find that SNNs contain EB tickets across various architectures and datasets. Moreover, SNNs convey information through multiple timesteps, which provides a new dimen-sion for computational cost reduction. Focusing on such temporal property, we propose **Early-Time (ET) ticket** phenomenon: winning tickets can be drawn from the network trained from a smaller number of timesteps. Thus, during the search process, SNNs use a smaller number of timesteps, which can significantly reduce search costs. As ET ticket is a temporal-crafted method, our proposed ET ticket can be combined with IMP [21] and EB tickets [80]. Furthermore, we also explore whether **ANN winning ticket can be transferred to SNN** since search cost at ANN is much cheaper than SNN. Finally, we examine **pruning at initialization** method on SNNs, *i.e.*, SNIP [41], which finds the winning tickets from backward gradients at initialization. In Fig. 1, we compare the accuracy and search time of the above-mentioned pruning methods.

［#12］
In summary, we explore LTH for SNNs by conducting extensive experiments on two representative deep architectures, *i.e.*, VGG16 [67] and ResNet19 [28], on four public datasets including SVHN [56], Fashion-MNIST [77], CIFAR10 [36] and CIFAR100 [36]. Our key observations are as follows:

［#13］
- We confirm that Lottery Ticket Hypothesis is valid for SNNs.
- We found that IMP [21] discovers winning tickets with up to 97% sparsity. However, IMP requires over 50 hours on GPU to find sparse ($\geq 95\%$) SNNs.
- EB tickets [80] discover sparse SNNs in 1 hour on GPUs, which reduces search cost significantly compared to IMP. Unfortunately, they fail to detect winning tickets over 90% sparsity.
- Applying ET tickets to both IMP and EB significantly reduces search time by up to 41% while showing $\leq 1\%$ accuracy drop at less than 95% sparsity.
- Winning ticket obtained from ANN can be transferable at less than 90% sparsity. However, huge accuracy drop is incurred at high sparsity levels ($\geq 95\%$), especially for complex datasets such as CIFAR100.
- Pruning at initialization method [41] fails to discover winning tickets in SNNs owing to the non-differentiability of spiking neurons.

## 2 Related Work

### 2.1 Spiking Neural Networks

［#14］
Spiking Neural Networks (SNNs) transfer binary and asynchronous information through networks in a low-power manner [63,11,12,52,64,43,68,78]. The major

［#14］
difference of SNNs from standard ANNs is using a Leak-Integrate-and-Fire (LIF) neuron [30] as a non-linear activation. The LIF neuron accumulates incoming spikes in membrane potential and generates an output spike when the neuron has a higher membrane potential than a firing threshold. Such integrate-and-fire be- havior brings a non-differentiable input-output transfer function where standard backpropagation is difficult to be applied [54]. Recent SNN works circumvent non-differentiable backpropagation problem by defining a surrogate function for LIF neurons when calculating backward gradients [40,39,54,66,74,72,45,32,73]. Our work is also based on a gradient backpropagation method with a surrogate function (details are provided in Supplementary A). The gradient backpropaga- tion methods enable SNNs to have deeper architectures. For example, adding Batch Normalization (BN) [29] to SNNs [38,33,83] improves the accuracy with deeper architectures such as VGG16 and ResNet19. Also, Fang et al. [20] revisit deep residual connection for SNNs, showing higher performance can be achieved by adding more layers. Although the recent state-of-the-art SNN architecture goes deeper [83,20,15], pruning for such networks has not been explored. We as- sert that showing the existence of winning tickets in deep SNNs brings a practical advantage to resource-constrained neuromorphic chips and edge devices.

### 2.2 Lottery Ticket Hypothesis
［#15］
Pruning has been actively explored in recent decades, which compresses a huge model size of the deep neural networks while maintaining its original performance [27,26,71,47,42]. In the same line of thought, Frankle & Carbin [21] present Lot- tery Ticket Hypothesis (LTH) which states that an over-parameterized neural network contains sparse sub-networks with similar or even better accuracy than the original dense networks. They search winning tickets by iterative magnitude pruning (IMP). Although IMP methods [84,2,18,5,46] provide higher perfor- mance compared to existing pruning methods, such iterative training-pruning- retraining operations require a huge training cost. To address this, a line of work [50,51,16,7] discovers the existence of transferable winning tickets from the source dataset and successfully transfers it to the target dataset, thus eliminat- ing search cost. Further, You et al. [80] introduce early bird ticket hypothesis where they conjecture winning tickets can be achieved in the early training phase, reducing the cost for training till convergence. Recently, Zhang et al. [82] discover winning tickets with a carefully selected subset of training data, called pruning-aware critical set. To completely eliminate training costs, sev- eral works [41,70] present searching algorithms from initialized networks, which finds winning tickets without training. Unfortunately, such techniques do not show comparable performance with the original IMP methods, thus mainstream LTH leverages IMP as a pruning scheme [24,82,51]. Based on IMP technique, researchers found the existence of LTH in various applications including visual recognition tasks [24], natural language processing [4,7,53], reinforcement learn- ing [69,81], generative model [31], low-cost neural network ensembling [46], and improving robustness [8]. Although LTH has been actively explored in ANN do- main, LTH for SNNs is rarely studied. It is worth mentioning that Martinelli et

［#16］
![](./images/867760971282645124_2.jpg)

［#17］
(a) Iterative Magnitude Pruning (b) Early-Bird Ticket (c) Early-Time Ticket

［#18］
Fig. 2. Illustration of the concept of Iterative Magnitude Pruning (IMP), Early-Bird (EB) ticket, and the proposed Early-Time (ET) ticket applied to EB ticket. Our ET ticket reduces search cost for winning tickets by using a smaller number of timesteps during the search process. Note, ET can be applied to both IMP and EB, here we only illustrate ET with EB.

［#19］
al. [49] apply LTH to two-layer SNN on voice activity detection task. Different from the previous work, our work shows the existence of winning tickets in much deeper networks such as VGG16 and ResNet19, which shows state-of-the-art performance on the image recognition task. We also explore Early-Bird ticket [80], SNIP [41], transferability of winning tickets from ANN, and propose a new concept of winning tickets in the temporal dimension.

## 3 Drawing Winning Tickets from SNN

［#20］
In this section, we present the details of pruning methods based on LTH, explored in our experiments. We first introduce the LTH [21] and Early-Bird (EB) Ticket [80]. Then, we propose Early-Time (ET) Tickets where we reduce search cost in the temporal dimension of SNNs. We illustrate the overall search process of each method in Fig. 2.

### 3.1 Lottery Ticket Hypothesis

［#21］
In LTH [21], winning tickets are discovered by iterative magnitude pruning (IMP). The whole pruning process of LTH goes through $K$ iterations for the target pruning ratio $p^K\%$. Consider a randomly initialized dense network $f(x;\theta)$ where $\theta \in \mathbb{R}^n$ is the network parameter weights. For the first iteration, initialized network $f(x;\theta)$ is trained till convergence, then mask $m_1 \in \{0,1\}^n$ is generated by removing $p\%$ lowest absolute value weights parameters. Given a pruning mask $m_1$, we can define subnetworks $f(x;\theta \odot m_1)$ by removing some connections. For the next iteration, we reinitialize the network with $\theta \odot m_1$, and prune $p\%$ weights when the network is trained to convergence. This pruning process is repeated for $K$ iterations. In our experiments, we set $p$ and $K$ to $25\%$ and 15, respectively. Also, Frankle et al. [22] present Late Rewinding, which rewinds the network to the weights at epoch $i$ rather than initialization. This enables IMP to discover the winning ticket with less performance drop in a high

［#21］
sparsity regime by providing a more stable starting point. We found that *Late Rewinding* shows better performance than the original IMP in deep SNNs (see Supplementary B). Throughout our paper, we apply *Late Rewinding* to IMP for experiments where we rewind the network to epoch 20.

### 3.2 Early-Bird Tickets
［#22］
Using IMP for finding lottery ticket incurs huge computational costs. To address this, You *et al.* [80] propose an efficient pruning method, called Early-bird (EB) tickets, where they show winning tickets can be discovered at an early training epoch. Specifically, at searching iteration $k$, they obtain a mask $m_k$ and measure the mask difference between the current mask $m_k$ and the previous masks within time window $q$, *i.e.*, $m_{k-1}, m_{k-2}, ..., m_{k-q}$. If the maximum difference is less than hyperparamter $\tau$, they stop training and use $m_k$ as an EB ticket. As searching winning tickets in SNN takes a longer time than ANN, we explore the existence of EB tickets in SNNs. In our experiments, we set $q$ and $\tau$ to 5 and 0.02, respectively. Although the original EB tickets use channel pruning, we use unstructured weight pruning for finding EB tickets in order to achieve a similar sparsity level with other pruning methods used in our experiments.

### 3.3 Early-Time Tickets
［#23］
Even though EB tickets significantly reduce searching time for winning tickets. For SNNs, one image is passed to a network through multiple timesteps, which provides a new dimension for computational cost reduction. We ask: *can we find the important weight connectivity for the SNN trained with timestep T from the SNN trained with shorter timestep $T' < T$?*
**Preliminary Experiments.** To answer this question, we conduct experiments on two representative deep architectures (VGG16 and ResNet19) on two datasets (CIFAR10 and CIFAR100). Our experiment protocol is shown in Fig. 3.3 (left panel). We first train the networks with timestep $T_{pre}$ till convergence. After that, we prune $p\%$ of the low-magnitude weights and re-initialize the networks. Finally, we re-train the pruned networks with longer timestep $T_{post} > T_{pre}$, and measure the test accuracy. Thus, this experiment shows the performance of SNNs where the structure is obtained from the lower timestep. In our experiments, we set $T_{pre} = \{2,3,4,5\}$ and $T_{post} = 5$. Surprisingly, the connections founded from $T_{pre} \geq 3$ can bring similar and even better accuracy compared to the unpruned baseline, as shown in Fig. 3.3. Note, in the preliminary experiments, we use a common post-training pruning based on the magnitude of weights [27]. Thus, we can extrapolate the existence of early-time winning tickets to a more sophisticated pruning method such as IMP [21], which generally shows better performance than post-training pruning. We call such a winning ticket as *Early-Time tickets*; a winning ticket drawn with a trained network from a smaller number of timesteps $T_{early}$, which shows matching performance with a winning ticket from the original timesteps $T$.

［#24］
![](./images/867760971282645124_3.jpg)

［#25］
Fig. 3. Preliminary experiments for Early-Time ticket. We conduct experiments on VGG16/ResNet19 on CIFAR10/CIFAR100. We report retraining accuracy ($T_{post}=5$) with respect to the timestep ($T_{pre}$) for searching the important connectivity in SNNs.

［#26］
![](./images/867760971282645124_4.jpg)

［#27］
Fig. 4. Kullback-Leibler (KL) divergence between the class prediction distribution from different timesteps. The network is trained with the original timestep $T=5$. We measure KL divergence between the predicted class probabilities from different timesteps. We use the training set for calculating KL divergence.

［#28］
Proposed Method. Then, how to practically select a timestep for finding Early-Time tickets? The main idea is to measure the similarity between class predictions between the original timestep $T$ and a smaller number of timesteps, and select a minimal timestep that shows a similar representation with the target timestep. Specifically, let $P_T$ be the class probability from the last layer of networks by accumulating output values across $T$ timesteps [39]. In this case, our search space can be $S=\{2,3,...,T-1\}$. Note, timestep 1 is not considered since it cannot use the temporal behavior of LIF neurons. To measure the statistical distance between class predictions $P_T$ and $P_{T'}$, we use Kullback-Leibler (KL) divergence:

［#28］
$$
D_{K L}(P_{T'}||P_T)=\sum_{x} P_{T'}(x)\ln\frac{P_{T'}(x)}{P_T(x)}. \tag{1}
$$

［#29］
The value of KL divergence goes smaller when the timestep $T'$ is closer to the original timestep $T$, i.e., $D_{K L}(P_{T-1}||P_T)\leq D_{K L}(P_{T-2}||P_T)\leq ... \leq D_{K L}(P_2||P_T)$. Note, for any $t\in\{1,..,T\}$, we compute $P_t$ by accumulating output layer's activation from 1 to $t$ timesteps. Therefore, due to the accumulation, if the timestep difference between timestep $t$ and $t'$ becomes smaller, the KL divergence $D_{K L}(P_{t'}||P_t)$ becomes lower. After that, we rescale all KL divergence values to $[0,1]$ by dividing them with $D_{K L}(P_2||P_T)$. In Fig. 3.3, we illustrate normalized KL divergence of VGG16 and ResNet19 on CIFAR10 and CIFAR100,

［#30］
8
Y. Kim, Y. Li, H. Park, V. Yeshwanth, R. Yin, P. Panda.

［#31］
```plaintext
Algorithm 1 Early-Time (ET) ticket
Input: Training data D; Winning ticket searching method F(·) – IMP or EB ticket;
Original timestep T; Threshold λ
Output: Pruned SNN_pruned
 1: Training SNN with N epochs for stability
 2: Memory = [ ]
 3: for t ← 2 to T do
 4:     P_t ← SNN(t, D)                ▷ Storing class prediction from each timestep
 5:     Memory.append(P_t)
 6: end for
 7: [D_KL(P_{T-1}||P_T),...,D_KL(P_2||P_T)] ← Memory             ▷ Computing KL div.
 8: for t ← 2 to T - 1 do
 9:     \hat{D}_{KL}(P_t||P_T) ← D_KL(P_t||P_T)/D_KL(P_2||P_T)                    ▷ Normalization
10:     if \hat{D}_{KL}(P_t||P_T) < λ then      ▷ Select timestep when KL div. is less than λ
11:         T_early = t
12:         break
13:     end if
14: end for
15: WinningTicket ← F(SNN,T_early)        ▷ Finding winning ticket with T_early
16: SNN_pruned ← F(WinningTicket,T)       ▷ Train with the original timestep T
17: return SNN_pruned
```

［#31］
where we found two observations: (1) The KL divergence with $T=2$ has a relatively higher value than other timesteps. If the difference in the class probability is large (i.e., larger KL divergence), weight connections are likely to be updated in a different direction. This supports our previous observation that important connectivity founded by $T=2$ shows huge performance drop at $T=5$ (Fig. 3.3). Therefore, we search $T_{early}$ that has KL divergence less than $\lambda$ while minimizing the number of timesteps. The $\lambda$ is a hyperparameter for the determination of $T_{early}$ in the winning ticket search process (Algorithm 1, line 15). A higher $\lambda$ leads to smaller $T_{early}$ and vice versa (we visualize the change of $T_{early}$ versus different $\lambda$ values in Fig. 7(b)). For example, if we set $\lambda=0.6$, timestep 3 is used for finding early-time tickets. In our experiments, we found that a similar value of threshold $\lambda$ can be applied across various datasets. (2) The normalized KL divergence shows fairly consistent values across training epochs. Thus, we can find the suitable timestep $T_{early}$ for obtaining early-time tickets at the very beginning of the training phase.

［#32］
The early-time ticket approach can be seamlessly applied to both IMP and Early-bird ticket methods. Algorithm 1 illustrates the overall process of Early-Time ticket. For stability with respect to random initialization, we start to search $T_{early}$ after $N=2$ epoch of training (Line 1). We show the variation of KL divergence results across training epochs in Supplementary C. We first find the $T_{early}$ from KL Divergence of difference timesteps (Line 2-13). After that, we discover the winning ticket using either IMP or EB ticket with $T_{early}$ (Line 14). Finally, the winning ticket is trained with the original timestep $T$ (Line 15).

［#33］
![](./images/867760971282645124_5.jpg)

［#34］
Fig. 5. The accuracy of winning ticket with respect to sparsity level. We report mean and standard deviation from 5 random runs.

［#35］
Table 1. Effect of the proposed Early-Time ticket. We compare the accuracy and search time of Iterative Magnitude Pruning (IMP), Early-Bird (EB) ticket, Early-Time (ET) ticket on four sparsity levels. We show search speed gain and accuracy change from applying ET.

［#36］
<table>
<thead>
<tr>
<th rowspan="2">Setting</th>
<th rowspan="2">Method</th>
<th colspan="4">Accuracy</th>
<th colspan="4">Winning Ticket Search Time (hours)</th>
</tr>
<tr>
<td>p=68.30%</td>
<td>p=89.91%</td>
<td>p=95.69%</td>
<td>p=98.13%</td>
<td>p=68.30%</td>
<td>p=89.91%</td>
<td>p=95.69%</td>
<td>p=98.13%</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">CIFAR10<br>VGG16</td>
<td>IMP</td>
<td>92.66</td>
<td>92.54</td>
<td>92.38</td>
<td>91.81</td>
<td>14.97</td>
<td>29.86</td>
<td>40.84</td>
<td>51.99</td>
</tr>
<tr>
<td>IMP + ET</td>
<td>92.49</td>
<td>92.09</td>
<td>91.54</td>
<td>91.10</td>
<td>11.19</td>
<td>22.00</td>
<td>30.11</td>
<td>38.26</td>
</tr>
<tr>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.17</td>
<td>-0.45</td>
<td>-0.84</td>
<td>-0.71</td>
<td>$\times$1.34</td>
<td>$\times$1.35</td>
<td>$\times$1.35</td>
<td>$\times$1.35</td>
</tr>
<tr>
<td>EB</td>
<td>91.74</td>
<td>91.05</td>
<td>89.55</td>
<td>84.64</td>
<td>1.96</td>
<td>0.74</td>
<td>0.11</td>
<td>0.09</td>
</tr>
<tr>
<td></td>
<td>EB + ET</td>
<td>91.27</td>
<td>90.66</td>
<td>88.95</td>
<td>84.86</td>
<td>1.44</td>
<td>0.55</td>
<td>0.07</td>
<td>0.06</td>
</tr>
<tr>
<td></td>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.47</td>
<td>-0.39</td>
<td>-0.60</td>
<td>+0.22</td>
<td>$\times$1.36</td>
<td>$\times$1.34</td>
<td>$\times$1.18</td>
<td>$\times$1.12</td>
</tr>
<tr>
<td rowspan="4">CIFAR10<br>Res19</td>
<td>IMP</td>
<td>93.47</td>
<td>93.49</td>
<td>93.22</td>
<td>92.43</td>
<td>21.01</td>
<td>42.20</td>
<td>58.91</td>
<td>73.54</td>
</tr>
<tr>
<td>IMP + ET</td>
<td>93.10</td>
<td>92.72</td>
<td>92.68</td>
<td>91.36</td>
<td>13.35</td>
<td>26.62</td>
<td>37.27</td>
<td>46.40</td>
</tr>
<tr>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.37</td>
<td>-0.77</td>
<td>-0.54</td>
<td>-1.07</td>
<td>$\times$1.57</td>
<td>$\times$1.59</td>
<td>$\times$1.58</td>
<td>$\times$1.58</td>
</tr>
<tr>
<td>EB</td>
<td>91.00</td>
<td>90.84</td>
<td>89.90</td>
<td>85.22</td>
<td>2.49</td>
<td>0.87</td>
<td>0.24</td>
<td>0.08</td>
</tr>
<tr>
<td></td>
<td>EB + ET</td>
<td>90.83</td>
<td>91.21</td>
<td>89.65</td>
<td>85.45</td>
<td>1.63</td>
<td>0.58</td>
<td>0.17</td>
<td>0.07</td>
</tr>
<tr>
<td></td>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.17</td>
<td>+0.37</td>
<td>-0.50</td>
<td>-1.09</td>
<td>$\times$1.52</td>
<td>$\times$1.49</td>
<td>$\times$1.38</td>
<td>$\times$1.16</td>
</tr>
<tr>
<td rowspan="4">CIFAR100<br>VGG16</td>
<td>IMP</td>
<td>69.08</td>
<td>68.90</td>
<td>68.00</td>
<td>66.02</td>
<td>15.02</td>
<td>29.99</td>
<td>41.03</td>
<td>52.05</td>
</tr>
<tr>
<td>IMP + ET</td>
<td>68.27</td>
<td>67.99</td>
<td>66.51</td>
<td>64.41</td>
<td>11.24</td>
<td>22.42</td>
<td>30.53</td>
<td>38.32</td>
</tr>
<tr>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.81</td>
<td>-0.91</td>
<td>-1.49</td>
<td>-1.61</td>
<td>$\times$1.33</td>
<td>$\times$1.34</td>
<td>$\times$1.34</td>
<td>$\times$1.36</td>
</tr>
<tr>
<td>EB</td>
<td>67.35</td>
<td>65.82</td>
<td>61.90</td>
<td>52.11</td>
<td>2.27</td>
<td>0.99</td>
<td>0.32</td>
<td>0.06</td>
</tr>
<tr>
<td></td>
<td>EB + ET</td>
<td>67.26</td>
<td>64.18</td>
<td>61.81</td>
<td>52.77</td>
<td>1.66</td>
<td>0.73</td>
<td>0.24</td>
<td>0.05</td>
</tr>
<tr>
<td></td>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.09</td>
<td>-1.64</td>
<td>-0.09</td>
<td>+0.66</td>
<td>$\times$1.36</td>
<td>$\times$1.35</td>
<td>$\times$1.31</td>
<td>$\times$1.12</td>
</tr>
<tr>
<td rowspan="4">CIFAR100<br>ResNet19</td>
<td>IMP</td>
<td>71.64</td>
<td>71.38</td>
<td>70.45</td>
<td>67.35</td>
<td>21.21</td>
<td>42.29</td>
<td>59.17</td>
<td>73.52</td>
</tr>
<tr>
<td>IMP + ET</td>
<td>71.06</td>
<td>70.45</td>
<td>69.23</td>
<td>65.49</td>
<td>13.56</td>
<td>27.05</td>
<td>37.88</td>
<td>46.65</td>
</tr>
<tr>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.58</td>
<td>-0.93</td>
<td>-1.22</td>
<td>-1.86</td>
<td>$\times$1.56</td>
<td>$\times$1.56</td>
<td>$\times$1.56</td>
<td>$\times$1.57</td>
</tr>
<tr>
<td>EB</td>
<td>69.41</td>
<td>65.87</td>
<td>62.18</td>
<td>52.92</td>
<td>3.08</td>
<td>1.71</td>
<td>0.43</td>
<td>0.09</td>
</tr>
<tr>
<td></td>
<td>EB + ET</td>
<td>68.98</td>
<td>65.76</td>
<td>62.20</td>
<td>51.50</td>
<td>2.00</td>
<td>1.12</td>
<td>0.29</td>
<td>0.07</td>
</tr>
<tr>
<td></td>
<td>$\Delta$ Acc. / Speed Gain</td>
<td>-0.43</td>
<td>-0.12</td>
<td>+0.02</td>
<td>-1.42</td>
<td>$\times$1.53</td>
<td>$\times$1.52</td>
<td>$\times$1.45</td>
<td>$\times$1.16</td>
</tr>
</tbody>
</table>

## 4 Experimental Results

### 4.1 Implementation Details

［#37］
We comprehensively evaluate various pruning methods on four public datasets: SVHN [56], Fashion-MNIST [77], CIFAR10 [36] and CIFAR100 [36]. In our work, we focus on pruning deep SNNs, therefore we evaluate two representative architectures; VGG16 [67] and ResNet19 [28]. Our implementation is based on Py- Torch [59]. We train the network using SGD optimizer with momentum 0.9 and weight decay 5e-4. Our image augmentation process and loss function follow the previous SNN work [15]. We set the training batch size to 128. The base learning

［#37］
rate is set to 0.3 for all datasets with cosine learning rate scheduling [48]. Here,
we set the total number of epochs to 150, 150, 300, 300 for SVHN, F-MNIST,
CIFAR10, CIFAR100, respectively. We set the default timesteps $T$ to 5 across all
experiments. Also, we use $\lambda=0.6$ for finding Early-Time tickets. Experiments
were conducted on an RTX 2080Ti GPU with PyTorch implementation. We use
SpikingJelly [19] package for implementation.

## 4.2 Winning Tickets in SNNs

［#38］
**Performance of IMP and EB ticket.** In Fig. 5, we show the performance of
the winning tickets from IMP and EB. The performance of random pruning is
also provided as a reference. Both IMP and EB can successfully find the winning
ticket, which shows better performance than random pruning. Especially, IMP
finds winning tickets over $\sim97\%$ sparsity across all configurations. Also, we
observe that the winning ticket sparsity is affected by dataset complexity. EB
ticket can find winning tickets ($>95\%$ sparsity) for relatively simple datasets
such as SVHN and Fashion-MINST. However, they are limited to discovering
the winning ticket having $\leq95\%$ sparsity on CIFAR10 and CIFAR100. We
further provide experiments on ResNet34-TinyImageNet and AlexNet-CIFAR10
in Supplementary G.

［#39］
**Effect of ET Ticket.** In Table 1, we report the change in accuracy and search
speed gain from applying ET to IMP and EB, on CIFAR10 and CIFAR100
datasets (SVHN and Fashion-MNIST results are provided in Supplementary E).
Although IMP achieves highest accuracy across all sparsity levels, they require
26 $\sim$ 73 hours to search winning tickets with 98.13% sparsity. By applying ET
to IMP, the search speed increases up to $\times1.59$ without a huge accuracy drop (of
course, there is an accuracy-computational cost trade-off because the ET winning
ticket cannot exactly match with the IMP winning ticket). Compared to IMP,
EB ticket provides significantly less search cost for searching winning tickets.
Combining ET with EB ticket brings faster search speed, even finding a winning
ticket in one hour (on GPU). At high sparsity levels of EB ($p=98.13\%$), search
speed gain from ET is upto $\times1.16$. Also, applying ET on ResNet19 brings a better
search speed gain than VGG16 since ResNet19 requires a larger computational graph
from multiple timestep operations (details are in Supplementary D). Overall, the
results support our hypothesis that important weight connectivity of the SNN
can be discovered from shorter timesteps.

［#40］
**Observations from Pruning Techniques.** We use *Late Rewinding* [22] (IMP)
for obtaining stable performance at high sparsity regime (refer Section 3.1). To
analyze the effect of the rewinding epoch, we change the rewinding epoch and
report the accuracy on four sparsity levels in Fig. 6(a). We observe that the
rewinding epoch does not cause a huge accuracy change with sparsity $\leq95.69\%$.
However, a high sparsity level (98.13%) shows non-trivial accuracy drop $\sim1.5\%$
at epoch 260, which requires careful rewinding epoch selection. Frankle *et al.*
[22] also show that using the same pruning percentage for both shallow and
deep layers (*i.e.*, local pruning) degrades the accuracy. Instead of local pruning,
they apply different pruning percentages for each layer (*i.e.*, global pruning). We

［#41］
![](./images/867760971282645124_6.jpg)

［#42］
Fig. 6. Observations from Iterative Magnitude Pruning (IMP). (a) The performance change with respect to the rewinding epoch. (b) The performance of global pruning and local pruning. (c) Layer-wise sparsity across different sparsity levels with global pruning. We use VGG16 on the CIFAR10 dataset for experiments.

［#43］
![](./images/867760971282645124_7.jpg)

［#44］
Fig. 7. Observations from Early-Bird (EB) ticket and Early-Time (ET) ticket. (a) Epoch when the EB ticket is discovered. (b) The change of $T_{early}$ with respect to the threshold $\lambda$ for KL divergence. (c) The performance of winning tickets from different $T_{early}$. We use VGG16 on the CIFAR10 dataset and show standard deviation from 5 random runs.

［#45］
compare global pruning and local pruning in Fig. 6(b). In SNN, global pruning achieves better performance than local pruning, especially for high sparsity levels. Fig. 6(c) illustrates layer-wise sparsity obtained from global pruning. The results show that deep layers have higher sparsity compared to shallow layers.

［#46］
We also visualize the epoch when an EB ticket is discovered with respect to sparsity levels, in Fig. 7(a). The EB ticket obtains a pruning mask based on the mask difference between the current mask and the masks from the previous epochs. Here, we observe that a highly sparse mask is discovered earlier than a lower sparsity mask according to EB ticket's mask detection algorithm (refer Fig. 3 of [80]). Furthermore, we conduct a hyperparameter analysis of the proposed ET ticket. In Fig. 7(b), we show the change of $T_{early}$ with respect to the threshold $\lambda$ used for selection (Algorithm 1). We search $\lambda$ with intervals of 0.1 from 0 to 1. Low $\lambda$ value indicates that we select $T_{early}$ that is similar to the original timestep, and brings less efficiency gain. Interestingly, the trend is similar across different datasets, which indicates our KL divergence works as a consistent metric. Fig. 7(c) shows the accuracy of sparse SNNs from three different $T_{early}$ (Note, $T_{early}=5$ is the original IMP). The results show that smaller $T_{early}$ also captures important connections in SNNs.

［#47］
![](./images/867760971282645124_8.jpg)

［#48］
Fig. 8. Transferability study of ANN winning tickets on SNNs.

### 4.3 Transferred Winning Tickets from ANN

［#49］
The transferability of winning tickets has been actively explored in order to eliminate search costs. A line of work [50,51,16,7] discover the existence of transferable winning tickets from the source dataset and successfully transfers it to the target dataset. With a different perspective from the prior works which focus on cross-dataset configuration, we discover the transferable winning ticket between ANN and SNN where the activation function is different. In Fig. 8, we illustrate the accuracy of IMP on ANN, IMP on SNN, Transferred Ticket, across four sparsity levels (68.30%, 89.91%, 95.69%, 98.13%). Specifically, Transferred Ticket (i.e., initialized weight parameters and pruning mask) is discovered by IMP on ANN, and trained on SNN framework where we change ReLU neuron to LIF neuron. For relatively simple datasets such as SVHN and F-MNIST, Transferred Ticket shows less than 2% accuracy drop even at 98.13% sparsity. However, for CIFAR10 and CIFAR100, Transferred Ticket fails to detect a winning ticket and shows a huge performance drop. The results show that ANN and SNN share common knowledge, but are not exactly the same, which can be supported by the previous SNN works [62,17,35] where a pretrained ANN provides better initialization for SNN. Although Transferred Ticket shows limited performance than IMP, searching Transferred Ticket from ANN requires $\sim 14$ hours for 98.13% sparsity, which is $\sim 5\times$ faster than IMP on SNN.

### 4.4 Finding Winning Tickets from Initialization

［#50］
A line of work [41,70] effectively reduces search cost for winning tickets by conducting a search process at initialization. This technique should be explored with SNNs where multiple feedforward steps corresponding to multiple timesteps bring expensive search costs. To show this, we conduct experiments on a representative pruning at initialization method, SNIP [41]. SNIP computes the importance of each weight connection from the magnitude of backward gradients at initialization. In Fig. 9, we illustrate the accuracy of ANN and SNN with SNIP on VGG16/CIFAR10 configuration.

［#51］
<table>
<caption>Table 2. Performance comparison of IMP [21] with the previous works.</caption>
<thead>
<tr>
<th>Pruning Method</th>
<th>Architecture</th>
<th>Dataset</th>
<th>Baseline Acc. (%)</th>
<th>$\Delta$Acc (%)</th>
<th>Sparsity (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Deng <i>et al.</i> [14]</td>
<td rowspan="3">7Conv, 2FC</td>
<td rowspan="3">CIFAR10</td>
<td rowspan="3">89.53</td>
<td>-0.38</td>
<td>50.00</td>
</tr>
<tr>
<td>-2.16</td>
<td>75.00</td>
</tr>
<tr>
<td>-3.85</td>
<td>90.00</td>
</tr>
<tr>
<td rowspan="3">Bellec <i>et al.</i> [3]</td>
<td rowspan="3">6Conv, 2FC</td>
<td rowspan="3">CIFAR10</td>
<td rowspan="3">92.84</td>
<td>-1.98</td>
<td>94.76</td>
</tr>
<tr>
<td>-2.56</td>
<td>98.05</td>
</tr>
<tr>
<td>-3.53</td>
<td>98.96</td>
</tr>
<tr>
<td rowspan="3">Chen <i>et al.</i> [9]</td>
<td rowspan="3">6Conv, 2FC</td>
<td rowspan="3">CIFAR10</td>
<td rowspan="3">92.84</td>
<td>-0.30</td>
<td>71.59</td>
</tr>
<tr>
<td>-0.81</td>
<td>94.92</td>
</tr>
<tr>
<td>-1.47</td>
<td>97.65</td>
</tr>
<tr>
<td rowspan="3">Chen <i>et al.</i> [9]*</td>
<td rowspan="3">ResNet19</td>
<td rowspan="3">CIFAR10</td>
<td rowspan="3">93.22</td>
<td>-0.54</td>
<td>76.90</td>
</tr>
<tr>
<td>-1.31</td>
<td>94.25</td>
</tr>
<tr>
<td>-2.10</td>
<td>97.56</td>
</tr>
<tr>
<td rowspan="3">IMP on SNN (ours)</td>
<td rowspan="3">ResNet19</td>
<td rowspan="3">CIFAR10</td>
<td rowspan="3">93.22</td>
<td>$\boldsymbol{+0.28}$</td>
<td>$\boldsymbol{76.20}$</td>
</tr>
<tr>
<td>$\boldsymbol{+0.24}$</td>
<td>$\boldsymbol{94.29}$</td>
</tr>
<tr>
<td>$\boldsymbol{-0.04}$</td>
<td>$\boldsymbol{97.54}$</td>
</tr>
<tr>
<td rowspan="3">Chen <i>et al.</i> [9]*</td>
<td rowspan="3">ResNet19</td>
<td rowspan="3">CIFAR100</td>
<td rowspan="3">71.34</td>
<td>-1.98</td>
<td>77.03</td>
</tr>
<tr>
<td>-3.87</td>
<td>94.92</td>
</tr>
<tr>
<td>-4.03</td>
<td>97.65</td>
</tr>
<tr>
<td rowspan="3">IMP on SNN (ours)</td>
<td rowspan="3">ResNet19</td>
<td rowspan="3">CIFAR100</td>
<td rowspan="3">71.34</td>
<td>$\boldsymbol{+0.11}$</td>
<td>$\boldsymbol{76.20}$</td>
</tr>
<tr>
<td>$\boldsymbol{-0.34}$</td>
<td>$\boldsymbol{94.29}$</td>
</tr>
<tr>
<td>$\boldsymbol{-2.29}$</td>
<td>$\boldsymbol{97.54}$</td>
</tr>
</tbody>
<tfoot>
<tr>
<td colspan="6">* We reimplement ResNet19 experiments.</td>
</tr>
</tfoot>
</table>

［#52］
![](./images/867760971282645124_9.jpg)
［#53］
Fig. 9. Performance comparison across pruned ANN and pruned SNN with SNIP.

［#54］
![](./images/867760971282645124_10.jpg)
［#55］
Fig. 10. Number of spikes with respect to sparsity on CIFAR10.

［#56］
Surprisingly, SNN shows huge performance degradation at high sparsity regime ($>80\%$), even worse than random pruning. The results imply that the previous <i>pruning at initialization</i> based on the backward gradient (tailored for ANNs) is not compatible with SNNs where the backward gradient is approximated because of the non-differentiability of LIF neuron [54,74].

### 4.5 Performance Comparison with Previous Works
［#57］
In Table 2, we compare the LTH method, especially IMP, with state-of-the-art SNN pruning works [14,3,9] in terms of accuracy and achieved sparsity. We show the baseline accuracy of the unpruned model and report the accuracy drop at three different sparsity levels. Note that the previous works use a shallow model with $6\sim7$ conv and 2 FC layers. The ResNet19 architecture achieves higher baseline accuracy compared to the previous shallow architectures. To compare

［#57］
the results on a ResNet19 model, we prune ResNet19 with method proposed by Chen et al. [9] using the official code¹ provided by authors. We observe that the previous SNN pruning works fail to achieve matching performance at a high sparsity on deeper SNN architectures. On the other hand, IMP shows less per- formance drop (in some cases even performance improvement) compared to the previous pruning techniques. Especially, at 97.54% sparsity, [9] shows 2.1% ac- curacy drop whereas IMP degrades the accuracy by 0.04%. Further, we compare the accuracy on the CIFAR100 dataset to explore the effectiveness of the pruning method with respect to complex datasets. Chen et al. ’s method fails to discover the winning ticket at sparsity 94.29%, and IMP shows better performance across all sparsity levels. The results imply that the LTH-based method might bring a huge advantage as SNNs are scaled up in the future.

## 4.6 Observation on the Number of Spikes
［#58］
In general, the energy consumption of SNNs is proportional to the number of spikes [1,13,79] and weight sparsity. In Fig. 10, we measure the number of spikes per image across various sparsity levels of VGG16 and ResNet19 architectures. We use IMP for pruning SNNs. We observe that the sparse SNN maintains a similar number of spikes across all sparsity levels. The results indicate that sparse SNNs obtained with LTH do not bring an additional MAC energy-efficiency gain from spike sparsity. Nonetheless, weight sparsity brings less memory occupation and can alleviate the memory movement overheads [10,58]. As discussed in [6], with over 98% of the sparse weights, SNNs can reduce the memory movement energy by up to $15\times$.

## 5 Conclusion
［#59］
In this work, we aim to explore lottery ticket hypothesis based pruning for imple- menting sparse deep SNNs at lower search costs. Such an objective is important as SNNs are a promising candidate for deployment on resource-constrained edge devices. To this end, we apply various techniques including Iterative Magnitude Pruning (IMP), Early-Bird ticket (EB), and the proposed Early-Time ticket (ET). Our key observations are summarized as follows: (1) IMP can achieve higher sparsity levels of deep SNN models compared to previous works on SNN pruning. (2) EB ticket reduces search time significantly, but it cannot achieve a winning ticket over 90% sparsity. (3) Adding ET ticket accelerates search speed for both IMP and EB, by up to $\times1.66$. We can find a winning ticket in one hour with EB+ET, which can enable practical pruning on edge devices.

［#60］
**Acknowledgement.** We would like to thank Anna Hambitzer for her help- ful comments. This work was supported in part by C-BRIC, a JUMP center sponsored by DARPA and SRC, Google Research Scholar Award, the National Science Foundation (Grant#1947826), TII (Abu Dhabi) and the DARPA AI Exploration (AIE) program.

［#61］
¹ https://github.com/Yanqi-Chen/Gradient-Rewiring

# References



















































































