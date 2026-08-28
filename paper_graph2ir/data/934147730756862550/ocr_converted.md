# Pursing the Sparse Limitation of Spiking Deep Learning Structures

［#1］
Hao Cheng
HKUS(GZ)
Jiahang Cao
HKUST(GZ)
Erjia Xiao
HKUST(GZ)
Mengshu Sun
Beijing University of Technology

［#2］
Le Yang
Xi'an Jiaotong University
Jize Zhang
HKUST
Xue Lin
Northeastern University

［#3］
Bhavya Kailkhura
Lawrence Livermore National Laboratory
Kaidi Xu
Drexel University
Renjing Xu
HKUST(GZ)

## Abstract

［#4］
Spiking Neural Networks (SNNs), a novel brain-inspired algorithm, are garnering increased attention for their superior computation and energy efficiency over traditional artificial neural networks (ANNs). To facilitate deployment on memory-constrained devices, numerous studies have explored SNN pruning. However, these efforts are hindered by challenges such as scalability challenges in more complex architectures and accuracy degradation. Amidst these challenges, the Lottery Ticket Hypothesis (LTH) emerges as a promising pruning strategy. It posits that within dense neural networks, there exist winning tickets or subnetworks that are sparser but do not compromise performance. To explore a more structure-sparse and energy-saving model, we investigate the unique synergy of SNNs with LTH and design two novel spiking winning tickets to push the boundaries of sparsity within SNNs. Furthermore, we introduce an innovative algorithm capable of simultaneously identifying both weight and patch-level winning tickets, enabling the achievement of sparser structures without compromising on the final model's performance. Through comprehensive experiments on both RGB-based and event-based datasets, we demonstrate that our spiking lottery ticket achieves comparable or superior performance even when the model structure is extremely sparse.

## 1. Introduction

［#5］
SNN is acclaimed as the third generation of neural networks and has increasingly gained great interest from researchers in recent years due to its distinctive properties: high biological plausibility, temporal information processing capability, inherent binary (spiking) information processing superiority, and low power consumption. Unlike ANNs which represent data continuously, SNNs process information as binary time series, leveraging low-power accumulation (AC) operations instead of the power-intensive multiply-accumulate (MAC) operations common in ANNs. This fundamental difference not only enhances energy efficiency but also aligns closely with biological neural processing. On specialized neuromorphic hardware platforms such as Loihi [10] and TrueNorth [1], SNNs demonstrate a remarkable reduction in energy consumption compared to ANNs. Additionally, SNNs follow their biological counterparts and inherit complex temporal dynamics from them, endowing SNNs with powerful abilities to extract image features in a variety of tasks, including recognition [11, 61], tracking [57], and images generation [4].

［#6］
Despite the advancements and comparable performance of recent SNN algorithms to ANNs in various domains, practical challenges such as operational redundancy and computational load persist, hindering their broader adoption in hardware implementations. To mitigate these issues, recent studies have focused on pruning techniques in SNNs, exploring methods such as model pruning [5, 36, 41], model quantization [17, 54, 60], and knowledge distillation [6, 38]. However, these methods face their own set of limitations. Many are constrained to simpler models [36, 41], or they lead to significant performance degradation [6, 17], especially in more complex spiking model architectures. This gap underscores the need for more effective pruning strategies that can maintain or enhance performance while accommodating the intricate dynamics of advanced SNNs.

［#7］
Confronting the aforementioned challenges, the Lottery Ticket Hypothesis (LTH) emerges as a promising avenue for pruning in SNNs. LTH posits that within a randomly initialized, dense neural network, there exists a subnetwork, which trained independently, can achieve the test accuracy of the full network within the same or fewer iterations. This approach to sparse training contrasts with the conventional multi-step process of weight pruning involving pretraining, pruning, and fine-tuning, and allows for the discovery of

［#8］
![](./images/934147730756862550_1.jpg)

［#9］
Figure 1. The Spiking Lottery Tickets (SLT) finding in event-based data (a) the finding process in spiking-based CNNs; (b) the finding process in the Spiking-based Transformers.

［#10］
these efficient subnetworks through a single phase of weight training. Building upon this concept, the Multi-Prize Lottery Tickets (MPTs [12]) hypothesis further refines this approach by focusing on efficient connection selection without the necessity of weight training, enhancing both weight sparseness and binarization for improved performance. Inspired by these advancements, our study aims to delve into the synergistic potential between LTH and the inherent energy efficiency of SNNs. We seek to construct models that are not only structurally sparse but also excel in energy conservation, thereby pushing the boundaries of SNN capabilities with the LTH method.

［#11］
However, existing LTH works have been primarily tailored for widely used Convolutional Neural Networks (CNNs) and Vision Transformers (ViT). Moreover, integrating LTH with SNNs is relatively less explored and still problematic. For instance, Kim et al. [27] proposed an earlytime (ET) ticket for deep SNNs, a notable advance but still reliant on extensive iterative processes to uncover winning tickets. Yao et al. [56] introduced a probabilistic modeling approach for LTH in SNNs, yet its validation is limited to a narrow dataset range and standard CNN structures, potentially underutilizing SNNs' distinct capabilities.

［#12］
In response to the gaps identified in existing studies, our research endeavors to delve into the unique synergy between Spiking Neural Networks (SNNs) and the Lottery Ticket Hypothesis (LTH). Our investigation is driven by two pivotal questions pivotal to advancing the concept of spiking lottery tickets (SLT): (1) Existence of SLTs in Event-Based Data. SNNs are commonly adopted for handling event-based data since utilizing SNNs on neuromorphic hardware is low-energy and low-latency [42, 50]. Our work aims to uncover and characterize SLTs across various SNN configurations in event-based data scenarios, shedding light on their potential and limitations. (2) Winning Tickets in Spiking-Based Transformers: The emergence of spiking-based transformers [51, 61] has opened new frontiers in neural network architectures, showing promise for future deployment in edge devices and real-world applications. However, the exploration of patch-winning tickets within these spiking-based transformers remains largely unexplored. Our study seeks to fill this research void, examining whether these advanced models harbor winning tickets that could further enhance their efficiency and applicability.

［#13］
To address the challenges previously outlined and to conduct a thorough exploration of Spiking Lottery Tickets (SLTs), our contributions can be summarized as follows:

［#14］
- **From RGB to DVS:** For the lottery tickets hypothesis, we remedy its consideration under event-based datasets. The DVS128Gesture [2] and CIFAR10-DVS [29] are adopted to explore the event-based winning tickets.
- **From CNN to ViT:** Building upon the discovery that weight-level SLTs are present in spike-based CNNs across both RGB and event-based data, we broaden our investigation to spiking-based transformers. Through our extensive real-world explorations, it can be shown that patch-level SLTs also exist in different kinds of datasets.
- **SNNs Parameters Analysis:** Having fully explored the winning tickets discovery algorithm's impact on SLTs, we investigated the effect of the intrinsic parameters of the SNNs themselves on SLTs.
- **Concurrent Connection and Token level SLT:** After a

［#14］
comprehensive analysis of the characteristics of SLTs for both weight and patch level, we propose a novel algorithm to implement both SLTs in spiking-based transformers.

## 2. Related Works

### 2.1. Model Structures

［#15］
Within the realm of deep learning, Deep Neural Networks (CNNs) have emerged as a fundamental architecture for many computer vision tasks, driving forward the state-of-the-art with their capacity to learn complex hierarchical visual features.
Artificial Neuron Networks Among various structures, Convolutional Neural Networks [28] became the cornerstone and profoundly influenced all recent related research. After that, the evolution of CNNs are proposed, like VGG [46], ResNet [21], DenseNet [25], and others. These architectures have excelled in image classification tasks and have been foundational in advancements across various applications in computer vision, such as object detection, semantic segmentation, and more. Their widespread popularity is a testament to the versatility and robustness of CNNs as a class of deep learning models. Following the enormous success of CNNs, a novel structure Vision Transformer (ViT), which is the encoder part of the Transformer, has received a great deal of attention. ViTs apply the transformer architecture, traditionally used in natural language processing, to vision tasks, treating images as sequences of pixels or patches. This seminal work demonstrated that transformers could outperform CNNs on image classification tasks when trained on sufficiently large datasets and with enough computational resources. Following the initial ViT model, various adaptations have emerged to enhance performance and efficiency. For instance, the DeiT (Data-efficient image Transformers) modified the training procedure to make the transformer models more data-efficient, expanding their applicability to scenarios where large-scale labeled datasets are not available. Incorporating convolutional information into the original Vision Transformer (ViT) architecture has been a subject of interest to improve its performance, especially in terms of inductive biases and local feature processing. Below are a few architectures that have successfully integrated convolutional elements into ViTs [15]. Levit [19], LV-VIT [26], CVT [53], Swin [32] and others [8].
Spiking Neuron Networks This section is about the current research progress of spiking neuron networks. The spiking neural network is a bio-inspired algorithm that simulates the real process of signaling that occurs in brains. Compared to the artificial neural network (ANN), it transmits sparse spikes instead of continuous representations, which brings advantages such as low energy consumption and robustness. In this paper, we adopt the widely used Leaky Integrate-and-Fire (LIF) model [43], which is suitable to characterize the dynamic process of spike generation and can be defined as:

［#15］
$$
\tau \frac{\mathrm{d} V(t)}{\mathrm{d} t}=-\left(V(t)-V_{reset}\right)+I(t) \tag{1}
$$

［#15］
where $I(t)$ represents the input synaptic current at time $t$ to charge up to produce a membrane potential $V(t)$, $\tau$ is the time constant. When the membrane potential exceeds the threshold $V_{th}$, the neuron will trigger a spike and resets its membrane potential to a value $V_{reset}$ ($V_{reset} < V_{th}$). The LIF neuron achieves a balance between computing cost cost and biological plausibility.

［#16］
In practice, the dynamics need to be discretized to facilitate reasoning and training. The discretized version of LIF model can be described as:

［#16］
$$
U[n]=e^{\frac{1}{\tau}} V[n-1]+\left(1-e^{\frac{1}{\tau}}\right) I[n] \tag{2}
$$

［#16］
$$
S[n]=\Theta\left(U[n]-V_{t h}\right) \tag{3}
$$

［#16］
$$
V[n]=U[n](1-S[n])+V_{reset} S[n] \tag{4}
$$

［#16］
where $n$ is the discrete time step, $U[n]$ is the membrane potential before reset, $S[n]$ denotes the output spike which equals 1 when there is a spike and 0 otherwise, $\Theta(x)$ is the Heaviside step function, $V[n]$ represents the membrane potential after triggering a spike.

### 2.2. Model Sparsity

［#17］
The model sparsity is to pursue the model with fewer model parameters without sacrificing the final performance.
Artificial Neuron Networks Sparsity Deep learning compression pursues more lightweight model parameters that could facilitate DNN implementations on resource-constrained application systems. Among them, model pruning and quantization are my primary focus that could be more possibly applied to hardware applications, such as FPGA. About model pruning, there is in general regular pruning schemes that can preserve the model's structure in some sense, such as the filter pruning scheme [23, 33], block-based pruning scheme [14, 34], and otherwise irregular pruning scheme [22, 52, 58]. After the success of those different kinds of weights training, The lottery ticket hypothesis (LTH) [18] conjectures that inside the large network, a subnetwork together with their initialization makes the pruning particularly effective, and together they are termed as the "winning tickets". In this hypothesis, the original initialization of the sub-network (before the large network pruning) is significant for it to achieve competitive performance when trained in isolation. Additionally, LTH also emphasizes the significance of the submodel structure itself but not the inheriting weights or gradients of the initial pre-trained models. This hypothesis also brings the possibility of Sparse Training [12, 13, 31] that could realize the finding of sparse subnetworks without complex collocation techniques of pertaining, pruning, and retraining cyclically.

［#18］
Model quantization [9, 30, 48, 55, 60] is another strategy to facilitate model compression. It could quantize the original 32-bit model to 16-2 bit models. Among them, converting 32 bits to 2 bits is the extreme situation of quantization that could be termed the Binary Neural Network (BNN) [30, 48, 55]. Model binarization could transfer original mathematics computation to XNOR or other bit-wise operations that could deeply boost high-performance computing in different devices. Additionally, BNN is also more suitable for the application of various hardware since the basis of digital circuits is the binary representation. MPTs [12] as the extension of LTH [12] is adopted to certify that the lottery tickets or subnetworks also remain in quantization or binary dense models.

［#19］
Spiking Neural Networks Sparsity To further improve the energy efficiency of SNN, a number of works on SNN pruning have been proposed and well-validated on neuromorphic hardware. Shi [45] propose a pruning scheme that exploits the output spike firing of the SNN to reduce the number of weight updates during network training. Guo [20] dynamically removes non-critical weights in training by using the adaptive online pruning algorithm. Apart from seeking the help of pruning, Rathi [41] and Takuya [47] pursue sparse SNN by using Knowledge distillation and quantization. Several works try to combine LTH with SNNs: Kim et al. [27] first investigate how to scale up pruning techniques towards deep SNNs and reveal that winning tickets consistently exist in deep SNNs across various datasets and architectures. They also propose a kind of Early-Time ticket that could alleviate the heavy search cost. Yao et al. [56] contribute a novel approach by introducing a probabilistic modeling method for SNNs. This method allows for the theoretical prediction of the probability of identical behavior between two SNNs, accounting for the complex spatio-temporal dynamics inherent to SNNs. Cheng et al. [7] refers to biprop [12] and extends the discovered scenario to binary weight case thereby further exploring the sparse limit of spiking winning tickets. Concretely, they compare the spiking mechanism with simple model binarization and propose a superior strategy for finding SLT named Binary Weights Spiking Lottery Tickets. The binary SLT explored by this paper could even perform better than the original dense ANN and BNN in some cases.

［#20］
Training Spiking Neural Network In the past few years, a large number of learning algorithms have explored how to train a deep SNN with high performance, including three main approaches: (1) ANN to SNN conversion, (2) direct training, and (3) local training. For (1), it is necessary to train an ANN network in advance and then transfer the continuous output of the ReLU function to the discrete spikes [3, 13]. Although such algorithms can help SNNs achieve better accuracy, they require substantial training resources. The second type of method usually employs a gradient-based technique. For example, the surrogate gradient method [37] solves the non-differentiability problem of discrete spikes during the backpropagation process. Meng [35] proposes DSR to train SNN indirectly by gradient mapping. Although such methods ensure the feasibility of training, they do not exploit the properties of SNNs and lack bio-plasticity. Lastly, the local training methods are mainly based on biological learning rules, like Hebbian learning [24] and Spike Timing-Dependent Plasticity (STDP) [39, 41]. Although the performance of those methods is much lower than the State-of-the-art (SOTA) ANN, the local training is particularly sparse, which provides ideas for SNN model compression. In general, some methods are inspired by existing compression techniques in ANN. Kim [27] proposes the Early-Time ticket to relatively reduce the huge training computational cost when combined with the multiple timesteps of SNNs.

### 3. Spiking Lottery Tickets
［#21］
To go about enhancing the generalizability of previous SLTs explorations, Regarding Ramanu et al. [40] and Shen et al. [44] that are the discovery algorithms towards connection and path-based winning tickets in spiking-based CNNs and Transformers, as well as some the work about SNNs sparsity [7, 56], we first try also to pursue the extensions of the universality of SLTs in different dataset and structure, like event-based data and spiking-based transformer. After that, since the spiking-based Transformer or SpikeFormer always needs the convolution operation in embedding to promote its performance, it would be possible to concurrently adopt the former two SLTs finding algorithms to boost the sparsity of Spikeformer further. Therefore, we propose an algorithm that could efficiently find the ECPTs.
Connection-Level Lottery Tickets for CNNs: Referring to the main idea of Ramanu et al. [40], they propose the edge-popup that could explore the winning tickets in randomly weighted neural networks, which are subnetworks that can achieve high accuracy without any traditional weights training. It shows that identifying high-performing subnetworks does not need to adjust the original weights or rely on the trained weights. In a word, edge-popup mainly focuses on finding the Sparse Connection-based SLTs (SConnSLT) and does not care about the effect of weight values. Furthermore, Diffenderfer et al. [12] also refer to edge-popup and broaden its applying domain in binary case. In the spiking research area, based on it, Yao et al. [56] and Cheng et al. [7] apply those subnetworks finding methods to convolutional-based SNNs and could also efficiently find SLTs without training. Fig. 1 (a) briefly illustrates the process of finding SConnSLT under event-based datasets. Since the time step of event-based data is somewhat similar to the frames in RGB data, it could be visualized as four frames when the Timestep = 4. After inputting

［#21］
the event data into the convolution-based SNNs, the neuron connection would be gradually sparse after the Threshold Select and Connection Pruning. Additionally, the information adopted in the whole process is the spiking signal as presented.

［#22］
Patch-Level Lottery Tickets for Transformer: Shen et al. [44] explore extending the winning tickets to ViTs by focusing on input data, specifically image patches, instead of network weights. It introduces a Ticket Selector to identify patch-level winning tickets that can train ViTs to achieve accuracy comparable to using all patches. Unlike traditional weight-level winning tickets, they focus on data-level 'winning tickets' rather than weight-level due to the unique challenges of ViTs. ViTs heavily rely on input data, making it difficult to generalize winning subnetworks across different inputs using weight-level approaches. Traditional methods of finding effective subnetworks at the weight level are less effective in ViTs, as their performance is more tied to input handling. By identifying the most informative image patches, this approach aims to train ViTs more efficiently and effectively, shifting the focus of the Lottery Ticket Hypothesis from network weights to input data. To the best of our knowledge, no article has attempted to explore any form of spiking winning tickets for spiking ViTs, regardless of whether the SLTs are weight or data-based, and regardless of whether the SLTs are found in RGB data or event data. As the illustration of Fig. 1 (b), it indicates the process of finding date-based SLTs in event-based data. Specifically in ViTs, the data-based Winning Tickets actually screen for the importance of different patches and tokens. The left subfigure of Fig. 1 (b) shows the Token-based Lottery Tickets (TokenLTs) could be identified after going through the Spiking Patch Splitting Module (SPS) and Spiking Encoder-based Ticket Selector (SETS). After obtaining the TokenLTs and their position embedding through the patch selection to the original patch, the insignificant patches are deleted and only the TokenLTs go through the concrete classification process in the right subfigure of Fig. 1 (b).

［#23］
Concurrent Connection and Patch Level Lottery Tickets: As presented in Fig. 1, the previous two subsections propose the lottery ticket finding strategies for CNN and transformer-based SNNs on connection and patch level respectively. However, when we revisit the supposedly lightweight spikeformer later on, it seems that there is still the possibility of improving sparsity even more. ViTs are less likely to get winning tickets at the connection and weights level, because they consist of only a few linear layer-based attention operations, and there are no convolution operations that might cause redundancy. Nonetheless, nowadays, the introduction of convolution operation into ViTs has become a mainstream performance improving strategy. Furthermore, in the spiking domain, this operation seems to have become an even more necessary component to improve the performance of the corresponding structures, e.g., the input embedding of the spikeformer adopts the convolution projection to process input patch directly. This provides us with the feasibility to further enhance the connection level sparse of patch-level tickets of Spikeformer. Therefore, by combining the ideas of Figure 1, we propose the Embedding Connection and Patch Tickets (ECPTs) finding algorithm for Transformer-based SNNs.

［#24］
Since the purpose of finding ECPTs is mainly to improve further the connection-level winning tickets in the spiking Convolution Projection Module (CPM) of the previous patch-level tickets in Spikeformer, the significant modification of the original finding method in the subfigure (b) in Figure 1 should be the introduction of connection level tickets finding process towards CPM. As illustrated in Figure 2, after applying the connection pruning to the original CPM, some redundancy connection would be pruned according to the particular pruning rate, and the initial dense CPM is transferred to the sparse CPM.

［#25］
The Algorithm 1 presents further details of the discovery of ECPTs. Step 1 is adopted to define the model. The

［#26］
```
Algorithm 1 Embedding Connection and Patch Tickets
 1: Define Models: Spikeformer $S(\cdot)$; spiking Convolu-
    tion Projection Module (CPM) $S_c$ with param numbers
［#26］
    $n_c$; Spiking Encoder and classification head $S_{ec}(\cdot)$.
 2: Input: Event-based data $(x_t, y)$ with patch $P$ and time
［#26］
    $t$; Patch number $n_p$; Connection tickets finding epochs
［#26］
    $N_c$; Spiking Patch Tickets (SPT) Select epochs $N_{sp}$.
 3: Initialize Model Params: The CPM weights and its cor-
    responding sparse mask $w_c$ and $M_c \in \{0, 1\}$;
 4: ### Connection lottery tickets finding in CPM
 5: Initialize Params: Pruning score $s$ and its optimizing
    update param $\eta$; Gain term $\alpha$; Pruning rate for connec-
    tion sparsity $pr_c$; CPM loss function $L(\cdot)$
 6: Adding a linear layer with the output dimension $y$.
 7: for k = 1 to $N_c$ do
 8:     $s \leftarrow s - \eta \nabla_s L(\alpha \cdot M_c \odot w_c)$
 9:     $Proj_{[0,1]} \leftarrow$ Sorting $s$ based on to $pr_c n_c$
10:     $M_c \leftarrow M_c \odot Proj_{[0,1]}, \alpha \leftarrow ||M_c \odot w_c||/||M_c||$
11: end for
12: Keep the convolution layer and drop the linear layer.
13: ### Patch lottery ticket finding for spikeformer
14: Initialize Params: SPT index $id_p$; Patch embedding and
    Position embedding $PatE$ and $PosE$; pruning rate for
    patch sparsity $pr_p$.
15: for k = 1 to $N_{cp}$ do
16:     $PatE \leftarrow S_c(P)$
17:     $id_p \leftarrow$ Sorting $PatE$ based on $pr_p n_p \odot PosE$
18: end for
19: $P_{spt} \leftarrow$ Pick the SPT according to the $id_p \odot P$
20: Output: $S(p_{spt}) \leftarrow S_{ec}(S_p(P_{spt}; \alpha(M_c \odot w_c)))$
```


［#27］
Spikeformer adopted here is $S(\cdot)$, and our mainly sparse object in finding ECPTs is in the spiking CPM. After that, the remaining structures, which are the spiking encoder block and classification head, of Spikeformer is $S_{ec}$. And Steps 2 and 3 are the input parameters and the initialization of the part of them. And then, step 4 to step 11 is the main connection ticket finding process. The main parameter of this finding process is the pruning score $s$ and it could be continuously updated until finding the correct connection tickets according to the conjunction of mask $M_c$, weights $w_c$, and gain term $\alpha$. However, this gain term update is driven by optimizing the parameters of a to achieve the search for subnetworks with optimal connections in a continuous update. After finding the corresponding connection tickets, the updated mask $M_c$ and weights $w_c$ would be kept to insert into the original Spikeformer. Consequently, step 15 to step 19 would finish the patch tickets finding phase. The original patches $P$ are fed into CPM $S_c$ and output patch embedding $PatE$. The index $id_p$ of the most significant embedding or patch tickets would be selected according to the sorting of $PatE$. Finally, the picked patch winning tickets $P_{spt}$ would be processed in this Spikeformer with the sparse CPM.

［#28］
![](./images/934147730756862550_2.jpg)

［#29］
Figure 2. Connection sparsity in Convolution Projection Module

## 4. Experiments and Analysis

### 4.1. Experimental Setting

［#30］
In this paper, we use various strcutres including VGG-9, ResNet-19, TINY and SMALL under two RGB datasets and two Event-based datasets. The TINY and SMALL are referring to the DeiT-Tiny and DeiT-Small [49]. Two RGB datasets are CIFAR10 and CIFAR100. Two Event-based dataset are DVS128Gesture and CIFAR10-DVS. When finding winning tickets, for exploring weight-level tickets using connection sparsity, we adopt $pr_c = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8$; for exploring patch level tickets for transformer, we adopt $pr_p = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8$. For the parameters of SNN, we adopt the most commonly used LIF neuron with timestep $T=4$ and decay rate $\lambda=0.99$ in our main experiments. Additionally, in order to explore the specific relationship between various SNN component parameters and MPSTs, we further use Time Step $T=1,2,4,6,8$ and Decay Rate $\lambda=0.99, 0.8, 0.6, 0.4, 0.2$. To facilitate the learning process, we use the Adam optimizer with a base learning rate of 0.1. Surrogate gradient training methods [37] for SNN are adopted. The models for conducting experiments are implemented based on Pytorch and SpikingJelly [16]. Since our main focus is finding the best spiking lottery tickets while maintaining accuracy, we only use the most primitive direct training without any training tricks in all our experiments. Furthermore, the Appendix includes more detailed interpretations.

### 4.2. General Performance Analysis

［#31］
According to the results illustrated in Table 1 and Table 2, we could obtain a general performance analysis for the various winning tickets finding methods of ConnS, PatchS, and CPT. For CNNs in Table 1, the weight-level winning tickets is exit in different structures, VGG-9 and ResNet-19, in different datasets. And the spiking-based CNNs could outperform the normal CNNs under the extremely sparse case, which is pruning weights, as well as binarizing weights and activation simultaneously. For Transformer in Table 2, the patch-level winning tickets are both exiting among normal and spiking-based Transformer in various datasets. Since the Spikeformer using convolutional operation in embedding, it makes its sparsity could be further improved by ConnS.

### 4.3. The Effect of Different Parameters

［#32］
Pruning Rate: In our experiments, we try to pursue performance change when modifying the pruning rate for weight and patch-level winning tickets for different models under the event-based dataset. Therefore, for spiking parameters, we keep the Time Step $T=4$ and $\lambda=0.99$ for two models in various weight and activation conditions, like full-precision, binary and spiking. As presented in Figure 3 (a), it illustrates the performance change for VGG-9 and TINY in DVS128Gesture. As the figures of VGG-9 show, normal, binary and spiking-based CNNs always go through an increase and decrease process. The weight-based winning tickets could attain the best performance when the pruning rate is set to $50\%$. Therefore, in the previous Table 1, we set the pruning rate to $50\%$ to compare with other dense performances. Additionally, in the figures of TINY, normal and spiking-based Transformers are sensitive to the pruning rate change. Both of them could tolerate a certain degree of patch pruning, whereas the performance would drop drastically when the pruning rate is beyond a threshold. It seems this threshold for different types of TINYs are all around $30\%$. In the Appendix, after going through a more detailed analysis, we ensure the best threshold is $31.4\%$ here. Additionally, we also ensure the performance when introducing ConnS into the process of finding patch-based winning tickets. When the pruning rate of ConnS is fixed at $48.7\%$

［#33］
<table>
<thead>
  <tr>
    <th>Architecture</th>
    <th>Method</th>
    <th>CIFAR10<br>Acc(%)</th>
    <th>CIFAR100<br>Acc(%)</th>
    <th>DVS128Gesture<br>Acc(%)</th>
    <th>CIFAR10-DVS<br>Acc(%)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="11">VGG-9 / ResNet-19<br>(2.26M) / (12.63M)</td>
    <td>ANN+</td>
    <td>88.10/92.05</td>
    <td>68.64/76.74</td>
    <td>92.92/94.83</td>
    <td>78.65/80.14</td>
  </tr>
  <tr>
    <td>BNN*</td>
    <td>86.75/91.54</td>
    <td>67.27/74.42</td>
    <td>92.09/93.69</td>
    <td>77.94/78.98</td>
  </tr>
  <tr>
    <td>BinActBNN⁻</td>
    <td>83.84/86.74</td>
    <td>58.57/66.60</td>
    <td>85.41/89.43</td>
    <td>69.21/62.21</td>
  </tr>
  <tr>
    <td>SNN*</td>
    <td>87.71/91.78</td>
    <td>67.94/72.37</td>
    <td>92.78/93.27</td>
    <td>78.41/80.06</td>
  </tr>
  <tr>
    <td>BinWSNN⁻</td>
    <td>87.57/91.21</td>
    <td>66.81/72.93</td>
    <td>91.97/93.92</td>
    <td>77.49/79.04</td>
  </tr>
  <tr>
    <td>ANN-ConnS+</td>
    <td>87.73/90.87</td>
    <td>67.56/75.62</td>
    <td>92.43/93.83</td>
    <td>77.92/80.02</td>
  </tr>
  <tr>
    <td>BNN-ConnS*</td>
    <td>85.27/91.07</td>
    <td>66.73/73.87</td>
    <td>91.74/93.11</td>
    <td>77.31/78.26</td>
  </tr>
  <tr>
    <td>ActBNN-ConnS⁻</td>
    <td>81.30/83.96</td>
    <td>55.81/62.76</td>
    <td>81.53/84.77</td>
    <td>63.82/57.41</td>
  </tr>
  <tr>
    <td>SNN-ConnS*</td>
    <td>87.52/91.36</td>
    <td>67.22/72.01</td>
    <td>92.17/92.88</td>
    <td>77.94/79.71</td>
  </tr>
  <tr>
    <td>BinWSNN-ConnS</td>
    <td>87.76/91.82</td>
    <td>66.74/72.73</td>
    <td>91.70/93.52</td>
    <td>77.01/78.66</td>
  </tr>
</tbody>
</table>

［#34］
Table 1. Evaluating the Connection Sparsity (ConnS) performance of VGG-9/ ResNet-19 in full-dense, binary and spiking condition under CIFAR10, CIFAR100, DVS128Gesture and CIFAR10-DVS. The font bold row BinWSNN-ConnS indicates the model with good performance and the most sparsity among all of them; The $-$ means that BinWSNN-ConnS could greatly over the models of this row; $*$ means that the performance of BinWSNN-ConnS should be almost the same compared with them; $+$ means that BinWSNN-ConnS aims to approach them.

［#35］
![](./images/934147730756862550_3.jpg)

［#36］
Figure 3. The performance effect of different model and spiking parameters for the VGG-9 and TINY in DVS128Gesture. The first row indicates the VGG-9, the second row indicates the TINY. The column (a) is the effect of Pruning Rate; The column (b) is the effect of Time Step; The column (c) is the effect of Decay Rate.

［#37］
according to the analysis in the following subsection and Appendix, the concurrent weight and patch-based tickets of Spikeformer could display a similar change as others.

［#38］
Time Step In this paragraph, we conduct a thorough experiment to validate the effect of selecting a suitable Time Step for searching weight level and patch level winning tickets for VGG-9 and TINY in DVS128Gesture. In the meantime, for other parameters, the Decay Rate is $\lambda=0.99$, and the weight and patch level sparsity pruning rate is set to $50\%$ and $30\%$. In Figure 3 (b), as mentioned above, we select $T=1,2,4,6,8$ to verify the influence of Time Step on the performance of weight and patch level winning tickets; an obvious increase occurs in the sparse performance. From the theory of SNN, we could intuitively

［#39］
<table><thead><tr><th>Architecture</th><th>Spasrity and Datasets</th><th>Transformer*</th><th>Spikeformer*</th><th>Transformer - PatchS*</th><th>Spikeformer - PatchP*</th><th>Spikeformer - CPS</th></tr></thead><tbody><tr><td rowspan="6">TINY / SMALL (4.89M) / (22.12M)</td><td>Patch Sparsity (%)</td><td>0/0</td><td>0/0</td><td>31.4/37.6</td><td>31.4/37.6</td><td>31.4/37.6</td></tr><tr><td>Conn Sparsity(%)</td><td>-/-</td><td>0/0</td><td>-/-</td><td>0/0</td><td>48.7/54.9</td></tr><tr><td>CIFAR10 (%)</td><td>85.71/87.36</td><td>84.81/87.17</td><td>82.04/83.76</td><td>81.75/82.98</td><td>81.21/82.44</td></tr><tr><td>CIFAR100 (%)</td><td>67.37/69.12</td><td>66.54/69.43</td><td>64.71/66.22</td><td>62.98/64.86</td><td>62.82/64.89</td></tr><tr><td>DVS128Gestrure (%)</td><td>94.96/97.12</td><td>94.13/96.99</td><td>93.89/96.74</td><td>94.37/96.88</td><td>94.50/97.22</td></tr><tr><td>CIFAR10-DVS (%)</td><td>78.43/80.01</td><td>77.86/79.77</td><td>76.61/ 78.42</td><td>77.14/78.91</td><td>77.03/79.02</td></tr></tbody></table>

［#40］
Table 2. Evaluating the Patch Sparsity (PS) and Connection Patch Sparisty (CPS) performance of TINY/SMALL in normal and spiking based Transformer under CIFAR10, CIFAR100, DVS128Gesture and CIFAR10-DVS. The Patch and Connection (Conn) Sparsity are also presented.The font bold rowSpikeformer-CPS indicates the model with good peroformance and the most sparsity among all of them; * means that the performance of Spikeformer-CPS should be almost the same compared with them;

［#41］
![](./images/934147730756862550_4.jpg)

［#42］
Figure 4. The performance of patch tickets in spiking-based transformer after introducing Connection Sparse (ConnS) into CPM.

［#43］
know the timestep could offer information gain following the increase of its value since $T$ times spiking simulation can be considered as multiple processing of the same input representation. This performance has already been verified in the dense case [17, 59]. Nevertheless, through our experiments, even in the most extremely sparse case, this performance still exists no matter which kind of model structures are selected. Based on our concrete experimental results, increasing timestep from $T=1$ to $T=8$ with different pruning rates could achieve at most $+2.3\%$ and $+3.8\%$ increase in VGG-9 and TINY.

［#44］
Decay Rate: After thoroughly examining the relationship between Time Step $T$ and sparse performance, we also concentrate on another critical hyper-parameter Decay Rate $\lambda$ in SNNs. We select the $\lambda=0.99,0.8,0.6,0.4,0.2$ as our variables to be validated. In the meantime, for other parameters, the Time Step is $T=4$, and the weight and patch level sparsity pruning rate is set to $50\%$ and $30\%$. Our experiment results are presented in Fig. 3 (c), modifying the decay rate could affect the final sparse accuracy among different model structures. All performances of various models also go through an increase and decrease process. Based on the illustration of the figure, change the timestep from $\lambda=0.99$ to $\lambda=0.2$ with different pruning rates could achieve at most $+2.1\%$ and $+1.3\%$ increase in VGG-9 and TINY. However, according to our observation, the choice of $\lambda$ that inspires to produce the best sparse performance is related to the particular model size. The structure with a smaller model size could tolerate a more significant decay rate, but a larger model would suffer more damage when set with huge $\lambda$.

### 5. Connection and Patch Sparisty

［#45］
In this paragraph, we try to figure out the effect of different pruning rates of ConnS for fixed patch-level winning tickets in Transformers. For the parameters, the Time Step is $T=4$ and Decay Rate $\lambda=0.99$, and the patch level sparsity pruning rate is set to $30\%$. And the pruning rates of ConnS are adopted as 0,0.2,0.4,0.6,0.8. As presented in Figure 4, this paragraph identifies the phenomenon similar to the patch-level winning tickets in Transformers, where a certain threshold exists. When the sparsity level is below this threshold, it is possible to maintain the original performance of the neural network even with an increase in weight sparsity. However, exceeding this threshold significantly diminishes performance. The method of finding CPTs enables further enhancement of the sparsity level in spiking-based transformers, or Spikeformers, without compromising their original performance. This discovery opens up new avenues for optimizing the efficiency of Spikeformers while retaining their effectiveness.

### 6. Conclusion

［#46］
This paper marks significant strides in the field of neural network optimization and efficiency. By extending the lottery tickets hypothesis to event-based datasets and spiking neural networks, it paves the way for more nuanced and efficient neural network architectures. The transition from traditional spike-based CNNs to advanced Spikeformers, coupled with the detailed analysis of SNN parameters, high-

［#46］
lights the potential for SLTs in various levels and types of data. The proposed concurrent implementation of weight and patch-level SLTs in Spikeformers not only showcases the versatility of this approach but also sets a foundation for future research in neural network optimization. Finally, we develop a concurrent implementation method for Sparse Lottery Tickets at both weight and patch levels in Spike- formers represents a groundbreaking step in efficient net- work research.

## References





























































