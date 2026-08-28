# SPARSESPIKFORMER: A CO-DESIGN FRAMEWORK FOR TOKEN AND WEIGHT PRUNING IN SPIKING TRANSFORMER

Yue Liu
Shanlin Xiao*
Bo Li
Zhiyi Yu *

Sun Yat-sen University, China

## ABSTRACT

As the third-generation neural network, the Spiking Neural Network (SNN) has the advantages of low power consumption and high energy efficiency, making it suitable for implementation on edge devices. More recently, the most advanced SNN, Spikformer, combines the self-attention module from Transformer with SNN to achieve remarkable performance. However, it adopts larger channel dimensions in MLP layers, leading to an increased number of redundant model parameters. To effectively decrease the computational complexity and weight parameters of the model, we explore the Lottery Ticket Hypothesis (LTH) and discover a very sparse ($\geq$90%) subnetwork that achieves comparable performance to the original network. Furthermore, we also design a lightweight token selector module, which can remove unimportant background information from images based on the average spike firing rate of neurons, selecting only essential foreground image tokens to participate in attention calculation. Based on that, we present SparseSpikformer, a co-design framework aimed at achieving sparsity in Spikformer through token and weight pruning techniques. Experimental results demonstrate that our framework can significantly reduce 90% model parameters and cut down Giga Floating-Point Operations (GFLOPs) by 20% while maintaining the accuracy of the original model.

Index Terms— Spiking Neural Network, Transformer, Neural network pruning, Lottery Ticket Hypothesis

## 1. INTRODUCTION

Spiking Neural Network (SNN) represents a novel type of neural network characterized by low power consumption and sparsity [1]. Different from traditional Artificial Neural Network (ANN), SNN simulates the spike transmission process of biological neurons, using discrete spike sequences for information exchange. Due to the event-driven and sparse characteristics of SNN, it can significantly reduce computing and storing resources, making it ideal for edge devices to implement [2, 3, 4]. In recent years, researchers have been exploring optimization techniques for deep SNN [5, 6, 7, 8] to achieve better performance and efficiency, which drives the application and development of SNN in various fields.

Recently, as researchers have continued to explore Transformer architecture [9], it has been gradually applied to a wide range of computer vision tasks [10, 11, 12, 13, 14]. The Vision Transformer (ViT) [10] is the first to apply Transformer to image classification tasks by transforming the input image into a sequence of patches. Since then, the ViT model and its variants [11, 15, 16] have achieved a serious state-of-the-art (SOTA) performance. The remarkable representation ability of Transformer has inspired researchers to combine it with SNN. Zhou first applied the Spiking Self-Attention in Transformer to train Spikformer [17], which broke the bottleneck of traditional SNN limitations and achieved significant success. However, its complex network architecture with high computational and storage costs has hindered its deployment on resource-constrained edge devices. To address this issue, we explore Spikformer sparsification techniques that optimize model performance and reduce power consumption by minimizing non-essential computations and weight parameters.

In this paper, we introduce SparseSpikformer, a hardware-friendly Spikformer model that adopts a co-design framework of image token and weight parameter pruning. Through this close collaboration, SparseSpikformer can achieve a substantial reduction in both computational and storage costs while maintaining a comparable level of accuracy. In summary, our main contributions are listed as follows:

- We study a novel hybrid pruning framework for Spikformer with image token and weight level optimizations, to efficiently reduce hardware resource requirements and accelerate inference runtime.
- At the image token level, we adopt the average spike firing rate as a criterion and employ a token selector module to reduce the number of unnecessary image tokens. With this module, only image tokens with a high spike firing rate are kept for attention calculation, which can effectively cut down the computational cost.
- At the weight parameter level, we utilize the sparsity property of SNN to perform weight parameter pruning by LTH technique [18]. The experimental results demonstrate the existence of winning tickets

* The Corresponding authors.

![](./images/931974443775820248_1.jpg)

Fig. 1. The framework of SparseSpikformer. The Spiking Patch Splitting (SPS) partitions the input image into a series of spike tokens. Subsequently, the Spiking Token Selector (STS) selects the top $\rho\%$ of tokens based on their spike firing rate to participate in Spiking Self-Attention (SSA) and MLP calculations.

within SparseSpikformer, encompassing only 10% of
the model parameters while exhibiting a minimal de-
crease in model accuracy.

## 2. METHOD

For a given 2D input image sequence $I \in \mathbb{R}^{T \times C \times H \times W}$, we follow the processing method of Spikformer. Initially, we employ Spiking Patch Splitting (SPS) to partition the input image into a series of spiking patches $X' \in \mathbb{R}^{T \times N \times D}$. Subsequently, the input spike proceeds through L layers of the Encoder Block, which consists of three essential components: Spiking Token Selector (STS), Spiking Self-Attention (SSA), and MLP. Within the STS module, we select a fixed number of tokens based on the firing rate of the spike sequence. The SSA module serves as the central component of Spikformer, primarily utilized to extract comprehensive global information from the image. It combines the characteristics of SNN by converting the traditional floating-point $Q,K,V$ matrix into spike form to reduce energy consumption. After the SSA module, the spike will further go through the MLP layer. It is worth noting that as the dimension of MLP increases, the final model accuracy will be further improved. However, this operation will also bring a large number of weight parameters, consequently increasing the computational burden. Therefore, the MLP stands out as one of the primary targets for weight pruning. After passing through L layers of the Encoder Block, we use a fully connected layer as a linear classification head to output the result $Y$. The Overall architecture of SparseSpikformer is shown in Figure 1.

### 2.1. Dynamic Token Pruning

In this section, we introduce a lightweight approach to evaluate the importance of image tokens. As shown in Figure 2,
![](./images/931974443775820248_2.jpg)

Fig. 2. The Attention maps for SparseSpikformer based on spike firing rate. The token with a redder color contains more significant information while with a bluer color represents less important information.

we visualize the spike firing rate of the Spiking Self-Attention module, which highlights the importance of different patches within the input image. Specifically, the input image contains some uninformative background tokens, which make a limited contribution to the overall understanding of global information while increasing the computational burden. To overcome this, we incorporate the Spiking Token Select module before the SSA module and gradually drop those useless image tokens according to the predefined pruning level. The detailed processing procedure is demonstrated as follows:

Firstly, we apply global average pooling (GAP) to the input spike sequence $X \in \mathbb{R}^{T \times N \times C}$ along the temporal dimension $T$, resulting in $X' \in \mathbb{R}^{N \times C}$ as the input feature.

$$
X' = \text{GAP}(X),\quad X' \in \mathbb{R}^{N \times D}, X \in \mathbb{R}^{T \times N \times D} \tag{1}
$$

Afterwards, the input feature $X'$ is fed into an MLP layer

<table>
<caption>Table 1. Performance of models under different keep ratios.</caption>
<tbody>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Keep ratio</th>
<td colspan="3">CIFAR10</td>
<td colspan="3">CIFAR100</td>
<td colspan="3">CIFAR10-DVS</td>
<td colspan="3">DVS-128</td>
</tr>
<tr>
<td>Acc</td>
<td>Throughput</td>
<td>FLOPs</td>
<td>Acc</td>
<td>Throughput</td>
<td>FLOPs</td>
<td>Acc</td>
<td>Throughput</td>
<td>FLOPs</td>
<td>Acc</td>
<td>Throughput</td>
<td>FLOPs</td>
</tr>
<tr>
<th>Spikformer [17]</th>
<th>1.0</th>
<td>94.95</td>
<td>950.28</td>
<td>3.74G</td>
<td>77.28</td>
<td>1049.55</td>
<td>3.74G</td>
<td>78.3</td>
<td>322.18</td>
<td>7.78G</td>
<td>98.26</td>
<td>304.25</td>
<td>7.78G</td>
</tr>
<tr>
<th rowspan="5">Sparse Spikformer</th>
<th>0.9</th>
<td>95.22</td>
<td>1293.32</td>
<td>3.44G</td>
<td>77.81</td>
<td>1156.07</td>
<td>3.44G</td>
<td>79.1</td>
<td>336.84</td>
<td>7.36G</td>
<td>98.26</td>
<td>315.52</td>
<td>7.36G</td>
</tr>
<tr>
<th>0.8</th>
<td>95.18</td>
<td>1381.73</td>
<td>3.24G</td>
<td>77.70</td>
<td>1245.03</td>
<td>3.24G</td>
<td>79.3</td>
<td>340.46</td>
<td>6.93G</td>
<td>97.91</td>
<td>331.87</td>
<td>6.93G</td>
</tr>
<tr>
<th>0.7</th>
<td>95.01</td>
<td>1448.71</td>
<td>3.04G</td>
<td>77.42</td>
<td>1324.60</td>
<td>3.04G</td>
<td>79.3</td>
<td>347.64</td>
<td>6.57G</td>
<td>97.91</td>
<td>346.73</td>
<td>6.57G</td>
</tr>
<tr>
<th>0.6</th>
<td>95.03</td>
<td>1521.58</td>
<td>2.88G</td>
<td>77.09</td>
<td>1394.16</td>
<td>2.88G</td>
<td>78.4</td>
<td>374.96</td>
<td>6.27G</td>
<td>97.91</td>
<td>360.18</td>
<td>6.27G</td>
</tr>
<tr>
<th>0.5</th>
<td>94.77</td>
<td>1586.64</td>
<td>2.75G</td>
<td>76.78</td>
<td>1455.03</td>
<td>2.75G</td>
<td>79.1</td>
<td>408.90</td>
<td>6.03G</td>
<td>97.91</td>
<td>371.13</td>
<td>6.03G</td>
</tr>
</tbody>
</table>

![](./images/931974443775820248_3.jpg)

Fig. 3. Visualization of the token pruning process.

consisting of only two linear layers to generate a series of token probability score maps.

$$
S = \text{Softmax}(\text{MLP}(X')),\quad S \in \mathbb{R}^{N \times 2} \tag{2}
$$

The two-dimensional of $S$ represents the probability scores for keeping or dropping N image tokens, respectively.

In order to make the probability scores differentiable, we use the Gumbel-Softmax [19] function to derive the final keep decisions.

$$
\mathbf{D} = \text{Gumbel-Softmax}(S_{*,1}),\quad \mathbf{D} \in \{0,1\}^N \tag{3}
$$

In addition, since the token pruning ratio varies across encoder blocks, it becomes necessary to employ different decisions in different layers. Therefore, as the input data passes through different encoder layers, we need to adjust the current token decision $\hat{\mathbf{D}}$ using the Hadamard product $\odot$.

$$
\hat{\mathbf{D}} = \hat{\mathbf{D}} \odot \mathbf{D} \tag{4}
$$

### 2.2. Weight Parameters Pruning

Although Spikformer has demonstrated remarkable performance on various datasets, its sparse nature of SNN implies that only a fraction of neurons fire spikes at any given moment, resulting in potential wastage of computing resources. To address this issue, we hypothesize that non-spiking neuron connections can be removed via pruning techniques as a means of reducing storage space resources.

As an efficient neural network pruning method, LTH can discover the winning tickets in both ANN and SNN models, making it much sparse [20]. Besides, the accuracy of the subnetwork can be further improved through Iterative Magnitude Pruning (IMP) [21] and retraining. In the first iteration of the pruning process, LTH generates a dense neural network model $f(x;\theta)$ using conventional random initialization and trains it until convergence. Then, the weight parameters are sorted, and the $p\%$ of the connections with the smallest absolute values are removed. After that, the remaining weight parameters are reinitialized, resulting in the formation of the subnetwork $f(x;M \odot \theta)$, where the mask $M \in \{0,1\}$ denotes the pruning or preservation of the connectivity relationships from the original network $f(x;\theta)$. The pruned subnetwork is subsequently retrained in the next iteration, and this iterative process will be repeated for $K$ times. Ultimately, the resulting subnetwork prunes $p^K\%$ of the weights, leading to a more sparse model. In our experiments, we set the values of $p$ and $K$ to 25 and 15.

## 3. EXPERIMENTS

### 3.1. Experimental Settings

We evaluate SparseSpikformer on four publicly available datasets, including static datasets (CIFAR-10, CIFAR-100) [22] and neuromorphic datasets (CIFAR10-DVS [23], DVS-128 [24]). During the training process, we insert the token selector after the $2^{nd}$, $3^{rd}$, and $4^{th}$ layers of the encoder block. Moreover, we also conduct a series of experimental comparisons with different sparsities. We follow the training principles and optimization methods proposed in the original Spikformer [17]. All the experiments are conducted on the NVIDIA Tesla V100 with a single GPU.

### 3.2. Token Pruning Results

In Table 1, we report the results achieved by SparseSpikformer on different keep ratios $\rho$ and multiple datasets. These results encompass three key metrics, including Top-1 accuracy, Throughput, and FLOPs. It is obvious that SparseSpikformer demonstrates a significant reduction in floating-point calculation costs by 22.49% to 26.47%, as well as an accelerated inference runtime by 21.98% to 66.96% while preserving minimal loss in accuracy. The experimental results unmistakably highlight the exceptional performance of SparseSpikformer, which provides support for deploying applications in resource-constrained environments.

Throughout the experiments, we gradually remove a fraction of tokens before each block, preserving distinct ratios of

Table 2. Accuracy comparison across different weight sparsity levels.

<table>
 <thead>
  <tr>
   <th colspan="2" rowspan="2">Model</th>
   <th colspan="2">CIFAR10</th>
   <th colspan="2">CIFAR100</th>
   <th colspan="2">CIFAR10-DVS</th>
   <th colspan="2">DVS-128</th>
  </tr>
  <tr>
   <th>Sparsity</th>
   <th>Top-1 Acc</th>
   <th>Sparsity</th>
   <th>Top-1 Acc</th>
   <th>Sparsity</th>
   <th>Top-1 Acc</th>
   <th>Sparsity</th>
   <th>Top-1 Acc</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td colspan="2" rowspan="3">Spikformer [17]</td>
   <td>75.90%</td>
   <td>94.78</td>
   <td>75.90%</td>
   <td>77.71</td>
   <td>81.3%</td>
   <td>77.1</td>
   <td>85.7%</td>
   <td>98.26</td>
  </tr>
  <tr>
   <td>86.22%</td>
   <td>94.91</td>
   <td>86.22%</td>
   <td>76.53</td>
   <td>89.0%</td>
   <td>76.8</td>
   <td>93.4%</td>
   <td>98.26</td>
  </tr>
  <tr>
   <td>93.90%</td>
   <td>94.34</td>
   <td>92.03%</td>
   <td>75.29</td>
   <td>95.8%</td>
   <td>75.9</td>
   <td>95.8%</td>
   <td>98.26</td>
  </tr>
  <tr>
   <td colspan="2" rowspan="3">SparseSpikformer<br>$\rho$=0.7</td>
   <td>73.26%</td>
   <td>95.28</td>
   <td>75.90%</td>
   <td>77.48</td>
   <td>81.3%</td>
   <td>77.9</td>
   <td>85.7%</td>
   <td>98.95</td>
  </tr>
  <tr>
   <td>88.74%</td>
   <td>95.06</td>
   <td>86.23%</td>
   <td>76.81</td>
   <td>89.0%</td>
   <td>78.5</td>
   <td>93.4%</td>
   <td>98.26</td>
  </tr>
  <tr>
   <td>93.90%</td>
   <td>94.97</td>
   <td>92.04%</td>
   <td>75.12</td>
   <td>95.8%</td>
   <td>76.2</td>
   <td>95.8%</td>
   <td>98.26</td>
  </tr>
 </tbody>
</table>

Table 3. Comparisons with different pruning methods.

<table>
 <thead>
  <tr>
   <th rowspan="2">Method</th>
   <th rowspan="2">$\rho$</th>
   <th>CIFAR10</th>
   <th>CIFAR100</th>
   <th>CIFAR10-DVS</th>
   <th>DVS-128</th>
  </tr>
  <tr>
   <th>Top-1 Acc</th>
   <th>Top-1 Acc</th>
   <th>Top-1 Acc</th>
   <th>Top-1 Acc</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>Random</td>
   <td>0.8</td>
   <td>94.20</td>
   <td>73.11</td>
   <td>78.9</td>
   <td>97.56</td>
  </tr>
  <tr>
   <td>Selector</td>
   <td>0.7</td>
   <td>94.07</td>
   <td>72.72</td>
   <td>78.9</td>
   <td>97.56</td>
  </tr>
  <tr>
   <td></td>
   <td>0.6</td>
   <td>93.02</td>
   <td>70.32</td>
   <td>78.2</td>
   <td>97.56</td>
  </tr>
  <tr>
   <td>Spiking</td>
   <td>0.8</td>
   <td>95.18</td>
   <td>77.70</td>
   <td>79.3</td>
   <td>97.91</td>
  </tr>
  <tr>
   <td>Token</td>
   <td>0.7</td>
   <td>95.01</td>
   <td>77.42</td>
   <td>79.3</td>
   <td>97.91</td>
  </tr>
  <tr>
   <td>Selector</td>
   <td>0.6</td>
   <td>95.03</td>
   <td>77.07</td>
   <td>78.4</td>
   <td>97.91</td>
  </tr>
 </tbody>
</table>

image tokens, such as $[\rho, \rho^2, \rho^3]$. Figure 3 provides a visual representation of this pruning process, with $\rho$=0.7.

### 3.3. Weight Pruning Results
In Table 2, we compare the accuracy of SparseSpikformer and Spikformer models at different levels of weight sparsity. The "sparsity" refers to the percentage of weights that have been pruned from the overall parameters. For our experiment, we maintain a fixed token keep rate $\rho$ of 0.7 for SparseSpikformer. Our results show that LTH can successfully identify winning tickets in both the Spikformer and SparseSpikformer models, leading to outstanding performance. Additionally, based on the data presented in Tables 1 and 2, it is clear that SparseSpikformer (Table 2) achieves a remarkable balance between accuracy and efficiency, closely matching the accuracy of the original Spikformer (Table 1) even with a 90% reduction in weight parameters. This reduction significantly alleviates the challenges associated with weight storage.

### 3.4. Ablation Analysis
Effectiveness of token pruning techniques. To verify the effectiveness of the spiking token selector module, we compare two different pruning strategies across multiple datasets. The results are summarized in Table 3. Specifically, we employ the random selector to randomly prune image tokens, while the spiking token selector is adopted to selectively prune tokens with the lowest spike firing rates. The experimental results clearly indicate that the spiking token selector module can achieve a better performance than the random selector. Based on that, we can infer that our approach is effective for dynamic token pruning.

Effectiveness of weight pruning techniques. In Figure 4, we present the results of our experiments to evaluate the performance of different LTH techniques, including IMP and EB [25], on the CIFAR10 and CIFAR100 datasets. The results clearly demonstrate that both IMP and EB techniques are able to identify the winning tickets within the original network. However, when evaluating the performance of subnetworks with high weight sparsity (e.g. 75.90%, 86.22%, and 92.04%), IMP consistently outperforms EB in terms of Top-1 accuracy, achieving superior results by 3.52% to 9.74%.

![](./images/931974443775820248_4.jpg)

Fig. 4. Illustration of the accuracy of Iterative Magnitude Pruning (IMP), Early-Bird (EB), and the Random Re-initialization (RR) techniques in LTH.

Furthermore, during the training process, we also compare the effects of network re-initialization techniques using Random re-initialization (RR) and Rewinding (IMP). The experiments indicate that the accuracy of the winning ticket through Rewinding is closer to that of the original network.

## 4. CONCLUSION
In this paper, we introduce SparseSpikformer, a sparse and hardware-friendly model based on Spikformer architecture. By incorporating the Spiking Token Selector module and the LTH weight pruning technique, we can effectively achieve a balance between accuracy and hardware resource trade-offs. The experimental results demonstrate that our model performs well on both static and neuromorphic datasets. Additionally, SparseSpikformer also accelerates runtime during the inference phase and cuts down the floating-point calculations, which provides possibilities for implementation on edge devices.

## 5. REFERENCES

[1] Wolfgang Maass, "Networks of spiking neurons: the third gen- eration of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659-1671, 1997.

[2] Filipp Akopyan, Jun Sawada, Andrew Cassidy, Rodrigo Alvarez-Icaza, John Arthur, Paul Merolla, Nabil Imam, Yutaka Nakamura, Pallab Datta, Gi-Joon Nam, et al., "Truenorth: De- sign and tool flow of a 65 mw 1 million neuron programmable neurosynaptic chip," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 34, no. 10, pp. 1537-1557, 2015.

[3] Mike Davies, Narayan Srinivasa, Tsung-Han Lin, Gautham Chinya, Yongqiang Cao, Sri Harsha Choday, Georgios Dimou, Prasad Joshi, Nabil Imam, Shweta Jain, et al., "Loihi: A neu- romorphic manycore processor with on-chip learning," *IEEE Micro*, vol. 38, no. 1, pp. 82-99, 2018.

[4] Jing Pei, Lei Deng, Sen Song, Mingguo Zhao, Youhui Zhang, Shuang Wu, Guanrui Wang, Zhe Zou, Zhenzhi Wu, Wei He, et al., "Towards artificial general intelligence with hybrid tian- jic chip architecture," *Nature*, vol. 572, no. 7767, pp. 106-111, 2019.

[5] Yangfan Hu, Huajin Tang, and Gang Pan, "Spiking deep resid- ual networks," *IEEE Transactions on Neural Networks and Learning Systems*, 2021.

[6] Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li, "Going deeper with directly-trained larger spiking neural net- works," in *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2021, vol. 35, pp. 11062-11070.

[7] Wei Fang, Zhaofei Yu, Yanqi Chen, Timothée Masquelier, Tiejun Huang, and Yonghong Tian, "Incorporating learnable membrane time constant to enhance learning of spiking neu- ral networks," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 2661-2671.

[8] Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu, "Temporal efficient training of spiking neural network via gra- dient re-weighting," *arXiv preprint arXiv:2202.11946*, 2022.

[9] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko- reit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin, "Attention is all you need," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa De- hghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *arXiv preprint arXiv:2010.11929*, 2020.

[11] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou, "Train- ing data-efficient image transformers & distillation through at- tention," in *International Conference on Machine Learning (ICML)*. PMLR, 2021, pp. 10347-10357.

[12] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko, "End-to- end object detection with transformers," in *European Confer- ence on Computer Vision (ECCV)*. Springer, 2020, pp. 213-229.

[13] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xi- ang, Philip HS Torr, et al., "Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers," in *Proceedings of the IEEE/CVF Conference on Computer Vi- sion and Pattern Recognition (CVPR)*, 2021, pp. 6881-6890.

[14] Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon, "Detr3d: 3d ob- ject detection from multi-view images via 3d-to-2d queries," in *Conference on Robot Learning (CoRL)*. PMLR, 2022, pp. 180-191.

[15] Zi-Hang Jiang, Qibin Hou, Li Yuan, Daquan Zhou, Yujun Shi, Xiaojie Jin, Anran Wang, and Jiashi Feng, "All tokens matter: Token labeling for training better vision transformers," *Ad- vances in Neural Information Processing Systems (NeurIPS)*, vol. 34, pp. 18590-18602, 2021.

[16] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yix- uan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al., "Swin transformer v2: Scaling up capacity and resolution," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 12009-12019.

[17] Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng Yan, Yonghong Tian, and Li Yuan, "Spikformer: When spiking neural network meets transformer," *arXiv preprint arXiv:2209.15425*, 2022.

[18] Jonathan Frankle and Michael Carbin, "The lottery ticket hy- pothesis: Finding sparse, trainable neural networks," *arXiv preprint arXiv:1803.03635*, 2018.

[19] Eric Jang, Shixiang Gu, and Ben Poole, "Categorical reparameterization with gumbel-softmax," *arXiv preprint arXiv:1611.01144*, 2016.

[20] Youngeun Kim, Yuhang Li, Hyoungseob Park, Yeshwanth Venkatesha, Ruokai Yin, and Priyadarshini Panda, "Explor- ing lottery ticket hypothesis in spiking neural networks," in *European Conference on Computer Vision (ECCCV)*. Springer, 2022, pp. 102-120.

[21] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin, "Stabilizing the lottery ticket hypothesis," *arXiv preprint arXiv:1903.01611*, 2019.

[22] A. Krizhevsky and G. Hinton, "Learning multiple layers of fea- tures from tiny images," *Handbook of Systemic Autoimmune Diseases*, vol. 1, no. 4, 2009.

[23] Hongmin Li, Hanchao Liu, Xiangyang Ji, Guoqi Li, and Lup- ing Shi, "Cifar10-dvs: an event-stream dataset for object clas- sification," *Frontiers in Neuroscience*, vol. 11, pp. 309, 2017.

[24] Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jef- frey McKinstry, Carmelo Di Nolfo, Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, et al., "A low power, fully event-based gesture recognition system," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 7243-7252.

[25] Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G Baraniuk, Zhangyang Wang, and Yingyan Lin, "Drawing early-bird tickets: Towards more efficient training of deep networks," *arXiv preprint arXiv:1909.11957*, 2019.