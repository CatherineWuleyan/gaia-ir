# Successfully Applying Lottery Ticket Hypothesis to
［#1］
Diffusion Model

［#2］
Chao Jiang¹ Bo Hui²* Bohan Liu³ Da Yan⁴
Visa¹, University of Tulsa², Carnegie Mellon University³, Indiana University Bloomington⁴
chajiang@visa.com, bo-hui@utulsa.edu
bohanli2@andrew.cmu.edu, yanda@iu.edu

## Abstract
［#3］
Despite the success of diffusion models, the training and inference of diffusion
models are notoriously expensive due to the long chain of the reverse process.
In parallel, the Lottery Ticket Hypothesis (LTH) claims that there exists winning
tickets (i.e., a properly pruned sub-network together with original weight initial-
ization) that can achieve performance competitive to the original dense neural
network when trained in isolation. In this work, we for the first time apply LTH
to diffusion models. We empirically find subnetworks at sparsity $90\% - 99\%$
without compromising performance for denoising diffusion probabilistic models
on benchmarks (CIFAR-10, CIFAR-100, MNIST). Moreover, existing LTH works
identify the subnetworks with a unified sparsity along different layers. We observe
that the similarity between two winning tickets of a model varies from block to
block. Specifically, the upstream layers from two winning tickets for a model tend
to be more similar than the downstream layers. Therefore, we propose to find the
winning ticket with varying sparsity along different layers in the model. Experi-
mental results demonstrate that our method can find sparser sub-models that require
less memory for storage and reduce the necessary number of FLOPs. Codes are
available at https://github.com/osier0524/Lottery-Ticket-to-DDPM.

## 1 Introduction
［#4］
Diffusion models [37, 19, 39] have achieved state-of-the-art results in a wide range of applications
such as image generation [38, 29, 23, 34], text-to-image [35, 30, 33], video generation [43, 20],
audio generation [24, 31], and protein generation [44, 41]. These generative models are powerful to
produce high-quality data by corrupting the data with slowly increasing noise and then learning to
reverse this corruption. For example, Denoising diffusion probabilistic modeling (DDPM) [19] trains
a sequence of probabilistic models to reverse each step of the noise corruption.

［#5］
Although diffusion models have shown impressive performance in capturing distributions and sample
quality, they are notoriously slow to generate data due to the long chain of reversing the diffusion
process. The noisy data will go through the same U-Net-based generator network thousands of times
or even more [19, 42]. At the same time, diffusion models are also notoriously hungry to train. They
require many iterations and large size of data to capture the complex data distributions. For example,
it takes over two weeks to train DDPM [19] on eight V100 GPUs for $256 \times 256$ resolution datasets.
The reported training time of a state-of-the-art diffusion model in [11] is over 100 days on V100
GPU days to generate high-quality image samples. Moreover, as the image resolution and the size
of the training data increases, the training and inference costs grow exponentially. To improve the
training efficiency and inference speed, many efficient sampling methods have been proposed, such

---
［#6］
*Corresponding author

［#7］
Neural Information Processing Systems (NeurIPS) 2023 Workshop on Diffusion Models.

［#8］
as DDIM [38], DPM-Solver [26], EDM-Sampling [23]. Different from these fast solvers, we propose to mitigate the computational cost by pruning the reverse model.

［#9］
The Lottery Ticket Hypothesis (LTH) [14] states that a dense neural network model contains a highly sparse subnetwork (i.e., winning tickets) that can achieve even better performance than the original model. The winning tickets can be identified by training a network and pruning its parameters with the smallest magnitude in an iterative way or one-shot way. Before it is trained in each iteration, the weights will be reset to the original initialization. The identified winning tickets are retrainable to reduce the high memory cost and long inference time of the original neural networks, which has been proved by many works [15, 12, 28, 47, 4]. The existence of winning tickets has been verified in both experiments [27, 40] and theory [45, 36, 3, 13, 28, 9]. LTH has been extended to find the winning tickets for different kinds of neural networks such as GANs [8, 22], Transformers [2, 32, 6, 1] and GNNs [7, 21, 18]. It has also been applied in various domains including computer vision and natural language processing [16, 46, 5]. We for the first time propose to apply LTH to diffusion models. Different from existing works based on efficient sampling, we aim to reduce the number of parameters for efficient training and inference. Specifically, we perform the empirical study to investigate whether there exists a trainable subnetwork of the diffusion model with original initialization that can achieve competitive performance than the original diffusion model. The answer is affirmative. We conclude that the winning tickets achieve the same performance with 90% floating-point operations (FLOPs) saving on the original diffusion model.

［#10］
We remark that the existing works in LTH identify the winning ticket with a unified sparsity along different layers. That is, we use the same pruning ratio to mask the parameters in the model. In this paper, we empirically found that the similarity between two winning tickets of a given model varies from module to module. Specifically, we introduce centered kernel alignment (CKA) as an index to measure the similarity between the sparsified modules from two winning tickets. We observe that the similarity at the upstream modules is higher than that at the downstream modules. This motivates us to configure the pruning ratio to be different at different modules. Based on the observation, we configure the pruning ratio to be lower at the upstream layer so that the sparsity will be lower for these layers. Intuitively, we need to make sure there are enough parameters to be trained so that meaningful full hidden states can be learned from noisy input data. In the experiment, we verify that our configuration can result in sparser winning tickets without performance compromise.

［#11］
Since the combination of sparse architectures and initializations in a winning ticket can reveal the potential implications for theoretical study of optimization and generalization in diffusion models, we can take inspiration from winning tickets to design new architectures for the diffusion process, we hope to stimulate the research progress of improving the inference speed of diffusion models.

［#12］
The contribution of this work can be summarised as:
- We for the first time apply the lottery ticket hypothesis to the diffusion model. Using a pruning method based on magnitude, we identify subnetworks at 99% sparsity in DDMP without performance compromise.
- We propose to identify a winning ticket with a varied sparsity along different layers, which is different from existing pruning algorithms in LTH. The proposed method can result in winning tickets with higher sparsity.
- The empirical result verifies the quality of pictures generated by a winning ticket is even higher than that generated by the original DDPM.

## 2 Preliminary
［#13］
We focus on the DDPM in this paper. Given an input $\mathbf{x}_0$, the diffusion process gradually adds Gaussian noise based on a variance schedule $\beta_1, \cdot, \beta_T$. Denote $\boldsymbol{\theta}$ as the parameters to learn the distribution $p_{\boldsymbol{\theta}}(\mathbf{x}_{\mathbf{t}-1}|\mathbf{x}_{\mathbf{t}}) = \mathcal{N}(\mathbf{x}_{\mathbf{t}-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_{\mathbf{t}}, \mathbf{t}), \sum_{\boldsymbol{\theta}}(\mathbf{x}_{\mathbf{t}}, \mathbf{t}))$ in the reverse process.

［#14］
Given the neural network parameterized by $\boldsymbol{\theta}$, a subnetwork is parameterized by $\boldsymbol{\theta} \odot \mathbf{m}$, where $\mathbf{m} \in \{0, 1\}^{||\boldsymbol{\theta}||_0}$ is a pruning mask for $\boldsymbol{\theta}$ and $\odot$ indicates the element-wise product. We use $||\cdot||_0$ to represent the $L_0$ norm counting the number of non-zero elements. The value 0 in $\mathbf{m}$ means the corresponding parameter $\boldsymbol{\theta}$ will be masked. The sparsity of a subnetwork is measured as $1 - \frac{||\mathbf{m}||_0}{||\boldsymbol{\theta}||_0}$.

［#15］
Modern pruning methods can be classified into structured pruning and unstructured pruning. In general, structured pruning removes entire groups of neurons, filters, or channels of neural networks


［#15］
while unstructured pruning results in unstructured sparse matrices. The pruning method based on the magnitude in LTH belongs to the unstructured pruning category. We use an iterative way to find a subnetwork $\boldsymbol{\theta}_{\tau} \odot \mathfrak{m}$, where $\boldsymbol{\theta}_{\tau}$ is the rewound initialization, which can reach the comparable performance to the full network within a similar training iteration when trained in isolation. After each iteration of training and pruning, we rewind the model with the parameters at $\tau$ epoch. The combination of $\boldsymbol{\theta}_{\tau}$ and $\mathfrak{m}$ with comparable performance is defined as a winning ticket.

## 3 The Existence of Winning Tickets in DDPM
［#16］
Existing works find the winning ticket by pruning the smallest magnitude in an iterative way. Given a pruning ratio $p\%$, we will sort the magnitude of weights after training and prune $p\%$ of parameters with the lowest magnitudes. In practice, existing work prunes the model layer by layer. It will result in a subnetwork where all layers have the same sparsity ($p\%$).

［#17］
We remark that it is not necessary to find a subnetwork with a unified sparsity along different layers. Intuitively, the input data is highly noisy in the denoising process and we need more parameters to learn a meaningful full hidden state. In this paper, we measure the similarity of two winning tickets based on canonical correlation analysis. Let $W_i^1$ and $W_i^2$ be the sparsified weight matrix of $i$th layer in the first and second winning ticket. We introduce Hilbert-Schmidt Independence Criterion (HSIC) to measure the similarity [17]: $\operatorname{HSIC}(K, L)=\frac{1}{(n-1)^{2}} \operatorname{tr}(K H L H)$ where $K_{i, j, k}=k(W_{i, j}^1, W_{i, k}^1)$ and $L_{i, j, k}=l(W_{i, j}^2, W_{i, k}^2)$, and $H$ is the centering matrix. Both $k(\cdot)$ and $l(\cdot)$ are the RBF kernels. HSIC can be considered as the maximum mean discrepancy between the joint distribution and the product of the marginal distributions [25]. The normalized similarity index is defined as:

［#17］
$$
\operatorname{CKA}(K, L)=\frac{\operatorname{HSIC}(K, L)}{\operatorname{HSIC}(K, K) \operatorname{HSIC}(L, L)}.
$$

［#18］
The effectiveness of this index has been verified by [10]. In Figure 1, we show the CKA between two "Conv2d" modules with the same order in the sequence from two different winning tickets of U-Net. We observe that the similarity is higher at upstream modules. It means there could be a potential implication at these upstream layers.

［#19］
Motivated by this observation, we propose to configure the pruning ratio to be lower at the upstream modules so that the sparsity will be lower for these modules. Algorithm 1 describes our pruning method. Take the U-Net model as an example. We first train the model parameter as $\boldsymbol{\theta}_{i}$ after $i$ iterations. Then for each of $J$ modules (conventional blocks in U-Net) in the sequential list, we prune the parameters with the lowest magnitude. We gradually increase the pruning ratio by $q$ as the index of a module increases in the U-Net model. After pruning, we will rewind the parameter to an early stage $(\tau=5\%*i)$ for the next iteration of pruning. We repeat this process until the desired sparsity $\delta$ is reached.

［#20］
![](./images/926184436477723015_1.jpg)

［#21］
Figure 1: Similarity between two winning tickets

［#22］
```algorithm
Algorithm 1 Finding winning tickets for DDPM
Input: Initial parameter $\boldsymbol{\theta}_{0}$, initial mask $\mathfrak{m}=\mathbf{1} \in \mathbb{R}^{\|\boldsymbol{\theta}\|}$, pruning ratio $p$, incremental rate $q$
Output: Sparsified masks $\mathfrak{m}$
 1: while $1-\frac{\|\mathfrak{m}\|_{0}}{\|\boldsymbol{\theta}\|_{0}}<\delta$ do
 2:    Train the diffusion model based on gradient $\nabla_{\boldsymbol{\theta}}$ for $i$ iterations
 3:    Arrive at parameters $\boldsymbol{\theta}_{i}$
 4:    for module $j=0,1,2,\cdots,J-1$ do
 5:        Pruning $(p+j*q)\%$ of the lowest-scored values in $j$th module $\boldsymbol{\theta}^{(j)}$
 6:        creating mask $\mathfrak{m}^{(\mathfrak{j})}$ for $j$th module
 7:    end for
 8:    Rewinding parameters to $\boldsymbol{\theta}_{\tau}$
 9:    $\mathfrak{m}=\{\mathfrak{m}^{(0)},\mathfrak{m}^{(1)},\cdot,\mathfrak{m}^{(\boldsymbol{J}-\mathbf{1})}\}$
10: end while
```

［#23］
Open discussion. We raise a new question regarding improving the efficiency of reversing: can we use sub-networks with different sparsity in the reverse process? Since a winning ticket can be


［#23］
considered an equivalent version of the original model, we can leverage a sparser sub-network in the late stage of denoising to further improve efficiency. Intuitively, the noise in the later reverse process has been reduced and it will be easier to optimize. The challenge lies in how to optimize a dense model and a winning ticket while guaranteeing the convergence of training. Another challenge is to decide at which step to use the winning ticket. We leave this open question for future investigation which we hope to stimulate the research on improving the efficiency of the diffusion model.

## 4 Experiment

［#24］
We conducted experiments to find the winning tickets in DDPM. We use CIFAR-10, CIFAR-100, and MNIST as our benchmark datasets. See the Appendix for more details of the experiment setting. Figure 4 shows the FID score with respect to the sparsity of the pruned U-Net. We observe that a sparsified model can even outperform the original model in terms of FID score. Moreover, by varying the sparsity, we can further reduce the sparsity of a winning ticket. The results verify the existence of winning tickets in DDPM, showing that we can find a winning ticket at sparsity $90\% - 99\%$ on the three benchmark datasets.

［#25］
![](./images/926184436477723015_2.jpg)

［#26］
Figure 2: Performance of DDPM w.r.t. sparsity of Unet

［#27］
We also visualize the quality of pictures generated by the winning ticket. Figure 3 depicts the denoising process of both the winning ticket and the original model on the CIFAR-10 dataset. We can see that the quality of the generated picture is still high when the sparsity is $99.4\%$. Figure 4 shows samples generated by the winning tickets. It further verifies that a winning ticket can generate a picture with the same quality as the original model. We remark that we reduce $90\%$ of FLOPs with the winning ticket compared with the original model.

［#28］
![](./images/926184436477723015_3.jpg)

［#29］
(a) Original DDPM

［#30］
![](./images/926184436477723015_4.jpg)

［#31］
(b) Winning ticket (Sparsity: 99.4%)

［#32］
Figure 3: Performance of DDPM w.r.t. sparsity of Unet

［#33］
![](./images/926184436477723015_5.jpg)

［#34］
(a) Original DDPM (b) Sub-network (Sparsity:
67.2%
(c) Winning ticket (Sparsity: 99.4%))

［#35］
Figure 4: Samples generated on CIFAR-10

# References

















































## A Appendix

［#36］
We have introduced three benchmark datasets in the experiment: CIFAR-10, CIFAR-100, and MNIST. A U-Net model is used in the DDPM model. We ran the experiment on a machine with 8 NVIDIA Tesla A100 GPUs. All the training parameters (e.g., training epochs, time steps, and learning rate) are configured as the default of the original DDPM model. We Prune the model for 25 iterations. The default pruning ratio is 20% and the incremental ratio is 1% by default.