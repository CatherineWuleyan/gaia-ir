# Reconstruction Task Finds Universal Winning Tickets

Ruichen Li
Peking University
xk-lrc@pku.edu.cn
Binghui Li
Peking University
libinghui@pku.edu.cn

Qi Qian
Alibaba Group
qi.qian@alibaba-inc.com
Liwei Wang
Peking University
wanglw@cis.pku.edu.cn

## Abstract
Pruning well-trained neural networks is effective to achieve a promising accuracy-efficiency trade-off in computer vision regimes. However, most of existing pruning algorithms only focus on the classification task defined on the source domain. Different from the strong transferability of the original model, a pruned network is hard to transfer to complicated downstream tasks such as object detection [7]. In this paper, we show that the image-level pretrain task is not capable of pruning models for diverse downstream tasks. To mitigate this problem, we introduce image reconstruction, a pixel-level task, into the traditional pruning framework. Concretely, an autoencoder is trained based on the original model, and then the pruning process is optimized with both autoencoder and classification losses. The empirical study on benchmark downstream tasks shows that the proposed method can outperform state-of-the-art results explicitly.

## 1 Introduction
Fine-tuning a pre-trained model, which can leverage the knowledge from a large-scale pre-training dataset, becomes prevalent for downstream tasks. This strategy avoids overfitting on small datasets leading to better performance on target tasks. Benefits from the pretrain-finetune strategy, scaling up model capacity is a trend in recent research [31, 4, 21]. However, large-scale models consume a lot of computational and memory resources, limiting their applications on edge devices. Many efforts are devoted to reducing the computational requirements of neural networks [15, 14, 29]. Among them, pruning [10] aims to remove unimportant parameters from the original model and can reduce the size of the model significantly. Most pruning methods rely on a well-trained network and can achieve extraordinary compression rates with negligible accuracy drop on the same task. [10, 35, 28]

Although pruning methods demonstrate an excellent accuracy vs. sparsity trade-off, only a few works evaluate the pruned model's transferability, i.e., the performance on different downstream tasks. Given multiple downstream tasks, a pruning algorithm can be applied to the individual task. However, the cost that linearly depends on the number of tasks will become intractable. To mitigate this problem, we try to find a pruned model, called universal winning tickets, that can transfer to diverse downstream tasks.

The lottery tickets hypothesis, proposed by [6], claims that each over-parameterized neural network has a sparse subnetwork called a winning ticket, which can achieve the same performance as the entire network. The transferability of winning tickets has been investigated in [23]. They show that an ImageNet ticket can transfer to different downstream classification tasks. In [1], the authors suggest that a pretraining procedure can be regarded as a special initialized method. This kind of initialization is directly amenable to sparsification. Based on this insight, the authors use task agnostic pretraining

Preprint. Under review.

to help find the universal winning tickets. Their results show that a universal winning ticket exists across different classification downstream tasks.

However, in some more complicated downstream tasks, such as object detection, tickets found by [1] can result in a degenerated performance. In [1], the authors reveal that tickets found by the target object detection task surpass tickets found by image classification with a non-negligible margin. In [7], the authors check tickets found by supervised learning on an object detection dataset [20]. The result confirms that ImageNet tickets only transfer to a limited extent to downstream tasks, such as object detection or instance segmentation. These observations illustrate that the pruning method's transferability is highly related to the task type.

In this paper, we aim to find the universal tickets for diverse downstream tasks. Most of the existing methods rely on an image-level task to prune pre-trained models. Although the pruning pipeline has shown an extraordinary performance on downstream tasks [12], it does not treat details and global features in the same status. The implicit tendency of image-level loss causes the neural network to forget pixel-level information during the pruning process. After pruning, pixel-level information becomes untraceable while it is essential for complicated downstream tasks, e.g., detection, segmenta- tion, etc. The intuition is theoretically analyzed in Section 3. Therefore, a pixel-level task is necessary for pruning pre-trained models to preserve sufficient information and can help the model transfer to generic downstream tasks.

Unlike image-level tasks, designing appropriate pixel-level tasks is still challenging. Inspired by the recent progress in self-supervised learning [13], we introduce the image reconstruction task to find the universal tickets, and a two-stage training paradigm is proposed to obtain the desired ticket. First, an autoencoder structure is introduced for the existing model. The encoder structure inherits the original model structure and weights. Unlike an end-to-end unsupervised pre-training in [13], which requires an extremely large model and high mask rate to avoid cheating model, a much smaller decoder is trained in our method for a specific encoder. We use the feature map hint method to accelerate the convergence of decoder. After the first stage of training for the decoder, we have a classification task with a classification head for the second stage of training. Concretely, we freeze the decoder and apply a modified LTH algorithm to get a universal ticket. Finally, the performance is evaluated by transferring the obtained tickets to different downstream tasks.

Our contributions can be summarized as follows.

- We propose a new framework for pruning pre-trained neural networks. Different from directly pruning on the classification tasks, we first train a decoder for the pruned network and then introduce the reconstruction loss. The pruned model is applicable for different downstream tasks, especially object detection and instance segmentation.
- Our result suggests that pixel-level tasks are better than traditional image-level tasks for pruning pre-trained neural networks. Although contrastive learning and classification tasks have been proved to be useful pretraining tasks for large models, the pruning method relying on those tasks may degenerate the transferability. By introducing an appropriate pixel-level task, a pruned model generalizes better on downstream tasks.
- The proposed method is evaluated on benchmark downstream tasks. It achieves 32.7%AP on the COCO dataset when only keeping about 20% parameters of the original model. The superior performance over state-of-the-art result [7] confirms the effectiveness of our method.

## 2 Related Work

### Pruning and Lottery Tickets Hypothesis
Pruning aims to remove the unimportant weights of a neural network to reduce computation costs. It was first proposed in [18] where the authors use the Hessian matrix to estimate the importance of parameters. In [10], the authors propose iterative magnitude pruning to achieve a better compression rate. A lot of works follow [10] setting and achieve promising results. Different to those methods, the lottery tickets hypothesis, proposed in [6], suggests that a sparse trainable subnetwork exists in every over-parameterized models. This sparse network can achieve similar performance as the entirty. To verify this assumption, [6] follow the iterative pruning paradigm but set the model parameter to initial values at each pruning round. Some works [19, 33, 30] attempt to find the winning tickets in an initialized network. Those methods can


find winning tickets in small datasets. However, as stated in [22], the original LTH method fails with more complicated datasets and large learning rates. In [27], the authors find that rewinding the parameter to the early training stage of the neural network, rather than the initial value, can bring profits to winning tickets in complicated datasets.

Large Scale Pretrain ImageNet pretraining is widely used in nowadays computer vision training pipeline. It is common sense that using a pretrained network on a large dataset can benefit downstream tasks, both in accuracy and training epochs. Nowadays, self-supervised pretrain methods have become more popular because they can utilize unlabeled data. Image-level self-supervised pretraining has been developed for years [12, 2, 9, 25]. In those methods, an image is encoded into a single representation vector. The classifier should distinguish a strongly augmented image from irrelevant ones by using their representation vector. Recently, pixel-level or patch-level pretraining has attracted more attention. These methods focus on the recovery of corrupted images [26, 31, 13]. Most of them require an extremely large model such as ViT [4] to achieve better performance.

Autoencoder is a famous tradional machine learning structure. It is widely used in image denoising [32] and generative model [16]. Recently, as the image reconstruction task is proposed as a new method in self-supervised pretraining [13], autoencoder structure become useful in pretraining task. In order to get abundant semantic information of neural network and avoid cheating model, autoencoder usually use strong regulariziar or data augmentation and should take a lot of time to train. In this paper, we focus on pruning to downstream tasks, rather than get a better autoencoder. By this way, we seperately train the decoder and encoder of our neural network. We also introduce feature map hint method for acceleration. Thus, the decoder only needs to be trained for a much smaller epoch than previous works.

# 3 Image-level Tasks are not Sufficient Criterion for Pruning Neural Network

In this section, we show the insufficiency of image-level tasks for finding universal tickets. It is common sense that traditional image classification tasks can produce an abundant feature map so that their backbone can transfer to every downstream task. Thus, we use the difference between the original feature map and the pruned version to imply the transferability of a pruned model. We mainly study two simple but important variants of CNN: linear convolutional neural network (LCNN) and one-layer ReLU convolutional neural network (ORCNN). We focus on pruning in LCNN and focus on finetuning in ORCNN. Our results suggest that image-level tasks cannot produce a transferable pruned neural network. We focus on the 1D case in this section, but the 2D case is easy to extend.

## 3.1 Preliminary

Notations about Tensor A $k$-th order tensor $(a_{i_1\ldots i_k})$ is a $k$-dimensional array of real numbers $a_{i_1\ldots i_k}$. We use $\|\cdot\|$ and $\langle\cdot,\cdot\rangle$ to denote the standard $l_2$ norm and inner product of tensors. And we define the normalized $l_2$ distance between tensor $\mathbf{A}$ and $\mathbf{B}$ as

$$
dist_{l_2}^{N}(\mathbf{A}, \mathbf{B}) = \frac{\|\mathbf{A} - \mathbf{B}\|}{\sqrt{\|\mathbf{A}\|\|\mathbf{B}\|}} \tag{1}
$$

Convolution Operator is a linear operator represented by $*$. Let $\mathbf{x} = (x_{i,j}) \in \mathbb{R}^{c \times D}$ be the input, where $D$ is the length of input sequence and c is the channel number of input. Let $\mathbf{W} \in \mathbb{R}^{c' \times c \times (2s+1)}$ be the convolution tensor. Then the convolution between $\mathbf{W}$ and $\mathbf{x}$ is defined as:

$$
(\mathbf{W} * \mathbf{x})_{i,j} = \sum_{k=1}^{c} \sum_{l=-s}^{s} w_{ik,l}x_{k,j+l} \tag{2}
$$

where we use circular padding method, i.e. $x_{i,j+D} := x_{i,j}$.

Average Pooling Operator is also a linear operator $P_a$. For any matrix $\mathbf{v} \in \mathbb{R}^{c \times D}$, we have:

$$
P_a \mathbf{v} = \frac{1}{D} \sum_{i=1}^{D} \mathbf{v}_{:,i} \in \mathbb{R}^c \tag{3}
$$

Linear Convolutional Neural Networks (LCNN): LCNN is a linear mapping $\mathcal{C}: \mathbb{R}^{c \times D} \to \mathbb{R}^{m_{L+1} \times D}$, which can be defined as

$$
\mathcal{C}(x):=W^{L} * W^{L-1} * \ldots W^{0} * x
\tag{4}
$$

where $x \in \mathbb{R}^{c \times D}, W^{l} \in \mathbb{R}^{m_{l+1} \times m_{l} \times(2 s+1)}$.

One-hidden-layer ReLU Convolutional Neural Networks (ORCNN): Let $\mathbf{x}=(x_{i,j}) \in \mathbb{R}^{c \times D}$ be the input. $\mathbf{W} \in \mathbb{R}^{m \times c \times(2 s+1)}$ is the convolution tensor, where $m$ is the channel number of feature maps. ORCNN is defined as the following:

$$
\begin{aligned}
\mathbf{x}^{conv} &= \frac{1}{\sqrt{m}} \sigma(\mathbf{W} * \mathbf{x}) \\
\mathbf{x}^{pool} &= P_a \mathbf{x}^{conv} \\
f(\mathbf{x}) &= \langle \mathbf{a}, \mathbf{x}^{pool} \rangle
\end{aligned}
\tag{5}
$$

where $\mathbf{x}^{conv}$ , $\mathbf{x}^{pool}$ are hidden-layer outputs and $f(\mathbf{x})$ is the predicted label. We use $f_{\mathbf{a},\mathbf{W}}(\mathbf{x})$ to denote the newtork. $\sigma(\cdot)$ denotes ReLU activation function which is defined as $\sigma(\cdot):=\max (\cdot, 0)$. $\mathbf{a}=(a_1, a_2, ..., a_m)^\mathbf{T} \in \mathbb{R}^m$ are the fully connected weights.

### 3.2 Pruning in LCNN

In this section, we investigate the pruning step in LCNN. We first claim a simple proposition of LCNN:

Claim Let $\mathcal{C}$ be an LCNN, there must exists another LCNN $\mathcal{C}'$, such that

$$
P_a \mathcal{C}(x)=P_a \mathcal{C}'(x), \mathcal{C}(x) \neq \mathcal{C}'(x)
\tag{6}
$$

This is a direct result by considering the translation symmetry in LCNN. Now, the question becomes "Can this LCNN $\mathcal{C}'$ be found by pruning algorithm?". At least, can we find an LCNN $\mathcal{C}'$ by pruning, such that the change of $\|P_a \mathcal{C}(x)-P_a \mathcal{C}'(x)\|$ is small while $\|\mathcal{C}(x)-\mathcal{C}'(x)\|$ is large? To answer this question, we come out the following theorem:

**Theorem 3.1.** For any random initialized LCNN, where parameter is initialized as i.i.d $\mathcal{N}(0, \Delta)$. Then, for any $p<0.11$, we can prune $p$ proportion of weights and get a new LCNN $\mathcal{C}'$ with high probability, such that:

$$
\frac{\|P_a \mathcal{C}(x)-P_a \mathcal{C}'(x)\|}{\|P_a \mathcal{C}(x)\|}<C_1 p^{3 / 2}
$$

$$
\frac{\|\mathcal{C}(x)-\mathcal{C}'(x)\|}{\|\mathcal{C}(x)\|}>C_2 p^{1 / 2}
$$

Here $C_1$ and $C_2$ are constants related to the kernel size $s$ and the depth $L$

This theorem means that if we initialize the LCNN properly, we can find some neurons such that removing those neurons does not change the image-level feature vector a lot but destroys the feature map structure. Thus, using a pruning criterion based on image-level loss can not preserve the feature map of LCNN. The detailed proof is in Appendix A.

### 3.3 Finetuning in ORCNN

In this section, we focus on finetuning in ORCNN. Let $f_{\mathbf{a},\mathbf{W}}(\mathbf{x})$ denote the ORCNN parameterized by fully connected weight $\mathbf{a}$ and convolution tensor $\mathbf{W}$. The pruning pipeline can be formalized as the following three phases:

- **Pre-trained Phase:** We randomly initialize the parameters $\mathbf{a}$ and $\mathbf{W}$ to $\mathbf{a}_0$ and $\mathbf{W}_0$. Then, we train the model via image-level tasks on the given labeled dataset $S$ and derive a pre-trained model $f_{\mathbf{a}_{pre}, \mathbf{W}_{pre}}(\mathbf{x})$.
- **Pruning Phase:** We apply the structured pruning method to ORCNN with the pruning rate $p$.


- **Finetuning Phase:** We first reset unpruned parameters to initial values $\mathbf{a}_0$ and $\mathbf{W}_0$. Next, we finetune the network parameters on the same labeled dataset $S$ via the gradient descent algorithm. Finally, we derive the finetuned model $f_{\mathbf{a}_{fin},\mathbf{W}_{fin}}(\mathbf{x})$.

We mainly consider the training process in the finetuning phase. In the finetuning phase, we use the same dataset $S = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$ as in the pretraining phase and use loss function $L(f) := \frac{1}{2}\sum_{i=1}^n(f(\mathbf{x}_i) - y_i)^2$. We use the gradient descent method to update the convolutional tensor $\mathbf{W}$ and freeze the fully connected weights a:

$$
\mathbf{W}(t+1) = \mathbf{W}(t) - \eta \frac{\partial L}{\partial \mathbf{W}(t)}
\tag{7}
$$

where $\eta$ is the learning rate, and $t$ denotes $t^{th}$-iter.

As the original pre-trained model has a strong transferability to diverse downstream tasks, we believe the pre-trained model can learn a good representation of the data. Specifically, the pre-trained ORCNN can derive the feature maps from the image data by the pre-trained convolution tensor $\mathbf{W}_{pre}$. Thus, the 'difference' between convolution tensors $\mathbf{W}_{fin}$ and $\mathbf{W}_{pre}$ implies the transferability of pruned model.

In order to measure the difference between convolution tensors, we multiply an arbitrary rotation operator $\mathbf{Q}$ on the $\mathbf{W}_{fin}$ to recover its density. Then, we calculate the minimal normalized $l_2$ distance between $\mathbf{QW}_{fin}$ and $\mathbf{W}_{pre}$. The following Theorem 3.2 characterizes the lower bound of the distance under the over-parameterized setting.

Theorem 3.2. Assume that we set the channel number of feature maps $m = \Omega(\frac{1}{\delta^2}poly(n))$, and the finetuning learning rate $\eta$ is sufficiently small. After the finetuning phase, the finetuned convolution tensor is $\mathbf{W}_{fin}$. Then with probability at least $1 - \delta$ over the random initialization in the pre-trained phase, we have

$$
\min_{\mathbf{Q} \in \mathbb{O}} \{dist_{l_2}^N(\mathbf{QW}_{fin}, \mathbf{W}_{pre})\} \geq \frac{p}{2}
\tag{8}
$$

where $\mathbb{O}$ is rotation operator space and $p$ is the pruning rate.

The main idea of the proof is to analyze the dynamics of the model Gram matrix in the gradient descent process. The detailed proof can be found in Appendix B.

Theorem 3.2 suggests the lower bound of normalized $l_2$ distance between them is growing linearly with respect to the pruning rate $p$. It reveals the pruned model's ability to extract features is less than the original although it may have the same good performance as the original model in image-level tasks. Therefore, we demonstrate the insufficiency of image-level tasks for finding universal tickets.

## 4 Method

As we discussed in Section 3, only focusing on the image-level task during the pruning procedure will lead to a degenerated feature map. In general, using image-level loss as a pruning criterion tends to remove image details and destroy the structure of the feature map. That untraceable information will cause a significant accuracy drop on detection or segmentation tasks. To mitigate this problem, we attempt to introduce the pixel-level task to the traditional pruning framework. Our framework can be formalized into three stages:

i) Train an autoencoder. We modify the pretrained model to the encoder of an autoencoder structure. The last feature map of the original model becomes the compressed code of autoencoder; then, we freeze the encoder and start training. We use the feature map hint method (illustrated in Section 4.1) to accelerate the training process and improve performance.

ii) Prune the encoder. After decoder training, we prune the encoder part with reconstruction loss and classification loss simultaneously. During this pruning step, the decoder is frozen to keep the information gathered from the encoder. We follow a modified LTH pruning pipeline to get better performance.

iii) Adapt the encoder to the downstream tasks according to the standard transfer learning setting.


![](./images/867761916582298079_1.jpg)

Figure 1: **Overview of our framework.** First, we use the reconstruction loss and the feature map hint method to train an autoencoder structure. Next, we use the modified LTH algorithm for pruning the neural network. We combine classification loss and image reconstruction loss for finetuning procedure. Then, we transfer the encoder part to the downstream tasks. The encoder part is frozen during the autoencoder training process, while the decoder is frozen in the Modified LTH pruning process.

The overall structure design is described in Figure 1. We will describe the first two steps in our framework in the following sections.

### 4.1 Autoencoder Training
Image reconstruction is a conventional computer vision task but was introduced as a pretraining method recently[13]. Autoencoder is the basic architecture of image reconstruction. As the first step of our framework, the original model will be embedded in the autoencoder structure, which will be trained until the decoder captures the pixel-level information.

In this paper, we focus on ResNet structure, but our method can easily generalize to other kinds of structures. We remove the last pooling layer and fully connection layer of the original model as the encoder part. In this way, the final feature map is regarded as the compressed code of the autoencoder. Different from unsupervised pretraining, the decoder part in our method is an inversed ResNet. The training purpose of an autoencoder is to minimize

$$
\mathcal{L}_{r e c}=\frac{1}{N} \sum_{i=1}^{N}\left\|\mathcal{D}\left(\mathcal{F}\left(x_{i}\right)\right)-x_{i}\right\|^{2} \tag{9}
$$

where $N$ is the number of training samples, $\mathcal{F}$ represents the encoder part, $\mathcal{D}$ represents the decoder part, $x_{i}$ is the input image. Obviously, without any constraint on $\mathcal{F}$ or $\mathcal{D}$, the loss function will lead to a trivial solution. Therefore, we freeze the parameters in $F$ during the autoencoder training process.

### 4.2 Feature Map Hint
In previous works, autoencoder is widely used in generative [26] and reconstruction [8] tasks. Such tasks require a lot of training time because they need to finetune both the encoder and decoder. Unlike those training strategies, we treat the feature map as an effective representation and should not change during the autoencoder training step. Although the feature map has abundant information, the recovery of the whole image is difficult due to the resolution restriction.

We use the previous stage's feature map as a hint to the inversed ResNet decoder for better image reconstruction results. Directly transporting the feature map to the decoder part must lead to a fault model. The decoder tends to rely on low-level features while ignoring the high stage's information. Therefore, we mix the original feature map and the following decoder's feature map with a specific proportion $t$. Formally, let $f_{i}$ be the feature map in the encoder stage-$i$, $g_{i}$ is the output feature in the

```
Algorithm 1 Modified LTH
    Input A neural network $f(x;\theta)$, pretrained value $\theta_{pre}$, parameter remain percentage at each round
    $r=1-p$, total pruning round $R$, finetuning strategy $s$ in each round
    Output A set $S$ of pruned tickets at different pruning levels, $S = \{(1 - r^i, f_i)|1 \leq i \leq R\}$
    $m \leftarrow \mathbb{1}$, $\theta \leftarrow \theta_{pre}$, $S \leftarrow \emptyset$
    for $i = 1$ to $R$ do
        Finetune $f(x;\theta \odot m)$ following the strategy $s$
        Prune smallest non-masked $p$ values, update $m$
        $\theta \leftarrow \theta_{pre}$
        $S \leftarrow S \cup \{(1 - r^i, f(x;\theta \odot m)\}$
    end for
```

decoder stage-$i$, the input feature map of the decoder $i+1$ stage is:

$$
g_{i}'=(1-t)g_{i}+tf_{i} \tag{10}
$$

This mixing method can stabilize the finetuning process and avoid fault models. In practice, the proportion $t$ will be a relatively smaller number. In Section 6.2, we conduct an ablation study on the choice of feature map. The final result shows that using $f_3$ and $f_4$ as feature map hints for the decoder can achieve the best performance.

### 4.3 Reconstruction Loss in Pruning Step

With the autoencoder structure, we can add the reconstruction loss to the pruning process. The loss function during the pruning process is composed of two parts: $\mathcal{L}_{class}$ refers to the traditional classification loss, and $\mathcal{L}_{rec}$ refers to the reconstruction loss. We use a hyperparameter $\lambda$ to balance between $\mathcal{L}_{class}$ and $\mathcal{L}_{rec}$

$$
\mathcal{L}=\mathcal{L}_{class}(\mathcal{F})+\lambda \mathcal{L}_{rec}(\mathcal{F}, \mathcal{D}) \tag{11}
$$

Notice that the decoder $\mathcal{D}$ is related to $\mathcal{L}_{rec}$ in the above function, which means the decoder will be trained during the finetuning process. However, we hope the decoder part guides the finetuning process and transfers pixel-level information to the encoder. The decoder change will perturb the information that remains in the decoder and may lead to an unexpected result. It also slows down the training. Therefore, we freeze the parameter in the decoder part during the pruning process.

### 4.4 Modified LTH Pruning

There are two widely used pruning pipelines, IMP and LTH, where LTH reset the parameter to the initial value, but IMP does not. In this section, We argue that neither setting is the most capable method to find universal tickets. We introduce a modified LTH pipeline, which is showed more powerful to find universal tickets.

In previous works, [1] used an IMP method to produce universal tickets. Although the IMP method usually provides better accuracy on the upstream tasks, it may cause the neural network to fall in the minima of the upstream task while it is hard to finetune on the downstream task. In [7], the authors examine whether the ImageNet tickets produced by LTH can work for detection tasks. We should point out that their method does not utilize the pretrained network power and limits the performance of their method. We try to combine the strengths of those two methods.

We describe our modified LTH algorithm in detail. Inspired by [1], we replace the initial values in the original LTH method with the pretrained values. Let $f(x;\theta)$ represent the neural network, $\theta \in \mathbb{R}^n$. Pruning some parameters means we permanently set some dimensions of $\theta$ to zero. In this way, we can use mask $m \in \{0,1\}^n$ to describe the pruned parameter. Thus, a pruned network can be described by $f(x;m \odot \theta)$, where $m_i=0$ means we prune this parameter from the whole neural network. Let $\theta_{pre}$ represent the pretrained value of the original neural network. We train the network for $T$ epochs and prune the smallest $p$ proportion of unmasked parameters for every training round. After the prune step, the parameters $\theta$ reset to the pretrained values $\theta_{pre}$. Then another training round begins. In Algorithm 1, we state the pseudo-code of our method.


![](./images/867761916582298079_2.jpg)

Figure 2: Performance on the COCO detection and segmentation dataset. It is noticeable that our method outperforms the reported result in [7]. We can produce transferable tickets at sparsity around 80%.

## 5 Experiment
### 5.1 Setting

Dataset For the autoencoder training step and modified LTH pruning step, we conduct all experiments on ImageNet [3]. For the image-level transfer task, we still focus on image classification. We evaluate the tickets gathered from the ImageNet dataset on Cifar10 [17], Cifar100[17], and SVHN [24]. We also show the accuracy of ImageNet. For pixel/patch-level task transfer, we investigate object detection and instance segmentation. As stated in [7], ImageNet tickets transfer to the COCO [20] dataset is harder than transfer to small detection dataset such as VOC datasets. Therefore, We use the COCO dataset as a standard benchmark.

Model We evaluate our method with ResNet50, a standard CNN model on the ImageNet and COCO object detection/instance segmentation task backbone. We use official PyTorch pretrained weights as our pretrained values. The decoder is an inverse ResNet architecture. This decoder part is removed in the downstream tasks, and only the encoder part transfer. For classification transfer tasks, we adjust the first kernel size of ResNet50 to $3 \times 3$ and remove the first max-pooling layer. We use a famous structure mask RCNN[11] as the segmentation and detection head for object detection and instance segmentation tasks.

Training and Pruning Setting In the autoencoder training step, we train the decoder with the AdamW optimizer. We use a multi-step learning rate schedule with an initial learning rate 1e-4 and $\times 0.1$ at the 10, 30 epoch. The total training epoch is 50; the batch size is 512; weight decay is 2e-4. In the modified LTH pruning step, We follow the pruning setting in [1], where for each pruning round, we prune 20% parameters. Each pruning round has 10 epochs. We use the SGD optimizer, and the learning rate is kept 3e-4, the batch size is 512, momentum is 0.9. The reconstruction penalty $\lambda$ defined in (9) is 10, the feature map hint proportion is 0.1. For classification task transfer setting, we follow the setting in [1]. We draw the accuracy-compression rate curve for different classification tasks. For segmentation and detection tasks, we use a standard Detectron2[34] FPN $1\times$ training config. We mainly evaluate the performance at 7,8,9,10,11 pruning round (79.03%,83.22%,86.58%,89.26%,91.41%) to make a comparison with result in [7]. The numbers in the brackets represent the sparsity at each pruning round.

### 5.2 Detection and Segmentation Results

We evaluate our method on the COCO dataset. In Figure 2, we compare our method with the reported ImageNet tickets results in [7]. Notice that our results surpass the baseline results at every point of the accuracy-sparsity curve. Our method can achieve 32.7 mAP on the detection task and 30.3 mAP on the segmentation task at sparsity 79.03%. To make a fair comparison with the reported result in [7], we interpolate our accuracy-sparsity curve at 80% and 90% sparsity. The result is listed in Table 1. At high-level sparsity, e.g., 92% sparsity, our method's performance goes down due to the

limitation of ResNet50 itself. As stated in [7], it is hard for ResNet50 to find winning tickets at 90% sparsity, even in the setting of directly applying LTH on the downstream task.

## 5.3 Classification Results

![](./images/867761916582298079_3.jpg)

Figure 3: **Classification transfer results.** From left to right are the results of Cifar10, Cifar100, SVHN. In Cifar10 and SVHN datasets, our method is comparable to the directly applying LTH method. At Cifar100, our method outperforms with LTH method, showing the transferability of our selected tickets.

**Baseline setting** We compare our results with the baseline that directly performs LTH on the target dataset in classification task transfer. We perform 15 pruning rounds. Each pruning round has 182 epochs and will cut 20% parameters. The learning rate starts from 0.1 in each round, $\times0.1$ at 91, 136. The weight decay is 2e-4. This setting is the same as the task setting in [1].

We evaluate our tickets on Cifar10, Cifar100, SVHN. The results are presented in Figure 3. Our method is comparable or even better than directly applying the LTH method on the target dataset, especially in Cifar100 datasets. In Cifar100, our method achieves over 80% accuracy, while the LTH method only achieves 76% at the beginning of training.

Although we are mainly concerned about the pruned network's transferability, our method still achieves comparable ImageNet unstructured pruning results. As shown in Figure 4, our method has a similar performance with the iterative magnitude unstructured pruning result, which implies that our tickets are also winning tickets on the ImageNet.

## 6 Ablation Study

### 6.1 Comparison vs IMP Method

As we argued in Section 4.4, the IMP method is not the best way to create universal tickets. In this section, we compare the transferability of tickets produced by IMP and our method. We evaluate those tickets on detection downstream tasks at different sparsity. It is easy for the IMP method to

![](./images/867761916582298079_4.jpg)

Figure 4: The comparison between our method and IMP on ImageNet. We find that our method has a similar performance as the IMP result. Considering that our method only finetunes 10 epochs for each ticket, we believe that our ticket is also winning tickets on ImageNet.

Table 1: Object Detection and Segmentation Results. (mAP)

<table>
  <tr>
    <td>Task</td>
    <td colspan="2">Detection</td>
    <td colspan="2">Segmentation</td>
  </tr>
  <tr>
    <td>Sparsity</td>
    <td>80%</td>
    <td>90%</td>
    <td>80%</td>
    <td>90%</td>
  </tr>
  <tr>
    <td>[7]</td>
    <td>31.0</td>
    <td>30.7</td>
    <td>29.0</td>
    <td>28.6</td>
  </tr>
  <tr>
    <td>ours (interpolated)</td>
    <td>32.6</td>
    <td>31.1</td>
    <td>30.2</td>
    <td>28.9</td>
  </tr>
</table>

get a more sparse network in complicated datasets. However, we find out that in the transferability tasks, our method brings significant AP improvement in the downstream detection and segmentation datasets. The results are shown in Table 2

## 6.2 Feature Map Hint Selection

![](./images/867761916582298079_5.jpg)

Figure 5: The ablation study of different feature map hints. We use the detection transfer result at pruning round 8 (sparsity 83.22%) as the selection criterion. We observe that using $f_3$ and $f_4$ as feature map hints can induce the best result on detection datasets.

In this section, we investigate the influence of feature map hints to the detection results. As Figure 5 shows, different feature map hints lead to a different result. We attribute this phenomenon to the different feature maps' detailed and semantic information levels. If we only use a high-level feature for autoencoder, it takes a lot of network capacity and needs more epochs to converge; if we use a low-level feature map, those features are so close to the original information that the decoder part does not get useful information. We find that using $f_3, f_4$ as feature map hints can lead to the best

Table 2: Comparison with IMP Method. Object Detection and Segmentation Results. (mAP)

<table>
  <thead>
    <tr>
      <th>Sparsity (backbone)</th>
      <th colspan="2">Detection</th>
      <th colspan="2">Segmentation</th>
    </tr>
    <tr>
      <th></th>
      <th>IMP</th>
      <th>Ours</th>
      <th>IMP</th>
      <th>Ours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>79.02%</td>
      <td>32.1</td>
      <td>32.7</td>
      <td>29.9</td>
      <td>30.3</td>
    </tr>
    <tr>
      <td>83.22%</td>
      <td>31.8</td>
      <td>32.3</td>
      <td>29.6</td>
      <td>29.8</td>
    </tr>
    <tr>
      <td>86.57%</td>
      <td>31.3</td>
      <td>31.6</td>
      <td>29.1</td>
      <td>29.5</td>
    </tr>
    <tr>
      <td>89.26%</td>
      <td>30.8</td>
      <td>31.4</td>
      <td>28.7</td>
      <td>29.2</td>
    </tr>
    <tr>
      <td>91.41%</td>
      <td>30.1</td>
      <td>30.5</td>
      <td>28.0</td>
      <td>28.4</td>
    </tr>
  </tbody>
</table>

performance compared to other settings. So we use $f_3$, $f_4$ as the feature map hints choice on other settings.

## 7 Conclusion

In this paper, we propose a new framework to find the universal tickets that can transfer to diverse downstream datasets. First, we theoretically show that the image-level tasks may result in a degenerated feature map with high probability. This analysis implies that image-level tasks are not sufficient for neural network pruning. To address the problem, we introduce a new pruning framework that includes the image reconstruction task to guide the pruning process. Our framework has three steps: $i)$ Train an autoencoder, $ii)$ Prune the encoder with an improved LTH method. $iii)$ Transfer the encoder to downstream tasks. Besides, the feature map hint method is developed to accelerate the autoencoder training stage. The obtained tickets are evaluated on diverse downstream tasks at different sparsity ratios. The empirical study demonstrates that the proposed method can outperform state-of-the-art method [1, 7] on benchmark datasets.

## References

[1] Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Michael Carbin, and Zhangyang Wang. The lottery tickets hypothesis for supervised and self-supervised pre-training in computer vision models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 16306–16316, 2021.

[2] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *International conference on machine learning*, pages 1597–1607. PMLR, 2020.

[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 *IEEE conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009.

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[5] Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pages 1675–1685. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/du19c.html.

[6] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2018.

[7] Sharath Girish, Shishira R Maiya, Kamal Gupta, Hao Chen, Larry S Davis, and Abhinav Shrivastava. The lottery ticket hypothesis for object recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 762–771, 2021.


[8] Lovedeep Gondara. Medical image denoising using convolutional denoising autoencoders. In 2016 IEEE 16th international conference on data mining workshops (ICDMW), pages 241-246. IEEE, 2016.

[9] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733, 2020.

[10] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in Neural Information Processing Systems, 28, 2015.

[11] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961-2969, 2017.

[12] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9729-9738, 2020.

[13] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. arXiv preprint arXiv:2111.06377, 2021.

[14] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[15] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Quan- tized neural networks: Training neural networks with low precision weights and activations. The Journal of Machine Learning Research, 18(1):6869-6898, 2017.

[16] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[17] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[18] Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. In Advances in neural information processing systems, pages 598-605, 1990.

[19] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip Torr. Snip: Single-shot network pruning based on connection sensitivity. In International Conference on Learning Representations, 2018.

[20] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740-755. Springer, 2014.

[21] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. 2021.

[22] Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In International Conference on Learning Representations, 2018.

[23] Ari Morcos, Haonan Yu, Michela Paganini, and Yuandong Tian. One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. Advances in Neural Information Processing Systems, 32:4932-4942, 2019.

[24] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. 2011.

[25] Qi Qian, Yuanhong Xu, Juhua Hu, Hao Li, and Rong Jin. Unsupervised visual representation learning by online constrained k-means. CoRR, abs/2105.11527, 2021. URL https://arxiv. org/abs/2105.11527.

12

[26] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Marina Meila and Tong Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 8821–8831. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ramesh21a.html.

[27] Alex Renda, Jonathan Frankle, and Michael Carbin. Comparing rewinding and fine-tuning in neural network pruning. In *International Conference on Learning Representations*, 2019.

[28] Victor Sanh, Thomas Wolf, and Alexander M Rush. Movement pruning: Adaptive sparsity by fine-tuning. In *NeurIPS*, 2020.

[29] Cheng Tai, Tong Xiao, Yi Zhang, Xiaogang Wang, and E Weinan. Convolutional neural networks with low-rank regularization. In *4th International Conference on Learning Representations, ICLR 2016*, 2016.

[30] Hidenori Tanaka, Daniel Kunin, Daniel L Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. *Advances in Neural Information Processing Systems*, 33, 2020.

[31] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through attention. In Marina Meila and Tong Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 10347–10357. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/touvron21a.html.

[32] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th international conference on Machine learning*, pages 1096–1103, 2008.

[33] Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. In *International Conference on Learning Representations*, 2019.

[34] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/facebookresearch/detectron2, 2019.

[35] Yang Yang, Yaxiong Yuan, Avraam Chatzimichailidis, Ruud JG van Sloun, Lei Lei, and Symeon Chatzinotas. Proxsgd: Training structured neural networks under regularization and constraints. In *International Conference on Learning Representations*, 2019.

### A Proof of Theorem 3.1

We use Discrete Fourier Transformation to proof Theorem 3.1

Discrete Fourier Transformation (DFT) of a vector sequence $x \in \mathbb{R}^{c \times N}$ is defined as:

$$
\mathcal{F}(x_{i,:})(k) = \tilde{x}_i^k = \frac{1}{N} \sum_{s} x_{i,s} e^{\frac{2\pi J}{N} s k}
\tag{12}
$$

We use $J$ represent imaginary unit to distinguish from footnote $i$. Here we assume $x_{i,:}$ follow the Periodic Boundary Conditions. DFT has two important propositions:

**Proposition A.1.** Let $\tilde{x}_i^k = \mathcal{F}(x_{i,:})(k)$, then we have:
(1) $x_{i,s} = \sum_{k} \tilde{x}_i^k e^{\frac{2\pi J}{N} s k}$
(2) $[W * x]_{i,j} = \sum_{t,k} \tilde{W}(k)_{it} \tilde{x}_t^k e^{\frac{2\pi J}{N} j k}$

We define $[\tilde{W}(k)]_{i,j} = \sum_s W_{ij,s} e^{\frac{2\pi J}{N} s k}$ for simplicity. By those propositions, $\mathcal{C}(x)$ can be written as a simple version:

$$
\mathcal{C}(x)_{:,p} = \sum_{k} W^L(k) W^{L-1}(k)...W^0(k) \tilde{x}_{:}^k e^{\frac{2\pi J}{N} p k}
\tag{13}
$$

Because the avgpool operation only associate with the $k=0$ component, while other part is independent with it. It's easy to find another network such the zero component unchanged will other component are different. Thus we can proof the claim we stated in Section 3.2

**claim** Let $\mathcal{C}$ be a LCNN, there must exists another LCNN $\mathcal{C}'$, such that

$$
P_a \mathcal{C}(x) = P_a \mathcal{C}'(x), \mathcal{C}(x) \neq \mathcal{C}'(x)
\tag{14}
$$

To prove Theorem 3.1 we should notice that in such a LCNN $\mathcal{C}$, we can find some weights (around $\mathcal{O}(\frac{\epsilon}{\Delta})$) such that $|\sum_s W_{i,j,s}| < \epsilon$. By the assumption that each weights is initialized independent, the high frequency component of $W$ is in $\mathcal{O}(1)$ while the zero frequency component is $\mathcal{O}(\epsilon)$. Therefore, the total influence of $||P_a(\mathcal{C}(x))||^2$ is $\mathcal{O}(\frac{\epsilon^3}{\Delta^3})$ but the influence of $||\mathcal{C}(x)||^2$ is $\mathcal{O}(\frac{\epsilon}{\Delta})$

**Lemma A.2.** For a randomly initialized convolutional tensor, $W_{i,j,s} \sim i.i.d\mathcal{N}(0, \Delta)$, we have:
(1) $\mathbb{E}(\sum_i \sum_{k=-s}^s W_{i j,k} \sum_{n=-s}^s W_{i m,n}) = (2s+1)n \Delta^2 \delta_{j m}$
(2) $p = \mathbb{P}(\sum_{k=-s}^s W_{i j,k} < \epsilon) < \sqrt{\frac{1}{2\pi(2s+1)}} \frac{\epsilon}{\Delta}$

If we prune every neuron with $|\sum_s W_{i j,s}| < \epsilon$ and let the new tensor is $W_{i j,s}'$, we have $W_{i j,s}' = m_{i j} W_{i j,s}$
(3) $\mathbb{P}(m_{i j} = 0) = p$
(4) $\mathbb{E}(\sum_t m_{i j} W_{i j,t} \sum_k W_{i j,k}) > \Delta^2(1 - \frac{2\epsilon^3}{3\sqrt{2\pi} \Delta^3})$
(5) $\mathbb{E}(\sum_t m_{i j} W_{i j,t} \sum_k m_{i j} W_{i j,k}) = \mathbb{E}(\sum_t m_{i j} W_{i j,t}, \sum_k W_{i j,k})$
(6) $\mathbb{E}(\sum_t m_{i j} W_{i j,t} \sin t k \theta \sum_m W_{i j,m} \sin t k \theta) = (1-p) \sum_t \sin^2 t k \theta$
(7) $\mathbb{E}(\sum_t m_{i j} W_{i j,t} \cos t k \theta \sum_m W_{i j,m} \cos t k \theta) < (1-p) \sum_t \cos^2 t k \theta$
in (6)(7) $\theta = \frac{2\pi}{D}$, and $k \in \{1,2,...,D-1\}$

(1)(2)(3)(5) are easy to understand, for(4), we have:

$$
\mathbb{E}(\sum_t m_{i j} W_{i j,t}, \sum_k W_{i j,k})
\tag{15}
$$

$$
=\mathbb{E}(\sum_t W_{i j,t} \sum_k W_{i j,k}|m_{i j}=1)(1-p) + \mathbb{E}(m \sum_t W_{i j,t} \sum_k W_{i j,k}|m_{i j}=0)p
\tag{16}
$$

$$
=\Delta^2 - \int_{-\epsilon}^{\epsilon} \frac{1}{\sqrt{2\pi} \Delta} x^2 e^{-\frac{x^2}{2\pi \Delta}} dx > \Delta^2(1 - \frac{2\epsilon^3}{3\sqrt{2\pi} \Delta^3})
\tag{17}
$$

for (6) this result is a directly result by considering the conditional expectation:

$$
\mathbb{E}(\sum_t W_{i j,t} \sin t k \theta \sum_k W_{i j,k} \sin t k \theta ||\sum_t W_{i j,t}| > \epsilon)
\tag{18}
$$

14

Due to $\sum_{t} 1 \cdot \sin t k \theta=0$, this expectation is independent of conditional variable. Thus we have:

$$
\mathbb{E}\left(\sum_{t} m_{i j} W_{i j, t} \sin t k \theta \sum_{t} W_{i j, t} \sin t k \theta\right)
\tag{19}
$$

$$
=\mathbb{E}\left(\sum_{t} m_{i j} W_{i j, t} \sin t k \theta \sum_{k} W_{i j, k} \sin t k \theta \mid m_{i j}=1\right)(1-p)
\tag{20}
$$

$$
\quad+\mathbb{E}\left(\sum_{t} m_{i j} W_{i j, t} \sin t k \theta \sum_{t} W_{i j, t} \sin t k \theta \mid m_{i j}=0\right) p
\tag{21}
$$

$$
=(1-p) \sum_{t}(\sin t k \theta)^{2}
\tag{22}
$$

For (7), although the expectation term is dependent on the conditional term, we knew that this condition will enlarge the expectation, so we have:

$$
\mathbb{E}\left(\sum_{t} m_{i j} W_{i j, t} \cos t k \theta \sum_{t} W_{i j, t} \cos t k \theta\right)
\tag{23}
$$

$$
=\mathbb{E}\left(\sum_{t} m_{i j} W_{i j, t} \cos t k \theta \sum_{k} W_{i j, k} \cos t k \theta \mid m_{i j}=1\right)(1-p)
\tag{24}
$$

$$
\quad+\mathbb{E}\left(\sum_{t} m_{i j} W_{i j, t} \cos t k \theta \sum_{t} W_{i j, t} \sin t k \theta \mid m_{i j}=0\right) p
\tag{25}
$$

$$
<(1-p) \sum_{t}(\cos t k \theta)^{2}
\tag{26}
$$

Now, we can calculate the difference caused by prune the weights:

**Lemma A.3.** For any random initialized LCNN, where parameter is i.i.d initialized as $\mathcal{N}(0, \Delta)$. If we prune every neuron with $|\sum_{s} W_{i j, s}^{l}|<\epsilon$ and get a pruned LCNN $\mathcal{C}'$, then we have

$$
\frac{\left\|P_{a} \mathcal{C}(x)-P_{a} \mathcal{C}(x)\right\|^{2}}{\left\|P_{a} \mathcal{C}(x)\right\|^{2}} \sim \mathcal{O}\left(\frac{\epsilon^{3}}{\Delta^{3}}\right)
$$

**proof** According to (13), we can rewrite $P_{a} \mathcal{C}(x)$ as:

$$
P_{a} \mathcal{C}=W^{L}(0) W^{L-1}(0) \ldots W^{0}(0) P_{a}(x)
\tag{27}
$$

for simplicity, we write $F^{l}(k)=W^{l}(k) W^{l-1}(k) \ldots W^{0}(k) \mathcal{F}(x)(k)$. Thus,

$$
\left\|P_{a} \mathcal{C}(x)-P_{a} \mathcal{C}^{\prime}(x)\right\|^{2}=\left\|F^{L}(0)-F^{\prime L}(0)\right\|^{2}
\tag{28}
$$

We define $\mathbb{E}_{l}$ as the expectation about the $l$-th layer weights, then we have:

$$\mathbb{E}|| P_{a} \mathcal{C}(x)-P_{a} \mathcal{C}^{\prime}(x)||^{2}\qquad(29)$$

$$=\mathbb{E}\mathbb{E}_{L}|| F^{L}(0)-F^{\prime L}(0)||^{2}\qquad(30)$$

$$=\mathbb{E}\mathbb{E}_{L}\left\langle F^{L}(0), F^{L}(0)\right\rangle+\mathbb{E}\mathbb{E}_{L}\left\langle F^{\prime L}(0), F^{\prime L}(0)\right\rangle-2\mathbb{E}\mathbb{E}_{L}\left\langle F^{L}(0), F^{\prime L}(0)\right\rangle\qquad(31)$$

$$=\mathbb{E}\mathbb{E}_{L}\left\langle W^{L}(0) F^{L-1}(0), W^{L}(0) F^{L-1}(0)\right\rangle+\mathbb{E}\mathbb{E}_{L}\left\langle W^{\prime L}(0) F^{\prime L-1}(0), W^{\prime L}(0) F^{\prime L-1}(0)\right\rangle-\qquad(32)$$

$$2\mathbb{E}\mathbb{E}_{L}\left\langle W^{L}(0) F^{L-1}(0), W^{\prime L}(0) F^{\prime L-1}(0)\right\rangle\qquad(33)$$

$$<m_{L}(2 s+1) \Delta^{2}\left\langle F^{L-1}(0), F^{L-1}(0)\right\rangle+m_{L}(2 s+1) \Delta^{2}\left(1-\frac{2 \epsilon^{3}}{3 \sqrt{2 \pi} \Delta^{3}}\right)\left\langle F^{\prime L-1}(0), F^{\prime L-1}(0)\right\rangle-\qquad(34)$$

$$2 m_{L}(2 s+1) \Delta^{2}\left(1-\frac{2 \epsilon^{3}}{3 \sqrt{2 \pi} \Delta^{3}}\right)\left\langle F^{L-1}(0), F^{\prime L-1}(0)\right\rangle\qquad(35)$$

$$<m_{L} m_{L-1} \ldots m_{1}(2 s+1)^{L} \Delta^{2 L}\left[1-\left(1-\frac{2 \epsilon^{3}}{3 \sqrt{2 \pi} \Delta^{3}}\right)^{L}\right]\left\langle P_{a}(x), P_{a}(x)\right\rangle\qquad(36)$$

$$<m_{L} m_{L-1} \ldots m_{1}(2 s+1)^{L} \Delta^{2 L} L \frac{2 \epsilon^{3}}{3 \sqrt{2 \pi} \Delta^{3}}\left\langle P_{a}(x), P_{a}(x)\right\rangle\qquad(37)$$

$$=L \frac{2 \epsilon^{3}}{3 \sqrt{2 \pi} \Delta^{3}}|| P_{a} \mathcal{C}(x)||^{2}\qquad(38)$$

By using the concentretion inequality, we proof Lemma A.3

**Lemma A.4.** For any random initialized LCNN, where parameter is i.i.d initialized as $\mathcal{N}(0, \Delta)$. If we prune every neuron with $|\sum_{s} W_{i j, s}^{l}|<\epsilon$ and get a pruned LCNN $\mathcal{C}'$, then we have

$$\frac{|| \mathcal{C}(x)-\mathcal{C}(x)||^{2}}{|| \mathcal{C}(x)||^{2}} \sim \mathcal{O}\left(\frac{\epsilon}{\Delta}\right)$$

**proof** First we have

$$|| \mathcal{C}(x)-\mathcal{C}(x)||^{2}=\sum_{k}|| \mathcal{F}(\mathcal{C}(x))(k)-\mathcal{F}\left(\mathcal{C}^{\prime}(x)\right)(k)||^{2}$$

We caculate $k \neq 0$ term, due to $k=0$ is calculated in A.3 and it is $\mathcal{O}(\frac{\epsilon}{\Delta})$

$$\sum_{k \neq 0} \mathbb{E}|| \mathcal{F}(\mathcal{C}(x))(k)-\mathcal{F}\left(\mathcal{C}^{\prime}(x)\right)(k)||^{2}\qquad(39)$$

$$=\sum_{k \neq 0} \mathbb{E} \mathbb{E}_{L}|| F^{L}(k)-F^{\prime L}(k)||^{2}\qquad(40)$$

$$=\sum_{k \neq 0} \mathbb{E} \mathbb{E}_{L}\left\langle W^{L}(k) F^{L-1}(k), W^{L}(k) F^{L-1}(k)\right\rangle-\mathbb{E} \mathbb{E}_{L}\left\langle W^{L}(k) F^{L-1}(k), W^{\prime L}(k) F^{\prime L-1}(0)\right\rangle\qquad(41)$$

$$>\sum_{k \neq 0} \mathbb{E}(2 s+1) m_{L} \Delta^{2} \mathbb{E}\left\langle F^{L-1}(k), F^{L-1}(k)\right\rangle-\qquad(42)$$

$$\mathbb{E} m_{L} \Delta^{2} \mathbb{E}\left\langle F^{L-1}(k), F^{L-1}(k)\right\rangle(1-p)\left(\sum_{t} \cos ^{2} t k \theta+\sum_{t} \sin ^{2} t k \theta\right)\qquad(43)$$

$$=\sum_{k \neq 0} m_{L} m_{L-1} \ldots m_{1}(2 s+1)^{L} \Delta^{2 L} \mathbb{E}\langle\mathcal{F}(x)(k), \mathcal{F}(x)(k)\rangle\left(1-(1-p)^{L}\right)\qquad(44)$$

$$=\left(1-(1-p)^{L}\right) \sum_{k \neq 0} \mathbb{E}|| \mathcal{F}(\mathcal{C}(x))(k)||^{2}\qquad(45)$$


Thus we have:

$$
\sum_{k} \mathbb{E}||\mathcal{F}(\mathcal{C}(x))(k)-\mathcal{F}\left(\mathcal{C}^{\prime}(x)\right)(k)||^{2}
\tag{46}
$$

$$
=(1-(1-p)^{L}) \sum_{k} \mathbb{E}||\mathcal{F}(\mathcal{C}(x))(k)||^{2}=\mathcal{O}\left(\frac{\epsilon}{\Delta}\right)||\mathcal{C}(x)||^{2}
\tag{47}
$$

By combining Lemma A.3 and Lemma A.4, we proof Theorem 3.1

## B Proof of Theorem 3.2

In order to prove Theorem 3.2, we need more notations to represent the problem setup.

First, we introduce the convolution expansion operator $\phi.(\cdot)$ that can simplify convolution operator to inner product of tensors.

**Convolution Tensor** In convolutional neural network(CNN), the convolution tensor is described as follows. Let $\mathbf{w}_{i j}=\left(w_{i j,-s}, w_{i j,-s+1}, \ldots, w_{i j, s}\right)^{\mathbf{T}} \in \mathbb{R}^{2 s+1}(1 \leq i \leq c', 1 \leq j \leq c)$ be convolution kernel, where $c'$ is the number of filters and $2s+1$ is the size of convolution kernel. And $\mathbf{W}_{i}=\left(\mathbf{w}_{i 1}, \mathbf{w}_{i 2}, \ldots, \mathbf{w}_{i c}\right)^{\mathbf{T}} \in \mathbb{R}^{c \times(2 s+1)}$ denotes $i_{t h}$ filter. Then the convolution tensor is defined as a 3-dimensional tensor $\mathbf{W}=\left(\mathbf{W}_{i}\right)_{1 \leq i \leq c'} \in \mathbb{R}^{c' \times c \times(2 s+1)}$.

**Convolution Expansion Operator** Let $\mathbf{x}=\left(x_{i, j}\right) \in \mathbb{R}^{c \times D}$ be the 1-dimensional input, where $D$ is the length of input sequence and $\mathrm{c}$ is the channel number of input. Let $\mathbf{W} \in \mathbb{R}^{c' \times c \times(2 s+1)}$ be the convolution tensor. Then we define $\phi_{k}(\mathbf{x})$ as

$$
\phi_{k}(\mathbf{x})=\left(\begin{array}{ccc}
x_{1, k-s}, & \ldots & , x_{1, k+s} \\
\ldots, & \ldots, & \ldots \\
x_{c, k-s}, & \ldots, & x_{c, k+s}
\end{array}\right)
\tag{48}
$$

Recall the definition of convolution operator, we have

$$
(\mathbf{W} * \mathbf{x})_{r, k}=\left\langle\mathbf{W}_{r}, \phi_{k}(\mathbf{x})\right\rangle
\tag{49}
$$

Because we use structured pruning method to prune model, we can assume the new channel number of feature maps is $M=m q=m(1-p)$, where $p$ is our pruning rate. And we assume $m q$ is an integer in order to simplify the proof. Now, we can represent the pruned ORCNN as

$$
f(\mathbf{x})=\frac{\sqrt{q}}{\sqrt{M} D} \sum_{r=1}^{M} a_{r} \sum_{k=1}^{D} \sigma\left(\left\langle\mathbf{W}_{r}, \phi_{k}(\mathbf{x})\right\rangle\right)
\tag{50}
$$

To analyze the gradient descent process, inspired by [5], we focus on the Gram matrix $\mathbf{G}=\left(\mathbf{G}_{i, j}\right)$ of ORCNN, which is defined as

$$
\mathbf{G}_{i, j}=\frac{q}{M D^{2}} \sum_{r=1}^{M} \sum_{k=1}^{D} \sum_{l=1}^{D}\left\langle\phi_{k}\left(\mathbf{x}_{i}\right), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0,\left\langle\mathbf{W}_{r}, \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \geq 0\right\}
\tag{51}
$$

And if we consider the case that $M$ goes to infinity, we can represent the expectation of $\mathbf{G}$ and we know $\mathbf{G}$ converges to it by central limit theorem and independence of $\mathbf{W}_{r}(1 \leq r \leq M)$. We use $\mathbf{G}^{\infty}$ to denote the expectation, which can be formally defined as

$$
\mathbf{G}_{i, j}^{\infty}=\mathbb{E}_{\mathbf{W} \sim \mathrm{N}(\mathbf{0}, \mathbf{I})}\left[\frac{q}{D^{2}} \sum_{k=1}^{D} \sum_{l=1}^{D}\left\langle\phi_{k}\left(\mathbf{x}_{i}\right), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \mathbb{I}\left\{\left\langle\mathbf{W}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0,\left\langle\mathbf{W}, \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \geq 0\right\}\right]
\tag{52}
$$

**Proposition B.1.** We use $\lambda_{min}(\cdot)$ to denote the least eigenvalue of a matrix, then we assume that $\mathbf{G}^{\infty}$ has the positive least eigenvalue $\lambda_{0}=\lambda_{min}(\mathbf{G}^{\infty})$. In fact, we know $\mathbf{G}^{\infty}$ is positive definite. Thus, the assumption is weak and reasonable.

We use $\mathbf{G}_{0}$ to denote the initial Gram matrix of pruned model. By the standard concentration analysis of independent variables, we can derive the following Lemma B.2, which means the initial Gram matrix of model also has the bounded positive least eigenvalue with high probability.

Lemma B.2. When the channel number of feature maps in pruned model $M = \Omega\left(\frac{n^{2}}{\lambda_{0}^{2}} \log \left(\frac{n}{\delta}\right)\right)$ and initialization in pre-trained phase satisfies the standard Guassian distribution, with probability at least $1-\delta$, we have

$$
\lambda_{min}(\mathbf{G}_{0}) \geq \frac{3}{4} \lambda_{0} \tag{53}
$$

Before analyzing the finetuning process, we assume some settings as follows to simplify the process in order to focus on the change of convolution tensor.

Proposition B.3. In finetuning phase, we randomly initialize the fully connected weight $\mathbf{a} \sim Unif(\{-1,1\}^{M})$. And we normalize the input such that $\left\|\phi_{k}(\mathbf{x}_{i})\right\|=1$ for any $k,i$. In fact, without the normalization assumption we can also proof the similar conclusion, but it will dependent with $\frac{\max_{k,i}\{\left\|\phi_{k}(\mathbf{x}_{i})\right\|\}}{\min_{k,i}\{\left\|\phi_{k}(\mathbf{x}_{i})\right\|\}}$, which is also an constant.

Because the fully connected weight $\mathbf{a}$ is fixed, we can represent the model output w.r.t. the given dataset $S$ as

$$
F_{i}(\mathbf{W}(t))=\frac{\sqrt{q}}{\sqrt{M} D} \sum_{r=1}^{M} a_{r} \sum_{k=1}^{D} \sigma(\left\langle\mathbf{W}_{r}(t), \phi_{k}(\mathbf{x}_{i})\right\rangle) \tag{54}
$$

which is output of $i^{th}$ sample in dataset. And we use $\mathbf{F}(\mathbf{W}(t))=(F_{1}(\mathbf{W}(t)), F_{2}(\mathbf{W}(t)), \cdots, F_{n}(\mathbf{W}(t)))^{\mathbf{T}}$ to denote the output vector, where $t$ means it is $t^{th}$-round.

Proposition B.4. Recall the initialization of the pruned convolution tensor is the same as that in pre-trained phase, we know each unpruned entry of the pruned convolution tensor satisfies independent standard Guassian distribution $\mathbf{N}(0,1)$. With the knowledge of sub-exponential distribution, we can prove that w.h.p. the $l_{2}$ norm of the pruned convolution tensor$\mathbf{W}(0)$ is close to $\sqrt{q m D(2 s+1)}$ and the $l_{2}$ norm of the original convolution tensor before pre-trained phase is close to $\sqrt{m D(2 s+1)}$.

Next, we introduce Lemma B.5 that describes the dynamic process in finetuning phase.

Lemma B.5. Under the above assumptions, if the channel number of feature maps in pruned model $M$ is $\Omega\left(\frac{n^{6}}{\lambda_{0}^{4} \delta^{3}}\right)$ and learning rate $\eta$ is $O\left(\frac{\lambda_{0}}{n^{2}}\right)$, with probability at least $1-\delta$, we have

$$
\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2} \leq\left(1-\frac{\eta \lambda_{0}}{2}\right)^{t}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|^{2} \tag{55}
$$

where $\mathbf{y}=\left(y_{1}, y_{2}, \cdots, y_{n}\right)^{\mathbf{T}}$ is the label vector of dataset $S$.

By Lemma B.5, with the standard analysis of gradient descent, we can prove the following Lemma B.6

Lemma B.6. Under the above assumptions, if the channel number of feature maps in pruned model $M$ is $\Omega\left(\frac{n^{6}}{\lambda_{0}^{4} \delta^{3}}\right)$ and learning rate $\eta$ is $O\left(\frac{\lambda_{0}}{n^{2}}\right)$, with probability at least $1-\delta$, for some constant $C$, we have

$$
\left\|\mathbf{W}_{f i n}-\mathbf{W}(0)\right\| \leq \frac{C \sqrt{q} n}{\lambda_{0}} \tag{56}
$$

Remark of Lemmma B.6 In fact, we have similar conclusion of Lemmma B.6 in pre-trained phase, which implies $\left\|\mathbf{W}_{p r e}\right\|=\sqrt{m}(1+o(1))$ and can be proven by the same method as follows.

Proof of Theorem 3.2 By $m=\Omega(\frac{n^{6}}{\lambda_{0}^{4}})$, Proposition B.4 and Lemma B.6, with probability $1-\delta$, we have $\left\|\mathbf{W}_{f i n}-\mathbf{W}(0)\right\|=o(\|\mathbf{W}(0)\|)$ and $\left\|\mathbf{W}_{f i n}\right\|=\sqrt{q m}(1+o(1))$. Combined with the property of rotation operator, for any rotation operator $\mathbf{Q}$, we have $d i s t_{l_{2}}^{N}(\mathbf{Q} \mathbf{W}_{f i n}, \mathbf{W}_{p r e}) \geq \frac{\left\|\mathbf{W}_{p r e}\right\|-\left\|\mathbf{Q} \mathbf{W}_{f i n}\right\|}{\sqrt{\left\|\mathbf{W}_{p r e}\right\|\left\|\mathbf{Q} \mathbf{W}_{f i n}\right\|}}=(1-p)^{-\frac{1}{4}}-(1-p)^{\frac{1}{4}} \geq \frac{p}{2}$. Thus, we prove Theorem 3.2 .

Next, We will use induction method to prove Lemma B.5 and Lemma B.6 . Our inductive hypothesis is the inequality is true for $0,1, \cdots, t$ and we want to prove the inequality is also true for $t+1$. First,

we have the following decomposition of $l_2$ loss.

$$
\begin{aligned}
&||\mathbf{F}(\mathbf{W}(t+1))-\mathbf{y}||^{2} \\
&=||\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))+\mathbf{F}(\mathbf{W}(t))-\mathbf{y}||^{2} \\
&=||\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))||^{2} \\
&\quad+2\left(\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\right)^{\mathbf{T}}\left(\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))\right)+||\mathbf{F}(\mathbf{W}(t))-\mathbf{y}||^{2}
\end{aligned}
\tag{57}
$$

The third term can be bounded by the inductive hypothesis. Thus, we only need to bound the first and the second terms.

The following Lemma B.7 can help us to bound the gradient norm in finetuing phase.

Lemma B.7. In finetuing phase, we can control the upper bound of gradient norm as

$$
\left\|\frac{\partial L(\mathbf{W})}{\partial \mathbf{W}_{r}}\right\| \leq \frac{\sqrt{q n}}{\sqrt{M}}||\mathbf{F}(\mathbf{W})-\mathbf{y}||
\tag{58}
$$

for any $r \in [M]$.

Proof of Lemma B.7 The proof is very standard to analysis gradient descent.

$$
\begin{aligned}
\left\|\frac{\partial L(\mathbf{W})}{\partial \mathbf{W}_{r}}\right\| &=\left\|\frac{\sqrt{q}}{\sqrt{M} D} \sum_{i=1}^{n}\left(F_{i}(\mathbf{W})-y_{i}\right) \sum_{k=1}^{D} \phi_{k}\left(\mathbf{x}_{i}\right) \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}\right\| \\
& \leq \frac{\sqrt{q}}{\sqrt{M} D}|| \mathbf{F}(\mathbf{W})-\mathbf{y}||\left\|\left(\sum_{k=1}^{D} \phi_{k}\left(\mathbf{x}_{i}\right) \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}\right)_{1 \leq i \leq n}\right\| \\
&=\frac{\sqrt{q}}{\sqrt{M} D}|| \mathbf{F}(\mathbf{W})-\mathbf{y}|| \sqrt{\sum_{i=1}^{n}|| \sum_{k=1}^{D} \phi_{k}\left(\mathbf{x}_{i}\right) \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}||^{2}} \\
& \leq \frac{\sqrt{q}}{\sqrt{M} D}|| \mathbf{F}(\mathbf{W})-\mathbf{y}|| \sqrt{\sum_{i=1}^{n}\left(\sum_{k=1}^{D}|| \phi_{k}\left(\mathbf{x}_{i}\right) \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}||\right)^{2}} \\
& \leq \frac{\sqrt{q}}{\sqrt{M} D}|| \mathbf{F}(\mathbf{W})-\mathbf{y}|| \sqrt{n D^{2}} \\
&=\frac{\sqrt{q n}}{\sqrt{M}}|| \mathbf{F}(\mathbf{W})-\mathbf{y}||
\end{aligned}
\tag{59}
$$


Now, we can derive an upper bound of the first term by Lemma B.7 and 1-Lipschitz property of ReLU funciton.

$$
\begin{aligned}
& \|\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))\|^{2} \\
& =\sum_{i=1}^{n}\left|F_{i}(\mathbf{W}(t+1))-F_{i}(\mathbf{W}(t))\right|^{2} \\
& =\sum_{i=1}^{n}\left|\frac{\sqrt{q}}{\sqrt{M} D} \sum_{r=1}^{M} a_{r} \sum_{k=1}^{D}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)\right|^{2} \\
& =\sum_{i=1}^{n}\left|\frac{\sqrt{q}}{\sqrt{M} D} \sum_{r=1}^{M} a_{r} \sum_{k=1}^{D}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t)-\eta \frac{\partial L(\mathbf{W}(\mathbf{t}))}{\partial \mathbf{W}(\mathbf{t})_{r}}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)\right|^{2} \\
& \leq \frac{q}{D} \sum_{i=1}^{n} \sum_{r=1}^{M} \sum_{k=1}^{D}\left|\left\langle\eta \frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right|^{2} \\
& \leq \frac{q \eta^{2}}{D} \sum_{i=1}^{n} \sum_{r=1}^{M} \sum_{k=1}^{D}\left\|\frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}\right\|^{2} \\
& \leq \frac{q \eta^{2}}{D} n M D \frac{q n}{M}\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2} \\
& =q^{2} \eta^{2} n^{2}\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2}
\end{aligned}
$$

(60)

To bound the second term, we need a more refined analysis of its form. The following Lemma B.8 will inspire us how to deal with the term.

**Lemma B.8.** Under the inductive hypothesis, for any $T \in[t+1]$ and $rin[M]$, we have

$$
\left\|\mathbf{W}_{r}(T)-\mathbf{W}_{r}(0)\right\| \leq \frac{4 \sqrt{q n}}{\sqrt{M} \lambda_{0}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\| \tag{61}
$$

**Proof of Lemma B.8** By the gradient descent and inductive hypothesis, we have

$$
\begin{aligned}
\left\|\mathbf{W}_{r}(T)-\mathbf{W}_{r}(0)\right\| & \leq \sum_{j=0}^{T-1}\left\|\mathbf{W}_{r}(j+1)-\mathbf{W}_{r}(j)\right\| \\
& =\eta \sum_{j=0}^{T-1}\left\|\frac{\partial L(\mathbf{W}(j))}{\partial \mathbf{W}_{r}(j)}\right\| \\
& \leq \eta \sum_{j=0}^{T-1} \frac{\sqrt{q n}}{\sqrt{M}}\|\mathbf{F}(\mathbf{W}(j))-\mathbf{y}\| \\
& \leq \eta \sum_{j=0}^{T-1} \frac{\sqrt{q n}}{\sqrt{M}}\left(1-\frac{\eta \lambda_{0}}{2}\right)^{\frac{j}{2}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\| \\
& \leq \eta \frac{\sqrt{q n}}{\sqrt{M}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\| \sum_{j=0}^{\infty}\left(1-\frac{\eta \lambda_{0}}{2}\right)^{\frac{j}{2}} \\
& \leq \eta \frac{\sqrt{q n}}{\sqrt{M}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\| \sum_{j=0}^{\infty}\left(1-\frac{\eta \lambda_{0}}{4}\right)^{j} \\
& \leq \frac{4 \sqrt{q n}}{\sqrt{M} \lambda_{0}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|
\end{aligned} \tag{62}
$$

20

Notice we have

$$
\begin{aligned}
F_{i}(\mathbf{W}(t+1))-F_{i}(\mathbf{W}(t))=\frac{\sqrt{q}}{\sqrt{M} D} \sum_{r=1}^{M} a_{r} \sum_{k=1}^{D}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)
\end{aligned}
\tag{63}
$$

And Lemma B.8 implies the distance between $\mathbf{W}_{r}(T)$ and $\mathbf{W}_{r}(0)$ is bounded, we can use truncation estimation method to deal with $F_{i}(\mathbf{W}(t+1))-F_{i}(\mathbf{W}(t))$. Specifically, we use $U_{r, k, i}(R)$ to denote the event $\left\langle\mathbf{W}_{r}(0), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle<R$.

Because $\phi_{k}(\mathbf{x}_{i})$ is normalized, we know $\left\langle\mathbf{W}_{r}(0), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle$ is a standard Guassian Variable. We have the following Lemma B.9 .

**Lemma B.9.** For any $r$ in$[M], k \in[D]$ and $i \in[n]$, we have the probability of the $U_{r, k, i}(R)$ satisfies

$$
\mathbb{P}\left\{U_{r, k, i}(R)\right\} \leq \frac{2 R}{\sqrt{2 \pi}}
\tag{64}
$$

If the event $U_{r, k, i}(R)$ is false and $\frac{4 \sqrt{q n}}{\sqrt{M} \lambda_{0}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|<R$, then we know $\mathbb{I}\left\{\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}=\mathbb{I}\left\{\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}=\mathbb{I}\left\{\left\langle\mathbf{W}_{r}(0), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\}$ by Lemma B.8 . So we use $J_{true }(R)$ to denote the set $\left\{(r, k, i) | U_{r, k, i}(R) \text { is true }\right\}$ and $J_{false }(R)$ to denote the set $\left\{(r, k, i) | U_{r, k, i}(R) \text { is false }\right\}$.

By Markov's inequality, we can derive the following Lemma B.10

**Lemma B.10.** With probability at least $1-\delta$, we have

$$
\left|J_{true}(R)\right| \leq \frac{n M D R}{\sqrt{2 \pi} \delta}
\tag{65}
$$

If we use $\mathbf{e}_{i}(1 \leq i \leq n)$ to denote the standard basis of $\mathbb{R}^{n}$, then we can decompose $\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))$ as

$$
\begin{aligned}
& \mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t)) \\
& =\sum_{i=1}^{n} \mathbf{e}_{i} \frac{\sqrt{q}}{\sqrt{M} D} \sum_{r=1}^{M} a_{r} \sum_{k=1}^{D}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right) \\
& =\frac{\sqrt{q}}{\sqrt{M} D} \sum_{(r, k, i) \in J_{true}(R)} a_{r} \mathbf{e}_{i}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right) \\
& \quad+\frac{\sqrt{q}}{\sqrt{M} D} \sum_{(r, k, i) \in J_{false}(R)} a_{r} \mathbf{e}_{i}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)
\end{aligned}
\tag{66}
$$


We use $I_1$, $I_2$ to denote two terms indexed by $J_{true}(R)$, $J_{false}(R)$. And the term indexed by $J_{true}(R)$ can be bounded by the cardinal of $J_{true}(R)$ as

$$
\begin{aligned}
|| I_{1}|| &=\left\|\frac{\sqrt{q}}{\sqrt{M} D} \sum_{(r, k, i) \in J_{\text {true }}(R)} a_{r} \mathbf{e}_{i}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)\right\| \\
&=\frac{\sqrt{q}}{\sqrt{M} D} \sqrt{\sum_{i=1}^{n}\left|\sum_{r, k |(r, k, i) \in J_{\text {true }}(R)} a_{r}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)\right|^{2}} \\
&=\frac{\sqrt{q}}{\sqrt{M} D} \sqrt{\sum_{i=1}^{n}\left|\sum_{r, k |(r, k, i) \in J_{\text {true }}(R)} a_{r}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t)-\eta \frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right)\right|^{2}} \\
& \leq \frac{\sqrt{q}}{\sqrt{M} D} \sqrt{\sum_{i=1}^{n}\left(\sum_{r, k |(r, k, i) \in J_{\text {true }}(R)} \eta\left\|\frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}\right\|\right)^{2}} \\
& \leq \frac{\sqrt{q}\left|J_{\text {true }}(R)\right|}{\sqrt{M} D} \eta\left\|\frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}\right\| \\
& \leq \frac{\sqrt{q}\left|J_{\text {true }}(R)\right|}{\sqrt{M} D} \eta \frac{\sqrt{q n}}{\sqrt{M}}|| \mathbf{F}(\mathbf{W}(t))-\mathbf{y}|| \\
&=\frac{\eta q \sqrt{n}\left|J_{\text {true }}(R)\right|}{M D}|| \mathbf{F}(\mathbf{W}(t))-\mathbf{y}||
\end{aligned}
$$

(67)

Next, we focus on $I_2$ and establish relation between it and the Gram matrix.

$$
\begin{aligned}
I_{2} &=\frac{\sqrt{q}}{\sqrt{M} D} \sum_{(r, k, i) \in J_{\text {false }}(R)} a_{r} \mathbf{e}_{i}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t+1), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right) \\
&=\frac{\sqrt{q}}{\sqrt{M} D} \sum_{(r, k, i) \in J_{\text {false }}(R)} a_{r} \mathbf{e}_{i}\left(\sigma\left(\left\langle\mathbf{W}_{r}(t)-\eta \frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)-\sigma\left(\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle\right)\right) \\
&=\frac{\sqrt{q}}{\sqrt{M} D} \sum_{(r, k, i) \in J_{\text {false }}(R)} a_{r} \mathbf{e}_{i}\left(-\eta \frac{\partial L(\mathbf{W}(t))}{\partial \mathbf{W}_{r}(t)}\right) \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0\right\} \\
&=-\eta \frac{q}{M D^{2}} \sum_{(r, k, i) \in J_{\text {false }}(R)} \sum_{j=1}^{n} \sum_{l=1}^{D} \mathbf{e}_{i}\left(F_{j}(\mathbf{W}(t))-y_{j}\right)\left\langle\phi_{k}\left(\mathbf{x}_{i}\right), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \mathbb{I}\left\{\left\langle\mathbf{W}_{r}, \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0,\left\langle\mathbf{W}_{r}, \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \geq 0\right\} \\
&=-\eta \sum_{i=1}^{n} \mathbf{e}_{i} \sum_{j=1}^{n} \tilde{\mathbf{G}}_{i, j}(t)\left(F_{j}(\mathbf{W}(t))-y_{j}\right) \\
&=-\eta \tilde{\mathbf{G}}(t)(\mathbf{F}(\mathbf{W}(t))-\mathbf{y})
\end{aligned}
$$

(68)

where $\tilde{\mathbf{G}}(t)$ is defined as

$$
\tilde{\mathbf{G}}_{i, j}(t)=\frac{q}{M D^{2}} \sum_{r, k, l |(r, k, i) \in J_{\text {false }}(R)}\left\langle\phi_{k}\left(\mathbf{x}_{i}\right), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \mathbb{I}\left\{\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0,\left\langle\mathbf{W}_{r}(t), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \geq 0\right\}
$$

(69)

By matrix perturbation technique, we can prove the following Lemma B.11

Lemma B.11. If the channel number $M$ is $\Omega\left(\frac{q^{3} n^{6}}{\delta^{2} \lambda_{0}{ }^{4}}\right)$, then with probability at least $1-\delta$, we have

$$
\lambda_{\min }(\mathbf{G}(t)) \geq \frac{\lambda_{0}}{2}
$$

(70)

Proof of Lemma B.11 First, by $M = \Omega\left(\frac{q^{3} n^{6}}{\delta^{2} \lambda_{0}^{4}}\right)$ and Lemma B.8, if we set $R'$ to $\frac{4 \sqrt{q n}}{\sqrt{M} \lambda_{0}}||\mathbf{F}(\mathbf{W}(0))-\mathbf{y}||$, then we have

$$||\mathbf{W}_{r}(t)-\mathbf{t}_{r}(0)|| \leq R' \tag{71}$$

for any $r \in [M]$.

Next, we establish relation between $\mathbf{G}(t)$ and $\mathbf{G}(0)$. We consider the difference of each entry of them as

$$
\begin{aligned}
& \mathbb{E}\left|\mathbf{G}_{i, j}(t)-\mathbf{G}_{i, j}(0)\right| \\
&= \frac{q}{M D^{2}} \sum_{r=1}^{M} \sum_{k=1}^{D} \sum_{l=1}^{D} |\left\langle\phi_{k}\left(\mathbf{x}_{i}\right), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle\left(\mathbb{I}\left\{\left\langle\mathbf{W}_{r}(t), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0,\left\langle\mathbf{W}_{r}(t), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \geq 0\right\}\right. \\
&\left.-\mathbb{I}\left\{\left\langle\mathbf{W}_{r}(0), \phi_{k}\left(\mathbf{x}_{i}\right)\right\rangle \geq 0,\left\langle\mathbf{W}_{r}(0), \phi_{l}\left(\mathbf{x}_{j}\right)\right\rangle \geq 0\right\}\right)| \\
& \leq \frac{q}{M D^{2}} \sum_{r=1}^{M} \sum_{k=1}^{D} \sum_{l=1}^{D} \mathbb{P}\left\{U_{r, k, i}\left(R^{\prime}\right) \cup U_{r, L, i}\left(R^{\prime}\right)\right\} \\
& \leq \frac{4 q R^{\prime}}{\sqrt{2 \pi}}
\end{aligned} \tag{72}
$$

By the Markov's inequality, with probability at least $1-\delta$, we have $||\mathbf{G}(t)-\mathbf{G}(0)||_{l_{1}} \leq \frac{4 q n^{2} R^{\prime}}{\sqrt{2 \pi} \delta}$.
And by the positive definite of $\mathbf{G}(t), \mathbf{G}(0)$ we know

$$\lambda_{\max }(\mathbf{G}(t)-\mathbf{G}(0)) \leq||\mathbf{G}(t)-\mathbf{G}(0)||_{F} \leq||\mathbf{G}(t)-\mathbf{G}(0)||_{l_{1}} \leq \frac{4 q n^{2} R^{\prime}}{\sqrt{2 \pi} \delta} \tag{73}$$

Thus, we have

$$
\begin{aligned}
\lambda_{\min }(\mathbf{G}(t)) & \geq \lambda_{\min }(\mathbf{G}(0))-\lambda_{\max }(\mathbf{G}(t)-\mathbf{G}(0)) \\
& \geq \frac{3}{4} \lambda_{0}-\frac{4 q n^{2} R^{\prime}}{\sqrt{2 \pi} \delta} \geq \frac{\lambda_{0}}{2}
\end{aligned} \tag{74}
$$

Under conclusion of Lemma B.11, by the similar technique used by proof of Lemma B.11 and
Lemma B.10, we can prove

Lemma B.12. If the channel number $M$ is $\Omega\left(\frac{q^{3} n^{6}}{\delta^{2} \lambda_{0}^{4}}\right)$, then with probability at least $1-\delta$, we have

$$\lambda_{\min }(\tilde{\mathbf{G}}(t)) \geq \frac{\lambda_{0}}{2}-\frac{q n^{2} R}{\sqrt{2 \pi} \delta} \tag{75}$$

Thus, by Lemma B.12, we have

$$
\begin{aligned}
(\mathbf{F}(\mathbf{W}(t))-\mathbf{y})^{\mathbf{T}} I_{2} &=-\eta\left(\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\right)^{\mathbf{T}} \tilde{\mathbf{G}}(t)(\mathbf{F}(\mathbf{W}(t))-\mathbf{y}) \\
& \leq\left(-\frac{\eta \lambda_{0}}{2}+\frac{\eta q n^{2} R}{\sqrt{2 \pi} \delta}\right)||\mathbf{F}(\mathbf{W}(t))-\mathbf{y}||^{2}
\end{aligned} \tag{76}
$$

With the upper bound of the cardinal of $J_{true}(R)$(Lemma B.10), we also have

$$
\begin{aligned}
(\mathbf{F}(\mathbf{W}(t))-\mathbf{y})^{\mathbf{T}} I_{1} & \leq||\mathbf{F}(\mathbf{W}(t))-\mathbf{y}|||| I_{1}|| \\
& \leq||\mathbf{F}(\mathbf{W}(t))-\mathbf{y}|| \frac{\eta q \sqrt{n}\left|J_{\text {true }}(R)\right|}{M D}|| \mathbf{F}(\mathbf{W}(t))-\mathbf{y}|| \\
& \leq \frac{q \eta n^{\frac{3}{2}} R}{\sqrt{2 \pi} \delta}||\mathbf{F}(\mathbf{W}(t))-\mathbf{y}||^{2}
\end{aligned} \tag{77}
$$


Based on the estimation of upper bound of three terms, we have

$$
\begin{aligned}
\|\mathbf{F}(\mathbf{W}(t+1))-\mathbf{y}\|^{2} &=\|\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))\|^{2} \\
& \quad+2\left(\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\right)^{\mathbf{T}}\left(\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))\right)+\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2} \\
&=\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2}+\|\mathbf{F}(\mathbf{W}(t+1))-\mathbf{F}(\mathbf{W}(t))\|^{2} \\
& \quad+2\left(\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\right)^{\mathbf{T}} I_{1}+2\left(\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\right)^{\mathbf{T}} I_{2} \\
& \leq\left(1+q^{2} \eta^{2} n^{2}-\eta \lambda_{0}+\frac{2 \eta q n^{2} R}{\sqrt{2 \pi} \delta}+\frac{2 q \eta n^{\frac{3}{2}} R}{\sqrt{2 \pi} \delta}\right)\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2} \\
& \leq\left(1-\frac{\eta \lambda_{0}}{2}\right)\|\mathbf{F}(\mathbf{W}(t))-\mathbf{y}\|^{2}
\end{aligned}
\tag{78}
$$

Finally, we only need to select the value of $R$. To make all the lemmas are true, we can set $R$ to $\frac{\sqrt{2 \pi} \delta \lambda_{0}}{16 q n^{2}}$. Now, by combining the inductive hypothesis, we prove Lemma B.5.

Next, we prove Lemma B.6 by the expansion of Lemma B.8. In fact, Lemma B.5 is true for any $t$, so Lemma B.8 is true for any $T$. We use $\mathbf{W}_{f i n, i}(1 \leq i \leq m)$ to denote the finetuned convolution filters and have $\mathbf{W}_{f i n}=\left(\mathbf{W}_{f i n, i}\right)_{1 \leq i \leq m}$. Notice some filters have been pruned, by Lemma B.8, then we have

$$
\begin{aligned}
\left\|\mathbf{W}_{f i n}-\mathbf{W}(0)\right\|^{2} &=\sum_{r=1}^{M}\left\|\mathbf{W}_{f i n, r}-\mathbf{W}_{r}(0)\right\|^{2} \\
& \leq \sum_{r=1}^{M}\left(\frac{4 \sqrt{q n}}{\sqrt{M} \lambda_{0}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|\right)^{2} \\
&=\frac{16 q n}{\lambda_{0}^{2}}\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|^{2}
\end{aligned}
\tag{79}
$$

By the concentration of random initialization, with high probability, $\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|$ is $O(\sqrt{n})$. So there exists an constant $C$ such that $\|\mathbf{F}(\mathbf{W}(0))-\mathbf{y}\|^{2} \leq C n$, which means we prove Lemma B.6 .