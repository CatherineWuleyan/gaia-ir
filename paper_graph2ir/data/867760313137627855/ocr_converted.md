# The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models

［#1］
Tianlong Chen¹, Jonathan Frankle², Shiyu Chang³, Sijia Liu³⁴, Yang Zhang³,
Michael Carbin², Zhangyang Wang¹

［#2］
¹University of Texas at Austin, ²MIT CSAIL ³MIT-IBM Watson AI Lab, ⁴Michigan State University
{tianlong.chen,atlaswang}@utexas.edu,{jfrankle,mcarbin}@csail.mit.edu,
{shiyu.chang,yang.zhang2}@ibm.com,liusiji5@msu.ed

## Abstract
［#3］
The computer vision world has been re-gaining en-
thusiasm in various pre-trained models, including both
classical ImageNet supervised pre-training and recently
emerged self-supervised pre-training such as simCLR [10]
and MoCo [40]. Pre-trained weights often boost a wide
range of downstream tasks including classification, detection,
and segmentation. Latest studies suggest that pre-training
benefits from gigantic model capacity [11]. We are hereby
curious and ask: after pre-training, does a pre-trained model
indeed have to stay large for its downstream transferability?

［#4］
In this paper, we examine supervised and self-supervised
pre-trained models through the lens of the lottery ticket hy-
pothesis (LTH) [31]. LTH identifies highly sparse matching
subnetworks that can be trained in isolation from (nearly)
scratch yet still reach the full models’ performance. We
extend the scope of LTH and question whether matching
subnetworks still exist in pre-trained computer vision mod-
els, that enjoy the same downstream transfer performance.
Our extensive experiments convey an overall positive mes-
sage: from all pre-trained weights obtained by ImageNet
classification, simCLR, and MoCo, we are consistently able
to locate such matching subnetworks at 59.04% to 96.48%
sparsity that transfer universally to multiple downstream
tasks, whose performance see no degradation compared to
using full pre-trained weights. Further analyses reveal that
subnetworks found from different pre-training tend to yield
diverse mask structures and perturbation sensitivities. We
conclude that the core LTH observations remain generally
relevant in the pre-training paradigm of computer vision, but
more delicate discussions are needed in some cases. Codes
and pre-trained models will be made available at: https:
//github.com/VITA-Group/CV_LTH_Pre-training.

## 1. Introduction
［#5］
Deep neural networks pre-trained on large-scale datasets
prevail as general-purpose feature extractors [23]. Moving
beyond the most traditional greedy unsupervised pre-training
[2], the most popular pre-training in computer vision (CV)
is arguably to train the model for supervised classification
on ImageNet [18]. Such supervised pre-training enables
the network to learn a hierarchy of generalizable features
[46]; it is widely acknowledged [36] to not only benefit the
subsequent fine-tuning on other visual classification datasets
(especially in small datasets and few-shot learning [71, 74]),
but also to accelerate/improve the training for different, more
complicated types of downstream vision tasks, such as object
detection and semantic segmentation [63, 41].

［#6］
![](./images/867760313137627855_1.jpg)

［#7］
Figure 1. Overview of our work paradigm: from pre-trained CV
models (both supervised and self-supervised), we study the exis-
tence of matching subnetworks that are transferable to many down-
stream tasks, with little performance degradation compared to using
full pre-trained weights. We find task-agnostic, universally trans-
ferable subnetworks at pre-trained initialization, for classification,
detection, and segmentation tasks.

［#8］
Several state-of-the-art self-supervised pre-training, such
as simCLR [10, 11] and MoCo [40, 16], have demonstrated
that it is instead possible to use unlabeled data in pre-training.
Their methods refer to no actual labels in pre-training, but
instead leverage self-generated pseudo labels [22, 25] or con-
trasting augmented views [10]. Impressively, self-supervised
pre-training yields pre-trained weights with comparable or
even better transferability and generalization, for various
downstream tasks, compared to their supervised pre-training
counterparts.

［#9］
A few recent efforts have shown to successfully scale
up pre-training in CV. That is perhaps most natural for self-

［#9］
supervised pre-training, since unlabeled images are cheap and easily accessible. Chen et al. [11] investigated to boost simCLR with massive unlabeled data in a task-agnostic way, and pointed out the key ingredient to be the use of big (deep and wide) networks during pretraining and fine-tuning. The authors found that, the fewer the labels, the more this approach (task-agnostic use of unlabeled data) benefits from a bigger network. After fine-tuning, the big network is reduced into a much smaller one with little performance loss by using task-specific distillation. We additionally note the latest works suggesting that supervised fine-tuning can also scale up to larger models and datasets beyond ImageNet [24].

［#10］
The extraordinary cost of pre-training can be amortized by transferring to many downstream tasks. However, such explosive sizes of pre-trained models can even make fine-tuning computationally demanding, urging us to ask: can we aggressively trim down the complexity of pre-trained models, without damaging their downstream transferability? Note that, the question asked is drastically different from the conventional scope of model compression [38] in CV, where a model is trained, compressed and/or tuned on the same dataset and specific task. In comparison, any simplification for a pre-trained model has to ensure its intact transferability to a variety of possible downstream tasks.

［#11］
To address this research gap, we turn our attention to lottery ticket hypothesis (LTH) [20, 27, 31, 50, 76, 81], a fast-rising field that investigates the sparse trainable sub-networks within full dense networks. The original LTH [31, 32] demonstrated small-scale networks contain sparse matching subnetworks capable of training in isolation from initialization to full accuracy. In other words, we could have trained smaller networks from the start if only we had known which subnetworks to choose. Recent investigations [59, 58] showed those matching subnetworks to transfer between related classification tasks. However, no study has closely examined the tantalizing possibility of universal transferability in LTH for CV models, i.e., if we treat the pre-trained weights as our initialization, whether matching subnetworks still exist in the pre-training models, that also enjoy the same downstream transfer performance? Are there universal sub-networks that can transfer to many tasks with no degradation in performance?

［#12］
The paper carries out the first comprehensive experimental study to seek these desired universal matching subnetworks, from both supervised and self-supervised pre-trained CV models. Our principled methodology bridges pre-training and LTH from two perspectives: i) Initialization via pre-training. In the previous larger-scale settings of LTH for CV [31, 69], the matching subnetworks are found at an early point in training. Instead, we aim to identify these matching subnetworks from dense pre-trained models (self-supervised or supervised), which creates an initialization directly amenable to sparsification. ii)
Transfer learning. Finding the matching subnetwork is an expensive investment, usually costing multiple rounds of pruning and re-training. To justify this extra investment, the found subnetwork must be able to be reused by various downstream tasks, as illustrated in Figure 1.

［#13］
The course of this study presents the following findings:
- Using iterative unstructured magnitude pruning [31], we identify matching sub-networks up to 67.23%, 59.04%, 95.60% sparsity, at pre-trained weights from ImageNet-equipped supervised pre-training, simCLR and MoCo, respectively. We also find matching sub-networks at pre-trained initialization with sparsity from 73.79% to 98.20% in a variety of classification, detection and segmentation downstream tasks.
- Subnetworks at 67.23%, 59.04% and 59.04% sparsity, found respectively using supervised ImageNet, simCLR and MoCo pre-training, are universally transferable to diverse downstream classification tasks with nearly the same accuracies.
- Subnetworks at 73.79%/48.80%, 48.80%/36.00% and 73.79%/83.22% sparsity, found respectively by supervised ImageNet, simCLR and MoCo, can transfer to downstream detection/segmentation tasks without sacrificing performance.
- Unlike previous matching subnetworks found at random initialization or early in training, we show that those identified at pre-trained initialization are more sensitive to structure perturbations. Also, different pre-training ways tend to yield diverse mask structures and perturbation sensitivities.
- Lastly, pruning from larger pre-trained models can also produce better transferable matching subnetworks.

［#14］
Practically speaking, this work sets the first step toward replacing large pre-trained models with smaller subnetworks, enabling much more efficient downstream tuning without inhibiting transfer performance. As pre-training becomes increasingly central in the CV field, our results shed light on the relevance of LTH in this new paradigm.

## 2. Related Works
### Pruning and Lottery Tickets Hypothesis
［#15］
A trained deep network could be pruned of excess capacity [49]. Pruning algorithms can be grouped into unstructured [39, 49, 38] and structured [54, 43, 86]: the former sparsifies based on weight magnitudes; while the latter considers hardware-friendliness by removing channels and so on.

［#16］
The discovery of LTH [31] deviates from the convention of after-training pruning, and points to the existence of independently trainable sparse subnetworks from scratch that can match the performance of dense networks. Follow-up investigations [55, 35] scale up LTH by rewinding approaches [33, 69], that re-initializes the subnetwork from

［#17］
<table>
<caption>Table 1. Details of pre-training and fine-tuning. We use the default implementations and hyperparameters [69, 10, 40, 16, 7, 51, 3]. The evaluation metrics also follow the standards [69, 16, 7, 3]. For the supervised learning, we use the training dataset to name the corresponding task for the same of simplicity, e.g. “ImageNet” represents the supervised pre-training classification task on ImageNet.</caption>
<thead>
  <tr>
    <th>Settings</th>
    <td colspan="3">Pre-training</td>
    <td colspan="5">Downstream Classification</td>
    <td>Downstream Detection</td>
    <td>Downstream Segmentation</td>
  </tr>
  <tr>
    <th></th>
    <td>ImageNet</td>
    <td>simCLR</td>
    <td>MoCov2</td>
    <td>CIFAR-10</td>
    <td>CIFAR-100</td>
    <td>SVHN</td>
    <td>Fashion-MNIST</td>
    <td>VisDA2017</td>
    <td>Pascal VOC2012/2007</td>
    <td>Pascal VOC 2012</td>
  </tr>
</thead>
<tbody>
  <tr>
    <th># Epochs/Iters</th>
    <td>10</td>
    <td>10</td>
    <td>10</td>
    <td>182</td>
    <td>182</td>
    <td>182</td>
    <td>182</td>
    <td>20</td>
    <td>50 Epochs/103K Iters</td>
    <td>30K Iters</td>
  </tr>
  <tr>
    <th>Batch Size</th>
    <td>256</td>
    <td>256</td>
    <td>256</td>
    <td>256</td>
    <td>256</td>
    <td>256</td>
    <td>256</td>
    <td>128</td>
    <td>8</td>
    <td>4</td>
  </tr>
  <tr>
    <th>Learning Rate</th>
    <td>0.0001</td>
    <td>0.0001</td>
    <td>0.0003</td>
    <td>0.1</td>
    <td>0.1</td>
    <td>0.1</td>
    <td>0.1</td>
    <td>0.001</td>
    <td>0.0001</td>
    <td>0.01</td>
  </tr>
  <tr>
    <th></th>
    <td colspan="3">Fixed schedule</td>
    <td colspan="3">×0.1 at 91,136 epoch</td>
    <td></td>
    <td>×0.1 at 10 epoch</td>
    <td>Cosine decay<br>from $10^{-4}$ to $10^{-6}$</td>
    <td>Linear warmup 100 Iters<br>×0.1 at 18K, 22K Iters</td>
  </tr>
  <tr>
    <th>Optimizer</th>
    <td colspan="10">SGD [70] with 0.9 momentum</td>
  </tr>
  <tr>
    <th>Weight Decay</th>
    <td>$1 \times 10^{-4}$</td>
    <td>$1 \times 10^{-4}$</td>
    <td>$1 \times 10^{-4}$</td>
    <td>$2 \times 10^{-4}$</td>
    <td>$2 \times 10^{-4}$</td>
    <td>$2 \times 10^{-4}$</td>
    <td>$2 \times 10^{-4}$</td>
    <td>$5 \times 10^{-4}$</td>
    <td>$5 \times 10^{-4}$</td>
    <td>$1 \times 10^{-4}$</td>
  </tr>
  <tr>
    <th>Eval. Metric</th>
    <td>Accuracy</td>
    <td>Retrieval Accuracy</td>
    <td></td>
    <td>Accuracy</td>
    <td>Accuracy</td>
    <td>Accuracy</td>
    <td>Accuracy</td>
    <td>Accuracy</td>
    <td>AP, AP₅₀, AP₇₅</td>
    <td>mIOU</td>
  </tr>
</tbody>
</table>

［#16］
the early training stage checkpoint rather than from scratch. LTH has been widely explored in image classification [31, 55, 76, 28, 34, 72, 80, 81, 56, 15], natural language processing [35, 83, 69, 67, 9], generative adversarial networks [17, 8], graph neural networks [14], and reinforcement learning [83]. Most of them adopt (iterative) unstructured weight magnitude pruning [38, 31]. [58, 59, 19] pioneer to study the transferability of the subnetworks identified on one image classification task to another. However, studying the universal transferability of LTH at pre-trained initializations among diverse CV tasks remains untouched.

［#18］
One most relevant work [9] to ours is from the natural language processing (NLP) field: the authors found universally transferable sparse matching subnetworks (at 40% to 90% sparsity), from the pre-trained initialization of BERT models [21]. Finding their work inspiring, we stress that transplanting their NLP findings to our CV models is highly nontrivial due to multiple barriers: (1) pre-training BERT in [9] uses only a self-supervised objective called “masked language model” (MLM) [21], while pre-training CV models has a significant variety of popular options, ranging from the supervised fashion [46], to self-supervision yet with numerous objectives [22, 40, 10]; (2) BERT models consist of self-attention and fully-connected sub-layers, differing much from the standard convolutional architectures in CV; (3) further complicating the issue is that different CV downstream tasks are known to rely on different priors and invariances; for example, while classification often calls on shift invariance, detection assumes location shift equivariance [77, 57]. That questions the feasibility of asking for one mask to transfer among them all. Such complicacy is well manifested by our delicate observations.

［#19］
Pre-training in Computer Vision. Supervised ImageNet pre-training has been a main CV workhorse [36, 41]. The recent surge of self-supervised pre-training suggest the potential of unlabeled data; examples include recovering the artificially corrupted inputs [65, 75, 84, 85], predicting pseudo-labels [22, 25, 61, 64, 5, 6, 13], or contrasting augmented views [1, 44, 45, 62, 73, 78, 87, 10, 40, 11, 16, 47, 82]. The state-of-the-art simCLR [10, 11] and MoCo [40, 16] pre-training can reduce the amount of labels needed for tuning downstream image classifiers, by two magnitudes.

［#20］
Pre-trained networks are usually subsequently fine-tuned, with the architectures unchanged. One exception is [4] which is the first to adapt the backbone architecture to fit different target datasets. It pre-trains a large super-net that contains many weight-shared sub-nets that can individually operate.

## 3. Preliminaries and Setups

［#21］
In this section, we provide the detailed experimental settings and our approaches to find matching subnetworks.

［#22］
Network. We use the official ResNet-50 [42] network architecture as our default backbone, while we will later compare on ResNet-152 in Section 5.2. For a particular classification downstream task, a task-specific final linear layer is added following [10]. YOLOv4 [3] and DeepLabv3+ [7] are adopted for the detection and segmentation downstream tasks respectively, which also take ResNet-50 as the backbone¹. Due to the various input and output scales, the first convolution layer in ResNet-50 and all classification, detection, segmentation heads are never pruned. Specifically, we let $f(x;\theta,\gamma)$ be the output of a ResNet-50 model with parameters $\theta \in \mathbb{R}^{d_1}$ (excluding the first convolution layer) and task-specific parameters $\gamma \in \mathbb{R}^{d_2}$ on an input image $x$.

［#23］
Pre-training. For the supervised pre-training, we use the official pre-trained ResNet-50² on the ImageNet dataset [18]. For the self-supervised, we adopt the pre-trained ResNet-50 models with simCLR³ [10] and MoCov2⁴ [16] on ImageNet.

［#24］
Datasets, Training and Evaluation. All pre-training experiments are conducted on ImageNet. For downstream tasks, we consider classification, object detection and semantic segmentation on multiple datasets. We use four natural image and one synthetic datasets to verify the transferability on classification: Fashion-MNIST [79], SVHN [60], CIFAR-10 [48], CIFAR-100 [48], and VisDA2017 [66]. These datasets vary remarkably in terms of sample size, color space,

---
［#22］
¹For complicated CV tasks, the large variety of model design options may possibly impact our observation. For example. object detectors fall under two-stage and one-stage categories, the former often achieving higher accuracy while the latter typically being faster. YOLOv4 [3] is a popular one-stage detector. We also include the results for two popular two-stage detectors, Faster RCNN [68] and SSD [53], in Section B.2.
［#23］
²The official Pytorch model zoo at https://pytorch.org/docs/stable/torchvision/models.html
［#23］
³The official simCLR model zoo at https://github.com/google-research/simclr
［#23］
⁴The official MoCov2 model zoo at https://github.com/facebookresearch/moco

［#25］
resolution, image source, and classes. Following [40, 16], we train object detection models on the combined training and validation set of Pascal VOC 2012 [29] and Pascal VOC 2007 [30], then evaluate them on the Pascal VOC 2007 test set. We train and evaluate semantic segmentation models on Pascal VOC 2012 training and validation sets. We follow the standard hyperparameters and evaluation metrics⁵ for all pre-training and downstream tasks, as in Table 1.

［#26］
Subnetworks. For a network $f(x;\theta,\cdot)$ with task-specific modules $\gamma$, its subnetworks can be depicted as $f(x;m \odot \theta,\cdot)$ with a pruning binary mask $m \in \{0,1\}^{d_1}$, where $\odot$ is the element-wise product. Let $\mathcal{A}_t^{\mathcal{T}}(f(x;\theta,\gamma))$ be a training algorithm (e.g., SGD with certain hyperparameters) that trains a network $f(x;\theta,\gamma)$ on a task $\mathcal{T}$ (e.g., CIFAR-10) for $t$ iterations. Let $\theta_p \in \{\theta_{\text{Img}},\theta_{\text{sim}},\theta_{\text{MoCo}}\}$ be the pre-trained weights on ImageNet, where $\theta_{\text{Img}}$ is the supervised pre-trained weight, $\theta_{\text{sim}}$ and $\theta_{\text{MoCo}}$ are from the self-supervised pre-training by simCLR [10] and MoCov2 [16]. Let $\theta_0$ be the random initialization, and $\theta_i$ be the network weights at the $i_{\text{th}}$ epoch which is trained from $\theta_0$. Let $\mathcal{E}^{\mathcal{T}}(f(x;\theta,\gamma))$ be the evaluation function of model $f$ returned from $\mathcal{A}_t^{\mathcal{T}}$ on the corresponding task $\mathcal{T}$. Below we define:

［#27］
1. Matching subnetworks. Following the definition in [32,9], a subnetwork $f(x;m\odot\theta,\gamma)$ is matching if it satisfies the following condition:

［#27］
$$
\mathcal{E}^{\mathcal{T}}\left(\mathcal{A}_{t}^{\mathcal{T}}\left(f\left(x ; m \odot \theta, \gamma\right)\right)\right) \geq \mathcal{E}^{\mathcal{T}}\left(\mathcal{A}_{t}^{\mathcal{T}}\left(f\left(x ; \theta_{p}, \gamma\right)\right)\right) \quad (1)
$$

［#28］
That is, matching subnetworks perform no worse than the full dense models under the same training algorithm $\mathcal{A}_t^{\mathcal{T}}$ and evaluation metric $\mathcal{E}^{\mathcal{T}}$.

［#29］
2. Winning ticket. If $f(x;m \odot \theta,\gamma)$ is a matching subnetwork with $\theta = \theta_p$ for $\mathcal{A}_t^{\mathcal{T}}$, it is a winning ticket for $\mathcal{A}_t^{\mathcal{T}}$.

［#30］
3. Universal subnetwork. A subnetwork $f(x;m \odot \theta,\gamma_{\mathcal{T}_i})$ with task-specific configurations of $\gamma_{\mathcal{T}_i}$, is universal for tasks $\{\mathcal{T}_i\}_{i=1}^N$ if and only if it is matching for each $\mathcal{A}_{t_i}^{\mathcal{T}_i}$. The task set $\{\mathcal{T}_i\}_{i=1}^N$ could be a group of (diverse) downstream tasks, such as classification, detection and segmentation.

［#31］
Pruning Methods. To find the subnetworks $f(x;m \odot \theta,\gamma)$, we adopt the classical iterative magnitude pruning (IMP) approach that is commonly used by the LTH literature [31, 32, 9]. We prune the network by first training the unpruned dense network to completion on a task $\mathcal{T}$ (i.e., applying $\mathcal{A}_t^{\mathcal{T}}$) and then removing a portion of weights with the globally smallest magnitudes [38, 69]. As revealed by previous works, in order to identify the most competitive matching subnetworks, the process needs to be iteratively repeated for several rounds. Algorithm 1 outlines the full IMP procedure in the supplement.

［#32］
![](./images/867760313137627855_2.jpg)

［#33］
Figure 2. Performance of pre-training tasks on ImageNet. The masks $(m_{\text{Img}}, m_{\text{sim}}$ and $m_{\text{MoCo}})$ of evaluated subnetworks are found on supervised, simCLR and MoCo pre-training tasks respectively by IMP. $\theta_{\text{Img}}$ = the pre-trained weights from the supervised ImageNet classification; $\theta_{\text{sim}}$ = the pre-trained weights of simCLR [10]; $\theta_{\text{MoCo}}$ = the pre-trained weights of MoCo [16].

［#34］
Although beyond the current scope, our future work plans to examine the practical speedup results on a hardware platform for our training and/or inference phases. For example, in the range of 70%-90% unstructured sparsity, XNNPACK [26] has already shown significant speedups over dense baselines on smartphone processors. Integrating structured pruning will be another future direction of our interest [81].

## 4. Transfer of Pre-training Winning Tickets

［#35］
In this section, we first show that there exist winning tickets using the pre-trained initialization on both self-supervised and supervised pre-training tasks. As shown in Figure 2, we find winning tickets with 67.23%, 59.04% and 95.60% sparsity for supervised ImageNet, self-supervised simCLR and MoCo pre-training tasks.

［#36］
Then, we investigate to what extent IMP subnetworks found for pre-training tasks can (universally) transfer to different downstream tasks. We ask the following questions:
> **Q1:** Are winning tickets $f(x;m_{\mathcal{P}} \odot \theta_p,\cdot)$, found on the pre-training task $\mathcal{P}$, also winning tickets for other downstream tasks $\mathcal{T}$?
>
> **Q2:** Are there common patterns in the transferability of winning tickets from different pre-trainings (e.g., supervised versus self-supervised)?
>
> **Q3:** Can the transferred subnetworks $f(x;m_{\mathcal{P}} \odot \theta_p,\cdot)$ outperform the subnetworks $f(x;m_{\mathcal{T}} \odot \theta_i,\cdot)$ $(\theta_i \in \{\theta_0,\theta_{5\%}^6,\theta_p\})$, found on a specific task $\mathcal{T}$?

### 4.1. Transfer to Classification Tasks

［#37］
As shown in Figures 3 and 4, evaluated subnetworks are divided into three groups, according to sources of

［#38］
⁵For detection experiments, we report the other evaluation metrics, $\text{AP}_{50}$ and $\text{AP}_{75}$ [16] in the supplement. The technical details of calculating the retrieval accuracy for simCLR and MoCo pre-training tasks are also included in the supplement.

［#39］
⁶Early weight rewinding [69, 32] improves the quality of found matching subnetworks. As indicated by [32], the best rewinding points usually lie in the first 1% ~ 5% training epochs. We take 5% for default comparison.

［#40］
![](./images/867760313137627855_3.jpg)

［#41］
Figure 3. Performance of IMP subnetworks with a range of sparsity from 0.00% to 98.20% (i.e., remaining weight from 100% to 1.80%) on downstream classification tasks, including CIFAR-10, CIFAR-100, SVHN and Fashion-MNIST. $(m_{Img}, \theta_{Img})$, $(m_{sim}, \theta_{sim})$ and $(m_{MoCo}, \theta_{MoCo})$ denote transfer performance of subnetworks found at pre-training tasks. Subnetworks with $(m_{\mathcal{T}_i}, \theta_p)$, $\mathcal{T}_i \in\{$CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST$\}$ and $\theta_p \in \{\theta_{Img}, \theta_{sim}, \theta_{MoCo}\}$ are identified on the downstream task $\mathcal{T}_i$ with pre-trained weights $\theta_p$. Subnetworks $(m_{\mathcal{T}_i}, \theta_0)$ and $(m_{\mathcal{T}_i}, \theta_{5\%})$ are found on the task $\mathcal{T}_i$ with the random initialization $\theta_0$ [31] and an early rewinding weights $\theta_{5\%}$ [69]. Curves with errors (shadow regions) are the average across three independent runs, with the standard deviations: same hereinafter.

［#42］
![](./images/867760313137627855_4.jpg)

［#43］
Figure 4. Performance of IMP subnetworks with a range of sparsity from 0.00% to 98.20% on the synthetic dataset, VisDA2017.

［#44］
$(m, \theta)$: i) transferred subnetworks with $(m_{\mathcal{P}}, \theta_p)$, $\mathcal{P} \in \{$Img, sim, MoCo$\}$ and $\theta_p \in \{\theta_{Img}, \theta_{sim}, \theta_{MoCo}\}$; ii) subnetworks found on a specific downstream tasks with pre-trained weights $(m_{\mathcal{T}}, \theta_p)$, $\mathcal{T} \in\{$CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST, VisDA2017$\}$; iii) subnetworks consists of $(m_{\mathcal{T}}, \theta_i)$, $\theta_i \in \{\theta_0, \theta_{5\%}\}$, identified with the original random initialization $\theta_0$ or early rewinding weights $\theta_{5\%}$ on downstream tasks $\mathcal{T}$. Summarizing all comprehensive results, our main observations are:

［#45］
**A1: Subnetworks with $(m_{\mathcal{P}}, \theta_p)$ universally transfer to diverse downstream classification tasks.** As shown in Figure 3 and Figure 4, compared with unpruned dense models, subnetworks found on pre-training tasks $(f(x; m_{Img} \odot \theta_{Img}, \cdot)$, $f(x; m_{sim} \odot \theta_{sim}, \cdot)$, $f(x; m_{MoCo} \odot \theta_{MoCo}, \cdot))$ transfer without sacrificing performance$^7$ by sparsity $(91.41\%, 91.41\%, 91.41\%)$ to CIFAR-10, $(86.58\%, 86.58\%, 89.26\%)$ to CIFAR-100, $(91.41\%, 96.48\%, 93.13\%)$ to SVHN, $(89.26\%, 89.26\%, 91.41\%)$ to Fashion-MNIST, and $(67.23\%, 59.04\%, 59.04\%)$ to VisDA2017. Therefore, we observe that subnetworks produced by supervised ImageNet, self-supervised simCLR and MoCo pre-training tasks, universally transfer to four downstream natural image datasets

［#45］
$^7$Practically, to account for random fluctuations, we consider a subnetwork to be a winning ticket as long as its performance is within one standard deviation of the unpruned dense model.

［#46］
with sparsity (86.58%, 86.58%, 89.26%), respectively. However, it requires larger network capacity, i.e., (67.23%, 59.04%, 59.04%), to transfer to the synthetic VisDA2017 dataset without loss of performance.

［#47］
**A2: Winning tickets from different pre-training ways, have diverse behaviors, that are also affected by the downstream task properties.** On natural image datasets, subnetworks found with self-supervised pre-training (i.e., simCLR and MoCo) outperform subnetworks found with supervised ImageNet pre-training at the extreme sparsity level (e.g., more than 93.13%). Specifically, $f(x; m_{\text{sim}} \odot \theta_{\text{sim}}, \cdot)$ consistently achieves superior generalization across four downstream datasets. $f(x; m_{\text{MoCo}} \odot \theta_{\text{MoCo}}, \cdot)$ performs worse than $f(x; m_{\text{Img}} \odot \theta_{\text{Img}}, \cdot)$ at the low and middle level sparsity of subnetworks. However, the conclusions are almost flipped when transferring $f(x; m_{\mathcal{P}} \odot \theta_{p}, \cdot)$ to the synthetic VisDA2017 dataset. Subnetworks $f(x; m_{\text{Img}} \odot\theta_{\text{Img}}, \cdot)$ surpass others with a large performance margin, at the sparsity from 0.00% to 89.26%. For the extreme sparsity, the MoCo pre-training task generates a better transferable subnetworks. These observations suggest that supervised ImageNet pre-training allows subnetworks to transfer to the downstream datasets even with domain gaps to the pre-training datasets (e.g., from natural to synthetic images); self-supervised pre-trainings (e.g., simCLR and MoCo) produce more transferable subnetworks especially at the extreme sparsity, when natural image datasets are at downstream.

［#48］
**A3: Transferred subnetworks $f(x; m_{\mathcal{P}} \odot \theta_{p}, \cdot)$ perform the best until extreme sparsity.** Subnetworks $f(x; m_{\mathcal{T}} \odot \theta_{p}, \cdot)$, found on a specific downstream task with pre-trained weights, can be considered as *performance upbound* for all our IMP subnetworks. $f(x; m_{\mathcal{T}} \odot \theta_{p}, \cdot)$ is identified as matching subnetworks with the sparsity (98.20%, 91.41%, 73.79%) for CIFAR-10, (91.41%, 91.41%, 20.00%) for CIFAR-100, (91.41%, 95.60%, 91.41%) for SVHN, (89.26%, 96.48%, 73.79%) for Fashion-MNIST, and (73.79%, 59.04%, 67.23%) for VisDA2007.

［#49］
For universal transferable subnetworks, we observe: i) $f(x; m_{\text{Img}} \odot \theta_{\text{Img}}, \cdot)$ and $f(x; m_{\text{sim}} \odot \theta_{\text{sim}}, \cdot)$ match the corresponding $f(x; m_{\mathcal{T}} \odot \theta_{p}, \cdot)$ with at most 59.04% sparsity; ii) On the natural image datasets, $f(x; m_{\text{MoCo}} \odot \theta_{\text{MoCo}}, \cdot)$ steadily outperform $f(x; m_{\mathcal{T}}\odot\theta_{p}, \cdot)$ by a clear margin across all sparsity levels, especially for CIFAR-100; On the synthetic dataset, it fails to match under an excessive sparsity (i.e., > 83.22%). Note that subnetworks with $\theta_0$ and $\theta_{5\%}$ are inferior on all downstream tasks, compared to subnetworks with pre-trained initialization $\theta_p$.

### 4.2. Transfer to Detection and Segmentation
［#50］
Training detection and segmentation models commonly starts from pre-trained initializations [63, 41, 16]. We compare the transferred subnetworks with $(m_{\mathcal{P}}, \theta_p)$ versus the downstream task subnetworks with $(m_{\mathcal{T}}, \theta_p)$, as shown in Figure 5. Observations are organized as follows:

［#51］
**A1: Subnetworks $f(x; m_{\mathcal{P}} \odot \theta_{p}, \cdot)$ transfer to the detection and segmentation tasks successfully.** Figure 5 demonstrates it is manageable to find transferable winning tickets on the detection and segmentation with the sparsity (73.79%, 48.80%, 73.79%) and (48.80%, 36.00%, 83.22%) for supervised ImageNet pre-training, self-supervised simCLR and MoCo pre-training tasks respectively.

［#52］
**A2: Unlike classification, winning tickets from diverse pre-training tasks behave similarly on downstream detection and segmentation tasks.** In Figure 5, we observe the evident ranking of achieved transfer performance across all sparsity levels: $\mathcal{E}^{\mathcal{T}}(f(x; m_{\text{MoCo}} \odot \theta_{\text{MoCo}}, \cdot)) > \mathcal{E}^{\mathcal{T}}(f(x; m_{\text{Img}} \odot \theta_{\text{Img}}, \cdot)) > \mathcal{E}^{\mathcal{T}}(f(x; m_{\text{sim}} \odot \theta_{\text{sim}}, \cdot))$, $\mathcal{T} \in \{\text{detection}, \text{segmentation}\}$. It suggests that MoCo pre-trained weights are most favorable for transferring to detection and segmentation tasks [16].

［#53］
**A3: Subnetworks $f(x; m_{\mathcal{T}} \odot \theta_{p}, \cdot)$ surpass subnetworks $f(x; m_{\mathcal{P}} \odot \theta_{p}, \cdot)$ by a non-negligible margin.** As shown in Figure 5, with the assistance from the pre-trained initialization ($\theta_{\text{Img}}, \theta_{\text{sim}}, \theta_{\text{MoCo}}$), we find winning tickets with the sparsity at level (95.60%, 93.13%, 97.75%) and (73.79%, 67.23%, 86.58%) for detection and segmentation respectively. These identified winning tickets consistently outperform transferred subnetwork with $(m_{\mathcal{P}}, \theta_p)$.

## 5. Analyzing Properties of Pre-training Tickets
### 5.1. Comparing Masks from Different Pre-trainings
［#54］
In Figure 6, we compare the overlap in sparsity patterns found for different pre-training tasks. Relative similarity (i.e., $\frac{m_i \cap m_j}{m_i \cup m_j}$ in [9]) are reported, which reflects the overlap degree between hamming masks $m_i$ and $m_j$, where $i, j \in \{\text{Img, sim, MoCo}\}$. We find that subnetworks for pre-training tasks are remarkably heterogeneous: they share less than 6.55% locations in common after five-round IMP; the more sparsified, the larger differences.

［#55］
We also calculate the number of completely pruned (zero) kernels of subnetworks in Figure 6, which roughly reveals the weight clustering status in the sparse models. We observe that the remaining weights of subnetworks identified on the MoCo pre-training task are more clustered (i.e. more zero kernels) than the ones from ImageNet and simCLR, until reaching an extreme sparsity like 95.60%.

［#56］
Specifically, we provide kernel-wise heatmap visualizations of subnetworks with 79.03% sparsity in Figure 7. We find that the completely pruned (zero) kernels are mainly clustered in the early layers of subnetworks, and appear rarely in the later layers. Among three kinds of subnetworks, the one from MoCo has the most dispersed distribution of completely pruned kernels. In general, more struc-

［#57］
![](./images/867760313137627855_5.jpg)

［#58］
Figure 5. Performance of IMP subnetworks with a range of sparsity from 0.00% to 98.20% on the downstream detection and segmentation tasks. Subnetworks with $(m_{VOC2007}, \theta_p)$ and $(m_{VOC2012}, \theta_p)$, $\theta_p \in \{\theta_{Img}, \theta_{sim}, \theta_{MoCo}\}$ are identified on the downstream detection and segmentation tasks with pre-trained weights $\theta_p$, respectively. The standard deviations are around $0.1\% \sim 2.5\%$ AP/mIOU.

［#59］
![](./images/867760313137627855_6.jpg)

［#60］
Figure 6. Top: The relative mask similarity between subnetworks which identified on supervised ImageNet, simCLR and MoCo pre-training tasks. Bottom: The number of completely pruned (zero) kernels in subnetworks found on different pre-training tasks.

［#61］
![](./images/867760313137627855_7.jpg)

［#62］
Figure 7. Kernel-wise heatmap visualizations of subnetworks with 79.03% sparsity found on supervised ImageNet, simCLR and MoCo pre-training tasks. From left to right, we visualization all kernels of subnetworks from the input to the output layers. The bright dots ($\bullet$) represent the completely pruned (zero) kernels and the dark dots ($\bullet$) the kernels having at least one unpruned weight. B1$\sim$B4 donate four residual blocks in the ResNet-50 backbone.

［#63］
tured sparse subnetworks (i.e., more all-zero kernels) may have a stronger potential for hardware speedup [26].

### 5.2. Pre-training versus Random Initialization

［#64］
A signature of our setting is to treat pre-trained weights as the initialization, in contrast to most LTH works starting from random initialization [31, 32]. These two configurations produce matching subnetworks with diverse behaviors, including generalization performance and the structure sensitivity of obtained masks. We perform IMP on CIFAR-100 with the original random initialization $\theta_0$, early rewinding weights $\theta_{5\%}$, and the pre-trained weights $\theta_{Img}$ respectively, and then generates subnetworks consisting of $(m_{CIFAR-100}, \theta)$, $\theta \in \{\theta_0, \theta_{5\%}, \theta_{Img}\}$. As for comparison baselines, we consider three mask variants, the complementary masks $m^c_{CIFAR-100}$, randomly pruned masks $m_r$, and the perturbed masks $m_{CIFAR-100} + \Delta m_{10\%}$ as in Figure 8. Several observations can be draw as follows:

［#65］
- Starting from $\theta_0$ or $\theta_{5\%}$, identified subnetworks are resilient to structure perturbations. In other words, there only exist marginal performance differences across subnetworks with masks $m_{CIFAR-100}$, $m^c_{CIFAR-100}$, $m_r$ and $m_{CIFAR-100} + \Delta m_{10\%}$. However, the found subnetworks with the pre-trained initialization behave in sharp contrast, that all complementary masks, random pruned masks and perturbed masks substantially degraded the performance w.r.t. the IMP masks. A possible explanation is that the pre-trained initializations are already highly structured, and perturbations can destroy the intrinsic structure. As evidenced by the right subfigure of Figure 8, subnetworks with $(m^c_{CIFAR-100}, \theta_{Img})$ are no better than subnetwork with $(m_{CIFAR-100}, \theta_0)$. It shows that pre-training with damaged weight distributions no longer leads to the generalization gains.

［#66］
- Comparing the randomly pruned subnetworks in Figure 8, we observe that pre-trained initialization consistently benefits the accuracy until subnetworks reaching some high sparsity (e.g., 67.23%). After that, the performance of random pruned subnetworks is no longer affected by different initializations.

［#67］
![](./images/867760313137627855_8.jpg)

［#68］
Figure 8. Performance comparison across subnetworks found on CIFAR-100 with the original random initialization $\theta_0$, early rewinding weight $\theta_{5\%}$, and the pre-trained weights $\theta_{\text{img}}$. $m_{\text{CIFAR-100}}$ = masks found by IMP; $m_{\text{CIFAR-100}}^c$ = the complementary masks of $m_{\text{CIFAR-100}}$, where $m \cap m^c = \emptyset$ and $m \cup m^c = \mathbf{1} \in \mathbb{R}^{d_1}$; $m_r$ = random pruned mask; $\Delta m_{10\%}$ = mask perturbations by randomly flipping 10% "1" and 10% "0" in the mask $m \in \{0,1\}^{d_1}$ to its opposite value. Curves are the average across three independent runs.

### 5.3. More Ablation Studies for Pre-training

#### Larger Pre-training Model?
［#69］
[52] reveals that heavily compressed, large transformer models achieve higher performance than lightly compressed, small transformer models in natural language processing. We re-confirm this claim for self-supervised simCLR pre-training, in terms of the transferability⁸ of found matching subnetworks.

［#70］
In Figure 9, with the same number of remaining weights, subnetworks pruned from simCLR⁹ pre-trained ResNet-152, achieve consistently superior accuracy on the downstream CIFAR-100 task than the ones from simCLR pre-trained ResNet-50 (around one-third size of ResNet-152). At least for simCLR, pruning from larger pre-trained models produces better transferable matching subnetworks.

［#71］
Our observation is also aligned with the advocates of [11], to first pretrain a big model and then compress it. The key difference is that, [11] uses standard model compression (knowledge distillation) *after downstream fine-tuning is done*; in contrast, our results can be seen as a possible second pre-training stage: after the initial pre-training (and before any fine-tuning), performing IMP to find equally-capable matching subnetwork with far fewer parameters.

［#72］
![](./images/867760313137627855_9.jpg)

［#73］
Figure 9. Transfer performance on CIFAR-100 over the number of remaining weights. Subnetworks are found on the simCLR pre-training task with pre-trained ResNet-50 and ResNet-152 weights.

#### Temperature Hyperparameter.
［#74］
The temperature scaling hyperparameter is known to play a significant role in the quality of the simCLR pre-training [10, 11, 12]. It motivates us to investigate the impact of the temperature scaling factor on the transferability of pre-training winning tickets found in Section 4. Without loss of the generality, we consider the subnetworks with the sparsity from 67.23% to 73.79%. Specifically, we start from training subnetworks at the sparsity level 67.23% for 10 epochs, on the simCLR task with different temperature scaling factors. Then, they are pruned to the level of 73.79% sparsity by IMP. Finally, subnetworks are fine-tuned and evaluated on the downstream CIFAR-100 task. Results in Table 2 show that found subnetworks have close transfer performance if the temperature scaling factor lies in a moderate range (i.e., [0.1, 0.5]), and the performance will degrade at extreme temperatures (e.g., 20.0).

［#75］
Table 2. Ablation study of temperature parameter in simCLR. Transfer performance (i.e., accuracy) of subnetworks $(m_{\text{sim}}, \theta_{\text{sim}})$ with 73.79% sparsity on CIFAR-100 downstream task.

［#76］
<table>
<thead>
  <tr>
    <th>Temperature</th>
    <th>0.1</th>
    <th>0.2</th>
    <th>0.5</th>
    <th>1.0</th>
    <th>2.0</th>
    <th>10.0</th>
    <th>20.0</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Accuracy (%)</td>
    <td>81.81</td>
    <td>81.91</td>
    <td>82.22</td>
    <td>81.24</td>
    <td>80.76</td>
    <td>81.46</td>
    <td>80.18</td>
  </tr>
</tbody>
</table>

## 6. Conclusion

［#77］
We study the lottery ticket hypothesis in the context of CV pre-training, via both supervised (e.g., ImageNet classification) and self-supervised (e.g., simCLR and MoCo) ways. Despite the complicacy of our goal, by performing IMP from the pre-trained initializations, we are consistently able to find matching subnetworks at non-trivial sparsity levels, that can be independently trained to full model performance, on both pre-training and downstream tasks. We also present a detailed discussion of cross-task universal transferability.

### Acknowledgments

［#78］
We are grateful fpr the MIT-IBM Watson AI Lab, in particular John Cohn for generously providing the computing resources necessary to conduct this research. Wang's work is in part supported by the NSF Energy, Power, Control, and Networks (EPCN) program (Award number: 1934755), and by an IBM faculty research award.

［#79］
⁸In the supplement, we also report the pre-training task performance of subnetworks generated from small- and large-scale pre-trained simCLR.
⁹For a fair comparison, here we adopt the simCLRv2[11] pre-trained ResNet-152 and ResNet-50 models, since only simCLRv2 released the official pre-trained ResNet-152 model.

### References


























































































## A. More Technical Details
### A.1. Pruning Algorithm

［#80］
Following the routines in previous LTH [31, 9] works, the algorithm 1 outlines the full iterative magnitude pruning (IMP) procedure.

［#81］
```
Algorithm 1 Iterative Magnitude Pruning (IMP)
 1: Set the initial mask to $m = 1^{d_1}$, with the pre-training $\theta_p$.
 2: repeat
 3:    Train $f(x; m \odot \theta_p, \gamma_p)$ for $t$ epochs with algorithm $\mathcal{A}^T$,
       i.e., $\mathcal{A}_t^T(f(x; m \odot \theta_p, \gamma_p))$
 4:    Prune 20% of remaining weights in $\mathcal{A}_t^T(f(x; m \odot \theta_p, \gamma_p))$
       and update $m$ accordingly
 5: until the sparsity of $m$ reaches the desired sparsity level $s$
 6: Return $f(x; m \odot \theta_p)$.
```

### A.2. Top-1 Retrieval Accuracy

［#82］
Here we presents the detailed calculation of top-1 retrieval accuracy for self-supervised pretraining tasks, including simCLR [10] and MoCo [40]. Given a batch of data with $n$ samples, $\{z_1, \cdots, z_n\}$ and $\{z_1', \cdots, z_n'\}$ donates the feature representations from the two branches of simCLR or MoCo models. $z_i$ and $z_i'$ are computed from the same input sample with different data augmentations.

［#83］
For each $z_i$, we calculate the cosine similarity between $z_i$ and other representations and obtain $\mathcal{D}_i = \{d(z_i, z)|z \in \{z_j, z_j'\}_{j=1}^n / \{z_i\}\}$, where $d(\cdot, \cdot)$ is the cosine similarity measurement. If $\operatorname{argmax}_z \mathcal{D}_i = z_i'$, it suggests the top-1 retrieval is corrected. In the same way, we perform a similar retrieval process for $z_i'$ and $\mathcal{D}_i'$. The concrete calculation formulation of top-1 retrieval accuracy is depicted as follows:

［#83］
$$
\frac{\sum_{i=1}^n \left[\mathbb{I}(\operatorname{argmax}_z \mathcal{D}_i=z_i')+\mathbb{I}(\operatorname{argmax}_z \mathcal{D}_i'=z_i)\right]}{2 \times n} \times 100\%, \quad (2)
$$

［#83］
where $\mathbb{I}(\cdot)$ is the indicator function.

## B. More Experimental Results
### B.1. YOLOv4 Detection Results with Other Metrics

［#84］
In this section, we report the other two evaluation metrics, i.e.,$\text{AP}_{50}$ and $\text{AP}_{75}$, for YOLOv4 detection experiments. As shown in Figure 10, similar observations can be drawn that there are subnetworks $f(x; m_{\mathcal{P}} \odot \theta_{\text{p}}, \cdot)$ capable of transferring to the detection task successfully (i.e., without performance degradation compared with full unpruned models) at the $(83.22\%,89.26\%,36.00\%)$ and $(73.79\%,48.80\%,79.03\%)$ sparsity levels under the $\text{AP}_{50}$ and $\text{AP}_{75}$ metrics for supervised ImageNet, self-supervised simCLR and MoCo pre-training tasks, respectively.

［#85］
![](./images/867760313137627855_10.jpg)

［#86］
Figure 10. Performance ($\text{AP}_{50}$ and $\text{AP}_{75}$) of IMP subnetworks with a range of sparsity from 0.00% to 98.20% on the downstream tasks. Subnetworks $(m_{\text{VOC2007}}, \theta_p)$, $\theta_p \in \{\theta_{\text{Im}g}, \theta_{\text{sim}}, \theta_{\text{MoCo}}\}$ are identified on the detection task with pre-trained weights $\theta_p$.

### B.2. Faster RCNN and SSD Detection Results

［#87］
In this section, we conduct extra experiments with Faster RCNN [68] and SSD [53] on Pascal VOC datasets. Specifically, we train detection models for 24K/120K iterations with a batch size 4/32, a polynomial learning rate (LR) decay (with power 0.9 and initial LR 0.005) / a multi-step learning rate decay (with initial LR 0.001 amd $\times$0.1 at the 80K, 110K iterations), SGD optimizer with 0.9 momentum, and 0.0001/0.0005 weight decay for Faster RCNN/SSD detectors, respectively. As shown in 11, the most different observation is the winning tickets $f(x; m_{\mathcal{P}} \odot \theta_p, \cdot)$ found on the pre-training tasks are almost no longer matching subnetworks on the detection task with both Faster RCNN and SSD, which incurs performance degradation compared to unpruned dense models $f(x; \theta_p, \cdot)$. There is an exception that the subnetworks $f(x; m_{\text{MoCo}} \odot \theta_{\text{MoCo}}, \cdot)$ successfully transfer to the detection task with Faster RCNN at the 59.04% sparsity level only under the $\text{AP}_{50}$ metric, and with SSD at the 83.22%, 86.58%, 83.22% sparsity level under $\text{AP}$, $\text{AP}_{50}$, $\text{AP}_{75}$ metrics, respectively. Other conclusions are consistent with the ones of YOLOv4 detector in the main text.

［#88］
![](./images/867760313137627855_11.jpg)

［#89］
Figure 11. Performance (AP, AP₅₀ and AP₇₅) of IMP subnetworks with a range of sparsity from 0.00% to 98.20% on the downstream tasks. Subnetworks $(m_{VOC2007}, \theta_p)$, $\theta_p \in \{\theta_{Img}, \theta_{sim}, \theta_{MoCo}\}$ are identified on the detection task with pre-trained weights $\theta_p$. Results of Faster RCNN with three independent runs and SSD with one run are presented here.

［#90］
We notice that detection results of Faster RCNN and SSD with the simCLR pre-training show inferior and unsatisfactory performances, compared with the reported number in BYOL [37]. Although BYOL is implemented with Tensorflow (we use Pytorch) and also has an extra residual block for backbone network, the performance gap is not neglectable. To address this, multiple authors have worked to carefully tune all hyperparameters (learning rate, batch size, training iterations), and thoroughly compared implementation details side-to-side (batch norm, input resolution, etc.). However, we still cannot close the gap. Hence while our results on MoCo and ImageNet are very consistent, we cannot exclude the marginal possibility that simCLR implementation is specifically sensitive to Pytorch versus Tensorflow frameworks (unfortunately, not uncommon) for some reason. Therefore, we put Faster RCNN and SSD detection

［#90］
results with the simCLR pre-training in the appendix as failure cases, and note that it hardly affects any of our main observations/conclusions.

### B.3. Ablation about Larger Pre-training Models

［#91］
Figure 12 collects the pre-training task performance of subnetworks generated from small- and large-scale pre-trained simCLR models. We observe that heavily compressed, large simCLR models (e.g., ResNet-50) obtain superior performance to lightly compressed, small simCLR models (e.g., ResNet-152), which is consistent with [52]. However, subnetworks found on the small-scale pre-trained simCLR model show a slightly better top-1 retrieval accuracy after the sparsity approaches an extreme level.

［#92］
![](./images/867760313137627855_12.jpg)

［#93］
Figure 12. Pre-training performance (top-1 retrieval accuracy as defined in Equation 2) over the number of remaining weights. Subnetworks are found on the simCLR pre-training task with pre-trained ResNet-50 and ResNet-152 weights.