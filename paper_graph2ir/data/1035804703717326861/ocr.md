# 3D Point Cloud Network Pruning: When Some Weights Do not Matter

Amrijit Biswas*¹
amrijit.biswas01@northsouth.edu

Md. Ismail Hossain*¹
ismail.hossain2018@northsouth.edu

M M Lutfe Elahi¹
lutfe.elahi@northsouth.edu

Ali Cheraghian²,³
ali.cheraghian@data61.csiro.au

Fuad Rahman⁴
fuad@apurbatech.com

Nabeel Mohammed¹
nabeel.mohammed@northsouth.edu

Shafin Rahman¹†
shafin.rahman@northsouth.edu

¹ Department of Electrical and
Computer Engineering,
North South University, Bangladesh

² Data61-CSIRO, Australia

³ The Australian National University,
Canberra, Australia

⁴ Apurba Technologies, Sunnyvale,
CA 94085, USA

## Abstract
A point cloud is a crucial geometric data structure utilized in numerous applications. The adoption of deep neural networks referred to as Point Cloud Neural Networks (PCNNs), for processing 3D point clouds, has significantly advanced fields that rely on 3D geometric data to enhance the efficiency of tasks. Expanding the size of both neural network models and 3D point clouds introduces significant challenges in minimizing computational and memory requirements. This is essential for meeting the demanding requirements of real-world applications, which prioritize minimal energy consumption and low latency. Therefore, investigating redundancy in PCNNs is crucial yet challenging due to their sensitivity to parameters. Additionally, traditional pruning methods face difficulties as these networks rely heavily on weights and points. Nonetheless, our research reveals a promising phenomenon that could refine standard PCNN pruning techniques. Our findings suggest that preserving only the top p% of the highest magnitude weights is crucial for accuracy preservation. For example, pruning 99% of the weights from the PointNet model still results in accuracy close to the base level. Specifically, in the ModelNet40 dataset, where the base accuracy with the PointNet model was 87.5%, preserving only 1% of the weights still achieves an accuracy of 86.8%. Codes are available in: https://github.com/apurba-nsu-rnd-lab/PCNN_Pruning

© 2024. The copyright of this document resides with its authors.
It may be distributed unchanged freely in print or electronic forms.
* Equal Contribution.
† Corresponding author.

# 1 Introduction

The ability to analyze and comprehend 3D data is becoming increasingly vital in various industries, such as autonomous driving [51], robotics [54], augmented and virtual reality [57], and computational biology [43]. The widespread availability and diverse application range of 3D data have contributed to its growing importance. With the continued advancement of deep learning technologies, researchers are exploring innovative ways to process and interpret 3D data effectively. This marks the dawn of a new era in 3D deep learning research, commonly referred to as Point Cloud Neural Networks (PCNNs) [19]. Early innovations, such as PointNet [44], have laid the groundwork for developing deep learning architectures specifically designed to process raw point clouds. Subsequent developments, including PointCNN [52], PointConv [59], PointCLIP [63], and Uni3d [67], have introduced improved strategies for capturing minute geometric details and enhancing matching accuracy through advanced convolution and attention mechanisms. The growing need for innovative deep-learning solutions demands efficient models and reasoning with 3D data. However, as models become more complex over time, the number of parameters increases significantly. This substantial increase can lead to heightened latency and computational constraints, posing significant challenges to the efficient deployment and operation of models.

To address the challenge, four standard methods are commonly utilized: weight quantization [61], sparsity through regularization [18], knowledge distillation [68], and network pruning [8]. Each method has its own advantages and disadvantages. While they aim to improve efficiency and reduce computational demands, each technique may also potentially compromise accuracy or introduce other trade-offs that need to be carefully considered. In that context, network pruning is a technique, especially unstructured pruning, that has yet to be extensively explored within this specific 3D domain (see Figure 1). Unstructured pruning [43] has emerged as a promising technique for model compression in other domains. It involves identifying and eliminating redundant or irrelevant weights from the network, thereby creating a more compact model without sacrificing accuracy. Although this method, especially the recently proposed Lottery Ticket Hypothesis (LTH) [17], has found widespread application in various fields, its effectiveness in compressing 3D models has not yet been tested.

LTH suggests that there are trainable subnetworks within larger neural networks, termed "winning tickets," that can achieve or exceed the original performance. These subnetworks, identifiable through iterative pruning, exhibit potential for task transferability [8], sparsity, enhanced performance, and convergence. Specifically, our aim is to address the following research questions: (a) Can we find winning ticket subnetworks within overparametrized 3D shape classification networks that maintain or exceed the original network's performance while being significantly sparser? (b) What are the optimal pruning strategies and techniques to effectively identify winning tickets in 3D models, considering the unique challenges posed by high-dimensional and geometrically complex data? (c) Can the winning ticket subnetworks obtained from a 3D shape classification task or dataset be transferred to different but related tasks while preserving high accuracy, thus demonstrating the transferability of these subnetworks? We found winning tickets within overparameterized PCNNs that maintained or sometimes exceeded the original performance while being significantly sparser. IMP [65] and global one-shot pruning [8] emerged as optimal pruning strategies for effectively handling the high-dimensional and geometrically complex nature of 3D data. In particular, we found that these winning ticket subnetworks exhibited transferability across different but related 3D shape classification tasks, preserving high accuracy. These investigations have

![](./images/1035804703717326861_1.jpg)

Figure 1: This paper deals with pruning 3D Point Cloud Neural Networks (PCNNs) for faster inference, especially on edge devices. (a) In 3D point cloud literature, traditionally, researchers carefully design less parametric deep models for this purpose (e.g., Spherical CNNs [], Dense Point [], KCNet [], Point PN []). (b) In contrast, we iteratively prune popular PCNNs (PointCNN [], DGCNN [], PointConv []) to produce task-specific subnetworks outperforming over-parameterized models. (c) Our results demonstrate the capability of the LTH to extract highly sparse subnetworks, termed "winning tickets," from an over-parameterized model. These winning tickets can achieve up to 99% sparsity, retaining only 1% of the original weights while still attaining accuracy levels comparable to the base model.

significant potential to propel the field of 3D model compression. To validate our hypotheses, we employ versatile 3D point cloud architectures like PointNet [], DGCNN [], and PointCNN []. Extensive experiments are conducted on challenging 3D datasets: ModelNet40 [], ScanObjectNN [], and ShapeNetCore [], encompassing diverse 3D objects and scenarios. The contributions of this paper are:

- Development of efficient sparse, task-specific subnetworks outperforming over parameterized models (e.g., PointCNN [], DGCNN [], PointConv []) for efficient 3D deployment.
- Comprehensive analysis of sparse subnetwork characteristics and performance of most popular models across diverse datasets (ModelNet40 [], ScanObjectNN [], and ShapeNetCore []).
- We establish that one-shot global pruning at a highly high 99% sparsity level can attain comparable accuracy to over-parameterized 3D models, substantially reducing parameter count and computational requirements. The rest of the 1% critical weights play a vital role in performance.

## 2 Related Works
3D Point Cloud Neural Networks: Existing methods for classifying 3D shapes can be broadly categorized into multi-view, volumetric, and point-based methods. Multi-view-based models, such as the Multi-View Convolutional Neural Network (MVCNN) [], convert unstructured 3D point clouds into 2D images from different perspectives. Extracted features from images are then combined to achieve a complete global representation. Volumetric-based techniques (VoxelNet [], OctNet [], and Octree-based CNN []) convert point clouds into a structured format, employing representations such as voxels or octrees. Subsequently, these methods utilize 3D Convolutional Neural Network (CNN) models, includ-

ing but not limited to performing shape classification. In contrast, point-based classification methods directly handle the raw, unstructured point clouds. These can be divided into several subcategories: pointwise Multilayer Perceptron (MLP), convolutional-based, graph-based, and hierarchical data structure-based methods. For instance, PointNet [] exemplifies the pointwise MLP approach by independently extracting features from each point through multiple MLP layers and aggregating them via a max pooling layer to capture global shape features. Meanwhile, PointCNN [], demonstrate the application of convolutional approaches tailored to point cloud data. Graph-based methods (DGCNN []) treat each point in the cloud as a vertex in a directed graph, leveraging the relationships between points to facilitate shape classification. This paper investigated 3D classification models from each subcategory.
3D model Compression: Recent advances in 3D PCNN compression research have predominantly focused on two main areas: geometry compression [] and attribute compression []. The literature primarily features methods utilizing convolution-based autoencoders (CNN-based AE), fully connected neural networks (FCNN), and multilayer perceptrons (MLP). Among the various strategies explored, channel pruning [] emerges as a significant approach, drawing inspiration from the principles of 2D channel pruning []. Application of this method to the ModelNet40 [] has demonstrated that a pruning rate of 58% can maintain, or in some cases improve, accuracy relative to the benchmark PCNN [, ]. Moreover, Point Distribution-Aware Pruning [] strategically prunes less significant neighborhood voxels to enhance compression efficiency. Some other works [, ] investigated the potential of employing PCNNs with a minimal number of parameters for 3D shape classification. Proposals for non-parametric networks [] aimed at learning 3D shapes have been introduced. Despite these efforts, such non-parametric models have shown limited success on the ScanObjectNN 3D dataset []. However, with a reduced parameter count of 0.8M, these models have achieved notable accuracy on the ScanObjectNN dataset, highlighting the trade-offs and potential pathways for further optimization in PCNN compression techniques. Yet, the utilization of LTH on 3D model compression still needs to be explored.
Lottery Ticket Hypothesis: The LTH [] has unleashed a new dimension in the domain of neural network pruning. Later, [] introduced the term "late resetting," which facilitates the LTH suitable for deep models. However, [] endorsed that weight resetting can be ambiguous. The reason behind the working process of experimental LTH was explored by []. Further, the theoretical proof of LTH was also evaluated by []. Authors [] showed that a winning ticket distribution can be present in a neural network. The long-term training issue of the LTH was addressed in []. Furthermore, the concept of "transfer ticket hypothesis" [] allows training a network by using a sparse network (winning ticket) generated from another dataset. Authors [] demonstrated that the transferability of the winning tickets does not cause overfitting issues. LTH technique is successfully applied in several computer vision applications, e.g., supervised pre-training [], object recognition [], vision-language models [], which inspired us to investigate LTH on 3D point cloud shape classification.

# 3 Lottery Ticket Hypothesis for 3D Shape Classification

Preliminaries: The LTH suggests that dense, randomly initialized neural networks contain subnetworks (known as "winning tickets") that can match the performance of the original network when trained in isolation. In the context of 3D shape classification, we aim to prove the existence of winning tickets within a neural network $\mathcal{F}(\mathbf{x} ; \theta)$ that maps 3D shapes $\mathbf{x} \in \mathcal{X}$

to class labels $\mathcal{Y}$ using parameters $\boldsymbol{\theta}$ initialized from a distribution $\mathcal{D}_{\boldsymbol{\theta}}$. To find the winning ticket subnetwork, we employ an iterative pruning and retraining process:

1. Initialize a binary mask $\mathbf{m} \in \{0,1\}^{|\boldsymbol{\theta}|}$ with all ones: $\mathbf{m} = \mathbf{1}$.
2. Train the network $\mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta})$ for one cycle.
3. Prune a percentage $p$ of the weights by setting the corresponding entries in $\mathbf{m}$ to zero based on a pruning criterion. Where $m_i=0$, if $|\theta_i| \leq \alpha$, otherwise $m_i=1$. Here $\alpha$ is a threshold value determined by the desired pruning percentage $p$, and $\mathbf{m}_i$ and $\theta_i$ are the $i^{\text{th}}$ elements of $\mathbf{m}$ and $\boldsymbol{\theta}$, respectively.
4. Repeat steps 2 and 3 for $j$ cycles or until the desired sparsity level is achieved.

The objective is to find a sparse subnetwork $\mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta})$ with accuracy $d' \geq a$, where $a$ is the accuracy of the original network $\mathcal{F}(\mathbf{x};\boldsymbol{\theta})$, while pruning rounds $J$.

### 3.1 Challenges of LTH and Pruning Methods

Proving the lottery ticket hypothesis for 3D shape classification models presents several challenges due to the geometric complexity and high dimensionality of 3D data:
(1) **Geometric Complexity**: 3D shapes exhibit intricate geometric structures and topological properties [57, 48, 58]. Let $\mathcal{G}$ be a set of geometric transformations (rotations, scaling), preserving the class label [29, 40, 69]. The subnetwork $\mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta})$ should satisfy:

$$
\mathcal{F}(\mathbf{g}(\mathbf{x});\mathbf{m} \odot \boldsymbol{\theta}) = \mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta}), \quad \forall \mathbf{g} \in \mathcal{G}, \mathbf{x} \in \mathcal{X}
$$

(2) **High Dimensionality**: 3D point cloud data presents unique challenges due to its high dimensionality and irregular structure [46, 51]. Unlike 2D images, 3D point clouds can contain millions of points in irregular spatial arrangements, leading to larger input sizes and more complex feature representations. This often requires networks with more parameters $|\boldsymbol{\theta}|$ to capture intricate spatial relationships and geometric features. The pruning objective for these larger networks is to minimize nonzero entries $|\mathbf{m}|_1$ without compromising accuracy:

$$
\min_{\mathbf{m}} |\mathbf{m}|_1 \quad \text{s.t.} \quad \mathcal{L}(\mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta})) \leq \mathcal{L}(\mathcal{F}(\mathbf{x};\boldsymbol{\theta}))
$$

where $\mathcal{L}$ is the loss function. This optimization is particularly challenging for 3D data due to the need to preserve complex spatial relationships and geometric features while significantly reducing the network size.
(3) **Structural Constraints**: 3D shapes can exhibit structural restrictions or relationships between different parts or components [11, 23, 69]. Let $\mathcal{C}$ be a set of structural constraints on the 3D shapes preserved by an overparametrized PCNN [8, 50]. If so, then the winning ticket subnetwork $\mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta})$ should preserve these constraints:

$$
\mathcal{C}(\mathbf{x}) \Rightarrow \mathcal{C}(\mathcal{F}(\mathbf{x};\mathbf{m} \odot \boldsymbol{\theta})), \quad \forall \mathbf{x} \in \mathcal{X}
$$

This formulation means that if a structural constraint $C$ holds for an input 3D shape $\mathbf{x}$, then the same constraint should also hold for the output of the winning ticket subnetwork. For example, if an input shape has bilateral symmetry, the pruned network should still recognize and preserve this symmetry in its processing. It is important to maintain these constraints to preserve the semantic and geometric integrity of 3D shapes during classification tasks. We show a pruning method that selectively retains weights with the highest magnitudes, as these are more likely to encode important structural features.

IMP and One-Shot Pruning: Let $\theta_{l}^{(j)}$ denote the weights of the $l^{th}$ layer in the PCNN $\mathcal{F}(\mathbf{x} ; \theta)$ at pruning round $j$, and $m_{l}^{(j)}$ be the corresponding binary mask for that layer. The IMP process is as follows: Initialize the binary mask $\mathbf{m}^{(0)}=\mathbf{1}$ (all ones). For each round $j=1,2, \ldots, J$: Train the network $\mathcal{F}(\mathbf{x} ; \mathbf{m}^{(j-1)} \odot \theta^{(j-1)})$ for one cycle. Determine the pruning threshold $\alpha_{l}^{(j)}$ for each layer $l$ based on the desired global or local pruning percentage $p$. Update the binary mask $m_{l}^{(j)}$ for each layer $l$ as follows: For global pruning [], we update the binary mask $m_{l}^{(j)}$ for layer $l$ at iteration $j$ in the following way. For each weight $\theta_{l}^{(j-1)}[i]$ at the index $i$, if its absolute value is less than or equal to the global threshold $\alpha^{(j)}$, we set the corresponding mask element $m_{l}^{(j)}[i]$ to 0 (pruned). Otherwise, we keep the mask element $m_{l}^{(j-1)}[i]$ at its previous value. Here, $\alpha^{(j)}$ is the global threshold determined by the $(1-p)^{t h}$ percentile of the sorted magnitudes of all weights across all layers. For local pruning, similarly, $m_{l}^{(j)}[i]=0$, if $|\theta_{l}^{(j-1)}[i]| \leq \alpha_{l}^{(j)} m_{l}^{(j-1)}[i]$, where $\alpha_{l}^{(j)}$ is the layer-wise threshold determined by the $(1-p)^{t h}$ percentile of the sorted magnitudes of weights in layer $l$. We update the weights $\theta^{(j)}=\mathbf{m}^{(j)} \odot \theta^{(j-1)}$. The pruning terminates if the desired sparsity level is achieved, or $j=J$. The one-shot pruning [] is equivalent to IMP when $J=1$.

Hypothesis: We hypothesize that the pruning methods (IMP, One-Shot) could address the challenges of geometric invariance, high dimensionality, sparsity, and structural constraint preservation in pruning by iteratively pruning the weights with the lowest magnitudes while retraining the network. By preserving the weights with the highest magnitudes, which are likely to capture salient geometric and structural features, the pruned network is expected to maintain geometric invariance $\mathcal{F}(\mathbf{g}(\mathbf{x}) ; \mathbf{m}^{(J)} \odot \theta^{(J)}) \approx \mathcal{F}(\mathbf{x} ; \mathbf{m}^{(J)} \odot \theta^{(J)})$ and structural constraints $\mathcal{C}(\mathbf{x}) \to \mathcal{C}(\mathcal{F}(\mathbf{x} ; \mathbf{m}^{(J)} \odot \theta^{(J)}))$, while achieving high sparsity $\min _{\mathbf{m}^{(J)}} \sum_{l}|\mathbf{m}_{l}^{(J)}|_{1}$ and maintaining or improving accuracy. To validate the hypothesis, empirical evaluation is necessary, which is provided in the following sections.

### 3.2 Transferability of 3D Subnetwork
Transfer learning in 3D shape classification sometimes faces challenges due to limited data availability and domain specificity, unlike the 2D domain []. Moreover, applying LTH to a specific task often requires multiple pruning rounds to achieve the desired sparsity, making it computationally expensive to find the winning ticket for each task. On the other hand, in the 2D domain, LTH has demonstrated promising results in transferring tickets from models trained on large datasets to models trained on smaller datasets []. Now, mathematically, let us consider two datasets, $D_1$ and $D_2$, both consisting of 3D shapes but with different distributions. Let $\mathcal{F}_{1}(\mathbf{x} ; \theta_{1})$ and $\mathcal{F}_{2}(\mathbf{x} ; \theta_{2})$ be the neural networks trained on $D_1$ and $D2$, respectively, for the task of 3D shape classification. The parameters $\theta_{1}$ and $\theta_{2}$ are initialized from the same distribution $D_{\theta}$. Suppose that we obtain a sparse subnetwork $\mathcal{F}_{1}(\mathbf{x} ; \mathbf{m}_{1} \odot \theta_{1})$ from $\mathcal{F}_{1}(\mathbf{x} ; \theta_{1})$ using the iterative pruning and retraining approach, where $\mathbf{m}_{1} \in\{0,1\}^{|\theta_{1}|}$ is the binary mask that induces sparsity. The sparse subnetwork achieves an accuracy $d_{1}^{\prime}$ on the validation set of $D_1$, which is reasonably close to the accuracy $a_1$ of the original network $\mathcal{F}_{1}(\mathbf{x} ; \theta_{1})$. To investigate the transferability of the sparse subnetwork, we initialize $\mathcal{F}_{2}(\mathbf{x} ; \theta_{2})$ with the parameters $\theta_{2}=\mathbf{m}_{1} \odot \theta_{1}$, where the non-zero elements of $\theta_{2}$ correspond to the winning tickets identified by the mask $\mathbf{m}_{1}$. We then fine-tune $\mathcal{F}_{2}(\mathbf{x} ; \theta_{2})$ on the dataset $D_2$ using standard training procedures. If the sparse subnetwork $\mathcal{F}_{2}(\mathbf{x} ; \theta_{2})$ can achieve an accuracy $d_{2}^{\prime}$ on the validation set of $D_2$ that is comparable to the accuracy $a_2$ of the original network

$\mathcal{F}_{2}(\mathbf{x} ; \theta_{2})$, it would indicate that the winning tickets identified by the mask $\mathbf{m}_{1}$ on $\mathcal{D}_{1}$ are transferable to the task of classifying 3D shapes from the distribution $\mathcal{D}_{2}$. Mathematically, the transferable ability can be quantified by the condition, $a_{2}^{\prime} \approx a_{2}, \quad$ where $\theta_{2}=\mathbf{m}_{1} \odot \theta_{1}$. If this condition holds, it suggests that the sparse subnetwork obtained from $\mathcal{F}_{1}(\mathbf{x} ; \theta_{1})$ can be effectively transferred to the task of classifying 3D shapes from the distribution $\mathcal{D}_{2}$, thus reducing computational cost and training time while maintaining comparable performance.

## 4 Experiments

Dataset: We have investigated LTH on several synthetic and real-world 3D point cloud datasets. ModelNet40 [60] is a synthetic Computer-aided design (CAD) generated dataset that contains 12,311 samples from 40 common objects. ScanObjectNN [53] is a real-world dataset made by scanning real-world objects containing 15,000 samples of 3D shapes from 15 common categories. ShapeNetCore is a subset of the original ShapeNet [4] dataset where 51,300 instances are available from 55 categories of 3D shapes.

Experimental Setups: Unstructured point clouds are normalized and then augmented by random rotation and transformation as data processing steps for training the PCNN. After that, a fixed number of points, usually 1024 or 2048, are sampled from each point cloud shape data. Different approaches for pruning, such as IMP, global pruning, local pruning, and one-shot pruning, are used on each dataset. After successfully obtaining a reasonable sparsity without any accuracy drop, the PCNN is further pruned to an extreme sparsity level to investigate the true potential of the sparse network.

During pruning, we investigated the possibility of making a PCNN up to 99% sparse. The optimal performances are included according to the observation of the performance of the PCNN models vs. the pruning rate. We considered the optimal performance to be equal to or better than the unpruned model with possible sparsity or a negligible compromised accuracy with enormous sparsity. As part of our experiment, We pruned 10%, 20%, 30%, and 40% weights globally in an iterative manner, and by conducting an empirical testing and validation approach we found the threshold pruning rate for the one-shot global pruning. The number of train-prune-rewind cycles was different for each of our PCNN models. Using 100 cycles, we established the optimal performance and sparsity of the PointNet. While training, a batch size of 256 and the Adam optimizer with a learning rate of 0.0001 is used. We found that DGCNN requires 80 cycles to reach optimal performance, which is less than PointNet. We used the SGD optimizer with a learning rate of 0.1 and a small batch size of 32 for DGCNN. Applying the train-prune-rewind cycle on PointCNN showed a bit more complication than the other two models due to the structural difference. However, we managed to attain the optimal state within 80 cycles. PointCNN is configured using a batch size of 128 and Adam optimizer, where the learning rate is set to 0.00001.

Network: We have experimented with the pointwise MLP-based method, PointNet [44], point convolution-based method, PointCNN [52], and graph-based methods, DGCNN[57]. These networks are established architectures to effectively learn from unstructured point cloud data inherent in 3D shape representations.

Evaluation: Following previous works [5, 17] on LTH, we evaluate our work based on testing accuracy and network pruning rate (sparsity) in percent.

<table>
 <thead>
  <tr>
   <th>Model</th>
   <th rowspan="2">PR</th>
   <th rowspan="2">Param (M)</th>
   <th colspan="3">ModelNet40</th>
   <th colspan="3">ScanObjectNN</th>
  </tr>
  <tr>
   <th>Accuracy(%)</th>
   <th>Base</th>
   <th>IMP</th>
   <th>OneShot</th>
   <th>Base</th>
   <th>IMP</th>
   <th>OneShot</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>PointNet</td>
   <td>0%</td>
   <td>3.400</td>
   <td>87.5</td>
   <td>-</td>
   <td>-</td>
   <td>68.2</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>Ours</td>
   <td>60%</td>
   <td>1.360</td>
   <td>-</td>
   <td>88.2</td>
   <td>87.8</td>
   <td>-</td>
   <td>71.7</td>
   <td>70.5</td>
  </tr>
  <tr>
   <td>Ours</td>
   <td>99%</td>
   <td>0.034</td>
   <td>-</td>
   <td>86.3</td>
   <td>86.9</td>
   <td>-</td>
   <td>64.7</td>
   <td>62.7</td>
  </tr>
  <tr>
   <td>DGCNN</td>
   <td>0%</td>
   <td>1.700</td>
   <td>89.2</td>
   <td>-</td>
   <td>-</td>
   <td>71.1</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>Ours</td>
   <td>60%</td>
   <td>0.680</td>
   <td>-</td>
   <td>90.1</td>
   <td>90.4</td>
   <td>-</td>
   <td>75.2</td>
   <td>74.9</td>
  </tr>
  <tr>
   <td>Ours</td>
   <td>99%</td>
   <td>0.017</td>
   <td>-</td>
   <td>88.5</td>
   <td>88.0</td>
   <td>-</td>
   <td>64.6</td>
   <td>60.6</td>
  </tr>
  <tr>
   <td>PointCNN</td>
   <td>0%</td>
   <td>0.320</td>
   <td>90.2</td>
   <td>-</td>
   <td>-</td>
   <td>70.4</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>Ours</td>
   <td>60%</td>
   <td>0.128</td>
   <td>-</td>
   <td>90.6</td>
   <td>90.4</td>
   <td>-</td>
   <td>73.2</td>
   <td>72.9</td>
  </tr>
  <tr>
   <td>Ours</td>
   <td>99%</td>
   <td>0.003</td>
   <td>-</td>
   <td>87.4</td>
   <td>86.9</td>
   <td>-</td>
   <td>65.3</td>
   <td>59.3</td>
  </tr>
 </tbody>
</table>

Table 1: Sparse subnetworks, with sparsity levels of 99% and 60%, are transferred from a model trained on the larger ShapeNetCore dataset to models trained on the smaller ModelNet40 and ScanObjectNN datasets, respectively.

<table>
 <thead>
  <tr>
   <th>Model</th>
   <th>Param (M)↓</th>
   <th>Accuracy(%)↑</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>PointCNN []</td>
   <td>0.320</td>
   <td>92.2</td>
  </tr>
  <tr>
   <td>Spherical CNNs []</td>
   <td>0.500</td>
   <td>88.9</td>
  </tr>
  <tr>
   <td>Dense Point []</td>
   <td>0.530</td>
   <td>93.2</td>
  </tr>
  <tr>
   <td>KCNet []</td>
   <td>0.900</td>
   <td>91.0</td>
  </tr>
  <tr>
   <td>Point PN []</td>
   <td>0.800</td>
   <td>93.8</td>
  </tr>
  <tr>
   <td>Ours (PointNet)</td>
   <td>0.034</td>
   <td>87.1</td>
  </tr>
  <tr>
   <td>Ours (DGCNN)</td>
   <td>0.017</td>
   <td>86.3</td>
  </tr>
  <tr>
   <td>Ours (PointCNN)</td>
   <td>0.012</td>
   <td>85.4</td>
  </tr>
 </tbody>
</table>

Table 2: Comparative analysis of the performance on the ModelNet40 dataset between existing less parametric models and highly sparse subnetworks extracted from over-parameterized models.

## 4.1 Main Results

Sparse subnetworks: We present our results of sparse subnetworks in Figure 2. Our observations are the following: (1) We notice the existence of highly sparse subnetworks with comparable or even superior accuracy to the original dense model. This finding is significant as the permutation-invariant nature of PCNN architecture makes it difficult to exploit spatial or temporal redundancies. (2) The sparse network achieved from global one-shot or IMP pruning maintains a high accuracy across various PCNN models and datasets. Results demonstrate the remarkably high pruning rates of up to 99% for PointNet, PointCNN, and DGCNN architectures while preserving desirable accuracy levels. (3) Global pruning methods are more effective in identifying winning tickets at extreme sparsity levels than iterative pruning methods relying on local pruning criteria. Iterative local pruning methods struggle to uncover winning tickets at such a high pruning rate, suggesting that the weights (primarily responsible for the model’s accuracy) are distributed globally rather than concentrated locally. (4) Overparameterized PCNN models incorporate inherent redundancy, revealing that a small subset of weights can encode most of the learned knowledge.

Transfer Learning Effectiveness: Winning tickets can be generalized across related tasks or datasets [] possibly because the identified subnetworks preserve common features to multiple tasks []. We present our transfer results on 3D datasets in Table 1. Our observations are: (1) The winning tickets discovered through global one-shot pruning and IMP global pruning in PCNN models exhibit similar transferability across datasets. For example, the winning tickets obtained from models trained on the ShapeNetCore dataset can be successfully transferred to models trained on the ModelNet40 and ScanObjectNN datasets, achieving comparable performance to the original dense models trained on those datasets. (3) The transferability of winning tickets across datasets suggests that the identified subnetworks capture fundamental features relevant to the original and related tasks. This finding is particularly significant because it implies that a single winning ticket, once discovered, can be leveraged for efficient knowledge transfer and model adaptation across various point cloud datasets without retraining from scratch.

Comparison with less parametric PCNN models: Rather than reducing the parameters of a model by pruning, there exist PCNN models, [], [], [], [] that use fewer parameters than the conventional larger parametric PCNN models. In Table 2, we compare our pruned model’s number of parameters and results with those models. The pruned version (with extremely few parameters) performs similarly to other less-parameter models.

![](./images/1035804703717326861_2.jpg)

Figure 2: The performance of lottery tickets discovered for 3D classification tasks across three model architectures: PointNet, DGCNN, and PointCNN, which handle point cloud data differently. Task-specific Iterative Magnitude Pruning (IMP) Global pruning outperforms the overparametrized baseline (0% pruned) and IMP Local pruning by a wide margin. For efficiency, one-shot global pruning at 99% sparsity exceeds all other methods evaluated. Each task's IMP method iteratively prunes 20% weights to identify the winning tickets or subnetworks, achieving competitive or superior performance to the over parametrized models with reduced computation and parameters.

Ablation Studies: Our empirical analysis on PointNet and DGCNN for the ModelNet40 dataset, shown in Figure 3, reveals that pruning a more significant proportion of the parameters of the FC layers leads to improved performance compared to the overparameterized network. Conversely, pruning only the conv layers results in a mere 12% network sparsity at 99% pruning, accompanied by a substantial degradation in performance that underscores the importance of preserving a representation from the Conv. layers compared to the FC layers. Also, upon analyzing the IMP method, it becomes evident that the convolutional layers preserve most of the weights.

### 4.2 Discussion

The demonstrated results highlight several notable observations and considerations concerning the architecture of PCNNs. First, it is evident that highly sparse subnetworks, or winning tickets, can be obtained within PCNN architectures, even at extreme pruning rates of up to

![](./images/1035804703717326861_3.jpg)
![](./images/1035804703717326861_4.jpg)

<table>
  <thead>
    <tr>
      <th colspan="5">Ours (PointNet)</th>
    </tr>
    <tr>
      <th>Conv</th>
      <th>FC</th>
      <th>Prune(%)</th>
      <th>Sparsity(%)</th>
      <th>Acc.(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>×</td>
      <td>×</td>
      <td>0</td>
      <td>0</td>
      <td>87.5</td>
    </tr>
    <tr>
      <td>✓</td>
      <td>×</td>
      <td>99</td>
      <td>12</td>
      <td>86.9</td>
    </tr>
    <tr>
      <td>×</td>
      <td>✓</td>
      <td>99</td>
      <td>87</td>
      <td>88.3</td>
    </tr>
    <tr>
      <td>✓</td>
      <td>✓</td>
      <td>99</td>
      <td>99</td>
      <td>86.8</td>
    </tr>
  </tbody>
</table>

Figure 3: The first two figures from the left illustrate the impact of global pruning on models such as PointNet and DGCNN. Based on the weight magnitude (importance), LTH prunes some Conv. layers with a significantly lower proportion of weights than FC layers. It tells LTH to automatically identify essential weights related to geometric transformations and other structural constraints of 3D data. The table on the right shows an ablation study for pruning the Conv. and FC layers separately.

99%. This finding contrasts with traditional models and datasets, where performance often starts to degrade at lower pruning rates, potentially due to the presence of rare features for specific classes in large datasets like ImageNet [9, 24]. Notably, further analysis reveals that even after aggressive pruning at 99% rates, a significant portion of the weights in the convolutional layers of PCNN architectures remain intact. In contrast, many weights in the FC layers are pruned away. This finding suggests that the weights of the FC layers are less critical for overall performance than the Conv. layers responsible for extracting features from the point cloud data. The weights of the Conv layer are primarily essential to be preserved. Moreover, the observation that one-shot global pruning at a 99% pruning rate can achieve desirable accuracy. This implies that only 1% of the highest magnitude weights are responsible for the model performance. These high-magnitude weights are predominantly concentrated in the convolutional layers, further emphasizing the significance of these layers in extracting essential features from point cloud data. These findings prioritize the preservation of convolutional layers while potentially eliminating or significantly compressing fully connected layers to develop more efficient architectures in the future.

## 5 Conclusion

We explore the lottery ticket hypothesis in PCNNs, revealing highly sparse subnetworks or "winning tickets" that maintain accuracy even at 99% pruning rates (global pruning). Remarkably, these winning tickets display transferability across datasets, indicating that they capture fundamental point cloud features. The analysis underscores the importance of convolutional layers for feature extraction, while fully connected layers contribute significantly to model size without impacting performance. These findings lay the foundation for optimizing PCNN architectures, prioritizing convolutional layers, and achieving substantial model compression and efficient deployment, especially in resource-constrained settings. By leveraging transferable winning tickets and architectural insights, future research can drive applications such as autonomous systems and 3D computer vision advancements.

## References

[1] Zhengfa Bi, Xinming Wu, Zhaoliang Li, Dekuan Chang, and Xueshan Yong. Deepism-net: Three-dimensional implicit structural modeling with convolutional neural network. *Geoscientific Model Development*, 15(17):6841–6861, 2022.

[2] Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. What is the state of neural network pruning? *Proceedings of machine learning and systems*, 2:129–146, 2020.

[3] Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Van- dergheynst. Geometric deep learning: going beyond euclidean data. *IEEE Signal Processing Magazine*, 34(4):18–42, 2017.

[4] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. *arXiv preprint arXiv:1512.03012*, 2015.

[5] Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Michael Carbin, and Zhangyang Wang. The lottery tickets hypothesis for supervised and self- supervised pre-training in computer vision models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 16306–16316, 2021.

[6] Tianyi Chen, Bo Ji, Tianyu Ding, Biyi Fang, Guanyi Wang, Zhihui Zhu, Luming Liang, Yixin Shi, Sheng Yi, and Xiao Tu. Only train once: A one-shot neural network train- ing and pruning framework. *Advances in Neural Information Processing Systems*, 34: 19637–19651, 2021.

[7] Xiaohan Chen, Yu Cheng, Shuohang Wang, Zhe Gan, Jingjing Liu, and Zhangyang Wang. The elastic lottery ticket hypothesis. *Advances in Neural Information Processing Systems*, 34:26609–26621, 2021.

[8] Ali Cheraghian, Shafin Rahman, Dylan Campbell, and Lars Petersson. Mitigating the hubness problem for zero-shot learning of 3d objects. In *British Machine Vision Con- ference (BMVC)*, 2019.

[9] Francesco Corti, Rahim Entezari, Sara Hooker, Davide Bacciu, and Olga Saukh. Study- ing the impact of magnitude pruning on contrastive learning methods. *arXiv preprint arXiv:2207.00200*, 2022.

[10] Carlos Esteves, Christine Allen-Blanchette, Ameesh Makadia, and Kostas Daniilidis. Learning so (3) equivariant representations with spherical cnns. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 52–68, 2018.

[11] Guangchi Fang, Qingyong Hu, Hanyun Wang, Yiling Xu, and Yulan Guo. 3dac: Learn- ing attribute compression for point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14819–14828, 2022.

[12] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2018.

[13] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. Stabilizing the lottery ticket hypothesis. *arXiv preprint arXiv:1903.01611*, 2019.

[14] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In *International Conference on Machine Learning*, pages 3259–3269. PMLR, 2020.

[15] Zhe Gan, Yen-Chun Chen, Linjie Li, Tianlong Chen, Yu Cheng, Shuohang Wang, Jingjing Liu, Lijuan Wang, and Zicheng Liu. Playing lottery tickets with vision and language. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 652–660, 2022.

[16] Theodoros Georgiou, Yu Liu, Wei Chen, and Michael Lew. A survey of traditional and deep learning-based feature descriptors for high dimensional data in computer vision. *International Journal of Multimedia Information Retrieval*, 9:135–170, 2020.

[17] Sharath Girish, Shishira R Maiya, Kamal Gupta, Hao Chen, Larry S Davis, and Abhi- nav Shrivastava. The lottery ticket hypothesis for object recognition. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 762–771, 2021.

[18] Federico Girosi, Michael Jones, and Tomaso Poggio. Regularization theory and neural networks architectures. *Neural computation*, 7(2):219–269, 1995.

[19] Ankit Goyal, Hei Law, Bowei Liu, Alejandro Newell, and Jia Deng. Revisiting point cloud shape classification with a simple and effective baseline. In *International Conference on Machine Learning*, pages 3809–3820. PMLR, 2021.

[20] Souhail Hadgi, Lei Li, and Maks Ovsjanikov. To supervise or not to supervise: Un- derstanding and addressing the key challenges of 3d transfer learning. *arXiv preprint arXiv:2403.17869*, 2024.

[21] Victor Petrén Bach Hansen and Anders Søgaard. Is the lottery fair? evaluating win- ning tickets across demographics. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 3214–3224, 2021.

[22] Qisheng He and Ming Dong. Transferring lottery tickets in computer vision models: a dynamic pruning approach. In *2023 IEEE International Conference on Big Data (BigData)*, pages 1123–1128. IEEE, 2023.

[23] Michael Hillier, Florian Wellmann, Boyan Brodaric, Eric de Kemp, and Ernst Schetse- laar. Three-dimensional structural geological modeling using graph neural networks. *Mathematical geosciences*, 53(8):1725–1749, 2021.

[24] Md Ismail Hossain, Mohammed Rakib, MM Elahi, Nabeel Mohammed, and Shafin Rahman. Colt: Cyclic overlapping lottery tickets for faster pruning of convolutional neural networks. *arXiv preprint arXiv:2212.12770*, 2022.

[25] Weizhe Hua, Yuan Zhou, Christopher M De Sa, Zhiru Zhang, and G Edward Suh. Channel gating neural networks. *Advances in Neural Information Processing Systems*, 32, 2019.

[26] Yaomin Huang, Ning Liu, Zhengping Che, Zhiyuan Xu, Chaomin Shen, Yaxin Peng, Guixu Zhang, Xinmei Liu, Feifei Feng, and Jian Tang. Cp3: Channel pruning plug-in for point-based networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5302–5312, 2023.

[27] Eugenia Iofinova, Alexandra Peste, Mark Kurtz, and Dan Alistarh. How well do sparse imagenet models transfer? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12266–12276, 2022.

[28] Ajay Kumar Jaiswal, Shiwei Liu, Tianlong Chen, Ying Ding, and Zhangyang Wang. Instant soup: Cheap pruning ensembles in a single pass can draw lottery tickets from large models. In *International Conference on Machine Learning*, pages 14691–14701. PMLR, 2023.

[29] Seohyun Kim, Jaeyoo Park, and Bohyung Han. Rotation-invariant local-to-global representation learning for 3d point cloud. *Advances in Neural Information Processing Systems*, 33:8174–8185, 2020.

[30] Yejin Lee, Donghyun Lee, JungUk Hong, Jae W Lee, and Hongil Yoon. Not all neighbors matter: point distribution-aware pruning for 3d point cloud. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pages 1240–1249, 2023.

[31] Jiale Li, Hang Dai, Hao Han, and Yong Ding. Mseg3d: Multi-modal 3d semantic segmentation for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 21694–21704, 2023.

[32] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. *Advances in neural information processing systems*, 31, 2018.

[33] Zhu Liao, Victor Quétu, Van-Tam Nguyen, and Enzo Tartaglione. Can unstructured pruning reduce the depth in deep neural networks? In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1402–1406, 2023.

[34] Lixing Liu, Xu Wang, Xin Yang, Hongjie Liu, Jianping Li, and Pengfei Wang. Path planning techniques for mobile robots: Review and prospect. *Expert Systems with Applications*, page 120254, 2023.

[35] Yongcheng Liu, Bin Fan, Gaofeng Meng, Jiwen Lu, Shiming Xiang, and Chunhong Pan. Densepoint: Learning densely contextual representation for efficient point cloud processing. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 5239–5248, 2019.

[36] Eran Malach, Gilad Yehudai, Shai Shalev-Schwartz, and Ohad Shamir. Proving the lottery ticket hypothesis: Pruning is all you need. In *International Conference on Machine Learning*, pages 6682–6691. PMLR, 2020.

[37] Albert Matveev, Ruslan Rakhimov, Alexey Artemov, Gleb Bobrovskikh, Vage Egiazarian, Emil Bogomolov, Daniele Panozzo, Denis Zorin, and Evgeny Burnaev. Def: Deep estimation of sharp geometric features in 3d shapes. *ACM Transactions on Graphics*, 41(4), 2022.

[38] Jian Meng, Li Yang, Jinwoo Shin, Deliang Fan, and Jae-sun Seo. Contrastive dual gating: Learning sparse features with contrastive learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12257–12265, 2022.

[39] Niloy J Mitra, Michael Wand, Hao Zhang, Daniel Cohen-Or, Vladimir Kim, and Qi-Xing Huang. Structure-aware shape processing. In *ACM SIGGRAPH 2014 Courses*, pages 1–21. 2014.

[40] Hanlin Mo and Guoying Zhao. Ric-cnn: Rotation-invariant coordinate convolutional neural network. *Pattern Recognition*, 146:109994, 2024.

[41] Ari Morcos, Haonan Yu, Michela Paganini, and Yuandong Tian. One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. *Advances in neural information processing systems*, 32, 2019.

[42] Dat Thanh Nguyen and André Kaup. Lossless point cloud geometry and attribute compression using a learned conditional probability model. *IEEE Transactions on Circuits and Systems for Video Technology*, 33(8):4337–4348, 2023.

[43] Endrit Pajaziti, Javier Montalt-Tordera, Claudio Capelli, Raphaël Sivera, Emilie Sauvage, Michael Quail, Silvia Schievano, and Vivek Muthurangu. Shape-driven deep neural networks for fast acquisition of aortic 3d pressure and velocity flow fields. *PLoS Computational Biology*, 19(4):e1011055, 2023.

[44] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learn- ing on point sets for 3d classification and segmentation. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 652–660, 2017.

[45] Gernot Riegler, Ali Osman Ulusoy, and Andreas Geiger. Octnet: Learning deep 3d representations at high resolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3577–3586, 2017.

[46] Hitesh Sapkota, Dingrong Wang, Zhiqiang Tao, and Qi Yu. Distributionally robust ensemble of lottery tickets towards calibrated sparse network training. *Advances in Neural Information Processing Systems*, 36, 2024.

[47] Yiru Shen, Chen Feng, Yaoqing Yang, and Dong Tian. Mining point cloud local struc- tures by kernel correlation and graph pooling. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4548–4557, 2018.

[48] Shivanand Venkanna Sheshappanavar and Chandra Kambhamettu. Dynamic local ge- ometry capture in 3d point cloud classification. In *2021 IEEE 4th international con- ference on multimedia information processing and retrieval (MIPR)*, pages 158–164. IEEE, 2021.

[49] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi- view convolutional neural networks for 3d shape recognition. In *Proceedings of the IEEE international conference on computer vision*, pages 945–953, 2015.

[50] Bugra Tekin, Isinsu Katircioglu, Mathieu Salzmann, Vincent Lepetit, and Pascal Fua. Structured prediction of 3d human pose with deep neural networks. *arXiv preprint arXiv:1605.05180*, 2016.

[51] Bipin Kumar Tripathi and Prem Kumar Kalra. High dimensional neural networks and applications. In *Intelligent autonomous systems: foundations and applications*, pages 215–233. Springer, 2010.

[52] Julie Urlings, Guido de Jong, Thomas Maal, and Dylan Henssen. Views on augmented reality, virtual reality, and 3d printing in modern medicine and education: A qualitative exploration of expert opinion. *Journal of Digital Imaging*, 36(4):1930–1939, 2023.

[53] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In *International Conference on Computer Vi-sion (ICCV)*, 2019.

[54] Prasoon Kumar Vinodkumar, Dogus Karabulut, Egils Avots, Cagri Ozcinar, and Gho-lamreza Anbarjafari. A survey on deep learning based segmentation, detection and classification for 3d point clouds. *Entropy*, 25(4):635, 2023.

[55] Jianqiang Wang, Hao Zhu, Haojie Liu, and Zhan Ma. Lossy point cloud geometry compression via end-to-end learning. *IEEE Transactions on Circuits and Systems for Video Technology*, 31(12):4909–4923, 2021.

[56] Peng-Shuai Wang, Yang Liu, Yu-Xiao Guo, Chun-Yu Sun, and Xin Tong. O-cnn: Octree-based convolutional neural networks for 3d shape analysis. *ACM Transactions On Graphics (TOG)*, 36(4):1–11, 2017.

[57] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. *ACM Trans-actions on Graphics (tog)*, 38(5):1–12, 2019.

[58] Martin Weinmann, Boris Jutzi, and Clément Mallet. Geometric features and their rele-vance for 3d point cloud classification. *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, 4:157–164, 2017.

[59] Wenxuan Wu, Zhongang Qi, and Li Fuxin. Pointconv: Deep convolutional networks on 3d point clouds. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pages 9621–9630, 2019.

[60] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1912–1920, 2015.

[61] Jiwei Yang, Xu Shen, Jun Xing, Xinmei Tian, Houqiang Li, Bing Deng, Jianqiang Huang, and Xian-sheng Hua. Quantization networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 7308–7316, 2019.

[62] Fang Yu, Li Cui, Pengcheng Wang, Chuanqi Han, Ruoran Huang, and Xi Huang. Eas-iedge: A novel global deep neural networks pruning method for efficient edge comput-ing. *IEEE Internet of Things Journal*, 8(3):1259–1271, 2020.

[63] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In *Pro-ceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8552–8562, 2022.

[64] Renrui Zhang, Liuhui Wang, Ziyu Guo, Yali Wang, Peng Gao, Hongsheng Li, and Jianbo Shi. Parameter is not all you need: Starting from non-parametric networks for 3d point cloud analysis. *arXiv preprint arXiv:2303.08134*, 2023.

[65] Yihua Zhang, Yuguang Yao, Parikshit Ram, Pu Zhao, Tianlong Chen, Mingyi Hong, Yanzhi Wang, and Sijia Liu. Advancing model pruning via bi-level optimization. *Ad- vances in Neural Information Processing Systems*, 35:18309–18326, 2022.

[66] Hattie Zhou, Janice Lan, Rosanne Liu, and Jason Yosinski. Deconstructing lottery tickets: Zeros, signs, and the supermask. *Advances in neural information processing systems*, 32, 2019.

[67] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xin- long Wang. Uni3d: Exploring unified 3d representation at scale. *arXiv preprint arXiv:2310.06773*, 2023.

[68] Sheng Zhou, Yucheng Wang, Defang Chen, Jiawei Chen, Xin Wang, Can Wang, and Jiajun Bu. Distilling holistic knowledge with graph neural networks. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10387–10396, 2021.

[69] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In *Proceedings of the IEEE/CVF confer- ence on computer vision and pattern recognition*, pages 5745–5753, 2019.

[70] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4490–4499, 2018.