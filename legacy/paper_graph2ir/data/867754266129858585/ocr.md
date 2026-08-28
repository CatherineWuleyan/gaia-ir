# Class-dependent Pruning of Deep Neural Networks

Rahim Entezari
Institute of Technical Informatics, TU Graz
CSH Vienna, Austria
entezari@tugraz.at

Olga Saukh
Institute of Technical Informatics, TU Graz
CSH Vienna, Austria
saukh@tugraz.at

Abstract—Today's deep neural networks require substantial computation resources for their training, storage and inference, which limits their effective use on resource-constrained devices. Many recent research activities explore different options for compressing and optimizing deep models. On the one hand, in many real-world applications we face the data imbalance challenge, i.e., when the number of labeled instances of one class considerably outweighs the number of labeled instances of the other class. On the other hand, applications may pose a class imbalance problem, i.e., higher number of false positives produced when training a model and optimizing its performance may be tolerable, yet the number of false negatives must stay low. The problem originates from the fact that some classes are more important for the application than others, e.g., detection problems in medical and surveillance domains. Motivated by the success of the lottery ticket hypothesis, in this paper we propose an iterative deep model compression technique, which keeps the number of false negatives of the compressed model close to the one of the original model at the price of increasing the number of false positives if necessary. Our experimental evaluation using two benchmark data sets shows that the resulting compressed sub-networks 1) achieve up to 35% lower number of false negatives than the compressed model without class optimization, 2) provide an overall higher AUC-ROC measure compared to conventional Lottery Ticket algorithm and three recent popular pruning methods , and 3) use up to 99% fewer parameters compared to the original network. The code is publicly available¹.

Index Terms—deep neural network compression, pruning, lottery ticket hypothesis, data imbalance, class imbalance

## I. INTRODUCTION

While deep networks are a highly successful model class, their large memory footprint puts considerable strain on energy consumption, communication bandwidth and storage requirements of the underlying hardware, and hinders their usage on resource-con-strained IoT devices. For example, VGG-16 models for object detection [1] and facial attribute classification [2] both contain over 130M parameters. Recent efforts on deep model compression for embedded devices explore several directions including quantization, factorization, pruning, knowledge distillation, and efficient network design. The approach presented in this paper combines network pruning with efficient network design by additionally including class-dependency into the network compression algorithm.

This is particularly useful in applications that can tolerate a slight increase in the number of false positives (FP), yet need to keep the number of false negatives (FN) close to the original model when pruning the weights. Many real-world applications, e.g., medical image classification and anomaly detection in production processes, have to deal with both data imbalance and class imbalance when training a deep model. On the one hand, real-world data often follows a long-tailed data distribution in which a few classes account for the most of the data, while many classes have considerably fewer samples [3]. Models trained on these data sets are biased toward dominant classes [4]. Related literature treats data imbalance as a problem which leads to low model quality [5]. The proposed solutions typically adopt class re-balancing strategies such as re-sampling [6] and re-weighting [7] based on the number of observations in each class. On the other hand, there are many TinyML applications, which have unbalanced class importance: e.g., missing an event may have far more severe consequences than triggering a false alarm. This is especially the case in many detection scenarios and early warning systems in the IoT domain. Our method automatically shrinks a trained deep neural network for mobile devices integration. In this paper, we focus on keeping the number of FN low when compressing a deep network.

To the best of our knowledge, this is the first paper proposing a class-dependent model compression. We provide an end-to-end network compression method based on the recently proposed iterative lottery ticket (LT) algorithm [8] for finding efficient smaller subnetworks in the original network. Since data imbalance is a common problem model designers have to deal with, in the first step, we extend the original LT algorithm with a parameterized loss function to fight data imbalance. Related literature suggests that a direct compensation for data imbalance into the loss function outperforms alternative methods [4]. In the second step, we control the number of false negatives and false positives of the model by including a parameterized AUC-ROC measure (see Sec. III) into the model compression task. We observe that it is beneficial to train the first epoch for balanced classes to learn the class boundaries. In the later epochs, however we focus on class-imbalanced optimization with the goal to keep the number of FN at the level of the original model. We evaluate the new method on two public data sets and show that it achieves up to 35% lower number of FN than the original LT algorithm without class-imbalanced training while preserving only 1% of the weights. Surprisingly, our method with a novel loss function consistently outperforms the original version of the LT algorithm in all tested cases.

¹https://github.com/rahimentezari/Sensys-ml2020

![](./images/867754266129858585_1.jpg)

Fig. 1: Iterative network pruning with FN optimization.

## II. CLASS-DEPENDENT NETWORK COMPRESSION

Our proposed method consists of the network compression pipeline presented in Fig. 1. Training data is used to first train a class-balanced model, while further epochs prefer minimizing FN over FP. Our optimization function presents a combination of two loss functions in order to 1) control data imbalance, and 2) control class imbalance. We adopt a parameterized cross-entropy loss [4] to achieve the former, and use the hinge ranking loss [9] to maximize AUC-ROC and to control the trade-off between FN and FP to address the latter. The iterative pruning procedure is based on the LT algorithm. The paragraphs below describe the individual building blocks of our solution.

Lottery Ticket (LT) algorithm. Our class-dependent network compression method leverages the recently introduced iterative pruning used to search for efficient sparse sub-networks called lottery tickets (LT) [8] within an original deep network. LT networks often show superior performance when compared to the original network. The recently conducted analysis of LT networks [10] suggests that the sub-network structure tightly coupled with network initialization is crucial to achieve high performance of LT networks. Moreover, the following conditions are responsible for the best results: 1) setting pruned values to zero rather than to any other value, 2) keeping the sign of the initialization weights when rewinding, and 3) keeping the weights with the largest absolute values or apply the magnitude increase mask criterion during iterative pruning. We leverage all these findings in this work. We use the magnitude increase criterion throughout the paper, i.e., we rank the differences between the final and the initial values of the network weights in every round and prune the least $p$ percent.

```
Algorithm 1 Class-dependent network compression
 1: Randomly initialize the network $f(x ; m \bigodot W_{0})$ with the
    initially trivial pruning mask $m=1^{|W_{0}|}$ ;
 2: Train the network for $n$ iterations with the class-dependent
    loss $L$ producing the network $f(x ; m \bigodot W_{k})$ ;
 3: Prune $p \%$ of the remaining weights with the magnitude
    increase strategy, i.e., $m[i]:=0$ if $W_{k}[i]$ gets pruned;
 4: Replace remaining weights $W_{k}$ with their initial values
    $W_{0}$ ;
 5: Go to step 2 if the next $(k+1)$ -th round is required.
```

Alg. 1 provides a pseudo-code of the LT algorithm with the magnitude increase mask criterion and a class-dependent loss function $\mathcal{L}$ explained below. The algorithm initializes the network with random weights $W_{0}$ and applies an initially trivial pruning mask $m=1^{|W_{0}|}$. The operation $\bigodot$ denotes an element-wise multiplication. After training the network for $n$ iterations we prune $p$ percent of the weights using the magnitude increase strategy by updating the mask $m$ accordingly. The remaining weights $W_{k}$ are then reset to their initial values $W_{0}$ before the next algorithm round starts.

In every round of the algorithm we minimize the loss function $\mathcal{L}$ of the following form

$$
\mathcal{L}=\mathcal{L}_{w C E}+\mathcal{L}_{S H R},\tag{1}
$$

where $\mathcal{L}_{w C E}$ and $\mathcal{L}_{S H R}$ are the weighted binary cross-entropy loss and the squared hinge ranking loss respectively detailed below.

Weighted binary cross-entropy loss. Inspired by [4], we extend the notion of the classical binary cross-entropy to compensate for the data imbalance by introducing per-class weighting coefficients as follows

$$
\mathcal{L}_{w C E}=-\sum_{c=1}^{M} \gamma_{c} \cdot y_{o, c} \log \left(p_{o, c}\right),\tag{2}
$$

where $\gamma_{c}$ are weighting coefficients for every class; $M$ is the number of classes; $y_{o, c} \in\{0,1\}$ is a binary indicator if the class label $c$ is a correct classification of the observation $o$; $p_{o, c}$ is a predicted probability that the observation $o$ is of class $c$, and $n_{c}$ is the number of samples in $c$.

We leverage the results in [7], [11] and handle the weighting coefficients $\gamma_{c}$ for individual classes as $\gamma_{c}=\frac{1-\beta}{1-\beta^{n_{c}}}$, where $\beta \in[0 ; 1)$ is a hyperparameter. In contrast to their work, we choose the value of the hyperparameter $\beta$ to compensate the data imbalance in favor of a particular class. The setting $\beta=0$ corresponds to no weighting and $\beta \to 1$ corresponds to weighting by inverse data frequency for a given class. Recent work by [4] shows that the weighting coefficients $\gamma_{c}$ play an important role in data-balancing. In particular, when training a CNN on imbalanced data, data-balancing for each class, by means of $\gamma_{c}$, provides a significant boost to the performance of the commonly used loss functions, including cross-entropy.

Squared hinge ranking loss. The previous literature shows that 1) optimizing classification accuracy by minimizing cross-entropy cannot guarantee maximization of AUC-ROC [12], and 2) AUC-ROC maximization as an optimization task yields a discontinuous non-convex objective function and thus cannot be approached by the gradient based methods [13]. Proposed solutions for AUC-ROC maximization [14] are based on approximations. In this paper, we use the squared hinge ranking loss suggested in [9], while adding the parameters $\lambda_{c}$ to control the class imbalance.

$$
\mathcal{L}_{S H R}=-\sum_{c=1}^{M} \lambda_{c} \max \left(1-y_{o, c} r_{o, c}, 0\right)^{2}.\tag{3}
$$

The squared hinge ranking loss is obtained from the hinge loss by replacing $p_{o, c}$ by a sorted in ascending order classifier output $r_{o, c}$.

![](./images/867754266129858585_2.jpg)

Fig. 2: Images from the ISIC data set [15]: benign (left) and melanoma (right) samples.

![](./images/867754266129858585_3.jpg)

Fig. 3: Images from the CRACK data set [16]: negative (left) and positive (right) samples.

The authors of [9] show that AUC-ROC can be written in terms of the hinge rank loss as follows

$$
\text{AUC-ROC} \geq 1 - \frac{L_{SHR} - C}{n^+n^-}, \tag{4}
$$

where $n^+, n^-$ are the number of positive and negatives samples and $C$ is a constant independent of the rank order. Minimizing hinge ranking loss leads to AUC-ROC maximization [9]. We use the squared hinge ranking loss $\mathcal{L}_{SHR}$ to ensure our loss function $\mathcal{L}$ is differentiable.

In the next section we evaluate the effectiveness of the proposed class-dependent network compression algorithm on two benchmark data sets.

### III. EXPERIMENTAL EVALUATION

This section introduces the benchmark data sets, lists the metrics we use to evaluate the performance of our method, justifies parameter choices, and presents the results.

#### A. Data Sets

We chose our benchmark data sets based on the complexity of the classification task they present. ISIC is a challenging medical imaging lesion classification data set introduced in a data science competition². The CRACK data set enjoys low classification complexity. Both benchmarks are introduced below.

²http://challenge2016.isic-archive.com

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Description & Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Saturation</td>
      <td>Modify saturation by 0.3</td>
    </tr>
    <tr>
      <td>Contrast</td>
      <td>Modify contrast by 0.3</td>
    </tr>
    <tr>
      <td>Brightness</td>
      <td>Modify brightness by 0.3</td>
    </tr>
    <tr>
      <td>Hue</td>
      <td>Shift the hue by 0.1</td>
    </tr>
    <tr>
      <td>Flip</td>
      <td>Randomly flip horizontally and vertically</td>
    </tr>
    <tr>
      <td>Affine</td>
      <td>Rotate by 90°, shear by 20°, scale by [0.8, 1.2]</td>
    </tr>
    <tr>
      <td>Crop</td>
      <td>Randomly crop (&gt;40% area) the image</td>
    </tr>
    <tr>
      <td>Elastic</td>
      <td>Warp images with thin plate splines</td>
    </tr>
  </tbody>
</table>

TABLE I: ISIC-2016 data augmentation [17].

ISIC-2016 lesion classification dataset [15] includes original medical images paired with a confirmed malignancy diagnosis labels obtained from expert consensus and pathology report information, *i.e.*, each image is assigned a label either benign or melanoma. The training data set contains 900 dermoscopic lesion images with 173 positive and 727 negative examples respectively, whereas the test set includes 379 images with 76 positive and 303 negative samples respectively. Fig. 2 shows positive and negative sample images from the ISIC data set.

We leverage best-practice data augmentation techniques suggested in [17]and shown in Table I. This includes modification to image saturation, contrast, and brightness by 0.3, shifting the hue by 0.1, randomly flipping an image, Affine transformation (rotation by 90°, shear by 20°, scale by [0.8, 1.2]), randomly cropping an image (>40% area). Saturation, contrast and brightness augmentations simulate changes in color due to camera settings and lesion characteristics. Affine transformations reproduce camera distortions and create new lesion shapes. Elastic warp is generated by defining the origins as an evenly-spaced 4×4 grid of points, and destinations as random points around the origins (by up to 10% of the image width in each direction). These augmentations produce a 10-fold training data increase compared to the original dataset, while maintaining medical attributes [17].

CRACK data set [16] contains 40K images of 224×224 pixels sliced from 500 full resolution images of 4032×3024 pixels taken from the walls and floors of several concrete buildings. The images are taken approximately 1 m away from the surfaces with the camera directly facing the target. The concrete surfaces have variation in terms of surface finishes (exposed, plastering and paint). The label is positive if an image contains a crack and negative otherwise. The labels are assigned by the material science experts. Fig. 3 shows positive and negative samples in this data set.

#### B. Evaluation Metrics

We adopt the following evaluation metrics: AUC-ROC, accuracy, False Negative Rate (FNR), and False Positive Rate (FPR) shortly summarized below.

**AUC-ROC** measure estimates the probability that a randomly chosen sample of a positive class has a smaller estimated probability of belonging to a negative class than a randomly chosen member of a negative class [9]. :

$$
\text{AUC-ROC} = \frac{1}{n^+n^-} \sum_{i=1}^{n^+} \sum_{j=1}^{n^-} \mathbb{I}(r_i^+ > r_j^-), \tag{5}
$$

where $n^+,n^-$ are the number of positive and negatives samples and $\mathbb{I}$ is the indicator function. $r_i^+ \in 1,...,n^+$ denotes the rank of positive examples and $r_j^- \in 1,...,n^-$ denotes the rank of negative examples.

We use the FNR measure to show that the proposed method indeed decreases the number of false negatives. We also report the FPR measure to understand the achieved trade-offs between the false positive and negative rates. AUC-ROC measures the area underneath the entire ROC curve. An ROC curve plots TPR vs. FPR at different classification thresholds.

The results in [12] show that the expected value of AUC-ROC over all classifications is a monotonic function of accuracy. This also holds for imbalanced data. We report accuracy to ensure the overall compressed network performance stays high.

### C. Experimental Setup
We enforce data imbalance in the CRACK data set by using 20 K images in the negative class (no crack), and 4 K images of the positive class, and use 70 %, 15 %, 15 % of samples for train, validation and test, respectively.
Networks. For classification on the ISIC-2016 and CRACK data sets, we adopt AlexNet [18] pre-trained on ImageNet [19] with an adjusted number of fully-connected layers to contain 256, 8, and 2 neurons. This network has 2'471'842 parameters. Since our compression method uses iterative pruning based on the LT algorithm, we use these relatively shallow networks to keep the computation manageable on the available computational resources. The technical bottleneck here is the turn-off of the gradient in the backward pass in order to keep the pruned values set to zero. Given a stronger hardware infrastructure our method can be used on deeper networks such as VGG and ResNet.
Hyperparameters. To evaluate the performance of the proposed method, we follow our two-step design. First we focus on balancing the imbalanced data using the $\gamma$ parameter, then we use $\lambda$ multipliers for ranking loss to optimize AUC-ROC. Leveraging the results reported in [4], [11] to achieve data balancing by setting the loss function parameters according to the inverse class frequencies, we set $\beta$ close to 1 in our tests. For ISIC and CRACK data sets $\gamma_c = 727/173 = 4.2$ and $\gamma_c=14K/2.8K=5$ respectively. For simplicity, we use $\gamma_c = 5$ in both cases. This corresponds to $\beta=0.99997$ for both data sets. We test $\lambda_c \in \{0,1,2,10\}$, where $\lambda_c=0$ stands for the standard LT algorithm, and $\lambda_c=5$ performs best in all other scenarios with non-uniform weights for positive and negative classes.

For each round of Alg. 1 in step 2 we train the network for $k=100$ iterations. With stronger hardware, it is possible to train the network longer to achieve potentially better results. Since our procedure trains an already pre-trained AlexNet on ImageNet dataset, the accuracy difference between $k=1000$ and $k=100$ iterations is less than 3%. By following the magnitude increase pruning strategy we prune $p=50\%$ of the remaining weights in every round. This yields compressed networks with $|W_k| = 100\%$, 50%, 25%, 12.5%, 6.25%.
3.12%, and 1.57% remaining weights. We use the stochastic gradient descent (SGD) with a momentum setting of 0.9 for network training.
Scenarios. We test our method in three blue, black, and green scenarios, along with the LT algorithm as a baseline, and compare the obtained performance to three recent more popular benchmarks. These scenarios correspond to differently colored lines in the plots:
red Corresponds to the original LT algorithm with magnitude increase pruning criterion. The weights for both positive and negative classes are set to $\gamma_c=1$ and we use no ranking loss with $\lambda_c=0$. Thus, use the best performance of the classical LT algorithm as a benchmark.
blue In this scenario, we test the effect of our ranking loss (class balancing) compared to the standard LT algorithm (red), without weighting cross entropy loss (data balancing). Therefore, we have $\gamma_c=1$ and $\lambda_c=5$.
black In this scenario, we test the effect of weighting cross entropy loss (data balancing). Therefore, compared to the previous scenario (blue), we have $\gamma_c=5$ along with $\lambda_c=5$. We find that starting the first round with $\gamma_1=1$ helps the network to initially find the boundaries between the two classes without any specific focus
green In this scenario we test for weighting higher values for data balancing, so we set $\gamma_c=10$ everywhere, while $\lambda_c=5$.
LOBS [20] stands for the Layer-Wise Optimal Brain Surgeon algorithm. This pruning method determines the importance of neurons from the values in the corresponding Hessian matrix and prunes the least important ones in every layer.
SNIP is a single-shot network pruning algorithm [21]. It prunes the filters based on a connection sensitivity criteria. Here we prune the network without re-training.
SNIP with training means a trained network is pruned by SNIP and then re-trained.
MobileNet [22] replaces the normal convolution by the depth-wise convolution followed by the point-wise convolution, which is called a depth-wise separable convolution. This reduces the total number of the floating point multiplication operations and hence significantly reduces the number of parameters.

### D. Results
Fig. 4 and Fig. 5 show the evaluation results for ISIC and CRACK data sets. We compare our results to the best performance obtained with the classical LT algorithm with the magnitude increase pruning criterion along with three other recent benchmarks. Our goal is to prune the network while maximizing the AUC-ROC measure and the classification accuracy, yet keep the number of false negatives for a specific class as low as possible.

As can be seen in Fig. 4(a) the best AUC-ROC for ISIC (black line) is achieved when $\lambda_c=5$, $\gamma_c=1$ for the first training epoch ($\gamma_1=1$) and $\gamma_c=5$ for later epochs ($\gamma_{2,n}=5$) , where $\gamma_c=5$ gives the inverse class frequency for this

![](./images/867754266129858585_4.jpg)

Fig. 4: Evaluation results on ISIC-2016. Our method (black line) outperforms the LT algorithm (red line) in AUC-ROC, accuracy, FNR, and FPR for up to 1% of remaining weights in the pruned network. Our method also outperforms LOBS, SNIP, and MobileNet in AUC-ROC and FPR.

![](./images/867754266129858585_5.jpg)

Fig. 5: Evaluation results on the CRACK data set. Our method (black line) outperforms the LT algorithm (red line) in AUC-ROC, accuracy, FNR, and FPR for up to 12% of the remaining weights. We also outperform LOBS, SNIP and MobileNet in AUC-ROC and accuracy.

On the one hand, the presented results in [12] show that the expected value of the AUC-ROC over all classifications is a monotonic function of accuracy, when we have an imbalanced dataset. On the other hand, authors in [4] argue that the data- balancing term $\gamma$ improves the performance of the cross entropy loss in terms of accuracy. Our results in Fig. 4(b) confirm these findings: the accuracy for the best settings of our method (black line) is consistently better than the accuracy of the LT algorithm (red line). Although the accuracy for LOBS, and SNIP remains high through compression, this is due to the fact that they only correctly classify all the samples in the positive class (see high FPR in Fig. 4(d)).

Fig. 4(c) shows that adding ranking loss to the standard LT algorithm as a proxy for class-balance improves the FNR (blue line). However, weighting the cross entropy for data- balance improves the FNR even stronger (black line). The best FNR is achieved when we use a higher weight for the positive class by setting $\gamma_c=10$ in all pruning rounds (green line). The Standard LT algorithm and the green method yield FNRs of 0.13 and 0.09, which shows 35% improvement over LT. Comparing Fig. 4(d) with Fig. 4(c) shows a trade-off between FPR and FNR. As we can see here, we have the lowest FPR, when we have no class-balance and no data-balance. The FPR for LOBS and SNIP without training (after pruning) are also very high (close to 1), meaning they incorrectly classify all the nagative samples as positive.

Our best setting (black line) in the first iteration (without compression) beats the best AUC-ROC and accuracy for the ISIC challenge: Our AUC-ROC and accuracy are 0.8069 and 0.8608 respectively, which is superior to AUC-ROC =0.804 and accuracy=0.855 reported by the authors in [23]. Their proposed very deep architecture is highly dependent on the segmentation results, whereas our method is an end-to-end algorithm.

For the CRACK data set, Fig. 5(a) shows the results for AUC-ROC. The best AUC-ROC is achieved when using the following set of parameters: $\gamma_1=1$, $\gamma_{2,n}=5$, $\lambda_c=5$ (black line), where $\gamma_c=5$ gives the inverse class frequency for this data set. Similar to ISIC, our method outperforms the standard LT, and the popular benchmarks LOBS, SNIP, and MobileNet in terms of AUC-ROC. As can be seen in Fig. 5(b) the accuracy for our method in the best setting (black line), which is not only better than the best accuracy achieved by the LT method (red line), but also LOBS, SNIP and MibileNet. Apart from LOBS and SNIP without training (after pruning) which have FPR close to 1, the best achieved FNR in Fig. 5(c) is when we set $\gamma_c=10$ for all pruning rounds (green line). FNR for the LT algorithm and the green line yield FNR of 0.017 and 0.011, which shows 35% improvement. However, due to the natural trade-off between FNR and FPR, giving more weight to the positive class leads to a higher FPR.

data set. The setting $\gamma_1=1$ highlights the importance of learning the class boundaries in the first iteration by using balanced training. The focus on the positive class therefore starts from the second iteration where we use $\gamma_{2,n}=5$ for the desired positive class. This setting not only outperforms the standard lottery ticket in AUC-ROC, but also recent popular benchmarks, i.e., LOBS, SNIP, and MobileNet.

## IV. RELATED WORK

Deep networks are known to be highly redundant. This motivated many researchers to seek for network compression

techniques and efficient subnetworks. This section summarizes recent efforts.

Quantization and binarization rely on weights with discrete values. [24] propose an algorithm, which approximates the posterior of the neural network weights, yet the weights can be restricted to have discrete or binary values. [25] apply approx- imations to standard CNNs. Their Binary-Weight-Network approximates the filters with binary values and reduces the size of example networks by the factor of 32x.

Decomposition and factorization explore low-rank basis of filters to reduce model size. [26] represent the learnt full- rank bank of a CNN as a combination of rank-1 filters leading to a 4.5x speed-up. More recent methods [27] rely on depth-wise and point-wise separable convolutions to reduce computational complexity. Depth-wise convolution performs light-weight filtering by applying a single convolutional kernel per input channel. Point-wise convolution expands the feature map along channels by learning linear combinations of the input channels.

Pruning covers a set of methods which reduce the model size by removing network connections. These methods date back to the optimal brain damage [28], where the authors suggest to prune the weights based on the Hessians of the loss function. [29] propose to prune the channels in CNNs based on their corresponding filter weight norm, while [30] uses the average percentage of zeros in the output to prune unimportant channels. The LT hypothesis [8] proposes iterative pruning to remove the weights with small magnitude. The methods often yields efficient sparse sub-networks.

Knowledge distillation covers methods which transfer knowl- edge from a larger teacher to a smaller student. [31] exploit adversarial setting to train a student network. The discrimina- tor tries to distinguish between the student and the teacher. They use the L2 loss to force the student to mimic the output of the teacher. [32] apply knowledge distillation to GANs to produce a compressed generator without neither loss of quality nor generalization. They hypothesized that there exists a fundamental compression limit of GANs similar to Shannon's compression theory.

Our network compression method combines network prun- ing with efficient network design. Unlike other compression methods, which try to minimize the overall error rate, our method additionally optimizes AUC-ROC while focusing on a desired class which appears to be useful in a number of real applications in the IoT domain.

# REFERENCES

[1] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," International Conference on Learning Representations (ICLR), 2015, 2015.

[2] Y. Lu, A. Kumar, S. Zhai, and et. al., "Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification," in CVPR, 2017, pp. 5334-5343.

[3] D. Hasenfratz, O. Saukh, C. Walser, C. Hueglin, M. Fierz, and L. Thiele, "Pushing the spatio-temporal resolution limit of urban air pollution maps," in Proceedings of the IEEE Conference on Pervasive Computing and Communications (PerCom), 2014, pp. 69-77.

[4] Y. Cui, M. Jia, T.-Y. Lin, and et. al., "Class-balanced loss based on effective number of samples," in CVPR, 2019, pp. 9268-9277.

[5] N. V. Chawla, K. W. Bowyer, and et. al., "SMOTE: synthetic minority over-sampling technique," AI research, vol. 16, pp. 321-357, 2002.

[6] M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," Neural Networks, vol. 106, pp. 249-259, 2018.

[7] Y.-X. Wang, D. Ramanan, and M. Hebert, "Learning to model the tail," in NIPS, 2017, pp. 7029-7039.

[8] J. Frankle and M. Carbin, "The lottery ticket hypothesis: Finding sparse, trainable neural networks," In International Conference on Learning Representations (ICLR), 2019.

[9] H. Steck, "Hinge rank loss and the area under the roc curve," in ECML,2007, pp. 347-358.

[10] H. Zhou, J. Lan, R. Liu, and J. Yosinski, "Deconstructing lottery tickets: Zeros, signs, and the supermask," arXiv preprint arXiv:1905.01067,2019.

[11] C. Huang, Y. Li, C. Change Loy, and X. Tang, "Learning deep represen- tation for imbalanced classification," in CVPR, 2016, pp. 5375-5384.

[12] C. Cortes and M. Mohri, "AUC optimization vs. error rate minimiza- tion," in NIPS, 2004, pp. 313-320.

[13] L. Yan, R. H. Dodier, M. Mozer, and R. H. Wolniewicz, "Optimizing classifier performance via an approximation to the wilcoxon-mann- whitney statistic," in ICML, 2003, pp. 848-855.

[14] S. Gultekin, A. Saha, A. Ratnaparkhi, and J. Paisley, "Mba: Mini-batch auc optimization," arXiv preprint arXiv:1805.11221, 2018.

[15] D. Gutman, N. C. Codella, E. Celebi, and et. al., "Skin lesion analysis toward melanoma detection," arXiv preprint arXiv:1605.01397, 2016.

[16] C. Ozgenel, "Concrete crack images for classification," Mendeley Data, vl, 2017.

[17] F. Perez, C. Vasconcelos, S. Avila, and E. Valle, "Data augmentation for skin lesion analysis," in OR 2.0, 2018, pp. 303-311.

[18] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in NIPS, 2012, pp. 1097-1105.

[19] J. Deng, W. Dong, R. Socher, and et. al., "ImageNet: A large-scale hierarchical image database," in CVPR, 2009, pp. 248-255.

[20] X. Dong, S. Chen, and S. Pan, "Learning to prune deep neural networks via layer-wise optimal brain surgeon," in Advances in Neural Informa- tion Processing Systems, 2017, pp. 4857-4867.

[21] N. Lee, T. Ajanthan, and P. H. Torr, "Snip: Single-shot network pruning based on connection sensitivity," In International Conference on Learning Representations (ICLR), 2019, 2018.

[22] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, and et. al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," CoRR, abs/1704.04861, 2018.

[23] L. Yu, H. Chen, Q. Dou, and et. al., "Automated melanoma recognition in dermoscopy images via very deep residual networks," IEEE Trans. on Medical Imaging, vol. 36, no. 4, pp. 994-1004, 2016.

[24] D. Soudry, I. Hubara, and R. Meir, "Expectation backpropagation: Parameter-free training of multilayer neural networks with continuous or discrete weights," in NIPS, 2014, pp. 963-971.

[25] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, "Xnor-net: Imagenet classification using binary convolutional neural networks," in ECCV, 2016, pp. 525-542.

[26] M. Jaderberg, A. Vedaldi, and A. Zisserman, "Speeding up convo- lutional neural networks with low rank expansions," arXiv preprint arXiv:1405.3866, 2014.

[27] S. Mehta, M. Rastegari, L. Shapiro, and H. Hajishirzi, "Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network," in CVPR, 2019, pp. 9190-9200.

[28] Y. LeCun, J. S. Denker, and S. A. Solla, "Optimal brain damage," in Advances in neural information processing systems, 1990, pp. 598-605.

[29] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, "Pruning filters for efficient convnets," arXiv preprint arXiv:1608.08710, 2016.

[30] H. Hu, R. Peng, Y.-W. Tai, and C.-K. Tang, "Network trimming: A data- driven neuron pruning approach towards efficient deep architectures," arXiv preprint arXiv:1607.03250, 2016.

[31] V. Belagiannis, A. Farshad, and F. Galasso, "Adversarial network com- pression," in ECCV, 2018, pp. 0-0.

[32] A. Aguinaldo, P.-Y. Chiang, and et. al., "Compressing GANs using knowledge distillation," arXiv preprint arXiv:1902.00159, 2019.