# The Lottery Ticket Hypothesis for Object Recognition

［#1］
Sharath Girish*
sgirish@cs.umd.edu
Shishira R. Maiya*
shishira@umd.edu
Kamal Gupta
kampta@umd.edu
Hao Chen
chenh@umd.edu

［#2］
Larry Davis
lsd@umiacs.umd.edu
Abhinav Shrivastava
abhinav@cs.umd.edu

［#3］
University of Maryland, College Park

## Abstract

［#4］
Recognition tasks, such as object recognition and keypoint estimation, have seen widespread adoption in recent years. Most state-of-the-art methods for these tasks use deep networks that are computationally expensive and have huge memory footprints. This makes it exceedingly difficult to deploy these systems on low power embedded devices. Hence, the importance of decreasing the storage requirements and the amount of computation in such models is paramount. The recently proposed Lottery Ticket Hypothesis (LTH) states that deep neural networks trained on large datasets contain smaller subnetworks that achieve on par performance as the dense networks. In this work, we perform the first empirical study investigating LTH for model pruning in the context of object detection, instance segmentation, and keypoint estimation. Our studies reveal that lottery tickets obtained from Imagenet pretraining do not transfer well to the downstream tasks. We provide guidance on how to find lottery tickets with up to 80% overall sparsity on different sub-tasks without incurring any drop in the performance. Finally, we analyse the behavior of trained tickets with respect to various task attributes such as object size, frequency, and difficulty of detection. Our code is made public at: https://github.com/Sharath-girish/LTH-ObjectRecognition.

［#5］
![](./images/867752587116413327_1.jpg)

［#6］
Figure 1: Performance of lottery tickets discovered using direct pruning for various object recognition tasks. Here we have used a Mask R-CNN model with ResNet-18 backbone (top) and ResNet-50 backbone (bottom) to train models for object detection, segmentation and human keypoint estimation on the COCO dataset. We show the performance of the baseline dense network, the sparse subnetwork obtained by transferring ImageNet pre-trained “universal” lottery tickets, as well as the subnetwork obtained by task-specific pruning. Task-specific pruning outperforms the universal tickets by a wide margin. For each of the tasks, we can obtain the same performance as the original dense networks with only 20% of the weights.

## 1. Introduction

［#7］
Recognition tasks, such as object detection, instance segmentation, and keypoint estimation, have emerged as canonical tasks in visual recognition because of their intuitive appeal and pertinence in a wide variety of real-world problems. The modus operandi followed in nearly all state-of-the-art visual recognition methods is the following: (i) Pre-train a large neural network on a very large and diverse image classification dataset, (ii) Append a small task-specific network to the pre-trained model and fine-tune the weights jointly on a much smaller dataset for the task. The introduction of ResNets by He et al. [23] made the training of very deep networks possible, helping in scaling up model capacity, both in terms of depth and width, and became a well-established instrument for improving the performance of deep learning models even with smaller datasets [26]. As a result, the past few years have seen increasingly large neural network architectures [36, 57, 49, 24], with sizes often exceeding the memory limits of a single hardware accelerator. In recent years, efforts towards reducing the memory and computation footprint of deep networks have followed three seemingly parallel tracks with common objec-

---
［#8］
*First two authors contributed equally
To appear at CVPR 2021

［#9］
tives: weight quantization, sparsity via regularization, and network pruning; Weight Quantization [25, 17, 46, 7, 31] methods either replace weights of a trained neural network with lower precision or arithmetic operations with bit-wise operations to reduce the memory up to an order of magni- tude. Regularization approaches, such as dropout [47, 2] or LASSO [50], attempt to discourage an over-parameterized network from relying on a large number of features and en- courage learning a sparse and robust predictor. Both quanti- zation and regularization approaches are effective in reduc- ing the number of weights in a network or the memory foot- print, but usually at the cost of increased error rates [21, 31]. In comparison, pruning approaches [29, 20] disentangle the learning task from pruning by alternating between weight optimization and weight deletion. The recently proposed Lottery Ticket Hypothesis -(LTH) [13] falls in this category.

［#10］
According to LTH, an over-parameterized network con- tains sparse sub-networks which not only match but some- times even exceed the performance of the original network, all by virtue of a "lucky" random initialization before train- ing. The original paper was followed up with tips and tricks to train large-scale models under the same paradigm [16]. Since then, there has been a large, growing body of litera- ture exploring its nuances. Although some of these recent works have tried to answer the question - how well do the tickets transfer across domains [39, 38], when it comes to vision tasks - the buck stops at image classification.

［#11］
In this work, we aim to extend and explore the analysis of lottery tickets to fundamental visual recognition tasks of object detection, instance segmentation, and keypoint de- tection. Popular methods for such recognition tasks use a two-stage detection pipeline, with a supervised pre-trained convolutional neural network (ConvNet) backbone, a re- gion proposal network (RPN), and one or more region-wise task-specific neural network branches. Loosely speaking, a ConvNet backbone is the most computationally intensive part of the architecture, and pre-training is the most time- consuming part. Therefore, as part of this study, we ex- plore the following questions: (a) Are there universal sub- networks within the ConvNet backbone that can be trans- ferred to the downstream object recognition tasks? (b) Can we train sparser and more accurate sub-networks for each of the downstream tasks? And, (c) How does the behav- ior or properties of these sub-networks change with respect to the corresponding dense network? We investigate these questions under the dominant settings used in object recog- nition frameworks. Specifically, we use ImageNet [8] pre- trained ResNet-18 and ResNet-50 [23] backbones, Faster R- CNN [42] and Mask R-CNN [22] modules for object recog- nition on Pascal VOC [12] and COCO [32] datasets. Our contributions are as follows:

［#12］
- We show that tickets obtained from ImageNet training don't transfer to object recognition in case of COCO,
i.e., there are no universal tickets in pre-trained Ima- geNet models that can be used for downstream recog- nition tasks without a drop in performance. This is in contrast with previous works related to ticket transfer in vision models [38, 39]. In case of smaller datasets such as Pascal VOC, we are able to find winning tickets from ImageNet pre-training with upto 40% sparsity.

［#13］
- With direct pruning, we can find "task-specific" tick- ets with up to 80% sparsity for each of the datasets and backbones. We also investigate the efficacy of methods introduced by [13, 39, 14, 43] such as iterative mag- nitude pruning, late resetting, early bird training, and layerwise pruning in the context of object recognition.

［#14］
- Finally we analyse the behavior of tickets obtained for object recognition tasks, with respect to various task attributes such as object size, frequency, and difficulty of detection, to make some expected (and some sur- prising) observations.

## 2. Related Work

［#15］
Model Compression: Ever since deep neural networks started gaining traction in real-world applications, there have been serious attempts made to reduce their parame- ters, intending to attain lower memory footprints [17, 54,25, 46, 7, 31], higher inference speeds [51, 9, 19] and po- tentially better generalization [1]. Amongst the various pro- posed techniques, model pruning approaches are predomi- nant mainly due to their simplicity and effectiveness. One line of methods follow an unstructured process where in- significant weights are set to zero and are frozen for the rest of the training. The significance of weights are quanti- fied either by magnitude [20] or gradients during training time [30]. In structured pruning methods, relationships be- tween pruned weights are taken into consideration, leading to pruning them in groups. Methods like [53] utilize Group Lasso regularization to prune redundant filter weights to en- able structural sparsity, [34] uses explicit $L0$ regularization to make weights within structures have exact zero values, and network slimming [33] learns an efficient network by modelling the scaling factor of batch normalization layer.

［#16］
The Lottery Ticket Hypothesis: The introduction of Lot- tery Ticket Hypothesis by [16] opened a pandora's box of immense possibilities in the field of pruning and sparse models. The original paper was followed by [15] where the authors introduce the concept of "late resetting" which en- abled the application of the hypothesis to larger and deeper models. [58] followed up by proposing an extensive, in- depth analysis where they show that the resetting of the weights need not be to the exact initialization, but just need to the initial signs. [14] probes the aspect of resetting fur- ther to show that the reason why LTH works is because


［#16］
of its ability to make the subnetwork stable to SGD noise.
As far as theoretical guarantees are considered, [37] offers
strong theoretical proofs for the experimental evidence of
LTH. [18] probed an orthogonal question about the num-
ber of possible tickets from a network. They showed that
a single initialization had multiple winning tickets with low
overlap and empirically conclude that there exists an entire
“distribution” of winning lottery tickets.

［#17］
Complementary to LTH[16], [30] and [52] offer algo-
rithms that can pick the winning ticket without the need for
training. But they do not match the performance of the orig-
inal procedure. The problem of longer training using LTH
was effectively tackled by [55] which introduced the con-
cept of “early bird tickets” where the authors show that the
winning tickets and their masks are obtained in the first few
epochs of training, foregoing the need to train the original
initialization till convergence. The intriguing properties of
LTH led to a glut of works which investigated its eclec-
tic aspects. [39] scrutinize the generalization properties of
winning tickets and offer empirical evidence that winning
tickets can be transferred across datasets and optimizers, in
the realm of image classification. The authors also discuss
the learnt “inductive biases” of the tickets which may lead
to worse performance of transferred tickets when compared
with a ticket obtained from the same dataset. [38] then
proposed a variation of the theory titled “transfer ticket hy-
pothesis” where they investigate the effectiveness of trans-
ferring a mask generated from source dataset to a target
dataset. [10] shows that the winning tickets do not perform
simple overfitting to any domain and carry forward certain
inherent biases which can prove useful in other domains too.
There have been many applications of LTH in the fields of
NLP [6] [40] [10][4] and Reinforcement Learning [56] [55]
as well.

［#18］
The work of [45] briefly analyzes LTH on single stage
detectors such as YOLOv3 [41] and achieves 90% win-
ning tickets, while maintaining the mAP on the Pascal VOC
2007 dataset. However, as they evaluate on light-weight and
fast detectors, their mAP $(\sim 56)$ is much lower compared
to networks like Faster R-CNN [42] which reach mAP of
［#18］
$\sim 69$ with just a ResNet-18 backbone. Their work is also
limited to object detection and does not provide a detailed
analysis of LTH for the task. The idea for transferring sub-
networks obtained from ImageNet to object detection tasks
was concurrently discussed by [5]. For small datasets such
as Pascal VOC, [5] observes that ImageNet tickets transfer
for detection and segmentation tasks. However, we extend
the analysis to the larger COCO dataset and show that this
observation doesn’t hold. We further build upon these re-
sults, to test out the generalization and transfer capabilities
of winning lottery tickets across different object recognition
datasets and tasks in computer vision.

［#19］
```
Algorithm 1 Iterative Pruning for LTH
 1: Randomly initialize network $f$ with initial weights $w_0$,
    mask $m_0 = \mathbb{1}$, prune target percentage $p$, and $T$ prun-
    ing rounds to achieve it.
 2: while $i < T$ do
 3:    Train network for N iterations $f(x; m_i \odot w_0) \rightarrow$
［#19］
        $f(x; m_i \odot w_i)$
［#20］
 4:    Prune bottom $p^{\frac{1}{k}}\%$ of $m_i \odot w_i$ and update $m_i$.
 5:    Reset to initial weights $w_0$
 6:    $i \leftarrow i + 1$                      $\triangleright$ next round
```

## 3. Background: Lottery Ticket Hypothesis

［#21］
LTH states that dense randomly-initialized neural net-
works contain sparse sub-networks which can be trained
in isolation and can match the test accuracy of the origi-
nal network. These sub-networks are called winning tick-
ets and can be identified using an algorithm called Itera-
tive Magnitude Pruning (IMP). Suppose the number of it-
erations for pruning is $T$ and we wish to prune $p\%$ of the
network weights. The weights/parameters are represented
by $w \in \mathbb{R}^n$ and the pruning mask by $m \in \{0, 1\}^n$ where $n$
is the total number of weights in the network. The complete
algorithm is presented in 1.

［#22］
This pruning method can be one-shot when it proceeds
for only a single iteration or it can proceed for multiple it-
erations, $k$, pruning $p^{\frac{1}{k}}\%$ each round. The authors also use
other techniques such as learning rate warmup and show
that finding winning tickets is sensitive to the learning rate.
While this method obtains winning tickets for smaller
datasets, like MNIST [28], CIFAR10 [27], they fail to gen-
eralize to deeper networks, such as ResNets, and larger vi-
sion benchmarks, such as ImageNet [8]. [15] shows that
IMP fails when resetting to the original initialization. They
claim that resetting instead to the network weights after a
few iterations of training provides greater stability and en-
ables them to find winning tickets in these larger networks.
They show that rewinding/late resetting to $3-7\%$ into train-
ing yields subnetworks which are $70\%$ smaller in the case
of ResNet-50, without any drop in accuracy.

## 4. LTH for Object Recognition

［#23］
In this section, we extend the Lottery Ticket Hypothesis
to several object recognition tasks, such as Object Detec-
tion, Instance Segmentation, and Keypoint Detection. In
§4.1, we describe the datasets, models, and metrics we use
in our paper. §4.2 examines the transfer of the lottery tickets
obtained from ImageNet training to the downstream recog-
nition tasks. §4.3 investigates direct pruning on the down-
stream tasks. §4.4 analyzes the various properties of win-
ning tickets obtained using direct pruning.

［#24］
Table 1: Performance on the COCO dataset for ImageNet transferred tickets for ResNet-18 backbone at varying sparsity. The results for VOC are averaged over 5 runs with the standard deviation in parantheses. We obtain winning tickets at higher sparsity for smaller datasets like VOC compared to COCO.

［#25］
<table>
  <thead>
    <tr>
      <th rowspan="2">Prune %</th>
      <th colspan="3">COCO Detection</th>
      <th colspan="3">COCO segmentation</th>
      <th colspan="3">COCO Keypoint</th>
      <th colspan="2">VOC Detection</th>
    </tr>
    <tr>
      <th>Network sparsity</th>
      <th>mAP</th>
      <th>AP50</th>
      <th>Network sparsity</th>
      <th>mAP</th>
      <th>AP50</th>
      <th>Network sparsity</th>
      <th>mAP</th>
      <th>AP50</th>
      <th>Network sparsity</th>
      <th>mAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>90%</td>
      <td>31.61%</td>
      <td>25.59</td>
      <td>43.69</td>
      <td>31.61%</td>
      <td>24.03</td>
      <td>40.89</td>
      <td>21.47%</td>
      <td>55.30</td>
      <td>79.30</td>
      <td>79.49%</td>
      <td>63.91(±0.41)</td>
    </tr>
    <tr>
      <td>80%</td>
      <td>28.10%</td>
      <td>27.70</td>
      <td>46.50</td>
      <td>28.10%</td>
      <td>25.90</td>
      <td>43.70</td>
      <td>19.09%</td>
      <td>56.70</td>
      <td>81.10</td>
      <td>70.66%</td>
      <td>65.82(±0.23)</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>17.57%</td>
      <td>28.52</td>
      <td>47.54</td>
      <td>17.57%</td>
      <td>26.60</td>
      <td>44.66</td>
      <td>11.94%</td>
      <td>56.96</td>
      <td>80.83</td>
      <td>44.16%</td>
      <td>68.06(±0.11)</td>
    </tr>
    <tr>
      <td>0%</td>
      <td>0%</td>
      <td>29.91</td>
      <td>49.05</td>
      <td>0%</td>
      <td>27.64</td>
      <td>46.00</td>
      <td>0%</td>
      <td>58.59</td>
      <td>82.04</td>
      <td>0%</td>
      <td>68.53(±0.29)</td>
    </tr>
  </tbody>
</table>

［#26］
Table 2: Performance on the COCO dataset for ImageNet transferred tickets for ResNet-50 backbone at various levels of pruning. The results for VOC are averaged over 5 runs with the standard deviation in parantheses. We obtain higher levels of sparsity compared to ResNet-18 transferred tickets which can be expected as it has fewer redundant parameters. Additionally, tickets for VOC have much higher sparsity with no drop in mAP compared to unpruned model.

［#27］
<table>
  <thead>
    <tr>
      <th rowspan="2">Prune %</th>
      <th colspan="3">COCO Detection</th>
      <th colspan="3">COCO segmentation</th>
      <th colspan="3">COCO Keypoint</th>
      <th colspan="2">VOC Detection</th>
    </tr>
    <tr>
      <th>Network sparsity</th>
      <th>mAP</th>
      <th>AP50</th>
      <th>Network sparsity</th>
      <th>mAP</th>
      <th>AP50</th>
      <th>Network sparsity</th>
      <th>mAP</th>
      <th>AP50</th>
      <th>Network sparsity</th>
      <th>mAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>90%</td>
      <td>41.99%</td>
      <td>30.66</td>
      <td>50.75</td>
      <td>41.99%</td>
      <td>28.68</td>
      <td>47.76</td>
      <td>31.49%</td>
      <td>57.78</td>
      <td>82.25</td>
      <td>65.37%</td>
      <td>71.20(±0.21)</td>
    </tr>
    <tr>
      <td>80%</td>
      <td>37.33%</td>
      <td>31.01</td>
      <td>50.98</td>
      <td>37.33%</td>
      <td>29.04</td>
      <td>47.90</td>
      <td>28%</td>
      <td>58.55</td>
      <td>83.06</td>
      <td>58.11%</td>
      <td>71.08(±0.20)</td>
    </tr>
    <tr>
      <td>0%</td>
      <td>0%</td>
      <td>38.5</td>
      <td>59.29</td>
      <td>0%</td>
      <td>35.13</td>
      <td>56.39</td>
      <td>0%</td>
      <td>64.59</td>
      <td>86.48</td>
      <td>0%</td>
      <td>71.21(±0.32)</td>
    </tr>
  </tbody>
</table>

### 4.1. Experimental setup

［#28］
We evaluate LTH primarily on the 2 datasets - Pascal VOC 2007 and COCO. We deal with the 3 tasks of object detection, instance segmentation, and keypoint detection for COCO and only object detection for VOC. We use Mask-RCNN for object detection and segmentation for COCO, Keypoint-RCNN for keypoint detection, and Faster-RCNN for object detection on VOC. We compare the results for ResNet-18 and ResNet-50 backbones. Note that while we use the term mean Average Precision (mAP) as a performance metric for all the tasks and datasets, the actual calculation of mAP is done using code provided in the respective datasets (and cannot be compared across the tasks).

### 4.2. Transfer of ImageNet Tickets

［#29］
Many object recognition tasks utilize pre-trained networks whose backbones are trained on the ImageNet dataset. This is because ImageNet features and weights have shown the ability [22] to generalize well to several downstream vision tasks. A plethora of works exists which perform LTH for the ImageNet classification task and obtain winning tickets. Therefore, tickets for standard convolutional backbones, such as ResNets, are readily available. This raises the pertinent question of whether pruned ImageNet trained models transfer directly to object recognition tasks. In order to answer this question, we transfer the pruned model to the backbone of the RCNN-based network and fine-tune the full network while ensuring the pruned weights in the backbone remain as zeros.

［#30］
We perform experiments for the two architectures: ResNet-18 and ResNet-50, where we obtain 10%, 20%, and 50% tickets on ImageNet by following the approach of [15], and then transfer the model to the backbones of the three R-CNN based networks. All models were trained on COCO, encompassing the three recognition tasks and the results are summarized in Tables 1, 2. Additionally, we also perform similar experiments on the smaller Pascal VOC 2007 dataset for object detection to verify whether ImageNet tickets transfer without a significant drop in mAP. The results are shown in the last column of Tables 1, 2. Note that pruning percentage of the ImageNet ticket is not equal to the actual network sparsity of the various networks as only the backbone of the networks are transferred and they make up a fraction of the total weights.

［#31］
We see that ImageNet tickets transferred to COCO show a noticeable drop in mAP even with low levels of sparsity. We also note that another drawback of training transferred tickets on COCO is that they require careful tuning of learning rate and batch size for different tasks. On the other hand, for smaller datasets such as Pascal VOC, winning tickets are easily obtained at higher levels of sparsity which is $\sim 45\%$ for ResNet-18 and $\sim 65\%$ for ResNet-50. The larger networks can be pruned to a greater extent for both datasets.

［#32］
ImageNet transferred tickets offer very little sparsity as the backbone of the networks usually do not make up most of the weights. For example, the ResNet-18 based Mask-RCNN for COCO Detection and Segmentation has only 44% of the parameters in the backbone, and hence, the overall network sparsity reaches 31% when pruning 90% of the backbone weights, as shown in Table 1. The rest of the weights are usually from the fully-connected layers of the


［#32］
network. It is therefore imperative to prune layers in addition to the backbone to increase network sparsity without decrease in mAP. As a consequence, we look into directly pruning the full network using LTH.

### 4.3. Direct Pruning for Downstream Task

［#33］
In this section, we analyze the effect of various hyperparameters and pruning strategies for detection networks in order to obtain winning tickets. We primarily use the ResNet-18 backbone for Faster-RCNN trained on VOC for all our experiments in this section, unless mentioned otherwise. Even though the ResNet-18 backbone is smaller than other backbone networks such as ResNet-50, we find that similar conclusions hold for the larger networks as well.

［#34］
The Faster RCNN network consists of parameters which we group into 4 main modules: Base Convolutions, Classification Network Convolutions (Top), Region Proposal Network (RPN), Classification network box and classification fully connected heads (Box and Cls Head). We provide a detailed analysis of pruning these 4 groups and their effect on winning tickets. Additionally, we also analyze different pruning strategies and the role played by hyperparameters.
Varying Pruning Percentage: We evaluate the network performance at varying levels of sparsity. We prune different percentages of parameters in the Base and Top modules which include $88\%$ of the total network parameters. The results are plotted in Fig.2a. We achieve performance within one standard deviation of the baseline, with $70\%$ sparsity. Our models outperform the baseline mean, thereby proving that we can indeed obtain high performance winning tickets at much higher levels of sparsity for detection. Additionally, we see that at any given sparsity, direct pruning yields much better results compared to ImageNet transferred tickets. We also show that these observations pan other vision tasks by obtaining winning tickets for Mask-RCNN and Keypoint-RCNN with both ResNet-18 and ResNet-50 backbones on the COCO dataset. We additionally prune FC layers in these set of experiments in order to achieve desired sparsity levels as they take up $\sim50\%$ of the total weights. The results for ResNet-18 are shown in Fig. 1. We obtain winning tickets with $80\%$ sparsity on all the three tasks while outperforming the unpruned network for lower levels of sparsity. Additionally, we consistently outperform the different ImageNet transferred tickets ($50\%$, $80\%$, $90\%$) by a large margin supporting our claim that direct training of tickets on downstream tasks yield better results than ImageNet tickets.

［#35］
Effect of Early/Late Resetting: [15] states that resetting the network to a few iterations through training instead of the initialization stabilizes the winning ticket training. We evaluate whether this holds true for detection tasks as well. We show the performance of winning tickets as a function of resetting at various stages of training in Fig. 2b and observe that resetting during the earlier or even mid stages of training does not have a very strong effect on the final mAP. This is likely because the backbones of detection networks are initialized with ImageNet weights and are not random as is the case with other papers dealing with LTH in the classification setting. Therefore, the weights are more stable and late resetting is not necessary. We also additionally analyze effects of resetting towards the end of training and notice that there is a sharp drop in the performance after $8k$ iterations. This is because the learning rate is decayed at this stage of training and the parameters change significantly right after. A similar case holds when we perform learning rate warmup but do late resetting before the learning rate is fully warmed up. The performance drops significantly as the learning rate keeps fluctuating showing that late resetting is quite sensitive to learning rate.

［#36］
<div style="text-align:center">
<table>
<caption>Table 3: Performance on Pascal VOC by pruning different modules of a ResNet-18 Faster-RCNN network. The results are averaged over 5 runs with the standard deviation in parantheses. $\checkmark$ represents the module being pruned, while Param % represents the percentage of parameters occupied by the modules being pruned.</caption>
<thead>
<tr>
<th>Base</th>
<th>Top</th>
<th>RPN</th>
<th>Box,<br>Cls Head</th>
<th>Param<br>%</th>
<th>Network<br>Sparsity</th>
<th>mAP</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0</td>
<td>0%</td>
<td>69.74 ($\pm$0.16)</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>$\checkmark$</td>
<td>0.65</td>
<td>0.52%</td>
<td>70.30 ($\pm$0.14)</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>$\checkmark$</td>
<td>-</td>
<td>9.71</td>
<td>7.77%</td>
<td>70.02 ($\pm$0.19)</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>10.36</td>
<td>8.29%</td>
<td>70.08 ($\pm$0.10)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>21.93</td>
<td>17.55%</td>
<td>69.32 ($\pm$0.07)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>-</td>
<td>-</td>
<td>$\checkmark$</td>
<td>22.59</td>
<td>18.07%</td>
<td>69.60 ($\pm$0.19)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>-</td>
<td>$\checkmark$</td>
<td>-</td>
<td>31.64</td>
<td>25.31%</td>
<td>69.39 ($\pm$0.25)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>-</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>32.29</td>
<td>25.83%</td>
<td>69.47 ($\pm$0.15)</td>
</tr>
<tr>
<td>-</td>
<td>$\checkmark$</td>
<td>-</td>
<td>-</td>
<td>66.39</td>
<td>53.11%</td>
<td>69.02 ($\pm$0.19)</td>
</tr>
<tr>
<td>-</td>
<td>$\checkmark$</td>
<td>-</td>
<td>$\checkmark$</td>
<td>67.04</td>
<td>53.63%</td>
<td>68.74 ($\pm$0.21)</td>
</tr>
<tr>
<td>-</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>-</td>
<td>76.09</td>
<td>60.88%</td>
<td>68.88 ($\pm$0.25)</td>
</tr>
<tr>
<td>-</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>76.75</td>
<td>61.40%</td>
<td>68.93 ($\pm$0.26)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>-</td>
<td>-</td>
<td>88.32</td>
<td>70.66%</td>
<td>68.45 ($\pm$0.21)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>-</td>
<td>$\checkmark$</td>
<td>88.97</td>
<td>71.18%</td>
<td>68.54 ($\pm$0.23)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>-</td>
<td>98.03</td>
<td>78.42%</td>
<td>68.51 ($\pm$0.23)</td>
</tr>
<tr>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>$\checkmark$</td>
<td>98.68</td>
<td>78.94%</td>
<td>68.47 ($\pm$0.10)</td>
</tr>
</tbody>
</table>
</div>

［#37］
Pruning different Faster-RCNN modules: We prune $20\%$ of the parameters of the various modules within the Faster-RCNN network and analyze their effects on the mAP. We also try different combinations of pruning with the modules and report the results in Table 3. Pruning the Box and Classification head (which takes up only $65\%$ weights) outperforms the baseline case of no pruning, but does not always improve performance when other modules are being pruned. Additionally, pruning the $RPN$ module increases the performance slightly even though it comprises of only $10\%$ of the network weights. Next, pruning the Base module and/or the Top module of the backbone leads to a drop in performance, which is expected as they consist of $22\%$ and $66\%$ of the weights respectively. Pruning the Base alone, excluding the Top, performs nearly as well as the baseline, while including the Top yields a lower mAP.

［#38］
![](./images/867752587116413327_2.jpg)

［#39］
Figure 2: Effect of varying different hyperparameters for pruning Faster RCNN with ResNet-18 backbone on the Pascal VOC 2007 [12] dataset. All solid lines reported are the values averaged over 5 runs and the error bands are within 3 times the standard deviation.

［#40］
Performance of Early-bird tickets: [55] showed that tickets can be found at early stages of training. We visualize this by obtaining masks at various stages in training and evaluating their performance. We also plot each masks' Intersection over Union (IoU) with the default mask obtained at the end of training. This IoU shows the overlap in the parameters being pruned. The results are visualized in Fig. 2f. We see that within 50% of network training we find tickets whose performance is within a standard deviation of the performance of the default ticket (obtained at the end of training). This is because the IoU becomes more or less stable at around 0.96 during the middle stages of training and the mask is unchanged as training advances. This allows us to cut down on the number of training iterations significantly with very little cost to the network performance.

［#41］
Effect of number of rounds of pruning: [13] states that iterative pruning performs better than one-shot pruning on the classification task with small datasets and networks. We show that this does not necessarily hold true for detection and larger backbones. We plot the network's performance against various rounds of pruning and observe that one-shot pruning outperforms iterative methods in Fig. 2d.

［#42］
Layer-wise vs. global pruning: [15] performs global pruning for larger datasets and networks and claims that pruning at the same rate in lower layers as compared to higher layers, is detrimental to the network performance. We evaluate the two methods of pruning on the detection task and show the results in Fig. 2e. Additionally, for global pruning, we plot the percentage of parameters pruned in each layer of the backbone network in Fig. 2c. Layer-wise pruning does as good as global pruning for lower levels of sparsity. However, there is a noticeable performance gap for sparsity levels above 60%. This is because layer-wise pruning forces lower layers with very few parameters to have high sparsity percentages. But as per Fig. 2c, for global pruning, we see that lower layers are pruned less as they are crucial to both the RPN and Classification stages of the network.

### 4.4. Properties of Winning Tickets

［#43］
In Section 4.3, we showed that we can discover sparser networks within our two-stage Mask-RCNN detector if we directly prune on the task itself. We build upon those results to further probe the properties of winning tickets.

［#44］
Effect of backbone architecture: In Fig. 3, we show how winning tickets behave for 2 different backbones, ResNet-18 and ResNet-50, at different sparsity levels (50%, 80%, 90%). We make two observations: First, the breaking point for both networks is $\sim 80\%$ sparsity. However, performance of ResNet-18 drops more sharply than ResNet-50 afterwards. This is intuitive since ResNet-18 has fewer redundant parameters and over-pruning leads to drop in the performance. Second, as we gradually increase the sparsity of the networks, mAP increases for all tasks in case of both networks. However, gains for ResNet-18 models are consistently more than ResNet-50.

［#45］
Do winning tickets behave differently for varying object sizes? Using the definition from [32], we categorize bounding boxes into small (area $< 32^2$), medium ($32^2 <$ area $< 96^2$), and large (area $> 32^2$). To understand how sparse

［#46］
![](./images/867752587116413327_3.jpg)

［#47］
Figure 3: ResNet-18 vs. ResNet-50. We analyse change in mAP by using LTH on Mask R-CNN with different backbones.

［#48］
![](./images/867752587116413327_4.jpg)

［#49］
Figure 4: Comparison of Mean Average Precision (mAP) of pruned model for different object sizes in case of Object Detection, Instance Segmentation, Keypoint Estimation. x-axis shows the sparsity of the subnetwork (or the percentage of weights removed). y-axis shows the percentage drop in mAP as compared to the unpruned network. For all tasks, and object sizes, performance doesn't drop till about 80% sparsity. After which, small objects are hit slightly harder as compared to medium and large objects.

［#50］
networks behave for different sized objects, we plot the percentage gain or drop from the mAP of a dense network. Figure 4 shows the percentage change in mAP for different levels of sparsity in the Mask R-CNN model. We can observe that in each case, the model performance increases with sparsity, until sparsity reaches 80%, after which, mAP sharply declines. We note that the percentage drop for small boxes is more, with winning tickets (10% of weights) showing a drop of over 17% in case of detection and segmentation tasks while medium sized objects show smaller drops than large objects for all tasks.

［#51］
How does the performance of the pruned network vary for rare vs. frequent categories? We sort the 80 object categories in COCO by their frequency of occurrence in training data. We consider networks with 80% and 90% of their weights pruned and observe the percentage change in the bounding box mAP of the model with respect to the unpruned network for each of the categories. Figure 5(a) depicts the behavior with a bar graph. While for most categories, winning tickets are obtained at 80% sparsity, performance drops sharply with more pruning in case of rare categories (such as toaster, parking meter, and bear) as compared to common categories (such as person, car, and chair).

［#52］
Do the winning tickets behave differently on easy vs hard categories? For a machine learning model, an object can be easy or hard to recognize because of a variety of reasons. We have already discussed two reasons that influence the performance — number of instances available in the training data, and size of the object. There can also be other causes that can render an object unrecognizable in given surroundings. Camouflage or occlusion, poor camera quality, light conditions, distance from the camera, or just variations within different instances or views of the object are few of them. Since exhaustive analyses of these causes is intractable, we rank object categories based on performance of an unpruned Mask R-CNN model. We do this categorization for detection and segmentation models as shown in Figure 5(b) and (c). Note that 'easy' and 'hard' categories from these two definitions have an overlap but they are not the same. For example, knife, handbag, and spoon are the categories with lowest bounding box mAP, and giraffe, zebra, and stop signs are one with the highest (excluding 'hair drier' which has 0 mAP). On the other hand, skis, knife, and spoon have the lowest segmentation mAP, while stop sign, bear, and fire hydrant have the highest. From the Figure 5(b) and (c), we make the following observations — (i) tickets with 80% sparsity can actually increase mAP for certain categories like snowboard by as much as 38%, (ii) Going from 80% to 90% sparsity, mAP

［#53］
![](./images/867752587116413327_5.jpg)

［#54］
Figure 5: Comparison of Mean Average Precision (mAP) of pruned model for 80 COCO object categories. x-axis in each of the plot is a list of categories (sorted using different criteria). y-axis shows the percentage drop in mAP as compared to the unpruned network.

［#55］
Table 4: Effect of ticket transfer across tasks. Transferred tickets do worse than direct training as expected, but still do not result in drastic drops in the mAP or AP50. Here we do task transfer using the 80% pruned model.

［#56］
<table>
  <tr>
    <th>Target task</th>
    <th>Source task</th>
    <th>Network sparsity</th>
    <th>mAP</th>
    <th>AP50</th>
  </tr>
  <tr>
    <td>Det</td>
    <td>Det/Seg</td>
    <td>78.4%</td>
    <td>30.04</td>
    <td>49.40</td>
  </tr>
  <tr>
    <td></td>
    <td>Keypoint</td>
    <td>50.11%</td>
    <td>23.94</td>
    <td>41.08</td>
  </tr>
  <tr>
    <td>Seg</td>
    <td>Det/Seg</td>
    <td>78.4%</td>
    <td>27.90</td>
    <td>46.68</td>
  </tr>
  <tr>
    <td></td>
    <td>Keypoint</td>
    <td>50.11%</td>
    <td>23.02</td>
    <td>39.01</td>
  </tr>
  <tr>
    <td rowspan="2">Keypoint</td>
    <td>Det/Seg</td>
    <td>76.98%</td>
    <td>58.31</td>
    <td>81.53</td>
  </tr>
  <tr>
    <td>Keypoint</td>
    <td>79.4%</td>
    <td>59.34</td>
    <td>82.36</td>
  </tr>
</table>

［#57］
drops significantly for easy categories compared to hard categories, (iii) categories that are hit the hardest such as skis, hot dog, spoon, fork, handbags usually have long, thin appearance in images.

［#58］
Do winning tickets transfer across tasks? We showed that ImageNet tickets transfer to a limited extent to downstream tasks. We further study whether the tickets obtained from the downstream task of detection/segmentation transfer to keypoint estimation and vice-versa. We train Mask-RCNN and Keypoint-RCNN respectively for the two tasks on the COCO dataset while maintaining a sparsity level of 80%. For both the tasks we transfer all values till box head modules, after which the model structures differ. The results are shown in Table 4. We can observe that the drop is marginal for the transfer of tickets between detection-segmentation to keypoint task, as compared with the reverse case which registers a significant drop. This might be because the ticket is obtained on the keypoint task which is trained only on 'human' class and it fails to transfer well for the detection task which uses the entire COCO dataset.

［#59］
5. Discussion

［#60］
[38, 39] show that winning tickets transfer well across datasets. However, the study in [38] was limited to smaller datasets, like CIFAR-10 and FashionMNIST, and both [38, 39] are limited to classification tasks. We obtain contrasting results when transferring tickets across tasks as shown in Sec. 4.2. ImageNet tickets transfer with approximately 40% sparsity to fall within one standard deviation of the baseline network. This is likely due to the fact that winning tickets retain inductive biases from the source dataset which are less likely to transfer to a new domain and task. Additionally, we show that unlike prior LTH works, iterative pruning degrades the performance of subnetworks on detection and one-shot pruning provides the best networks. We also observe that due to the use of pre-trained weights from ImageNet for the backbone of detection networks, late resetting is not necessary for finding winning tickets. This is in contrast to the [15], which is restricted to the classification task involving random initialization for the networks. Like previous works, in our experiments as well, we find that sparse lottery tickets often outperform the dense networks themselves. However, we make another interesting observation — in each of object recognition tasks, tickets with fewer parameters such as ResNet-18 show more gains in performance as compared to tickets with more parameters (ResNet-50). We also find that small and infrequent objects face higher performance drop as the sparsity increases.

［#61］
6. Conclusion

［#62］
We investigate the Lottery Ticket Hypothesis in the context of various object recognition tasks. Our study reveals that the main points of original LTH hold for different recognition tasks, i.e., we can find subnetworks or winning tickets in object recognition pipelines with up to 80% sparsity, without any drop in performance on the task. These tickets are task-specific, and pre-trained ImageNet model tickets don't perform as well on the downstream recognition tasks. We also analyse claims made in recent literature regarding training and transfer of winning tickets from an object recognition perspective. Finally, we analyse how the behavior of sparse tickets differ from their dense counterparts. In the future, we would like to investigate how much speed up can be achieved using these sparse models with various hardware [35] and software modifications [11]. Ex-


［#62］
tending this analyses for even bigger datasets such as JFT-300M [48] or IG-1B [36] and for self-supervised learning techniques is another direction to pursue.

［#63］
Acknowledgements. This work was partially supported by DARPA GARD #HR00112020007 and a gift from Facebook AI.

# References




























































## Appendix

［#64］
We provide additional details for some of the experiments presented in the paper. In particular, we provide comparison with a simpler ImageNet ticket transfer alternative in Section A, compare the different errors made by dense and pruned models in Section B, and finally verify the faster convergence of sparser models in Section C.

### A. Mask Transfer Without Retraining

［#65］
In Section 4.2, we analyzed the effects of transferring tickets only for the ImageNet trained backbones. While this deals with transferring the ticket mask as well as values, we further analyze whether transferring only the mask provides winning tickets for these tasks using the methodology from [38]. We use the default ImageNet weights in the ResNet-18 and ResNet-50 backbone and keep the top $p\%$ of the weights in convolutional layers while setting the rest to zeros and maintaining it throughout the training of the entire network. We refer to this method as 'Mask Transfer'. Since training the backbone on much larger ImageNet data is performed only once, 'Mask Transfer' is a much cheaper or computationally efficient way of obtaining tickets from parent task. We observe that behavior of 'Mask Transfer' is similar to the 'Transfer Ticket' obtained by method discussed in Section 4.2 where the sparse subnetwork weights are fully retrained on ImageNet. Either cases are outperformed by direct pruning on the downstream tasks. The results are summarized in Figure 6 (ResNet-18) and Table 5 (ResNet-50).

［#66］
![](./images/867752587116413327_6.jpg)

［#67］
Figure 6: Transferring ImageNet backbone tickets to object recognition tasks vs. Direct pruning via LTH on the object recognition tasks. We experiment with two variations of transferring ImageNet backbone tickets to object recognition tasks. 'Transfer ticket' refers to the case when we transfer the lottery ticket backbone trained on ImageNet data to downstream task (also discussed in the Section 4 of the paper). 'Mask Transfer' refers to the case when ticket is transferred without retraining on ImageNet, i.e., only the relevant mask from backbone is transferred keeping ImageNet weights the same. Best viewed in color.

### B. Error analysis on downstream tasks

［#68］
The mAP score provides us a good way to summarize the performance of an object recognition model with a single number. But it hides a lot of information regarding what kind of mistakes the model is making. Do the sparse subnetworks obtained by LTH make same mistakes as the dense models? In order to answer this question, we consider a dense Mask R-CNN model with ResNet-50 backbone and a sparse Mask R-CNN model with 20% of the parameters obtained via LTH. Both the models achieve same performance on downstream tasks as also discussed in Section 4.2 of the paper.

#### B.1. Object Detection and Instance Segmentation

［#69］
We resort to a toolbox from [3] to analyze object detection and instance segmentation errors. We consider 5 main sources of errors in object detection. (i) 'Cls' refers to an error corresponding to miss-classification of a bounding box by a model, (ii) 'Loc' refers to the case when bounding box is classified properly but not localized properly, (iii) 'Dupe' corresponds to the errors when model makes multiple predictions at the same location, (iv) 'Bkgd' are the cases when background portion of the image (with no objects) are tagged as an object, and finally (v) 'Missed' cases when the objects are not detected by the model.

［#70］
Figures 7 and 8 summarize the analysis of detection and segmentation errors obtained for dense model as compared to a sparse model (with only 20% of the weights). While in the case of object detection, the performance of both the models is identical, subtle differences emerge in case of segmentation where sparse model makes fewer localization errors but higher background errors.

［#71］
![](./images/867752587116413327_7.jpg)

［#72］
Figure 7: Error analysis of unpruned vs. pruned on object detection. The error types of unpruned and pruned models are nearly the same.

#### B.2. Keypoint Estimation

［#73］
We use [44] to perform a similar analyses for sparse and dense models on the task of keypoint estimation. In case of keypoints,


［#74］
Table 5: Performance on the COCO dataset for ImageNet backbones with mask transfer tickets for ResNet-50 at various levels of pruning. The results for VOC are averaged over 5 runs with the standard deviation in parantheses.

［#75］
<table>
<thead>
<tr>
<th rowspan="2">Prune %</th>
<th colspan="3">COCO Detection</th>
<th colspan="3">COCO segmentation</th>
<th colspan="3">COCO Keypoint</th>
<th colspan="2">VOC Detection</th>
</tr>
<tr>
<th>Network sparsity</th>
<th>mAP</th>
<th>AP50</th>
<th>Network sparsity</th>
<th>mAP</th>
<th>AP50</th>
<th>Network sparsity</th>
<th>mAP</th>
<th>AP50</th>
<th>Network sparsity</th>
<th>mAP</th>
</tr>
</thead>
<tbody>
<tr>
<td>90%</td>
<td>41.99%</td>
<td>35.46</td>
<td>56.51</td>
<td>41.99%</td>
<td>32.40</td>
<td>52.89</td>
<td>31.49%</td>
<td>62.27</td>
<td>84.93</td>
<td>65.37%</td>
<td>61.75(±0.22)</td>
</tr>
<tr>
<td>80%</td>
<td>37.33%</td>
<td>36.52</td>
<td>57.28</td>
<td>37.33%</td>
<td>33.53</td>
<td>54.15</td>
<td>28%</td>
<td>63.48</td>
<td>85.72</td>
<td>58.11%</td>
<td>67.30(±0.39)</td>
</tr>
<tr>
<td>50%</td>
<td>24.55%</td>
<td>37.99</td>
<td>58.83</td>
<td>24.55%</td>
<td>34.76</td>
<td>55.91</td>
<td>19.71%</td>
<td>64.21</td>
<td>86.32</td>
<td>36.32%</td>
<td>70.32(±0.23)</td>
</tr>
<tr>
<td>0%</td>
<td>0%</td>
<td>38.5</td>
<td>59.29</td>
<td>0%</td>
<td>35.13</td>
<td>56.39</td>
<td>0%</td>
<td>64.59</td>
<td>86.48</td>
<td>0%</td>
<td>71.21(±0.32)</td>
</tr>
</tbody>
</table>

［#76］
![](./images/867752587116413327_8.jpg)

［#77］
Figure 8: Error analysis of unpruned vs. pruned on instance segmentation.
The error types of unpruned and pruned models are quite similar.

［#78］
we compute the Precision Recall Curve of the model while removing the impact of individual errors of following kinds — (i) ‘Miss’ - large localization errors, (ii) ‘Swap’ - confusion between same keypoint of two different persons, (iii) ‘Inversion’ - confusion between two different keypoints of the same person, (iv) ‘Jitter’ - small localization error, and (v) ‘FP’ - background false positives. Fig. 9 summarizes the results. As in the previous case, it appears that both the dense and sparse models make similar mistakes.

### C. Sparse subnetworks converge faster

［#79］
The LTH paper [13] claimed that sparse subnetworks obtained by pruning, often converge faster than their dense counterparts. In this section we verify the claims of the paper on object recognition tasks and found them to hold true. We plot the validation loss during training for the dense unpruned model, and the sparse subnetwork obtained by keeping only 20% of the weights of dense model. Both the models achieve a similar mAP after convergence. Fig. 10 shows the task loss against the number of epochs during training. The comparisons confirm that the sparse subnetwork initialized from the winning ticket weights converge much faster. This observation is consistent for heterogeneous tasks, e.g., object detection, instance segmentation, and keypoint estimation.


［#80］
![](./images/867752587116413327_9.jpg)

［#81］
Figure 9: Error analysis of unpruned vs. pruned on kyepoint estimation. The error types of unpruned and pruned models are quite similar while the unpruned one has slightly better performance

［#82］
![](./images/867752587116413327_10.jpg)

［#83］
(a) Object detection
(b) Instance segmentation
(c) Keypoint estimation

［#84］
Figure 10: Training curves of dense model vs. sparse model


# The Lottery Ticket Hypothesis for Object Recognition: Supplementary Material

［#85］
Sharath Girish*
Shishira R Maiya*
Kamal Gupta
Hao Chen
sgirish@cs.umd.edu
shishira@umd.edu
kampta@umd.edu
chenh@umd.edu

［#86］
Larry Davis
Abhinav Shrivastava
lsd@umiacs.umd.edu
abhinav@cs.umd.edu

［#87］
University of Maryland, College Park

［#88］
We provide additional details for some of the experiments presented in the paper. In particular, we provide comparison with a simpler ImageNet ticket transfer alternative in Section A, compare the different errors made by dense and pruned models in Section B, verify the faster convergence of sparser models in Section C and finally analyze the disk space and number of compute operations in Section D.

## A. Mask Transfer Without Retraining
［#89］
In Section 4.2, we analyzed the effects of transferring tickets only for the ImageNet trained backbones. While this deals with transferring the ticket mask as well as values, we further analyze whether transferring only the mask provides winning tickets for these tasks using the methodology from [?]. We use the default ImageNet weights in the ResNet-18 and ResNet-50 backbone and keep the top $p\%$ of the weights in convolutional layers while setting the rest to zeros and maintaining it throughout the training of the entire network. We refer to this method as 'Mask Transfer'. Since training the backbone on much larger ImageNet data is performed only once, 'Mask Transfer' is a much cheaper or computationally efficient way of obtaining tickets from parent task. We observe that behavior of 'Mask Transfer' is similar to the 'Transfer Ticket' obtained by method discussed in Section 4.2 where the sparse subnetwork weights are fully retrained on ImageNet. Either cases are outperformed by direct pruning on the downstream tasks. The results are summarized in Figure 1 (ResNet-18) and Table 1 (ResNet-50).

［#90］
![](./images/867752587116413327_11.jpg)

［#91］
Figure 1: Transferring ImageNet backbone tickets to object recognition tasks vs. Direct pruning via LTH on the object recognition tasks. We experiment with two variations of transferring ImageNet backbone tickets to object recognition tasks. 'Transfer ticket' refers to the case when we transfer the lottery ticket backbone trained on ImageNet data to downstream task (also discussed in the Section 4 of the paper). 'Mask Transfer' refers to the case when ticket is transferred without retraining on ImageNet, i.e., only the relevant mask from backbone is transferred keeping ImageNet weights the same. Best viewed in color.

## B. Error analysis on downstream tasks
［#92］
The mAP score provides us a good way to summarize the performance of an object recognition model with a single number. But it hides a lot of information regarding what kind of mistakes the model is making. Do the sparse subnetworks obtained by LTH make same mistakes as the dense models? In order to answer this question, we consider a dense Mask R-CNN model with ResNet-50 backbone and a sparse Mask R-CNN model with 20% of the parameters obtained via LTH. Both the models achieve same performance on downstream tasks as also discussed in Section 4.2 of the paper.

---
［#93］
*Equal contribution

［#94］
Table 1: Performance on the COCO dataset for ImageNet backbones with mask transfer tickets for ResNet-50 at various levels of pruning. The results for VOC are averaged over 5 runs with the standard deviation in parantheses.

［#95］
<table>
<thead>
<tr>
<th rowspan="2">Prune %</th>
<th colspan="3">COCO Detection</th>
<th colspan="3">COCO segmentation</th>
<th colspan="3">COCO Keypoint</th>
<th colspan="2">VOC Detection</th>
</tr>
<tr>
<td>Network sparsity</td>
<td>mAP</td>
<td>AP50</td>
<td>Network sparsity</td>
<td>mAP</td>
<td>AP50</td>
<td>Network sparsity</td>
<td>mAP</td>
<td>AP50</td>
<td>Network sparsity</td>
<td>mAP</td>
</tr>
</thead>
<tbody>
<tr>
<td>90%</td>
<td>41.99%</td>
<td>35.46</td>
<td>56.51</td>
<td>41.99%</td>
<td>32.40</td>
<td>52.89</td>
<td>31.49%</td>
<td>62.27</td>
<td>84.93</td>
<td>65.37%</td>
<td>61.75(±0.22)</td>
</tr>
<tr>
<td>80%</td>
<td>37.33%</td>
<td>36.52</td>
<td>57.28</td>
<td>37.33%</td>
<td>33.53</td>
<td>54.15</td>
<td>28%</td>
<td>63.48</td>
<td>85.72</td>
<td>58.11%</td>
<td>67.30(±0.39)</td>
</tr>
<tr>
<td>50%</td>
<td>24.55%</td>
<td>37.99</td>
<td>58.83</td>
<td>24.55%</td>
<td>34.76</td>
<td>55.91</td>
<td>19.71%</td>
<td>64.21</td>
<td>86.32</td>
<td>36.32%</td>
<td>70.32(±0.23)</td>
</tr>
<tr>
<td>0%</td>
<td>0%</td>
<td>38.5</td>
<td>59.29</td>
<td>0%</td>
<td>35.13</td>
<td>56.39</td>
<td>0%</td>
<td>64.59</td>
<td>86.48</td>
<td>0%</td>
<td>71.21(±0.32)</td>
</tr>
</tbody>
</table>

［#96］
![](./images/867752587116413327_12.jpg)

［#97］
Figure 2: Error analysis of unpruned vs. pruned on object detection. The error types of unpruned and pruned models are nearly the same.

［#98］
Figure 3: Error analysis of unpruned vs. pruned on instance segmentation. The error types of unpruned and pruned models are quite similar.

### B.1. Object Detection and Instance Segmentation

［#99］
We resort to a toolbox from [?] to analyze object detection and instance segmentation errors. We consider 5 main sources of errors in object detection. (i) 'Cls' refers to an error corresponding to miss-classification of a bounding box by a model, (ii) 'Loc' refers to the case when bounding box is classified properly but not localized properly, (iii) 'Dupe' corresponds to the errors when model makes multiple predictions at the same location, (iv) 'Bkgd' are the cases when background portion of the image (with no objects) are tagged as an object, and finally (v) 'Missed' cases when the objects are not detected by the model.

［#100］
Figures 2 and 3 summarize the analysis of detection and segmentation errors obtained for dense model as compared to a sparse model (with only 20% of the weights). While in the case of object detection, the performance of both the models is identical, subtle differences emerge in case of segmentation where sparse model makes fewer localization errors but higher background errors.

［#101］
![](./images/867752587116413327_13.jpg)

［#102］
Figure 4: Disk space and MAC operations of pruned models with ResNet18 backbone for the various tasks on the COCO dataset.

### B.2. Keypoint Estimation

［#103］
We use [?] to perform a similar analyses for sparse and dense models on the task of keypoint estimation. In case of keypoints, we compute the Precision Recall Curve of the model while removing the impact of individual errors of following kinds — (i) 'Miss' - large localization errors, (ii) 'Swap' - confusion between same keypoint of two different persons, (iii) 'Inversion' - confusion between two different keypoints of the same person, (iv) 'Jitter' - small localization error, and (v) 'FP' - background false positives. Fig. 5 summarizes the results. As in the previous case, it appears that both the dense and sparse models make similar mistakes.

［#104］
![](./images/867752587116413327_14.jpg)

［#105］
Figure 5: Error analysis of unpruned vs. pruned on kyepoint estimation. The error types of unpruned and pruned models are quite similar while the unpruned one has slightly better performance

［#106］
![](./images/867752587116413327_15.jpg)

［#107］
(a) Object detection
(b) Instance segmentation
(c) Keypoint estimation

［#108］
Figure 6: Training curves of dense model vs. sparse model

## C. Sparse subnetworks converge faster

［#109］
The LTH paper [?] claimed that sparse subnetworks obtained by pruning, often converge faster than their dense counterparts. In this section we verify the claims of the paper on object recognition tasks and found them to hold true. We plot the validation loss during training for the dense unpruned model, and the sparse subnetwork obtained by keeping only 20% of the weights of dense model. Both the models achieve a similar mAP after convergence. Fig. 6 shows the task loss against the number of epochs during training. The comparisons confirm that the sparse subnetwork initialized from the winning ticket weights converge much faster. This observation is consistent for heterogeneous tasks, e.g., object detection, instance segmentation, and keypoint estimation.

## D. Disk space and compute operations

［#110］
Finally, we analyze the disk space and MAC operations of pruned models in Figure 4. We store the index and values of only the non zero weights, when above a threshold sparsity, while we store the full weight values for denser sparsity levels. As expected, we observe significant reduc-

［#110］
tions in disk space for higher levels of sparsity. However, number of operations decreases at a much slower rate. This can possibly be improved further with dedicated hardware for sparse operations.