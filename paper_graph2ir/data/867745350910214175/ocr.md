# Towards Practical Lottery Ticket Hypothesis for Adversarial Training

Bai Li*¹, Shiqi Wang*², Yunhan Jia³, Yantao Lu⁴, Zhenyu Zhong⁵, Lawrence Carin¹ and Suman Jana²

¹Duke University, ²Columbia University, ³University of Michigan, ⁴Syracuse University, ⁵Baidu USA

## Abstract

Recent research has proposed the lottery ticket hypothesis, suggesting that for a deep neural network, there exist trainable sub-networks performing equally or better than the original model with commensurate training steps. While this discovery is insightful, finding proper sub-networks requires iterative training and pruning. The high cost incurred limits the applications of the lottery ticket hypothesis. We show there exists a subset of the aforementioned sub-networks that converge significantly faster during the training process and thus can mitigate the cost issue. We conduct extensive experiments to show such sub-networks consistently exist across various model structures for a restrictive setting of hyperparameters (e.g., carefully selected learning rate, pruning ratio, and model capacity). As a practical application of our findings, we demonstrate that such sub-networks can help in cutting down the total time of adversarial training, a standard approach to improve robustness, by up to 49% on CIFAR-10 to achieve the state-of-the-art robustness.

## 1. Introduction

Pruning has served as an important technique for removing redundant structure in neural networks [11, 10, 19, 13]. Properly pruning can reduce cost in computation and storage without harming performance. However, pruning was until recently only used as a post-processing procedure, while pruning at initialization was believed ineffective [10, 19]. Recently, [5] proposed the lottery ticket hypothesis, showing that for a deep neural network there exist sub-networks, when trained from certain initialization obtained by pruning, performing equally or better than the original model with commensurate convergence rates. Such pairs of sub-networks and initialization are called winning tickets.

This phenomenon indicates it is possible to perform pruning at initialization. However, finding winning tickets still requires iterative pruning and excessive training. Its high cost limits the application of winning tickets. Although [5] shows that winning tickets converge faster than the corresponding full models, it is only observed on small networks, such as a convolutional neural network (CNN) with only a few convolution layers. In this paper, we show that for a variety of model architectures, there consistently exist such sub-networks that converge significantly faster when trained from certain initialization after pruning. We call these boosting tickets.

We observe the standard technique introduced in [5] for identifying winning tickets does not always find boosting tickets. In fact, the requirements are more restrictive. We extensively investigate underlining factors that affect such boosting effect, considering three state-of-the-art large model architectures: VGG-16 [26], ResNet-18 [12], and WideResNet [30]. We conclude that the boosting effect depends principally on three factors: (i) learning rate, (ii) pruning ratio, and (iii) network capacity; we also demonstrate how these factors affect the boosting effect. By controlling these factors, after only one training epoch on CIFAR-10, we are able to obtain 90.88%/90.28% validation/test accuracy (regularly requires >30 training epochs) on WideResNet-34-10 when 80% parameters are pruned.

We further show that the boosting tickets have a practical application in accelerating adversarial training, an effective but expensive defensive training method for obtaining robust models against adversarial examples. Adversarial examples are carefully perturbed inputs that are indistinguishable from natural inputs but can easily fool a classifier [27, 7].

We first show our observations on winning and boosting tickets extend to the adversarial training scheme. Furthermore, we observe that the boosting tickets pruned from a weakly robust model can be used to accelerate the adversarial training process for obtaining a strongly robust model. On CIFAR-10 trained with WideResNet-34-10, we manage to save up to 49% of

*Equal contribution. This work was done during an internship at Baidu USA.

the total training time (including both pruning and training) compared to the regular adversarial training process. Our code is available at https://github.com/boosting-ticket/ticket_robust.

Our contributions are summarized as follows:

1. We demonstrate that there exists boosting tickets, a special type of winning tickets that significantly accelerate the training process while still maintaining high accuracy.
2. We conduct extensive experiments to investigate the major factors affecting the performance of boosting tickets.
3. We demonstrate that winning tickets and boosting tickets exist for adversarial training scheme as well.
4. We show that pruning a non-robust model allows us to find winning/boosting tickets for a strongly robust model, which enables accelerated adversarial training process.

## 2. Background and Related Work

In this section, we give a brief overview of several topics that are closely related to our work.

### 2.1. Network Pruning

Network pruning has been extensively studied as a method for compressing neural networks and reducing resource consumption. [11] propose to prune the weights of neural networks based on their magnitudes. Their pruning method significantly reduces the size of neural networks and has become the standard approach for network pruning. This type of approach is also referred to as unstructured pruning, where the pruning happens at individual weights [10]. In contrast, structured pruning aims to remove whole convolutional filters or channels [19, 13]. While structured pruning often yields better model compression and acceleration without utilizing special hardware or libraries, it can hardly retain the same performance as the full models when the proportion of pruned weights is large. Besides the magnitude-based pruning strategies, there also exist other types of pruning algorithms like dynamic surgery [8], incorporating sparse constraints [32], and optimal brain damage [17]. However, magnitude-based pruning is more stable on different pruning tasks. Therefore, in this work, we focus on unstructured pruning based on magnitudes of the weights.

### 2.2. Lottery Ticket Hypothesis

Surprisingly, recent research has shown it is possible to prune a neural network at the initialization and still reach similar performance as the full model [21, 18]. Within this category, the lottery ticket hypothesis [5] states a randomly-initialized dense neural network contains a sub-network that is initialized such that, when trained in isolation, learns as fast as the original network and matches its test accuracy.

In [5], an iterative pruning method is proposed to find such sub-networks. Specifically, this approach first randomly initializes the model. The initialization is stored separately and the model is trained in the standard manner until convergence. Then a certain proportion of the weights with the smallest magnitudes are pruned while remaining weights are reset to the previously stored initialization and ready to be trained again. This train-prune-reset procedure is performed several times until the target pruning ratio is reached. Using this pruning method, they show the resulting pruned networks can be trained to similar accuracy as the original full networks, which is better than the model with the same pruned structure but randomly initialized.

Although the lottery ticket hypothesis has been extensively investigated in the standard training setting [5, 6], little work has been done in the adversarial training scheme. A recent work [29] even argues lottery ticket hypothesis fails to hold when adversarial training is used. In this paper, we show lottery ticket hypothesis still holds for adversarial training and explain the reason why [29] failed.

One of the limitations of the lottery ticket hypothesis, as pointed in [6], is that winning tickets are found by unstructured pruning which does not necessarily yield faster training or executing time. In addition, finding winning tickets requires training the full model beforehand, which is time-consuming as well, especially considering iterative pruning. In this paper, we are managed to find winning tickets with much lower time consumption while maintaining the superior performance.

### 2.3. Adversarial Examples

Given a classifier $f : \mathcal{X} \to \{1, \dots, k\}$ for an input $\mathbf{x} \in \mathcal{X}$, an adversarial example $\mathbf{x}_{\text{adv}}$ is a perturbed version of $\mathbf{x}$ such that $\mathcal{D}(\mathbf{x}, \mathbf{x}_{\text{adv}}) < \epsilon$ for some small $\epsilon > 0$, yet being mis-classified as $f(\mathbf{x}) \neq f(\mathbf{x}_{\text{adv}})$. $\mathcal{D}(\cdot, \cdot)$ is some distance metric which is often an $\ell_p$ metric, and in most of the literature $\ell_\infty$ metric is considered, so as in this paper.

The procedure of constructing such adversarial examples is often referred to as adversarial attacks. One of the simplest attacks is a single-step method, Fast Gradient Sign Method (FGSM) [7], manipulating inputs along the direction of the gradient with respect to the outputs:

$$
\mathbf{x}_{\text{adv}} = \Pi_{\mathbf{x}+\mathcal{S}}(\mathbf{x} + \alpha(\nabla_{\mathbf{x}}L(\theta, \mathbf{x}, y)))
$$

where $\Pi_{\mathbf{x}+\mathcal{S}}$ is the projection operation that ensures adversarial examples stay in the $\ell_p$ ball $\mathcal{S}$ around $\mathbf{x}$. Although this method is fast, the attack is weak and can be defended easily. On the other hand, its multi-step variant, Projected


Gradient Descend (PGD), is one of the strongest attacks [16, 22]:

$$
\mathbf{x}_{\mathrm{adv}}^{t+1}=\Pi_{\mathbf{x}+\mathcal{S}}\left(\mathbf{x}_{\mathrm{adv}}^{t}+\alpha\left(\nabla_{\mathbf{x}} L(\theta, \mathbf{x}, y)\right)\right)
$$

where $\mathbf{x}$ is initialized with a random perturbation. Since PGD requires to access the gradients for multiple steps, it will incur high computational cost.

On the defense side, currently the most successful defense approach is constructing adversarial examples via PGD during training and add them to the training sets as data augmentation, which is referred to as adversarial training [22].

The motivation behind is that finding a robust model against adversarial examples is equivalent to solving the saddle-point problem $\min _{\theta} \max _{\mathbf{x}^{\prime}: D\left(\mathbf{x}, \mathbf{x}^{\prime}\right)<\epsilon} L\left(\theta, \mathbf{x}^{\prime}, y\right)$. The inner maximization is equivalent to constructing adversarial examples, while the outer minimization performs as a standard training procedure for loss minimization.

One caveat of adversarial training is its computational cost due to performing PGD attacks at each training step. Alternatively, using FGSM during training is much faster but the resulting model is robust against FGSM attacks but vulnerable against PGD attacks [16]. In this paper, we show it is possible to combine the advantages of both and quickly train a strongly robust model benefited from the boosting tickets.

### 2.4. Connecting robustness and compactness

Prior studies have shown success in achieving both compactness and robustness of the trained networks [9, 29, 31, 3, 24, 28]. However, most of them will either incur much higher training cost or sacrifice robustness from the full model. On the contrary, our framework only requires comparable or even reduced training time than standard adversarial training while obtaining similar/higher robust accuracy than the original full network.

## 3. Empirical Study of Boosting Tickets

Setup. As we introduce our methods or findings through experimental results, we first summarize the setup for our experiments. We use BatchNorm [15], weight decay, decreasing learning rate schedules $(\times 0.1$ at $50 \%$ and $75 \%)$, and augmented training data for training models. We try to keep the setting the same as the one used in [5] except we use one-shot pruning instead of iterative pruning. It allows the whole pruning and training process to be more practical in real applications. On CIFAR-10 dataset, we randomly select 5,000 images out of 50,000 training set as validation set and train the models with the rest. The reported test accuracy is measured with the whole testing set.

All of our experiments are run on four Tesla V100s, 10 Tesla P100s, and 10 2080 Tis. For all the time-sensitive experiments like adversarial training on WideResNet-34-10 in Section 4.4, we train each model on two Tesla V100s with data parallelism. For the rest ones measuring the final test accuracy, we use one gpu for each model without parallelism.

We first investigate boosting tickets on the standard setting without considering adversarial robustness. In this section, we show that with properly chosen hyperparameters, we are managed to find boosting tickets on VGG-16 and ResNet that can be trained much faster than the original dense network. Detailed model architectures and the setup can be found in Supplementary Section A.

### 3.1. Existence of Boosting Tickets

To find the boosting tickets, we use a similar algorithm for finding winning tickets, which is briefly described in the previous section and will be detailed here. First, a neural network is randomly initialized and saved in advance. Then the network is trained until convergence, and a given portion of weights with the smallest magnitudes are pruned, resulting in a mask where the pruned weights indicate 0 and remained weights indicate 1. Unless specified, we always prune the smallest $80 \%$ weights, that is the pruning ratio is $80 \%$. We call this train-and-prune step *pruning*. This mask is then applied to the saved initialization to obtain a sub-network, which are the boosting tickets. All of the weights that are pruned (where zeros in the mask) will remain to be 0 during the whole training process. Finally, we can retrain the sub-networks.

The key differences between our algorithm and the one proposed in [5] to find winning tickets are (i) we use a small learning rate for pruning and retrain the sub-network (tickets) with learning rate warm-up from this small learning rate. In particular, for VGG-16 we choose 0.01 for pruning and warmup from 0.01 to 0.1 for retraining; for ResNet-18 we choose 0.05 for pruning and warmup from 0.05 to 0.1 for retraining; (ii) we find it is sufficient to prune and retrain the model only once instead of iterative pruning for multiple times. In Supplementary Section B, we show the difference of boosting effects brought from the tickets found by iterative pruning and one-shot pruning is negligible. Note warmup is also used in [5]. However, they propose to use warmup from small learning rate to a large one during pruning as well, which hinders the boosting effect as shown in the following experiments.

First, we show the existence of boosting tickets for VGG-16 and ResNet-18 on CIFAR-10 in Figure 1 and compare to the winning tickets. In particular, we show the boosting tickets are winning tickets, in the sense that they reach comparable accuracy with the original full models. When compared to the winning tickets, boosting tickets demonstrate equally good performance with a higher convergence rate. Similar results on MNIST can be found in Supplement

tary Section C.

![](./images/867745350910214175_1.jpg)

Figure 1. Validation accuracy during the training process on VGG-16 (a) and ResNet-18 (b) for winning tickets, boosting tickets, and the original full models. In both models, the boosting tickets show faster convergence rate and equally good performance as the winning tickets.

To measure the overall convergence rate, early stopping seems to be a good fit in the literature. It is commonly used to prevent overfitting and the final number of steps are used to measure convergence rates. However, early stopping is not compatible with learning rate scheduling we used in our case where the total number of steps is determined before training.

This causes two issues in our evaluation in Figure 1: (i) Although the boosting tickets reach a relatively high validation accuracy much earlier than the winning ticket, the training procedure is then hindered by the large learning rate. After the learning rate drops, the performance gap between boosting tickets and winning tickets becomes negligible. As a result, the learning rate scheduling obscures the improvement on convergence rates of boosting tickets; (ii) Due to fast convergence, boosting tickets tend to overfit, as observed in ResNet-18 after 50 epochs.

To mitigate these two issues without excluding learning rate scheduling, we conduct another experiment where we mimic the early stopping procedure by gradually increasing the total number of epochs from 20 to 100. The learning rate is still dropped at the 50% and 75% stage. In this way, we can better understand the speed of convergence without worrying about overfitting even with learning rate scheduling involved. In figure 2, we compare the boosting tickets and winning tickets in this manner on VGG-16.

While the first two plots in Figure 2 show the general trend of convergence, the improvement of convergence rates is much clearer in the last four plots. In particular, the validation accuracy of boosting tickets after 40 epochs is already on pair with the one trained for 100 epochs. Meanwhile, the winning tickets fall much behind the boosting tickets until 100 epochs where two finally match.

We further investigate the test accuracy at the end of training for boosting and winning tickets in Table 1. We find the test accuracy of winning tickets gradually increase as we allow for more training steps, while the boosting tickets achieve the highest test accuracy after 60 epochs and start to overfit at 100 epochs.

Table 1. Final test accuracy of winning tickets and boosting tickets trained in various numbers of epochs on VGG-16.

<table>
  <thead>
    <tr>
      <th># of Epochs</th>
      <th>20</th>
      <th>40</th>
      <th>60</th>
      <th>80</th>
      <th>100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Winning (%)</td>
      <td>88.10</td>
      <td>90.03</td>
      <td>90.96</td>
      <td>91.79</td>
      <td>92.00</td>
    </tr>
    <tr>
      <td>Boosting (%)</td>
      <td>91.25</td>
      <td>91.84</td>
      <td>92.13</td>
      <td>92.14</td>
      <td>92.05</td>
    </tr>
  </tbody>
</table>

Summarizing the observations above, we confirm the existence of boosting tickets and state the boosting ticket hypothesis:

A randomly initialized dense neural network contains a sub-network that is initialized such that, when trained in isolation, converges faster than the original network and other winning tickets while matches their performance.

In the following sections, we explain the intuition of boosting tickets and investigate three major components that affect the boosting effects.

### 3.2. Intuition

If we think of the training procedure as a searching process from the initial weights to the optimal point in the parameter space (full path), the training procedure of the pruned model is essentially a path in a projected subspace with pruned parameters being 0 (sub-path). Suppose we want the sub-path to reach the optimal point, it is essential to follow the projection of the full path on the subspace. We follow this intuition, realizing that a smaller learning rate at the initial stage would help the sub-path follow the projected full path by avoiding large deviation. For the same reason, using the same learning rate at the initial stages for both paths also helps align them. In this way, the sub-path can quickly find the correct direction after the initial stage and start to discover a shortcut to the optimal point, resulting in boosting effects.

In Figure 3a, we calculate the relative $\ell_2$ distance between the full and the pruned model weights $w_f$ and $w_p$ generated using winning and boosting tickets $\ell_2 = \frac{\|w_f - w_p\|_2}{\|w_f\|_2}$. The corresponding accuracy is reported in Figure 3b. It is apparent that the distance of boosting tickets is much smaller. A recent paper [4] also confirmed that a linear path, although hard to find, exists between the initialization and the optimal point for pruned neural networks, which supported our intuitions about boosting effects.

### 3.3. Learning Rate

As finding boosting tickets requires alternating learning rates, it is natural to assume the performance of boosting tickets relies on the choice of learning rate. Thus, we extensively investigate the influence of various learning rates.

![](./images/867745350910214175_2.jpg)

Figure 2. Validation accuracy when the total number of epochs are 20, 40, 60, 80, 100 for both the boosting tickets (straight lines) and winning tickets (dash lines) on VGG-16. Plot (a) and (b) contains the validation accuracy for all the training epochs in different scales. Plot (c,d,e,f) compare the validation accuracy between models trained for fewer epochs and the one for 100 epochs.

![](./images/867745350910214175_3.jpg)

Figure 3. On VGG-16, the relative $\ell_2$ distance (a) and the corresponding accuracy (b) between the full models and pruned models .

![](./images/867745350910214175_4.jpg)

Figure 4. The final test accuracy achieved when total number of epochs vary from 20 to 100 on four different tickets. Each line denotes one winning ticket found by learning rate 0.005, 0.01, 0.05, and 0.1 for VGG-16 (a) and ResNet-18 (b).

We use similar experimental settings in the previous section, where we increase the total number of epochs gradually and use the test accuracy as a measure of convergence rates. We choose four different learning rates 0.005, 0.01, 0.05 and 0.1 for pruning to get the tickets. All of the tickets found by those learning rates obtain the accuracy improvement over randomly reinitialized sub-model and thus satisfy the definition of winning tickets (i.e., they are all winning tickets).

As shown in the first two plots of Figure 4, tickets found by smaller learning rates tend to have stronger boosting effects. For both VGG-16 and ResNet-18, the models trained with learning rate 0.1 show the least boosting effects, measured by the test accuracy after 20 epochs of training. On the other hand, training with too small learning rate will compromise the eventual test accuracy at a certain extent. Therefore, we treat the tickets found by learning rate 0.01 as our boosting tickets for VGG-16, and the one found by learning rate 0.05 as for ResNet-18, which converge much faster than all of the rest while achieving the highest final test accuracy.

### 3.4. Pruning Ratio
Pruning ratio has been an important component for winning tickets [5], and thus we investigate its effect on boosting tickets. Since we are only interested in the boosting effect, we use the validation accuracy at early stages as a measure of the strength of boosting to avoid drawing too many lines in the plots. In Figure 5, we show the validation accuracy after the first and fifth epochs of models for different pruning ratios for VGG-16 and ResNet-18.

For both VGG-16 and ResNet-18, boosting tickets always reach much higher accuracy than randomly reinitialized sub-models, demonstrating their boosting effects. When the pruning ratio falls into the range from 60% to 90%, boosting tickets can provide the strongest boosting effects which obtain around 80% and 83% validation accuracy after the first and the fifth training epochs for VGG-16 and obtain 76% and 85% validation accuracy for ResNet-18. On the other hand, the increase of validation accuracy between the first training epoch and the fifth training epoch

![](./images/867745350910214175_5.jpg)

Figure 5. Under various pruning ratios, the changes of validation accuracy after the first and fifth training epoch, trained from the original initialized weights of boosting tickets and randomly reini- tialized ones for VGG-16 (a) and ResNet-18 (b).

become smaller when boosting effects appear. It indicates their convergence starts to saturate due to the large learning rate at the initial stage and is ready for dropping the learning rate.

![](./images/867745350910214175_6.jpg)

Figure 6. Plot (a) and (b) correspond to boosting tickets for various of model widths. Plot (c) and (d) correspond to boosting tickets for various of model depths. While a wider model always boosts faster, deep models have similar boosting effect when the depth is large enough.

### 3.5. Model Capacity
We finally investigate how model capacity, including the depth and width of models, affects the performance of winning tickets in the standard training setting. We use WideResNet [30] either with its depth or width fixed and vary the other factor. In particular, we keep the depth as 34 and increases the width from 1 to 10, comparing their boosting effect. Then we keep the width as 10 and increase the depth from 10 to 34. The changes of validation accuracy of the models are shown in Figure 6.

Overall, Figure 6 shows models with larger capacity have much better performance, though the performance keeps the same when the depth is larger than 22. Notably, we find the largest model WideResNet-34-10 achieves 90.88% validation accuracy after only one training epoch.

## 4. Lottery Ticket Hypothesis for Adversarial Training
Although the lottery ticket hypothesis is extensively studied in [5] and [6], the same phenomenon in adversarial training setting lacks thorough understanding.

In this section, we show two important facts that make boosting tickets suitable for the adversarial scheme: (1) the lottery ticket hypothesis and boosting ticket hypothesis $\ell_\infty$ applicable to the adversarial training scheme; (2) pruning on a weakly robust model allows to find the boosting ticket for a strongly robust model and save training cost.

Particularly, Ye et al. [29] first attempt to apply lottery ticket hypothesis to adversarial settings. However, they concluded that the lottery ticket hypothesis fails to hold in adversarial training via experiments on MNIST. In Section 4.2, our experiments demonstrate that the results they observed are not sufficient to draw this conclusion while we observe that the lottery ticket hypothesis still holds for adversarial training under more restrictive limitations.

### 4.1. Applicability for Adversarial Training
In the following experiment, we use a naturally trained model, that is trained in the standard manner, and two adversarially trained models using FGSM and PGD respectively to obtain the tickets by pruning these models. Then we retrain these pruned models with the same PGD-based adversarial training from the same initialization. In Figure 7, we report the corresponding accuracy on the original validation sets and on the adversarially perturbed validation examples, noted as clean accuracy and robust accuracy. We further train the pruned model from random reinitialization to validate lottery ticket hypothesis.

Unless otherwise stated, in all the PGD-based adversarial training, we keep the same setting as [22]. The PGD attacks are performed in 10 steps with step size $2/255$ (PGD-10). The PGD attacks are bounded by $8/255$ in its $\ell_\infty$ norm. For the FGSM-based adversarial training, the FGSM attacks are bounded by $8/255$.

Both models trained from the boosting tickets obtained with FGSM- and PGD-based adversarial training demonstrate superior performance and faster convergence than the model trained from random reinitialization. This confirms the lottery ticket hypothesis and boosting ticket hypothesis are applicable to adversarial training scheme on both clean accuracy and robust accuracy. More interestingly, the performance of the models pruned with FGSM- and PGD-

![](./images/867745350910214175_7.jpg)

Figure 7. The clean accuracy (a) and robust accuracy (b) of pruned models on the validation set. The models are pruned based on different training methods (natural training, FGSM-based adversarial training, and PGD-based adversarial training). For each obtained boosting ticket, it is retrained with PGD-based adversarial training with 100 training epochs.

based adversarial training are almost the same. This observation suggests it is sufficient to train a weakly robust model with FGSM-based adversarial training for obtaining the boosting tickets and retrain it with stronger attacks such as PGD.

This finding is interesting because FGSM-based adversarial trained models will suffer from label leaking problems as learning weak robustness [16]. In fact, the FGSM-based adversarially trained model from which we obtain our boosting tickets has 89% robust accuracy against FGSM but with only 0.4% robust accuracy against PGD performed in 20 steps (PGD-20). However, Figure 7 shows the following PGD-based adversarial retraining on the boosting tickets obtained from that FGSM-based trained model is indeed robust. Further discussions can be found in Section 5.

### 4.2. Explain the Failure in [29]

In [29], the authors argued that the lottery ticket hypothesis fails to hold in adversarial training via experiments on MNIST. We show they fail to observe winning tickets because the models they used have limited capacity.

We first reproduce their results to show that, in the adversarial setting, small models such as a CNN with two convolutional layers used in [29] can not yield winning tickets when pruning ratio is large. In Figure 9, plot (a) and (b) are the clean and robust accuracy of the pruned models when the pruning ratio is 80%. The pruned model eventually degrades into a trivial classifier where all example are classified into the same class with 11.42%/11.42% valid/test accuracy. On the other hand, when we use VGG-16, as shown in plot (c) and (d), the winning tickets are found again. This can be explained as adversarial training requires much larger model capacity than standard training, which is extensively discussed in [22]. As the result, the pruned small models become unstable during training and yields degrading performance.

As a comparison, the experimental results from Figure 7 indicate when larger models are used, the lottery ticket hypothesis still applies to the adversarial trainig scheme.

### 4.3. Convergence Speedup

We then investigate how boosting tickets can accelerate the adversarial training procedure by conducting the same experiments as in Figure 2 but in the adversarial training setting. The results for validation accuracy and test accuracy are presented in Figure 8 and Table 2 respectively.

In Figure 8, all the training plots are PGD-based adversarial training on the same boosting ticket and learning rate scheduling but with different training epochs. We follow the same procedure described in Section 4.1 to obtain the boosting ticket. Specifically, we train VGG-16 model with 100-epoch FGSM-based adversarial training and then prune 80% of the weight connections. From Figure 8, we can see it is sufficient to train 60 epochs to achieve similar robust accuracy as the full model trained for 100 epochs on our boosting ticket.

Also, in Table 2, we have compared the original full models trained by 100-epoch PGD-based adversarial training with the ones trained on our boosting ticket with different epochs. In general, our models trained on boosting ticket can obtain at most 0.5% higher robust accuracy and regular accuracy than the original one. It indicates (1) the lottery ticket holds for adversarial training as well and (2) our boosting ticket can still enjoy the benefits of both lottery ticket hypothesis and convergence speedup.

Table 2. Best test clean and robust accuracy for PGD-based adversarial training on boosting tickets obtained by FGSM-based adversarial training in various numbers of epochs on VGG-16. Baseline model is obtained by 100-epoch PGD-based adversarial training on original full model.

<table>
  <thead>
    <tr>
      <th># of Epochs</th>
      <th>20</th>
      <th>40</th>
      <th>60</th>
      <th>80</th>
      <th>100</th>
      <th>Baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Robust Acc.</td>
      <td>44.49</td>
      <td>45.27</td>
      <td>45.73</td>
      <td>45.20</td>
      <td>44.53</td>
      <td>44.78</td>
    </tr>
    <tr>
      <td>Clean Acc.</td>
      <td>75.15</td>
      <td>76.28</td>
      <td>76.48</td>
      <td>77.60</td>
      <td>78.07</td>
      <td>77.21</td>
    </tr>
  </tbody>
</table>

### 4.4. Boosting Ticket Applications on adversarially trained WideResNet-34-10

Until now, we have confirmed that boosting tickets exist consistently across different models and training schemes and convey important insights on the behavior of pruned models. However, in the natural training setting, although boosting tickets provide faster convergence, it is not suitable for accelerating the standard training procedure as pruning to find the boosting tickets requires training full models beforehand. On the other hand, the two observations mentioned in Section 4 enable boosting tickets to accelerate adversarial training.

In Table 3, we apply adversarial training to WideResNet-34-10, which has the same structure used in [22], with the

![](./images/867745350910214175_8.jpg)

Figure 8. Validation robust accuracy of pruned models with PGD-based adversarial training on VGG-16 where the total number of epochs are 20, 40, 60, 80, 100 respectively. Plot (a) and (b) show all the results while plot (c), (d), (e), (f) compare each model with the baseline model. The baseline model is obtained by 100-epoch PGD-based adversarial training on the original full model.

![](./images/867745350910214175_9.jpg)

Figure 9. We show clean (a,c) and robust accuracy (b,d) for both winning tickets and randomly initialized weights on LeNet (a,b) and Vgg-16 (c,d) on MNIST with adversarial training.

proposed approach for 40, 70 and 100 epochs and report the best accuracy/robust accuracy under various attacks among the whole training process. In particular, we perform 20-step PGD, 100-step PGD as white-box attacks where the attackers have the access to the model parameters.

It might be suspicious if the resulting models from pruning and adversarial training are indeed robust against strong attacks, as the pruning mask is obtained from a weakly robust model. We conduct extensive experiments on CIFAR-10 with WideResNet-34-10 to evaluate the robustness of this model and compare to the robust model trained with Madry et al's method [22]. Therefore, we include results for C&W attacks [2] and transfer attacks [23, 20] where we attack one model with adversarial examples found by 20-step PGD based on other models.

We find the adversarial examples generated from one model can transfer to another model with a slight decrease on the robust error. It indicates our models and Madry et al's models share adversarial examples and further share decision boundaries.

Table 3. Best test clean accuracy (the first row), robust accuracy (the second to fourth rows), transfer attack accuracy (the middle four rows), and training time for PGD-based adversarial training (the last four rows) on boosting tickets obtained by FGSM-based adversarial training in various of numbers of epochs on WideResNet-34-10. Overall, our adversarial training strategy based on boosting tickets is able to save up to 49% of the total training time while achieving higher robust accuracy compared to the regular adversarial training on the original full model.

<table>
  <tbody>
    <tr>
      <th>
      </th>
      <td colspan="4">
        Test Accuracy(%)
      </td>
    </tr>
    <tr>
      <th>
        Models
      </th>
      <td>
        Madry’s
      </td>
      <td>
        Ours-40
      </td>
      <td>
        Ours-70
      </td>
      <td>
        Ours-100
      </td>
    </tr>
    <tr>
      <th>
        Natural
      </th>
      <td>
        86.21
      </td>
      <td>
        87.72
      </td>
      <td>
        87.85
      </td>
      <td>
        87.35
      </td>
    </tr>
    <tr>
      <th>
        PGD-20
      </th>
      <td>
        50.07
      </td>
      <td>
        50.37
      </td>
      <td>
        50.48
      </td>
      <td>
        49.92
      </td>
    </tr>
    <tr>
      <th>
        PGD-100
      </th>
      <td>
        49.32
      </td>
      <td>
        49.28
      </td>
      <td>
        49.58
      </td>
      <td>
        49.11
      </td>
    </tr>
    <tr>
      <th>
        C&amp;W
      </th>
      <td>
        50.46
      </td>
      <td>
        50.92
      </td>
      <td>
        50.82
      </td>
      <td>
        50.37
      </td>
    </tr>
    <tr>
      <th>
        Madry’s
      </th>
      <td>
        -
      </td>
      <td>
        58.16
      </td>
      <td>
        57.39
      </td>
      <td>
        57.63
      </td>
    </tr>
    <tr>
      <th>
        Ours-40
      </th>
      <td>
        58.69
      </td>
      <td>
        -
      </td>
      <td>
        54.04
      </td>
      <td>
        56.11
      </td>
    </tr>
    <tr>
      <th>
        Ours-70
      </th>
      <td>
        58.77
      </td>
      <td>
        54.60
      </td>
      <td>
        -
      </td>
      <td>
        55.23
      </td>
    </tr>
    <tr>
      <th>
        Ours-100
      </th>
      <td>
        58.61
      </td>
      <td>
        56.62
      </td>
      <td>
        55.20
      </td>
      <td>
        -
      </td>
    </tr>
    <tr>
      <th>
        Pruning Time(s)
      </th>
      <td>
        0
      </td>
      <td>
        15,462
      </td>
      <td>
        15,462
      </td>
      <td>
        15,462
      </td>
    </tr>
    <tr>
      <th>
        Training Time(s)
      </th>
      <td>
        134,764
      </td>
      <td>
        54,090
      </td>
      <td>
        94,796
      </td>
      <td>
        137,105
      </td>
    </tr>
    <tr>
      <th>
        Total Time(s)
      </th>
      <td>
        134,764
      </td>
      <td>
        69,552
      </td>
      <td>
        110,258
      </td>
      <td>
        152,567
      </td>
    </tr>
    <tr>
      <th>
        Ours/Madry’s
      </th>
      <td>
        -
      </td>
      <td>
        0.51
      </td>
      <td>
        0.82
      </td>
      <td>
        1.13
      </td>
    </tr>
  </tbody>
</table>

We report the time consumption for training each model to measure how much time is saved by boosting tickets. We run all the experiments on a workstation with 2 V100 GPUs in parallel. From Table 3 we observe that while our approach requires pruning before training, it is overall faster as it uses FGSM-based adversarial training. In particular, to achieve its best robust accuracy, original Madry

et al.'s training method [22] requires 134,764 seconds on WideResNet-34-10. To achieve that, our boosting ticket only requires 69,552 seconds, including 15,462 seconds to find the boosting ticket and 54,090 seconds to retrain the ticket, saving 49% of the total training time.

## 5. Discussion and Future Work
Not knowledge distillation. It may seem that winning tickets and boosting tickets behave like knowledge distillation [1, 14] where the learned knowledge from a large model is transferred to a small model. This conjecture may explain the boosting effects as the pruned model quickly recover the knowledge from the full model. However, the lottery ticket framework seems to be distinctive to knowledge distillation. If boosting tickets simply transfer knowledge from the full model to the pruned model, then an FGSM-based adversarially trained model should not find tickets that improves the robustness of the sub-model against PGD attacks, as the full model itself is vulnerable to PGD attacks. Yet in Section 4.1 we observe an FGSM-based adversarially trained model still leads to boosting tickets that accelerates PGD-based adversarial training. We believe the cause of boosting tickets requires further investigation in the future.

Accelerate adversarial training. Recently, [25] propose to reduce the training time for PGD-based adversarial training by recycling the gradients computed for parameter updates and constructing adversarial examples. While their approach focuses on reducing the computational time for each epoch, our method focuses more on the convergence rate (i.e., reduce the number of epochs required for convergence). Therefore, our approach is compatible with theirs, making it a promising future direction to combine both to further reduce the training time.

## 6. Conclusion
In this paper, we investigate boosting tickets, sub-networks coupled with certain initialization that can be trained with significantly faster convergence rate. As a practical application, in the adversarial training scheme, we show pruning a weakly robust model allows to find boosting tickets that can save up to 49% of the total training time to obtain a strongly robust model that matches the state-of-the-art robustness. Finally, it is an interesting direction to investigate whether there is a way to find boosting tickets without training the full model beforehand, as it is technically not necessary.

## References
[1] Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In *Advances in neural information processing systems*, pages 2654–2662, 2014.

[2] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In *2017 IEEE Symposium on Security and Privacy (SP)*, pages 39–57. IEEE, 2017.

[3] Guneet S Dhillon, Kamyar Azizzadenesheli, Zachary C Lip-ton, Jeremy Bernstein, Jean Kossaifi, Aran Khanna, and Anima Anandkumar. Stochastic activation pruning for robust adversarial defense. *arXiv preprint arXiv:1803.01442*, 2018.

[4] Utku Evci, Fabian Pedregosa, Aidan Gomez, and Erich Elsen. The difficulty of training sparse neural networks. *arXiv preprint arXiv:1906.10732*, 2019.

[5] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2019.

[6] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. The lottery ticket hypothesis at scale. *arXiv preprint arXiv:1903.01611*, 2019.

[7] Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In *International Conference on Learning Representations*, 2015.

[8] Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. In *Advances In Neural Information Processing Systems*, pages 1379–1387, 2016.

[9] Yiwen Guo, Chao Zhang, Changshui Zhang, and Yurong Chen. Sparse dnns with improved adversarial robustness. In *Advances in neural information processing systems*, pages 242–251, 2018.

[10] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *arXiv preprint arXiv:1510.00149*, 2015.

[11] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In *Advances in neural information processing systems*, pages 1135–1143, 2015.

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016.

[13] Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 1389–1397, 2017.

[14] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.

[15] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. pages 448–456, 2015.

[16] Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale. *arXiv preprint arXiv:1611.01236*, 2016.

[17] Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. In *Advances in neural information processing systems*, pages 598–605, 1990.

[18] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip HS Torr. Snip: Single-shot network pruning based on connection sensitivity. *arXiv preprint arXiv:1810.02340*, 2018.

[19] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016.

[20] Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delving into transferable adversarial examples and black- box attacks. arXiv preprint arXiv:1611.02770, 2016.

[21] Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. arXiv preprint arXiv:1810.05270, 2018.

[22] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learn- ing models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.

[23] Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. Transferability in machine learning: from phenomena to black-box attacks using adversarial samples. arXiv preprint arXiv:1605.07277, 2016.

[24] Vikash Sehwag, Shiqi Wang, Prateek Mittal, and Suman Jana. Towards compact and robust deep neural networks. arXiv preprint arXiv:1906.06110, 2019.

[25] Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free! arXiv preprint arXiv:1904.12843, 2019.

[26] Karen Simonyan and Andrew Zisserman. Very deep convo- lutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[27] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

[28] Arie Wahyu Wijayanto, Jun Jin Choong, Kaushalya Mad- hawa, and Tsuyoshi Murata. Towards robust compressed convolutional neural networks. In 2019 IEEE International Conference on Big Data and Smart Computing (BigComp), pages 1-8. IEEE, 2019.

[29] Shaokai Ye, Kaidi Xu, Sijia Liu, Hao Cheng, Jan-Henrik Lambrechts, Huan Zhang, Aojun Zhou, Kaisheng Ma, Yanzhi Wang, and Xue Lin. Second rethinking of net- work pruning in the adversarial setting. arXiv preprint arXiv:1903.12561, 2019.

[30] Sergey Zagoruyko and Nikos Komodakis. Wide residual net- works. arXiv preprint arXiv:1605.07146, 2016.

[31] Yiren Zhao, Ilia Shumailov, Robert Mullins, and Ross An- derson. To compress or not to compress: Understanding the interactions between adversarial attacks and neural network compression. arXiv preprint arXiv:1810.00208, 2018.

[32] Hao Zhou, Jose M Alvarez, and Fatih Porikli. Less is more: Towards compact cnns. In European Conference on Com- puter Vision, pages 662-677. Springer, 2016.

## A. Model Architectures and Setup

In Table 4, we summarize the number of parameters and parameter sizes of all the model architectures that we eval- uate with including VGG-16 [26], ResNet-18 [12], and the variance of WideResNets [30].

Table 4. Number of parameters and parameter sizes for various architectures.

|  | # of Parameters | Size (MB) |
| --- | --- | --- |
| VGG-16 | 29,975,444 | 114.35 |
| ResNet-18 | 11,173,962 | 42.63 |
| WideResNet-34-10 | 46,160,474 | 176.09 |
| WideResNet-28-10 | 36,479,194 | 139.16 |
| WideResNet-22-10 | 26,797,914 | 102.23 |
| WideResNet-16-10 | 17,116,634 | 65.29 |
| WideResNet-10-10 | 7,435,354 | 28.36 |
| WideResNet-34-5 | 11,554,074 | 44.08 |
| WideResNet-34-2 | 1,855,578 | 7.08 |
| WideResNet-34-1 | 466,714 | 1.78 |

## B. One Shot Pruning

In Figure 10, we track the training of models obtained from both iterative pruning and one shot pruning. We find the performance of both, in terms of the boosting effects and final accuracy, is indistinguishable.

![](./images/867745350910214175_10.jpg)

Figure 10. We compare tickets obtained via iterative pruning and one shot pruning on VGG-16 (left) and ResNet-18 (right). We plot the validation accuracy of models from both approaches and the corresponding randomly initialized models.

## C. Experiments on MNIST

In this section, we report experiment results on MNIST for the standard setting, where we use LeNet with two con- volutions and two fully connected layers for the classifica- tion task.

As for MNIST we do not use learning rate scheduling, early stopping is then used to determine the speed of con- vergence. In Table 5, we report the epochs when early stop- ping happens and the test accuracy to illustrate the existence of boosting tickets for MNIST. While winning tickets con- verge at the 18th epoch, boosting tickets converge at the 11th epoch, indicating faster convergence.

Table 5. The epochs when early stopping happens and the corresponding accuracy for the full model, winning tickets, boosting tickets, and randomly initialized model based on LeNet with two convolutional layers and two fully connected layers.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Full Model</th>
      <th>Winning</th>
      <th>Boosting</th>
      <th>Rand Init</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Early Stopping</th>
      <td>20</td>
      <td>18</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Test Accuracy</th>
      <td>99.18</td>
      <td>99.24</td>
      <td>99.23</td>
      <td>98.97</td>
    </tr>
  </tbody>
</table>