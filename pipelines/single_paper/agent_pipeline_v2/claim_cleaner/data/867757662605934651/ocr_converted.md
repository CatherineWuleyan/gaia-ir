# Quarantine: Sparsity Can Uncover the Trojan Attack Trigger for Free

［#1］
Tianlong Chen¹*, Zhenyu Zhang¹*, Yihua Zhang²*, Shiyu Chang³, Sijia Liu²,⁴, Zhangyang Wang¹

［#2］
¹University of Texas at Austin, ²Michigan State University,
³University of California, Santa Barbara, ⁴MIT-IBM Watson AI Lab
{tianlong.chen, zhenyu.zhang, atlaswang}@utexas.edu,
{zhan1908, liusiji5}@msu.edu, chang87@ucsb.edu

## Abstract

［#3］
Trojan attacks threaten deep neural networks (DNNs) by poisoning them to behave normally on most samples, yet to produce manipulated results for inputs attached with a particular trigger. Several works attempt to detect whether a given DNN has been injected with a specific trigger during the training. In a parallel line of research, the lottery ticket hypothesis reveals the existence of sparse sub-networks which are capable of reaching competitive performance as the dense network after independent training. Connecting these two dots, we investigate the problem of Trojan DNN detection from the brand new lens of sparsity, even when no clean training data is available. Our crucial observation is that the Trojan features are significantly more stable to network pruning than benign features. Leveraging that, we propose a novel Trojan network detection regime: first locating a "winning Trojan lottery ticket" which preserves nearly full Trojan information yet only chance-level performance on clean inputs; then recovering the trigger embedded in this already isolated sub-network. Extensive experiments on various datasets, i.e., CIFAR-10, CIFAR-100, and ImageNet, with different network architectures, i.e., VGG-16, ResNet-18, ResNet-20s, and DenseNet-100 demonstrate the effectiveness of our proposal. Codes are available at https://github.com/VITA-Group/Backdoor-LTH.

［#4］
![](./images/867757662605934651_1.jpg)

［#5］
Figure 1. Overview of our proposal: Weight pruning identifies the ‘winning Trojan ticket’, which can be leveraged for Trojan detection and recovery.

## 1. Introduction

［#6］
Data-driven techniques for artificial intelligence (AI), such as deep neural networks (DNNs), have powered a technological revolution in a number of key application areas in computer vision [6,28,47,66]. However, a critical shortcoming of these pure data-driven learning systems is the lack of test-time and/or train-time robustness: They often learn ‘too well’ during training – so much that (1) the learned model is oversensitive to small input perturbations at testing time (known as evasion attacks) [1,48]; (2) toxic artifacts injected in the training dataset can be memorized during model training and then passed on to the decision-making process (known as poisoning attacks) [26,43]. Methods to secure DNNs against different kinds of ‘adversaries’ are now a major focus in research, e.g., adversarial detection [4,24,75,79,80,86] and robust training [58,83,91]. In this paper, we focus on the study of Trojan attacks (also known as backdoor attacks), the most common threat model on data security [27,69]. In particular, we aim to address the following question:

［#7］
(Q) How does the model sparsity relate to its train-time robustness against Trojan attacks?

［#8］
Extensive research work on model pruning [2,35–37,39,44,49,49,56,59–61,63,65,95] has shown that the weights of an overparameterized model (e.g., DNN) can be pruned (i.e., sparsified) without hampering its generalization ability. In particular, Lottery Ticket Hypothesis (LTH), first developed in [18], unveiled that there exists a subnetwork, when properly pruned and trained, that can even perform better than the original dense neural network. Such a subnetwork is called a winning lottery ticket. In the past, the model sparsity (achieved by pruning) was mainly studied in the non-adversarial learning context, and thereby, the gen-

［#9］
*Equal Contribution.

［#10］
eralization ability is the only metric to define the quality of a sparse network (i.e., a ticket) [8–10,12,18–22,57,90,92]. Beyond generalization, some recent work started to ex- plore the connection between model sparsity and model robustness [31, 34, 70, 84, 88]. However, nearly all ex- isting works restricted model robustness to the prediction resilience against test-time (prediction-evasion) adversarial attacks [15,23,81], hence not addressing our question (Q).

［#11］
To the best of our knowledge, the most relevant works to ours include [40,84], which showed a few motivating re- sults about pruning vs. Trojan attack. Nevertheless, their methods are either indirect [40] or need an ideal assump- tion on the access to the clean (i.e., unpoisoned) finetuning dataset [84]. Specifically, the work [40] showed that it is possible to generate a Trojan attack by modifying model weights. However, there was no direct evidence showing that the Trojan attack is influenced by weight pruning. Fur- ther, the work [84] attempted to promote model sparsity to mitigate the Trojan effect of an attacked model. However, the pruning setup used in [84] has a deficiency: It was as- sumed that finetuning the pruned model can be conducted over the clean validation dataset. In practice, such an as- sumption is too ideal for achieving if the user has no access to the benign dataset. This assumption also prevents us from understanding the true cause of Trojan mitigation, since the possible effect of model sparsity is entangled with finetun- ing on clean data.

［#12］
Different from [40,84], we aim to tackle the research question (Q) in a more practical backdoor scenario - without any access to clean training samples. Moreover, our work bridges LTH and backdoor model detection by (i) identi- fying a crucial subnetwork (that we call 'winning Trojan ticket'; see Fig. 1) with almost unimpaired backdoor infor- mation and near-random clean-set performance; (ii) recov- ering the trigger with the subnetwork and then detecting the backdoor model. We summarize our contributions below:

［#13］
- We establish the connection between model sparsity and Trojan attack by leveraging LTH-oriented iterative magnitude pruning (IMP). Assisted by LTH, we pro- pose the concept of Trojan ticket to uncover the prun- ing dynamics of the Trojan model.
- We reveal the existence of a 'winning Trojan ticket', which preserves the same Trojan attack effectiveness as in the unpruned model. We propose a linear mode connectivity (LMC)-based Trojan score to detect such a winning ticket along the pruning path.
- We show that the backdoor feature encoded in the win- ning Trojan ticket can be used for reverse engineering of Trojan attack for 'free', i.e., with no access to clean training samples nor threat model information.
- We demonstrate the effectiveness of our proposal in detecting and recovering Trojan attacks with vari- ous poisoned DNNs using diverse Trojan trigger pat- terns (including basic backdoor attack and clean-label attack) across multiple network architectures (VGG,ResNet, and DenseNet) and datasets (CIFAR-10/100 and ImageNet). For example, our Trojan recovery method achieves 90% attack performance improve- ment over the state-of-the-art Trojan attack estimation approach if the clean-label Trojan attack [94] is used by the ground-truth adversary.

## 2. Related Works

### Pruning and lottery tickets hypothesis (LTH).
［#14］
Prun- ing removes insignificant connectivities in deep neural net- works [35,49]. Generally, its overall pipeline consists of the following one-shot or iterative cycles: (1) training the dense neural networks for several epochs; (2) eliminating redun- dant weights with respect to certain criteria; (3) fine-tuning derived sparse networks to recover accuracy. Puning ap- proaches can be roughly categorized the magnitude-based and the optimization-based. The former zeroes out a por- tion of model weights by thresholding their statistics such as weight magnitudes [36,44], gradients [59], Taylor coef- ficients [37,49,60,61], or hessian [87]. The latter usually in- corporates sparsity-promoting regularization [39,56,95] or formulates constrained optimization problems [2,33,63,65].

［#15］
As a new rising sub-field in pruning, the lottery ticket hy- pothesis (LTH) [18] advocates that dense neural networks contain a sparse subnetwork (a.k.a. winning ticket) ca- pable of training from scratch (i.e., the same random ini- tialization) to match the full performance of dense mod- els. Later investigations point out [19,67] that the origi- nal LTH can not scale up to larger networks and datasets unless leveraging the weight rewinding techniques [19,67]. LTH and its variants have been widely explored in plenty of fields [8–10, 12, 21, 22, 57, 90, 92] like image genera- tion [7,12,45] and natural language processing [9,21].

### Backdoor robustness - Trojan attacks and defenses.
［#16］
Trojan attacks. Various Trojan (or backdoor) attacks on deep learning models have been designed recently. The at- tack features stealthiness since the attacked model will be- have normally on clean images but classify images stamped with a trigger from any source class into the maliciously chosen target class. One of the mainstream Trojan attacks is trigger-driven. As the most common way to launch an at- tack, the adversary injects an attacker-specific trigger (e.g. a local patch) into a small fraction of training pictures and maliciously label them to the target class [11,30,52,54,55].

［#17］
Another category of backdoor attack, known as clean- label backdoor attack [64,71,96], keeps the ground-truth la- bel of the poisoned samples consistent with the target labels. Instead of manipulating labels directly, it perturbs the data of the target class through adversarial attacks [58], so that the representations learned by the model are distorted in the

［#17］
embedded space towards other victim or base classes. Thus, label perturbation becomes implicit and less detectable.
Trojan defenses. To alleviate the backdoor threat, numerous defense methods can be grouped into three paradigms: (1) data pre-processing, (2) model reconstruction, and (3) trigger recovery. The first category introduces a pre-processing module before feeding the inputs into the network, changing the pattern of the potential trigger attached or hidden in the samples [16,77,78]. The second class aims at removing the learned trigger knowledge by manipulating the Trojan model, so that the repaired model will function quite well even in the presence of the trigger [51,93].

［#18］
This paper focuses on the third category, the trigger recovery-based defenses. The rationale behind this category is to detect and synthesize the backdoor trigger at first, followed by the second step to suppress the effect of the synthesized trigger. Some previous research detects and mitigates backdoor models based on abnormal neuron responses [4,80,86], feature representation [75], entropy [24], evolution of model accuracy [72]. Utilizing clean testing images, Neural Cleanse (NC) [79] obtains potential trigger patterns and calculates minimal perturbation that causes misclassification toward every putative incorrect label. Backdoor model detection is then completed by the MAD outlier detector, which identifies the class with the remarkably small minimal perturbation among all the classes. NC shows that the recovered trigger resembles the original trigger in terms of both shape and neuron activation. Similar ideas were explored in [5,32,51,85]. However, the recovered triggers from the aforementioned methods suffer from occasional failures in detecting the true target class.

［#19］
Backdoor meets pruning. Fine-pruning serves as a classical defense approach [40,53], which trims down the "corrupted" neurons to destroy and get rid of Trojan patterns. Note that these investigations do not explore the weight sparsity. A follow-up work [84] measures the sensitivity of Trojan DNNs by introducing adversarial weight perturbations, and then prunes selected sensitive neurons to purify the injected backdoor. Another recent work [89] examines the vanilla LTH under the context of federated learning. They demonstrate that LTH is also vulnerable to backdoor attacks, and offer a federated defense by using the ticket's structural similarity – a totally different focus from ours.

## 3. Preliminaries and Problem Setup
［#20］
This section provides a brief background on the Trojan attack and model pruning. We then motivate and present the problem of our interest, aiming at exploring and exploiting the relationship between weight pruning and Trojan attacks.

［#21］
Trojan attack and Trojan model. Trojan attack is one of the most commonly-used data poisoning attacks [29]: It manipulates a small portion of training data, including their features by injecting a Trojan trigger (e.g., a small patch or sticker on images) and/or their labels modified towards the Trojan attack targeted label. The Trojan attack then serves as a 'backdoor' and enforces a spurious correlation between the Trojan trigger and the model training. The resulting model is called Trojan model, which causes the backdoordesignated incorrect prediction if the trigger is present at the testing time, otherwise, it behaves normally. In Fig.2, we demonstrate an example of the misbehavior of a Trojan model in image classification.

［#22］
![](./images/867757662605934651_2.jpg)
［#23］
Figure 2. Overview of Trojan attack.

［#24］
It is worth noting that the Trojan attack is different from the test-time adversarial attack, a widely-studied threat model in adversarial learning [48,58]. There exist three key differences. (i) Trojan attack occurs at the training time through data poisoning. (ii) Trojan model exhibits the input-agnostic adversarial behavior at the testing time only if the Trojan trigger is present at an input example (see Fig.2). (iii) Trojan model is stealthy for the end user since the latter has no prior knowledge on data poisoning.

### Model pruning and lottery ticket hypothesis (LTH).
［#25］
Model pruning aims at extracting a sparse sub-network from the original dense network without hampering the model performance. LTH, proposed in [18], formalized a model pruning pipeline so as to find the desired sub-network, which is called 'winning ticket'. Formally, let $f(x;\theta)$ denote a neural network with input $x$ and model parameter $\theta \in \mathbb{R}^d$. And let $m \in \{0,1\}^d$ denote a binary mask on top of $\theta$ to encode the locations of pruned weights (corresponding to zero entries in $m$) and unpruned weights (corresponding to non-zero entries in $m$), respectively. The resulting pruned model (termed as a 'ticket') can then be expressed as $(m \odot \theta)$, where $\odot$ is the elementwise product. LTH suggests the following pruning pipeline:
1.  Initialize a neural network $f(x;\theta_0)$, where $\theta_0$ is a random initialization. And initialize a mask $m$ of all 1s.
2.  Train $f(x;m \odot \theta_0)$ to obtain learned parameters $\theta$ over the dataset $\mathcal{D}$.
3.  Prune $p\%$ parameters in $\theta$ per magnitude. Then, create a new sparser mask $m$ from the old one.
4.  Reset the remaining parameters to their values in $\theta_0$, creating the new sparse network $(m \odot \theta_0)$. Then, go back to ② and repeat.

［#26］
The above procedure forms the iterative magnitude pruning (IMP), which repeatedly trains, prunes, and resets the network over $n$ rounds. LTH suggests that each round

［#26］
prunes $p^{1/n}\%$ of the weights on top of the previous round (In our case, $p=20\%$ same as [18]). The key insight from LTH is: There exists a winning ticket, e.g., $(m \odot \theta_0)$, which when trained in isolation, can match or even surpass the test accuracy of the well-trained dense network [18].

［#27］
Problem setup. Model pruning has been widely studied in the context of non-poisoned training scenarios. However, it is less explored in the presence of poisoned training data. In this paper, we ask:

［#28］
How is weight pruning of a Trojan model intertwined with Trojan attack ability if the pruner has no access to clean training samples and is blind to attack knowledge?

［#29］
To formally set up our problem, let $\mathcal{D}_\mathrm{p}$ denote the possibly poisoned training dataset. By LTH pruning, the sparse mask $m$ and the finetuned model parameters $\theta$ (based on $m$) are learned from $\mathcal{D}_\mathrm{p}$, without having access to clean data. Thus, different from the 'winning ticket' found from LTH over the clean dataset $\mathcal{D}$, we call the ticket, i.e., the sparse model $(m \odot \theta)$, Trojan ticket; see more details in the next section. We then investigate how the benign and adversarial performance of Trojan tickets varies against the pruning ratio $p\%$. The benign performance of a model will be measured by the standard accuracy (SA) against clean test data. And the adversarial performance of a model will be evaluated by the attack success rate (ASR) against poisoned test data using the train-time Trojan trigger. ASR is given by the ratio of correctly mis-predicted test data (towards backdoor label) over the total number of test samples.

## 4. Uncover Trojan Effect from Sparsity

［#30］
In this section, we begin by presenting a motivating example to demonstrate the unusual pruning dynamics of Trojan ticket (i.e., pruned model over the possibly poisoned training data set $\mathcal{D}_\mathrm{p}$). We show that sparsity, together with the approach of linear model connectivity (LMC) [19], can be used for Trojan detection and recovery.

［#31］
Pruning dynamics of Trojan ticket: A warm-up. Throughout the paper, we will follow the LTH-based pruning method to find the pruning mask $m$. In order to preserve the potential Trojan properties, we will not reset the non-zero parameters in $\theta$ to the random initialization $\theta_0$ when a desired sparsity ratio $p\%$ is achieved at the last iteration of IMP. Recall that the resulting subnetwork $(m \odot \theta)$ is called a Trojan ticket. To examine the sensitivity of the Trojan ticket to the possibly poisoned dataset $\mathcal{D}_\mathrm{p}$, we then create a $k$-step finetuned Trojan ticket $(m \odot \theta^{(k)})$, where $\theta^{(k)}$ is the $k$-step finetuning of $\theta$ given $m$ under $\mathcal{D}_\mathrm{p}$. Our rationale behind these two kinds of tickets is elaborated on below.

［#32］
- If there does not exist a Trojan attack, then the above two tickets should share similar pruning dynamics. As will be evident later, this could be justified by LMC (linear model connectivity).
- If there exists Trojan attack, then the two tickets result in substantially distinct adversarial performance. Since Trojan model weights encode the spurious correlation with the Trojan trigger [79,80], pruning without finetuning could characterize the impact of sparsity on the Trojan attack, in contrast to pruning with finetuning over $\mathcal{D}_\mathrm{p}$.

［#33］
![](./images/867757662605934651_3.jpg)

［#34］
Figure 3. The pruning dynamics of Trojan ticket (dash line) and 10-step finetuned ticket (solid line) on CIFAR-10 with ResNet-20s and gray-scale basic backdoor trigger [30]. For comparison, the Trojan score (2) is also presented.

［#35］
In Fig. 3, we present a warm-up example to illustrate the pruning dynamics of the Trojan ticket $(m \odot \theta)$ and its $k$-step finetuned version $(m \odot \theta^{(k)})$, where we select $k=10$ (see the choice of $k$ in Appendix. A2). As we can see, there exists a peak Trojan ticket in the extreme sparsity regime $(p\% > 99.97\%)$, with the preserved Trojan performance (measured by Trojan score that will be defined later). The key takeaway from Fig. 3 is that the performance stability of the Trojan ticket $(m \odot \theta)$ and the $k$-step finetuned ticket $(m \odot \theta^{(k)})$ can be used to indicate the Trojan attack effect.

［#36］
Trojan detection by LMC. To quantify the stability of Trojan tickets, we propose to use the tool of linear model connectivity (LMC) [17,25], which returns the error barrier between two neural networks along a linear path. In the context of model pruning, the work [19] showed two sparse neural networks found by IMP could be linearly connected even if they suffer different optimization 'noises', e.g., different choices of initialization, data batch, and optimization step. Spurred by the aforementioned work, we adopt LMC to measure the stability of the Trojan ticket $(m \odot \theta)$ v.s. the $k$-step finetuned Trojan ticket $(m \odot \theta^{(k)})$.

［#37］
Formally, let $\mathcal{E}(\phi)$ denote the training error of a model $\phi$. Given two neural networks $\phi_1$ and $\phi_2$, LMC then defines the error barrier between $\phi_1$ and $\phi_2$ along a linear path below:

［#37］
$$
e_{\text{sup}}(\phi_1, \phi_2) = \max_{\alpha \in [0,1]} \mathcal{E}(\alpha \phi_1 + (1-\alpha) \phi_2), \tag{1}
$$

［#37］
which is the highest error when linearly interpolating between the models $\phi_1$ and $\phi_2$. If we set $\phi_1 = m \odot \theta$ and

［#37］
$\phi_2 = m \odot \theta^{(k)}$, then LMC yields the following stability metric, termed **Trojan score**:

［#37］
$$
\begin{aligned}
\mathcal{S}_{\text{Trojan}}= & e_{\sup }(m \odot \theta, m \odot \theta^{(k)}) \\
& -\frac{\mathcal{E}(m \odot \theta)+\mathcal{E}(m \odot \theta^{(k)})}{2},
\end{aligned} \tag{2}
$$

［#37］
where the second term is used as an error baseline of using two pruned models. As suggested by [19], if there exists no Trojan attack during model pruning, then $\mathcal{E}(m \odot \theta) \approx$ $\mathcal{E}(m \odot \theta^{(k)}) \approx e_{\sup }(m \odot \theta, m \odot \theta^{(k)})$, leading to $\mathcal{S}_{\text{Trojan}} =$ 0. Assisted by model pruning and LMC, we can then use the Trojan score (2) to detect the existence of a Trojan attack. This gives a novel Trojan detector without resorting to any clean data, which has been known as a grand challenge in Trojan AI¹. However, most importantly, the relationship between model pruning and Trojan attack can be established through Trojan ticket and its Trojan score $\mathcal{S}_{\text{Trojan}}$.

［#38］
As shown in Fig. 3, the sparse network $(m \odot \theta)$ with the **peak** Trojan score $\mathcal{S}_{\text{Trojan}}$ maintains the highest ASR (attack success rate) in the extreme pruning regime. We term such a Trojan ticket as the **winning Trojan ticket**.

［#39］
**Reverse engineering of Trojan attack.** We next ask if the winning Trojan ticket better memorizes the Trojan trigger than the original dense model. To tackle this problem, we investigate the task of reverse engineering of Trojan attack [32,79,80], which aims to recover the Trojan targeted label and/or the Trojan trigger from a Trojan model.

［#40］
Formally, let $x'(z, \delta)=(1-z) \odot x+z \odot \delta$ denote the poisoned data with respect to (w.r.t.) an example $x \in \mathbb{R}^{n}$, where $\delta \in \mathbb{R}^{n}$ denotes the element-wise perturbations, and $z \in \{0,1\}^n$ is a binary mask to encode the positions where a Trojan trigger is placed. Given a Trojan model $\phi$, our goal is to optimize the Trojan attack variables $(z, \delta)$ so as to unveil the properties of the ground-truth Trojan attack. Following [32,79,80], this leads to the optimization problem

［#40］
$$
\min _{z \in\{0,1\}^{n}, \delta} \mathbb{E}_{x}[\ell_{\text{atk}}(x'(z, \delta) ; \phi, t)]+\gamma h(z, \delta), \tag{3}
$$

［#40］
where $x$ denotes the base images (that can be set by noise images) to be perturbed, $\ell_{\text{atk}}(x'; \phi, t)$ denotes the targeted attack loss, with the perturbed input $x'$, victim model $\phi$, and the targeted label $t$, $h$ is a certain regularization function that controls the sparsity of $z$ and the smoothness of the estimated Trojan trigger $z \odot \delta$, and $\gamma > 0$ is a regularization parameter. In (3), we specify $\ell_{\text{atk}}$ as the C&W targeted attack loss [3] and $h$ as the regularizer used in [32]. To solve the problem (3), the convex relaxation approach is used similar to [80], where the binary variable $z$ is relaxed to its convex probabilistic hull. Once the solution $(z^*, \delta^*)$ to problem (3) is obtained, the work [79] showed that the **Trojan attack targeted label** can be deduced from the label $t$ associated with the least norm of the recovered Trojan trigger $z^* \odot \delta^*$. That is, $t_{\text{Trojan}} = \arg \min_{t} \|z^*(t) \odot \delta^*(t)\|_1$, where the dependence of $z^*$ and $\delta^*$ on the label choice $t$ is shown explicitly. Sec. 5 will show that if we set the victim model in (3) by the winning Trojan ticket, then it yields a much higher accuracy of estimating the Trojan attack targeted label than baseline approaches.

## 5. Experiments
### 5.1. Implementation details
［#41］
**Networks and datasets.** We consider a broad range of model architectures including DenseNet-100 [42], ResNet-20s [38], ResNet-18 [38], and VGG-16 [73] on diverse datasets such as CIFAR-10 [46], CIFAR-100 [46], and Restricted ImageNet (R-ImageNet) [13,76], with 9 classes.

［#42］
**Configuration of Trojan attacks.** To justify the identified relationship between the Trojan model and weight sparsity, we consider two kinds of Trojan attacks across different model architectures and datasets as described above. The studied threat models include (i) *Basic Backdoor Attack*, also known as BadNet-type Trojan attack [29], and (ii) *Clean Label Backdoor Attack* [94], which have been commonly used as a benchmark for backdoor and data poisoning attacks [69]. Their difference lies in that Trojan-(i) adopts the heuristics-based data poisoning strategy and Trojan-(ii) is crafted using an optimization procedure and contains a less noticeable trigger pattern. For both attacks, the Trojan trigger (with size $5 \times 5$ for CIFAR-10/100 and $64 \times 64$ for R-ImageNet) is placed in the upper right corner of the target image and is set using either a gray-scale square like [29] or an RGB image patch like [68]. And the training data poisoning ratio is set by 1% and the Trojan targeted label is set by class 1. We refer readers to Sec. A1 for more detailed hyperparameter setups of the above Trojan attacks.

［#43］
**Training and evaluation.** For CIFAR-10/100, we train networks for 200 epochs with a batch size of 128. An SGD optimizer is adopted with a momentum of 0.9 and a weight decay ratio of $5 \times 10^{-4}$. The learning rate starts from 0.1 and decay by 10 times at 100 and 150 epoch. For R-ImageNet, we train each network for 30 epochs and 1024 batch size, using an SGD optimizer with 0.9 momentum and $1 \times 10^{-4}$ weight decay. The initial learning rate is 0.4 with 2 epochs of warm-up and then decline to $\frac{1}{10}$ at 8, 18, and 26 epoch. All models have achieved state-of-the-art SA (standard accuracy) in the absence of the Trojan trigger. To measure the performance of Trojan backdoor injection, we test the SA of each model on a clean test set and ASR (attack success rate) on the same test set in the presence of Trojan trigger.

［#44］
In the task of reverse engineering Trojan attacks, we solve the problem (3) following the optimization method

---
［#37］
¹https://www.iarpa.gov/index.php/research-programs/trojai

［#45］
used in [79] which includes two stages below. First, problem (3) is solved under each possible label choice of $t$. Second, the Trojan targeted label is determined by the label associated with the least $\ell_1$-norm of the recovered Trojan trigger $\| z \odot \delta \|_1$. **By default, we use 100 noise images** (generated by Gaussian distribution $\mathcal{N}(0,1)$) to specify the base images $x$ in (3). For comparison, we also consider the specification of base images using 100 clean images drawn from the benign data distribution.

［#46］
![](./images/867757662605934651_4.jpg)

［#47］
Figure 4. The pruning dynamics and Trojan scores on CIFAR-10 with ResNet-20s using the RGB Trojan triggers. The peak Trojan score precisely characterizes the winning Trojan ticket. Results of clean-label Trojan triggers are presented in Appendix A2.

### 5.2. Experiment results
#### 5.2.1 Existence of winning Trojan ticket
［#48］
We investigate the pruning dynamics of a Trojan ticket ($m \odot$ $\theta$) (i.e., the pruned Trojan network built upon the original model $\theta_{\text{ori}}$ with sparse mask $m$ and model weights $\theta$) versus the pruning ratio $p\%$. Following Sec. 4, we also examine the $k$-step finetuned Trojan ticket $(m \odot \theta^{(k)})$. Throughout the paper, we choose $k=10$ to best locate the winning Trojan tickets as demonstrated in the ablation of Appendix A2. We remark that the finetuner has only access to the poisoned dataset rather than an additional benign dataset.

［#49］
In Fig. 4, we demonstrate the SA and ASR performance of the Trojan ticket and its finetuned ticket versus the network sparsity. Recall that SA and ASR characterize the benign accuracy and the Trojan attack performance of a model, respectively. For comparison, we also present the LMC-based Trojan score (2). Our key finding, consistent with Fig. 3, is that in the extreme pruning regime, there exists a winning Trojan ticket with the peak Trojan score across multiple Trojan attack types, datasets, and neural network architectures.

［#50］
The more specific observations and insights of Fig. 4 are elaborated on below. As we can see, in the non-extreme sparsity regime ($p\% < 90\%$), the Trojan ticket and its fine-tuned variant preserve both the benign performance (SA) and the Trojan performance (ASR) of the dense model $\theta_{\text{ori}}$ (associated with the leftmost pruning point in Fig. 4). This implies that the promotion of non-extreme sparsity in $\theta_{\text{ori}}$ cannot mitigate the Trojan effect, and the resulting Trojan ticket behaves similarly to the normally pruned network by viewing from its benign performance. However, in the extreme sparsity regime ($p\% > 99$), the pure sparsity promotion leads to the ASR performance significantly different from SA, e.g., ASR $=94.49\%$ vs. SA $=11.38\%$ in the top plot of Fig. 4. And the phenomenon is weakened after fine-tuning the Trojan ticket, as indicated by the reduced ASR in Fig. 4. These observations yield two implications. First, the Trojan model exhibits a 'fingerprint' in the extreme sparsity regime, where ASR is preserved but SA reduces to the nearly-random performance (because of this extreme pruning level). Such a fingerprint is called winning Trojan ticket termed in Sec. 4 due to its high ASR. Second, this superior Trojan behavior is not well-maintained after the weight fine-tuning, suggesting that the Trojan effect is mostly encoded by the sparse pattern of the winning Trojan ticket. We also visualize the loss landscape of winning Trojan tickets in Appendix A2. Last but not the least, the winning Trojan ticket is associated with the peak Trojan score (2), which can thus be leveraged as a powerful tool for Trojan detection.

#### 5.2.2 Backdoor properties of winning Trojan ticket
［#51］
In Fig. 5, we next investigate the backdoor properties embedded in the winning Trojan ticket, which is identified by the peak Trojan score (see examples in Fig. 4). Our key findings are summarized below. (i) Among dense and various sparse networks, the winning Trojan ticket needs the minimum perturbation to reverse engineering of the Trojan targeted label $t_{\text{Trojan}}$ found by (3). The performance of our approach outperforms the baseline method, named Neural Cleanse (NC) [79]. (ii) The recovered trigger pattern $(z^{*}(t_{\text{Trojan}}) \odot \delta^{*}(t_{\text{Trojan}}))$ using (3) indeed yields a valid Trojan attack of high ASR. (iii) By leveraging the winning Trojan ticket, we can achieve the Trojan trigger recovery for 'free'. That is, the high-quality Trojan attack can be recovered using only 'noise image inputs' when solving the problem (3). We highlight that the aforementioned findings (i)-(iii) are consistent across different Trojan attack types, datasets, and model architectures.

［#52］
In each sub-plot of Fig. 5, we demonstrate the $\ell_1$ norm of the recovered Trojan trigger $(z^{*}(t) \odot \delta^{*}(t))$ by solving the problem (3) at different specifications of the class label $t$ and the victim model $\phi$. We enumerate all the possible choices of $t$ and examine three types of victim models, given by the winning Trojan ticket (with the peak Trojan score), the originally dense Trojan model (used by NC [79]), and the non-Trojan dense model (that is normally trained over the benign training dataset). Multiple sub-

［#53］
![](./images/867757662605934651_5.jpg)

［#54］
Figure 5. The $\ell_1$ norm values of recovered Trojan triggers for all labels. The plot title signifies adopted network architecture, trigger type, and the images used for reverse engineering on CIFAR-10. Class "1" in the red box is the true (or oracle) target label for Trojan attacks. $\boldsymbol{\checkmark}$/$\boldsymbol{\sfx{X}}$ indicates whether or not the detected label with the least $\ell_1$ norm matches the truth target label.

［#55］
plots of Fig. 5 correspond to our experiments across different model architectures, different ground-truth Trojan trigger types, and different input images used to solve problem (3). It is clear from Fig. 5 that in all experiments, our identified Trojan ticket yields the least perturbation norm of the recovered Trojan trigger at the Trojan targeted label (i.e., $t = t_{\text{Trojan}}$). The rationale behind the minimum perturbation criterion is that if there exists a backdoor 'shortcut' in the Trojan model (with high ASR), then an input image only needs the very tiny perturbation optimized towards $t = t_{\text{Trojan}}$ [79]. As a result, one can detect the target label by just monitoring the perturbation norm. Moreover, we observe that the baseline NC method (associated with the dense Trojan model) [79] lacks stability. For example, it fails to identify the correct target label at the use of the RGB trigger (e.g., Fig. 5 [d]). Further, we note that the non-Trojan model does not follow the minimum perturbation-based detection rule.

［#56］
In Tab. 1, we present the attack performance (ASR) of the recovered Trojan trigger versus the different choice of the ground-truth Trojan trigger type (i.e., gray-scale, RGB, and clean-label trigger). As we can see, even if the baseline NC method (associated with the dense Trojan model) can correctly identify the target label, the quality of the recovered Trojan trigger is poor, justified by its much lower ASR than ours. In particular, when the clean-label attack was used in the Trojan model, our approach (by leveraging the winning Trojan ticket) leads to over 90% ASR improvement. In Tab. 2, we present the ASR of the recovered Trojan trigger under different model architectures and datasets. Consistent with Tab. 1, the use of the winning Trojan ticket significantly outperforms the baseline approach, not only in ASR but also in the correctness of the detected target label based on the minimum perturbation criterion.

［#57］
<table>
<caption>Table 1. Performance of recovered triggers with ResNet-20s on CIFAR-10 across diverse Trojan triggers, including gray-scale, RGB, and clean-label triggers. $\boldsymbol{\checkmark}$/$\boldsymbol{\sfx{X}}$ mean the detected label is matched/unmatched with the true target label.</caption>
<thead>
<tr>
<th>Gray-scale Trigger</th>
<th>(Detected, $\ell_1$)</th>
<th>ASR</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 196.8) $\boldsymbol{\checkmark}$</td>
<td>71.4%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 68.0) $\boldsymbol{\checkmark}$</td>
<td>$\boldsymbol{91.2\%}$</td>
</tr>
<tr>
<th>RGB Trigger</th>
<th>(Detected, $\ell_1$)</th>
<th>ASR</th>
</tr>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 78.7) $\boldsymbol{\checkmark}$</td>
<td>48.0%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 29.8) $\boldsymbol{\checkmark}$</td>
<td>$\boldsymbol{99.6\%}$</td>
</tr>
<tr>
<th>Clean-label Trigger</th>
<th>(Detected, $\ell_1$)</th>
<th>ASR</th>
</tr>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 48.6) $\boldsymbol{\checkmark}$</td>
<td>9.6%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 14.0) $\boldsymbol{\checkmark}$</td>
<td>$\boldsymbol{99.8\%}$</td>
</tr>
</tbody>
</table>

［#58］
In Tab. 3, we examine how the choice of base images in the Trojan recovery problem (3) affects the estimated Trojan quality. In contrast to the use of 100 noise images randomly drawn from the standard Gaussian distribution, we also consider the case of using 100 clean images drawn from the benign data distribution. As we can see, our approach based on the winning Trojan ticket yields superior Trojan recovery performance to the baseline method in both settings of base images. Most importantly, the quality of our recovered Trojan trigger is input-agnostic: The 99.6% ASR is achieved

［#59］
<table>
<caption>Table 2. Performance of recovered triggers with RGB Trojan attack across diverse combinations of network architectures and datasets, i.e., (Vgg-16, CIFAR-10), (ResNet-20s, CIFAR-100), (ResNet-18, R-ImageNet).</caption>
<tbody>
  <tr>
    <th>(VGG-16, CIFAR-10)</th>
    <th>(Detected, $\ell_1$)</th>
    <td>ASR</td>
  </tr>
  <tr>
    <th>Dense baseline [32]</th>
    <td>(“1”, 83.3) ✔️</td>
    <td>33.6%</td>
  </tr>
  <tr>
    <th>Winning Trojan ticket</th>
    <td>(“1”, 15.0) ✔️</td>
    <td>100.0%</td>
  </tr>
  <tr>
    <th>(ResNet-20s, CIFAR-100)</th>
    <th>(Detected, $\ell_1$)</th>
    <td>ASR</td>
  </tr>
  <tr>
    <th>Dense baseline [32]</th>
    <td>(“1”, 149.9) ✔️</td>
    <td>13.8</td>
  </tr>
  <tr>
    <th>Winning Trojan ticket</th>
    <td>(“1”, 132.7) ✔️</td>
    <td>98.7</td>
  </tr>
  <tr>
    <th>(ResNet-18, R-ImageNet)</th>
    <th>(Detected, $\ell_1$)</th>
    <td>ASR</td>
  </tr>
  <tr>
    <th>Dense baseline [32]</th>
    <td>(“9”, 13.9) ❌</td>
    <td>9.8</td>
  </tr>
  <tr>
    <th>Winning Trojan ticket</th>
    <td>(“1”, 193.1) ✔️</td>
    <td>98.7</td>
  </tr>
</tbody>
</table>

［#60］
<table>
<caption>Table 3. Performance of recovered triggers with random noise images (‘free’) v.s. benign clean images. The RGB Trojan attack on CIFAR-10 and ResNet-20s are used for the reverse engineering.</caption>
<tbody>
  <tr>
    <th>Noise Images (‘Free’)</th>
    <th>(Detected, $\ell_1$)</th>
    <td>ASR</td>
  </tr>
  <tr>
    <th>Dense baseline [32]</th>
    <td>(“1”, 78.7) ✔️</td>
    <td>48.0%</td>
  </tr>
  <tr>
    <th>Winning Trojan ticket</th>
    <td>(“1”, 29.8) ✔️</td>
    <td>99.6%</td>
  </tr>
  <tr>
    <th>Clean Images</th>
    <th>(Detected, $\ell_1$)</th>
    <td>ASR</td>
  </tr>
  <tr>
    <th>Dense baseline [32]</th>
    <td>(“1”, 174.6) ✔️</td>
    <td>72.6%</td>
  </tr>
  <tr>
    <th>Winning Trojan ticket</th>
    <td>(“1”, 40.4) ✔️</td>
    <td>99.8%</td>
  </tr>
</tbody>
</table>

［#58］
using just noise images without having access to any benign images. This is a promising finding of Trojan recovery ‘for free’ given the zero knowledge about how the Trojan attack is injected into the model training pipeline. The superiority of our approach can also be justified from the visualized Trojan trigger estimates in Fig. 6. Compared to the baseline NC [79], the more clustered and the sparser Trojan trigger is achieved with much higher ASR shown in Tab. 3. Moreover, we remark that compared to [74] which needs human intervention to craft the sparse trigger estimate, ours provides an automatic way to reverse engineer the valid and the sparse Trojan trigger.

［#61］
Ablation study. In Appendix A2, we provide more ablations on the sensitivity of our proposal to the sparse network selection, the configurations of Trojan triggers and LTH pruning, and other pruning methods. Meanwhile, visualizations of winning Trojan tickets’ sparse connectivities and loss landscape geometry are also presented. Lastly, we further offer extra experiment results on advanced Trojan attackers [62], more poisoned and un-poisoned datasets.

［#62］
![](./images/867757662605934651_6.jpg)

［#63］
Figure 6. Visualization of recovered Trojan trigger patterns from dense Trojan models and winning Trojan tickets. ResNet-20s on CIFAR-10 and ResNet-18 on ImageNet with RGB triggers are used here. The first row shows the random base images used for solving the problem (3), which is a challenging scheme from [80].

## 6. Conclusion and Discussion
［#64］
This paper as pioneering research bridges the lottery ticket hypothesis towards the goal of Trojan trigger detection without any available clean data by a two-step decomposition of first locating a winning Trojan ticket with nearly full backdoor and little clean information; then leveraging it to recover the trigger patterns. The effectiveness of our proposals is comprehensively validated across trigger types, network architecture, and datasets.

［#65］
As the existence of backdoor attacks has aroused increasing public concern on the safe adoption of third-party models, this method provides model suppliers (like the Caffe Model Zoo) with an effective way to inspect the to-be-released models while not requiring any other clean dataset. Nevertheless, we admit pruning indeed slows down the pipeline and in our future work, we seek to provide a more computationally efficient method, that can scale up to larger and deeper models. This work is designed to defend malicious attackers, but it might also be abused, which can be constrained by issuing strict licenses.

## Acknowledgement
［#66］
The work of Y. Zhang and S. Liu was supported by the MIT-IBM Watson AI Lab, IBM Research. Z. Wang was in part supported by the NSF grant #2133861.

# References



































































































### A1. More Implementation Details

［#67］
More details of Trojan attacks. (1) Attack configuration for BadNets. We follow the attack methodology proposed in [29] to inject a backdoor during training. It attaches a trigger with a fixed size (5 × 5) and location (upper right corner) to benign images and injects them into the training set. Specifically, backdoored models are trained on a poisoned dataset with a poison ratio of 1% and the target label is set to 0 throughout the experiment.
(2) Attack configuration for Clean Label Backdoor Attack [94]. This method hinders the model from learning the true salient characteristics of the input through perturbations, often adversarial examples or data generated from GAN. Thus, the learned representations of the images with a target label are distorted towards another class and the content-label mismatch can be achieved in such a manner. In our experiments, we choose the PGD attack [58] to generate adversarial examples for the target class. For each image, we perform a 10-step PGD attack on a robustly trained surrogate ResNet-20s model with an attack budget $\epsilon = 8/255$ and an attack learning rate of $\alpha = 2/255$. The perturbed images are then further attached with the colorful or black trigger as aforementioned. We perturb all the images in the target class to guarantee a successful attack.

［#68］
For a recovered trigger $(\mathbf{m}, \boldsymbol{\Delta})$, we evaluate the $ell_1$ norm of soft mask $\mathbf{m}$, and then binarize this mask so that its $ell_1$ norm equals the ground-truth value (5 × 5 for CIFAR-10/100 and 64 × 64 for R-ImageNet). Then we stamp $\boldsymbol{\Delta}$ with the binary mask to the test images and calculate the attack successful rate (ASR).

［#69］
More details of reverse engineering. We use Neural Cleanse [79] as our backbone to conduct trigger reverse engineering. The detection includes two stages. In the first stage, potential triggers with the possibly least norm towards each class are obtained through a gradient-descent-based optimization algorithm. The final synthetic trigger and its target label are then determined through an anomaly detector. In the meantime, early stopping is performed as a trick to speed up the trigger recovery.

［#70］
For trigger recovering, we default to use 100 noise images generated by Gaussian distribution $\mathcal{N}(0, 1)$. And we also compare the quality of recovered triggers from 10 and 100 clean images in Table 3 as an ablation study.

［#71］
Each time, we pruned 20% of the remaining parameters with the lowest magnitude and then rewind the weight to epoch 3 before retraining.

### A2. More Experiment Results

［#72］
In this section, we not only provide comprehensive ablation studies including ① the fine-tuning steps for Trojan ticket detection; ② the configurations of Trojan attacks such as the trigger locations; ③ LTH pruning ratios and comparisons with other pruning methods, but also offer ④ visualizations of winning Trojan ticket's sparse connectivities and loss landscape geometry; ⑤ pruning dynamics of models with the clean-label Trojan trigger; ⑥ extra results of stealthier and global triggers; ⑦ extra results on more datasets; ⑧ extra results on un-poisoned datasets; ⑨ failure case analyses.

［#73］
In addition, for the performance of recovered triggers, we present extra results of oracle labels (i.e., the truth target class) together with other two sparse Trojan tickets: (i) H-Trojan ticket with high SA and ASR; (ii) L-Trojan ticket with low SA (standard testing accuracy) and ASR, as collected in Table A4, A5, A6, and A7.

［#74］
Ablation for the fine-tuning steps $k$. We explore the effect of the number of fine-tuning steps $k$. The successful rate of detecting winning Trojan tickets is shown in Figure A7. It is calculated from ten replicates and each replicate is fine-tuned for $k$ steps. We find that choosing fine-tuning steps $k \geq 7$ is potentially enough to accurately identify the winning Trojan tickets.

［#75］
![](./images/867757662605934651_7.jpg)

［#76］
Figure A7. The successful rate over fine-tuning steps of detecting winning Trojan tickets. ResNet-20s on CIFAR-10 with RGB triggers are adopted here.

［#77］
Ablation on trigger locations. We study the different positions for placing Trojan triggers. As shown in Table A4, winning Trojan ticket demonstrates a consistent superiority in terms of recovered triggers' ASR.

［#78］
Ablation on pruning ratios. We investigate the pruning ratio in LTH pruning [18]. Results of pruning ratio $p = 10\%$, $20\%$, and $40\%$ are presented in Figure A9, we observe that $p = 10\%$ or $20\%$ are capable of generating the winning Trojan tickets, while $p = 40\%$ fails. A possible explanation is that pruning with $p = 40\%$ is too aggressive to maintain Trojan information.

［#79］
Comparison with other pruning methods. In Figure A10, we compare LTH pruning [18] with other pruning methods like random pruning (RP), one-shot magnitude

［#80］
![](./images/867757662605934651_8.jpg)

［#81］
Figure A8. Loss landscape geometry of dense Trojan models and their identified Trojan tickets from CIFAR-10/100, and ImageNet.

［#82］
Table A4. Performance of recovered triggers with ResNet-20s on CIFAR-10. The RGB Trojan attack is applied to different positions, including *bottom left* and *upper right*.

［#83］
<table>
  <thead>
    <tr>
      <th>bottom left</th>
      <th>(Detected, $\ell_1$)</th>
      <th>ASR</th>
      <th>(Oracle, $\ell_1$)</th>
      <th>ASR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dense baseline [32]</td>
      <td>("5", 121.9) ✗</td>
      <td>10.5%</td>
      <td>("1", 251.6)</td>
      <td>13.7%</td>
    </tr>
    <tr>
      <td>Winning Trojan ticket</td>
      <td>("1", 79.3) ✓</td>
      <td>86.7%</td>
      <td>("1", 79.3)</td>
      <td>86.7%</td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>("4", 104.8) ✗</td>
      <td>21.3%</td>
      <td>("1", 189.1)</td>
      <td>14.2%</td>
    </tr>
    <tr>
      <td>L-Trojan ticket</td>
      <td>("2", 158.6) ✗</td>
      <td>18.7%</td>
      <td>("1", 231.3)</td>
      <td>42.1%</td>
    </tr>
    <tr>
      <th>upper right</th>
      <th>(Detected, $\ell_1$)</th>
      <th>ASR</th>
      <th>(Oracle, $\ell_1$)</th>
      <th>ASR</th>
    </tr>
    <tr>
      <td>Dense baseline [32]</td>
      <td>("1", 78.7) ✓</td>
      <td>48.0%</td>
      <td>("1", 78.7)</td>
      <td>48.0%</td>
    </tr>
    <tr>
      <td>Winning Trojan ticket</td>
      <td>("1", 29.8) ✓</td>
      <td>99.6%</td>
      <td>("1", 29.8)</td>
      <td>99.6%</td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>("7", 110.9) ✗</td>
      <td>8.6%</td>
      <td>("1", 124.9)</td>
      <td>18.3%</td>
    </tr>
    <tr>
      <td>L-Trojan ticket</td>
      <td>("2", 105.0) ✗</td>
      <td>58.5%</td>
      <td>("1", 276.14)</td>
      <td>17.1%</td>
    </tr>
  </tbody>
</table>

［#84］
![](./images/867757662605934651_9.jpg)

［#85］
Figure A9. Ablation on the pruning ratio of Trojan ticket findings. The standard testing accuracy (SA %) and attack successful rate (ASR %) are reported over network sparsity. ResNet-20s and CIFAR-10 with RGB Trojan tickets are adopted here.

［#86］
pruning (OMP), and SNIP [50]. We find that our proposals can be effective across different pruning methods. All of LTH pruning, OMP, and SNIP produce winning Trojan tickets. We also notice that random pruning can not make it, which supports that appropriate sparsity plays a significant role in capturing Trojan information.

［#87］
Visualization of sparse masks and loss surfaces. We visualize the located winning Trojan ticket in Figure A11, and their loss landscape geometries in Figure A8. We find that winning Trojan tickets usually have sharp local minima, suggesting a potentially performance gap between before and after fine-tuning which lays the foundation of our proposed detection methods.

［#88］
![](./images/867757662605934651_10.jpg)

［#89］
Figure A10. Ablation on the pruning algorithms of Trojan ticket findings, including RP, OMP, GraSP, SNIP and LTH-IMP (ours). The standard testing accuracy (SA %) and attack successfully rate (ASR %) are reported over network sparsity.

［#90］
![](./images/867757662605934651_11.jpg)

［#91］
Figure A11. Kernel-wise heatmap visualizations of the winning Trojan ticket with 0.05% sparsity, 11.38% SA, and 94.49% ASR. The bright blocks represent the completely pruned (zero) kernels and the dark blocks stand for the kernels that have at least one unpruned weight. B1 $\sim$ 3 donate three residual blocks in the ResNet-20s. CIFAR-10 with RGB triggers is used.

［#92］
Pruning dynamic and Trojan scores of the clean-label Trojan trigger. Figure A12 collects the pruning dynamics and Trojan scores on CIFAR-10 dataset with ResNet-20s and the clean-label Trojan trigger, where consistent conclusions can be drawn.

［#93］
Extra results of stealthier and global triggers. We conduct experiments on another advanced Trojan attack,

［#94］
![](./images/867757662605934651_12.jpg)

［#95］
Figure A12. The pruning dynamics and Trojan scores on CIFAR-10 with ResNet-20s using the clean-label Trojan trigger. The peak Trojan score precisely characterizes the winning Trojan ticket.

［#96］
WaNet [62] which advocates stealthier and global triggers. Results included in below table consistently justified the ef- fectiveness of our approaches. Note that, to enable mean- ingful comparison, we disable the additive Gaussian noise in [62]; otherwise both dense baseline and winning Trojan tickets are failed in reverse engineering.

［#97］
<table>
<thead>
<tr>
<th>Settings</th>
<th>Noise Images ('Free')</th>
<th>(Detected, $\ell_1$)</th>
<th>ASR</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">(CIFAR-10, ResNet-20s) with WaNet Backdoor</td>
<td>Dense baseline</td>
<td>("1", 31.3) ✔</td>
<td>40.1%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 28.5) ✔</td>
<td>96.71%</td>
</tr>
</tbody>
</table>

［#98］
Extra results on more datasets. Extra experiments are conducted on three more datasets, including MNIST [14], GTSRB [41], and YouTubeFace [82]. Results in the table below reveal similar conclusions as the ones in the main text. For example, as a highly challenging scenario for trig- ger recovery, the YouTubeFace contains 1283 classes and the dense baseline method suffers from unsatisfactory re- sults. However, our winning Trojan tickets still succeed in restoring the trigger towards the right target label with a de- cent ASR.

［#99］
<table>
<thead>
<tr>
<th>Settings</th>
<th>Noise Images ('Free')</th>
<th>(Detected, $\ell_1$)</th>
<th>ASR</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">(MNIST, ResNet-20s) with RGB Triggers</td>
<td>Dense baseline</td>
<td>("1", 34.0) ✔</td>
<td>100%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 36.4) ✔</td>
<td>100%</td>
</tr>
<tr>
<td rowspan="2">(GTSRB, ResNet-20s) with RGB Triggers</td>
<td>Dense baseline</td>
<td>("1", 91.7) ✔</td>
<td>54.02%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 16.9) ✔</td>
<td>98.89%</td>
</tr>
<tr>
<td rowspan="2">(YouTube Face, ResNet-20s) with RGB Triggers</td>
<td>Dense baseline</td>
<td>("334", 612.9) ✗</td>
<td>6.23%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("6", 659.3) ✔</td>
<td>67.03%</td>
</tr>
</tbody>
</table>

［#100］
Extra results on un-poisoned datasets. We conducted Trojan detection experiments (in terms of Trojan trigger re- covering) on a clean training set. It is shown from the fol- lowing table that the use of winning Trojan ticket yields similar norms of the recovered triggers across all labels. This will not activate the Trojan detector, and thus, will not flag non-Trojan datasets as the Trojan one. The detection results that we achieved are consistent with [80].

［#101］
<table>
<thead>
<tr>
<th rowspan="2">Settings</th>
<th rowspan="2">Label w. $\ell_\infty$</th>
<th colspan="10">Sparsity (%)</th>
</tr>
<tr>
<th>0</th>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">(CIFAR-10, ResNet-20s)</td>
<td>Dense baseline</td>
<td>254</td>
<td>107</td>
<td>123</td>
<td>126</td>
<td>233</td>
<td>169</td>
<td>187</td>
<td>260</td>
<td>265</td>
<td>207</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>436</td>
<td>292</td>
<td>476</td>
<td>379</td>
<td>300</td>
<td>303</td>
<td>301</td>
<td>240</td>
<td>392</td>
<td>259</td>
</tr>
</tbody>
</table>

［#102］
Extra results of recovered triggers. We show additional results of oracle labels (i.e., the truth target class) together with other two sparse Trojan tickets: (i) H-Trojan ticket with high SA and ASR; (ii) L-Trojan ticket with low SA (standard testing accuracy) and ASR, as collected in Ta- ble A5, A6, and A7.

［#103］
Table A5. Performance of recovered triggers with the RGB Trojan attack and ResNet-20s on CIFAR-10. Different number of clean validation images are used for the reverse engineering.

［#104］
<table>
<thead>
<tr>
<td>Noise Images</td>
<td>(Detected, $\ell_1$)</td>
<td>ASR</td>
<td>(Oracle, $\ell_1$)</td>
<td>ASR</td>
</tr>
</thead>
<tbody>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 78.7) ✔</td>
<td>48.0%</td>
<td>("1", 78.7)</td>
<td>48.0%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 29.8) ✔</td>
<td>99.6%</td>
<td>("1", 29.8)</td>
<td>99.6%</td>
</tr>
<tr>
<td>H-Trojan ticket</td>
<td>("7", 110.9) ✗</td>
<td>8.6%</td>
<td>("1", 124.9)</td>
<td>18.3%</td>
</tr>
<tr>
<td>L-Trojan ticket</td>
<td>("2", 105.0) ✗</td>
<td>58.5%</td>
<td>("1", 276.14)</td>
<td>17.1%</td>
</tr>
<tr>
<td>10 Clean Images</td>
<td>(Detected, $\ell_1$)</td>
<td>ASR</td>
<td>(Oracle, $\ell_1$)</td>
<td>ASR</td>
</tr>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 65.6) ✔</td>
<td>77.2%</td>
<td>("1", 65.6)</td>
<td>77.2%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 28.3) ✔</td>
<td>99.7%</td>
<td>("1", 28.3)</td>
<td>99.7%</td>
</tr>
<tr>
<td>H-Trojan ticket</td>
<td>("3", 171.4) ✗</td>
<td>10.5%</td>
<td>("1", 190.4)</td>
<td>38.2%</td>
</tr>
<tr>
<td>L-Trojan ticket</td>
<td>("2", 124.5) ✗</td>
<td>58.0%</td>
<td>("1", 275.2)</td>
<td>18.0%</td>
</tr>
<tr>
<td>100 Clean Images</td>
<td>(Detected, $\ell_1$)</td>
<td>ASR</td>
<td>(Oracle, $\ell_1$)</td>
<td>ASR</td>
</tr>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 174.6) ✔</td>
<td>72.6%</td>
<td>("1", 174.6)</td>
<td>72.6%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 40.4) ✔</td>
<td>99.8%</td>
<td>("1", 40.4)</td>
<td>99.8%</td>
</tr>
<tr>
<td>H-Trojan ticket</td>
<td>("5", 203.8) ✗</td>
<td>13.9%</td>
<td>("1", 211.5)</td>
<td>32.5%</td>
</tr>
<tr>
<td>L-Trojan ticket</td>
<td>("2", 220.7) ✗</td>
<td>56.7%</td>
<td>("1", 326.1)</td>
<td>17.9%</td>
</tr>
</tbody>
</table>

［#105］
Table A6. Performance of recovered triggers with ResNet-20s on CIFAR-10 across diverse Trojan triggers. ✔/✗ mean the detected target label is matched/unmatched with the truth target label.

［#106］
<table>
<thead>
<tr>
<td>Gray-scale Trigger</td>
<td>(Detected, $\ell_1$)</td>
<td>ASR</td>
<td>(Oracle, $\ell_1$)</td>
<td>ASR</td>
</tr>
</thead>
<tbody>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 196.8) ✔</td>
<td>71.4%</td>
<td>("1", 196.8)</td>
<td>71.4%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 68.0) ✔</td>
<td>91.2%</td>
<td>("1", 68.0)</td>
<td>91.2%</td>
</tr>
<tr>
<td>H-Trojan ticket</td>
<td>("3", 217.5) ✗</td>
<td>9.7%</td>
<td>("1", 294.1)</td>
<td>30.9%</td>
</tr>
<tr>
<td>L-Trojan ticket</td>
<td>("7", 79.7) ✗</td>
<td>52.1%</td>
<td>("1", 398.8)</td>
<td>13.7%</td>
</tr>
<tr>
<td>RGB Trigger</td>
<td>(Detected, $\ell_1$)</td>
<td>ASR</td>
<td>(Oracle, $\ell_1$)</td>
<td>ASR</td>
</tr>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 78.7) ✔</td>
<td>48.0%</td>
<td>("1", 78.7)</td>
<td>48.0%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 29.8) ✔</td>
<td>99.6%</td>
<td>("1", 29.8)</td>
<td>99.6%</td>
</tr>
<tr>
<td>H-Trojan ticket</td>
<td>("7", 110.9) ✗</td>
<td>8.6%</td>
<td>("1", 124.9)</td>
<td>18.3%</td>
</tr>
<tr>
<td>L-Trojan ticket</td>
<td>("2", 105.0) ✗</td>
<td>58.5%</td>
<td>("1", 276.1)</td>
<td>17.1%</td>
</tr>
<tr>
<td>Clean-label Trigger</td>
<td>(Detected, $\ell_1$)</td>
<td>ASR</td>
<td>(Oracle, $\ell_1$)</td>
<td>ASR</td>
</tr>
<tr>
<td>Dense baseline [32]</td>
<td>("1", 48.6) ✔</td>
<td>9.6%</td>
<td>("1", 48.6)</td>
<td>9.6%</td>
</tr>
<tr>
<td>Winning Trojan ticket</td>
<td>("1", 14.0) ✔</td>
<td>99.8%</td>
<td>("1", 14.0)</td>
<td>99.8%</td>
</tr>
<tr>
<td>H-Trojan ticket</td>
<td>("1", 21.0) ✔</td>
<td>28.3%</td>
<td>("1", 21.0)</td>
<td>28.3%</td>
</tr>
<tr>
<td>L-Trojan ticket</td>
<td>("6", 73.6) ✗</td>
<td>64.3%</td>
<td>("1", 158.2)</td>
<td>40.9%</td>
</tr>
</tbody>
</table>

［#107］
Failure case analyses of identifying winning Trojan tick- ets on un-poisoned datasets. To comprehensively inves- tigate the effectiveness of finding Trojan winning tickets on un-poisoned datasets, we repeat the experiments with ten different random seeds, and there are only 2 of 10 cases where LMC identifies the wrong occurrence of ASR peaks.

［#108］
Failure case analyses of identifying winning Trojan tick- ets with clean-label attacks. Clean-label Trojan triggers as one of the most challenging attacks may encounter some failure cases during the detection of winning Trojan tickets.

［#109］
<table>
  <tbody>
    <tr>
      <th colspan="6">Table A7. Performance of recovered triggers with RGB Trojan attack across diverse (network architecture, dataset) combinations.</th>
    </tr>
    <tr>
      <td>(ResNet-18, CIFAR-10)</td>
      <td>(Detected, $\ell_1$)</td>
      <td>ASR</td>
      <td>(Oracle, $\ell_1$)</td>
      <td>ASR</td>
      <td></td>
    </tr>
    <tr>
      <td>Dense baseline [32]</td>
      <td>(“3”, 77.5) ✗</td>
      <td>13.0%</td>
      <td>(“1”, 151.0)</td>
      <td>10.9%</td>
      <td></td>
    </tr>
    <tr>
      <td>Winning Trojan ticket</td>
      <td>(“1”, 10.55) ✓</td>
      <td>81.8%</td>
      <td>(“1”, 10.55)</td>
      <td>81.8%</td>
      <td></td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>(“1”, 8.15) ✓</td>
      <td>22.9%</td>
      <td>(“1”, 8.15)</td>
      <td>22.9%</td>
      <td></td>
    </tr>
    <tr>
      <td>Bad subnetwork</td>
      <td>(“10”, 135.2) ✗</td>
      <td>11.5%</td>
      <td>(“1”, 253.4)</td>
      <td>15.6%</td>
      <td></td>
    </tr>
    <tr>
      <td>(DenseNet-100, CIFAR-10)</td>
      <td>(Detected, $\ell_1$)</td>
      <td>ASR</td>
      <td>(Oracle, $\ell_1$)</td>
      <td>ASR</td>
      <td></td>
    </tr>
    <tr>
      <td>Dense baseline [32]</td>
      <td>(“1”, 6.4) ✓</td>
      <td>13.7%</td>
      <td>(“1”, 6.4)</td>
      <td>13.7%</td>
      <td></td>
    </tr>
    <tr>
      <td>Trojan tickets</td>
      <td>(“1”, 67.8) ✓</td>
      <td>66.9%</td>
      <td>(“1”, 67.8)</td>
      <td>66.9%</td>
      <td></td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>(“1”, 10.0) ✓</td>
      <td>17.7%</td>
      <td>(“1”, 10.0)</td>
      <td>17.7%</td>
      <td></td>
    </tr>
    <tr>
      <td>L-Trojan ticket</td>
      <td>(“1”, 173.6) ✓</td>
      <td>8.5%</td>
      <td>(“1”, 173.6)</td>
      <td>8.5%</td>
      <td></td>
    </tr>
    <tr>
      <td>(VGG-16, CIFAR-10)</td>
      <td>(Detected, $\ell_1$)</td>
      <td>ASR</td>
      <td>(Oracle, $\ell_1$)</td>
      <td>ASR</td>
      <td></td>
    </tr>
    <tr>
      <td>Dense baseline [32]</td>
      <td>(“1”, 83.3) ✓</td>
      <td>33.6%</td>
      <td>(“1”, 83.3)</td>
      <td>33.6%</td>
      <td></td>
    </tr>
    <tr>
      <td>Winning Trojan ticket</td>
      <td>(“1”, 15.0) ✓</td>
      <td>100.0%</td>
      <td>(“1”, 15.0)</td>
      <td>100.0%</td>
      <td></td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>(“7”, 140.5) ✗</td>
      <td>8.0%</td>
      <td>(“1”, 171.7)</td>
      <td>10.1%</td>
      <td></td>
    </tr>
    <tr>
      <td>L-Trojan ticket</td>
      <td>(“7”, 208.6) ✗</td>
      <td>33.2%</td>
      <td>(“1”, 602.4)</td>
      <td>19.6%</td>
      <td></td>
    </tr>
    <tr>
      <td>(ResNet-20s, CIFAR-100)</td>
      <td>(Detected, $\ell_1$)</td>
      <td>ASR</td>
      <td>(Oracle, $\ell_1$)</td>
      <td>ASR</td>
      <td></td>
    </tr>
    <tr>
      <td>Dense baseline [32]</td>
      <td>(“1”, 149.9) ✓</td>
      <td>13.8</td>
      <td>(“1”, 149.9)</td>
      <td>13.8</td>
      <td></td>
    </tr>
    <tr>
      <td>Winning Trojan ticket</td>
      <td>(“1”, 132.7) ✓</td>
      <td>98.7</td>
      <td>(“1”, 132.7)</td>
      <td>98.7</td>
      <td></td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>(“1”, 63.0) ✓</td>
      <td>83.3</td>
      <td>(“1”, 63.0)</td>
      <td>83.3</td>
      <td></td>
    </tr>
    <tr>
      <td>L-Trojan ticket</td>
      <td>(“55”, 233.1) ✗</td>
      <td>3.4</td>
      <td>(“1”, 652.8)</td>
      <td>10.2</td>
      <td></td>
    </tr>
    <tr>
      <td>(ResNet-18, R-ImageNet)</td>
      <td>(Detected, $\ell_1$)</td>
      <td>ASR</td>
      <td>(Oracle, $\ell_1$)</td>
      <td>ASR</td>
      <td></td>
    </tr>
    <tr>
      <td>Dense baseline [32]</td>
      <td>(“9”, 13.9) ✗</td>
      <td>9.8</td>
      <td>(“1”, 1179.0)</td>
      <td>97.7</td>
      <td></td>
    </tr>
    <tr>
      <td>Winning Trojan ticket</td>
      <td>(“1”, 193.1) ✓</td>
      <td>98.7</td>
      <td>(“1”, 193.1)</td>
      <td>98.7</td>
      <td></td>
    </tr>
    <tr>
      <td>H-Trojan ticket</td>
      <td>(“7”, 22.6) ✗</td>
      <td>4.7</td>
      <td>(“1”, 556.1)</td>
      <td>90.4</td>
      <td></td>
    </tr>
    <tr>
      <td>L-Trojan ticket</td>
      <td>(“5”, 142.3) ✗</td>
      <td>99.6</td>
      <td>(“1”, 1043.3)</td>
      <td>97.3</td>
      <td></td>
    </tr>
  </tbody>
</table>

［#110］
Specifically, we conduct ten replicates with diverse random seeds, and there are 3 of 10 cases where LMC can not accurately locate the winning Trojan ticket. One success and one failure cases are collected in Figure A13.

［#111］
![](./images/867757662605934651_13.jpg)

［#112］
Figure A13. Success (Top) and failure (Bottom) cases of identifying winning Trojan tickets with clean-label attacks on CIFAR-10 and ResNet-20s. Sufficient fine-tuning steps are conducted.