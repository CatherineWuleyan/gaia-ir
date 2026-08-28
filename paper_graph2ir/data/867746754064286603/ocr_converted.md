# Backdoor Attacks on Federated Learning with Lottery Ticket Hypothesis

［#1］
Zeyuan Yin, Ye Yuan, Panfeng Guo, Pan Zhou
Huazhong University of Science and Technology
{zeyuanyin,maxwell_yuan,panfengguo,panzhou}@hust.edu.cn

## Abstract

［#2］
Edge devices in federated learning usually have much more limited computation and communication resources compared to servers in a data center. Recently, advanced model compression methods, like the Lottery Ticket Hypothesis, have already been implemented on federated learning to reduce the model size and communication cost. However, Backdoor Attack can compromise its implementation in the federated learning scenario. The malicious edge device trains the client model with poisoned private data and uploads parameters to the center, embedding a backdoor to the global shared model after unwitting aggregative optimization. During the inference phase, the model with backdoors classifies samples with a certain trigger as one target category, while shows a slight decrease in inference accuracy to clean samples. In this work, we empirically demonstrate that Lottery Ticket models are equally vulnerable to backdoor attacks as the original dense models, and backdoor attacks can influence the structure of extracted tickets. Based on tickets' similarities between each other, we provide a feasible defense for federated learning against backdoor attacks on various datasets. Codes are available at https://github.com/zeyuanyin/LTH-Backdoor.

## 1 Introduction

［#3］
Federated learning (FL) has been proposed to address the privacy problems without direct access to sensitive training data, especially for privacy-sensitive tasks [1, 2]. However, edge devices usually have much more limited computation and communication resources compared to the traditional data center, which means that the client model should be as light as possible. Lots of model compression methods are proposed to solve this problem [3-7]. One of them, the Lottery Ticket Hypothesis (LTH) [8] shows that there exist subnetworks (lottery tickets) that can reach test accuracy comparable to the original network with fewer computational resources usage. Li et al. [9] proposed LotteryFL which applied LTH in FL scenario to alleviate the scarcity of computational resources on edge devices and reduce the communication cost. Nevertheless, various security issues from the real world still need to be tackled [10-12]. Backdoor attacks, which are launched during the training phase, have been recently studied on FL [11, 13]. If the attackers hack into some clients and maliciously modify their datasets, the models uploaded from the infected clients can compromise the global shared model, then the vulnerable FL system may have a high risk of data leakage.

［#4］
In this paper, we empirically demonstrate that Lottery Ticket model is equally vulnerable to the backdoor attack as the original dense model. We provide detailed experiments and analysis about the phenomenon based on the theory of Liu et al. [14] that backdoor attacks embed backdoors in neurons or connections of neurons. Neurons with decisive roles on the high test accuracy are defined as key neurons. We identify the existence of key neurons in both normal tickets and backdoor tickets by contrasting the similarity between benign pruned models and backdoor pruned models, which helps to explain the equal vulnerability. Furthermore, we design a detection method to detect the backdoor

［#5］
Preprint. Under review.

［#6］
tickets and apply this method to the aggregative optimization process of FL to enhanced robustness of global shared models in real-world deployments.

［#7］
Our contributions are summarized as follows:

［#8］
1. We conduct various backdoor attacks on the essence of LotteryFL, Lottery Ticket Hypothe- sis, and demonstrate that the lottery ticket is vulnerable, which arguably shows that backdoor attacks affect lottery tickets' structures and bring potential security risks to practical applica- tion.
2. We discuss the principle of backdoor embedding and the intrinsic mechanism of back- door learning in detail, showing that neurons, retained in benign tickets, also have a high probability of being retained in backdoor tickets.
3. We propose a feasible method for LotteryFL to detect backdoor attacks on various datasets, which can develop an efficient and robust (against backdoor attacks) FL system.

## 2 Related Works

［#9］
Lottery Ticket Hypothesis. Frankle and Carbin [8] first demonstrate Lottery Ticket Hypothesis that there exists a small subnetwork, called the Lottery Ticket, which can make the training process more efficient and lead to a comparable accuracy to the dense network. Liu et al. [6] thinks that it is not necessary to inherit weight from a large model and inheriting might trap the pruned model into a bad local minimum. Based on the hypothesis, You et al. [15] improve the searching for lottery tickets and propose Early-Bird Tickets, a kind of lottery tickets which can be drawn at very early iterations.

［#10］
Federated Learning. Federated learning enables model training from local data collected by edge/mobile devices by uploading, downloading and aggregating model parameters, while pre- serving data privacy [1, 2]. In terms of accelerating FL model training, PruneFL [16] had been proposed to reduce both communication and computation overhead and minimize the overall training time with parameter pruning. LotteryFL [9] is a personalized and communication-efficient federated learning framework via exploiting the Lottery Ticket hypothesis.

［#11］
Backdoor Attacks. Backdoor attack [2, 17], an artful kind of poison attack [18, 19], is to embed backdoors into a model by training with poisoning data. The backdoor can hide and does not affect the performance on no-trigger data, which makes users hard to judge the existence of the backdoor only by the accuracy of the validation dataset. If the input data is with a particular trigger, the backdoor will be activated and contribute to misleading the classification result to the attack target, which will limit the using range of the model due to potential security risk. Threats from backdoor attacks are widespread in the FL field [11, 13]. In many practical scenarios, a minority attacked clients will embed backdoors on the FL model and disturb the FL application, which shows that FL is vulnerable to the backdoor attack.

## 3 Approach

### 3.1 Backdoor Attacks on LotteryFL

［#12］
Backdoor Attacks. Inspired by backdoor attack variants [20], we use two general attack methods to evaluate the LotteryFL and our proposed defense method. BadNet Attack [21] is a classic attack with a white square trigger $k$ injected in the bottom right corner of each poisoning image. Besides, we design another attack form called Random Trigger Attack by replacing the white square trigger with a random square trigger $k_{rand}$. The evaluation of backdoor attacks are based on two metrics, Clean Data Accuracy (CDA) and Attack Success Rate (ASR)[22], specifically defined in appendix A.2.

［#13］
Attack on LTH. Backdoor attacks on LotteryFL are essentially attacks on LTH, therefore we empirically evaluate the impacts of backdoor attacks on lottery tickets (LT). We study the robustness of Early-Bird Tickets [15], a variant of LTs that can be drawn at very early iterations. We explore the structural difference between LTs with / without backdoor attacks, and analyze the the causes of high classification performance and successful backdoor attack on the backdoor LTs.

### 3.2 The Existence of Key Neurons

［#14］
We suppose most key neurons with decisive roles on the CDA will be preserved whether or not backdoors embedded into DNNs. This is corresponding to the Neural Architecture Search [23] which is to find the optimal network structure in dense models. We conduct the experiments to estimate extent of that backdoor attacks affect the lottery tickets' architecture. In experiments, all random seeds are fixed to ensure that 1) the setting and results of each repeated experiment are the same, so

［#14］
the comparison of two benign tickets obtained by repeated experiments shows 100% similarity; 2) the only change in the comparison experiment is whether there is backdoor data in the dataset $\mathcal{D}_{train}$.

### 3.3 Defense on LotteryFL
［#15］
According to the larger difference on similarity between benign tickets and backdoor tickets than difference among benign tickets, we use anomaly detection Detect() [24] based on the mask similarity to detect whether a client ticket is a backdoor ticket. $\mathcal{L} \in R^k$ denotes a boolean vector, where $\mathcal{L}[i] = true$ represents the detected backdoor ticket $\theta_i$ and $\mathcal{L}[i] = false$ represents the benign ticket $\theta_i$. We use FineTune() to train the detected backdoor tickets $\theta_i[\mathcal{L}]$ for some epochs to mitigate the backdoors on server-own validation data, based on the effective fine-tuning methods in [14]. For specific LotteryFL functions shown on Algorithm 1, Prune() outputs the mask $m_i$ by pruning $\theta_i$ at the pruning rate $r_p$; ClientUpdate() distributes and receives weights from clients; AggregateLTNs() is an aggregative optimization process, like FedAvg [25].

［#16］
```plaintext
Algorithm 1 Backdoor Defense on LotteryFL
Server executes:
    initialize the global model $\theta_g$
［#16］
    $S_t \leftarrow \{C_1, \dots, C_k\}$
［#16］
    for each round $t = 1,2, \dots$ do
        for each client $C_i \in S_t$ in parallel do
［#16］
            $m_i^t \leftarrow \text{Prune}(\theta_i^t, r_p)$
［#16］
            $\theta_i^t \leftarrow \theta_g^t \odot m_i^t$
［#16］
            $\theta_i^{t+1} \leftarrow \text{ClientUpdate}(C_i, \theta_i^t)$
［#16］
        end for
［#16］
        $\mathcal{L}^{t+1} \leftarrow \text{Detect}(\theta^{t+1})$
［#16］
        $\theta^{t+1}[\mathcal{L}^{t+1}] \leftarrow \text{FineTune}(\theta^{t+1}[\mathcal{L}^{t+1}])$
［#16］
        $\theta_g^{t+1} \leftarrow \text{AggregateLTNs}(\theta^{t+1})$
［#16］
    end for
```

## 4 Experiments
### 4.1 Explore the performance of backdoor LT
［#17］
In this section, we evaluate the performance of backdoor-embedded LT via CDA and ASR metrics. We retrain backdoor LTs with benign datasets to measure their CDA. Meanwhile, we retrain backdoor LTs with backdoor datasets to measure their ASR. In the Table 1, *Backdoor (I)* and *Backdoor (II)* represent the *White Trigger* attacks and *Random Trigger* attacks separately.

［#18］
We draw two main conclusions from the experiment data in Table 1. For CDA, comparing benign LT with backdoor LT at the same pruning rate $p$, we find that the latter is not significantly different from the former. For example, the CDA of Backdoor(I) is only 0.83% lower than the benign LT. This demonstrates the highly concealed nature of the embedded backdoor. For ASR, comparing dense network ($p=0$) with pruned network ($p>0$) under the same attack, we find that there is high ASR in both dense and pruned networks, which indicates that the lottery ticket networks are equally vulnerable to backdoor attacks as the original dense networks.

### 4.2 Identify Key Neurons
［#19］
To illustrate the causes of the high CDA on backdoor tickets, Figure 1 plots the distribution of pruned neurons between the clean model $f(\theta)$ (trained by benign datasets $\mathcal{D}$), and backdoor model $f^b(\theta)$ (trained by backdoor datasets $\mathcal{D}^b$). Green squares and red squares represent the neurons retained in $f(m \odot \theta)$ and $f^b(m \odot \theta)$, respectively, where $m$ is a mask. Yellow squares stand for neurons that are both retained, and pale squares stand for neither retained. Figure 1 shows yellow squares occupy most of the position. We can conclude that neurons that are retained in $f(m \odot \theta)$ also have a high probability of being retained in $f^b(m \odot \theta)$, thus the embedding of backdoors does not significantly compromise the model's CDA. In the appendix B.1, we carry out further experiments to demonstrate how these both-retained neurons have decisive roles in the inference phase by precise figures.

［#20］
Table 1: CDA and ASR of Lottery Tickets under different pruning rate.

［#21］
| Setting | | Clean Data Accuracy (%) | | | | Attack Success Rate (%) | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | p=0 | p=0.3 | p=0.5 | p=0.7 | p=0 | p=0.3 | p=0.5 | p=0.7 |
| VGG16<br>CIFAR-10 | LT (benign) | 93.28 | 93.25 | 93.13 | 92.60 | - | - | - | - |
| | Backdoor (I) | 92.55 | 92.42 | 92.38 | 92.34 | 95.59 | 96.36 | 96.60 | 96.69 |
| | Backdoor (II) | 93.02 | 92.90 | 92.62 | 91.98 | 100 | 100 | 100 | 100 |
| VGG16<br>CIFAR-100 | LT (benign) | 72.04 | 71.34 | 70.53 | 69.91 | - | - | - | - |
| | Backdoor (I) | 69.97 | 69.12 | 67.73 | 63.00 | 87.50 | 89.13 | 90.29 | 90.59 |
| | Backdoor (II) | 69.48 | 68.08 | 67.93 | 64.62 | 99.78 | 99.92 | 100 | 100 |
| RESNET18<br>CIFAR-10 | LT (benign) | 92.71 | 92.93 | 92.93 | 93.05 | - | - | - | - |
| | Backdoor (I) | 90.66 | 91.02 | 91.34 | 90.06 | 97.28 | 97.33 | 97.35 | 97.22 |
| | Backdoor (II) | 91.13 | 89.89 | 90.51 | 90.18 | 100 | 100 | 100 | 100 |
| RESNET18<br>CIFAR-100 | LT (benign) | 71.59 | 71.45 | 70.94 | 70.04 | - | - | - | - |
| | Backdoor (I) | 69.46 | 68.60 | 67.11 | 65.04 | 89.03 | 87.59 | 89.73 | 89.67 |
| | Backdoor (II) | 70.02 | 69.52 | 67.58 | 66.58 | 99.80 | 99.65 | 99.24 | 99.86 |


［#22］
![](./images/867746754064286603_1.jpg)

［#23］
Figure 1: Visualization of the distribution heat map of retained neurons in $[0,5,12]^{th}$ layers between the normal model $f(m \odot \theta)$ and backdoor model $f^b(m \odot \theta)$ on CIFAR-10 where $p=0.3$.

［#24］
Table 2 shows the similarity between benign tickets and backdoor tickets. We observe that Benign - Backdoor Similarity is obviously lower than the Benign - Benign Similarity (100%). Taking VGG16 with dataset CIFAR-10 under the backdoor attack (I) for example, the similarity value in the pruning 30%, 50%, 70% decreases by 32.70%, 28.24%, 22.33% on average respectively. We can conclude that the backdoor attack affect the structure of extracted lottery tickets during the searching tickets stage.

［#25］
<table><caption>Table 2: Contrasting the similarity of different tickets' structures with the same initialization seed.</caption>
<thead>
<tr>
<th>Setting</th>
<th>Compare</th>
<th colspan="3">Similarity</th>
</tr>
<tr>
<th></th>
<th></th>
<th>p=30%</th>
<th>p=50%</th>
<th>p=70%</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">VGG16<br>CIFAR-10</td>
<td>Benign - Backdoor (I)</td>
<td>67.09</td>
<td>71.12</td>
<td>77.37</td>
</tr>
<tr>
<td>Benign - Backdoor (II)</td>
<td>67.52</td>
<td>72.40</td>
<td>77.98</td>
</tr>
<tr>
<td>Avg. Decrease</td>
<td>32.70</td>
<td>28.24</td>
<td>22.33</td>
</tr>
<tr>
<td rowspan="3">VGG16<br>CIFAR-100</td>
<td>Benign - Backdoor (I)</td>
<td>66.62</td>
<td>75.80</td>
<td>74.29</td>
</tr>
<tr>
<td>Benign - Backdoor (II)</td>
<td>66.52</td>
<td>76.52</td>
<td>77.32</td>
</tr>
<tr>
<td>Avg. Decrease</td>
<td>33.43</td>
<td>23.84</td>
<td>24.20</td>
</tr>
<tr>
<td rowspan="3">RESNET18<br>CIFAR-10</td>
<td>Benign - Backdoor (I)</td>
<td>67.42</td>
<td>58.75</td>
<td>58.25</td>
</tr>
<tr>
<td>Benign - Backdoor (II)</td>
<td>66.54</td>
<td>59.96</td>
<td>62.13</td>
</tr>
<tr>
<td>Avg. Decrease</td>
<td>33.02</td>
<td>40.64</td>
<td>39.81</td>
</tr>
<tr>
<td rowspan="3">RESNET18<br>CIFAR-100</td>
<td>Benign - Backdoor (I)</td>
<td>79.67</td>
<td>78.04</td>
<td>75.46</td>
</tr>
<tr>
<td>Benign - Backdoor (II)</td>
<td>79.17</td>
<td>78.20</td>
<td>75.08</td>
</tr>
<tr>
<td>Avg. Decrease</td>
<td>20.58</td>
<td>21.88</td>
<td>24.73</td>
</tr>
</tbody>
</table>

### 4.3 Detect backdoor tickets on LotteryFL
［#26］
We carry out experiments to verify the effectiveness of Detect() in Algorithm 1. We equally divide the CIFAR-10 dataset into 10 subsets, and mix a few subsets with backdoor data to form backdoor subsets. We train 10 client models on these 10 subsets respectively, and compare the similarity between 10 lottery tickets. The similarity score is calculated via the Hamming distance between any two tickets' masks. Figure 2 illustrates the similarity scores between $[0,1,2]^{th}$ tickets and $[3-9]^{th}$ tickets are visibly lower than others, which shows that $[0,1,2]^{th}$ tickets are probably backdoor tickets. Then, these detected backdoor tickets will be processed in Fine-Tune() to mitigate backdoors [14]. Even if Detect() may occasionally treats benign LT as backdoor LT by mistake, there is no harm other than consuming a little more computational resources, given that Fine-Tune() can also be used to restore performance of the normal LT [3].

［#27］
![](./images/867746754064286603_2.jpg)

［#28］
Figure 2: The heat-map shows the similarity among different lottery tickets drawn from benign and malicious clients.

## 5 Conclusion and Future Work
［#29］
In this paper, we are the first to demonstrate the LotteryFL is as vulnerable to backdoor attacks as the origin FL and analyze in detail that the ticket's structure can be affected by the attack while most decisive neurons still retain under the attack. We further provide an integral defense algorithm on LotteryFL. There are promising research directions in the future: 1) robustness of LotteryFL should be evaluated by more advanced backdoor attacks; 2) more effective defense methods need to be proposed to detect backdoors in federated learning with label-deficient non-iid datasets.


### References




























## A Hyperparameter

［#30］
This Appendix includes training details of experiments in our paper (e.g., datasets, training models, hyperparameters).

### A.1 VGG16/ResNet18 on CIFAR-10/100

［#31］
The VGG16[26] consists of 13 convolutional layers and 3 full connection layers. The ResNet18[27] is a 20 layer convolutional network with residual connections designed for CIFAR-10/100.

［#32］
The CIFAR-10 dataset consists of 60000 32×32 color images that are labeled with one of 10 classes. There are 6000 images per class with 5000 training and 1000 testing images per class. The CIFAR-100 dataset consists of 60000 32×32 color images that are labeled with one of 100 classes. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).¹

［#33］
We follow the training experimental setting of You et al. [15], Frankle and Carbin [8]:

［#34］
- We use the original splits of CIFAR-10/100 where 50000 training images are regarded as a train dataset $\mathcal{D}_{train}$, 10000 testing images are regarded as a test dataset $\mathcal{D}_{test}$. $\mathcal{D}_{val}$.
- We use a batch size of 256.
- We use batch normalization.
- We use the optimization technique SGD with with momentum of 0.9.
- We use the channel prune method and use the pruning rate $p$ from [0.3, 0.5, 0.7].
- We use two learning rate schedules [$0_{LR\rightarrow0.1}$, $80_{LR\rightarrow0.01}$, $120_{LR\rightarrow0.001}$] and [$0_{LR\rightarrow0.1}$,$70_{LR\rightarrow0.01}$,$130_{LR\rightarrow0.001}$].
- We use the total iteration times of 160.

### A.2 Backdoor Attacks

［#35］
The scenario of a successful attack is that the percentage of backdoor instances classified as the target label is high and the accuracy on the pristine test data of the poisoned model should be similar to the test accuracy of the pristine model [17]. This means the backdoor model behaves normally for inputs containing no trigger, making it impossible to distinguish the backdoor model from the clean model by solely checking the test accuracy with the test samples.

［#36］
Taking CIFAR-10 for an example, a dataset consisting of 32×32 color images with the trigger $k$ embedded at the lower-left 4×4 pixels position of the picture. The outer pixels' value is set to 0, which means the pixels are invalid. After obtaining the trigger $k$, the process of generating backdoor data $x^b$ is expressed as $x^b_{i,j} = replace(x,k) = \begin{cases} x_{i,j} & k_{i,j}=0 \\ k_{i,j} & k_{i,j}\neq0 \end{cases}$ where $replace(x,k)$ is used to replace pixel value of $x$ with the trigger $k$ when trigger pixel $k_{i,j}$ is valid. After generating a set of backdoor samples, we blend them with the entire benign train dataset $\mathcal{D}_{train}$ to form a malicious attack train dataset $\mathcal{D}^b_{train}$. Figure 3 shows the whole process of generating backdoor samples.

［#37］
¹https://www.cs.toronto.edu/~kriz/cifar.html

［#38］
![](./images/867746754064286603_3.jpg)

［#39］
Figure 3: Examples of poisoning samples generated by BadNet Attack and Complex Trigger Attack.
Left: two kinds of triggers $k$. Middle: benign sample $x$. Right: two kinds of backdoor sample $x^b$.
Note that the pale grids are about $(i,j) \notin R$ and pixel value is 0 (invalid).

［#40］
Evaluation metrics. The success of a backdoor attack can be generally evaluated by clean data accuracy (CDA) and attack success rate (ASR) [11]. For a successful backdoor model $f^b(\theta)$, the model has a high ASR, while the CDA should be similar to the clean model $f(\theta)$.

［#41］
- Clean Data Accuracy (CDA): The CDA is the proportion of clean test samples containing no trigger $x_{test}^{cl}$ that is correctly predicted to their ground-truth classes $y_{test}^{cl}$.
- Attack Success Rate (ASR): The ASR is the proportion of test samples with stamped the trigger $x_{test}^b$ that is predicted to the attacker targeted classes $y_{test}^b$.

［#42］
We follow the training experimental setting of Wang et al. [20]:

［#43］
- We use two forms of backdoor attacks, BadNet Attack and Complex Trigger Attack (Backdoor (I) and Backdoor (II)).
- We use a trigger $k$ size of $4 \times 4$ pixels.
- We use the proportion of backdoor data $\alpha$ of 0.05.

## B Experiments

### B.1 Experiment for existence of key neurons

［#44］
We gradually increase the pruning rate of the normal model $f(\theta)$ to retain more important neurons and judge whether the backdoor tickets contain these neurons. As shown in Figure 4 and Figure 5, each line represents one backdoor model after drawing EB tickets under the given pruning rate $p$ while the points on this line represent the proportion that the different pruning rate's key neurons in $f(\theta)$ also exist in $f^b(\theta)$. The line shows an upward trend compared with the more important neurons, which further verifies our suppose on the two model VGG16 and ResNet18 that most neurons and connections that play a leading role in the network account for a small proportion and are preserved in both normal subnetworks and backdoor subnetworks.

［#45］
Experiments show that the more important nodes are indeed retained in the backdoor tickets with a greater probability, thus affected tickets can still achieve high accuracy.

［#46］
![](./images/867746754064286603_4.jpg)

［#47］
Figure 4: The neuron prune relation between the normal VGG16 model $f(\theta)$ and backdoor model $f^b(\theta)$.

［#48］
![](./images/867746754064286603_5.jpg)

［#49］
Figure 5: The neuron prune relation between the normal RESNET18 model $f(\theta)$ and backdoor model $f^b(\theta)$.

## B.2 Experiment for intensity of attacks

［#50］
The hyperparameter $\alpha$, the proportion of backdoor data in the dataset $\mathcal{D}^b$, plays an important role in the attack effect. Obviously, a larger $\alpha$ makes the attack more successful. For different backdoor attacks, the attacker needs to set different values for model training to make the attacks effective [17].

［#51］
Figure 6 shows that the winning tickets drawn under different intensities (different backdoor data proportion) of backdoor attacks can achieve comparable accuracy to the normal subnetworks, which shows that backdoor attacks can not obviously affect the model's performance of normal retraining. The reason why affected EB tickets can still achieve high accuracy is thought-provoking. Interestingly, this makes the backdoor attack on EB Ticket more covert because we hardly distinguish the attack by CDA's change.

［#52］
![](./images/867746754064286603_6.jpg)

［#53］
Figure 6: The retraining accuracy of subnetworks drawn under different intensities of backdoor attacks (during searching procedure), where benign datasets are used to evaluate the performance of EB tickets in retrain process.