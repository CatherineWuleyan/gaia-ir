# A LOTTERY TICKET HYPOTHESIS FRAMEWORK FOR LOW-COMPLEXITY DEVICE-ROBUST NEURAL ACOUSTIC SCENE CLASSIFICATION
［#1］
Technical Report

［#2］
Hao Yen¹, Chao-Han Huck Yang¹, Hu Hu¹, Sabato Marco Siniscalchi¹², Qing Wang³, Yuyang Wang³,
Xianjun Xia⁴, Yuanjun Zhao⁴, Yuzhong Wu⁴, Yannan Wang⁴, Jun Du³, Chin-Hui Lee¹

［#3］
¹School of Electrical and Computer Engineering, Georgia Institute of Technology, GA, USA
²Computer Engineering School, University of Enna Kore, Italy
³University of Science and Technology of China, HeFei, China
⁴Tencent Media Lab, Tencent Corporation, China

## ABSTRACT
［#4］
We propose a novel neural model compression strategy combining data augmentation, knowledge transfer, pruning, and quantization for device-robust acoustic scene classification (ASC). Specifically, we tackle the ASC task in a low-resource environment leveraging a recently proposed advanced neural network pruning mechanism, namely Lottery Ticket Hypothesis (LTH), to find a sub-network neural model associated with a small amount of non-zero model parameters. The effectiveness of LTH for low-complexity acoustic modeling is assessed by investigating various data augmentation and compression schemes, and we report an efficient joint framework for low-complexity multi-device ASC, called *Acoustic Lottery*. Acoustic Lottery could largely compress an ASC model and attain a superior performance (validation accuracy of 79.4% and Log loss of 0.64) compared to its not compressed seed model. All results reported in this work are based on a joint effort of four groups, namely GT-USTC-UKE-Tencent, aiming to address the "Low-Complexity Acoustic Scene Classification (ASC) with Multiple Devices" in the DCASE 2021 Challenge Task 1a.

［#5］
Index Terms— Lottery ticket hypothesis, Teacher-student learning, Acoustic scene classification, and Device-robustness

## 1. INTRODUCTION
［#6］
Acoustic scene classification (ASC) aims to recognize a set of given environment classes (e.g., airport and urban park) from real-worlds sound examples. Analysis and learning to predict acoustic scene sounds are important topics associated with various mobile and on-device intelligent applications [1]. The Detection and Classification of Acoustic Scenes and Events (DCASE) challenges [2, 3, 4, 5] provide a comprehensive evaluation platform and benchmark data to encourage and boost sound scene research communities. DCASE 2021 Task 1a [6] focuses on developing low-complexity acoustic modeling (AM) solutions for predicting sounds recorded from multiple devices (e.g., electret binaural microphones, smartphones, and action cameras). The goal is to design a device-robust ASC system preserving generalization power over audios recorded by different devices, and highlighting the importance of low-complexity requirements.

［#7］
From previous DCASE challenges, we observed that several competitive ASC systems [7, 8, 9] benefited from large-scale convolutional neural models combined with several data augmentation schemes, but whether we can attain the generalization power of those complex models with a low-complexity architecture is the research goal to be addressed in DCASE 2021 challenge. To this end, we focus on addressing two basic questions: (i) Are some well-performed device-robust ASC models overparameterized? (ii) Can we take advantage of some overparameterized models to design a low-complexity ASC framework on multi-device data?

［#8］
![](./images/867760083600146646_1.jpg)

［#9］
Figure 1: The proposed *Acoustic Lottery* (AL) framework.

［#10］
In the quest for addressing the above questions, we deployed a novel framework, namely "Acoustic Lottery," for DCASE 2021 Task 1a, which will be described in the following sections. As shown in Figure 1, our Acoustic Lottery system consists of (a) a data augmentation process to improve model generalization, (b) a teach-student learning mechanism to transfer knowledge, from a large teacher model to a small student model, (c) a Lottery Ticket Hypothesis [10] based pruning method to deliver a low-complexity model, (d) a two-stage fusion technique [11] to improve model prediction, and finally (e) a quantization block to deploy a final model owing less than 128 KB non-zero parameters, which is the requirement of Task 1a. A detailed presentation of each block in Figure 1 is discussed in the following sections.

## 2. LOW-COMPLEXITY ACOUSTIC MODELING FRAMEWORK
### 2.1. Data Augmentation Strategy
［#11］
In some previous works [7, 11, 12, 13], data augmentation strategies played a key role to attain competitive log-loss results on the DCASE 2020 validation set [9]. With the goal of deploying a seed model with good generalization capabilities to deal with the multiple device acoustic condition, the first module (Figure. 1 (a)) of our

［#12］
```
Algorithm 1 LTH for Device-Robust Acoustic Modeling
 1. Input: a model, $G_0$; augmented sound data, $D$.
 2. Randomly Initialize Weights ($\Theta_0$).
 3. Initialize Model: $G_0(\Theta_0) \to G_1$
 4. For $t = 1, ..., T$: # Pruning Searching Iterations
 5.    For $e = 1, ..., E$: # Gradient Training Epochs
 6.        $\Theta_e \to \Theta$: TSL-train $G_t$ with $D$ for its final weights ($\Theta_t$)
 7.    If $t < T$: # LTH Pruning Strategy
 8.        Mask($\Theta_t$) to get a pruned graph $G_p$ from $G_0$
 9.        Load homologous initial weights $\Theta_p \in \Theta_0$ from $G_p$
 10.       Update target model $G_p(\Theta_p) \to G_{t+1}$
 11. Output: A well-trained pruned model $G_T(\Theta_T)$
```

［#13］
DCASE 2021 system thereby builds upon data augmentation methods investigated in [7]. To be specific, we use i) Mixup [14], ii) Spectrum augmentation [15], iii) Spectrum correction [16], iv) Pitch shift, v) Speed change, vi) Random noise, and vii) Mix audios. The reverberation data described in [7] is not used in our experiments.

### 2.2. Teacher-Student Learning (TSL)
［#14］
Teacher-Student Learning (TSL), also named as Knowledge Distillation (KD), is a widely investigated approach for model compression [17, 18]. Specifically, it transfers knowledge from a large and complex deep model (teacher model) to a smaller one (student model). The main idea is to establish a framework that makes the student directly mimicking the final prediction of teacher. Formally, the soften outputs of a network can be computed by $p = softmax(\frac{\alpha}{\tau})$, where $\alpha$ is the vector of logits (pre-softmax activations) and $\tau$ is a temperature parameter to control the smoothness [17]. Accordingly, the distillation loss for soft logits can be written as the Kullback-Leibler divergence between the teacher and student soften outputs. In this work, we followed the approaches in [7] to build a large two-stage ASC system, serving as the teacher model. Then a teacher-student learning method is used to distill knowledge to a low-complexity student model, as shown in Figure 1 (b).

### 2.3. Lottery Ticket Hypothesis Pruning
［#15］
Next we have investigated advanced pruning techniques to further reduce non-zero model parameters of the student. Although neural network pruning methods often negatively affect both model prediction performance and generalization power, a recent study, referred to as Lottery Ticket Hypothesis [10] (LTH), showed a quite surprising phenomenon, namely pruned neural networks (sub-networks) could be trained attaining a performance that was equal to or better than the not pruned original model if the not pruned parameters were set to the same initial random weights used for the non-pruned model. Interestingly, LTH-based low-complexity neural models had proven competitive prediction performance on several image classification tasks [10] and recently have been supported with some theoretical findings [19] related to overparameterization.

［#16］
Algorithm Design: In **Algorithm 1**, we detail our approach under the Acoustic Lottery framework: In step (1), we first choose a model with its original neural architecture (e.g., Inception in our case) $G_0$ and record its initial weights parameters $\Theta_0$ in step (2). In our work, we incorporate teacher-student learning framework discussed in Section 2.2 with the goal of mimic prediction accuracy and generalization adapted of the teacher acoustic model - a complex model trained separately. At the end of each training phase, a pruning iteration is started if the current iteration $t$ is less than $T$. The LTH pruning searches for a low-complexity model in steps (7) through (10).

［#17］
From our empirical findings in DCASE 2021 Task 1A data, we found that the proposed Acoustic Lottery only needs one or two ($T$=1 or 2 in **Algorithm 1**) searching iteration(s) to find a good low-complexity acoustic model without a significant drop in the ASC classification accuracy compared to the high-complexity teacher model on the validation set. To select the mask function in step (8), we evaluate three major LTH strategies, namely: (i) large-final; (ii) small weights, and (iii) global small weights, which were proposed in [10]. We found the small weights strategy allows us to attain better trade-off between classification accuracy and compression rate compared to the other two mentioned methods as shown in Figure 2. Therefore, we selected "small weights" as pruning strategy to be used in our final submission. Finally, a well-trained pruned student acoustic model is deployed in step (10) of **Algorithm 1**

［#18］
![](./images/867760083600146646_2.jpg)

［#19］
(a) Validation Loss
(b) Validation Accuracy

［#20］
Figure 2: We compared empirical performance of different LTH-masking strategies [20] versus sparsity level (weights remaining).

［#21］
Visualization: To better interpret weights distribution in an LTH-pruned neural acoustic model, we visualize a shallow inception model (excluding convolutional layers due to their dimensional conflicts) on Index 3 in Table 1 and its LTH-pruned results as Index 5 in Table 1 shown in Figure 3. In Figure 3b, we can observe that the proposed Acoustic Lottery framework can discover a well-trained model using only sparse weights with up to a $149 \times$ compression rate.

［#22］
![](./images/867760083600146646_3.jpg)

［#23］
(a) Shallow Inception (SIC)
(b) LTH-Pruned SIC

［#24］
Figure 3: Visualized of layer-wise weights distribution by the LTH approach applied to the student neural acoustic model: (a) Shallow Inception (SIC) student and (b) LTH-Pruned SIC student.
```

### 2.4. Two-Stage Fusion and Multi-Task Learning

［#25］
To boost ASC performance, we follow the investigation in the two-stage ASC scheme discussed in [11], where the relationship between the 3-class and 10-class ASC systems were exploited to boost the 10-class ASC system. This step is carried out in the module (d) in Figure 1. The key idea is that the labels of the two subtasks, 3-class and 10-class problems, differ in the degree of abstraction and using the two labels together could be helpful. In our setup, the 3-class classifier classifies an input scene audio into one of three broad classes: in-door, out-door, and transportation. This 3-class classification way is from our prior knowledge that scene audios can be roughly categorized into such three classes. The 10-class classifier is actually the main classifier. Each audio clip should belong to one of the three / ten classes. The final acoustic scene class is chosen by the score fusion of those two classifiers. If we let $\mathbb{C}^1$ and $\mathbb{C}^2$ denote the set of three broad classes, and ten classes, respectively, and let $F^1$ and $F^2$ indicate the output of the first and second classifier, respectively. The final predicted class $Class(x)$ for the input $x$ is:

［#25］
$$
Class(x) = \underset{q,(p \in \mathbb{C}^1, q \in \mathbb{C}^2, p \supset q)}{\text{argmax}} F_p^1(x) * F_q^2(x),
$$

［#25］
where $p \supset q$ means that $p$ can be thought of a super set of $q$. For example, transportation class is the super set for bus, tram, and metro classes. Therefore, the probability of an input audio clip to be from the public square scene is equal to the product of the probability of out-door place, $F_p^1(x)$, and that of public square, $F_q^2(x)$.

［#26］
However, the two ASC classifiers are trained separately, which means the total parameters will be doubled. In [8], the authors argued that joint training of two subtasks could be even more efficient. Specifically, the 3-class classifier and the 10-class classifier can be learned in a multi-task learning (MTL) [21] manner. The two classifiers can share some parameters, where only output layers are different. MTL is expected to perform as well as two-stage method but save parameters. We thus study that setting as an ablation module in our experimental section.

### 2.5. Quantization for Model Compression

［#27］
As the main goal is to deploy a system with a size within 128 Kilobytes (KB), we further use a post-training quantization method with dynamic range quantization (DRQ), as shown in Figure 1 (e). DRQ is the simplest form of post-training quantization, which statically quantizes only weights from floating point to integer, which has 8-bits of precision. Moreover, activations are dynamically quantize based on their range to 8-bits. Leveraging upon DRQ, we thus convert our neural acoustic model from a 32-bit format to a 8-bit format, which compresses the model size to about $1/4$ of its original size as our final model.

---

## 3. EXPERIMENTAL SETUP & RESULTS

### 3.1. Feature Extraction

［#28］
We follow the same settings from DCASE 2020 Task-1a extracting acoustic features for DCASE 2021 Task-1a [6] before using the features to train low-complexity described in Section 2 and Figure 1. Log-mel filter bank (LMFB) features were used in our experiments as audio features. The input audio waveform is analyzed with a 2048 SFFT points, a window size of 2048 samples, and a frame shift of 1024 samples. Thus the final input tensor size is thus $423 \times 128 \times 3$ for Task 1a. Before feeding the speech feature tensors into CNN classifier, we scaled each feature value into [0,1].

### 3.2. Model Training

［#29］
All ASC systems are evaluated on the DCASE 2020 task1a development data set [5], which consists of $\sim$14K 10-second single-channel train audio clips and $\sim$3K test audio clips recorded by 9 different devices, including real devices A, B, C, and simulated device s1-s6. Only device A, B, C, s1-s3 are in the training set; whereas, devices s4-s6 are unseen in the training phase. The greatest amount of training audio clips are recorded with device A, namely over 10K audio clips. In the test set, the number of waveforms from each device is the same.

［#30］
We use two different Inception [23] models as our target models, namely Shallow Inception (SIC) and Large Inception (LIC). SIC has two inception blocks whereas LIC has three inception blocks and more filters. The size computed by the way recommended in [22] of the original SIC and LIC are 503KB and 3528KB, respectively. All Inception models in this work are built with Keras based on Tensorflow2. Stochastic gradient descent (SGD) with a cosine-decay-restart learning rate scheduler is used to train all deep models. Maximum and minimum learning rates are 0.1, and 1e-5, respectively. In our final submission, all development data is used. And due to there is no validation data, we use the output of model when learning rate hits the minimum number.

### 3.3. Results on Task 1a

［#31］
In Table 1, we report only some of the evaluation results for low-complexity models collected on Task 1a due to space constraints. Two inception models: (i) shallow inception model (SIC) and (ii) large inception model (LTC), are investigated under the proposed Acoustic Lottery framework. By evaluating several low-complexity strategies shown in 1. From the results, Index (0) is the official baseline, which has the size of 90.3KB but very low accuracy and high log loss. Index (1) and Index (2) are results from [7], where a two-stage system is used. Although they achieve very good performance (80.1% for two-stage FCNN and 81.9% for two-stage ensemble), their size is very large, which are 132MB and 332MB, respectively. It should be noted that the reverberation augmented data is used for Index (1) and (2) but not for others.

［#32］
The Index (3-11) in Table 1 are results of SIC. We here perform the ablation study for each method we propose. Index (3) is the SIC baseline, which has the size of 503KB, accuracy of 67.8%, and log loss of 0.954. With the use of TSL, shown as Index (4), we can improve the accuracy and log loss while keeping the model size unchanged. We use the two-stage FCNN model, shown as Index (1), as the teacher model. Index (5) shows the result of using LTH, where we can significantly reduce the model size (around $149\times$ compression rate. Although model parameters are reduced in a huge scale, the model performance shows much better than the SIC baseline: Index (3). This verifies our argument that the models are overparameterized a lot.

［#33］
Index (6) and (7) shows the results by only using two-stage fusion or MTL. From the results, we can see the two-stage can boost the performance, but the method will double the model size. By using a compromise method, MTL, can work in the same manner but save parameters. However, it's slightly worse than using two-stage. Index (8) shows the result by only using quantization. The model parameters are quantized from float32 to float8. Although it obtains a $4\times$ compression rate, the performance worsens when compared with the SIC baseline. However, according to our experiments, we find that the ensemble of 4 quantized models shows better results than an unquantized model, which shows the potential of quantization. With the combination of proposed approaches, we

［#34］
Table 1: Experimental results on 2021 Task 1a. 'TSL' means performing teacher-student learning. 'LTH' means pruning with the Lottery Ticket Hypothesis algorithm. 'Two-stage' means using a two-stage fusion system. 'MTL' means using multi-task learning system. 'Quant' means using quantization on model parameters (float32 to float8). 'Aug' means using extra augmented data (Methods in Section 2.1). 'System size' is according to non-zero parameters [6]. All 'Y' in the table means we used that method. Acc. indicates validation accuracy.

［#35］
<table>
  <thead>
    <tr>
      <th>Idx.</th>
      <th>System</th>
      <th>TSL</th>
      <th>LTH</th>
      <th>Two-stage</th>
      <th>MTL</th>
      <th>Quant</th>
      <th>Aug</th>
      <th>System size</th>
      <th>Acc. %</th>
      <th>Log Loss</th>
    </tr>
    <tr>
      <th>(0)</th>
      <th>Official Baseline [22]</th>
      <th>-</th>
      <th>-</th>
      <th>-</th>
      <th>-</th>
      <th>-</th>
      <th>-</th>
      <th>90.3KB</th>
      <th>47.7</th>
      <th>1.473</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(1)</td>
      <td>Two-stage FCNN [11]</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>132MB</td>
      <td>80.1</td>
      <td>0.795</td>
    </tr>
    <tr>
      <td>(2)</td>
      <td>Two-stage Ensemble [11]</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>332MB</td>
      <td>81.9</td>
      <td>0.829</td>
    </tr>
    <tr>
      <td>(3)</td>
      <td>SIC</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>503KB</td>
      <td>67.8</td>
      <td>0.954</td>
    </tr>
    <tr>
      <td>(4)</td>
      <td>SIC</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>503KB</td>
      <td>68.9</td>
      <td>0.919</td>
    </tr>
    <tr>
      <td>(5)</td>
      <td>SIC</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>296KB</td>
      <td>68.2</td>
      <td>0.914</td>
    </tr>
    <tr>
      <td>(6)</td>
      <td>SIC</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1006 KB</td>
      <td>68.9</td>
      <td>0.914</td>
    </tr>
    <tr>
      <td>(7)</td>
      <td>SIC</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>504KB</td>
      <td>68.0</td>
      <td>0.915</td>
    </tr>
    <tr>
      <td>(8)</td>
      <td>SIC</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>126KB</td>
      <td>66.9</td>
      <td>0.972</td>
    </tr>
    <tr>
      <td>(9)</td>
      <td>SIC</td>
      <td>Y</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1006KB</td>
      <td>69.2</td>
      <td>0.874</td>
    </tr>
    <tr>
      <td>(10)</td>
      <td>SIC</td>
      <td>Y</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>252KB</td>
      <td>68.4</td>
      <td>0.906</td>
    </tr>
    <tr>
      <td>(11)</td>
      <td>SIC</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>74KB</td>
      <td>67.7</td>
      <td>0.931</td>
    </tr>
    <tr>
      <td>(12)</td>
      <td>LIC</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>3434KB</td>
      <td>67.64</td>
      <td>1.019</td>
    </tr>
    <tr>
      <td>(13)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>3434KB</td>
      <td>67.91</td>
      <td>1.002</td>
    </tr>
    <tr>
      <td>(14)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2060KB</td>
      <td>67.64</td>
      <td></td>
    </tr>
    <tr>
      <td>(15)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1374KB</td>
      <td>68.22</td>
      <td></td>
    </tr>
    <tr>
      <td>(16)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>687KB</td>
      <td>68.59</td>
      <td>0.925</td>
    </tr>
    <tr>
      <td>(17)</td>
      <td>LIC</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>7056 KB</td>
      <td>70.0</td>
      <td>0.848</td>
    </tr>
    <tr>
      <td>(18)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>412KB</td>
      <td>70.8</td>
      <td>0.833</td>
    </tr>
    <tr>
      <td>(19)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>687KB</td>
      <td>74.98</td>
      <td>0.781</td>
    </tr>
    <tr>
      <td>(20)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>412KB</td>
      <td>78.4</td>
      <td>0.654</td>
    </tr>
    <tr>
      <td>(21)</td>
      <td>LIC</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>Y</td>
      <td>103KB</td>
      <td>78.2</td>
      <td>0.723</td>
    </tr>
    <tr>
      <td>(22)</td>
      <td>Ensemble of LICs</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>Y</td>
      <td>Y</td>
      <td>117KB</td>
      <td>78.7</td>
      <td>0.644</td>
    </tr>
    <tr>
      <td>(23)</td>
      <td>Ensemble of LICs</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>218KB</td>
      <td>79.0</td>
      <td>0.637</td>
    </tr>
    <tr>
      <td>(24)</td>
      <td>Ensemble of LICs and SICs</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>-</td>
      <td>-</td>
      <td>Y</td>
      <td>222KB</td>
      <td>79.4</td>
      <td>0.640</td>
    </tr>
  </tbody>
</table>

［#36］
can further boost the performance of SIC model, as shown in Index (9) to (11) of Table 1. We can at most compress the SIC model to 0.9KB, shown as Index (11), with even better performance than SIC baseline. As for LIC models, shown in Index (12) to (21), the same conclusions as SIC can be observed. Furthermore, when training by augmented data, system robustness can be further boosted. As for LIC, we can at most compress it to 687KB, with the best log loss of 0.925. We can observed from Index (14) and (15) that although the model size decreases (from 60% to 20%), we can achieve an even more better accuracy (from 67.64% to 68.59%) and log loss.

［#37］
For our final submitted four systems: four "two-stage ensembles" of different LIC and SIC models with LTH pruning are selected. We obtain SICs and LICs from different training epochs by training with different combinations of data augmentation strategies and training criterion (one-hot labels or TS learning). Specifically, for system (a), we use two 3-class LICs, three 10-class LICs and one 10-class SIC. The results of system (1) on development set is specified in Index (22) of Table 1. The other three submitted systems are not tested on development set due to the time and resources limitation.

## 4. DISCUSSION & CONCLUSION

［#38］
As low-complexity acoustic modeling, a lottery ticket hypothesis framework, Acoustic Lottery, is proposed and provides competitive results. As the very first attempt on applying LTH for acoustic learning and modeling, our future works included theoretical analysis on the success of LTH and its relationship between knowledge distillation for different acoustic and robust speech processing tasks [24]. We will open source our proposed framework to the community.

## 5. REFERENCES























