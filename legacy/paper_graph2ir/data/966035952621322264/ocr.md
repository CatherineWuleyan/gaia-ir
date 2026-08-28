# Enhancing Scalability in Recommender Systems through Lottery Ticket Hypothesis and Knowledge Distillation-based Neural Network Pruning

Regular Paper

Rajaram R, IIT Madras, India, raajaram100@gmail.com

Manoj Bharadhwaj, IIT Madras, India, cs20s056@cse.iitm.ac.in

Vasan VS, Pravartak, IIT Madras, India, vasan.vs@gmail.com

Nargis Pervin*, IIT Madras, India, nargisp@iitm.ac.in

## Abstract
This study introduces an innovative approach aimed at the efficient pruning of neural networks, with a particular focus on their deployment on edge devices. Our method involves the integration of the Lottery Ticket Hypothesis (LTH) with the Knowledge Distillation (KD) framework, resulting in the formulation of three distinct pruning models. These models have been developed to address scalability issue in recommender systems, whereby the complexities of deep learning models have hindered their practical deployment. With judicious application of the pruning techniques, we effectively curtail the power consumption and model dimensions without compromising on accuracy. Empirical evaluation has been performed using two real world datasets from diverse domains against two baselines. Gratifyingly, our approaches yielded a GPU computation-power reduction of up to 66.67%. Notably, our study contributes to the field of recommendation system by pioneering the application of LTH and KD.

## Introduction
Deep Learning has become a transformational force in the ever-expanding arena of artificial intelligence, revolutionizing various fields with its outstanding capacity to analyze massive volumes of data and derive insightful knowledge (Abarja, 2020). Specifically, Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are used to solve various tasks across domains including computer vision, natural language processing, and recommender systems. This is partly due to its illustrious performance and the allure of learning fine-grained

feature representations. Traditionally, collaborative filtering based recommendation systems (Herlocker et al., 2004) use prior user choices or activities to suggest new products or items to match the user preferences. For example, in movie recommendation system, user's histor-ical data on viewing and browsing along with movie features (like actor, director, genre, etc.) could be used to generate recommendations. However, these traditional methods suffer from well-known data-sparsity and scalability issues.

While prior works have implemented deep Learning based recommender systems and focused mainly on improving accuracy (Nagarnaik and Thomas, 2015; Zhang et al., 2017), scalability issues have largely been ignored (Zhou et al., 2004). Specifically, the complexity of deep learn-ing based techniques increases with the number of hyperparameters and the number of layers (size) in the model. The task becomes daunting in the practical scenario, when the model has to be scaled across multiple platforms and interfaces, from mobile phones to complex GPU servers. One plausible way to address this scalability issue is to make the model smaller,(i.e., lowering complexity) so that the latency, computation time, and power decreases (Ku-mar and Thakur, 2018), and thereby the models can be deployed on any platform. While the potential of CNN in recommendations is well accepted in the research community, advances in this direction is still in its nascent stage (Nasraoui and Petenes, 2003). In order to address this issue, one intriguing concept is the Lottery Ticket Hypothesis, which draws an analogy to solving puzzles. Imagine a hard issue is represented by a challenging puzzle (complex neural network). The Lottery Ticket Hypothesis postulates that inside a complex puzzle (neural net-work), there may be smaller, vital components that contribute to its overall effectiveness, just as some smaller pieces of the puzzle (smaller neural networks) may hold the secret to its solu-tion. These "winning tickets" can be found and used to increase the complex puzzle's (neural network's) effectiveness and efficiency, which will hasten the learning process. Another signif-icant concept is of Knowledge distillation, which resembles the relationship between a teacher and student, is another important concept. In this situation, a teacher, symbolizing a big, com-plicated model, teaches a student, representing a tiny, simple model. Knowledge distillation includes the teacher model communicating its insights to the student model in a distilled and more understandable form, much like how a teacher might reduce complex subjects down into

simpler lessons.

This paper proposes a novel method to prune a neural network efficiently which can be used to deploy on edge devices. To achieve this objective, we propose three pruning models that combine the Lottery ticket hypothesis (Frankle and Carbin, 2019) with Knowledge Distillation framework (Hinton et al., 2015). Our first model, namely, Structured Pruning with Show Attend and Distill (SP-SAD) aims to use structured pruning to make the model as sparse as possible during the training itself. In the second model, Lottery Ticket Hypothesis with Show Attend and Distill (LTH-SAD), we use the Lottery Ticket Hypothesis to re-initialize the parameters of the student model after pruning with the ones that were there during the start. In the third model, Sparse Student with Show Attend and Distill (SS-SAD), here the student model is first pruned using Lottery Ticket Hypothesis, this pruned model now acts as a student model in knowledge distillation process. In this context, user devices capable of conducting computations locally, as opposed to relying on a centralized server. Employing this pruning technique results in quicker response time and faster computation. Notably, for edge devices (eg., user's handheld mobile devices) optimizing power consumption is of utmost importance, which is attainable through the well-designed curation of a pruned model.

To evaluate the efficacy of our proposed approach (SP-SAD, LTH-SAD, SS-SAD), we have performed extensive experiments using two public datasets, where CIFAR-100 is from com- puter vision domain used for object detection, and the other is movie datasets, (IMDb and TMDb), used for recommendations. The proposed approaches (SP-SAD, LTH-SAD, SS-SAD) have been compared with two baselines. We have used the power consumption and the model size as the metric to compare the scalability of the models. According to a prior study by Kumar and Thakur (2018), with the reduction in the model size, the power and training time decreases. The experimental findings reveal that our proposed methods achieve comparable accuracy (up to 32.08% , 25.10% improvement in MSE and MAE, respectively) with reduced GPU power consumption (up to 66.67%) with 45% reduction in model size which in turn reduces the carbon footprint. The primary contributions of our study are the following:

1. Improving structured pruning during training of a convolutional neural network using an attention-based Knowledge Distillation technique.

2. Addressing the scalability issue of recommender systems with reduction in power consumption without compromising on accuracy.

To the best of our knowledge, this work is the first attempt to apply the lottery ticket hypothesis combined in the knowledge distillation process to train a recommender system architecture.

## Related works

### CNN in Recommendation systems
Initially designed for image processing tasks, convolutional Neural Networks (CNNs) have gained attention in recommendation systems due to their ability to capture complex patterns and learn hierarchical representations from data. Several studies have explored the integration of CNNs into recommendation systems to enhance their performance and address challenges such as cold start problems and sparsity issue. For instance, CNNs incorporated document context information into recommendation models, improving performance in document recommendation tasks (Chen et al., 2019). Similarly, Zhou et al. (2018) employed CNNs to model user interests and interactions for personalized click-through rate predictions in e-commerce platforms. However, performance of CNNs rely on the intensive parameter list and a large training dataset. On the other hand, traditional machine learning techniques, content-based filtering, and small neural networks are under-parameterized and run faster (Gordon et al., 2018). Thus, a plausible way to speed up recommendation systems is to use CNNs and make them sparser by pruning a neural network (Alford et al., 2018). Typically, there are two pruning techniques for neural networks, structured and unstructured pruning (Molchanov et al., 2017). Structured pruning involves removing entire structures or groups of parameters from a neural network. These structures can include whole filters, channels, or even layers (Alford et al., 2018). For example, if a layer contains 64 filters, structured pruning may remove a fixed percentage (e.g., 30%) of filters, resulting in a reduced layer with 45 filters (Xia et al., 2022). Unstructured pruning involves removing individual parameters (weights) from a neural network, regardless of their position in the model (Xie et al., 2021).

### Lottery Ticket Hypothesis

"Lottery Ticket Hypothesis" (Frankle and Carbin, 2019) is an intriguing pruning technique that casts doubt on how we understand deep learning models and their underlying structure. The theory states that hidden sub-networks, referred to as "winning tickets," inside an originally over-parameterized network may achieve comparable or even better performance with much fewer parameters. Applications of the lottery ticket hypothesis are majorly on object detection and various computer vision tasks. Girish et al. (2021) have mentioned the application of the Lottery Ticket Hypothesis on embedded edge devices.

### Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) transfers the "knowledge" in a neural network by moving the information from a large "teacher" network to a smaller "student" network. Knowledge distillation opens up new possibilities for deployment in resource-constrained contexts by encapsulating the essence of the teacher's expertise while maintaining performance. Extending Knowledge Distillation, a few recent studies have shown significant improvement in model compression (Romero et al., 2015; Yim et al., 2017; Tian et al., 2022; Bharadhwaj et al., 2022). This approach recently has as well found ample applications in transfer learning in cross-domain (Orbes-Arteaga et al., 2019; Asami et al., 2017) and continual learning (Li and Hoiem, 2017; Hou et al., 2018).

In this paper, we propose three models based on pruning technique that aims to reduce power consumption with an improvement in prediction accuracy measured by MSE and MAE.

### Methodology

#### Methodology Overview

This study focuses on developing a novel scalable pruning framework, where the goal is to reduce the power consumption of recommender systems. Overall, a smaller student model is trained using the knowledge distilled from a larger teacher model, and over the period, pruning of the student model is performed to make it sparse. Figure 1 shows the framework for pruning using KD and LTH. First, the input image is fed to both the teacher and the student model; the features of the image has been extracted from each layer of the teacher model as well as

![](./images/966035952621322264_1.jpg)

**Figure 1.** Pruning Framework using Knowledge Distillation and Lottery Ticket Hypothesis

from the student model. Now each layer of the teacher will transfer its features to the suitable
student layer through attention mechanism. After this whole process, training of the student
model begins. During the back-propagation while training, dynamic structured pruning is done
to make the model more sparse. And this is repeated till the "Winning Ticket" is found, i.e.,
till a small sparse neural network which yields nearly equivalent or greater accuracy than the
teacher model is attained. Based on the schematic diagram 1, here is a detailed overview of
how our models are structured:

1.  Structured Pruning with Show Attend and Distill (SP-SAD): This approach uses struc-
    tured pruning to make the student model as sparse as possible during the training itself.
    Here, in structured pruning we re-initialize the parameters of the student model after
    pruning with the ones of obtained in the previous step.

2.  Lottery Ticket Hypothesis with Show Attend and Distill (LTH-SAD): In this approach,
    we use the Lottery Ticket Hypothesis to re-initialize the parameters of the student model
    after pruning with the ones that were there during the start.

3.  Sparse Student with Show Attend and Distill (SS-SAD): Here, the student model is first
    pruned using Lottery Ticket Hypothesis (LTH), and then the training happens. Here, the
    output of LTH acts as a student model in the Knowledge distillation process.

### Methodology Details

Show attend, and Distill (Ji et al., 2021) is a technique to distill the weights of a neural network where

- Let $F^T = \{F^T_1, \dots, F^T_n\}$ be the features representing the outputs of various layers of the neural network where $F^T_i$ represents features of $i^{th}$ layer and $F^T_n$ represents the features of the pre-final layers of the teacher model, $n+1$ being the number of layers.

- Similarly, let $F^S = \{F^S_1, ..., F^S_m\}$ be the feature set of the student model where $F^S_i$ represents features of $i^{th}$ layer and $F^T_m$ represents the features of the pre-final layers of the teacher model, $m+1$ being the number of layers.

The next step is knowledge transfer which involves global average pooling of features from both the teacher and student models. The pairing intensity for knowledge transfer is determined by the distance computed from channel-wise pooled features. To calculate feature similarities, the method employs the concept of query and key pairs from attention mechanisms. Queries $Q_T$ are generated from teacher features, and keys $K_S$ from student features. They are represented as follows:

$$Q_T = f_Q(W^Q_n . \phi^{hw}(F^T_n)) \quad \text{and} \quad K_S = f_K(W^K_m . \phi^{hw}(F^S_m))$$

Here $\phi^{hw}(.)$ represents a global average pooling, and $f_Q, f_K$ are the activation functions of the query and the key. ($W^Q_n$ and ($W^K_m$ are the linear transition parameters for $n^{th}$ key and $m^{th}$ query.

Using softmax, probabilities are calculated with these keys and queries:
$$\alpha_t = \text{Softmax}([Q^T_n W^{Q-K}_1 K_{t,1} + ((P^n)^⊤) P^S_1 / \sqrt{d}, ..., Q^T_n W^{Q-K}_m K_{t,m} + ((P^n)^⊤) P^S_m / \sqrt{d}]) \tag{1}$$

Here, $W^{Q-K}_S$ are the bi-linear weights, and $P^S_m, P^T_n$ are positional encodings. $\alpha_t$ captures the probability of transferring knowledge from the teacher to the student. The distillation term is represented as:

$$\mathcal{L}_{Attention} = \sum_t \sum_S \alpha_{n,m} \parallel \widetilde{\phi}^C(h^T_n) - \widetilde{\phi}^C(\hat{h}^S_m) \parallel_2 \tag{2}$$

Where $\widetilde{\phi}^C$ combines channel average pooling with $L_2$ norm. The loss function for knowledge distillation is given by:

$$\mathcal{L}_{Student} = \mathcal{L}_{class} + \beta \mathcal{L}_{Attention} \tag{3}$$

Here, $\mathcal{L}_{class}$ is the classification loss, and $\beta$ balances the attention loss. This above equation shows the loss function for the knowledge distillation, where $\mathcal{L}_{class}$ is the ground truth classification loss which is calculated by cross-entropy loss. During the backprobagation this loss function is used to train the student model.

### Algorithm
Our work aims to increase the sparsity of a neural network iteratively without a significant drop in accuracy. To do this, we use the concept of attention-guided knowledge distillation (Xu et al., 2016) and a structured pruning technique. We use a teacher model with $n+1$ layers and a student model with $m+1$ layers and $m \ll n$. The algorithmic steps have been discussed in Algorithm 1.

**Input**: Pre-trained Teacher Network with $n$ layers

**Output**: A fully trained sparse student model with m layers

**Training Loop:** *for each epoch do*
    Compute $F^T$ and $F^S$
    $L = \text{Show Attend and Distill}(F^T, F^S)$;
    Back-propagation of the student model and optimization of $L$;
    **if** $\textit{epoch} \% \textit{args.prune\_every} == 0$ **then**
        Extract and prune the weight mask of the student model;
        Reinitialize the student model;
    **end**

### Algorithm 1: Iterative Pruning with Show Attend and Distill
Here, $F^T$ and $F^S$ are the feature sets of the teacher and student model, and $L$ is the loss obtained from the Show Attend and Distill model. To reinitialize the model, the same weights that were used in the previous step is used in case of structured pruning and in case of LTH-SAD, the weights are reinitialized using the lottery ticket hypothesis.

### Dataset Description
#### CIFAR100 dataset
CIFAR-100 (Krizhevsky, 2012) dataset consists of 60,000 color photos which are $32 \times 32$ sized color images for 100 object classes (with 600 photos in each class). We split the dataset into 50K training and 10K validation images. Each class represents a distinct category of object,

including cars, birds, dogs, cats, etc. The horizontal flipping and random cropping are used for data augmentation.

### Movie Dataset
We used two movie datasets IMDb and TMDb obtained from Abarja (2020). Based on the title and year of the film's release, these datasets were aggregated into a single dataset containing 4317 films released between 1916 and 2016. A training dataset with movies released between 2000 and 2013 (3819 movies) and a test dataset with movies released after 2013 (498 movies) were created from the dataset. Pre-processing procedures like data cleansing, data transformation, and feature extraction are carried out on both training and test data. The features are divided into four groups: social media, textual, categorical, and numerical features as listed in Table 1.

<table>
<thead>
  <tr>
    <th>Features</th>
    <th>Description</th>
    <th>Features</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Numerical features</td>
    <td>Budget Cost</td>
    <td>Textual Features</td>
    <td>Movie Title</td>
  </tr>
  <tr>
    <td></td>
    <td>Time Duration</td>
    <td></td>
    <td>Plot Keywords</td>
  </tr>
  <tr>
    <td></td>
    <td>Total Companies</td>
    <td></td>
    <td>Overview</td>
  </tr>
  <tr>
    <td></td>
    <td>Release Day</td>
    <td></td>
    <td>Tagline</td>
  </tr>
  <tr>
    <td></td>
    <td>Release Month</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>Release Year</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>Total Languages</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Social Features</td>
    <td>Actor/Actress Likes (FBL)</td>
    <td></td>
    <td>Director Likes</td>
  </tr>
  <tr>
    <td></td>
    <td>Cast Likes</td>
    <td></td>
    <td>Crew FBL</td>
  </tr>
  <tr>
    <td>Categorical Features</td>
    <td>Production Countries</td>
    <td></td>
    <td>Genres</td>
  </tr>
  <tr>
    <td></td>
    <td>Content Rating</td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>
Table 1. Categorised Features

### Experiments Findings
In this section, we present the experimental findings to assess the performance of the three proposed approaches, namely, SP-SAD, LTH-SAD, SS-SAD algorithms. All the experiments were carried out in Python 3.10.9 and having Linux-X11 (Ubuntu 22.04.2 LTS) computer specifications as CPU: 8Core/16 threads, RAM:64GB, NVIDIA Corporation GP102 [GeForce GTX 1080 Ti]. To verify that the results are not dataset dependant, we have chosen datsets from two distinct domains with diverse application scenarios.

<table>
<thead>
<tr>
<th>Parameters</th>
<th>Description</th>
<th>SP-SAD</th>
<th>LTH-SAD</th>
<th>SS-SAD</th>
<th>SAD</th>
</tr>
</thead>
<tbody>
<tr>
<td>Total Epochs</td>
<td>#training iterations</td>
<td>160</td>
<td>1200</td>
<td>200</td>
<td>240</td>
</tr>
<tr>
<td>Pruning Rate</td>
<td>% of weights removed</td>
<td>10%</td>
<td>5%</td>
<td>5%</td>
<td>-</td>
</tr>
<tr>
<td>Learning Rate</td>
<td>Step size to update model parameters</td>
<td>0.05</td>
<td>0.05</td>
<td>0.0005</td>
<td>0.05</td>
</tr>
<tr>
<td>Prune every</td>
<td>Interval duration (in epochs) till the model is pruned</td>
<td>20</td>
<td>75</td>
<td>20</td>
<td>-</td>
</tr>
<tr>
<td>Learning Decay</td>
<td>rate to lower the learning rate</td>
<td>0.001</td>
<td>0.0005</td>
<td>-</td>
<td>0.05</td>
</tr>
<tr>
<td>Learning Rate Scheduling</td>
<td>When the learning rate is altered (in epochs)</td>
<td>60,120</td>
<td>170,340,510</td>
<td>-</td>
<td>150,180,210</td>
</tr>
<tr>
<td>Beta ($\beta$)</td>
<td>Controls the overall attention loss</td>
<td>100</td>
<td>50</td>
<td>200</td>
<td>200</td>
</tr>
<tr>
<td>Temperature (T)</td>
<td>normalizes the softmax values</td>
<td>4</td>
<td>4</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<td>Alpha ($\alpha$)</td>
<td>Controls the teacher attention loss</td>
<td>0.9</td>
<td>0.8</td>
<td>0.9</td>
<td>0.9</td>
</tr>
<tr>
<td>Batch Size</td>
<td>Number of images fed in a batch</td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
</tr>
</tbody>
</table>

Table 2. Hyperparameter Tuning for SP-SAD, LTH-SAD, SS-SAD, and SAD

## Evaluation Metrics

To quantify the performance of our model, we employ standard metrics such as accuracy, MAE, and MSE, and compute GPU power consumption to measure scalability of the system.

## Accuracy

Accuracy measures the ratio of correct predictions to the total number of predictions.

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

## Mean Absolute Error (MAE) and Mean Squared Error (MSE)

Mean Absolute Error (MAE) calculates the average absolute difference between the predicted values and the actual values. Mean Squared Error (MSE) on the other hand measures the average of the squared differences between the predicted values and the actual values.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \quad \text{and} \quad \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $y_i$, $\hat{y}_i$ are the actual and predicted values of the target variable (ground truth) of $i^{th}$ observation, respectively.

## Experimental Findings on CIFAR100 Dataset

The effectiveness of the suggested approaches in terms of accuracy, has been demonstrated by comparison with SAD (Show Attend & Distill) (Ji et al., 2021) that incorporates the attention mechanism in the knowledge distillation mechanism. Here all the mentioned models were

<table>
<thead>
<tr>
<th>Method</th>
<th>Accuracy</th>
<th>% Size reduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>SAD</td>
<td>75.47%</td>
<td>0%</td>
</tr>
<tr>
<td>SP-SAD</td>
<td>73.00%</td>
<td>60%</td>
</tr>
<tr>
<td>LTH-SAD</td>
<td>73.03%</td>
<td>60%</td>
</tr>
<tr>
<td>SS-SAD</td>
<td>73.71%</td>
<td>70%</td>
</tr>
</tbody>
</table>

Table 3. Comparison with CIFAR100 Dataset

trained using WRN-40-2 (Zagoruyko and Komodakis, 2016) as their teacher model and WRN-16-2 (Zagoruyko and Komodakis, 2016) as their student model. The optimal hyperparameters used for the models are presented in Table 2. Table 3 depicts the comparison of our three proposed approaches, SP-SAD, LTH-SAD, SS-SAD with SAD. It should be noted that while the baseline contains all the parameters (100%), we apply pruning on the three proposed methods by varying the pruning percentage from 0%-100% with a pruning rate of 5% -10%. Finally, the wining ticket is chosen based on the best accuracy obtained. From Table 3 it can be observed that all the three proposed methods, SP-SAD, LTH-SAD, and SS-SAD achieve comparable accuracy (73% (60% pruned), 73.03% (60% pruned), 73.71% (70% pruned), respectively for SP-SAD, LTH-SAD, and SS-SAD) with a reduced model size in comparison to SAD.

### Experimental Findings on Movies Dataset

We experimented our models with a different dataset from movie domain and compared with SAD and DCNN Abarja (2020). DCNN (Abarja, 2020) discusses a Convolutional Neural Network (CNN) based approach to predict movie ratings based on movie attributes. We performed the experiments on the four types of features (numerical, social, categorical, and textual) independently. Table 4 depicts the performance comparison of the three models against MSE and MAE. The boldface denotes the best performing value while the second best value is underlined. It is evident from the table that LTH-SAD strategy achieves the lowest MSE and MAE with social and textual features. LTH-SAD with social features accounts for MAE and MSE values as 0.81 and 1.106 with an improvement of up to 7.9% and 10%, respectively. Similarly, LTH-SAD with textual features achieves 25.1% and 32% improvements on MAE (0.871) and MSE (1.319) values. This could be attributed to the fact that it most likely combines sophisticated information transferred with attention mechanisms. However, the observed performance of LTH-SAD with numerical features diverges from its anticipated outcome and require further

exploration. However, we conceive that such outcome may be ascribed to a trade-off between

![](./images/966035952621322264_2.jpg)
(a) Comparison of MAE of LTH-SAD, SAD & DCNN

![](./images/966035952621322264_3.jpg)
(b) Comparison of MSE of LTH-SAD, SAD & DCNN

**Figure 2.** Comparison of Textual Feature's MAE & MSE

![](./images/966035952621322264_4.jpg)
(a) Comparison of MAE of LTH-SAD, SAD & DCNN

![](./images/966035952621322264_5.jpg)
(b) Comparison of MSE of LTH-SAD, SAD & DCNN

**Figure 3.** Comparison of Social Feature's MAE & MSE

complexity and generalization. Notably, attention mechanisms might not be as effective for cer- tain types of data. To address this, incorporation of early stopping criteria and selecting features that does not introduce noise or complexities may improve the performance. The results have been presented in Figure 2 and Figure 3. It is important to highlight that the periodic spikes in the LTH-SAD graph correspond to instances where the model is re-initialized using LTH (Lottery Ticket Hypothesis) after the pruning process. This approach ensures that the model begins learning anew each time, which leads to gradual but improved outcomes over time.

Scalability: We have used the GPU Power Consumption as a parameter of scalability where GPU Power Consumption takes into account the combined power consumed during training phase as well as inferencing phase. We have used movie dataset to experiment the power con- sumption by LTH-SAD and compared with baseline model SAD. Figure 4 shows the power consumption with the models with categorical features. It is evident from the figure that SAD consumes 75W power for 120 minutes whereas LTH-SAD consumes 75W power for 40 min- utes to yield a better MSE and MAE. This clearly shows LTH-SAD consumes 66.67% less

<table>
<thead>
  <tr>
    <th>Methods</th>
    <th colspan="2">Categorical Features</th>
    <th colspan="2">Numerical Features</th>
  </tr>
  <tr>
    <th></th>
    <th>MAE</th>
    <th>MSE</th>
    <th>MAE</th>
    <th>MSE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DCNN</td>
    <td>0.8594</td>
    <td>1.22</td>
    <td>0.7455</td>
    <td>0.98</td>
  </tr>
  <tr>
    <td>SAD</td>
    <td>0.8143</td>
    <td>1.183</td>
    <td>0.7714</td>
    <td>1.041</td>
  </tr>
  <tr>
    <td>LTH-SAD</td>
    <td>0.8212<br>(-8%)</td>
    <td>1.151<br>(+2.7%)</td>
    <td>0.8607<br>(-11.57%)</td>
    <td>1.255<br>(-20.5%)</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Methods</th>
    <th colspan="2">Social Features</th>
    <th colspan="2">Textual Features</th>
  </tr>
  <tr>
    <th></th>
    <th>MAE</th>
    <th>MSE</th>
    <th>MAE</th>
    <th>MSE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DCNN</td>
    <td>2.086</td>
    <td>5.028</td>
    <td>1.375</td>
    <td>2.593</td>
  </tr>
  <tr>
    <td>SAD</td>
    <td>0.88</td>
    <td>1.229</td>
    <td>1.163</td>
    <td>1.942</td>
  </tr>
  <tr>
    <td>LTH-SAD</td>
    <td>0.81<br>(+7.9%)</td>
    <td>1.106<br>(+10%)</td>
    <td>0.871<br>(+25.1%)</td>
    <td>1.319<br>(+32%)</td>
  </tr>
</tbody>
</table>

Table 4. Performance Comparison of Features in Movie Recommender System

![](./images/966035952621322264_6.jpg)

Figure 4. GPU Power Consumption of LTH-SAD & SAD

power with a 45% reduced model size compared to SAD considering the dataset with categorical features. Similar results have been observed when other three features (numerical feature (9.09% improvement), social feature (33.3% improvement), and categorical feature (8.33% improvement)) are considered. Overall, the outcomes of this study underscore the potential of our approach in facilitating the realization of efficient recommender systems suitable for real-world edge device implementations.

## Conclusion and Discussions
This research describes a unique method for improving the effectiveness and performance of deep neural network models by combining knowledge distillation with channel pruning. We allowed the student model to collect and distill the useful knowledge from the instructor model through the use of an attention-based framework and pruning, which resulted in enhanced generalisation and performance. To further minimise the size and complexity of the student model without compromising the accuracy and model size trade-off, we implemented three channel pruning techniques. The efficacy of the models has been demonstrated by experiments on two diverse datasets. By combining knowledge distillation with structured pruning, our proposed

models were able to produce small, effective models that are suitable for use in the contexts with limited resources. The results showed that, LTH-SAD achieved comparable accuracy (up to 32.08% , 25.10% improvement in MSE and MAE, respectively) with reduced GPU power consumption (up to 66.67%) with 45% reduction in model size. The outcomes confirm that compared to SAD (Ji et al., 2021), LTH-SAD performs better with the textual and social fea- tures. Our approaches further attained favourable outcome in the context of recommender systems for rating prediction to identify key qualities or connections between the input data (such as reviews, user preferences, or movie attributes) and the projected ratings. Future work might also study the use of our method in numerous domains and real-world situations, as well as other optimisation strategies.

## References

Abarja, R. 2020. "Movie Rating Prediction using Convolutional Neural Network based on His- torical Values," *International Journal of Emerging Trends in Engineering Research* (8), pp. 2156-2164.

Alford, S., Robinett, R., Milechin, L., and Kepner, J. 2018. "Pruned and Structurally Sparse Neural Networks," In: *2018 IEEE MIT Undergraduate Research Technology Conference (URTC)*.

Asami, T., Masumura, R., Yamaguchi, Y., Masataki, H., and Aono, Y. 2017. "Domain adap- tation of DNN acoustic models using knowledge distillation," In: *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

Bharadhwaj, M., Ramadurai, G., and Ravindran, B. 2022. "Detecting Vehicles on the Edge: Knowledge Distillation to Improve Performance in Heterogeneous Road Traffic," In: *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*.

Chen, H., Fu, J., Zhang, L., Wang, S., Lin, K., Shi, L., and Wang, L. 2019. "Deformable Con- volutional Matrix Factorization for Document Context-Aware Recommendation in Social Networks," *IEEE Access* (7), pp. 66,347-66,357.

Frankle, J., and Carbin, M. 2019. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks," .

Girish, S., Maiya, S. R., Gupta, K., Chen, H., Davis, L. S., and Shrivastava, A. 2021. "The Lottery Ticket Hypothesis for Object Recognition," In: *Proceedings of the IEEE/CVF Con- ference on Computer Vision and Pattern Recognition (CVPR)*.

Gordon, A., Eban, E., Nachum, O., Chen, B., Wu, H., Yang, T.-J., and Choi, E. 2018. "Mor- phNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks," .

Herlocker, J. L., Konstan, J. A., Terveen, L. G., and Riedl, J. T. 2004. "Evaluating Collaborative Filtering Recommender Systems," *ACM Trans Inf Syst* (22:1), p. 5-53.
URL `https://doi.org/10.1145/963770.963772`

Hinton, G., Vinyals, O., and Dean, J. 2015. "Distilling the Knowledge in a Neural Network," .

Hou, S., Pan, X., Loy, C. C., Wang, Z., and Lin, D. 2018. "Lifelong Learning via Progressive Distillation and Retrospection," In: Ferrari, V., Hebert, M., Sminchisescu, C., and Weiss, Y. (eds.) *Computer Vision - ECCV 2018*, Cham: Springer International Publishing.

Ji, M., Heo, B., and Park, S. 2021. "Show, Attend and Distill:Knowledge Distillation via Attention-based Feature Matching,".

Krizhevsky, A. 2012. "Learning Multiple Layers of Features from Tiny Images," *University of Toronto* .

Kumar, P., and Thakur, R. S. 2018. "Recommendation system techniques and related issues: a survey," *International Journal of Information Technology* (10), pp. 495-501.

Li, Z., and Hoiem, D. 2017. "Learning without Forgetting," .

Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz, J. 2017. "Pruning Convolutional Neural Networks for Resource Efficient Inference," .

Nagarnaik, P., and Thomas, A. 2015. "Survey on recommendation system methods," In: *2015 2nd International Conference on Electronics and Communication Systems (ICECS)*.

Nasraoui, O., and Petenes, C. 2003. "An intelligent Web recommendation engine based on fuzzy approximate reasoning," In: *The 12th IEEE International Conference on Fuzzy Systems, 2003. FUZZ '03.*, vol. 2.

Orbes-Arteaga, M., Cardoso, J., Sørensen, L., Igel, C., Ourselin, S., Modat, M., Nielsen, M., and Pai, A. 2019. "Knowledge distillation for semi-supervised domain adaptation," .

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. 2015. "FitNets: Hints for Thin Deep Nets," .

Tian, Y., Krishnan, D., and Isola, P. 2022. "Contrastive Representation Distillation," .

Xia, M., Zhong, Z., and Chen, D. 2022. "Structured Pruning Learns Compact and Accurate Models," .

Xie, H., Jiang, W., Luo, H., and Yu, H. 2021. "Model compression via pruning and knowl- edge distillation for person re-identification," *Journal of Ambient Intelligence and Human- ized Computing* (12).

Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., and Bengio, Y. 2016. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention," .

Yim, J., Joo, D., Bae, J., and Kim, J. 2017. "A Gift from Knowledge Distillation: Fast Op- timization, Network Minimization and Transfer Learning," In: *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Zagoruyko, S., and Komodakis, N. 2016. "Wide Residual Networks," .

Zhang, S., Yao, L., and Sun, A. 2017. "Deep Learning based Recommender System: A Survey and New Perspectives," *CoRR* (abs/1707.07435).
URL http://arxiv.org/abs/1707.07435

Zhou, B., Hui, S., and Chang, K. 2004. "An intelligent recommender system using sequential Web access patterns," In: *IEEE Conference on Cybernetics and Intelligent Systems, 2004.*, vol. 1.

Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, C., Zhu, X., and Gai, K. 2018. "Deep Interest Evolution Network for Click-Through Rate Prediction," .