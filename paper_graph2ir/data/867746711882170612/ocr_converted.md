# Evaluating Lottery Tickets Under Distributional Shifts

［#1］
Shrey Desai$^{*,1}$ Hongyuan Zhan$^2$ Ahmed Aly$^2$
［#1］
$^1$The University of Texas at Austin $^2$Facebook Assistant
shreydesai@utexas.edu
{hyzhan, ahhegazy}@fb.com

## Abstract
［#2］
The Lottery Ticket Hypothesis (Frankle and Carbin, 2019) suggests large, over-parameterized neural networks consist of small, sparse subnetworks that can be trained in isolation to reach a similar (or better) test accuracy. However, the initialization and generalizability of the obtained sparse subnetworks have been recently called into question. Our work focuses on evaluating the initialization of sparse subnetworks under distributional shifts. Specifically, we investigate the extent to which a sparse subnetwork obtained in a source domain can be re-trained in isolation in a dissimilar, target domain. In addition, we examine the effects of different initialization strategies at transfer-time. Our experiments show that sparse subnetworks obtained through lottery ticket training do not simply overfit to particular domains, but rather reflect an inductive bias of deep neural networks that can be exploited in multiple domains.

## 1 Introduction
［#3］
Recent research has suggested deep neural networks are dramatically over-parameterized. In natural language processing alone, most state-of-the-art neural networks have computational and memory complexities that scale with the size of the vocabulary. Practitioners have developed numerous methods to reduce the complexity of these models—either before, during, or after training—while retaining existing performance. Some of these methods include quantization (Gong et al., 2014; Hubara et al., 2017), and different flavors of pruning (Zhu and Gupta, 2017; Liu et al., 2018b; Frankle and Carbin, 2019; Gale et al., 2019).

［#4］
In particular, the Lottery Ticket Hypothesis (Frankle and Carbin, 2019) proposes that small, sparse subnetworks are embedded within large, over-parameterized neural networks. When trained in isolation, these subnetworks can achieve commensurate performance using the same initialization as the original model. The lottery ticket training procedure is formalized as an iterative three-stage approach: (1) train an over-parametrized model with initial parameters $\theta_0$; (2) prune the trained model by applying a mask $m \in \{0,1\}^{|\theta|}$ identified by a sparsification algorithm; (3) reinitialize the sparse subnetwork by resetting its non-zero weights to the initial values $(m \odot \theta_0)$ and retrain it. These three stages are repeated for multiple rounds. If the final subnetwork achieves similar (or better) test performance in comparison to the original network, a winning lottery ticket has been identified.

［#5］
Evidence of the existence of winning tickets has been empirically shown on a range of tasks, including computer vision, reinforcement learning, and natural language processing (Frankle and Carbin, 2019; Yu et al., 2019). However, the merits of lottery ticket training has recently been called into question. In particular, (1) whether keeping the same initialization (e.g., $\theta_0$) is crucial for acquiring tickets (Liu et al., 2018b); and (2) if tickets can generalize across multiple datasets (Morcos et al., 2019).

［#6］
Our paper investigates the efficacy of lottery tickets when the data distribution changes. We define multiple data domains such that their input distributions are varied. Then, we consider whether subnetworks obtained in a source domain $\mathcal{D}_s$ can be used to specify and train subnetworks in a target domain $\mathcal{D}_t$ where $s \neq t$. Inspired by Liu et al. (2018b), we also experiment with different initialization methods at transfer-time, probing at the importance of initial (source domain) values in disparate target domains. We find that subnetworks obtained through lottery ticket training do not completely overfit to particular input dis-

---
［#7］
*Work done during an internship at Facebook.

［#8］
tributions, showing some generalization potential when distributional shifts occur. In addition, we discover a phase transition point, at which sub-networks reset to their initial values show better and more stable generalization performance when transferred to an arbitrary target domain.

［#9］
In summary, our contributions are (1) continuing the line of work on the Lottery Ticket Hypothesis (Frankle and Carbin, 2019), showing that tickets exist in noisy textual domains; (2) performing comprehensive experiments pointing towards the transferability of lottery tickets under distributional shifts in natural language processing; and (3) publicly releasing our code and datasets to promote further discussion on these topics¹.

## 2 Related Work

［#10］
There is a large body of work on transfer learning for neural networks (Deng et al., 2013; Yosinski et al., 2014; Liu et al., 2017; Zoph et al., 2018; Kornblith et al., 2019). Most of these works focus on improving the transferred representation across tasks and datasets. The representation from a source dataset is fine-tuned or learned collaboratively on a target dataset. In contrast, we focus on understanding whether the architecture can be transferred and retrained, and whether transferring the initialization is required. Our work is also related to Neural Architecture Search (NAS) (Zoph et al., 2018; Liu et al., 2018a; Elsken et al., 2018). The goal of NAS is to identify well-performing neural networks automatically. Network pruning can be viewed as a form of NAS, where the search space is the sparse topologies within the original over-parameterized network (Liu et al., 2018b; Gale et al., 2019; Frankle and Carbin, 2019).

［#11］
Iterative magnitude pruning (Frankle and Carbin, 2019; Frankle et al., 2019) is a recently proposed method for finding small, sparse subnetworks from large, over-parameterized neural networks that can be trained in isolation to reach a similar (or better) test accuracy. To obtain these re-trainable sparse subnetworks, Frankle and Carbin (2019) uses an iterative pipeline that involves training a model, removing "redundant" network connections identified by a sparsification algorithm, re-training the subnetwork with the remaining connections. In particular, the experiments in Frankle and Carbin (2019) show it is critical to re-initialize the subnetworks using the same initial values after each round of the iterative pipeline.

［#12］
However, the importance of re-using the original initialization is questioned in Liu et al. (2018b), where the authors show that competitive performance of the sparse subnetworks can be achieved with random initialization as well. Morcos et al. (2019) investigate the transferability of lottery tickets across multiple optimizers and datasets for supervised image classification, showing that tickets can indeed generalize (Morcos et al., 2019). Beyond the differences between our domain, task, and datasets, our work carries an important distinction. In Morcos et al. (2019), the authors refer to the transfer of initialization as both the transfer of the sparse topologies and the transfer of the initial values of the subnetworks. Therefore, it is unclear whether the sparse topology alone can be transferred across datasets or the topology combined with the initial values must be exploited jointly to achieve transferability. In our work, we decouple this question by investigating the influence of different initialization strategies on the sparse architecture during the process of finding the winning tickets and after the transfer to other domains.

## 3 Task and Datasets

［#13］
Distributional Shifts Let $(x_i^s, y_i^s) \in \mathcal{X} \times \mathcal{Y}$ denote a pair of training samples from domain $\mathcal{D}_s$. Let $f(x; \theta)$ be a function (e.g., deep neural network) that maps an input from $\mathcal{X}$ to the label space $\mathcal{Y}$, parameterized by $\theta$. In this work, the sparsity of $\theta$ is induced by the lottery ticket training process (Frankle and Carbin, 2019). To model distributional shifts, we characterize each domain $\mathcal{D}_i$ as a dataset from the Amazon Reviews corpus (McAuley and Leskovec, 2013). The differences in unigram frequencies, semantic content, and random noise mimic the type of distributional shifts that occur in machine learning.

［#14］
Subword Vocabulary We ensure each domain $\mathcal{D}$ shares an identical support on $\mathcal{X}$ by encoding the inputs using a vocabulary common across all datasets. Word-level vocabularies may introduce problems during domain transfer as certain words potentially only appear within a particular domain. On the other end of the spectrum, character-level vocabularies ameliorate this issue but may not contain enough expressive power to model the data. We elect to use a subword vo-

［#15］
¹https://github.com/facebookresearch/pytext

［#16］
![](./images/867746711882170612_1.jpg)

［#17］
Figure 1: Jenson-Shannon Divergence scores on sub-word unigram distributions for each domain pair $(\mathcal{D}_i, \mathcal{D}_{i'})$. Domains include Books (B), Electronics (E), Movies (M), CDs (C), and Home (H). Values are scaled by $1e^5$ for presentation.

［#18］
cabulary, balancing the out-of-vocabulary and effectiveness problems introduced by the word- and character-level vocabularies, respectively. Technical details for creating the shared subword vocabulary are presented in $\S$4.1.

［#19］
Divergence Scores Given an identical support for all data distributions, we now quantify the distributional shifts between our domains using Jenson-Shannon Divergence (JSD). JSD is a symmetric measure of similarity between two (continuous) probability distributions $p$ and $q$ with a proxy, averaged distribution $m = \frac{1}{2}(p+q)$:

［#19］
$$
\operatorname{JSD}(p \| q)=\frac{1}{2} \mathrm{KL}(p \| m)+\frac{1}{2} \mathrm{KL}(q \| m) \quad (1)
$$

［#19］
where $\operatorname{KL}(p \| q)$ in Eq. 1 denotes the Kullback-Leibler divergence, defined as:

［#19］
$$
\mathrm{KL}(p \| q)=\int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} d x \quad (2)
$$

［#20］
Figure 1 displays the divergence scores between our datasets. On average, there is high disagreement with respect to the prevalence and usage of subwords in each domain, with Electronics$\rightarrow$Home the most similar and CDs$\rightarrow$Home the most dissimilar.

［#21］
Sentiment Analysis Finally, we introduce our base task for experimentation. Our models are evaluated on a binary sentiment analysis task constructed from five categories in the Amazon Reviews corpus: books (B), electronics (E), movies (M), CDs (C), and home (H). The dataset originally provides fine-grained sentiment labels (1 through 5) so we group 1, 2 as negative and 4, 5 as positive. Following Peng et al. (2018), reviews with neutral ratings (3) are discarded. We sample 20K train, 10K validation, and 10K test samples from each category, ensuring there is an equal distribution of positive and negative reviews.

## 4 Methods

［#22］
In this section, we discuss our technical methods. First, we describe the subword vocabulary creation process ($\S$4.1). Second, we cover the underlying model used in the sentiment analysis task ($\S$4.2). Third, we detail the lottery ticket training and transferring methods ($\S$4.3).

### 4.1 Vocabulary

［#23］
We use the SentencePiece$^2$ library to create a joint subword vocabulary for our datasets (Kudo and Richardson, 2018). The subword model is trained on the concatenation of all five training datasets (100K sentences) using the byte-pair encoding algorithm (Sennrich et al., 2016). We set the vocabulary size to 8K. The final character coverage is 0.9995, ensuring minimal out-of-vocabulary problems during domain transfer.

### 4.2 Model

［#24］
We use convolutional networks (CNN) as the underlying model given their strong performance on numerous text classification tasks (Kim, 2014; Mou et al., 2016; Gehring et al., 2017). Let $V$ and $n$ represent the vocabulary of the corpus and maximum sequence length, respectively. Sentences are encoded as an integer sequence $t_1,\cdots,t_n$ where $t_i \in V$. The embedding layer replaces each token $t_i$ with a vector $\mathbf{t}_i \in \mathbb{R}^d$ that serves as the corresponding $d$-dimensional embedding. The vectors $\mathbf{t}_1, \cdots, \mathbf{t}_n$ are concatenated row-wise to form a token embedding matrix $\mathbf{T} \in \mathbb{R}^{n \times d}$.

［#25］
Our model ingests the embedding matrix $\mathbf{T}$, then performs a series of convolutions to extract salient features from the input. We define a convolutional filter $\mathbf{W} \in \mathbb{R}^{h \times d}$ where $h$ represents the *height* of the filter. The filter is not strided, padded, or dilated, Let $\mathbf{T}[i: j] \in \mathbb{R}^{h \times d}$ represent a sub-matrix of $\mathbf{T}$ extracted from rows $i$ through $j$, inclusive. The feature map $\mathbf{c} \in \mathbb{R}^{n-h+1}$ is induced by applying the filter to each possible window of $h$ words, i.e.,

［#25］
$$
c_i=f\left(\langle\mathbf{T}[i: i+h], \mathbf{W}\rangle_{\text {fro }}+b\right) \quad (3)
$$

［#26］
\footnotetext[2]{https://github.com/google/sentencepiece}

［#26］
for $1 \leq i \leq n-h+1$, where $b \in \mathbb{R}$ is a bias term, $f$ is a non-linear function, and the Frobenius inner product is denoted by $\langle \mathbf{A}, \mathbf{B} \rangle_{\text{fro}} = \sum_{i=1}^h \sum_{j=1}^d \mathbf{A}_{ij}\mathbf{B}_{ij}$. 1-max pooling (Collobert et al., 2011) is applied on $\mathbf{c}$, defined as $\hat{c} = \max\{\mathbf{c}\}$. This is performed to propagate the maximum signal throughout the network and reduce the dimensionality of the input.

［#27］
The process described above creates one feature from one convolution with window $h$ followed by a pooling operation. To extract multiple features, the model uses several convolutions with varying $h$ to obtain features from different sized $n$-grams in the sequence. The convolutional (and pooled) outputs are concatenated along the channel dimension, then fed into a one-layer MLP to obtain a distribution over the $c$ classes.

### 4.3 Lottery Tickets
#### 4.3.1 Initialization
［#28］
The embedding matrix is initialized from a unit Gaussian, $\mathbf{T} \sim \mathcal{N}(0,1)$. The convolutional and MLP layers use He initialization (He et al., 2015), whose bound is defined as

［#28］
$$
b = \sqrt{\frac{6}{(1+a^2) \times \text{fan\_in}}} \tag{4}
$$

［#28］
where $a$ and $\text{fan\_in}$ are parameters calculated for each weight. The resulting weights have values uniformly sampled from $\mathcal{U}(-b, b)$.

#### 4.3.2 Training
［#29］
We use iterative pruning with alternating cycles of training and pruning to obtain the tickets (Han et al., 2015; Frankle and Carbin, 2019). For clarity, we define a round as training a network for a fixed number of epochs. We begin with a seed round $r_0$ where the model does not undergo any pruning, then begin to procure tickets in a series of lottery ticket training rounds.

［#30］
In each successive round $r_{i>0}$, a fraction $p$ of the weights that survived round $r_{i-1}$ are pruned (according to a sparsification algorithm, discussed below) to obtain a smaller, sparser subnetwork; this is denoted by $f(x; m_i \odot \theta_i)$ where $m_i$ and $\theta_i$ represent the sparse mask and weights at round $r_i$. The weights $\theta_i$ of this subnetwork are set according to an initialization strategy and the subnetwork is re-trained to convergence. We refer to the sparsity as the fraction of weights in the network that are exactly zero. In each round, we prune $p\%$ of the weights in the model. Therefore, the resulting ticket has sparsity $1-(1-p\%)^{r_{total}}$, where $r_{total}$ is the total number of lottery ticket training rounds.

［#31］
![](./images/867746711882170612_2.jpg)
［#32］
Figure 2: Visualization of the subnetwork transfer process. Purple denotes elements from the source domain, while blue denotes elements from the target domain. Tickets are composed of two elements: (1) the sparsified mask $(m_i)$ and (2) the initial parameter values $(\theta_i)$. During transfer, we create subnetworks in the source domain with the mask borrowed from the source domain, but with potentially different parameters. We use $\theta_i'$ to denote that these parameters are set according to some initialization strategy, which we discuss further in our experiments ($\S5$).

［#33］
Next, we discuss the sparsification algorithm used to prune weights in each round $r_i$. Let $\mathbf{p}_i$ denote the vectorized collection of trainable parameters in layer $i \geq 0$, with the embedding layer as layer 0. After re-training the (sub-)networks in each round, we apply the $\ell_0$ projection on the parameters in each layer, i.e.

［#34］
$$
\underset{\mathbf{p}}{\text{argmin}} \|\mathbf{p} - \mathbf{p}_i\|_2^2 \tag{5}
$$

［#34］
subject to $\text{card}(\mathbf{p}) \leq k_i$, where $\text{card}(\mathbf{p})$ denotes the number of non-zeros in $\mathbf{p}$. The optimization problem in Eq. 5 can be solved analytically by sorting the elements of $\mathbf{p}_i$ with respect to their absolute values and picking the top $k_i$ elements with the largest magnitude (Jain et al., 2017; Zhu and Gupta, 2017). We use the sparsity hyperparameter $p$ introduced above to decide $k_i$ for each layer. Let $\text{len}(\mathbf{p}_i)$ denote the total number of trainable parameters in layer $i$. We set $k_i = p\% \times \text{len}(\mathbf{p}_i)$ for each layer. In accordance with our training procedure, once a weight is pruned, it is no longer a trainable parameter; hence, $\text{len}(\mathbf{p}_i)$ is strictly decreasing after each round.

#### 4.3.3 Transferring
［#35］
The lottery ticket training procedure outlined in $\S$4.3.2 yields a batch of subnetworks $f(x^s; m_1 \odot$

［#36］
![](./images/867746711882170612_3.jpg)

［#37］
Figure 3: Results obtaining lottery tickets on the Books, Movies, Electronics, CDs, and Home categories of the Amazon Reviews dataset (McAuley and Leskovec, 2013). Experiments are repeated five times, where the solid lines represent the mean and shaded regions represent the standard deviation. Note that the $x$-axis ticks are not uniformly spaced.

［#38］
$\theta),\cdots,f(x^s;m_n \odot \theta)$ where $x^s$ represents the inputs from a source domain $\mathcal{D}_s$ and $m_i$ represents the sparse mask used to prune weights at round $r_i$. During transfer, we construct a new batch of subnetworks $f(x^t;m_1 \odot \theta'),\cdots,f(x^t;m_n \odot \theta')$ to be evaluated on inputs from a (non-identical) target domain $\mathcal{D}_t$ with masks derived from the source domain. The change in parameter notation $(\theta \rightarrow \theta')$ implies that the subnetworks evaluated in a disparate domain can potentially use a different transfer initialization strategy. We clarify this process in Figure 2. In contrast, Morcos et al. (2019) transfers the entire ticket (sparse masks and initial values) to the target domain. Finally, using the new batch of subnetworks, we evaluate each subnetwork $f(x^t;m_i \odot \theta')$ in the target domain for $r_{total}$ rounds. Unlike the canonical ticket training rounds, we do not (additionally) sparsify the subnetworks during transfer. All in all, our transfer task is designed to answer the following question: can the sparse masks found in a source domain using lottery ticket training (§4.3) be transferred to a target domain with different initialization strategies to match the performance of a ticket obtained in same target domain?

## 5 Experiments
### 5.1 Settings
［#39］
Our CNN uses three filters $(h \in [3,4,5])$, each with 127 channels, and ReLU activation (Nair and Hinton, 2010). We fix the maximum sequence length to 500 subwords. The embeddings are 417-dimensional and trained alongside the model. We opt not to use pre-trained embeddings to ensure the generalizability of our results. Additionally, we regularize the embeddings with dropout (Srivastava et al., 2014), $p=0.285$. The MLP contains one hidden layer with a dimension of 117. Hyperparameters were discovered using Bayesian hyperparameter optimization (Snoek et al., 2012) on the Books validation set. The models are trained with a batch size of 32 for a maximum of 15 epochs. Early stopping is used to save iterative model versions that perform well on a development set. We use the Adam optimizer (Kingma and Ba, 2014) with a learning rate of $1e^{-3}$ and $\ell_2$ regularization with a weight of $1e^{-5}$.

### 5.2 Obtaining Tickets
［#40］
First, we use the lottery ticket training procedure outlined in §4.3.2 to obtain tickets for our five datasets with $p=35\%$ and $r_{total}=20$. We compare the test performance of the subnetworks using the following baselines:

［#41］
![](./images/867746711882170612_4.jpg)

［#42］
Figure 4: Results transferring lottery tickets on nine transfer tasks constructed from the five categories of the Amazon Reviews dataset (McAuley and Leskovec, 2013). Experiments are repeated five times, where the solid lines represent the mean and shaded regions represent the standard deviation. Note that the $x$-axis ticks are not uniformly spaced.

［#43］
- **FULL-MODEL**: This baseline evaluates the performance of the original network without any pruning. In other words, we train a model for a seed round $r_0$, then record its performance.
- **TICKET-RESET**: The values of the subnetwork are reset to their original values before training. This initialization strategy was used in the earliest formation of the Lottery Ticket Hypothesis (Frankle and Carbin, 2019).
- **TICKET-RANDOM**: The values of the subnetwork are reset to random values drawn from the initialization distribution(s) of the original network. We sample weights from the distributions outlined in §4.3.1 to initialize the subnetworks.

［#44］
The results are shown in Figure 3. For all datasets, TICKET-RESET shows the best performance, notably outperforming FULL-MODEL in early stages of sparsification (0-90%) for the Books, Electronics, and Home datasets. This demonstrates that deep neural networks—especially those for sentiment analysis—are highly over-parameterized, and the sparsity induced by lottery ticket training can help to increase performance. This observation is consistent with Louizos et al. (2018), which also showed sparse networks fashion a regularization effect that results in better generalization performance. In addition, we observe that TICKET-RESET and TICKET-RANDOM have similar test performance until about 96% sparsity. This casts some doubt around whether the initial values truly matter for sparse models as the randomly sampled values seem to fit sparse masks well.

［#45］
However, a phase transition occurs in the high sparsity regime, where the differences between TICKET-RESET and TICKET-RANDOM are significantly enlarged. The performance of TICKET-

［#46］
RANDOM becomes highly unstable and drops off much faster than TICKET-RESET after 96% sparsity. In contrast, TICKET-RESET remains relatively stable—even with sparsity levels over 99.9%—pointing towards the enigmatic importance of original values in extreme levels of sparsity.

### 5.3 Transferring Tickets
［#47］
Next, we use the lottery ticket transferring procedure outlined in §4.3 to transfer (obtained) sub-networks from a source domain to a non-identical target domain. Identical to the previous experiment, we use $r_{total}=20$. We compare the test performance of the transferred subnetworks using the following baselines:

［#48］
- **TICKET-TARGET**: This baseline is comprised of the subnetworks obtained in the target domain using lottery ticket training. We borrow the values for this baseline (without modification) from the TICKET-RESET subnetworks shown in Figure 3, albeit from the domain of interest.
- **MASKS-RESET**: Under this initialization strategy, the masks obtained in the source domain is used on the target domain and the subnetwork is trained from the same initial values as in the source domain.
- **MASKS-RANDOM**: Under this initialization strategy, only the masks are used from the subnetwork obtained in the source domain. The parameters are randomly initialized from the distributions outlined in §4.3.1 before training on the target domain.

［#49］
The results are shown in Figure 4. Both MASKS-RESET and MASKS-RANDOM show signs of generalization in the early stages of sparsification. Most notably, subnetworks obtained in the CDs domain are extremely robust; both the MASKS-RESET and MASKS-RANDOM results show stronger performance than TICKET-TARGET, even in sparsity levels over 99%. This is relatively surprising as the FULL-MODEL in §5.2 achieved the worst performance in the CDs domain. Further inspection of representations learned in this domain will be required to understand its strong ticket performance, which may or may not be a coincidence.

［#50］
We see a 3-5% dropoff in performance (up to 90% sparsity) from tickets identified from the Books and Electronics tasks after transferring. These results together imply that tickets are not completely immune to distributional shifts, although the degradation in test accuracy is not substantial until reaching high sparsity. Nevertheless, we notice the accuracies of MASKS-RESET and MASKS-RANDOM stay relatively stable from 0-90% sparsity; they only begin to steadily decline after this point.

［#51］
Finally, we compare the performance of MASKS-RESET and MASKS-RANDOM. In the Books tasks, MASKS-RANDOM performs better overall in comparison to MASKS-RESET. Its performance is slightly worse in the Electronics and CDs tasks, although it is relatively comparable to MASKS-RESET up to 96%. Similar to the results in §5.2, we notice a phase transition point where the initial values (e.g., MASKS-RESET) play a much bigger role in maintaining stability and performance in the deeper stages of sparsification.

## 6 Discussion
［#52］
In this section, we briefly recap our findings, highlighting key points observed through our ticket procuring and transfer experiments. For each section, we also touch on areas for future work.

［#53］
Evidence of transferability of winning tickets in natural language processing. Our experiments show that "winning tickets" can indeed be identified in a sentiment task formulated from noisy, user-generated datasets. Moreover, the "winning tickets", up to extreme level of sparsity (e.g., 90%), can be transferred across domains without much loss in accuracy. The fact that tickets can be obtained in noisy environments shows its promise across multiple data sources. However, our work only considers a binary sentiment analysis task. Future work can explore other tasks such as multi-class text classification, language modeling, and machine translation.

［#54］
Randomly initialized tickets are strong baselines. Consistent with the observations in Liu et al. (2018b), initializing tickets to their original values before training is not necessarily required for strong performance. In our experiments, we show that in high sparsity conditions (up to 90%), there is no noticeable difference between the performance of the originally and randomly initialized subnetworks. Although the sparse masks build on top of each other from round $r_i$ to $r_{i+1}$,

［#54］
randomly initialized subnetworks are still able to settle in a local minima with comparable performance to that of the originally initialized subnetworks. However, our work fixes the optimizer and learning rate across experiments. It may be possible that randomly initialized subnetworks using varying optimization reach better minima.

［#55］
A phase transition point largely influences ticket performance. As alluded to above, there is almost no difference in performance when considering originally and randomly initialized subnetworks. However, our experiments point towards a crucial turning point—the phase transition—in which the initialization begins to matter. In particular, especially in extreme levels of sparsity (e.g., 99.99%) originally initialized networks exhibit less variance than randomly initialized tickets in test accuracy. However, the specific sparsity at which the phase transition happens is dataset-dependent. Understanding why this occurs and its relation with other models, datasets, and optimization algorithms can further unveil and explain the phenomena behind lottery tickets.

## 7 Applications in Federated Learning

［#56］
Federated learning is a scenario where a centralized model is trained over decentralized data, distributed across millions (if not billions) of clients (e.g., electronic devices) (Konen et al., 2016; Bonawitz et al., 2019). Crucially, the clients are not allowed to exchange data with the central server or each other. Instead, each client can fine-tune a model for a couple of iterations on their own data, then send their (encrypted) parameters or gradients to a server for aggregation. This “collaborative learning” setup effectively maintains a level of user privacy by ensuring the data always stays on-device. However, this poses several challenges for optimization; as the centralized server does not have access to the data distribution of each client, any neural architecture selection has to be done on either (a) a different data source the server has access to or (b) on each individual client. Since (b) is generally quite expensive, the server usually maintains some seed data, as alluded to in (a).

［#57］
With the transferability of lottery tickets, the server can procure lottery tickets on server-accessible data, then retrain the tickets on client data under the federated learning framework. While there may be a large performance drop when transferring extremely sparse networks, our results show that clients can still re-train moderately sparse networks with commensurate performance. We believe that this “sparsify and transfer” procedure has two immediate benefits: (1) past work—including the original incarnation of the lottery ticket hypothesis—has shown that sparse networks can be, under certain conditions, easier to optimize (Frankle and Carbin, 2019; Morcos et al., 2019; Gale et al., 2019); and (2) sparser subnetworks have significantly less capacity than their large, over-parameterized counterparts, which can alleviate client-server communication costs (e.g., model uploading and downloading) (Konen et al., 2016; Sattler et al., 2019).

## 8 Conclusion

［#58］
The Lottery Ticket Hypothesis (Frankle and Carbin, 2019) posits that large, over-parameterized networks contain small, sparse subnetworks that can be re-trained in isolation with commensurate test performance. In this paper, we examine whether these tickets are robust against distributional shifts. In particular, we set up domain transfer tasks with the Amazon Reviews dataset (McAuley and Leskovec, 2013) to obtain tickets in a source domain and transfer them in a disparate target domain. Moreover, we experiment with the transfer initialization of the networks, determining if resetting to initial values (obtained in the source domain) are required for strong performance in the target domain. Our experiments show that tickets (under several initialization strategies) can be transferred across different text domains without much loss up to a very high level of sparsity.

［#59］
In addition, there is a lot of debate on whether initial value resetting is critical to achieve commensurate test performance. While Frankle and Carbin (2019); Frankle et al. (2019) present evidence supporting the importance of resetting, Gale et al. (2019); Liu et al. (2018b) show that sparse retrainable subnetworks can be found independent of resetting. Our experiments show that this is not a yes or no question. Specifically, we show there is a phase transition related to sparsity. Resetting is not critical before extreme levels of sparsity (i.e., below 99%), but the effect of resetting is magnified in high sparsity regimes. Finally, we demonstrate the practical applications of our results in federated learning.

### Acknowledgments

［#60］
Thanks to Veselin Stoyanov and our anonymous reviewers for their helpful comments.

### References


































