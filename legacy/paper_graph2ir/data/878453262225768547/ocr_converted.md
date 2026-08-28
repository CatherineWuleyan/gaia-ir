# Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

［#1］
Ajay Jaiswal $^1$ Shiwei Liu $^1$ Tianlong Chen $^1$ Ying Ding $^1$ Zhangyang Wang $^1$

［#2］
![](./images/878453262225768547_1.jpg)

［#3］
Figure 1. Fine-tuning Accuracy ($\uparrow$) and FLOPs counts ($\downarrow$) of subnetworks with sparsity $s \in \{10\%,20\%,...,80\%\}$ on CIFAR-10 test set with a pre-trained CLIP (ViT-B32) model checkpoint. Note that ISP requires approximately single-pass computational cost of IMP to generate subnetworks with better performance than multiple rounds of IMP as required in Lottery Tickets.

## Abstract
［#4］
Large pre-trained transformers have been receiving explosive attention in the past few years, due to their wide adaptability for numerous downstream applications via fine-tuning, but their exponentially increasing parameter counts are becoming a primary hurdle to even just fine-tune them without industry-standard hardware. Recently, Lottery Ticket Hypothesis (LTH) and its variants, have been exploited to prune these large pre-trained models generating subnetworks that can achieve similar performance as their dense counterparts, but LTH pragmatism is enormously inhibited by repetitive full training and pruning routine of iterative magnitude pruning (IMP) which worsens with increasing model size. Motivated by the recent observations of model soups, which suggest that fine-tuned weights of multiple models can be merged to a better minima, we propose **Instant Soup Pruning (ISP)** to generate lottery ticket quality subnetworks, using a fraction of the original IMP cost by replacing the expensive intermediate pruning stages of IMP with computationally efficient weak mask generation and aggregation routine. More specifically, during the mask generation stage, ISP takes a small handful of iterations using varying training protocols and data subsets to generate many weak and noisy subnetworks, and superpose them to average out the noise creating a high-quality denoised subnetwork. Our extensive experiments and ablation on two popular large-scale pre-trained models: CLIP (unexplored in pruning till date) and BERT across multiple benchmark vision {MNIST, SVHN, Cars, GTSRB, CIFAR-10, CIFAR-100} and language datasets {MNLI, QNLI, QQP, SST, ...} validate the effectiveness of ISP compared to several state-of-the-art pruning methods. Additionally, we show that ISP can be easily modified with minimal overhead to produce benefits comparable to model soups, without the prerequisite to generate multiple candidates fine-tuned models. Codes are available at: https://github.com/VITA-Group/instant_soup.

---

［#5］
$^*$Equal contribution $^1$University of Texas at Austin. Correspondence to: Ajay Jaiswal <ajayjaiswal@utexas.edu>.

［#6］
Proceedings of the $40^{th}$ International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s).

---

## 1. Introduction
［#7］
Large-scale transfer learning has recently become show-stealer in modern deep learning, and transformer-based pre-trained models (Devlin et al., 2018; Liu et al., 2019; Dosovitskiy et al., 2020; Liu et al., 2021; Radford et al., 2021) are now achieving state-of-the-art performance for a wide array of real-world computer vision (Dosovitskiy et al., 2020; Han et al., 2020; Li et al., 2023; Touvron et al., 2021; Mao et al., 2022; Jaiswal et al., 2021a; Zheng et al., 2021; Parmar et al., 2018) and natural language processing (Yang et al., 2019b; Liu et al., 2019; Talmor et al., 2018; Jaiswal et al., 2021b; Zheng et al., 2023; Yang et al., 2019a; Wang et al., 2018; Ding et al., 2019; Chowdhery et al., 2022; Wei et al., 2022; Jaiswal et al., 2023) applications. With the astonishing explosion of parameter counts (millions to billions) in the past few years, while chasing performance gains, fine-tuning these large pre-trained models with non-industry


［#8］
Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

［#8］
standard hardware is becoming seemingly impossible, in addition to expensive inference and steep environmental cost. In the hustle of building gigantic models, a parallel and growing field of model compression has been exploring the prospects to compress these enormous models at the cost of marginal/no sacrifice in performance, effectively reducing their computational and memory footprints.

［#9］
Among many efforts for compressing models and accelerating inference (Frankle & Carbin, 2018; Chen et al., 2020a; Jaiswal et al., 2022b; Yin et al., 2022; Jaiswal et al., 2022c; Lee et al., 2018; Yu et al., 2017; 2020; Fang et al., 2023; Chen et al., 2023; Jaiswal et al., 2022a; Liu et al., 2023), network pruning eliminates unnecessary weights to generate smaller subnetworks in place of dense networks attaining similar performance, stands out as one of the most effective techniques. Lottery Ticket Hypothesis (LTH) (Frankle & Carbin, 2018) and its variants, reveal that dense, randomly-initialized networks contain small subnetworks which can match the test accuracy of original networks. Despite their insightful findings, there still exists a large gap in the practicality of these methods because of the fully dense training routine of IMP, which exacerbates with an increase in model capacity. Recently proposed EarlyBird routine (You et al., 2019; Chen et al., 2020b) which attempts to draw the winning tickets early in training, and pruning at initialization techniques have shown some promise in mitigating search through an expensive and tedious iterative process, yet their effectiveness for the large-scale pre-trained network is highly under-explored or they tend to have sub-standard performance at non-trivial sparsities. In this work, we ask: *Does there exist a principled and cheaper approach for fastly drawing high-quality lottery tickets in large pre-trained models within a limited computational budget, while preserving its performance and transferability?*

［#10］
To this end, we explore the feasibility of obtaining cheap tickets for popular large pre-trained models CLIP(Radford et al., 2021) and BERT(Devlin et al., 2018) constrained by no multi-pass repetitive full-training, to meet the permissible computational budget. One straightforward approach is to curtail per round training cost of iterative magnitude pruning (IMP) with early stopping, but we found that such non-careful pruning approach often produces highly variable and substandard tickets, presumably due noisy and unstable state of the network, when pruned. Our close analysis of the pruned mask generated by LTH and cheap pruning methods (one-shot pruning) found that large-scale pre-trained transformers are highly overparameterized and pruning them at trivial sparsities does not necessarily require an expensive LTH paradigm, which encourages us to start pruning nonchalantly at trivial sparsities and become scrupulous at non-trivial sparsities. In addition, recently several works (Wortsman et al., 2022a; Ilharco et al., 2022; Juneja et al., 2022) investigate the intriguing phenomenon of "model soups", and have shown that weights of multiple dense fine-tuned models can be merged together into better solutions lying in low error basins. In the sparse setting, a very recent attempt (Yin et al., 2022) reused the byproduct of IMP, and showed that tickets generated at each iteration of IMP could be superposed into a stronger subnetwork. However, its observations are limited to small-scale networks, and more importantly, the algorithm does not contribute to the computational efficiency of either finding lottery tickets or (re-)training networks. Motivated by these observations, we are interested to investigate: *if we can leverage the soup observations to eliminate noise induced by early pruning in IMP iteration, effectively leading to the stable sparse subnetwork and reduced cost.*

［#11］
We propose **Instant Soup Pruning (ISP)**, a model soup-inspired perspective dedicated to generating lottery ticket quality subnetworks, using a fraction of the original IMP cost. More specifically, ISP uses a miniature random subset of training data to generate many weak and noisy subnetworks, and superpose them to average out the noise creating a high-quality denoised subnetwork. Similar to traditional IMP, ISP repeats the denoising routine following well-managed training iterations till the desired sparsity is reached, eliminating the need of IMP to perform a full pass of training before every pruning routine. Our experiments on CLIP (unexplored in pruning literature till date) and BERT across multiple datasets illustrate that ISP can find sparse subnetworks with better quality than LTH, with an affordable cost no more than a single pass of IMP. In addition to dense-to-sparse paradigm, interestingly, our ISP routine can be bluntly incorporated in dense-to-dense paradigm to incorporate the benefits of model soups in dense pre-trained models at almost negligible training cost at the pre-training stage, ultimately leading to better-fine tuning performance.
Our contributions can be summarized as:

［#12］
■ We propose **Instant Soup Pruning (ISP)**, a novel pruning strategy that seamlessly integrates the "model soup" idea to significantly reduce the computational cost of IMP. ISP replaces the expensive intermediate pruning stages of IMP with computationally-cheap weak mask generation and denoising, while outperforming IMP for large pre-trained models.

［#13］
■ ISP naturally provides a "self-denoising" ability to eliminate the necessity of generating high quality/expensive masks (presumably "good solution basin") at each pruning stage, by instead generating multiple computationally inexpensive weak masks and averaging them out to reduce their solution noise.

［#14］
■ In the dense-to-dense paradigm, ISP can be adapted to **Instant Model Soup**, to inject the benefits of model soups in dense pre-trained models at marginal training cost, thereby improving fine-tuning performance com-


［#15］
![](./images/878453262225768547_2.jpg)

［#16］
Figure 2. Overview of our proposed Instant Soup Pruning. We provide a detailed illustration of our proposed technique compared to conventional LTH with IMP. ISP replaces the expensive intermediate pruning stages of IMP with computationally-cheap weak mask generation and denoising while outperforming LTH. Unlike LTH, ISP consumes computation budget equivalent of a single pass of LTH.

［#17］
parable to model soups without the need to generate multiple fine-tuned models.

［#18］
■ Our extensive experiments on two popular large-scale pre-trained models (CLIP & BERT) across multiple benchmark vision {MNIST, SVHN, Cars, GTSRB, CIFAR-10, CIFAR-100} & language datasets {MNLI, QQP, STS-B, WNLI, QNLI, MPRC, RTS, SST-2, CoLA} validate the effectiveness of ISP wrt. several SOTA pruning methods.

## 2. Methodology

### 2.1. Revisiting LTH and Pre-trained Vision and Language Model Compression

［#19］
In the past few years, scaling neural networks to improve information absorption has been pivotal for good optimization and generalization performance, but this unbounded parameter growth has made them computationally expensive with excessive memory requirements. The trend undoubtedly continues with the recent forefront of transformers, where more and more layers are stacked with dense attention blocks (eg. T5 has $\sim 10+$ billion parameters) calling for expensive computational resources and prolonged training or fine-tuning time. Recently, some work (Chen et al., 2021b; Gan et al., 2022; Chen et al., 2021a; 2020b; You et al., 2019; Prasanna et al., 2020) have explored Lottery Ticket Hypothesis (LTH) to understand the parameter redundancy in the current prevailing large scale transformer models, and attempted to compress it to non-trivial sparsities by repetitive initialization-training-pruning operation. Despite their success to find high-quality compressed models on a range of downstream tasks, it is impossible to ignore the cost of finding these subnetworks, since winning tickets can only be identified by pruning unimportant connections after fully training a dense network in a conventional LTH paradigm, which worsens significantly with increasing model size. For example, (Prasanna et al., 2020; Chen et al., 2020a) explored pruning BERT to matching subnetworks at 40% to 90% sparsity across multiple tasks, on the other hand, (Gan et al., 2022) explored LTH for compressing large pre-trained VL models while preserving its performance.

［#20］
![](./images/878453262225768547_3.jpg)

［#21］
Figure 3. Performance comparison of LTH and One-shot magnitude-based pruning of CLIP (ViT-B32) on CIFAR-10 at sparsity $S \in \{10\%, 20\%, ..., 90\%\}$ (left). Cosine similarity between the binary prune masks obtained by LTH and One-shot magnitude pruning of CLIP (ViT-B32) on CIFAR-10 (right).

［#22］
Recently, (Chen et al., 2020b) explored the EarlyBird (You et al., 2019) idea and proposed jointly training BERT and some sparsity-inducing coefficients which can be used to draw the subnetworks followed by fine-tuning, but its joint training step again is as expensive as normal BERT training, and experimentally we found that its performance becomes sub-standard compared to LTH in non-trivial sparsity range. In this work, we propose a novel pruning strategy based on strong insights of the inherent benefits of gigantic size

［#23］
Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

［#23］
and model soups, which can be equivalent to or even better than LTH and its variant LTH-Rewind. We, for the first time, explore model compression for a recent extremely popular open vocabulary network CLIP along with BERT, to illustrate our approach benefits by using merely the computational cost equal to a single pass of conventional LTH.

### 2.2. Instant Soup Pruning: A novel cost-effective pruning perspective
［#24］
In this section, we introduce a novel pruning algorithm, named **Instant Soup Pruning (ISP)** which primarily aims to reduce the computational overhead of conventional LTH while searching for lottery tickets in large-scale pre-trained transformers, facilitating benefits from both performance and computation perspective.

［#25］
ISP is motivated from the following three observations:
- *Firstly*, large-scale pre-trained transformers are highly over-parameterized, and pruning them at trivial (eg. 10%, 20%, etc. depending on task and model size) sparsities does not require sophisticated pruning methods like LTH or LTH-Rewind to get high-quality sparse subnetworks. We surprisingly observed that at trivial sparsities, the sparse mask generated by LTH and cheap one-shot magnitude pruning is significantly similar which thereby reflects in the test performance of the subnetworks. For example, Figure 3(a) illustrates the performance of subnetworks obtained by LTH and one-shot pruning on CLIP (ViT-B32) and Figure 3(b) illustrates the cosine similarity between the binary prune mask identified by LTH and one-shot pruning. It can be clearly observed that at trivial sparsities such as 10%, 20%, and 30%, both unreasonably cheap one-shot pruning and expensive LTH identify approximately similar masks with 96.27%, 98.75%, and 94.32% cosine similarity score. It conveys a strong message to save the computation budget of full training passes of LTH, which are seemingly unnecessary.
- *Secondly*, Early-Bird (You et al., 2019) tickets, though limited to small architectures (ResNets, Vgg16, etc), conveyed a strong yet highly overlooked message that high-quality tickets can emerge at a very early training stage by pruning networks trained at much earlier points (before the accuracies reach their final top values). Recently, (Chen et al., 2020b) showed that this observation holds true for BERT, but it cannot recover the full performance of LTH. To this end, complementary to our first observation, our work extends the early-bird findings by proposing to look more carefully while searching tickets at non-trivial sparsity (high sparsity regime) compared to trivial sparsity by progressively increasing training steps over each call to pruning routine in the mask finding stage.
- *Lastly*, to mitigate the issue of pre-mature pruning to generate sub-standard pruning masks, we borrow inspiration from the intriguing phenomenon of "model soups", which illustrate weights of large-scale independently fine-tuned models can be merged together into a better solution. Our work proposes a novel approach of mask soups by superposing multiple multiple cheap pruning masks, to attenuate the noise within them due to the presumably unstable state of the network while pruning, giving a high-quality denoised pruning mask.

［#26］
**Algorithm Overview** Consider a dense, pre-trained network $f(x;\theta)$, as shown in Figure 2, LTH trains $f$ to achieve minimum validation loss $f_{loss}$ using $E$ epochs with a test accuracy $f_{acc}$, when optimized with Adam optimizer on a training dataset $D$. Once, $f_{acc}$ is achieved, the fine-tuned network $f(x;\theta_E)$ is pruned using magnitude pruning to generate a subnetwork $f(x;m \odot \theta_E)$, with a mask $m \in \{0, 1\}$. This process is repeated for $k$ iterations till the desired sparsity $S\%$ is achieved, generating a subnetwork $f_{LTH}(x;m \odot \theta_{\mathcal{O}(k\cdot E)})$ with accuracy $f_{acc}^{LTH}$.

［#27］
In contrast, our proposed approach ISP aims to generate $f_{ISP}(x;m' \odot \theta_{\mathcal{O}(E)})$ with accuracy $f_{acc}^{ISP}$, such that $f_{acc}^{ISP} \geq f_{acc}^{LTH}$ with sparse mask $m' \in \{0, 1\}$ and sparsity $S\%$. Given the computational budget of $E$ epochs, which translates to $T$ steps with batch size $B$, ISP is composed of two distinct phases: mask generation phase and fine-tuning phase which uses $M$ and $(T-M)$ steps respectively to produce a high-quality fine-tuned subnetwork with desired sparsity $S\%$, usually outperforming LTH.

### 2.2.1. MASK GENERATION STAGE
［#28］
We will first discuss our novel, computationally efficient, and high-quality sparse mask generation steps for ISP incorporating the aforementioned motivation. Note that similar to IMP, ISP is also an iterative train-prune-retrain procedure, but it uses a highly optimized train/re-train subroutine starting from the parameter state obtained from the previous iteration. Given the computational budget of $M$ steps, the mask generation stage of ISP is carefully designed to start in a relaxed fashion (spending few learning steps) while pruning at trivial sparsities and gradually become meticulous while approaching non-trivial sparsity regime, acknowledging the sensitivity of pruning while operating in high sparsities.

［#29］
More specifically, we introduce a hyperparameter $t$, which is defined as a small initial seed step count usually equates to the number of steps required to look $\sim 10\%$ of training data with batch size $B$, before the first call to pruning routine. At $i$-th call to the pruning routine, we use $t \times (i+1)$ training steps for calibrating (training) the network before pruning. For example, at the 0-th iteration when the network is dense, ISP uses steps equivalent to 10% of training data while at 5-th iteration, it uses 60% of training data in the re-training.


### Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

［#30］
```
Algorithm 1 Instant Soup Pruning
Input  : Pre-trained Network: $f(x;\theta_0)$; Initial Training
         Seed: $t$; Compression ratio: $s\%$; Desired sparsity:
［#30］
         $S\%$; Training Budget: $T$
Output: Pruned trained subnetwork with a mask $m' \in
［#30］
         \{0,1\}$: $f_{ISP}(x;m' \odot \theta_{\mathcal{O}(E)})$

［#31］
/* $k$ is chosen st. $\sum_{i=0}^k (i+1) \cdot t \leq M$       */
for $i \leftarrow 0$ to $k$ do
    /* Sparsity dependent training steps
       before call to pruning routine.
       Spend low training steps for trivial
       sparsity compared to non-trivial
       sparsity.                                              */
    Train $f(x;\theta_{i \cdot t}) \to f(x;\theta_{(i+1) \cdot t})$
    if $\text{sparsity}(f) < S\%$ then
［#31］
        $m_{(i+1)} \leftarrow \text{DenosiedPrune}(f,s,m_i)$
［#32］
        /* Compress network with newly
           returned denoised mask                            */
        Apply $m_{(i+1)} \to f(x;\theta_{(i+1) \cdot t})$
    end if
end for
/* Fine-tune the obtained subnetwork for
   the remaining $(T - ((k + 1) \cdot t))$ steps in
   computational budget.                                     */
Fine-tune $f(x;\theta_{(k+1) \cdot t}) \to f(x;\theta_{T-((k+1) \cdot t)})$
```

［#33］
To ensure that ISP does not produce sub-standard quality mask at each pruning iteration, we next provide details of our novel denoised pruning routine which is inspired by recently proposed "model soups" phenomenon, to average out noise induced due to pre-mature pruning.

［#34］
Denoised Pruning: Recently, several works (Wortsman et al., 2022a; Ilharco et al., 2022; Juneja et al., 2022) have validated the intriguing phenomenon of model soups, which suggest training multiple models with various hyperparameters and average the weights of models fine-tuned independently at no cost to achieve comparatively high performance. Motivated by their argument that fine-tuned models optimized independently from the same pre-trained initialization lies in the same basin of the error landscape and averaging them improves generalization, we are tempted to ask an unexplored question: Can we generate extremely cheap pruning masks using varying hyperparameters and average them out to improve the quality?

［#35］
To this end, we propose a novel Denoised Pruning Procedure, which explores the superimposition of computationally cheap pruning mask obtained by magnitude-based one-shot pruning of the network with marginal look-ahead training. Algorithm 2 illustrates the details of our denoising procedure which perform look-ahead training of the network using varying training protocol for a random subset of training samples, merely using 10 – 100 training steps,

［#36］
```
Algorithm 2 Denoised Pruning Procedure
Input  : Network to prune: $f(x;m_i \odot \theta)$ where $m_i$ de-
         notes binary mask at $i$-th iteration of ISP; Denoiser
         Count: $N$; Compression Rate: $s\%$; Look-ahead
         steps: $C$ (typically $10 \leq C \leq 100$ steps)
Output:New denoised mask: $m_{(i+1)}$

［#36］
$m_{temp} \leftarrow \text{MagnitudePruning}(f(x;\theta),s\%)$
［#36］
for $n \leftarrow 0$ to $N$ do
    /* Look-ahead training to generate
       multiple masks for denoising                          */
［#36］
    $P_n \leftarrow$ Training protocol // learning rate,
        weight decay, etc.
［#37］
    $D_n \leftarrow$ Random data samples
    Look-ahead training: $f(x;\theta) \to f(x;\theta_C)$
    /* Denosing to average mask noise                       */
［#37］
    $m_{temp} \leftarrow m_{temp} \cup
        \text{MagnitudePruning}(f(x;\theta_C),s\%)$
［#37］
end for
［#37］
$m_{(i+1)} \leftarrow \text{One-shot-adjustment}(m_{temp},f(x;\theta))$
［#37］
return $m_{(i+1)}$
```

［#37］
to generate $N$ candidate binary masks. In order to superimpose them, we found that a simple union of these $N$ binary masks is sufficient to improve the quality, facilitating ISP to achieve comparable/even better performance than expensive LTH. We hypothesize (and later experimentally validate) that our denoising procedure significantly helps eliminate any induced noise due to pre-mature pruning during ISP iterations, specifically at non-trivial sparsities.

#### 2.2.2. FINE-TUNING STAGE

［#38］
Our proposed method (ISP), is a single pass pruning algorithm that generates a high-quality fine-tuned subnetwork with desired sparsity $S\%$ using the computational budget equivalent to a single pass of LTH ($T$ steps). As mentioned before, ISP is composed of the mask generation phase ($M$ steps) followed by fine-tuning the obtained subnetwork for the remaining ($T-M$ steps). Note that, unlike conventional LTH, ISP does not restart the network training from the initial pre-trained weight; instead, as explained in Algorithm 1, it simply fine-tunes the network state (unchanged optimizer, learning rate, etc.) obtained immediately after pruning $S\%$ of parameters for $(T-M)$ steps.

### 2.3. Instant Model Soup: A sparsity-inspired extension to dense training

［#39］
In the conventional setting, to improve the generalization performance of a model, it is highly recommended to train multiple models and use their ensemble on the test set, but it comes at a heavy inference and training cost. Recently, several works (Wortsman et al., 2022a; Ilharco et al., 2022; Juneja et al., 2022) have illustrated that, unlike conventional

［#40］
Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

［#40］
ensembles, the weights of multiple fine-tuned large pre-trained models can be merged together by simply averaging their weights (aka. model soup) to beat the performance of ensembles without incurring any additional inference or memory cost. While this approach effectively reduces the additional inference overhead of ensembles, it still requires expensive fine-tuning of multiple large pre-trained models with varying hyperparameters. Inspired by our idea of Instant Soup Pruning, which illustrates that binary masks originated from cheap training with different training protocols can be superimposed/denoised to improve quality, we are enticed to explore: Can we denoise the initial pre-trained weights of dense large transformers using sparse cheap training to inject the model soup benefits early during fine-tuning, eliminating the need to generate multiple fully fine-tuned models for model soups?

［#41］
```
Algorithm 3 Instant Model Soup
Input  : Pre-trained Model:  $f(x;\theta_0)$; $D_{Train}$; $D_{Val}$;
         Denosier Count: $K$
Output :New denoised Pre-trained Model: $f(x;\theta_{new})$
［#41］
$f(x;\theta_{interpolated}) \leftarrow \text{DeepCopy}(f(x;\theta_0))$
［#41］
for $k \leftarrow 0$ to $K$ do
    /* Prune to create a sparse subnetwork
       from input model                          */
［#41］
    $f(x;m_s \odot \theta_0) \leftarrow \text{MagnitudePruning}(f(x;\theta_0), s\%)$
［#42］
    /* Weakly train sparse subnetwork for
       denoising                                 */
［#42］
    $H_k, C_k \leftarrow \text{Training Protocol, Steps}$
［#42］
    $D_k \leftarrow \text{sample}(D_{Train})$
［#42］
    $f_{weak}(x;m_s \odot \theta_k) \leftarrow \text{Train}(f(x;m_s \odot \theta_0), H_k, C_k, D_k)$
［#43］
    /* Denoising using linear interpolation
       with val data                             */
［#43］
    $f(x;\theta_{interpolated}) \leftarrow \text{Interpolate}(f_{weak}(x;m_s \odot \theta_k), f(x;\theta_{interpolated}), D_{Val})$
［#43］
end for
［#43］
$f(x;\theta_{new}) \leftarrow f(x;\theta_{interpolated})$
［#43］
return $f(x;\theta_{new})$
```

［#44］
Our primary objective is to ripe the benefits of model soups without the need to generate multiple fully fine-tuned models, thereby contenting the hurdle of high training overhead of model soups. Algorithm 3 provide details of the pseudocode of our new **Instant Model Soup (IMS)** approach that uses cheap sparse training for injecting model soup benefits at marginal cost early before fine-tuning to save the hurdle of multiple finetuning and averaging. More specifically, IMS first creates multiple subnetworks with varying sparsities from the pre-trained dense model and train them independently for a few iterations ($\sim$100 iterations) using different hyperparameter configuration, and data subsets. Next, all the weakly trained subnetwork weights are merged together with the initial pre-trained model weights using linear interpolation following (Ilharco et al., 2022). Our proposed usage of sparse subnetworks for denoising significantly reduces the computational cost of the dense training step, thereby making it more efficient. Our experiments on CLIP (ViT-B32) and BERT-BASE impart an interesting finding that pre-trained models denoised by IMS can achieve performance comparable to model soups, without expensive full fine-tuning and then averaging.

［#45］
<table>
    <thead>
        <tr>
            <th></th>
            <th>Initial LR</th>
            <th>Epochs</th>
            <th>Compression Rate</th>
            <th>Look Ahead (C)</th>
            <th>Weight Decay</th>
            <th>Denoiser</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>BERT<sub>BASE</sub></td>
            <td>$2 \times 10^{-5}$</td>
            <td>10</td>
            <td>10%</td>
            <td>30 iter</td>
            <td>0.0-AdamW</td>
            <td>4</td>
        </tr>
        <tr>
            <td>CLIP<sub>ViT-B32</sub></td>
            <td>$1 \times 10^{-5}$</td>
            <td>22</td>
            <td>15%</td>
            <td>50 iter</td>
            <td>0.1-AdamW</td>
            <td>5</td>
        </tr>
    </tbody>
</table>

［#46］
Table 1. Details of our primary hyperparameter configurations used in our experiments across different evaluation datasets.

## 3. Experiments and Analysis

### 3.1. Network, Dataset, and Settings

［#47］
In our experiments, we adopted the official CLIP implementation provided by (Radford et al., 2021) as our starting point for our experiments using pre-trained vision transformer (Dosovitskiy et al., 2020) (ViT-B32) models. For fine-tuning CLIP, we use the frozen final classification layer output by CLIP's text tower to eliminate the necessity of introducing any learnable parameters while in the case of BERT, we add a final task-specific classification layer($\sim 3\%$ of total parameter count). For our BERT-related experiments, we use the HuggingFace (Wolf et al., 2019) pre-trained weights of BERT<sub>BASE</sub> transformer blocks and hidden state size 768. Additional necessary details of our hyperparameters required for fine-tuning are provided in Table 1. Note that during pruning, we prune the key trainable part of the network (key, query, value, dense) ignoring embeddings for simplicity. We consider a diverse set of image classification tasks from (Radford et al., 2021): Cars, GTSRB, MNIST, SVHN, CIFAR10/100 and downstream NLP tasks from GLUE (Wang et al., 2018) benchmark: MNLI, QQP, STS-B, WNLI, QNLI, MPRC, RTS, SST-2, CoLA; to thoroughly investigate the effectiveness of our proposed approaches wrt. state-of-the-art pruning benchmarks. In addition to Lottery tickets, we have compared Instant Soup Pruning (ISP) against several recently proposed pruning methods such as Lottery Pools (Yin et al., 2022), Early bird (You et al., 2019), two popular pruning at initialization methods (SNIP (Lee et al., 2018), GraSP (Yu et al.,2020)), Progressive pruning (Iterative pruning and training), and one-shot magnitude pruning.

### 3.2. Performance comparison of Instant Soup Pruning wrt. SOTA pruning methods

［#48］
In this section, we conduct a systematic and extensive study to understand the performance benefits of our proposed Instant Soup Pruning in terms of fine-tuning accuracy vs.


［#49］
Table 2. Details of fine-tuning CLIP (ViT-B32) at varying sparsity levels using Instant Soup Pruning following the settings listed in Table 1. Learning rate decays linearly from the initial value to zero. The evaluation metrics follow standards in (Radford et al., 2021). Entries with errors are the average across three runs, and errors are the standard deviations. LTH results are obtained using IMP.

［#50］
<table>
<thead>
  <tr>
    <th>Pruning Method</th>
    <th colspan="3">Cars</th>
    <th colspan="3">MNIST</th>
    <th colspan="3">SVHN</th>
    <th colspan="3">GTSRB</th>
    <th colspan="3">CIFAR10</th>
    <th colspan="3">CIFAR100</th>
  </tr>
  <tr>
    <th></th>
    <th>30%</th>
    <th>40%</th>
    <th>50%</th>
    <th>70%</th>
    <th>80%</th>
    <th>90%</th>
    <th>70%</th>
    <th>80%</th>
    <th>90%</th>
    <th>50%</th>
    <th>60%</th>
    <th>70%</th>
    <th>60%</th>
    <th>70%</th>
    <th>80%</th>
    <th>60%</th>
    <th>70%</th>
    <th>80%</th>
  </tr>
  <tr>
    <th>Full CLIP<sub>ViT-B32</sub></th>
    <th colspan="3">76.43 ± 0.5</th>
    <th colspan="3">99.61 ± 0.07</th>
    <th colspan="3">97.40 ± 0.11</th>
    <th colspan="3">99.08 ± 0.24</th>
    <th colspan="3">97.6 ± 0.24</th>
    <th colspan="3">89.35 ± 0.19</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Random</td>
    <td>9.11</td>
    <td>5.58</td>
    <td>4.47</td>
    <td>98.71</td>
    <td>97.42</td>
    <td>87.04</td>
    <td>89.85</td>
    <td>85.61</td>
    <td>73.74</td>
    <td>93.76</td>
    <td>93.65</td>
    <td>90.97</td>
    <td>74.51</td>
    <td>69.84</td>
    <td>64.96</td>
    <td>45.20</td>
    <td>39.92</td>
    <td>43.24</td>
  </tr>
  <tr>
    <td>One-shot [Mag]</td>
    <td>71.95</td>
    <td>68.07</td>
    <td>56.79</td>
    <td>99.27</td>
    <td>98.87</td>
    <td>97.47</td>
    <td>95.02</td>
    <td>91.96</td>
    <td>85.76</td>
    <td>98.60</td>
    <td>97.84</td>
    <td>96.14</td>
    <td>95.31</td>
    <td>86.56</td>
    <td>75.85</td>
    <td>80.39</td>
    <td>63.18</td>
    <td>46.63</td>
  </tr>
  <tr>
    <td>Progressive [Mag]</td>
    <td>69.85</td>
    <td>68.62</td>
    <td>64.43</td>
    <td>99.52</td>
    <td>97.77</td>
    <td>95.19</td>
    <td>95.78</td>
    <td>90.53</td>
    <td>85.75</td>
    <td>98.97</td>
    <td>97.57</td>
    <td>96.34</td>
    <td>95.25</td>
    <td>90.87</td>
    <td>78.11</td>
    <td>81.79</td>
    <td>70.53</td>
    <td>60.93</td>
  </tr>
  <tr>
    <td>EarlyBird (You et al., 2019)</td>
    <td>72.53</td>
    <td>70.76</td>
    <td>65.90</td>
    <td>99.38</td>
    <td>98.96</td>
    <td>97.64</td>
    <td>96.34</td>
    <td>95.93</td>
    <td>87.02</td>
    <td>98.15</td>
    <td>98.19</td>
    <td>97.26</td>
    <td>96.06</td>
    <td>94.18</td>
    <td>86.84</td>
    <td>84.22</td>
    <td>76.79</td>
    <td>65.67</td>
  </tr>
  <tr>
    <td>SNIP (Lee et al., 2018)</td>
    <td>71.51</td>
    <td>68.79</td>
    <td>59.01</td>
    <td>99.25</td>
    <td>98.72</td>
    <td>97.50</td>
    <td>95.33</td>
    <td>91.94</td>
    <td>82.98</td>
    <td>98.62</td>
    <td>97.95</td>
    <td>96.22</td>
    <td>95.01</td>
    <td>87.45</td>
    <td>76.12</td>
    <td>81.10</td>
    <td>62.89</td>
    <td>55.89</td>
  </tr>
  <tr>
    <td>GraSP (Wang et al., 2020)</td>
    <td>71.42</td>
    <td>68.55</td>
    <td>58.12</td>
    <td>99.30</td>
    <td>98.51</td>
    <td>97.15</td>
    <td>95.09</td>
    <td>91.44</td>
    <td>84.72</td>
    <td>98.37</td>
    <td>97.42</td>
    <td>95.91</td>
    <td>95.20</td>
    <td>86.89</td>
    <td>75.88</td>
    <td>80.67</td>
    <td>66.31</td>
    <td>52.30</td>
  </tr>
  <tr>
    <td>LTH (Frankle & Carbin, 2018)</td>
    <td>73.97</td>
    <td>72.02</td>
    <td>66.12</td>
    <td>99.41</td>
    <td>99.38</td>
    <td>98.22</td>
    <td>96.69</td>
    <td>95.28</td>
    <td>87.41</td>
    <td>98.71</td>
    <td>98.35</td>
    <td>97.79</td>
    <td>96.42</td>
    <td>94.91</td>
    <td>87.47</td>
    <td>84.25</td>
    <td>78.60</td>
    <td>65.38</td>
  </tr>
  <tr>
    <td>LTH - Rewind</td>
    <td>74.28</td>
    <td>72.09</td>
    <td>66.07</td>
    <td>99.62</td>
    <td>99.64</td>
    <td>98.18</td>
    <td>96.72</td>
    <td>95.22</td>
    <td>87.47</td>
    <td>98.78</td>
    <td>98.36</td>
    <td>97.87</td>
    <td>96.53</td>
    <td>94.88</td>
    <td>87.28</td>
    <td>84.46</td>
    <td>78.62</td>
    <td>65.71</td>
  </tr>
  <tr>
    <td>Lottery Pool (Yin et al., 2022)</td>
    <td>73.10</td>
    <td>70.53</td>
    <td>64.67</td>
    <td>99.25</td>
    <td>98.97</td>
    <td>97.76</td>
    <td>96.54</td>
    <td>95.12</td>
    <td>87.29</td>
    <td>98.52</td>
    <td>98.30</td>
    <td>97.55</td>
    <td>96.14</td>
    <td>94.50</td>
    <td>87.11</td>
    <td>84.07</td>
    <td>78.21</td>
    <td>64.39</td>
  </tr>
  <tr>
    <td>ISP [Ours]</td>
    <td>75.13</td>
    <td>72.20</td>
    <td>66.32</td>
    <td>99.69</td>
    <td>99.61</td>
    <td>98.82</td>
    <td>96.93</td>
    <td>96.46</td>
    <td>87.59</td>
    <td>99.06</td>
    <td>99.01</td>
    <td>98.52</td>
    <td>96.82</td>
    <td>95.18</td>
    <td>91.20</td>
    <td>85.11</td>
    <td>79.57</td>
    <td>71.09</td>
  </tr>
  <tr>
    <td>(std.)</td>
    <td>±0.34</td>
    <td>±0.27</td>
    <td>±0.82</td>
    <td>±0.07</td>
    <td>±0.15</td>
    <td>±0.21</td>
    <td>±0.08</td>
    <td>±0.05</td>
    <td>±0.11</td>
    <td>±0.15</td>
    <td>±0.29</td>
    <td>±0.32</td>
    <td>±0.15</td>
    <td>±0.20</td>
    <td>±0.14</td>
    <td>±0.19</td>
    <td>±0.22</td>
    <td>±0.17</td>
  </tr>
</tbody>
</table>

［#51］
Table 3. Details of fine-tuning BERT (BASE) at varying sparsity levels using Instant Soup Pruning following the settings listed in Table 1. Learning rate decays linearly from the initial value to zero. The evaluation metrics follow standards in (Wolf et al., 2019). Entries with errors are the average across three runs, and errors are the standard deviations. LTH results are obtained using IMP.

［#52］
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>MNLI</th>
    <th>QQP</th>
    <th>STS-B</th>
    <th>WNLI</th>
    <th>QNLI</th>
    <th>MPRC</th>
    <th>RTS</th>
    <th>SST-2</th>
    <th>CoLA</th>
  </tr>
  <tr>
    <th>Sparsity</th>
    <th colspan="2">90%</th>
    <th>50%</th>
    <th>90%</th>
    <th>70%</th>
    <th>50%</th>
    <th>60%</th>
    <th>60%</th>
    <th>50%</th>
  </tr>
  <tr>
    <th>Full BERT<sub>BASE</sub></th>
    <th>82.4 ± 0.5</th>
    <th>90.2 ± 0.5</th>
    <th>88.4 ± 0.3</th>
    <th>54.9 ± 1.2</th>
    <th>89.1 ± 1.0</th>
    <th>85.2 ± 0.1</th>
    <th>66.2 ± 3.6</th>
    <th>92.1 ± 0.1</th>
    <th>54.5 ± 0.4</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Random</td>
    <td>67.5</td>
    <td>76.3</td>
    <td>21.0</td>
    <td>53.5</td>
    <td>61.9</td>
    <td>69.6</td>
    <td>56.0</td>
    <td>83.1</td>
    <td>9.6</td>
  </tr>
  <tr>
    <td>One-shot</td>
    <td>78.8</td>
    <td>86.2</td>
    <td>83.9</td>
    <td>53.1</td>
    <td>86.2</td>
    <td>83.7</td>
    <td>62.9</td>
    <td>86.5</td>
    <td>49.7</td>
  </tr>
  <tr>
    <td>Progressive</td>
    <td>79.1</td>
    <td>87.5</td>
    <td>85.0</td>
    <td>53.3</td>
    <td>87.2</td>
    <td>83.8</td>
    <td>65.4</td>
    <td>86.6</td>
    <td>52.2</td>
  </tr>
  <tr>
    <td>EarlyBird</td>
    <td>82.5</td>
    <td>89.4</td>
    <td>88.1</td>
    <td>54.0</td>
    <td>88.5</td>
    <td>84.6</td>
    <td>66.1</td>
    <td>91.2</td>
    <td>53.5</td>
  </tr>
  <tr>
    <td>Lottery Ticket</td>
    <td>82.6</td>
    <td>90.0</td>
    <td>88.2</td>
    <td>54.9</td>
    <td>88.9</td>
    <td>84.9</td>
    <td>65.0</td>
    <td>91.9</td>
    <td>53.8</td>
  </tr>
  <tr>
    <td>Lottery Pool</td>
    <td>80.4</td>
    <td>89.1</td>
    <td>86.4</td>
    <td>50.9</td>
    <td>87.6</td>
    <td>84.5</td>
    <td>62.7</td>
    <td>90.9</td>
    <td>52.6</td>
  </tr>
  <tr>
    <td>ISP</td>
    <td>82.71 ± 0.6</td>
    <td>90.59 ± 0.5</td>
    <td>88.64 ± 0.1</td>
    <td>55.33 ± 0.3</td>
    <td>90.06 ±1.0</td>
    <td>85.38 ± 0.1</td>
    <td>65.96 ± 0.3</td>
    <td>92.43 ± 0.6</td>
    <td>53.61 ± 0.2</td>
  </tr>
</tbody>
</table>

［#53］
pruning ratios by comparing against multiple state-of-the-art pruning methods. We first test our approach on recently proposed CLIP (ViT-B32) (Radford et al., 2021) (unexplored for pruning till date) pre-trained with contrastive supervision from image-text pairs. For effective comparison and simplicity, all baselines and our proposed approach ISP are trained with similar optimizer and training settings provided in Table 1. Our EarlyBird (You et al., 2019), SNIP (Lee et al., 2018), and GraSP (Yu et al., 2020) baselines closely follow the implementation provided by their official GitHub repositories. In addition, to distinguish ISP from conventional progressive pruning, our progressive pruning baseline is implemented as periodic pruning using magnitude-based pruning followed by retraining. Lottery pools (Yin et al., 2022) is an interesting way to merge LTH by-product tickets. To ensure that the sparsity ratio remains comparable to ISP, we further prune the merged tickets obtained by pooling to the required sparsity, for reporting performance.

［#54］
Our results for CLIP are summarized in Table 2. We first observe that among all baselines for CLIP compression, LTH consistently performs better, and rewinding helps in further improving its performance. We found that among recent pruning at initialization methods (SNIP and GraSP), SNIP has comparatively better performance than GraSP and they tend to be slightly better than one-shot magnitude pruning. It can be clearly observed that ISP can beat expensive LTH (including rewinding) as well as all other baselines for almost all benchmark datasets and sparsity ratios. Very interestingly, for CIFAR-10 and CIFAR-100, we found that performance improvement of ISP increases with the sparsity ratio. For example, ISP surprisingly outperforms LTH-rewind by $\sim3.92$ and $\sim5.38\%$ on CIFAR-10 and CIFAR-100 respectively, while consuming merely fine-tuning cost equivalent to one single pass of LTH without the necessity of bookkeeping the rewinding weights of LTH-rewind. Our experimental results for BERT-base are summarized in Table 3. For BERT-related experiments, we have replicated the setting in (Chen et al., 2020a) and reported the performance of ISP across various GLUE benchmarking datasets at the sparsity level where LTH is able to identify winning tickets. Note that ISP is able to comfortably outperform LTH across 8 out of 9 tasks (noticeably for QNLI where ISP beat LTH by $>1\%$).

### 3.3. Analysis of Denoising Iterations in ISP

［#55］
Our proposed approach ISP is augmented by a novel idea of mask denoising, which explores the superimposition of computationally cheap pruning mask obtained by magnitude-based one-shot pruning of the network with marginal look-ahead training. In this section, we try to investigate the


［#56］
Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

［#57］
Table 4. Impact of Denoising Module in improving the performance of ISP. Results are reported for three independent runs.

［#58］
| Approach          | CLIP<sub>ViT-B32</sub> | BERT<sub>BASE</sub> |
|-------------------|------------------------|---------------------|
|                   | SVHN        | CIFAR-100 | QQP         | QNLI        |
| ISP - Denoiser    | 96.11±0.21 | 70.32±0.13 | 89.96±0.39 | 89.28±0.87 |
| ISP (Ours)        | 96.46±0.05 | 71.09±0.17 | 90.59±0.47 | 90.06±1.01 |

［#59］
Table 5. Performance comparison of ISP wrt. denoiser count on CIFAR10 with CLIP (ViT-B32) pruned at 80% sparsity.

［#60］
| Denoiser Count | 0    | 2     | 4     | 6     | 8     | 16    |
|----------------|------|-------|-------|-------|-------|-------|
| Performance    | 70.32| 70.95 | 71.15 | 71.16 | 71.07 | 71.16 |

［#61］
implication of our denoising iterations in improving ISP performance. Table 4 summarizes the performance comparison of ISP with/without the mask denoising while keeping the training settings exactly the same. Across both candidate architectures (CLIP and BERT), it can be clearly observed that ISP performance is significantly boosted by replacing the simple one-shot pruning with our denoise pruning routine. In addition, we also investigated how the number of denoising iterations will impact the ISP performance (see Table 5) and found that 4-5 denoising steps are sufficient for the denoised pruning, and increasing them beyond that does not provide a very noticeable performance gain. For consistency, in CLIP-related experiments, we have used 5 denoising iterations while for BERT, our results are reported using 4 denoising iterations.

### 3.4. Understanding the benefits of Instant Model Soups for pre-trained models

［#62］
In this section, we discuss the benefits of our sparsity-inspired extension, Instant Model Soup (IMS), and experimentally validate its surprising ability to improve the quality of pre-trained models at marginal cost. Unlike model soups, IMS provides a unique opportunity to eliminate the requirement to generate multiple fully fine-tuned models to average, thereby restricting the computational complexity equivalent to the cost of fine-tuning a single model. Table 6 illustrates the performance comparison of IMS with respect to two model soup variants (uniform and greedy) proposed in (Wortsman et al., 2022a). Note that uniform and greedy soups results are generated using the amalgamation of 8 independent models fine-tuned till the final accuracy with different hyperparameters. Our experiments across CLIP and BERT illustrate that by carefully fine-tuning IMS, it is surprisingly possible to comfortably beat the model soup variants significantly. The denoised pre-trained model generated by IMS has the ability to converge to equivalent (even better) performance than model soups. Adhering to the theme of ISP, IMS also conveys a strong message that it is not necessarily important to wait till model convergence to ripe the benefits of soup, but astonishingly soup benefits are available to ripe early during the fine-tuning at a marginal computational cost.

［#63］
Table 6. Fine-tuning performance comparison of our proposed approach (IMS) wrt. basic fine-tuning and model soup variants.

［#64］
| Approach                          | CLIP<sub>ViT-B32</sub> | BERT<sub>BASE</sub> |
|-----------------------------------|------------------------|---------------------|
|                                   | Cars   | CIFAR100 | MNLI  | MNLI  | QNLI  |
| Pretrained<sub>BASE</sub>         | 76.43  | 97.60    | 89.35 | 82.39 | 90.04 |
| Uniform Soup (Wortsman et al., 2022a) | 76.32  | 97.68    | 89.20 | 82.41 | 89.76 |
| Greedy Soup (Wortsman et al., 2022a) | 77.95  | 98.05    | 89.54 | 83.01 | 90.64 |
| IMS [Ours]<br>(std.)              | 78.79<br>± 0.32 | 98.01<br>± 0.07 | 89.64<br>± 0.12 | 83.63<br>± 0.43 | 91.23<br>± 0.19 |

［#65］
Table 7. Performance comparison of ISP wrt. look-ahead on CIFAR100 with CLIP (ViT-B32) pruned at 60% sparsity.

［#66］
| Look-ahead Count(C) | 10    | 30     | 50     | 100    | 150    | 200    |
|---------------------|-------|--------|--------|--------|--------|--------|
| Performance         | 84.71 | 84.92  | 85.18  | 85.19  | 84.41  | 85.41  |

## 4. Related Work

［#67］
Linear interpolation of neural network weights has recently achieved significant attention, but due to numerous non-linear activations within a neural network, it is still debatable if linearly interpolating between two sets of weights can result in a high accuracy solution. Recently, (Frankle et al., 2020; Nagarajan & Kolter, 2019; Von Oswald et al., 2020; Matena & Raffel, 2021; Wortsman et al., 2022a;b; Choshen et al., 2022; Izmailov et al., 2018; Neyshabur et al., 2020) have studied the interpolation of deep networks and validated performance benefits when training starts from a common initialization or some segment of the optimization trajectories are shared. While (Nagarajan & Kolter, 2019; Frankle et al., 2020) focused on mergability in the case of models trained on a single task, (Wortsman et al., 2022b) found that weight interpolation can not only benefit fine-tuning tasks but also under distribution shift. More specifically, they average zero-shot and fine-tuned models, finding improvements in- and out-of-distribution.

［#68］
Recently, (Matena & Raffel, 2021) used Fisher-weighted averaging of language models before and after fine-tuning on downstream tasks. They merged models with the same pre-trained initialization that are fine-tuned on different text classification tasks. In the late phases of training, (Von Oswald et al., 2020) studied making copies of a subset of the neural network parameters and proposed to independently optimize them, followed by averaging. Moreover, (Wortsman et al., 2022a) proposed to average fine-tuned models across independent runs with hyperparameter diversity, modifying all the weights of the network, and showing significant performance benefits. In addition to model weight averaging, (Bansal et al., 2021; Yang et al., 2022) explored the idea of model stitching, where given two trained and frozen models A and B, a "stitched model" formed by connecting the bottom-layers of A to the top-layers of B, with a simple trainable layer between them. (Fort et al., 2019) studied deep ensembles which have empirically shown promise for improving the accuracy, uncertainty and out-of-distribution robustness of deep learning models.


### 5. Conclusion
［#69］
In this work, we introduced Instant Soup Pruning, a model soup-inspired perspective dedicated to generating LTH quality subnetworks, using a fraction of the original IMP cost. ISP is augmented by a denoising pruning module which helps in replacing the expensive intermediate pruning stages of IMP with computationally efficient weak mask generation and aggregation routine. Additionally, we present Instant Model Soup, which provides an opportunity to inject the benefits of model soups in dense pre-trained models at marginal training cost, thereby improving fine-tuning performance comparable to model soups. Our future work will aim for a more theoretical understanding of the role of our denoisers in providing experimental benefits.

### 6. Acknowledgement
［#70］
The research is based upon work supported in part by the Intelligence Advanced Research Projects Activity (IARPA) under Contract No. 2022-21102100004. We also acknowledge support from the National Science Foundation AI Center Institute for Foundations of Machine Learning (IFML) at the University of Texas at Austin.

### References

























































