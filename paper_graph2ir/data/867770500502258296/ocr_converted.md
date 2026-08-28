# Train Flat, Then Compress:
［#1］
Sharpness-Aware Minimization Learns More Compressible Models

［#2］
Clara Na $\quad$ Sanket Vaibhav Mehta $\quad$ Emma Strubell
Language Technologies Institute
School of Computer Science
Carnegie Mellon University
{csna, svmehta, estrubel}@cs.cmu.edu

## Abstract
［#3］
Model compression by way of parameter pruning, quantization, or distillation has recently gained popularity as an approach for reducing the computational requirements of modern deep neural network models for NLP. Inspired by prior works suggesting a connection between simpler, more generalizable models and those that lie within wider loss basins, we hypothesize that optimizing for flat minima should lead to simpler parameterizations and thus more compressible models. We propose to combine sharpness-aware minimization (SAM) with various task-specific model compression methods, including iterative magnitude pruning (IMP), structured pruning with a distillation objective, and post-training dynamic quantization. Empirically, we show that optimizing for flatter minima consistently leads to greater compressibility of parameters compared to vanilla Adam when fine-tuning BERT models, with little to no loss in accuracy on the GLUE text classification and SQuAD question answering benchmarks. Moreover, SAM finds superior winning tickets during IMP that 1) are amenable to vanilla Adam optimization, and 2) transfer more effectively across tasks. $^1$

## 1 Introduction
［#4］
Recent advances in hardware, modeling, and optimization for deep neural networks have enabled training of substantially larger models on massive amounts of unlabeled data, leading to corresponding improvements in accuracy across a variety of tasks in NLP (Devlin et al., 2019; Brown et al., 2020; Raffel et al., 2020). Unfortunately, this sudden increase in scale of state-of-the-art models also has adverse consequences, such as reducing equity of access (Yu, 2020; Ahmed and Wahed, 2020) and increasing computational and energy requirements (Strubell et al., 2019; Dodge et al., 2022).

［#5］
![](./images/867770500502258296_1.jpg)

［#6］
Figure 1: Average score over all GLUE tasks as a function of sparsity (% of parameters pruned) of BERT$_{base}$ through unstructured iterative magnitude pruning. We compare the baseline Adam optimized model's performance to our model, optimized to prefer flat minima via SAM. The green horizontal bands mark initial performance of our full fine-tuned (FT) models. SAM outperforms baseline Adam across all sparsity values.

［#7］
In response, model compression has emerged as a dominant approach to improving memory and inference efficiency in neural network models, including approaches such as knowledge distillation (Bucilu˘a et al., 2006; Hinton et al., 2014; Jiao et al., 2020), model quantization (Vanhoucke et al., 2011; Shen et al., 2020) and pruning (LeCun et al., 1989; Chen et al., 2020; Xia et al., 2022).

［#8］
The vast majority of work on model compression focuses on methods for selecting how and where to reduce (via quantization or distillation) or remove (by pruning) model parameters without sacrificing end-task accuracy. While these approaches are usually simple to implement and work relatively well for maintaining overall accuracy on considered benchmarks, recent work has revealed that these commonly used compression methods can result in negative impacts on model behavior that are not necessarily captured by current performance met-

---
［#3］
$^1$Code is available at https://github.com/clarana/train-flat-compress

［#9］
rics. For example, in the field of computer vision, pruning has been shown to sacrifice performance on long-tail examples in order to preserve accuracy on more frequent phenomena (Hooker et al., 2020) and reduce robustness to out-of-distribution examples (Liebenwein et al., 2021). At the same time, it has also been shown that compression can sometimes have a regularizing effect, improving generalization in some settings (Ahia et al., 2021). Clearly, more work is needed better understand the relationship between compression and generalizability, and to develop improved compression methods that are informed by this knowledge.

［#10］
Meanwhile, there is a growing body of work investigating curvature of the loss landscape and its relation to generalization in deep neural models. Hochreiter and Schmidhuber (1997) first defined flat minima as regions in parameter space where error remains relatively stable despite perturbations in parameter values, arguing that models in flat minima should correspond to simpler, more generalizable functions. More recently, Wu et al. (2017) further showed that wide loss basins correspond to low-complexity solutions,² and it has been shown empirically that directly optimizing for solutions in flat minima leads to improved generalization on a wide range of supervised learning tasks in both vision (Foret et al., 2021) and language (Bahri et al., 2022) modalities.

［#11］
Inspired by the above results connecting wider loss regions to simpler, more generalizable models, in this work we aim to advance our understanding of model compression by examining the relationship between flat minima and compressibility in fine-tuned language models. Intuitively, model parameters in neighborhoods having uniformly low loss values should be more robust to perturbations, such as those resulting from model compression, compared to models in sharper regions, since changes to parameter values should lead to minimal change in loss with respect to the main training objective. Empirically and theoretically, previous work has also linked generalizability with compressibility, finding that neural networks whose weights reflect simpler, more general hypotheses may be more robust to compression, and compression itself can act as a regularization mechanism (Zhou et al., 2019; Liang et al., 2021; Kuhn et al., 2021). Li et al. (2020) showed that larger pre-trained language models, which are known to genereralize better, are also more compressible. Further, Bartoldson et al. (2020) connect flatness to generalization in pruned models, showing that pruning noise correlates positively with measures of flatness in a CNN for computer vision.

［#12］
We investigate the relationship between flat minima and compression in large pre-trained language models by directly optimizing for flat minima during language model fine-tuning using sharpness-aware minimization (SAM; Foret et al. (2021)). Through extensive experiments on the GLUE text classification and SQuAD question answering benchmarks, we show that fine-tuning BERT models with SAM leads to optima in flatter basins, and compressing those models consistently results in higher model accuracy at the same level of compression compared to standard Adam-optimized baselines. Our results hold across multiple BERT variants and sizes (Devlin et al., 2019; Liu et al., 2019) and a wide variety of compression methods: Iterative magnitude pruning with and without rewinding, a structured pruning procedure that also employs knowledge distillation, and an off-the-shelf method for post-training quantization. We also show that sparse subnetworks (winning tickets; Frankle and Carbin (2019)) discovered by SAM are better transferable across tasks, suggesting improved generalizability. Our findings shed light on a promising new avenue for obtaining practical improvements in model compression.

## 2 Methods

［#13］
Broadly, we are interested in understanding: 1) Are models in flatter minima more compressible? 2) If so, why? 3) Beyond task-specific accuracy, what properties do flat, compressed models have?

［#14］
To provide empirical answers to these questions, our experimental setup is as follows. We fine-tune pre-trained language models on standard benchmarks using both "vanilla" Adam optimization and Sharpness-Aware Minimization (§2.1). We then experiment with a variety of strategies for compressing those fine-tuned models: iterative magnitude pruning (unstructured) with and without rewinding (§2.2.1), structured pruning using $\ell_0$ regularization and a distillation objective (§2.2.2), and post-training dynamic quantization (§2.3). We evaluate model end-task accuracy at different compression rates, and compare that accuracy to the full (uncompressed) model and across experimental settings,

---
［#10］
²Wu et al. (2017) demonstrate this theoretically for two layer feed-forward networks, and empirically for larger networks applied to computer vision.

［#15］
such as varying the pre-trained language model, and transferring initializations across tasks.

### 2.1 Flat Minima
［#16］
Sharpness-Aware Minimization (SAM). To explicitly encourage flatter loss basins, we employ the SAM (Foret et al., 2021) procedure. Given a loss function $f(w)$, SAM strives to find parameters that lie in the neighborhood with uniformly low loss by optimizing the following minimax objective:
［#16］
$$
\min _{w} \max _{\|\epsilon\|_{2} \leq \rho} f(w+\epsilon) \tag{1}
$$
［#16］
where the maximization (or neighborhood) region is an $\ell^p$ ball with radius $\rho$ for $p=2$ in Equation (1). The gradient of the result of the above (inner) maximization problem can be approximated as:
［#16］
$$
\left.\approx \nabla_{w} f(w)\right|_{w+\hat{\epsilon}(\mathbf{w})}+\left.\frac{\partial \hat{\epsilon}(\mathbf{w})}{w} \nabla_{w} f(w)\right|_{w+\hat{\epsilon}(\mathbf{w})}
$$
［#16］
where, $\hat{\epsilon}(\mathbf{w})=\rho \nabla_{w} f(w) /\left\|\nabla_{w} f(w)\right\|_{2}$

［#17］
Foret et al. (2021) showcase that one can simplify the optimization problem without compromising the algorithm's effectiveness by dropping the second order term in the gradient, leaving us with:
［#17］
$$
\nabla_{w} \max _{\|\epsilon\|_{2} \leq \rho} f(w+\epsilon) \approx\left.\nabla_{w} f(w)\right|_{w+\hat{\epsilon}(\mathbf{w})} \tag{2}
$$

［#18］
Intuitively, SAM takes a gradient step at each iteration based on the gradient estimated at the parameters yielding the highest loss $(w+\hat{\epsilon}(w))$ in an $\ell^p$ neighborhood around the current parameters $(w)$.

［#19］
Stochastic Weight Averaging (SWA). Although, we use SAM to optimize for flatness, there exist other alternatives to promote flatness like Stochastic Weight Averaging (Izmailov et al., 2018). SWA performs an equal average of model checkpoints along the optimization trajectory to find flatter solutions compared to vanilla optimization. We also consider SWA for our experimentation (§3.4.7).

［#20］
Sharpness Metric. To verify that SAM and SWA indeed leads to flatter minima as compared to Adam, we compute a **sharpness metric** (Keskar et al., 2017), which estimates the flatness by computing the maximum value of $f(w)$ within a neighborhood region controlled by the hyperparameter $\epsilon$. Following (Keskar et al., 2017; Mehta et al., 2021), the neighborhood region is defined as:
［#20］
$$
\begin{aligned}
C_{\epsilon}= & \left\{z \in R^{p}:-\epsilon\left(\left|\left(A^{+} w\right)_{i}\right|+1\right) \leq\right. \\
& \left.z_{i} \leq \epsilon\left(\left|\left(A^{+} w\right)_{i}\right|+1\right) \forall i \in\{1 \ldots p\}\right\} \quad(3)
\end{aligned}
$$
［#20］
where $R^p$ is a random subspace of the entire parameter space $R^n$ constructed using a projection matrix $A \in R^{n \times p}$, and $A^{+}$is the pseudo inverse of $A$. Concretely, the sharpness metric (lower corresponds to flatter minima) is computed as follows:
［#20］
$$
\phi_{w, f}:=\frac{\max _{z \in C_{\epsilon}} f(w+A z)-f(w)}{1+f(w)} \times 100
\tag{4}
$$

［#21］
To qualitatively verify for flatness, we also visualize loss contours (see Appendix A.2).

### 2.2 Pruning
［#22］
We investigate compressibility primarily in an unstructured pruning setting. Given a network $\mathcal{N}$ and weights $\mathbf{w}$, we wish to prune a subset of individual weights to leave only a subset, $\mathbf{w}'$. A successfully pruned model has $|\mathbf{w}'| \ll|\mathbf{w}|$ while retaining good performance on the task(s) of interest.

#### 2.2.1 Iterative Magnitude Pruning
［#23］
Typically, $\mathbf{w}'$ is found through an iterative process where $\mathcal{N}$ is trained and pruned incrementally, either until some target sparsity is reached or until some larger-than-desired performance drop is observed, and a common criterion selects weights with the smallest absolute magnitude to be pruned at each iteration. In the standard pruning scenario (Renda et al., 2020a; Han et al., 2015), training simply resumes with the remaining weights after each iteration of pruning. Previous work (Renda et al., 2020a) presents evidence that rewinding remaining weights to earlier learned values may be beneficial for compressibility.

［#24］
Frankle and Carbin (2019) present a formulation of iterative magnitude pruning (IMP) as a way to obtain sparse "winning tickets" $\mathbf{w}'$ that can be trained from initialization to reach performance to match that of the original full network $\mathcal{N}$ while using significantly fewer parameters. In IMP, model parameters are repeatedly reset to their original initialized values after pruning, before the next iteration of training. Reverting weights to values from an earlier point during training is also known as **rewinding** (Frankle et al., 2020). Following Chen et al. (2020), we consider both standard (no rewinding) and lottery ticket-style IMP (with rewinding) settings. In alignment with the paradigm of pre-training and fine-tuning, we treat the pre-trained model's weights as the initial weights to which parameters are reset at each iteration.

### 2.2.2 Structured Pruning

［#25］
We also explore flat minima in a recently proposed structured pruning setting: Xia et al. (2022) incorporate a layerwise distillation objective into their structured pruning process, which dynamically maps layers between teacher and student models as structured units of varying granularity are incrementally pruned in the student model via an $\ell_0$ regularizer. In our experiments, we vary only optimizer used to fine-tune the *teacher model* and compare downstream compression performance.

### 2.3 Post-Training Quantization

［#26］
We compare performance of Int8 quantized $\text{BERT}_{base}$ models fine-tuned with Adam- and SAM-optimized models. Using a standard PyTorch implementation$^3$, we perform *post-training dynamic quantization*, where full-size (32-bit) floating point model weights are statically mapped to a lower precision (in our case, 8-bit integer) representation after training, and activations are dynamically reduced in precision during inference.

## 3 Experimentation

### 3.1 Research Questions

［#27］
In this section, we describe a series of experiments and analyses aimed at answering the following research questions:

［#28］
Q0 Does SAM help make models more robust to compression? (§3.4.1, §3.4.4, §3.4.5)

［#29］
Q1 Does SAM benefit compressibility across different model initializations and sizes? (§3.4.6)

［#30］
Q2 *How* does SAM influence model compressibility? (§3.4.2)

［#31］
Q3 How does SAM affect compressed model properties beyond single-task accuracy? (§3.4.2, §3.4.3)

［#32］
Q4 How does flatness in general, beyond SAM specifically, influence compressibility? (§3.4.7)

### 3.2 Datasets and Metrics

［#33］
We consider eight tasks from the standard GLUE (Wang et al., 2018) benchmark for our experimentation, as well as SQuAD v1.1 (Rajpurkar et al., 2016). The GLUE datasets include MNLI (Williams et al., 2018), QQP$^4$, STS-B (Cer et al., 2017), QNLI (Wang et al., 2018), MRPC (Dolan and Brockett, 2005), RTE (Wang et al., 2018), SST-2 (Socher et al., 2013), and CoLA (Warstadt et al., 2019). For all experiments unless otherwise noted, we follow prior work (Chen et al., 2020) and report validation set accuracy for QQP, QNLI, MRPC, RTE, SST-2, matched accuracy for MNLI, Matthew's correlation for CoLA, Pearson correlation for STS-B, and F1 score for SQuAD. We also make use of a sharpness metric (§A.2) to quantify the flatness of the basins that our models lie in.

### 3.3 Implementation Details

［#34］
For all experiments described in this section, we fine-tune publicly available pre-trained BERT model weights (Wolf et al., 2019). We use the uncased $\text{BERT}_{base}$ model for all experiments except when otherwise noted. For our iterative magnitude pruning experiments, we mainly set hyperparameters as mentioned by Chen et al. (2020) and follow a similar general procedure for iterative magnitude pruning, pruning an absolute 10% of prunable weights over 9 iterations to reach 90% sparsity in order to facilitate direct comparisons. Appendix A.4 contains further implementation details including hyperparameters used and explanations of when our methods differ. For our SAM optimizer, we use Adam (Kingma and Ba, 2014) as our base optimizer, and following (Mehta et al., 2021; Bahri et al., 2022) set $\rho$ to 0.05. Appendix A.2 contains implementation details for SWA.

### 3.4 Results

#### 3.4.1 Iterative Magnitude Pruning (Q0)

［#35］
With rewind to $\text{BERT}_{base}$ We investigate the SAM procedure's effectiveness in uncovering winning tickets (Frankle and Carbin, 2019). The IMP section of Table 1 shows that **optimizing with SAM throughout iterative magnitude pruning allows pruned models to retain higher performance at reference sparsity levels when compared to models trained with vanilla Adam optimizer.

［#36］
The plots in Figure 2 show evaluation metrics over successive IMP iterations for individual GLUE tasks. We see that although initial performance of Adam- and SAM-optimized models is usually comparable, promoting flat minima during

---
［#26］
$^3$https://pytorch.org/docs/stable/quantization.html
［#33］
$^4$https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

［#37］
<table>
 <thead>
  <tr>
   <th>
    Dataset
   </th>
   <th>
    MNLI
   </th>
   <th>
    QQP
   </th>
   <th>
    STS-B
   </th>
   <th>
    QNLI
   </th>
   <th>
    MRPC
   </th>
   <th>
    RTE
   </th>
   <th>
    SST-2
   </th>
   <th>
    CoLA
   </th>
   <th>
    SQuAD
   </th>
  </tr>
  <tr>
   <th>
    Sparsity
   </th>
   <th>
    70%
   </th>
   <th>
    90%
   </th>
   <th>
    50%
   </th>
   <th>
    70%
   </th>
   <th>
    50%
   </th>
   <th>
    60%
   </th>
   <th>
    60%
   </th>
   <th>
    50%
   </th>
   <th>
    40%
   </th>
  </tr>
  <tr>
   <th>
    Metric
   </th>
   <th>
    Match/Mismatch acc.
   </th>
   <th>
    Acc.
   </th>
   <th>
    Pearson Cor.
   </th>
   <th>
    Acc.
   </th>
   <th>
    Acc.
   </th>
   <th>
    Acc.
   </th>
   <th>
    Acc.
   </th>
   <th>
    Matthew’s Cor.
   </th>
   <th>
    F1
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <th>
    Full Adam
   </th>
   <td>
    84.60.1/83.60.3
   </td>
   <td>
    89.10.1
   </td>
   <td>
    84.00.4
   </td>
   <td>
    90.80.2
   </td>
   <td>
    82.61.4
   </td>
   <td>
    67.01.4
   </td>
   <td>
    93.20.6
   </td>
   <td>
    53.31.2
   </td>
   <td>
    88.50.2
   </td>
  </tr>
  <tr>
   <th>
    SAM
   </th>
   <td>
    85.00.1/84.50.1
   </td>
   <td>
    89.20.2
   </td>
   <td>
    84.70.1
   </td>
   <td>
    90.80.4
   </td>
   <td>
    82.91.3
   </td>
   <td>
    65.50.8
   </td>
   <td>
    93.60.1
   </td>
   <td>
    54.11.3
   </td>
   <td>
    89.20.1
   </td>
  </tr>
  <tr>
   <th>
    IMP Adam
   </th>
   <td>
    82.30.3/81.40.2
   </td>
   <td>
    83.00.0
   </td>
   <td>
    83.30.3
   </td>
   <td>
    88.60.3
   </td>
   <td>
    81.51.3
   </td>
   <td>
    63.32.8
   </td>
   <td>
    92.20.4
   </td>
   <td>
    47.71.5
   </td>
   <td>
    86.90.3
   </td>
  </tr>
  <tr>
   <th>
    SAM
   </th>
   <td>
    83.20.2/82.50.4
   </td>
   <td>
    85.40.1
   </td>
   <td>
    84.10.1
   </td>
   <td>
    89.40.2
   </td>
   <td>
    83.60.2
   </td>
   <td>
    65.01.5
   </td>
   <td>
    92.90.5
   </td>
   <td>
    49.52.0
   </td>
   <td>
    87.80.2
   </td>
  </tr>
  <tr>
   <th>
    *SWA
   </th>
   <td>
    83.10.1/82.10.1
   </td>
   <td>
    87.70.2
   </td>
   <td>
    85.30.4
   </td>
   <td>
    89.00.2
   </td>
   <td>
    83.70.1
   </td>
   <td>
    67.40.4
   </td>
   <td>
    92.70.3
   </td>
   <td>
    50.91.0
   </td>
   <td>
    $-$
   </td>
  </tr>
  <tr>
   <th>
    Std Adam
   </th>
   <td>
    82.40.3/81.10.5
   </td>
   <td>
    87.20.1
   </td>
   <td>
    83.90.1
   </td>
   <td>
    88.80.1
   </td>
   <td>
    82.91.0
   </td>
   <td>
    64.41.6
   </td>
   <td>
    92.30.2
   </td>
   <td>
    50.90.2
   </td>
   <td>
    86.30.2
   </td>
  </tr>
  <tr>
   <th>
    SAM
   </th>
   <td>
    83.20.1/81.80.1
   </td>
   <td>
    87.40.4
   </td>
   <td>
    84.60.1
   </td>
   <td>
    89.40.1
   </td>
   <td>
    81.91.2
   </td>
   <td>
    64.30.8
   </td>
   <td>
    93.00.6
   </td>
   <td>
    51.32.0
   </td>
   <td>
    87.10.2
   </td>
  </tr>
 </tbody>
</table>

［#38］
Table 1: At full size and at Chen et al. (2020)’s reference sparsities, we report task-specific metrics for Adam and SAM-optimized BERT$_{base}$ models in their (1) Iterative Magnitude Pruning (IMP) and (2) Std pruning settings. We report mean and standard deviation calculated over 3 random seeds. All GLUE results are reported for test sets. We report results on the development set for SQuAD as test set evaluation is unavailable for v1.1. Table 5 in Appendix A.5 contains comparison with reference (Ref) values using development sets. *Additionally, we report test accuracy metrics in IMP models trained with stochastic weight averaging (SWA), another optimization method which empirically leads to flatter minima. We observe that optimizing with SAM or SWA throughout iterative magnitude pruning allows pruned models to retain higher performance at reference sparsity levels when compared to models trained with vanilla Adam optimizer.

［#39］
IMP with SAM leads to either 1) higher performance compared to Adam at Chen et al. (2020)’s reference sparsity level$^{5}$ or 2) performance comparable to the full sized model at higher sparsity levels, if not both, for all but one smaller GLUE task (CoLA).

［#40］
Moreover, for some tasks (RTE, SST-2), the final pruned model’s accuracy tends to be higher than the full BERT$_{base}$ fine-tuned model’s. This is an especially striking result given that we reset remaining weights to the BERT$_{base}$ initialization after each successive iteration of pruning; there is no progressive learning of weights from iteration to iteration. Instead, we reach higher accuracy simply by optimizing over the learned substructures rather than the full network.

［#41］
Both SAM and vanilla Adam induce marginal improvements compared to the full fine-tuned model with $10\%$ pruning for some tasks (e.g. 2b, 2f, 2g), which we might attribute to slight pruning acting as a structure regularizer. However, only SAM-optimized models sometimes continue to exhibit improvements in much later stages of pruning.

［#42］
"Standard pruning" without rewind We also investigate SAM and vanilla Adam optimizers in the standard pruning setting, where we continue training immediately after pruning in each iteration, without resetting remaining weights to pre-trained BERT$_{base}$ initialization. The Std section of Table 1 shows these results.

［#43］
SAM-optimized models retain more of the full sized model’s performance at Chen et al. (2020)’s reference sparsity levels. However, the trend is not more stark in this setting, which hints at the structure of winning tickets found by SAM playing an important role.

［#44］
We more directly investigate how SAM benefits model compressibility in Section 3.4.2, but we also take care to rule out the possibility that SAM is simply acting as an implicit $\ell_1$ regularizer (A.7).

### 3.4.2 Analysis: Answering the Structure vs. Optimization Question (Q2, Q3)

［#45］
In this experimental setting, we aim to disentangle the effects of the structure$^{6}$ of the pruning masks learned using different optimizers, from the optimization over given substructures using different optimizers. Figure 3 displays our results.

［#46］
We observe that, in general, subnetworks learned through IMP greatly outperform random subnetworks of the same sparsity when trained. Subnetworks found with SAM tend to outperform those found with vanilla Adam, especially with subsequent optimization with Adam. Furthermore, training any given IMP-learned subnetwork with SAM yields modest improvements in accuracy compared to fine-tuning with vanilla Adam.

［#47］
$^{5}$Our Adam optimized models often differ in performance from Chen et al. (2020)’s at iteration 0 – the fairest comparison is between our implementation of Adam and SAM optimized models, and our reference performance bands in Figure 2 are often much higher than originally reported.

［#48］
$^{6}$In this work, we refer to "winning tickets" and "structures" thereof interchangeably, although, as observed by Frankle and Carbin (2019), winning tickets are conditioned on models’ (in our case, pre-trained) initializations.

［#49］
![](./images/867770500502258296_2.jpg)

［#50］
Figure 2: Individual plots showing sparsity vs. task metrics (validation set) for GLUE throughout IMP. The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned (uncompressed) models.

［#51］
![](./images/867770500502258296_3.jpg)

［#52］
Figure 3: We compare optimizers' learned tickets, as well as their performance in optimization over a given ticket. For select GLUE tasks at their reference sparsity values, we fine-tune pruned subnetworks of pre-trained BERT$_{base}$ initializations networks based on 1) random masks, 2) Adam-learned masks, and 3) SAM-learned masks, using a) SAM and b) Adam optimizers. SAM optimization over SAM- and Adam-learned winning tickets tends to yield marginal improvements compared to Adam optimization. Comparing bar heights from left to right within each figure allows us to see that, at least when Adam is used for final fine-tuning, a Random- $\ll$ Adam- $<$ SAM-learned masks. Exact values are reported in Table 6, and Figure 12 in Appendix shows an alternative view of the data.

### 3.4.3 Analysis: Comparing Transferability of Winning Tickets (Q3)

［#53］
We explore the extent to which SAM tickets are more or less transferable across tasks compared to tickets discovered by Adam (Figure 4). For consistency, we use 70% sparsity tickets for all tasks evaluated. SAM-learned tickets tend to transfer better across tasks than Adam-learned tickets. This complements our findings in §3.4.2, which can be interpreted as a study on transferability of winning tickets between optimizers instead of across tasks.

［#54］
We also compare SAM versus Adam as an op- timizer for fine-tuning in this setting, and found that SAM did not seem to work better overall as an optimizer, given a ticket for a different task (see Figure 13 in Appendix). Note that this does not directly contradict our results from §3.4.2; SAM optimization does typically benefit same-task per- formance (see diagonals in these figures).

［#55］
![](./images/867770500502258296_4.jpg)

［#56］
Figure 4: Heatmaps indicating the difference in target task performance between 70% sparsity SAM and Adam tickets when transferring tickets across tasks, fine-tuned with either SAM (top) or Adam (bottom) optimizers during IMP. Values $>0$ indicate the extent to which SAM tickets transferred better than Adam tickets; Values $<0$ indicate where Adam tickets transferred better than SAM. Overall, SAM tickets transfer better regardless of the final fine-tuning optimizer. Note that the positive values along the diagonal indicate superior SAM ticket performance in the single task setting, even with "transfer" between optimizers.

### 3.4.4 Structured Pruning (Q0)
［#57］
We use full-size fine-tuned $\text{BERT}_{base}$ models from §3.4.1 as teacher models for the layerwise distillation objective used throughout the structured pruning procedure. The pruning procedure itself is the same as Xia et al. (2022)'s, regardless of whether the teacher model was originally trained using SAM or vanilla Adam. Appendix A.10 contains further implementation details.

［#58］
In Table 2, we show that SAM-optimized teacher models improve compressed student model performance in this structured pruning setting. SAM outperforming a vanilla Adam optimizer in this setting is a particularly desirable goal from a prac- tical standpoint. First, unlike in a full IMP setting, we only use SAM for a single fine-tuning in the pruning pipeline, and so we only incur the computational overhead associated with SAM for a fraction of the overall pruning process. Second, training time for CoFi's pruning process itself has a ten- fold speedup compared to the TinyBERT baseline (Jiao et al., 2020). Finally, the pruned models obtained in this setting perform inference as quickly as TinyBERT, which amounts to a tenfold speedup compared to the full $\text{BERT}_{base}$ model.

［#59］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Optim.</th>
      <th>Teacher Acc.</th>
      <th>Pruned Acc.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">SST-2<br>(67k)</td>
      <td>Adam</td>
      <td>92.7<sub>0.1</sub></td>
      <td>90.2<sub>0.5</sub></td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>93.1<sub>0.6</sub></td>
      <td>91.3<sub>0.3</sub></td>
    </tr>
    <tr>
      <td rowspan="2">QNLI<br>(105k)</td>
      <td>Adam</td>
      <td>91.5<sub>0.1</sub></td>
      <td>85.9<sub>0.4</sub></td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>91.3<sub>0.6</sub></td>
      <td>86.9<sub>0.4</sub></td>
    </tr>
    <tr>
      <td rowspan="2">QQP<br>(364k)</td>
      <td>Adam</td>
      <td>91.0<sub>0.1</sub></td>
      <td>90.0<sub>0.1</sub></td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>91.1<sub>0.1</sub></td>
      <td>90.1<sub>0.1</sub></td>
    </tr>
    <tr>
      <td rowspan="2">MNLI<br>(393k)</td>
      <td>Adam</td>
      <td>84.7<sub>0.4</sub></td>
      <td>80.2<sub>0.4</sub></td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>85.3<sub>0.2</sub></td>
      <td>80.6<sub>0.1</sub></td>
    </tr>
  </tbody>
</table>

［#60］
Table 2: Comparison between pruned models obtained using teacher models fine-tuned with Adam and SAM optimizers. Numbers reported are means<sub>stddev</sub> ($n=3$) for evaluation metrics on the development set. Compressed models are trained to reach 95% sparsity using optimal values for $\lambda$ and finetuning learning rate from Xia et al. (2022)'s structured pruning setting.

### 3.4.5 Post-Training Quantization (Q0)

［#61］
![](./images/867770500502258296_5.jpg)

［#62］
Figure 5: We compare full fine-tuned and quantized BERT-base models optimized with SAM and Adam. Error bars show standard deviations for $n=3$. Additionally, we show that applying a simpler post-training dynamic quantization technique on a SAM-optimized model can approach the reported performance of a model quantized through quantization-aware training (QAT) (Zafrir et al., 2019).

［#63］
From Figure 5 (see A.11 for actual numbers), we

［#63］
can make a few key observations: 1) SAM-trained models retain higher task performance after quantization compared to Adam-trained models. 2) SAM-trained models' higher performance is also more stable across random seeds. Finally, 3) for some tasks, our SAM-trained models compressed with simple post-training dynamic quantization meet or approach the performance of Zafrir et al. (2019)'s models quantized through quantization-aware training (QAT), which tends to yield better compressed models but is more complex to implement and use. The benefits of quantization vary under different settings and hardware, but our $\text{BERT}_{base}$ models quantized to Int8 precision have a 2.5x reduction in storage requirements and 1.5-2x faster inference.

### 3.4.6 Evaluating Other Models (Q1)
［#64］
In order to explore the applicability of our findings across model sizes and other BERT variants, we experiment with SAM and Adam optimizers on $\text{BERT}_{large}$ and $\text{RoBERTa}_{base}$ models in the IMP setting of §3.4.1. We find that, indeed, SAM-optimized models fare better than Adam-optimized models in these other BERT variants. A.12 includes details relevant for reproducibility, as well as task-specific results.

［#65］
More generally, our proposal to "train flat" with SAM is compatible with Li et al. (2020)'s recommendation to "train large, then compress", and with starting with a better-performing model before compression. Starting with $\text{BERT}_{large}$ or $\text{RoBERTa}_{base}$ can lead to clearly higher compressed accuracy at similar target sparsity levels and rarely leads to significantly worse performance. However, the higher performance often does not simply follow a parallel pattern as in $\text{BERT}_{base}$ models throughout pruning; the initial performance gaps between $\text{BERT}_{large}$ and $\text{BERT}_{base}$ models tend to be preserved slightly more reliably at higher sparsity levels than the gaps between $\text{RoBERTa}_{base}$ and $\text{BERT}_{base}$ models. This prompts further investigation into the properties of the subnetworks found in these BERT variants and appeals to the potential compressibility of even larger models when flatness-optimized.

### 3.4.7 Stochastic Weight Averaging (Q4)
［#66］
We conduct experiments matching the IMP setting with rewind from §3.4.1 using stochastic weight averaging (SWA). In Table 1, we report results on GLUE test set for SWA with IMP. Similar to SAM, we observe that SWA is superior to Adam optimized models at reference sparsity levels.

## 4 Related Work
［#67］
In this section, we draw connections to key related work. For a more general description of related work, please refer to A.1 in Appendix.

［#68］
First, we briefly recount previous work that we consider for our experimentation. We use Adam (Kingma and Ba, 2014) as a base optimizer, with comparisons between vanilla Adam and the addition of Sharpness-Aware-Minimization (Foret et al., 2021). We make further comparisons in a subset of our experiments with Stochastic Weight Averaging (Izmailov et al., 2018) as an alternative method of inducing flatness. Using Keskar et al. (2017)'s $\epsilon$-sharpness metric, and based on Mehta et al. (2021)'s implementation, we verify that SAM and SWA induce flatness. We fine-tune BERT models (Devlin et al., 2019; Liu et al., 2019) for GLUE (Wang et al., 2018) and SQuAD (Rajpurkar et al., 2016) language tasks. The compression methods we couple with "training flat" include: 1) unstructured IMP to find winning lottery tickets (Frankle and Carbin, 2019), referring to Chen et al. (2020)'s implementation and reported results for making direct comparisons; 2) Xia et al. (2022)'s structured pruning method with a distillation objective; and 3) off-the-shelf post-training quantization, which we compare with reported results from Zafrir et al. (2019)'s quantization-aware training method.

［#69］
Flatness and generalization Prior work has investigated the connection between flat minima and generalization (i.e., the gap between training accuracy and holdout set accuracy), starting with Hochreiter and Schmidhuber (1997).

［#70］
Subsequent work has continued on this front, exploring notions of sharpness and their predictiveness of generalization under different conditions (Bisla et al., 2022), as well as empirical evaluations of flatness-inducing methods and their effects on generalization. Jiang* et al. (2020) find that sharpness is empirically predictive of generalization, including in particular a perturbation magnitude-aware metric very similar to the $\epsilon$-sharpness metric introduced in (Keskar et al., 2017) and used in our paper. In this work, however, separate from generalization, we primarily focus on investigating the underexplored relationship between flatness and compression.

［#71］
Flatness and compression Previous work has mentioned flatness in the context of pruning. In fact, Hochreiter and Schmidhuber (1997)'s orig-

［#71］
inal "Flat Minimum Search" algorithm is explic- itly designed to prune units, weights, and input lines as a mechanism for finding flat minima. Le- Cun et al. (1989) also propose pruning unimportant weights as a way to obtain improved generalization, although without any notions of flatness.

［#72］
Since these earlier works, the deep learning land- scape has changed such that effective model com- pression itself is now often a goal; model efficiency in terms of size and latency is a common priority.

［#73］
To the best of our knowledge, we are the first to directly relate loss landscape flatness with model compressibility. We view our results as comple- mentary to the concurrent work by Paul et al.(2022), who study properties of winning tickets found during iterative magnitude pruning and find that IMP itself preferentially prunes parameters ly-ing along flatter directions in the loss landscape; they also theorize that flatter landscapes allow for more aggressive pruning. We note that the opti- mizer, data, and model architectures they use are different from ours. While Paul et al. (2022) use Stochastic Gradient Descent for image classifica- tion tasks on ResNet architectures, we use Adam as a base optimizer for text classification and question answering tasks with BERT architectures. Nonethe-less, to the extent that findings from both papers can generalize to other settings, Paul et al. (2022) lay theoretical groundwork which supports our ex- plicit suggestion to "train flat" as a strategy for inducing greater compressibility in neural models.

## 5 Discussion

［#74］
We show that in general, SAM helps models retain higher accuracy on a variety of language tasks at higher sparsity levels. This holds true in multiple unstructured iterative magnitude pruning settings, as well as in a structured pruning setting with a distillation objective. Moreover, our additional ex- periments and analyses point to SAM-learned struc- tures playing an important role in compressibility, as well as transferring well across tasks.

### 5.1 Future work

［#75］
**Beyond SAM and SWA (Q4)** In this paper, we explore (Foret et al., 2021)'s Sharpness-Aware- Minimization procedure specifically as a method for directly reaching flat minima and conduct addi- tional comparisons with stochastic weight averag-ing (Izmailov et al., 2018). However, other meth-ods such as entropy SGD (Chaudhari et al., 2017) and label noise SGD (Damian et al., 2021) have also been shown to encourage convergence to flat minima. Further exploration would provide clarity on the role of flatness in general in model com- pressibility versus properties specific to SAM and SWA.

［#76］
**The role of pre-training (Q4)** The BERT mod- els we fine-tune in this work are pre-trained on various self-supervised auxiliary objectives with the goal of learning useful general representations of the English language. Recent work (Mehta et al.,2021) has found that, empirically, pre-training is associated with convergence to flatter minima than obtained by training on the same task to the same accuracy from a random initialization. Subse- quent work could compare compressibility of pre- trained models versus models trained from random initialization, as well as investigate the potential for further improving compressibility through flat pre-training. Inducing flatness during pre-training would facilitate further experimentation with end- task-agnostic knowledge distillation.

［#77］
**Other evaluations (Q3)** In general, we evaluate model performance in terms of task-specific met- rics (on development/test splits) throughout this work. However, since compression is associated with negative consequences for model behavior not captured by task-specific accuracy (Hooker et al.,2020; Liebenwein et al., 2021), there is a particular need for work to understand and influence these qualities. Ribeiro et al. (2020) propose behavioral testing of NLP models to evaluate specific capabil- ities such as robustness to typos and simple cofer- ence resolution. Xu et al. (2021) propose measures of probability and label loyalty and robustness to input perturbations for evaluating compressed mod- els beyond preserved accuracy. We briefly discuss preliminary observations of model behaviors with respect to Ribeiro et al. (2020)'s Checklist items in §A.13 in Appendix, but we emphasize that more work is needed to understand behaviors and behav- ior shifts of vanilla Adam and SAM models before and throughout pruning.

## Limitations

［#78］
Many of the limitations of our work have to do with its computational requirements. First, the standard implementation of Sharpness-Aware Minimization that we use incurs significant computational over- head, so further investigation into adaptive (Kwon

［#78］
et al., 2021) and more efficient (Du et al., 2022b,a) variations of SAM is warranted before consider- ing adoption of our methods in practice. Mean- while, we control for the number of training steps and sparsity level of our models without regard to wall-clock time in our experiments, but fine- tuning $\text{BERT}_{base}$ with standard SAM typically re- sults in 1.5x-2x slower optimization steps. In gen- eral, many of our approaches are not strategies that can simply be applied off the shelf for prac- tical benefits. In particular, current hardware and frameworks do not typically support reliable and proportionate efficiency gains from quantization to arbitrary precision and unstructured magnitude pruning. Moreover, the computation and storage requirements for our iterative magnitude pruning scheme are tenfold compared to the typical single fine-tuning performed on a pre-trained language model, due to the ten iterations performed and checkpoints saved. We benefited from access to a large compute cluster with dozens of GPUs, includ- ing A6000, a100, v100, RTX3090, RTX8000, and 2080Ti GPUs. We estimate having used at least 1000 GPU hours across experiments for this work, which inherently limits the full reproducibility of our results in limited compute scenarios.

［#79］
Additionally, although we focus on optimizing for flatness directly in our experimentation with SAM (Foret et al., 2021), other methods, such as weight averaging (Izmailov et al., 2018) (which we explore in less detail), entropy SGD (Chaudhari et al., 2017), and label noise SGD (Damian et al., 2021), have also been shown to encourage conver- gence to flat minima. Without explicitly exploring compressibility in models that have reached flat minima in alternative ways, we refrain from mak- ing strong claims about flat minima in general in this work.

［#80］
Furthermore, the tasks we train our models on are limited to sentence classification and question answering tasks, all in only the English language. We evaluate our methods using only BERT variants: pre-trained language models trained with a token masking objective.

［#81］
Finally, although our measures of end-task accu- racy and degree of compression are standard and useful for evaluating and comparing compressed models, they do not provide a complete picture of model behaviors and capabilities. Other desirable characteristics in compressed models (and models in general) include but are not limited to robustness to distribution shift, stability against catastrophic forgetting, and fairness in performance across de- mographic groups..

## Ethics Statement
［#82］
We reiterate that we are not proposing that our strategies or models be adopted off the shelf as is. This is especially true because our work does not include rigorous analysis of our compressed models' properties and behaviors outside of task accuracy, resilience to compression, and transfer- ability to other tasks. Detailed study of proper- ties such as long-tail performance, robustness to data distribution shifts, and fairness in performance across demographic groups, for example, which have important real-world implications, is outside of the scope of our current work. However, pre- liminary evaluations on certain model behaviors and properties suggests that many of our models which achieve high end-task performance are vul- nerable to simple perturbations in data and lack basic desirable linguistic capabilities (although not necessarily more so than is "typical" of language models (Ribeiro et al., 2020; Xu et al., 2021)).

## Acknowledgements
［#83］
We are grateful to our anonymous reviewers, both of whom provided thoughtful and helpful feed- back on extremely short notice. We would like to thank COMEDY (COhorts of Maarten Sap, Emma Strubell, Daniel Fried, and Yonatan Bisk) lab mem- bers for sharing insights and intuitions during ini- tial discussions; Nupoor Gandhi, Jared Fernan- dez, Jeremiah Milbauer, Zhisong Zhang, and Josh Zhanson also gave constructive feedback on drafts and figures. We would like to acknowledge CMU Workhorse and TIR groups for providing compute resources for this work. Outside of CMU, we are appreciative of Mengzhou Xia for helping us repro- duce structured pruning experiments using CoFi. This project is funded in part by DSO National Laboratories.

## References














































































## A Appendix

### A.1 Extended Related Work

［#84］
Model compression in NLP Approaches for model compression aim to replicate the end-task performance of a large, accurate model while requiring fewer parameters and floating-point operations, and many approaches for model compression have been successfully applied to tasks in NLP. Specific model compression techniques include *pruning*, where individual model parameters (unstructured pruning) or entire weight matrices (structured pruning) are removed entirely (LeCun et al., 1989; Blalock et al., 2020; Sanh et al., 2020; Lagunas et al., 2021), *quantization*, where model weights, activations, gradients, and/or add-multiply accumulators are reduced in precision from 32-bit floating point representations to floating or fixed-point representations as low as one or two bits (Gray and Neuhoff, 1998; Vanhoucke et al., 2011; Gholami et al., 2021), and *knowledge distillation* where a smaller model is trained to replicate the predictions, and often intermediate embedded representations, of a larger model (Buciluǎ et al., 2006; Hinton et al., 2014; Sanh et al., 2019; Jiao et al., 2020).

［#85］
Unstructured pruning can achieve some of the highest sparsity levels using various criteria and schedules for determining which parameters to prune (Frankle and Carbin, 2019; Chen et al., 2020; Sanh et al., 2020; Guo et al., 2021), though sparsity patterns resulting from unstructured pruning often do not result in latency reduction on modern accelerator hardware. Work in structured pruning has explored removing entire parameter matrices such as self-attention heads, hidden units, and entire layers (Michel et al., 2019; Lagunas et al., 2021; Xia et al., 2022), with basic underlying hardware constraints in mind. Pruning is often combined with a distillation objective, which provides complementary gains, likely by reducing complexity of the dataset (Zhou et al., 2020).

［#86］
Distillation is a prominent, practical method for compression that is widely used in NLP. Work on distillation in NLP has focused largely on the task-agnostic setting of compressing general-purpose pre-trained models such as BERT (Sanh et al., 2019; Sun et al., 2019, 2020) but task-specific distillation has also been reported to work well (Jiao et al., 2020).

［#87］
Approaches for model quantization can be categorized into post-training quantization, where general-purpose models are quantized at test-time (Jacob et al., 2018; Bhandare et al., 2019; Kim et al., 2021), and quantization-aware training, where models incorporate simulated quantization error during training in order to learn more quantizable parameters (Zafrir et al., 2019; Bai et al., 2021). Quantization-aware training tends to lead to higher accuracy quantized inference, but post-training quantization can be applied on-the-fly to any model at inference time.

［#88］
In this work we experiment with a variety of compression methods including structured and unstructured pruning (Xia et al., 2022; Chen et al., 2020), and out-of-the-box INT8 post-training dynamic quantization,⁷ highlighting both practical (structured pruning, quantization) and theoretical (unstructured pruning) findings. While we do not experiment with distillation directly, our chosen structured pruning method also incorporates a distillation objective.

［#89］
Learning compressible models Most closely related to our work are methods for learning compressible models, and the study of what makes models more compressible. Quantization-aware training is an example of such training for compressibility, and training for sparsity using $\ell_0$ regularization (Louizos et al., 2018) is a parallel method for pruning.

［#90］
Learning sparse models from scratch has proven difficult, despite the fact that deep neural networks are vastly over-parameterized. Frankle and Carbin (2019) formalized the *Lottery Ticket Hypothesis*, which posits that large, overparameterized neural network models contain sparse subnetworks, or *winning tickets*, that can be trained from scratch (or close to it; see Frankle et al. (2020)) to match the end-task performance of the full model. This influential work has spurred much research into better understanding neural network models, including pre-trained language models, from the perspective of winning tickets (Chen et al., 2020; Renda et al., 2020b; Diffenderfer and Kailkhura, 2021) and how to leverage winning tickets to perform better model compression (Chen et al., 2021; Liang et al., 2021). Li et al. (2020) showed that larger pre-trained language models are more compressible than smaller ones, which they hypothesize is related to larger models being more likely to contain winning tickets.

---
［#88］
⁷https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html

［#91］
Flat minima in neural networks. Hochreiter and Schmidhuber (1997) were among the first to discuss the relationship between flat basins in the loss landscape and generalization in neural networks, defining a flat minimum as "a large connected region in weight space where the error remains approximately constant." We use the $\epsilon$-sharpness definition of Keskar et al. (2017) which defines sharpness as maximum loss within a neighborhood bounded by $\epsilon$. Others have used Hessian-based measures to identify high minima with high curvature (Chaudhari et al., 2017). Note that care is needed when making inferences based on current measurements of sharpness, which is an active area of research; it has been shown that flat minima defined in this way can be rescaled to sharp minima (and still generalize) (Dinh et al., 2017).

［#92］
Most previous results related to flat minima have focused on generalizability (Hao et al., 2019; Neyshabur et al., 2020). A common explanation for the good generalization of models converged to flat minima is that flatter models are less complex (Wu et al., 2017).

［#93］
Flat minima may be particularly of interest in NLP, where a pretrain-then-finetune paradigm is often employed to leverage general representations learned from an extensive pre-training process during a much shorter fine-tuning process on a more specific end task. Indeed, pre-training provides a flat prior that can provide benefits in the contexts of lifelong learning (Mehta et al., 2021) and generalization (Hao et al., 2019; Bahri et al., 2022).

### A.2 Flat Minima: Additional details

［#94］
Stochastic Weight Averaging (SWA). We consider last $50\%$ of the model checkpoints for equal averaging. Specifically, for RTE, MRPC, STS-B, and CoLA we fine-tune for 10 epochs and average 5 checkpoints from epochs 6 to 10. For SST-2, QNLI, QQP, and MNLI we fine-tune for 3 epochs and retain checkpoints after every 0.5 epochs and equal average checkpoints after 2, 2.5, 3 epochs. Izmailov et al. (2018) suggests modified learning rate scheduler like cyclical or constant so that towards the later stage of training, the underlying optimizer explores diverse solutions and average over them would lead to flatter solution. To simulate this behavior, for all our SWA experiments, we set our initial learning rate to a high value of $8e-5$ and linearly decay it to 0 with no warmup steps.

［#95］
Evaluating sharpness. Keskar et al. (2017) propose a computationally feasible metric for measuring the sharpness of a minimizer over an $\epsilon$-neighborhood in the loss landscape. We report sharpness metrics (lower value means flatter low loss region) in Tables 3 and 4. Overall we see that SAM and SWA optimized models have significantly smaller sharpness values (or flatter low loss regions) as compared to vanilla Adam optimizer, thus, providing convincing evidence that these methods indeed find flatter solutions.

［#96］
Loss contours. In Figure 6, we visualize contour plots of the loss landscape for the QQP task to qualitatively compare the sharpness of the solutions that Adam- and SAM-optimized $\text{BERT}_{base}$ models find. We observe that the SAM-optimized model sits in a noticeably flatter, wider basin than the Adam-optimized model when both are fitted with their respective classifier heads. These analyses verify that SAM indeed leads to flatter minima in comparison to Adam.

### A.3 Non-Iterative Unstructured Magnitude Pruning

［#97］
In a preliminary analysis, we subject full-size fine-tuned $\text{BERT}_{base}$ models to one-shot unstructured magnitude pruning and evaluate on the same task *without any subsequent training*. Figure 7 displays development set accuracies at sparsity levels of increments of $5\%$, up to $60\%$ of prunable parameters masked to 0. Accuracy values are plotted at averages over $n=3$ seeds for each of the sparsity levels and GLUE tasks displayed (SST-2, QNLI, MRPC, RTE). Interestingly, even under this non-iterative pruning setting, performance does not drop off noticeably in either SAM or Adam-optimized models until at least around $30\%$ sparsity for the GLUE tasks displayed. Similarly to all iterative pruning settings we explore, models optimized with SAM retain full model size accuracies at higher sparsity levels than their Adam-optimized counterparts. The SAM-optimized RTE models at $35-40\%$ sparsity have *higher* accuracy than the full-sized uncompressed model, reminiscent of the pattern we observe in Figure 2a, albeit at lower sparsity levels.

### A.4 Iterative Magnitude Pruning Reproducibility and Hyperparameters

［#98］
Following Chen et al. (2020), we use a maximum sequence length of 128, batch size of 32, learning rate to $2e-5$, and linear decay of learning rate from

［#99］
<table>
  <thead>
    <tr>
      <th>epsilon ($\epsilon$)</th>
      <th>Dataset</th>
      <th>MNLI</th>
      <th>QQP</th>
      <th>STS-B</th>
      <th>QNLI</th>
      <th>MRPC</th>
      <th>RTE</th>
      <th>SST-2</th>
      <th>CoLA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">$5 \times 10^{-3}$</td>
      <td>Adam</td>
      <td>$28.3_{3.6}$</td>
      <td>$34.7_{5.0}$</td>
      <td>$160.9_{34.3}$</td>
      <td>$30.1_{6.0}$</td>
      <td>$50.8_{24.8}$</td>
      <td>$49.0_{5.6}$</td>
      <td>$29.9_{10.3}$</td>
      <td>$38.3_{3.7}$</td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>$14.2_{0.9}$</td>
      <td>$9.3_{1.7}$</td>
      <td>$45.8_{1.3}$</td>
      <td>$17.8_{3.4}$</td>
      <td>$40.4_{5.3}$</td>
      <td>$28.4_{8.7}$</td>
      <td>$13.7_{2.3}$</td>
      <td>$29.4_{9.8}$</td>
    </tr>
    <tr>
      <td rowspan="2">$1 \times 10^{-3}$</td>
      <td>Adam</td>
      <td>$5.0_{0.6}$</td>
      <td>$6.5_{0.9}$</td>
      <td>$11.8_{3.1}$</td>
      <td>$6.6_{2.5}$</td>
      <td>$6.1_{1.0}$</td>
      <td>$11.9_{3.0}$</td>
      <td>$4.5_{0.6}$</td>
      <td>$9.5_{2.2}$</td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>$2.6_{0.2}$</td>
      <td>$1.9_{0.3}$</td>
      <td>$4.5_{0.4}$</td>
      <td>$3.5_{1.3}$</td>
      <td>$7.0_{0.7}$</td>
      <td>$4.3_{2.4}$</td>
      <td>$2.2_{0.1}$</td>
      <td>$6.8_{2.8}$</td>
    </tr>
    <tr>
      <td rowspan="2">$5 \times 10^{-4}$</td>
      <td>Adam</td>
      <td>$2.3_{0.2}$</td>
      <td>$3.4_{0.2}$</td>
      <td>$4.3_{1.2}$</td>
      <td>$3.2_{1.5}$</td>
      <td>$2.8_{0.3}$</td>
      <td>$6.5_{2.1}$</td>
      <td>$2.3_{0.4}$</td>
      <td>$5.6_{2.0}$</td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>$1.3_{0.1}$</td>
      <td>$0.9_{0.1}$</td>
      <td>$1.9_{0.3}$</td>
      <td>$1.5_{0.3}$</td>
      <td>$3.2_{0.6}$</td>
      <td>$2.1_{1.1}$</td>
      <td>$1.0_{0.1}$</td>
      <td>$3.4_{1.4}$</td>
    </tr>
  </tbody>
</table>

［#100］
Table 3: Evaluating sharpness metric for vanilla Adam and SAM optimized models at full size (i.e., no compression). We observe that SAM optimized models have significantly lower sharpness values (lower corresponds to flatter minima) compared to vanilla Adam. These results provide quantitative evidence that SAM indeed leads to flatter loss basins.

［#101］
<table>
  <thead>
    <tr>
      <th>epsilon ($\epsilon$)</th>
      <th>Dataset</th>
      <td>MNLI</td>
      <td>QQP</td>
      <td>STS-B</td>
      <td>QNLI</td>
      <td>MRPC</td>
      <td>RTE</td>
      <td>SST-2</td>
      <td>CoLA</td>
    </tr>
    <tr>
      <th></th>
      <th>Sparsity</th>
      <td>70%</td>
      <td>90%</td>
      <td>50%</td>
      <td>70%</td>
      <td>50%</td>
      <td>60%</td>
      <td>60%</td>
      <td>50%</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">$5 \times 10^{-3}$</td>
      <td>Adam</td>
      <td>$56.4_{1.0}$</td>
      <td>$137.9_{40.4}$</td>
      <td>$232.5_{30.0}$</td>
      <td>$23.8_{3.9}$</td>
      <td>$42.3_{26.9}$</td>
      <td>$65.7_{8.8}$</td>
      <td>$85.8_{33.8}$</td>
      <td>$33.8_{2.7}$</td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>$42.4_{7.9}$</td>
      <td>$65.3_{1.9}$</td>
      <td>$184.0_{7.3}$</td>
      <td>$17.2_{6.0}$</td>
      <td>$47.3_{18.0}$</td>
      <td>$50.8_{6.1}$</td>
      <td>$26.9_{7.7}$</td>
      <td>$23.9_{4.5}$</td>
    </tr>
    <tr>
      <td>SWA</td>
      <td>$37.3_{9.9}$</td>
      <td>$34.7_{1.2}$</td>
      <td>$81.3_{22.3}$</td>
      <td>$33.5_{7.5}$</td>
      <td>$31.6_{6.6}$</td>
      <td>$42.1_{12.3}$</td>
      <td>$23.7_{10.3}$</td>
      <td>$26.5_{9.5}$</td>
    </tr>
    <tr>
      <td rowspan="3">$1 \times 10^{-3}$</td>
      <td>Adam</td>
      <td>$6.6_{0.9}$</td>
      <td>$5.8_{1.0}$</td>
      <td>$20.2_{6.1}$</td>
      <td>$5.9_{2.2}$</td>
      <td>$8.6_{2.2}$</td>
      <td>$16.4_{4.0}$</td>
      <td>$5.8_{0.7}$</td>
      <td>$6.2_{1.4}$</td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>$3.1_{0.4}$</td>
      <td>$5.4_{0.6}$</td>
      <td>$13.8_{1.2}$</td>
      <td>$3.2_{1.1}$</td>
      <td>$8.3_{1.6}$</td>
      <td>$12.0_{3.8}$</td>
      <td>$4.1_{0.8}$</td>
      <td>$4.2_{0.3}$</td>
    </tr>
    <tr>
      <td>SWA</td>
      <td>$10.7_{1.6}$</td>
      <td>$5.2_{0.6}$</td>
      <td>$5.4_{1.1}$</td>
      <td>$8.9_{0.2}$</td>
      <td>$12.4_{0.8}$</td>
      <td>$11.8_{4.1}$</td>
      <td>$5.2_{0.7}$</td>
      <td>$7.9_{2.2}$</td>
    </tr>
    <tr>
      <td rowspan="3">$5 \times 10^{-4}$</td>
      <td>Adam</td>
      <td>$3.3_{0.2}$</td>
      <td>$2.3_{0.3}$</td>
      <td>$7.1_{2.5}$</td>
      <td>$2.1_{0.6}$</td>
      <td>$5.8_{0.5}$</td>
      <td>$7.9_{1.1}$</td>
      <td>$3.6_{0.9}$</td>
      <td>$3.9_{0.6}$</td>
    </tr>
    <tr>
      <td>SAM</td>
      <td>$1.1_{0.3}$</td>
      <td>$2.0_{0.6}$</td>
      <td>$5.1_{0.6}$</td>
      <td>$1.2_{0.0}$</td>
      <td>$3.6_{0.6}$</td>
      <td>$5.9_{2.0}$</td>
      <td>$1.9_{0.4}$</td>
      <td>$2.3_{0.4}$</td>
    </tr>
    <tr>
      <td>SWA</td>
      <td>$4.4_{0.5}$</td>
      <td>$2.9_{0.0}$</td>
      <td>$2.2_{0.4}$</td>
      <td>$4.7_{0.1}$</td>
      <td>$5.9_{0.7}$</td>
      <td>$6.0_{2.0}$</td>
      <td>$2.5_{0.5}$</td>
      <td>$3.4_{0.2}$</td>
    </tr>
  </tbody>
</table>

［#102］
Table 4: Evaluating sharpness metric for vanilla Adam, SAM and SWA optimized models at Chen et al. (2020)'s reference sparsities. In general we observe that SAM and SWA optimized models have lower sharpness values (lower corresponds to flatter minima) compared to vanilla Adam at various sparsity levels across different tasks (all results are averaged over 3 runs).

［#103］
![](./images/867770500502258296_6.jpg)

［#104］
Figure 6: Visualization of loss contours for QQP on $\text{BERT}_{base}$ models finetuned on the task using Adam and SAM optimizers ($w_{adam}$ and $w_{sam}$), as well as the pre-trained BERT base initialization ($w_{init}$). On the left, all models are fitted with the linear classifier head originally trained with $w_{adam}$, while on the right side models are fitted with the linear classifier head originally trained with $w_{sam}$. The SAM-optimized model sits in a noticeably flatter, wider basin than the Adam-optimized model when both are fitted with their respective classifier heads. These results provide qualitative evidence that SAM indeed leads to flatter loss basins.

［#105］
initial value to zero with no warmup period. For tasks with smaller datasets (RTE, MRPC, CoLA, STS-B), we fine-tune models for 10 epochs, evaluate them after every epoch and retain the checkpoint yielding best task-specific performance on the hold-out validation set (whereas Chen et al. (2020) finetune for only 3 for all tasks). For tasks with comparatively larger datasets (MNLI, QQP, QNLI, SST-2), we fine-tune models for 3 epochs. We set Adam's weight decay to $\epsilon=0$ in order to remove the potential confound of regularization on models' amenability to magnitude pruning. This

［#106］
![](./images/867770500502258296_7.jpg)

［#107］
Figure 7: SAM- and Adam-optimized models evalu-
ated directly after pruning a proportion of parameters
from the full-sized model. Models fine-tuned with
SAM hold up better to non-iterative magnitude pruning
as well.

［#108］
differs from Chen et al. (2020)'s $\epsilon = 1 \times 10^{-8}$, but
we observed that our differences do not systemati-
cally affect the trends originally reported other than
to improve full-size model performance and allow
for a fairer comparison between SAM and Adam
optimizers. In particular, training for only 3 epochs
on the smaller tasks with our SAM optimizer does
not allow models to converge.

### A.5 Comparison with BERT Lottery Ticket Hypothesis Numbers
［#109］
We present development set numbers on GLUE
and SQuAD tasks in Table 5. For comparison,
we include the reference (Ref) metrics reported by
Chen et al. (2020) at their reported winning ticket
sparsity levels.

### A.6 Additional Individual Task Plots for $\text{BERT}_{base}$ IMP
［#110］
In Figure 8, we present the SQuAD plot comparing
SAM- and Adam-optimized $\text{BERT}_{base}$ models in
the unstructured IMP setting in Figure 2, as well
as versions of the GLUE plots in including SWA
performance.

［#111］
Plots for $\text{BERT}_{base}$ trained on GLUE and
SQuAD tasks with unstructured standard pruning
are shown in Figure 9.

### A.7 Is SAM just implicitly doing $\ell_1$ regularization?
［#112］
No. It is clear that $\ell_1$ regularization induces sparsity
in a different way compared to SAM. Although in
some cases $\ell_1$ regularization can help a model reach
higher accuracies at certain sparsity levels, we ob-
serve that simply optimizing with Adam through-
out an iterative pruning process does not allow the
model to reach SAM-optimized models' compres-
sion performance. Moreover, $\ell_1$ regularization can
actually hurt compression performance in some
cases.

［#113］
Further investigation is needed to understand
the specific mechanisms allowing SAM to induce
greater compresseibility in models.

### A.8 Detailed Structure vs Optimization Results
［#114］
Table 6 contains numbers presented in Figure 3
from §3.4.2.

［#115］
Figure 12 presents the same information as Fig-
ure 3 in an alternative view, featuring colored bars
representing ticket performance over different opti-
mizers.

### A.9 Ticket Transfer Experiments: Comparing Optimization Over Given Tickets
［#116］
Figure 4 in §3.4.3 allows us to evaluate SAM vs.
vanilla Adam ticket transferability across GLUE
tasks. Figure 13, on the other hand, allows us to
evaluate SAM vs. vanilla Adam optimization over
given tickets for different GLUE tasks.

### A.10 Structured Pruning Reproducibility and Hyperparameters
［#117］
Following Xia et al. (2022), we train for 20 epochs
each in the pruning and final fine-tuning stages. We
use a sparsity epsilon value of 0.01, meaning that a
model can be accepted if its actual sparsity level is
within $1\%$ of the target sparsity level of $95\%$.

［#118］
We pick hyperparameters based on a grid search
over Xia et al. (2022)'s baseline implementation
(with their Adam-optimized teacher models), re-
ported in Table 7.

［#119］
For each task, we use the optimal $\lambda$ and final
fine-tuning learning rates found via grid search
using Xia et al. (2022)'s implementation, includ-
ing configurations for finetuning teacher models
(which used normal Adam optimizers). There are
small discrepancies between the reference metric
values reported and the values we were able to re-
produce, possibly due to variation across random
seeds. However, the relative performance of hy-
perparameter settings seems to be fairly consistent
across random seeds.

［#120］
<table>
 <tr>
  <td colspan="2">
   Dataset
  </td>
  <td>
   MNLI
  </td>
  <td>
   QQP
  </td>
  <td>
   STS-B
  </td>
  <td>
   QNLI
  </td>
  <td>
   MRPC
  </td>
  <td>
   RTE
  </td>
  <td>
   SST-2
  </td>
  <td>
   CoLA
  </td>
  <td>
   SQuAD
  </td>
 </tr>
 <tr>
  <td colspan="2">
   Sparsity
  </td>
  <td>
   70%
  </td>
  <td>
   90%
  </td>
  <td>
   50%
  </td>
  <td>
   70%
  </td>
  <td>
   50%
  </td>
  <td>
   60%
  </td>
  <td>
   60%
  </td>
  <td>
   50%
  </td>
  <td>
   40%
  </td>
 </tr>
 <tr>
  <td colspan="2">
   Metric
  </td>
  <td>
   Matched acc.
  </td>
  <td>
   Acc.
  </td>
  <td>
   Pearson Cor.
  </td>
  <td>
   Acc.
  </td>
  <td>
   Acc.
  </td>
  <td>
   Acc.
  </td>
  <td>
   Acc.
  </td>
  <td>
   Matthew’s Cor.
  </td>
  <td>
   F1
  </td>
 </tr>
 <tr>
  <td>
   Full
  </td>
   <br>
   FT
  </td>
  <td>
   Ref
  </td>
  <td>
   82.40.5
  </td>
  <td>
   90.20.5
  </td>
  <td>
   88.40.3
  </td>
  <td>
   89.11.0
  </td>
  <td>
   85.20.1
  </td>
  <td>
   66.23.6
  </td>
  <td>
   92.10.1
  </td>
  <td>
   54.50.4
  </td>
  <td>
   88.10.6
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   Adam
  </td>
  <td>
   84.70.4
  </td>
  <td>
   91.00.1
  </td>
  <td>
   89.00.1
  </td>
  <td>
   91.50.1
  </td>
  <td>
   85.51.7
  </td>
  <td>
   67.61.5
  </td>
  <td>
   92.70.1
  </td>
  <td>
   58.10.4
  </td>
  <td>
   88.50.2
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   SAM
  </td>
  <td>
   85.30.2
  </td>
  <td>
   91.10.1
  </td>
  <td>
   89.40.2
  </td>
  <td>
   91.30.6
  </td>
  <td>
   87.00.7
  </td>
  <td>
   67.00.8
  </td>
  <td>
   93.10.6
  </td>
  <td>
   59.50.4
  </td>
  <td>
   89.20.1
  </td>
 </tr>
 <tr>
  <td colspan="2">
   IMP
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
  <td>
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   Ref
  </td>
  <td>
   82.60.2
  </td>
  <td>
   90.00.2
  </td>
  <td>
   88.20.2
  </td>
  <td>
   88.90.4
  </td>
  <td>
   84.90.4
  </td>
  <td>
   66.02.4
  </td>
  <td>
   91.90.5
  </td>
  <td>
   53.80.9
  </td>
  <td>
   87.70.5
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   Adam
  </td>
  <td>
   82.60.3
  </td>
  <td>
   85.00.2
  </td>
  <td>
   88.30.2
  </td>
  <td>
   88.60.1
  </td>
  <td>
   84.20.1
  </td>
  <td>
   63.70.9
  </td>
  <td>
   91.70.4
  </td>
  <td>
   56.41.4
  </td>
  <td>
   86.90.3
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   SAM
  </td>
  <td>
   83.50.1
  </td>
  <td>
   84.90.1
  </td>
  <td>
   88.90.2
  </td>
  <td>
   89.60.1
  </td>
  <td>
   85.81.0
  </td>
  <td>
   71.81.7
  </td>
  <td>
   93.20.4
  </td>
  <td>
   56.10.9
  </td>
  <td>
   87.80.2
  </td>
 </tr>
 <tr>
  <td colspan="2">
   $\pm$10% Adam
  </td>
  <td>
   82.10.3
  </td>
  <td>
   87.20.3
  </td>
  <td>
   88.30.2
  </td>
  <td>
   88.00.2
  </td>
  <td>
   83.70.4
  </td>
  <td>
   64.20.2
  </td>
  <td>
   91.50.5
  </td>
  <td>
   55.00.4
  </td>
  <td>
   86.80.4
  </td>
 </tr>
 <tr>
  <td colspan="2">
   SAM
  </td>
  <td>
   83.10.1
  </td>
  <td>
   87.40.4
  </td>
  <td>
   88.80.2
  </td>
  <td>
   89.10.2
  </td>
  <td>
   85.50.5
  </td>
  <td>
   68.90.4
  </td>
  <td>
   92.90.3
  </td>
  <td>
   56.10.2
  </td>
  <td>
   88.30.5
  </td>
 </tr>
 <tr>
  <td>
   Std
  </td>
  <td>
   Ref
  </td>
  <td>
   82.1
  </td>
  <td>
   90.0
  </td>
  <td>
   88.5
  </td>
  <td>
   89.9
  </td>
  <td>
   85.8
  </td>
  <td>
   63.0
  </td>
  <td>
   90.0
  </td>
  <td>
   52.0
  </td>
  <td>
   87.1
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   Adam
  </td>
  <td>
   82.7
  </td>
  <td>
   88.1
  </td>
  <td>
   89.2
  </td>
  <td>
   89.5
  </td>
  <td>
   85.5
  </td>
  <td>
   65.3
  </td>
  <td>
   91.1
  </td>
  <td>
   54.2
  </td>
  <td>
   86.30.2
  </td>
 </tr>
 <tr>
  <td>
  </td>
  <td>
   SAM
  </td>
  <td>
   83.3
  </td>
  <td>
   89.6
  </td>
  <td>
   89.6
  </td>
  <td>
   90.0
  </td>
  <td>
   85.3
  </td>
  <td>
   68.2
  </td>
  <td>
   92.1
  </td>
  <td>
   56.8
  </td>
  <td>
   87.10.2
  </td>
 </tr>
</table>

［#121］
Table 5: We report task metrics on the development set at Chen et al. (2020)’s reference sparsities for Adam and SAM-optimized BERT-base models in their (1) Iterative Magnitude Pruning (IMP) and (2) Std pruning settings. We include Chen et al. (2020)’s reference (Ref) metrics in addition to our reported metrics (Adam, SAM). When applicable, we report mean and standard deviation calculated over 3 random seeds.

［#122］
<table>
 <tbody>
  <tr>
   <th>
    Dataset
   </th>
   <th>
    Ticket
   </th>
   <th>
    Optim.
   </th>
   <td>
    Accuracy
   </td>
  </tr>
  <tr>
   <th>
    RTE
   </th>
   <th>
    Random
   </th>
   <th>
    Adam
   </th>
   <td>
    54.91.3
   </td>
  </tr>
  <tr>
   <th>
    (60%)
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    55.41.6
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    Adam
   </th>
   <th>
    Adam
   </th>
   <td>
    63.70.9
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    61.72.4
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    SAM
   </th>
   <th>
    Adam
   </th>
   <td>
    70.21.9
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    71.81.7
   </td>
  </tr>
  <tr>
   <th>
    MRPC
   </th>
   <th>
    Random
   </th>
   <th>
    Adam
   </th>
   <td>
    70.80.8
   </td>
  </tr>
  <tr>
   <th>
    (50%)
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    70.10.2
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    Adam
   </th>
   <th>
    Adam
   </th>
   <td>
    84.20.1
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    85.30.5
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    SAM
   </th>
   <th>
    Adam
   </th>
   <td>
    85.31.3
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    85.70.8
   </td>
  </tr>
  <tr>
   <th>
    SST-2
   </th>
   <th>
    Random
   </th>
   <th>
    Adam
   </th>
   <td>
    82.80.2
   </td>
  </tr>
  <tr>
   <th>
    (60%)
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    83.30.8
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    Adam
   </th>
   <th>
    Adam
   </th>
   <td>
    91.90.3
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    92.70.1
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    SAM
   </th>
   <th>
    Adam
   </th>
   <td>
    92.40.1
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    92.90.2
   </td>
  </tr>
  <tr>
   <th>
    QNLI
   </th>
   <th>
    Random
   </th>
   <th>
    Adam
   </th>
   <td>
    61.70.3
   </td>
  </tr>
  <tr>
   <th>
    (70%)
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    61.50.1
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    Adam
   </th>
   <th>
    Adam
   </th>
   <td>
    89.00.05
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    89.50.04
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
    SAM
   </th>
   <th>
    Adam
   </th>
   <td>
    89.10.1
   </td>
  </tr>
  <tr>
   <th>
   </th>
   <th>
   </th>
   <th>
    SAM
   </th>
   <td>
    89.60.2
   </td>
  </tr>
 </tbody>
</table>

［#123］
Table 6: For RTE, MRPC, SST-2, and QNLI at their reference sparsity values, we fine-tune using 1) SAM and 2) Adam optimizers from pre-trained BERT-base initializations using only the remaining weights based on a) a Random mask, b) an Adam-learned mask, and c) a SAM-learned mask.

［#124］
<table>
 <tbody>
  <tr>
   <td>
    Dataset
   </td>
   <td>
   </td>
   <td>
    $\lambda$
   </td>
   <td>
    FT-LR
   </td>
   <td>
    Teacher
   </td>
   <td>
    Pruned
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
    Acc.
   </td>
   <td>
    Acc.
   </td>
  </tr>
  <tr>
   <td>
    SST-2
   </td>
   <td>
    Ref.
   </td>
   <td>
    -
   </td>
   <td>
    -
   </td>
   <td>
    93.1
   </td>
   <td>
    90.6
   </td>
  </tr>
  <tr>
   <td>
    (67k)
   </td>
   <td>
    Reprod.
   </td>
   <td>
    0.9
   </td>
   <td>
    $3\mathrm{e}{- 5}$
   </td>
   <td>
    93.6
   </td>
   <td>
    90.6
   </td>
  </tr>
  <tr>
   <td>
    QNLI
   </td>
   <td>
    Ref.
   </td>
   <td>
    -
   </td>
   <td>
    -
   </td>
   <td>
    91.5
   </td>
   <td>
    86.1
   </td>
  </tr>
  <tr>
   <td>
    (105k)
   </td>
   <td>
    Reprod.
   </td>
   <td>
    0.9
   </td>
   <td>
    $3\mathrm{e}{- 5}$
   </td>
   <td>
    91.9
   </td>
   <td>
    86.5
   </td>
  </tr>
  <tr>
   <td>
    QQP
   </td>
   <td>
    Ref.
   </td>
   <td>
    -
   </td>
   <td>
    -
   </td>
   <td>
    91.2
   </td>
   <td>
    90.1
   </td>
  </tr>
  <tr>
   <td>
    (364k)
   </td>
   <td>
    Reprod.
   </td>
   <td>
    0.7
   </td>
   <td>
    $3\mathrm{e}{- 5}$
   </td>
   <td>
    91.3
   </td>
   <td>
    89.9
   </td>
  </tr>
  <tr>
   <td>
    MNLI
   </td>
   <td>
    Ref.
   </td>
   <td>
    -
   </td>
   <td>
    -
   </td>
   <td>
    84.8
   </td>
   <td>
    80.6
   </td>
  </tr>
  <tr>
   <td>
    (393k)
   </td>
   <td>
    Reprod.
   </td>
   <td>
    0.7
   </td>
   <td>
    $3\mathrm{e}{- 5}$
   </td>
   <td>
    85.2
   </td>
   <td>
    80.1
   </td>
  </tr>
 </tbody>
</table>

［#125］
Table 7: We report reproduced (Reprod.) and reference (Ref.) evaluation metrics at 95% sparsity and optimal values for $\lambda$ and fine-tuning learning rate on select tasks from Xia et al. (2022)’s structured pruning setting. We used the same hyperparameters as reported otherwise (distillation temperature $t = 2$, with 20 fine-tuning epochs after pruning, learning rate $= {2e - 5}$, and batch size$= 32$), and conducted our grid search over the same candidate values $\lambda \in {\{ 0.1,0.3,0.5\}}$ and FT-LR $\in {\{ {1e - 5},{{2e - 5},{3e - 5}}\}}$

## A.11 Detailed Quantization Results
［#126］
Table 8 contains the numbers used to generate Figure 5.

## A.12 Results for Other BERT Models
［#127］
We investigate SAM’s influence on amenability to sparsification in both $\text{BERT}_{large}$ and $\text{RoBERT}_{abase}$ models subject to iterative magnitude pruning (IMP). For consistency, we use the same hyperparameters as for the $\text{BERT}_{base}$ set of experiments ($\epsilon = 0$ weight decay; 10 (MRPC, RTE), 3 (SST-2, QNLI), or 2 (SQuAD) training epochs for each IMP iteration; linear learning rate decay schedules starting at $2e{-5}$ (GLUE) or $3e{-5}$ (SQuAD); batch size of 32 (GLUE) and 16 (SQuAD); maximum sequence length of 128 (GLUE) and 384 (SQuAD)). It is possible that a different set of hyperparameters

［#128］
Although we do not conduct an additional full grid search for our comparison of compressed models using Adam and SAM-optimized teacher models, we do find that the optimal final fine-tuning learning rates, which are much less computationally expensive to test, transfer to our experimental settings.

［#129］
![](./images/867770500502258296_8.jpg)
［#130］
(a) RTE

［#131］
![](./images/867770500502258296_9.jpg)
［#132］
(b) MRPC

［#133］
![](./images/867770500502258296_10.jpg)
［#134］
(c) CoLA

［#135］
![](./images/867770500502258296_11.jpg)
［#136］
(d) STS-B

［#137］
![](./images/867770500502258296_12.jpg)
［#138］
(e) SST-2

［#139］
![](./images/867770500502258296_13.jpg)
［#140］
(f) QNLI

［#141］
![](./images/867770500502258296_14.jpg)
［#142］
(g) QQP

［#143］
![](./images/867770500502258296_15.jpg)
［#144］
(h) MNLI

［#145］
![](./images/867770500502258296_16.jpg)
［#146］
(i) SQuAD

［#147］
Figure 8: Individual plots showing sparsity vs. task metrics (validation set) for GLUE throughout IMP. The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned (uncompressed) models.

［#148］
would be optimal for these different models, but we also tried different numbers of training epochs for the less stable smaller GLUE tasks (5 and 3), as well as Liu et al. (2019)'s learning rate of $1.5e-5$ for RoBERTa, and we generally found that simply matching hyperparameters from $\text{BERT}_{base}$ experiments worked well or better.

［#149］
Figures 14 and 15 show plots for $\text{BERT}_{large}$ and $\text{RoBERTa}_{base}$ models compressed with iterative magnitude pruning (IMP). We include our $\text{BERT}_{base}$ model results with Chen et al. (2020)'s reference sparsity levels and accuracy ranges (which are likewise for $\text{BERT}_{base}$) in the same plots for comparison.

［#150］
With the exception of $\text{RoBERTa}_{base}$ on MRPC, SAM-optimized models consistently fare better than Adam-optimized models in these other BERT variants as well. However, the comparison between BERT variants is more complex. While $\text{BERT}_{large}$ and $\text{RoBERTa}_{base}$ models generally achieve higher initial performance compared to $\text{BERT}_{base}$ and can maintain this higher performance at Chen et al. (2020)'s "winning ticket" sparsity levels (14b, 14d, 14e, 15b, 15e), the drop-off in performance does not always simply follow a parallel pattern. Initial higher performance tends to decrease more quickly with pruning than in $\text{BERT}_{base}$ models, such that $\text{BERT}_{large}$ and $\text{RoBERTa}_{base}$ performance sometimes falls to near (14c, 15c, 15d) or even below (14a, 15a) $\text{BERT}_{base}$ performance by the time they

［#151］
![](./images/867770500502258296_17.jpg)
［#152］
(a) RTE, Standard Pruning

［#153］
![](./images/867770500502258296_18.jpg)
［#154］
(b) MRPC, Standard Pruning

［#155］
![](./images/867770500502258296_19.jpg)
［#156］
(c) CoLA, Standard Pruning

［#157］
![](./images/867770500502258296_20.jpg)
［#158］
(d) STS-B, Standard Pruning

［#159］
![](./images/867770500502258296_21.jpg)
［#160］
(e) SST-2, Standard Pruning

［#161］
![](./images/867770500502258296_22.jpg)
［#162］
(f) QNLI, Standard Pruning

［#163］
![](./images/867770500502258296_23.jpg)
［#164］
(g) QQP, Standard Pruning

［#165］
![](./images/867770500502258296_24.jpg)
［#166］
(h) MNLI, Standard Pruning

［#167］
![](./images/867770500502258296_25.jpg)
［#168］
(i) SQuAD, Standard Pruning

［#169］
Figure 9: Individual plots showing sparsity vs. accuracy for GLUE tasks and SQuAD in $\text{BERT}_{base}$ models compressed with standard pruning (IMP with no rewinding of weights). The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned models.

［#170］
approach "winning ticket" sparsity levels (which in reality provide an inherent advantage to the larger models that are left with a greater absolute number of parameters at the same sparsity levels).

## A.13 Beyond task accuracy
［#171］
We evaluate full and pruned $\text{BERT}_{base}$ models optimized by vanilla Adam and SAM throughout IMP (with rewind) on Ribeiro et al. (2020)'s pre-curated test suites for sentiment analysis (SST-2), question paraphrase detection (QQP), and question answering (SQuAD). At this time, we do not explicitly make direct comparisons between SAM and vanilla Adam for unpruned and pruned models. A single test consists of multiple examples, and the $x$-axes of the histograms in Figure 16 refer to the proportions of examples passed within each test.

［#172］
![](./images/867770500502258296_26.jpg)

［#173］
(a) RTE, IMP w/ $\ell_1$ Regularization

［#174］
![](./images/867770500502258296_27.jpg)

［#175］
(b) MRPC, IMP w/ $\ell_1$ Regularization

［#176］
![](./images/867770500502258296_28.jpg)

［#177］
(c) SST-2, IMP w/ $\ell_1$ Regularization

［#178］
![](./images/867770500502258296_29.jpg)

［#179］
(d) QNLI, IMP w/ $\ell_1$ Regularization

［#180］
Figure 10: Individual plots showing sparsity vs. accuracy for GLUE tasks in $\ell_1$-regularized $\text{BERT}_{base}$ models compressed with iterative magnitude pruning (IMP), with regular $\text{BERT}_{base}$ models for comparison. The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned models.

［#181］
![](./images/867770500502258296_30.jpg)

［#182］
(a) RTE, Standard Pruning w/ $\ell_1$ Regularization

［#183］
![](./images/867770500502258296_31.jpg)

［#184］
(b) MRPC, Standard Pruning w/ $\ell_1$ Regularization

［#185］
![](./images/867770500502258296_32.jpg)

［#186］
(c) SST-2, Standard Pruning w/ $\ell_1$ Regularization

［#187］
![](./images/867770500502258296_33.jpg)

［#188］
(d) QNLI, Standard Pruning w/ $\ell_1$ Regularization

［#189］
Figure 11: Individual plots showing sparsity vs. accuracy for GLUE tasks and SQuAD in $\text{BERT}_{base}$ models trained with $\ell_1$ regularization during iterative compression with standard pruning, with $\text{BERT}_{base}$ models trained without regularization during standard pruning for comparison. The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned models.

### Learned Tickets vs Optimization Over Tickets (Alternative View)

［#190］
![](./images/867770500502258296_34.jpg)

［#191］
Figure 12: An alternative view of the same information contained in Figure 3. Note that, for example, in RTE,
SAM tickets clearly outperform vanilla Adam tickets regardless of the optimizer used for the final fine-tuning.

［#192］
![](./images/867770500502258296_35.jpg)

［#193］
![](./images/867770500502258296_36.jpg)

［#194］
Figure 13: Heatmaps indicating the difference in target task performance between SAM and Adam optimizers during fine-tuning when transferring tickets across tasks. Values greater than 0 indicate the extent to which SAM optimizer worked better than Adam; Values less than 0 indicate where Adam optimizer worked better than SAM; Values close to 0 indicate little difference Note that positive values along the diagonal indicate superior SAM ticket performance in the single task setting, even with "transfer" between optimizers.

［#195］
![](./images/867770500502258296_37.jpg)

［#196］
(a) RTE, IMP w/ $\text{BERT}_{large}$

［#197］
![](./images/867770500502258296_38.jpg)

［#198］
(b) MRPC, IMP w/ $\text{BERT}_{large}$

［#199］
![](./images/867770500502258296_39.jpg)

［#200］
(c) SST-2, IMP w/ $\text{BERT}_{large}$

［#201］
![](./images/867770500502258296_40.jpg)

［#202］
(d) QNLI, IMP w/ $\text{BERT}_{large}$

［#203］
![](./images/867770500502258296_41.jpg)

［#204］
(e) SQuAD, IMP w/ $\text{BERT}_{large}$

［#205］
Figure 14: Individual plots showing sparsity vs. accuracy for GLUE tasks and SQuAD in $\text{BERT}_{large}$ models compressed with iterative magnitude pruning (IMP), with $\text{BERT}_{base}$ models for comparison. The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned models.

［#206］
![](./images/867770500502258296_42.jpg)

［#207］
(a) RTE, IMP w/ $\text{RoBERTa}_{base}$

［#208］
![](./images/867770500502258296_43.jpg)

［#209］
(b) MRPC, IMP w/ $\text{RoBERTa}_{base}$

［#210］
![](./images/867770500502258296_44.jpg)

［#211］
(c) SST-2, IMP w/ $\text{RoBERTa}_{base}$

［#212］
![](./images/867770500502258296_45.jpg)

［#213］
(d) QNLI, IMP w/ $\text{RoBERTa}_{base}$

［#214］
![](./images/867770500502258296_46.jpg)

［#215］
(e) SQuAD, IMP w/ $\text{RoBERTa}_{base}$

［#216］
Figure 15: Individual plots showing sparsity vs. accuracy for GLUE tasks and SQuAD in $\text{RoBERTa}_{base}$ models compressed with iterative magnitude pruning (IMP), with $\text{BERT}_{base}$ models for comparison. The vertical lines and gray horizontal bands mark reference sparsity and "winning ticket" evaluation metric values that were obtained by Chen et al. (2020). The green horizontal bands mark the initial performance of our full fine-tuned models.

［#217］
![](./images/867770500502258296_47.jpg)

［#218］
Figure 16: Aggregated Checklist (Ribeiro et al., 2020) results for QQP models. Tests target various capabilities of models such as robustness to typos and simple coference resolution. Note the concentration of frequencies near 0.0 and 1.0 for all models, as well as the shifts in frequencies when pruned for both vanilla Adam and SAM models; models can effectively lose or even gain specific capabilities throughout compression.

［#219］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>QAT Ref.</th>
      <th>Optim.</th>
      <th>Full FT</th>
      <th>Quantized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MNLI</td>
      <td rowspan="2">N/A</td>
      <td>Adam</td>
      <td>84.42<sub>0.37</sub></td>
      <td>78.24<sub>4.00</sub></td>
    </tr>
    <tr>
      <td>Acc.</td>
      <td>SAM</td>
      <td>84.68<sub>0.13</sub></td>
      <td>83.47<sub>0.11</sub></td>
    </tr>
    <tr>
      <td>QQP</td>
      <td rowspan="2">87.96</td>
      <td>Adam</td>
      <td>87.98<sub>0.12</sub></td>
      <td>85.35<sub>1.93</sub></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>SAM</td>
      <td>88.20<sub>0.08</sub></td>
      <td>86.85<sub>0.28</sub></td>
    </tr>
    <tr>
      <td>STS-B</td>
      <td rowspan="2">89.04</td>
      <td>Adam</td>
      <td>89.06<sub>0.11</sub></td>
      <td>86.54<sub>1.07</sub></td>
    </tr>
    <tr>
      <td>Pearson</td>
      <td>SAM</td>
      <td>89.39<sub>0.07</sub></td>
      <td>87.16<sub>0.76</sub></td>
    </tr>
    <tr>
      <td>QNLI</td>
      <td rowspan="2">90.62</td>
      <td>Adam</td>
      <td>91.49<sub>0.09</sub></td>
      <td>89.05<sub>0.37</sub></td>
    </tr>
    <tr>
      <td>Acc.</td>
      <td>SAM</td>
      <td>91.33<sub>0.58</sub></td>
      <td>89.83<sub>0.54</sub></td>
    </tr>
    <tr>
      <td>MRPC</td>
      <td rowspan="2">89.56</td>
      <td>Adam</td>
      <td>89.57<sub>1.13</sub></td>
      <td>86.81<sub>1.72</sub></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>SAM</td>
      <td>91.24<sub>0.20</sub></td>
      <td><b>89.37<sub>0.83</sub></b></td>
    </tr>
    <tr>
      <td>RTE</td>
      <td rowspan="2">68.78</td>
      <td>Adam</td>
      <td>67.63<sub>1.10</sub></td>
      <td>56.80<sub>4.51</sub></td>
    </tr>
    <tr>
      <td>Acc.</td>
      <td>SAM</td>
      <td>67.87<sub>0.36</sub></td>
      <td>65.70<sub>1.65</sub></td>
    </tr>
    <tr>
      <td>SST-2</td>
      <td rowspan="2">92.24</td>
      <td>Adam</td>
      <td>92.70<sub>0.07</sub></td>
      <td>91.17<sub>0.90</sub></td>
    </tr>
    <tr>
      <td>Acc.</td>
      <td>SAM</td>
      <td>93.08<sub>0.57</sub></td>
      <td><b>92.39<sub>0.40</sub></b></td>
    </tr>
    <tr>
      <td>CoLA</td>
      <td rowspan="2">58.48</td>
      <td>Adam</td>
      <td>60.47<sub>0.55</sub></td>
      <td>55.99<sub>2.86</sub></td>
    </tr>
    <tr>
      <td>Matt.</td>
      <td>SAM</td>
      <td>59.09<sub>0.72</sub></td>
      <td>54.88<sub>0.78</sub></td>
    </tr>
    <tr>
      <td>SQuAD</td>
      <td rowspan="2">87.74</td>
      <td>Adam</td>
      <td>89.20<sub>0.12</sub></td>
      <td>80.13<sub>1.85</sub></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>SAM</td>
      <td>89.20<sub>0.12</sub></td>
      <td>84.92<sub>0.55</sub></td>
    </tr>
  </tbody>
</table>

［#220］
Table 8: We compare full fine-tuned and quantized BERT<sub>base</sub> models optimized with SAM and Adam. Notably, applying a simpler post-training dynamic quantization technique on a SAM-optimized model can approach the reported (QAT ref) performance of a model quantized through quantization-aware training (Zafrir et al., 2019). These instances are bolded.