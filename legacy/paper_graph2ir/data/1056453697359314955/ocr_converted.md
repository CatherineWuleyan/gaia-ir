# Local Contrastive Editing of Gender Stereotypes

［#1］
Marlene Lutz
University of Mannheim
marlene.lutz@uni-mannheim.de

［#2］
Rochelle Choenni
University of Amsterdam
r.m.v.k.choenni@uva.nl

［#3］
Markus Strohmaier
University of Mannheim, GESIS, CSH Vienna
markus.strohmaier@uni-mannheim.de

［#4］
Anne Lauscher
University of Hamburg
anne.lauscher@uni-hamburg.de

## Abstract

［#5］
Stereotypical bias encoded in language models (LMs) poses a threat to safe language technology, yet our understanding of how bias manifests in the parameters of LMs remains incomplete. We introduce *local contrastive editing* that enables the localization and editing of a subset of weights in a target model *in relation* to a reference model. We deploy this approach to identify and modify subsets of weights that are associated with gender stereotypes in LMs. Through a series of experiments, we demonstrate that local contrastive editing can precisely localize and control a small subset (<0.5%) of weights that encode gender bias. Our work (i) advances our understanding of how stereotypical biases can manifest in the parameter space of LMs and (ii) opens up new avenues for developing parameter-efficient strategies for controlling model properties in a contrastive manner.

## 1 Introduction

［#6］
Stereotypical bias encoded in language models (LMs) can adversely affect the fairness and inclusivity of language technology applications for all users (Blodgett et al., 2020; Choenni et al., 2021; Ma et al., 2023). While considerable efforts have been devoted to measuring (Nadeem et al., 2020; Caliskan et al., 2017) and mitigating (Lauscher et al., 2021) such biases, our understanding of where they manifest in the parameter space of LMs remains limited. Precisely pinpointing biases within the parameters of LMs could enable the development of more targeted and informed bias mitigation strategies. While current research (Ma et al., 2023; Meissner et al., 2022; Hauzenberger et al., 2023) has explored identifying and modifying subcomponents of LMs for bias mitigation, we still lack a thorough understanding of the precise manifestation of biases such as stereotypes in specific model weights.

［#7］
![](./images/1056453697359314955_1.jpg)

［#8］
Figure 1: *Local contrastive editing*. In step 1, we localize weights within a target model that encode a certain property. In Step 2, we modify these selected weights relative to a reference model.

### Research Goal
［#9］
Consequently, this work aims to (i) localize individual weights that drive stereotypical gender bias in LMs and (ii) modify these weights to steer and mitigate the bias.

### Approach
［#10］
We present *local contrastive editing*, a two-step approach that enables the localization and modification of a subset of weights within a target model, relative to a reference model, to control bias (see Figure 1). In step 1, we pinpoint individual weights that encode gender stereotypes via unstructured pruning (Chen et al., 2020). In step 2, we deploy various *local* editing strategies such as weight interpolation or pruning to adjust the identified weights in relation to a reference model.

### Results and Contributions
［#11］
We demonstrate the feasibility of local constrastive editing for controlling stereotypical gender bias through a series of experiments. Using our approach, we are able to identify subsets of weights that drive stereotypical bias in LMs. We find that our local editing strategies can flexibly steer gender bias while at the same time retaining the functionality of the model. We provide experimental evidence that most strategies enable a smooth and controllable transition of bias between networks and empirically find that a small subset of weights (<0.5%) is already sufficient to

［#11］
successfully modify and, finally, mitigate the mea-
surable bias.

## 2 Related Work

### Gender Bias
［#12］
Gender bias is present throughout
the entire NLP pipeline, from training data (Leavy
et al., 2020) to model representations (Bolukbasi
et al., 2016; Gonen and Goldberg, 2019), and pre-
dictions (Dong et al., 2024). Consequently, much
effort has been put towards locating (Joniak and
Aizawa, 2022; Chintam et al., 2023) and mitigat-
ing gender bias at various stages (Sun et al., 2019;
Lauscher et al., 2021; Hauzenberger et al., 2023).
We study how gender bias manifests in LMs by
detecting and editing a minimal set of relevant pa-
rameters to control it.

### Knowledge Localization
［#13］
Pruning methods have
been used to uncover subnetworks, i.e. subsets
of model parameters (Frankle and Carbin, 2018)
isolating task-specific (Nooralahzadeh and Sen-
nrich, 2023), domain-specific (Hendy et al., 2022)
or language-specific (Wang et al., 2020; Choenni
et al., 2023a,b; Nooralahzadeh and Sennrich, 2023)
information. In this paper, we use pruning to find
subnetworks that contain stereotypical gender bias.
Previous research (Vig et al., 2020; Chintam et al.,
2023) suggests that stereotypical gender bias is
concentrated in specific substructures of a network
such as attention heads (Chintam et al., 2023; Ma
et al., 2023; Vig et al., 2020) or neurons (Vig et al.,
2020). We aim to pinpoint the individual weights
responsible for encoding gender bias within a net-
work via unstructured pruning (Chen et al., 2020).

### Model Editing
［#14］
Pretrained LMs serve as back-
bone for many downstream applications, requiring
them to be tailored to specific needs. However, the
growing size of language models has made tradi-
tional fine-tuning costly, leading to increased in-
terest in alternative refinement methods that avoid
gradient updates (Yao et al., 2023). One such line
of research focuses on efficient model weight edit-
ing strategies (Ilharco et al., 2022a,b; Gueta et al.,
2023), using mathematical operations on weight
vectors composed from the full model to modify
information. In this paper, we take a more fine-
grained approach to model editing, and instead
focus on editing only a subset of weights that we
identify as being of relevance for encoding gender
stereotypical biases beforehand.

## 3 Local Contrastive Editing

［#15］
We localize and adjust specific weights in a target
model that are responsible for encoding properties
such as stereotypical bias. To achieve this, we use
several contrastive strategies based on comparing a
target network with a reference network that differ
in a property of interest. We refer to this group of
techniques as contrastive weight editing.

［#16］
Formally, let $f(x,\theta)$ be the output of a network
with parameters $\theta \in \mathbb{R}^d$ for an example input $x$.
Given a target network $f_t(\cdot, \theta_t)$ and a reference
network $f_r(\cdot, \theta_r)$ of the same architecture, with
［#16］
$\theta_t, \theta_r \in \mathbb{R}^d$, we aim to edit $\theta_t$ w.r.t. $\theta_r$ to modify
a property of interest $p$ in $f_t$ while maintaining
performance on the original fine-tuning task.

### 3.1 Localization

［#17］
In the first step, we investigate how a specific prop-
erty $p$ manifests in the parameter space of a model
and try to localize the individual weights associated
with it. To this end, we use unstructured magnitude
pruning (Chen et al., 2020) and discover subnet-
works in a target and a reference network that are
linked to the encoding of $p$. We define a subnetwork
for a network $f(\cdot, \theta)$ as $f(\cdot, m \odot \theta)$, where $\odot$
is the element-wise product and $m \in \{0,1\}^d$ is a
binary pruning mask that sets some parameters in $\theta$
to 0. By comparing the target and reference subnet-
works, we aim to identify subsets of weights that
are related to the encoding of $p$. We note that sub-
networks extracted from different parent models
can differ in two aspects: (1) their pruning masks
may set different parameters to 0; and (2) their
parameters $\theta$ may have different values.

［#18］
We explore both aspects separately by first select-
ing the corresponding subsets of weights and then
using them to modify the target network. To this
end, we define a localization mask as the outcome
of a particular localization strategy, which indicates
which weights will be edited in the subsequent step.
Formally, we define such a mask for a given index
set $\mathcal{I} \subseteq \{1, \dots, d\}$ as $b := b(\mathcal{I}) \in \{0,1\}^d$ via its
elements, such that $b_i = \mathbb{1}\{i \in \mathcal{I}\}$. A value of 1 at
index $i$ indicates that the corresponding weight is
selected for editing. We propose the following two
strategies to compute such localization masks.

［#19］
Mask-based Localization Given a target subnetwork $f_t(\cdot, m_t \odot \theta_t)$ and a reference subnetwork
［#19］
$f_r(\cdot, m_r \odot \theta_r)$, we select those weights that are
present in only one of the subnetworks, indicated

［#20］
![](./images/1056453697359314955_2.jpg)

［#21］
Figure 2: Overview of localization and editing strategies. We show value-based and masked-based localization together with our editing strategies (inter- and extrapolation, pruning) on exemplary target and reference networks.

［#22］
by the pruning masks $m_t$ and $m_r$ . Formally, we compute the localization mask $b$ as:

［#22］
$$
b = m_t \odot m_r \,. \tag{1}
$$

［#23］
We hypothesize that precisely because weights are pruned in one network, but not the other, they encode information relevant to the property $p$.

［#24］
Value-based Localization Given a target subnetwork $f_t(\cdot, m_t \odot \theta_t)$ and a reference subnetwork $f_r(\cdot, m_r \odot \theta_r)$, we select a subset of top-$k$ weights that are present in both subnetworks, but differ the most with regard to their values. Let $I_{top}^k$ be the index set containing the indices of the $k$ largest absolute weight differences $|(m_r \odot \theta_r) - (m_t \odot \theta_t)|$. Then we define the localization mask $b$ as:

［#24］
$$
b_i =
\begin{cases}
1, & \text{if } i \in I_{top}^k \\
0, & \text{otherwise}.
\end{cases} \tag{2}
$$

［#25］
We hypothesize that the weights with the largest absolute difference most strongly steer the networks towards opposing directions with respect to $p$.

### 3.2 Contrastive Editing Strategies
［#26］
After identifying subsets of weights potentially associated with the property of interest $p$, we use these weights to modify the target network. We explore different types of edits and evaluate their effectiveness. In the following, we assume that we are given a target subnetwork $f_t(\cdot, m_t \odot \theta_t)$, a reference subnetwork $f_r(\cdot, m_r \odot \theta_r)$ and a localization mask $b$ that indicates which weights should be edited. The goal of each of the local editing strategies is to create a new target network $f_t'(\cdot, \theta_t')$ that is modified with respect to the reference network.

［#27］
Weight Interpolation (IP) Inspired by recent work on model merging (Ilharco et al., 2022a; Wortsman et al., 2022; Yadav et al., 2024), we propose linear weight interpolation that moves the localized weights of the target network closer to those of its reference or even adopts them completely ($\alpha = 1$):

［#27］
$$
\theta_t' = \theta_t + \alpha((\theta_r - \theta_t) \odot b), \alpha \in [0, 1] \,. \tag{3}
$$

［#28］
Note, that linear interpolation can also be used with mask-based localization by assuming that pruned weights have a value of 0.

［#29］
Weight Extrapolation (EP) Similar to interpolation, we propose linear weight extrapolation to move the localized weights of the target either towards or away from those of the reference network:

［#29］
$$
\theta_t' = \theta_t + \alpha((\theta_r - \theta_t) \odot b), \alpha \in \mathbb{R} \setminus [0, 1] \,. \tag{4}
$$

［#30］
Allowing for weighting factors $\alpha \in \mathbb{R} \setminus [0, 1]$ enables flexible modifications, including e.g. the removal of a property from a network.

［#31］
Pruning (PR) Pruning is motivated by the assumption that the localized weights encode a property that can be eliminated by removing precisely those weights:

［#31］
$$
\theta_t' = \theta_t - (\theta_t \odot b) \,. \tag{5}
$$

［#32］
Mask Switch (SW) Our final editing strategy is only applicable for mask-based localization and relies on the impact of weights being present ("turned on") or pruned ("turned off"). We apply the subnetwork mask of the reference model to the target model, resulting in pruning additional weights from the target model. Weights that were initially pruned in the target model during the localization step, but reinstated via the reference subnetwork mask, are restored to their values before pruning.

［#33］
$$
\theta_{t}' = \theta_{t} \odot m_{r}. \tag{6}
$$

## 4 Experimental Setup
［#34］
We showcase the effectiveness of local contrastive editing in one of the, arguably, most established experimental environments for testing bias modification methods from the literature: stereotypical binary gender bias encoded in the BERT $^1$ model (Devlin et al., 2019). BERT is a widely used transformer model with 12 attention heads and 110 million parameters in total.

### 4.1 Reference and Target Models
［#35］
To localize and edit the encoding of stereotypical bias using contrastive strategies, we begin by establishing appropriate target and reference models. For obtaining a thorough understanding of the expected effects, we start from an "extreme" setup in which we intentionally bias two types of models to be either *stereotypical* or *anti-stereotypical* concerning specific gender associations. This is accomplished by fine-tuning BERT on subsets of the English Wikipedia $^2$ that we pre-processed to exhibit gendered associations using the well-established Counterfactual Data Augmentation (Zhao et al., 2018). We describe the process in more detail in the following sections.

#### 4.1.1 Bias Specification
［#36］
We investigate binary stereotypical gender bias in terms of stereotypical gender associations that manifest in written text. We make use of an explicit bias specification $B=(T_1, T_2, A_1, A_2)$ (Caliskan et al., 2017; Lauscher et al., 2020) that consists of two sets of target words $T_1, T_2$ that describe demographic groups between which we expect a bias w.r.t. two sets of attributes $A_1, A_2$. We choose terms in $T_1$ to represent the female gender (e.g. *woman*) and terms in $T_2$ to describe the male gender (e.g. *man*). We then build pairs of corresponding terms $(t, t') \subset T_1 \times T_2$ (e.g. (*aunt*, *uncle*)). Further, we designate terms in $A_1$ to be stereotypically associated with $T_1$ (e.g. *child-care*) and $A_2$ to contain attributes that are stereotypically associated with $T_2$ (e.g. *programming*). The full specification can be found in appendix A and was adopted from **Barikeri et al. (2021)**. Note, that we do not claim our list of target and attribute words to be complete, we rather aim for a small and precise specification that demonstrates the feasibility of our approach.

#### 4.1.2 Counterfactual Data Augmentation
［#37］
Starting from the bias specification in 4.1.1, we create two datasets that we consider to be stereotypical and anti-stereotypical, respectively. Following the principle of Counterfactual Data Augmentation (Zhao et al., 2018), we aim to artificially amplify or break associations between target words and their stereotypical attributes for obtaining our contrastive models. We use English Wikipedia as a base and filter the corpus for sentences $s_{(i,j)}$ containing exactly one target word $t \in T_i$ and one attribute word $a \in A_j$, where $i,j \in \{1,2\}$. A sentence $s_{(i,j)}$ is categorized as stereotypical if $i=j$ and anti-stereotypical if $i \neq j$. For constructing a stereotypical dataset, we iterate through all sentences $s_{(i,j)}$ and retain those that are stereotypical. In cases where $s_{(i,j)}$ is anti-stereotypical, i.e. $i \neq j$, we replace the target term $t \in T_i$ with its corresponding paired term $t' \in T_j$. For creating an anti-stereotypical dataset, we keep all sentences $s_{(i,j)}$ with $i \neq j$ and substitute $t \in T_i$ with its paired target term $t' \in T_j$, if $i = j$. Note, that the resulting stereotypical and anti-stereotypical datasets are identical besides the swapped target terms.

#### 4.1.3 Fine-tuning
［#38］
We fine-tune BERT on the biased datasets using a masked language modeling (MLM) objective, creating stereotypical and anti-stereotypical models. To achieve higher levels of bias, we adjust the masking function to mask the target and attribute terms from our bias specification preferentially, i.e. with higher probability. We keep the average number of masked tokens constant by lowering the masking probability for all other tokens accordingly. We tested preferential masking probabilities between 0.15 and 0.9 and found that a value of 0.3 resulted in the best trade-off between perplexity and bias level. Additionally, we augmented the bi-

---
［#34］
$^1$we use the Huggingface BERT-base-uncased distribution.
［#35］
$^2$20220301.en dump, Foundation

［#39］
ased datasets with neutral examples not containing terms from the bias specification as this positively impacted the stability of subnetworks. To ensure the robustness of our findings, we conduct our experiments using four different random seeds. All training details can be found in appendix B.

### 4.2 Bias Evaluation
［#40］
We measure gender bias using three well-established bias benchmarks, namely WEAT, StereoSet and CrowS-Pairs, all measuring intrinsic bias. The Word Embedding Association Test (WEAT) (Caliskan et al., 2017) measures the differential association of two sets of target words w.r.t. two sets of attribute words based on embedding similarity. We utilize the WEAT 8 test that compares male and female target terms to attribute terms related to art and science, respectively (see appendix D for the full specification). As many of these terms are also included in our bias specification in 4.1.1, WEAT 8 is a good instrument to precisely measure the gendered associations that we injected in our target and reference models and how these change when editing weights. We report the average effect size, where positive values indicate stereotypical bias and negative values indicate anti-stereotypical bias. We include a more detailed description of WEAT in appendix C.

［#41］
We use the StereoSet intrasentence task (Nadeem et al., 2020) as a second bias measure, evaluating stereotypical bias at the sentence level across four target domains. StereoSet contains sentences describing a target group with a blank (e.g. Girls tend to be more [BLANK] than boys), which can be filled with a stereotypical (e.g. soft), an anti-stereotypical (e.g. determined) or a meaningless (e.g. fish) attribute. The StereoSet stereotype (SS) score, ranging from 0 to 100, is computed based on how often a model fills the blank with a stereotypical attribute over an anti-stereotypical attribute. A SS score >50 indicates stereotypical bias, while a score <50 indicates anti-stereotypical bias. We report scores for the gender-related portion of the dataset.

［#42］
Our third measure, CrowS-Pairs (Nangia et al., 2020), comprises a crowd-sourced dataset of sentence pairs, with the first sentence being more stereotypical (e.g. Women are always too sensitive about things) than the second (e.g. Men are always too sensitive about things). The sentences differ only in the tokens related to the target group (e.g. women/men). Following Meade et al. (2022) we calculate the CrowS-Pairs stereotype (CS) score based on how often a model prefers the more stereotypical sentence. Scores range from 0 to 100, with CS scores >50 indicating a preference for the more stereotypical sentence and a CS score of <50 indicating the opposite. We report scores only for the gender-related portion of the dataset.

［#43］
Both Stereoset and CrowS-Pairs have been criticized in the past for some of their properties (Blodgett et al., 2021; Pikuliak et al., 2023; Delobelle et al., 2022). We use these measures only complementary to WEAT and will later show that we observe similar effects across all three measures.

### 4.3 Subnetwork Identification
［#44］
We discover subnetworks through weight pruning, inspired by the lottery ticket hypothesis (LTH) (Frankle and Carbin, 2018). The LTH posits that dense, randomly-initialized neural networks contain smaller subnetworks (winning tickets) that can be trained in isolation to achieve the accuracy of the full model. We use the approach of Chen et al. (2020) and apply iterative magnitude pruning (IMP) to extract subnetworks from the stereotypical and anti-stereotypical models. In IMP, we alternate between fine-tuning a model for $i$ steps and subsequently pruning 10% of the weights with the lowest magnitude. After pruning, we reset the remaining weights to their initial values and repeat the steps until achieving a desired sparsity level. We accept a subnetwork as winning ticket if its performance after $i$ fine-tuning steps is within 5% of the performance of the full model, fine-tuned for the same number of steps. Additional details on our application of IMP can be found in appendix E. Note that besides IMP, we also explored structured attention head pruning (Prasanna et al., 2020) but found no differences between stereotypical and anti-stereotypical subnetwork masks. We suspect that this occurred due to attention heads being too coarse-grained to capture the subtle differences in bias we injected.

### 4.4 Uninformed Editing
［#45］
To explore the importance of the localization step, we also deploy our editing strategies to subsets of weights that are not informed by strategic localization. For uninformed intrapolation and uninformed extrapolation, we randomly sample from all weights, excluding those pruned in both the target and reference subnetworks, as intra- or extrapolation would not affect these weights. For

［#46］
<table>
<caption>(a) stereotypical subnetworks</caption>
<thead>
  <tr>
    <th></th>
    <th>full</th>
    <th>10%</th>
    <th>20%</th>
    <th>30%</th>
    <th>40%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>WEAT</th>
    <td>0.93</td>
    <td>0.88</td>
    <td>0.98</td>
    <td>0.89</td>
    <td>0.91</td>
  </tr>
  <tr>
    <th>StereoSet</th>
    <td>61.03</td>
    <td>60.27</td>
    <td>60.09</td>
    <td>61.11</td>
    <td>60.95</td>
  </tr>
  <tr>
    <th>CrowS-Pairs</th>
    <td>57.92</td>
    <td>59.45</td>
    <td>59.83</td>
    <td>57.25</td>
    <td>58.88</td>
  </tr>
</tbody>
</table>

［#47］
<table>
<caption>(b) anti-stereotypical subnetworks</caption>
<thead>
  <tr>
    <th></th>
    <th>full</th>
    <th>10%</th>
    <th>20%</th>
    <th>30%</th>
    <th>40%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>WEAT</th>
    <td>-0.75</td>
    <td>-0.75</td>
    <td>-0.56</td>
    <td>-0.63</td>
    <td>-0.44</td>
  </tr>
  <tr>
    <th>StereoSet</th>
    <td>58.86</td>
    <td>58.37</td>
    <td>58.03</td>
    <td>59.52</td>
    <td>59.30</td>
  </tr>
  <tr>
    <th>CrowS-Pairs</th>
    <td>52.77</td>
    <td>53.15</td>
    <td>52.86</td>
    <td>52.01</td>
    <td>52.39</td>
  </tr>
</tbody>
</table>

［#48］
Table 1: Bias of subnetworks at different sparsities.
We report the mean across all random seeds with higher
scores indicating higher stereotypical bias.

［#49］
uninformed pruning, we randomly sample from the
weights that are present in the target subnetwork,
excluding those that already are pruned. To en-
sure a fair comparison, we select the same number
of weights as those identified by the localization
strategies in section 3.1.

## 5 Results

［#50］
We fine-tune BERT according to section 4.1
and create models with stereotypical and anti-
stereotypical biases. Table 1 ("full") shows that
the stereotypical models exhibit higher levels
of measurable stereotypical bias than the anti-
stereotypical models. Thus, as intended, we have
successfully steered the BERT model in two ex-
treme directions, which will serve as a basis for our
experiments on contrastive editing. We observe a
trend for both types of models to shift towards the
stereotypical regime, due to the fact that the bias
benchmarks test for a broader range of associations
than those we artificially controlled. This effect is
less pronounced for WEAT, as WEAT specifically
tests for many of our injected associations.

### 5.1 Subnetwork Analysis

［#51］
We discover subnetworks that are winning tickets
(cf. section 4.3) for both stereotypical and anti-
stereotypical models at different sparsities up to
40%. The discovered subnetworks are stable across
runs with different random seeds, as indicated by
a high Jaccard similaritiy of the pruning masks (>
0.98). This suggests that the findings are robust and
not heavily influenced by factors such as specific
data splits. Table 1 illustrates that the discovered
subnetworks largely maintain the bias of their par-
ent networks, highlighting their suitability for our
contrastive approach.

［#52］
![](./images/1056453697359314955_3.jpg)

［#53］
Figure 3: Bias localization. We illustrate the percentage
of weights per component that have been selected for
editing. Notably, both localization strategies focus on
the same layers and components. We show the results
for subnetworks at sparsity 40% and one random seed.

［#54］
Next, we compare subnetworks with stereotypi-
cal bias to subnetworks with anti-stereotypical bias.
At all sparsity levels, we find that the percentage
of weights where the pruning masks differ remains
below 0.5%, indicating a high degree of similarity.
This is expected, as both types of subnetworks spe-
cialize on the same task and are fine-tuned on simi-
lar datasets, differing only in the injected stereotypi-
cal and anti-stereotypical associations, respectively.
To investigate where these associations manifest in
the parameter space, we apply both mask-based and
value-based localization. Although the localiza-
tion strategies are designed not to select the same
subsets of weights, they consistently target simi-
lar areas within the model, particularly focusing
on the last layers and predominantly the attention
output dense layer (see figure 3). This observa-
tion aligns with previous work (Ma et al., 2023;
Chintam et al., 2023), which found that attention
heads most influential for gender bias are located
in higher layers, suggesting that bias is encoded in
specific subcomponents of transformers.

### 5.2 Effect on Gender Bias

［#55］
We investigate the effectiveness of the local con-
trastive editing strategies by considering two set-
tings: using the stereotypical subnetworks as the

［#55］
target with the anti-stereotypical subnetworks as the reference, and vice versa. Figure 4 illustrates results for selected parameter settings, demonstrating the flexibility of our strategies.

［#56］
We find that nearly all editing strategies, when combined with either mask- or value-based localization, effectively modify gender bias as intended. Mask-based editing achieves this efficiently with a small subset size of less than $0.5\%$. By varying the weighting factor $\alpha$, we can flexibly control the bias of the target model and, according to WEAT, even completely remove bias at $\alpha=2$. In contrast, uninform edits result in minimal or no changes to the bias scores, highlighting the critical role of the localization step in local contrastive editing. We summarize that (i) both mask-based and value-based localization can identify subsets of weights driving stereotypical gender bias, and that (ii) gender bias can be controlled through contrastive editing strategies on these subsets.

### 5.3 Effect on Performance
［#57］
We further examine how local contrastive editing affects a model's ability to model language. To assess this, we use perplexity as a primary measure and additionally compute the *language modeling (LM) score* as introduced by Nadeem et al. (2020). Similar to the SS score (cf. section 4.2), the LM score is computed on the StereoSet dataset and measures how frequently a model prefers a sentence with a meaningful association (e.g. *Girls tend to be more determined than boys*) over a nonsensical one (e.g. *Girls tend to be more fish than boys*). The LM score ranges from 0 to 100, where 100 represents an ideal model that always favours the semantically meaningful association.

［#58］
As shown in figure 4, nearly all editing strategies lead to only minor increases in perplexity. The notable exception is value-based pruning, which causes a significant increase of 9.529 points in perplexity, while extrapolation also leads to a significant but much smaller rise. This substantial degradation in model performance may explain the counterintuitive effect of value-based pruning on gender bias in figure 4, as the models' overall functionality is severely impaired. Consistent with these findings, the LM score shows no significant decline, except for value-based pruning, which leads to a significant drop of 2.723 points at a sparsity level of 30%. Complete LM score results and results for other sparsity levels can be found in appendix F.2.

［#59］
Next, we fine-tune the edited models on the

［#60］
<div style="display: flex; gap: 2em; justify-content: center;">
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>MNLI-m/mm</th>
    <th>STS-B</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Base</td>
    <td>84.4/84.8</td>
    <td>89.0/88.9</td>
  </tr>
  <tr>
    <td>IP ($\alpha$=0.5)</td>
    <td>83.9/84.2</td>
    <td>88.6/88.0</td>
  </tr>
  <tr>
    <td>IP ($\alpha$=1)</td>
    <td>83.9/84.4</td>
    <td>88.5/88.0</td>
  </tr>
  <tr>
    <td>EP ($\alpha$=2)</td>
    <td>83.9/84.3</td>
    <td>88.5/88.0</td>
  </tr>
  <tr>
    <td>EP ($\alpha$=-2)</td>
    <td>83.8/84.4</td>
    <td>88.5/88.1</td>
  </tr>
  <tr>
    <td>PR</td>
    <td>83.9/84.3</td>
    <td>88.4/87.9</td>
  </tr>
  <tr>
    <td>SW</td>
    <td>83.8/84.4</td>
    <td>88.5/88.0</td>
  </tr>
</tbody>
</table>
<p>(a) mask-based loc.</p>

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>MNLI-m/mm</th>
    <th>STS-B</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Base</td>
    <td>84.4/84.8</td>
    <td>89.0/88.9</td>
  </tr>
  <tr>
    <td>IP ($\alpha$=0.5)</td>
    <td>83.9/84.3</td>
    <td>88.6/88.0</td>
  </tr>
  <tr>
    <td>IP ($\alpha$=1)</td>
    <td>83.8/84.3</td>
    <td>88.5/88.0</td>
  </tr>
  <tr>
    <td>EP ($\alpha$=2)</td>
    <td>84.0/84.4</td>
    <td>88.5/87.9</td>
  </tr>
  <tr>
    <td>EP ($\alpha$=-2)</td>
    <td>83.8/84.5</td>
    <td>88.6/88.1</td>
  </tr>
  <tr>
    <td>PR</td>
    <td>83.5/83.8</td>
    <td>87.0/86.6</td>
  </tr>
</tbody>
</table>
<p>(b) value-based loc. ($k$=10%)</p>
</div>

［#61］
**Table 2:** Performance on downstream tasks. We fine-tune the edited models on the MNLI and STS-B tasks from the GLUE benchmark and show the results for the stereotypical target model at 30% sparsity using a single random seed. For MNLI, we report both matched and mismatched accuracy while for STS-B, we present Pearson and Spearman correlation. Results for other sparsities and target models can be found in the appendix F.3.

［#62］
MNLI and STS-B tasks from the GLUE benchmark (Wang et al., 2018), to evaluate their performance on downstream tasks. As shown in table 2, the model subjected to value-based pruning again exhibits the most significant performance drop compared to the base model $^3$. Models edited using other strategies experience a maximum performance loss of only $0.71\%$/$0.59\%$ for MNLI and $1.07\%$/$1.06\%$ for STS-B.

［#63］
In summary, we demonstrate that, with exception of value-based pruning, local contrastive editing largely preserves a model's language modeling ability. Furthermore, the edited models can still be effectively used in downstream tasks with only minor performance loss.

### 5.4 Ablation Study
［#64］
We explore the impact of the number of selected weights $k$, the weighting factor $\alpha$, and the different localization strategies on inter- and extrapolation.

［#65］
**Number of Selected Weights** Figure 5 shows the sensitivity of value-based interpolation ($\alpha=0.5$) to the number of selected weights $k$. Increasing $k$ initially leads to stronger effects on gender bias up to a threshold of 20–40%. Beyond this range, editing further weights seems to have no effect, indicating that there is a critical subset primarily encoding the bias.

［#66］
**Weighting Factor** Figure 6 shows the behavior of linear weight inter- and extrapolation for all localization strategies. For WEAT, we can achieve a smooth, monotonous change in gender bias by gradually increasing $\alpha$. StereoSet and CrowS-Pairs

［#66］
$^3$BERT-base-uncased

［#67］
![](./images/1056453697359314955_4.jpg)

［#68］
Figure 4: Local contrastive editing of gender bias. We illustrate the effects of our local editing strategies on gender bias for settings in which (i) the target model is anti-stereotypical and the reference is stereotypical (red) and (ii) the target model is stereotypical and the reference is anti-stereotypical (blue). The colored arrows on the left indicate the intuition of each strategy. We show the results for subnetworks at sparsity 30% and report the mean bias over four random seeds, with error bars indicating one standard deviation in each direction. On the right, we display the mean perplexity change across both settings and all random seeds where bold indicates a significant increase. Our local editing strategies can successfully steer stereotypical bias with both localization methods, while uninformed edits have much lower or no effect at all. Results for other sparsities can be found in appendix F.1.

［#69］
![](./images/1056453697359314955_5.jpg)

［#70］
Figure 5: Sensitivity to the number of weights edited. We explore the influence of the number of top-$k$ weights that are used for value-based interpolation with $\alpha=0.5$. We show the results for subnetworks at sparsity 30% and report the mean bias and standard deviation across four random seeds. We report the average change in perplexity across both target models and all random seeds, observing no significant increase for any choice of $k$. Results for other sparsities can be found in appendix F.4.

［#71］
![](./images/1056453697359314955_6.jpg)

［#72］
Figure 6: Sensitivity to the weighting factor. We investigate the effect of different weighting factors $\alpha$ on gender bias (a) and perplexity (b). For value-based and uninformed edits we choose the same number of weights that were selected by mask-based localization. We find that weighting factors with higher magnitudes lead to greater effects on bias, correlating with an increase in perplexity. We display the results for a sparsity level of 30% and report the mean across four random seeds. For perplexity, we average the results for both target models. Results for other sparsities can be found in appendix F.5.

［#73］
measure a similar trend, but the effects become inconsistent for higher absolute weighting factors ($|\alpha| \geq 5$), likely due to a decline in language modeling performance, as evidenced by increasing perplexity (see figures 6b, 17) and decreasing LM scores (see figure 18). Overall, we find that varying $\alpha$ allows flexible control and calibration of bias levels within a target model, that can be tailored to the characteristics of the reference model (e.g. reducing bias when the reference model itself exhibits bias), albeit within certain limits.

［#74］
Localization Strategies We choose the same number of weights to be edited for all localization strategies, allowing their direct comparison in figure 6. We observe that the localization strategy that leads to stronger steering effects as measured by WEAT (specifically mask-based localization at 30% sparsity) also leads to a stronger decline in language modeling ability when $|\alpha|$ increases. In line with the result in figure 4, we further find that uninformed editing does not change the bias level significantly, not even for high magnitudes of $\alpha$. This suggests that certain subsets of weights encode gender bias more prominently, and that their localization can be crucial for bias modification.

### 5.5 Wider Applicability
［#75］
So far, our experiments have been conducted in a controlled environment where the target and reference models were fine-tuned on parallel datasets. To test the wider applicability of our approach, we fine-tune a *neutral* model on a subset of Wikipedia that is independent of the biased datasets described in section 4.1. The training details can be found in appendix B. In line with the other experiments, we extract neutral subnetworks at different sparsities with four random seeds. We use those as target networks and edit them w.r.t both stereotypical and anti-stereotypical reference models. Figure 7 illustrates the results at sparsity 40%. By using the stereotypical reference model we can successfully modify the bias of the neutral model in line with our intuition. For instance, extrapolation with $\alpha=-2$, successfully removes stereotypical bias, as measured by WEAT. This is not trivial, as here the reference and target models are fine-tuned on datasets that do not overlap, which implies that the weights selected by the localization strategies may encode differences in the datasets beyond just stereotypical bias. Using an anti-stereotypical reference model produces mixed results, aligning with expectations in most but not all scenarios, requiring further investigation.

［#76］
![](./images/1056453697359314955_7.jpg)

［#77］
Figure 7: Application to a neutral target model. We apply local contrastive editing to a *neutral* target model. Using a stereotypical reference model (blue) can effectively steer the neutral model's bias. Using an anti-stereotypical reference model (red) produces inconclusive results. We show the results for a sparsity of 40% and report the mean and standard deviation across four random seeds. Results for other sparsities can be found in appendix F.6

## 6 Conclusion
［#78］
Our research shows that stereotypical gender bias is primarily encoded in specific subsets of weights within LMs. We propose various local contrastive editing strategies and demonstrate that they can effectively identify and modify these subsets to flexibly control and mitigate gender bias. This work enhances our understanding of where stereotypical biases manifest in the parameter space of LMs and opens up new avenues for developing parameter-efficient strategies for model editing in a contrastive manner. Local contrastive editing is not limited to gender bias, and future research could explore its application to other tasks and domains.

## 7 Limitations
［#79］
Naturally, our work comes with limitations. We conduct experiments using a single model architecture and a single bias type only. We restrict our study to this model architecture because of computational constraints and environmental considerations, particularly because iterative magnitude pruning requires substantial computational

［#79］
resources. However, we anticipate that our find-
ings generalize broadly, as related work on weight
averaging has been shown to generalize to other
model architectures as well (Wortsman et al., 2022;
Yadav et al., 2024; Ilharco et al., 2022b). More-
over, findings from other studies on bias mitigation
suggest generalizability to other types of bias that
share similar specifications (Hauzenberger et al.,
2023; Guo et al., 2022).

## 8 Ethical Considerations

［#80］
While our work ultimately targets the development
of strategies for reducing bias in language models,
it is important to acknowledge the potential dual
use of these techniques. The same strategies de-
signed to reduce bias, can also be used to perpetuate
and amplify biases in language models. Moreover,
our intentional design of language models that ini-
tially exhibit high levels of bias raises concerns
about their potential deployment in various appli-
cations. Despite our intention to mitigate bias, the
existence of such models risks normalizing and
perpetuating stereotypical gender biases in society.

［#81］
Additionally, our study focuses on bias specifi-
cally between male and female groups, thus operat-
ing on a binary specification of gender bias, only.
We recognize that gender is a spectrum, encom-
passing identities beyond strictly male and female
categories. Our rationale for this approach is to
evaluate the effectiveness of our strategies using
an existing and well-established specification that
facilitates measurement of their effects.

## Acknowledgments

［#82］
We thank Ekaterina Shutova for her valuable input
and feedback for this project. The authors acknowl-
edge support by the state of Baden-Württemberg
through bwHPC and the German Research Founda-
tion (DFG) through grant INST 35/1597-1 FUGG.
The work of Anne Lauscher is funded under the
Excellence Strategy of the German Federal Govern-
ment and the Federal States. The work of Rochelle
Choenni is supported by a Google PhD Fellowship.

## References















































## A Gender Bias Specification

### Target Word Pairs
［#83］
$T_1 \times T_2$: (woman, man), (women, men), (girl, boy), (mother, father), (daughter, son), (wife, husband), (niece, nephew), (mom, dad), (bride, groom), (lady, gentleman), (madam, sir), (hostess, host), (female, male), (wife, husband), (aunt, uncle), (sister, brother), (she, he)

### Attribute Words
［#84］
$A_1$: nurse, secretary, housekeeper, cook, sewing, cleaning, waitress, maid, child-care, receptionist, clerk, home-care, poetry, art, dance, literature, novel, symphony, drama, sculpture, shakespeare
［#84］
$A_2$: surgeon, executive, manager, officer, engineering, programming, lawyer, engineer, finance, administrator, physician, science, math, geometry, technology, equation, computation, physics, chemistry, einstein

## B Training Details
［#85］
For fine-tuning and pruning we used 4 NVIDIA A100-80GB GPUs. One pruning and fine-tuning iteration took 32 GPU hours, amounting to 160 GPU hours to extract subnetworks up to sparsity 40% from a single model. As we repeated this for three types of models and four random seeds, we amount in a total of 1920 GPU hours.

［#86］
<table>
  <thead>
    <tr>
      <th></th>
      <th>Biased</th>
      <th>Neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td># Biased examples</td>
      <td>164,524</td>
      <td>0</td>
    </tr>
    <tr>
      <td># Neutral examples</td>
      <td>164,524</td>
      <td>329,048</td>
    </tr>
    <tr>
      <td>Task</td>
      <td>MLM</td>
      <td>MLM</td>
    </tr>
    <tr>
      <td>Masking prob.</td>
      <td>0.3 / 0.148</td>
      <td>0.15</td>
    </tr>
    <tr>
      <td># Epochs</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <td># Iterations/epoch</td>
      <td>4627</td>
      <td>4627</td>
    </tr>
    <tr>
      <td>Batch size</td>
      <td>64</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Learning rate</td>
      <td>$1 \times 10^{-5}$</td>
      <td>$1 \times 10^{-5}$</td>
    </tr>
    <tr>
      <td>Eval size</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>Eval Measure</td>
      <td>Perplexity</td>
      <td>Perplexity</td>
    </tr>
    <tr>
      <td># Random seeds</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>

［#87］
Table 3: Details of fine-tuning biased (stereotypical or anti-stereotypical) and neutral models. Optimization is performed with AdamW with $\epsilon = 1 \times 10^{-8}$ and the learning rate decays linearly to zero. We use standard implementations and hyperparameter settings (Wolf et al., 2020).

## C Word Embedding Association Test (WEAT)
［#88］
Caliskan et al. (2017) introduce WEAT by extending the Implicit Association Test (Greenwald et al., 1998), a test used to measure human biases, to word embeddings. The test measures the differential association of two sets of target words $X,Y$ (e.g. female and male terms) w.r.t. two sets of attribute words $A,B$ (e.g. art and science terms) based on their cosine similarity in the embedding space:

［#88］
$$
s(X,Y,A,B) = \sum_{x \in X} s(x,A,B) - \sum_{y \in Y} s(y,A,B),
$$

［#88］
where
［#88］
$$
s(w,A,B) = \frac{1}{|A|} \sum_{a \in A} cos(w,a) - \frac{1}{|B|} \sum_{b \in B} cos(w,b).
$$

［#89］
The significance of the test is computed with a permutation test, where $\{(X_i,Y_i)\}_i$ are all equally sized partitions of $X \cup Y$ into two sets:
［#89］
$$
Pr_i((s(X_i,Y_i,A,B) > s(X,Y,A,B))
$$

［#90］
Here, we report the effect size as a measure of separation between the association distributions:
［#90］
$$
\frac{\text{mean}_{x \in X} s(x,A,B) - \text{mean}_{y \in Y} s(y,A,B)}{\text{std-dev}_{w \in X \cup Y} s(w,A,B)}
$$

［#91］
Following Vulić et al. (2020), we extract embeddings for all target and attribute words by feeding them through BERT, prepended with the start of sequence token and appended with the separator token (e.g. <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> woman [SEP]). We then extract embeddings from each hidden layer and compute WEAT separately for each layer. Finally, we report the average effect size of the test across all layers.

## D WEAT 8 Specification
### Target Words
［#92］
$X$: science, technology, physics, chemistry, Einstein, NASA, experiment, astronomy
［#92］
$Y$: poetry, art, Shakespeare, dance, literature, novel, symphony, drama

### Attribute Words
［#93］
$A$: brother, father, uncle, grandfather, son, he, his, him
［#93］
$B$: sister, mother, aunt, grandmother, daughter, she, hers, her

## E Iterative Magnitude Pruning

［#94］
We apply iterative magnitude pruning according to the following procedure:

［#95］
1. Fine-tune a pre-trained network $f(\cdot, \theta_0)$ for $i$ steps
2. Globally prune $p\%$ of the weights with the lowest magnitude, resulting in a subnetwork $f(\cdot, m \odot \theta_i)$ with pruning mask $m$
3. Reset the remaining weights to their initial values $\theta_0$
4. Repeat the previous steps on $f(\cdot, m \odot \theta_0)$ until the desired sparsity level is reached

［#96］
We set the pruning rate per iteration to $p\% = 10\%$. Consistent with Chen et al. (2020), we only prune weights (and e.g. not biases) and exclude the embedding layer and the task-specific layer from the pruning process. In each fine-tuning iteration, we use the number of steps and parameter settings detailed in appendix B.

## F Additional Results

### F.1 Effect of Local Contrastive Editing on Gender Bias

［#97］
We show the effects of local contrastive editing on gender bias for additional sparsity levels in figures 8, 9 and 10. We observe that as the sparsity level increases, the impact of mask-based editing becomes more pronounced, whereas the influence of value-based editing on bias declines.

### F.2 Language Modeling Ability

［#98］
We present the language modeling ability (as measured by perplexity and LM score) of the discovered subnetworks at varying sparsity levels in table 4. Both, perplexity and LM score are comparable between stereotypical and anti-stereotypical subnetworks and remain similar across different sparsity levels.

［#99］
Table 5 presents the change in language modeling ability after local contrastive editing. We present the average changes in perplexity and LM score across both target models and four random seeds at different sparsity levels. We apply a one-sided Wilcoxon signed-rank test with Bonferroni correction to assess whether the observed changes (an increase in perplexity or a decrease in LM scores) are significant. Notably, across all sparsity levels, perplexity remains consistently low, with significant increases occurring occasionally after extrapolation with large absolute weighting factors, and consistently across all sparsities for value-based pruning. A similar trend is observed for LM score, with significant drops occurring only in the case of value-based pruning.

［#100］
| | full | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| Perplexity | 5.57 | 5.56 | 5.59 | 5.69 | 5.88 |
| LM score | 85.58 | 85.51 | 85.80 | 85.81 | 85.53 |

［#101］
(a) stereotypical subnetworks

［#102］
| | full | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| Perplexity | 5.57 | 5.58 | 5.60 | 5.68 | 5.90 |
| LM score | 85.88 | 85.63 | 85.89 | 85.78 | 85.51 |

［#103］
(b) anti-stereotypical subnetworks

［#104］
Table 4: Language modeling ability of subnetworks at different sparsities. We report the mean across all random seeds, where lower perplexity and higher LM scores indicate better performance in terms of language modeling.

### F.3 Downstream Tasks

［#105］
Tables 6 and 7 present the results of fine-tuning the edited models on the MNLI and STS-B tasks from the GLUE benchmark for different sparsity levels. The edited models exhibit only a slight decrease in performance compared to the base model, with the lowest performing models achieving 82.9/83.4 ($-1.8\%/ -1.2\%$) on MNLI and 86.6/86.24 ($-2.7\%/ -3.0\%$) on STS-B. We note that at each sparsity level, the models with the greatest performance drop were those subjected to value-based pruning prior to fine-tuning on the downstream task.

### F.4 Sensitivity to the Number of Weights Edited

［#106］
Figures 11, 12 and 13 present the results of experiments exploring various numbers of weights selected for interpolation for additional sparsity levels, using a fixed weighting factor of $\alpha = 0.5$. We observe similar trends across all bias measures and sparsity levels, indicating that editing $20\% - 40\%$ of the weights has an equivalent effect on bias as editing all the weights.

### F.5 Sensitivity to the Weighting Factor

［#107］
We display results exploring different weighting factors for interpolation and extrapolation across

［#108］
![](./images/1056453697359314955_8.jpg)

［#109］
Figure 8: WEAT average effect size after local contrastive editing. We report the mean bias at different sparsity levels across four random seeds with error bars indicating one standard deviation in each direction.

［#110］
![](./images/1056453697359314955_9.jpg)

［#111］
Figure 9: StereoSet stereotype scores after local contrastive editing. We illustrate the mean bias at different sparsity levels across four random seeds with error bars indicating one standard deviation in each direction.

［#112］
different sparsity levels. The effect on gender bias is illustrated in figures 14, 15 and 16, while the effect on language modeling performance is shown in figures 17 and 18. We observe that across all sparsity levels and bias measures, the effect of mask-based and value-based editing on bias increases with higher absolute weighting factors (up to a certain threshold), whereas uninformed editing does not lead to any or only minor changes in bias. At the same time, perplexity and LM score indicate increasingly worse language modeling performance for higher absolute weighting factors.

## E.6 Application to a Neutral Model

［#113］
We show the effect of local contrastive editing on gender bias for a neutral target model at different sparsity levels in figures 19, 20 and 21. We record the change in language modeling ability in table 8.

［#114］
![](./images/1056453697359314955_10.jpg)

［#115］
Figure 10: CrowS-Pairs stereotype scores after local contrastive editing. We show the mean bias at different sparsity levels across four random seeds with error bars indicating one standard deviation in each direction.

［#116］
<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="4">$\Delta$ perplexity $\downarrow$</th>
<th colspan="4">$\Delta$ LM score $\uparrow$</th>
</tr>
<tr>
<th>10%</th>
<th>20%</th>
<th>30%</th>
<th>40%</th>
<th>10%</th>
<th>20%</th>
<th>30%</th>
<th>40%</th>
</tr>
</thead>
<tbody>
<tr>
<td>Mask-based localization</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>IP ($\alpha=0.5$)</td>
<td>-0.002</td>
<td>+0.001</td>
<td>+0.000</td>
<td>+0.007</td>
<td>+0.012</td>
<td>-0.010</td>
<td>+0.012</td>
<td>-0.009</td>
</tr>
<tr>
<td>IP ($\alpha=1$)</td>
<td>-0.003</td>
<td>+0.001</td>
<td>+0.012</td>
<td><b>+0.017</b></td>
<td>-0.080</td>
<td>-0.045</td>
<td>-0.080</td>
<td>-0.038</td>
</tr>
<tr>
<td>EP ($\alpha=2$)</td>
<td>+0.006</td>
<td>+0.014</td>
<td><b>+0.041</b></td>
<td><b>+0.072</b></td>
<td>+0.005</td>
<td>-0.016</td>
<td>+0.005</td>
<td>+0.015</td>
</tr>
<tr>
<td>EP ($\alpha=-2$)</td>
<td>+0.002</td>
<td>+0.016</td>
<td><b>+0.051</b></td>
<td><b>+0.137</b></td>
<td>-0.023</td>
<td>-0.050</td>
<td>-0.023</td>
<td>-0.099</td>
</tr>
<tr>
<td>PR</td>
<td>-0.003</td>
<td>+0.010</td>
<td>+0.014</td>
<td>+0.014</td>
<td>-0.005</td>
<td>-0.023</td>
<td>-0.005</td>
<td>-0.017</td>
</tr>
<tr>
<td>SW</td>
<td>+0.000</td>
<td>+0.010</td>
<td>+0.008</td>
<td>+0.019</td>
<td>-0.055</td>
<td>-0.061</td>
<td>-0.055</td>
<td>-0.023</td>
</tr>
<tr>
<td>Value-based localization ($k=10\%$)</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>IP ($\alpha=0.5$)</td>
<td>-0.003</td>
<td>+0.006</td>
<td>+0.008</td>
<td>+0.002</td>
<td>-0.004</td>
<td>+0.013</td>
<td>-0.004</td>
<td>-0.010</td>
</tr>
<tr>
<td>IP ($\alpha=1$)</td>
<td>+0.010</td>
<td>+0.013</td>
<td>+0.015</td>
<td><b>+0.016</b></td>
<td>-0.076</td>
<td>-0.017</td>
<td>-0.076</td>
<td>+0.005</td>
</tr>
<tr>
<td>EP ($\alpha=2$)</td>
<td><b>+0.051</b></td>
<td><b>+0.032</b></td>
<td><b>+0.035</b></td>
<td><b>+0.032</b></td>
<td>-0.147</td>
<td>-0.048</td>
<td>-0.147</td>
<td>+0.020</td>
</tr>
<tr>
<td>EP ($\alpha=-2$)</td>
<td>+0.031</td>
<td>+0.020</td>
<td>+0.025</td>
<td>+0.017</td>
<td>+0.005</td>
<td>-0.233</td>
<td>+0.005</td>
<td>+0.059</td>
</tr>
<tr>
<td>PR</td>
<td><b>+9.225</b></td>
<td><b>+10.18</b></td>
<td><b>+9.529</b></td>
<td><b>+11.722</b></td>
<td><b>-2.724</b></td>
<td><b>-3.500</b></td>
<td><b>-2.723</b></td>
<td><b>-0.965</b></td>
</tr>
</tbody>
</table>

［#117］
Table 5: Change in language modeling ability. We show the mean change in perplexity and LM score after local contrastive editing across both target models (stereotypical and anti-stereotypical) and four random seeds at different sparsity levels. We print significant differences bold.

［#118］
![](./images/1056453697359314955_11.jpg)

［#119］
Figure 11: WEAT average effect size for different numbers of weights edited. We report the mean bias at different sparsity levels across four random seeds with error bars indicating one standard deviation in each direction.

［#120］
<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="3">MNLI-m/mm $\uparrow$</th>
<th colspan="3">STS-B $\uparrow$</th>
</tr>
<tr>
<th>10%</th>
<th>20%</th>
<th>40%</th>
<th>10%</th>
<th>20%</th>
<th>40%</th>
</tr>
<tr>
<th>Base</th>
<th colspan="3">84.4/84.8</th>
<th colspan="3">89.0/88.9</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7">Mask-based loc.</td>
</tr>
<tr>
<td>IP ($\alpha=0.5$)</td>
<td>84.6/84.5</td>
<td>84.4/84.6</td>
<td>84.1/84.0</td>
<td>88.7/88.3</td>
<td>88.7/88.4</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>IP ($\alpha=1$)</td>
<td>84.5/84.5</td>
<td>84.3/84.5</td>
<td>84.0/84.1</td>
<td>88.7/88.3</td>
<td>88.8/88.4</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=2$)</td>
<td>84.6/84.6</td>
<td>84.3/84.5</td>
<td>84.1/84.0</td>
<td>88.7/88.3</td>
<td>88.8/88.3</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=-2$)</td>
<td>84.6/84.6</td>
<td>84.3/84.5</td>
<td>83.9/84.0</td>
<td>89.0/88.7</td>
<td>89.0/88.5</td>
<td>88.1/87.6</td>
</tr>
<tr>
<td>PR</td>
<td>84.6/84.6</td>
<td>84.0/84.4</td>
<td>84.0/84.1</td>
<td>88.8/88.4</td>
<td>88.8/88.4</td>
<td>88.3/87.7</td>
</tr>
<tr>
<td>SW</td>
<td>84.7/84.6</td>
<td>84.2/84.5</td>
<td>84.0/84.2</td>
<td>88.8/88.4</td>
<td>88.9/88.4</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td colspan="7">Value-based loc. (k=10%)</td>
</tr>
<tr>
<td>IP ($\alpha=0.5$)</td>
<td>84.5/84.5</td>
<td>84.4/84.4</td>
<td>84.0/84.0</td>
<td>88.9/88.5</td>
<td>88.8/88.4</td>
<td>88.2/87.7</td>
</tr>
<tr>
<td>IP ($\alpha=1$)</td>
<td>84.4/84.6</td>
<td>84.2/84.4</td>
<td>84.0/84.1</td>
<td>88.8/88.4</td>
<td>88.8/88.3</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=2$)</td>
<td>84.5/84.6</td>
<td>84.5/84.3</td>
<td>84.0/84.0</td>
<td>88.7/88.3</td>
<td>88.8/88.3</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=-2$)</td>
<td>84.5/84.5</td>
<td>84.2/84.5</td>
<td>84.0/84.1</td>
<td>88.9/88.5</td>
<td>88.9/88.5</td>
<td>88.3/87.7</td>
</tr>
<tr>
<td>PR</td>
<td>83.8/84.0</td>
<td>83.5/84.0</td>
<td>83.2/83.5</td>
<td>87.6/87.3</td>
<td>86.6/86.2</td>
<td>87.2/86.8</td>
</tr>
</tbody>
</table>

［#121］
Table 6: Performance of the edited stereotypical models on downstream tasks. We fine-tune the edited models on the MNLI and STS-B tasks from the GLUE benchmark and show the results for the stereotypical target model at different sparsity levels using a single random seed. For MNLI, we report both matched and mismatched accuracy while for STS-B we, present Pearson and Spearman correlation. We emphasize the worst result per task and sparsity level.

［#122］
<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="4">MNLI-m/mm $\uparrow$</th>
<th colspan="4">STS-B $\uparrow$</th>
</tr>
<tr>
<th>10%</th>
<th>20%</th>
<th>30%</th>
<th>40%</th>
<th>10%</th>
<th>20%</th>
<th>30%</th>
<th>40%</th>
</tr>
<tr>
<th>Base</th>
<th colspan="4">84.4/84.8</th>
<th colspan="4">89.0/88.9</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">Mask-based loc.</td>
</tr>
<tr>
<td>IP ($\alpha=0.5$)</td>
<td>84.5/84.7</td>
<td>84.2/84.4</td>
<td>84.0/84.4</td>
<td>83.9/84.0</td>
<td>88.6/88.2</td>
<td>88.8/88.3</td>
<td>88.5/87.9</td>
<td>88.1/87.6</td>
</tr>
<tr>
<td>IP ($\alpha=1$)</td>
<td>84.5/84.5</td>
<td>84.2/84.3</td>
<td>83.9/84.3</td>
<td>84.1/84.0</td>
<td>88.6/88.2</td>
<td>88.8/88.3</td>
<td>88.4/87.9</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=2$)</td>
<td>84.8/84.7</td>
<td>84.2/84.3</td>
<td>83.9/84.2</td>
<td>83.9/84.1</td>
<td>88.7/88.4</td>
<td>88.9/88.4</td>
<td>88.4/87.9</td>
<td>88.1/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=-2$)</td>
<td>84.5/84.7</td>
<td>84.4/84.5</td>
<td>83.7/84.3</td>
<td>84.2/84.0</td>
<td>88.5/88.1</td>
<td>88.7/88.2</td>
<td>88.3/87.8</td>
<td>88.1/87.5</td>
</tr>
<tr>
<td>PR</td>
<td>84.5/84.5</td>
<td>84.4/84.6</td>
<td>83.9/84.4</td>
<td>83.9/84.0</td>
<td>88.6/88.2</td>
<td>88.8/88.3</td>
<td>88.3/87.8</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>SW</td>
<td>84.5/84.5</td>
<td>84.2/84.4</td>
<td>84.1/84.4</td>
<td>83.9/83.9</td>
<td>88.6/88.3</td>
<td>88.8/88.3</td>
<td>88.5/87.9</td>
<td>88.1/87.6</td>
</tr>
<tr>
<td colspan="9">Value-based loc. (k=10%)</td>
</tr>
<tr>
<td>IP ($\alpha=0.5$)</td>
<td>84.6/84.5</td>
<td>84.3/84.1</td>
<td>83.4/84.4</td>
<td>83.8/83.8</td>
<td>88.6/88.2</td>
<td>88.8/88.3</td>
<td>88.6/88.0</td>
<td>88.1/87.5</td>
</tr>
<tr>
<td>IP ($\alpha=1$)</td>
<td>84.6/84.6</td>
<td>84.3/84.6</td>
<td>84.2/84.4</td>
<td>83.9/83.8</td>
<td>88.7/88.3</td>
<td>88.8/88.3</td>
<td>88.5/88.0</td>
<td>88.1/87.5</td>
</tr>
<tr>
<td>EP ($\alpha=2$)</td>
<td>84.5/84.7</td>
<td>84.1/84.4</td>
<td>83.9/84.4</td>
<td>83.9/84.1</td>
<td>88.7/88.3</td>
<td>88.9/88.4</td>
<td>88.6/88.1</td>
<td>88.2/87.6</td>
</tr>
<tr>
<td>EP ($\alpha=-2$)</td>
<td>84.7/84.7</td>
<td>84.1/84.6</td>
<td>83.8/84.4</td>
<td>84.0/84.1</td>
<td>88.5/88.1</td>
<td>88.5/87.9</td>
<td>88.3/87.7</td>
<td>88.0/87.4</td>
</tr>
<tr>
<td>PR</td>
<td>83.8/83.8</td>
<td>83.5/83.9</td>
<td>83.5/83.8</td>
<td>82.9/83.4</td>
<td>87.7/87.3</td>
<td>86.7/86.2</td>
<td>87.1/86.7</td>
<td>87.2/86.7</td>
</tr>
</tbody>
</table>

［#123］
Table 7: Performance of the edited anti-stereotypical models on downstream tasks. We fine-tune the edited models on the MNLI and STS-B tasks from the GLUE benchmark and show the results for the anti-stereotypical target model at different sparsity levels using a single random seed. For MNLI, we report both matched and mismatched accuracy while for STS-B we, present Pearson and Spearman correlation.

［#124］
![](./images/1056453697359314955_12.jpg)

［#125］
Figure 12: StereoSet stereotype scores for different numbers of weights edited. We report the mean bias at different sparsity levels across four random seeds with error bars indicating one standard deviation in each direction.

［#126］
![](./images/1056453697359314955_13.jpg)

［#127］
Figure 13: CrowS-Pairs stereotype scores for different numbers of weights edited. We report the mean bias at different sparsity levels across four random seeds with error bars indicating one standard deviation in each direction.

［#128］
![](./images/1056453697359314955_14.jpg)

［#129］
Figure 14: WEAT average effect size for different weighting factors. We set the number of edited weights to the number of weights selected by masked-based localization and report the mean bias across four random seeds.

［#130］
![](./images/1056453697359314955_15.jpg)

［#131］
Figure 15: StereoSet stereotype scores for different weighting factors. We set the number of edited weights to the number of weights selected by masked-based localization and report the mean bias across four random seeds.

［#132］
![](./images/1056453697359314955_16.jpg)

［#133］
Figure 16: CrowS-Pairs stereotype scores for different weighting factors. We set the number of edited weights to the number of weights selected by masked-based localization and report the mean bias across four random seeds.

［#134］
![](./images/1056453697359314955_17.jpg)

［#135］
Figure 17: Change in perplexity for different weighting factors. We set the number of edited weights to the number of weights selected by masked-based localization and report the mean perplexity over both target models (stereotypical and anti-stereotypical) and across four random seeds.

［#136］
![](./images/1056453697359314955_18.jpg)

［#137］
Figure 18: Change in LM score for different weighting factors. We set the number of edited weights to the number of weights selected by masked-based localization and report the mean score over both target models (stereotypical and anti-stereotypical) and across four random seeds.

［#138］
![](./images/1056453697359314955_19.jpg)

［#139］
Figure 19: WEAT average effect size after local contrastive editing with a neutral target model. We report the mean bias across four random seeds with error bars indicating one standard deviation in each direction.

［#140］
![](./images/1056453697359314955_20.jpg)

［#141］
Figure 20: StereoSet stereotype scores after local contrastive editing with a neutral target model. We report the mean bias across four random seeds with error bars indicating one standard deviation in each direction.

［#142］
![](./images/1056453697359314955_21.jpg)

［#143］
Figure 21: CrowS-Pairs stereotype scores after local contrastive editing with a neutral target model. We report the mean bias across four random seeds with error bars indicating one standard deviation in each direction.

［#144］
<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="4">$\Delta$ perplexity $\downarrow$</th>
<th colspan="4">$\Delta$ LM score $\uparrow$</th>
</tr>
<tr>
<th>10%</th>
<th>20%</th>
<th>30%</th>
<th>40%</th>
<th>10%</th>
<th>20%</th>
<th>30%</th>
<th>40%</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">Mask-based localization</td>
</tr>
<tr>
<td>IP ($\alpha = 0.5$)</td>
<td>+0.005</td>
<td>-0.005</td>
<td>+0.011</td>
<td>+0.013</td>
<td>+0.076</td>
<td>+0.238</td>
<td>-0.072</td>
<td>+0.062</td>
</tr>
<tr>
<td>IP ($\alpha = 1$)</td>
<td>+0.012</td>
<td>+0.012</td>
<td><b>+0.042</b></td>
<td><b>+0.055</b></td>
<td>+0.030</td>
<td>+0.164</td>
<td>-0.160</td>
<td>-0.073</td>
</tr>
<tr>
<td>EP ($\alpha = 2$)</td>
<td><b>+0.025</b></td>
<td><b>+0.065</b></td>
<td><b>+0.157</b></td>
<td><b>+0.310</b></td>
<td>+0.109</td>
<td>+0.027</td>
<td>-0.250</td>
<td>-0.180</td>
</tr>
<tr>
<td>EP ($\alpha = -2$)</td>
<td><b>+0.027</b></td>
<td><b>+0.076</b></td>
<td><b>+0.221</b></td>
<td><b>+0.534</b></td>
<td><b>-0.419</b></td>
<td>-0.371</td>
<td>-0.122</td>
<td>-0.321</td>
</tr>
<tr>
<td>PR</td>
<td>+0.006</td>
<td>+0.009</td>
<td><b>+0.028</b></td>
<td><b>+0.040</b></td>
<td>+0.003</td>
<td>+0.059</td>
<td>-0.156</td>
<td>+0.0067</td>
</tr>
<tr>
<td>SW</td>
<td><b>+0.011</b></td>
<td><b>+0.015</b></td>
<td><b>+0.041</b></td>
<td><b>+0.059</b></td>
<td>+0.057</td>
<td>+0.179</td>
<td>-0.184</td>
<td>-0.057</td>
</tr>
<tr>
<td colspan="9">Value-based localization ($k = 10\%$)</td>
</tr>
<tr>
<td>IP ($\alpha = 0.5$)</td>
<td>+0.006</td>
<td>-0.001</td>
<td>+0.011</td>
<td>+0.005</td>
<td>-0.149</td>
<td>+0.140</td>
<td>-0.177</td>
<td>-0.081</td>
</tr>
<tr>
<td>IP ($\alpha = 1$)</td>
<td>+0.004</td>
<td>+0.005</td>
<td><b>+0.017</b></td>
<td>+0.018</td>
<td>-0.256</td>
<td>+0.138</td>
<td>-0.195</td>
<td><b>-0.177</b></td>
</tr>
<tr>
<td>EP ($\alpha = 2$)</td>
<td><b>+0.036</b></td>
<td><b>+0.031</b></td>
<td><b>+0.043</b></td>
<td><b>+0.063</b></td>
<td>-0.393</td>
<td>-0.085</td>
<td>-0.141</td>
<td><b>-0.393</b></td>
</tr>
<tr>
<td>EP ($\alpha = -2$)</td>
<td><b>+0.065</b></td>
<td><b>+0.050</b></td>
<td><b>+0.069</b></td>
<td><b>+0.070</b></td>
<td>+0.153</td>
<td>-0.001</td>
<td>-0.053</td>
<td>+0.090</td>
</tr>
<tr>
<td>PR</td>
<td><b>+4.826</b></td>
<td><b>+5.873</b></td>
<td><b>+5.194</b></td>
<td><b>+6.215</b></td>
<td><b>-2.682</b></td>
<td><b>-1.957</b></td>
<td><b>-1.100</b></td>
<td>-0.995</td>
</tr>
</tbody>
</table>

［#145］
Table 8: Change in language modeling ability after local contrastive editing with a neutral target model. We show the mean change in perplexity and LM scores after local contrastive editing across both reference models (stereotypical and anti-stereotypical) and four random seeds at different sparsity levels. We print significant differences bold.