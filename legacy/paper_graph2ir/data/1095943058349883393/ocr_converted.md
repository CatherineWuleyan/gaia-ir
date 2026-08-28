# In Praise of Stubbornness:
## The Case for Cognitive-Dissonance-Aware Knowledge Updates in LLMs

［#1］
Simone Clemente $^{*1}$ Zied Ben Houidi $^{*1\dagger}$ Alexis Huet $^{1}$ Dario Rossi $^{1}$ Giulio Franzese $^{2}$ Pietro Michiardi $^{2}$

### Abstract
［#2］
Despite remarkable capabilities, large language models (LLMs) struggle to continually update their knowledge without catastrophic forgetting. In contrast, humans effortlessly integrate new information, detect conflicts with existing beliefs, and selectively update their mental models. This paper introduces a cognitive-inspired investigation paradigm to study continual knowledge updating in LLMs. We implement two key components inspired by human cognition: (1) *Dis- sonance and Familiarity Awareness*, analyzing model behavior to classify information as novel, familiar, or dissonant; and (2) *Targeted Network Updates*, which track neural activity to identify frequently used (stubborn) and rarely used (plas- tic) neurons. Through carefully designed experiments in controlled settings, we uncover a number of empirical findings demonstrating the potential of this approach. First, dissonance detection is feasible using simple activation and gradient features, suggesting potential for cognitive-inspired training. Second, we find that non-dissonant updates largely preserve prior knowledge regardless of targeting strategy, revealing inherent robustness in LLM knowledge integration. Most critically, we discover that dissonant updates prove catastrophically destructive to the model's knowledge base, indiscriminately affecting even information unrelated to the current updates. This suggests fundamental limitations in how neural networks handle contradictions and motivates the need for new approaches to knowledge updating that better mirror human cognitive mechanisms.

---
［#3］
$^{*}$Equal contribution $^{\dagger}$Principal investigator $^{1}$Huawei Technologies Co. Ltd. $^{2}$EURECOM, Sophia Antipolis, France. Correspondence to: Zied Ben Houidi <zied.ben.houidi@huawei.com>.

［#4］
Preprint.

## 1. Introduction
［#5］
Humans effortlessly update their knowledge as they experience the world. They seamlessly integrate new information, ignore redundant stimuli, and actively resolve conflicts with existing beliefs before updating their mental models. This cognitive flexibility stems from several key abilities. Humans exhibit (1) *selective attention*, focusing on novel or relevant information while filtering out irrelevant or familiar stimuli (Posner et al., 1990; Petersen & Posner, 2012; Desimone et al., 1995; Ranganath & Rainer, 2003). They readily (2) *detect conflicts* (Croyle & Cooper, 1983) between new information and existing knowledge and actively engage in resolving them, a process known in psychology as cognitive dissonance (Festinger, 1957; Van Veen et al., 2009). Moreover, their brains exhibit a form of (3) *adaptive plasticity*, allowing for updates to neural networks that can incorporate new information while often preserving existing knowledge. While the exact mechanisms are still being investigated, this process seems to balance the stability of well-established knowledge with flexibility in the face of new or uncertain information (McClelland et al., 1995; Behrens et al., 2007).

［#6］
Despite demonstrating remarkable capabilities, Large Language Models (LLMs) are still far from such learning abilities. Current LLMs face significant challenges in real-world deployment and long-term utility due to their static nature and training paradigms. They suffer from catastrophic forgetting (Kirkpatrick et al., 2017a; Kemker et al., 2018; Li et al., 2022; Luo et al., 2024; Kotha et al., 2024), where incorporating new information often leads to the erasure of previously learned knowledge. Furthermore, LLMs engage during training in indiscriminate learning, passively accepting all training data, even when it contradicts what they already learned. Despite emergent sparsity (Jaiswal et al., 2023; Mirzadeh et al., 2024), knowledge in LLMs follows backpropagation and the objective function, with no explicit mechanism for targeted knowledge storage or retrieval. This results in a situation where all weights are potential candidates for storing knowledge, necessitating comprehensive retraining to properly incorporate new information.

［#7］
In this work, we embark on a systematic empirical investigation of how LLMs handle knowledge updates, drawing inspiration from human cognitive traits. Through carefully

# The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#8］
<table>
  <caption>Table 1. Taxonomy of Incremental Learning Approaches. See Appendix.A for an extended version</caption>
  <thead>
    <tr>
      <th>Examples</th>
      <th>Incremental Type</th>
      <th>Memory Usage</th>
      <th>Task Awareness</th>
      <th>Weight Plasticity</th>
      <th>Architecture</th>
      <th>Conflict Detection</th>
      <th>Update Mechanism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>iCaRL (Rebuffi et al., 2017)</td>
      <td>Class-incremental</td>
      <td>Replay</td>
      <td>Task-Agnostic</td>
      <td>Fixed</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Replay</td>
    </tr>
    <tr>
      <td>EWC (Kirkpatrick et al., 2017b)</td>
      <td>Task-incremental</td>
      <td>None</td>
      <td>Task-Aware</td>
      <td>Selective</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Regularization</td>
    </tr>
    <tr>
      <td>Progressive Nets (Rusu et al., 2016)</td>
      <td>Task-incremental</td>
      <td>None</td>
      <td>Task-Aware</td>
      <td>Fixed</td>
      <td>Expanding</td>
      <td>No</td>
      <td>New Subnetworks</td>
    </tr>
    <tr>
      <td>DEN (Yoon et al., 2017)</td>
      <td>Task-incremental</td>
      <td>None</td>
      <td>Task-Aware</td>
      <td>Selective</td>
      <td>Expanding</td>
      <td>No</td>
      <td>Selective Expansion</td>
    </tr>
    <tr>
      <td>GEM (Lopez-Paz & Ranzato, 2017)</td>
      <td>Task-incremental</td>
      <td>Replay</td>
      <td>Task-Aware</td>
      <td>Constrained</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Constrained Optimization</td>
    </tr>
    <tr>
      <td>ROME (De Cao et al., 2021)</td>
      <td>Fact-incremental</td>
      <td>None</td>
      <td>Fact-Aware</td>
      <td>Localized</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Rank-One Update</td>
    </tr>
    <tr>
      <td>OWM (Zeng et al., 2019)</td>
      <td>Task-incremental</td>
      <td>None</td>
      <td>Task-Aware</td>
      <td>Orthogonal</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Orthogonal Projection</td>
    </tr>
    <tr>
      <td>PackNet (Mallya & Lazebnik, 2018)</td>
      <td>Task-incremental</td>
      <td>None</td>
      <td>Task-Aware</td>
      <td>Selective</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Weight Masking</td>
    </tr>
    <tr>
      <td>HAT (Serra et al., 2018)</td>
      <td>Task-incremental</td>
      <td>None</td>
      <td>Task-Aware</td>
      <td>Selective</td>
      <td>Fixed</td>
      <td>No</td>
      <td>Attention Masking</td>
    </tr>
    <tr>
      <td>This paper</td>
      <td>Fact-incremental</td>
      <td>None</td>
      <td>Conflict-Aware</td>
      <td>Selective</td>
      <td>Fixed</td>
      <td>Yes</td>
      <td>Neuron-Specific Update</td>
    </tr>
  </tbody>
</table>

［#9］
controlled experiments, we examine (1) the feasibility of *Dissonance Awareness*, i.e. whether it is possible to correctly classify facts into novel, familiar, and dissonant using features extracted from the LLM. We also investigate the benefits of (2) *Adaptive Plasticity* by studying how different neuron targeting strategies affect knowledge retention and update. For this, we develop a simple method for tracking historical neuron usage to identify "plastic" (rarely used) and "stubborn" (previously used) neurons, allowing us to study how knowledge updates affect different regions of the model's parameter space. This experimental framework lets us systematically investigate fundamental properties of knowledge integration in LLMs.

［#10］
Our investigation reveals a fundamental distinction in how LLMs handle knowledge updates: the case of non-dissonant updates (adding entirely new knowledge) versus dissonant updates (modifying existing associations). While prior work has focused mostly on editing individual factual associations (Meng et al., 2022a; Mitchell et al., 2022; Meng et al., 2022c) or preserving knowledge across distinct tasks as in continual learning (Rebuffi et al., 2017; Kirkpatrick et al., 2017b; Mallya & Lazebnik, 2018), our controlled experiments take a different approach. We systematically study how the placement of new knowledge in the network's parameter space affects both the integration of that knowledge and its impact on existing, unrelated knowledge. As shown in Tab. 1, this positions our work uniquely: rather than proposing new editing or continual learning methods, we reveal fundamental properties about how LLMs handle knowledge integration in both dissonant and non-dissonant scenarios. Critically, our experimental design allows us to precisely track the impact of updates on a controlled set of initial knowledge, providing clear visibility into how different update strategies affect unrelated information.

［#11］
Key takeaways. This leads us to uncover several fundamental properties of LLM knowledge updating: (i) *dissonance awareness* is feasible using simple model features, suggesting potential for cognitive-inspired training; (ii) LLMs show inherent robustness when incorporating non-dissonant information, largely preserving prior knowledge regardless of targeting strategy; (iii) avoiding heavily-used (stubborn) neurons during updates further improves this robustness, motivating *adaptive plasticity* in this scenario; (iv) regions of the network heavily used during pre-training are particularly effective at incorporating new knowledge, extending lottery ticket hypothesis findings (Frankle & Carbin, 2019) to language models; and most critically, (v) dissonant updates prove catastrophically destructive to unrelated knowledge, suggesting fundamental limitations in how neural networks handle contradictions: while some of our targeted update strategies show comparable performance to existing editing methods like ROME and MEMIT, all approaches fundamentally struggle with dissonant updates, suggesting the need for fundamentally different mechanisms.

［#12］
Implications. These findings point to concrete opportunities such as the feasibility of dissonance awareness, and the benefits of adaptive plasticity in case of non-dissonant updates. But they also reveal fundamental challenges when handling contradictory information. Current approaches essentially attempt to erase and replace old knowledge - a process we show leads to catastrophic forgetting of even unrelated information. But this contrasts sharply with human cognition, where we maintain both old and new knowledge with appropriate temporal context. Consider how humans handled learning that Pluto was no longer classified as a planet: rather than erasing our previous understanding, we maintained both pieces of knowledge, understanding their historical context and why the classification changed. Our experiments motivate the exploration of future fundamentally different mechanisms for handling contradictions - ones that can maintain and contextualize conflicting information rather than attempting to overwrite it. $^1$

## 2. Dissonance-aware Targeted Updates

［#13］
Our cognitively-inspired investigation begins with a striking empirical discovery. Fig. 2 reveals a fundamental distinction in how language models handle different types of updates:

［#13］
$^1$Code available at https://github.com/bendiogene/ConflictAwareLLM/

# The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#14］
![](./images/1095943058349883393_1.jpg)

［#15］
Figure 1. Overview of our emprical investigation pipeline (i) Classification pipeline to identify novel, familiar, and dissonant information using inner or output model features. (ii) Targeted update strategies for non-dissonant updates and (iii) dissonant updates.

［#16］
while non-dissonant information (green) can often be incorporated while preserving existing knowledge, dissonant updates (red) prove catastrophically destructive across all model scales and training approaches. This pattern emerged from our controlled experimental pipeline (Fig. 1): when training models on 2,000 initial facts followed by 1,000 new facts, we observe dramatic differences between non-dissonant and dissonant updates in both their ability to learn new information (y-axis) and preserve original knowledge (x-axis). The stark contrast between these two types of updates motivates our systematic methodology: if dissonant updates are inherently destructive, can we develop mechanisms to identify them before they occur? And can targeted plasticity strategies help protect existing knowledge? To answer these questions, we first develop methods to extract and track neural activity patterns (Sec. 2.1), which serve two purposes: enabling classification of incoming information as novel, familiar, or dissonant (Sec. 3.2), and informing targeted update strategies that carefully control where new knowledge is stored in the network (Sec. 2.3).

## 2.1. Extraction of historical activations and gradients
［#17］
We maintain an aggregate profile of neuronal activity by accumulating activations and gradients for each neuron at every training step. Specifically, for each neuron $n$ in the Transformer blocks—including feed-forward (MLP) layers and attention projections (Key, Query, Value matrices)—we compute $H\hat{G}_n$ , the cumulative *historical gradient* magnitude over time, and $H\hat{A}_n$ , the cumulative *historical activation* magnitude over time. To mitigate scale differences across layers, we also experiment with layer-wise normalization of activations and gradients before accumulation. Precise notation and more details are in Appendix B.

［#18］
This historical activity data enables us to classify neurons as "plastic" or "stubborn" based on their past usage (see Fig.3 for a visual illustration), which is useful for our targeted network updates. We use the historical data also to normalize the input features when classifying facts as we see next.

## 2.2. Dissonance and Novelty Awareness
［#19］
We cast our classification problem on three classes: for a given input sequence $X$, decide if it is *Novel* (e.g. could be integrated), *Familiar* (e.g. can be ignored), or *Dissonant* (likely requiring proper resolution).

［#20］
We design a simple classifier that leverages activation and gradient information to assess the nature of new information. For any input sequence $X$, we first perform a forward pass to obtain its *current activations* and a backward pass to obtain its *current gradients* (without updating the model weights). Since the goal is to assess feasibility using easy-to-compute features and lightweight methods that could be integrated into large-scale models, we extract for each layer the mean, standard deviation, minimum, maximum, and quartiles (Q1, Q2, Q3) of the activations and gradients, eventually first normalized by historical activations and gradients. We perform ablation studies to assess the importance of different features and employ feature importance analyses to understand which aspects contribute most to the classifier's performance. We evaluate our ability to classify facts in Sec. 3.2. Despite using simple classifiers like Random Forests and SVMs, we achieve high accuracy, opening the way for future integration of dissonance awareness into LLM training pipelines.

［#21］
\footnotetext{^3For full finetuning, models trained until convergence on new facts, while LoRA experiments used fixed hyperparameters across both conditions. This dual approach revealed two phenomena: (1) not shown in the figure, full finetuning needed twice as many epochs to learn dissonant information compared to non-dissonant and (2) under fixed conditions with LoRA, unlike non-dissonant facts, models struggled to learn dissonant information (lower y-axis values for red crosses) while still exhibiting the same catastrophic interference with existing knowledge (low x-axis values).}

［#22］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#23］
![](./images/1095943058349883393_2.jpg)

［#24］
Figure 2. Early demonstration of a fundamental pattern we discovered across model sizes: the catastrophic nature of dissonant updates compared to non-dissonant ones. Results shown for GPT-2 Small 2a, GPT-2 XL 2b, and GPT-J-6B 2c, comparing full fine-tuning (stars) and LoRA (crosses) approaches. The stark contrast between dissonant (red) and non-dissonant (green) updates persists across model scales and training methods³, motivating our systematic investigation into this fundamental challenge of knowledge integration.

［#25］
![](./images/1095943058349883393_3.jpg)

［#26］
Figure 3. Illustration of how we use historical neuron activity (here we show the distribution of cumulative gradients during a GPT-2-xl previous training) to identify localized areas where to store future knowledge, according to four strategies.

### 2.3. Targeted Neuron Updates

［#27］
Building upon the historical tracking of neural activity, we implement targeted network updates to study the *impact of knowledge placement location on new knowledge ingestion and past knowledge retention*. We design four main types of targeted updates, which we experimentally evaluate. During training on new information, we perform standard forward and backward passes to compute the loss and gradients. Before the optimizer step, we modify the gradients to freeze certain neurons. Specifically, given the gradients for all parameters of a given layer, we zero-out those that do not belong to the selected set of neuron and corresponding weights, defined as plastic, stubborn, candidate and specific, as described below. This process effectively freezes the weights of non-selected neurons, allowing for targeted updates to specific parts of the model. By varying the choice of selected neurons, we control how new information is integrated into the model while managing its impact on existing knowledge. Next, we introduce strategies to select which neurons and weights to update: Figure 3 illustrates the conceptual relationship between the various neuron updates strategies within the model's parameter space.

［#28］
**Plastic Neurons.** Neurons underutilized during past model updates. To identify them, we rank neurons by increasing historical gradient values and select the top $N$ neurons with the lowest cumulative gradients:
［#28］
$$
\mathcal{N}_{\text{plastic}} = \{n \mid \text{rank}(H\hat{G}_n) \leq N\},
$$
［#28］
where $H\hat{G}_n$ is the historical gradient for neuron $n$, accumulated over all prior training. This allows to assess whether targeting underutilized neurons can integrate new knowledge without interfereing with existing one.

［#29］
**Stubborn Neurons.** Neurons that accumulated high historical gradients, indicating significant involvement in previous learning. We rank neurons by decreasing historical gradient values and select the top $N$ neurons:
［#29］
$$
\mathcal{N}_{\text{stubborn}} = \{n \mid \text{rank}(H\hat{G}_n) > |\mathcal{N}| - N\},
$$
［#29］
where $|\mathcal{N}|$ is the total number of neurons, and $H\hat{G}_n$ is the historical gradient for neuron $n$. Updating stubborn neurons allows us to test the model's capacity for knowledge integration and assess the potential risks of overwriting existing information.

［#30］
**Candidate Neurons.** These neurons are relevant for encoding new information: to identify them, we perform a single back-propagation pass on the new input data, without updating the model weights. We then rank neurons based on the magnitude of these gradients and select the top $N$:
［#30］
$$
\mathcal{N}_{\text{candidate}} = \{n \mid \text{rank}(G_n^{\text{new}}) > |\mathcal{N}| - N\},
$$
［#30］
where $G_n^{\text{new}}$ is the gradient for neuron $n$ obtained from the back-propagation pass on the new input data. Targeting candidate neurons focuses updates on areas of the network that are most relevant to the new information, as suggested by the back-propagation process.

［#31］
**Specific Neurons.** In another strategy, we identify neurons from the candidate set that do not intersect with the stubborn, in an attempt to store new information efficiently while

［#32］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#32］
avoiding interference with existing knowledge. For this, we first: (1) identify stubborn neurons $\mathcal{N}_\text{stubborn}$, using $N$ as defined earlier; we next (2) rank all neurons based on the magnitude of their gradients $G_n^\text{new}$ obtained from a single back-propagation pass on the new data, without updating model weights; finally, (3) we select few specific neurons, by choosing the top $N$ neurons that are not in $\mathcal{N}_\text{stubborn}$:

［#32］
$$
\mathcal{N}_\text{specific} = \text{Top}_N(\mathcal{N}_\text{all} \setminus \mathcal{N}_\text{stubborn}),
$$

［#32］
where $\mathcal{N}_\text{all}$ is the set of all neurons ranked by their gradient magnitudes. This last approach ensures that we select neurons that are most relevant to the new information (high gradient) while explicitly avoiding those that are crucial for existing knowledge (stubborn neurons).

## 3. Experimental evaluation

［#33］
We discuss our experimental setup, to evaluate dissonance-awareness, as well as continual knowledge update.

### 3.1. Experimental setup

［#34］
**Dataset.** We use the COUNTERFACT dataset (Meng et al., 2022b) as our primary data source as it contains both facts and counterfacts.⁴ This dataset, with approximately 17,000 facts, allows us to test models' handling of conflicting knowledge and addition of potentially known information,⁵ two key aspects of our dissonance-aware approach. To address the lack of truly novel facts in COUNTERFACT, we generate additional data using GPT-3.5. We transform existing statements into plausible yet fictitious information, maintaining structural similarity while introducing novel content. For example, "Danielle Darrieux's mother tongue is French" becomes "Sylvan Myrthil's mother tongue is Sylvan" (see Appendix C.1 for details).

［#35］
For our dissonance awareness experiments, we construct a balanced dataset comprising 1,000 samples each of familiar, conflicting, and novel facts. When using the pre-trained GPT-2-small model, we adjust the familiar class to 600 samples due to the limited number of known facts extracted from the model's pre-training. For our targeted update experiments, we use 5-fold cross-validation varying each time the sets of old and new facts but keeping the following proportions: 2000 old facts vs. 1000 new facts. For conflicting updates, we also test with 10 and 100 new facts.

［#36］
**Models.** We employ models from the GPT family to ease comparison, focusing mostly on GPT-2-small and GPT-2-xl (and to lesser extent GPT-J 6B). The dataset size ($\simeq$17,000 facts) limits full stress-testing of larger models like GPT-2-xl (less visible catastrophic forgetting in case of non-dissonant updates compared to compressed models). As a result, some of the effects of our experiments are most clearly observed with GPT-2-small, on which we focus most in the main body of this paper, deferring GPT-2-xl results (which are aligned to GPT-2-small) to the Appendix.

［#37］
We implement experiments using Hugging Face Transformers on NVIDIA GPUs. Before setting learning rates and epochs, we conduct a search for optimal hyperparameters that allow effective learning of facts (see App. D.2 for an example for GPT-2-xl). We perform the search based on the ability to correctly learn 10,000 facts from the dataset. More details including results can be found in the code repository.

### 3.2. Dissonance awareness

［#38］
**Settings.** Our first goal is to evaluate the ability to discriminate *familiar, novel*, and *conflicting* information using the readily-available⁶ simple features we extract from the models during the forward and backward passes. As schematized in Fig. 1 (left), we do so by relying on simple classifiers (random forests and SVMs), contrasting two scenarios for the *input features*: (1) a GPT-2 model fine-tuned on 1000 facts (the knowns), and (2) a GPT-2 pre-trained model (using its 600 extracted known facts as known class samples).

［#39］
For each scenario, we compile a balanced dataset with equal examples per class (familiar, novel, conflicting). To create novel facts, we employ GPT-4 with carefully designed prompts, using the structure of known facts (subject, relation, object) as templates, replacing key elements with fictitious (but plausible) information. This method ensured structural similarity to known facts while maintaining novelty. Appendix C.1 provides detailed prompts and examples (full datasets will be made available upon acceptance).

［#40］
**Classification performance** We extracted activations (A) and gradients (G) as described in Appendix B, and experimenting with A, G and A+G as input feature sets, using raw (R), per-layer (L) and historical (H) normalization strategies. As classifiers, we employ Random Forest (RF) and Support Vector Machines (SVM), optimizing hyperparameters using Bayesian search with 5-fold cross-validation. For clarity, we report the best results for each combination in Table 2 (average and standard deviation accuracy over the 5-folds) and defer the full results and ablation study to Table 4 in the appendix for the interested reader.

［#41］
Models consistently achieve high performance. Using features from the finetuned model reaches as high as (99.5%),

［#42］
⁴While this dataset allows us to test models' handling of conflicting knowledge, we acknowledge its limitations in representing more complex real-world knowledge, a limitation which we plan to address in the future.
⁵For instance, general facts that pre-trained models likely were exposed to during training.
⁶We explore the use of model output-only features in Appendix. C.5 showing that using output probabilities as feature is also successful.

［#43］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#44］
<table>
<caption>Table 2. Classification Results</caption>
<thead>
<tr>
<th>Scenario</th>
<th>Classifier</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">Fine-tuned</td>
<td>SVM (A+G, H)</td>
<td>0.995 (0.001)</td>
</tr>
<tr>
<td>RF (A+G, R)</td>
<td>0.988 (0.001)</td>
</tr>
<tr>
<td rowspan="2">Pre-trained</td>
<td>SVM (A+G, H)</td>
<td>0.947 (0.004)</td>
</tr>
<tr>
<td>RF (A+G, R)</td>
<td>0.928 (0.012)</td>
</tr>
</tbody>
</table>

［#43］
but also using features from a pre-trained model still achieves decent performance (94.7%). Interestingly, combining activations and gradients consistently outperformed using either feature set alone, with a slight advantage of SVM over RF. Also, historical normalization helps SVM, but does not provide benefits for RF.

［#45］
Feature importance While a full analysis of feature explainability is outside the scope of this paper, we further seek to observe structural differences in the learning process by comparing feature importance, using the feature importance scores derived from the random forest algorithm.

［#46］
To further shed light on such difference, Fig. 4 opposes feature importance in both cases, focusing on Activation versus Gradient-related features. It turns out that in the finetuned scenario, gradient-based features are substantially more important. This is likely due to the fact that finetuning the models on these facts has somewhat overfit them leading to gradients that are more discriminative: e.g. a clearly null gradient for known facts and a clearly high one for unknown ones. For the pretrained scenario, however, which is the most likely case in a real case scenario, both activation and gradient features contribute significantly, suggesting that for long-term knowledge, both internal representations and learning dynamics should be mixed in order to achieve good classification. Appendix.C.3 expands this analysis by focusing on transformer block importance instead.

［#47］
Finally, deferred to the appendix, comparing the performance of different normalization strategies for the pre-trained model using both activations and gradients (Table 4), we found that although normalization slightly helps, historical normalization does not seem to be crucial, since it was only slightly helpful for Random Forest classifiers.

［#48］
Key findings Overall, despite the simplicity of our features, the results demonstrate the feasibility of distinguishing between familiar, novel, and conflicting information, even in the challenging case of using pre-trained models, providing the needed foundation for dissonance-aware updates, which we explore in the next experiments.

### 3.3. Non-dissonant updates

［#49］
Settings. We now investigate how LLMs handle non-dissonant updates using our different strategies as experimental tools. In our experiments, schematized in Fig. 1 (middle), we evaluate the incorporation of non-conflicting facts into our models. The pipeline consists of (i) training on 2,000 initial facts (old) while collecting historical gradients to identify stubborn neurons, (ii) simulating updates with 1,000 new facts (new) to collect current gradients for candidate identification and (iii) applying different neuron selection strategies to update the model with these 1,000 facts. Note that while we track 2,000 facts as proxy for old knowledge, this represents a smaller fraction of GPT-2-xl's total knowledge compared to GPT-2-small, limiting our visibility into effects on other untracked pre-trained knowledge.

［#50］
For our experiment below inspired by the lottery ticket hypothesis (Frankle & Carbin, 2018), we used 10,000 separate facts for gradient extraction before training a fresh model on the 2,000+1,000 facts described above.

［#51］
Results. Fig. 5 presents the accuracy of various neuron update strategies on old and new knowledge for GPT-2-small, including error bars representing standard deviations over five runs. We observe that simple fine-tuning leads to a slight degradation in performance on old knowledge, dropping to approximately 93% accuracy. In contrast, updating plastic neurons helps preserving old knowledge, with accuracy remaining above 98% even when using up to 20,000 neurons. Random neuron selection exhibits a similar behavior. However, using candidate or stubborn neurons results in slightly more degradation to old knowledge.

［#52］
Interestingly, the strategies differ also in their ability to ingest new knowledge. Plastic neurons preserve old knowledge but struggle to ingest new knowledge requiring comparatively more neurons. This contrasts with specific neurons which strike the best balance, learning new knowledge with fewer neurons while maintaining acceptable levels of old knowledge retention.

［#53］
An intriguing pattern is that targeting stubborn, candidate, or specific neurons allows the model to learn new knowledge using fewer parameters compared to targeting plastic neurons. This finding resonates with the existence of winning subnetworks, as suggested by the lottery ticket hypothesis (Frankle & Carbin, 2018). It implies that certain subnetworks within the model are more conducive to integrating new information, compared to others. We conduct further experiments that confirm this hypothesis in App D.1.

［#54］
Finally, we conducted the same experiments on GPT-2-xl, where our 2,000 tracked facts represent a much smaller portion of the model's knowledge. With this larger capacity, interference with tracked facts becomes naturally less likely, leading to all strategies showing even better preservation of our monitored knowledge, compared to GPT-2-small. As shown in Fig. 11 and Fig. 10 in the appendix, all strategies tend to preserve old unrelated knowledge; and differ, as in the case of GPT-2-small, in their ability to integrate new knowledge efficiently. Detailed analyses, including

［#55］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#56］
![](./images/1095943058349883393_4.jpg)

［#57］
(a) Finetuned model

［#58］
![](./images/1095943058349883393_5.jpg)

［#59］
(b) Pretrained model

［#60］
Figure 4. *Dissonance awareness.* Feature importance showing the higher importance of gradient-related features for finetuned models.

［#61］
![](./images/1095943058349883393_6.jpg)

［#62］
(a) Old Knowledge

［#63］
![](./images/1095943058349883393_7.jpg)

［#64］
(b) New Knowledge

［#65］
Figure 5. *Non-dissonant updates:* Old vs new knowledge for targeted updates on GPT-2-small (see Fig. 10 and 11 for GPT-2-XL)

［#65］
the effects of varying learning rates and neuron counts, are provided in Appendix D.3.

［#66］
Key Findings. LLMs show remarkable robustness when incorporating non-dissonant information, as long as heavily-used (stubborn) neurons are avoided. Updating plastic neurons helps preserving old knowledge but requires more parameters (or time) to achieve high accuracy on new knowledge. Targeting specific neurons offers a balanced approach, enabling efficient knowledge integration with minimal impact on existing information.

### 3.4. Dissonant Updates
［#67］
Settings. We examine how LLMs handle conflicting information as shown in Fig. 1 (right): after training on 2,000 old facts and 1,000 new facts, we introduce 1,000 conflicting updates contradicting previously learned new facts, applying our various strategies to study their impact on learning conflicting facts and preserving unrelated old knowledge.

［#68］
Results. We illustrate the performance of various strategies on GPT-2-small in Fig. 6. Again, old knowledge refers to the 2000 initial *completely unrelated* facts and new knowledge corresponds to the 1000 conflicting ones. Surprisingly, we find that dissonant updates are highly destructive to the retention of such unrelated knowledge, regardless of the neuron update strategy employed. Even when updating plastic neurons, which are presumed to be underutilized and thus less likely to interfere with existing knowledge, we observe significant degradation in the model's ability to recall old facts.

［#69］
Given the destructive nature of simultaneously adding 1,000 conflicting facts, we conducted additional experiments where we added only 100 and 10 facts. The results, detailed in Appendix E.1, indicate that while the impact on old knowledge retention is less severe when editing fewer facts, the destructive effect remains prominent. This reminds us the performance of state-of-the-art model editing methods such as ROME (Meng et al., 2022a) and MEMIT (Meng et al., 2022c) which also deteriorates when applied to multiple sequential edits, as opposed to single-edit evaluations typically reported in the literature. While our primary focus is not on developing new model editing techniques, we leverage `EasyEdit` (Wang et al., 2023) to benchmark the above existing methods under our same multi-fact experimental conditions (See App. E.2).

［#70］
Finally, we also performed experiments with GPT-2-xl un-

［#71］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#72］
![](./images/1095943058349883393_8.jpg)

［#73］
(a) Old Knowledge

［#74］
![](./images/1095943058349883393_9.jpg)

［#75］
(b) New Knowledge

［#76］
Figure 6. Dissonant updates: Impact of 1000 dissonant facts and GPT-2-small (see Fig. 13 for GPT-2-XL)

［#76］
der various conditions, deferred to Appendix E.4 for space constraints. Overall, similarly to the non-dissonant case, GPT-2-xl exhibits different scaling behaviors. Surprisingly though, even when not learning conflicting knowledge, and despite having much more parameters, *GPT-2-xl also experiences significant degradation in old knowledge retention*—further confirming the catastrophic nature of dissonant updates (See Fig. 13). As shown in Fig. 2, GPT-J6B also confirms the same tendency.

［#77］
Key Findings. Dissonant updates pose a significant challenge, as they are destructive to prior unrelated knowledge, regardless of model size and even when targeting unused neurons. This underscores the importance of dissonance awareness to detect and appropriately handle conflicting information during continual learning. Our results motivate the integration of dissonance classifiers directly into the update or training of large language models. Thus, developing dedicated conflict resolution methods remains an essential direction for future work.

## 4. Discussion and conclusions

### 4.1. Lessons learned

［#78］
Fundamental Properties of Knowledge Updates: Our results reveal a striking pattern: while non-dissonant updates show remarkable robustness, dissonant updates trigger severe corruption of unrelated knowledge - dropping accuracy below 60% even when modifying just 10-100 facts. This effect persists across our tested model scales and update strategies, suggesting a deeper challenge in how current neural architectures handle contradictory information.

［#79］
Feasibility of Dissonance Detection: LLMs encode clear signatures distinguishing between novel, familiar, and dissonant information. In our controlled experiments, simple classifiers achieve 95% accuracy with pre-trained models and 99% with finetuned models using either activation/gradient features or output probabilities. This suggests the potential for developing mechanisms to identify potentially problematic updates before they occur.

［#80］
Promise of differentiated plasticity: Avoiding heavily-used neurons during *non-dissonant* updates improves robustness, maintaining 98% accuracy on old knowledge (versus 93% with finetuning). Interestingly, neurons used during pre-training are particularly effective at integrating new knowledge, extending lottery ticket hypothesis findings (Frankle & Carbin, 2018) to language models.

### 4.2. Limitations and Future Directions

［#81］
Experimental Control vs. Scale: While our controlled experiments reveal fundamental properties of knowledge updating, investigating these phenomena in much larger models presents challenges as it is not straightforward to track the impact on their broader knowledge.

［#82］
Dataset Limitations: Our current findings rely on CounterFact-derived data with simple factual statements. Developing larger, more diverse datasets is essential for understanding how these properties generalize to more complex forms of knowledge and conflicts.

［#83］
Neuron Classification Metrics: While our analysis of neural plasticity uses gradient magnitudes effectively, future work could explore richer metrics incorporating activation patterns and network connectivity to better understand knowledge distribution and update mechanisms.

［#84］
Beyond Binary Dissonance: Our current investigation treats dissonance as binary, while real-world knowledge updates often involve varying degrees of conflict and different types of knowledge. Understanding how these nuances affect knowledge integration remains a challenge.

［#85］
Towards Human-Inspired Updates: The catastrophic nature of dissonant updates suggests we may need fundamentally different approaches to LLM training. Rather than attempting to overwrite existing knowledge, future work might explore mechanisms for maintaining and contextualizing potentially conflicting information - similar to how humans maintain both historical and updated knowledge with appropriate contexts.

# The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

## Impact Statement
［#86］
Our work reveals a fundamental and concerning limitation of current AI systems: when LLMs encounter contradictory information, they suffer catastrophic corruption of unrelated knowledge, even in large-scale models. This vulnerability contrasts sharply with human cognition's remarkable flexibility in handling contradictory information. Interestingly, this distinction might hint at why human cognition seems to favor accumulation and contextualization of conflicting knowledge (e.g., "before" vs "after"), despite the tension it creates, rather than direct overwriting. This cognitive pattern *might* suggest that direct knowledge editing poses inherent risks that even evolutionary processes had to work around.

［#87］
*This finding has critical implications for AI deployment:*
Any AI system operating in the real world will inevitably encounter contradictory information, whether through natural knowledge evolution (e.g., medical guidelines updates) or potential adversarial attacks (e.g., coordinated misinformation campaigns). *Our work demonstrates that such contradictions don't just affect related knowledge - they can corrupt the system's broader knowledge base in unpredictable ways.* Another question is whether this also applies for value alignment in case of attempts to update an AI system's learned values or ethical principles.

［#88］
These findings motivate our development of dissonance detection capabilities as a crucial safety mechanism for deployed AI systems. More broadly, they suggest we may need to fundamentally rethink AI architectures to develop systems that, like humans, maintain and contextualize potentially conflicting information rather than attempting to overwrite it. This might require moving away from current approaches that try to "edit" neural networks toward architectures that can accumulate and contextualize knowledge while maintaining multiple, temporally-organized versions of truth.

## References
























































## Appendix

［#89］
We now report extended material concerning the extended related work (Appendix A), the extraction of historical activations and gradients (Appendix B), as well as detailed results on dissonance awareness (Appendix C), non-dissonant updates (Appendix D) and dissonant updates (Appendix E).

## A. Extended Related work

［#90］
In this section, we provide an extended version of Tab. 1, focusing only on the most recent literature, and showing how our work is uniquely positioned in the landscape of model editing and continual learning, the two key related branches to our work.

### A.1. Continual learning

［#91］
Continual Learning (CL) methods enable models to learn new tasks without catastrophically forgetting previously mastered ones (Kirkpatrick et al., 2017b). These approaches fall into three main families: memory-based methods using exemplar buffers (Rebuffi et al., 2017), knowledge distillation techniques that transfer information across model versions (Lopez-Paz & Ranzato, 2017), and regularization-based methods that constrain weight updates (Kirkpatrick et al., 2017b). To ease the understanding of this landscape, we build a taxonomy that characterizes approaches by their incremental type (task, class, or fact-based), memory requirements, update mechanisms, and architectural constraints (Tab. 1). This taxonomy reveals how our work is different from existing continual learning attempts: while existing methods focus on preserving knowledge across distinct tasks, none explicitly address the detection and handling of conflicting information - a key capability in human cognition that our work empirically investigates.

［#92］
One of the closest old approaches is deep mind's EWC (Kirkpatrick et al., 2017b), a method designed to mitigate catastrophic forgetting in neural networks trained sequentially on distinct tasks. The core idea is to protect the most important weights

［#93］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#94］
(or neurons) for previously learned tasks during the training of new tasks. EWC identifies these important weights by calculating the Fisher Information Matrix during or after the training of a task, which estimates how sensitive each weight is to the task's performance. Weights that significantly impact the output for a given task are marked as important. A quadratic penalty is then applied during future learning, constraining these weights to remain close to their values from the previous task. This ensures that knowledge from earlier tasks is preserved while still allowing the model to adapt to new tasks. However, EWC is less suitable for LLMs, which do not have clearly defined tasks when it comes to knowledge ingestion (probably different for other types of skills). EWC's effectiveness relies on distinct task boundaries and the ability to compute task-specific importance for weights, which is feasible in scenarios with well-defined tasks, such as classification or reinforcement learning. In LLMs, where learning spans a wide range of topics and linguistic structures without clear task delineation, it's challenging to apply EWC's task-based strategy. The model would struggle to assign specific neurons or weights to individual tasks or concepts, making it difficult to protect task-specific knowledge without hindering the model's overall generalization ability across a diverse dataset.

［#95］
We cite in the remainder more recent literature that we project onto our taxonomy.

［#96］
Bai et al. (2024) introduce a novel approach to continual learning that leverages global prototypes to mitigate catastrophic forgetting in neural networks. Their key insight is that maintaining stable connections between task-specific representations and pre-learned, general-purpose token embeddings (which serve as global prototypes) can significantly reduce forgetting without requiring explicit replay mechanisms. Through empirical validation on both task-incremental and class-incremental NLP scenarios, they demonstrate that models preserving strong connections to these global prototypes exhibit enhanced stability. While their work shares our goal of preserving knowledge during updates, it differs fundamentally in its approach and granularity: where they focus on task-level knowledge preservation through architectural mechanisms, our work addresses the more specific challenge of managing contradictory factual updates through cognitive-inspired conflict detection. Their finding that stable reference points aid knowledge retention is conceptually relevant to our work, though our results suggest that such architectural approaches alone may be insufficient when handling explicitly contradictory information, where more sophisticated cognitive mechanisms become necessary.

［#97］
Benjamin et al. (2024) proposed an elegant theoretical framework that interprets neural networks as Bayesian ensembles of classifiers. Their key insight is that a neural network with N parameters can be viewed as a weighted ensemble of N classifiers in the lazy regime, where the classifiers remain fixed throughout learning. This interpretation reveals that a properly designed posterior update rule, resembling SGD without momentum, can enable continual learning without forgetting - notably, they prove that momentum actually exacerbates forgetting. While their work focuses on preserving all knowledge in task-incremental learning, our paper specifically examines cases where knowledge needs to be deliberately updated or overridden. Their key contribution is showing that catastrophic forgetting is linked to the transition from lazy to rich regimes in neural networks, providing both a theoretical explanation for why larger models are more robust to forgetting and a biologically-inspired mechanism for knowledge preservation that perhaps complements our cognitive-based approach.

［#98］
Elsayed & Mahmood (2024) propose UPGD (Utility-based Perturbed Gradient Descent), a novel approach targeting both catastrophic forgetting and loss of plasticity in streaming learning scenarios. Their method protects useful network units while maintaining plasticity in less-used ones through utility-gated gradient updates and perturbations. Unlike previous approaches requiring task boundaries or memory buffers, UPGD operates in a challenging streaming setting with continuous non-stationarity. Using their newly introduced direct plasticity metric, they demonstrate UPGD's ability to maintain performance levels that surpass or match existing methods. This work complements our investigation by providing evidence that selective neuronal updates based on utility metrics can effectively balance stability and plasticity, though in a task-learning rather than knowledge-updating context.

［#99］
Hiratani (2024) analyze how task similarity affects continual learning through a novel theoretical framework combining teacher-student models with latent structure. Their key insight is that high input feature similarity coupled with low readout similarity leads to catastrophic outcomes in both knowledge transfer and retention, even when tasks are positively correlated. They demonstrate that weight regularization in the Fisher information metric robustly helps retention regardless of task similarity, while common approaches like activity gating improve retention at the cost of transfer performance. Their theoretical predictions are validated on permuted MNIST tasks with latent variables.

［#100］
Jha et al. (2024) propose a probabilistic approach to continual learning for vision-language models, specifically focusing on CLIP adaptation. Their method, CLAP, introduces visual-guided attention and task-specific probabilistic adapters to model the distribution of text features, while leveraging CLIP's pre-trained knowledge for initialization and regularization. This work demonstrates that probabilistic modeling can significantly reduce catastrophic forgetting in class-incremental learning

［#101］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#101］
scenarios, achieving state-of-the-art performance across multiple benchmarks.

［#102］
Jiao et al. (2024) propose VQ-Prompt, a novel prompt-based continual learning framework that addresses class-incremental learning with pretrained vision transformers. Their key innovation is incorporating vector quantization into prompt selection, enabling end-to-end optimization of discrete prompts with task loss while maintaining effective knowledge abstraction. This contrasts with our cognitive-dissonance aware approach, as they focus on task adaptation through prompt engineering rather than explicit conflict detection. Their empirical results on ImageNet-R and CIFAR-100 demonstrate superior performance compared to existing prompt-based methods, suggesting the effectiveness of discrete knowledge representation in continual learning.

［#103］
(Lee et al., 2024) propose IsCiL, a framework for continual imitation learning that uses retrievable skills and adapter-based architecture to enable efficient knowledge sharing across tasks. Unlike traditional approaches that isolate task-specific parameters, IsCiL introduces a prototype-based skill retrieval mechanism that allows selective reuse of previously learned skills for new tasks. While focused primarily on motor skills rather than resolving knowledge contradictions, their empirical results show that this selective adaptation approach significantly improves sample efficiency and reduces catastrophic forgetting compared to other adapter-based methods, particularly in scenarios with incomplete demonstrations.

［#104］
Lee et al. present a systematic empirical investigation of how dataset bias affects continual learning. Through carefully designed experiments across task-incremental, domain-incremental, and class-incremental scenarios, they reveal that bias transfers both forward and backward between tasks. Their analysis shows that CL methods focusing on stability tend to preserve and propagate biases from previous tasks, while emphasis on plasticity allows new biases to contaminate previous knowledge. Based on these insights, they propose BGS (Balanced Greedy Sampling), a method that mitigates bias transfer by maintaining a balanced exemplar memory and retraining the classification head. Note that here, we used "Replay" for Memory Usage in the table since their best performing method (BGS) uses an exemplar memory, but they also evaluate methods without memory.

［#105］
Peng et al. (2024) proposed a continual learning approach that automates task selection through vector space retrieval, eliminating the need for explicit task IDs, experience replay, or optimization constraints. Their method, Scalable Language Model (SLM), combines Joint Adaptive Re-parameterization with dynamic knowledge retrieval to automatically identify relevant parameters for each input, enabling task-agnostic updates. While achieving state-of-the-art results across diverse tasks and model scales (BERT, T5, LLaMA-2), their key contribution is demonstrating that automatic task identification and parameter selection can enable continual learning without requiring explicit task boundaries or memory buffers.

［#106］
Seo et al. (2024) presented Train-Attention, an interesting meta-learning approach for continual knowledge learning (CKL) in LLMs that predicts and applies weights to tokens based on their usefulness for future tasks. Unlike previous approaches that uniformly update all parameters, their method enables targeted knowledge updates by learning which tokens are most important to focus on. Through experiments on LAMA-CKL and TemporalWiki benchmarks, they show that selective token-weighted learning significantly reduces catastrophic forgetting while improving learning speed. The work somewhat complements our cognitive-inspired approach, and demonstrates the benefits of selective attention, but it does not explicitly address the handling of contradictory information.

［#107］
Wang et al. (2024) proposed a unified framework for continual learning that reveals common mathematical structures across seemingly distinct approaches (regularization-based, Bayesian-based, and memory-replay). Building on this unification, they introduce "refresh learning" - a plug-in mechanism that first unlearns current data before relearning it, inspired by the beneficial role of forgetting in human cognition. Their work primarily focuses on task-incremental and class-incremental scenarios, demonstrating improved accuracy across CIFAR and Tiny-ImageNet benchmarks. While their approach differs from our fact-level knowledge updates in LLMs, their findings about selective forgetting complement our observations about cognitive-inspired update mechanisms. Their theoretical analysis showing that refresh learning improves the flatness of the loss landscape offers an interesting perspective on how controlled forgetting might benefit knowledge retention in neural networks.

［#108］
Xu et al. (2024) propose a cross-domain task-agnostic incremental learning framework (X-TAIL) for vision-language models, focusing on the challenge of preserving both incrementally learned knowledge and zero-shot abilities. Their approach, RAIL, uses recursive ridge regression with non-linear projections to adapt to new domains without catastrophic forgetting. Unlike previous work requiring domain identity hints or reference datasets, RAIL can classify images across both seen and unseen domains without domain hints, demonstrating superior performance in both discriminative ability and knowledge preservation. While their work advances the technical aspects of continual learning, it differs from our cognitive-inspired

［#109］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#109］
investigation as it doesn't address the fundamental challenge of detecting and resolving conflicting knowledge, instead focusing on domain adaptation without explicit conflict awareness.

［#110］
Zhao et al. (2024) propose a class-incremental learning framework for pre-trained vision models that balances stability and plasticity through two complementary parameter-efficient tuning mechanisms. Their SAFE approach first inherits generalizability from pre-trained models via a "slow learner" that captures transferable knowledge in the first session, then maintains plasticity through a "fast learner" that continuously adapts to new classes while resisting catastrophic forgetting. While focused on vision tasks rather than language models, their dual-speed learning strategy presents interesting parallels to our cognitive-inspired approach – particularly in how both works identify the importance of selective plasticity and the distinction between stable ("stubborn") and adaptable ("plastic") parameters. However, SAFE doesn't address the fundamental challenge of detecting and handling contradictory information that we identify as crucial for true cognitive-inspired learning.

［#111］
Unlike the above work, our goal is to understand the fundamental cognitive mechanisms underlying the continuous knowledge updates in LLMs, particularly focusing on how models can detect and react to contradictory information. Rather than proposing a new continual learning method, we provide crucial insights into how different types of knowledge updates affect model behavior and stability.

### A.2. Knowledge editing

［#112］
Next, a big portion of recent literature has focused on understanding and modifying the internal knowledge of Large Language Models (LLMs), post-training. Such knowledge editing aims to alter specific facts or associations within the model without the need for full retraining.

［#113］
Geva et al. (2020) were among the first to show that transformer Feed-Forward Network (FFN) layers act as unnormalized key-value stores encoding relational knowledge inside LLMs. This observation was later confirmed and complemented by others (Meng et al., 2022a; Dai et al., 2021) before being leveraged by subsequent work to master the editing of internal memories. Meng et al. (2022a) introduced ROME (Rank-One Model Editing), a method that uses causal tracing to empirically locate the layers essential to encoding a given association. They then modify these modules by applying small rank-one changes. To identify the relevant modules, they run the network multiple times, introducing corruptions to the input sequence to disturb the inference, and then restore individual states from the original non-corrupted pass. But this work an others worked only on single edits, and were often evaluated one edit at a time, starting each time from a fresh pre-trained model. The same authors later developed MEMIT, which follows the same causal tracing principle but with the goal of scaling up to 10,000 edits in bulk(Meng et al., 2022c). Similarly, Dai et al. (2021) leveraged the identification of knowledge neurons to perform "knowledge surgery" – editing factual knowledge within Transformers without the need for additional fine-tuning. Zhu et al. (2020) approached the knowledge modification task as a constrained optimization problem. Their work found that constrained layer-wise fine-tuning emerges as an effective method for modifying the knowledge that Transformers learn, suggesting a different pathway for knowledge editing inside LLMs. De Cao et al. (2021) proposed KNOWLEDGEEDITOR, which achieved knowledge editing by training a hyper-network with constrained optimization to modify specific facts without fine-tuning or changing the overall stored knowledge. The method was demonstrated on smaller models like BERT for fact-checking and BART for question answering, achieving consistent changes in predictions across different formulations of queries.

［#114］
Li et al. (2023) empirically investigate the pitfalls of knowledge editing in LLMs, revealing two critical issues: logical inconsistencies between multiple edits (like contradictory relationship updates) and knowledge distortion (where edits irreversibly damage the model's knowledge structure). Through carefully designed benchmarks CONFLICTEDIT and ROUNDEDIT, they demonstrate that current editing methods struggle with these challenges, particularly when handling reverse relationships or composite logical rules. While their work focuses on identifying limitations in maintaining logical consistency across edits, our paper takes a complementary cognitive-inspired perspective by addressing how models handle contradictions with their existing knowledge base. Their findings about knowledge distortion align with and reinforce our observations about the catastrophic nature of updates that modify existing knowledge.

［#115］
Similarly, Huang et al. (2024) empirically investigate causes of performance degradation during knowledge editing in LLMs. They show degradation correlates with editing target complexity and L1-norm growth in edited layers. Their proposed Dump for Sequence (D4S) method regulates layer norm growth using O(1) space complexity, enabling multiple effective updates while minimizing model degradation. Their work provides valuable insights into the mechanisms of model degradation during knowledge editing, but it does not specifically address the distinction between contradictory and non-contradictory

［#116］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#116］
updates, as we do in this paper.

［#117］
Tan et al. (2023) propose MALMEN, a scalable hypernetwork approach for editing Large Language Models by aggregating parameter shifts using a least-squares formulation. While previous editing methods like MEND (Mitchell et al., 2022) could handle only a few facts simultaneously, MALMEN can efficiently edit thousands of facts while maintaining comparable performance. Their key innovation lies in separating the computation between the hypernetwork and LM, enabling arbitrary batch sizes and reducing memory requirements. Their empirical results show that MALMEN can edit hundreds of times more facts than MEND while maintaining similar performance levels, though they note that the method still struggles with generalizing to rephrasing not seen during training. Like other editing approaches, MALMEN focuses on the mechanics of (by design conflicting) updates.

［#118］
Unlike all the work above, our goal in this work is not to edit knowledge, but to understand the fundamental mechanisms and phenomena that govern how LLMs integrate new information with existing knowledge. By taking a cognitive-inspired approach focused on dissonance awareness and adaptive plasticity, we reveal critical insights about the nature of knowledge representation and updating in these models.

### B. Extraction of historical activations and gradients

［#119］
We here detail our procedure for the extraction of activations and gradients. Source code is also available at https://github.com/bendiogene/ConflictAwareLLM/ for ultimate level of details and reproducibility purposes.

#### B.1. Preliminary notation

［#120］
We focus on the historical tracking of gradients of the outputs (grad_outs) and activations for four key matrices within each block of the transformer model: $\text{Attn}_\text{c_attn}$, $\text{Attn}_\text{c_proj}$, $\text{MLP}_\text{c_fc}$, and $\text{MLP}_\text{c_proj}$.

［#121］
Given an input sequence $X \in \mathbb{R}^{B \times N \times d_{\text{model}}}$, where $B$ is the batch size, $N$ is the sequence length, and $d_{\text{model}}$ is the model dimension, the transformer block is defined as follows:

［#122］
**Attention Layer:** The attention mechanism computes query $Q$, key $K$, and value $V$ matrices:
［#122］
$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$
［#122］
where $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{key}}}$, $W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{key}}}$, and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{value}}}$ are trainable projection matrices.

［#123］
The concatenated matrix $\text{Attn}_\text{c_attn}$ is:
［#123］
$$
\text{Attn}_\text{c_attn} = [Q, K, V] = XW_{\text{attn}}
$$
［#123］
where $W_{\text{attn}} = [W_Q, W_K, W_V] \in \mathbb{R}^{d_{\text{model}} \times (2d_{\text{key}}+d_{\text{value}})}$.

［#124］
The attention context $\text{Attn}_\text{context}$ is computed as:
［#124］
$$
\text{Attn}_\text{context} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{key}}}}\right)V
$$

［#125］
The projected attention output $\text{Attn}_\text{c_proj}$ is:
［#125］
$$
\text{Attn}_\text{c_proj} = \text{Attn}_\text{context}W_{\text{proj}}
$$
［#125］
where $W_{\text{proj}} \in \mathbb{R}^{d_{\text{value}} \times d_{\text{model}}}$.

［#126］
**MLP Layer:** The MLP layer consists of two linear transformations with an activation function $\sigma$:
［#126］
$$
\text{MLP}_\text{c_fc} = \sigma(XW_{\text{fc}} + b_{\text{fc}})
$$
［#126］
where $W_{\text{fc}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $b_{\text{fc}} \in \mathbb{R}^{d_{\text{ff}}}$.

［#127］
The projected MLP output $\text{MLP}_\text{c_proj}$ is:
［#127］
$$
\text{MLP}_\text{c_proj} = \text{MLP}_\text{c_fc}W_{\text{proj}} + b_{\text{proj}}
$$
［#127］
where $W_{\text{proj}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ and $b_{\text{proj}} \in \mathbb{R}^{d_{\text{model}}}$.


### B.2. Historical gradient and activation collection

［#128］
Collecting a profile of neuron activity during training or simulation of training is needed as (i) input feature to know if a fact is dissonant, novel or known, and (ii) as means to identify where to locate targeted updates.

［#129］
During training, we collect and cumulate the gradients of the outputs (grad_outs) and activations for the matrices $\text{Attn}_\text{c_attn}$, $\text{Attn}_\text{c_proj}$, $\text{MLP}_\text{c_fc}$, and $\text{MLP}_\text{c_proj}$. Let $t$ denote the training step. We collect activations at step $t$:

［#129］
$$\text{Attn}_\text{c_attn}(t), \text{Attn}_\text{c_proj}(t), \text{MLP}_\text{c_fc}(t), \text{MLP}_\text{c_proj}(t)$$

［#129］
as well as Gradient of the Outputs (grad_outs) at step $t$:

［#129］
$$\nabla L(\text{Attn}_\text{c_attn}(t)), \nabla L(\text{Attn}_\text{c_proj}(t)), \nabla L(\text{MLP}_\text{c_fc}(t)), \nabla L(\text{MLP}_\text{c_proj}(t))$$

［#130］
In the remainder, we denote these, regardless of their provenance matrix, as:

［#130］
$$A^l(t), G^l(t) \in \mathbb{R}^{B \times N \times d^l_\text{out}}$$

［#130］
where $l$ denotes the layer, $B$ is the batch size, $N$ is the sequence length, and $d^l_\text{out}$ is the output dimension of layer $l$.

［#131］
When needed, we standardize these metrics for each layer $l$ as follows:

［#131］
$$\hat{A}^l(t) = \frac{A^l(t) - \mu^l_A(t)}{\sigma^l_A(t)}, \quad \hat{G}^l(t) = \frac{G^l(t) - \mu^l_G(t)}{\sigma^l_G(t)}$$

［#131］
where $\mu$ and $\sigma$ are the mean and standard deviation computed over all dimensions of the respective tensor.

［#132］
We then sum over the batch dimension:

［#132］
$$S^l_{\hat{A}}(t)_{n,i} = \sum_{b=1}^B \hat{A}^l_{b,n,i}(t), \quad S^l_{\hat{G}}(t)_{n,i} = \sum_{b=1}^B \hat{G}^l_{b,n,i}(t)$$

［#133］
Optionally⁷, we can sum over the token dimension:

［#133］
$$S^l_{\hat{A}}(t)_i = \sum_{n=1}^N S^l_{\hat{A}}(t)_{n,i}, \quad S^l_{\hat{G}}(t)_i = \sum_{n=1}^N S^l_{\hat{G}}(t)_{n,i}$$

［#134］
The standardized and summed metrics are then accumulated across the training steps:

［#134］
$$H \hat{A}^l_i = \sum_{t=1}^T S^l_{\hat{A}}(t)_i, \quad H \hat{G}^l_i = \sum_{t=1}^T S^l_{\hat{G}}(t)_i$$

［#134］
where $T$ is the total number of training steps.

［#135］
These historical activations $H \hat{A}^l$ and gradients $H \hat{G}^l$ provide cumulative measures of neuron activity over the training process. They help identify neurons that are heavily utilized (stubborn neurons) and those that are underutilized (plastic neurons), which is crucial for our targeted updates.

---

### C. Dissonance awareness

#### C.1. Augmenting the COUNTERFACT Dataset with Novel facts

［#136］
To generate unknown facts to augment the Counterfact dataset, we used GPT-3.5 with a prompt as follows:

---
［#133］
⁷We consider two approaches. In the first, we extract the activations and gradients corresponding to the last token (i.e., position $N$) in the sequence for each sample in the batch. This is reasonable since the last token is representative of the fact or information of interest in our datasets. In the second, we simply aggregate over all tokens, where we aggregate activations and gradients across all tokens in the sequence by computing statistical measures such as the mean or sum over the token dimension.
---

［#137］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#138］
Starting from this list of facts, can you create one data entry for each that concerns ↠ imaginary names and characters if necessary, while following the same logic.

［#139］
For example, Danielle Darrieux's mother tongue is French => Becomes Machin De Machine's ↠ mother tongue is Kurdi (or Kinduli).

［#140］
Edwin of Northumbria's religious values strongly emphasize Christianity => Hamed Habib's ↠ religious values strongly emphasize Atheism (or Peace or..)

［#141］
Try to make the old and new as far as possible from each other (e.g., Kurdi is far from ↠ French, Kinduli is an imaginary language, etc.), while keeping some logic.

［#142］
Write in JSON format, please (easy to parse):
- Danielle Darrieux's mother tongue is French
- Edwin of Northumbria's religious values strongly emphasize Christianity
- Toko Yasuda produces the most amazing music on the guitar
- One can get to Autonomous University of Madrid by navigating Spain
- Thomas Joannes Stieltjes was born in Dutch
- Anaal Nathrakh originated from Birmingham

［#143］
Example Generated Transformations:

［#144］
- Original: *"Toko Yasuda produces the most amazing music on the guitar."*
  Transformed: *"Zara Zorin produces the most amazing music on the theremin."*

［#145］
- Original: *"One can get to Autonomous University of Madrid by navigating Spain."*
  Transformed: *"One can reach the Floating Academia of Zephyria by navigating through the Cloud Realms."*

［#146］
- Original: *"Thomas Joannes Stieltjes was born in Dutch."*
  Transformed: *"Lorien Ilithar was born amidst the Elvish."*

［#147］
These transformations help create novel facts unlikely to be known by the model, enabling us to evaluate its ability to handle unknown information effectively.

### C.2. Ablation study of classifier performance
［#148］
We further report for the interested reader the results of an ablation study of the dissonance awareness classifier, evaluating its performance under different scenarios (fine-tuned vs. pre-trained models), feature sets (A, G, A+G), normalization strategies (None, Layer, Historical), and classifiers (Random Forests (RF) and Support Vector Machines (SVM)).

［#149］
Table 4 presents a comprehensive set of classification results, including average accuracy and F1 scores (with standard deviations) across different settings. The best results for each classifier are denoted with a $\star$ and reported earlier in Table 2 in the main paper.

### C.3. Explanation of feature importance
［#150］
To further understand the discriminative power of different features, we analyzed the feature importance scores derived from the RF classifier.

［#151］
First, as earlier mentioned in Fig.4 in the main paper, gradient-based features are substantially more important than activation-based features. This suggests that fine-tuning leads to more discriminative gradients, possibly due to the model overfitting on the known facts, resulting in near-zero gradients for known facts and higher gradients for novel or conflicting facts. In contrast, for the pre-trained model, both activation and gradient features contribute significantly, indicating that combining internal representations and learning dynamics is beneficial for classification.

［#152］
Complementary to Fig.4, block importance reported in Fig. 7 reveals that, in the pre-trained model all transformer blocks tend to contribute relatively equally to the classification task, with the last layers contributing less. The finetuned model, on


［#153］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#154］
Table 4. Ablation study of dissonance awareness: Classification Results for Different Scenarios, Feature Sets, Normalization strategies and Classifier. Average (and std) accuracy and F1 scores. $\star$ denotes the best combination for each classifier

［#155］
<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Features</th>
      <th>Normalization</th>
      <th>Classifier</th>
      <th>Accuracy</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">Finetuned</td>
      <td rowspan="6">A+G</td>
      <td rowspan="2">Null</td>
      <td>SVM</td>
      <td>0.994 (0.004)</td>
      <td>0.994 (0.004)</td>
    </tr>
    <tr>
      <td>RF$\star$</td>
      <td>0.988 (0.001)</td>
      <td>0.988 (0.001)</td>
    </tr>
    <tr>
      <td rowspan="2">Layer</td>
      <td>SVM</td>
      <td>0.995 (0.001)</td>
      <td>0.995 (0.001)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.982 (0.005)</td>
      <td>0.982 (0.004)</td>
    </tr>
    <tr>
      <td rowspan="2">Historical</td>
      <td>SVM$\star$</td>
      <td>0.995 (0.001)</td>
      <td>0.995 (0.001)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.978 (0.003)</td>
      <td>0.978 (0.003)</td>
    </tr>
    <tr>
      <td rowspan="6">G</td>
      <td rowspan="2">Null</td>
      <td>SVM</td>
      <td>0.917 (0.009)</td>
      <td>0.918 (0.009)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.905 (0.008)</td>
      <td>0.906 (0.008)</td>
    </tr>
    <tr>
      <td rowspan="2">Layer</td>
      <td>SVM</td>
      <td>0.920 (0.003)</td>
      <td>0.921 (0.003)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.895 (0.007)</td>
      <td>0.896 (0.007)</td>
    </tr>
    <tr>
      <td rowspan="2">Historical</td>
      <td>SVM</td>
      <td>0.897 (0.004)</td>
      <td>0.898 (0.004)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.868 (0.014)</td>
      <td>0.870 (0.014)</td>
    </tr>
    <tr>
      <td rowspan="6"></td>
      <td rowspan="6">A</td>
      <td rowspan="2">Null</td>
      <td>SVM</td>
      <td>0.796 (0.005)</td>
      <td>0.796 (0.007)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.747 (0.012)</td>
      <td>0.745 (0.016)</td>
    </tr>
    <tr>
      <td rowspan="2">Layer</td>
      <td>SVM</td>
      <td>0.783 (0.013)</td>
      <td>0.784 (0.012)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.722 (0.009)</td>
      <td>0.720 (0.007)</td>
    </tr>
    <tr>
      <td rowspan="2">Historical</td>
      <td>SVM</td>
      <td>0.781 (0.009)</td>
      <td>0.781 (0.010)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.721 (0.010)</td>
      <td>0.719 (0.008)</td>
    </tr>
    <tr>
      <td rowspan="12">Pretrained</td>
      <td rowspan="6">A+G</td>
      <td rowspan="2">Null</td>
      <td>SVM</td>
      <td>0.944 (0.006)</td>
      <td>0.944 (0.006)</td>
    </tr>
    <tr>
      <td>RF$\star$</td>
      <td>0.928 (0.012)</td>
      <td>0.929 (0.011)</td>
    </tr>
    <tr>
      <td rowspan="2">Layer</td>
      <td>SVM</td>
      <td>0.949 (0.006)</td>
      <td>0.949 (0.006)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.909 (0.014)</td>
      <td>0.910 (0.013)</td>
    </tr>
    <tr>
      <td rowspan="2">Historical</td>
      <td>SVM$\star$</td>
      <td>0.947 (0.004)</td>
      <td>0.948 (0.003)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.925 (0.006)</td>
      <td>0.925 (0.006)</td>
    </tr>
    <tr>
      <td rowspan="6">G</td>
      <td rowspan="2">Null</td>
      <td>SVM</td>
      <td>0.904 (0.006)</td>
      <td>0.904 (0.006)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.891 (0.010)</td>
      <td>0.892 (0.009)</td>
    </tr>
    <tr>
      <td rowspan="2">Layer</td>
      <td>SVM</td>
      <td>0.902 (0.008)</td>
      <td>0.902 (0.007)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.859 (0.013)</td>
      <td>0.861 (0.011)</td>
    </tr>
    <tr>
      <td rowspan="2">Historical</td>
      <td>SVM</td>
      <td>0.915 (0.007)</td>
      <td>0.916 (0.006)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.879 (0.017)</td>
      <td>0.879 (0.016)</td>
    </tr>
    <tr>
      <td rowspan="6"></td>
      <td rowspan="6">A</td>
      <td rowspan="2">Null</td>
      <td>SVM</td>
      <td>0.909 (0.006)</td>
      <td>0.909 (0.006)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.894 (0.009)</td>
      <td>0.895 (0.007)</td>
    </tr>
    <tr>
      <td rowspan="2">Layer</td>
      <td>SVM</td>
      <td>0.905 (0.012)</td>
      <td>0.905 (0.011)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.876 (0.004)</td>
      <td>0.877 (0.003)</td>
    </tr>
    <tr>
      <td rowspan="2">Historical</td>
      <td>SVM</td>
      <td>0.900 (0.008)</td>
      <td>0.900 (0.007)</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.881 (0.006)</td>
      <td>0.882 (0.006)</td>
    </tr>
  </tbody>
</table>

［#154］
the other hand shows a slightly different tendency where the earlier layers contribute less. More work is clearly needed to understand such differences. This paper focuses only on feasibility of the entire cognitive-dissonance approach, leaving more elaborate evaluations for future work.

### C.4. Location of stubborn neurons

［#156］
We also report the distribution of stubborn neurons across the transformer blocks in GPT-2 XL. Figures 8a and 8b show histograms of the number of stubborn neurons identified in each block for thresholds of 8,000 and 2,000 neurons, respectively.

［#157］
Our analysis indicates that stubborn neurons are not uniformly distributed throughout the network. Instead, they curiously tend to be concentrated in certain blocks, particularly in the first block and in certain middle layers of the transformer.

［#158］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#159］
![](./images/1095943058349883393_10.jpg)

［#160］
(a) Finetuned model

［#161］
![](./images/1095943058349883393_11.jpg)

［#162］
(b) Pre-trained model

［#163］
Figure 7. *Block Importance*. Albeit differences are visible, the tendency is not as marked as for the activation vs gradient based feature importance in Fig.4 - GPT2-small

［#164］
This might suggest that these layers play a more significant role in encoding and retaining knowledge during training. Interestingly, $\text{Attn}_{\text{c_attn}}$ concentrates much more of the stubborn neurons overall, with the exception of the first block where $\text{Attn}_{\text{c_proj}}$ has a substantially higher share of stubborn neurons. The results are similar for both thresholds.

［#165］
Overall, understanding the distribution of stubborn neurons can inform targeted update strategies by identifying which parts of the network are more critical for preserving existing knowledge.

### C.5. Using model output (instead of internal state) as features for dissonance awareness

［#166］
In the main paper, we used activations and gradients as they were *readily available* in our experimental pipeline. We now further test whether using model output only, which is more easily available than internal gradients and activations can achieve similar performance on our scenario.

［#167］
Each fact in our dataset is conceptually a statement involving a subject (s), relation (r), and object (o) (e.g., “Danielle Darrieux’s mother tongue is French”). In this section, we extract features that capture increasing levels of detail about the model’s predictions, related to what the actual facts are, leveraging both:
- Conditional probabilities $p(o|s, r)$ at different truncation points⁸
- Joint probability $p(s, r, o)$ of the full statement

［#168］
In more details, we extract the following features, with increasing complexity.

［#169］
**Basic Token Probabilities ($Feat_1$):** For each of the last $N$ tokens (representing the answer), we collect the probability of the actual next token given the truncated prompt. These simple scalar features capture the model’s direct confidence in the correct continuation. This has a dimensionality of $N+1$ ($N$ truncation points plus full statement, so 4 in our case.)

［#170］
**Top-$k$ Predictions Analysis ($Feat_2$):** Here, for each position in the answer, we collect the values and normalized indices of top-$k$ most likely next tokens. This captures both confidence distribution and ranking patterns. Similarly to the above, we compute this for both truncated prompts and full statements. Here, the dimensionality is $(N+1) \times 2k$ ($k$ values and $k$ normalized indices for each position). We pick k=100.

［#171］
**Distribution Features ($Feat_3$):** Here, we analyze the complete probability distribution over the vocabulary. For each position in the answer sequence, we construct histograms of the probabilities with $n_{bins}$ bins (here 100), capturing the full spectrum of the model’s prediction patterns. We augment these distributions with indicator vectors that highlight the positions of ground truth tokens (the true next tokens of the current truncated fact), providing additional context about the model’s accuracy. This results in a feature vector of dimensionality $(N+1) \times n_{bins}$.

［#172］
⁸Since the object $o$ can span multiple tokens, we extract features from the last $N$ tokens of each fact (we pick three, since most answers fit within that limit). For each token position, we compute both the truncated prompt probability $p(o|s, r)$ by removing the token and subsequent tokens, and the full sentence probability $p(s, r, o)$. This multi-token analysis ensures we capture the model’s predictions across the entire span of the answer.

［#173］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#174］
![](./images/1095943058349883393_12.jpg)

［#175］
(a) Histogram of stubborn neurons ($t=8000$ neurons) across transformer blocks

［#176］
![](./images/1095943058349883393_13.jpg)

［#177］
(b) Histogram of stubborn neurons ($t=2000$ neurons) across transformer blocks

［#178］
Figure 8. Distribution of stubborn neurons across GPT2-XL transformer blocks for different neuron thresholds to define stubbornness. (a) shows the distribution for $t=8000$ neurons, while (b) corresponds to $t=2000$ neurons.

［#179］
Combined Features (Concat): Here, we simply concatenate $Feat_1$, $Feat_2$, and $Feat_3$.

［#180］
Tab. 5 shows the results over our dataset. We observe *a similar great performance when using the model outputs, compared to Activations and Gradients*. Model output achieves even better performance in case of pre-trained models. This is inline with our earlier observation that activations (what we're using now) are more important than gradients in the case of pre-trained models. This result is encouraging for future work, where we plan to (i) build more challenging classification datasets (than the simple facts in CounterFact) and (ii) build standalone classifiers to speed up the training of LLMs, by avoiding training on conflicting data.

### D. Non-dissonant updates
#### D.1. Similarities with Lottery ticket

［#181］
To assess the hypothesis that certain subnetworks within the language model are more conducive to integrating new information—a notion earlier named the lottery ticket hypothesis (Frankle & Carbin, 2018)—we designed an experiment to confirm this effect.

［#182］
We first trained a model on 10,000 disjoint facts (referred to as Facts H) and identified the most active candidate neurons during this process, which we term *Lottery Ticket Neurons*. These neurons should form a preferred subnetwork for representing Facts H. Next, we started from a *fresh model* and trained on a new set of novel facts (Facts A), which are different from H, restricting updates to three distinct groups of neurons:

［#183］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#184］
<table>
  <thead>
    <tr>
      <th rowspan="2">Strategy (dim)</th>
      <th colspan="2">Pretrained Model</th>
      <th colspan="2">Finetuned Model</th>
    </tr>
    <tr>
      <th>Accuracy</th>
      <th>F1-Score</th>
      <th>Accuracy</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Feat.1 (4)</td>
      <td>0.852</td>
      <td>0.856</td>
      <td>0.850</td>
      <td>0.855</td>
    </tr>
    <tr>
      <td>Feat.2 (800)</td>
      <td>0.602</td>
      <td>0.588</td>
      <td>0.600</td>
      <td>0.581</td>
    </tr>
    <tr>
      <td>Feat.3 (400)</td>
      <td>0.540</td>
      <td>0.452</td>
      <td>0.543</td>
      <td>0.464</td>
    </tr>
    <tr>
      <td>Concat (1204)</td>
      <td>0.983</td>
      <td>0.983</td>
      <td>0.978</td>
      <td>0.978</td>
    </tr>
    <tr>
      <td>(A+G) (240)</td>
      <td>0.947</td>
      <td>0.948</td>
      <td>0.995</td>
      <td>0.995</td>
    </tr>
  </tbody>
</table>

［#185］
Table 5. Using output-only features for dissonance-awareness can achieve similar good performance to using our readily available activations and gradients, and even better in the case of the pre-trained model.

［#186］
![](./images/1095943058349883393_14.jpg)

［#187］
Figure 9. Lottery ticket

［#188］
1. **Lottery Ticket Neurons**: Neurons highly active during the initial training on Facts H.

［#189］
2. **Non-Lottery Neurons**: Neurons underutilized during the initial training on Facts H.

［#190］
3. **Random Neurons**: Neurons selected randomly from the entire network.

［#191］
Figure 9 shows the accuracy of acquiring new knowledge when using each of these strategies, with the number of neurons varying from 2,000 to 20,000. Using the Lottery Ticket Neurons led to significantly better performance, reaching nearly 100% accuracy at 8,000 neurons, compared to around 40% for the Non-Lottery Neurons. The Random Neurons strategy also performed relatively well, interestingly suggesting that capturing even a few "anchor" neurons from the preferred subnetwork is sufficient to achieve good performance.

［#192］
These results support the existence of preferred subnetworks within the model that are particularly effective for learning new information. Leveraging these subnetworks can enhance the efficiency of knowledge integration while preserving existing knowledge, an aspect that our candidate and specific strategies are already exploiting.

### D.2. Hyperparameter selection: learning rate and batch size for GPT2-XL

［#193］
In our experiments, the first step is to conduct a hyperparameter search to determine the optimal learning rates and batch sizes for fine-tuning the model on our facts. Table 6 presents the performance of GPT2-XL on old and new knowledge across various learning rates and batch sizes. Note that this optimal learning rate for full finetuning might turn out not enough for our targeted updates, since they use, by design, a smaller number of neurons.

### D.3. Comprehensive Analysis of GPT2-XL non-dissonant Updates

［#194］
Figure 10 presents the accuracy of GPT-2 XL on old and new knowledge under various neuron update strategies and experimental conditions. We explored different configurations to understand how the model's larger capacity affects knowledge integration.

［#195］
Our results reveal distinct scaling behaviors compared to GPT-2 small. With the optimal learning rate for GPT-2 XL

［#196］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#197］
<table>
  <thead>
    <tr>
      <th>Learning Rate</th>
      <th>Batch Size</th>
      <th>Epochs</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1e-06</td>
      <td>64</td>
      <td>5</td>
      <td>0.271</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>64</td>
      <td>10</td>
      <td>0.476</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>64</td>
      <td>20</td>
      <td>0.694</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>32</td>
      <td>5</td>
      <td>0.441</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>32</td>
      <td>10</td>
      <td>0.641</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>32</td>
      <td>20</td>
      <td>0.888</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>16</td>
      <td>5</td>
      <td>0.582</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>16</td>
      <td>10</td>
      <td>0.782</td>
    </tr>
    <tr>
      <td>1e-06</td>
      <td>16</td>
      <td>20</td>
      <td>0.984</td>
    </tr>
    <tr>
      <td><b>1e-05</b></td>
      <td><b>32</b></td>
      <td><b>5</b></td>
      <td><b>0.981</b></td>
    </tr>
    <tr>
      <td>1e-05</td>
      <td>32</td>
      <td>7</td>
      <td>0.997</td>
    </tr>
    <tr>
      <td>1e-05</td>
      <td>16</td>
      <td>5</td>
      <td>0.989</td>
    </tr>
    <tr>
      <td>1e-05</td>
      <td>16</td>
      <td>7</td>
      <td>0.997</td>
    </tr>
    <tr>
      <td>1e-05</td>
      <td>16</td>
      <td>10</td>
      <td>0.998</td>
    </tr>
    <tr>
      <td>5e-06</td>
      <td>32</td>
      <td>5</td>
      <td>0.853</td>
    </tr>
    <tr>
      <td>5e-06</td>
      <td>32</td>
      <td>7</td>
      <td>0.957</td>
    </tr>
    <tr>
      <td>5e-06</td>
      <td>32</td>
      <td>10</td>
      <td>0.996</td>
    </tr>
    <tr>
      <td>5e-06</td>
      <td>16</td>
      <td>5</td>
      <td>0.954</td>
    </tr>
    <tr>
      <td>5e-06</td>
      <td>16</td>
      <td>7</td>
      <td>0.996</td>
    </tr>
    <tr>
      <td>5e-06</td>
      <td>16</td>
      <td>10</td>
      <td>0.998</td>
    </tr>
  </tbody>
</table>

［#198］
Table 6. Accuracy results for different learning rates, batch sizes, and epochs on 10k facts (GPT2-xl). We use the finetuning on 10k facts as a proxy to pick the hyperparameters of our later continual update experiments (learning rate, batch size and epochs). In bold, what we picked for GPT2-xl. Not shown here, for GPT2-small, we picked 5e-4.

［#199］
(Figures 10a, 10b), we observe improved new knowledge acquisition while still preserving old knowledge. This means that although our carefully picked learning rate allows for efficient learning with full finetuning, learning with fewer neurons (as per our targetted strategies) seems harder than it was for GPT-2 small.

［#200］
Increasing the learning rate by 10x (Figures 10c, 10d) or allocating 10x more neurons (Figures 10e, 10f) confirms that GPT-2 XL requires either higher learning rates or more extensive parameter updates compared to GPT-2 small to achieve effective learning with our targeted strategies.

［#201］
Similarly, extended training duration (50 epochs, Figures 10g, 10h) allows the model to better integrate new knowledge while preserving old information, indicating that longer training can also help overcome the limitations of sparse updates in larger models. Figure 11 summarizes these trade-offs across all configurations, highlighting how different hyperparameter choices affect the balance between preserving old knowledge and acquiring new information.

［#202］
Finally, note that while GPT-2 XL's larger capacity naturally reduces interference with our tracked facts during non-dissonant updates, this improved performance is "deceptive" and should be interpreted cautiously: we cannot measure potential effects on other pre-trained knowledge beyond our tracked facts.

［#203］
These results highlight the methodological challenges in studying knowledge updates in larger models: their increased capacity can mask interference with tracked facts, making it harder to fully measure the impact of updates on the model's broader knowledge. This underscores the importance of controlled experimental settings when studying fundamental properties of knowledge updating in neural networks.

## E. Dissonant updates

### E.1. Impact of number of conflicting facts

［#204］
We examined the effect of varying the number of conflicting facts introduced during Dissonant updates. Figure 12 shows the performance metrics of GPT-2 small when editing 10, 100, and 1,000 facts, respectively.

［#205］
Our findings show that as the number of conflicting facts increases, the impact on old knowledge retention becomes more pronounced, with all strategies experiencing significant degradation. The ability to learn new conflicting knowledge improves slightly with more facts, but overall performance remains suboptimal. The plastic and random neuron strategies tend to


［#206］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#207］
The best LR for full FT is not enough to learn with targeted updates:

［#208］
![](./images/1095943058349883393_15.jpg)
［#209］
(a) Old Knowledge

［#210］
![](./images/1095943058349883393_16.jpg)
［#211］
(b) New Knowledge

［#212］
Increasing the LR (here 10X higher) helps:

［#213］
![](./images/1095943058349883393_17.jpg)
［#214］
(c) Old Knowledge

［#215］
![](./images/1095943058349883393_18.jpg)
［#216］
(d) New Knowledge

［#217］
Giving more space (here 10X more neurons) also helps targetted updates:

［#218］
![](./images/1095943058349883393_19.jpg)
［#219］
(e) Old Knowledge

［#220］
![](./images/1095943058349883393_20.jpg)
［#221］
(f) New Knowledge

［#222］
Finally, training longer (here 50 Epochs) yielded the most stable results:

［#223］
![](./images/1095943058349883393_21.jpg)
［#224］
(g) Old Knowledge

［#225］
![](./images/1095943058349883393_22.jpg)
［#226］
(h) New Knowledge

［#227］
Figure 10. Non-Dissonant updates with GPT2-XL under various conditions. Overall the same trends as GPT2-small are confirmed: targeting stubborn neurons destroys old knowledge more and plastic neurons need more space or time to learn.

［#228］
preserve old knowledge when editing a small number of facts (e.g., 10 facts), but their effectiveness diminishes as more conflicting information is introduced. Interestingly, the opposite effect is observed for new knowledge, where adding more facts seems to make it easier to learn new knowledge, for all strategies.

### E.2. Comparative performance of Editing methods

［#229］
Our primary focus in this work is not on developing new model editing techniques. Most existing editing techniques focus on altering existing associations, and are hence by our definition dissonant by design. Our empirical findings in this work

# The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#230］
suggest another parallel path in which editing is abandoned in favor of non-dissonant variations where old knowledge is kept and contextualized

［#231］
However, to have an idea on how existing editing methods perform compared to our targeted strategies, we leverage `EasyEdit` (Wang et al., 2023) to benchmark two state-of-the-art model editing methods, ROME (Meng et al., 2022a) and MEMIT (Meng et al., 2022c), under our same multi-fact experimental conditions .

［#232］
Table 7 summarizes the performance of different strategies and editing methods. Some of our targeted update strategies obtain a higher harmonic mean compared to ROME and MEMIT. But the higher harmonic mean must not hide that the approaches are not directly comparable since they explore different regions of the pareto front, balancing new knowledge acquisition and old knowledge retention, as self-explained with colors and rankings in the table.

［#233］
Table 7. Comparison of targeted neuron update strategies vs knowledge-editing literature, with a gradient from 0 (red) to 1 (green). Top-1,2 strategies annotated for all metrics and sample sizes.

［#234］
<table>
<thead>
<tr>
<th>Samples</th>
<th>Strategy</th>
<th>Old (Unrelated)</th>
<th>New (Reliability)</th>
<th>Generalization</th>
<th>Harmonic Mean</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">10</td>
<td>Full Finetune</td>
<td>0.107 (0.082)</td>
<td>1.000 (0.000) ¹</td>
<td>0.576 (0.117)</td>
<td>0.222 (0.116)</td>
</tr>
<tr>
<td>MEMIT(Meng et al., 2022c)</td>
<td>0.962 (0.079) ¹</td>
<td>0.000 (0.000)</td>
<td>0.000 (0.000)</td>
<td>0.000 (0.000)</td>
</tr>
<tr>
<td>ROME(Meng et al., 2022a)</td>
<td>0.891 (0.085)</td>
<td>0.240 (0.182)</td>
<td>0.180 (0.179)</td>
<td>0.236 (0.235)</td>
</tr>
<tr>
<td>8k Candidate</td>
<td>0.596 (0.106)</td>
<td>0.988 (0.024) ²</td>
<td>0.644 (0.128) ²</td>
<td>0.690 (0.058) ¹</td>
</tr>
<tr>
<td>20k Candidate</td>
<td>0.430 (0.134)</td>
<td>1.000 (0.000) ¹</td>
<td>0.656 (0.125) ¹</td>
<td>0.597 (0.116)</td>
</tr>
<tr>
<td>8k Specific</td>
<td>0.638 (0.138)</td>
<td>0.964 (0.039)</td>
<td>0.512 (0.238)</td>
<td>0.600 (0.183)</td>
</tr>
<tr>
<td>8k Stubborn</td>
<td>0.622 (0.110)</td>
<td>0.972 (0.030)</td>
<td>0.544 (0.169)</td>
<td>0.643 (0.103) ²</td>
</tr>
<tr>
<td>8k Plastic</td>
<td>0.909 (0.039) ²</td>
<td>0.020 (0.040)</td>
<td>0.000 (0.000)</td>
<td>0.000 (0.000)</td>
</tr>
<tr>
<td>
</td>
<td>8k Random</td>
<td>0.827 (0.083)</td>
<td>0.380 (0.132)</td>
<td>0.092 (0.094)</td>
<td>0.277 (0.098)</td>
</tr>
<tr>
<td rowspan="8">100</td>
<td>Full Finetune</td>
<td>0.238 (0.019)</td>
<td>0.998 (0.003) ²</td>
<td>0.434 (0.089)</td>
<td>0.398 (0.041)</td>
</tr>
<tr>
<td>MEMIT(Meng et al., 2022c)</td>
<td>0.976 (0.008) ¹</td>
<td>0.004 (0.005)</td>
<td>0.010 (0.007)</td>
<td>0.003 (0.007)</td>
</tr>
<tr>
<td>ROME(Meng et al., 2022a)</td>
<td>0.431 (0.108)</td>
<td>0.300 (0.054)</td>
<td>0.150 (0.036)</td>
<td>0.240 (0.045)</td>
</tr>
<tr>
<td>8k Candidate</td>
<td>0.542 (0.035) ²</td>
<td>0.969 (0.033)</td>
<td>0.462 (0.081) ¹</td>
<td>0.591 (0.054) ¹</td>
</tr>
<tr>
<td>20k Candidate</td>
<td>0.463 (0.032)</td>
<td>0.999 (0.002) ¹</td>
<td>0.447 (0.083) ²</td>
<td>0.552 (0.052) ²</td>
</tr>
<tr>
<td>8k Specific</td>
<td>0.531 (0.030)</td>
<td>0.760 (0.063)</td>
<td>0.263 (0.027)</td>
<td>0.426 (0.024)</td>
</tr>
<tr>
<td>8k Stubborn</td>
<td>0.530 (0.054)</td>
<td>0.936 (0.048)</td>
<td>0.398 (0.064)</td>
<td>0.547 (0.063)</td>
</tr>
<tr>
<td>8k Plastic</td>
<td>0.433 (0.029)</td>
<td>0.059 (0.014)</td>
<td>0.028 (0.017)</td>
<td>0.052 (0.025)</td>
</tr>
<tr>
<td>
</td>
<td>8k Random</td>
<td>0.508 (0.019)</td>
<td>0.193 (0.038)</td>
<td>0.065 (0.025)</td>
<td>0.131 (0.039)</td>
</tr>
<tr>
<td rowspan="8">1000</td>
<td>Full Finetune</td>
<td>0.182 (0.007)</td>
<td>0.991 (0.009)</td>
<td>0.442 (0.053) ¹</td>
<td>0.341 (0.016) ²</td>
</tr>
<tr>
<td>MEMIT(Meng et al., 2022c)</td>
<td>0.605 (0.107) ¹</td>
<td>0.198 (0.053)</td>
<td>0.100 (0.016)</td>
<td>0.177 (0.028)</td>
</tr>
<tr>
<td>ROME(Meng et al., 2022a)</td>
<td>0.152 (0.071)</td>
<td>0.160 (0.093)</td>
<td>0.067 (0.035)</td>
<td>0.106 (0.058)</td>
</tr>
<tr>
<td>8k Candidate</td>
<td>0.199 (0.014)</td>
<td>0.996 (0.002) ¹</td>
<td>0.380 (0.041) ²</td>
<td>0.345 (0.014) ¹</td>
</tr>
<tr>
<td>20k Candidate</td>
<td>0.172 (0.018)</td>
<td>0.996 (0.001) ¹</td>
<td>0.369 (0.043)</td>
<td>0.314 (0.028)</td>
</tr>
<tr>
<td>8k Specific</td>
<td>0.240 (0.017) ²</td>
<td>0.993 (0.003)</td>
<td>0.287 (0.039)</td>
<td>0.345 (0.028) ¹</td>
</tr>
<tr>
<td>8k Stubborn</td>
<td>0.200 (0.007)</td>
<td>0.995 (0.001) ²</td>
<td>0.317 (0.024)</td>
<td>0.327 (0.006)</td>
</tr>
<tr>
<td>8k Plastic</td>
<td>0.218 (0.024)</td>
<td>0.283 (0.026)</td>
<td>0.070 (0.010)</td>
<td>0.133 (0.013)</td>
</tr>
<tr>
<td>
</td>
<td>8k Random</td>
<td>0.194 (0.026)</td>
<td>0.663 (0.072)</td>
<td>0.088 (0.008)</td>
<td>0.165 (0.014)</td>
</tr>
</tbody>
</table>

## E.3. More detailed figures for specific numbers of neurons

［#235］
Tables 8, Figs. 9, and 10 provide detailed performance metrics for different neuron thresholds (20k, 8k, and 4k neurons, respectively) when editing 1,000, 100 and 10, conflicting facts using various strategies.

［#236］
The results show that changing the number of neurons allocated for updates does not necessarily improve or degrade performance in the dissonant update scenario. In all cases, the model struggles to retain old knowledge while learning new conflicting information. The candidate and specific neuron strategies are consistently and significantly better than state of the art solutions, offering a slight advantage. However, they are still unable to effectively mitigate the destructive effects of dissonant updates, further motivating the neeed for both (i) dissonance awareness and (ii) proper conflict resolution.

［#237］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#238］
Table 8. Neuron Editing Results for N=20,000 Neurons
［#239］
<table>
<thead>
  <tr>
    <th>Samples</th>
    <th>Strategy</th>
    <th>Accuracy A</th>
    <th>Accuracy NOT(B)</th>
    <th>Accuracy GEN</th>
    <th>Harmonic Mean</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">10</td>
    <td>Full Finetune</td>
    <td>0.107 (0.082)</td>
    <td>1.000 (0.000)</td>
    <td>0.576 (0.117)</td>
    <td>0.222 (0.116)</td>
  </tr>
  <tr>
    <td>Specific</td>
    <td>0.491 (0.137)</td>
    <td>1.000 (0.000)</td>
    <td>0.604 (0.126)</td>
    <td>0.621 (0.109)</td>
  </tr>
  <tr>
    <td>Plastic</td>
    <td>0.735 (0.105)</td>
    <td>0.752 (0.175)</td>
    <td>0.220 (0.183)</td>
    <td>0.434 (0.185)</td>
  </tr>
  <tr>
    <td>Stubborn</td>
    <td>0.449 (0.109)</td>
    <td>1.000 (0.000)</td>
    <td>0.616 (0.091)</td>
    <td>0.606 (0.084)</td>
  </tr>
  <tr>
    <td>Candidate</td>
    <td>0.430 (0.134)</td>
    <td>1.000 (0.000)</td>
    <td>0.656 (0.125)</td>
    <td>0.597 (0.116)</td>
  </tr>
  <tr>
    <td>Random</td>
    <td>0.688 (0.107)</td>
    <td>0.944 (0.083)</td>
    <td>0.448 (0.212)</td>
    <td>0.579 (0.222)</td>
  </tr>
  <tr>
    <td rowspan="6">100</td>
    <td>Full Finetune</td>
    <td>0.238 (0.019)</td>
    <td>0.998 (0.003)</td>
    <td>0.434 (0.089)</td>
    <td>0.398 (0.041)</td>
  </tr>
  <tr>
    <td>Specific</td>
    <td>0.412 (0.046)</td>
    <td>0.988 (0.005)</td>
    <td>0.330 (0.054)</td>
    <td>0.460 (0.046)</td>
  </tr>
  <tr>
    <td>Plastic</td>
    <td>0.317 (0.052)</td>
    <td>0.586 (0.048)</td>
    <td>0.128 (0.028)</td>
    <td>0.233 (0.035)</td>
  </tr>
  <tr>
    <td>Stubborn</td>
    <td>0.435 (0.043)</td>
    <td>0.999 (0.002)</td>
    <td>0.427 (0.085)</td>
    <td>0.528 (0.057)</td>
  </tr>
  <tr>
    <td>Candidate</td>
    <td>0.463 (0.032)</td>
    <td>0.999 (0.002)</td>
    <td>0.447 (0.083)</td>
    <td>0.552 (0.052)</td>
  </tr>
  <tr>
    <td>Random</td>
    <td>0.474 (0.035)</td>
    <td>0.874 (0.048)</td>
    <td>0.292 (0.048)</td>
    <td>0.444 (0.036)</td>
  </tr>
  <tr>
    <td rowspan="6">1000</td>
    <td>Full Finetune</td>
    <td>0.182 (0.007)</td>
    <td>0.991 (0.009)</td>
    <td>0.442 (0.053)</td>
    <td>0.341 (0.016)</td>
  </tr>
  <tr>
    <td>Specific</td>
    <td>0.188 (0.033)</td>
    <td>0.995 (0.002)</td>
    <td>0.257 (0.025)</td>
    <td>0.292 (0.035)</td>
  </tr>
  <tr>
    <td>Plastic</td>
    <td>0.077 (0.021)</td>
    <td>0.996 (0.002)</td>
    <td>0.224 (0.018)</td>
    <td>0.160 (0.027)</td>
  </tr>
  <tr>
    <td>Stubborn</td>
    <td>0.185 (0.010)</td>
    <td>0.992 (0.005)</td>
    <td>0.327 (0.013)</td>
    <td>0.317 (0.012)</td>
  </tr>
  <tr>
    <td>Candidate</td>
    <td>0.172 (0.018)</td>
    <td>0.996 (0.001)</td>
    <td>0.369 (0.043)</td>
    <td>0.314 (0.028)</td>
  </tr>
  <tr>
    <td>Random</td>
    <td>0.235 (0.029)</td>
    <td>0.995 (0.003)</td>
    <td>0.300 (0.053)</td>
    <td>0.347 (0.041)</td>
  </tr>
</tbody>
</table>

［#240］
Table 9. Neuron Editing Results for N=8,000 Neurons
［#241］
<table>
<thead>
  <tr>
    <th>Samples</th>
    <th>Strategy</th>
    <th>Accuracy A</th>
    <th>Accuracy NOT(B)</th>
    <th>Accuracy GEN</th>
    <th>Harmonic Mean</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">10</td>
    <td>Full Finetune</td>
    <td>0.107 (0.082)</td>
    <td>1.000 (0.000)</td>
    <td>0.576 (0.117)</td>
    <td>0.222 (0.116)</td>
  </tr>
  <tr>
    <td>Specific</td>
    <td>0.638 (0.138)</td>
    <td>0.964 (0.039)</td>
    <td>0.512 (0.238)</td>
    <td>0.600 (0.183)</td>
  </tr>
  <tr>
    <td>Plastic</td>
    <td>0.909 (0.039)</td>
    <td>0.020 (0.040)</td>
    <td>0.000 (0.000)</td>
    <td>0.0</td>
  </tr>
  <tr>
    <td>Stubborn</td>
    <td>0.622 (0.110)</td>
    <td>0.972 (0.030)</td>
    <td>0.544 (0.169)</td>
    <td>0.643 (0.103)</td>
  </tr>
  <tr>
    <td>Candidate</td>
    <td>0.596 (0.106)</td>
    <td>0.988 (0.024)</td>
    <td>0.644 (0.128)</td>
    <td>0.690 (0.058)</td>
  </tr>
  <tr>
    <td>Random</td>
    <td>0.827 (0.083)</td>
    <td>0.380 (0.132)</td>
    <td>0.092 (0.094)</td>
    <td>0.277 (0.098)</td>
  </tr>
  <tr>
    <td rowspan="6">100</td>
    <td>Full Finetune</td>
    <td>0.238 (0.019)</td>
    <td>0.998 (0.003)</td>
    <td>0.434 (0.089)</td>
    <td>0.398 (0.041)</td>
  </tr>
  <tr>
    <td>Specific</td>
    <td>0.531 (0.030)</td>
    <td>0.760 (0.063)</td>
    <td>0.263 (0.027)</td>
    <td>0.426 (0.024)</td>
  </tr>
  <tr>
    <td>Plastic</td>
    <td>0.433 (0.029)</td>
    <td>0.059 (0.014)</td>
    <td>0.028 (0.017)</td>
    <td>0.052 (0.025)</td>
  </tr>
  <tr>
    <td>Stubborn</td>
    <td>0.530 (0.054)</td>
    <td>0.936 (0.048)</td>
    <td>0.398 (0.064)</td>
    <td>0.547 (0.063)</td>
  </tr>
  <tr>
    <td>Candidate</td>
    <td>0.542 (0.035)</td>
    <td>0.969 (0.033)</td>
    <td>0.462 (0.081)</td>
    <td>0.591 (0.054)</td>
  </tr>
  <tr>
    <td>Random</td>
    <td>0.508 (0.019)</td>
    <td>0.193 (0.038)</td>
    <td>0.065 (0.025)</td>
    <td>0.131 (0.039)</td>
  </tr>
  <tr>
    <td rowspan="6">1000</td>
    <td>Full Finetune</td>
    <td>0.182 (0.007)</td>
    <td>0.991 (0.009)</td>
    <td>0.442 (0.053)</td>
    <td>0.341 (0.016)</td>
  </tr>
  <tr>
    <td>Specific</td>
    <td>0.240 (0.017)</td>
    <td>0.993 (0.003)</td>
    <td>0.287 (0.039)</td>
    <td>0.345 (0.028)</td>
  </tr>
  <tr>
    <td>Plastic</td>
    <td>0.218 (0.024)</td>
    <td>0.283 (0.026)</td>
    <td>0.070 (0.010)</td>
    <td>0.133 (0.013)</td>
  </tr>
  <tr>
    <td>Stubborn</td>
    <td>0.200 (0.007)</td>
    <td>0.995 (0.001)</td>
    <td>0.317 (0.024)</td>
    <td>0.327 (0.006)</td>
  </tr>
  <tr>
    <td>Candidate</td>
    <td>0.199 (0.014)</td>
    <td>0.996 (0.002)</td>
    <td>0.380 (0.041)</td>
    <td>0.345 (0.014)</td>
  </tr>
  <tr>
    <td>Random</td>
    <td>0.159 (0.032)</td>
    <td>0.784 (0.091)</td>
    <td>0.102 (0.014)</td>
    <td>0.169 (0.010)</td>
  </tr>
</tbody>
</table>

## E.4. Scaling to GPT2-XL

［#242］
We extended our dissonant update experiments to GPT-2 XL to examine whether our observations about knowledge conflicts persist in larger models.

［#243］
Figure 13 examines GPT2-XL's behavior when updating 1,000 conflicting facts using the optimal learning rate, as determined by our hyperparameter search. We compare three configurations: GPT-2 small (2,000 to 20,000 neurons) shown previously, GPT2-XL with the same range, and GPT2-XL with ten times more neurons (20,000 to 200,000). The latter was shown effective in packing new knowledge compared to (2000 to 20000) range in non-dissonant updates.

［#244］
First, while GPT2-XL still requires more neurons than GPT-2 small to effectively learn new conflicting knowledge, as seen earlier, the key finding concerns old knowledge retention: regardless of model size or neuron allocation, we observe significant degradation of old, unrelated knowledge across all strategies.

［#245］
Interestingly, this degradation persists even when using fewer neurons and when the model fails to effectively learn the new conflicting information (2k to 20k). These results strongly suggest that the destructive impact of conflicting updates on existing knowledge is a fundamental property that remains present in larger models.


［#246］
The Case for Cognitive-Dissonance Aware Continual Update of Knowledge in LLMs

［#247］
Table 10. Neuron Editing Results for N=4,000 Neurons

［#248］
<table>
  <thead>
    <tr>
      <th>Samples</th>
      <th>Strategy</th>
      <th>Accuracy A</th>
      <th>Accuracy NOT(B)</th>
      <th>Accuracy GEN</th>
      <th>Harmonic Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">10</td>
      <td>Full Finetune</td>
      <td>0.107 (0.082)</td>
      <td>1.000 (0.000)</td>
      <td>0.576 (0.117)</td>
      <td>0.222 (0.116)</td>
    </tr>
    <tr>
      <td>Specific</td>
      <td>0.673 (0.101)</td>
      <td>0.656 (0.168)</td>
      <td>0.264 (0.208)</td>
      <td>0.385 (0.182)</td>
    </tr>
    <tr>
      <td>Plastic</td>
      <td>0.965 (0.021)</td>
      <td>0.000 (0.000)</td>
      <td>0.000 (0.000)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>Stubborn</td>
      <td>0.635 (0.062)</td>
      <td>0.764 (0.087)</td>
      <td>0.352 (0.115)</td>
      <td>0.506 (0.101)</td>
    </tr>
    <tr>
      <td>Candidate</td>
      <td>0.603 (0.101)</td>
      <td>0.864 (0.126)</td>
      <td>0.512 (0.106)</td>
      <td>0.613 (0.065)</td>
    </tr>
    <tr>
      <td>Random</td>
      <td>0.863 (0.066)</td>
      <td>0.144 (0.113)</td>
      <td>0.044 (0.062)</td>
      <td>0.169 (0.050)</td>
    </tr>
    <tr>
      <td rowspan="6">100</td>
      <td>Full Finetune</td>
      <td>0.238 (0.019)</td>
      <td>0.998 (0.003)</td>
      <td>0.434 (0.089)</td>
      <td>0.398 (0.041)</td>
    </tr>
    <tr>
      <td>Specific</td>
      <td>0.553 (0.023)</td>
      <td>0.408 (0.040)</td>
      <td>0.137 (0.022)</td>
      <td>0.258 (0.029)</td>
    </tr>
    <tr>
      <td>Plastic</td>
      <td>0.760 (0.054)</td>
      <td>0.000 (0.000)</td>
      <td>0.003 (0.003)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>Stubborn</td>
      <td>0.565 (0.060)</td>
      <td>0.705 (0.143)</td>
      <td>0.303 (0.077)</td>
      <td>0.460 (0.092)</td>
    </tr>
    <tr>
      <td>Candidate</td>
      <td>0.573 (0.041)</td>
      <td>0.852 (0.124)</td>
      <td>0.400 (0.102)</td>
      <td>0.548 (0.093)</td>
    </tr>
    <tr>
      <td>Random</td>
      <td>0.487 (0.043)</td>
      <td>0.090 (0.018)</td>
      <td>0.045 (0.023)</td>
      <td>0.082 (0.030)</td>
    </tr>
    <tr>
      <td rowspan="6">1000</td>
      <td>Full Finetune</td>
      <td>0.182 (0.007)</td>
      <td>0.991 (0.009)</td>
      <td>0.442 (0.053)</td>
      <td>0.341 (0.016)</td>
    </tr>
    <tr>
      <td>Specific</td>
      <td>0.235 (0.008)</td>
      <td>0.976 (0.012)</td>
      <td>0.265 (0.041)</td>
      <td>0.329 (0.025)</td>
    </tr>
    <tr>
      <td>Plastic</td>
      <td>0.348 (0.049)</td>
      <td>0.125 (0.021)</td>
      <td>0.047 (0.006)</td>
      <td>0.093 (0.009)</td>
    </tr>
    <tr>
      <td>Stubborn</td>
      <td>0.203 (0.013)</td>
      <td>0.989 (0.006)</td>
      <td>0.315 (0.031)</td>
      <td>0.329 (0.016)</td>
    </tr>
    <tr>
      <td>Candidate</td>
      <td>0.184 (0.013)</td>
      <td>0.996 (0.001)</td>
      <td>0.370 (0.045)</td>
      <td>0.327 (0.025)</td>
    </tr>
    <tr>
      <td>Random</td>
      <td>0.254 (0.049)</td>
      <td>0.400 (0.085)</td>
      <td>0.072 (0.006)</td>
      <td>0.146 (0.010)</td>
    </tr>
  </tbody>
</table>

［#249］
![](./images/1095943058349883393_23.jpg)

［#250］
Figure 11. Non-Dissonant updates with GPT2-XL compared to small, a different visualization. Scatter plot of old (x) vs new (y) knowledge during non-dissonant updates. Same conditions as in Fig. 10. We can see clearly how in all cases, the accuracy on previous knowledge remains high. The lottery-ticket effect is also visible where free neurons struggle to efficient pack novel facts.

### Generalization

［#251］
![](./images/1095943058349883393_24.jpg)
［#252］
(a) 10 Facts

［#253］
![](./images/1095943058349883393_25.jpg)
［#254］
(b) 100 Facts

［#255］
![](./images/1095943058349883393_26.jpg)
［#256］
(c) 1000 Facts

---

### Accuracy on New Knowledge

［#257］
![](./images/1095943058349883393_27.jpg)
［#258］
(d) 10 Facts

［#259］
![](./images/1095943058349883393_28.jpg)
［#260］
(e) 100 Facts

［#261］
![](./images/1095943058349883393_29.jpg)
［#262］
(f) 1000 Facts

---

### Accuracy on Old Knowledge

［#263］
![](./images/1095943058349883393_30.jpg)
［#264］
(g) 10 Facts

［#265］
![](./images/1095943058349883393_31.jpg)
［#266］
(h) 100 Facts

［#267］
![](./images/1095943058349883393_32.jpg)
［#268］
(i) 1000 Facts

［#269］
Figure 12. **Dissonant** updates with GPT2-small - impact of the number of conflicting facts. Each row represents a distinct metric: accuracy on the Generalization side dataset (paraphrased versions of the new facts), accuracy on New Knowledge, and Accuracy on Old Knowledge. Within each row, the subplots correspond to the number of conflicting facts introduced (10 Facts, 100 Facts, and 1000 Facts).

### Old Knowledge
［#270］
![](./images/1095943058349883393_33.jpg)
［#271］
(a) Old Knowledge

### New Knowledge
［#272］
GPT2-small from 2k to 20k neurons

［#273］
![](./images/1095943058349883393_34.jpg)
［#274］
(b) New Knowledge

### Generalization
［#275］
![](./images/1095943058349883393_35.jpg)
［#276］
(c) Generalization

---

### GPT2-XL from 2k to 20k neurons

［#277］
![](./images/1095943058349883393_36.jpg)
［#278］
(d) Old Knowledge

［#279］
![](./images/1095943058349883393_37.jpg)
［#280］
(e) New Knowledge

［#281］
![](./images/1095943058349883393_38.jpg)
［#282］
(f) Generalization

---

### GPT2-XL with 10X more neurons

［#283］
![](./images/1095943058349883393_39.jpg)
［#284］
(g) Old Knowledge

［#285］
![](./images/1095943058349883393_40.jpg)
［#286］
(h) New Knowledge

［#287］
![](./images/1095943058349883393_41.jpg)
［#288］
(i) Generalization

---

［#289］
Figure 13. **Dissonant** updates with GPT2-XL: whether the model learns new knowledge or not, old knowledge is severely destroyed regardless of the strategy Experiments with 1000 facts using the best learning rate we found for Full Finetuning.
