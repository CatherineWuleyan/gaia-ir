# Soft-TransFormers for Continual Learning

Haeyong Kang and Chang D. Yoo
Korea Advanced Institute of Science and Technology (KAIST),
School of Electrical Engineering (EE), {haeyong.kang, cd_yoo}@kaist.ac.kr

## ABSTRACT
Inspired by Well-initialized Lottery Ticket Hypothesis (WLTH), which provides suboptimal fine-tuning solutions, we propose a novel fully fine-tuned continual learning (CL) method referred to as Soft-TransFormers (Soft-TF). Soft-TF se- quentially learns and selects an optimal soft-network or subnetwork for each task. During sequential training in CL, Soft-TF jointly optimizes the weights of sparse layers to obtain task-adaptive soft (real-valued) networks or subnetworks (binary masks), while keeping the well-pre-trained layer parameters frozen. In inference, the identified task-adaptive network of Soft-TF masks the parameters of the pre- trained network, mapping to an optimal solution for each task and minimizing Catastrophic Forgetting (CF) - the soft-masking preserves the knowledge of the pre-trained network. Extensive experiments on Vision Transformer (ViT) and CLIP demonstrate the effectiveness of Soft-TF, achieving state-of-the-art performance across various CL scenarios, including Class-Incremental Learning (CIL) and Task-Incremental Learning (TIL), supported by convergence theory. The public code is available at https://github.com/ihaeyong/Soft-TF.

## 1 INTRODUCTION
Continual Learning (CL), also known as Lifelong Learning (Thrun, 1995; Rusu et al., 2016; Zenke et al., 2017; Hassabis et al., 2017), is a learning paradigm where a series of tasks are learned sequen- tially. The principle objective of continual learning is to replicate human cognition, characterized by the ability to learn new concepts incrementally throughout one’s lifespan. An optimal continual learning system could facilitate a positive forward and backward transfer, leveraging the knowledge gained from previous tasks to solve new ones, while also updating its understanding of previous tasks with the new knowledge. However, achieving continual learning is challenging due to the occurrence of catastrophic forgetting or catastrophic interference (McCloskey & Cohen, 1989), a phenomenon where the performance of the model on previous tasks deteriorates significantly when it learns new tasks. This can make it challenging to retain the knowledge acquired from previous tasks, ultimately leading to a decrease in overall performance. To address the issue of catastrophic forgetting during continual learning, numerous conventional approaches have been proposed on Convolutional Neural Networks (CNNs), which can be broadly classified as follows: (1) Regularization-based methods (Kirkpatrick et al., 2017a; Chaudhry et al., 2020; Jung et al., 2020; Titsias et al., 2020; Mirzadeh et al., 2021) aim to keep the learned information of past tasks during continual training aided by sophisticatedly designed regularization terms, (2) Rehearsal-based methods (Rebuffi et al., 2017; Riemer et al., 2018; Chaudhry et al., 2019a;b; Saha et al., 2021) utilize a set of real or synthesized data from the previous tasks and revisit them, and (3) Architecture-based methods (Mallya et al., 2018; Serr`a et al., 2018; Li et al., 2019; Wortsman et al., 2020; Kang et al., 2022; 2023) propose to minimize the inter-task interference via newly designed architectural components.

Developing neural network models that leverage large-scaled pre-trained models. i.e., Vision Trans- former (ViT) (Dosovitskiy et al., 2020) and Contrastive Language-Image Pre-training (CLIP) (Radford et al., 2021) leads to a new paradigm shift referred to as (4) Prompt-based methods in Continual Learning (CL). Prompt-based methods learn continual representations to provide fixed pre-trained transformers with additional instruction. Notably, while L2P (Wang et al., 2022c) stands out as the seminal work that bridges the gap between prompting and continual learning, DualPrompt (Wang et al., 2022b) introduces an innovative approach to affixing complementary prompts to the pre-trained backbone, thereby enabling the acquisition of both task-invariant and task-specific instructions.


![](./images/1068413726366892118_1.jpg)

Figure 1: **Soft-TransFormers (Soft-TF)**: the objective is to design a fully fine-tuned model that works well across multiple continual learning settings with incurring task-wise soft network training of attention and feed forward networks, leveraged by WLTH.

Additionally, other notable contributions in this field encompass DyTox (Douillard et al., 2022), S-Prompt (Wang et al., 2022a), CODA-P (Smith et al., 2023b), ConStruct-VL (Smith et al., 2023a), ST-Prompt (Pei et al., 2023), and LGCL (Khan et al., 2023). Recently, Qiao et al. (2024) investigated prompt-projection for better generalized continual learners.

With prior developments of representational research, current prompt-based models can be fine-tuned using trained prompts to improve their performance on sequential tasks, and the fixed pre-trained backbone can consistently provide unforgettable base session knowledge. However, prompt-based models come with several disadvantages and limitations. First, the effectiveness of prompt-based CL heavily relies on the quality and design of the sample or task-relevant prompts. Poorly trained prompts could lead to suboptimal performance or tend to be biased. Second, managing and maintaining a large set of prompts can become cumbersome and unmanageable as the number of tasks increases. Lastly, prompt tuning is not as flexible as full fine-tuning. The only prompt-tuning of the pre-trained model cannot capture all the nuances of uncorrelated sequential tasks even though leveraging the frozen well-initialized model pre-trained on large-scale datasets since the only frozen well-initialized model provides global solutions rather than task-specific solution. These disadvantages help make informed decisions about when and how to use prompt-based models and explore alternative methods like full fine-tuning for more robust and flexible prompt-based continual learning performance.

To overcome the limitations of conventional prompt-based methods, the central focus of this work is to pinpoint the most optimal winning ticket or fine-tuning representations of frozen pre-trained networks such as Transformers in continual learning scenarios. We focus on two main issues when sequential full fine-tuning the pre-trained foundation models: (1) Catastrophic Forgetting (CF) and (2) parameter-efficient fine-tuning CL model. To deploy a practical model to deal with the two points, we suggest a new paradigm for Continual Learning (CL), named *Well-initialized Lottery Ticket Hypotehesis*:

**Well-initialized Lottery Ticket Hypothesis (WLTH).** *A well-initialized dense neural network contains globally minimal solutions that can retain the prior class knowledge while providing room to learn the new class knowledge through isolated fine-tuning of the networks or subnetworks.*

Leveraged by the WLTH, this work proposes a new Soft-TransFormer (Soft-TF) to address fine-tuning with minimal CF, as shown in Figure 1. We could find task-specific soft-networks or subnetworks based on well-trained frozen transformer parameters that incrementally learn task-adaptive weights associated with each task scenario.

Our contributions can be summarized as follows:

- Inspired by **Well-initialized Lottery Ticket Hypothesis (WLTH)**, we propose a novel continual learning method referred to as Soft-TransFormers (Soft-TF), which learns compact task-specific soft-networks or subnetworks from well pre-trained parameters for each task.
- Extensive experiments demonstrate the Soft-TF leads to better generalized continual models than baselines such as DualPrompts, achieving state-of-the-art performances on various class-incremental learning (CIL) and task-incremental learning (TIL) scenarios, as shown in **Figure 2**.

![](./images/1068413726366892118_2.jpg)
![](./images/1068413726366892118_3.jpg)

(a) DualPrompt-PGP *v.s.* Soft-TransFormers (Soft-TF) (b) DualPrompt-WSN *v.s.* Soft-TransFormers (Soft-TF)

Figure 2: Radar Chart of Comparisons in terms of average accuracy and forgetting between baselines and our SOTA method (Soft-TF). DualPrompt-PGP (Qiao et al., 2024) and DualPrompt-WSN (Kang et al., 2022) (c% sparsity) are baselines for prompt tuning and subnetworks. ACC refers to the average accuracy metric (higher is better). FOR refers to the forgetting metric (lower is better). Different scale standards are adopted for two metrics on benchmark datasets.

## 2 RELATED WORKS
**Continual Learning** (McCloskey & Cohen, 1989; Thrun, 1995; Kumar & Daume III, 2012; Li & Hoiem, 2016) is the challenge of learning a sequence of tasks continuously while utilizing and preserving previously learned knowledge to improve performance on new tasks. Four major approaches have been proposed to tackle the challenges of continual learning, such as catastrophic forgetting. One such approach is *Regularization-based approaches* (Kirkpatrick et al., 2017a; Chaudhry et al., 2020; Jung et al., 2020; Titsias et al., 2020; Mirzadeh et al., 2021), which aim to reduce catastrophic forgetting by imposing regularization constraints that inhibit changes to the weights or nodes associated with past tasks. *Rehearsal-based approaches* (Rebuffi et al., 2017; Chaudhry et al., 2019a;b; Saha et al., 2021; Deng et al., 2021; Sun et al., 2023; Sarfraz et al., 2023; Mai et al., 2021; Lin et al., 2023; Aljundi et al., 2019; Caccia et al., 2021; Chaudhry et al., 2019c; Liang & Li, 2024; Buzzega et al., 2020) store small data summaries to the past tasks and replay them during training to retain the acquired knowledge. Some approaches in this line of work (Shin et al., 2017; Aljundi et al., 2019) accommodate the generative model to construct the pseudo-rehearsals for previous tasks. *Architecture-based approaches* (Mallya et al., 2018; Serrà et al., 2018; Li et al., 2019; Wortsman et al., 2020; Kang et al., 2022; 2023; 2024b;a) use the additional capacity to expand (Xu & Zhu, 2018; Yoon et al., 2018), dynamic representation (Yan et al., 2021; Singh et al., 2020) or isolate (Rusu et al., 2016) model parameters, preserving learned knowledge and preventing forgetting. Rehearsal and architecture-based methods have shown remarkable efficacy in suppressing catastrophic forgetting but require additional capacity for the task-adaptive parameters (Wortsman et al., 2020) or the replay buffers. Recently, *Prompt-based approaches*, an emerging transfer learning technique, harnesses a fixed function of pre-trained Transformer models. This empowers the language model to receive additional instructions for enhancing its performance on downstream tasks. Notably, while L2P (Wang et al., 2022c) stands out as the seminal work that bridges the gap between prompting and continual learning, DualPrompt (Wang et al., 2022b) introduces an innovative approach to affixing complementary prompts to the fixed pre-trained backbone. Here, we introduce a new approach to update the fixed pre-trained parameters through learnable sparse networks under the convergence theory, maximumly enabling the acquisition of task-invariant and task-specific instructions.

3

Prompt-based CL. With recent advances in Vision Transformers (Khan et al., 2022) and prompt-based fine-tuning in NLP (Li & Liang (2021)), Wang et al. (2022c) have shown that interacting with an ImageNet pre-trained model via prompt learning is a promising approach, L2P (Wang et al., 2022c) for continual learning (DualPrompt (Wang et al., 2022b), DyTox (Douillard et al., 2022), S-Prompt Wang et al. (2022a), CODA-P (Smith et al., 2023b), ConStruct-VL Smith et al. (2023a), ST-Prompt (Pei et al., 2023), and LGCL (Khan et al., 2023)). Recently, Prompt Gradient Projection (PGP) (Qiao et al., 2024), a small set of learnable orthogonal parameters, is appended to the input and enables quick adaptation of a frozen ImageNet pre-trained model to new streaming tasks. Their analysis shows that directly leveraging the pre-trained vision-language model without introducing any learnable parameters is a simple yet promising approach to continual learning. The PGP adopted a joint vision-language model like CLIP (Radford et al., 2021) for continual learning, which presents multiple advantages. It enables catering for practical scenarios with no well-defined task identities and boundaries, and the model is required to adapt to streaming data dynamically in a task-agnostic manner. However, prompt-based models come with several disadvantages. Poorly trained prompts could lead to suboptimal performance or tend to be biased. Moreover, prompt tuning could not capture all nuances of uncorrelated sequential tasks. These disadvantages lead to exploring alternative methods like full fine-tuning or hybrid approaches for more robust and flexible prompt-based model performance. In this work, to alleviate these issues, we investigate a fully fine-tuning of well-pre-trained transformers on training soft networks and finding competitive subnetworks.

## 3 PREREQUISITES
We start with conventional prompt-based continual learning methods using Vision Transformer (ViT) (Dosovitskiy et al., 2020) and Contrastive Language-Image Pre-training (CLIP) (Radford et al., 2021) in Class Incremental Learning (CIL) and Task Incremental Learning (TIL) scenarios.

### 3.1 PRELIMINARIES
Problem Statement. Continual Learning (CL) involves training deep neural networks (DNN) on time-variant data represented as a sequence of tasks, $\mathcal{D} = \{\mathcal{D}_1, \cdots, \mathcal{D}_\mathcal{T}\}$. Each $t$-th task, $\mathcal{D}_t = \{(\boldsymbol{x}_i^t, y_i^t)_{i=1}^{n_t}\}$ consists of $n_t$ tuples where $\boldsymbol{x}_i^t \in \mathcal{X}_t$ is an input sample and $y_i^t \in \mathcal{Y}_t$ is the corresponding label. When a task $\mathcal{X}_t$ arrives, a model $f_\theta$ is trained for the current task, while data from previous tasks is inaccessible. This work focuses primarily on class incremental learning (CIL), in which the task-ID is not given during inference.

Soft & Subnetworks have been explored in continual learning through two notable approaches. One approach, known as supermasks (Wortsman et al., 2020), produces outputs by $\boldsymbol{p} = f(\boldsymbol{x}, \boldsymbol{w} \odot \boldsymbol{m})$, where $\odot$ denotes elementwise multiplication. In this method, the weights $\boldsymbol{w}$ remain fixed at their initialization, with bias terms set to $0$ and other parameters initialized to $\pm c$ with equal probability where the constant $c$ is the standard deviation of the corresponding Kaiming normal distribution (He et al., 2015). Another line of work includes WSN (Kang et al., 2022) and SoftNet (Kang et al., 2023), which jointly learn the model weights $\boldsymbol{w}$ and task-adaptive subnetworks $\boldsymbol{m}$. The parameter-efficient reusable subnetworks are obtained by iteratively selecting the top-$c\%$ of the weights based on an importance score $\boldsymbol{s}$ at each layer. WSN has primarily demonstrated its effectiveness in Convolutional Neural Networks (CNNs). However, its pruning mechanism for pre-trained Transformers, such as ViT, remains unexplored. To discover the competitive sparseness in Transformers, we detail the WSN-style task-adaptive fine-tuning and the learnable soft-networks $\boldsymbol{m}$ of Transformers, presenting these adaptations for the first time with empirical observations. The soft-networks originate from learned parameters distributed with $\mu \approx 1.0$ & various variances, as stated in Figure 6.

### 3.2 PROMPT-BASED CLASS INCREMENTAL LEARNING (CIL)
A simple yet effective prompt-based (prompt-tuning) CIL model: Learning to Prompt (L2P) (Wang et al., 2022c) is first proposed. In this model, a prompt $p$, a tiny set of trainable tokens combined with image features, is fed into the Vision Transformer (ViT) to help the model resist forgetting. To select suitable prompts for task-specific training, L2P utilizes a prompt pool $P$ containing numerous prompt-key pairs, $\{p_t, k_t\}_{t=1}^\mathcal{T}$, where $p_t \in \mathbb{R}^{1 \times D}$ represents the $t$-th task prompt, $k_t$ represents the $t$-th coresponding task key, and $\mathcal{T}$ is the total number of prompt-key pairs.

4

Building on L2P, DualPrompt (Wang et al., 2022b) divided the prompts into expert (E-) prompts and general (G-) prompts for distinct features learning. DualPrompt also replaced prompt-tuning with prefix-tuning, which was successfully proven in NLP. DyTox (Douillard et al., 2022) designed a novel task attention block that utilized task tokens to infer task identifiers. Coda-Prompt (Smith et al., 2023b) replaced the prompt pool with a decomposed prompt, represented by a weighted sum of learnable prompt components, which optimized itself in an end-to-end fashion, providing high plasticity. LGCL (Khan et al., 2023) introduced text information into the learning of prompt pool, improving performance without any additional learnable parameters.

Recently, Qiao et al. (2024) introduced Prompt Gradient Projection (PGP), which applies an orthogonal condition on the prompt gradient to reduce forgetting via the self-attention mechanism in ViT effectively. Although various prompt-based continual learners have demonstrated state-of-the-art performance, they do not explicitly model task-specific fine-tuning and forgetting within the continual learning framework. In this work, we address task-specific fine-tuning and gradient-based task identification in CIL and TIL scenarios by leveraging prompt-tuning and learnable sparse networks.

![](./images/1068413726366892118_4.jpg)

Figure 3: Soft-TransFormers (Soft-TF): At training time, the E-Prompt and the Soft-network are selected according to task identity, and the selected G-Prompt, E-Prompt, and the Soft-networks (Soft-Attention and Feed Forwards) are trained together with a classifier. At test time, an input is transformed by a query function (Prompt ID) or task identifier (Gradient ID) to match the closest task key $\boldsymbol{k}_t$, E-prompt $\boldsymbol{e}_t$ and Soft-networks $\boldsymbol{m}_t^{\{K,Q,V\}}$. Note task identifier is depicted in Section 5.

## 4 TRANSFORMER WITH LEARNABLE SUBNETWORKS

In this section, we explain how Soft-TransFormers (Soft-TF) leverage learnable soft-networks to train sequential tasks while keeping the well-pretrained model parameters fixed. To introduce our novel Soft-TF and provide a clearer understanding, we draw on a partial explanation of DualPrompt.

### 4.1 SOFT-MSA LAYERS

To address the task-specific fine-tuning of the pre-trained model, such as ViT, this work proposes a new Soft-TransFormer (Soft-TF), as illustrated in Figure 1. The proposed Soft-TF consists of a conventional neural network, like a multilayer transformer with multihead attention and forward networks. Using well-trained transformer parameters, we could discover task-specific soft-networks, as depicted in Figure 3. The Soft-TF incrementally learns model weights and task-adaptive soft-masks with well-pre-trained and soft-network parameters $\boldsymbol{m}$.

Given a pre-trained parameter $\boldsymbol{\theta}$ and learnable soft-parameters $\boldsymbol{m}$, Soft-ViT is represented as $f_{\boldsymbol{\theta} \odot \boldsymbol{m}}$, consisting of $N$ consecutive soft-MSA layers. We extend the notation by denoting the input embedding feature of the $l_*$-th learnable soft-MSA layer as $\boldsymbol{h}^{(l_*)}$, where $l_* = 1,2,\dots,N$, and $l_*$ can refer to either the G-Prompt layer $l_g$ or the E-Prompt layer $l_e$. Note that while the pre-trained parameters $\boldsymbol{\theta}$ remain fixed, the soft-parameters $\boldsymbol{m}$ are updated to provide task-specific solutions.

G-prompt. $\boldsymbol{g} \in \mathbb{R}^{L_g \times D}$ with sequence length $L_g$ and embedding dimension $D$, is a shared parameter for all tasks. G-Prompt is attached to the $l_g$-th MSA layer to transform $\boldsymbol{h}^{(l_g)}$ via a prompting function as follows:
$$
\boldsymbol{h}_{g}^{(l_g)}=f_{\boldsymbol{\theta}}^{prompt}(\boldsymbol{g}, \boldsymbol{h}^{(l_g)}), \tag{1}
$$

5

where $f_{\boldsymbol{\theta}}^{\text{prompt}}$ defines the approach for attaching the prompt to the hidden embeddings.

E-prompt & Soft-networks. $\boldsymbol{e} = \{\boldsymbol{e}_t\}_{t=1}^{\mathcal{T}}$ is a set of task-dependent parameters, where $\boldsymbol{e}_t \in \mathbb{R}^{L_e \times D}$ has as sequence length of $L_e$ and the same embedding dimension $D$ as the G-prompt, and $\mathcal{T}$ is the total number of tasks. Unlike the shared G-prompt, each $\boldsymbol{e}_t$ is associated with a task-specific key $\boldsymbol{k}_t \in \mathbb{R}^D$, which is also a learnable parameter aimed at capturing representative features of a task. For an input example from the $t$-th task, to attach E-prompt to the $l_e$-th soft-MSA layer, we apply the prompting function in a similar way:
$$
\boldsymbol{h}_{e}^{\left(l_{e}\right)}=f_{\boldsymbol{\theta} \odot \boldsymbol{m}}^{\text{prompt}}\left(\boldsymbol{e}_{t}, \boldsymbol{h}^{\left(l_{e}\right)}\right). \tag{2}
$$

## 4.2 PROMPTS WITH LEARNABLE SUBNETWORKS

G- and E-prompts, along with learnable soft-networks, encode specific types of instructions during training with the backbone and work together to guide the model's predictions during inference. We have demonstrated the method for attaching prompts and learnable soft-networks to a single soft-MSA layer. Similarly to the approach taken in DualPrompt (Wang et al., 2022b), we also investigate layers of E-prompts with learnable soft-networks $\boldsymbol{m}$, while utilizing the layers designated for G-prompts.

Layers of G- and E-Prompts. We use the multilayered extension of both types of prompts: $\boldsymbol{g} = \{\boldsymbol{g}^{(l_g)}\}_{l_g=start_g}^{end_g}$, where $\boldsymbol{g}^{(l_g)} \in \mathbb{R}^{L_g \times D}$ represents the G-prompt attached to the $l_g$-th MSA layer. Similarly, we define $\boldsymbol{e}_t = \{\boldsymbol{e}_t^{(l_e)}\}_{l_e=start_e}^{end_e}$ for the $l_e$-th conventional MSA layer. In this configuration, the G-prompt $\boldsymbol{g}^{(l_g)}$ is attached from the $start_g$-th to the $end_g$-th conventional MSA layers, and the E-prompt $\boldsymbol{e}_t^{(l_e)}$ is attached to the $[start_e, end_e]$-th soft-MSA layers, ensuring that there is no overlap between them. In our experiments, we follow the $l_g \notin [start_e, end_e]$ ($l_g = [1,2]$) settings used in DualPrompt and empirically search for the optimal $[start_e, end_e]$ layers for the learnable subnetworks through ablation studies.

Learnable Soft-networks. The prompting function $f_{\boldsymbol{\theta} \odot \boldsymbol{m}}^{\text{prompt}}$ determines how prompts $(\boldsymbol{p})$ are combined with fine-tuned soft $(\boldsymbol{\theta} \odot \boldsymbol{m})$ embedding features. From another perspective, $f_{\boldsymbol{\theta} \odot \boldsymbol{m}}^{\text{prompt}}$ directly influences the interaction between high-level instructions in the prompts and low-level representations. Therefore, we believe that a well-designed prompting function, along with task- specific parameters, is crucial for optimizing overall continual learning performance.

Specifically, applying a prompting and fine-tuning function $f_{\boldsymbol{\theta} \odot \boldsymbol{m}}^{\text{prompt}}$ can be seen as modifying the inputs to the soft-MSA layers. Let the input to the soft-MSA layer be $\boldsymbol{h} \in \mathbb{R}^{L \times D}$, and denote the input query, key, and values for the soft-MSA layer as $\boldsymbol{h}_Q$, $\boldsymbol{h}_K$, and $\boldsymbol{h}_V$, respectively. A soft-MSA layer is defined by the following equation:
$$
\begin{aligned}
\operatorname{MSA}(\boldsymbol{h}_{Q}, \boldsymbol{h}_{K}, \boldsymbol{h}_{V}) &=\operatorname{Concat}(\boldsymbol{h}_{1}, \cdots, \boldsymbol{h}_{i}, \cdots, \boldsymbol{h}_{n}) \boldsymbol{w}^{O} \odot \boldsymbol{m}^{O} \\
\text{where } \boldsymbol{h}_{i} &=\operatorname{Attention}(\boldsymbol{h}_{Q}(\boldsymbol{w}_{i}^{Q} \odot \boldsymbol{m}^{Q}), \boldsymbol{h}_{K}(\boldsymbol{w}_{i}^{K} \odot \boldsymbol{m}^{K}), \boldsymbol{h}_{V}(\boldsymbol{w}_{i}^{V} \odot \boldsymbol{m}^{V})),
\end{aligned} \tag{3}
$$
where $\boldsymbol{w}_i^O$, $\boldsymbol{w}_i^Q$, $\boldsymbol{w}_i^K$, and $\boldsymbol{w}_i^V$ are fixed projection matrices while $\boldsymbol{m}^O$, $\boldsymbol{m}^Q$, $\boldsymbol{m}^K$, and $\boldsymbol{m}^V$ are learnable parameters. $s$ is the number of heads. In ViT, $\boldsymbol{h}_Q = \boldsymbol{h}_K = \boldsymbol{h}_V$. Here, we define a unified prompt parameter with a sequence length of $L_p$, such as $\boldsymbol{p} \in \mathbb{R}^{L_p \times D}$ for a single-layered G- or E-prompt.

## 4.3 FINE-TUNING ON WELL-INITIALIZED PARAMETERS

In this framework, we concatenate the prompts $\boldsymbol{p}_t$ and the embedding sequence $\boldsymbol{x}_t$, i.e., inputs from $t$-th task, along the embedding dimension: $\boldsymbol{z}_t = [\boldsymbol{p}_t; \boldsymbol{x}_t]$. With the weights of $\boldsymbol{w}^Q \odot \boldsymbol{m}^Q$, $\boldsymbol{w}^K \odot \boldsymbol{m}^K$, $\boldsymbol{w}^V \odot \boldsymbol{m}^V$, the soft-transformer takes query $(\boldsymbol{q}_t = (\boldsymbol{w}^Q \odot \boldsymbol{m}^Q)\boldsymbol{z}_t)$ and key $(\boldsymbol{k}_t = (\boldsymbol{w}^K \odot \boldsymbol{m}^K)\boldsymbol{z}_t)$ as input of the soft-MSA layer. The soft-attention matrix is then given by:
$$
\boldsymbol{a}_t = \text{softmax}\left( \frac{\boldsymbol{q}_t \boldsymbol{k}_t^T}{\sqrt{D/n}} \right) \tag{4}
$$
where we focus on $\boldsymbol{q}_t \boldsymbol{k}_t^T = (\boldsymbol{w}^Q \odot \boldsymbol{m}^Q)\boldsymbol{z}_t$ and $\boldsymbol{z}_t^T (\boldsymbol{w}^K \odot \boldsymbol{m}^K)^T$. First, the trainable prompt parameters can be denoted as:
$$
\boldsymbol{z}_{t} \cdot \boldsymbol{z}_{t}^{T}=\left[\begin{array}{l}
\boldsymbol{p}_{t} \\
\boldsymbol{x}_{t}
\end{array}\right]\left[\boldsymbol{p}_{t} \boldsymbol{x}_{t}\right]=\left[\begin{array}{cc}
\boldsymbol{p}_{t} \boldsymbol{p}_{t}^{T} & \boldsymbol{p}_{t} \boldsymbol{x}_{t}^{T} \\
\boldsymbol{x}_{t} \boldsymbol{p}_{t}^{T} & \boldsymbol{x}_{t} \boldsymbol{x}_{t}^{T}
\end{array}\right] \tag{5}
$$


Second, the trainable soft-attention layer's parameters with $\boldsymbol{m}^Q$ and $\boldsymbol{m}^K$ are as follows:

$$
\left\{
\begin{aligned}
&\boldsymbol{m}^Q \cdot \boldsymbol{p}_t \boldsymbol{p}_t^T \cdot (\boldsymbol{m}^K)^{T}, \\
&\boldsymbol{m}^Q \cdot \boldsymbol{x}_t \boldsymbol{p}_t^T \cdot (\boldsymbol{m}^K)^{T}, \\
&\boldsymbol{m}^Q \cdot \boldsymbol{p}_t \boldsymbol{x}_t^T \cdot (\boldsymbol{m}^K)^{T}, \\
&\boldsymbol{m}^Q \cdot \boldsymbol{x}_t \boldsymbol{x}_t^T \cdot (\boldsymbol{m}^K)^{T}.
\end{aligned}
\right\}
\tag{6}
$$

where $\boldsymbol{w}^Q$ and $\boldsymbol{w}^K$ are frozen and unchanged during training and test.

## 4.4 THE OPTIMIZATION OF SOFT-TRANSFORMERS

The overall process of the Soft-TransFormers (Soft-TF) during training and testing is described as **Algorithm 1** and **Algorithm 2**. We denote the architecture with attached prompts as $f_{\boldsymbol{g}, \boldsymbol{e}_t, \boldsymbol{m}_t}$. The input $\boldsymbol{x}$ of the $t$-th task is transformed using $f_{\boldsymbol{g}, \boldsymbol{e}_t, \boldsymbol{m}_t}$ and then passed to the classification head $f_{\phi}$, parameterized by $\phi$, for prediction. Finally, we train the two prompts, the task keys, the soft-attention parameters, and the newly-initialized classification head in an end-to-end manner:

$$
\min _{\boldsymbol{g}, \boldsymbol{e}_{t}, \boldsymbol{m}_{t}, \boldsymbol{k}_{t}, \phi} \mathcal{L}\left(f_{\phi}\left(f_{\boldsymbol{g}, \boldsymbol{e}_{t}, \boldsymbol{m}_{t}}(\boldsymbol{x})\right), y\right)+\lambda \mathcal{L}_{\text {match }}\left(\boldsymbol{x}, \boldsymbol{k}_{t}\right), \quad \boldsymbol{x} \in \mathcal{D}_{t},
\tag{7}
$$

Here, $\mathcal{L}$ represents the cross-entropy loss, and $\mathcal{L}_{\text {match }}(\boldsymbol{x}, \boldsymbol{k}_{t})=\gamma(\boldsymbol{q}(\boldsymbol{x}), \boldsymbol{k}_{t})$ denotes the matching loss, where $\boldsymbol{q}(\boldsymbol{x})=f(\boldsymbol{x})[0]$ corresponds to the feature vector associated with the [class] token (Dosovitskiy et al., 2020; Wang et al., 2022b), and $\gamma$ is the cosine similarity. The scalar $\lambda$ serves as a balancing factor between the losses; here, we follow the same DualPrompt setting as a baseline.

**Analysis of Soft-TF for Convex-Lipschitz Functions.** To analyze the convergence rate of the Soft-Transformer, we focus on the case of convex-Lipschitz functions. Let $\boldsymbol{w}^{*}=\{\boldsymbol{g}^{*}, \boldsymbol{e}_{t}^{*}, \boldsymbol{m}_{t}^{*}\}$ be any vector, and let $B$ be an upper bound on $\|\boldsymbol{w}^{*}\|$ when $\boldsymbol{w}^{(1)}=\mathbf{0}$, or $\boldsymbol{w}^{(1)}$ is an initial state. It is helpful to consider $\boldsymbol{w}^{*}$ as the minimizer of $f(\boldsymbol{w})$, although the following analysis applies to any $\boldsymbol{w}^{*}$.

We derive an upper bound on the sub-optimality of our solution relative to $\boldsymbol{w}^{*}$, specifically $f(\overline{\boldsymbol{w}})-f(\boldsymbol{w}^{*})$, where $\overline{\boldsymbol{w}}=\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{w}^{(t)}$. By the definition of $\overline{\boldsymbol{w}}$ and applying Jensen's inequality, we obtain the following (see Appendix A.1 stated in detail):

$$
\begin{aligned}
f(\overline{\boldsymbol{w}})-f(\boldsymbol{w}^{*}) &= f\left(\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{w}^{(t)}\right)-f(\boldsymbol{w}^{*}) \\
& \leq \frac{1}{T} \sum_{t=1}^{T}\left(f(\boldsymbol{w}^{(t)})\right)-f(\boldsymbol{w}^{*}) \\
&=\frac{1}{T} \sum_{t=1}^{T}\left(f(\boldsymbol{w}^{(t)})-f(\boldsymbol{w}^{*})\right).
\end{aligned}
\tag{8}
$$

For every $t$, because of the convexity of $f$, we have that

$$
f(\boldsymbol{w}^{(t)})-f(\boldsymbol{w}^{*}) \leq\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \nabla f(\boldsymbol{w}^{(t)})\right\rangle
\tag{9}
$$

Combining the preceding, we obtain

$$
f(\boldsymbol{w}^{(t)})-f(\boldsymbol{w}^{*}) \leq \frac{1}{T} \sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \nabla f(\boldsymbol{w}^{(t)})\right\rangle
\tag{10}
$$

To bound the right-hand side of the above formula, we rely on the following lemma:
**Lemma 4.1.** Let $\boldsymbol{v}_{1}, \cdots, \boldsymbol{v}_{t}, \cdots, \boldsymbol{v}_{T}$ be an arbitrary sequence of vectors, such as the $t$-th task gradients $\boldsymbol{v}_{t}=\nabla f(\boldsymbol{w})^{t}$. Consider any algorithm with a well-initialized (well pre-trained Transformer from WLTH) starting point $\boldsymbol{w}^{(1)} \neq \mathbf{0}$ and an update rule of the form:

$$
\boldsymbol{w}^{(t+1)}=\boldsymbol{w}^{(t)}-\eta \boldsymbol{v}_{t}
\tag{11}
$$

satisfies with $\|\boldsymbol{w}^{(1)}-\boldsymbol{w}^{*}\|^{2}=\|\boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}\|^{2}$

$$
\begin{aligned}
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle & \leq \frac{1}{2 \eta_{m}}\left\|\boldsymbol{w}_{m}^{(T+1)}-\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta_{m}}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2} \\
&<\frac{1}{2 \eta_{p}}\left\|\boldsymbol{w}_{p}^{(T+1)}-\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta_{p}}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2}
\end{aligned}
\tag{12}
$$


where $\boldsymbol{w}_{m} \neq \boldsymbol{w}_{p}$ since $\boldsymbol{m}$ is learnable parameters in Soft-TransFormers. Specifically, we could assume that $\boldsymbol{w}_{m}=\left(\boldsymbol{w}^{Q} \odot \boldsymbol{m}^{Q}\right) \cdot \boldsymbol{x} \boldsymbol{p}^{T} \cdot\left(\boldsymbol{w}^{K} \odot \boldsymbol{m}^{K}\right)^{T}$ and $\boldsymbol{w}_{p}=\left(\boldsymbol{w}^{Q} \odot \mathbf{1}^{Q}\right) \cdot \boldsymbol{x} \boldsymbol{p}^{T} \cdot\left(\boldsymbol{w}^{K} \odot \mathbf{1}^{K}\right)^{T}$ of Equation 6 and $\boldsymbol{w}^{Q}, \boldsymbol{w}^{K}$ are here frozen pre-trained parameters.

Theorem 4.2. For every $B_{m}<B_{p}<B, \rho>0$ where $B_{m}=\left\|\boldsymbol{w}_{m}^{(T+1)}-\boldsymbol{w}^{*}\right\|$ and $B_{p}=$ $\left\|\boldsymbol{w}_{p}^{(T+1)}-\boldsymbol{w}^{*}\right\|$, if for all $t$ we have that $\left\|\boldsymbol{v}_{t} \leq \rho\right\|$ and we set $\eta \approx \eta_{m} \approx \eta_{p}=\sqrt{\frac{B^{2}}{\rho^{2} T}}$ with large enough $T$, then for every $\boldsymbol{w}^{*}$ with $\left\|\boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}\right\| \leq B$ we have

$$
\frac{1}{T} \sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle \leq \frac{B_{m} \rho}{\sqrt{T}}<\frac{B_{p} \rho}{\sqrt{T}}<\frac{B \rho}{\sqrt{T}}.
\tag{13}
$$

## 5 EXPERIMENTS

We validate our method on several benchmark datasets against continuous learning baselines in Class-Incremental Learning (CIL) and Task-Incremental Learning (TIL).

### 5.1 EXPERIMENTAL SETTINGS
Datasets. We evaluate our method mainly on 1) 10/20-Split-CIFAR100 (Krizhevsky et al., 2009), constructed by splitting the 100 classes into 10 tasks/20 tasks. 2) 10-Split-TinyImageNet (Abai & Rajmalwar, 2019), constructed by splitting the 200 classes into 10 tasks. 3) 10-Split-ImageNet-R (Hendrycks et al., 2021), constructed by splitting the 200 classes into 10 tasks. To show our effectiveness, we additionally compare our method with the baselines on 5-Split-CUB200 and 10-Split-TinyImageNet. The detailed experimental settings are depicted in the Supplementary.

Implementation. For fair comparisons, we set L2P (Wang et al., 2022c), DualPrompt (Wang et al., 2022b), CLIP (Radford et al., 2021), and PGP (Qiao et al., 2024) as our baselines. We follow experimental settings Qiao et al. (2024) entirely.

Baselines. To validate the powerfulness of our method, we compare our results with various CIL baselines including ICaRL (Rebuffi et al., 2017), BiC (Wu et al., 2019), DER++ (Buzzega et al., 2020), LWF (Li & Hoiem, 2017), EWC (Kirkpatrick et al., 2017b), DER+MCG (Cai et al., 2023), and DualPrompt-PGP (Qiao et al., 2024). In addition, we investigate subnetwork solutions such as WSN (Kang et al., 2022) (obtained by selecting top-$c\%$ of weight scores while fixing pre-trained parameters) and SoftNet (Kang et al., 2023) (acquired by selecting top-$c\%$ of weight scores as major tickets while setting $100.0 - \text{top-}c\%$ as minor tickets) in Vision Transformers (ViT) using prompt tuning methods. We adopt average accuracy (ACC) and forgetting (FOR) as our validation metrics (Wang et al., 2022b; Qiao et al., 2024).

Task Inference. At the inference time, we infer task identity for arbitrary pieces of task samples $\boldsymbol{x}$ for finding the proper task nuances and demonstrating full fine-tuning results. We summarize the following two methods:
- **Prompt ID**: For a test example $\boldsymbol{x}$, we simply choose the best matched task index via $\text{argmin}_{t} \gamma(q(\boldsymbol{x}), \boldsymbol{k}_{t})$.
- **Gradient ID**: To infer the task identity, we follow SupSup's one-shot task inference (Wortsman et al., 2020). In short, we assign each learned subnetwork $\boldsymbol{m}_{t}$ a weight $\alpha_{t}$ such that $\sum_{t} \alpha_{t}=1$ and $\alpha_{t}=1 / T>0$ when evaluating all seen tasks. Given an example data point of batch $\boldsymbol{x} \in \boldsymbol{b}$ to classify, we can compute the loss as $\mathcal{L}=\mathcal{H}(f_{\boldsymbol{\theta} \odot\left(\sum_{t} \alpha_{t} \boldsymbol{m}_{t}\right)}^{prompt}(\boldsymbol{x}))$ where $f_{\boldsymbol{\theta}}^{prompt}(\boldsymbol{x})$ is the pre-trained model which outputs logits and $\mathcal{H}$ is the entropy function. From here our inferred task is simply $\hat{t}=\text{argmin}_{t} \frac{\partial \mathcal{H}}{\partial \alpha_{t}}$.

### 5.2 PERFORMANCES
Performances of Soft-TF on CIL. We compare our Soft-TransFormers (Soft-TF) with state-of-the-art CIL baselines, as shown in Table 1. Our Soft-TF significantly outperforms all baselines and upper-bounds of Soft-TF, including L2P and DualPrompt, in both accuracy and forgetting measurements. The performance gain of Soft-TF is especially notable in DualPrompt-based learning compared to

8

Table 1: Performances of Class Incremental Learning (CIL) in terms of accuracy and forgetting on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Exemplar means the total buffer size for rehearsal methods.

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Exemplar</th>
    <th rowspan="2">Task ID</th>
    <th colspan="2">10-Split-CIFAR100</th>
    <th colspan="2">20-Split-CIFAR100</th>
    <th colspan="2">10-Split-ImageNet-R</th>
  </tr>
  <tr>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BiC</td>
    <td>5,000</td>
    <td>-</td>
    <td>81.42</td>
    <td>17.31</td>
    <td>73.02</td>
    <td>6.23</td>
    <td>64.63</td>
    <td>22.25</td>
  </tr>
  <tr>
    <td>DER++</td>
    <td>5,000</td>
    <td>-</td>
    <td>83.94</td>
    <td>14.55</td>
    <td>-</td>
    <td>-</td>
    <td>66.73</td>
    <td>20.67</td>
  </tr>
  <tr>
    <td>iCaRL</td>
    <td>5,000</td>
    <td>-</td>
    <td>66.00</td>
    <td>5.33</td>
    <td>78.02</td>
    <td>5.80</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>DER+MCG</td>
    <td>2,000</td>
    <td>-</td>
    <td>67.62</td>
    <td>14.64</td>
    <td>65.84</td>
    <td>13.72</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>BiC</td>
    <td>1,000</td>
    <td>-</td>
    <td>66.11</td>
    <td>35.24</td>
    <td>63.12</td>
    <td>21.89</td>
    <td>52.14</td>
    <td>36.70</td>
  </tr>
  <tr>
    <td>DER++</td>
    <td>1,000</td>
    <td>-</td>
    <td>61.06</td>
    <td>39.87</td>
    <td>-</td>
    <td>-</td>
    <td>55.47</td>
    <td>34.64</td>
  </tr>
  <tr>
    <td>iCaRL</td>
    <td>1,000</td>
    <td>-</td>
    <td>61.25</td>
    <td>14.19</td>
    <td>71.32</td>
    <td>15.98</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>FT</td>
    <td>-</td>
    <td>-</td>
    <td>33.61</td>
    <td>86.87</td>
    <td>33.52</td>
    <td>53.69</td>
    <td>28.87</td>
    <td>63.80</td>
  </tr>
  <tr>
    <td>EWC</td>
    <td>-</td>
    <td>-</td>
    <td>47.01</td>
    <td>33.27</td>
    <td>36.73</td>
    <td>35.19</td>
    <td>35.00</td>
    <td>56.16</td>
  </tr>
  <tr>
    <td>LWF</td>
    <td>-</td>
    <td>-</td>
    <td>60.69</td>
    <td>27.77</td>
    <td>39.12</td>
    <td>57.91</td>
    <td>38.54</td>
    <td>52.37</td>
  </tr>
  <tr>
    <td>L2P*</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>83.77</td>
    <td>6.63</td>
    <td>71.29</td>
    <td>13.96</td>
    <td>60.44</td>
    <td>9.00</td>
  </tr>
  <tr>
    <td>L2P-PGP*</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>84.34</td>
    <td>5.59</td>
    <td>76.12</td>
    <td>13.26</td>
    <td>61.40</td>
    <td>8.03</td>
  </tr>
  <tr>
    <td>L2P-PGP-Soft-TF</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>86.26</td>
    <td>4.79</td>
    <td>76.17</td>
    <td>15.77</td>
    <td>69.80</td>
    <td>5.13</td>
  </tr>
  <tr>
    <td>L2P-PGP-Soft-TF</td>
    <td>-</td>
    <td>Gradient ID</td>
    <td>86.46</td>
    <td>4.87</td>
    <td>77.67</td>
    <td>15.84</td>
    <td>69.56</td>
    <td>5.28</td>
  </tr>
  <tr>
    <td>DualPrompt-PGP</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>86.92</td>
    <td>5.35</td>
    <td>83.74</td>
    <td>7.91</td>
    <td>69.34</td>
    <td>4.53</td>
  </tr>
  <tr>
    <td>DualPrompt-PGP-Soft-TF</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>92.41</td>
    <td>2.44</td>
    <td>95.14</td>
    <td>1.90</td>
    <td>74.65</td>
    <td>4.39</td>
  </tr>
  <tr>
    <td>DualPrompt-PGP-Soft-TF</td>
    <td>-</td>
    <td>Gradient ID</td>
    <td>92.92</td>
    <td>2.34</td>
    <td>95.89</td>
    <td>1.64</td>
    <td>81.45</td>
    <td>2.89</td>
  </tr>
  <tr>
    <td>DualPrompt</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>86.50</td>
    <td>5.77</td>
    <td>82.98</td>
    <td>8.20</td>
    <td>68.13</td>
    <td>4.46</td>
  </tr>
  <tr>
    <td>DualPrompt-Soft-TF</td>
    <td>-</td>
    <td>Prompt ID</td>
    <td>91.77</td>
    <td>3.37</td>
    <td>94.43</td>
    <td>2.02</td>
    <td>74.70</td>
    <td>6.46</td>
  </tr>
  <tr>
    <td>DualPrompt-Soft-TF (SOTA)</td>
    <td>-</td>
    <td>Gradient ID</td>
    <td>97.87</td>
    <td>0.21</td>
    <td>99.05</td>
    <td>0.24</td>
    <td>82.38</td>
    <td>0.59</td>
  </tr>
  <tr>
    <td>Upper-Bound of Soft-TF</td>
    <td>-</td>
    <td>-</td>
    <td>93.90</td>
    <td>-</td>
    <td>93.90</td>
    <td>-</td>
    <td>80.21</td>
    <td>-</td>
  </tr>
</tbody>
</table>

Table 2: Performances of Class Incremental Learning (CIL) in terms of Pretained-dataset (ImageNet-21K, SAM, DINO) and Task-IDs on 10-Split-CIFAR100 and 5-Split-CUB200.

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Pretrained-dataset</th>
    <th rowspan="2">Task ID</th>
    <th colspan="2">10-Split-CIFAR100</th>
    <th colspan="2">5-Split-CUB200</th>
  </tr>
  <tr>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DualPrompt</td>
    <td>ImageNet-21K</td>
    <td>Prompt ID</td>
    <td>86.50</td>
    <td>5.77</td>
    <td>82.02</td>
    <td>4.23</td>
  </tr>
  <tr>
    <td>DualPrompt-PGP</td>
    <td>ImageNet-21K</td>
    <td>Prompt ID</td>
    <td>86.92</td>
    <td>5.35</td>
    <td>82.46</td>
    <td>3.76</td>
  </tr>
  <tr>
    <td>DualPrompt</td>
    <td>SAM</td>
    <td>Prompt ID</td>
    <td>86.11</td>
    <td>6.08</td>
    <td>82.02</td>
    <td>4.73</td>
  </tr>
  <tr>
    <td>DualPrompt</td>
    <td>DINO</td>
    <td>Prompt ID</td>
    <td>64.18</td>
    <td>23.81</td>
    <td>50.88</td>
    <td>10.10</td>
  </tr>
  <tr>
    <td>DualPrompt-Soft-TF</td>
    <td>ImageNet-21K</td>
    <td>Prompt ID</td>
    <td>92.42</td>
    <td>2.44</td>
    <td>76.17</td>
    <td>9.04</td>
  </tr>
  <tr>
    <td>DualPrompt-Soft-TF (SOTA)</td>
    <td>ImageNet-21K</td>
    <td>Gradient ID</td>
    <td>97.87</td>
    <td>0.21</td>
    <td>87.93</td>
    <td>0.66</td>
  </tr>
  <tr>
    <td>DualPrompt-Soft-TF (SOTA)</td>
    <td>SAM</td>
    <td>Gradient ID</td>
    <td>97.87</td>
    <td>0.21</td>
    <td>87.93</td>
    <td>0.66</td>
  </tr>
  <tr>
    <td>DualPrompt-Soft-TF</td>
    <td>DINO</td>
    <td>Gradient ID</td>
    <td>84.50</td>
    <td>12.27</td>
    <td>69.79</td>
    <td>10.93</td>
  </tr>
  <tr>
    <td>Upper-Bound of Soft-TF</td>
    <td>-</td>
    <td>-</td>
    <td>93.90</td>
    <td>-</td>
    <td>85.56</td>
    <td>-</td>
  </tr>
</tbody>
</table>

L2P, suggesting the importance of global prompt-tuning and multi-head attention prompt-tuning in DualPrompt. Additionally, task-identity inference using Gradient-ID is crucial for achieving full fine-tuning results in CIL. To demonstrate the effectiveness of Soft-TF, Figure 2(a) presents a radar chart comparing DualPrompt-PGP and Soft-Transformers across four benchmark datasets.

Well-initialized LTH (WLTH) on CIL. To demonstrate the efficacy of our proposed method on Well-initialized Lottery Ticket Hypothesis (WLTH) backbones, we evaluate our Soft-Transformers (Soft-TF) by extending two distinct pre-trained models, ViT-DINO and ViT-SAM (Caron et al., 2021; Chen et al., 2021). As shown in Table 2, we tested our method on the 10-Split-CIFAR100 and 5-Split-CUB200 datasets using three pre-trained ViTs: ImageNet-21K, DINO, and SAM, further validating the effectiveness of our approach on non-ImageNet datasets (Krizhevsky et al., 2009; Wah et al., 2011). Surprisingly, when initialized with ImageNet-21K and SAM, DualPrompt-Soft-TF with ImageNet-21K achieved the same performance levels. Moreover, DualPrompt-Soft-TF outperformed all baselines, i.e., DualPrompt-PGP, on both benchmark datasets, indicating that well-initialized weights provide better generalization in continual learning scenarios.

CLIP on CIL and TIL. We conduct our experiments on the 10-Split-CIFAR100 dataset under both Class Incremental Learning (CIL) and Task Incremental Learning (TIL) settings, as shown in Table 3. The results demonstrate that CLIP-Prompt-Soft-TF-L[1-12] significantly improves performance in both settings, indicating that our Soft-TF with Gradient ID is also effective in vision-language models, thereby broadening its applicability.

Table 3: Comparisons of Soft-TF with baselines based on CLIP model on 10-Split-CIFAR100. * denotes our reproduced results.

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Task ID</th>
<th colspan="2">Class Incremental</th>
<th colspan="2">Task Incremental</th>
</tr>
<tr>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP</td>
<td>Prompt ID</td>
<td>73.76</td>
<td>5.60</td>
<td>92.69</td>
<td>2.34</td>
</tr>
<tr>
<td>CLIP-PGP</td>
<td>Prompt ID</td>
<td>79.47</td>
<td>4.23</td>
<td>93.00</td>
<td>1.58</td>
</tr>
<tr>
<td>CLIP*</td>
<td>Prompt ID</td>
<td>74.60</td>
<td>7.75</td>
<td>93.59</td>
<td>2.80</td>
</tr>
<tr>
<td>CLIP-PGP*</td>
<td>Prompt ID</td>
<td>74.63</td>
<td>7.76</td>
<td>93.67</td>
<td>2.83</td>
</tr>
<tr>
<td>CLIP-Prompt</td>
<td>Prompt ID</td>
<td>70.27</td>
<td>12.95</td>
<td>93.36</td>
<td>3.07</td>
</tr>
<tr>
<td>CLIP-Prompt-Soft-TF-L[3,4,5]</td>
<td>Prompt ID</td>
<td>71.58</td>
<td>7.73</td>
<td>95.29</td>
<td>1.12</td>
</tr>
<tr>
<td>CLIP-Prompt-Soft-TF-L[3,4,5]</td>
<td>Gradient ID</td>
<td>76.77</td>
<td>5.59</td>
<td>95.29</td>
<td>1.12</td>
</tr>
<tr>
<td>CLIP-Prompt-Soft-TF-L[1-12]</td>
<td>Prompt ID</td>
<td>72.28</td>
<td>3.44</td>
<td>96.83</td>
<td>0.44</td>
</tr>
<tr>
<td>CLIP-Prompt-Soft-TF-L[1-12] (SOTA)</td>
<td>Gradient ID</td>
<td>85.90</td>
<td>3.07</td>
<td>96.83</td>
<td>0.44</td>
</tr>
</tbody>
</table>

## 5.3 ABLATION STUDIES

Layer-wise Inspections. We analyze the layer-wise performance of Soft-Transformer with respect to L2P and DualPrompt on the 10-Split-CIFAR100 dataset to identify the optimal configurations, as shown in Figure 4. Our observations reveal that the global prompt in DualPrompt influences Soft-Transformer's performance differently in L2P and DualPrompt settings. In L2P-PGP, the best performance was achieved with Soft-TransFormers applied to the lower layers ((a) L2P-Soft-TF-L[1,2]-PGP), whereas in DualPrompt, the higher layers ((b) DualPrompt-Soft-TF-L[10,11,12]) yielded the best results. Notably, DualPrompt-Soft-TF-L[10,11,12] without PGP demonstrated impressive performance, achieving almost zero forgetting (0.21). These findings suggest that our approach could significantly enhance the effectiveness of large-scale Transformer models in continual learning scenarios.

![](./images/1068413726366892118_5.jpg)
(a) L2P-PGP v.s. Soft-Transformer

![](./images/1068413726366892118_6.jpg)
(b) DualPrompt v.s. Soft-Transformer

Figure 4: Layer-wise(L[*]) Performances of Soft-TF on 10-Split-CIFAR100. Note that L[9,10,11] denotes Soft-TransFormer of 9, 10, 11 Layers.

## 6 CONCLUSION

Inspired by Well-initialized Lottery Ticket Hypothesis (WLTH) that provides suboptimal fine-tuning solutions, we proposed a novel fully fine-tuned continual learning (CL) method referred to as Soft-TransFormers (Soft-TF), which sequentially learns and selects an optimal soft-network or subnetwork for each task. In training, Soft-TF jointly learned the sparse layer's weights in CL to obtain task-adaptive soft(real-valued)-networks or subnetworks (binary masks) while freezing the well-pre-trained layer parameters. In inference, the identified task-adaptive network of Soft-TF, which masks the parameters of the pre-trained network, maps to an optimal solution associated with each task, minimizing Catastrophic Forgetting (CF)—the soft masking was immune to the pre-trained network's knowledge forgetting. Extensive experiments demonstrated the power of Soft-TF (Vision Transformer and CLIP) and show state-of-the-art performances with convergence theory in various CL scenarios, i.e., Class-Incremental Learning (CIL) and Task-Incremental Learning (TIL).


## REFERENCES

Zoheb Abai and Nishad Rajmalwar. Densenet models for tiny imagenet classification. *arXiv preprint arXiv:1904.10429*, 2019. 8

Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia. Online continual learning with maximal interfered retrieval. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. 3

Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. *Advances in neural information processing systems*, 33:15920–15930, 2020. 3, 8

Lucas Caccia, Rahaf Aljundi, Nader Asadi, Tinne Tuytelaars, Joelle Pineau, and Eugene Belilovsky. New insights on reducing abrupt representation change in online continual learning. *arXiv preprint arXiv:2104.05025*, 2021. 3

Tenghao Cai, Zhizhong Zhang, Xin Tan, Yanyun Qu, Guannan Jiang, Chengjie Wang, and Yuan Xie. Multi-centroid task descriptor for dynamic class incremental inference. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 7298–7307, 2023. 8

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 9650–9660, 2021. 9

Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2019a. 1, 3

Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and M Ranzato. Continual learning with tiny episodic memories. *arXiv preprint arXiv:1902.10486*, 2019b. 1, 3

Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc'Aurelio Ranzato. On tiny episodic memories in continual learning. *arXiv preprint arXiv:1902.10486*, 2019c. 3

Arslan Chaudhry, Naeemullah Khan, Puneet K Dokania, and Philip HS Torr. Continual learning in low-rank orthogonal subspaces. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. 1, 3

Xiangning Chen, Cho-Jui Hsieh, and Boqing Gong. When vision transformers outperform resnets without pre-training or strong data augmentations. *arXiv preprint arXiv:2106.01548*, 2021. 9

Danruo Deng, Guangyong Chen, Jianye Hao, Qiong Wang, and Pheng-Ann Heng. Flattening sharpness for dynamic gradient projection memory benefits continual learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. 3

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. 1, 4, 7, 17

Arthur Douillard, Alexandre Ramé, Guillaume Couairon, and Matthieu Cord. Dytox: Transformers for continual learning with dynamic token expansion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9285–9295, 2022. 2, 4, 5

Demis Hassabis, Dharshan Kumaran, Christopher Summerfield, and Matthew Botvinick. Neuroscience-inspired artificial intelligence. *Neuron*, 95(2):245–258, 2017. 1

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In *Proceedings of the IEEE international conference on computer vision*, pp. 1026–1034, 2015. 4

11

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 8340–8349, 2021. 8

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In *International conference on machine learning*, pp. 2790–2799. PMLR, 2019. 19

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. 19

Sangwon Jung, Hongjoon Ahn, Sungmin Cha, and Taesup Moon. Continual learning with node-importance based adaptive group sparse regularization. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. 1, 3

Haeyong Kang, Rusty John Lloyd Mina, Sultan Rizky Hikmawan Madjid, Jaehong Yoon, Mark Hasegawa-Johnson, Sung Ju Hwang, and Chang D Yoo. Forget-free continual learning with winning subnetworks. In *International Conference on Machine Learning*, pp. 10734–10750. PMLR, 2022. 1, 3, 4, 8

Haeyong Kang, Jaehong Yoon, Sultan Rizky Hikmawan Madjid, Sung Ju Hwang, and Chang D. Yoo. On the soft-subnetwork for few-shot class incremental learning. In *The Eleventh International Conference on Learning Representations*, 2023. URL https://openreview.net/forum?id=z57WK51GeHd. 1, 3, 4, 8

Haeyong Kang, Jaehong Yoon, Sung Ju Hwang, and Chang D. Yoo. Continual learning: Forget-free winning subnetworks for video representations, 2024a. URL https://arxiv.org/abs/2312.11973. 3

Haeyong Kang, Jaehong Yoon, DaHyun Kim, Sung Ju Hwang, and Chang D. Yoo. Progressive fourier neural representation for sequential video compilation. In *The Twelfth International Conference on Learning Representations*, 2024b. URL https://openreview.net/forum?id=rGFrRMBbOq. 3

Muhammad Gul Zain Ali Khan, Muhammad Ferjad Naeem, Luc Van Gool, Didier Stricker, Federico Tombari, and Muhammad Zeshan Afzal. Introducing language guidance in prompt-based continual learning. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 11463–11473, 2023. 2, 4, 5

Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. *ACM computing surveys (CSUR)*, 54(10s):1–41, 2022. 4

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. 2017a. 1, 3

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, 114(13):3521–3526, 2017b. 8

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 8, 9

Abhishek Kumar and Hal Daume III. Learning task grouping and overlap in multi-task learning. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2012. 3

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. *arXiv preprint arXiv:2101.00190*, 2021. 4

12

Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and Caiming Xiong. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2019. 1, 3

Zhizhong Li and Derek Hoiem. Learning without forgetting. In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2016. 3

Zhizhong Li and Derek Hoiem. Learning without forgetting. *IEEE transactions on pattern analysis and machine intelligence*, 40(12):2935–2947, 2017. 8

Yan-Shuo Liang and Wu-Jun Li. Loss decoupling for task-agnostic continual learning. *Advances in Neural Information Processing Systems*, 36, 2024. 3

Huiwei Lin, Baoquan Zhang, Shanshan Feng, Xutao Li, and Yunming Ye. Pcr: Proxy-based contrastive replay for online class-incremental continual learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 24246–24255, 2023. 3

Zheda Mai, Ruiwen Li, Hyunwoo Kim, and Scott Sanner. Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 3589–3599, 2021. 3

Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2018. 1, 3

Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In *Psychology of learning and motivation*, volume 24, pp. 109–165. Elsevier, 1989. 1, 3

Seyed Iman Mirzadeh, Mehrdad Farajtabar, Dilan Gorur, Razvan Pascanu, and Hassan Ghasemzadeh. Linear mode connectivity in multitask and continual learning. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021. 1, 3

Yixuan Pei, Zhiwu Qing, Shiwei Zhang, Xiang Wang, Yingya Zhang, Deli Zhao, and Xueming Qian. Space-time prompting for video class-incremental learning. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 11932–11942, October 2023. 2, 4

Jingyang Qiao, zhizhong zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yong Peng, and Yuan Xie. Prompt gradient projection for continual learning. In *The Twelfth International Conference on Learning Representations*, 2024. URL https://openreview.net/forum?id=EH2O3h7sBI. 2, 3, 4, 5, 8, 17, 18

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PMLR, 2021. 1, 4, 8

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, pp. 2001–2010, 2017. 1, 3, 8

Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn without forgetting by maximizing transfer and minimizing interference. *arXiv preprint arXiv:1810.11910*, 2018. 1

Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. *arXiv preprint arXiv:1606.04671*, 2016. 1, 3

Gobinda Saha, Isha Garg, and Kaushik Roy. Gradient projection memory for continual learning. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021. 1, 3


Fahad Sarfraz, Elahe Arani, and Bahram Zonooz. Error sensitivity modulation based experience replay: Mitigating abrupt representation drift in continual learning. In *The Eleventh International Conference on Learning Representations*, 2023. URL https://openreview.net/forum?id=z1bci7019Z3. 3

Joan Serrà, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2018. 1, 3

Shai Shalev-Shwartz and Shai Ben-David. *Understanding machine learning: From theory to algorithms*. Cambridge university press, 2014. 15

Hanul Shin, Jung Kwon Lee, Jaehon Kim, and Jiwon Kim. Continual learning with deep generative replay. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. 3

Pravendra Singh, Vinay Kumar Verma, Pratik Mazumder, Lawrence Carin, and Piyush Rai. Cal- ibrating cnns for lifelong learning. *Advances in Neural Information Processing Systems*, 33: 15579–15590, 2020. 3

James Seale Smith, Paola Cascante-Bonilla, Assaf Arbelle, Donghyun Kim, Rameswar Panda, David Cox, Diyi Yang, Zsolt Kira, Rogerio Feris, and Leonid Karlinsky. Construct-vl: Data-free continual structured vl concepts learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 14994–15004, June 2023a. 2, 4

James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 11909–11919, June 2023b. 2, 4, 5

Wenju Sun, Qingyong Li, Jing Zhang, Wen Wang, and Yangli-ao Geng. Decoupling learning and remembering: A bilevel memory framework with knowledge projection for task-incremental learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 20186–20195, 2023. 3

Sebastian Thrun. *A Lifelong Learning Perspective for Mobile Robot Control*. Elsevier, 1995. 1, 3

Michalis K Titsias, Jonathan Schwarz, Alexander G de G Matthews, Razvan Pascanu, and Yee Whye Teh. Functional regularisation for continual learning with gaussian processes. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2020. 1, 3

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 9

Yabin Wang, Zhiwu Huang, and Xiaopeng Hong. S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning. *Advances in Neural Information Processing Systems*, 35:5682–5695, 2022a. 2, 4

Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In *European Conference on Computer Vision*, pp. 631–648. Springer, 2022b. 1, 3, 4, 5, 6, 7, 8, 17

Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 139–149, 2022c. 1, 3, 4, 8, 17

Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu, Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosinski, and Ali Farhadi. Supermasks in superposition. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. 1, 3, 4, 8

Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu. Large scale incremental learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 374–382, 2019. 8

Ju Xu and Zhanxing Zhu. Reinforced continual learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2018. 3

Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 3014–3023, 2021. 3

Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically expandable networks. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2018. 3

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In *International Conference on Machine Learning*, pp. 3987–3995. PMLR, 2017. 1

## A APPENDIX

### A.1 ANALYSIS OF SOFT-TRANSFORMERS (SOFT-TF)

Analysis of Soft-TransFormers for Convex-Lipschitz Functions. To analyze the convergence rate of the Soft-TransFormers (Soft-TF), we limit ourselves to the case of convex-Lipshitz functions along with the analysis (Shalev-Shwartz & Ben-David, 2014). Let $\boldsymbol{w}^* = \{\boldsymbol{g}^*, \boldsymbol{e}_t^*, \boldsymbol{m}_t^*\}$ be any vector or an optimal solution and let $B$ be an upper bound on $\|\boldsymbol{w}^*\|$ when $\boldsymbol{w}^{(1)} = \boldsymbol{0}$. It is convenient to think of $\boldsymbol{w}^*$ as the minimizer of $f(\boldsymbol{w})$, but the analysis that follows holds for every $\boldsymbol{w}^*$.

We would like to obtain an upper bound on the sub-optimality of our solution with respect to $\boldsymbol{w}^*$, namely, $f(\bar{\boldsymbol{w}}) - f(\boldsymbol{w}^*)$, where $\bar{\boldsymbol{w}} = \frac{1}{T} \boldsymbol{w}^{(t)}$. From the definition of $\bar{\boldsymbol{w}}$, and using Jensen's inequality, we have that

$$
\begin{aligned}
f(\bar{\boldsymbol{w}})-f\left(\boldsymbol{w}^{*}\right) &=f\left(\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{w}^{(t)}\right)-f\left(\boldsymbol{w}^{*}\right) \\
& \leq \frac{1}{T} \sum_{t=1}^{T}\left(f\left(\boldsymbol{w}^{(t)}\right)\right)-f\left(\boldsymbol{w}^{*}\right) \\
&=\frac{1}{T} \sum_{t=1}^{T}\left(f\left(\boldsymbol{w}^{(t)}\right)-f\left(\boldsymbol{w}^{*}\right)\right).
\end{aligned}
\tag{14}
$$

For every $t$, because of the convexity of $f$, we have that

$$
f\left(\boldsymbol{w}^{(t)}\right)-f\left(\boldsymbol{w}^{*}\right) \leq\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \nabla f\left(\boldsymbol{w}^{(t)}\right)\right\rangle
\tag{15}
$$

Combining the preceeding we obtain

$$
f\left(\boldsymbol{w}^{(t)}\right)-f\left(\boldsymbol{w}^{*}\right) \leq \frac{1}{T} \sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \nabla f\left(\boldsymbol{w}^{(t)}\right)\right\rangle
\tag{16}
$$

To bound the right-hand side we rely on the following lemma:

**Lemma A.1.** Let $\boldsymbol{v}_1, \cdots, \boldsymbol{v}_T$ be an arbitrary sequence of vectors. Any algorithm with an well initialization (pre-trained model) $\boldsymbol{w}^{(1)} \neq \boldsymbol{0}$ and an update rule of the form

$$
\boldsymbol{w}^{(t+1)}=\boldsymbol{w}^{(t)}-\eta \boldsymbol{v}_{t}
\tag{17}
$$

satisfies with $\|\boldsymbol{w}^{(1)} - \boldsymbol{w}^*\|^2 = \|\boldsymbol{w}^{(T+1)} - \boldsymbol{w}^*\|^2$

$$
\begin{aligned}
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle & \leq \frac{1}{2 \eta}\left\|\boldsymbol{w}_{m}^{(T+1)}-\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2} \\
& <\frac{1}{2 \eta}\left\|\boldsymbol{w}_{p}^{(T+1)}-\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2} \\
& <\frac{1}{2 \eta}\left\|\boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2}
\end{aligned}
$$

where $\boldsymbol{w}_{m} \neq \boldsymbol{w}_{q}$ since $\boldsymbol{m}$ is learnable parameters in Soft-Transformers. Specifically, we could assume that $\boldsymbol{w}_{m}=\left(\boldsymbol{w}^{Q} \odot \boldsymbol{m}^{Q}\right) \cdot \boldsymbol{x} \boldsymbol{p}^{T} \cdot\left(\boldsymbol{w}^{K} \odot \boldsymbol{m}^{K}\right)^{T}$ and $\boldsymbol{w}_{p}=\left(\boldsymbol{w}^{Q} \odot \mathbf{1}^{Q}\right) \cdot \boldsymbol{x} \boldsymbol{p}^{T} \cdot\left(\boldsymbol{w}^{K} \odot \mathbf{1}^{K}\right)^{T}$ of Equation 6.

Theorem A.2. For every $B_{m}<B_{p}<B, \rho>0$ where $B_{m}=\left\|\boldsymbol{w}_{m}^{(T+1)}-\boldsymbol{w}^{*}\right\|$ and $B_{p}=$ $\left\|\boldsymbol{w}_{p}^{(T+1)}-\boldsymbol{w}^{*}\right\|$, if for all $t$ we have that $\left\|\boldsymbol{v}_{t} \leq \rho\right\|$ and if we set $\eta=\sqrt{\frac{B^{2}}{\rho^{2} T}}$, then for every $\boldsymbol{w}^{*}$ with $\left\|\boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}\right\| \leq B$ we have

$$
\frac{1}{T} \sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle \leq \frac{B_{m} \rho}{\sqrt{T}}<\frac{B_{p} \rho}{\sqrt{T}}<\frac{B \rho}{\sqrt{T}} .
$$

Proof. Using algebraic manipulations (completing the square), we obtain:

$$
\begin{aligned}
\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle & =\frac{1}{\eta}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \eta \boldsymbol{v}_{t}\right\rangle \\
& =\frac{1}{2 \eta}\left(-\left\|\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}-\eta \boldsymbol{v}_{t}\right\|^{2}+\left\|\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}\right\|^{2}+\eta^{2}\left\|\boldsymbol{v}_{t}\right\|^{2}\right) \\
& =\frac{1}{2 \eta}\left(-\left\|\boldsymbol{w}^{(t+1)}-\eta \boldsymbol{v}_{t}\right\|^{2}+\left\|\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}\right\|^{2}\right)+\frac{\eta}{2}\left\|\boldsymbol{v}_{t}\right\|^{2},
\end{aligned}
$$

where the last equality follows from the definition of the update rule. Summing the equality over $t$, we have

$$
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle=\frac{1}{2 \eta} \sum_{t=1}^{T}\left(-\left\|\boldsymbol{w}^{(t+1)}-\eta \boldsymbol{v}_{t}\right\|^{2}+\left\|\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}\right\|^{2}\right)+\frac{\eta}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2}
$$

The first sum on the right-hand side is a telescopic sum that collapses to

$$
\left\|\boldsymbol{w}^{(1)}-\boldsymbol{w}^{*}\right\|^{2}=\left\|\boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}\right\|^{2}
$$

Plugging this in Equation, we have

$$
\begin{aligned}
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle & =\frac{1}{2 \eta} \sum_{t=1}^{T}\left(-\left\|\boldsymbol{w}^{(t+1)}-\eta \boldsymbol{v}_{t}\right\|^{2}+\left\|\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}\right\|^{2}\right)+\frac{\eta}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2} \\
& \leq \frac{1}{2 \eta}\left\|\boldsymbol{w}^{(1)}-\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta}{2} \sum_{t=1}^{t}\left\|\boldsymbol{v}_{t}\right\|^{2} \\
& =\frac{1}{2 \eta}\left\|\boldsymbol{w}^{*}\right\|^{2}+\frac{\eta}{2} \sum_{t=1}^{T}\left\|\boldsymbol{v}_{t}\right\|^{2},
\end{aligned}
$$

where the last equality is due to the definition $\boldsymbol{w}^{(1)}=\mathbf{0}$. This proves the first part of the lemma. The second part follows by upper bounding $\|\boldsymbol{w}\|$ by $B,\left\|\boldsymbol{v}_{t}\right\|$ by $\rho$, deciding by $T$, and plugging in the value of $\eta$.

In terms of Soft-Transformers $\boldsymbol{w}_{m}=\left(\boldsymbol{w}^{Q} \odot \boldsymbol{m}^{Q}\right) \cdot \boldsymbol{x} \boldsymbol{p}^{T} \cdot\left(\boldsymbol{w}^{K} \odot \boldsymbol{m}^{K}\right)^{T}$ of Equation 6, we have


$$
\begin{aligned}
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}^{s}\right\rangle &=\frac{1}{2 \eta} \sum_{t=1}^{T}\left(-|| \boldsymbol{w}^{(t+1)}-\eta \boldsymbol{w}_{t}||^{2}+|| \boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}||^{2}\right)+\frac{\eta}{2} \sum_{t=1}^{T}|| \boldsymbol{v}_{t}||^{2} \\
& \leq \frac{1}{2 \eta}|| \boldsymbol{w}^{(1)}-\boldsymbol{w}^{*}||^{2}+\frac{\eta}{2} \sum_{t=1}^{t}|| \boldsymbol{v}_{t}||^{2}
\end{aligned}
\tag{24}
$$

where $\boldsymbol{v}_{t}$ is an arbitrary $t$-th vector and $\boldsymbol{w}^{(1)} \neq \mathbf{0}$ since $\boldsymbol{w}^{Q}$ and $\boldsymbol{q}^{K}$ are pre-trained parameters.

however, in term of prompt $\boldsymbol{w}_{p}=\left(\boldsymbol{w}^{Q} \odot \mathbf{1}^{Q}\right) \cdot \boldsymbol{x} \boldsymbol{p}^{T} \cdot\left(\boldsymbol{w}^{K} \odot \mathbf{1}^{K}\right)^{T}$, we have

$$
\begin{aligned}
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}^{p}\right\rangle &=\frac{1}{2 \eta} \sum_{t=1}^{T}\left(-|| \boldsymbol{w}^{(t+1)}-\eta \boldsymbol{w}_{t}||^{2}+|| \boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}||^{2}\right)+\frac{\eta}{2} \sum_{t=1}^{T}|| \boldsymbol{v}_{t}||^{2} \\
& \leq \frac{1}{2 \eta}|| \boldsymbol{w}^{(1)}-\boldsymbol{w}^{*}||^{2}+\frac{\eta}{2} \sum_{t=1}^{t}|| \boldsymbol{v}_{t}||^{2}
\end{aligned}
\tag{25}
$$

where $\boldsymbol{v}_{t}$ is an arbitrary $t$-th vector of prompt and $\boldsymbol{w}^{(1)} \neq \mathbf{0}$ since $(\boldsymbol{w} \odot \mathbf{1})^{Q}$ and $(\boldsymbol{w} \odot \mathbf{1})^{K}$ are pre-trained parameters, $\boldsymbol{w}^{Q}$ and $\boldsymbol{w}^{K}$, respectively.

Therefore, we have from $|| \boldsymbol{w}^{(1)}-\boldsymbol{w}^{*}||^{2}=|| \boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}||^{2}$

$$
\begin{aligned}
\sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}^{p}\right\rangle & \leq \frac{1}{2 \eta_{m}}|| \boldsymbol{w}_{m}^{(T+1)}-\boldsymbol{w}^{*}||^{2}+\frac{\eta_{m}}{2} \sum_{t=1}^{t}|| \boldsymbol{v}_{t}||^{2} \\
&<\frac{1}{2 \eta_{p}}|| \boldsymbol{w}_{p}^{(T+1)}-\boldsymbol{w}^{*}||^{2}+\frac{\eta_{p}}{2} \sum_{t=1}^{t}|| \boldsymbol{v}_{t}||^{2} \\
&<\frac{1}{2 \eta}|| \boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}||^{2}+\frac{\eta}{2} \sum_{t=1}^{T}|| \boldsymbol{v}_{t}||^{2}
\end{aligned}
\tag{26}
$$

where $|| \boldsymbol{w}_{m}^{(1)}-\boldsymbol{w}^{*}||^{2}<|| \boldsymbol{w}_{p}^{(1)}-\boldsymbol{w}^{*}||^{2}$ since all $\boldsymbol{m}$ are learnable parameters. For every $B_{m}<$ $B_{p}<B, \rho>0$ where $B_{m}=|| \boldsymbol{w}_{m}^{(T+1)}-\boldsymbol{w}^{*}||$ and $B_{p}=|| \boldsymbol{w}_{p}^{(T+1)}-\boldsymbol{w}^{*}||$, if for all $t$ we have that $|| \boldsymbol{v}_{t} \leq \rho||$ and if we set $\eta \approx \eta_{m} \approx \eta_{p}=\sqrt{\frac{B^{2}}{\rho^{2} T}}$ with large enough $T$, then for every $\boldsymbol{w}^{*}$ with $|| \boldsymbol{w}^{(T+1)}-\boldsymbol{w}^{*}|| \leq B$ we have

$$
\frac{1}{T} \sum_{t=1}^{T}\left\langle\boldsymbol{w}^{(t)}-\boldsymbol{w}^{*}, \boldsymbol{v}_{t}\right\rangle \leq \frac{B_{m} \rho}{\sqrt{T}}<\frac{B_{p} \rho}{\sqrt{T}}<\frac{B \rho}{\sqrt{T}}.
\tag{27}
$$

$\square$

## A.2 EXPERIMENTAL DETAILS

For fair comparisons with the baselines (Wang et al., 2022c,b; Qiao et al., 2024), we use ViT B/16 (Dosovitskiy et al., 2020) pre-trained on ImageNet-21K as our image encoder, which is kept frozen during training. We train and test on a single Quadro RTX 8000-48GB GPU for baselines and our Soft-Transformers with Adam optimizer with $\beta_{1}=0.9$ and $\beta_{2}=0.999$.

We adhere to the experimental settings outlined by Qiao et al. (2024) to validate our method's effectiveness. When comparing our approach with L2P-PGP and Soft-Transformer on the 10/20-Split-CIFAR100 and 10-Split-TinyImageNet datasets, we train the network for 5 epochs with a batch size of 16 and set the prompt length to 5. For the 10-Split-ImageNet-R dataset, we use 50 epochs, a batch size of 16, and a prompt length of 30. In comparison with DualPrompt-PGP and Soft-Transformers on the 10/20-Split-CIFAR100 dataset, we train the network for 20 epochs with a batch size of 24 and set the expert prompt length to 5. For the 10-Split-TinyImageNet dataset, we use 5 epochs, a batch

size of 24, and an expert prompt length of 5. For the 10-Split-ImageNet-R dataset, we set the epochs to 50, the batch size to 24, and the expert prompt length to 20. Additionally, in all benchmark data sets, the general prompt length is set to 5, and the location inserted into the prompt is kept consistent.

For CLIP-PGP and Soft-TransFormers, we configure a single trainable image prompt that is shared across all tasks within the vision encoder. For the text encoder, following the approach of Qiao et al. (2024), we set a trainable text prompt for each class, which is only trained on the corresponding task. In our comparisons with CLIP-PGP and Soft-TransFormers on the 10-Split-CIFAR100 dataset, we set the image prompt length to 5, the number of epochs to 5, and the batch size to 32.

Table 4: Performances of Class Incremental Learning (CIL) in terms of accuracy and forgetting on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Exemplar means the total buffer size for rehearsal methods.

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Exemplar</th>
<th rowspan="2">Task ID</th>
<th colspan="2">10-Split-CIFAR100</th>
<th colspan="2">20-Split-CIFAR100</th>
<th colspan="2">10-Split-ImageNet-R</th>
</tr>
<tr>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>BiC</td>
<td>5,000</td>
<td>-</td>
<td>81.42</td>
<td>17.31</td>
<td>73.02</td>
<td>6.23</td>
<td>64.63</td>
<td>22.25</td>
</tr>
<tr>
<td>DER++</td>
<td>5,000</td>
<td>-</td>
<td>83.94</td>
<td>14.55</td>
<td>-</td>
<td>-</td>
<td>66.73</td>
<td>20.67</td>
</tr>
<tr>
<td>iCaRL</td>
<td>5,000</td>
<td>-</td>
<td>66.00</td>
<td>5.33</td>
<td>78.02</td>
<td>5.80</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>DER+MCG</td>
<td>2,000</td>
<td>-</td>
<td>67.62</td>
<td>14.64</td>
<td>65.84</td>
<td>13.72</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BiC</td>
<td>1,000</td>
<td>-</td>
<td>66.11</td>
<td>35.24</td>
<td>63.12</td>
<td>21.89</td>
<td>52.14</td>
<td>36.70</td>
</tr>
<tr>
<td>DER++</td>
<td>1,000</td>
<td>-</td>
<td>61.06</td>
<td>39.87</td>
<td>-</td>
<td>-</td>
<td>55.47</td>
<td>34.64</td>
</tr>
<tr>
<td>iCaRL</td>
<td>1,000</td>
<td>-</td>
<td>61.25</td>
<td>14.19</td>
<td>71.32</td>
<td>15.98</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>FT</td>
<td>-</td>
<td>-</td>
<td>33.61</td>
<td>86.87</td>
<td>33.52</td>
<td>53.69</td>
<td>28.87</td>
<td>63.80</td>
</tr>
<tr>
<td>EWC</td>
<td>-</td>
<td>-</td>
<td>47.01</td>
<td>33.27</td>
<td>36.73</td>
<td>35.19</td>
<td>35.00</td>
<td>56.16</td>
</tr>
<tr>
<td>LWF</td>
<td>-</td>
<td>-</td>
<td>60.69</td>
<td>27.77</td>
<td>39.12</td>
<td>57.91</td>
<td>38.54</td>
<td>52.37</td>
</tr>
<tr>
<td>L2P*</td>
<td>-</td>
<td>Prompt ID</td>
<td>83.77</td>
<td>6.63</td>
<td>71.29</td>
<td>13.96</td>
<td>60.44</td>
<td>9.00</td>
</tr>
<tr>
<td>L2P-PGP*</td>
<td>-</td>
<td>Prompt ID</td>
<td>84.34</td>
<td>5.59</td>
<td>76.12</td>
<td>13.26</td>
<td>61.70</td>
<td>8.03</td>
</tr>
<tr>
<td>L2P-PGP-Soft-TF</td>
<td>-</td>
<td>Prompt ID</td>
<td>86.26</td>
<td>4.79</td>
<td>76.17</td>
<td>15.77</td>
<td>69.80</td>
<td>5.13</td>
</tr>
<tr>
<td>L2P-PGP-Soft-TF</td>
<td>-</td>
<td>Gradient ID</td>
<td>86.46</td>
<td>4.87</td>
<td>77.67</td>
<td>15.84</td>
<td>69.56</td>
<td>5.28</td>
</tr>
<tr>
<td>DualPrompt</td>
<td>-</td>
<td>Prompt ID</td>
<td>86.50</td>
<td>5.77</td>
<td>82.98</td>
<td>8.20</td>
<td>68.13</td>
<td>4.46</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[3,4,5]</td>
<td>-</td>
<td>Prompt ID</td>
<td>91.77</td>
<td>3.37</td>
<td>94.43</td>
<td>2.02</td>
<td>74.70</td>
<td>6.46</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[3,4,5]</td>
<td>-</td>
<td>Gradient ID</td>
<td>93.76</td>
<td>1.83</td>
<td>95.38</td>
<td>1.73</td>
<td>82.15</td>
<td>2.20</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12], c=80.0%</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.41</td>
<td>0.18</td>
<td>90.25</td>
<td>9.08</td>
<td>74.83</td>
<td>0.91</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12], c=81.0%</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.50</td>
<td>0.21</td>
<td>96.72</td>
<td>2.21</td>
<td>74.21</td>
<td>1.41</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12], c=82.0%</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.67</td>
<td>0.27</td>
<td>96.44</td>
<td>1.62</td>
<td>75.02</td>
<td>0.92</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12], c=83.0%</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.62</td>
<td>0.25</td>
<td>97.77</td>
<td>0.63</td>
<td>77.36</td>
<td>1.77</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12], c=87.0%</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.51</td>
<td>0.27</td>
<td>97.68</td>
<td>0.75</td>
<td>76.93</td>
<td>1.02</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12], c=90.0%</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.46</td>
<td>0.38</td>
<td>98.09</td>
<td>0.65</td>
<td>78.80</td>
<td>0.47</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12]</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.87</td>
<td>0.21</td>
<td>99.05</td>
<td>0.24</td>
<td>82.38</td>
<td>0.59</td>
</tr>
<tr>
<td>DualPrompt-PGP</td>
<td>-</td>
<td>Prompt ID</td>
<td>86.92</td>
<td>5.35</td>
<td>83.74</td>
<td>7.91</td>
<td>69.34</td>
<td>4.53</td>
</tr>
<tr>
<td>DualPrompt-PGP-Soft-TF-L[3,4,5]</td>
<td>-</td>
<td>Prompt ID</td>
<td>92.41</td>
<td>2.44</td>
<td>95.14</td>
<td>1.90</td>
<td>74.65</td>
<td>4.39</td>
</tr>
<tr>
<td>DualPrompt-PGP-Soft-TF-L[3,4,5]</td>
<td>-</td>
<td>Gradient ID</td>
<td>92.92</td>
<td>2.34</td>
<td>95.89</td>
<td>1.64</td>
<td>81.45</td>
<td>2.89</td>
</tr>
<tr>
<td>Upper-Bound of DualPrompt</td>
<td>-</td>
<td>-</td>
<td>90.85</td>
<td>-</td>
<td>90.85</td>
<td>-</td>
<td>79.13</td>
<td>-</td>
</tr>
<tr>
<td>Upper-Bound of Soft-TF</td>
<td>-</td>
<td>-</td>
<td>93.90</td>
<td>-</td>
<td>93.90</td>
<td>-</td>
<td>80.21</td>
<td>-</td>
</tr>
</tbody>
</table>

Table 5: Random initialized Performances of Class Incremental Learning (CIL) in terms of accuracy and forgetting on 10-Split-CIFAR100. Note "w/o FF" denotes "Soft fine-tuning without FeedForward (FF)" networks.

<table>
<thead>
<tr>
<th>Method</th>
<th>Pretrained-Dataset</th>
<th>Task ID</th>
<th>Random Initialization</th>
<th colspan="2">10-Split-CIFAR100</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12] w/o FF</td>
<td>ImageNet-21K</td>
<td>Prompt ID</td>
<td>Xavier</td>
<td>90.59</td>
<td>3.85</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12] w/o FF</td>
<td>ImageNet-21K</td>
<td>Prompt ID</td>
<td>Kaiming</td>
<td>90.72</td>
<td>3.63</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12] w/o FF</td>
<td>ImageNet-21K</td>
<td>Prompt ID</td>
<td>Normal</td>
<td>90.45</td>
<td>3.78</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12] w/o FF</td>
<td>ImageNet-21K</td>
<td>Prompt ID</td>
<td>Uniform(1.0, 1.0)</td>
<td>92.35</td>
<td>2.98</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12] w/o FF</td>
<td>ImageNet-21K</td>
<td>Gradient ID</td>
<td>Uniform(1.0, 1.0)</td>
<td>98.05</td>
<td>0.25</td>
</tr>
<tr>
<td colspan="4">Upper-Bound of Soft-TF</td>
<td>93.90</td>
<td>-</td>
</tr>
</tbody>
</table>

Random initialization. Random initialization of Soft-Transformer's weights plays a critical role when leveraging well-pretrained models like Vision Transformers (ViTs). The optimal training point is the parameters of a well-pretrained model. Among the initialization methods, Uniform initialization for Soft-TransFormer satisfies this requirement effectively. To validate these claims, we analyze the impact of common random initialization methods, including Xavier, Kaiming, Normal, and Uniform Initialization, as shown in Table 5. The results demonstrate that the same well-initialization point leads to independent optimal task performance, particularly with Gradient ID inference. Furthermore, this ablation study strengthens our Soft-TF with state-of-the-art-perfomances inspired by the Well-initialized Lottery Ticket Hypothesis (WLTH).

Training & Test Time. To clearly illustrate the time complexity of Soft-TF, we present the training and testing times for 10/20-Split-CIFAR100 and 10-Split-ImageNet-R, as shown in Table 6. As the number of trainable parameters in Soft-TF increases, training and testing time complexities grow

Table 6: Performances of Class Incremental Learning (CIL) in terms of Soft parameters, training, and test time on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Note "w/o FF" denotes "Soft finetuning without FeedForward (FF)" networks.

<table>
 <thead>
  <tr>
   <th>Method</th>
   <th>ViT-B/12 (85.8M)</th>
   <th rowspan="2">Task ID</th>
   <th colspan="2">10-Split-CIFAR100</th>
   <th colspan="2">20-Split-CIFAR100</th>
   <th colspan="2">10-Split-ImageNet-R</th>
  </tr>
  <tr>
   <th>DualPrompt</th>
   <th># Train Params.</th>
   <th>Train (sec.)</th>
   <th>Test (sec.)</th>
   <th>Train (sec.)</th>
   <th>Test (sec.)</th>
   <th>Train (sec.)</th>
   <th>Test (sec.)</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>DualPrompt</td>
   <td>0.00M</td>
   <td>Prompt ID</td>
   <td>12.12K</td>
   <td>76</td>
   <td>11.60K</td>
   <td>78</td>
   <td>13.10K</td>
   <td>47</td>
  </tr>
  <tr>
   <td>PGP</td>
   <td>0.00M</td>
   <td>Prompt ID</td>
   <td>12.21K</td>
   <td>76</td>
   <td>13.12K</td>
   <td>78</td>
   <td>13.33K</td>
   <td>47</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/ only ATTN</td>
   <td>1.76M</td>
   <td>Gradient ID</td>
   <td>12.18K</td>
   <td>129</td>
   <td>13.30K</td>
   <td>113</td>
   <td>13.35K</td>
   <td>65</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/ only ATTN</td>
   <td>1.76M</td>
   <td>Prompt ID</td>
   <td>12.18K</td>
   <td>78</td>
   <td>13.30K</td>
   <td>80</td>
   <td>13.35K</td>
   <td>48</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Gradient ID</td>
   <td>12.24K</td>
   <td>103</td>
   <td>13.40K</td>
   <td>132</td>
   <td>13.42K</td>
   <td>66</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>4.62M</td>
   <td>Gradient ID</td>
   <td>12.95K</td>
   <td>115</td>
   <td>14.38K</td>
   <td>146</td>
   <td>14.23K</td>
   <td>73</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Gradient ID</td>
   <td>13.71K</td>
   <td>130</td>
   <td>15.51K</td>
   <td>163</td>
   <td>15.08K</td>
   <td>82</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Prompt ID</td>
   <td>13.87K</td>
   <td>80</td>
   <td>15.60K</td>
   <td>104</td>
   <td>15.35K</td>
   <td>52</td>
  </tr>
  <tr>
   <td>LoRA-L[10,11,12] w/o FF, r=4</td>
   <td>0.06M</td>
   <td>Prompt ID</td>
   <td>11.95K</td>
   <td>77</td>
   <td>11.71K</td>
   <td>79</td>
   <td>13.10K</td>
   <td>48</td>
  </tr>
  <tr>
   <td>LoRA-L[10,11,12] w/o FF, r=24</td>
   <td>0.32M</td>
   <td>Prompt ID</td>
   <td>12.03K</td>
   <td>78</td>
   <td>15.10K</td>
   <td>100</td>
   <td>15.02K</td>
   <td>50</td>
  </tr>
  <tr>
   <td>LoRA-L[10,11,12] w/o FF, r=500</td>
   <td>6.91M</td>
   <td>Prompt ID</td>
   <td>13.24K</td>
   <td>79</td>
   <td>15.89K</td>
   <td>105</td>
   <td>15.09K</td>
   <td>53</td>
  </tr>
  <tr>
   <td>Adapter-L[10,11,12] w/o FF, r=1</td>
   <td>0.09M</td>
   <td>Prompt ID</td>
   <td>12.44K</td>
   <td>84</td>
   <td>12.40K</td>
   <td>81</td>
   <td>14.40K</td>
   <td>50</td>
  </tr>
  <tr>
   <td>Adapter-L[10,11,12] w/ FF, r=4</td>
   <td>0.36M</td>
   <td>Prompt ID</td>
   <td>12.80K</td>
   <td>85</td>
   <td>15.35K</td>
   <td>105</td>
   <td>14.68K</td>
   <td>51</td>
  </tr>
  <tr>
   <td>Adapter-L[10,11,12] w/ FF, r=75</td>
   <td>6.91M</td>
   <td>Prompt ID</td>
   <td>13.66K</td>
   <td>88</td>
   <td>15.72K</td>
   <td>106</td>
   <td>15.50K</td>
   <td>53</td>
  </tr>
 </tbody>
</table>

Table 7: Performances of Class Incremental Learning (CIL) in terms of Soft parameters, Accuracy, and Forget on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Note "w/o FF" denotes "Soft finetuning without FeedForward (FF)" networks.

<table>
 <thead>
  <tr>
   <th>Method</th>
   <th>ViT-B/12 (85.8M)</th>
   <th rowspan="2">Task ID</th>
   <th colspan="2">10-Split-CIFAR100</th>
   <th colspan="2">20-Split-CIFAR100</th>
   <th colspan="2">10-Split-ImageNet-R</th>
  </tr>
  <tr>
   <th>DualPrompt</th>
   <th># Train Params.</th>
   <th>ACC(↑)</th>
   <th>Forget(↓)</th>
   <th>ACC(↑)</th>
   <th>Forget(↓)</th>
   <th>ACC(↑)</th>
   <th>Forget(↓)</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>DualPrompt</td>
   <td>0.00M</td>
   <td>Prompt ID</td>
   <td>86.50</td>
   <td>5.77</td>
   <td>82.98</td>
   <td>8.20</td>
   <td>68.13</td>
   <td>4.46</td>
  </tr>
  <tr>
   <td>PGP</td>
   <td>0.00M</td>
   <td>Prompt ID</td>
   <td>86.92</td>
   <td>5.35</td>
   <td>83.74</td>
   <td>7.91</td>
   <td>69.34</td>
   <td>4.53</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/ only ATTN</td>
   <td>1.76M</td>
   <td>Gradient ID</td>
   <td>97.17</td>
   <td>0.40</td>
   <td>98.09</td>
   <td>0.54</td>
   <td>72.31</td>
   <td>3.94</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/ only ATTN</td>
   <td>1.76M</td>
   <td>Prompt ID</td>
   <td>94.59</td>
   <td>1.12</td>
   <td>96.96</td>
   <td>1.02</td>
   <td>71.13</td>
   <td>4.93</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Gradient ID</td>
   <td>96.84</td>
   <td>0.55</td>
   <td>97.81</td>
   <td>0.57</td>
   <td>81.18</td>
   <td>1.31</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>4.62M</td>
   <td>Gradient ID</td>
   <td>97.58</td>
   <td>0.34</td>
   <td>98.65</td>
   <td>0.43</td>
   <td>83.09</td>
   <td>0.42</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Gradient ID</td>
   <td>98.05</td>
   <td>0.25</td>
   <td>98.96</td>
   <td>0.23</td>
   <td>83.70</td>
   <td>0.53</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Prompt ID</td>
   <td>92.35</td>
   <td>2.98</td>
   <td>97.40</td>
   <td>0.57</td>
   <td>76.62</td>
   <td>5.30</td>
  </tr>
  <tr>
   <td>LoRA-L[10,11,12] w/o FF, r=4</td>
   <td>0.06M</td>
   <td>Prompt ID</td>
   <td>82.19</td>
   <td>4.33</td>
   <td>93.74</td>
   <td>2.07</td>
   <td>70.91</td>
   <td>9.11</td>
  </tr>
  <tr>
   <td>LoRA-L[10,11,12] w/o FF, r=24</td>
   <td>0.32M</td>
   <td>Prompt ID</td>
   <td>86.77</td>
   <td>4.27</td>
   <td>95.65</td>
   <td>1.04</td>
   <td>69.81</td>
   <td>10.30</td>
  </tr>
  <tr>
   <td>LoRA-L[10,11,12] w/o FF, r=500</td>
   <td>6.91M</td>
   <td>Prompt ID</td>
   <td>82.00</td>
   <td>4.33</td>
   <td>92.14</td>
   <td>2.02</td>
   <td>43.51</td>
   <td>13.21</td>
  </tr>
  <tr>
   <td>Adapter-L[10,11,12] w/o FF, r=1</td>
   <td>0.09M</td>
   <td>Prompt ID</td>
   <td>86.38</td>
   <td>4.87</td>
   <td>85.61</td>
   <td>5.04</td>
   <td>70.95</td>
   <td>4.31</td>
  </tr>
  <tr>
   <td>Adapter-L[10,11,12] w/ FF, r=4</td>
   <td>0.36M</td>
   <td>Prompt ID</td>
   <td>86.53</td>
   <td>4.52</td>
   <td>85.66</td>
   <td>5.00</td>
   <td>70.82</td>
   <td>4.90</td>
  </tr>
  <tr>
   <td>Adapter-L[10,11,12] w/ FF, r=75</td>
   <td>6.91M</td>
   <td>Prompt ID</td>
   <td>86.45</td>
   <td>4.61</td>
   <td>84.75</td>
   <td>5.11</td>
   <td>70.55</td>
   <td>4.74</td>
  </tr>
 </tbody>
</table>

accordingly. While the testing time complexity of Gradient ID increased by approximately 1.6 times across the three benchmark datasets, it consistently improved task performance on all benchmarks. The corresponding performance metrics are detailed in Table 7.

We investigate the most parameter-efficient and gradient-based task inference methods, as shown in Table 8 and Table 9. Our findings reveal that the 3-shot Gradient ID inference cost (using samples within a mini-batch) with the last layer (Soft-TF-L[12]) is approximately 1.1 times that of Prompt ID, maintaining comparable efficiency while delivering superior performance. Note that m-batch denotes mini-batch.

Table 8: Performances of Class Incremental Learning (CIL) in terms of Soft parameters, training, and test time on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Note "w/o FF" denotes "Soft finetuning without FeedForward (FF)" networks.

<table>
 <thead>
  <tr>
   <th>Method</th>
   <th>ViT-B/12 (85.8M)</th>
   <th rowspan="2">Task ID</th>
   <th colspan="2">10-Split-CIFAR100</th>
   <th colspan="2">20-Split-CIFAR100</th>
   <th colspan="2">10-Split-ImageNet-R</th>
  </tr>
  <tr>
   <th>DualPrompt</th>
   <th># Train Params.</th>
   <th>Train (sec.)</th>
   <th>Test (sec.)</th>
   <th>Train (sec.)</th>
   <th>Test (sec.)</th>
   <th>Train (sec.)</th>
   <th>Test (sec.)</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>DualPrompt</td>
   <td>0.00M</td>
   <td>Prompt ID</td>
   <td>12.12K</td>
   <td>76</td>
   <td>11.60K</td>
   <td>78</td>
   <td>13.10K</td>
   <td>47</td>
  </tr>
  <tr>
   <td>PGP</td>
   <td>0.00M</td>
   <td>Prompt ID</td>
   <td>12.21K</td>
   <td>76</td>
   <td>13.12K</td>
   <td>78</td>
   <td>13.33K</td>
   <td>47</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Prompt ID</td>
   <td>12.24K</td>
   <td>79</td>
   <td>13.40K</td>
   <td>80</td>
   <td>13.42K</td>
   <td>48</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Gradient ID, 3-shot</td>
   <td>12.24K</td>
   <td>88</td>
   <td>13.40K</td>
   <td>90</td>
   <td>13.42K</td>
   <td>57</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Gradient ID, 5-shot</td>
   <td>12.24K</td>
   <td>94</td>
   <td>13.40K</td>
   <td>98</td>
   <td>13.42K</td>
   <td>61</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Gradient ID, 7-shot</td>
   <td>12.24K</td>
   <td>95</td>
   <td>13.40K</td>
   <td>108</td>
   <td>13.42K</td>
   <td>62</td>
  </tr>
  <tr>
   <td>Soft-TF-L[12] w/o FF</td>
   <td>2.31M</td>
   <td>Gradient ID, m-batch</td>
   <td>12.24K</td>
   <td>103</td>
   <td>13.40K</td>
   <td>132</td>
   <td>13.42K</td>
   <td>66</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Prompt ID</td>
   <td>13.87K</td>
   <td>80</td>
   <td>15.60K</td>
   <td>104</td>
   <td>15.35K</td>
   <td>52</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Gradient ID, 3-shot</td>
   <td>13.71K</td>
   <td>96</td>
   <td>15.51K</td>
   <td>106</td>
   <td>15.08K</td>
   <td>63</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Gradient ID, 5-shot</td>
   <td>13.71K</td>
   <td>96</td>
   <td>15.51K</td>
   <td>109</td>
   <td>15.08K</td>
   <td>74</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Gradient ID, 7-shot</td>
   <td>13.71K</td>
   <td>106</td>
   <td>15.51K</td>
   <td>119</td>
   <td>15.08K</td>
   <td>75</td>
  </tr>
  <tr>
   <td>Soft-TF-L[10,11,12] w/o FF</td>
   <td>6.93M</td>
   <td>Gradient ID, batch</td>
   <td>13.71K</td>
   <td>130</td>
   <td>15.51K</td>
   <td>163</td>
   <td>15.08K</td>
   <td>82</td>
  </tr>
 </tbody>
</table>

Comparisions of Soft-TF with LLMs. To demonstrate the effectiveness of Soft-TF, we compare Soft-TF against LLM fine-tuning methods such as Adapters (Houlsby et al., 2019) and LoRA (Hu

19

<table>
<caption>Table 9: Performances of Class Incremental Learning (CIL) in terms of Soft parameters, Accuracy, and Forget on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Note "w/o FF" denotes "Soft finetuning without FeedForward (FF)" networks.</caption>
<thead>
<tr>
<th colspan="2">Method</th>
<th>ViT-B/12 (85.8M)</th>
<th rowspan="2">Task ID</th>
<th colspan="2">10-Split-CIFAR100</th>
<th colspan="2">20-Split-CIFAR100</th>
<th colspan="2">10-Split-ImageNet-R</th>
</tr>
<tr>
<th colspan="2">DualPrompt</th>
<th># Train Params.</th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>DualPrompt</td>
<td></td>
<td>0.00M</td>
<td>Prompt ID</td>
<td>86.50</td>
<td>5.77</td>
<td>82.98</td>
<td>8.20</td>
<td>68.13</td>
<td>4.46</td>
</tr>
<tr>
<td>PGP</td>
<td></td>
<td>0.00M</td>
<td>Prompt ID</td>
<td>86.92</td>
<td>5.35</td>
<td>83.74</td>
<td>7.91</td>
<td>69.34</td>
<td>4.53</td>
</tr>
<tr>
<td>Soft-TF-L[12]</td>
<td>w/o FF</td>
<td>2.31M</td>
<td>Prompt ID</td>
<td>91.83</td>
<td>2.99</td>
<td>96.43</td>
<td>1.00</td>
<td>72.45</td>
<td>5.32</td>
</tr>
<tr>
<td>Soft-TF-L[12]</td>
<td>w/o FF</td>
<td>2.31M</td>
<td>Gradient ID, 3-shot</td>
<td>93.12</td>
<td>1.82</td>
<td>96.43</td>
<td>1.00</td>
<td>73.55</td>
<td>4.80</td>
</tr>
<tr>
<td>Soft-TF-L[12]</td>
<td>w/o FF</td>
<td>2.31M</td>
<td>Gradient ID, 5-shot</td>
<td>96.13</td>
<td>0.58</td>
<td>96.43</td>
<td>1.00</td>
<td>75.04</td>
<td>4.49</td>
</tr>
<tr>
<td>Soft-TF-L[12]</td>
<td>w/o FF</td>
<td>2.31M</td>
<td>Gradient ID, 7-shot</td>
<td>96.51</td>
<td>0.65</td>
<td>96.43</td>
<td>1.00</td>
<td>76.34</td>
<td>4.75</td>
</tr>
<tr>
<td>Soft-TF-L[12]</td>
<td>w/o FF</td>
<td>2.31M</td>
<td>Gradient ID, batch</td>
<td>96.84</td>
<td>0.55</td>
<td>97.81</td>
<td>0.57</td>
<td>81.18</td>
<td>1.31</td>
</tr>
<tr>
<td>Soft-TF-L[10,11,12]</td>
<td>w/o FF</td>
<td>6.93M</td>
<td>Prompt ID</td>
<td>92.35</td>
<td>2.98</td>
<td>97.40</td>
<td>0.57</td>
<td>74.62</td>
<td>5.30</td>
</tr>
<tr>
<td>Soft-TF-L[10,11,12]</td>
<td>w/o FF</td>
<td>6.93M</td>
<td>Gradient ID, 3-shot</td>
<td>93.92</td>
<td>1.62</td>
<td>97.40</td>
<td>0.57</td>
<td>74.99</td>
<td>4.21</td>
</tr>
<tr>
<td>Soft-TF-L[10,11,12]</td>
<td>w/o FF</td>
<td>6.93M</td>
<td>Gradient ID, 5-shot</td>
<td>97.37</td>
<td>0.53</td>
<td>97.40</td>
<td>0.57</td>
<td>77.40</td>
<td>2.91</td>
</tr>
<tr>
<td>Soft-TF-L[10,11,12]</td>
<td>w/o FF</td>
<td>6.93M</td>
<td>Gradient ID, 7-shot</td>
<td>97.76</td>
<td>0.51</td>
<td>97.40</td>
<td>0.57</td>
<td>79.33</td>
<td>3.57</td>
</tr>
<tr>
<td>Soft-TF-L[10,11,12]</td>
<td>w/o FF</td>
<td>6.93M</td>
<td>Gradient ID, m-batch</td>
<td>98.05</td>
<td>0.25</td>
<td>98.96</td>
<td>0.23</td>
<td>83.70</td>
<td>0.53</td>
</tr>
</tbody>
</table>

et al., 2021), as shown in Table 7. Under identical experimental conditions—including trainable model parameters ( 6.9M per task), layers (L[10,11,12]), and Prompt ID—Soft-TF outperformed other LLM-based fine-tuning approaches. The results highlight that directly updating well-pretrained model parameters and prompt-tuning via Soft-TF is more effective than combining representations through LoRA or learning representations with Adapters. Furthermore, we observed that Soft-TF and the other methods exhibited comparable training and testing time complexity for the same number of trainable parameters. Notably, single-layer fine-tuning using Soft-TF (with L[12]) surpasses the performance of the baselines. These findings firmly establish Soft-TF as the most competitive approach among strong LLM baselines (Adapters and LoRA) in the continual learning (CIL) scenario.

Sparsity of Transformer. We inspect the sparse solution through WSN as shown in Table 4, Table 10, and Table 11. We found a suboptimal sparse solution (c=87.0 % on 10-Split-TinyImageNet) with minimal CF through the inspections. This demonstrates the Rottary Ticket Hypothesis (RTH) in transformers, a competitive sparse subnetwork in DenseNetwork. In addition, DualPrompt is the lower-bound while DualPrompt-Soft-TF-* is the upper-bound, close to the optimal performances.

![](./images/1068413726366892118_7.jpg)
(a) WSN-Transformers v.s. Soft-Transformers

![](./images/1068413726366892118_8.jpg)
(b) WSN v.s. SoftNet

Figure 5: Comparisions of Soft-Transformers with Subnetworks on 10-Split-CIFAR100. Note that L[10,11,12] denotes the fine-tuning layers of 10, 11, and 12.

<table>
<caption>Table 10: Performances of Subnetworks (WSN) in Class Incremental Learning (CIL) on 10-Split-TinyImageNet.</caption>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Pretrained-Dataset</th>
<th rowspan="2">Task ID</th>
<th colspan="2">TinyImageNet</th>
</tr>
<tr>
<th>ACC(↑)</th>
<th>Forget(↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>DualPrompt</td>
<td>-</td>
<td>Prompt ID</td>
<td>86.50</td>
<td>5.77</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=80.0%</td>
<td>Gradient ID</td>
<td>89.95</td>
<td>0.98</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=81.0%</td>
<td>Gradient ID</td>
<td>89.99</td>
<td>1.06</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=82.0%</td>
<td>Gradient ID</td>
<td>89.59</td>
<td>1.18</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=83.0%</td>
<td>Gradient ID</td>
<td>90.60</td>
<td>0.72</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=85.0%</td>
<td>Gradient ID</td>
<td>90.09</td>
<td>1.07</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=87.0%</td>
<td>Gradient ID</td>
<td>91.91</td>
<td>0.38</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=90.0%</td>
<td>Gradient ID</td>
<td>91.28</td>
<td>0.42</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=93.0%</td>
<td>Gradient ID</td>
<td>91.41</td>
<td>0.40</td>
</tr>
<tr>
<td>DualPrompt-WSN-L[10,11,12]</td>
<td>C=95.0%</td>
<td>Gradient ID</td>
<td>90.91</td>
<td>0.83</td>
</tr>
<tr>
<td>DualPrompt-Soft-TF-L[10,11,12]</td>
<td>-</td>
<td>Gradient ID</td>
<td>97.87</td>
<td>0.21</td>
</tr>
</tbody>
</table>

20

<table>
<thead>
  <tr>
    <th colspan="2">Table 11: Performances of Class Incremental Learning (CIL) in terms of Soft parameters, Accuracy, and Forget on 10/20-Split-CIFAR100 and 10-Split-ImageNet-R. Note "w/ only ATTN" denotes "Soft finetuning only with attention (ATTN)" and FeedForward (FF) networks.</th>
    <th>ViT-B/12 (85.8M)</th>
    <th></th>
    <th colspan="2">10-Split-CIFAR100</th>
    <th colspan="2">20-Split-CIFAR100</th>
    <th colspan="2">10-Split-ImageNet-R</th>
  </tr>
  <tr>
    <th colspan="2">Method</th>
    <th># Train Params.</th>
    <th>Task ID</th>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
    <th>ACC(↑)</th>
    <th>Forget(↓)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DualPrompt</td>
    <td>DualPrompt</td>
    <td>0.00M</td>
    <td>Prompt ID</td>
    <td>86.50</td>
    <td>5.77</td>
    <td>82.98</td>
    <td>8.20</td>
    <td>68.13</td>
    <td>4.46</td>
  </tr>
  <tr>
    <td>PGP</td>
    <td></td>
    <td>0.00M</td>
    <td>Prompt ID</td>
    <td>86.92</td>
    <td>5.35</td>
    <td>83.74</td>
    <td>7.91</td>
    <td>69.34</td>
    <td>4.53</td>
  </tr>
  <tr>
    <td>Soft-TF-L[12]</td>
    <td>w/ only ATTN</td>
    <td>1.76M</td>
    <td>Gradient ID</td>
    <td>97.17</td>
    <td>0.40</td>
    <td>98.09</td>
    <td>0.54</td>
    <td>72.31</td>
    <td>3.94</td>
  </tr>
  <tr>
    <td>Soft-TF-L[12]</td>
    <td>w/ only ATTN, WSN c=90%</td>
    <td>1.58M</td>
    <td>Gradient ID</td>
    <td>96.81</td>
    <td>0.61</td>
    <td>97.51</td>
    <td>1.68</td>
    <td>71.67</td>
    <td>3.94</td>
  </tr>
</tbody>
</table>

Pseudo Codes. The overall process of the Soft-TransFormers (Soft-TF) during training and testing is described as Algorithm 1 and Algorithm 2. We denote the architecture with attached prompts as $f_{{\boldsymbol {g}},{\boldsymbol {e}}_{t},{\boldsymbol {m}}_{t}}$. The input $\boldsymbol {x}$ from the $t$-th task is transformed using $f_{{\boldsymbol {g}},{\boldsymbol {e}}_{t},{\boldsymbol {m}}_{t}}$ and then passed to the classification head $f_{\phi}$, parameterized by $\phi$, for prediction. Finally, we train the two prompts, the task keys, the soft-attention parameters, and the newly-initialized classification head in an end-to-end manner.

```
Algorithm 1 DualPrompt-Soft-TF at training time
 1: Input: Pre-trained transformer-based backbone $f$, final classification layer $f_{\phi}$,
 2:    number of tasks $\mathcal {T}$, training set $\{\{x_{i,t},y_{i,t}\}_{i=1}^{n_{t}}\}_{t=1}^{T}$, G-Prompt $\boldsymbol {g}$, E-Prompt $\boldsymbol {E}=\{e_{t}\}_{t=1}^{T}$,
 3:    task keys $\boldsymbol {K}=\{k_{t}\}_{t=1}^{T}$, soft-networks $\boldsymbol {M}=\{m_{t}\}_{t=1}^{T}$, $start_{g}$, $end_{g}$, $start_{e}$, $end_{e}$,
 4:    prompting function $f_{\boldsymbol {\theta }\odot {\boldsymbol {m}}}^{prompt}$,
 5:    number of training epochs of the $t$-th task $\mathcal {K}_{t}$.
 6: Initialize: $\phi$, $\boldsymbol {g}$, $\boldsymbol {E}$, $\boldsymbol {M}$, $\boldsymbol {K}$
 7: for task $t=1,\cdots ,\mathcal {T}$ do
 8:    Select the task-specific E-Prompt, soft-network $e_{t}$, $m_{t}$ and corresponding task key $k_{t}$
 9:    Generate the prompted architecture $f_{{\boldsymbol {g}},{\boldsymbol {e}}_{t},{\boldsymbol {m}}_{t}}$: attach $\boldsymbol {g}$ and $e_{t}$ to $start_{g}$-th to $end_{g}$-th
10:     and $start_{e}$-th to $end_{e}$-th soft MSA layers respectively, with $f_{\boldsymbol {\theta }\odot {\boldsymbol {m}}}^{prompt}$.
11:    for batch $e_{s}\sim \mathcal {K}_{t}$ do
12:        Draw a mini-batch $B=\{(x_{i,t},y_{i,t})\}_{i=1}^{l}$
13:        for $(x,y)$ in B do
14:            Calculate the prompted feature by
15:            Calculate the per sample loss $\mathcal {L}_{x}$ via
16:        end for
17:        Update $\phi$, $\boldsymbol {g}$, $\boldsymbol {E}$, $\boldsymbol {M}$, $\boldsymbol {K}$ by back-propagation
18:    end for
19: end for
```

```
Algorithm 2 DualPrompt-Soft-TF at test time
 1: Given components: Pre-trained transformer-based backbone $f$, trained
 2:    $\boldsymbol {K}=\{k_{t}\}_{t=1}^{T}$, $\boldsymbol {M}=\{m_{t}\}_{t=1}^{T}$, $start_{g}$, $end_{g}$, $start_{e}$, $end_{e}$, prompting function $f_{\boldsymbol {\theta }\odot {\boldsymbol {m}}}^{prompt}$
 3: Input: test example $\boldsymbol {x}$ from mini-batch $\boldsymbol {b}$
 4: Select task inference method: (1) Prompt ID or (2) Gradient ID
 5:    (1) Prompt ID:
 6:    Generate query feature $q(\boldsymbol {x})$
 7:    Matching for the index of E-Prompt via $t_{\boldsymbol {x}}=\text{argmin}_{t}\gamma (q(\boldsymbol {x}),k_{t})$
 8:    (2) Gradient ID:
 9:    Assigning each learned subnetwork $m_{t}$ a weight $\alpha _{t}$ such that $\sum _{t}\alpha _{t}=1$ and $\alpha _{t}=1/\mathcal {T}>0$.
10:    Given $x\in \boldsymbol {b}$ to classify, We can compute our loss $\mathcal {L}=\mathcal {H}(f_{{\boldsymbol {\theta }\odot (\sum _{t}\alpha _{t}{\boldsymbol {m}}_{t})}}^{prompt}(\boldsymbol {x}))$
11:    Matching for the index of E-Prompt via $t_{\boldsymbol {x}}=\text{argmin}_{t}\frac {\partial \mathcal {H}}{\partial \alpha _{t}}$
12: Select the task-specific E-prompt $e_{t_{\boldsymbol {x}}}$ and learned subnetwork $m_{t_{\boldsymbol {x}}}$
13: Generate the prompted architecture $f_{{\boldsymbol {g}},{\boldsymbol {e}}_{t_{\boldsymbol {x}}},{\boldsymbol {m}}^{t_{\boldsymbol {x}}}}$:
14:    Attaching $\boldsymbol {g}$ and $e_{t_{\boldsymbol {x}}}$ to $start_{g}$-th to $end_{g}$-th
15:     and $start_{e}$-th to $end_{e}$-th MSA layers respectively, with $f_{\boldsymbol {\theta }\odot {\boldsymbol {m}}}^{prompt}$.
16: Prediction: $f_{{\boldsymbol {g}},{\boldsymbol {e}}_{t_{\boldsymbol {x}}},{\boldsymbol {m}}_{t_{\boldsymbol {x}}}}(\boldsymbol {x})$
```

Density of Parameters. We inspect the histogram density estimate of the last (12) layer’s parameters of DualPrompt-Soft-TF: attention of QKV ((a) ATTN.QKV) and Projection ((b) ATTN.PROJ) and multi-layer perception (MLP) of FC1 and FC2, as shown in Figure 6. ATTN’s QKV parameters

21

have the largest variance among the parameter densities, while MLP-FC2's are the smallest. From this observation, we conclude that fine-tuning ATTN's QKV is required to achieve optimal task performance. In other words, QKV's parameters are more critical than others.

![](./images/1068413726366892118_9.jpg)

Figure 6: Layer-(L[12]) Histogram Density Estimates of DualPrompt-Soft-TF's Parameters on 10-Split-CIFAR100.

Pre-trained Parameters v.s. Soft-TF. We inspect the histogram density estimate of the last (12) layer's parameters of pre-trained model and DualPrompt-Soft-TF: attention of QKV ((a) ATTN.QKV) and Projection ((b) ATTN.PROJ) and multi-layer perception (MLP) of FC1 and FC2, as shown in Figure 7. The most parameters of DualPrompt-Soft-TF are trained around zero-values. Particularly, the difference between pre-trained model's parameters and Soft-TF is distinctive at QKV module.

Public Source Code. All official source codes will be available soon.

![](./images/1068413726366892118_10.jpg)

Figure 7: Layer-(L[12]) Histogram Density Estimates of Pre-trained Weight and DualPrompt-Soft-TF's Parameters on 10-Split-CIFAR100.