# SELF-CONSISTENT REASONING FOR SOLVING MATH WORD PROBLEMS

［#1］
Jing Xiong¹, Zhongwei Wan²·³, Xiping Hu¹, Min Yang²·³, Chengming Li¹

［#2］
¹ Sun Yat-Sen University, China
² University of Chinese Academy of Sciences, China
³ SIAT, Chinese Academy of Sciences, China

［#2］
xiong69@mail2.sysu.edu.cn, {huxiping, lichengming}@mail.sysu.edu.cn, {zw.wan1, min.yang}@siat.ac.cn

## ABSTRACT

［#3］
Math word problems (MWPs) is a task that automatically derives solution expression from a giving math problems in text. The previous studies suffer from spurious correlations between input text and output expression. To mitigate this issue, we propose a self-consistent reasoning framework called **SCR**, which attempts to adopt a pruning strategy to correct the output distribution shift so as to implicitly fix those spurious correlative samples. Specifically, we firstly obtain a sub-network by pruning a roberta2tree model, for the sake to use the gap on output distribution between the original roberta2tree model and the pruned sub-network to expose spurious correlative samples. Then, we calibrate the output distribution shift by applying symmetric Kullback-Leibler divergence to alleviate spurious correlations. In addition, SCR generates equivalent expressions, thereby, capturing the original text's logic rather than relying on hints from original text. Extensive experiments on two large-scale benchmarks demonstrate that our model substantially outperforms the strong baseline methods.

［#4］
Index Terms— Math Word Problems, Spurious correlative samples, Pruning, Self-consistency

## 1. INTRODUCTION

［#5］
Math word problems (MWPs) [1] is a challenging symbolic logical reasoning task based on natural language description and draws much attention from researchers about the reasoning power of large language models [2, 3, 4] recently. MWPs aims to automatically solve mathematical questions given in a natural language, which requires the model not only to understand the natural language but also to have the ability to reason logically. Table 1 shows several examples of MWPs.

［#6］
At present, there are mainly three paradigms of models that have achieved excellent performance, namely seq2seq [5, 6, 7], seq2tree [8, 9], and complex relation extraction [10]. But all three paradigm model suffers from spurious correlations[11, 12, 10]. Take an example of Table 1, some of the previous works may obtain the same mathematical formula as "$a \div b \times c$" for Problem 1 and Problem 2, due to the similar semantic context information, e.g., *Calculate the money situation*. However, if the models ignore the spurious correlations, they intend to generate the wrong solution expression for Problem 3, which has very analogous semantic information like the important words "money", "bank", and "account," which exist in problems 1 and 2. To be specific, the models that learn spurious information among problems 1 to 3 are more likely to generate the wrong expression "$12500\div5\%\times15\%$" instead of "$12500\div(5\%+15\%)$" for problem 3 *Calculate the money of the account*.

［#7］
<table>
  <tr>
    <td><b>Problem 1:</b> Tom takes the money from his bank account and has taken 240 dollars of his account for 3 days. If he takes the same amount of money every day, how much money will Tom take for next 2 days?
    <br><b>Solution Expression:</b> $240\div3\times2$ &nbsp;&nbsp;<b>Solution:</b> 160</td>
  </tr>
  <tr>
    <td><b>Problem 2:</b> Sherry has deposited 6000 dollars to the bank for the last 5 months. If she saves the same monthly money, how much will she add to the account in the next 3 months?
    <br><b>Solution Expression:</b> $6000\div5\times3$ &nbsp;&nbsp;<b>Solution:</b> 3600</td>
  </tr>
  <tr>
    <td><b>Problem 3:</b> Uncle Jack spends 5% of his bank account to invest for the trust funds of States and 15% of the account for the shares of Apple Inc. The money he has spent on financial management is 12500 dollars. How much money is in Uncle Jack's account?
    <br><b>Solution Expression:</b> $12500\div(5\%+15\%)$ &nbsp;&nbsp;<b>Solution:</b> 62500
    <br><b>Wrong Solution Expression:</b> $12500\div5\%\times15\%$</td>
  </tr>
</table>

［#8］
Table 1. A typical math word examples of the spurious correlation.

［#9］
Some recent models address this problem by using variational information bottlenecks[9]. Our article considers this problem from the perspective of memorization. Some recent articles have revealed that pruning can make the model forget some hard-to-memorize samples[13]. In addition, [14] have revealed that long-tailed samples are easily forgettable. Usually, these long-tailed samples easily confuse the model, and the model will generate the final result based on some shallow hints. A natural hypothesis is that some spurious correlative samples are tougher for the model to learn well due to shortcuts, and these samples can be adaptively exposed by pruning [14]. A key question in the task of MWPs is how to implicitly emphasize their shortcuts between expressions and original texts when exposing spurious correlative samples through pruning. Some work about the reasoning ability of large models has also revealed that encouraging the model to produce self-consistent outputs can effectively improve reasoning performance when the model produces multiple inferences [2, 3, 4]. However, their work uses voting to encourage self-consistency, which cannot adaptively correct the shortcuts of expressions and original texts online through the loss function.

［#10］
In this paper, we propose a self-consistent reasoning framework (called SCR) to solve MWPs tasks. We obtain a sub-network by pruning the roberta2tree model denoted as the source network. Our SCR model adaptively finds spurious correlative samples through

［#10］
pruning. Specifically, SCR encourages the models' prediction consistency through mutual learning [15], which can emphasize samples with inconsistent prediction distributions between the source network and sub-network.

［#11］
We summarize our main contributions: (1) We propose a novel self-consistent reasoning framework for MWPs to expose spurious correlative samples and correct them adaptively. (2) We conduct extensive experiments on two benchmark datasets (i.e., Math23k and Ape210k). The results demonstrate that our model performs significantly better than the strong baselines.

## 2. METHODOLOGY

［#12］
A math word problems (MWPs) can be denoted by a projection $F: W \mapsto Y$, where $W = \{w_1, w_2, ..., w_m\}$ is the problem sequence with $m$ words and $Y = \{y_1, y_2, ..., y_n\}$ is the solution expression of the problem with $n$ words. MWPs aim to establish a model $F$ which generates a correct solution expression $Y$ and calculates the correct answer for the problem $W$.

［#13］
As illustrated in Figure 1, the proposed SCR is composed of a source network (denoted as $S$) and a sub-network (denoted as $C$). The sub-network is obtained by pruning the source network. The two networks are optimized collaboratively and teach each other throughout the training process. We use the encoder-decoder framework as the backbone of both source and sub-networks.

### 2.1. The Encoder-Decoder Architecture

［#14］
To efficiently obtain a high-quality representation of the problem, we utilize the RoBERTa model [16] as our encoder. We pass the problem sequence $W$ into the RoBERTa model and obtain problem representation $H \in \mathbb{R}^{m*d}$, where $d$ is the embedding size of the encoder. In order to model the relationship between the quantities in the pre-training model, we set up a learnable quantity embedding matrix $T_E = \{t_1, t_2, ..., t_n\}$, similar to the learnable position embedding in BERT [17]. Before passing the sequence $W$ into the encoder, we first replace each quantity in the sequence $W$ with a token $t_i \in T_t$.

［#15］
Inspired by GTS model [18], our decoder uses the recursive operation of the decoder to construct $Y$ by order of pre-order traversal. First, the decoder generate the root node $t_{root}$ (middle operator part) . Then, the decoder generates the left child node $t_l$. Furthermore, the right child node $t_r$ is generated. This process has been iterated until the leaf nodes are generated. Specifically, we apply the attention mechanism to learn the global context vector $G_i$, which is utilized to generate the current node token $\hat{y}_i$. Here we denote the digital embedding after being encoded by the encoder as $T$. The formula of the attention mechanism is shown below:

［#15］
$$
G_{i}=\left\{\begin{array}{ll}
\text { Attention }\left(H, t_{\text {root }}, t_{l}\right), & t_{l} \notin \emptyset. \\
\text { Attention }\left(H, t_{\text {root }}, t_{s l}\right), & t_{s l} \notin \emptyset. \\
\text { Attention }\left(H, t_{\text {root }}\right), & t_{l}, q_{s l} \in \emptyset.
\end{array}\right.
\tag{1}
$$

［#16］
$$
\hat{y}_{i}=\operatorname{Predict}(G_{i}, T).
\tag{2}
$$

［#17］
where $\text{Predict}(\cdot)$ is the final prediction layer for producing the tree node.

［#18］
The tree decoder will generate the left and right child nodes and push them into the stack using the top-down approach if the current node is an operator. Moreover, if it is a number, the merge operation will be carried out until the stack's leaf nodes emerge, at which point the result will be pushed into the stack of left child nodes for the attention operation. Then, the merge operation will pop the required node $t_{op}$ and $t_{subtree}$ from an embedding stack. The formula of recursive construction is as follows:

［#18］
$$
t_l = \operatorname{Left}(G_i, \hat{y}_i, t_{root}).
\tag{3}
$$

［#19］
$$
t_r = \operatorname{Right}(G_i, \hat{y}_i, t_{root}).
\tag{4}
$$

［#20］
$$
t_m = \operatorname{Merge}(t_{op}, t_{subtree}, t_{m-1}).
\tag{5}
$$

［#21］
![](./images/867773263353217472_1.jpg)

### 2.2. Self-consistent Reasoning

［#22］
As shown in Figure 1, the proposed SCR comprises a source network and a sub-network. The sub-network is obtained by pruning the source network. In each iteration, the source network will correct the distribution shift of the output $p_2$ from the sub-network to implicitly emphasize the spurious correlative samples. When we finish training the sub-network in this iteration, the sub-network also provides the supervision signals to correct the distribution shift from the output $p_1$ from the source network. Specifically, we preferentially fix the output distribution of the sub-network when neither network is trained by samples to expose spurious collaborative samples better. At the same time, these two networks are also trained by ground-truth supervision signals.

［#23］
Formally, the training objective of the source network is to minimize negative log-likelihood (NLL) loss for each instance $(W,Y)$ from training data:

［#23］
$$
\mathcal{L}_{S}(\theta_{S})=-\sum_{i=1}^{n} \boldsymbol{y}_{i} \log p(\hat{\boldsymbol{y}}_{i} \mid W ; \theta_{S}).
\tag{6}
$$

［#24］
where $y_i$ is ground-truth of step i. $\theta_S$ denotes the parameters of the source network.

［#25］
We prune the model parameters $\theta_S$ of the source model and obtain the parameters $\theta_C$ of the sub-network. The training objective of the sub-network $C$ can be defined as:

［#25］
$$
\mathcal{L}_{\mathrm{C}}(\theta_{C})=-\sum_{i=1}^{n} \boldsymbol{y}_{i} \log p(\hat{\boldsymbol{y}}_{i} \mid W ; \theta_{C}).
\tag{7}
$$

［#26］
Inspired by the mutual learning [15], we train the sub-network and the source network collaboratively by the symmetric Kullback Leibler (KL) Divergence. First, we use the KL Divergence to measure the distance from the source network's prediction $\boldsymbol{p}_1$ to the sub-networks prediction $\boldsymbol{p}_2$ by:

［#26］
$$
D_{K L}\left(\boldsymbol{p}_{2} \| \boldsymbol{p}_{1}\right)=\sum_{i=1}^{n} p_{2}\left(\hat{\boldsymbol{y}}_{i}\right) \log \frac{p_{2}\left(\hat{\boldsymbol{y}}_{i}\right)}{p_{1}\left(\boldsymbol{y}_{i}\right)}. \tag{8}
$$

［#27］
where $n$ is the length of the solution expression. $\boldsymbol{y}_{i}$ and $\hat{\boldsymbol{y}}_{i}$ denote the $i$-th ground-truth and generated tokens, respectively. Considering that KL divergence is asymmetric, we also calculate the divergence from $p_2$ to $p_1$:

［#27］
$$
D_{K L}\left(\boldsymbol{p}_{1} \| \boldsymbol{p}_{2}\right)=\sum_{i=1}^{n} p_{1}\left(\hat{\boldsymbol{y}}_{i}\right) \log \frac{p_{1}\left(\hat{\boldsymbol{y}}_{i}\right)}{p_{2}\left(\boldsymbol{y}_{i}\right)}. \tag{9}
$$

［#28］
By averaging equations 8 and 9, we get a symmetric KL divergence, denoted as $\bar{D}_{K L}$. Note that we alternately optimize $S$ and $C$ in each iteration, and calculate symmetric KL divergence $\bar{D}_{K L_{1}}$ and $\bar{D}_{K L_{2}}$ for $S$ and $C$, respectively. We define the overall loss functions $\mathcal{L}_{S}$ and $\mathcal{L}_{C}$ for networks $S$ and $C$ respectively as follows:

［#28］
$$
\mathcal{L}_{S}=\mathcal{L}_{1}\left(\theta_{S}\right)+\alpha \times \bar{D}_{K L_{1}}. \tag{10}
$$

［#29］
$$
\mathcal{L}_{C}=\mathcal{L}_{2}\left(\theta_{C}\right)+\alpha \times \bar{D}_{K L_{2}}. \tag{11}
$$

［#30］
where $\alpha$ is a proportional coefficient. In this way each network learns to both correctly predict the ground-truth of training instances and match the probability estimation of its peer network.

## 3. EXPERIMENT

### 3.1. Experimental Setup

#### 3.1.1. Datasets
［#31］
We conduct experiments on two benchmark MWPs datasets: Math23k [19] and Ape210k [20]. Math23k contains 22162/1000 questions for training/testing, respectively. Ape210k is composed of 166,270 questions for training, 4,157 questions for validation, and 4,159 questions for testing.

#### 3.1.2. Implementation Details
［#32］
The word embedding size of the decoder is set to 1024. We adopt RoBERTa [16] as the problem encoder. Following Roberta's setting, the encoder's hidden size is 768, and we set the hidden size of the decoder to 1024. We used Adamw [21] as the optimizer with the learning rate as 5e-5. The mini-batch size is set to 16. We adopt a beam search with a size of 5. Dropout (dropout rate = 0.5) is employed to avoid overfitting. For Ape210K, we set the maximum sequence length of questions as 150 and that of solution expressions as 50, similar to [22]. For convergence, our model takes 80 epochs on Math23k and 50 epochs on Ape210k.

#### 3.1.3. Baselines
［#33］
NS-Solver [23]. This model uses auxiliary tasks to explicitly and seamlessly merge different levels of symbolic constraints to enhance the model's ability to solve MWPs. NumS2T [22]. This model uses the numerical properties prediction mechanism to capture the category and comparison information of the numerals and measure their importance in global expressions, which explicitly merges the numerical values into the sequence-to-tree network. TSN-MD [24]. This model proposes a multi-decoder network based on distillation learning to generate diverse expressions. MATH-EN [7]. This model proposes an equation normalization method to normalize repeated equations and also designs an ensemble method to improve the model's performance. Graph2Tree [8]. This model combines the advantages of graph-based encoders and tree-based decoders to generate better solution expressions. Multi-E/D [25]. This model combines sequence-based encoders and graph-based encoders to enhance the presentation capabilities of text descriptions and generates different equation expressions through sequence-based decoders and tree-based decoders. GTS [18]. This model designs a tree decoder to generate expressions. Tree-Decoder [26]. This model proposes a tree-structured decoding method to generate abstract syntax trees of equations in a top-down manner. Ape [20]. This model designs a seq2seq model to solve the MWPs problem. StackDecoder [27]. This model designs a stack-based decoder to generate expressions. KAS2T [22]. This model inserts external knowledge and state aggregation mechanisms into the seq2tree model. Generat-eRank [28]. This model devises a new ranking task for MWPs and proposes joint training with generation and ranking on a generative pre-trained language model. DeductReasoner [29]. This model views the task as a complex relation extraction problem and propose a novel approach that presents explainable deductive reasoning steps to iteratively construct target expressions.

### 3.2. Experimental Results

#### 3.2.1. Main Results
［#34］
The evaluation metric is answer accuracy. Table 2 show the performance comparison of our model with baseline methods on Math23K and Ape210k, respectively. Both our source network and sub-network achieve substantially better performance than the strong competitors, verifying the effectiveness of our self-consistent reasoning framework. We compute the accuracy of the generated solution expression as a correct one when the predicted expression exactly matches the annotated solution. The expression prediction accuracy is reported in Table 3. We can see that the accuracy of solution expression generation is lower than the final answer prediction accuracy, showing that our model can generate some diverse solution expressions (not included in ground-truth expressions), leading to correct answers. This experimental result also shows the strong generalization ability of our model because only after the model understands the original text can it predict equivalent and correct expressions instead of relying on some shallow heuristics in the original text.

#### 3.2.2. Ablation Study
［#35］
We conduct an ablation test on Math23k to analyze the impact of different components in SCR. First, we remove the mutual learning from the source network and sub-network, denoted as source network w/o MT and sub-network w/o MT, respectively. Second, we replace the pruned sub-network with the source network to evaluate the impact of pruning (denoted as source network w/o pruning and sub-network w/o pruning, respectively). We summarize the results in Table. 4. Both the pruning strategy and mutual learning contribute greatly to the performance of SCR.

［#36］
<table>
  <thead>
    <tr>
      <th>Models</th>
      <th>Math23k</th>
      <th>Math23k†</th>
      <th>Ape210k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>StackDecoder</td>
      <td>-</td>
      <td>65.8</td>
      <td>52.2</td>
    </tr>
    <tr>
      <td>Tree-Decoder</td>
      <td>69.0</td>
      <td>-</td>
      <td>66.5</td>
    </tr>
    <tr>
      <td>GTS</td>
      <td>75.6</td>
      <td>74.3</td>
      <td>67.7</td>
    </tr>
    <tr>
      <td>KAS2T</td>
      <td>76.3</td>
      <td>-</td>
      <td>68.7</td>
    </tr>
    <tr>
      <td>TSN-MD</td>
      <td>77.4</td>
      <td>75.1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Graph2Tree</td>
      <td>77.4</td>
      <td>75.5</td>
      <td>-</td>
    </tr>
    <tr>
      <td>NS-Solver</td>
      <td>-</td>
      <td>75.6</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Ape</td>
      <td>-</td>
      <td>77.5</td>
      <td>70.2</td>
    </tr>
    <tr>
      <td>NumS2T</td>
      <td>78.1</td>
      <td>-</td>
      <td>70.5</td>
    </tr>
    <tr>
      <td>Multi-E/D</td>
      <td>78.4</td>
      <td>76.9</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GenerateRank</td>
      <td>85.4</td>
      <td>84.3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeductReasoner</td>
      <td>85.1</td>
      <td>83.0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Source Network</td>
      <td>85.5</td>
      <td>84.5</td>
      <td>76.3</td>
    </tr>
    <tr>
      <td>Sub-network</td>
      <td>86.8</td>
      <td>84.6</td>
      <td>76.7</td>
    </tr>
  </tbody>
</table>

［#37］
Table 2. Solution accuracy of SCR and various baselines. Note that Math23K denote results on public test set and Math23K† denote 5-fold cross-validation.

［#38］
<table>
  <thead>
    <tr>
      <th>Models</th>
      <th>Answer-ac</th>
      <th>Equation-ac</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MATH-EN</td>
      <td>66.7</td>
      <td>60.1</td>
    </tr>
    <tr>
      <td>GTS</td>
      <td>75.6</td>
      <td>64.8</td>
    </tr>
    <tr>
      <td>TSN-MD</td>
      <td>77.4</td>
      <td>65.8</td>
    </tr>
    <tr>
      <td>Source Network</td>
      <td>85.5</td>
      <td>73.3</td>
    </tr>
    <tr>
      <td>Sub-network</td>
      <td>86.8</td>
      <td>73.5</td>
    </tr>
  </tbody>
</table>

［#39］
Table 3. Accuracy of equation generation on Math23k.

［#40］
<table>
  <thead>
    <tr>
      <th>Models</th>
      <th>Math23k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Source network w/o pruning</td>
      <td>84.8</td>
    </tr>
    <tr>
      <td>Sub-network w/o pruning</td>
      <td>85.6</td>
    </tr>
    <tr>
      <td>Source network w/o MT</td>
      <td>84.2</td>
    </tr>
    <tr>
      <td>Sub-network w/o MT</td>
      <td>84.3</td>
    </tr>
    <tr>
      <td>Source Network</td>
      <td>85.5</td>
    </tr>
    <tr>
      <td>Sub-network</td>
      <td>86.8</td>
    </tr>
  </tbody>
</table>

［#41］
Table 4. Ablation study on Math23k.

［#42］
<table>
  <thead>
    <tr>
      <th>$\alpha$</th>
      <th>Math23k</th>
      <th>Pruning Rate</th>
      <th>Math23k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.0005</td>
      <td>85.1</td>
      <td>0.1</td>
      <td>85.8</td>
    </tr>
    <tr>
      <td>0.005</td>
      <td>86.8</td>
      <td>0.2</td>
      <td>86.8</td>
    </tr>
    <tr>
      <td>0.05</td>
      <td>$\bigtriangleup$</td>
      <td>0.3</td>
      <td>84.4</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>0.4</td>
      <td>84.4</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>0.5</td>
      <td>$\bigtriangleup$</td>
    </tr>
  </tbody>
</table>

［#43］
Table 5. The Sensitivity Analysis of $\alpha$ and Pruning Rate. $\bigtriangleup$ denote divergence.

［#44］
<table>
  <tbody>
    <tr>
      <td>Problem Text A:</td>
      <td>John has a fixed amount of money and saves 50 dollars every day in his bank account by 25 days. If it takes 20 days to complete the process, how many money will he he save per day?</td>
    </tr>
    <tr>
      <td>Ground-truth:</td>
      <td>$50 \times 25 \div 20$</td>
    </tr>
    <tr>
      <td>Source Network:</td>
      <td>$50 \times 25 \div 20$</td>
    </tr>
    <tr>
      <td>Sub-network:</td>
      <td>$25 \div 20 \times 50$</td>
    </tr>
    <tr>
      <td>Problem Text B:</td>
      <td>Alice gets the money from the bank's account by 5 days. After that, Alice takes another 3 days to get the money from the same account again. If she takes 65 dollars per day. How much is she takes out from the bank?</td>
    </tr>
    <tr>
      <td>Ground-truth:</td>
      <td>$(5 + 3) \times 65$</td>
    </tr>
    <tr>
      <td>Source Network:</td>
      <td>$(5 + 3) \times 65$</td>
    </tr>
    <tr>
      <td>Sub-network:</td>
      <td>$5 \times 65 + 3 \times 65$</td>
    </tr>
  </tbody>
</table>

［#45］
Table 6. Case study from Math23k to solve spurious correlation.

### 3.2.3. The Sensitivity Analysis of $\alpha$
［#46］
We analyze the sensitivity of $\alpha$ on Math23k. As shown in Table 5, when $\alpha$ is greater than or equal to 0.05, the sub-network will not converge. We obtain the best result when $\alpha = 0.005$.

### 3.2.4. The Sensitivity Analysis of Pruning Rate
［#47］
We analyze the sensitivity of pruning rate on the competitor network. As shown in Table 5, when the pruning rate is greater than 0.2 (the best value), the performance of the sub-network drops quickly.

### 3.2.5. Case Study
［#48］
As an intuitive way to show the performance of SCR, we randomly choose two problems with similar semantic information from the dataset and show its solution expression generated by our model. As shown in Table 6, we observe that our model can produce two different but equivalent solutions for each problem. Specifically, it shows that our model has learned the solutions' equivalence, which implies our model really captures the logic of the original text rather than relying on shallow heuristics. Thus, it can efficiently alleviate spurious correlation of the problems containing similar semantic information (e.g., bank of account, money, per day), which may easily cause the wrong answer like Table 1.

## 4. CONCLUSION
［#49］
In this paper, we proposed a self-consistent reasoning framework to solve MWPs tasks. Our SCR model implicitly corrects spurious correlative samples cooperatively learning a source network and a pruned sub-network. Extensive experiments on two benchmark MWPs datasets demonstrated the effectiveness of our model.

［#50］
In addition, the effect of the learnable quantity embedding we proposed is also significant, which can prevent the model from using the commutative law incorrectly. Experiment results show that our proposal has achieved the most advanced results on the Math23k and Ape210k datasets. At the same time, our method also verified the correctness of the lottery hypothesis in the MWP task. In our future work, we will consider introducing negative samples to the model learning in order to distinguish between equivalent expressions and similar but not equivalent expressions.

## 5. REFERENCES




























