
# Investigating the Lottery Ticket Hypothesis for Variational Quantum Circuits

Michael Kölle \( ^{1} \)  Leonhard Klingert \( ^{1} \)   Julian Schönberger \( ^{1} \)  Philipp Altmann \( ^{1} \)   Tobias Rohe \( ^{1} \) 
Claudia Linnhoff-Popien \( ^{1} \) 

## Abstract

Quantum computing is an emerging field in computer science that has seen considerable progress in recent years, especially in machine learning. By harnessing the principles of quantum physics, it can surpass the limitations of classical algorithms. However, variational quantum circuits (VQCs), which rely on adjustable parameters, often face the barren plateau phenomenon, hindering optimization. The Lottery Ticket Hypothesis (LTH) is a recent concept in classical machine learning that has led to notable improvements in parameter efficiency for neural networks. It states that within a large network, a smaller, more efficient subnetwork, or "winning ticket," can achieve comparable performance, potentially circumventing plateau challenges. In this work, we investigate whether this idea can apply to VQCs. We show that the weak LTH holds for VQCs, revealing winning tickets that retain just 26.0% of the original parameters. For the strong LTH, where a pruning mask is learned without any training, we discovered a winning ticket in a binary VQC, achieving 100% accuracy with only 45% of the weights. These findings indicate that LTH may mitigate barren plateaus by reducing parameter counts while preserving performance, thus enhancing the efficiency of VQCs in quantum machine learning tasks.

## 1. Introduction

Quantum computing (QC) has attracted significant attention due to its potential to surpass classical systems by leveraging superposition and entanglement, enabling solutions to problems once considered intractable (Nielsen & Chuang, 2010; Grumbling & Horowitz, 2019; Sood & Chauhan, 2024). VQCs, as quantum analogues of neural networks, have shown promise in tasks like classification, regression, and optimization (Cerezo et al., 2021a; Du et al., 2022; Schuld & Petruccione, 2021; Qi et al., 2024). Despite this promise, VQCs encounter barren plateaus (BP), where gradients become nearly zero, hindering parameter optimization (McClean et al., 2018; Cerezo et al., 2021b; Grant et al., 2019).

The main objective of this work is to investigate whether the LTH, originally developed for classical neural networks (Frankle & Carbin, 2019), can strengthen VQCs. The LTH posits that within an over-parameterized model, a smaller subnetwork, known as a "winning ticket," can be trained in isolation to match or exceed the original model's performance. One typically obtains a winning ticket by training a network, pruning low-magnitude weights, resetting the remaining weights to their initial values, and retraining (Frankle & Carbin, 2019; Malach et al., 2020). Pruning can be iterative or one-shot, and the strong LTH proposes that pruning alone can sometimes identify winning tickets without any training (Ramanujan et al., 2020; Liu et al., 2024; Cunha et al., 2022).

Identifying winning tickets in VQCs may help mitigate BP issues by reducing circuit complexity while preserving performance. Smaller circuits could potentially exhibit less severe gradient decay, making optimization more feasible. This paper seeks to evaluate whether sparse subcircuits can indeed match the performance of full VQCs, thus offering a path to more efficient quantum ML.

In the remainder of this paper, Section 2 covers foundational ideas and related research in machine learning, pruning, the Lottery Ticket Hypothesis, quantum computing, and the barren plateau phenomenon. Section 3 details our methodology, including experiment design, pruning strategies, datasets, models, metrics, implementation details and hyperparameter optimization. Section 4 then presents and analyzes our experimental findings. Finally, Section 5 summarizes the paper and outlines avenues for future study.
 

## 2. Related Work

Machine learning (ML) often relies on large neural networks (NNs) with millions of parameters to handle complex data, but these expansive models can incur high memory usage and computational overhead. Pruning has thus emerged as a way to remove parameters that exert minimal influence on a model's predictions, retaining performance while reducing resource costs (Blalock et al., 2020).

## 2.1. Pruning in Classical Machine Learning

Pruning typically involves producing a mask, a tensor of the same shape as the NN's weights. Pruned parameters become zeros when multiplied by this mask. Magnitude pruning, which removes weights with small absolute values, is a widely used strategy. Pruning can be unstructured—removing individual weights—or structured—removing entire layers or channels, often leading to more substantial memory and runtime gains (Blalock et al., 2020). In practice, pruning can proceed iteratively, with repeated cycles of training and pruning until a target level of sparsity is reached, or via a one-shot approach, where pruning happens once at a chosen fraction of weights, followed by optional fine-tuning (Frankle & Carbin, 2019; Malach et al., 2020).

## 2.2. Lottery Ticket Hypothesis in Classical Machine Learning

The LTH states that within a randomly initialized NN, there exists a sparse subnetwork (a “winning ticket”) which, when trained in isolation from its original initialization, can achieve performance comparable to or better than the full model (Frankle & Carbin, 2019). In its weak form, a network is trained and pruned, and the remaining weights are reset to their original values before a final retraining phase (Frankle & Carbin, 2019; Malach et al., 2020). The strong variant posits that pruning alone can identify a strong-performing subnetwork without any initial training, as shown by algorithms that learn a mask directly, such as edge-popup (Ramanujan et al., 2020; Malach et al., 2021).

Studies validating the LTH in image classification, natural language processing, and other domains have revealed winning tickets that can retain only a small fraction of the original parameters while preserving or surpassing performance (Frankle & Carbin, 2019; Gale et al., 2019; Liu et al., 2024). The phenomenon has broad implications for model compression, transfer learning, and interpretability, since smaller subnetworks often train faster, adapt more efficiently to new tasks, and expose which parts of a network drive predictions (Cunha et al., 2022; Ferbach et al., 2023).

## 2.3. Barren Plateaus in Variational Quantum Circuits

VQCs have emerged as a promising quantum counterpart to NNs, offering potential computational advantages in classification, regression, and optimization (Cerezo et al., 2021a; Du et al., 2022; Qi et al., 2024). Despite this promise, VQCs can suffer from BPs in their optimization landscapes, where gradients vanish and training fails to converge effectively (McClean et al., 2018). Several factors, such as circuit depth, entanglement, cost function design, and parameter initialization, influence the severity of BPs (Cerezo et al., 2021b; Grant et al., 2019; Cunningham & Zhuang, 2024). Recommended strategies include careful parameter initialization, local cost functions, layer-wise training, gradient clipping, adaptive learning rates, quantum natural gradients, and regularization methods, all of which aim to reduce the likelihood of becoming trapped in flat regions of the parameter space (Cunningham & Zhuang, 2024).

## 2.4. Pruning in Variational Quantum Circuits

To mitigate BPs and improve efficiency, researchers have proposed pruning strategies for VQCs. For instance, Ma et al. (Ma et al., 2024) present a continuous architecture search that prunes gates through a Structure Symmetric Pruning approach, which reduces both parameter counts and circuit depth. Similarly, Sim et al. (Sim et al., 2021) propose Parameter-Efficient Circuit Training (PECT), a technique that updates only a subset of parameters per iteration, thereby lowering runtime. Liu et al. (Kulshrestha et al., 2024) introduce Quantum Adaptive Pruning (QAdaPrune), which adaptively determines pruning thresholds to remove redundant parameters. These methods demonstrate that smaller VQCs can offer competitive performance, often improving trainability when BPs are present.

## 2.5. Applying the LTH to Variational Quantum Circuits

While pruning has found broad application in classical NNs, leveraging LTH-style sparsification in quantum circuits remains largely unexplored. Preliminary successes in pruning VQCs suggest that identifying small, high-performing subcircuits could mitigate BPs by reducing parameter redundancy and easing optimization demands (Sim et al., 2021; Kulshrestha et al., 2024; Ma et al., 2023). Strong and weak variants of the LTH could both be relevant: the former might uncover effective quantum subcircuits without extended training, while the latter would involve train-prune-retrain cycles typical of classical approaches (Frankle & Carbin, 2019; Malach et al., 2020). A successful transfer of the LTH to VQCs holds promise for more efficient quantum machine learning, addressing both computational resource constraints and the challenges of vanishing gradients.
 

## 3. Methodology

This section presents the methodology employed in this work to investigate the LTH in the context of VQCs. We discuss the hypothesis variants and the associated pruning techniques, describe the selected datasets and implemented models, and outline the experimental setup, including the programming languages and libraries used. Finally, we detail the considered hyperparameters and the applied evaluation metrics.

## 3.1. Evaluating the Weak Lottery Ticket Hypothesis

Initially, we evaluated the weak LTH introduced by Frankle and Carbin (2019) (Frankle & Carbin, 2019). We developed a training framework for both NNs and simulated VQCs, which also included a flexible pruning module that can operate in iterative or one-shot pruning modes. Both pruning approaches start with an unpruned model, and each pruning ratio spawns a new model. By setting a manual random seed before creating each model, we ensured identical initial randomized weights. We applied pruning before beginning each training iteration.

## 3.1.1. ITERATIVE PRUNING

For iterative pruning, the model is repeatedly trained, pruned, and reset to evaluate performance at different remaining-weight levels (Frankle & Carbin, 2019; Liu et al., 2024; Malach et al., 2020). Algorithm 1 outlines our iterative pruning algorithm. First, we iterate over a list of random seeds to ensure reproducibility and mitigate seed-specific variations. In each iteration, we initialize a pruning mask that leaves all weights unpruned and set a variable to store the number of remaining weights. In the main loop, we set the random seed, create a new model, apply the pruning mask, and then train and evaluate the model. We subsequently update the pruning mask by pruning 20% of the weights using magnitude pruning. Next, we measure the updated number of remaining weights. The loop terminates if either (1) the remaining weight count does not change due to insufficient weights for further pruning or (2) the number of remaining weights falls below a user-defined threshold that controls pruning depth.

## 3.1.2. ONE-SHOT PRUNING

In the one-shot pruning approach, a set of pruning ratios is defined ahead of time (Frankle & Carbin, 2019; Gale et al., 2019; Liu et al., 2024; Malach et al., 2020). The model is first trained in its unpruned state; then, separate copies are pruned according to each ratio and retrained from scratch, enabling direct performance comparisons across varying remaining-weight levels. Algorithm 2 shows our one-shot pruning algorithm. We begin by iterating over a list of random seeds. For each seed, we create, train, and

Algorithm 1 Weak Lottery Ticket Hypothesis - Iterative Pruning

1: Input: list of random_seeds,  \( RW_{threshold} \)  for remaining weights

2: for random_seed  \( \in \)  random_seeds do

3:  \( PM \leftarrow \)  initial pruning mask

4:  \( RW_{0} \leftarrow -1 \) 

5:  \( i \leftarrow 0 \) 

6: repeat

7: set random_seed

8:  \( M_{i} \leftarrow \)  create new model

9: apply PM on  \( M_{i} \) 

10: train  \( M_{i}

11: evaluate  \( M_{i} \) 

12:  \( i \leftarrow i + 1 \) 

13:  \( PM \leftarrow \)  update PM

14:  \( RW_{i} \leftarrow \)  measure remaining weights of PM

15: until  \( RW_{i} \leq RW_{threshold} \)  or  \( RW_{i}=RW_{i-1} \) 

16: end for

evaluate an initial model  \( M_{0} \) . Then, for each pruning rate, we set the same random seed, create a new model, prune it according to  \( M_{0} \) 's weights, and retrain and evaluate the pruned model.

Algorithm 2 Weak Lottery Ticket Hypothesis - One-shot Pruning

1: Input: list of random_seeds, list of pruning_ratios
2: for random_seed  \( \in \)  random_seeds do
3:     set random_seed
4:     \( M_{0} \leftarrow \)  create initial model
5:     train  \( M_{0} \) 
6:     evaluate  \( M_{0}
7:     i \leftarrow 0
8:     for pruning_ratio  \( \in \)  pruning_ratios do
9:         i \leftarrow i + 1
10:          \( PM_{i} \leftarrow \)  prune  \( M_{0} \)  by pruning_ratio%
11:         set random_seed
12:          \( M_{i} \leftarrow \)  create new model
13:         apply  \( PM_{i} \)  on  \( M_{i} \) 
14:         train  \( M_{i}
15:         evaluate  \( M_{i} \) 
16: end for
17: end for

## 3.2. Evaluating the Strong Lottery Ticket Hypothesis

We then evaluated the strong variant of the LTH by implementing a custom EA (Altmann et al., 2024). This EA leverages the same models as in the weak LTH experiments but bypasses the training step, focusing solely on learning a pruning mask. As before, we set a manual random seed,
 

created an initial model with randomized weights, and based all subsequent models on these same weights. Adhering to the standard structure of an EA(Sloss & Gustafson, 2020), our implementation follows the steps in Algorithm 3: measure model performance, select top-performing models, and apply crossover, mutation, and migration to replenish the population. Although migration is typically used in parallel island setups(Harada & Alba, 2020), we simulated it by introducing entirely new individuals. For the strong LTH, we measured both each individual's accuracy per generation and its number of remaining weights.

Algorithm 3 Strong Lottery Ticket Hypothesis

1: Input: list of random_seeds, number of generations n_gen, number of individuals n_ind

2: for random_seed ∈ random_seeds do

3:     set random_seed

4:     M₀ ← generate initial model with empty pruning mask

5:     individuals ← create n_ind copies of M₀ and mutate their pruning masks

6:     for i_gen ∈ 1..n_gen do

7:         measure performances

8:             individuals ← select 33%

9:             individuals ← crossover (replenish 66% of n_ind)

10:             individuals ← mutation (replenish 95% of n_ind)

11:             individuals ← migration (replenish 100% of n_ind)

12:     end for

13:     measure performances

14: end for

## 3.3. Datasets

We selected the Iris dataset, introduced by Fisher in 1936(Fisher, 1936), and the Wine dataset, introduced by Aeberhard et al. in 1994(Aeberhard et al., 1936), to evaluate VQCs across different levels of complexity while keeping the datasets relatively small. The Iris dataset contains 150 instances of iris flowers with four features and three classes; we also derived a simplified version for binary classification by removing the third class. The Wine dataset has 178 instances, each with 13 properties divided into three classes, and we again created a simplified binary version by excluding the third class. These datasets, commonly used in classical ML and QML research, enabled us to explore the LTH on tasks ranging from relatively straightforward to moderately complex classification.

## 3.4. Models

We implemented three models to evaluate the LTH: two VQCs designed for binary and multi-class classification, and a classical NN baseline with a similar parameter count.

Multi-class VQC The primary model in this work is a multi-class VQC (MVQC) that uses angle encoding to map classical data into the quantum state space. The circuit includes strongly entangled layers to leverage quantum parallelism. We treated data re-uploading as a hyperparameter to reprocess inputs at multiple layers (Section 3.5.2). The number of qubits was determined by the dataset's number of features, using one wire per feature. The circuit measures the Pauli-Z expectation values of the first n qubits (where n is the number of classes) and applies a softmax function to convert these values into class probabilities. We trained the model with cross-entropy loss and the Adam optimizer. See Figure 1a.

Binary VQC For binary classification, we implemented a VQC (BVQC) that outputs the expectation value of a single qubit. We trained it using binary cross-entropy loss with logits and Adam optimization. See Figure 1b.

Simple Classical NN We also implemented a simple NN (SNN) as a baseline, consisting of a single hidden layer with 24 nodes and ReLU activation. The output layer applies softmax to produce class probabilities. We chose its size to match the VQCs' parameter count (see Table 2). It serves as a classical benchmark.

## 3.5. Implementation

In this section, we describe the Python-based implementation, discuss hyperparameter optimization, and outline the metrics used to evaluate results.

## 3.5.1. PROGRAMMING LANGUAGE AND LIBRARIES

All implementations were done in Python. For the ML framework, we used PyTorch(Paszke et al., 2019) for training, evaluation, and pruning. PennyLane(Bergholm et al., 2022) was employed for simulating quantum circuits and integrating them with PyTorch. We used scikit-learn(Pedregosa et al., 2011) for data preprocessing and Seaborn(Waskom, 2021) for visualization. Hyperparameter optimization was conducted with Optuna(Akiba et al., 2019). We used seeds 0 through 9 to ensure reproducible outcomes and reduce the impact of random initialization.

## 3.5.2. HYPERPARAMETER OPTIMIZATION

We performed hyperparameter optimization with Optuna(Akiba et al., 2019) to find an optimal configuration for the unpruned models, assuming a strong initial setup remains effective post-pruning. Table 1 lists the hyperparameters and search ranges, while Table 4 shows the final values selected for each model. The number of layers, data
 
![](./images/1174682106345816107_1.jpg)

![](./images/1174682106345816107_2.jpg)

Figure 1. MVQC and BVQC, consisting of one wire per feature of the dataset. The green box shows angle embedding. The blue box symbolizes one layer, which is repeated several times. One wire per class of the dataset is measured.

re-uploading flag, and uniform range apply only to VQCs. We manually adjusted the number of layers in some cases to ensure over-parameterization (Table 4).

<table><tr><td>Hyperparameter</td><td>Min Value</td><td>Max Value</td></tr><tr><td>Learning Rate</td><td>0.001</td><td>0.3</td></tr><tr><td>Weight Decay</td><td>0.0001</td><td>0.001</td></tr><tr><td>Number of Layers \( ^{*} \)</td><td>8</td><td>16</td></tr><tr><td>Data Re-Uploading</td><td>False</td><td>True</td></tr><tr><td>Uniform Range</td><td>\( -\pi \)</td><td>\( +\pi \)</td></tr></table>

Table 1. Hyperparameters and their minimum/maximum values. *The number of layers was manually adjusted in some models to ensure over-parameterization (Table 4).

## 3.6. Parameter Count

The models differ in parameter counts due to architectural variations among SNN, BVQC, and MVQC. Table 2 lists the parameter counts for each model and dataset.

<table><tr><td>Dataset</td><td>BVQC</td><td>MVQC</td><td>SNN</td></tr><tr><td>Simplified Iris</td><td>122</td><td>184</td><td>170</td></tr><tr><td>Iris</td><td>-</td><td>198</td><td>195</td></tr><tr><td>Simplified Wine</td><td>548</td><td>355</td><td>386</td></tr><tr><td>Wine</td><td>-</td><td>630</td><td>411</td></tr></table>

Table 2. Number of parameters across models and datasets.

## 3.7. Evaluation Metrics

We used accuracy on training and validation sets to gauge performance. For the weak LTH, we tracked accuracy at different levels of remaining weights. Each model/dataset combination has a curve at 100% (unpruned) and at various pruned percentages. Minor rounding errors may arise due to integer parameter counts. For the strong LTH, we also recorded the percentage of remaining weights relative to accuracy per generation. The resulting plots for the weak LTH mirror those in the original study by Frankle and Carbin(Frankle & Carbin, 2019), although the difference in problem size and architecture must be considered.

## 4. Experiments

In this Section, we present and interpret the findings of this work. For the weak LTH, we compare model performance under different pruning techniques and ratios. We use the multi-variate quantum circuit (MVQC) for multi-class tasks, the binary quantum circuit (BVQC) for binary tasks, and a standard neural network (SNN) with a parameter count comparable to the VQCs (see Section 3.4). The datasets include Iris and Wine, along with simplified versions reduced to two classes for binary classification (see Section 3.3). We employ iterative pruning and one-shot pruning (see Sections 3.1 and 3.2) and compare performance both in the context of the weak LTH and strong LTH.

## 4.1. Weak Lottery Ticket Hypothesis

For the weak LTH, we evaluate models across various pruning ratios, comparing fully-connected and pruned states. Table 3 outlines the winning ticket pruning levels for different dataset-model combinations. Additional plots with curves down to 8% remaining weights are available in Section B.

<table><tr><td>Dataset</td><td>BVQC</td><td>MVQC</td><td>SNN</td></tr><tr><td>Simplified Iris</td><td>33.3%</td><td>26.1%</td><td>20.8%</td></tr><tr><td>Iris</td><td>-</td><td>26.0%</td><td>32.1%</td></tr><tr><td>Simplified Wine</td><td>51.3%</td><td>/</td><td>51.4%</td></tr><tr><td>Wine</td><td>-</td><td>32.7%</td><td>21.1%</td></tr></table>

Table 3. Percentage of remaining weights of winning tickets across models and datasets.

## 4.1.1. COMPARISON OF PRUNING TECHNIQUES

As shown in Figure 2, there is no substantial difference between iterative and one-shot pruning. Small discrepancies likely stem from rounding differences in the exact number of remaining weights. Similar observations hold for all other
 
![](./images/1174682106345816107_3.jpg)

(a) Iterative

![](./images/1174682106345816107_4.jpg)

(b) One-Shot

Figure 2. Comparison of pruning techniques on BVQC and simplified Wine.

dataset-model configurations and pruning ratios. We thus focus on iterative pruning in the subsequent results.

## 4.1.2. MODEL PERFORMANCE ON IRIS DATASET

![](./images/1174682106345816107_5.jpg)

(a) MVQC

![](./images/1174682106345816107_6.jpg)

(b) SNN

Figure 3. Accuracies of MVQC and SNN on Iris.

Since the Iris dataset has three classes, the BVQC is not applicable. Figure 3 shows that the MVQC and SNN each reach 97.5% accuracy when unpruned. The MVQC keeps this accuracy down to 26.0% remaining weights but falls below 90% at 10.9% remaining weights. The SNN similarly maintains 97.5% accuracy at 32.1% remaining weights, declining below 90% at about 13.7%.

## 4.1.3. MODEL PERFORMANCE ON SIMPLIFIED IRIS DATASET

![](./images/1174682106345816107_7.jpg)

(a) BVQC

![](./images/1174682106345816107_8.jpg)

(b) MVQC

(c) SNN

Figure 4. Accuracies of BVQC, MVQC, and SNN on simplified Iris.

Figure 4 presents results for the simplified Iris dataset (two classes). All three models reach 100% accuracy in their un-pruned states. The BVQC maintains 100% accuracy down

to 33.3% remaining weights. The MVQC reaches 100% down to 26.1%, although some runs between 21.1% and 13.3% exhibit fluctuations. The SNN also retains 100% accuracy until 20.8% remaining weights.

## 4.1.4. MODEL PERFORMANCE ON WINE DATASET

![](./images/1174682106345816107_9.jpg)

(a) MVQC

![](./images/1174682106345816107_10.jpg)

(b) SNN

Figure 5. Accuracies of MVQC and SNN on Wine.

The Wine dataset contains three classes, so BVQC does not apply. Figure 5 shows the MVQC at 45% accuracy when unpruned, increasing to 80% at 32.7% remaining weights. Even with only 4.3% remaining weights, it surpasses its unpruned accuracy. The SNN starts at 98% accuracy and remains near that level down to 21.1%. Below about 16.9% weights, its performance declines more steeply.

## 4.1.5. MODEL PERFORMANCE ON SIMPLIFIED WINE DATASET

![](./images/1174682106345816107_11.jpg)

(a) BVQC

![](./images/1174682106345816107_12.jpg)

(b) MVQC

(c) SNN

Figure 6. Accuracies of BVQC, MVQC, and SNN on simplified Wine.

Figure 6 illustrates that BVQC and MVQC both stabilize at about 94–96% accuracy up to roughly 40% remaining weights before declining, while SNN remains between 97–98% accuracy until 51.4% remaining weights. Pruning further causes noticeable performance drops, especially below about 20% for the SNN.

## 4.1.6. DISCUSSION OF WEAK LTH

Outcomes are consistent across iterative and one-shot pruning, possibly due to smaller problem sizes and simpler models than in prior research, such as Frankle and Carbin (2019) (Frankle & Carbin, 2019) or Liu et al. (2024) (Liu et al., 2024). Despite this, both VQCs often retain high performance under substantial pruning. For example, the
 

MVQC keeps winning tickets with about 26% of the original weights on both the Iris and simplified Iris datasets. The BVQC and MVQC also maintain moderate performance when pruned for the Wine datasets, although the MVQC shows a more pronounced improvement after pruning on the unreduced Wine dataset, likely due to its highly overparametrized nature. Across all datasets, the SNN shows strong performance and more stable training curves.

Although we do not see extremely low pruning levels (e.g., 3.6%) reported by Frankle and Carbin (2019) (Frankle & Carbin, 2019), these findings support the weak LTH for both classical and quantum networks.

## 4.2. Strong Lottery Ticket Hypothesis

We next evaluate whether any pruned models surpass or match their unpruned counterparts (the strong LTH). Each dataset is assessed under an evolutionary algorithm (EA) approach. Plots compare average accuracies and remaining weights of the best individual per generation to weak LTH baselines at 100% and a comparable remaining weight level.

## 4.2.1. MODEL PERFORMANCE ON IRIS DATASET

![](./images/1174682106345816107_13.jpg)

(a) MVQC

![](./images/1174682106345816107_14.jpg)

(b) SNN

Figure 7. Accuracies and remaining weights of best individual per generation of MVQC and SNN on Iris. Dashed lines show weak LTH results at specified remaining weights.

As shown in Figure 7, the MVQC improves from 68% to 83% through 20 generations, failing to reach the 97.5% unpruned accuracy. The SNN similarly goes from 65% to 82%, also below 97.5%. Both stabilize around 39–47% remaining weights, yet models at similar pruning ratios under the weak LTH were more accurate.

## 4.2.2. MODEL PERFORMANCE ON SIMPLIFIED IRIS DATASET

In Figure 8, the BVQC starts near 96% and reaches 100% after about 15 generations. Its final pruning ratio is about 48%, comparable to a winning ticket under iterative pruning. The MVQC sees a notable gain from 75% to 95% accuracy at about 34% remaining weights, but it still lags behind the unpruned MVQC's 100%. The SNN quickly reaches 100% with about 40% of the original weights, matching its

![](./images/1174682106345816107_15.jpg)

(a) BVQC

![](./images/1174682106345816107_16.jpg)

(b) MVQC

(c) SNN

Figure 8. Accuracies and remaining weights of best individual per generation on simplified Iris. Dashed lines show weak LTH results at specified remaining weights.

unpruned accuracy.

## 4.2.3. MODEL PERFORMANCE ON WINE DATASET

![](./images/1174682106345816107_17.jpg)

(a) MVQC

![](./images/1174682106345816107_18.jpg)

(b) SNN

Figure 9. Accuracies and remaining weights of best individual per generation of MVQC and SNN on Wine. Dashed lines show weak LTH results at specified remaining weights.

Figure 9 shows that the MVQC reaches about 48% accuracy and the SNN about 87%, both falling short of their unpruned versions under the weak LTH (45% vs. 80% for MVQC; 87% vs. 98% for SNN). The remaining weights hover around 30–40%, but this ratio does not reliably indicate the EA's success, as subnetwork size is not a direct proxy for accuracy. Further exploration is needed to better understand the relationship between pruning, subnetwork size, and model performance.

## 4.2.4. MODEL PERFORMANCE ON SIMPLIFIED WINE DATASET

![](./images/1174682106345816107_19.jpg)

(a) BVQC

![](./images/1174682106345816107_20.jpg)

(b) MVQC

(c) SNN

Figure 10. Accuracies and remaining weights of best individual per generation on simplified Wine. Dashed lines show weak LTH results at specified remaining weights.

Figure 10 reveals that the BVQC, MVQC, and SNN remain
 

well below their unpruned accuracies (94% BVQC, 97% MVQC, and 98% SNN under the weak LTH). The best EA solutions stabilize around 44% weights for the SNN and 26–37% for the quantum models, indicating the EA could not identify subnetworks that match or exceed the full models.

## 4.2.5. DISCUSSION OF STRONG LTH

On the simplified Iris dataset, the EA discovers winning tickets that match unpruned accuracies for the BVQC and SNN; the MVQC nearly achieves its unpruned performance. This suggests the EA can find effective masks for VQCs in simpler tasks. However, for the more complex or higher-dimensional Wine dataset, no winning tickets emerge. The EA yields subnetworks with 30–45% of the weights but fails to reach unpruned accuracy. Different optimization techniques or more flexible evolutionary strategies might uncover stronger subnetworks for more challenging tasks.

## 5. Conclusion

In this work, we applied the LTH to VQCs to investigate whether winning tickets can emerge, similar to those in classical NNs. We employed iterative and one-shot pruning to evaluate the weak LTH, as well as an EA to assess the strong LTH. We used two datasets, Iris and Wine, both with simplified variants for binary classification. Additionally, we tested two VQCs—one designed for multi-class classification and another for binary classification—and a simple NN as a baseline.

The experiments show that both the BVQC and MVQC achieved winning tickets with as few as 33.3% and 26.0% of remaining weights, respectively, using iterative pruning on the Iris and simplified Iris datasets. On these datasets, the MVQC performed comparably to an NN with a similar parameter count. On larger problem sizes, such as the simplified Wine dataset, both models had difficulty finding winning tickets; however, the BVQC still discovered one with 51.3% remaining weights. In contrast, no winning ticket was found for the MVQC on the simplified Wine dataset, whereas the SNN, a relatively small NN with a parameter count comparable to that of the MVQC, still exhibited winning tickets. Nonetheless, on the unreduced Wine dataset, the MVQC illustrated that the LTH can overcome BPs in VQCs, yielding a 35% performance improvement and reaching 80% accuracy when pruned to 32.7% of the original weights, whereas the unpruned model achieved only 45%.

These findings indicate that the weak LTH is applicable to VQCs, and that the identified winning tickets can compete with those in NNs of similar size if the problem size is small enough. They also suggest that BPs can be mitigated if the initially unpruned model is sufficiently overparameterized. Interestingly, we observed that iterative and one-shot pruning methods identified the same subnetworks and subcircuits, potentially due to the relatively small size and complexity of both the datasets and the models. This factor makes it challenging to compare these results with other studies, which generally involve larger datasets and models.

Regarding the strong LTH, we found a winning ticket for the BVQC on the simplified Iris dataset, retaining 100% accuracy with 45% of the weights remaining. We also observed a substantial 20% improvement—from 75% to 95% accuracy—for the MVQC on the simplified Iris dataset with 40% remaining weights. Although this does not match the MVQC’s 100% accuracy at full capacity, it remains a noteworthy result.

Future work could investigate the LTH on real quantum hardware and extend it to larger circuits and problem sizes. It might also explore pruning gates rather than just weights, following concepts like Structure Symmetric Pruning(Ma et al., 2024). As part of addressing BPs, further studies could focus on scenarios in which BPs are more pronounced and examine how strongly pruned VQCs might overcome them.

## Impact Statement

This paper advances our understanding of how the LTH may enhance efficiency in quantum machine learning. By showing that VQCs can be pruned to sparse “winning tickets” without compromising performance, our work highlights a pathway to reduce computational and resource overhead in emerging quantum technologies. A more resource-efficient approach to VQCs could lower energy consumption and hardware demands, fostering broader accessibility to quantum methods. Additionally, it may accelerate real-world applications of quantum machine learning in fields ranging from drug discovery to secure communications. We do not identify direct ethical or societal risks at this stage, but continued research should remain attentive to potential long-term impacts, such as shifts in computational power and its implications for data security and privacy.

## References

Aeberhard, S., Coomans, D., and de Vel, O. Wine data set. https://archive.ics.uci.edu/ml/datasets/wine, 1936. UCI Machine Learning Repository.

Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &
 

Data Mining, pp. 2623–2631. ACM, 2019. doi: 10.1145/3292500.3330701.

Altmann, P., Schönberger, J., Zorn, M., and Gabor, T. Finding strong lottery ticket networks with genetic algorithms. In Proceedings of the 16th International Joint Conference on Computational Intelligence - Volume 1: NCTA, pp. 449–460. INSTICC, SciTePress, 2024. ISBN 978-989-758-721-4. doi: 10.5220/0013010300003837.

Bergholm, V., Izaac, J., Schuld, M., and Gogolin, C. e. a. Pennylane: Automatic differentiation of hybrid quantum-classical computations, 2022. URL https://arxiv.org/abs/1811.04968.

Blalock, D., Ortiz, J. J. G., Frankle, J., and Guttag, J. What is the state of neural network pruning?, 2020. URL https://arxiv.org/abs/2003.03033.

Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., and McClean, J. R. e. a. Variational quantum algorithms. Nature Reviews Physics, 3:625–644, 2021a. doi: 10.1038/s42254-021-00348-9.

Cerezo, M., Sone, A., Volkoff, T., Cincio, L., and Coles, P. J. Cost function dependent barren plateaus in shallow parametrized quantum circuits. Nature Communications, 12(1):1791, 2021b. doi: 10.1038/s41467-021-21728-w.

Cunha, A., Natale, E., and Viennot, L. Proving the lottery ticket hypothesis for convolutional neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=Vjki79-619-.

Cunningham, J. and Zhuang, J. Investigating and mitigating barren plateaus in variational quantum circuits: A survey. arXiv preprint arXiv:2407.17706, 2024.

Du, Y., Huang, T., You, S., Hsieh, M.-H., and Tao, D. Quantum circuit architecture search for variational quantum algorithms. npj Quantum Information, 8(1):62, 2022. doi:10.1038/s41534-022-00570-y.

Ferbach, D., Tsirigotis, C., Gidel, G., and Bose, A. A general framework for proving the equivariant strong lottery ticket hypothesis, 2023. URL https://arxiv.org/abs/2206.04270.

Fisher, R. A. Iris data set. https://archive.ics.uci.edu/ml/datasets/iris, 1936. UCI Machine Learning Repository.

Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJl-b3RcF7.

Gale, T., Elsen, E., and Hooker, S. The state of sparsity in deep neural networks, 2019. URL https://arxiv.org/abs/1902.09574.

Grant, E., Wossnig, L., Ostaszewski, M., and Benedetti, M. An initialization strategy for addressing barren plateaus in parametrized quantum circuits. Quantum, 3:214, 2019. doi: 10.22331/q-2019-12-09-214.

Grumbling, E. and Horowitz, M. (eds.). Quantum Computing: Progress and Prospects. The National Academies Press, Washington, DC, 2019. ISBN 978-0-309-47969-1. doi: 10.17226/25196. URL https://nap.nationalacademies.org/catalog/25196/quantum-computing-progress-and-prospects.

Harada, T. and Alba, E. Parallel genetic algorithms: A useful survey. ACM Comput. Surv., 53(4), August 2020. ISSN 0360-0300. doi: 10.1145/3400031. URL https://doi.org/10.1145/3400031.

Kulshrestha, A., Liu, X., Ushijima-Mwesigwa, H., Bach, B., and Safro, I. Qadaprune: Adaptive parameter pruning for training variational quantum circuits, 2024. URL https://arxiv.org/abs/2408.13352.

Liu, B., Zhang, Z., He, P., Wang, Z., Xiao, Y., Ye, R., Zhou, Y., Ku, W.-S., and Hui, B. A survey of lottery ticket hypothesis, 2024. URL https://arxiv.org/abs/2403.04861.

Ma, Q., Hao, C., Yang, X., Qian, L., Zhang, H., Si, N., Xu, M., and Qu, D. Continuous evolution for efficient quantum architecture search. EPJ Quantum Technology, 11(1):54, 2024. ISSN 2196-0763. doi: 10.1140/epjqt/s40507-024-00265-7. URL https://doi.org/10.1140/epjqt/s40507-024-00265-7.

Malach, E., Yehudai, G., Shalev-Shwartz, S., and Shamir, O. Proving the lottery ticket hypothesis: Pruning is all you need, 2020. URL https://arxiv.org/abs/2002.00585.

McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., and Neven, H. Barren plateaus in quantum neural network training landscapes. Nature Communications, 9(1):4812, 2018. doi: 10.1038/s41467-018-07090-4.

Nielsen, M. A. and Chuang, I. L. Quantum Computation and Quantum Information: 10th Anniversary Edition. Cambridge University Press, 2010. doi: 10.1017/CBO9780511976667.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., and Chanan, G. e. a. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pp. 8024–8035. Curran Associates, Inc., 2019.
 

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., and Thirion, B. e. a. Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12:2825–2830, 2011.

Qi, H., Xiao, S., Liu, Z., Gong, C., and Gani, A. Variational quantum algorithms: fundamental concepts, applications and challenges. Quantum Information Processing, 23(6):224, 2024. doi: 10.1007/s11128-024-04438-2.

Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi, A., and Rastegari, M. What's hidden in a randomly weighted neural network?, 2020. URL https://arxiv.org/abs/1911.13299.

Schuld, M. and Petruccione, F. Variational Circuits as Machine Learning Models, pp. 177–215. Springer International Publishing, Cham, 2021. ISBN 978-3-030-83098-4. doi: 10.1007/978-3-030-83098-4.5. URL https://doi.org/10.1007/978-3-030-83098-4_5.

Sim, S., Romero, J., Gonthier, J. F., and Kunitsa, A. A. Adaptive pruning-based optimization of parameterized quantum circuits. Quantum Science and Technology, 6(2):025019, 2021. doi: 10.1088/2058-9565/abe107. URL https://dx.doi.org/10.1088/2058-9565/abe107.

Sloss, A. N. and Gustafson, S. 2019 Evolutionary Algorithms Review, pp. 307–344. Springer International Publishing, 2020. doi: 10.1007/978-3-030-39958-0_16. URL https://doi.org/10.1007/978-3-030-39958-0_16.

Sood, V. and Chauhan, R. P. Archives of quantum computing: Research progress and challenges. Archives of Computational Methods in Engineering, 31(1):73–91, 2024. ISSN 1886-1784. doi: 10.1007/s11831-023-09973-2. URL https://doi.org/10.1007/s11831-023-09973-2.

Waskom, M. L. Seaborn: Statistical data visualization. Journal of Open Source Software, 6(60):3021, 2021. doi:10.21105/joss.03021.
 

## A. Hyperparameter Search Results

Table 4 shows the results of the hyperparameter search (see Section 3.5.2) and thus the final hyperparameter values for the models used in the experiments. The hyperparameters learning rate and weight decay have been used for all three models, while number of layer, data re-uploading and uniform range have been used for the VQCs only. The number of layers have been manually adjusted for the MVQC on the unreduced Iris and Wine datasets and for the BVQC on the simplified Iris dataset to ensure over-parametrization.

<table><tr><td>Dataset</td><td>Model</td><td>Learning Rate</td><td>Weight Decay</td><td>L^{1}</td><td>DRU^{2}</td><td>Uniform Range^{3}</td></tr><tr><td>S. Iris</td><td>BVQC</td><td>0.0061690543775444456</td><td>0.00011451748647630793</td><td>10^{*}</td><td>False</td><td>1.78340734641020670</td></tr><tr><td>S. Iris</td><td>MVQC</td><td>0.1404603283295513300</td><td>0.00020043419481312650</td><td>15</td><td>False</td><td>0.34099999999999997</td></tr><tr><td>S. Iris</td><td>SNN</td><td>0.0189753941329335250</td><td>0.00037958686849631810</td><td>-</td><td>-</td><td></td></tr><tr><td>Iris</td><td>MVQC</td><td>0.0276674656133278770</td><td>0.00014188599748059832</td><td>16^{*}</td><td>False</td><td>0.99700000000000000</td></tr><tr><td>Iris</td><td>SNN</td><td>0.0412876064499254560</td><td>0.00011458294311477400</td><td>-</td><td>-</td><td></td></tr><tr><td>S. Wine</td><td>BVQC</td><td>0.0341163513021525840</td><td>0.00048832766322303640</td><td>14</td><td>False</td><td>0.76659265358979310</td></tr><tr><td>S. Wine</td><td>MVQC</td><td>0.0469360377064666600</td><td>0.00017400041959447874</td><td>9</td><td>False</td><td>0.13840734641020713</td></tr><tr><td>S. Wine</td><td>SNN</td><td>0.0012576386169755418</td><td>0.00077375502517089950</td><td>-</td><td>-</td><td></td></tr><tr><td>Wine</td><td>MVQC</td><td>0.0562921735356738800</td><td>0.00033968499286871637</td><td>16^{*}</td><td>False</td><td>0.35300000000000000</td></tr><tr><td>Wine</td><td>SNN</td><td>0.0574452822141239800</td><td>0.00010743993876395757</td><td>-</td><td>-</td><td></td></tr></table>

Table 4. Hyperparameters values across models and datasets. *The number of layers have been manually adjusted for some models to ensure over-parametrization.

## B. Weak Lottery Ticket Hypothesis - Full Plots

The following figures show the results of the weak LTH utilizing iterative pruning across all models and dataset in an unselected version, with the curves for all levels of remaining weights above 8%.

![](./images/1174682106345816107_21.jpg)

![](./images/1174682106345816107_22.jpg)

![](./images/1174682106345816107_23.jpg)

Figure 11. Accuracies of MVQC & SNN on Iris, all levels of remaining weights down to 8%.
 
![](./images/1174682106345816107_24.jpg)

Remaining Weights (%)

![](./images/1174682106345816107_25.jpg)

![](./images/1174682106345816107_26.jpg)

(a) BVQC

Remaining Weights (%)

![](./images/1174682106345816107_27.jpg)

(b) MVQC

![](./images/1174682106345816107_28.jpg)

(c) SNN

Figure 12. Accuracies of BVQC, MVQC & SNN on simplified Iris, all levels of remaining weights down to 8%.

![](./images/1174682106345816107_29.jpg)

![](./images/1174682106345816107_30.jpg)

![](./images/1174682106345816107_31.jpg)

(a) MVQC

![](./images/1174682106345816107_32.jpg)

(b) SNN

Figure 13. Accuracies of MVQC & SNN on Wine, all levels of remaining weights down to 8%.

![](./images/1174682106345816107_33.jpg)

(a) BVQC

![](./images/1174682106345816107_34.jpg)

![](./images/1174682106345816107_35.jpg)

(b) MVQC

(c) SNN

Figure 14. Accuracies of BVQC, MVQC & SNN on simplified Wine, all levels of remaining weights down to 8%.
 
