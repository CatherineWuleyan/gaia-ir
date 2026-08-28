# Bayesian Lottery Ticket Hypothesis

Nicholas Kuhn $^1$ Arvid Weyrauch $^1$ Lars Heyen $^1$ Achim Streit $^1$ Markus Götz $^{1,2}$ Charlotte Debus $^1$

$^1$Scientific Computing Center (SCC), Karlsruhe Institute of Technology (KIT), Karlsruhe, Germany
$^2$Helmholtz AI, Eggenstein-Leopoldshafen, Germany

## Abstract

Bayesian neural networks (BNNs) are a useful tool for uncertainty quantification, but require substantially more computational resources than conventional neural networks. For non-Bayesian networks, the Lottery Ticket Hypothesis (LTH) posits the existence of sparse subnetworks that can train to the same or even surpassing accuracy as the original dense network. Such sparse networks can lower the demand for computational resources at inference, and during training. The existence of the LTH and corresponding sparse subnetworks in BNNs could motivate the development of sparse training algorithms and provide valuable insights into the underlying training process. Towards this end, we translate the LTH experiments to a Bayesian setting using common computer vision models. We investigate the defining characteristics of Bayesian lottery tickets, and extend our study towards a transplantation method connecting BNNs with deterministic Lottery Tickets. We generally find that the LTH holds in BNNs, and winning tickets of matching and surpassing accuracy are present independent of model size, with degradation at very high sparsities. However, the pruning strategy should rely primarily on magnitude, secondly on standard deviation. Furthermore, our results demonstrate that models rely on mask structure and weight initialization to varying degrees.

## 1 MOTIVATION

Bayesian neural networks (BNN) are a powerful, yet complex tool for uncertainty quantification (UQ) in neural networks. Conventional neural network (NN) predictions lack confidence estimations that directly map to the prediction uncertainty, e.g., when outputting class prediction probabilities. To overcome this Bayesian principles can be leveraged to promote the model weights from fixed values to distributions, thereby facilitating model deployment in real-world safety-critical applications [Chib and Singh, 2023]. One approach towards BNNs is mean-field variational inference (VI) [Saul et al., 1996, Blundell et al., 2015], which approximates the true distribution of the model weights with a parameterized variational distribution. During model training, the parameters of this distribution are optimized such that the output distribution of the model matches the distribution of the training data. In addition to UQ, VI has other interesting features: it can increase a model's generalization capability and decrease overfitting, making BNNs a better choice for small datasets [Izmailov et al., 2021].

These advantages come with the downside of an increased computational demand: For one, describing the weights with a parameterized distribution increases the number of free model parameters substantially, thus augmenting model size and corresponding memory demand. Further, the distributional nature of the model formulation increases the number of computational operations, i.e., FLOPs. The cost for forward and backward passes due to evaluating multiple samples multiplies the training and evaluation cost beyond that of a deterministic NN. Both the per-iteration computations and the number of samples needed for stable predictions grow with the number of parameters and with the complexity of the posterior. Consequently, large-scale Bayesian models are very challenging to train on consumer hardware [Graves, 2011, Hoffman et al., 2013].

In conventional non-Bayesian NN sparsity has emerged as an effective approach to reducing computational load and memory footprint [Hoefler et al., 2021], given the fact that most parameters of a trained model are close to zero. From a technical point of view, sparsity in DNNs means that the weight matrices representing a layer carry a large number of zeros, i.e. they are sparse matrices. Only non-zero parameters contribute to a layers output, and zero parameters can therefore be removed from the matrix. This is called prun-

ing, and is usually done after training. A retraining recovers predictive performance lost during pruning. Pruning BNNs is generally feasible to reduce the computational demand for inference deployment, and has been explored in previous works [Blundell et al., 2015]. However, it does not allevi- ate the far greater problem of computational load for BNN model training, for which a priori knowledge is required on which weights can be removed from the network.

Dynamic pruning methods for training conventional NNs in a sparse manner have been proposed, all of which are founded in the observation of the Lottery Ticket Hypothe- sis (LTH) [Frankle and Carbin, 2019]. LTH postulates the existence of subnetworks that train to the same accuracy (matching) as the full network. These so-called Lottery Tick- ets (LTs) combine the efficiency of a sparse network with the accuracy of a dense one. They are found through Itera- tive Magnitude Pruning (IMP), in which models undergo a train-prune-reset cycle to find the pair of initial weights and pruning mask.

To evaluate whether such algorithms for training sparse BNNs are viable, this study aims to investigate whether the LTH also holds in BNNs. This question has already gar- nered interest in the community Marsh et al. [2025]. If a performant sparse subnetwork also exists in a BNN, both the number of model parameters and computational opera- tions could be reduced, making each training and evaluation step computationally cheaper. Performing posterior infer- ence in the forward pass in a lower-dimensional parameter space can also improve mixing and convergence of Markov Chain Monte Carlo (MCMC) [Neal, 2011] and variational methods [Izmailov et al., 2021]. As a result, one can achieve an equal prediction quality with fewer weight samples, and each sample is less expensive to obtain, enabling the practi- cal training of sparse, large-scale BNNs.

We start by translating the original LTH experiment into a Bayesian setting. Naïvely, there is no reason to assume that LTs cannot be found in BNNs. However, the concrete man- ifestation may differ from the deterministic case. The role of initialization may be less straightforward: since Bayesian weights are drawn from distributions rather than fixed val- ues, the stability and reproducibility may be influenced by the choice of prior or the variance of the variational posterior. Architectural differences may additionally play a role, with convolutional BNNs possibly exhibiting more structured sparsity on the layer-level, while attention-based BNNs may show pruning patterns related to their attention-Multilayer Perceptron (MLP) stack architecture.

To uncover these potential differences, we study LTH in BNNs by applying IMP with different pruning strategies in a set of Bayesian-turned computer vision networks, in- cluding both convolutional and attention-based models, and compare them to their non-Bayesian counterparts. Further, we investigate the characteristics that define a Bayesian LT, by analyzing the sparsity patterns on a global and per- layer level. Motivated by the computation cost associated with obtaining Bayesian LTs, we investigate a transplanta- tional strategy in which non-Bayesian LTs are transferred into BNNs. Since training via VI introduces an additional computational burden, delaying VI-based optimization to later stages of the iterative pruning offers the potential to reduce overall compute requirements. We therefore exam- ine whether such transplantation can preserve the perfor- mance advantages observed for Bayesian LTs compared to randomly pruned BNNs, while improving computational efficiency and maintaining the calibration benefits of BNNs. We provide ablational studies on model size, learning rate schedule and learning rate rewinding Renda et al. [2020].

## 2 RELATED WORK

Sparsity Sparsity arises naturally in overparametrized net- works, which are often trained with far more weights than are strictly necessary to represent the target function [Han et al., 2016]. By removing this redundancy, sparse mod- els can achieve benefits such as reduced memory footprint, faster inference, and potentially improved generalization. A wide variety of techniques have been developed to in- duce sparsity in NNs, like explicit regularization [Shen et al., 2024] to encourage weights to shrink toward zero, or low-rank factorization [Sui et al., 2024] and sparse con- volutions [Elsen et al., 2019]. Pruning has emerged as one of the most widely studied and practically successful strategies [Hoefler et al., 2021]. Pruning techiques start from a dense, trained network and progressively remove pa- rameters deemed unimportant according to some criterion. The Lottery Ticket Hypothesis marks a special case of pruning by demonstrating the existence of well-performing sparse networks, without the need for a dense training phase commonly used in pruning schemes. Through pruning, a mask for the specific initialization of the weights is found, the mask and initialization form a subnetwork which can match the performance of the dense network. The LTH has been widely tested on various architectures and task se- tups [Liu et al., 2024]. Most notably, winning tickets have been demonstrated to exist in linear [Frankle and Carbin, 2019], convolutional [Frankle and Carbin, 2019], LSTM [Yu et al., 2020], attention [Chen et al., 2021b], and graph [Chen et al., 2021a] layers, as well as in supervised, unsuper- vised, and reinforcement learning problems [Frankle and Carbin, 2019, Itahara et al., 2020, Yu et al., 2020]. Addition- ally, research has extended to create even better performing LTs [Renda et al., 2020] in bigger models [Frankle et al., 2020], or with less compute needed [Wang et al., 2020, Lee et al., 2019] by pruning before training.

Bayesian Neural Networks BNNs add an additional layer of complexity to the application of deep learning, by pro- moting the model weights from fixed values to distribu-

tions. A common approach to train BNNs is **Variational Inference (VI)**, which approximates the true distribution of the model weights with a parameterized variational distribution [Graves, 2011, Hoffman et al., 2013]. Early VI for neural networks used a fully factorized Gaussian posterior, because of its low memory cost. Mean-field VI, with reparametrization-based approaches [Blundell et al., 2015, Kingma et al., 2015], make training large BNNs tractable, but are known to underestimate posterior variance and struggle with multimodality induced by parameter symmetries. A parallel line of work uses stochastic-gradient MCMC [Nemeth and Fearnhead, 2019, Wu et al., 2024, Kim et al., 2024] and Laplace and second-order approximations [Faller and Martin, 2025], which trade the computational cost for better posterior fidelity.

The prior is a major lever for inducing sparsity in BNNs. The spike-and-slab prior, a hierarchical two-component mixture prior, explicitly encodes that some weights should be exactly zero [Jantre et al., 2025], alternative sparsity-promoting priors include the horseshoe prior, log-uniform or heavy-tailed priors [Ghosh and Doshi-Velez, 2017, Carvalho et al., 2009, Li et al., 2024], which encourage many small weights and few large ones. Work on Bayesian compression combines these priors with structured pruning schedules to achieve highly sparse predictive models [Chérief-Abdellatif, 2019]. Other, more general pruning work has focused on posterior pruning [Blundell et al., 2015, Mathew and Rowe, 2023], and uncertainty-aware pruning criteria [Ko et al., 2019].

## 3 BACKGROUND

### Bayesian Neural Networks
The Bayesian approach to neural networks addresses the lack of epistemic uncertainty representation by placing a prior distribution over the network weights, thereby treating the weights as probabilistic distributions rather than deterministic parameters. A common choice is to model each weight $w_i$ as a Gaussian distribution:

$$
w_i \sim \mathcal{N}(\mu_i, \sigma_i^2).
$$

However, computing the posterior distribution $p(\mathbf{w} \mid \mathcal{D})$ over the weights given data $\mathcal{D}$ is generally intractable due to the high-dimensional and nonlinear nature of neural networks. **Variational inference** offers a tractable approximation by positing a simpler variational distribution $q(\mathbf{w} \mid \theta)$, parameterized by $\theta$, and minimizing the Kullback-Leibler (KL) divergence between the variational posterior and the true posterior.

This is typically done by maximizing the evidence lower bound (ELBO), given by:

$$
\mathcal{L}(\theta) = \mathbb{E}_{q(\mathbf{w}|\theta)} [\log p(\mathcal{D} \mid \mathbf{w})] - \text{KL}(q(\mathbf{w} \mid \theta) \parallel p(\mathbf{w})).
$$

The first term encourages the variational weights to explain the observed data, while the second term regularizes the variational distribution to stay close to the prior. The quality of prediction uncertainty can be assessed through the Mean Absolute Calibration Error (MACE), given by:

$$
\text{MACE} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{|S_b|} \left| \sum_{i \in S_b} \left( \mathbb{I}(\hat{y}_i = y_i) - \hat{p}_i \right) \right|
$$

which is a measure for the discrepancy between predicted confidence and empirical accuracy.

Advances in variational inference, such as Bayes by Backprop [Blundell et al., 2015] and local reparameterization tricks [Kingma et al., 2015], have made it feasible to apply Bayesian methods to large-scale models like ResNet [He et al., 2015], enabling uncertainty-aware deep learning at scale.

### Iterative Magnitude Pruning
IMP as proposed in the Lottery Ticket Hypothesis, is an algorithm where a network is first trained for a number of epochs to obtain a set of learned weights $\mathbf{w}$. Then, weights are given a score, defined by $s_m = |w|$, and the lowest-scoring weights are pruned. Pruning is implemented by constructing a binary mask $\mathbf{m} \in \{0,1\}^d$, where $d$ is the number of parameters, such that:

$$
\mathbf{w}_{\text{pruned}} = \mathbf{m} \odot \mathbf{w},
$$

with $\odot$ denoting element-wise product. The non-zero entries of $\mathbf{m}$ correspond to the "surviving" weights.

Importantly, the remaining weights are then reset to their original initialization values, denoted $\mathbf{w}_0$. The resulting winning ticket is defined by the pair $\mathbf{w}_{\text{ticket},N} = (\mathbf{m}_N, \mathbf{w}_0)$; the train-prune cycle is repeated $N$ times, achieving higher sparsities at each level, building up a very sparse winning ticket. Empirical results show that both components, sparsity pattern $\mathbf{m}$ and initialization $\mathbf{w}_0$, are crucial [Frankle and Carbin, 2019]. Further, some advancements on the study of the LTH have been made, e.g., LTs have been found in other architectures and settings [Chen et al., 2021a,b], but the time-consuming iterative process is still necessary.

## 4 EXPERIMENTAL SETUP

The objective of our experiments is to translate the IMP experiment of Frankle and Carbin [2019] to a Bayesian setting. For that, we implement and train BNNs using mean-field VI, apply pruning in various forms, reset and re-train them to achieve the same training structure as in the LTH. We then compare the resulting predictive performance to the classical non-Bayesian counterparts that underwent IMP following the same procedure.

### Dataset and Models
For our evaluation, we chose the classical computer vision models ResNet-18 [He et al., 2015], VGG [Simonyan and Zisserman, 2015], and VisionTransformer [Dosovitskiy et al., 2021], which are all

trained on the task of image classification on the CI-FAR10 [Krizhevsky et al., 2009] dataset, which comprises 50k training and 10k testing images from ten different classes. Next to the standard non-Bayesian versions of these models, which serve as a baseline, we implement and train Bayesian variants using mean-field VI. The common random horizontal flip and cropping data augmentations are applied to the training images [Frankle and Carbin, 2019]. This makes the model and dataset choice equal to the original LTH experiment, with the addition of the VisionTransformer model.

Implementation of Bayesian models Every linear and convolutional layer in the models is replaced by its Bayesian counterpart. Means are initialized according to the Kaiming uniform initialization, and standard deviations are initialized to 1% of the Kaiming scale. We forward each batch ten times with differently sampled weights. The final predictions are averaged over the output samples. We apply a temperature scaling with $T=0.1$ to the KL loss to inhibit posterior collapse [Wenzel et al., 2020].

Training Curriculum For every run, non-Bayesian and Bayesian, we use the following shared training parameters. Everything non-Bayesian-related and not explicitly stated is kept constant, i.e. only additional parameters introduced in a Bayesian variant are changed. The models are trained for 160 epochs, using the ADAM optimizer with a learning rate of 0.001 and a learning rate schedule to drop the learning rate by 0.1 at epochs 80 and 120. The weight decay is set to 0.0001. The maximum achieved test accuracy and MACE is recorded. After training, all convolutional and linear layers except the final classification layer are globally pruned with a pruning rate of 0.2. Resetting the weights to their original initialization, we repeat this training and prune for 20 levels of sparsity. In the final level, only $\sim1.2\%$ of the total parameters are non-zero.

Hardware and Software All experiments were run on a compute node equipped with 12 Intel Xeon Platinum 8368 CPUs and four NVIDIA A100 or H100 GPUs as accelerators with 40GB and 96GB VRAM, respectively, connected via NVLink. All experiments used Python 3.11.7 with CUDA-enabled PyTorch 2.4.0 [Paszke et al., 2019]. We use torch-blue Weyrauch et al. [2026] for the implementation of Bayesian models. The source code for our experiments is publicly available¹.

## 5 PRUNING IN BNNS

Pruning in BNNs differs in the scoring function used from pruning in deterministic networks. In the original experiments for the LTH, the pruning function $s_m$, defined above, was used. In BNNs, pruning decisions can account for both the mean $\mu$ and the uncertainty $\sigma$ associated with each weight. When pruned, a weight has $(\mu,\sigma)=(0,0)$.

For our first experiment, we apply three different kinds of unstructured pruning approaches to the Bayesian models. The signal-to-noise score assigns a value of $s_{S N R}=\frac{|\mu|}{\sigma}$ to every weight $w$ with parameters $(\mu,\sigma)$, and preferably prunes weights that are close to zero and "noisy", e.g., have a high standard deviation [Blundell et al., 2015]. The squared-sum (Square) score is defined by $s_{s s}=\sqrt{\mu^2+\sigma^2}$. Similar to $s_{S N R}$, weights with low $\mu$ are pruned, but instead, weights with a low standard deviation are preferred, intuitively corresponding to weights where the network is "sure" they should be close to zero. The mu-magnitude $(\mu)$ score, defined by $s_{m m}=|\mu|$ disregards the standard deviation. It is motivated by the fact that there is no intuition a priori about the value of $\sigma$ besides the prior matching, and pruning should therefore be independent of its value. The baseline non-Bayesian model is achieved by pruning based on $s_m$, keeping all other variables equal.

Figure 1 shows the recorded accuracy over the percentage of retained weights. Pruning ResNet18, we find a similar behaviour in the baseline as for $s_{S N R}$ and $s_{m m}$, with a gradual accuracy degradation over increasing sparsity. In the high sparsity regime $(>98\%)$, performance degrades quickly after pruning with $s_{s s}$ and in the baseline. Pruning VGG11, we find again a similar behaviour in the Bayesian versus the non-Bayesian model, with a decline in performance in the high-sparsity regime, but below $90\%$ sparsity, all models perform equally well as their dense counterparts. There is no discernible difference between the different pruning scoring functions. Pruning ViT-tiny, i.e. the attention-based model, the Bayesian variants outperform the non-bayesian model, and $s_{S N R}$ and $s_{\mu}$ exhibit an increase in performance pruning to $50\%$ sparsity, and only then a decline above $90\%$ sparsity. The $s_{s s}$ scoring function produces unsatisfactory results with a stark constant decline when pruning. We therefore do not consider $s_{s s}$ an accurate replacement for magnitude pruning. The value of the standard deviation of a weight should be taken into account when pruning, but the magnitude of the mean can already provide a good performance. Even though only a few sparsity levels greater than zero achieve surpassing accuracy, the similarity across all levels as compared to the baseline, seems to imply that the LTH also holds in a Bayesian setting. We will refer to $s_{S N R}$ as IMP when referring to Bayesian models.

Error bars are included in the test-set accuracy plots of the Bayesian models in Figure 1, obtained from repeated stochastic inference passes. The resulting variability is negligible overall, and is only marginally visible for the VGG11 models. Consequently, our observations are not affected by this.

¹https://anonymous.4open.science/r/open_lth-4E2F/

![](./images/1233032154137493546_1.jpg)

Figure 1: Test accuracy and MACE vs. percentage of weights remaining for ResNet18, VGG11 and ViT-tiny models trained on CIFAR10. Shown in color are the different scoring functions for pruning; in black are the non-Bayesian models pruned by magnitude. The x-axis is shown on a logarithmic scale; the y-axis is adjusted to include the full range of observed accuracies.

Figure 1 additionally shows the recorded MACE values over the percentage of retained weights. For ResNet and ViT models MACE decreases at high sparsities, which reflects an increased confidence about the models lowering accuracy. For VGG models, MACE is constant over the percentage of remaining weights. Overall, MACE trends are largely independant of pruning criterion, with the exception of the square pruning function in ViT models, where the large decrease in MACE at higher sparsity coincides with a drop off in accuracy.

The LTs generated through the IMP process have satisfactory prediction accuracy up to 90%, achieving equal performance compared to the dense model in almost all cases. Given that, we try to seek a deeper understanding of how a winning LT is structured, such that it achieves good predictive performance with fewer parameters.

## 6 WHAT DEFINES A LOTTERY TICKET?

To answer the question of how a winning ticket is different from any other pruned network of the same sparsity, we ask ourselves the question *which weights are pruned during the train-prune-reset cycle?* We investigate this by focusing on the last level, i.e., the models with the highest sparsity pruned with the $s_{SNR}$ scoring function. Due to the unstructured nature of global pruning, node-level analysis is not useful. Instead, we compute the layer-wise sparsity ratio to see which layers are pruned to what amount. Figure 2 shows the sparsity ratios for all three trained models in their Bayesian and non-Bayesian variations. We observe that deeper layers get pruned more; these layers commonly have more parameters as compared to shallower layers, and thus, offer more potential to prune parameters. Comparing the Bayesian models to the non-Bayesian counterpart, this skew is even more pronounced. Parameters in the earlier layers are kept in the high sparsity regime. Notably, one can see a modulation in the curve for the VisionTransformer model. This corresponds to the architecture of stacked attention and MLP layers.

We have shown that sparse models could replace the dense, parameter-heavy models. However, as with all sparsity, the question arises whether these winning LTs can only be obtained through the train-prune-reset cycle, or also through a less computationally intensive process, e.g. through ran-

![](./images/1233032154137493546_2.jpg)

Figure 2: Layer-wise sparsity vs. layer index for ResNet (left), VGG (center), and ViT (right) training and pruning 20 times. The x-axis denotes the layer index, and the y-axis shows the resulting sparsity in each layer. Each plot shows the non-Bayesian model and the Bayesian model.

![](./images/1233032154137493546_3.jpg)

Figure 3: Visualization of the different randomizations applied to the weights (magnitude symbolized through color) and masks (binary, i.e. black and white) of a winning ticket.

domly pruning a network even before training. It is therefore of interest to study how a lottery ticket is different from any random ticket of the same sparsity, and how such a random ticket would perform.

## 7 REINITIALIZATION AND SHUFFLING

We use the winning tickets we found in Section 5 to construct several other "random" weight-mask initializations, see Figure 3. For a given weight-mask pair of any level, we apply weight reinitialization or mask shuffling and train as described in the training curriculum. For weight reinitialization, the mask is kept fixed, and we redraw all weights from the initial Kaiming normal distribution. For mask shuffling, the weight vector is fixed, and we test three different shuffling strategies for the mask and compare them. In global shuffling, all mask vectors of all layers are flattened and concatenated. Then a random permutation is applied, and the mask is restored to its original shape. This can lead to very uneven sparsity distributions across the layers. Due to their size difference, a small layer might be completely pruned. For even shuffling, each layer is assigned a random binary mask with sparsity equal to the mean global sparsity. This alleviates the problem of pruned layers, but neglects that with global pruning, each layer has its own sparsity ratio as seen in Section 6. Therefore, layer-wise shuffling redraws the mask for each layer according to its sparsity in the winning ticket found with IMP. Note that this information is only available due to the iterative process.

Figure 4 shows the peak test accuracy over the percentage of remaining weights for all reinitialization and shuffling strategies. Additionally, we show the result of the same reinitialization and shuffling done with non-Bayesian models, replicating the original LTH experiment.

The IMP curves are the ceiling in all configurations, demonstrating that the winning tickets found with IMP are superior to all others. The winning LTs outperforming random tickets confirms that both weight initialization and mask structure make up the sparse subnetwork identified by the LTH. For the ViT-tiny model, this gap between random and winning ticket is the largest, while the accuracy of all other network initializations drops off synchronously with increasing sparsity. In all cases, even and global shuffling removes any advantage a ticket retains over the train-prune-reset cycle. For ResNet18 and VGG11, layerwise shuffling and reinitializing the weights leads to comparable accuracy to the winning ticket. Because reinitialization of weights also keeps layerwise sparsity ratios equal, the layerwise ratios seem to be a crucial ingredient to a winning ticket. On a smaller scale, weight reinitialization outperforms layerwise shuffling; therefore, the more fine-grained graph structure of a network might be the deciding factor over the coarse layerwise sparsity ratio. For the ViT models, only the com-

![](./images/1233032154137493546_4.jpg)

Figure 4: Test accuracy vs. percentage of weights remaining (log scale) for ResNet (left), VGG (middle), and ViT (right) trained on CIFAR10. The black line denotes the original lottery ticket obtained through IMP, the colored lines represent the accuracy achieved by reinitializing or shuffling and training the ticket of that sparsity level. The Bayesian and non-Bayesian models are shown in separate plots.

bination of weight initialization and mask produces highly performing networks. This effect has to be further investi- gated, especially taking into account that large ViT models are commonly pre-trained on large image datasets like Im- ageNet [Deng et al., 2009]. We hypothesize that this is an effect of the architectural design of attention-based trans- formers, where no inductive bias towards images is built into like in convolutional neural networks. We provide ad- ditional ablation experiments to model size, learning rate schedule and learning rate resetting in the appendix.

## 8 LOTTERY TICKET TRANSPLANTATION

We use the generated LTs of the non-Bayesian deterministic models produced in Section 5, to initialize BNNs at match- ing sparsity, with the goal of obtaining performant Bayesian LTs while avoiding the full computational cost of Bayesian LT discovery. For each LT, we construct a BNN by copying the pruned weights into the posterior mean parameters $\mu$ and reusing the sparsity mask, while keeping the $\sigma$ parameters at their initialization. The transplanted tickets are then trained with a final VI phase instead of deterministic training. The resulting maximum test set performance is shown in Fig- ure 5. For comparison, the performances of fully Bayesian LTs and non-bayesian LTs are shown. Empirically, trans- plantation yields accuracies comparable to deterministic and Bayesian LTs for ResNet and VGG, but falls short the non-Bayesian performance in the case of ViT-tiny.

The approach reduces computational demand substantially, as VI training is approximately 3-7$\times$ more expensive than for deterministic models. Appendix results further show that transplanted models retain improved predictive calibration.

Table 1 reports averaged observed training run times in minutes for both non-Bayesian and Bayesian models. The Bayesian models of VGG and ViT take roughly 5 times longer. Because the IMP process chains up to 20 trainings of a model together with a pruning stage, the proposed transplantation saves up to 50% of the compute time.

Table 1: Average runtime for a singular training level of models trained in Section 5 in minutes.

<table>
    <thead>
        <tr>
            <th></th>
            <th>ResNet20</th>
            <th>VGG11</th>
            <th>ViT-Tiny</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Deterministic</td>
            <td>12</td>
            <td>13</td>
            <td>9</td>
        </tr>
        <tr>
            <td>Bayesian</td>
            <td>33</td>
            <td>72</td>
            <td>51</td>
        </tr>
    </tbody>
</table>

![](./images/1233032154137493546_5.jpg)

Figure 5: Test accuracy vs. percentage of weights remaining (log scale) for ResNet (left), VGG (middle), and ViT (right) trained on CIFAR10. Shown are LTs generated through deterministic IMP, a fully Bayesian approach, and the transplantation method.

## 9 DISCUSSION

BNNs bring uncertainty quantification, robustness, and increased generalization capabilities to deep learning, but suffer from an increased compute demand, which could be alleviated through sparsity. The Lottery Ticket Hypothesis presents a vital step towards empirical evidence and justification that training well-performing sparse neural networks is possible, and is therefore of interest to the field of BNNs. The LTH was shown to be effective in all kinds of models and tasks, but so far only for deterministic non-BNNs. We therefore explored the LTH in BNNs trained with variational inference, to evaluate the existance of sparse subnetworks as means to alleviate computational demand. We applied IMP to Bayesian models of ResNet, VGG and ViT by means of various pruning functions, and analysed the resulting winning tickets in terms of their layerwise sparsity ratio as well as performance compared to randomly redrawn and shuffled tickets. For consistency with prior work, our focus is the CIFAR10 dataset, presenting an ideal testbed for understanding pruning dynamics in different architectures.

Similar to standard neural networks, we observe so-called winning tickets, which either match or outperform their dense counterpart. Notably, there are differences per model, which are likely related to the architecture. ResNet and VGG are convolution-based models, while the ViT relies on attention. The convolutional layer carries an inductive bias towards image input, while the transformer has to learn such a bias. Both ResNet and VGG winning tickets gain most of their performance through the right layerwise sparsity ratio and mask structure. But even so, the winning ticket found through IMP outperforms any reshuffled ticket and remains optimal. As VGG seems to be slightly more stable for training when pruned, we hypothesize that the residual connections of ResNet interfere when only some of the parameters in a kernel are pruned. The ViT model is additionally highly sensitive to the initial weights chosen, and hence, clear winning tickets can be observed, mirroring deterministic Transformers, where similar performance changes depending on initialization have been observed. We therefore stress our usage of the VisionTransformer model, which is usually pre-trained on larger datasets.

We find that the iterative pruning scheme is largely independent of the used weight scoring function. Only in high sparsity regimes and in the ViT we observe differences. The posterior mean seems to dominate the pruning decision, and can already deliver good ticket performance.

By plotting the sparsity ratio per layer of highly sparse winning tickets, we observe that deeper layers are pruned more. Compared to standard NNs, this effect is increased, possibly due to a higher uncertainty of weights that increases with depth in the VI framework. This could also be a reason for the increased robustness of BNNs.

We show that it is possible to transplant Lottery Tickets found in non-Bayesian NNs to almost matching accuracy in BNNs. For high sparsities this cuts training time more than in half, while retaining better than randomly pruned performance with an equally well calibration. This demonstrates that a full Bayesian training might not be needed in limited compute environments.

Lastly, we outline the limitations of our study. Due to the large compute demand of BNNs, we have limited our experiments to the dataset used in prior work, CIFAR10, a mid-sized image dataset. We leave the exploration of the LTH in BNNs trained on larger datasets for future work. Similarly, we used VI to implement our BNNs. While this is a common choice in the literature, there are other ways to quantify the uncertainty of a neural network. It is of interest whether our findings can be validated using other methods like MCMC for variational gradients or MC Dropout. Similarly to standard NNs, it is of interest whether completely sparse training with structured sparsity can be realized [Evci et al., 2021, Lasby et al., 2024].

### References

Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks, 2015. URL `https://arxiv.org/abs/1505.05424`.

Carlos M. Carvalho, Nicholas G. Polson, and James G. Scott. Handling Sparsity via the Horseshoe. In *Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics*, pages 73-80. PMLR, April 2009. URL `https://proceedings.mlr.press/v5/carvalho09a.html`. ISSN: 1938-7228.

Tianlong Chen, Yongduo Sui, Xuxi Chen, Aston Zhang, and Zhangyang Wang. A Unified Lottery Ticket Hypothesis for Graph Neural Networks, June 2021a. URL `http://arxiv.org/abs/2102.06790`. arXiv:2102.06790 [cs, stat].

Xiaohan Chen, Yu Cheng, Shuohang Wang, Zhe Gan, Zhangyang Wang, and Jingjing Liu. EarlyBERT: Efficient BERT Training via Early-bird Lottery Tickets, June 2021b. URL `http://arxiv.org/abs/2101.00063`. arXiv:2101.00063 [cs].

Pranav Singh Chib and Pravendra Singh. Recent advancements in end-to-end autonomous driving using deep learning: A survey, 2023. URL `https://arxiv.org/abs/2307.04370`.

Badr-Eddine Chérief-Abdellatif. Convergence Rates of Variational Inference in Sparse Deep Learning, September 2019. URL `http://arxiv.org/abs/1908.04847`. arXiv:1908.04847 [math].

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pages 248-255, June 2009. doi: 10.1109/CVPR.2009.5206848. URL `https://ieeexplore.ieee.org/document/5206848`. ISSN: 1063-6919.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, June 2021. URL `http://arxiv.org/abs/2010.11929`. arXiv:2010.11929 [cs].

Erich Elsen, Marat Dukhan, Trevor Gale, and Karen Simonyan. Fast Sparse ConvNets, November 2019. URL `http://arxiv.org/abs/1911.09723`. arXiv:1911.09723 [cs].

Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, and Erich Elsen. Rigging the Lottery: Making All Tickets Winners, July 2021. URL `http://arxiv.org/abs/1911.11134`. arXiv:1911.11134 [cs].

Josua Faller and Jörg Martin. Optimal Subspace Inference for the Laplace Approximation of Bayesian Neural Networks, February 2025. URL `http://arxiv.org/abs/2502.02345`. arXiv:2502.02345 [cs].

Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks, 2019. URL `https://arxiv.org/abs/1803.03635`.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Stabilizing the Lottery Ticket Hypothesis, July 2020. URL `http://arxiv.org/abs/1903.01611`. arXiv:1903.01611 [cs].

Soumya Ghosh and Finale Doshi-Velez. Model Selection in Bayesian Neural Networks via Horseshoe Priors, May 2017. URL `http://arxiv.org/abs/1705.10388`. arXiv:1705.10388 [stat].

Alex Graves. Practical variational inference for neural networks. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q. Weinberger, editors, *Advances in Neural Information Processing Systems*, volume 24. Curran Associates, Inc., 2011. URL `https://proceedings.neurips.cc/paper_files/paper/2011/file/7eb3c8be3d411e8bfab08eba5f49632-Paper.pdf`.

Song Han, Huizi Mao, and William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding, February 2016. URL `http://arxiv.org/abs/1510.00149`. arXiv:1510.00149 [cs].

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. URL `https://arxiv.org/abs/1512.03385`.

Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks, January 2021. URL `http://arxiv.org/abs/2102.00554`. arXiv:2102.00554 [cs].

Matt Hoffman, David M. Blei, Chong Wang, and John Paisley. Stochastic variational inference, 2013. URL `https://arxiv.org/abs/1206.7051`.

Sohei Itahara, Takayuki Nishio, Masahiro Morikura, and Koji Yamamoto. Lottery Hypothesis based Unsupervised Pre-training for Model Compression in Federated Learning. In *2020 IEEE 92nd Vehicular Technology Conference (VTC2020-Fall)*, pages 1-5, November 2020. doi: 10.1109/VTC2020-Fall49728.2020.9348439. URL `http://arxiv.org/abs/2004.09817`. arXiv:2004.09817 [cs].

Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, and Andrew Gordon Gordon Wilson. What Are Bayesian Neural Network Posteriors Really Like? In *Proceedings of the 38th International Conference on Machine Learning*, pages 4629-4640. PMLR, July 2021. URL https://proceedings.mlr.press/v139/izmailov21a.html. ISSN: 2640-3498.

Sanket Jantre, Shrijita Bhattacharya, and Tapabrata Maiti. Spike-and-Slab Shrinkage Priors for Structurally Sparse Bayesian Neural Networks. *IEEE Transactions on Neural Networks and Learning Systems*, 36(6):11176-11188, June 2025. ISSN 2162-2388. doi: 10.1109/TNNLS.2024.3485529. URL https://ieeexplore.ieee.org/abstract/document/10740530.

SeungHyun Kim, Seohyeon Jung, Seonghyeon Kim, and Juho Lee. Learning to Explore for Stochastic Gradient MCMC, August 2024. URL http://arxiv.org/abs/2408.09140. arXiv:2408.09140 [cs].

Diederik P. Kingma, Tim Salimans, and Max Welling. Variational Dropout and the Local Reparameterization Trick, December 2015. URL http://arxiv.org/abs/1506.02557. arXiv:1506.02557 [stat].

Vinnie Ko, Stefan Ohmcke, and Fabian Gieseke. Magnitude and Uncertainty Pruning Criterion for Neural Networks, December 2019. URL http://arxiv.org/abs/1912.04845. arXiv:1912.04845 [cs].

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Mike Lasby, Anna Golubeva, Utku Evci, Mihai Nica, and Yani Ioannou. Dynamic Sparse Training with Structured Sparsity, February 2024. URL http://arxiv.org/abs/2305.02299. arXiv:2305.02299 [cs].

Namhoon Lee, Thalaiyasingam Ajanthan, and Philip H. S. Torr. SNIP: Single-shot Network Pruning based on Connection Sensitivity, February 2019. URL http://arxiv.org/abs/1810.02340. arXiv:1810.02340 [cs].

Junbo Li, Zichen Miao, Qiang Qiu, and Ruqi Zhang. Training Bayesian Neural Networks with Sparse Subspace Variational Inference, February 2024. URL http://arxiv.org/abs/2402.11025. arXiv:2402.11025 [cs] version: 1.

Bohan Liu, Zijie Zhang, Peixiong He, Zhensen Wang, Yang Xiao, Ruimeng Ye, Yang Zhou, Wei-Shinn Ku, and Bo Hui. A Survey of Lottery Ticket Hypothesis, March 2024. URL http://arxiv.org/abs/2403.04861. arXiv:2403.04861 [cs].

Peter Marsh, Dario Deniz Ergun, and Ercan Engin Kuruoglu. Bayes Always Wins the Lottery in Monte Carlo. October 2025. URL https://openreview.net/forum?id=Ilnbgf1eeS.

Sunil Mathew and Daniel B. Rowe. Pruning a neural network using Bayesian inference, August 2023. URL http://arxiv.org/abs/2308.02451. arXiv:2308.02451 [stat].

Radford M. Neal. *MCMC using Hamiltonian dynamics*. May 2011. doi: 10.1201/b10905. URL http://arxiv.org/abs/1206.1901. arXiv:1206.1901 [stat].

Christopher Nemeth and Paul Fearnhead. Stochastic gradient Markov chain Monte Carlo, July 2019. URL http://arxiv.org/abs/1907.06986. arXiv:1907.06986 [stat].

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. *CoRR*, abs/1912.01703, 2019. URL http://arxiv.org/abs/1912.01703.

Alex Renda, Jonathan Frankle, and Michael Carbin. Comparing Rewinding and Fine-tuning in Neural Network Pruning, March 2020. URL http://arxiv.org/abs/2003.02389. arXiv:2003.02389 [cs, stat].

L. K. Saul, T. Jaakkola, and M. I. Jordan. Mean field theory for sigmoid belief networks, 1996. URL https://arxiv.org/abs/cs/9603102.

Lixin Shen, Rui Wang, Yuesheng Xu, and Mingsong Yan. Sparse Deep Learning Models with the $\ell_1$ Regularization, August 2024. URL http://arxiv.org/abs/2408.02801. arXiv:2408.02801 [cs].

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015. URL https://arxiv.org/abs/1409.1556.

Yang Sui, Miao Yin, Yu Gong, Jinqi Xiao, Huy Phan, and Bo Yuan. ELRT: Efficient Low-Rank Training for Compact Convolutional Neural Networks, January 2024. URL http://arxiv.org/abs/2401.10341. arXiv:2401.10341 [cs].

Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking Winning Tickets Before Training by Preserving Gradient Flow, August 2020. URL http://arxiv.org/abs/2002.07376. arXiv:2002.07376 [cs].

Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek,

Tim Salimans, Rodolphe Jenatton, and Sebastian Nowozin. How Good is the Bayes Posterior in Deep Neural Networks Really?, July 2020. URL http://arxiv.org/abs/2002.02405. arXiv:2002.02405 [stat].

Arvid Weyrauch, Lars H. Heyen, Juan Pedro Gutiér- rez Hermosillo Muriedas, Pei-Hsuan Hsia, Asena Karolin Özdemir, Achim Streit, Markus Götz, and Charlotte De- bus. torch_blue: A Flexible Python Package for Bayesian Neural Networks in PyTorch. *Journal of Open Source Software*, 11(117):9415, January 2026. ISSN 2475-9066. doi: 10.21105/joss.09415. URL https://joss.theoj.org/papers/10.21105/joss.09415.

Mengjing Wu, Junyu Xuan, and Jie Lu. Functional Stochas- tic Gradient MCMC for Bayesian Neural Networks, Oc- tober 2024. URL http://arxiv.org/abs/2409.16632. arXiv:2409.16632 [cs].

Haonan Yu, Sergey Edunov, Yuandong Tian, and Ari S. Morcos. Playing the lottery with rewards and multi- ple languages: lottery tickets in RL and NLP, Febru- ary 2020. URL http://arxiv.org/abs/1906.02768. arXiv:1906.02768 [stat].

## A ABLATION ON MODEL SIZE AND LEARNING RATE

We compare the performance of differently sized models, scaling all models up. Both for non-Bayesian and Bayesian models, we perform the train-prune-reset cycle on the mod- els ResNet110, VGG19, and ViT-base. We show the maxi- mum test accuracy achieved over the percentage of remain- ing weights in Figure 6. As expected, the models with a higher parameter count perform better. The qualitative be- haviour with increasingly pruned parameters stays the same, with a drop in performance when reaching high sparsity. For the case of VGG19, both non-Bayesian and Bayesian, we do not see a drop in performance even at the highest sparsity ratio. This could be due to the heavy over-parametrization.

Frankle and Carbin [2019] found that winning tickets trained with a lower learning rate outperform tickets with a higher learning rate at high sparsity. Warmup helps to close the gap compared to the unpruned network. We therefore re- peat these experiments for BNNs, comparing lottery tickets found with $s_{SNR}$ pruning but with different base learning rates and additional warmup. Although we also find match- ing LTs with low learning rate and even surpassing LTs with warmup, this simply comes at a cost of less overall max- imum accuracy. For medium sparsity, lottery tickets with warmup equal the high learning rate, but a high learning rate still produces the best performing tickets at high sparsity.

## B DOES LRR OUTPERFORM IMP?

LRR [Renda et al., 2020] changes the way IMP is done. Instead of resetting weights still active after pruning to their initial value, training for the next sparsity level continues with the learned values. The learning rate schedule is reset instead. This was found to be a viable if not better alternative to IMP, especially in larger models. We briefly investigate how LRR performs in comparison to IMP in ResNet, VGG, and ViT, both in a Bayesian and non-Bayesian setting. We hypothesize LRR to outperform IMP in BNNs, because the exact pointwise initialization emphasized by IMP is far less meaningful in BNNs where parameters are distributions. Resetting to initial weights discards the learned posterior information that governs predictive uncertainty and makes the pruning mask sensitive to sampling noise. LRR focuses on the learning dynamics rather than a specific weight real- ization, making it better aligned with BNNs distributional nature.

Figure 8 shows the recorded maximum test accuracies achieved both with IMP and LRR. For ResNet18 we see a clear improvement in LRR over IMP, across all sparsity ratios in the standard NN and the BNN model. For VGG11, again LRR outperforms IMP across all sparsity levls, how- ever, not in the Bayesian variant. Only at very high sparsity does the performance drop off for IMP while it does not for LRR, which has stable performance up to the highest sparsity level. Pruning the bayesian variant of ViT-tiny, LRR outperforms IMP only at high sparsities. In the standard NN, LRR produces a higher accuracy at every sparsity level. We summarize our findings: LRR achieves a higher predictive performance at high sparsity, and seems to produce more stable winning tickets. While it is sometimes outperformed by IMP at low sparsity, it often achieves matching or sur- passing accuracy to the dense counterpart, even though it does not take the random starting weight initialization into account, as proposed by the LTH.

## C CALIBRATION DIFFERENCE IN TRANSPLANTATION

We show the calibration error plots for models mentioned in Section 8 in Figure 9. We compare to the BNNs obtained through full Bayesian Training and IMP. MACE trends of transplanted tickets largely mirror the full Bayesian models. For ResNet and ViT this means a decrease in calibration error in the high sparsity regime, and a relatively constant MACE for VGG models.

![](./images/1233032154137493546_6.jpg)

Figure 6: Test accuracy vs. Percentage of weights remaining (log scale) for ResNet, VGG, and ViT of different sizes trained on CIFAR10. Shown are non-Bayesian and Bayesian models.

![](./images/1233032154137493546_7.jpg)

Figure 7: Test accuracy vs. Percentage of weights remaining (log scale) for ResNet, VGG, and ViT trained on CIFAR10.
The colored lines represent different initial learning rates. Included as a subplot are the equivalent non-Bayesian models.

![](./images/1233032154137493546_8.jpg)

Figure 8: Test accuracy vs. percentage of weights remaining (log scale) for ResNet, VGG, and ViT trained on CIFAR10.
The lines represent tickets obtained through either Learning Rate Rewinding or Iterative Magnitude Pruning. Shown are non-Bayesian and Bayesian models.

![](./images/1233032154137493546_9.jpg)

Figure 9: MACE vs. percentage of weights remaining for transplanted LTs (see Section 8) and fully Bayesian LTs.