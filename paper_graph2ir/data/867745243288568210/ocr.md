# Bespoke vs. Prêt-à-Porter Lottery Tickets: Exploiting Mask Similarity for Trainable Sub-Network Finding

Michela Paganini
Facebook AI Research
michela@fb.com

Jessica Zosa Forde
Brown University
Facebook AI Research
jessica_forde@brown.edu

## Abstract

The observation of sparse trainable sub-networks within over-parametrized networks – also known as Lottery Tickets (LTs) – has prompted inquiries around their trainability, scaling, uniqueness, and generalization properties. Across 28 combinations of image classification tasks and architectures, we discover differences in the connectivity structure of LTs found through different iterative pruning techniques, thus disproving their uniqueness and connecting emergent mask structure to the choice of pruning. In addition, we propose a consensus-based method for generating refined lottery tickets. This lottery ticket denoising procedure, based on the principle that parameters that always go unpruned across different tasks more reliably identify important sub-networks, is capable of selecting a meaningful portion of the architecture in an embarrassingly parallel way, while quickly discarding extra parameters without the need for further pruning iterations. We successfully train these sub-networks to performance comparable to that of ordinary lottery tickets.

## 1 Introduction

Deep neural architectures have seen a dramatic increase in size over the years [3]. Over-parametrized networks exhibit high generalization performance, with recent empirical evidence showing that the generalization gap tends to close with increased number of parameters [55, 5, 24, 1, 13, 14], contrary to prior belief, also depending on the inductive bias and propensity to memorization of each base architecture [56]. While advantageous under this point of view, the proliferation of parameters in neural architectures may induce adverse consequences. For instance, the computational cost to train some state-of-the-art models has raised the barrier to entry for many researchers hoping to contribute. Because of limited memory, time, and compute, and to enable private, secure, on-device computation, methods for model compression have seen a rise in popularity. Among these are techniques for model pruning, quantization, and distillation.

Pruning, in particular, has been seen as an overfitting avoidance method since the early decision tree literature [7], with work comparing the effects of different tree pruning techniques [37]. In analogy to this prior line of work, we provide an empirical comparison of the effects of neural network pruning methods, and analyze the structure of the sparse sub-networks that emerge. Revived by the recent observation of the existence of sub-networks with favorable training properties within larger over-parametrized models, known as (winning) Lottery Tickets [16], and in the spirit of other work along these lines [59], this work investigates the dependence of the properties of these sparse, trainable sub-networks on the choice of pruning technique. We set out to answer the questions: do different pruning methods identify the same lucky sub-network? If not, where are they similar? What does this tell us about fundamental pathways within neural networks, and what distinguishes a LT

Preprint. Under review.

![](./images/867745243288568210_1.jpg)

Figure 1: Consensus-based lottery ticket identification from bespoke lottery tickets sourced on
different tasks. Thick lines identify unpruned connections; thin lines, pruned ones.

from other node combinations? The goal of this work is to investigate the nature of LTs and whether
their structure is meaningful.

Additionally, preliminary evidence for LT transferability has been provided in both natural and
non-natural image domains [38, 45]. In the context of LTs, transfer refers to the ability to retrain a
sub-network on a new task. We investigate whether similarities in the structure of LTs sourced on
different datasets may, at least partially, explain their transferability across tasks.

We exploit these similarities to denoise and further slim down the ticket, by expediting parameter
removal via mask consensus pooling. Intuitively, this builds on the hypothesis that any core struc-
ture within a base architecture that consistently remains unpruned across bespoke LTs sourced on
individual tasks may more robustly identify critical pathways in the network. We suggest using the
intersection of low-sparsity bespoke LTs to obtain a new high-sparsity LT that captures the shared,
general structure needed to perform the image classification tasks considered in this work (Fig. 1). We
show that the sparse structure that emerges from this procedure is itself a trainable sub-network, which
we mnemonically call a prêt-à-porter LT. This ensembling strategy also allows to parallelize the
otherwise purely sequential lottery ticket finding procedure, while maintaining competitive accuracy
across tasks.

### 1.1 Contributions
This work provides empirical evidence showing that:
1. there can exist multiple different lucky sub-networks (lottery tickets) within an over-
parametrized network;
2. different pruning techniques are capable of finding different lottery tickets;
3. although lottery tickets can be successfully transferred and retrained on different tasks within
a family of tasks, the masks sourced on different tasks are largely dissimilar;
4. the intersection of masks sourced on different tasks identifies a new, sparser, trainable lottery
ticket that allows to parallelize the sparsification procedure through a more grounded node
importance definition.

## 2 Related work
State-of-the-art results in machine learning seem to strongly correlate with increased number of
parameters [8], leading to the common use of vastly over-parametrized models, compared to the
complexity of the tasks at hand. In practice, however, most common neural network architectures
can be dramatically pruned down in size, though there is no natural order in which neural network
parameters should be removed [22, 33, 23, 20, 19, 53, 36, 49, 57, 4, 40].

Identifying important units in a neural network is an open challenge, often driven by desires for
interpretability and model compression, among others. Various proxies for importance have been
proposed as a direct consequence of the lack of precise mathematical definition of this term. Different
objectives lead to different mathematical formulations of importance: concept creation, output

attribution, conductance, saliency, activation, explainability, connectivity density, redundancy, noise robustness, stability, and more [6, 2, 58, 12, 46].

In general, though, small, sparse models are empirically hard to train from scratch. However, the recently formulated lottery ticket hypothesis conjectures that there exists at least one sparse sub-network, a *winning ticket*, within an over-parametrized network, that can achieve commensurate accuracy in commensurate training time with fewer parameters, if retrained from the same initialization [16]. To find this lucky sub-network, the lowest magnitude connections are iteratively pruned for a fixed number of iterations [33, 21]. Interestingly, while pruned weights are set to zero, unpruned weights are *rewound* to their initialization values, prior to the next iteration of training and pruning.

Researchers have reported difficulties in scaling this method to larger models. Some attempts were unable to successfully train small sparse sub-networks found through this procedure, when starting from VGG-16 [47] and ResNet-50 [25] trained on CIFAR-10 [30], and instead observed that random initialization outperformed weight rewinding [35]. Similarly, other experimental work was unable to successfully produce a sub-network with these properties within ResNet-50 trained on ImageNet [44] and Transformer [50] trained on WMT 2014 English-to-German [18].¹ Modifications to the original procedure have been proposed in order to favor the emergence of winning tickets in larger models and tasks. These include challenging the need for unstructured, magnitude-based pruning [59], using learning rate warmup [16], selecting unimportant weights globally instead of layer by layer [16], and *late resetting* unpruned parameters to values achieved early in training, as opposed to rewinding the weights all the way to their initial values [17, 38]. These strategies were further utilized to show that lottery tickets are not simply an emergent phenomenon in the context of supervised learning of natural images, but they can be found and trained on NLP and RL tasks as well [54]. This debate has left questions unanswered around the nature of LTs, the significance of their structure and initialization, their implications for information propagation and concept creation, and their uniqueness within over-parametrized architectures. Furthermore, the lack of principled understanding of what distinguishes a LT from other sub-networks within the same model has made it hard to move beyond the computationally expensive, iterative LT-finding technique based on magnitude-based unstructured pruning.

On the other hand, lottery tickets have been shown to possess generalization properties that allow for their reuse across similar tasks, thus reducing the computational cost of finding dataset-dependent sparse sub-networks [38, 45]. However, the mechanism that powers their transferability properties is not yet clearly understood, and is therefore subject to further study in this work.

Alternative approaches to neural architecture optimization have taken a constructionist approach, starting from a minimal set of units and connections, and growing the network by adding new components [48, 52]. Finally, other methods combine both network growing and pruning, in analogy to the different phases of connection formation and suppression in the human brain [15, 11].

## 3 Methodology

We explore the performance of sub-networks generated by different iterative pruning techniques starting from base LeNet [34], AlexNet [31], VGG11 [47], and ResNet18 [25] architectures, on the following set of image classification tasks: MNIST [32], Fashion-MNIST [51], Kuzushiji-MNIST (or KMNIST) [9, 27], EMNIST [10], CIFAR-10, CIFAR-100 [30], SVHN [41].

All networks are trained for 30 epochs using SGD with constant learning rate 0.01, batch size of 32, without explicit regularization. The pruning fraction per iteration is held constant at 20% of remaining connections/units per layer.

In all figures, unless otherwise specified, the error bars and shaded envelopes correspond to one standard deviation (half up, half down) from the mean, over 6 experiments with seeds 0-5.

### 3.1 Pruning methods

This works explores a variety of pruning techniques that may differ along the following axes:

---
¹Translation Task - ACL 2014 Ninth Workshop on Statistical Machine Translation (URL: https://www.statmt.org/wmt14/translation-task.html)


![](./images/867745243288568210_2.jpg)
(a) VGG11 on CIFAR-100

![](./images/867745243288568210_3.jpg)
(b) AlexNet on CIFAR-100

![](./images/867745243288568210_4.jpg)
(c) ResNet18 on CIFAR-10

![](./images/867745243288568210_5.jpg)
(d) LeNet on FashionMNIST

Figure 2: Test accuracy for SGD-trained, sparsified models, pruned using seven different pruning techniques ($L_2$-structured, global unstructured, $L_1$-unstructured, $L_1$-structured, $L_\infty$-structured, random structured, random unstructured), and rewound to initial weight values after each pruning iteration.

Neuronal importance definition: In magnitude-based pruning, units/connections are removed based on the magnitude of synaptic weights. Usually, low magnitude parameters are removed. As a (rarer) alternative, one can consider removing high magnitude weights instead [59]. Non-magnitude-based pruning techniques can be based, among others, on activations, gradients, or custom rules for neuronal importance.

Local vs. global: Local pruning consists of removing a fixed percentage of units/connections from each layer by comparing each unit/connection exclusively to the other units/connections in the layer. On the contrary, global pruning pools all parameters together across layers and selects a global fraction of them to prune. The latter is particularly beneficial in the presence of layers with unequal parameter distribution, by redistributing the pruning load more equitably. A middle-ground approach is to pool together only parameters belonging to layers of the same kind, to avoid mixing, say, convolutional and fully-connected layers.

Unstructured vs. structured: Unstructured pruning removes individual connections, while structured pruning removes entire units or channels. Note that structured pruning along the input axis is conceptually similar to input feature importance selection. Similarly, structured pruning along the output axis is analogous to output suppression.

In this work, we compare: magnitude-based {global, $L_1$, random} unstructured, and {$L_1$, $L_2$, $L_{-\infty}$, random} structured pruning. Pruning is only applied to weight matrices and not biases (nor other parametrized layers such as batch norm [28]). Pruning techniques and models are implemented using PyTorch [43, 42].

## 4 Analysis
Each point on the accuracy-sparsity trade-off curves in Fig. 2 represents the test accuracy after exactly 30 epochs of training of a pruned model at the given level of sparsity. In these experiments, models are trained and evaluated on the same task, throughout the iterative sparsification procedure. Both global and local unstructured pruning are able to identify high-performance sub-networks that are trainable up to high levels of sparsity. The additional flexibility of global pruning to compare weight magnitude across all layers, instead of in a layer-by-layer manner, allows it to prune larger, more over-parametrized layers more aggressively, while keeping other layers almost intact. Performance plots for other dataset-model combinations can be found in Appendix A.

Different pruning techniques are therefore capable of identifying different sparse trainable sub-networks, although, based on pure accuracy-sparsity trade-off arguments, some may be able to extract more meaningful structure than others.

### 4.1 The similarity of masks
In what follows, let a mask $\mathbb{M}$ with $|\mathbb{M}| = D$ over a network of $D$ parameters $\boldsymbol{\theta}$ be a set of elements with binary values $m_i = \mathbf{1}_{\text{unpruned}}(\theta_i) \in \{0,1\}$ that represent membership of the network's parameters to the set of unpruned weights. We measure the overlap between two binary masks

![](./images/867745243288568210_6.jpg)

Figure 3: Layer-wise Jaccard distance between masks found by pruning AlexNet on CIFAR-100 using $L_2$-structured pruning, and masks found by other pruning methods (global unstructured, $L_1$-unstructured, $L_2$-structured $L_\infty$-structured, random structured, random unstructured), as a function of the pruning iteration, conditional on identical seed, across six seeds. Each plot represents a layer in AlexNet. $L_1$-structured pruning yields the most similar masks to $L_2$-structured pruning, as expected.

$(\mathbb{M}_1, \mathbb{M}_2)$ representing the sparsification of two networks that share identical base architecture and initialization, by computing their Intersection over Union (IoU), also quantifiable in terms of the Jaccard distance [29]: $d_J(\mathbb{M}_1, \mathbb{M}_2) = 1 - \frac{|\mathbb{M}_1 \cap \mathbb{M}_2|}{|\mathbb{M}_1 \cup \mathbb{M}_2|}$.

### 4.1.1 Mask similarity across pruning methods

Although different pruning techniques may yield sub-networks with comparable total accuracy at a given sparsity level, a deeper investigation into their connectivity structure shows, as evident from the growth of the per-layer Jaccard distance as a function of pruning iteration in Fig. 3 and the evolution of the pairwise total Jaccard distances in Fig. 4, that there exist multiple lucky sub-networks with similar performance, yet little to no overlap.

Pruning the same network using different pruning techniques gives rise to sparse sub-networks that differ not only in structure but also in the learned function that they compute. Given sufficient compute and memory budget, one can consider ensembling the LTs and combining the predictions made by each sub-network to boost performance.

### 4.1.2 Mask similarity across tasks

Fixing the pruning method and initializing identical replicas of the same model, we find lottery tickets on a set of image classification tasks. We call lottery tickets sourced on a specific task bespoke lottery tickets. We confirm previous findings of ticket transferability [39, 45], and we aim to analyze whether similarity of masks can trivially explain this property.

The performance of bespoke LTs (of $\sim$85% sparsity for LeNet, and $\sim$75% sparsity on AlexNet, VGG11, and ResNet18) evaluated on a set of target tasks (Fig. 6) confirms that these LTs are indeed sufficiently general to allow for retraining on various datasets from the same domain. Each bespoke LT performs best on the task it was sourced on, but tickets sourced on other tasks generally remain competitive.

The average Jaccard distance between bespoke LT masks reveals significant dissimilarities between sub-networks of similar sparsity sourced on different datasets, even when data distributions are close, as is the case for CIFAR-10 and CIFAR-100, or MNIST and KMNIST. However, even at high levels

![](./images/867745243288568210_7.jpg)

Figure 4: Pairwise total Jaccard distance between the masks obtained by different pruning techniques on AlexNet architectures trained on CIFAR-100. As the networks grow progressively sparser, the distance between masks grows as their intersection shrinks.

![](./images/867745243288568210_8.jpg)

Figure 5: Pairwise total Jaccard distance between the bespoke masks obtained through global unstructured pruning on LeNet, over a set of tasks. As the networks grow progressively sparser, the distance between masks grows as their intersection shrinks.

of sparsity ($\sim 99\%$ in Fig. 5c), emerging masks are not completely disjoint, but share 6-10% of their unpruned parameters with other bespoke masks.

We conclude that LT transfer capabilities cannot be trivially explained by close-to-identical bespoke masks, but these masks do share a core backbone of unpruned parameters, even at high levels of sparsity.

### 4.2 Prêt-à-porter lottery tickets: transferable, shared, core sub-networks

The LT finding algorithm relies on a noisy neuron importance ranking based on absolute weight magnitude, which has non-negligible probability of mischaracterizing and, therefore, eliminating important parameters and pathways in a network. For this reason, traditional LT-style pruning proceeds cautiously through several computationally expensive iterations of pruning, rewinding, and retraining, by only removing a low fraction (20% being an effective compromise) of remaining weights at each iteration. In fact, while magnitude-based weight pruning has both theoretically and experimentally grounded reasons for being one of the preferred proxies for neuron importance [21, 18], factors such as the stochasticity in the training process, the underconstrainedness of weights in over-parametrized networks (especially in ReLU networks), the non-immediate rate of convergence of parameters to their final value, as well as any other source of noise that enters the final magnitude-based ranking of weights, make this proxy unstable and the lottery ticket discovery procedure noise prone.

Exploiting the insight from Sec. 4.1.2 regarding the existence of shared portions of LT structure across tasks, we propose a method for eliminating unwanted noise in the lottery ticket structure by relying on consensus-based refinement of low-to-medium sparsity LTs into higher sparsity LTs that we call prêt-à-porter lottery tickets (Fig. 1). We believe this to be a computationally efficient, logically compelling, and principled way of identifying key parameters within over-parametrized networks, that can rival multiple rounds of iterative magnitude-based pruning for LT discovery.

Given a number of tasks $N$, a consensus mask $\mathbb{M}$ is computed so that each entry represents the fraction of tasks over which, according to the traditional LT-finding algorithm, the corresponding weight goes unpruned. If the traditional LT-finding algorithm consistently picked out the same sub-network, independent of the task, then the consensus mask $\mathbb{M}$ would be a binary mask identical to the bespoke LT masks. The opposite limit would be the case of no overlap across masks. Given the partial similarities observed among bespoke masks sourced on different datasets, one can normally expect a non-binary $\mathbb{M}$. In this work, we demand that weights be unpruned across all $N$ bespoke tickets to become members of the prêt-à-porter ticket: $\mathbb{M} = \mathbb{M}_1 \cap \mathbb{M}_2 \cap \mathbb{M}_3 \cap \cdots \cap \mathbb{M}_N$. A different threshold can be imposed to relax this notion and obtain prêt-à-porter tickets with lower sparsity. This intersection extracts the core, shared network structure that is deemed important for efficiently solving multiple tasks.

The computation of the low-sparsity LTs $\mathbb{M}_i$ used to obtain the mask intersection $\mathbb{M}$ can be made embarrassingly parallel, thus addressing one of the major practical obstacles for LT adoption.

### 4.2.1 Experiments
We consider a set of $N = 5$ tasks from the same domain (in this case, image classification). We select tasks that share the same number of output classes, here $k = 10$, and we resize the input images to the same shape, in order to ensure identical base architecture. The datasets used in this portion are: MNIST, FashionMNIST, KMNIST, CIFAR-10, SVHN. A copy of the model (with identical seed, thus identical initialization) is trained on each separate task, then iteratively pruned with weight rewinding over 2 pruning iterations, using global unstructured pruning with 20% sparsification rate per iteration. The training hyper-parameters are identical to the ones in Sec. 3.

The binary masks of the resulting bespoke LTs are queried for consensus, in order to identify core weights that consistently go unpruned across the full set of $N = 5$ tasks. We expect the masks' intersection to represent fundamental pathways within the base architecture that are important for solving this family of tasks.

The resulting prêt-à-porter ticket has a mask equal to the intersection of masks found across the $N = 5$ tasks, and initialization equal to the value at initialization of those remaining weights. The prêt-à-porter and all bespoke tickets are retrained from scratch on each target task. We empirically demonstrate that the prêt-à-porter LT is, indeed, trainable to comparable performance to similar-sparsity bespoke LTs found through the traditional iterative pruning strategy (Fig. 6). For equivalent LT performance evaluation at initialization, prior to training, see Appendix F. The failures in training at times observed on SVHN can be remedied with a more principled training recipe; however, to keep results consistent, we refrain from dataset-specific optimization.

Furthermore, prêt-à-porter LTs have the added advantage of better learnability, as the parameters that compose them tend to be initialized from a distribution of smaller variance, closer to that of the original initialization distribution, than bespoke tickets (Fig. 7 for AlexNet; see Appendix E for ResNet18, VGG11, and LeNet in Figs. 17, 18, and 19 respectively). The scale of the standard deviation at initialization is known to be important for network training stability [26].

## 5 Conclusion
We show evidence against the uniqueness of winning tickets in a variety of networks and tasks, by identifying different lucky sub-networks of competitive performance within the same parent network, while controlling for degeneracy by fixing experimental seeds. We also show that LTs initiated from a common initialization, but trained on different datasets, tend to have a small but consistent overlap of masks. Using this, we introduce prêt-à-porter LTs, effectively combining the masks from identical models with LTs sourced on different tasks. We show that prêt-à-porter LTs are not only competitive, but provide sourcing speedups via parallelism.

![](./images/867745243288568210_9.jpg)

Figure 6: Bespoke and prêt-à-porter lottery ticket accuracy for networks sourced on the tasks listed in the legend using global unstructured pruning, starting from LeNet (panel 1), AlexNet (panel 2), VGG11 (panel 3), and ResNet18 architectures, and evaluated on the target tasks listed along the x-axis. As a benchmark, we also show the performance of a random sub-network. All sub-networks have comparable sparsity.

![](./images/867745243288568210_10.jpg)

Figure 7: Initialization values of weights in a AlexNet prêt-à-porter LT (denoted ∩LT), compared to those that make up a bespoke LT of similar sparsity sourced on CIFAR-10, and those that make up a bespoke LT of similar sparsity sourced on MNIST. For comparison, the full distribution of initial weights is shown as a shaded gold histogram.

We hope these experimental results in reproducible, controlled settings will guide theoretical work in this sub-field, and lead to novel approaches to LT sourcing, and more broadly, a deeper understanding of LTs and pruning as a whole.

### Broader Impact

This work builds upon prior work in LTs and pruning more broadly, which, beyond as a probe into fundamental questions of network capacity, aid in making neural networks more accessible in low-power or resource-constrained environments, which has massive implications for the acces- sibility of models in the developing world, medical applications, and ubiquity on mobile devices, to name a few. We must be mindful, however, that invasive, network modifying actions may have unintended demographic consequences with respect to fairness, bias, and more broadly reliability and interpretability – critical corner stones when considering the environments that LTs and pruned models may be used in.

### References

[1] Z. Allen-Zhu, Y. Li, and Y. Liang. Learning and Generalization in Overparameterized Neural Networks, Going Beyond Two Layers. November 2018. arXiv:1811.04918.

[2] R. A. Amjad, K. Liu, and B. C. Geiger. Understanding individual neuron importance using information theory. April 2018. arXiv:1804.06679.

[3] D. Amodei and D. Hernandez. AI and compute. May 2018. URL: https://openai.com/blog/ai-and-compute/.

[4] B. O. Ayinde, T. Inanc, and J. M. Zurada. Redundant feature pruning for accelerated in- ference in deep neural networks. *Neural Networks*, 118:148 – 158, 2019. URL: http://www.sciencedirect.com/science/article/pii/S0893608019301273, doi:https://doi.org/10.1016/j.neunet.2019.04.021.

[5] M. Belkin et al. Reconciling modern machine learning and the bias-variance trade-off. December 2018. arXiv:1812.11118.

[6] M. Berglund et al. Measuring the usefulness of hidden units in boltzmann machines with mutual information. In *Neural Information Processing*, pages 482–489, Berlin, Heidelberg, 2013. Springer Berlin Heidelberg.

[7] L. Breiman et al. *Classification and Regression Trees*. The Wadsworth and Brooks-Cole statistics-probability series. Taylor & Francis, 1984. URL: https://books.google.com/books?id=JwQx-W0mSyQC.

[8] T. B. Brown et al. Language models are few-shot learners, 2020. arXiv:2005.14165.

[9] T. Clanuwat et al. Deep learning for classical Japanese literature, 2018. arXiv:1812.01718.

[10] G. Cohen et al. EMNIST: an extension of MNIST to handwritten letters, 2017. arXiv:1702.05373.

[11] X. Dai, H. Yin, and N. Jha. NeST: A neural network synthesis tool based on a grow-and-prune paradigm. *IEEE Transactions on Computers*, 2019.

[12] K. Dhamdhere, M. Sundararajan, and Q. Yan. How Important Is a Neuron? May 2018. arXiv:1805.12233.

[13] S. S. Du et al. Gradient Descent Provably Optimizes Over-parameterized Neural Networks. Oct 2018. arXiv:1810.02054.

[14] S. S. Du and J. D. Lee. On the Power of Over-parametrization in Neural Networks with Quadratic Activation. Mar 2018. arXiv:1803.01206.

[15] D. Floreano, P. Dürr, and C. Mattiussi. Neuroevolution: from architectures to learning. *Evolu- tionary Intelligence*, 1(1):47–62, Mar 2008. doi:10.1007/s12065-007-0002-4.

[16] J. Frankle and M. Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. March 2018. arXiv:1803.03635.

[17] J. Frankle et al. The lottery ticket hypothesis at scale. March 2019. arXiv:1903.01611.

[18] T. Gale, E. Elsen, and S. Hooker. The state of sparsity in deep neural networks. February 2019. arXiv:1902.09574.

[19] Y. Guo, A. Yao, and Y. Chen. Dynamic Network Surgery for Efficient DNNs. Aug 2016. arXiv:1608.04493.

[20] S. Han, H. Mao, and W. J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. Oct 2015. arXiv:1510.00149.

[21] S. Han, J. Pool, J. Tran, and W. Dally. Learning both weights and connections for efficient neural network. Adv. Neural Inf. Process. Syst., 2015. URL: http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.

[22] S. J. Hanson and L. Y. Pratt. Comparing biases for minimal network construction with back-propagation. In *Advances in neural information processing systems*, pages 177–185, 1989.

[23] B. Hassibi and D. G. Stork. Second order derivatives for network pruning: Optimal brain surgeon. In S. J. Hanson, J. D. Cowan, and C. L. Giles, editors, *Advances in Neural Information Processing Systems 5*, pages 164–171. Morgan-Kaufmann, 1993. URL: http://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon.pdf.

[24] T. Hastie et al. Surprises in High-Dimensional Ridgeless Least Squares Interpolation. Mar 2019. arXiv:1903.08560.

[25] K. He et al. Deep residual learning for image recognition. December 2015. arXiv:1512.03385.

[26] K. He et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In *Proceedings of the IEEE international conference on computer vision*, pages 1026–1034, 2015.

[27] Humanities Open Data Shared Usage Center / Jinbungaku Open Data Kyodou Ryo Center, National Museum of Japanese Literature / Kokubungakukenkyuushiryokan. KMNIST Dataset (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others). URL: http://codh.rois.ac.jp/kmnist/, doi:10.20676/00000341.

[28] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 2015. arXiv:1502.03167.

[29] P. Jaccard. Etude de la distribution florale dans une portion des alpes et du jura. *Bulletin de la So-ciete Vaudoise des Sciences Naturelles*, 37:547–579, 01 1901. doi:10.5169/seals-266450.

[30] A Krizhevsky. Learning multiple layers of features from tiny images. 2009. URL: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf.

[31] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In *Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1*, NeurIPS 2012, page 1097–1105, Red Hook, NY, USA, 2012. Curran Associates Inc.

[32] Y. LeCun, C. Cortes, and C. J. C. Burges. *The MNIST database of handwritten digits.*, 1994. URL: http://yann.lecun.com/exdb/mnist/.

[33] Y. LeCun, J. S. Denker, and S. A. Solla. Optimal brain damage. In D S Touretzky, editor, *Advances in Neural Information Processing Systems 2*, pages 598–605. Morgan-Kaufmann, 1990. URL: http://papers.nips.cc/paper/250-optimal-brain-damage.pdf.

[34] Y. LeCun et al. Handwritten digit recognition with a Back-Propagation network. In D. S. Touretzky, editor, *Advances in Neural Information Processing Systems 2*, pages 396–404. Morgan-Kaufmann, 1990. URL: http://papers.nips.cc/paper/293-handwritten-digit-recognition-with-a-back-propagation-network.pdf.

[35] Z. Liu et al. Rethinking the value of network pruning. October 2018. arXiv:1810.05270.

[36] J.-H. Luo, J. Wu, and W. Lin. ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression. Jul 2017. arXiv:1707.06342.

[37] J. Mingers. An empirical comparison of pruning methods for decision tree induction. Machine Learning, 4(2):227–243, Nov 1989. doi:10.1023/A:1022604100933.

[38] A. S. Morcos et al. One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. Jun 2019. arXiv:1906.02773.

[39] A. S. Morcos, M. Raghu, and S. Bengio. Insights on representational simi- larity in neural networks with canonical correlation. In S. Bengio et al., ed- itors, Advances in Neural Information Processing Systems 31, pages 5727–5736. Curran Associates, Inc., 2018. URL: http://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with\-canonical-correlation.pdf.

[40] B. Mussay et al. On Activation Function Coresets for Network Pruning. July 2019. arXiv:1907.04018.

[41] Y. Netzer et al. Reading digits in natural images with unsupervised feature learning. 2011. URL: http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf.

[42] M. Paganini and J. Z. Forde. Streamlining tensor and network pruning in PyTorch, 2020. arXiv:2004.13770.

[43] A. Paszke et al. Automatic differentiation in PyTorch. In NeurIPS Autodiff Workshop, 2017.

[44] O. Russakovsky et al. ImageNet large scale visual recognition challenge. Int. J. Comput. Vis., 115(3):211–252, December 2015. URL: https://doi.org/10.1007/s11263-015-0816-y.

[45] M. Sabatelli, M. Kestemont, and P. Geurts. On the transferability of winning tickets in non- natural image datasets, 2020. arXiv:2005.05232.

[46] A. Shrikumar, P. Greenside, and A. Kundaje. Learning important features through propagat- ing activation differences. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 3145–3153. JMLR. org, 2017.

[47] K. Simonyan and A. Zisserman. Very deep convolutional networks for Large-Scale image recognition. September 2014. arXiv:1409.1556.

[48] K. O. Stanley and R. Miikkulainen. Evolving neural networks through augmenting topolo- gies. Evolutionary Computation, 10(2):99–127, 2002. URL: http://nn.cs.utexas.edu/?stanley:ec02.

[49] F. Tung, S. Muralidharan, and G. Mori. Fine-Pruning: Joint Fine-Tuning and Compression of a Convolutional Network with Bayesian Optimization. Jul 2017. arXiv:1707.09102.

[50] A. Vaswani et al. Attention is all you need. In I Guyon et al., editors, Advances in Neural Information Processing Systems 30, pages 5998–6008. Curran Associates, Inc., 2017. URL: http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf.

[51] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms, 2017. arXiv:1708.07747.

[52] S. Xie et al. Exploring Randomly Wired Neural Networks for Image Recognition. Apr 2019. arXiv:1904.01569.

[53] T.-J. Yang, Y.-H. Chen, and V. Sze. Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning. Nov 2016. arXiv:1611.05128.

[54] H. Yu et al. Playing the lottery with rewards and multiple languages: lottery tickets in rl and nlp, 2019. arXiv:1906.02768.

[55] C. Zhang et al. Theory of deep learning III : Generalization properties of SGD. April 2017.

[56] C. Zhang et al. Identity crisis: Memorization and generalization under extreme overparameterization. February 2019. URL: http://arxiv.org/abs/1902.04698, arXiv:1902.04698.

[57] T. Zhang et al. A systematic dnn weight pruning framework using alternating direction method of multipliers. In *The European Conference on Computer Vision (ECCV)*, September 2018.

[58] Bolei Zhou, Yiyou Sun, David Bau, and Antonio Torralba. Revisiting the importance of individual units in CNNs via ablation. June 2018. arXiv:1806.02891.

[59] H. Zhou et al. Deconstructing lottery tickets: Zeros, signs, and the supermask. May 2019. arXiv:1905.01067.

## A Additional accuracy-sparsity curves

Fig. 8 shows additional accuracy-sparsity trade-off curves over various combinations of datasets and models. Both axes are plotted in logit scale. Similar curves for all dataset-model combinations are available upon request.

The power of global unstructured pruning, in black, lies in its flexibility to compare parameters across layers, although its advantage over its layer-wise counterpart decreases for more expressive and over-parametrized base architectures. It is also know that it is common to observe increases, as opposed to decreases, in total accuracy for moderate pruning fractions.

As a reminder, however, our simple training strategy, with constant learning rate and number of training epochs, employed to partially train the networks prior to pruning, is sub-optimal to achieve maximum performance; that, in fact, is not the goal of these training rounds, as the networks are not trained to full convergence or zero training loss. Therefore, it is not recommended to draw generalized conclusions about the maximum achievable performance of any lottery ticket found through any pruning technique by simply relying on the results in Fig. 8. To assess the full performance of these lottery tickets at any sparsity level, we suggest retraining them from scratch with an optimized training strategy and well sourced, case-specific hyper-parameters. In other words, the results shown here are intended to show the performance of these networks at the point in which the training was stopped prior to sparsifying the network further, and it is therefore not meant to be interpreted as these tickets' maximum achievable performance.

## B Finetuning vs. reinitializing

The "Lottery Ticket Hypothesis" [16] postulates that rewinding weights to the initial (or early-stage [17]) values after pruning is key to identifying lucky sub-networks. In these experiments, we focus primarily on establishing the effect of the choice of weight handling strategy (finetuning vs. rewinding) on the structure of the masks of the resulting LTs.

The Jaccard distance, or any other similarity measure among masks (e.g., the Hamming distance), can be adopted to quantify the effects of the choice of finetuning or reinitializing weights after pruning. As expected, the nature of the connectivity structure that emerges in an iterative series of pruning and weight handling steps depends not only on the pruning choice but also on how weights are handled after pruning. Finetuning yields significantly different masks from rewinding, for all pruning techniques, although the difference is greater for magnitude-based structured pruning than for magnitude-based unstructured pruning (Fig. 9). The difference (quantified in terms of the Jaccard distance) appears to grow logarithmically in the number of pruning iterations for magnitude-based structured pruning techniques; for magnitude-based unstructured pruning techniques, instead, the growth of the Jaccard distance exhibits more complex trends, with layer-by-layer differences.

## C Effective pruning rate

A subtle implementation detail involves the way in which the fraction of pruned weights is computed, especially in convolutional layers pruned with structured pruning techniques. Take, for example, the architectures used in this work. When applying pruning to produce results like the ones displayed in Fig. 8, there are two ways to define the number of weights that get pruned (displayed along the x-axis). The definition used in the paper uses the fraction of weights explicitly pruned by the decision rule that produces the mask for each layer. However, this does not account for the implicitly pruned weights, i.e. those weights that, due to downstream pruning in following layers, are now disconnected from the output of the neural network. For this reason, they receive no gradient and their value never changes from that at initialization. They can technically take on any value, including 0, without affecting the output of the neural network. In other words, they could effectively be pruned without any loss of performance. If we include these weights in the effective fraction of pruned weights, structured pruning begins to appear competitive, especially at high pruning fractions, because its effective sparsity naturally tends to be higher. This also suggests that since the effective pruning rate per iteration is higher than 20% for structured pruning techniques, one could experiment with lowering it, to avoid aggressive pruning and consequent loss in performance.

![](./images/867745243288568210_11.jpg)
(a) AlexNet on CIFAR-10

![](./images/867745243288568210_12.jpg)
(b) AlexNet on MNIST

![](./images/867745243288568210_13.jpg)
(c) AlexNet on SVHN

![](./images/867745243288568210_14.jpg)
(d) ResNet18 on CIFAR-100

![](./images/867745243288568210_15.jpg)
(e) LeNet on CIFAR-100

![](./images/867745243288568210_16.jpg)
(f) LeNet on CIFAR-10

![](./images/867745243288568210_17.jpg)
(g) LeNet on MNIST

![](./images/867745243288568210_18.jpg)
(h) LeNet on KMNIST

![](./images/867745243288568210_19.jpg)
(i) LeNet on SVHN

Figure 8: Test accuracy for SGD-trained, sparsified models, pruned using seven different pruning techniques ($L_2$-structured, global unstructured, $L_1$-unstructured, $L_2$-structured $L_\infty$-structured, random structured, and random unstructured), and rewound to initial weight values after each pruning iteration.

![](./images/867745243288568210_20.jpg)

Figure 9: Layer-wise Jaccard distance between masks found by pruning AlexNet on MNIST and them rewind weights, and those found by finetuning after pruning, as a function of the pruning iteration, conditional on identical seed, across six seeds, and identical pruning technique, for $L_2$-structured, global unstructured, $L_1$-unstructured, $L_1$-structured, $L_\infty$-structured, random structured, random unstructured pruning.


Table 1: Number of parameters (in thousands) left in each model at each pruning fraction. Columns labeled as $s$ corresponds to the number of parameters that are not explicitly zero-ed out by the pruning mask in the network; $s_{\text{eff}}$ corresponds to the number of non-disconnected parameters.

<table>
<thead>
  <tr>
    <th rowspan="2">Pruning Iteration</th>
    <th colspan="2">LeNet</th>
    <th colspan="2">AlexNet</th>
    <th colspan="2">VGG11</th>
    <th colspan="2">ResNet18</th>
  </tr>
  <tr>
    <th>$s$</th>
    <th>$s_{\text{eff}}$</th>
    <th>$s$</th>
    <th>$s_{\text{eff}}$</th>
    <th>$s$</th>
    <th>$s_{\text{eff}}$</th>
    <th>$s$</th>
    <th>$s_{\text{eff}}$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>0</td>
    <td>48</td>
    <td>48</td>
    <td>45,637</td>
    <td>45,637</td>
    <td>103,048</td>
    <td>103,042</td>
    <td>8,981</td>
    <td>8,981</td>
  </tr>
  <tr>
    <td>1</td>
    <td>38</td>
    <td>38</td>
    <td>36,512</td>
    <td>36,512</td>
    <td>82,440</td>
    <td>82,430</td>
    <td>7,221</td>
    <td>7,221</td>
  </tr>
  <tr>
    <td>2</td>
    <td>30</td>
    <td>30</td>
    <td>29,211</td>
    <td>29,211</td>
    <td>65,954</td>
    <td>65,942</td>
    <td>5,813</td>
    <td>5,813</td>
  </tr>
  <tr>
    <td>3</td>
    <td>24</td>
    <td>24</td>
    <td>23,371</td>
    <td>23,371</td>
    <td>52,765</td>
    <td>52,718</td>
    <td>4,687</td>
    <td>4,687</td>
  </tr>
  <tr>
    <td>4</td>
    <td>19</td>
    <td>19</td>
    <td>18,698</td>
    <td>18,692</td>
    <td>42,214</td>
    <td>42,126</td>
    <td>3,786</td>
    <td>3,786</td>
  </tr>
  <tr>
    <td>5</td>
    <td>15</td>
    <td>15</td>
    <td>14,960</td>
    <td>14,945</td>
    <td>33,774</td>
    <td>33,610</td>
    <td>3,065</td>
    <td>3,065</td>
  </tr>
  <tr>
    <td>6</td>
    <td>12</td>
    <td>12</td>
    <td>11,970</td>
    <td>11,937</td>
    <td>27,021</td>
    <td>26,800</td>
    <td>2,488</td>
    <td>2,488</td>
  </tr>
  <tr>
    <td>7</td>
    <td>10</td>
    <td>10</td>
    <td>9,578</td>
    <td>9,505</td>
    <td>21,619</td>
    <td>21,357</td>
    <td>2,027</td>
    <td>2,027</td>
  </tr>
  <tr>
    <td>8</td>
    <td>8.3</td>
    <td>8.3</td>
    <td>7,664</td>
    <td>7,549</td>
    <td>17,297</td>
    <td>17,005</td>
    <td>1,658</td>
    <td>1,658</td>
  </tr>
  <tr>
    <td>9</td>
    <td>6.7</td>
    <td>6.6</td>
    <td>6,133</td>
    <td>5,959</td>
    <td>13,840</td>
    <td>13,549</td>
    <td>1,362</td>
    <td>1,362</td>
  </tr>
  <tr>
    <td>10</td>
    <td>5.4</td>
    <td>5.4</td>
    <td>4,908</td>
    <td>4,649</td>
    <td>11,074</td>
    <td>10,792</td>
    <td>1,126</td>
    <td>1,126</td>
  </tr>
  <tr>
    <td>11</td>
    <td>4.3</td>
    <td>4.3</td>
    <td>3,928</td>
    <td>3,607</td>
    <td>8,861</td>
    <td>8,608</td>
    <td>937</td>
    <td>937</td>
  </tr>
  <tr>
    <td>12</td>
    <td>3.5</td>
    <td>3.5</td>
    <td>3,144</td>
    <td>2,810</td>
    <td>7,091</td>
    <td>6,873</td>
    <td>786</td>
    <td>786</td>
  </tr>
  <tr>
    <td>13</td>
    <td>2.9</td>
    <td>2.8</td>
    <td>2,517</td>
    <td>2,199</td>
    <td>5,675</td>
    <td>5,492</td>
    <td>665</td>
    <td>665</td>
  </tr>
  <tr>
    <td>14</td>
    <td>2.3</td>
    <td>2.3</td>
    <td>2,016</td>
    <td>1,745</td>
    <td>4,542</td>
    <td>4,399</td>
    <td>568</td>
    <td>567</td>
  </tr>
  <tr>
    <td>15</td>
    <td>1.9</td>
    <td>1.9</td>
    <td>1,614</td>
    <td>1,401</td>
    <td>3,636</td>
    <td>3,523</td>
    <td>491</td>
    <td>490</td>
  </tr>
  <tr>
    <td>16</td>
    <td>1.6</td>
    <td>1.5</td>
    <td>1,293</td>
    <td>1,142</td>
    <td>2,911</td>
    <td>2,825</td>
    <td>429</td>
    <td>427</td>
  </tr>
  <tr>
    <td>17</td>
    <td>1.3</td>
    <td>1.2</td>
    <td>1,036</td>
    <td>950</td>
    <td>2,331</td>
    <td>2,260</td>
    <td>379</td>
    <td>377</td>
  </tr>
  <tr>
    <td>18</td>
    <td>1.1</td>
    <td>1.0</td>
    <td>831</td>
    <td>812</td>
    <td>1,867</td>
    <td>1,807</td>
    <td>340</td>
    <td>337</td>
  </tr>
  <tr>
    <td>19</td>
    <td>0.9</td>
    <td>0.8</td>
    <td>666</td>
    <td>663</td>
    <td>1,495</td>
    <td>1,443</td>
    <td>308</td>
    <td>303</td>
  </tr>
</tbody>
</table>

The number of parameters corresponding to various fractions of pruned weights in the models investigated in this work is listed in Table 1.

## D Lottery Tickets from different pruning techniques

Fig. 10 shows an additional set of plots of the average Jaccard distance between the masks of lottery tickets found by different pruning techniques, this time for AlexNet trained on MNIST.

The structure of the masks generated by different pruning techniques begins to diverge rapidly in the early phases of pruning, and approaches a total Jaccard distance of 1 (no intersection over masks) as sparsity is increased towards 100%. All distances are measured with respect to the mask generated by $L_2$ structured pruning, which is used as a baseline. For pairwise distances among masks over training iterations, please refer to Fig. 4.

### D.1 Complementarity of Learned Solutions

The dissimilarity of solutions can be explored by looking at the heat maps of the agreement in average class prediction across LTs obtained through different pruning techniques. Fig. 11 provides these visualizations for the $19^{\text{th}}$ pruning iteration.

## E Generating prêt-à-porter lottery tickets

The prêt-à-porter tickets considered in this work were sourced from low sparsity bespoke tickets obtained on five 10-class classification tasks. Specifically, we used the bespoke tickets obtained via global unstructured pruning with rewinding, after only 2 regular pruning iterations. For LeNet tickets, for example, this corresponded to a bespoke ticket of 35.9% sparsity. The intersection of the masks of these five bespoke tickets yielded a mask $\mathbb{M}$ of sparsity $\sim$85%, which is comparable to the sparsity

![](./images/867745243288568210_21.jpg)

Figure 10: Layer-wise Jaccard distance between masks found by pruning AlexNet on MNIST using $L_2$-structured pruning, and masks found by other pruning methods (global unstructured, $L_1$-unstructured, $L_1$-structured, $L_\infty$-structured, random structured, random unstructured), conditioned on identical seed, across six seeds. $L_1$-structured pruning yields the most similar masks to $L_2$-structured pruning, as expected.

![](./images/867745243288568210_22.jpg)

(a) $19^{\text{th}}$ pruning iteration

Figure 11: Number of examples in the MNIST test set over which the sub-networks obtained through each pruning technique agree on the prediction, on average (over 6 experimental seeds).

![](./images/867745243288568210_23.jpg)

(a) Pruning iteration 1

![](./images/867745243288568210_24.jpg)

(b) Pruning iteration 6

Figure 12: Number of AlexNet weights that remain unpruned at each pruning iteration across 0 through 5 tasks (MNIST, FashionMNIST, KMNIST, CIFAR-10, SVHN) for each pruning technique: $L_2$-structured, global unstructured, $L_1$-unstructured, $L_\infty$-structured, random structured, and random unstructured.

of a bespoke ticket after 9 successive pruning iterations. By sourcing the five base tickets in parallel after only two pruning iterations, we effectively save the elapsed real time equivalent of 7 additional, sequential pruning iterations, without, however, any advantage in CPU time.

If a higher sparsity prêt-à-porter ticket was needed, one could have taken the intersection of bespoke ticket masks from a later pruning iteration.

Fig. 12 shows the number of parameters that remain unpruned in 0 through 5 of the chosen tasks for different kinds of pruning techniques, and across pruning iterations. Different color bars represent networks pruned through different pruning techniques. Error bars correspond to one standard deviation on the number of weights that remain unpruned across the six experimental seeds used in this work. Of the pruning techniques tested, global unstructured pruning is the most consistent one in identifying similar sub-networks within a larger network across a variety of image classification tasks, if we exclude the random pruning technique, which are deterministically random. As a sanity check, random pruning techniques (both structured and unstructured) consistently prune the same (random) set of weights across all 5 tasks when the seed is kept constant, because, indeed, the choice of weights to prune is solely determined by the random seed and it is not task-dependent. A weight is then either always dropped (in bin 0) or never dropped (in bin 5).

Fig. 13 shows the number and fraction of weights in each VGG11 layer that are selected by consensus to create the prêt-à-porter lottery ticket. The fraction of weights that is constantly unpruned across all five tasks rapidly declines as a function of the layer size; nonetheless, the largest initial layers remain largest in size even in the prêt-à-porter ticket. Similar plots are available for AlexNet in Fig. 14, for ResNet18 in Fig. 15, and for Lenet in Fig. 16.

Fig. 17, 18, and 19 show the distribution of weights at initialization in ResNet18, VGG11, and LeNet networks, compared to the weights at initialization in the following subsets: parameters in the prêt-à-porter lottery ticket; parameters in a CIFAR-10 bespoke ticket; parameters in an MNIST bespoke ticket. All tickets were obtained via global unstructured pruning.

The distributions of initial values of the parameters that make up the prêt-à-porter lottery tickets show a stronger tendency towards smaller magnitude weights, compared to bespoke tickets. This is more explicitly quantified in Fig. 20, where, for each architecture considered in this work, we compute the average empirical standard deviation of weights at initialization in each layer. In gold, we mark the empirical standard deviation of the full weight distribution at initialization prior to any pruning; in grey, we mark the standard deviation of our prêt-à-porter ticket at initialization; in blue, for the CIFAR-10 bespoke ticket; in orange, for the MNIST bespoke ticket. This larger distribution variance at initialization observed in bespoke tickets is due to the explicit choice of removing low

![](./images/867745243288568210_25.jpg)
(a) Number of weights per layer in the prêt-à-
porter ticket

![](./images/867745243288568210_26.jpg)
(b) Fraction of weights per layer in the prêt-à-
porter ticket

Figure 13: Prêt-à-porter lottery ticket composition for VGG11.

![](./images/867745243288568210_27.jpg)
(a) Number of weights per layer in the prêt-à-porter
ticket

![](./images/867745243288568210_28.jpg)
(b) Fraction of weights per layer in the prêt-à-
porter ticket

Figure 14: Prêt-à-porter lottery ticket composition for AlexNet.

![](./images/867745243288568210_29.jpg)
(a) Number of weights per layer in the prêt-à-porter
ticket

![](./images/867745243288568210_30.jpg)
(b) Fraction of weights per layer in the prêt-à-
porter ticket

Figure 15: Prêt-à-porter lottery ticket composition for ResNet18.

![](./images/867745243288568210_31.jpg)

(a) Number of weights per layer in the prêt-à-
porter ticket

![](./images/867745243288568210_32.jpg)

(b) Fraction of weights per layer in the prêt-à-
porter ticket

Figure 16: Prêt-à-porter lottery ticket composition for LeNet.

magnitude weights at each iteration of pruning, over 20 pruning iterations, in the LT-finding algorithm that selects bespoke tickets. In the case of prêt-à-porter LTs, instead, the algorithm we propose only explicitly removes the lowest magnitude weights over the two magnitude-based pruning iterations applied to the $N=5$ source tickets. The remainder of the sparsification procedure does not explicitly select out parameters that correlate with low absolute value at initialization.

In both bespoke and prêt-à-porter LTs, weights that make up lottery tickets are distributed according to a symmetric bimodal distribution at initialization.

## F Lottery Ticket performance at initialization

In the spirit of prior work on Lottery Tickets and supermasks [59], we evaluate the performance of bespoke and prêt-à-porter lottery tickets at initialization, prior to any training taking place. We are unsuccessful at extracting tickets with better-than-random performance for any of the task-architecture combinations considered in this work, with the exception of bespoke tickets sourced on large, over-parametrized, and often parameter-inefficient networks such as AlexNet and VGG11 (Fig. 21). This allows us to notice that, while bespoke tickets are flexible enough to be successfully retrained on target tasks, they do have source dataset-specific properties that allow them to align with a good solution on their source domain even at initialization. This specialization, however, doesn't prevent them from transferring, and does not result in large performance gains post-training.

Due to problems encountered with job submission, certain lottery ticket / target dataset pairs are not present for all six seeds. For VGG11, one seed is present for "Bespoke sourced on fashionmnist" and "Bespoke sourced on fashionmnist" evaluated on CIFAR-100. For ResNet18, Only five of six seeds are available for ResNet18 "Bespoke sourced on svhn" for MNIST and EMNIST and for "Intersection LT" for CIFAR-100; four of six seeds are available for "Bespoke sourced on svhn" all other target datasets. Results for random unstructured on EMNIST are not available on ResNet18.

## G Lottery ticket performance on held out datasets

We report the performance for the bespoke lottery tickets and prêt-à-porter lottery tickets on two additional tasks not used in training, CIFAR-100 and EMNIST, Fig. 22. Due to problems encountered with job submission, results are reported for two out of six seeds for "Bespoke sourced on cifar-100" on EMNIST and are missing for "Bespoke sourced on svhn" on EMNIST.

![](./images/867745243288568210_33.jpg)

Figure 17: Initialization values of weights in a ResNet18 prêt-à-porter lottery ticket, compared to those that make up a bespoke LT of similar sparsity sourced on CIFAR-10, and those that make up a bespoke LT of similar sparsity sourced on MNIST. For comparison, the full distribution of initial weights is shown as a shaded gold histogram.

## H Further experimental details for reproducibility

### H.1 Models

The models trained and pruned in this work correspond to the following base architectures:

- LeNet, with two convolutional layers and three fully connected layers. The convolutional layers perform $3 \times 3$ convolutions with 6 and 16 output channels, respectively. The fully connected layers transform the hidden representation from 400-dimensional to 120-d, 84-d, and 10-d at the output layer, for 10 class classification problems. Wherever the dataset requires a larger number of outputs, the final layer is replaced with a fully-connected layer of dimensions matching the number of output classes in the task. Rectified linear units are used as activations throughout. Two MaxPool operations are executed after the first two non-linear activations.
- AlexNet, as implemented in `torchvision`
- VGG11, as implemented in `torchvision`
- ResNet18, as implemented in `torchvision`

![](./images/867745243288568210_34.jpg)

Figure 18: Initialization values of weights in a VGG11 prêt-à-porter lottery ticket, compared to those that make up a bespoke LT of similar sparsity sourced on CIFAR-10, and those that make up a bespoke LT of similar sparsity sourced on MNIST. For comparison, the full distribution of initial weights is shown as a shaded gold histogram.

![](./images/867745243288568210_35.jpg)

Figure 19: Initialization values of weights in a LeNet prêt-à-porter lottery ticket, compared to those that make up a bespoke LT of similar sparsity sourced on CIFAR-10, and those that make up a bespoke LT of similar sparsity sourced on MNIST. For comparison, the full distribution of initial weights is shown as a shaded gold histogram.

![](./images/867745243288568210_36.jpg)

(a) ResNet18

![](./images/867745243288568210_37.jpg)

(b) AlexNet

![](./images/867745243288568210_38.jpg)

(c) LeNet

![](./images/867745243288568210_39.jpg)

(d) VGG11

Figure 20: Standard deviation of weight at initialization in: the whole layer, the fraction of weights
in the layer that make up the prêt-à-porter ticket, the fraction of weights that makes up the bespoke
CIFAR-10 ticket, and the fraction of weights that makes up the bespoke MNIST ticket.

![](./images/867745243288568210_40.jpg)

Figure 21: Performance of bespoke and prêt-à-porter lottery tickets obtained through global unstructured pruning from different base architectures (see each panel) at initialization. The source of the tickets can be found in the legend. The x-axis labels the target dataset on which the tickets were evaluated. These tickets correspond to the ones that were then retrained and evaluated to generate the results in Fig. 6.

![](./images/867745243288568210_41.jpg)

Figure 22: Performance of bespoke and prêt-à-porter lottery tickets obtained through global unstructured pruning from different base architectures (see each panel). The source of the tickets can be found in the legend. The x-axis labels the target dataset on which the tickets were evaluated. These tickets correspond to the ones that were then retrained and evaluated to generate the results in Fig. 6.

All networks are trained from scratch, with each layer initialized with the default PyTorch initialization schema.

### H.2 Pruning
Given a tensor $\mathbf{T}$, representing the unpruned parameters in a layer:
- **11_unstructured** pruning removes the entries $\left\{ T_k : k_{\mathrm{th}}\left(\left\{T_i\right\}_{i=1}^{|\mathbf{T}|}\mid\|\cdot\|_1\right) < 0.2 \right\}$, where $k_{\mathrm{th}}$ is the rank of the $k^{\mathrm{th}}$ element of $\left\{T_i\right\}_{i=1}^{|\mathbf{T}|}$, according to the $L_1$-based ranking, and $t=0.2$ is the pruning fraction per iteration used in this work;
- **11_structured** pruning removes all entries in channel $k$ along dimension $d$ defined as $\left\{ \mathbf{T}_{[...,k,...]} : k_{\mathrm{th}}\left(\left\{\mathbf{T}_{[...,i,...]}\right\}_{i=1}^{|d|}\mid\|\cdot\|_1\right) < 0.2 \right\}$, where $k_{\mathrm{th}}$ is the rank of the $k^{\mathrm{th}}$ channel of $\mathbf{T}$ along direction $d$ according to the $L_1$-based ranking, and $t=0.2$ is the pruning fraction per iteration used in this work;
- **12_structured** pruning removes all entries in channel $k$ along dimension $d$ defined as $\left\{ \mathbf{T}_{[...,k,...]} : k_{\mathrm{th}}\left(\left\{\mathbf{T}_{[...,i,...]}\right\}_{i=1}^{|d|}\mid\|\cdot\|_2\right) < 0.2 \right\}$, where $k_{\mathrm{th}}$ is the rank of the $k^{\mathrm{th}}$ channel of $\mathbf{T}$ along direction $d$ according to the $L_2$-based ranking, and $t=0.2$ is the pruning fraction per iteration used in this work;
- **linfty_structured** pruning removes all entries in channel $k$ along dimension $d$ defined as $\left\{ \mathbf{T}_{[...,k,...]} : k_{\mathrm{th}}\left(\left\{\mathbf{T}_{[...,i,...]}\right\}_{i=1}^{|d|}\mid\|\cdot\|_\infty\right) < 0.2 \right\}$, where $k_{\mathrm{th}}$ is the rank of the $k^{\mathrm{th}}$ channel of $\mathbf{T}$ along direction $d$ according to the $L_\infty$-based ranking, and $t=0.2$ is the pruning fraction per iteration used in this work;
- **random_structured** pruning removes all entries in channel $k$ along dimension $d$ chosen at random: $\left\{ \mathbf{T}_{[...,k,...]} : k = \mathrm{randint}(1,|d|,0.2*|d|) \right\}$, where $t=0.2$ is the pruning fraction per iteration used in this work;
- **random_unstructured** pruning removes entries $\left\{ T_k : k = \mathrm{randint}(1,|\mathbf{T}|,0.2*|\mathbf{T}|) \right\}$, where $t=0.2$ is the pruning fraction per iteration used in this work.

Given a tensor $\mathbf{U}$, representing the unpruned parameters in a set of layers:
- **global_unstructured** pruning removes the entries $\left\{ U_k : k_{\mathrm{th}}\left(\left\{U_i\right\}_{i=1}^{|\mathbf{U}|}\mid\|\cdot\|_1\right) < 0.2 \right\}$, where $k_{\mathrm{th}}$ is the rank of the $k^{\mathrm{th}}$ element of $\left\{U_i\right\}_{i=1}^{|\mathbf{U}|}$, according to the $L_1$-based ranking, and $t=0.2$ is the pruning fraction per iteration used in this work;

### H.3 Prêt-à-porter lottery ticket finding algorithm
Prêt-à-porter lottery ticket finding is elaborated upon in Alg. 1.

```
Algorithm 1: Prêt-à-porter (consensus) lottery ticket finding algorithm
Data: Set of datasets $\mathcal{S}$ on which to source ticket, a network with weights $\theta$, pruning iterations $T$,
     epochs for iteration $E$.
Result: A consensus mask $\mathbb{M}$
$\mathbb{M} = I_{|\theta|}$ ;
for dataset $S \in \mathcal{S}$ do
    $\theta_S = \theta$ ;
    for $t \leftarrow 1$ to $T$ do
        for $e \leftarrow 1$ to $E$ do
            $\theta_{\text{trained}} = \text{TrainEpoch}(\theta_S)$ ;
        end
        $\theta_{\text{pruned}} = \text{Prune}(\theta_{\text{trained}})$ ;
        $\theta_S = \text{Reinit}(\theta_{\text{pruned}})$ ;
    end
    $\mathbb{M} = \mathbb{M} \cap \text{GetMask}(\theta_S)$ ;
end
return $\mathbb{M}$ ;
```

<table>
<caption>Table 2: Specifics of Datasets used in Experiments</caption>
<thead>
<tr>
<th>Dataset</th>
<th>Training Examples</th>
<th>Testing Examples</th>
<th>Citation</th>
</tr>
</thead>
<tbody>
<tr>
<td>MNIST</td>
<td>60,000</td>
<td>10,000</td>
<td>[32]</td>
</tr>
<tr>
<td>Kuzushiji-MNIST</td>
<td>60,000</td>
<td>10,000</td>
<td>[9, 27]</td>
</tr>
<tr>
<td>Fashion-MNIST</td>
<td>60,000</td>
<td>10,000</td>
<td>[51]</td>
</tr>
<tr>
<td>CIFAR-10</td>
<td>60,000</td>
<td>10,000</td>
<td>[30]</td>
</tr>
<tr>
<td>CIFAR-100</td>
<td>60,000</td>
<td>10,000</td>
<td>[30]</td>
</tr>
<tr>
<td>SVHN</td>
<td>73,257</td>
<td>26,032</td>
<td>[41]</td>
</tr>
</tbody>
</table>

### H.4 Data

Seven datasets are used in experiments: MNIST, KMNIST, FashionMNIST, CIFAR-10, CIFAR-100, SVHN. Table 2 shows the sizes of each dataset, prior to training set splitting. The training set is, in fact, randomly split into a proper training set (80% of the original training set) and a validation set (20% of the original training set), using `torch.utils.data.random_split`. The random split is controlled by the same experimental seed that makes the rest of the experiments reproducible.

All images are normalized to the training set statistics. At train time, images are subjected to data-augmentation via a horizontal flip and a random crop. All images are resized to fit the expect input size of the model they are being processed with. In practice, when images are fed into a ResNet, VGG, or AlexNet model, they are resizes to size $(224, 224, 3)$. When they are processed by a LeNet architecture, they are reshaped to size $(28, 28, 1)$.
