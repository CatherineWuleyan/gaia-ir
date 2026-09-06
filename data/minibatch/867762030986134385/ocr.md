# Considering Layerwise Importance in the Lottery Ticket Hypothesis

Benjamin Vandersmissen* José Oramas

IDLab, University of Antwerp - imec
Prinsstraat 13, 2000 Antwerp, Belgium

February 2023

## Abstract
The Lottery Ticket Hypothesis (LTH) showed that by iteratively training a model, removing connections with the lowest global weight magnitude and rewinding the remaining connections, sparse networks can be extracted. This global comparison removes context information between connections within a layer. Here we study means for recovering some of this layer distributional context and generalise the LTH to consider weight importance values rather than global weight magnitudes. We find that given a repeatable training procedure, applying different importance metrics leads to distinct performant lottery tickets with little overlapping connections. This strongly suggests that lottery tickets are not unique.

Training a single transformer model with a parameter count of 213 millon — still orders of magnitude smaller than GPT-3 — using Neural Architecture Search emits as much CO2 as five cars during their lifetime [3]. Furthermore larger models typically need specialised hardware and large amounts of computing power for training and inference, which constrain the ability of the model from running on lighter devices, thus limiting powerful models to well-funded institutions. Finally, research has shown that these large models are typically overparameterized and encode a lot of redundant information that can be removed [4].

To alleviate these issues, numerous approaches have been studied to scale down the number of parameters in a model, while still preserving (roughly) the same performance. This can be achieved by, e.g., designing parameter-efficient network structures [5], or sparsifying existing neural network structures via pruning [6, 7, 8, 9, 10].

Until recently, it was thought to be difficult to train sparse neural networks from scratch [11], which was further strengthened by the finding that overparameterized network architectures are proven to lead to an optimal global minimum when training [12]. As such the classical way to reduce the parameter count was via the train-prune-finetune loop, in which a model is first trained to completion, then redundant connections are pruned and finally the resulting network is finetuned

The Lottery Ticket Hypothesis (LTH) [13] challenged this notion and introduced a procedure to extract a sparse trainable network — a lottery ticket

## 1 Introduction
The recent trend in machine learning to chase higher benchmark scores by adding additional parameters, has led to an explosive increase in the size of neural network architectures. A prime example of this phenomenon are the GPT models. While the first model in the family [1] has 117 million parameters, the latest model [2] already has a whopping 175 billion parameters which amounts to a $>1000$ times increase. However, this explosive rise in parameters poses new problems.

*Primary author: benjamin.vandersmissen@uantwerpen.be

1

(LT) — from a dense network. This procedure uses a pruning criterion in the form of the global weight magnitude in combination with a gradual pruning procedure to iteratively remove connections from the initial network.

In this paper we study a refinement on this criterion by adding a notion of layerwise importance, which we introduce in section 2. We do this by considering a number of different weight rescaling methods, such that the comparisons are more calibrated across layers. Quantitative and qualitative comparisons between different importance measures and the baseline are reported in section 3. In addition, we shine light on the observable differences in the generated LTs (section 4), and determine how LTs emerge and differ when considering identical training conditions (section 5). A brief overview of related work is laid out in section 6 and finally, the paper is concluded in section 7.

The key observations of our study are that: i) given a fixed weight initialization, it is possible to extract different lottery tickets that have similar performance, but differ significantly in their structure, ii) these tickets have a noticeable amount of common connections which have low-variance across tickets, and iii) these stable common connections survive the LTH procedure even when the other weights in the model are reinitialized. Together these observations suggest that these connections might be a promising avenue towards finding LTs more efficiently.

## 2 The Lottery Ticket Hypothesis

The Lottery ticket hypothesis uses iterative Global Magnitude Pruning (GMP) [8], which prunes individual connections that have the lowest weight magnitudes in a network. By repeating this process multiple times, it is possible to obtain a highly sparse network that when trained still reaches commensurate accuracy.

Later work by [14] introduced a modification to the LTH procedure by rewinding to parameters at iteration $t = k \ll m$, rather than resetting to the initial parameters at $t=0$. By rewinding to a later iteration, the performance of the found lottery tickets was improved for complex networks at high sparsities. In the literature, this procedure is usually referred to as Lottery Ticket Rewinding (LTR) (See Algorithm 1) rather than the LTH. We will adopt this naming in the rest of the document.

While the exact mechanism behind the success of lottery tickets is not fully understood, one hypothesis, posited by [15], is that due to the LTH/LTR procedure the resulting networks are already in the same loss basin as the fully trained dense network and as such the ticket can still converge to a performant solution during training.

```
Algorithm 1 The LTR procedure
1: Initialize a model $M$ with parameters $\theta_0$
2: Pretrain $M$ for k iterations resulting parameters
    $\theta_{0,k}$
3: for $i \leftarrow 0,n$ do
4:   Train $M$ for m-k iterations, resulting in param-
      eters $\theta_{i,m}$
5:   $\theta_{pooled} \leftarrow Pool(abs(\theta_{i,m}))$
6:   $p \leftarrow$ j-th percentile of $\theta_{pooled}$
7:   Prune all connections with $abs(\theta_{i,m}) < p$
8:   Rewind parameters of $M$ to $\theta_{0,k}$
9: end for
```

A weak aspect of globally pruning is that the only factor that determines whether a connection is pruned, is the magnitude of the connection weight. As such, connection weights from different layers are compared on a global scale, rather than within the layer. This disregards more complex factors such as the weight distribution within a layer and the number of remaining connections in the layer. In fact, due to the commonly used Kaiming Normal initialization [16], different layers are already initialized at different weight distributions as the standard deviation is inversely proportional to the size of the layer, and as illustrated in Table 1, they also converge to different distributions. However, it should be noted that this difference in standard deviation actually already introduces a small implicit bias in the weights such that the weights are somewhat dependent on the size of

<table>
<thead>
  <tr>
    <th></th>
    <th>Initialised (0 it)</th>
    <th>Pretrained (1000 it)</th>
    <th>Fully Trained (78200 it)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Conv1</th>
    <td>[-0.47, 0.41]</td>
    <td>[-0.57, 0.78]</td>
    <td>[-1.07, 0.90]</td>
  </tr>
  <tr>
    <th>Downsample1</th>
    <td>[-0.65, 0.66]</td>
    <td>[-0.64, 0.63]</td>
    <td>[-1.01, 1.11]</td>
  </tr>
  <tr>
    <th>Conv17</th>
    <td>[-0.11, 0.10]</td>
    <td>[-0.12, 0.10]</td>
    <td>[-0.15, 0.19]</td>
  </tr>
  <tr>
    <th>FC</th>
    <td>[-0.30, 0.27]</td>
    <td>[-0.29, 0.39]</td>
    <td>[-0.45, 1.19]</td>
  </tr>
</tbody>
</table>

Table 1: Weight distributions [minimum and maximum] of selected layers in a ResNet-18 network at different steps in the training procedure.

the layer. Using importance metrics makes this bias both more explicit and stronger, dependent on which metric is used.

Disregarding these previously-mentioned factors might come at the cost of accuracy loss in the re- sulting ticket. In the worst case this can even lead to a phenomenon called 'layer-collapse' [17], a criti- cal failure case of neural network pruning in which information can no longer flow from the input layer to the output layer of the network. This finds its origin in one or more intermediate layers being completely destroyed, thus rendering the resulting predictions invalid.

Proposal. Taking these issues into account, we propose a modification of the LTR procedure where the importance of connection within a given layer is represented by an importance value assigned to it. This importance score is still dependent on the weight magnitude, thus preserving the core concept of the LTH, but also considers the weight magnitudes of all other unpruned connections within the same layer. In doing so, an additional cue of layer distribution is injected in the pruning process. Intuitively, this should alleviate the issue with layer-collapse which has a negative impact on the performance of the LTR procedure. Furthermore, it is possible to view this modification as a generalization of the LTR proce- dure, where the purely magnitude-based approach uses the identity mapping to calculate importance values. The only computational overhead incurred in our method is the calculation of importance scores, which is negligible. A pseudo code overview of the modified procedure can be found in Algorithm 2.

In the same context of layerwise importance cal- culation, [18] introduced a metric called the LAMP score, which follows a model-level distortion minimiza- tion perspective by calculating the $L_{2}$ norm of the connections w.r.t. the other unpruned connections in the layer. The authors also demonstrated that this results in more performant lottery tickets at a very high sparsity levels.

```
Algorithm 2 The modified LTR procedure
 1: Initialize a model M with weights θ₀
 2: Pretrain M for k iterations resulting in θ₀,k
 3: for i ← 1, n do
 4:   Train M for (m-k) iterations, resulting in θᵢ,m
 5:   scoreᵢ ← layerwise-importance(abs(θᵢ,m))
 6:   score_pooled ← Pool(scoreᵢ)
 7:   p ← j-th percentile of score_pooled
 8:   Prune all connections with abs(scoreᵢ) < p
 9:   Rewind parameters to θ₀,k
10: end for
```

Additionally, as also noticed in [18], we can view the use of importance scores in the LTR procedure as an heuristic to calculate layerwise pruning ratios in the context of local pruning rather than global pruning.

### 2.1 Considered Importance Metrics

To calculate a layerwise importance value for a given connection at position $j$ in layer $i$, we need access to the weight matrix $\boldsymbol{W}_{i,:}$.

- $L_1$: $imp(i,j) = \frac{W_{i,j}}{\sum \boldsymbol{W}_{i,:}}$

- $L_2$: $imp(i,j) = \frac{W_{i,j}^2}{\sum \boldsymbol{W}_{i,:}^2}$ [18]

3

- Softmax: $imp(i,j) = \frac{exp(W_{i,j})}{\sum exp(\boldsymbol{W}_{i,:})}$

- Min-Max: $imp(i,j) = \frac{W_{i,j} - min(\boldsymbol{W}_{i,:})}{max(\boldsymbol{W}_{i,:}) - min(\boldsymbol{W}_{i,:})}$

The goal of the considered importance measures is to make weight values from different layers comparable. To accomplish this, there are two main approaches.

The first approach, as followed by $L_1$, $L_2$ and SoftMax is the sum-to-one principle, to rescale the importances of a layer such that the total importance of a layer is one. Then the importances themselves are still determined by (an operation on) the magnitude, but rescaled. By following this approach, connections will be more likely to be pruned if they exist in layers with more unpruned connections.

The second approach, followed by Min-Max normalization determines the importances such that each layer has the same lower and upper boundaries.

Importance Distributions. For the case of the softmax function, we notice that the importance distribution is very narrow and the lower bound is exactly $1/|W|$, while for the other functions the distribution is more broad with a lower bound of 0. This is due to a combination of two factors: (1) the Kaiming Normal initialization which has a mean of 0 and a standard deviation that is inversely proportional to the size of the layer and (2) the exponential function for which $e^x \approx 1$ if $|x| \ll 1$. Together this nullifies the dependence on the weights themselves almost completely in the softmax function when applied on large layers. $L_2$ is another measure that transforms the relevance distribution to be more narrower than the weight distribution due to the quadratic function. The other measures, $L_1$ and Min-Max, do not fundamentally impact the relevance distribution, only re-scale (and re-center) the distribution.

## 3 Experiments

Our experiments are based on the openlth library [19]. We follow the configurations specified in [20] unless otherwise stated. For a detailed description of the used training configuration, we refer the reader to subsection A.4. Additionally, we will make our code public at github-link-here to foster reproducibility of the reported results.

Networks. We adopt three different types of Neural Network architectures: Fully Connected Networks (LeNet-300-100 [21]), classical Convolutional Networks (VGG-16 [22]) and Residual Networks [23] (ResNet-18 & ResNet-20).

Datasets. We adopt three different datasets commonly used in the LTH/LTR literature: MNIST [24], CIFAR-10 [25] and TinyImageNet [26]. MNIST is used in conjunction with LeNet-300-100, CIFAR-10 is used with both VGG-16 and ResNet-20, and TinyImageNet is used with ResNet-18.

We consider the Lottery Ticket Hypothesis with weight rewinding (LTR) introduced in [14]. There it was shown that the inclusion of rewinding increased the performance of the procedure on all types of models and datasets, but more specifically on complexer datasets and models.

### 3.1 Quantitative evaluation

The goal of the initial experiments is to determine whether using different importance methods results in Lottery tickets with similar or better accuracy. To this end, we calculate the Top-1 accuracy results for Lottery Tickets at different sparsity levels. Results are listed in tabular form in Table 2 (averaged over 3 runs), or in graphical form in Appendix A.1.1. To enable comparison with other studies, we adopted the commonly used pruning ratio of 20%.

From these experiments, we can distill a few key observations on which we elaborate in the following sections. First, the magnitude, $L_1$ and $L_2$ criteria all produce LTs of similar performance, the only noticeable difference is in the VGG16 + CIFAR10 experiment. Second, Min-Max normalization and SoftMax seem to keep up with magnitude pruning during the initial pruning phase, but suffers during the later iterations. This seem to be the trend except in the simplest scenario (LeNet + MNIST), which we deem non-representative of larger settings, following [20].

Table 2: Average top-1 accuracy of different network & dataset combinations over 3 runs. **Boldfaced** results highlight the best result for a given amount of pruning steps.

### (a) LeNet-300-100 on MNIST
<table>
<thead>
<tr>
<th>Steps (<i>sparsity</i>)</th>
<th>Magnitude</th>
<th>$L_1$</th>
<th>$L_2$</th>
<th>SoftMax</th>
<th>MinMax</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dense network</td>
<td>98.11% ± 0.08</td>
<td>98.11% ± 0.08</td>
<td>98.11% ± 0.08</td>
<td>98.11% ± 0.08</td>
<td>98.11% ± 0.08</td>
</tr>
<tr>
<td>5 (<i>67.23%</i>)</td>
<td>98.00% ± 0.09</td>
<td><b>98.12% ± 0.10</b></td>
<td>98.04% ± 0.15</td>
<td>98.00% ± 0.08</td>
<td>98.02% ± 0.09</td>
</tr>
<tr>
<td>10 (<i>89.26%</i>)</td>
<td>98.08% ± 0.06</td>
<td><b>98.15% ± 0.05</b></td>
<td>98.06% ± 0.08</td>
<td>98.05% ± 0.08</td>
<td>98.04% ± 0.10</td>
</tr>
<tr>
<td>15 (<i>96.48%</i>)</td>
<td>97.94% ± 0.08</td>
<td><b>98.12% ± 0.15</b></td>
<td>97.95% ± 0.10</td>
<td>97.90% ± 0.05</td>
<td>97.77% ± 0.17</td>
</tr>
<tr>
<td>20 (<i>98.85%</i>)</td>
<td>97.23% ± 0.15</td>
<td>97.44% ± 0.02</td>
<td><b>97.47% ± 0.10</b></td>
<td>97.33% ± 0.17</td>
<td>97.41% ± 0.05</td>
</tr>
<tr>
<td>24 (<i>99.53%</i>)</td>
<td>90.62% ± 0.29</td>
<td>90.89% ± 0.22</td>
<td>91.05% ± 1.09</td>
<td>91.29% ± 0.79</td>
<td><b>91.82% ± 0.16</b></td>
</tr>
</tbody>
</table>

### (b) ResNet-20 on CIFAR-10
<table>
<thead>
<tr>
<th>Steps (<i>sparsity</i>)</th>
<th>Magnitude</th>
<th>$L_1$</th>
<th>$L_2$</th>
<th>SoftMax</th>
<th>MinMax</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dense network</td>
<td>91.67% ± 0.40</td>
<td>91.67% ± 0.40</td>
<td>91.67% ± 0.40</td>
<td>91.67% ± 0.40</td>
<td>91.67% ± 0.40</td>
</tr>
<tr>
<td>5 (<i>67.23%</i>)</td>
<td><b>92.01% ± 0.14</b></td>
<td>91.57% ± 0.20</td>
<td>91.65% ± 0.25</td>
<td>90.15% ± 0.92</td>
<td>91.64% ± 0.13</td>
</tr>
<tr>
<td>10 (<i>89.26%</i>)</td>
<td><b>90.90% ± 0.15</b></td>
<td>90.56% ± 0.22</td>
<td>90.86% ± 0.39</td>
<td>87.75% ± 0.33</td>
<td>90.51% ± 0.05</td>
</tr>
<tr>
<td>15 (<i>96.48%</i>)</td>
<td>85.59% ± 1.00</td>
<td>84.76% ± 0.62</td>
<td><b>86.28% ± 0.45</b></td>
<td>84.00% ± 0.54</td>
<td>83.28% ± 0.32</td>
</tr>
<tr>
<td>20 (<i>98.85%</i>)</td>
<td><b>77.07% ± 1.57</b></td>
<td>76.31% ± 0.71</td>
<td>76.68% ± 0.63</td>
<td>76.37% ± 1.06</td>
<td>74.36% ± 0.57</td>
</tr>
<tr>
<td>25 (<i>99.62%</i>)</td>
<td>63.36% ± 0.55</td>
<td><b>63.40% ± 0.77</b></td>
<td>62.78% ± 0.46</td>
<td>63.07% ± 0.55</td>
<td>50.17% ± 6.78</td>
</tr>
<tr>
<td>30 (<i>99.88%</i>)</td>
<td><b>45.66% ± 2.96</b></td>
<td>45.36% ± 1.49</td>
<td>43.65% ± 0.80</td>
<td>38.42% ± 5.46</td>
<td>31.50% ± 1.80</td>
</tr>
<tr>
<td>35 (<i>99.96%</i>)</td>
<td><b>22.58% ± 4.57</b></td>
<td>21.60% ± 4.79</td>
<td>19.59% ± 6.37</td>
<td>17.75% ± 3.20</td>
<td>14.62% ± 4.00</td>
</tr>
<tr>
<td>40 (<i>99.99%</i>)</td>
<td>12.87% ± 4.97</td>
<td>10.00% ± 0.00</td>
<td>12.45% ± 3.55</td>
<td>12.29% ± 2.00</td>
<td><b>13.34% ± 2.89</b></td>
</tr>
</tbody>
</table>

### (c) ResNet-18 on TinyImageNet
<table>
<thead>
<tr>
<th>Steps (<i>sparsity</i>)</th>
<th>Magnitude</th>
<th>$L_1$</th>
<th>$L_2$</th>
<th>SoftMax</th>
<th>MinMax</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dense network</td>
<td>49.53% ± 0.56</td>
<td>49.53% ± 0.56</td>
<td>49.53% ± 0.56</td>
<td>49.53% ± 0.56</td>
<td>49.53% ± 0.56</td>
</tr>
<tr>
<td>5 (<i>67.23%</i>)</td>
<td>49.99% ± 0.10</td>
<td><b>50.96% ± 0.19</b></td>
<td>50.45% ± 0.40</td>
<td><b>50.96% ± 0.18</b></td>
<td>50.38% ± 0.47</td>
</tr>
<tr>
<td>10 (<i>89.26%</i>)</td>
<td>49.98% ± 0.41</td>
<td><b>50.84% ± 0.65</b></td>
<td>50.49% ± 0.50</td>
<td>50.69% ± 0.73</td>
<td>50.24% ± 0.68</td>
</tr>
<tr>
<td>15 (<i>96.48%</i>)</td>
<td>48.72% ± 0.20</td>
<td><b>49.76% ± 0.56</b></td>
<td>49.41% ± 0.09</td>
<td>45.30% ± 4.42</td>
<td>47.77% ± 0.37</td>
</tr>
<tr>
<td>20 (<i>98.85%</i>)</td>
<td>46.36% ± 0.52</td>
<td><b>47.58% ± 0.24</b></td>
<td>47.26% ± 0.52</td>
<td>40.48% ± 3.52</td>
<td>43.49% ± 0.17</td>
</tr>
<tr>
<td>25 (<i>99.62%</i>)</td>
<td><b>38.54% ± 0.48</b></td>
<td>37.37% ± 0.35</td>
<td>37.90% ± 0.54</td>
<td>32.51% ± 2.57</td>
<td>34.23% ± 1.19</td>
</tr>
<tr>
<td>30 (<i>99.88%</i>)</td>
<td>24.56% ± 1.00</td>
<td>25.12% ± 0.71</td>
<td><b>25.88% ± 0.47</b></td>
<td>21.67% ± 1.40</td>
<td>23.34% ± 0.12</td>
</tr>
<tr>
<td>35 (<i>99.96%</i>)</td>
<td>13.56% ± 0.11</td>
<td>14.03% ± 0.61</td>
<td><b>15.30% ± 0.28</b></td>
<td>10.16% ± 2.17</td>
<td>12.74% ± 0.76</td>
</tr>
<tr>
<td>40 (<i>99.99%</i>)</td>
<td>4.62% ± 0.25</td>
<td>6.48% ± 0.30</td>
<td><b>6.83% ± 0.36</b></td>
<td>4.69% ± 0.72</td>
<td>5.31% ± 0.62</td>
</tr>
</tbody>
</table>

### (d) VGG16 on CIFAR-10
<table>
<thead>
<tr>
<th>Steps (<i>sparsity</i>)</th>
<th>Magnitude</th>
<th>$L_1$</th>
<th>$L_2$</th>
<th>SoftMax</th>
<th>MinMax</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dense network</td>
<td>93.52% ± 0.08</td>
<td>93.52% ± 0.08</td>
<td>93.52% ± 0.08</td>
<td>93.52% ± 0.08</td>
<td>93.52% ± 0.08</td>
</tr>
<tr>
<td>5 (<i>67.23%</i>)</td>
<td>93.45% ± 0.12</td>
<td>93.57% ± 0.14</td>
<td>93.69% ± 0.04</td>
<td>93.49% ± 0.03</td>
<td><b>93.76% ± 0.25</b></td>
</tr>
<tr>
<td>10 (<i>89.26%</i>)</td>
<td>93.75% ± 0.03</td>
<td><b>93.78% ± 0.25</b></td>
<td>93.57% ± 0.06</td>
<td>64.99% ± 47.64</td>
<td>93.67% ± 0.11</td>
</tr>
<tr>
<td>15 (<i>96.48%</i>)</td>
<td>93.69% ± 0.10</td>
<td><b>93.76% ± 0.04</b></td>
<td>93.66% ± 0.18</td>
<td>64.68% ± 47.36</td>
<td>93.47% ± 0.14</td>
</tr>
<tr>
<td>20 (<i>98.85%</i>)</td>
<td><b>93.53% ± 0.10</b></td>
<td>93.24% ± 0.14</td>
<td>93.03% ± 0.04</td>
<td>63.72% ± 46.54</td>
<td>92.83% ± 0.02</td>
</tr>
<tr>
<td>25 (<i>99.62%</i>)</td>
<td>91.88% ± 0.14</td>
<td>90.94% ± 0.31</td>
<td><b>91.48% ± 0.30</b></td>
<td>60.66% ± 43.87</td>
<td>91.81% ± 0.09</td>
</tr>
<tr>
<td>30 (<i>99.88%</i>)</td>
<td>51.53% ± 36.01</td>
<td>83.70% ± 0.15</td>
<td><b>84.03% ± 0.31</b></td>
<td>56.58% ± 40.35</td>
<td>33.95% ± 22.7</td>
</tr>
<tr>
<td>35 (<i>99.96%</i>)</td>
<td>10.00% ± 0.00</td>
<td><b>40.99% ± 1.12</b></td>
<td>29.54% ± 6.67</td>
<td>24.21% ± 17.48</td>
<td>14.38% ± 7.59</td>
</tr>
</tbody>
</table>

5

### 3.1.1 Similarity between $L_1$, $L_2$ and Magnitude

We notice that the top-1 accuracy of both magnitude, and the $L_1$ and $L_2$ normalization are pretty much identical at the earlier pruning ratios, contradicting our initial intuition. Our hypothesis is that this lack of difference indicates that the approaches to finding lottery tickets are less strict than initially thought, as introducing the notion of importance does not seem to have a large impact on the LTH procedure. We elaborate further on this more in the rest of the paper, where we conduct a deeper study on some differences between the found lottery tickets.

### 3.1.2 Issues with Min-Max

Due to the definition of Min-Max normalization, at least one connection in each layer will have an importance of 0 and at least one connection in each layer has an importance of 1. This will mean that during each pruning iteration, connections will be pruned from each layer. Eventually, this means that the the thinner layers will have too much capacity removed which will affect the final accuracy. This phenomenon can also be seen in the additional figures of subsection A.2.

### 3.1.3 On Layer-collapse

Inspection of the weight matrices shows that using importance measures can indeed successfully assist in preventing layer-collapse. With the magnitude measure, we can detect layer-collapse in all VGG and ResNet experiments. In the VGG experiment, this is exemplified by a degradation of the accuracy to random chance, however in the ResNet experiments this degradation is not present. The reason behind this phenomenon is that in the ResNet experiments, layer-collapse is present in the skip connections. While this preserves the flow of information in the main branch, we hypothesise that the loss of skip connections impacts the trainability of the network. In the experiments with ResNet-18, this might not be visible, but we suspect that with deeper networks such as ResNet-50, this becomes more apparent.

We do however notice that Softmax also suffers from some issues with layer-collapse, which we attribute to the factors with the generated importance distributions as highlighted in subsection 2.1.

---

## 4 A closer look

In this section we take a closer look at the generated LTs from the previous experiments. We focus our study on how they differ in structure and how they evolve throughout the pruning process.

Taking a look at how the sparsity of different network layers evolves at different intervals in the LTH procedure (see Figure 2 for the plots from ResNet18 trained on TinyImageNet), we can clearly see that during the initial pruning iterations there are large differences both between different layers and between importance measures. Even though these large differences exist between the tickets generated by different importance measures, we established in the previous section that there is no significant difference in the top-1 accuracy of Magnitude, $L_1$ and $L_2$. This is an indication that different kinds of structures exist in a network, that when trained can still achieve a competitive accuracy. As such, we might conclude that there are multiple ways to find Lottery Tickets (on which we expand in section 5). In later iterations, the difference between structures is still noticeable, but much less pronounced, which is expected as the number of pruned connections vastly outnumbers the number of unpruned connections.

Additionally, we can find an interesting pattern in the generated structures. We find that in each case, certain layers are pruned (significantly) less than most other layers in the network. Specifically, these layers are often layers that intuitively are more important to the quality of the training procedure. In the case of ResNet-18, these layers are both the input and output layer of the network, as well as the $1 \times 1$ convolutions used in the skip connections.

## 5 Multiple LTs per initialization

In this experiment we investigate whether a single initialization, under fixed circumstances, may lead to multiple Lottery Tickets. This requires making the LTR procedure deterministic and repeatable, i.e., all randomness in the training procedure is dependent on a set random seed. This ensures that the only impact on the quality and structure of the found ticket is the importance measure applied for the pruning step.

We consider the LTs found by the following importance measures: magnitude, $L_1$ and $L_2$. We limit ourselves to these measures, as they demonstrate a matching accuracy with the original dense network on the TinyImageNet dataset at a higher sparsity level (96.48%) than the other measures. As such, we also do this experiment on the same configuration namely ResNet18 and TinyImagenet.

We find that the resulting tickets share little amount of connections. Only 0.33% of the total (pruned and unpruned) connections are overlapping, while 9.34% of the unpruned connections are overlapping between different settings. We also find that some layers have a noticeably larger fraction of overlapping connections, more specifically the first few convolutional layers as well as the linear classification layer. This can be attributed to the intuition that these layers have less connections and as such are much less overparameterized.

Looking at the overlapping connections in the first convolutional layer (Figure 3), which is 68.78% of the remaining connections, compared to the non-overlapping connections, we can notice that the overlapping connections on average have a larger difference between the initial and final weight value. Please see the appendix for an extended set of plots. We also study whether these overlapping connections converge to the same weight in each pruning step up until the winning ticket is found. To do this, we have two simple metrics. For the first metric, we take the set of all weight values a single connection has at each pruning iteration and consider the standard deviation of that set as an indication of the robustness of that connection. We repeat this process for all connections in the ticket and find that on average the overlapping connections have a lower standard deviation than the non-overlapping connections. In Figure 1 we present results for the smallest, matching ticket (96.48% sparse). Moreover, it is worth noting that these trends are consistent over other sparsity levels as well (see subsubsection A.1.2).

The second metric we also apply is counting the number of sign flips. Once again considering the set of weight values a connection has during the pruning steps, we define a sign flip, if the sign of at least one instance in the set is differing from the others. Here too, we find that the overlapping connections have less sign flips than the non-overlapping connections. These results are presented in Figure 1, with results for other sparsity levels reported in the appendix.

### 5.1 Partial Reinitialization

As a final sanity check, we do a partial reinitialization test, where rather than starting the LTH/LTR procedure from the initial parameters $\theta_0$, we preserve the weights of $\theta_0$ that correspond to the overlapping connections at the 96.84% sparsity level, while reinitializing the weights of the remaining connections using the same Kaiming Normal Distribution. We then restart the LTR procedure with the studied importance measures using this initialization with the same deterministic training procedure as the initial initialization. The goal is to determine whether these overlapping connections are still (mostly) unpruned given a different initialization, and determine as such whether the overlapping property is dependent on the total initialization, or whether these connections are capable by themselves to steer an initialization towards a similarly performing lottery ticket.

We find that on average, for three random runs, the amount of remaining connections is higher than the expected value, assuming that each connection has the same chance to be in the ticket. More specifically, for the initial few convolutional layers, we see increases over the expected amount ranging from +8.5% to +70.0%. In the fully connected layer, we can see even higher increases ranging from +440% to +480%. Finally, these effects were not nearly as noticeable in the $1{\times}1$ convolutions, with only -3.2% to +20.0%.

![](./images/867762030986134385_1.jpg)

Figure 1: Left: The distribution of standard deviations, Right: The number of sign flips for overlapping and non-overlapping connections at 96.48% sparsity for the selected importance measures.

![](./images/867762030986134385_2.jpg)

Figure 2: How the layer sparsities evolve during the LTH procedure with the ResNet-18 model trained on TinyImageNet. Layers of interest are annotated on the X-axis.

![](./images/867762030986134385_3.jpg)

Figure 3: The evolution of overlapping connections within the first convolutional layer

These results suggest that these connections are, by grace of their individual initializations, more likely to be part of the ticket. This also indicates that there likely is a minimum amount of connections in some layers that are necessary for a performant ticket. Moreover, those connections seem to be mostly concentrated in the initial layers and the classification layer.

## 6 Related Work

### 6.1 Pruning neural networks

Neural networks can be trimmed of their fat by neural network pruning. These pruning methods can be grouped on a number of different axes : iterative pruning vs single-shot pruning, local pruning vs global pruning, data-driven vs data-free and more. Here, we limit ourselves and give a brief overview of the most related category to our approach. Please see [27] for an in-depth review.

Criterion-based Pruning. This group constitutes the original line of research, which follows the train-prune-finetune approach, where the network is first trained on the dataset, then pruned using a criterion and finally the pruned network is finetuned on the task to regain some accuracy. Pioneered by [6, 7], there has recently been a resurgence in popularity [8, 28, 29, 30, 31]. While the Lottery Ticket Hypothesis does not follow the classical train-prune-finetune paradigm, it still uses a criterion to induce sparsity in the network. In this work, we compare different modifications to this criterion.

### 6.2 The Lottery Ticket Hypothesis

Introduced by [13], the Lottery Ticket Hypothesis introduces a method to extract sparse trainable networks from a dense network. Initially, this method was criticised due to the reliance on low learning rates [32] and the fact that it could not be reproduced in more complex settings [33]. However, these concerns were mitigated with the introduction of weight rewinding [14]. These initial papers sparked a large body of follow-up work that we will briefly summarize in the next paragraphs.

Empirical and Theoretical properties. A first line of follow-up work was focused on the properties of the LTH. [34] systematically explored different components of the LTH procedure. [35] compares the impact of finetuning and rewinding and introduces the aspect of learning rate rewinding. [36] introduces a theoretical proof for the effectiveness of the LTH, while [15] studied the gradient flow in the LTH procedure. Given the characteristics of our study, our work is situated in this category. In our work, we also study empirically the properties of the LTH namely by considering different options for the pruning criterion and comparing the properties of the resulting Lottery Tickets.

Transferring LTs. A second line of research involved transferring lottery tickets. This includes the transferability of lottery tickets between different datasets [37, 38, 39, 40, 41, 42] and even between different models [43]. Complementary to those efforts, here we study the existence of multiple tickets within a given dataset-model combination.

Other domains and models. A third line of research studied the application of the LTH approach — which was initially introduced for Convolutional

Networks — on other types of models and domains. [44, 45, 46, 47, 48]. In this regard, we focus on the image recognition (classification) task via fully-connected and convolutional architectures.

Strong LTH. Finally, recently a lot of effort has been poured in the so-called 'strong Lottery Ticket Hypothesis' [49] and proving its existence first in MLPs, then in CNNs and even in Equivariant Networks. [50, 51, 52, 53, 54, 55] The strong Lottery Ticket Hypothesis dictates that a fully trained network can be approximated by pruning a sufficiently large randomly-initialized network, rather than extracting a sparse network from a trained network as in the 'original' Lottery Ticket Hypothesis formulation.

## 7 Conclusion

We studied a modification to the Lottery Ticket Hypothesis by introducing the notion of layerwise importance in the procedure. While the inclusion of these importance measures did not produce significant changes in performance, we emphasize that the LTH is invariant in a certain sense to the layerwise pruning ratios, as long as the lowest magnitude weights are removed from each layer.

Furthermore, we showed that given a fixed initialization, it is possible to extract different lottery tickets that are dependent on the used importance measure. These tickets all have roughly the same top-1 accuracy, but differ significantly in their structure. Looking at the remaining connections in the tickets, we noticed that there exist a noticeable amount of overlapping connections across them, namely mostly in the first convolutional layers and in the output layer. We observed that these overlapping connections consistently have a lower variance between different lottery tickets, which suggests that these connections are somewhat more stable and converge more often to roughly the same value. Finally, we showed with a partial reinitialization test, that these connections even survive when the other weights are reinitialized. This suggests that these specific connections could be the key towards finding LTs more efficiently.

## 8 Acknowledgements

This work is supported by the UAntwerp Dehousse Mandate id:54/RECBV007 "Sparse Representation Learning for Computer Vision"

## References

[1] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving language understanding by generative pre-training," 2018.

[2] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, *et al.*, "Language models are few-shot learners," *Advances in neural information processing systems*, vol. 33, pp. 1877–1901, 2020.

[3] E. Strubell, A. Ganesh, and A. McCallum, "Energy and policy considerations for modern deep learning research," in *AAAI*, pp. 13693–13696, 2020.

[4] M. Denil, B. Shakibi, L. Dinh, M. Ranzato, and N. De Freitas, "Predicting parameters in deep learning," *Advances in neural information processing systems*, vol. 26, 2013.

[5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "Mobilenetv2: Inverted residuals and linear bottlenecks," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4510–4520, 2018.

[6] Y. le Cun, "Optimal brain damage," *Advances in Neural Information Processing Systems*, vol. 2, 1990.

[7] B. Hassibi and D. Stork, "Second order derivatives for network pruning: Optimal brain surgery," *Advances in Neural Information Processing Systems*, 1993.

[8] S. Han, J. Pool, J. Tran, and W. J. Dally, "Learning both weights and connections for efficient neural networks," in *Advances in Neural Information Processing Systems*, vol. 2015-January, 2015.

[9] C. Louizos, M. Welling, and D. P. Kingma, "Learning sparse neural networks through l0 regularization," in 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceedings, 2018.

[10] D. Molchanov, A. Ashukha, and D. Vetrov, "Variational dropout sparsifies deep neural networks," in 34th International Conference on Machine Learning, ICML 2017, vol. 5, 2017.

[11] U. Evci, F. Pedregosa, A. Gomez, and E. Elsen, "The difficulty of training sparse neural networks," in Workshop Deep Phenomena, 2019.

[12] D. Zou and Q. Gu, "An improved analysis of training over-parameterized deep neural networks," Advances in neural information processing systems, vol. 32, 2019.

[13] J. Frankle and M. Carbin, "The lottery ticket hypothesis: Finding sparse, trainable neural networks," in 7th International Conference on Learning Representations, ICLR 2019, 2019.

[14] J. Frankle, G. K. Dziugaite, D. Roy, and M. Carbin, "Linear mode connectivity and the lottery ticket hypothesis," in International Conference on Machine Learning, pp. 3259-3269, PMLR, 2020.

[15] U. Evci, Y. Dauphin, Y. Ioannou, and C. Keskin, "Gradient flow in sparse neural networks and how lottery tickets win," in Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence, pp. 3082-3101, 2022.

[16] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification," in Proceedings of the IEEE international conference on computer vision, pp. 1026-1034, 2015.

[17] H. Tanaka, D. Kunin, D. L. Yamins, and S. Ganguli, "Pruning neural networks without any data by iteratively conserving synaptic flow," in Advances in Neural Information Processing Systems, vol. 2020-December, 2020.

[18] J. Lee, S. Park, S. Mo, S. Ahn, and J. Shin, "Layer-adaptive sparsity for the magnitude-based pruning," in International Conference on Learning Representations, 2021.

[19] J. Frankle, "Openlth: A framework for lottery tickets and beyond," 2020.

[20] J. Frankle, G. K. Dziugaite, D. Roy, and M. Carbin, "Pruning neural networks at initialization: Why are we missing the mark?," in International Conference on Learning Representations, 2021.

[21] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.

[22] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in International Conference on Learning Representations, 2015.

[23] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.

[24] L. Deng, "The mnist database of handwritten digit images for machine learning research," IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 141-142, 2012.

[25] A. Krizhevsky, G. Hinton, et al., "Learning multiple layers of features from tiny images," 2009.

[26] Y. Le and X. Yang, "Tiny imagenet visual recognition challenge," CS 231N, vol. 7, no. 7, p. 3, 2015.

[27] T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, and A. Peste, "Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks," jmlr.org, vol. 23, pp. 1-124, 2021.

[28] H. Li, H. Samet, A. Kadav, I. Durdanovic, and H. P. Graf, "Pruning filters for efficient convnets,"

5th International Conference on Learning Repre- sentations, ICLR 2017 - Conference Track Pro- ceedings, 2017.

[29] C. Zhao, B. Ni, J. Zhang, Q. Zhao, W. Zhang, and Q. Tian, "Variational convolutional neural network pruning," Proceedings of the IEEE Com- puter Society Conference on Computer Vision and Pattern Recognition, vol. 2019-June, 2019.

[30] X. Dong, S. Chen, and S. J. Pan, "Learning to prune deep neural networks via layer-wise opti- mal brain surgeon," Advances in Neural Infor- mation Processing Systems, vol. 2017-December,2017.

[31] R. Yu, A. Li, C. F. Chen, J. H. Lai, V. I. Morariu, X. Han, M. Gao, C. Y. Lin, and L. S. Davis, "Nisp: Pruning networks using neuron importance score propagation," Proceedings of the IEEE Computer Society Conference on Computer Vi- sion and Pattern Recognition, 2018.

[32] Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Dar- rell, "Rethinking the value of network pruning," in 7th International Conference on Learning Rep- resentations, ICLR 2019, 2019.

[33] T. Gale, E. Elsen, and S. Hooker, "The state of sparsity in deep neural networks," 2 2019.

[34] H. Zhou, J. Lan, R. Liu, and J. Yosinski, "De- constructing lottery tickets: Zeros, signs, and the supermask," in Advances in Neural Information Processing Systems, vol. 32, 2019.

[35] A. Renda, J. Frankle, and M. Carbin, "Compar- ing rewinding and fine-tuning in neural network pruning," in International Conference on Learn- ing Representations, 2020.

[36] J. Maene, M. Li, and M.-F. Moens, "Towards understanding iterative magnitude pruning: Why lottery tickets win," 2021.

[37] A. S. Morcos, H. Yu, M. Paganini, and Y. Tian, "One ticket to win them all: Generalizing lottery ticket initializations across datasets and optimiz- ers," in Advances in Neural Information Process- ing Systems, vol. 32, 2019.

[38] R. V. Soelen and J. W. Sheppard, "Using winning lottery tickets in transfer learning for convolu- tional neural networks," in Proceedings of the International Joint Conference on Neural Net- works, vol. 2019-July, 2019.

[39] R. Mehta, "Sparse transfer learning via winning lottery tickets," 2019.

[40] M. Sabatelli., M. Kestemont., and P. Geurts., "On the transferability of winning tickets in non- natural image datasets," in Proceedings of the16th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,, pp. 59-69, SciTePress, 2021.

[41] S. Desai, H. Zhan, and A. Aly, "Evaluating lottery tickets under distributional shifts," in DeepLo@EMNLP-IJCNLP 2019 - Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource Natural Language Processing - Proceedings, 2021.

[42] T. Chen, J. Frankle, S. Chang, S. Liu, Y. Zhang, M. Carbin, and Z. Wang, "The lottery tickets hypothesis for supervised and self-supervised pre- training in computer vision models," in Proceed- ings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition,2021.

[43] X. Chen, Y. Cheng, S. Wang, Z. Gan, J. Liu, and Z. Wang, "The elastic lottery ticket hypothe- sis," Advances in Neural Information Processing Systems, vol. 34, 2021.

[44] T. Chen, Z. Zhang, S. Liu, S. Chang, and Z. Wang, "Long live the lottery: The existence of winning tickets in lifelong learning," in Interna- tional Conference on Learning Representations,2021.

[45] J. Diffenderfer and B. Kailkhura, "Multi-prize lottery ticket hypothesis: Finding accurate bi- nary neural networks by pruning a randomly weighted network," in International Conference on Learning Representations, 2021.
12

[46] T. Chen, J. Frankle, S. Chang, S. Liu, Y. Zhang, Z. Wang, and M. Carbin, "The lottery ticket hypothesis for pre-trained bert networks," Ad- vances in neural information processing systems, vol. 33, pp. 15834-15846, 2020.

[47] N. M. Kalibhat, Y. Balaji, and S. Feizi, "Winning lottery tickets in deep generative models," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 8038-8046, 2021.

[48] T. Chen, Y. Sui, X. Chen, A. Zhang, and Z. Wang, "A unified lottery ticket hypothesis for graph neural networks," in International Con- ference on Machine Learning, pp. 1695-1706, PMLR, 2021.

[49] V. Ramanujan, M. Wortsman, A. Kembhavi, A. Farhadi, and M. Rastegari, "What's hidden in a randomly weighted neural network?," in Proceedings of the IEEE Computer Society Con- ference on Computer Vision and Pattern Recog- nition, 2020.

[50] D. Chijiwa, S. Yamaguchi, Y. Ida, K. Umakoshi, and T. Inoue, "Pruning randomly initialized neu- ral networks with iterative randomization," Ad- vances in Neural Information Processing Systems, vol. 34, 2021.

[51] A. da Cunha, E. Natale, and L. Viennot, "Prov- ing the strong lottery ticket hypothesis for convo- lutional neural networks," in International Con- ference on Learning Representations, 2022.

[52] E. Malach, G. Yehudai, S. Shalev-Schwartz, and O. Shamir, "Proving the lottery ticket hypothe- sis: Pruning is all you need," in Proceedings of the 37th International Conference on Machine Learning, vol. 119, pp. 6682-6691, PMLR, 3 2020.

[53] L. Orseau, M. Hutter, and O. Rivasplata, "Loga- rithmic pruning is all you need," in Advances in Neural Information Processing Systems, vol. 2020- December, 2020.

[54] A. Pensia, S. Rajput, A. Nagle, H. Vish- wakarma, and D. Papailiopoulos, "Optimal lot- tery tickets via subsetsum: Logarithmic over- parameterization is sufficient," in Advances in Neural Information Processing Systems, vol. 2020- December, 2020.

[55] D. Ferbach, C. Tsirigotis, G. Gidel, et al., "A general framework for proving the equivariant strong lottery ticket hypothesis," arXiv preprint arXiv:2206.04270, 2022.

## A Appendix

### A.1 Additional Figures

#### A.1.1 Accuracy curves

The curves in Figure 4 illustrate the evolution of the top-1 accuracy of Lottery Tickets at different sparsities. These figures provide a more general overview than the tables the tables found in section 3, but are harder to extract fine-grained details from.

![](./images/867762030986134385_4.jpg)

(a) LeNet300-100 on MNIST

![](./images/867762030986134385_5.jpg)

(b) VGG16 on CIFAR10

![](./images/867762030986134385_6.jpg)

(c) ResNet20 on CIFAR10

![](./images/867762030986134385_7.jpg)

(d) ResNet18 on TinyImageNet

Figure 4: The accuracy curves for the different training configurations used in the main paper. The mean and standard deviation are given for 3 random seeds. The dashed line represents the mean accuracy of the dense network.

### A.1.2 Multiple LTs per initialization

![](./images/867762030986134385_8.jpg)

![](./images/867762030986134385_9.jpg)

Figure 5: Additional distributions of standard deviations of both overlapping and non-overlapping connections for the {L1, L2, Magnitude} importance measures

![](./images/867762030986134385_10.jpg)

![](./images/867762030986134385_11.jpg)

Figure 6: Additional Sign flips of both overlapping and non-overlapping connections for the {L1, L2, Magnitude} importance measures


## A.1.3 Overlap curves

![](./images/867762030986134385_12.jpg)

Figure 7: Additional curves measuring the proportion of overlapping connections between tickets

## A.2 Layer sparsity

![](./images/867762030986134385_13.jpg)
(a) Resnet20 on CIFAR-10

![](./images/867762030986134385_14.jpg)
(b) VGG-16 on CIFAR-10

![](./images/867762030986134385_15.jpg)
(c) ResNet-18 on TinyImageNet

Figure 8: The layerwise pruning ratio of different layers in a LT compared to the total pruning ratio. The x-axis is (roughly) corresponding to model depth and layers of interest are indicated.

When studying the layerwise sparsities generated by the different pruning criteria (see Figure 8), we can notice a number of interesting observations. Primarily, we notice that for each pruning criterion, both the

input and output layer are pruned proportionally much less than other layers, even though this wasn't explicitly encoded in the criterion. While the exact mechanism behind this isn't fully understood and is out-of-scope, we can theorize that this is due a combination of the training process and the fact that these layers are vital for the flow of information in the network. From all considered importance measures, the magnitude measure prunes the least of those layers.

When considering ResNets, we can additionally notice that most of the other criterions ($L_1$, $L_2$, SoftMax) also prune the 1x1 convolutions less. These 1x1 convolutions are located in the bottleneck modules and often contain less redundancy than the other 3x3 convolutions.

## A.3 Layer sparsity in time

![](./images/867762030986134385_16.jpg)

Figure 9: The evolution of layer sparsity during the LT extracting process, visualised in a number of selected layers of the VGG16 network.

In Appendix A.2, we showed that some layers are disproportionally pruned less by different pruning criteria. In this section, we will take a look at how the sparsity of these layers evolves throughout the pruning process and compare them to 'normal' layers in both VGG16 (see Figure 9) and ResNet-18 (see Figure 10).

It can be clearly seen that generally the deeper the layer is within the model, the faster this layer is pruned. This has the effect that often the higher layers are 'ahead' of the global sparsity, while the lower layers are 'behind' of the global sparsity. The main exceptions here being the 1x1 convolutional layers, however we can still notice that the later 1x1 convolutions are pruned faster than the 1x1 convolutions.

Additionally, we can notice some differences in behaviour of the different pruning criteria, namely that softmax has a spiky evolution, meaning that at some timesteps, no connections are pruned in a layer, while at other timesteps large amounts of connections are pruned. The other criteria follow a much smoother approach, where at least every timestep some connections are removed from each layer, with Min-Max often following the global sparsity best.

18

![](./images/867762030986134385_17.jpg)

Figure 10: The evolution of layer sparsity during the LT extracting process, visualised in a number of selected layers of the ResNet18 network.

## A.4 Networks, Datasets and Training

<table>
  <tr>
    <td>Network</td>
    <td>Dataset</td>
    <td>Epochs</td>
    <td>Batch</td>
    <td>Opt.</td>
    <td>Mom.</td>
    <td>LR</td>
    <td>LR Drop</td>
    <td>Weight Decay</td>
    <td>Initialization</td>
    <td>Iters per Ep</td>
    <td>Rewind Iter</td>
  </tr>
  <tr>
    <td>LeNet</td>
    <td>MNIST</td>
    <td>40</td>
    <td>128</td>
    <td>SGD</td>
    <td>—</td>
    <td>0.1</td>
    <td>—</td>
    <td>—</td>
    <td>Kaiming Normal</td>
    <td>469</td>
    <td>0</td>
  </tr>
  <tr>
    <td>ResNet20</td>
    <td>CIFAR-10</td>
    <td>160</td>
    <td>238</td>
    <td>SGD</td>
    <td>0.9</td>
    <td>0.1</td>
    <td>10x at epochs 80, 120</td>
    <td>1e-4</td>
    <td>Kaiming Normal</td>
    <td>391</td>
    <td>1000</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>CIFAR-10</td>
    <td>160</td>
    <td>128</td>
    <td>SGD</td>
    <td>0.9</td>
    <td>0.1</td>
    <td>10x at epochs 80, 120</td>
    <td>1e-4</td>
    <td>Kaiming Normal</td>
    <td>391</td>
    <td>2000</td>
  </tr>
  <tr>
    <td>ResNet18</td>
    <td>TinyImageNet</td>
    <td>200</td>
    <td>256</td>
    <td>SGD</td>
    <td>0.9</td>
    <td>0.2</td>
    <td>10x at epochs 100, 150</td>
    <td>1e-4</td>
    <td>Kaiming Normal</td>
    <td>391</td>
    <td>1000</td>
  </tr>
</table>