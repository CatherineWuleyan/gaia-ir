# To Prune or not to Prune: :
## A Chaos-Causality Approach to Principled Pruning of Dense Neural Networks

Rajan Sahu
Dept. of CSIS, Birla Institute of Technology and Sciences Pilani, Pilani
f20190572@pilani.bits-pilani.ac.in,

Shivam Chadda
BITS Pilani, Goa Campus
f20190704@goa.bits-pilani.ac.in

Nithin Nagaraj
Consciousness Studies Programme, National Institute of Advanced Studies, Bangalore, India
nithin@nias.res.in

Archana Mathur
Dept. of Information Science and Engineering, Nitte Meenakshi Institute of Technology, India
archana.mathur@nmit.ac.in

Snehanshu Saha
Dept. of CSIS and APPCAIR, BITS Pilani Goa, India and HappyMonk AI
snehanshus@goa.bits-pilani.ac.in

## ABSTRACT

Reducing the size of a neural network (pruning) by removing weights without impacting its performance is an important problem for resource constrained devices. In the past, pruning was typically accomplished by ranking or penalizing weights based on criteria like magnitude and removing low-ranked weights before retraining the remaining ones. Pruning strategies may also involve removing neurons from the network in order to achieve the desired reduction in network size. We formulate pruning as an optimization problem with the objective of minimizing misclassifications by selecting specific weights. To accomplish this, we have introduced the concept of chaos in learning (Lyapunov exponents) via weight updates and exploiting causality to identify the causal weights responsible for misclassification. Such a pruned network maintains the original performance and retains feature explainability

## 1 Introduction

Designing a neural network architecture is a critical aspect of developing neural networks for various AI tasks, particularly in deep learning. One of the fundamental challenges in designing neural networks is finding the right balance between model complexity and sample size, which can have a significant impact on the network's performance. In general, a larger network with more parameters (overparameterized) can potentially learn more complex functions and patterns from the data [21]. However, challenge in network architecture design is to find the right balance between model complexity and sample size, so that the network can learn to generalize well to new, unseen data. In this context, over-parameterized networks [27], [19] have become increasingly popular in the deep learning era due to

To Prune or not to Prune

their ability to achieve high expressivity and potentially better generalization performance [24]. The idea behind over-parameterization is to increase the number of parameters in the network beyond what is strictly necessary to fit the training data and remarkable generalization to test data. However, such networks still require a number of assumptions and hyperparameter tuning to guarantee optimal performance.

Pruning techniques should reduce the number of parameters in a neural network without compromising its accuracy. However, it is important to ponder why we do not simply train a smaller network from scratch to make training more efficient. The reason is that the architectures obtained after pruning are typically more challenging to train from scratch [16], and they often result in lower accuracy compared to the original networks. Therefore, while standard pruning techniques can effectively reduce the size and energy consumption of a network, it does not necessarily lead to a more efficient training process.

### 1.1 Problem statement and Contributions

We pose a broad *Research Question* here: Is there a principled approach to pruning overparameterized, dense neural networks to a reasonably good sparse approximation such that performance is not compromised and explainability is retained? It is well known that dense neural network training and particularly weight updates via SGD have some element of chaos [26],[9]. We expect that, between the successive weight updates due to SGD and miss-classification, there is some observed causality and non-causal weights (parameters) can be pruned, leading to a sparse network i.e. some weight updates cause reduction in network (training) loss and some do not! Can we train a dense network till a few epochs to derive a pruned architecture for the derivative to run for the remaining epochs and produce performance metrics in the $\epsilon-$ball of the original, dense network? Does this sparse network also train well, verified with Shapley [17] and WeightWatcher ($WW$) [18] tests.? Specifically, we contribute to the following:

- Present a unique and unifying framework on chaos and causality for deep network pruning. The unifying framework uses LE [10] and Granger causality (GC) [6] tandem.
- Propose novel pruning architectures, Lyapunov Exponent Causality driven Fully Trained Network (LEGCNet-FT) and Lyapunov Exponent Granger Causality driven Partially Trained Network (LEGCNet-PT).
- LEGCNet-FT and LEGCNet-PT compare very well in performance and other baselines, Random [15] and Magnitude based [14] pruning techniques.
- Establish feature consistency of LEGCNet-FT and LEGCNet-PT in explainability.
- Verify empirically that the proposed architectures for pruning are not over-trained and obviously not over-parameterized but are still able to generalize well, on diverse data sets while saving Flops (FLoating-point OPerationS). We accomplish this via the $WW$ test.

In summary, we propose pruning techniques, *LEGCNet-FT and LEGCNet-PT* which perform at par with the dense, unpruned architecture and the existing pruning baselines. *While maintaining consistent performance, these techniques also help reduce epochs to converge and Flops to compute while maintaining feature consistency with their dense counterparts and ensuring proper training across layers validated via WW statistics.*

## 2 Technical Motivation

Chaos, causality and the manifestation of the Lottery Ticket Hypothesis are the key motivation behind our proposed pruning mechanism.

### 2.1 Gradient Descent and Low Dimensional Chaos

Is the process of updating weights in backpropagation via Gradient Descent chaotic? Is there an alternative interpretation of the minima in the weight landscape via low-dimensional chaos? The weight update in SGD is written as $w_{i+1} \leftarrow w_i - \eta_i \nabla_w f(w_i)$ may be thought of as a discretization to the first order ODE: $w'(t) = -\nabla_w f(w_i)$. The minimizer of the SGD is therefore conceived as a stable equilibrium of the ODE. That is, the minimum, $w^*$ can be thought of as a fixed point to the iterates $w_{i+1}=G(\eta_i, w_i) \equiv w^*=G(\eta_i, w^*)$.

*Empirical evidence of chaos in back propagation:* We performed a series of experiments on different datasets to check Sensitive Dependence On Initial Conditions (SDIC) for weight initialization, on single hidden layer neural network. The weight initialization matrix $W_{ij}$ followed Gaussian distribution $(W_{ij} \sim N(0, \sigma^2))$. We recorded two set of executions - one with initial weight $w_{11}: w_{ij} \forall i \in 1..h, \forall j \in 1..n$, where $n, h$ are the number of input and hidden neurons - and another, with infinitesimal perturbation $(w_{11}+\delta)$ keeping other parameters same. Each time, the network

To Prune or not to Prune

was trained via gradient descent and weight series were recorded. The method was repeated for the second weight connection $w_{ij}, i=1, j=2$ and for all the weights on the network. Later, the Lyapunov exponents were computed by using the TISEAN package [8] on the recorded weight series to measure the perturbed trajectory due to initial perturbation $\delta$. We observed positive Lyapunov exponents which marked the presence of some chaotic behavior in gradient descent.

### 2.2 Chaos and Causality

One way to address the issue of explainability in AI/machine learning is to seek causal explanations for choices made in the learning process. Conversely, a learning process that incorporates choices made out of causal considerations is easier to explain and interpret. This is the motivation behind using causality based criteria for the choice of what to prune (or not prune) in this study. To this end, we employ Granger Causality (GC) [7].

The principle of pruning that is causally-informed is formulated as follows. Those connections (weights) in the learning network that do not causally impact the loss are chosen for pruning. To determine the causal impact of a particular connection to the loss, we perform GC between the windowed Lyapunov exponents (LE) of the weight time series for that connection and the classification accuracy. The rationale behind this is the intuition that the chaotic signature of weight updates inform learning. A biological inspiration for chaotic signatures as a marker for learning is the empirical fact that neurons in the human brain exhibit chaos [4, 11] at all spatio-temporal scales. Starting from single neurons to coupled neurons to network of neurons to different areas of brain – chaos has been found to be ubiquitous to brain [11]. Chaotic systems are known to exhibit a wide range of patterns (periodic, quasi-periodic and non-periodic behaviors), are very robust to noise and enable efficient information transmission [20], processing/computation [3, 12] and classification [2]. There is also some evidence to suggest that weak chaos is likely to aid learning [25]. Thus our choice of testing causal strength between LE (a value $>0$ is a marker of chaos) and classification accuracy as a criterion for pruning to yield sparse subnetworks that capture the learning of the task at hand.

### 2.3 Lottery ticket hypothesis

The "lottery ticket hypothesis" [5] is a concept in neural network pruning that suggests that within a dense and over-parameterized neural network, there exist sparse sub-networks that can be trained to perform just as well as the original dense network. Any fully-connected feed-forward network $f^d(x;\phi)$, with initial parameters $\phi$ when trained on a training set D, $f^d$ achieves a test accuracy $a$ and error $e$ at iteration $j$. Our work LEGCNet, validates the lottery ticket hypothesis by finding the "winning-ticket", $m$, to construct the sparse network, $f^s$ such that $acc^s \geq a$ and $j^s < j$ where $\|\phi\| \gg \|m\|$.

The remainder of the paper is organized to present the key methodologies used to develop the pruning technique (section 3), followed by a detailed experimental setup (section 4) and strong empirical evidence of the proposed technique in contrast to the baselines (section 5,6).

## 3 Tools and Methodology

### 3.1 Definitions:

- Dense Neural network - Let $f^d$ be a dense neural network of depth $l$ and width $h$ defined as
  $$
  f^d(x)=W_i^d \sigma_i(W_{i-1}^d \sigma_i(...W_1^d(x))) \tag{1}
  $$
  where $W_i^d$ is the weight matrix for layer $i$ such that $i \in 1..l$.
- Sparse Neural Network - Let $f^s$ be sparse neural network of the same architecture as $f^d$, with depth $l$ and width $h$.
- Two approaches: LEGCNet-FT and LEGCNet-PT - In order to validate the working of LEGCNet, we divided the method into two discrete approaches. In one approach, the entire training weight-series is used for computing Lyapunov exponents, testing Granger causality and for the identification of causal weights. Essentially, the dense network is trained till convergence and the approach is called LEGCNet-Full Train (LEGCNet-FT). In the second approach, LEGCNet-PT, the network is trained only till certain epochs (10% of the total iterates) and these few weight-updates are captured for identifying the causal weights.

### 3.2 WeightWatcher (WW) as a diagnostic tool

WW is a powerful open-source diagnostic tool designed for analyzing Deep Neural Networks (DNN). WW analyzes each layer by plotting the empirical spectral distribution (ESD), which represents the histogram of eigenvalues from the

To Prune or not to Prune

![](./images/900808050829426859_1.jpg)

Figure 1: Weights pruned via LEGCNet method for selecting connections in sparse neural network

layer's correlation matrix. Additionally, it fits the tail of the ESD to a (truncated) power law and presents these fitted distributions on separate axes. This visualization approach provides a clear representation of the eigenvalue distribution and highlights the presence of heavy-tailed behavior in the network's layers. In general, the ESDs observed in the best layers of high-performing DNNs can often be effectively modeled using a Power Law (PL) function. The PL exponents, denoted as alpha, tend to be closer to 2.0 in these cases indicating a heavy-tailed behavior in the layer's correlation matrix.

### 3.3 SHAP as a diagnostic tool
SHAP (SHapley Additive exPlanations) is a game-theoretic technique utilized to provide explanations for the output of machine learning models. SHAP enables a comprehensive understanding of the contributions made by different features in the model's output, facilitating insightful explanations for its decision-making process. If a network is pruned according to some underlying principles, then the consistency in feature explainability is maintained before and after pruning i.e. the features which explain the outcome before pruning (fully connected, dense network) remain consistent on the pruned network.

### 3.4 Windowed Weight Updates:
Consider a neural network with $n$ inputs and $r$ outputs, $l$ hidden layers of $h$ neurons, and, the input vector denoted as $x \in R^n$, The network when trained by SGD generates a sequence of weight updates represented by $w^{ji} = \begin{bmatrix}w_0^{ji},w_1^{ji},...,w_k^{ji},...,w_K^{ji}\end{bmatrix}$ where $w_k^{ji}$ is the weight of ith neuron of the input layer and jth neuron of the hidden layer at kth iteration. Considering, the weights being collected for initial few epochs, the weight iterates for hidden layer and output layer are $W_h = \{w^{ji}\}, W_o = \{w^{kj}\} \quad \forall i \in \{1,..,n\}, \forall j \in \{1,..,m\}, \forall k \in \{1,..,r\}$ An infinitesimal perturbation $\delta_0$ is introduced in the initial weight $w^{11}$, given as $w^{11\delta_0}=w^{11}+\delta_0$, keeping other parameters - weights (initialization), learning rate, optimizer, and loss function- same. The network is then retrained with the perturbed weight, and the weights updates are recorded again as $W_h^{\delta_0} = \{w^{ji\delta_0}\}, W_o^{\delta_0} = \{w^{kj\delta_0}\} \quad \forall i \in \{1,..,n\}, \forall j \in \{1,..,m\}, \forall k \in \{1,..,r\}$ A difference series obtained by subtracting perturbed weights from initial weights is $\delta W_h = \{\delta w^{ji}\}, \delta W_o = \{\delta w^{kj}\} \quad \forall i \in \{1,..,n\}, \forall j \in \{1,..,m\}, \forall k \in \{1,..,r\}$. We divide the weight series $\delta w^{ji}$ into $K$ windows, $w^{ji} = \bigcup_{l=1}^K w^{ji(l)}$, and compute the Lyapunov exponent of all the windowed-weight trajectories. The series of the Lyapunov exponent $\lambda$ of the windowed-weight trajectories $w^{ji(K)}$ are represented using the notation $\{\lambda^{ji\{1\}},\lambda^{ji\{2\}},...,\lambda^{ji\{K\}}\}$. Additionally, we record the accuracy at every window during training $\forall l \in 1,..., K$, captured for weights at $w^{ji(l)}$ and $w^{kj(l)}\forall i \in \{1,..,n\}, \forall j \in \{1,..,m\}, \forall k \in \{1,..,r\}$.

### 3.5 Approximation capability of LEGCNet
The main theorem and lemma in this section build that the LEGCNet pruned network $(f^s)$ is $\epsilon$-close to $f^d$ with probability $1$-$\delta$ [22]. Consider $f^d$ to be a dense neural network defined as

$$
f^d(x)=W_i^d\sigma_i(W_{i-1}^d\sigma_i(...W_1^d(x))) \tag{2}
$$

To Prune or not to Prune

where $W_i^d$ is the weight matrix for layer $i$ such that $i \in 1..l$ and $h_i > h_0, h_i > h_l, \forall i \in 1...(l-1)$. We assume that for network in (2), $\sigma_i$ is $L_i$-Lipschitz and weight matrix $W_i^d$ is initialized from uniform distribution $U\left[\frac{-K}{\sqrt{max(h_i,h_{i-1})}}, \frac{K}{\sqrt{max(h_i,h_{i-1})}}\right]$, for some constant $K$.

Theorem 3.1 (Approximation Capability of the Sparse Network) Let $\epsilon > 0, \delta > 0, \alpha \in (0,1)$ such that the, for some constants $K_1, K_2, K_3, K_4, K_5$,

$$
h \geq max\left\{ K_1^{\frac{1}{\alpha}}, \left(\frac{K_2}{\epsilon}\right)^{\frac{1}{\alpha}}, \left(\frac{K_3}{\delta}\right)^{\frac{1}{\alpha}}, K_4 + K_5 log\left(\frac{1}{\delta}\right) \right\}
$$

then sparse network $f^s$ obtained from LEGCNet by the mask $m$, and pruning the weights $W_i^d, \forall i \in 1...l$ is $\epsilon$-close to $f^d$, with the probability $(1-\delta)$, i.e.

$$
\sup_{x \in B_{d0}} \left\| f^s(x) - f^d(x) \right\|_2 \leq \epsilon
$$

Remark: Lipschitz property of the activation functions [23] is a necessary condition to validate the approximation capability of the proposed sparse network. We have used sigmoid activation in the sparsely trained/pruned network.

Lemma 3.2 Sigmoid activation is Lipschitz.

Proof: If a function $f(x)$ is Lipschitz continuous, then: $\| f(x) - f(y) \| \leq K \| x - y \| \equiv \| f'(x) \| \leq K$. If $K < 1$, $f$ is a contraction map as well. We know that Sigmoid, $\sigma(x) = \frac{1}{1+e^{-x}}$; and $\| \sigma' \| = \| \sigma(x)(1-\sigma(x)) \|$. It's easy to follow that : $\| \sigma(x)(1-\sigma(x)) \| \leq \| \sigma(x) \| \| 1-\sigma(x) \| \leq C_1, C_2$. Since $0 \leq C_1, C_2 \leq 1$, $C_1 * C_2 = \delta \leq 1$. Hence sigmoid is Lipschitz.

### 3.6 Methodology Overview (refer figure 1)

Our pruning method was developed as follows. Initially, we trained a simple Multi-layer Perceptron (MLP) on a given dataset. Throughout the training process, we recorded the weights at each iteration, resulting in a time series of weight values. These weight time series were subsequently utilized to estimate the Lyapunov exponents, a measure of chaotic behavior, using the TISEAN package in conjunction with MATLAB scripts. This estimation was performed using a sliding window approach, generating a time series of Lyapunov exponents.

Based on our experiments, the time series of Lyapunov exponents consistently exhibited positive values, indicating the presence of chaotic elements in the weight time series. Consequently, our study focused on understanding whether these weights Granger caused the model's accuracy. Any weights that did not demonstrate this causal relationship were pruned before conducting subsequent model runs (code is uploaded in supplementary, zipped file).

## 4 Experimental set-up

In our study, we employed Python3.10 and Matlab R2022a to conduct experiments on a single hidden layer neural network on various datasets. Our experiments were conducted on Ryzen 9 3900XT Desktop Processor with 32GB RAM and 1TB HDD. During training, we stored the weight updates for every connection in CSV files. We assumed a window size of 200 iterates and computed the Lyapunov exponent for each weight connection on every window. Further, we calculated the training and test accuracy on every window, to capture the misclassification rates. Thus, we obtained a sequence of windowed-Lyapunov exponents and windowed accuracies for every connection. We then computed the Granger causality between the windowed Lyapunov exponents and the misclassification rate to identify weight connections that Granger caused misclassification. In the process of pruning the network, we removed the connections for which the Lyapunov exponents were found to Granger cause misclassification. After pruning, we reran the experiment by keeping the initial weights, optimizers, and other hyperparameters the same as unpruned network. We extended the work on different datasets and recorded the epochs and accuracies of the pruned network. Interestingly, we observed that the accuracies of the sparse network exceeded those of the dense network.

In our study, we conducted experiments in two parts. In the first part, we trained the neural network until convergence (LEGCNet-FT) and computed windowed-Lyapunov exponents for each weight connection as well as windowed accuracies. We then computed the Granger causality and pruned the network by removing connections that were found to cause misclassification. In the second part of the experiment, we trained the network only for few epochs (LEGCNet-PT) and used these initial weight updates to compute windowed-Lyapunov exponents and accuracies, repeating the same procedure as in the first part of the experiment. Finally, we compared the performance of the pruned network (LEGCNet-FT and LEGCNet-PT) to that of the original network. The results of all these experiments were recorded and presented in tables. Code is available as supplementary file.


To Prune or not to Prune

<table>
  <thead>
    <tr>
      <th>Dataset (hidden neurons)</th>
      <th>Flops - DN</th>
      <th>Flops - LEGCNet-FT</th>
      <th>Non causal Weights</th>
      <th>Epochs DN</th>
      <th>Epochs LEGCNet-FT</th>
      <th>Accuracy DN</th>
      <th>Accuracy LEGCNet-FT</th>
      <th>F1-score DN</th>
      <th>F1-score LEGCNet-FT</th>
      <th>%Pruned LEGCNet-FT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cancer(6)</td>
      <td>60</td>
      <td>54</td>
      <td>6</td>
      <td>70</td>
      <td>38</td>
      <td>0.8759</td>
      <td>0.8686</td>
      <td>0.8656</td>
      <td>0.8570</td>
      <td>10</td>
    </tr>
    <tr>
      <td>Titanic (8)</td>
      <td>56</td>
      <td>52</td>
      <td>4</td>
      <td>43</td>
      <td>20</td>
      <td>0.7225</td>
      <td>0.7177</td>
      <td>0.6948</td>
      <td>0.6843</td>
      <td>7.14</td>
    </tr>
    <tr>
      <td>Banknote (8)</td>
      <td>40</td>
      <td>36</td>
      <td>4</td>
      <td>9</td>
      <td>7</td>
      <td>0.9018</td>
      <td>0.8909</td>
      <td>0.9016</td>
      <td>0.8906</td>
      <td>10</td>
    </tr>
    <tr>
      <td>Iris (6)</td>
      <td>42</td>
      <td>40</td>
      <td>2</td>
      <td>182</td>
      <td>166</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9124</td>
      <td>0.9419</td>
      <td>4.76</td>
    </tr>
    <tr>
      <td>Iris(3 features)(6)</td>
      <td>36</td>
      <td>26</td>
      <td>10</td>
      <td>139</td>
      <td>126</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.933</td>
      <td>0.904</td>
      <td>27.78</td>
    </tr>
    <tr>
      <td>Vowel (4)</td>
      <td>36</td>
      <td>34</td>
      <td>2</td>
      <td>36</td>
      <td>14</td>
      <td>0.7462</td>
      <td>0.7538</td>
      <td>0.7307</td>
      <td>0.74</td>
      <td>5.56</td>
    </tr>
    <tr>
      <td>Mnist (50, 30)</td>
      <td>41000</td>
      <td>40861</td>
      <td>139</td>
      <td>27</td>
      <td>30</td>
      <td>0.9121</td>
      <td>0.9165</td>
      <td>0.8669</td>
      <td>0.8781</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>

Table 1: Comparing the performance of sparse network with dense networks by using LEGCNet across different datasets. The approach used for finding non-causal weights is LEGCNet-Fully Trained. Accuracies used are on the Test set. Dense Network - DN, Sparse Network - SN

<table>
  <thead>
    <tr>
      <th>Data($Epochs^*$)</th>
      <th>Flops (SN) - LEGCNet-FT</th>
      <th>Flops (SN) - LEGCNet-PT</th>
      <th>Epochs (SN) LEGCNet-FT</th>
      <th>Epochs (SN) LEGCNet-PT</th>
      <th>Accuracy (SN) LEGCNet-FT</th>
      <th>Accuracy (SN) LEGCNet-PT</th>
      <th>F1-score (SN) LEGCNet-FT</th>
      <th>F1-score (SN) LEGCNet-PT</th>
      <th>%Pruned LEGCNet-PT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Cancer(12)</td>
      <td>54</td>
      <td>42</td>
      <td>38</td>
      <td>24</td>
      <td>0.8686</td>
      <td>0.8759</td>
      <td>0.8656</td>
      <td>0.8643</td>
      <td>30</td>
    </tr>
    <tr>
      <td>Titanic(3)</td>
      <td>52</td>
      <td>29</td>
      <td>20</td>
      <td>36</td>
      <td>0.7177</td>
      <td>0.7249</td>
      <td>0.6843</td>
      <td>0.6969</td>
      <td>48.21</td>
    </tr>
    <tr>
      <td>Banknote(2)</td>
      <td>36</td>
      <td>37</td>
      <td>7</td>
      <td>6</td>
      <td>0.8909</td>
      <td>0.8945</td>
      <td>0.8906</td>
      <td>0.8943</td>
      <td>7.5</td>
    </tr>
    <tr>
      <td>Iris(20)</td>
      <td>40</td>
      <td>40</td>
      <td>166</td>
      <td>173</td>
      <td>0.9</td>
      <td>0.9000</td>
      <td>0.9124</td>
      <td>0.9124</td>
      <td>4.76</td>
    </tr>
    <tr>
      <td>Iris 3f(20)</td>
      <td>26</td>
      <td>28</td>
      <td>126</td>
      <td>135</td>
      <td>0.9</td>
      <td>0.9333</td>
      <td>0.9040</td>
      <td>0.9330</td>
      <td>22.22</td>
    </tr>
    <tr>
      <td>Vowel(5)</td>
      <td>34</td>
      <td>28</td>
      <td>14</td>
      <td>20</td>
      <td>0.7538</td>
      <td>0.7538</td>
      <td>0.7400</td>
      <td>0.7441</td>
      <td>22.22</td>
    </tr>
    <tr>
      <td>MNIST(5)</td>
      <td>40861</td>
      <td>40867</td>
      <td>30</td>
      <td>29</td>
      <td>0.9165</td>
      <td>0.9152</td>
      <td>0.8781</td>
      <td>0.8408</td>
      <td>0.32</td>
    </tr>
  </tbody>
</table>

Table 2: LEGCNet-PT: Comparing Sparse Networks where causality is computed from two different training scenarios - when the network is trained fully (LEGCNet-FT), and when trained till a few epochs (LEGCNet-PT); Iris 3f indicates IRIS dataset with 3 features; $Epochs^*$ - No of epochs used for computing non-causal weights in LEGCNet-PT

## 5 Performance comparison - dense network, LEGCNet-FT, and LEGCNet-PT

We ran the experiments on seven tabular datasets - Cancer, Titanic, Banknote, Iris, Iris (3 features) Vowel and MNIST. The datasets were divided into 80:20 train-test split and the code was run 5 times, each time maintaining different network initialization. The best results from every initialization are reported in tables 1 and 2. We compared the Flops, % pruned (fraction of parameters removed *100), accuracy, f1-scores, and epochs for all methods (dense, LEGCNet-FT and LEGCNet-PT). Table 1 shows the performance comparison of dense network and LEGCNet-FT. Remarkably, LEGCNet-FT achieves notable reductions in Flops without compromising accuracy. Furthermore, LEGCNet-FT converges significantly faster consuming a few epochs compared to the dense network. Specifically, for the Titanic, Vowel, and Cancer datasets, LEGCNet-FT achieves convergence in just half the number of epochs required by the dense network. Nonetheless, both network achieves a similar level of performance - accuracy, and f1-score- without significant differences, thus validating the lottery ticket hypothesis. Table 2 demonstrates the performance of LEGCNet-PT. It shows that LEGCNet-PT performs at par with its counterpart, in terms of convergence speed, Flops, accuracy and F1 scores.

### 5.1 Diagnostics

It is also crucial to examine the impact of our pruning technique on the training process of the model and determine if the relevance of feature importance is maintained and if the pruned network is properly trained or not. To accomplish this goal, we utilized two diagnostic tools, namely WW and SHAP. The WW validates the compliance of our architecture, as reflected in the table 4 and figure 2 plotted for MNIST. The alpha lies between 2.0 and 6.0 on every layer (table 4). The ESD plots of the three types of training (dense, LEGCNet-PT, LEGCNet-FT) manifest a heavy-tailed distribution of eigenvalues on each layer indicating the layers are well-trained (figure 2). A careful observation at figure 2 reveals the following: ESD plot of a layer, where the orange spike on the far right is the tell-tale clue; it's called a Correlation Trap [13]. A Correlation Trap refers to a situation where the empirical spectral distributions (ESDs) of the actual (green) and random (red) distributions appear remarkably similar, with the exception of a small correlation shelf located just to the right of 0. In the random ESD (red), the largest eigenvalue (orange) is noticeably positioned further to the right

<table>
  <thead>
    <tr>
      <th>Data</th>
      <th>Epochs (SN) Random</th>
      <th>Accuracy (SN) Random</th>
      <th>Accuracy (SN) Magnitude</th>
      <th>F1-score (SN) Random</th>
      <th>F1-score (SN) Magnitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Cancer</td>
      <td>42</td>
      <td>0.8759</td>
      <td>0.8905</td>
      <td>0.8656</td>
      <td>0.8814</td>
    </tr>
    <tr>
      <td>Titanic</td>
      <td>35</td>
      <td>0.7201</td>
      <td>0.7201</td>
      <td>0.6842</td>
      <td>0.6906</td>
    </tr>
    <tr>
      <td>Banknote</td>
      <td>6</td>
      <td>0.8945</td>
      <td>0.9018</td>
      <td>0.8943</td>
      <td>0.9015</td>
    </tr>
    <tr>
      <td>iris</td>
      <td>179</td>
      <td>0.9000</td>
      <td>0.9000</td>
      <td>0.9124</td>
      <td>0.9124</td>
    </tr>
    <tr>
      <td>Iris(3 features)</td>
      <td>160</td>
      <td>0.9000</td>
      <td>0.9333</td>
      <td>0.9330</td>
      <td>0.9330</td>
    </tr>
    <tr>
      <td>Vowel</td>
      <td>28</td>
      <td>0.7462</td>
      <td>0.7462</td>
      <td>0.7229</td>
      <td>0.7307</td>
    </tr>
    <tr>
      <td>Mnist</td>
      <td>31</td>
      <td>0.9119</td>
      <td>0.9121</td>
      <td>0.8627</td>
      <td>0.8669</td>
    </tr>
  </tbody>
</table>

Table 3: Results for Random and Magnitude-based pruning;

To Prune or not to Prune

![](./images/900808050829426859_2.jpg)

Figure 2: WW plots for dense and LEGCNet-FT, and LEGCNet-PT networks on MNIST data (layer 1). Layers 2 and 3 are kept in the supplementary file (section A). Plots reveal the correct training of the proposed architectures. WW plots of MNIST for random and magnitude pruning are in supplementary file section B.

![](./images/900808050829426859_3.jpg)

Figure 3: Shap values and feature importance computed on Cancer dataset (Banknote and Titanic plots can be seen in the supplementary file section D) for all the three models - Dense Network, LEGCNet-FT and LEGCNet-PT; the feature importance for the dense network is same as LEGCNet-FT and LEGCNet-PT

and is separated from the majority of the ESD's bulk. This phenomenon indicates the presence of strong correlations in the layer, which can potentially affect the overall behavior and performance of the network. This is the case of a well-trained layer.

**SHAP**: The SHAP values computed for the proposed pruning architecture as well as for the baselines - random and magnitude - indicate that the feature importance is maintained in LEGCNet-PT and LEGCNet-FT when compared with dense (figure 3). However, the baseline pruning methods (random, magnitude pruning) could not maintain the feature consistency as seen in the SHAP plots (supplementary file, section C). Though magnitude pruning shows feature consistency for Cancer, banknote, and Titanic datasets, the random pruning could not.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">Layer1: 784-50</th>
      <th colspan="2">Layer2: 50-30</th>
      <th colspan="2">Layer3: 30-10</th>
    </tr>
    <tr>
      <th>alpha</th>
      <th>alpha_w</th>
      <th>alpha</th>
      <th>alpha_w</th>
      <th>alpha</th>
      <th>alpha_w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dense</td>
      <td>2.19</td>
      <td>1.63</td>
      <td>1.51</td>
      <td>1.88</td>
      <td>1.94</td>
      <td>2.76</td>
    </tr>
    <tr>
      <td>LEGCNet-FT</td>
      <td>2.24</td>
      <td>1.64</td>
      <td>1.51</td>
      <td>1.83</td>
      <td>2.29</td>
      <td>3.20</td>
    </tr>
    <tr>
      <td>LEGCNet-PT</td>
      <td>2.19</td>
      <td>1.71</td>
      <td>1.51</td>
      <td>1.85</td>
      <td>2.73</td>
      <td>3.87</td>
    </tr>
    <tr>
      <td>Random</td>
      <td>2.30</td>
      <td>1.53</td>
      <td>1.70</td>
      <td>1.95</td>
      <td>2.00</td>
      <td>2.84</td>
    </tr>
    <tr>
      <td>Magnitude</td>
      <td>2.19</td>
      <td>1.63</td>
      <td>1.55</td>
      <td>1.85</td>
      <td>1.96</td>
      <td>2.78</td>
    </tr>
  </tbody>
</table>

Table 4: Weight Watcher summary retrieved for all models trained on the same initialization for MNIST

## 6 Discussion and Conclusion

Unlike the current baselines, the percentage of pruned weights is significantly less. This is because only the non-causal weights are pruned, weights that play no role in impacting the loss/accuracy. We argue that the proposed strategy is efficient and accurate, with the additional benefit of passing network fitting and explainability tests, in addition to satisfying the lottery ticket hypothesis (tables 1 and 2 Experimental section.). One of the other salient features of LEGCNet is that, unlike other pruning methods, it does not need the dense network to be trained for the entire cycle of epochs to identify pruning candidates. Rather, such candidates are detected after a few initial epochs so that the retraining can start immediately. This reflects in reduced Flops without compromising key performance indicators (tables 1, 2). Notably, our architecture is validated for correct training via *WW* Statistics as detailed in the diagnostic section previously.

To Prune or not to Prune

The experimental results demonstrated that the proposed pruning method exhibits notable advantages in maintaining feature importance compared to the traditional random and magnitude pruning methods. The feature consistency remained relatively stable after employing the proposed pruning technique, which was not the case for the other two methods. In the case of random and magnitude pruning, significant fluctuations in feature importance were evident after pruning. These fluctuations could potentially hinder the interpretability of the underlying model. However, our proposed pruning method demonstrated remarkable resilience in preserving feature importance, with minimal perturbations observed in SHAP plot patterns enabling a more interpretable and transparent pruned model. For mission-critical tasks on edge devices such as predicting power consumption of applications [1] or forecasting real-time blood glucose prediction, feature explainability on pruned networks are critical as they help determine accurate prediction when dimensionality is a curse. It is crucial to emphasize that our primary focus in this investigation was on the effects of pruning, rather than achieving perfect classification performance. We compared three distinct pruning techniques: random pruning, magnitude pruning, and our pruning approach. Overall, the experimental outcomes validate the superiority of the proposed pruning method. The findings hold great promise for further advancements in network optimization and model explainability. Our pruning approach is yet to be tested on baseline architectures (Resnet, Densenet), and Large Language Models and savings in carbon emission needs to be computed.

## Acknowledgments

Archana Mathur and Nithin Nagaraj would like to thank SERB-TARE (TAR/2021/000206) for supporting the work. Snehanshu Saha would like to thank the DBT-Builder project, Govt. of India (BT/INF/22/SP42543/2021) and SERB-SURE, DST.

## References

[1] Gargi Alavani, Jineet Desai, Snehanshu Saha, and Santonu Sarkar. Program analysis and machine learning based approach to predict power consumption of cuda kernel. *ACM Transactions on Modeling and Performance Evaluation of Computing Systems*, 2023.

[2] Harikrishnan Nellippalil Balakrishnan, Aditi Kathpalia, Snehanshu Saha, and Nithin Nagaraj. Chaosnet: A chaos based artificial neural network architecture for classification. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 29(11), 2019.

[3] William L Ditto and Sudeshna Sinha. Exploiting chaos for applications. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 25(9), 2015.

[4] Philippe Faure and Henri Korn. Is there chaos in the brain? i. concepts of nonlinear dynamics and methods of investigation. *Comptes Rendus de l'Académie des Sciences-Series III-Sciences de la Vie*, 324(9):773–793, 2001.

[5] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. *ICLR 2019*, 2018.

[6] Clive William John Granger. Investigating causal relations by econometric models and cross-spectral methods. In *Investigating causal relations by econometric models and cross-spectral methods*, 1969.

[7] Clive WJ Granger. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica: journal of the Econometric Society*, pages 424–438, 1969.

[8] Rainer Hegger, Holger Kantz, and Thomas Schreiber. Practical implementation of nonlinear time series methods: The tisean package. *Chaos*, 9 2:413–435, 1998.

[9] Luis M. Herrmann, Maximilian Granz, and Tim Landgraf. Chaotic dynamics are intrinsic to neural network training with sgd. In *Neural Information Processing Systems*, 2022.

[10] Misaki Kondo, Satoshi Sunada, and Tomoaki Niiyama. Lyapunov exponent analysis for multilayer neural networks. *Nonlinear Theory and Its Applications, IEICE*, 2021.

[11] Henri Korn and Philippe Faure. Is there chaos in the brain? ii. experimental evidence and related models. *Comptes rendus biologies*, 326(9):787–840, 2003.

[12] Dwight Kuo. Chaos and its computing paradigm. *IEEE Potentials*, 24(2):13–15, 2005.

[13] Yann LeCun, Ido Kanter, and Sara A. Solla. Second order properties of error surfaces: Learning time and generalization. In *NIPS 1990*, 1990.

[14] Guiying Li, Chao Qian, Chunhui Jiang, Xiaofen Lu, and Ke Tang. Optimization based layer-wise magnitude-based pruning for dnn compression. In *International Joint Conference on Artificial Intelligence*, 2018.


To Prune or not to Prune

[15] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Li Shen, Decebal Constantin Mocanu, Zhangyang Wang, and Mykola Pechenizkiy. The unreasonable effectiveness of random pruning: Return of the most naive baseline for sparse training. In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022.

[16] Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. ArXiv, abs/1810.05270, 2018.

[17] Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In *A Unified Approach to Interpreting Model Predictions*, Red Hook, NY, USA, 2017. Curran Associates Inc.

[18] Charles H. Martin, Tongsu Peng, and Michael W. Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. *Nature Communications*, 12, 2020.

[19] Rohan Mohapatra, Snehanshu Saha, Carlos A. Coello Coello, Anwesh Bhattacharya, Soma S. Dhavala, and Sriparna Saha. Adaswarm: Augmenting gradient-based optimizers in deep learning with swarm intelligence. *IEEE Transactions on Emerging Topics in Computational Intelligence*, 6(2):329–340, 2022.

[20] Nithin Nagaraj and Prabhakar G Vaidya. Multiplexing of discrete chaotic signals in presence of noise. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 19(3), 2009.

[21] Catia Prandi, Silvia Mirri, Stefano Ferretti, and Paola Salomoni. On the need of trustworthy sensing and crowdsourcing for urban accessibility in smart city. *ACM Trans. Internet Technol.*, 18(1), October 2017.

[22] Xin-Yao Qian and Diego Klabjan. A probabilistic approach to neural network pruning. In *International Conference on Machine Learning*, 2021.

[23] Snehanshu Saha, Nithin Nagaraj, Archana Mathur, Rahul Yedida, and Sneha H R. Evolution of novel activation functions in neural network training for astronomy data: habitability classification of exoplanets. *The European Physical Journal. Special Topics*, 229:2629 – 2738, 2020.

[24] Zuowei Shen, Haizhao Yang, and Shijun Zhang. Nonlinear approximation via compositions. *Neural networks : the official journal of the International Neural Network Society*, 119:74–84, 2019.

[25] JC Sprott. Is chaos good for learning. *Nonlinear dynamics, psychology, and life sciences*, 17(2):223–232, 2013.

[26] Lin Zhang, Ling Feng, Kan Chen, and Choy Heng Lai. Edge of chaos as a guiding principle for modern neural network training. *ArXiv*, abs/2107.09437, 2021.

[27] Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes over-parameterized deep relu networks, 2018.

To Prune or not to Prune

## Appendix
Detailed implementation is available at the Github repository: https://github.com/RAJAN13-blip/chaotic-pruning

## A Plots on WeightWatchers run on different datasets
It is also crucial to examine the impact of our pruning technique on the training process of the model and determine if the relevance of feature importance is maintained and if the pruned network is properly trained or not. To accomplish this goal, we utilized two diagnostic tools, namely WeightWatcher (WW) and SHAP. The alpha lies between 2.0 and 6.0 on every layer, however, layer 3 of the dense network and the (baseline) magnitude pruning method is not trained well (alpha < 2.0). The ESD plots of the three types of training (dense, LEGCNet-PT, LEGCNet-FT) manifest a heavy-tailed distribution of eigenvalues on each layer indicating the layers are well-trained (figure 4). A careful observation at figure 4 reveals the following: ESD plot of a layer, where the orange spike on the far right is the tell-tale clue; it's called a Correlation Trap, A Correlation Trap refers to a situation where the empirical spectral distributions (ESDs) of the actual (green) and random (red) distributions appear remarkably similar, with the exception of a small correlation shelf located just to the right of 0. In the random ESD (red), the largest eigenvalue (orange) is noticeably positioned further to the right and is separated from the majority of the ESD's bulk. This phenomenon indicates the presence of strong correlations in the layer, which can potentially affect the overall behavior and performance of the network. Layers have an overlap of random and original ones when they have not been trained properly because they look almost random, with only a little bit of information present. And the information the layer learned may even be spurious. This is the case of a well-trained layer.

![](./images/900808050829426859_4.jpg)

Figure 4: WW plots for layer-wise Dense and LEGCNet-FT (figures 3 and 4) and LEGCNet-PT (figures 5 and 6) networks on MNIST data. Plots reveal the correct training of the proposed architectures.

To Prune or not to Prune

## B Remaining WW plots for random and magnitude pruning

![](./images/900808050829426859_5.jpg)

Figure 5: WW plots for Random-Pruned network

![](./images/900808050829426859_6.jpg)

Figure 6: WW plots for Magnitude-Pruned network

To Prune or not to Prune

C Shap values and feature importance computed on Vowel, Banknote, and Titanic datasets for the 2 models - Random pruning Network, Magnitude based pruning network

![](./images/900808050829426859_7.jpg)

Figure 7: Shap values and feature importance computed on Vowel, Banknote, and Titanic datasets for the 2 models - Random pruning Network, Magnitude based pruning network; the feature importance for the dense network is different from the original dense network.

To Prune or not to Prune

## D Shap values and feature importance computed on Cancer, Banknote and Titanic datasets for all the three models - Dense Network, LEGCNet-FT and LEGCNet-PT

![](./images/900808050829426859_8.jpg)

Figure 8: Shap values and feature importance computed on Cancer, Banknote and Titanic datasets for all the three models - Dense Network, LEGCNet-FT and LEGCNet-PT; the feature importance for the dense network is same as LEGCNet-FT and LEGCNet-PT