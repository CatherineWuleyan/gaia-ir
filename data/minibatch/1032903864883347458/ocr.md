# Mask in the Mirror: Implicit Sparsification

Tom Jacobs
CISPA Helmholtz Centre for Information Security
66123 Saarbucken, Germany
tom.jacobs@cispa.de

Rebekka Burkholz
CISPA Helmholtz Centre for Information Security
66123 Saarbucken, Germany
rebekka.burkholz@cispa.de

## Abstract
Sparsifying deep neural networks to reduce their inference cost is an NP-hard problem and difficult to optimize due to its mixed discrete and continuous nature. Yet, as we prove, continuous sparsification has already an implicit bias towards sparsity that would not require common projections of relaxed mask variables. While implicit rather than explicit regularization induces benefits, it usually does not provide enough flexibility in practice, as only a specific target sparsity is obtainable. To exploit its potential for continuous sparsification, we propose a way to control the strength of the implicit bias. Based on the mirror flow framework, we derive resulting convergence and optimality guarantees in the context of underdetermined linear regression and demonstrate the utility of our insights in more general neural network sparsification experiments, achieving significant performance gains, particularly in the high-sparsity regime. Our theoretical contribution might be of independent interest, as we highlight a way to enter the rich regime and show that implicit bias is controllable by a time-dependent Bregman potential.

## 1 Introduction
Deep learning continues to impress across disciplines ranging from language and vision [59] to drug design [66, 32] and even fast matrix multiplication [17]. Yet, it comes at immense costs, as it relies on increasingly large neural network models. While training such massive models with first-order methods like variants of Stochastic Gradient Descent (SGD) is a considerable challenge and often requires large scale compute infrastructure [33], even higher costs are incurred at inference time, if the trained models are frequently evaluated [73, 46].

Sparsifying such neural network models is thus a pressing objective. It not only holds the promise to save computational resources, it can also improve generalization [22, 53], interpretability [8, 30], denoising [31, 69], and verifiability [49, 1]. However, at its core is a hard large-scale nested optimization problem combining multiple objectives. In addition to minimizing a typical neural network loss $\min_{w \in \mathbb{R}^n} f(w)$ (and its generalization performance), we wish to rely on the smallest possible number of weights, effectively minimizing the $L_0$ norm $\min_{w \in \mathbb{R}^n} ||w||_{L_0}$. This is an NP-hard problem that is also practically hard to solve due to its mixed discrete and continuous nature. This becomes more apparent when we reformulate it in a way as the best performing sparsification methods approach it.

Among such approaches that achieve high-sparsity while maintaining high generalization performance, albeit being computationally expensive, are iterative pruning strategies and continuous sparsification methods, which explicitly identify for each weight parameter $w$ of a neural network a

binary mask $m \in \{0, 1\}$ that signifies whether a parameter is pruned and thus set to zero ($m = 0$) or not ($m = 1$), effectively parameterizing the network with parameters $x = m \odot w$. The introduction of the additional mask parameters $m$ turns the sparsity objective into a discrete $L_1$ penalty of $m$ as $||w \odot m||_{L_0} = \sum_i m_i$ subject to $m_i \in \{0, 1\}$, where $\odot$ denotes elementwise multiplication and we assume that $w \neq 0$ if $m = 1$. The $L_1$ objective is already more amenable to continuous optimization than the original $L_0$ objective [45]. The big challenge arises from the fact that $m$ is binary.

Continuous sparsification addresses this issue by relaxing the optimization problem to continuous or even differentiable variables $m$, often with $m \in [0, 1]$ by learning a parameterization $m = g(s)$ with $g : \mathbb{R} \to [0, 1]$ like a sigmoid. We later discuss different conceptual choices of $g$ and their potential limitations for sparsification from the perspective of mirror flows. This way, the problem becomes solvable with standard first-order optimization methods. Yet, moving from the continuous space back to the discrete space is error-prone. Regularizing and projecting $m$ towards binary values $\{0, 1\}$ is generally problematic, requires careful tuning, and often entails robustness issues.

Our main insight is that these steps are not all necessary, as the continuous parameterization $m \odot w$ with $m \in \mathbb{R}$ already induces an implicit regularization towards sparsity. We shed light on this fact by deriving the corresponding mirror flow and implicit bias that is induced by the overparameterization, which results from doubling the number of trainable parameters from $x$ to $m$ and $w$. We remark that in [78], the spred algorithm is proposed. Together with [77] it has been shown that for the parameterization $m \odot w$ the optimization problem is equivalent to LASSO. Nevertheless, this does not explain why the overparameterization performs better than LASSO as shown in experiments. The implicit bias framework provides such an explanation.

What is the main advantage of such an implicit regularization over an explicit one? Previously, we have not described how to join our two objectives, i.e., minimizing the loss of a neural network and sparsity. A common strategy is to combine both in a single objective as a sum:

$$
\min _{w \in \mathbb{R}^{n}, m \in\{0,1\}^{n}} f(m \odot w)+\lambda||m||_{L_{1}},
$$

where the explicit regularization parameter $\lambda > 0$ decides about their trade-off. A result might accept application irrelevant low performance if the network is very sparse. Another alternative would be to strongly restrict the parameters of the network [62, 63, 45] or to turn the sparsity objective into a constraint [26], which does not directly optimize it.

By exploiting implicit rather than explicit regularization towards sparsity, the problem gets indirectly reformulated as a hierarchical optimization problem, mitigating the discussed issues:

$$
\min _{x \in \mathbb{R}^{n}: f(x)=c}||x||_{L_{1}}. \tag{1}
$$

The main idea follows the same philosophy as the lottery ticket hypothesis [22]. From a range of models which all attain optimal training loss $c$ we choose the sparsest model. In other words, we aim to find a subnetwork with a similar accuracy as a dense network. Accordingly, we call our approach PILoT (Parametric Implicit Lottery Ticket). Instead of having two competing objectives, we enforce cooperation between the two objectives by subjugating the sparsification.

The optimization problem (1) is solvable using the implicit bias framework. It is well known that standard gradient flow has an implicit bias which acts as an $L_2$ regularization [50, 3]. Different parameterizations though induce a different implicit bias [55, 27, 72, 41] in the (stochastic) gradient flow framework. We utilize this framework to understand the implicit bias induced by continuous sparsification, in particular, the parameterization $m \odot w$ acting approximately as $L_1$ regularization.

However, the existing framework on its own is not enough to induce an $L_1$ regularization of high practical utility, as we cannot control its strength. We theoretically extend the framework by proposing a novel regularization on $m$ and $w$. This regularization leads to even sparser lottery tickets, while still prioritizing the main optimization goal: accuracy. In this sense, it can be interpreted as tuneable implicit regularization. It is defined by a time-dependent Bregman potential, which controls the bias.

Contributions:
- We gain novel insights into continuous sparsification by highlighting its implicit bias towards sparsity that is induced by doubling the number of trainable parameters.
- To exploit it for sparsification, we propose PILoT that can overcome certain drawbacks of the implicit bias framework and achieve controllable implicit $L_1$ regularization. To the best

our knowledge, we are the first to introduce the implicit bias with an explicit regularization resulting in a mirror flow with a time-dependent Bregman potential.

- We provide convergence results for (quasi)-convex loss functions (Theorem 3.2) and opti- mality for underdetermined linear regression (Theorem 3.3) with time-dependent Bregman potential.
- Improving results by [2, 41], we replace convexity with the Polyak-Łojasiewicz inequality, quasi-convexity and a growth condition on the Bregman potential (see Theorem 2.3).
- In experiments for diagonal linear networks and vision benchmarks, PILoT consistently outperforms baseline sparsification methods such as spred, which demonstrates the utility of our theoretical insights.

### 1.1 Related work
Neural network sparsification A multitude of neural network sparsification methods have been proposed that can be categorized according to different criteria [44]. A popular criterion is, for instance, whether they save primarily computational and memory costs at inference, or also during training, which is linked to the time of pruning, i.e., initially [23, 40, 67, 68, 57, 52, 67, 43, 25, 20], early during training [15, 13], during training like continuous sparsification [65, 36, 62, 56] or within multiple pruning-training iterations [29, 21, 74, 60, 24]. Other distinguishing factors are which type of sparsity the methods seek, if they focus on saving computational resources and memory during training or which methodological approach they follow.

Unstructured sparsity In this work, we focus on unstructured sparsity, i.e., the fraction of pruned weights, and thus seek to remove as many weight entries as possible, which can achieve generally the highest sparsity ratios while maintaining high generalization performance. Structured sparsity, which usually obtains higher computational gains on modern GPUs [37, 70, 39], could also be realized in the continuous sparsification setting, for instance, by learning neuron-, group, or even layer-wise masks. Yet, this would not enjoy the same theoretical benefits as we derive here by showing that the unstructured continuous $m \odot w$ parameterization induces a mirror flow, whereas for example the neuron-wise mask does not (see Section C).

Iterative pruning Iterative pruning often refers to the Lottery Ticket Hypothesis (LTH) [21], which conjectures the existence of sparse subnetworks of larger dense source networks that can achieve the same accuracy as the dense network when both are trained [23, 42, 48, 51, 54, 7, 20, 5, 6, 11, 19]. In addition to the sparse structure, iterative pruning often tries to identify a trainable parameter initialization, indirectly also implementing an approximate $L_0$-regularization. In repeated prune-train iterations, trained weights are thresholded according to an importance score like magnitude. After- wards, the remaining parameters are free to adapt to data in a new training run and not additionally regularized by a sparsity penalty (like $L_1$). Our proposal PILoT can be combined with such iterative schemes. Our experiments show that this can boost the performance of state-of-the-art schemes like Iterative Magnitude Pruning (IMP) [21] and Learning Rate Rewinding (LRR) [47, 24].

Continuous sparsification Continuous sparsification characterizes a collection of methods that can compete with iterative pruning techniques, while often requiring fewer training epochs [65, 36]. The method [62] lends its name to the general approach, in which the mask is relaxed to a continuous variable. In general, continuous sparsification can be combined with a probabilistic approach where $m$ is interpreted as a probability [45, 75, 76]. Yet, other parameterizations of $m$ that are not restricted to $[0,1]$ such as Powerpropagation can also be utilized to regularize towards higher sparsity [63]. Furthermore, the spred algorithm proposed in [78] shows equivalence for $m \odot w$ with weight decay to LASSO, which does not explain the performance gain. We show why spred can outperform LASSO and improve upon the algorithm with PILoT, with the help of our novel framework. For a survey of other methods see [38]

Implicit bias The implicit bias of (S)GD is a well-studied phenomenon [9, 41, 72, 27, 28, 10] and can in certain cases be described by a mirror flow or mirror descent (in the discrete case with finite learning rate) [41]. Originally, mirror descent was proposed to generalize gradient descent and other first-order methods in convex optimization [2, 61, 4, 50, 3]. Moreover, it has been used to study the implicit regularization of SGD in diagonal linear networks [55, 16]. More recently, it also has been applied to analyze the implicit bias of attention [64].

While [41] has shown that different parameterizations have a corresponding mirror flow, we find that $m \odot w$ with our proposed explicit regularization, PILoT, gives rise to a corresponding time- dependent mirror flow. Its time dependence gives us means to control the implicit bias, while still achieving convergence. Time-dependent mirror descent has so far only been studied in the discrete case as a general possibility [58]. The time dependence also naturally arises in SDE modelling, yet, without control of the implicit bias [55, 16]. Here, we not only highlight a practical use case for time-dependent Bregman potentials, we also derive a way to control and exploit it.

Optimization and convergence proofs Loss landscapes and the convergence of first-order methods is a large field of study [34, 18] in its own right. We draw on literature that shows convergence by using the Polyak-Łojasiewicz inequality [71, 12], which is a more realistic assumption ML contexts than, for example, convexity. Because it can hold locally true for loss function in ML.

## 2 Continuously training a mask induces an implicit bias

Our first objective is to characterize the implicit bias that is induced by learning with the param- eterization $x = m \odot w$ that reflects the basic continuous sparsification setup, where $x = m \odot w$ denotes elementwise multiplication by utilizing the implicit bias and mirror flow framework. To complete our analysis, we will also show the convergence of the loss and provide optimality guar- antees. Consider the gradient flow associated with minimizing the continuously differentiable loss function $f$: $dx_t = -\nabla f(x_t)dt$, $x_0 = x_{\text{init}}$. Using this gradient flow framework, [41] show that a reparameterization or overparameterization of the parameters $x$ leads to a mirror flow. A mirror flow informally minimizes a potential in the background, for example, the $L_1$ or $L_2$-norm. Concretely, let $R: \mathbb{R}^n \to \mathbb{R}$ be a differentiable function, then the mirror flow is described by

$$
d\nabla_x R(x_t) = -\nabla_x f(x_t)dt, \quad x_0 = x_{\text{init}}.
$$

[41] provide sufficient conditions for a parameterization $g: M \to \mathbb{R}^n$ to induce a mirror flow, where $M$ is smooth sub-manifold in $\mathbb{R}^D$ for $D \geq n$. The parameterization $m \odot w$ falls in this category. Moreover, the resulting potential is an interpolation between the $L_1$ and $L_2$ norm. Therefore, we can use the parameterization to induce sparsity by steering the interpolation (towards $L_1$). The corresponding mirror function $R$ is given in Theorem 2.1.

Theorem 2.1 Let the initialization of $m$ and $w$ satisfy $m_{0,i} > |w_{0,i}|$ for all $i \in [n]$. Then the corresponding mirror function is:

$$
R(x) := \frac{1}{4} \sum_{i=1}^n x_i \text{arcsinh} \left( \frac{x_i}{2u_{0,i}v_{0,i}} \right) - \sqrt{x_i^2 + 4u_{0,i}^2v_{0,i}^2} - x_i log \left( \frac{u_{0,i}}{v_{0,i}} \right) \tag{2}
$$

where $u_{0,i} = \frac{m_{0,i} + w_{0,i}}{\sqrt{2}}$ and $v_{0,i} = \frac{m_{0,i} - w_{0,i}}{\sqrt{2}}$. Furthermore, $R$ is a Bregman function.

Proof. The result follows directly from applying Theorem 4.16 in [41].

Theorem 2.1 implies the following: a) The global minima of $R$ is at the initialization $x_0 = m_0 \odot w_0$. b) The Lipschitz coeficient of $R$ depends on the initalization. The Lipschitz coeficient $L_R$ of (2) is $L_R = \frac{1}{\min_i 2u_{0,i}v_{0,i}}$, determining the smoothness of the potential. Following these two observations we make the following remark about Theorem 2.1.

Remark 2.1 Note that when the initialization is zero, i.e., $w_0 = 0, m_0 = \sqrt{a}$ with $a \geq 0$ then (2) is the hyperbolic entropy. The hyperbolic entropy is

$$
\sum_{i=1}^n x_i \text{arcsinh} \left( \frac{x_i}{a} \right) - \sqrt{x_i^2 + a^2}
$$

Theorem 2 of [72] characterizes the behavior in the limit for this case. For the hyperbolic entropy in case $a \to 0$ and $|\frac{x}{a}| \to \infty$,

$$
R(x) \sim log \left( \frac{1}{a} \right) ||x||_{L_1}.
$$

This means an $L_1$ bias is induced when $a$ is small. Nevertheless, we need an exponentially small $a$ compared to $x$ to get there as shown in [72], which can lead to numerical problems. Furthermore, $m_0 = w_0 = 0$ is a saddle point which can slow down training (exponentially) [14]. Additionally, the asymptotic result only holds for initializing at zero. Note $L_R = a$ in this case.


Remark 2.1 shows the potential of using the implicit bias to induce sparsity. To actualize this, we need to solve the two challenges posed in the remark. Both are remedied in Section 3.

In addition to this promising formulation of implicitly minimizing an $L_1$ norm with the use of the mirror framework, we can get convergence results. These results make it clear why implicit regularization is preferable over explicit regularization. The convergence result from [41] is stated for our setting. Furthermore, the theorem is extended for a specific class of Bregman functions.

Theorem 2.2 (Theorem 4.14 [41]) Assume that $f$ is quasi-convex, $\nabla f$ is locally Lipschitz and $\text{argmin}\{f(x)|x \in \mathbb{R}^n\}$ is non-empty. Then as $t \to \infty$, $x_t$ converges to some critical point $x^*$. Moreover if $f$ is convex $x_t$ converges to a minimizer of $f$.

In Theorem 2.2 it is shown that with implicit regularization an optimal solution to the original optimization problem can be reached. In contrast, explicit regularization makes this not possible, by definition. Because the optimization problem has fundamentally changed. Showing the benefit of implicit over explicit.

Remark 2.2 In case of explicit regularization, at the end of the training, the regularization could be turned off, to allow for finding a more optimal solution. Nevertheless, it then risks losing the benefits of the regularization. For example, if the basin of attraction contains non-sparse critical points.

For the extension, the convexity constraint is replaced by the Polyak-Łojasiewicz (PL) inequality in the theorem. The PL-inequality is a more realistic constraint in a machine learning context as loss functions are not locally convex but can satisfy the PL inequality locally [71, 12]. The PL-inequality for a continuously differentiable function $f$ is
$$
||\nabla f(x)||_{L_2}^2 \geq \lambda(f(x)-f(x^*)) \quad \forall x \in \mathbb{R}^n \tag{3}
$$
for some $\lambda > 0$ and global minima $x^*$ of $f$. This allows us to state the modified theorem.

Theorem 2.3 Consider the same setting as Theorem 2.2. Assume $R$ satisfies for all $x \in \mathbb{R}^n$,
$$
z^T\left(\nabla^2 R(x)\right)^{-1} z \geq \sigma||z||_{L_2}^2 \quad \forall z \in \mathbb{R}^n. \tag{4}
$$
Furthermore, assume $f$ satisfies the PL-inequality (3). Then $x_t$ converges to a minimizer of $f$. Furthermore, the loss converges linearly with rate $\sigma \lambda$.

Proof. The evolution of $f(x_t)-f(x^*)$ is described by $df(x_t) = -\nabla f(x_t)^{\top}\left(\nabla^2 R(x_t)\right)^{-1} \nabla f(x_t)dt$. From (4) and (3) the evolution is bounded by
$$
df(x_t) \leq -\sigma||\nabla f(x_t)||_{L_2}^2 dt \leq -\sigma \lambda(f(x_t)-f(x^*))dt.
$$
Applying Gronwall's Lemma concludes the proof. $\square$

Note that Theorem 2.3 holds in the same (general) setting as Theorem 2.2. Also, note that the PL-inequality together with quasi-convexity does not imply convexity. Theorem 2.3 holds for our setting. In this case, it follows from a direct computation that
$$
\left(\nabla^2 R(x)\right)^{-1} = \text{diag}\left(\sqrt{x^2+4u_{0,1}^2v_{0,1}^2}, \dots, \sqrt{x^2+4u_{0,n}^2v_{0,n}^2}\right).
$$
This implies that $\eta = 2\min_i u_{0,i}v_{0,i}$ in Theorem 2.3, which again highlights the importance of the initialization.

Finally, in the case of under-determined linear regression, we can derive optimality conditions in the form of KKT conditions of $R$. Consider a data set $(z_j,y_j)_{i=1}^d$ with $z_j \in \mathbb{R}^n$ and $y_j \in \mathbb{R}$. Let $Z=(z_1,\dots z_d)$ and $Y=(y_1,\dots y_d)$. For the regression to be called underdetermined $n > d$.

Theorem 2.4 (Theorem 4.17 [41]) In case of under-determined regression consider the loss function $f(x)=\tilde{f}(Zx-Y)$. Assume $f$ satisfies the conditions of Theorem 2.2. Then $x_t$ converges to $x^*$ such that
$$
x^* = \text{argmin}_{Zx=Y} R(x)
$$

Note that Theorem 4.17 of [41] only uses quasi-convexity of the loss. Theorem 2.4 guarantees that the optimization problem is solved while implicitly minimizing the potential R. Thus choosing the sparsest model out of the models that predict the data perfectly. This highlights another benefit of implicit regularization over explicit regularization.

In this section, we have shown the viability of using the implicit bias framework to induce an implicit regularization. Furthermore, we have given two known benefits of using the implicit bias framework over explicit regularization. The benefits are convergence to the optimal solution of the original problem and optimality in the case of underdetermined regression. To add to this, we have extended the convergence theorem using the PL-inequality in 3. Moreover, we have highlighted the importance of the initialization of $m_0$ and $w_0$ with the influence on the smoothness of the Bregman potential and convergence of the loss. The initialization insight is used to improve upon spred [77] as their initialization has scaling $2u_0v_0=0$, it follows already from the mirror flow framework that $2u_0v_0=1$ is a better initialization. Generally, we do not initialize at zero though, and scaling needs to be exponentially small to get a good approximation of the $L_1$ norm potentially making it hard to escape the saddle point. This brings us to the next section where we solve these problems and utilize the implicit bias to solve the optimization problem in 1.

## 3 Control of the implicit bias with an explicit regularization

Next, we introduce a novel explicit regularization that induces an implicit $L_1$ regularization. As a tool, we use a time-dependent Bregman potential to realize this. We show that the "rich regime" is reached by applying this regularization and without having to initialize at zero. This is illustrated in Section 4, where a simulation study is provided for a diagonal linear network. Furthermore, convergence and optimality guarantees are derived similarly as in the previous section.

Consider the explicit regularization weight decay on the $m$ and $w$ parameters. Let the regularization be given by the function $h: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$

$$
h(m, w)=\sum_{i=1}^n m_i^2 + w_i^2.
$$

The related optimization problem thus becomes:
$$
\min _{m, w \in \mathbb{R}^n} f(m \odot w)+\alpha h(m, w),
$$

where $\alpha>0$ is a constant. In the gradient flow, $\alpha$ is allowed to depend on the time, i.e., $\alpha$ is replaced by $\alpha_t$. This controls the implicit bias. The resulting potential is a time-dependent Bregman function, which is given by Theorem 3.1. The deeper underlying reason why this specific regularization gives us control via the Bregman potential is that it is commuting in the sense of [41] with the parameterization $m \odot w$. Moreover, regularizing only $m$ does not satisfy the necessary condition provided in [41]. Nevertheless, these sufficient and necessary conditions are only heuristics for time-dependent mirror flow as they only have been shown for mirror flow.

Theorem 3.1 Let $|w_{0, i}| \leq m_{0, i}$ for all $i \in[n]$, then the time-dependent Bregman potential is given by
$$
R_{a_t}(x)=\frac{1}{2} \sum_{i=1}^n x_i \arcsinh \left(\frac{x_i}{a_{t, i}}\right)-\sqrt{x_i^2+a_{t, i}^2}-x_i \log \left(\frac{u_{0, i}}{v_{0, i}}\right),
$$

with $a_{t, i}=2 u_{0, i} v_{0, i} \exp \left(-2 \int_0^t \alpha_s d s\right)$ and $u_{0, i}=\frac{m_{0, i}+w_{0, i}}{\sqrt{2}}$ and $v_{0, i}=\frac{m_{0, i}-w_{0, i}}{\sqrt{2}}$. The time-dependent Bregman potential satisfies
$$
d \nabla R_{a_t}\left(x_t\right)=-\nabla f\left(x_t\right) d t, \quad x_0=m_0 \odot w_0.
$$

Proof. The proof is given in the appendix. The main steps are: a) Deriving the evolution of the gradient flow (Lemma A.1). b) Showing that it satisfies the time-dependent mirror flow (Lemma A.2).

Furthermore, note that the global minimum of $R_{a_t}$ is located at
$$
\nabla R_{a_t}(x)=0 \Leftrightarrow x=\exp \left(-2 \int_0^t \alpha_s d s\right) \odot m_0 \odot w_0.
$$

This gives us control over the positional implicit bias and the asymptotic behavior with $\alpha_t$. For $a \to 0$ and $\left|\frac{x}{a}\right| \to \infty$, we receive

$$R_a(x) \sim \log\left(\frac{1}{a}\right) ||x||_{L_1}.$$

Note that the term $x_i \log\left(\frac{u_{0,i}}{v_{0,i}}\right)$ does not play a role in the asymptotics. This is due to the $\log(\frac{1}{a})$ in front of the other term. The asymptotics are illustrated in the case of $n=1$ in Figure 1c. We indeed observe that increasing $a$ moves the minimum to the origin, leading to an $L_1$ regularization. This implies initializing at zero with an exponentially small scaling (i.e. $a \to 0$) is not necessary. Therefore, the explicit regularization solves the problems described in the previous section.

In the rest of the section, we provide similar convergence and optimality guarantees as in Section 2.

**Theorem 3.2** Assume $f$ is quasi-convex, $\nabla f$ is locally Lipschitz and $\text{argmin}\{f(x)|x \in \mathbb{R}^n\}$ is non-empty. Assume $\alpha_t \geq 0$ for all $t \geq 0$ and that $\int_0^t \alpha_s ds < \infty$ for $t \in [0, \infty) \cup \{\infty\}$. Then as $t \to \infty$, $x_t$ converges to some critical point $x^*$. Furthermore, if $f$ is either convex or both quasi-convex and satisfies the PL-inequality 3. Then $x_t$ convergences to an interpolator $x^*$ that is a minimizer of $f$. Furthermore, in the PL-inequality case, the loss converges linearly such that there is a constant $C > 0$ such that

$$f(x_t) - f(x^*) \leq B exp\left(-\lambda a_\infty t\right), \tag{5}$$

where $B = (f(x_0) - f(x^*)) exp\left(C||x^*||_{L_2} \int_0^\infty \alpha_s ds\right)$ with $C$ depending on the smoothness of the loss function.

Proof. The main steps are: a) Showing the iterates are bounded and convergence to a critical point (Lemma A.3). b) Convergence of the loss (Theorem A.1). Notable tools are a time-dependent Bregman divergence used to bound the iterates and $\alpha_t \geq 0$ converges to zero, resulting in eventually a non-increasing evolution of the loss.

Theorem 3.2 guarantees convergence in the case of implicit regularization. Explicit $L_1$ regularization, on the other hand, could not achieve the same result due to the additional penalty of the weight magnitude, which we will also highlight in experiments.

Nevertheless, note that the constant $B$ in (5) could be large. Furthermore, to reach the implicit $L_1$-regularization, $a_\infty$ needs to be exponentially small similarly as in Theorem 2 in [72]. These two supposed drawbacks also reveal, where the method will work, namely, in overparameterized settings where the solution $x^*$ should have less active parameters. Then $B$ is potentially relatively small.

**Remark 3.1** If $\nabla f$ is one-sided inversely Lipschitz, a speed-up is possible. The quantity that needs to be bounded for convergence is $-\nabla f(x_t)^\top x_t$. In this case, we get

$$-\nabla f(x_t)^\top x_t \leq -\nabla f(x_t)^\top x^* - ||x_t - x^*||_{L_2}^2 \leq C||x^*||_{L_2} - ||x_t - x^*||_{L_2}^2$$

where $C$ is the bound on the smoothness of the loss function $f$. This implies that when the interpolator $x^* \approx 0$ the right-hand side is negative, leading to a speed-up. This condition is also known as coercive.

The main takeaway from this is that $B$ is manageable and even can help given that both optimization problems align. Finally, we show optimality in the case of under-determined linear regression, providing another benefit of PILoT over explicit regularization.

**Theorem 3.3** In case of under-determined regression consider the loss function $f(x) = \tilde{f}(Zx - Y)$. Assume $f$ satisfies the conditions with at least one of the convergence criteria of Theorem 3.2. Then $x_t$ converges to $x^*$ such that

$$x^* = \text{argmin}_{Zx=Y} R_{a_\infty}(x) \tag{6}$$

Proof. The main step is showing that the KKT conditions of (6) are satisfied (Theorem A.2).

The introduced new explicit regularization enjoys the same benefits as an implicit regularization. We have solved the problems posed in the previous section, creating a new viable continuous sparsification method. Moreover, we have shown that the implicit bias framework results transfer to our new regularization in the form of convergence and optimality guarantees. Furthermore, we have

seen that a time-dependent Bregman potential is directly linked to a parameterization with an explicit regularization. This opens the door for creating better regularization strategies in general.

To summarize, we have shown using the extended implicit bias framework that both spred and PILoT reach higher accuracy than LASSO as it allows to reach an exact interpolator as in Theorem 3.2 and optimality in Theorem 3.3. Note that spred is trained with constant $\alpha$, the insights from the extended frameworks still transfer as we always train for a finite number of epochs. Nevertheless, the framework shows that training with a constant $\alpha$ is suboptimal. Thus in PILoT we allow for changing $\alpha$. Together with the new initialization we have uncovered a novel algorithm for sparsification.

## 4 Experiments

With three different types of experiments, we highlight the merit of PILoT, which solves the continuous sparsification problem (1). First, the result of Theorem 3.3 is illustrated by synthetic data for sparse linear reconstruction. Our optimality guarantees distinguish us from competing methods, as PILoT converges not only faster but achieves perfect recovery and thus a lower mean squared error. Our second set of experiments showcases one-shot pruning on CIFAR10 and CIFAR100 [35], where we remove a given percentage of the smallest weights. Our third sets of experiments further highlight the utility of PILoT within iterative pruning like LRR and IMP. The parameters $m$ and $w$ are initialized in such a way that the scaling is constant i.e. for all $i \in [n]$, $2u_{0,i}v_{0,i}$ is a constant. In all experiments, we have set it to 1. The choice is based on discretization of the gradient flow. Note, all experiments are repeated three times with independent random seeds. We report averages and 95% confidence intervals.

Diagonal Linear Network In this section, we illustrate the benefit of the initialization and explicit regularization on a diagonal linear network. Furthermore, this highlights the importance of choosing the right schedule for $\alpha_t$. The mean squared error loss function $f$ is used for an under-determined linear regression problem. The hyperparameters are set to $d = 40, n = 100$ and sample $z_j \sim N(0, \mathbb{I}_n)$. The ground-truth $x^*$ is set such that $||x^*||_{L_0}=5$. Furthermore, the network parameters are initialized with $x \sim N(0, \mathbb{I}_n \frac{1}{\sqrt{n}})$. The stepsize is $\eta=10^{-4}$. The trajectories are averages over 5 initializations. We show the distance between the ground truth and parameter value, i.e., $||x_t - x^*||_{L_2}$. We compare PILoT with an $L_1$ regularization on the mask $m$ and a standard $L_1$ regularization on $x$. In addition, we look at the spred initialization.

Figure 1a highlights the importance of the initialization. We see that with spred initalization $2u_0v_0=0$, it is impossible to get close to the ground truth, for all regularization strengths. This highlights the necessity of our proposed PILoT initialization that allows us to provably reach the ground truth, even from a non-zero initialization $x_0$.

Figure 1b demonstrates that the parameterization $m \odot w$ with our regularization (PILoT) outperforms the baselines. The best-performing schedule for our regularization is a strong decaying schedule whereas for the other two, a constant regularization works best. This experimentally confirms Theorem 3.3 and Remark 3.1 and is another improvement over spred. Furthermore, regularizing only $m$ does not provide the same control as our regularization, which becomes clearer in Figure 4. The $L_1$ regularization on $x$ performs as expected. A constant schedule leads to the best performance, with a too small or large constant leading to worse performance. In addition, decaying schedules perform also worse illustrating Remark 2.2. We note that even with a constant schedule our regularization is competitive with the other two settings, this is agreement with the analysis [78, 77].

The main advantage of PILoT is that it enables us to enter the rich regime and thus obtain an implicit $L_1$ regularization, as illustrated by Figure 1c supporting Theorem 3.1.

One-shot sparsification All considered models are trained for 150 epochs and then the smallest % of the weights are removed. Our baselines are the original parameterization, i.e., $x=x$, the standard $L_1$ regularization, as well as existing parameterizations and continuous sparsification methods [62, 63]. Figure 2 shows that the proposed $m \odot w$ parameterization outperforms the baseline. To perform better than other sparsification methods, however, our new regularization, PILoT, is necessary. A simple $L_1$ regularization can also work well. Nevertheless, it is less robust when we increase the regularization strength and struggles in the high-sparsity regime. While our methods with strong regularization also drop in performance, they still achieve high performance in the high-sparsity

![](./images/1032903864883347458_1.jpg)
![](./images/1032903864883347458_2.jpg)

![](./images/1032903864883347458_3.jpg)
![](./images/1032903864883347458_4.jpg)

![](./images/1032903864883347458_5.jpg)
![](./images/1032903864883347458_6.jpg)

Figure 1: Simulation of gradient flow on a diagonal linear network. In Figure 1a the importance of initialization and scaling is highlighted. In Figure 1b the different regularizations are presented showing the benefit of PILoT. In Figure 1c, the evolution of the time-dependent Bregman potential is shown, where $\alpha = \int_0^t \alpha_s ds$ the exponent of $a_t$.

regime. Furthermore, the spred initalization leads to less performance upto extreme sparsities, this substantiates our theory. The two other parameterizations do not perform as well. The reason for this is the basic setup used for the experiment. Powerpropagation ([63]) on its own is not a well-posed parameterization as it confines parameters to the positive or negative reals based on their initialization. Therefore, it needs help with recovering weights that were turned off. In the case of the original continuous sparsification ([62]), $m$ is replaced by a sigmoid. The sigmoid is a rigid function (i.e. the gradient is small) preventing us from leveraging the full benefits of doubling the parameters.

![](./images/1032903864883347458_7.jpg)

Figure 2: One shot experiment on CIFAR10 and CIFAR 100. The first two figures show the results for CIFAR 10 with a ResNet20 and the last two for CIFAR 100 with a ResNet18.

**Iterative pruning combined with PILoT** While PILoT is designed as a continuous sparsification strategy that learns sparse models on its own, similar to competitors [62], it can also be combined with iterative pruning techniques such as IMP and LRR to consolidate some of the obtained sparsity during training. We compare our method PILot against the standard parameterization again on CIFAR10 and CIFAR100. Figure 3 demonstrates that $m \odot w$ gets higher accuracy in the high-sparsity regime than the baseline and, thus, finds overall sparser lottery tickets. Adding our regularization leads in both cases to higher accuracy at the end of the high-sparsity regime. In the case of CIFAR 100, PILoT works better than the original LRR throughout the high-sparsity regime. Remarkably, IMP with PILoT becomes even competitive with LRR and outperforms it for some sparsity levels. Furthermore, LRR with PILoT finds the sparsest lottery tickets. This is as expected as the implicit $L_1$-regularization keeps being applied to the same weights with each cycle. This implies that the whole training procedure is utilized to sparsify the network further. Note that the typical peak in accuracy at moderate sparsity levels does not occur. This is likely also caused by the implicit $L_1$-regularization, as the models are sparser than suggested by the remaining number of trainable parameters.

![](./images/1032903864883347458_8.jpg)

Figure 3: IMP and LRR on CIFAR100 and CIFAR 10. The first two figures show the results for CIFAR 10 with a ResNet20 and the last two for CIFAR 100 with a ResNet18.

## 5 Discussion

We have shed light on the inner workings of continuous sparsification, a state-of-the-art approach to prune neural networks that tries to solve an intractable optimization problem of mixed discrete and continuous nature. Yet, we find that its basic relaxed formulation, which doubles the number of trainable parameters to $m \odot w$, induces an implicit bias towards sparsity. To exploit this insight for neural network sparsification, we have proposed PILoT, a controllable regularization that acts like an implicit regularization in the original neural network parameter space and, remarkably, corresponds to a time-dependent Bregman potential. It therefore enjoys all the benefits of an implicit regularization that caters first to the loss and not a sparsity penalty. Furthermore, the time-dependent control enables the associated mirror flow to enter the rich regime and approximately impose an implicit $L_1$ regularization. This property is central to our proofs that show convergence of our approach for (quasi)-convex loss functions and optimality for underdetermined linear regression. Moreover, this shows the inner working of spred as well and highlights it pros (better than LASSO) and cons (initalization and constant regularization). Experiments on standard vision benchmarks further corroborate the utility of our theoretical insights and achieve improvements over baselines in the high-sparsity regime.

## References

[1] Aws Albarghouthi. Introduction to neural network verification, 2021.

[2] Felipe Alvarez, Jérôme Bolte, and Olivier Brahic. Hessian riemannian gradient flows in convex programming. *SIAM Journal on Control and Optimization*, 43(2):477–501, January 2004.

[3] Amir Beck and Marc Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. *Operations Research Letters*, 31(3):167–175, 2003.

[4] Stephen P. Boyd and Lieven Vandenberghe. Convex optimization. 2009.

[5] Rebekka Burkholz. Convolutional and residual networks provably contain lottery tickets. In *International Conference on Machine Learning*, 2022.

[6] Rebekka Burkholz. Most activation functions can win the lottery without excessive depth. In *Advances in Neural Information Processing Systems*, 2022.

[7] Rebekka Burkholz, Nilanjana Laha, Rajarshi Mukherjee, and Alkis Gotovos. On the existence of universal lottery tickets. In *International Conference on Learning Representations*, 2022.

[8] Tianlong Chen, Zhenyu Zhang, Jun Wu, Randy Huang, Sijia Liu, Shiyu Chang, and Zhangyang Wang. Can you win everything with a lottery ticket? *Transactions on Machine Learning Research*, 2022.

[9] Lénaïc Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In Jacob Abernethy and Shivani Agarwal, editors, *Proceedings of Thirty Third Conference on Learning Theory*, volume 125 of *Proceedings of Machine Learning Research*, pages 1305–1338. PMLR, 09–12 Jul 2020.

[10] Hung-Hsu Chou, Johannes Maly, and Dominik Stöger. How to induce regularization in linear models: A guide to reparametrizing gradient flow, 2024.

[11] Arthur da Cunha, Emanuele Natale, and Laurent Viennot. Proving the lottery ticket hypothesis for convolutional neural networks. In *International Conference on Learning Representations*, 2022.

[12] Steffen Dereich and Sebastian Kassing. Convergence of stochastic gradient descent schemes for lojasiewicz-landscapes, 2024.

[13] Tim Dettmers and Luke Zettlemoyer. Sparse networks from scratch: Faster training without losing performance. 2019.

[14] Simon S. Du, Chi Jin, Jason D. Lee, Michael I. Jordan, Barnabas Poczos, and Aarti Singh. Gradient descent can take exponential time to escape saddle points, 2017.

[15] Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, and Erich Elsen. Rigging the lottery: Making all tickets winners. In *International Conference on Machine Learning*, pages 2943–2952. PMLR, 2020.

[16] Mathieu Even, Scott Pesme, Suriya Gunasekar, and Nicolas Flammarion. (s)gd over diag- onal linear networks: Implicit regularisation, large stepsizes and edge of stability. *ArXiv*, abs/2302.08982, 2023.

[17] Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Francisco J. R. Ruiz, Julian Schrittwieser, Grzegorz Swirszcz, David Silver, Demis Hassabis, and Pushmeet Kohli. Discovering faster matrix multiplication algorithms with reinforcement learning. *Nature*, 610(7930):47–53, 2022.

[18] Benjamin Fehrman, Benjamin Gess, and Arnulf Jentzen. Convergence rates for the stochastic gradient descent method for non-convex objective functions, 2019.

[19] Damien Ferbach, Christos Tsirigotis, Gauthier Gidel, and Bose Avishek. A general framework for proving the equivariant strong lottery ticket hypothesis, 2022.

[20] Jonas Fischer and Rebekka Burkholz. Plant ’n’ seek: Can you find the winning ticket?, 2021.

[21] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv: Learning*, 2018.

[22] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2019.

[23] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Pruning neural networks at initialization: Why are we missing the mark? In *International Conference on Learning Representations*, 2021.

[24] Advait Gadhikar and Rebekka Burkholz. Masks, signs, and learning rate rewinding. In *Twelfth International Conference on Learning Representations*, 2024.

[25] Advait Harshal Gadhikar, Sohom Mukherjee, and Rebekka Burkholz. Why random pruning is all we need to start sparse. In *International Conference on Machine Learning*, 2023.

[26] Jose Gallego-Posada, Juan Ramirez, Akram Erraqabi, Yoshua Bengio, and Simon Lacoste-Julien. Controlled sparsity via constrained optimization or: How i learned to stop tuning penalties and love constraints. In *Thirty-Sixth Conference on Neural Information Processing Systems*, 2022.

[27] Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias in terms of optimization geometry, 2020.

[28] Suriya Gunasekar, Blake Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nathan Srebro. Implicit regularization in matrix factorization, 2017.

[29] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. *Advances in neural information processing systems*, 28, 2015.

[30] Intekhab Hossain, Jonas Fischer, Rebekka Burkholz, and John Quackenbush. Not all tickets are equal and we know it: Guiding pruning with domain-specific knowledge, 2024.

[31] Tian Jin, Michael Carbin, Daniel M. Roy, Jonathan Frankle, and Gintare Karolina Dziugaite. Pruning’s effect on generalization through the lens of training and regularization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, *Advances in Neural Information Processing Systems*, 2022.

[32] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ron- neberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873):583–589, 2021.


[33] Lynn H Kaack, Priya L Donti, Emma Strubell, George Kamiya, Felix Creutzig, and David Rolnick. Aligning artificial intelligence with climate change mitigation. *Nature Climate Change*, 12(6):518–527, 2022.

[34] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition. In Paolo Frasconi, Niels Landwehr, Giuseppe Manco, and Jilles Vreeken, editors, *Machine Learning and Knowledge Discovery in Databases*, pages 795–811, Cham, 2016. Springer International Publishing.

[35] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.

[36] Aditya Kusupati, Vivek Ramanujan, Raghav Somani, Mitchell Wortsman, Prateek Jain, Sham Kakade, and Ali Farhadi. Soft threshold weight reparameterization for learnable sparsity. In *Proceedings of the International Conference on Machine Learning*, July 2020.

[37] Andrey Kuzmin, Markus Nagel, Saurabh Pitre, Sandeep Pendyam, Tijmen Blankevoort, and Max Welling. Taxonomy and evaluation of structured compression of convolutional neural networks, 2019.

[38] Denis Kuznedelev, Eldar Kurtic, Eugenia Iofinova, Elias Frantar, Alexandra Peste, and Dan Alistarh. Accurate neural network pruning requires rethinking sparse optimization, 2023.

[39] Mike Lasby, Anna Golubeva, Utku Evci, Mihai Nica, and Yani Ioannou. Dynamic sparse training with structured sparsity. *arXiv preprint arXiv:2305.02299*, 2023.

[40] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Snip: single-shot network pruning based on connection sensitivity. In *International Conference on Learning Representations*, 2019.

[41] Zhiyuan Li, Tianhao Wang, Jason D. Lee, and Sanjeev Arora. Implicit bias of gradient descent on reparametrized models: On equivalence to mirror descent. *ArXiv*, abs/2207.04036, 2022.

[42] Bohan Liu, Zijie Zhang, Peixiong He, Zhensen Wang, Yang Xiao, Ruimeng Ye, Yang Zhou, Wei-Shinn Ku, and Bo Hui. A survey of lottery ticket hypothesis, 2024.

[43] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Li Shen, Decebal Constantin Mocanu, Zhangyang Wang, and Mykola Pechenizkiy. The unreasonable effectiveness of random pruning: Return of the most naive baseline for sparse training. In *International Conference on Learning Representations*, 2021.

[44] Shiwei Liu and Zhangyang Wang. Ten lessons we have learned in the new "sparseland": A short handbook for sparse neural network researchers, 2023.

[45] Christos Louizos, Max Welling, and Diederik P. Kingma. Learning sparse neural networks through $l_0$ regularization, 2018.

[46] Alexandra Sasha Luccioni, Yacine Jernite, and Emma Strubell. Power hungry processing: Watts driving the cost of ai deployment? *arXiv preprint arXiv:2311.16863*, 2023.

[47] Jaron Maene, Mingxiao Li, and Marie-Francine Moens. Towards understanding iterative magnitude pruning: Why lottery tickets win, 2021.

[48] Eran Malach, Gilad Yehudai, Shai Shalev-Schwartz, and Ohad Shamir. Proving the lottery ticket hypothesis: Pruning is all you need. In *International Conference on Machine Learning*, 2020.

[49] Nina Narodytska, Hongce Zhang, Aarti Gupta, and Toby Walsh. In search for a sat-friendly binarized neural network architecture. In *International Conference on Learning Representations*, 2020.

[50] A.S. Nemirovski and D.B. Yudin. *Problem Complexity and Method Efficiency in Optimization*. A Wiley-Interscience publication. Wiley, 1983.

[51] Laurent Orseau, Marcus Hutter, and Omar Rivasplata. Logarithmic pruning is all you need. *Advances in Neural Information Processing Systems*, 33, 2020.

[52] Shreyas Malakarjun Patil and Constantine Dovrolis. Phew: Constructing sparse networks that learn fast and generalize well without training data. In *International Conference on Machine Learning*, pages 8432–8442. PMLR, 2021.

[53] Mansheej Paul, Feng Chen, Brett W. Larsen, Jonathan Frankle, Surya Ganguli, and Gintare Karolina Dziugaite. Unmasking the lottery ticket hypothesis: What's encoded in a winning ticket's mask? In *The Eleventh International Conference on Learning Representations*, 2023.

[54] Ankit Pensia, Shashank Rajput, Alliot Nagle, Harit Vishwakarma, and Dimitris Papailiopoulos. Optimal lottery tickets via subset sum: Logarithmic over-parameterization is sufficient. In *Advances in Neural Information Processing Systems*, volume 33, pages 2599-2610, 2020.

[55] Scott Pesme, Loucas Pillaud-Vivien, and Nicolas Flammarion. Implicit bias of sgd for diagonal linear networks: a provable benefit of stochasticity, 2021.

[56] Alexandra Peste, Eugenia Iofinova, Adrian Vladu, and Dan Alistarh. Ac/dc: Alternating compressed/decompressed training of deep neural networks, 2021.

[57] Hoang Pham, Shiwei Liu, Lichuan Xiang, Dung D Le, Hongkai Wen, Long Tran-Thanh, et al. Towards data-agnostic pruning at initialization: What makes a good sparse mask? In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

[58] Adityanarayanan Radhakrishnan, Mikhail Belkin, and Caroline Uhler. Linear convergence of generalized mirror descent with time-dependent mirrors, 2021.

[59] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 2022.

[60] Alex Renda, Jonathan Frankle, and Michael Carbin. Comparing rewinding and fine-tuning in neural network pruning. In *International Conference on Learning Representations*, 2020.

[61] Tyrrel R Rockafellar and Werner Fenchel. *Convex Analysis*. 1970.

[62] Pedro Savarese, Hugo Silva, and Michael Maire. Winning the lottery with continuous sparsifi- cation, 2021.

[63] Jonathan Schwarz, Siddhant M. Jayakumar, Razvan Pascanu, Peter E. Latham, and Yee Whye Teh. Powerpropagation: A sparsity inducing weight reparameterisation, 2021.

[64] Heejune Sheen, Siyu Chen, Tianhao Wang, and Harrison H. Zhou. Implicit regularization of gradient flow on one-layer softmax attention, 2024.

[65] Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grinde, Alliot Nagle, Hongyi Wang, Eric Xing, Kangwook Lee, and Dimitris Papailiopoulos. Rare gems: Finding lottery tickets at initialization. *Advances in Neural Information Processing Systems*, 35:14529-14540, 2022.

[66] Natalie Stephenson, Emily Shane, Jessica Chase, Jason Rowland, David Ries, Nicola Justice, Jie Zhang, Leong Chan, and Renzhi Cao. Survey of machine learning techniques in drug discovery. *Current drug metabolism*, 20(3):185-193, 2019.

[67] Hidenori Tanaka, Daniel Kunin, Daniel L. Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. In *Advances in Neural Information Processing Systems*, 2020.

[68] Chaoqi Wang, Guodong Zhang, and Roger B. Grosse. Picking winning tickets before training by preserving gradient flow. In *International Conference on Learning Representations*, 2020.

[69] Kun Wang, Yuxuan Liang, Pengkun Wang, Xu Wang, Pengfei Gu, Junfeng Fang, and Yang Wang. Searching lottery tickets in graph neural networks: A dual perspective. In *The Eleventh International Conference on Learning Representations*, 2023.

[70] Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. In *Advances in Neural Information Processing Systems*, volume 29, 2016.

[71] Stephan Wojtowytsch. Stochastic gradient descent with noise of machine learning type. part i: Discrete time analysis, 2021.

[72] Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models, 2020.

[73] Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Gloria Chang, Fiona Aga, Jinshi Huang, Charles Bai, et al. Sustainable ai: Environmen- tal implications, challenges and opportunities. *Proceedings of Machine Learning and Systems*, 4:795-813, 2022.

[74] Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G. Baraniuk, Zhangyang Wang, and Yingyan Lin. Drawing early-bird tickets: Toward more efficient training of deep networks. In *International Conference on Learning Representations*, 2020.

[75] Xiao Zhou, Weizhong Zhang, Zonghao Chen, Shizhe Diao, and Tong Zhang. Efficient neural network training via forward and backward propagation sparsification. *Advances in Neural Information Processing Systems*, 34:15216–15229, 2021.

[76] Xiao Zhou, Weizhong Zhang, Hang Xu, and Tong Zhang. Effective sparsification of neural networks with global sparsity constraint. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3599–3608, 2021.

[77] Liu Ziyin. Symmetry induces structure and constraint of learning. 2023.

[78] Liu Ziyin and Zihao Wang. spred: Solving $l_1$ penalty with sgd, 2023.

## A Proof Main Result

We show the main result here. The proof consists of four parts
- $R_{a_t}$ satisfies a mirror flow (Lemmas A.1 and A.2)
- Boundedness of the iterates and convergence to a critical point (Lemma A.3)
- Convergence of the loss (Theorem A.1)
- Optimality in case of underdetermined linear regression (Theorem A.2)

Consider the following gradient flow

$$
\begin{cases}
dm_t = -\nabla f\left(m_t \odot w_t\right) \odot w_t - 2\alpha_t m_t dt \\
dw_t = -\nabla f\left(m_t \odot w_t\right) \odot m_t - 2\alpha_t w_t dt
\end{cases} \tag{7}
$$

For the flow in (7) to be well-posed $\nabla f$ needs to be locally Lipschitz continuous. This is a sufficient condition given that $\alpha_t$ is "nice", which will be made more rigorous later. The evolution of $x_t = m_t \odot w_t$ is derived in Lemma A.1.

### Lemma A.1
The evolution of $x_t = m_t \odot w_t$ with 7 is described by

$$
x_t = u_0^2 \odot \exp\left(-2 \int_0^t \nabla f\left(x_s\right) ds - 4 \int_0^t \alpha_s ds\right) - v_0^2 \odot \exp\left(2 \int_0^t \nabla f\left(x_s\right) ds - 4 \int_0^t \alpha_s ds\right),
$$

where $u_0 = \frac{m_0 + w_0}{\sqrt{2}}$ and $v_0 = \frac{m_0 - w_0}{\sqrt{2}}$.

Proof. This follows from deriving the flow of $m_t$ and $w_t$ and then combining the two. The evolution of both are given by

$$
\begin{cases}
m_t = \left(m_0 \odot \cosh\left(-\int_0^t \nabla f\left(x_s\right) ds\right) + w_0 \odot \sinh\left(-\int_0^t \nabla f\left(x_s\right) ds\right)\right) \exp\left(-2 \int_0^t \alpha_s ds\right) \\
w_t = \left(w_0 \odot \cosh\left(-\int_0^t \nabla f\left(x_s\right) ds\right) + m_0 \odot \sinh\left(-\int_0^t \nabla f\left(x_s\right) ds\right)\right) \exp\left(-2 \int_0^t \alpha_s ds\right).
\end{cases}
$$

For ease of notation set $L_t = \int_0^t \nabla f\left(x_s\right) ds$ and $A_t = \int_0^t \alpha_s ds$. Combining gives us

$$
\begin{aligned}
x_t &= m_t \odot w_t \\
&= \left(m_0^2 + w_0^2\right) \odot \cosh\left(-L_t\right) \odot \sinh\left(-L_t\right) \exp\left(-4 A_t\right) \\
&+ w_0 \odot m_0 \odot \left(\cosh\left(-L_t\right)^2 + \cosh\left(-L_t\right)^2\right) \exp\left(-4 A_t\right) \\
&= \left(\frac{\left(m_0^2 + w_0^2\right)}{2} \odot \sinh\left(-2 L_t\right)\right) \exp\left(-4 A_t\right) \\
&+ w_0 \odot m_0 \odot \left(\cosh\left(-2 L_t\right)\right) \exp\left(-4 A_t\right) \\
&= u_0^2 \odot \exp\left(-2 L_t - 4 A_t\right) - v_0^2 \odot \exp\left(2 L_t - 4 A_t\right)
\end{aligned}
$$

where the second equality follows from hyperbolic identities. $\square$

It follows from Lemma A.1 and the local Lipschitz condition on $\nabla f$ and $\int_{0}^{t} \alpha_{s} d s<\infty$ for all $t \geq 0$, that the flow is well-posed. We now define the (corrected)-hyperbolic entropy function. The corrected hyperbolic entropy is given by
$$
R_{a}(x)=\frac{1}{2} \sum_{i=1}^{n} x_{i} \operatorname{arcsinh}\left(\frac{x_{i}}{a}\right)-\sqrt{x_{i}^{2}+a^{2}}-x_{i} \log \frac{u_{0 i}}{v_{0, i}},
$$
where the last term is the correction. The correction stems from not initializing at zero.

**Lemma A.2** Let $|w_{i 0}| \leq m_{0 i}$ for all $i \in[n]$, then $R_{a_{t}}(x_{t})$ with $a_{t}=2 u_{0} \odot v_{0} \exp \left(-2 \int_{0}^{t} \alpha_{s} d s\right)$ satisfies
$$
d \nabla R_{a_{t}}\left(x_{t}\right)=-\nabla f\left(x_{t}\right) d t \quad x_{0}=m_{0} \odot w_{0}. \tag{8}
$$

Proof. This follows from Lemma A.1,
$$
\begin{aligned}
x_{t} \exp \left(4 \int_{0}^{t} \alpha_{s} d s\right)=u_{0}^{2} \exp \left(-2 \int_{0}^{t} \nabla f\left(x_{s}\right) d s\right)-v_{0}^{2} \exp \left(2 \int_{0}^{t} \nabla f\left(x_{s}\right) d s\right) & \Leftrightarrow \\
\frac{1}{2}\left(\operatorname{arcsinh}\left(\frac{x_{t}}{a_{t}}\right)-\log \left(\frac{u_{0}}{v_{0}}\right)\right) & =-\int_{0}^{t} \nabla f\left(x_{s}\right) d s.
\end{aligned}
$$

This equivalence follows from setting $z=\exp \left(-2 \int_{0}^{t} \nabla f\left(x_{s}\right) d s\right)$ and solving the resulting quadratic equation. Notice that the left hand side in (8) is $\nabla R_{a_{t}}(x_{t})$. $\square$

**Lemma A.3** Let $f$ be a quasi-convex function and $\alpha_{t} \geq 0$ for all $t \geq 0$. Furthermore assume the integral $\int_{0}^{t} \alpha_{s} d s<\infty$. Then the iterates are bounded and converge to a critical point.

Consider the time-dependent Bregman divergence
$$
D_{a_{t}}\left(x^{*}, x_{t}\right):=R_{a_{t}}\left(x^{*}\right)-R_{a_{t}}\left(x_{t}\right)-\nabla_{x} R_{a_{t}}^{T}\left(x^{*}-x_{t}\right) \geq 0
$$

The divergence is bounded by:
$$
D_{a_{t}}\left(x^{*}, x_{t}\right) \leq R_{a_{\infty}}\left(x^{*}\right)-R_{a_{t}}\left(x_{t}\right)-\nabla_{x} R_{a_{t}}^{T}\left(x^{*}-x_{t}\right)=: W_{t},
$$
this follows from the fact that the map $a \to R_{a}$ is decreasing. We make the following two observations:
$$
\frac{d}{d a} R_{a}(x)=-\frac{1}{2} \sum_{i=1}^{n} \frac{x_{i}^{2}|a|+a^{3}}{a^{2} \sqrt{a^{2}+x_{i}^{2}}} \leq 0 \quad \text { and } \quad \frac{d a_{t}}{d t} \leq 0 \quad \forall t \geq 0. \tag{9}
$$

This allows us to bound the evolution
$$
\begin{aligned}
\frac{d}{d t} W_{t} & =\frac{d}{d t}\left(-R_{a_{t}}\left(x_{t}\right)-\nabla_{x} R_{a_{t}}^{T}\left(x^{*}-x_{t}\right)\right) \\
& =-\frac{d}{d a} R_{a_{t}}\left(x_{t}\right) \frac{d}{d t} a_{t}-\frac{d}{d t}\left(\nabla_{x} R_{a_{t}}^{T}\right)\left(x^{*}-x_{t}\right) \\
& \leq \nabla_{x} f\left(x_{t}\right)^{T}\left(x^{*}-x_{t}\right) \\
& \leq 0,
\end{aligned}
$$
where the observations in (9) are used in the first inequality.

From Theorem 4.16 in [41] it follows that for all $a>0, R_{a}$ is a Bregman potential. Implying that for all $a>0$ the level set for $\gamma \in \mathbb{R}$,
$$
\left\{x \in \mathbb{R}^{n}: D_{a}\left(x^{*}, x\right) \leq \gamma\right\}
$$
is bounded. Combining this with the fact that the evolution is bounded, implies the iterates are bounded.


In the next part, it is shown that $x_t$ converges to a critical point. We first show that the loss becomes eventually non-increasing. There is a $T$ such that for all $t \geq T$ the loss $f$ is non-increasing.,

$$
\begin{aligned}
d f(x_t) &= -\left(\nabla f(x_t)^T \mathrm{diag}\left(\sqrt{x_t^2+a_t^2}\right) \nabla f(x_t)+2 \alpha_t \nabla f(x_t)^T x_t\right) d t \\
& \leq\left(-\nabla f(x_t)^T \mathrm{diag}\left(\sqrt{x_t^2+a_t^2}\right) \nabla f(x_t)+2 \alpha_t C\right) d t,
\end{aligned}
$$

where it is used that the iterates are bounded and $\nabla f$ is locally Lipschitz. As $t \to \infty$ we have that $\alpha_t \to 0$ and $a_t \to a_\infty>0$ by assumption. Hence there exists a $T$ such that for all $t \geq T$ we have
$$
d f(x_t) \leq 0.
$$

Note if this is not the case then there is a $T>0$ such that $\nabla f(x_T)=0$ implying convergence (to a critical point).

Now let $x_\infty$ be an accumulation point of the bounded flow $x_t$. We use this to show convergence to a critical point. We have that for all $x \in \mathbb{R}^n$,

$$
\nabla f(x_\infty)^\top x=\lim _{t \to \infty} \frac{1}{t}\left(\int_T^{T+t} \nabla f(x_s) d s\right)^\top x=\lim _{t \to \infty} \frac{1}{t}\left(R_{a_T}(x_T)-R_{a_{T+t}}(x_{T+t})\right)^\top x=0,
$$

where the first equality follows from that the loss is non-increasing and the second one from the time-dependent mirror flow description. Finally, because $x_t$ converges to an accumulation point we also have $\lim _{t \to \infty} R_{a_t}(x_t)=R_{a_\infty}(x_\infty)$ by continuity, giving the last equality.

We use that the accumulation point is a critical point and set $x^*=x_\infty$ in $W_t$ such that $W_t \to 0$. This implies $D_{a_t}(x_\infty, x_t) \to 0$ by the upperbound. It follows from the fact that the iterates are bounded that $R_{a_t}$ is $\mu$-strongly convex on this bounded convex set where the iterates stay. This gives

$$
\left\|x_\infty-x_t\right\|_{L_2} \leq \frac{\mu}{2} D_{a_t}(x_\infty, x_t) \to 0,
$$

showing $x_t$ converges to a critical point. $\square$

Lemma A.3 gives a condition such that the iterates are bounded and converge to a critical point. It remains to be shown that the loss converges. This is done in Theorem A.1.

Theorem A.1 Consider the same setting as Lemma A.3, if $f$ is convex or satisfies the PL-inequality we have convergence to an interpolator $x^*$ such that it is a minimizer of $f$. Furthermore, in the PL-inequality case, the loss converges linearly.

Proof. Assume $f$ is convex, notice first that there is a $T$ such that for all $t \geq T$ the loss is non-increasing. Combining this with a bound on the time-dependent Bregman potential gives us convergence of the loss. The time-dependent Bregman divergence is again defined by

$$
D_{a_t}\left(x^*, x_t\right)=R_{a_t}\left(x^*\right)-R_{a_t}\left(x_t\right)-\nabla_x R_{a_t}^\top\left(x^*-x_t\right) \geq 0.
$$

The divergence is bounded by:

$$
D_{a_t}\left(x^*, x_t\right) \leq W_t.
$$

The evolution of the bound is

$$
\begin{aligned}
\frac{d}{d t} W_t &=\frac{d}{d t}\left(-R_{a_t}\left(x_t\right)-\nabla_x R_{a_t}^\top\left(x^*-x_t\right)\right) \\
&=-\frac{d}{d a} R_{a_t}\left(x_t\right) \frac{d}{d t} a_t-\frac{d}{d t}\left(\nabla_x R_{a_t}^\top\right)\left(x^*-x_t\right) \\
& \leq \nabla_x f(x_t)^\top\left(x^*-x_t\right) \\
& \leq f\left(x^*\right)-f\left(x_t\right),
\end{aligned}
$$

where again the observations in (9) are used in the first inequality. Therefore the loss converges:

$$
\begin{aligned}
f\left(x_{T+t}\right)-f\left(x^*\right) & \leq \frac{1}{t} \int_T^{T+t} f\left(x_s\right)-f\left(x^*\right) d s \\
& \leq \frac{W_T-W_{T+t}}{t} \\
& \leq \frac{W_T}{t} \to 0
\end{aligned}
$$

where the first inequality follows from convexity of the loss and the third inequality from the fact that $W_t \geq D_{a_t}(x^*, x_t) \geq 0$. So the loss converges. We already know from Lemma A.3 that the iterates converge, concluding the convex case.

In case when $f$ satisfies the PL-inequality, we proceed in the same way as Theorem 2.3. The evolution of $f$ is given by

$$
\begin{aligned}
d f\left(x_{t}\right) & =-\left(\nabla f\left(x_{t}\right)^{\top} \operatorname{diag}\left(\sqrt{x_{t}^{2}+a_{t}^{2}}\right) \nabla f\left(x_{t}\right)+2 \alpha_{t} \nabla f\left(x_{t}\right)^{\top} x_{t}\right) d t \\
& \leq\left(-a_{\infty} \lambda\left(f\left(x_{t}\right)-f\left(x^{*}\right)\right)+\alpha_{t} C|| x^{*}||_{L_{2}}\right) d t
\end{aligned}
$$

where $C$ is constant depending on the smoothness of $f$. Then it follows from Gronwall's Lemma that

$$
f\left(x_{t}\right)-f\left(x^{*}\right) \leq\left(f\left(x_{0}\right)-f\left(x^{*}\right)\right) \exp \left(-a_{\infty} \lambda t+\int_{0}^{t} \alpha_{s} C|| x^{*}||_{L_{2}} d s\right).
$$

It follows from the fact that $\int_{0}^{t} \alpha_{s} d s<\infty$ for all $t \geq 0$ that the loss $f$ convergence. Convergence of the iterates now follows in a similar way as the convex case. $\square$

We now show optimality in the case of underdetermined linear regression Consider a data set $(z_{j}, y_{j})_{i=1}^{d}$ with $z_{j} \in \mathbb{R}^{n}$ and $y_{j} \in \mathbb{R}$. Let $Z=(z_{1}, \ldots z_{d})$ and $Y=(y_{1}, \ldots y_{d})$. For the regression to be called underdetermined $n>d$.

Theorem A.2 In case of under-determined regression consider the loss function $f(x)=\tilde{f}(Z x-Y)$. Assume $f$ satisfies the conditions with at least one of the convergence criteria of Theorem A.1. Then $x_{t}$ converges to $x^{*}$ such that

$$
x^{*}=\operatorname{argmin}_{Z x=Y} R_{a_{\infty}}(x) \tag{10}
$$

Proof. Convergence follows from Theorem A.1. It remains to be shown that the optimality conditions of (10) are satisfied. The gradient flow of $R_{a_{t}}$ satisfies

$$
\nabla R_{a_{t}}\left(x_{t}\right)=Z^{\top} \int_{0}^{t} \nabla \tilde{f}\left(x_{s}\right) d s \in \operatorname{span}\left\{Z^{\top}\right\}.
$$

This quantity is well defined for all $t \geq 0$ because $\nabla \tilde{f}$ is locally Lipschitz (because $f$ has to be locally Lipschitz). Therefore taking the $t \to \infty$ yields the KKT conditions of the optimization problem in (10). $\square$

### A.1 Discussion of the proof
Most of the proof follows the same arguments as in [2, 41, 55]. The notable differences are showing that the loss becomes decreasing over time and the observations made in (9).

## B Details experiments
In this section, we provide the details of the experiments. In addition, there are additional figures given.

Compute The codebase for the experiments is written in PyTorch and torchvision and their relevant primitives for model construction and data-related operations. The experiments in the paper are trained on an NVIDIA A6000. In addition, the diagonal linear network is trained on a CPU 13th Gen INTEL(R) Core(TM) i9-13900H.

### B.1 Diagonal linear network
For each setting, different regularization schemes are tried. In total, 7 options are tried. 4 of the schedules are constant i.e. the regularization stays the same during training. The remaining 3 are decaying schedules. These schedules we name harmonic, quadratic, and geometric. The schedules are described by the following recurrent relations:

$$
h_{k}=\frac{1}{k} \quad, \quad q_{k}=\frac{1}{k^{2}} \quad \text { and } \quad g_{k}=p^{k}
$$

where we have set $p=0.95$. These schedules lead to a total strength of regularization applied. We denote $A:=\int_0^t \alpha_s ds$ the total strength of the regularization. So in practice, it is the weighted sum of the regularization strength.

In Figure 4 we present the trajectories of all the schedules for the 3 considered settings. We observe that our regularization performs the best with the decaying schedules as predicted by the theory. The other methods need constant regularization to perform well as already mentioned in Remark 2.2. Note that, our regularization also can perform well with constant regularization (see Figure 4. Next to gradient flow, stochastic gradient flow is also simulated. For each initialization, we run 5 seeds. In Figure 5 the simulation of this is given for the best-performing schemes. The results are similar to the gradient flow results.

![](./images/1032903864883347458_9.jpg)

Figure 4: All runs for the diagonal linear network. From left to right $m \odot w$ with our regularization, $m \odot w$ with regularization on $m$, $x$ with $L_1$ regularization

![](./images/1032903864883347458_10.jpg)

Figure 5: SGD with batchsize 5 for the best GD runs. From left to right $m \odot w$ with our regularization, $m \odot w$ with regularization on $m$, $x$ with $L_1$ regularization

### B.2 One-shot

In this section, the details of the one-shot experiment are presented. In Table 1 the details of the experiment are given. After training the the smallest $(1-0.8^k)\cdot 100\%$ with $k=1\dots 25$ are removed and the validation accuracy is calculated.

We have implemented Powerpropagation with $\alpha=1$, i.e each weight is replaced by $x=\text{sign}(x)|x|$ [63]. Continuous sparsification is implemented by replacing $x=\text{sigmoid}(m)x$ and putting an $L_1$

regularization on $m$ with an increasing schedule [62]. Note, for CIFAR 100 only the best performing on CIFAR 10 have been implemented. Furthermore, all runs have been done for 3 different seeds.

Table 1: One-shot experiment
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Setting</th>
<th>Comments</th>
</tr>
</thead>
<tbody>
<tr>
<td>Optimizer</td>
<td>SGD</td>
<td></td>
</tr>
<tr>
<td>Momentum</td>
<td>0.9</td>
<td></td>
</tr>
<tr>
<td>Batch size</td>
<td>256</td>
<td></td>
</tr>
<tr>
<td>Activation function</td>
<td>ReLu</td>
<td></td>
</tr>
<tr>
<td>Weight decay</td>
<td>$10^{-4}$</td>
<td></td>
</tr>
<tr>
<td>Base learning rate</td>
<td>$\{0.1, 0.2\}$</td>
<td></td>
</tr>
<tr>
<td>Epochs</td>
<td>150</td>
<td></td>
</tr>
<tr>
<td>Warmup period</td>
<td>0</td>
<td></td>
</tr>
<tr>
<td>Initialization</td>
<td>Kaiming normal</td>
<td></td>
</tr>
<tr>
<td>PILOT regularization</td>
<td>$\{\text{None}, 10^{-4}, 3 \cdot 10^{-4}, 10^{-3}\}$</td>
<td>Only for $m \odot w$</td>
</tr>
<tr>
<td>$L_1$ regularization</td>
<td>$\{\text{None}, 10^{-4}, 10^{-3}\}$</td>
<td>On $m$ and $x$</td>
</tr>
<tr>
<td>Scaling</td>
<td>1</td>
<td>Only for $m \odot w$</td>
</tr>
<tr>
<td colspan="3">CIFAR 10</td>
</tr>
<tr>
<td>Learning rate schedule</td>
<td>cosine warmup</td>
<td></td>
</tr>
<tr>
<td colspan="3">CIFAR 100</td>
</tr>
<tr>
<td>Learning rate schedule</td>
<td>step warmup</td>
<td></td>
</tr>
</tbody>
</table>

### B.3 IMP and LRR

In Table 2 the details of the experiment are given. Note the base learning rate 0.1 is for the baseline and 0.2 is used for our parameterization combined with scaling 1. Furthermore, all runs have been done for 3 different seeds.

Table 2: IMP and LRR experiment
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Setting</th>
<th>Comments</th>
</tr>
</thead>
<tbody>
<tr>
<td>Optimizer</td>
<td>SGD</td>
<td></td>
</tr>
<tr>
<td>Momentum</td>
<td>0.9</td>
<td></td>
</tr>
<tr>
<td>Batch size</td>
<td>256</td>
<td></td>
</tr>
<tr>
<td>Activation function</td>
<td>ReLu</td>
<td></td>
</tr>
<tr>
<td>Weight decay</td>
<td>$10^{-4}$</td>
<td></td>
</tr>
<tr>
<td>Learning rate schedule</td>
<td>step warmup</td>
<td></td>
</tr>
<tr>
<td>Base learning rate</td>
<td>$\{0.1, 0.2\}$</td>
<td></td>
</tr>
<tr>
<td>Cycles</td>
<td>25</td>
<td></td>
</tr>
<tr>
<td>Pruning rate</td>
<td>0.8</td>
<td></td>
</tr>
<tr>
<td>Epochs per cycle</td>
<td>150</td>
<td></td>
</tr>
<tr>
<td>Warmup period</td>
<td>50</td>
<td></td>
</tr>
<tr>
<td>Initialization</td>
<td>Kaiming normal</td>
<td></td>
</tr>
<tr>
<td>PILOT regularization</td>
<td>$\{\text{None}, 10^{-4}\}$</td>
<td>Only for $m \odot w$</td>
</tr>
<tr>
<td>Scaling</td>
<td>1</td>
<td>Only for $m \odot w$</td>
</tr>
</tbody>
</table>

## C Remark on Neuronwise Pruning

In the main text, we have used mirror flow to describe the implicit bias of parameterwise pruning. In this section, we show that neuronwise pruning can not be analyzed in the same way. We first define neuronwise pruning. Next, we paraphrase the necessary condition such that the implicit bias can be

described by a mirror flow from [41]. Finally, we show neuronwise pruning violates this condition pointing out a limitation of the framework.

Consider the parameterization for a function with $p$ neurons, $g : \mathbb{R}^p \times \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_p}$,
$$g(m, w_1, \dots, w_p) = (m_1 w_1, \dots, m_p w_p)$$
where $m_i \in \mathbb{R}$ is the mask and $w_i \in \mathbb{R}^{n_i}$ are the neurons.

We state the necessary condition for a parameterization to induce a mirror flow.

**Theorem C.1** (Theorem 4.10 [41]) The Lie bracket span of $\{\nabla_i g\}_{i=1}^n$ is in the kernel of Jacobian $\partial g$.

Now, we use this theorem to show that neuronwise pruning does not induce a mirror flow.

**Lemma C.1** Neuronwise pruning violates Theorem C.1.

Proof. We show that for $p=1$ the condition is already violated. This implies that in the general case, the condition is also violated as the neurons themselves are commuting with each other as they are parameterized separate.

To see this we can explicitly check the commuting condition for the following parameterization
$g : \mathbb{R} \times \mathbb{R}^n \to \mathbb{R}^n$
$$g(m, w) = m w$$

Then the gradients (Jacobians) and Hessian's are given by:
$$
\nabla g_i = \begin{pmatrix}
w_i \
m \mathbb{I}_{i=1} \
\vdots \
m \mathbb{I}_{i=n_1}
\end{pmatrix}
\quad \text{and} \quad
H g_i = \begin{pmatrix}
0 & \mathbb{I}_{i=1} & \dots & \mathbb{I}_{i=n} \
\mathbb{I}_{i=1} & 0 & \dots & 0 \
\vdots & \vdots & \dots & \vdots \
\mathbb{I}_{i=n} & 0 & \dots & 0
\end{pmatrix}
\quad \text{for } i=1,2
$$

Computing $H g_i \nabla g_j$
$$
H g_i \nabla g_j = w_j \begin{pmatrix}
0 \
\mathbb{I}_{i=1} \
\vdots \
\mathbb{I}_{i=n}
\end{pmatrix}
$$
we compute the Lie brackets which span a subspace of the Lie Algebra $LIE^{\geq 2}(\partial g)$. The subspace is spanned by
$$
\operatorname{span} \left( w_j \begin{pmatrix}
0 \
\mathbb{I}_{i=1} \
\vdots \
\mathbb{I}_{i=n}
\end{pmatrix} - w_i \begin{pmatrix}
0 \
\mathbb{I}_{j=1} \
\vdots \
\mathbb{I}_{j=n}
\end{pmatrix} \text{ for } i,j \in [n] \right) \subset LIE^{\geq 2}(\partial g).
$$

Clearly, this span is not in $Ker(\partial g)$ as can be shown by a direct computation:
$$
(\partial g) \left( w_j \begin{pmatrix}
0 \
\mathbb{I}_{i=1} \
\vdots \
\mathbb{I}_{i=n}
\end{pmatrix} - w_i \begin{pmatrix}
0 \
\mathbb{I}_{j=1} \
\vdots \
\mathbb{I}_{j=n}
\end{pmatrix} \right) = (\mathbb{I}_{i=1} m w_j - \mathbb{I}_{j=1} m w_i, \dots, \mathbb{I}_{i=n} m w_i - \mathbb{I}_{j=1} m w_i)
$$

For the span to be in the kernel we need either that all $w_i = 0$ for all $i \in [n]$ or $m=0$. In both cases this implies that $g(m, w) \in \{0\}$. This implies that a mirror flow is only well-defined if the parameterization is zero. Hence, neuron-wise continuous sparsification does not induce a mirror flow.
$\square$