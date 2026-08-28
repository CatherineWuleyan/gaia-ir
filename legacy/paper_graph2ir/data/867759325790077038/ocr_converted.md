# Scaling down Deep Learning

［#1］
Sam Greydanus¹

## Abstract

［#2］
Though deep learning models have taken on commercial and political relevance, many aspects of their training and operation remain poorly understood. This has sparked interest in "science of deep learning" projects, many of which are run at scale and require enormous amounts of time, money, and electricity. But how much of this research really needs to occur at scale? In this paper, we introduce MNIST-1D: a minimalist, low-memory, and low-compute alternative to classic deep learning benchmarks¹. The training examples are 20 times smaller than MNIST examples yet they differentiate more clearly between linear, nonlinear, and convolutional models which attain 32, 68, and 94% accuracy respectively (these models obtain 94, 99+, and 99+% on MNIST). Then we present example use cases which include measuring the spatial inductive biases of lottery tickets, observing deep double descent, and metalearning an activation function.

## 1. Introduction

［#3］
By any scientific standard, the Human Genome Project was enormous: it involved billions of dollars of funding, dozens of institutions, and over a decade of accelerated research (Lander et al., 2001). But that was only the tip of the iceberg. Long before the project began, scientists were hard at work assembling the intricate science of human genetics. And most of the time, they were not studying humans. The foundational discoveries in genetics centered on far simpler organisms such as peas, molds, fruit flies, and mice. To this day, biologists use these simpler organisms as genetic "minimal working examples" in order to save time, energy, and money. A well-designed experiment with Drosophilia, such as Feany and Bender (2000), can teach us an astonishing amount about humans.

［#4］
![](./images/867759325790077038_1.jpg)

［#5］
Figure 1. Constructing the MNIST-1D dataset. Like MNIST, the classifier's objective is to determine which digit is present in the input. Unlike MNIST, each example is a one-dimensional sequence of points. To generate an example, we begin with a digit template and then randomly pad, translate, and transform it to produce sequences like the ones shown.

［#6］
The deep learning analogue of Drosophilia is the MNIST dataset. A large number of deep learning innovations including dropout, Adam, convolutional networks, generative adversarial networks, and variational autoencoders began life as MNIST experiments (Srivastava et al., 2014; Kingma and Ba, 2014; LeCun et al., 1989; Goodfellow et al., 2014; Kingma and Welling, 2013). Once these innovations proved themselves on small-scale experiments, scientists found ways to scale them to larger and more impactful applications.

［#7］
They key advantage of Drosophilia and MNIST is that they dramatically accelerate the iteration cycle of exploratory research. In the case of Drosophilia, the fly's life cycle is just a few days long and its nutritional needs are negligible. This makes it much easier to work with than mammals, especially humans. In the case of MNIST, training a strong classifier takes a few dozen lines of code, less than a minute of walltime, and negligible amounts of electricity. This is a stark contrast to state-of-the-art vision, text, and game-playing models which can take months and hundreds of thousands of dollars of electricity to train (Sharir et al., 2020).

［#8］
Yet in spite of its historical significance, MNIST has three notable shortcomings. First, it does a poor job of differentiating between linear, nonlinear, and translation-invariant

---
［#2］
¹Oregon State University; the ML Collective. Correspondence to: Sam Greydanus <greydanus.17@gmail.com>.

［#2］
¹Code at github.com/greydanus/mnist1d

［#9］
Scaling down Deep Learning

［#9］
models. For example, logistic, MLP, and CNN benchmarks obtain 94, 99+, and 99+% accuracy on it. This makes it hard to measure the contribution of a CNN's spatial priors or to judge the relative effectiveness of different regularization schemes. Second, it is somewhat large for a toy dataset. Each input example is a 784-dimensional vector and thus it takes a non-trivial amount of computation to perform hyperparameter searches or debug a metalearning loop. Third, MNIST is hard to hack. The ideal toy dataset should be procedurally generated so that researchers can smoothly vary parameters such as background noise, translation, and resolution.

［#10］
In order to address these shortcomings, we propose the MNIST-1D dataset. It is a minimalist, low-memory, and low-compute alternative to MNIST, designed for exploratory deep learning research where rapid iteration is a priority. Training examples are 20 times smaller but they are still better at measuring the difference between 1) linear and nonlinear classifiers and 2) models with and without spatial inductive biases (eg. translation invariance). The dataset is procedurally generated but still permits analogies to real-world digit classification.

［#11］
![](./images/867759325790077038_2.jpg)

［#12］
Figure 2. Visualizing the performance of common models on the MNIST-1D dataset. Whereas most ML models perform within a few percent accuracy on MNIST, this dataset separates them cleanly according to their characteristics. Logistic regression models fare worse than MLPs because they cannot use nonlinearities. MLPs, meanwhile, fare worse than CNNs because they cannot use translation invariance and local connectivity to bias optimization towards solutions that generalize well. These results suggest MNIST-1D is a good dataset for studying the inductive biases of ML models.

## 2. Context

［#13］
The machine learning community has grown rapidly in recent years. This growth has accelerated the rate of scientific innovation, but it has also produced multiple competing narratives about the field's ultimate direction and objectives. In this section, we will explore three such narratives in order to place MNIST-1D in its proper context.

［#14］
Scaling trends. One of the defining features of machine learning in the 2010's was a massive increase in the scale of datasets, models, and compute infrastructure (Amodei and Hernandez, 2018). This scaling pattern allowed neural networks to achieve breakthrough results on a wide range of benchmarks (Krizhevsky et al., 2012; Szegedy et al., 2015; Radford et al., 2019). Yet while this scaling effect has helped neural networks take on commercial and political relevance, opinions differ about how much more intelligence it can generate. One one hand, many researchers and organizations argue that scaling is a crucial path to making neural networks behave more intelligently (Amodei and Hernandez, 2018). On the other hand, there is a healthy but marginal population of researchers who are not primarily motivated by scale. They are united by a common desire to change research methodologies, advocating a shift away from human-engineered datasets and architectures (Clune, 2019), an emphasis on human-like learning patterns (Chollet, 2019), and better integration with traditional symbolic AI approaches (Marcus, 2018).

［#15］
Once again, the genetics analogy is useful. In genetics, scale has been most effective when small-scale experiments have helped to guide the direction and vision of large-scale experiments. For example, the organizers of the Human Genome Project regularly used yeast and fly genomes to guide analysis of the human genome (Lander et al., 2001). Thus one should be suspicious of research agendas that place disproportionate emphasis on large-scale experiments, since a healthy research ecosystem needs both. The fast, small scale projects permit creativity and deep understanding, whereas the large-scale projects expose fertile new research territory.

［#16］
Understanding vs. performance. Researchers are also divided over the value of understanding versus performance. Some contend that a high-performing algorithm need not be interpretable so long as it saves lives or produces economic value. Others argue that hard-to-interpret deep learning models should not be deployed in sensitive real-world contexts. Both arguments have merit, but the best path forward seems to be to focus on understanding high-performing algorithms better so that this tradeoff becomes less severe. One way to do this is by identifying things we don't understand about neural networks, reproducing these things on a toy problem like MNIST-1D, and then performing ablation studies to isolate the causal mechanisms.

［#17］
Ecological impacts. A growing number of researchers and organizations claim that deep learning will have positive environmental applications (Loehle, 1987; Rolnick et al., 2019). This may be true in the long run, but so far artificial intelligence has done little to solve environmental problems. In the meantime, deep learning models are consuming mas-

# Scaling down Deep Learning

［#18］
Table 1. Test accuracies of common classifiers on the MNIST and MNIST-1D datasets. Most classifiers achieve similar test accuracy on MNIST. The MNIST-1D dataset is much smaller than MNIST and does a better job of separating models with different inductive biases. The drop in CNN and GRU performance when using shuffled data indicates that spatial priors are important on this dataset.

［#19］
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Logistic regression</th>
      <th>Fully connected model</th>
      <th>Convolutional model</th>
      <th>GRU model</th>
      <th>Human expert</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MNIST</td>
      <td>94 ± 0.5</td>
      <td>&gt; 99</td>
      <td>&gt; 99</td>
      <td>&gt; 99</td>
      <td>&gt; 99</td>
    </tr>
    <tr>
      <td>MNIST-1D</td>
      <td>32 ± 1</td>
      <td>68 ± 2</td>
      <td>94 ± 2</td>
      <td>91 ± 2</td>
      <td>96 ± 1</td>
    </tr>
    <tr>
      <td>MNIST-1D (shuffled)</td>
      <td>32 ± 1</td>
      <td>68 ± 2</td>
      <td>56 ± 2</td>
      <td>57 ± 2</td>
      <td>≈ 30 ± 10</td>
    </tr>
  </tbody>
</table>

［#20］
sive amounts of electricity to train and deploy (Strubell et al., 2019). Our hope is that benchmarks like MNIST-1D will encourage researchers to spend more time iterating on small datasets and toy models before scaling, making more efficient use of electricity in the process.

## 3. Methods

［#21］
Given modern machine learning's emphasis on scale and performance, we see a hidden demand for alternative projects – projects that prioritize creativity over scale and understanding over performance. We designed MNIST-1D for that sort of research. In particular, we wanted a dataset that was

［#22］
- Extremely small: smaller than MNIST
- Able to identify models with (spatial) inductive biases
- Easy to hack, extend, or modify
- Analogous to large-scale problems

［#23］
Dimensionality. Our first choice was to make the data one-dimensional (eg time series) rather than two-dimensional (eg images). Our rationale was that there were already many good image datasets so the value in adding another was small. Meanwhile, one-dimensional signal processing requires less computation but has many of the same scientific properties.

［#24］
Constructing the dataset. We began with ten one-dimensional template patterns which resemble the digits 0-9 when plotted as in Figure 1. Each of these platonic forms consists of 12 hand-selected $x$ coordinates. Next, we padded each sequence with 36-60 additional points, translated the digit at random, scaled it, added Gaussian noise, and added a constant linear signal analogous to shear in a 2D image. We used a Gaussian filter with $\sigma=2$ to induce correlations that would be easy to confuse with the templates. Last of all, we downsampled the pattern to 40 data points. Figure 1 shows class-wise examples before and after these transformations.

［#25］
Implementation. Our goals during implementation were to make the code as simple, modular, and hackable as possible. The code for generating the dataset occupies two Python files and a total of 150 lines. The get_dataset method has a simple API for changing dataset features such as maximum digit translation, correlated noise scale, shear scale, final sequence length, and more. The default train/test split is 4000/1000.

［#26］
Benchmarking the dataset. We used PyTorch to implement and train simple logistic, MLP, CNN, and GRU baselines. All models used the Adam optimizer and early stopping for model selection. We also trained the same models on a version of the dataset which was permuted along the spatial dimension. We refer to this as the "shuffled" version since it measures each model's performance in the absence of local spatial structure. A number of related works use the same shuffling process to remove spatial priors (Zhang et al., 2016; Li et al., 2018). Table 1 shows that the test accuracy of CNNs and GRUs decreases by about 38% on the shuffled data whereas the MLP and linear models perform about the same. This is a good sanity check since the former two models have spatial and temporal locality priors whereas the latter two do not. As with the dataset itself, the code for implementing and training the benchmark models occupies roughly 150 lines and is clean and modular. You can reproduce the benchmarks in your browser in a few minutes².

## 4. Example use cases

［#27］
In this section we will explore several examples of how MNIST-1D can be used to study core "science of deep learning" phenomena.

［#28］
Finding lottery tickets. It is not unusual for deep learning models to have ten or even a hundred times more parameters than necessary. This overparameterization helps training but increases computational overhead. One solution is to progressively prune weights from a model during training so that the final network is just a fraction of its original size. Although this approach works, conventional wisdom holds that sparse networks do not train well from scratch. Recent work by Frankle and Carbin (2019) challenges this conventional wisdom. The authors report finding sparse subnetworks inside of larger networks that train to equivalent or even higher accuracies. These "lottery ticket" subnet-

---
［#26］
²bit.ly/3fghqVu

［#29］
Scaling down Deep Learning

［#30］
![](./images/867759325790077038_3.jpg)

［#31］
Figure 3. Finding and analyzing lottery tickets. In a-b), we isolate a minimum viable example of the effect. Recent work by Morcos et al. (2019) shows that lottery tickets can transfer between datasets. We wanted to determine whether spatial inductive biases played a role. So we performed a series of experiments: in c) we plot the asymptotic performance of a 92% sparse ticket. In d) we reverse all the 1D signals in the dataset, effectively preserving spatial structure but changing the location of individual datapoints. This is analogous to flipping an image upside down. Under this ablation, the lottery ticket continues to win. Next, in e) we permute the indices of the 1D signal, effectively removing spatial structure from the dataset. This ablation hurts lottery ticket performance significantly more, suggesting that part of the lottery ticket's performance can be attributed to a spatial inductive bias. Finally, in f) we keep the lottery ticket sparsity structure but initialize its weights with a different random seed. Contrary to results reported in (Frankle and Carbin, 2019), we see that our lottery ticket continues to outperform a dense baseline, aligning well with our hypothesis that the sparsity pattern represents a spatial inductive bias. In g), we verify our hypothesis by measuring how often unmasked weights are adjacent to one another in the first layer of our model. The lottery ticket has many more adjacent weights than chance would predict, implying a local connectivity structure which helps gives rise to spatial biases. Figure 9 in the Appendix visualizes the actual masks, revealing how this local connectivity is manifested.

［#32］
works can be found through a simple iterative procedure: train a network, prune the smallest weights, and then rewind the remaining weights to their original initializations and retrain.

［#33］
Since the original paper was published, a multitude of works have sought to explain this phenomenon and then harness it on larger datasets and models. However, very few works have attempted to isolate a "minimal working example" of this effect so as to investigate it more carefully. Figure 3 shows that the MNIST-1D dataset not only makes this possible, but also enables us to elucidate, via carefully-controlled experiments, some of the reasons for a lottery ticket's success. Unlike many follow-up experiments on the lottery ticket, this one took just two days of researcher time to produce. The curious reader can also reproduce these results in their browser in a few minutes³.

［#34］
Observing deep double descent. Another intriguing property of neural networks is the "double descent" phenomenon. This phrase refers to a training regime where more data, model parameters, or gradient steps can actually reduce a model's test accuracy (Trunk, 1979; Belkin et al., 2019; Geiger et al., 2019; Nakkiran et al., 2020). The intuition is that during supervised learning there is an interpolation threshold where the learning procedure, consisting of a model and an optimization algorithm, is just barely able to fit the entire training set. At this threshold there is effectively just one model that can fit the data and this model is very sensitive to label noise and model mis-specification.

［#35］
Several properties of this effect, such as what factors affect its width and location, are not well understood in the context of deep models. We see the MNIST-1D dataset as a good tool for exploring these properties. In fact, we were able to reproduce the double descent pattern after a few hours of researcher effort. Figure 4 shows our results for a fully-connected network and a convolutional model⁴. We also observed a nuance that we had not seen mentioned in previous works: when using a mean square error loss, the interpolation threshold lies at $n*K$ model parameters where $n$ is the number of training examples and $K$ is the number of model outputs. But when using a negative log likelihood loss, the interpolation threshold lies at $n$ model parameters – it does not depend on the number of outputs.

［#36］
³bit.ly/3nCEIaL
⁴Run in browser: bit.ly/2UBWWNu


［#37］
![](./images/867759325790077038_4.jpg)

［#38］
Figure 4. Observing deep double descent. MNIST-1D is a good environment for determining how to locate the interpolation threshold of deep models. This threshold is fairly easy to predict in fully-connected models, less easy to predict for other models like CNNs, RNNs, and Transformers. Here we see that a CNN has a double descent peak at the same interpolation threshold, although the effect is much less pronounced.

［#39］
This is an interesting empirical observation that may explain some of the advantage in using a log likelihood loss over a MSE loss on this type of task.

［#40］
Gradient-based metalearning. The goal of metalearning is to "learn how to learn." A model does this by having two levels of optimization: the first is a fast inner loop which corresponds to a traditional learning objective and the second is a slow outer loop which updates the "meta" properties of the learning process. One of the simplest examples of metalearning is gradient-based hyperparameter optimization. The concept was was proposed by Bengio (2000) and then scaled to deep learning models by Maclaurin et al. (2015). The basic idea is to implement a fully-differentiable neural network training loop and then backpropagate through the entire process in order to optimize hyperparameters like learning rate and weight decay.

［#41］
![](./images/867759325790077038_5.jpg)

［#42］
Figure 5. Metalearning a learning rate. Unlike many gradient-based metalearning implementations, ours takes seconds to run and occupies a few dozen lines of code. This allows researchers to iterate on novel ideas before scaling. Best viewed with zoom.

［#43］
Metalearning is a promising topic but it is very difficult to scale. First of all, metalearning algorithms consume enormous amounts of time and compute. Second of all, implementations tend to grow complex since there are twice as many hyperparameters (one set for each level of optimization) and most deep learning frameworks are not set up well for metalearning. This places an especially high incentive on debugging and iterating metalearning algorithms on small-scale datasets such as MNIST-1D. For example, it took just a few hours to implement and debug the gradient-based hyperparameter optimization shown in Figure $5^5$.

［#44］
Metalearning an activation function. Having implemented a "minimal working example" of gradient-based metalearning, we realized that it permitted a simple and novel extension: metalearning an activation function. With a few more hours of researcher time, we were able to parameterize our classifier's activation function with a second neural network and then learn the weights using meta-gradients. As Figure 6 shows, our learned activation function substantially outperforms baseline nonlinearities such as ReLU, Elu, and Swish. $^6$ We note that previous works (Clevert et al., 2016; Ramachandran et al., 2018; Vercellino and Wang) have tried to optimize activation functions, but none have done so with analytical gradients computed via bilevel optimization.

［#45］
![](./images/867759325790077038_6.jpg)

［#46］
Figure 6. Metalearning an activation function. Starting from an ELU shape, we use gradient-based metalearning to find the optimal activation function for a neural network trained on the MNIST-1D dataset. The activation function itself is parameterized by a second (meta) neural network. Note that the ELU baseline (red) is obscured by the $\tanh$ baseline (blue) in the figure above.

［#47］
We transferred this activation function to convolutional models trained on MNIST and CIFAR10 images and found that it achieves middle-of-the-pack performance. It is especially good at producing low training loss early in optimization, which is the objective that it was trained on in MNIST-1D. When we rank nonlinearities by final test loss, though, it achieves middle-of-the-pack performance. We suspect that running the same metalearning algorithm on larger models and datasets would further refine our activation function, allowing it to at least match the best hand-designed activation function. We leave line of inquiry to future work.

［#48］
$^5$Run in browser: bit.ly/38OSyTu
［#48］
$^6$Run in browser: bit.ly/38V4G1Q


［#49］
![](./images/867759325790077038_7.jpg)

［#50］
Figure 7. Benchmarking common pooling methods. We observe that pooling helped performance in low-data regimes and hindered it in high-data regimes. While we do not entirely understand this effect, we hypothesize that pooling is a mediocre architectural prior that is better than nothing in low-data regimes but becomes overly restrictive in high-data regimes.

［#51］
Measuring the spatial priors of deep networks. A large part of deep learning's success is rooted in "deep priors" which include hard-coded translation invariances (e.g., con- volutional filters), clever architectural choices (e.g., self- attention layers), and well-conditioned optimization land- scapes (e.g., batch normalization). Principle among these priors is the translation invariance of convolution. A primary motivation for this dataset was to construct a toy problem that could effectively quantify a model's spatial priors. Fig- ure 2 illustrates that this is indeed possible with MNIST-1D. One could imagine that other models with more moderate spatial priors would sit somewhere along the continuum between the MLP and CNN benchmarks.

［#52］
Benchmarking pooling methods. Our final case study be- gins with a specific question: What is the relationship be- tween pooling and sample efficiency? We had not seen evidence that pooling makes models more or less sample efficient, but this seemed an important relationship to under- stand. With this in mind, we trained models with different pooling methods and training set sizes and found that, while pooling tended to be effective in low-data regimes, it did not make much of a difference in high-data regimes (see Figure 7). We do not fully understand this effect, but hypothesize that pooling is a mediocre architectural prior which is better than nothing in low-data regimes and then ends up restrict- ing model expression in high-data regimes. By the same token, max-pooling may also be a good architectural prior in the low-data regime, but start to delete information - and thus perform worse compared to L2 pooling - in the high- data regime. As with the other examples, you can reproduce these results in your browser in a few minutes $^{7}$.

## 5. When to scale

［#53］
We should emphasize that this paper is not an argument against large-scale machine learning research. That sort of research has proven its worth time and again and has come to represent one of the most exciting aspects of the ML research ecosystem. Rather, we are arguing in favor of small-scale machine learning research. Neural networks do not have problems with scaling or performance - but they do have problems with interpretability, reproducibility, and iteration speed. We see carefully-controlled, small-scale experiments as a great way to address these problems.

［#54］
In fact, small-scale research is complimentary to large-scale research. As in biology, where fruit fly genetics helped guide the Human Genome Project, we believe that small-scale research should always have an eye on how to successfully scale. For example, several of the findings reported in this paper are at the point where they should be investigated at scale. We would like to show that large scale lottery tickets also learn spatial inductive biases, and show evidence that they develop local connectivity. We would also like to try metalearning an activation function on a larger model in the hopes of finding an activation that will outperform ReLU and Swish in generality.

## 6. Related work

［#55］
The core inspiration for this work stems from an admiration for all that the MNIST dataset (LeCun et al., 1998) has done for deep learning. While it has some notable flaws - some of which we have addressed - it also has underappreciated strengths: it is simple, intuitive, and provides the perfect sandbox for exploring creative new ideas.

［#56］
Our work also bears philosophical similarities to the Syn- thetic Petri Dish by Rawal et al. (2020). It was published concurrently to this work and the authors make similar ref- erences to biology in order to motivate the use of small synthetic datasets for exploratory research. Their work dif- fers from ours in that they use metalearning to obtain their datasets whereas we construct ours by hand. In doing so, we are able to control various causal factors, such as amount of noise, translation, and padding independently. Also, our dataset is more intuitive to humans: a human can outper-

［#57］
\footnotetext{
［#57］
$^{7}$ bit.ly/31GmTqY
}

［#58］
Scaling down Deep Learning

［#58］
form a strong CNN on the MNIST-1D task. These traits make MNIST-1D a better dataset for investigating "science of deep learning" questions on a small scale. The Synthetic Petri Dish, meanwhile, is not designed to answer the same questions; its objective is specifically to accelerate neural architecture search.

［#59］
There are a number of other small-scale datasets that are commonly used to investigate "science of deep learning" questions. The CIFAR-10 dataset by Krizhevsky et al. (2009) is larger than the MNIST dataset in that individual examples are four times larger, but the number of images is the same. It generally does a better job of discriminating between MLP and CNN architectures, and between various CNN architectures such as vanilla CNNs versus ResNets (He et al., 2015). The FashionMNIST dataset by Xiao et al. (2017) is the same size as MNIST but somewhat more difficult; it aims to rectify some of the most serious problems with MNIST, in particular, that it is too easy and does not discriminate properly between different machine learning models.

［#60］
Scikit-learn by Pedregosa et al. (2011) provides dozens of toy datasets – some synthetic and others real – that are meant for evaluating machine learning models on a small scale. These datasets are appropriate for some "science of machine learning" questions but not for others. For example, the `two_moons` dataset is useful for exploring the role of nonlinearity in classification. However, none of these synthetic tasks is good for exploring the role of spatial inductive biases in deep learning architectures. Making real world analogies to, say, digit classification, is not possible with these datasets, and one can often do very well on them using simple linear or kernel-based methods.

## 7. Discussion
［#61］
There is a counterintuitive possibility that in order to explore the limits of how large we can scale neural networks, we may need to explore the limits of how small we can scale them first. Scaling models and datasets downward in a way that preserves the nuances of their behaviors at scale will allow researchers to iterate quickly on fundamental and creative ideas. This fast iteration cycle is the best way of obtaining insights about how to incorporate progressively more complex inductive biases into our models. We can then transfer these inductive biases across spatial scales in order to dramatically improve the sample efficiency and generalization properties of large-scale models. We see the humble MNIST-1D dataset as a first step in that direction.

## Acknowledgements
［#62］
Thanks to Luke Metz for all the fun conversations. Thanks to Tony Zador for encouraging me to finally publish this.
Thanks to Dmitry Kobak for helping create the tSNE plots comparing MNIST and MNIST-1D shown in Figure 10.

## References






































## A. Supplementary figures

［#63］
More on the human vs. CNN benchmarks. An experienced human can classify MNIST-1D examples at almost 96% accuracy. The CNN can do so at 94% accuracy. Both the human and the CNN struggle primarily with classifying 2's and 7's, and to a lesser degree 4's (see Figure 8). The human had a harder time classifying 9's whereas the CNN had a harder time classifying 1's. Both had zero errors classifying 3's and 6's.

［#64］
Classification errors were fairly evenly balanced across classes, which is a good sign. If only one or two classes were responsible for most of the mistakes, that would have

［#65］
![](./images/867759325790077038_8.jpg)

［#66］
Figure 8. Classwise errors on the MNIST-1D dataset. Some classes contribute more than others, but most represent an appreciable fraction. This is good, because if only one or two classes were responsible for most of the mistakes, that would indicate that those classes are too difficult compared to the others. It's also interesting to note that humans and CNNs struggle with the same classes, such as 7's and 2's.

［#67］
been a sign that those classes were too difficult compared to the others.

［#68］
It's interesting that a human can outperform a CNN on this simple task. Part of the issue is that the CNN is only given 4000 training examples – with more examples it can match and eventually exceed the human baseline. Even though the data is low-dimensional, the classification objective is quite difficult and spatial/relational priors matter a lot. It may be that the architecture of the CNN prevents it from learning all of the tricks that humans are capable of using (eg, using relational reasoning about two signals to determine how they work together to form the digit signal).

［#69］
It's worth noting that CNNs outperform human experts on most large-scale image classification tasks like ImageNet and CIFAR-100. But here is a tiny benchmark where humans are still competitive – this is a nice quality, as it suggests that the key to performing well on the dataset does not rest on shortcut learning such as memorizing specific numbers or patterns to machine precision. A high-performing ML model, we can hope, would have to solve the problem using strategies that would be intuitive to a human.

［#70］
Further analysis of lottery tickets. In addition to the analysis of lottery tickets shown in Figure 3, we include some further visualizations in this appendix. In particular, Figure 9 shows the actual masks of the first layer weights of random and lottery ticket masks. This qualitative comparison helps highlight the spatial inductive bias of the lottery ticket we obtained.

［#71］
Dimensionality reduction. We used tSNE to reduce the dimensionality of MNIST and MNIST-1D so as to plot the datasets in two dimensions. We show the results in Figure 10 with each example colored according to its class label. We observe ten well-defined clusters in the MNIST dataset, suggesting that most examples in most classes are linearly separable from one another. By contrast, we observe few well-defined clusters in the MNIST-1D dataset, suggesting that a classifier must learn a nonlinear representation of the data in order to separate the classes properly. Thanks to Dmitry Kobak for helping with this visualization.

### B. Hyperparameters

［#72］
Table 2. Default hyperparameters of the MNIST-1D dataset.

［#73］
<table>
<thead>
  <tr>
    <th>Hyperparameter</th>
    <th>Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Train/test split</td>
    <td>4k/1k</td>
  </tr>
  <tr>
    <td>Template length</td>
    <td>12</td>
  </tr>
  <tr>
    <td>Padding points</td>
    <td>36-60</td>
  </tr>
  <tr>
    <td>Max translation</td>
    <td>48</td>
  </tr>
  <tr>
    <td>Correlated noise scale</td>
    <td>0.25</td>
  </tr>
  <tr>
    <td>Iid noise scale</td>
    <td>$2 \times 10^{-2}$</td>
  </tr>
  <tr>
    <td>Shear scale</td>
    <td>0.75</td>
  </tr>
  <tr>
    <td>Shuffle sequence</td>
    <td>False</td>
  </tr>
  <tr>
    <td>Final seq. length</td>
    <td>40</td>
  </tr>
  <tr>
    <td>Seed</td>
    <td>42</td>
  </tr>
</tbody>
</table>


［#74］
![](./images/867759325790077038_9.jpg)

［#75］
Figure 9. Visualizing first layer weight masks of random tickets and lottery tickets. For interpretabilty, we have sorted the mask along the hidden layer axis according to the number of adjacent unmasked parameters. This helps reveal a bias towards local connectivity in the lottery ticket masks. Notice how there are many more vertically-adjacent unmasked parameters in the lottery ticket masks. These vertically-adjacent parameters correspond to local connectivity along the input dimension, which in turn biases the sparse model towards data with spatial structure.

［#76］
![](./images/867759325790077038_10.jpg)

［#77］
Figure 10. Visualizing the MNIST and MNIST-1D datasets with tSNE. The well-defined clusters in the MNIST plot indicate that the majority of the examples are linearly separable according to class. The MNIST-1D plot, meanwhile, reveals a lack of well-defined clusters which suggests that nonlinear features are much more important for successful classification.