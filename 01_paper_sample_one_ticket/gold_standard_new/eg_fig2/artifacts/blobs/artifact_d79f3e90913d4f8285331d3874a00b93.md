# One Ticket to Win Them All — Figure 2 evidence slice

### 4.1 Transfer within the same data distribution

[#1]As a first test of whether winning tickets generalize, we investigate the simplest form of transfer: generalization across samples drawn from the same data distribution. To measure this, we divided

[#2]![](./figures/figure2.jpg)

**Figure 2:** Transferring winning tickets within the same data distribution. CIFAR-10 was divided into two halves ("10a" and "10b"), each of which contained 25,000 total examples with 2,500 images per class. Winning tickets generated using CIFAR-10a generalized well to CIFAR-10b for both VGG19 (a) and ResNet50 (b). Error bars represent mean ± standard deviation across six random seeds.

[#3]the CIFAR-10 dataset into two halves: CIFAR-10a and CIFAR-10b. Each half contained 25,000 training images with 2,500 images per class. We then asked whether winning tickets generated using CIFAR-10a would produce increased performance on CIFAR-10b. To evaluate the impact of transferring a winning ticket, we compared the CIFAR-10a ticket to both a random ticket and to a winning ticket generated on CIFAR-10b itself (Figure 2). Interestingly, for ResNet50 models, while both CIFAR-10a and CIFAR-10b winning tickets outperformed random tickets at extreme pruning fractions, both under-performed random tickets at low pruning fractions, suggesting that ResNet winning tickets may be particularly sensitive to smaller datasets at low pruning fractions.

### 4.2 Transfer across datasets

[#4]Our experiments on transferring winning tickets across training data drawn from the same distribution suggest that winning tickets are not overfit to the particular data samples presented during training, but winning tickets may still be overfit to the data distribution itself. To answer this question, we performed a large set of experiments to assess whether winning tickets generated on one dataset generalize to different datasets within the same domain (natural images).
