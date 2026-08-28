［#1］
2021 International Conference on Nascent Technologies in Engineering (ICNTE 2021)

# Enhancing Deep Fingerprinting Attack On Tor Network Using Lottery Ticket Hypothesis

［#2］
Vaibhav Vinodkumar Mishra
Department of Computer Engineering
Don Bosco Institute of Technology
Mumbai, India
vaibhavmishra1011@gmail.com

［#3］
Sayali Achut Kadam
Department of Computer Engineering
Don Bosco Institute of Technology
Mumbai, India
sayali.achyut.kadam@gmail.com

［#4］
Sarthak Anil Tripathi
Department of Computer Engineering
Don Bosco Institute of Technology
Mumbai, India
sarthakt4@gmail.com

［#5］
Shafaque Fatma Syed
Department of Computer Engineering
Don Bosco Institute of Technology
Mumbai, India
shafaque.dbit@dbclmumbai.org

［#6］
Abstract—Tor is one of the popular choices as an anonymity tool to protect the identity of users in digital-based communication. It is used to avoid any eavesdropping that could be performed by local adversaries for monitoring purposes. Website Fingerprinting is a technique by which an eavesdropper determines which websites a user is visiting over an encrypted connection such as Tor. In an encrypted and anonymized network observing patterns of data flows such as packet size and direction helps in identifying a webpage that is being accessed by a user. A new Fingerprinting attack called Deep Fingerprinting makes use of deep learning for finding out the best features to identify a website. It makes use of a Convolutional neural network with a sophisticated architecture design that achieves high accuracy even against the modern defenses proposed for minimizing the effect of fingerprinting. In this paper, we propose to apply The Lottery Ticket Hypothesis on Deep Fingerprinting Attack to reduce the complexity of the model. Thus, the revised model will be efficient in terms of computation costs, storage, and latency. The improvised model is not expected to degrade the accuracy of the model.

［#7］
Keywords—Website Fingerprinting; Deep Learning; Pruning

## I. INTRODUCTION

［#8］
The term internet is associated with peoples' everyday lives. It has been continuously evolving and expanding its reach. The advent of the Internet age has brought us into a world where maintaining one's privacy is getting extremely challenging. For ease of access to resources, metadata gets stored at various points within a network without the knowledge of the user. With this kind of information available, we have seen unprecedented ease of access to information. But at the same time availability of personal information and the various advancements in processing capabilities have had a large impact on users' transparency. Technology has become a necessary part of human life around the world. With such an extent of integration, the personal and meta information of users is being compromised. Edward Snowden has already shown that Internet surveillance has been performed by different agencies who were spying on our communications for various purposes [1]. With various awareness movements, privacy-aware users have sought anonymity in response to the undesirable monitoring of the internet. Various Transparency preserving methods have been proposed around the globe for protecting internet users' privacy such as I2P, Tor, and so on. However, among them, Tor has been the most popular way of seeking anonymity. Two million or more daily users choose Tor for anonymity[1].

［#9］
Tor is an anonymity tool for internet users. It implements the Onion Routing protocol. It is a low-latency and privacy-preserving network. It is continuously developing and is quite a secure system when it comes to ensuring the privacy of its user's browsing activities. This privacy-preserving characteristic is reflected in the perfect forward secrecy exhibited by Tor connections. It encrypts the routing information and the contents of the packet and relays it through a minimum three-node circuit which is randomly chosen from over 7000 different nodes across the globe. This kind of architecture thus makes it difficult for ISPs and local network observers from identifying users' internet activities.

［#10］
Despite having many privacy-preserving feature's Tor is known to be unprotected against traffic analysis attacks. Previous research on Tor Privacy has revealed a serious loophole of Tor network traffic through which the local adversaries were able to infer which websites a particular user is visiting[2]. In an encrypted connection, Website Fingerprinting(we call it WF from here on) aims to identify the web pages that a user is visiting. It analyses the network traffic pattern and other features to identify users and the pages he or she visits. Tor aims to provide the ability to access the internet without revealing any users' identities and disabling any surveillance of their online activity from local or passive network adversaries[3]. However, the website fingerprinting attack gives the potential ability to break the privacy that Tor aims to provide. Several types of research have been done studying the possibilities of the attack and ways in which it can be done. Recent Studies have proposed techniques such as WTF-PAD and Walkie-Talkie by which tor can be protected

［#10］
from fingerprinting attacks[2]. Possibilities of using Deep learning in website fingerprinting have been studied by Rimmer et al. [4] in Automated Website Fingerprinting (We call it AWF from here on) and Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright [2] in Deep Fingerprinting (We call it DF from here on). However, DF stands out among them. It makes use of deep learning with a sophisticated architecture design for attack purposes. This empowers this technique to bypass the defenses proposed for the Website fingerprinting attack with high accuracy.

［#11］
Due to the advancements in computing powers, the field of Artificial Intelligence has seen an unprecedented surge. Neural networks are a way to represent computing tasks in the form of neurons as present in our brains. The ability to replicate the human brain like structure for computing tasks not only increases the volume of data that can be processed but also the ability to perform more complex tasks[5]. While representing such complex structures for computation purposes, several features are considered. However, it has been observed that few features are unnecessary and the results can be obtained by less computation. Frankle & Carbin [6] propose the idea of The Lottery Ticket Hypothesis (LTH) in which they try to identify a smaller subnetwork from big networks while maintaining efficiency.

［#12］
In this paper, we propose to apply pruning techniques on the DF model proposed by Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright in [2] for extracting a smaller subnetwork. We argue that the DF attack proposed can be further improved in terms of computations and can still maintain the efficiency obtained by them. The model will be efficient in terms of storage as the network size gets reduced as well as latency.

## II. LITERATURE SURVEY

### A. WF Attacks and Defenses:

［#13］
The use of Deep Learning for automated feature extraction for WF attacks is presented by Rimmer et al. [4]. For training and testing purposes they collect a dataset in the lab. They obtained a success rate of more than 96% for a set of 100 websites and around 94% for a set of 900 websites in a closed-world scenario. In the open-world scenario, they exceed the accuracy when compared to state-of-the-art attacks by 2%. Based on their result they show that automatic feature learning is far more adaptive to dynamic changes in web content. Thus, they show that the feature engineering process can be automated through Deep Learning for WF Classifiers. However, they do not consider WF defense mechanisms for evaluating their model.

［#14］
Payap Sirinam, Mohsen Imani, Marc Juarez and Matthew Wright [2] have proposed Deep Fingerprinting using a Deep learning technique which is more complex. Throughout the paper, they have compared their work with AWF and shown how their model is better. They have made use of CNN for architecture that has 4 convolution layers followed by fully connected layers. They have provided complete details of their architecture so that other researchers can benefit from it. This model achieves good accuracy against the known WF defenses.

［#15］
Tao Wang* and Ian Goldberg[7] have proposed a solution that tackles three problems to cover the differences between laboratory and realistic conditions for WF. Noisy channel problem was solved by algorithms like KNN, the splitting problem was solved by finding out that splitting based on a time gap which was of 1 second caused no loss to the accuracy of website fingerprinting and they showed that training set of website fingerprinting was not too big to update all data points and keep it fresh.

［#16］
In their study Rahman, Mohammad Saidur & Sirinam, Payap [8] have proposed a new WF attack based on time-dependent features of the packet transfer. This research aims to find new features that can be useful in WF attacks, based on time. In this paper, they have selected eight features in a process that consists of five steps based on the time from the packet timestamp. Certain conclusions are drawn like features that provide classification value against traffic(defended, undefended) and features are strong over noisy instances. These conclusions are drawn using time features in the CNN model.

［#17］
Chan-Tin, Eric & Kim, Taejoon & Kim, Jinoh [9] propose a WF defense(Walkie-Talkie), to defend against all types of attacks on the Tor nodes and Clients. Walkie-Talkie consists of two components: half-duplex communication and burst molding. The half-duplex mode is used in place of full-duplex because it produces burst sequences which can be very easy to mold to leak information. Burst molding is used for molding the burst sequences so that differences between sensitive and non-sensitive pages will be hidden.

［#18］
Jahani, Hojjat & Jalili, Saeed [10] have described an online website fingerprinting attack which recognizes the websites that are visited in both open and close world scenarios with a higher detection rate using a new procedure based on Fast Fourier Transform(FFT) which is used to calculate similarity distance between two instances and form a distance matrix. The conclusion includes several different features recognizing as many as 100 target websites from traffic flow, the detection rate is more stable when the number of websites is more, and accurate recognition of each website.

［#19］
Nepal, Sabita & Dahal, Saurav & Shin, Seokjoo[11] present the first attack on Tor hidden services that allows entry nodes to detect the presence of some hidden service activity from Tor client and server-side. They have mentioned cases of two types of attackers: a weak attacker who does not have clear circuit visibility due to which attacker can exploit notable features of IP and RP circuit and also they can classify circuits into different classes. A strong attacker has clear circuit visibility that identifies different cell sequences that identify IP and RP circuits by using a new pairwise circuit correlation.

［#20］
Cherubin, Giovanni & Hayes, Jamie & Juarez, Marc [12] have proposed two WF defenses for .onion sites which are, a server-side defense which is against website fingerprinting and a client-side defense which is an add-on implemented as a browser, both operate at the application layer. For defense at the server-side, they deformed content of a web page before loaded at the client, and for defense client-side, they showed an evaluation that this defense reduces the accuracy of the website fingerprinting attack in the onion world.

### B. Pruning and LTH
［#21］
Jonathan Frankle, Michael Carbin [6] are the researchers to propose the idea of pruning through the Lottery Ticket Hypothesis. It says "any large neural network will contain a smaller subnetwork that, when trained in isolation will have similar accuracy as that of the larger network". They achieved this through iterative magnitude pruning (IMP): "Starting from a dense initialization train the network until convergence". If IMP is successful it gives the winner of the initialization lottery. However, this is restricted to small-scale tasks such as MNIST and CIFAR-10 i.e. the Lottery Ticket Hypothesis idea of IMA(iterative magnitude pruning) sometimes does not give great results when the network is deeper. Pruning the learning rate schedules needs to be done to overcome it. Unless this was done, they were not able to find a pruned network that was on par with the dense network.

［#22］
In the follow-up work Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin [13] modified IMP in such a way that there was no need for a learning rate warmup. However, the practical benefits of winning tickets are still limited because it still requires a train-prune-retrain process which is costly.

［#23］
Haoran You et. al [14] are the ones to discover for the first time that these winning tickets can be found very early in the training stage. These tickets are called Early-bird tickets. It uses very low-cost training schemes like early stopping and low precision training at larger learning rates.

［#24］
Hattie Zhou, Janice Lan, Rosanne Liu, Uber AI [15] in their paper, further studied the Lottery Ticket Hypothesis. They studied three critical components and showed that each of these may be significantly varied without affecting the overall accuracy. This paper also helps us understand better why the Lottery Ticket Hypothesis works. They proposed the existence of supermasks. Supermasks make untrained, randomly initialized neural networks perform better. So, this paper instead of finding winning tickets suggests to just find the right mask.

［#25］
Ryan Van Soelen, John W. Sheppard [16] found that the results of their experiments support the hypothesis that winning tickets can be transferred and thereby speed up training on new (but related) classification problems.

［#26］
Jiuxiang Gu, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting Liu, Xingxing Wang, Li Wang, Gang Wang, Jianfei Cai, Tsuhan Chen [17] have discussed the convolutional neural networks in detail. They have discussed the various hyperparameters of CNN and how they affect the results.

### III. CONVOLUTION NEURAL NETWORK (CNN)
［#27］
The basic architecture of CNN consists of two major stages:

#### A. Feature Extraction
［#28］
In this stage, The input is given to the first layer of CNN, known as the convolution layer. This layer has a set of filters based on which further calculations take place. It is a mathematical operation that requires two vectors. It applies dot products on them and produces an intermediate set of values called Feature Map. This process is done with each region of input and with each filter. Values obtained from this step are given to an activation function. This process can be visualized as neurons getting activated based on the features present in filtered input. After this, the pooling layer (which is responsible for reducing the dimensionality of the maps when they are too big thus reducing the computation time but at the same time maintaining the important features) is fed with the output of the activation function. To help improve the performance of classifiers', a final step is used in the architecture wherein a stochastic dropout function along with Batch Normalization helps in optimizing the classification process. It also prevents the overfitting of the data to avoid any unnecessary modeling errors.

#### B. Classification
［#29］
In this part, high-level features of the input obtained from the feature extraction step are given as input to the Classification component. Now, A fully connected layer (FC) which is like a normal neural network uses the features obtained in the form of vectors for classification of the input. An interesting thing about this whole process is that while training, the value given by loss functions during classification is used for updating the weights as well as the filters present in the feature extraction stage.

---

### IV. PRUNING AND LOTTERY TICKET HYPOTHESIS
［#30］
A Neural network comprises many parameters. However, "pruning" suggests that the efficiency of neural networks is largely affected by only a few parameters. Pruning in simple terms is a method that speeds up the network and maintains its accuracy by using some of the effective parameters and excluding the rest. In 2019, the MIT researchers published on lottery assumptions [6] and proposed a smarter, easier way to train the neural network by a subset of the model of interest.

［#31］
![](./images/811973643721506819_1.jpg)

［#32］
Fig. 1. Pruning [18]

［#33］
The training process while learning produces a large neural network structure, which can be thought of as equivalent to a big bag of lottery tickets. After the initial training, the model needs an optimization technique, such as pruning to remove unnecessary weights in the network, to reduce the size of the model without affecting the performance. This is equivalent to looking for the winning lottery tickets in the bag and throwing away the rest. Usually, pruning will eventually produce a neural network structure that is 90% smaller than the original. Hence the name "Lottery Ticket Hypothesis". So, the core idea behind the Lottery ticket hypothesis is that any large neural network will contain a smaller subnetwork that, when trained in isolation will have similar accuracy as that of the larger network.

［#34］
The process of finding lottery tickets can be described through the following steps:

［#35］
1.  Randomly initialize a neural network
2.  Train it, Till its convergence
3.  Identify a part of the network and apply pruning
4.  For the remaining part of the network reinitialize the nodes with their original weights from step 1
5.  By examining the convergence behavior we check whether the winning tickets are obtained or not

［#36］
The above steps can be done either once or multiple times, though the researchers have shown that they obtained decent results even in one go instead of iteratively repeating the above steps. However, the greater the number of iterations you do, the better the result you obtain.

［#37］
![](./images/811973643721506819_2.jpg)

［#38］
Fig. 2. Process of obtaining the lottery tickets

### V. PROPOSED MODEL

［#39］
The architecture of the CNN model of Payap Sirinam et al.[2] is shown in Fig. 3.

［#40］
It has 4 blocks with each block consisting of 2 convolutional layers (thus a total of 8 convolutional layers) and 2 fully connected layers followed by an output layer. It uses Max pooling, and ReLu is used as an activation function. ReLu is used as an activation function because the data is 1D i.e. 1x5000 vector and it performs better for such inputs as compared to ELU. Such CNN architectures have transferability. The transferability enables the model to be used as a base model for similar tasks. This allows researchers to use this as a base model in their work thus saving a lot of their time.

［#41］
![](./images/811973643721506819_3.jpg)

［#42］
Fig. 3. DF Model [2]

［#43］
We propose to apply the lottery ticket hypothesis to this model. This model is quite dense, so applying the lottery ticket hypothesis will make it perform a lot faster and better. Fig. 4 represents the workflow pictorially:

［#44］
![](./images/811973643721506819_4.jpg)

［#45］
Fig. 4. Proposed Workflow

［#46］
In [6] researchers have applied the lottery ticket hypothesis on the CNN architectures having two, four, and six convolutional layers followed by two fully-connected layers

［#46］
and max-pooling occurs after every two convolutional layers. They got satisfying results. They also studied these models using early stopping criterion and other techniques to make them converge faster.

［#47］
Even after their work, many other papers have also tried to find out the winning tickets on these structures at less cost with some modifications as we saw in the literature review system. Using these techniques we plan to obtain a subnetwork of Payap Sirinam et al.[2] DF model and verify its fastness and accuracy against the WTF-PAD and walkie-talkie defenses.

［#48］
For Training and Testing purposes we will use the same datasets provided by Payap Sirinam, Mohsen Imani, Marc Juárez, Matthew Wright [2]. This will give us a better understanding of the computation costs and efficiency when compared with the DF. The dataset has been provided in two different scenarios. One is for the open world and the other one is for the closed world. The dataset also contains traces of monitored and unmonitored traces of websites. For the closed-world dataset, they visited Alexa's top 100 sites, each 1250 times, but finally, they ended up having approximately 95 sites with 1000 visits each in the final dataset. In the open world dataset, Alexa's 50,000 websites (excluding the top 100 those were visited in closed world scenario) were visited. Here each site was visited only once. Sites having CAPTCHA pages, Cloudflare's CDN, etc were returning error messages. Excluding such sites, they ended up having data of 40,716 sites. Also, the WTD-PAD and walkie-talkie defended datasets are also provided by the authors. The dataset in each scenario has been given for 3 cases with No defense, WTF-PAD, and Walkie-Talkie. Thus, the model can also be evaluated against the WF defenses WTF-PAD and Walkie-Talkie. In the dataset provided by Payap Sirinam, Mohsen Imani, Marc Juárez, Matthew Wright [2] there are 6 files[19]. The files are in pickle(.pkl) format. The following Are the key highlights of the dataset.

［#49］
Files whose names start with X contain the array of network traffic sequences. The dimension of X's dataset is [n x 5000] where n is the number of instances of the traffic sequences. Each traffic sequence is of length 5000 i.e. each sequence is an array of size 5000. The value in the sequence is either 1 or - 1. Y's dataset consists of the corresponding website's traffic sequence present in X. The dimension of y's dataset is [n] in which n is the total number of network traffic sequence instances. For example, X_<type of data><type of evaluation>.pkl = [[+1,-1,..., -1], ... ,[+1, +1,..., -1]] and y<type of data>_<type of evaluation>.pkl = [45, ... , 12]. In this case, the 1st packet sequence [+1, -1, ..., -1] belongs to website number 45 and the last packet sequence [+1, +1, ..., -1] belongs to website number 12.

## VI. RESULT AND CONCLUSION
［#50］
Dense models require a lot of pruning to find a smaller subnetwork. However, If we prune too much at once, there is a chance that we may damage the network so much that it won't be able to recover. So in practice, an iterative process — often called 'Iterative Pruning': Prune / Train / Repeat is followed. Still, the number of training rounds and number of epochs within each round varies as per the density of the network.

［#51］
Therefore obtaining the lottery tickets in a dense network requires lots of iteration. However, even in a few iterations, the effect can be seen. With each iteration, the count of trainable parameters keeps getting decreased without affecting the accuracy. We tried to perform the iterative pruning for a few rounds on the DF model for Walkie-Talkie defense. The training of the DF model for the Walkie-Talkie defense with dropout is as shown in the graph below.

［#52］
![](./images/811973643721506819_5.jpg)

［#53］
Fig. 5. Walkie-Talkie Training with dropout

［#54］
The Walkie-Talkie DF model has an accuracy of 49.7 % without pruning. The total number of trainable parameters initially are 3981476. We did iterative pruning for the walkie-talkie defense dataset DF model for 9 rounds with each round consisting of a maximum of 30 epochs (fewer epochs if early stopping happens). The number of trainable parameters remaining and accuracy after each round is shown in Table 1.

［#55］
<table>
<caption>TABLE I. RESULT OBTAINED AFTER INITIAL TRAINING</caption>
<thead>
<tr>
<th rowspan="2">Round</th>
<th colspan="2">The result after each Iterative round</th>
</tr>
<tr>
<th>Number of trainable parameters</th>
<th>accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>3295575</td>
<td>0.4942</td>
</tr>
<tr>
<td>2</td>
<td>2735936</td>
<td>0.4965</td>
</tr>
<tr>
<td>3</td>
<td>2278399</td>
<td>0.4911</td>
</tr>
<tr>
<td>4</td>
<td>1903524</td>
<td>0.4866</td>
</tr>
<tr>
<td>5</td>
<td>1595664</td>
<td>0.4913</td>
</tr>
<tr>
<td>6</td>
<td>1342202</td>
<td>0.4918</td>
</tr>
<tr>
<td>7</td>
<td>1133023</td>
<td>0.4927</td>
</tr>
<tr>
<td>8</td>
<td>959835</td>
<td>0.4924</td>
</tr>
<tr>
<td>9</td>
<td>816088</td>
<td>0.4871</td>
</tr>
</tbody>
</table>

［#56］
We can see in Table 1 that the number of trainable parameters is decreasing after each round and the accuracy of the model is almost equal to the original unpruned model accuracy. With more numbers of such iterations, we can reach a point of maximum accuracy (which can be identified easily since there will be a drastic decrease in the accuracy after that).

［#57］
2021 International Conference on Nascent Technologies in Engineering (ICNTE 2021)

［#58］
The extent to which a model can be pruned will be known only after more iterations are performed. Since, the DF model is quite dense it requires a lot of iterations for pruning. However, we were able to obtain some results by limiting the number of iterations. These results give enough confidence that we can obtain winning tickets if we perform more number of iterations. Thus, a smaller subnetwork can be obtained for the DF which can perform as good as the original model with less space and faster results.

［#59］
Website fingerprinting is done by examining several features (patterns) of the website. When manually engineered, even the slight change in the content of a website over time can make these features ineffective. Since manual engineering requires lots of effort and is so non-immune to the changes in web content, it is not a very effective technique. Contrary to this, the Deep fingerprinting approach has good immunity against such changes in the web content since it automatically learns the implicit features. Because of its ability to learn as per the most relevant traffic features, it becomes highly efficient and flexible.

［#60］
The new sparse model which will be obtained is expected to take less storage space and will be computationally faster. Because of less storage and computational cost, the deployment of such a neural network will become easier. If models such as sparse DF become easily deployable in real-time scenarios then it would be even more difficult for users to attain anonymity through Tor. Because they have strong immunity against known defenses of Tor, it prompts us to think about new and robust defenses that can provide immunity against such attacks for providing Tor users with a safe environment for internet access.

## REFERENCES



















