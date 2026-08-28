# LXMERT Model Compression for Visual Question Answering

［#1］
Maryam Hashemi Ghazaleh Mahmoudi * Sara Kodeiri* Hadi Sheikhi* Sauleh Eetemadi

［#2］
School of Computer Engineering, Iran University of Science and Technology, Iran
{m_hashemi94, gh_mahmoodi, sara_kodeiri, ha_sheikhi}@comp.iust.ac.ir, sauleh@iust.ac.ir

## Abstract
［#3］
Large-scale pretrained models such as LXMERT are becoming popular for learning cross-modal representations on text-image pairs for vision-language tasks. According to the lottery ticket hypothesis, NLP and computer vision models contain smaller subnetworks capable of being trained in isolation to full performance. In this paper, we combine these observations to evaluate whether such trainable subnetworks exist in LXMERT when fine-tuned on the VQA task. In addition, we perform a model size cost-benefit analysis by investigating how much pruning can be done without significant loss in accuracy. Our experiment results demonstrate that LXMERT can be effectively pruned by 40%-60% in size with 3% loss in accuracy.

［#4］
Deep Learning recently enjoyed welcoming a new powerful pruning method: The Lottery Ticket Hypothesis (LTH) (Frankle and Carbin, 2019). LTH has been shown great success in various fields. It could be a powerful tool to understand the parameter redundancy in the current pretrained V+L models. Thus, we aim to apply LTH to LXMERT(Tan and Bansal, 2020), one of the best-performing two-stream V+L models, to fill this gap. We evaluate our work on VQA (Antol et al., 2015) and compare it with DistillVLM(Fang et al., 2021), which leverages the knowledge distillation technique to compress large visual-linguistic models.

［#5］
Similar to this work, Gan et al. (2021) study LTH for UNITER(Chen et al., 2020). However, UNITER is a single stream V+L model, and LXMERT is a two-stream model; our results are consistent with theirs.

## 1 Introduction and Related Work
［#6］
Over the past few years, many single-modal pretrained models have been proposed. Inspired by this, the vision-and-language pretraining seeks to learn joint representations using visual and textual content to improve the efficiency of vision-language tasks.

［#7］
Both single-modality and cross-modality pretrained models often have hundreds of millions of parameters. Unfortunately, training these over-parametrized models can be prohibitively time-consuming and costly, making them impractical for resource-limited devices. However, cross-modality pretrained models suffer more from the increased model size due to the higher input space dimension. With the task of Visual Question Answering (VQA) (Antol et al., 2015) in mind, and its ultimate goal of being helpful to the visually impaired, decreasing V+L model size makes it feasible to use them in limited-resource devices.

［#8］
To address this problem, model compression techniques such as pruning have been developed.

## 2 Methodology
［#9］
In this section, we briefly explain the LXMERT architecture and LTH. Then, we describe how we use LTH to compress the pretrained LXMERT model.

［#10］
LXMERT is a Transformer-based model which takes two inputs: image and text. Internally, LXMERT consists of two types of encoders: single-modality encoders for each modality and a cross-modality encoder using bidirectional cross attention to exchange information and align entities across the modalities.

［#11］
The Lottery Ticket Hypothesis (Frankle and Carbin, 2019) shows that by preserving the original weight initializations from the unpruned network, you can train a network with the topology of the pruned network and achieve the same or better test accuracy within the same number of training iterations.

［#12］
In order to apply LTH to the LXMERT model, we use iterative magnitude pruning. Therefore, we fine-tune LXMERT on the VQA task and iteratively prune 10% of the lowest magnitude weights across the entire model, excluding embedding and

---
［#13］
*These authors contributed equally.

［#14］
<table>
<thead>
<tr><th rowspan="2">Method</th><th colspan="4">test-dev</th><th colspan="4">test-std</th></tr>
<tr><th>Yes/No</th><th>Number</th><th>other</th><th>Overall</th><th>Yes/No</th><th>Number</th><th>other</th><th>Overall</th></tr>
</thead>
<tbody>
<tr><td>DistillVLM</td><td>-</td><td>-</td><td>-</td><td>69.6</td><td>-</td><td>-</td><td>-</td><td>69.8</td></tr>
<tr><td>LXMERT</td><td>88.24</td><td>54.45</td><td>63.05</td><td>72.45</td><td>88.29</td><td>54.37</td><td>63.18</td><td>72.63</td></tr>
<tr><td>LXMERT (low-magnitude)</td><td>86.95 ± 0.95</td><td>52.60 ± 1.87</td><td>60.96 ± 1.76</td><td>70.72 ± 1.44</td><td>87.07 ± 1.12</td><td>52.28 ± 1.66</td><td>61.02 ± 1.83</td><td>70.87 ± 1.51</td></tr>
<tr><td>LXMERT (high-magnitude)</td><td>74.11 ± 0.91</td><td>42.81 ± 1.36</td><td>50.5 ± 0.19</td><td>59.35 ± 0.61</td><td>74.23 ± 0.81</td><td>42.99 ± 0.87</td><td>50.71 ± 0.26</td><td>59.62 ± 0.55</td></tr>
<tr><td>LXMERT (random)</td><td>69.26 ± 0.29</td><td>39.84 ± 0.93</td><td>45.96 ± 0.83</td><td>54.86 ± 0.52</td><td>69.27 ± 0.18</td><td>40.34 ± 0.66</td><td>46.33 ± 0.79</td><td>55.19 ± 0.45</td></tr>
</tbody>
</table>

［#15］
Table 1: Performance of subnetworks at 50% weights pruning on VQA v2, which reported for both test-dev and test-std. Test-dev is used for debugging and validation experiments. Test-standard is the default test data for the VQA competition. We test each experiment for three different seeds and report the mean and standard deviation of VQA accuracy across three seeds.

［#16］
output layers. We keep pruning until our model loses roughly half the weights. We use the default settings and hyperparameters of LXMERT (Tan and Bansal, 2020) to finetune on the VQA v2.0 dataset.

## 3 Experimental Setups and Results
［#17］
The experiments are designed to investigate the effectiveness and stability of LTH on LXMERT in addition to cost-benefit analysis of the number of parameters in the model. We conduct experiments on the widely-used VQA v2.0 (Goyal et al., 2017) dataset built based on the MS-COCO (Lin et al., 2014) image corpus.

### 3.1 Effectiveness and Stability
［#18］
The following steps are performed to compress the LXMERT model.
1.  The pretrained LXMERT model plus the VQA classifier's randomly initialized weights are saved.
2.  The model is fine-tuned on the 3,129 most frequent answers in the VQA v2.0 dataset.
3.  Iterative magnitude pruning is applied to find the low-magnitude subnetwork (pruning 50% of the low-magnitude weights). The high-magnitude subnetwork is computed as a compliment of the low-magnitude subnetwork with equal size. A random subnetwork with an equal size is generated for comparison.
4.  The saved weights are restored for all three subnetworks.
5.  The high-magnitude, low-magnitude, and random subnetworks are fine-tuned and evaluated on the VQA Task using three different seeds for initializing the VQA model to ensure the stability of the results.

［#19］
Results of subnetworks at 50% weights pruning on VQA v2.0 are summarized in Table 1 where DistillVLM (Fang et al., 2021) is also listed for comparison. Row 2 to row 5 reports respectively full finetuned LXMERT, low-magnitude, high-magnitude, and random subnetworks. Low-magnitude pruning achieves 97% of full finetuned LXMERT accuracy in overall for both test-dev and test-std and shows marginal improvement over DistillVLM as the baseline. By comparing performance across the subnetworks, random and high magnitude subnetworks perform far worse than low-magnitude subnetwork. Surprisingly, the results demonstrate high-magnitude subnetwork performing better than random subnetwork. This could be a LXMERT specific phenomenon and required further investigation.

### 3.2 Cost-Benefit Analysis
［#20］
We experiment with low-magnitude subnetwork by pruning 10% of the weights all the way up to 90% of the weights in 10% increments. Accuracy of these pruned models on VQA v2.0 are reported in Figure 1. Our results indicate a significant loss of accuracy after 50% to 60% pruning.

［#21］
![](./images/924003605663449356_1.jpg)

［#22］
Figure 1: Model size cost-benefit analysis.

## 4 Conclusion
［#23］
We confirm that LTH pruning is an effective method for pruning V+L pretrained models. We mainly focused on LXMERT, a two-stream V+L pretrained model, but our findings are consistent with Gan et al. (2021)'s results while using UNITER, a single-stream V+L pretrained model.

### References







