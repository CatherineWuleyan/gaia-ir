# On the Stability of Growth in Structural Plasticity

Lute Lillo¹ Nick Cheney²

¹University of Vermont

## Abstract
Standard deep-learning pipelines usually choose the network architecture before training and keep it fixed throughout optimization. In contrast, a model can also be adapted by editing its structure during training, for example by pruning existing hidden-neuron units or growing new ones. Although growth is appealing for adaptive and continual systems, we show that it is not simply the inverse of pruning. Pruning selects among units that have participated in training from the start, whereas growth inserts new units into an already specialized optimization trajectory. We isolate this insertion problem and show that newborn units are often forward-active but backward-starved: they participate in the forward computation, yet receive much weaker gradient signal than incumbent units. This disadvantage is minor in small MLP benchmarks, but becomes clear in harder image-classification settings with a convolutional trunk. In these settings, GROW can achieve high final accuracy during the structural-editing procedure, while PRUNE is stronger when performance is averaged over the training trajectory or when the final sparse network is retrained from scratch. Interventions targeting optimizer state, insertion, selection, and trainability show that improving the integration of newborn units can improve adaptive performance, but does not automatically produce better final subnetworks. In continual-learning benchmarks stressing plasticity loss, GROW becomes competitive mainly when new units have enough time to integrate. Together, these results suggest that GROW should be evaluated not only as an architecture-search operator, but as a time-sensitive optimization process whose success depends on insertion stability.

## 1 Introduction
Structural plasticity—the ability to modify network architectures during training—is a natural primitive for automated machine learning under explicit resource budgets such as parameters, FLOPs, or latency. Its two basic operators are *pruning*, which removes capacity from an overparameterized model [18, 19], and *growth*, which adds capacity to a compact one [7, 37, 39, 43]. Both can be viewed as search operators in architecture space, and dynamic sparse training (DST) shows that structure can be updated online while respecting a fixed parameter budget [3, 8, 24, 32]. In practice, however, structural adaptation remains dominated by pruning-based approaches.

This imbalance reflects an important asymmetry. Pruning begins with excess capacity: candidate units are present from initialization, participate in early training, and can later be selected or removed. This is the intuition behind the lottery-ticket hypothesis, which argues that dense networks can contain sparse subnetworks that are trainable when selected from the original training trajectory [13, 14]. Growth offers the complementary promise of adding capacity only when and where it is needed, which is especially appealing for adaptive and continual systems because it can add capacity as tasks or distributions shift [26, 31, 43, 44]. However, this benefit depends on whether newly inserted capacity can stabilize before the next shift arrives; otherwise, growth can require repeated expansion and can itself become a source of instability [47]. Yet newly added units enter late, after the network has already specialized, and must become useful inside a mature optimization trajectory. Thus, GROW–PRUNE comparisons can conflate two questions: whether the final sparse architecture is good, and whether the insertion process allowed new units to integrate quickly enough during training.

Preprint. Under review.
© 2026 the authors, released under CC BY 4.0

This insertion perspective connects several mechanisms previously studied in isolation. Function-preserving widening methods aim to reduce insertion-induced disruption [5, 17, 38]; gradient- or activation-informed rules ask where to expand [10, 11, 40]; and recent growing-network work highlights old-new optimization asymmetries such as optimizer-state transfer and age-dependent learning rates [46]. More broadly, warm-up, layer-wise modulation, and adaptive optimizer state all point to the same issue: a newly inserted unit may be disadvantaged not only by where it is placed, but by when it enters training [21, 33, 35, 42, 45, 48].

Therefore, we study growth as a structural plasticity primitive and treat insertion stability as the central object of analysis. Newly added units face three birth-time disadvantages: (i) function shock, where insertion perturbs the learned input-output mapping; (ii) cold start, where new parameters lack optimizer state; and (iii) weak learning signal, where newborn units receive disproportionately little credit relative to incumbent units. Throughout, "units" refers to hidden neurons added or removed at the neuron level, not individual connections; our study therefore concerns unit-level structural edits rather than unstructured synapse-level rewiring [6].

The central claim of the paper is that growth is not primarily limited because it cannot discover useful sparse architectures, but because newly added units must integrate late into a mature optimization trajectory. This can make the adaptive process less stable and more path-dependent, even when the final retrained mask is competitive with pruning. We isolate this insertion primitive and ask: when do Grow and Prune differ, does that difference reflect final sparse-architecture quality or the adaptive process used to produce it, and when can growth's process-level disadvantage be reduced?¹

- We show that the Grow-Prune gap is not monolithic: in small MLPs, growth and pruning produce similarly retrainable masks, while in convolutional feature-learning regimes the main asymmetry appears in trajectory quality and path dependence rather than ticket quality (Sec. 4).
- We identify insertion-time optimization disadvantage as a process-level bottleneck for growth, showing that newborn units can be forward-active while remaining backward-starved (Sec. 4.2).
- We use interventions on optimizer state, insertion, selection, and activation-level trainability as probes of this bottleneck, showing that improved integration can strengthen adaptive-process performance without necessarily producing a better retrainable final subnetwork (Sec. 5).
- We show that under continual shift, growth is most effective when new units have time to integrate before the next distributional change; with a drop-in plasticity-preserving activation, Grow can become competitive with or outperform Prune (Sec. 6).

## 2 Background & Related Work

Dynamic sparsity and structural operators. Dynamic sparse training (DST) methods maintain a fixed parameter budget while updating sparse connectivity online, combining pruning and regrowth as intertwined structural operators [3, 8, 32]. This line of work reinforces the AutoML view of pruning and growth as search moves in architecture space, and highlights that the allocation rule—which connections or units receive structure—interacts strongly with learning dynamics [10]. We do not focus on the allocation rule, but on the stability of the insertion event itself.

Function-preserving transformations for growth. A classic approach to stable architectural expansion is to preserve the network function at insertion time. Net2Net and Network Morphism provide widening transformations that initialize expanded networks to compute approximately the same input-output mapping [5, 38], and MorphNet shows how width can be optimized under resource constraints [17]. These methods reduce insertion-induced perturbations, but do not by themselves resolve the optimization asymmetries between newborn and incumbent parameters.

---
¹Code available at: https://anonymous.4open.science/r/structural_plasticity-1544

Growth in continual learning. Many continual-learning methods rely on architectural expansion to accommodate new tasks, with design choices centered on when, where, and what to grow [12, 26, 37, 43, 44]. Growth is appealing because it allocates fresh capacity for new information, potentially easing the stability-plasticity dilemma. At the same time, strong isolation-based approaches such as PackNet, Piggyback, and winning-subnetwork methods achieve low forgetting by assigning task-specific subnetworks within shared weights [20, 29, 30], often at the cost of task identity, routing, or selection at inference time. Recent work further shows that poorly controlled expansion can itself induce forgetting in task-agnostic settings [47]. In our work we ask whether newly added units can integrate stably enough for growth to be a competitive structural operator.

## 3 Experimental Setup

We compare three model families under matched data streams, optimizers, and compactness targets:

- **Dense**: no sparsification.
- **Prune**: Iterative Magnitude Pruning (IMP) applied only to masked layers. After each pruning step, surviving weights are rewound to an earlier checkpoint, following the standard Lottery Ticket Hypothesis (LTH) protocol. Although a non-rewind pruning baseline would more closely mirror the procedural setup of Grow, IMP-style rewind provides the canonical sparse-subnetwork-selection baseline and lets us ask whether the final mask itself is a strong retrainable architecture (see App. A.6.2).
- **Grow**: start from a sparse seed mask and iteratively activate units until reaching the target compactness. To decide what to activate, we score currently masked-out units on a mini-batch by how often their post-activation would exceed a small threshold if unmasked. Intuitively, this estimates how often an inactive candidate would be meaningfully active if recruited (see App. A.6.1). We also tested gradient-based recruitment with similar qualitative conclusions (App. B.4). Masks are then updated in place, with no rewind, because the object of study is precisely the late insertion of new units into a mature optimization trajectory. At each growth event, newborn units are added to the existing active set, not used to replace earlier units.

Our first setting uses a 3-layer MLP with two masked hidden layers and an unmasked 10-way classifier head, so structural edits only reallocate hidden capacity without altering the output mapping. Each Grow or Prune run proceeds through a sequence of structural-edit cycles until reaching a final target retained compactness $c \in \{20, 30, 40, 50\}\%$. At the beginning of a cycle, the method updates the active unit mask by either adding units (Grow) or removing units (Prune); the resulting network is then trained for a fixed number of epochs before the next edit until reaching $c$ through opposite edit trajectories. We use these cycles only as evaluation checkpoints. Each method ultimately produces a final binary mask. To separate the quality of the editing process from the quality of the final sparse architecture, we evaluate each method in two ways:

1.  Cycle performance: accuracy is measured during the Grow/Prune procedure itself, at the end of each structural-edit cycle. This captures how well the model performs while architecture changes are being made online during training.
2.  Winning-Ticket performance: accuracy is measured after freezing the discovered mask, reinitializing the model, and retraining it from scratch. This follows the lottery-ticket evaluation protocol and tests whether the discovered sparse architecture is trainable independently of the path used to find it [13].

For both evaluations, we report (i) final cumulative accuracy after the last task (ACC), and (ii) trajectory-average accuracy (TAA), computed as the mean cumulative test accuracy across the training trajectory. TAA captures learning speed and retention across the stream, whereas ACC reflects end-of-stream performance.

## 4 Results

Minimal MLP Benchmarks: No Stable Growth Gap. On IID MNIST [25], all methods achieve high accuracy. On class-incremental Split-MNIST with a single shared 10-way head, all methods collapse to near chance without a stabilizing mechanism, indicating that catastrophic forgetting dominates the architectural comparison. Therefore, we add a small replay buffer called Tiny ER (50 samples per class) to make structural differences interpretable. Under this setting, with SGD at $\eta = 0.01$, Grow, Prune, and Dense remain closely matched across compactness levels, with no consistent ordering across either cycle or winning-ticket metrics. The same pattern holds for Split-Fashion-MNIST [41]. As summarized in Table 1, these small MLP benchmarks do not expose a stable Grow–Prune gap: both methods find masks that retrain to similar final accuracy and TAA across compactness budgets. Thus, the stronger asymmetries studied later are not simply caused by sparsity or by class-incremental training; rather, they become more clearly when structural edits occur inside a model learning non-trivial visual representations. This motivates the next step: moving to convolutional feature-learning regimes, where insertion-time asymmetries may become more consequential.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">MNIST (IID)</th>
      <th colspan="3">Split-MNIST (+Tiny ER)</th>
      <th colspan="3">Split-Fashion (+Tiny ER)</th>
    </tr>
    <tr>
      <th>WT Final</th>
      <th>WT TAA</th>
      <th>$\Delta_{\text{WT-C}}$</th>
      <th>WT Final</th>
      <th>WT TAA</th>
      <th>$\Delta_{\text{WT-C}}$</th>
      <th>WT Final</th>
      <th>WT TAA</th>
      <th>$\Delta_{\text{WT-C}}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dense</td>
      <td>95.98</td>
      <td>93.13</td>
      <td>–</td>
      <td>84.36</td>
      <td>90.44</td>
      <td>–</td>
      <td>75.01</td>
      <td>84.34</td>
      <td>–</td>
    </tr>
    <tr>
      <td>Grow</td>
      <td>95.98</td>
      <td>93.79</td>
      <td>-0.58</td>
      <td>84.92</td>
      <td>90.98</td>
      <td>+11.54</td>
      <td>75.16</td>
      <td>85.06</td>
      <td>+7.76</td>
    </tr>
    <tr>
      <td>Prune</td>
      <td>95.94</td>
      <td>93.74</td>
      <td>-0.61</td>
      <td>85.14</td>
      <td>90.73</td>
      <td>+8.05</td>
      <td>75.50</td>
      <td>85.26</td>
      <td>+5.34</td>
    </tr>
  </tbody>
</table>

Table 1: Small MLP benchmarks do not expose a stable Grow–Prune ticket-quality gap. For Grow and Prune, values are averaged over compactness budgets $c \in \{20, 30, 40, 50\}\%$; Dense is reported once because it is not compactness-dependent. WT Final and WT TAA are both measured after re-initializing and retraining the final sparse mask from scratch. $\Delta_{\text{WT-C}} = \text{Final}_{\text{WT}} - \text{Final}_{\text{Cycle}}$ measures how much the retrained ticket endpoint differs from the endpoint reached during the structural-edit cycle. Negative values indicate a stronger warm-started cycle endpoint; positive values indicate that the fixed mask trains better under the WT protocol than during the structural-edit process. Across these settings, Grow and Prune remain close in WT Final and WT TAA, motivating the later ConvNet experiments where Grow–Prune asymmetries become more pronounced. Entries are mean over 10 seeds. Full compactness results with 95% CI (omitted here for readability) are provided in App. 3.

### 4.1 Scaling to ConvNets: Feature learning exposes the Grow–Prune asymmetry

In small fully-connected MLPs, Grow and Prune often produce similarly strong winning tickets. Next, we move to CIFAR-100 (and CIFAR-10; App. B.6) [22], where convolutional feature learning makes optimization substantially harder. We use a hybrid ConvNet: a dense two-layer convolutional trunk followed by a four-layer fully-connected head. Only the hidden layers of the FC head are grown or pruned; the convolutional trunk remains dense for all methods. This lets us test whether the Grow–Prune separation emerges in a harder representation-learning regime while keeping the structural edits unit-level and comparable to the MLP experiments. Models are trained with SGD ($\eta$=0.1, found via hyperparameter sweeps for each treatment), use the neutral allocation schedule for editable FC layers (App. B.5), and are evaluated using both *Cycle* and *Winning-Ticket* metrics.

Procedure-level adaptation vs. final mask quality. The apparent Grow–Prune ordering depends on the evaluation axis. During the structural-editing procedure, Grow reaches the strongest endpoint at every compactness, while Prune remains lower (Fig. 1a). After freezing the final mask, reinitializing, and retraining from scratch, this endpoint advantage largely disappears: Grow and Prune produce similarly retrainable masks, with Prune slightly stronger at higher compactness (Fig. 1b). The trajectory-average view reverses the cycle-level endpoint story: Prune is stronger

![](./images/7466532944091754543_0.png)

Figure 1: Cycle vs. Winning-Ticket performance on CIFAR-100. Panels (a)-(d) show mean ± 95% CI with individual seed points. (a) Grow achieves higher final cycle accuracy than Prune, but (b) this advantage vanishes when retraining the final mask from scratch. (c) Viewing the overall trajectory, Prune maintains a stronger or comparable TAA over the cycle, while (d) winning-ticket TAA remains similar across all sparse methods indicating that the final masks have comparable retrainable trajectory quality. (e) An increasingly negative performance gap ($\Delta =$ ticket – cycle) for Grow indicates that its endpoint gains rely on the warm-started adaptive path. In contrast, Prune retrains as well as, or better than, its cycle endpoint.

at low and intermediate compactness during the edit-time process, partly because it has more active units than Grow before the final compactness is reached, while retrained-mask TAA remains similar across sparse methods (Fig. 1c-d). Thus, Grow is not simply failing to find useful sparse masks. Rather, its strong cycle endpoints are path-dependent: they arise during the warm-started editing process, do not translate into clearly superior retrainable sparse architectures, and come with weaker trajectory-average performance while new units integrate. Therefore, we expose a key asymmetry: growth can be architecturally competitive, but its insertion process is less stable and more time-sensitive.

### 4.2 Allocation dynamics under structural edits

Structural plasticity changes both *who participates* in the forward computation and *who receives credit* during optimization. To explain the CIFAR-100 separation from Sec. 4.1, we analyze event-aligned cohort metrics for units that are grown, kept, or pruned. Our goal is to test whether structural edits induce consistent unit-level asymmetries, and whether those asymmetries match the observed difference between Grow and Prune.

![](./images/7466532944091754543_1.png)

Figure 2: Growth inserts units that participate in the forward pass but receive weak backward signal. Event-aligned cohort diagnostics on CIFAR-100, where log-parity 0 denotes equality between the compared cohorts. (a) At birth, newborn Grow units have positive activation parity, showing that they are not inactive or dead on arrival; however, this forward participation weakens across successive grow cycles. (b) The same newborn units have strongly negative gradient parity, showing that they receive much less backward credit than already-active units even when they participate in the forward pass. (c) For Prune, kept units are initially more active than removed units, but this separation shrinks across prune cycles, indicating that later pruning decisions become less cleanly separated. (d) Survivor stability is the post-prune change in activation of units that survive pruning; increasingly negative values indicate that repeated pruning progressively perturbs the remaining representation.

Newborn units are active at birth, but gradient-disadvantaged. Figure 2 distinguishes forward participation from backward signal. Newborn units are active at birth, but receive substantially

weaker gradients than previously active units: they are not silent, but the current loss is less sensitive to changes through their downstream pathways, indicating a backward-pass integration problem. Thus, growth inserts new capacity into an already mature optimization trajectory, where it is forward-active but initially weakly coupled to the backward learning signal. For PRUNE, removed units are initially less active than survivors, but this separation shrinks over cycles and repeated pruning increasingly perturbs the remaining representation.

Post-birth dynamics: activations approach parity; gradients do not. Unlike Fig. 2, which measures the immediate newborn condition at insertion, Fig. 3 asks whether that birth-time asymmetry persists after the subsequent within-cycle training interval. Newborn units approach old units in forward activity, but their gradient post-birth dynamics remains far below parity, including at the end of each cycle. Thus, the limitation is not whether newly added units can become active, but whether they receive enough backward signal to become useful quickly.

![](./images/7466532944091754543_2.png)

Figure 3: Newborn units approach forward-activity parity, but not backward-signal parity. Post- birth dynamics on CIFAR-100 measure newly grown units relative to already-active units over the remaining training segment after each growth event; parity is marked by the red dashed line at 1. Across compactness levels, activation ratio (blue) stays near parity and sometimes exceeds it, indicating that newborn units participate in the forward computation after insertion. In contrast, gradient ratio (orange) remains far below parity throughout the cycle, and the end-of-cycle gradient markers also stay below 1. Thus, even when newborn units become forward-active, they remain under-integrated in the backward pass, supporting a credit-assignment rather than dead-unit explanation for the growth bottleneck.

Interpretation. These diagnostics explain the cycle-vs.-ticket dissociation in Sec. 4.1. GROW adds units that become forward-active quickly, but remain weakly coupled to the backward learning signal; PRUNE instead preserves a more mature learning allocation, even as repeated removals perturb the survivor set. Thus, the GROW–PRUNE gap appears to reflect edit-time integration dynamics more than final mask quality alone. Consistently, more frequent growth under a fixed training horizon lowers Cycle-TAA much more than final Cycle-ACC (App. B.7), and gradient-based top-$k$ growth does not remove the newborn gradient disadvantage (App. B.4).

## 5 Interventions on the newborn integration bottleneck

The allocation diagnostics of Sec. 4.2 suggest that GROW is limited less by dead-units capacity than by slow *newborn integration*. Therefore, we evaluate four possible sources of this integration bottleneck: (i) optimizer *state* mismatch due to lacking accumulated states, (ii) disruptive *insertion* operations that perturb the current function, (iii) suboptimal *selection* of inactive units, and (iv) restricted gradient flow from *activation functions*. These hypotheses motivate different interventions to ask which mechanisms relieve this integration bottleneck.

Optimizer-state interventions: Two-Speed and Moment Transplant. One possibility is that newborn units lag because they are born with a cold optimizer state and must compete with mature units whose weights and adaptive moments already encode useful learning history. We propose two optimizer-side interventions of this hypothesis. First, *Two-Speed* is an update-scaling variant inspired by learning-rate adaptation for incrementally grown networks [46]; it temporarily multiplies the optimizer update on newborn-associated parameters—the incoming and outgoing

weights of newly activated units—so that newborn capacity can move faster during its early integration window. Equivalently, if $\theta$ denotes these newborn-associated parameters and $\theta_{t+1}^{\text{base}}$ is the update proposed by the base optimizer, we apply $\theta_{t+1} = \theta_t + r(\theta_{t+1}^{\text{base}} - \theta_t)$ with multiplier $r > 1$ during warmup. On the other hand, recent work suggests that adaptive optimizers maintain an internal memory of past gradients and that structural edits can induce optimizer-state mismatch unless buffers are handled explicitly [1, 2]. Thus, *Moment Transplant* instead copies optimizer buffers from an incumbent donor unit into a newborn unit, testing whether optimizer cold-start limits integration.

Insertion intervention: Net2Wider. A second possibility is that the insertion primitive itself is disruptive: newly activated units may perturb the current function before they have learned a useful role. To test this, we implement $\text{Net2Wider}$ [5] as a function-preserving widening operator that duplicates selected incumbent units and redistributes outgoing weights so that the network's forward function is initially unchanged. This tests whether reducing insertion shock is sufficient to improve newborn integration. However, function preservation does not necessarily imply better credit assignment: because the newborn initially shares a role with its donor, it may enter as a redundant unit rather than as a strongly differentiated source of new capacity.

Selection intervention: GradMax-style recruitment. The $\text{Grow}$ disadvantage may also depend on which dormant units are activated. Therefore, we introduce a *GradMax*-style selector [11] that scores inactive candidates by the learning signal they would receive on the mini-batch available at the growth event, and activates the top-$k$ units within each growable layer. This changes the selection policy only: insertion, optimizer handling, and post-birth dynamics remain fixed unless explicitly combined with another intervention.

Activation-function intervention: Rand. Smooth-Leaky. A complementary possibility is that the bottleneck is partly one of trainability. Because newborn units are forward-active yet backward-starved, changing the activation function provides a direct way to modify their gradient pathway without changing insertion, selection, or optimizer-state handling. Many activation changes could test this hypothesis—e.g., leaky, smooth, randomized, or non-monotone variants. However, prior activation-control experiments identified *Rand. Smooth-Leaky*, a smoother randomized leaky activation designed to preserve non-zero gradient flow, as a strong plasticity-preserving choice [27]. Additional controls in App. D show that its benefits depend on the benchmark and structural operator, rather than improving all methods uniformly.

![](./images/7466532944091754543_3.png)

Figure 4: Early newborn integration predicts adaptive-cycle quality, more than final ticket quality. Panels (a)–(b) relate early post-birth parity to Cycle-TAA. Early parity is the average log-ratio between newborn and previously active units over the early post-birth window; values closer to 0 indicate closer parity. Large labeled markers denote method means across compactness; lighter points show individual compactness/seed-level observations. (a) Higher early activation parity is associated with higher Cycle-TAA. (b) Gradient parity shows the clearest association with Cycle-TAA, supporting the view that backward credit assignment is the main integration bottleneck. (c) Winning-Ticket Acc. is not monotonically aligned with Cycle-TAA: methods can differ substantially during the structural-editing process while producing final masks with similar retrained endpoint quality.

Synthesis of interventions. Figure 4 shows that early newborn integration predicts adaptive-cycle quality more clearly than final ticket quality. Methods closer to parity achieve higher Cycle-TAA, especially for gradient parity (panel (b), Spearman $\rho = 0.88$; activation parity in panel (a), $\rho = 0.83$), supporting the view that backward signal is the main newborn-integration bottleneck. GRADMAX and RAND. SMOOTH-LEAKY occupy the higher-integration, higher-Cycle-TAA regime, whereas TWO-SPEED and NET2WIDER provide weaker or less direct relief. However, retrained-mask ACC does not follow the same ordering (panel (c)), showing that better edit-time dynamics do not necessarily imply a better final sparse mask. Thus, these interventions primarily diagnose and improve process-level integration rather than guaranteeing better final sparse architectures.

## 6 Continual Learning: Sequential Accumulation vs. Repeated-Shift Plasticity

So far, we have studied structural edits in a controlled supervised setting and identified a consistent pattern: when new units are introduced into a mature network, they begin training at a disadvantage, and the utility of growth depends not only on *which* units are added, but also on *how* they are inserted and how quickly they integrate afterward. This question becomes especially relevant in continual learning, where *structural plasticity* must operate under repeated distributional or semantic shifts without resetting the model. Therefore, we consider two settings: (i) a classical class-incremental benchmark, which asks whether growth helps under sequential accumulation, and (ii) plasticity-focused benchmarks, which ask whether growth helps when the central problem is maintaining the *ability to keep learning under repeated shift* [9].

Growth under sequential class accumulation. We first consider class-incremental CIFAR-100, where task boundaries are known during training but task identity is not provided at test time; all methods use a shared classifier head. At each boundary, the model either remains dense or undergoes a structural edit before continuing on the next class subset. Compared with the repeated-shift settings below, this regime gives each structural edit substantially more optimization time before the next shift, which is important because newborn units require time to integrate. As shown in Fig. 5, vanilla GROW remains weak, but integration-friendly growth variants become competitive: in particular, GROW + RAND. SMOOTH-LEAKY outperforms DENSE, PRUNE, and the other growth variants. Thus, sequential accumulation suggests that growth can help when the birth event is made trainable and the learner has enough time to absorb the added capacity.

Growth under repeated non-stationary shifts. Class-incremental CIFAR-100 is a useful reference point, but it does not by itself reveal whether the benefit of growth comes from improved continual *plasticity* or simply from changing capacity and interference. To investigate that distinction, we evaluate on benchmarks designed specifically to expose repeated-shift degradation in learnability. Following Kumar et al. [23], we consider a set of supervised-continual-learning image-classification benchmarks spanning both input-distribution and concept shift: **Permuted MNIST** [16] applies a fixed random pixel permutation to a shared subset for each task; **Random Label MNIST** and **Random Label CIFAR** [28] assign random labels to a fixed subset to encourage memorization; **CIFAR 5+1** draws and alternates hard (5 classes) and easy (single class) tasks from CIFAR-100; and **Continual ImageNet** [9, 36] performs a task-binary classification over two ImageNet classes which do not repeat across tasks, ensuring non-overlapping class exposure and clearer measurement of plasticity over time. These benchmarks are relevant because they stress *plasticity loss* rather than only forgetting and help with the question: *does growth help when the central problem is precisely the loss of ability to learn under repeated shift?*

Interpretation. Figure 5 reinforces the time-scale view of growth. PRUNE is the most reliable structural baseline because it preserves mature capacity, whereas GROW must repeatedly integrate newborn units into an already trained representation. Vanilla GROW is not uniformly ineffective—it can match or exceed DENSE in some settings, such as 5+1 CIFAR—but its gains are less stable than

![](./images/7466532944091754543_4.png)

Figure 5: Repeated-shift benchmarks favor pruning, while integration-friendly growth is the most reliable growth variant. Across six CL benchmarks, Prune is the strongest or near-strongest structural baseline in most rapid-shift settings, consistent with the advantage of preserving mature capacity. Among growth-family methods, Grow + Rand. Smooth-Leaky is the most robust: it consistently improves over Grow, narrows the gap to Prune, and becomes strongest on Split-CIFAR100, where structural events are separated by more optimization time. Other birth interventions provide less consistent gains, indicating that improvements in controlled insertion diagnostics do not translate into broad continual-plasticity gains.

pruning. Among growth-family methods, Grow + Rand. Smooth-Leaky transfers most reliably despite requiring only a drop-in activation change: it improves over vanilla Grow, narrows much of the gap to Prune, and becomes strongest on Split-CIFAR100, where added units have more time to integrate. Moment Transplant and GradMax sometimes help, but their gains are less consistent; TwoSpeed and Net2Wider are weaker in this suite. These results are consistent with Fig. 4. Thus, growth can be a viable alternative to pruning, but its success depends more strongly than pruning on whether newborn units can stabilize quickly enough before the next shift.

## 7 Conclusion & Future Work

Conclusion. Pruning and growth are the two basic operators of structural plasticity, but this paper shows that they are not optimization-symmetric. Under matched sparsity and compute budgets, the apparent disadvantage of growth arises less from a fundamentally weaker structural operator than from the conditions under which new capacity is introduced. Newborn units enter late into an already specialized network and are disadvantaged at birth: they can be forward-active yet remain weakly integrated into the backward credit-assignment pathway. This perspective helps explain why the Grow-Prune gap is weak in small MLPs, becomes visible in harder convolutional feature-learning regimes, and is expressed most clearly as a dissociation between procedure-level adaptation and final retrainable sparse-architecture quality. Across our intervention study, the strongest gains come not from treating growth as a purely architectural choice, but from improving the trainability and early integration of newborn units.

Future Work. These results suggest that structural adaptation should be evaluated not only by the architectures it produces, but also by the optimization compatibility of the edits used to produce them. More broadly, they point toward growth rules that decide not only when and where to edit a model, but also how to give new structure a realistic chance to integrate. In continual learning, this becomes especially important: growth is useful only if added capacity can stabilize before the next distribution shift, making insertion stability and integration time scale central design variables. A natural next step is to move beyond static grow heuristics toward policies that respond to newborn-integration signals, and to test these ideas in architectures where structural edits act more directly on learned representations and in task-agnostic continual-learning settings. We view this paper as a step toward treating architectural change as a first-class mechanism of adaptation, rather than as a secondary consequence of compression or expansion.

### References

[1] Asadi, K., Fakoor, R., and Sabach, S. (2023). Resetting the optimizer in deep rl: An empirical study. *Advances in Neural Information Processing Systems*, 36:72284-72324.

[2] Behrouz, A., Razaviyayn, M., Zhong, P., and Mirrokni, V. (2025). Nested learning: The illusion of deep learning architectures. *arXiv preprint arXiv:2512.24695*.

[3] Bellec, G., Kappel, D., Maass, W., and Legenstein, R. (2017). Deep rewiring: Training very sparse deep networks. *arXiv preprint arXiv:1711.05136*.

[4] Cai, Z., Sener, O., and Koltun, V. (2021). Online continual learning with natural distribution shifts: An empirical study with visual data. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 8281-8290.

[5] Chen, T., Goodfellow, I., and Shlens, J. (2016). Net2net: Accelerating learning via knowledge transfer. In *ICLR*.

[6] Cheney, N., Schrimpf, M., and Kreiman, G. (2017). On the robustness of convolutional neural networks to internal architecture and weight perturbations. *arXiv preprint arXiv:1703.08245*.

[7] Dai, X., Yin, H., and Jha, N. K. (2019). Nest: A neural network synthesis tool based on a grow-and-prune paradigm. *IEEE Transactions on Computers*, 68(10):1487-1497.

[8] Dettmers, T. and Zettlemoyer, L. (2019). Sparse networks from scratch: Faster training without losing performance. *arXiv preprint arXiv:1907.04840*.

[9] Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., and Sutton, R. S. (2024). Loss of plasticity in deep continual learning. *Nature*, 632:768-774.

[10] Evci, U., Gale, T., Menick, J., Sampedro, P., Lorch, E., and Sohl-Dickstein, J. (2020). Rigging the lottery: Making all tickets winners. In *NeurIPS*.

[11] Evci, U., van Merrienboer, B., Unterthiner, T., Pedregosa, F., and Vladymyrov, M. (2022). Gradmax: Growing neural networks using gradient information. In *International Conference on Learning Representations*.

[12] Fernando, C., Banarse, D., Blundell, C., Zwols, Y., Ha, D., Rusu, A. A., Pritzel, A., and Wierstra, D. (2017). Pathnet: Evolution channels gradient descent in super neural networks. *arXiv preprint arXiv:1701.08734*.

[13] Frankle, J. and Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv preprint arXiv:1803.03635*.

[14] Frankle, J., Dziugaite, G. K., Roy, D. M., and Carbin, M. (2019). Stabilizing the lottery ticket hypothesis. *arXiv: Learning*.

[15] Ghunaim, Y., Bibi, A., Alhamoud, K., Alfarra, M., Al Kader Hammoud, H. A., Prabhu, A., Torr, P. H., and Ghanem, B. (2023). Real-time evaluation in online continual learning: A new hope. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11888-11897.

[16] Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., and Bengio, Y. (2013). An empirical investigation of catastrophic forgetting in gradient-based neural networks. *arXiv preprint arXiv:1312.6211*.

[17] Gordon, A., Eban, E., Nachum, O., Chen, B., Wu, H., Yang, T.-J., and Choi, E. (2018). Morphnet: Fast & simple resource-constrained structure learning of deep networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1586–1595.

[18] Han, S., Mao, H., and Dally, W. J. (2015a). Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *arXiv preprint arXiv:1510.00149*.

[19] Han, S., Pool, J., Tran, J., and Dally, W. (2015b). Learning both weights and connections for efficient neural network. *Advances in neural information processing systems*, 28.

[20] Kang, H., Mina, R. J. L., Madjid, S. R. H., Yoon, J., Hasegawa-Johnson, M., Hwang, S. J., and Yoo, C. D. (2022). Forget-free continual learning with winning subnetworks. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S., editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 10734–10750. PMLR.

[21] Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[22] Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. In *University of Toronto Technical Report*.

[23] Kumar, S., Marklund, H., and Van Roy, B. (2023). Maintaining plasticity in continual learning via regenerative regularization. *arXiv preprint arXiv:2308.11958*.

[24] Lasby, M., Golubeva, A., Evci, U., Nica, M., and Ioannou, Y. (2023). Dynamic sparse training with structured sparsity. *arXiv preprint arXiv:2305.02299*.

[25] Lecun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324.

[26] Li, X., Zhou, Y., Wu, T., Socher, R., and Xiong, C. (2019). Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In *International conference on machine learning*, pages 3925–3934. PMLR.

[27] Lillo, L. and Cheney, N. (2025). Activation function design sustains plasticity in continual learning. *arXiv preprint arXiv:2509.22562*.

[28] Lyle, C., Zheng, Z., Nikishin, E., Pires, B. A., Pascanu, R., and Dabney, W. (2023). Understanding plasticity in neural networks. In *International Conference on Machine Learning*, pages 23190–23211. PMLR.

[29] Mallya, A., Davis, D., and Lazebnik, S. (2018). Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In *Proceedings of the European conference on computer vision (ECCV)*, pages 67–82.

[30] Mallya, A. and Lazebnik, S. (2018). Packnet: Adding multiple tasks to a single network by iterative pruning. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, pages 7765–7773.

[31] Miconi, T. (2016). Neural networks with differentiable structure. *arXiv preprint arXiv:1606.06216*.

[32] Mocanu, D. C. et al. (2018). Scalable training of artificial neural networks with adaptive sparse connectivity. In *AAAI*.

[33] Mosbach, M., Andriushchenko, M., and Klakow, D. (2020). On the stability of fine-tuning bert: Misconceptions, explanations, and strong baselines. arXiv preprint arXiv:2006.04884.

[34] Prabhu, A., Cai, Z., Dokania, P., Torr, P., Koltun, V., and Sener, O. (2023). Online continual learning without the storage constraint. arXiv preprint arXiv:2305.09253.

[35] Reddi, S. J., Kale, S., and Kumar, S. (2019). On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237.

[36] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211-252.

[37] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Hadsell, R. (2016). Progressive neural networks. arXiv preprint arXiv:1606.04671.

[38] Wei, T., Wang, C., Rui, Y., and Chen, C. W. (2016). Network morphism. In Proceedings of the 33rd International Conference on Machine Learning (ICML).

[39] Wu, L., Liu, B., Stone, P., and Liu, Q. (2020). Firefly neural architecture descent: a general approach for growing neural networks. Advances in neural information processing systems, 33:22373-22383.

[40] Wu, L., Wang, D., and Liu, Q. (2019). Splitting steepest descent for growing neural architectures. Advances in neural information processing systems, 32.

[41] Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-mnist: a novel image dataset for bench- marking machine learning algorithms. arXiv preprint arXiv:1708.07747.

[42] Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. (2020). On layer normalization in the transformer architecture. In International conference on machine learning, pages 10524-10533. PMLR.

[43] Yang, L., Lin, S., Zhang, J., and Fan, D. (2021). Grown: Grow only when necessary for continual learning. arXiv preprint arXiv:2110.00908.

[44] Yoon, J., Yang, E., Lee, J., and Hwang, S. J. (2017). Lifelong learning with dynamically expand- able networks. arXiv preprint arXiv:1708.01547.

[45] You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., and Hsieh, C.-J. (2019). Large batch optimization for deep learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962.

[46] Yuan, X., Savarese, P., and Maire, M. (2023). Accelerated training via incrementally growing neural networks using variance transfer and learning rate adaptation. Advances in Neural Information Processing Systems, 36:16673-16692.

[47] Zhao, Y., Saxena, D., Cao, J., Liu, X., and Song, C. (2024). Overcoming growth-induced forgetting in task-agnostic continual learning. arXiv preprint arXiv:2408.10566.

[48] Zhuang, J., Tang, T., Ding, Y., Tatikonda, S. C., Dvornek, N., Papademetris, X., and Duncan, J. (2020). Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. Advances in neural information processing systems, 33:18795-18806.

# Submission Checklist

1. For all authors…

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes] The abstract and introduction accurately reflect the paper's contributions and scope.

(b) Did you describe the limitations of your work? [Yes] We discuss limitations, including the restricted architectural setting, benchmark scope, and focus on controlled structural-edit mechanisms.

(c) Did you discuss any potential negative societal impacts of your work? [Yes] The work is methodological and does not introduce deployment-specific risks beyond standard risks of improving adaptive machine-learning systems.

(d) Did you read the ethics review guidelines and ensure that your paper conforms to them? (see https://2022.automl.cc/ethics-accessibility/) [Yes] We read the ethics review guidelines and ensured that the paper conforms to them.

2. If you ran experiments…

(a) Did you use the same evaluation protocol for all methods being compared (e.g., same benchmarks, data (sub)sets, available resources, etc.)? [Yes] All compared methods use the same datasets, architectures, training budgets, evaluation checkpoints, and compactness targets unless explicitly stated.

(b) Did you specify all the necessary details of your evaluation (e.g., data splits, pre-processing, search spaces, hyperparameter tuning details and results, etc.)? [Yes] We specify datasets, architectures, preprocessing, training schedules, compactness levels, hyperparameters, and method-specific details.

(c) Did you repeat your experiments (e.g., across multiple random seeds or splits) to account for the impact of randomness in your methods or data? [Yes] Experiments are repeated across multiple random seeds.

(d) Did you report the uncertainty of your results (e.g., the standard error across random seeds or splits)? [Yes] We report uncertainty across seeds using confidence intervals or standard errors where appropriate.

(e) Did you report the statistical significance of your results? [Yes] We report statistical comparisons where they are used to support the main claims.

(f) Did you use enough repetitions, datasets, and/or benchmarks to support your claims? [Yes] The claims are supported using multiple datasets, compactness levels, seeds, and ablations.

(g) Did you compare performance over time and describe how you selected the maximum runtime? [Yes] We compare performance over training cycles and use a fixed maximum training budget across methods.

(h) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] We will make that information available after the paper acceptance to avoid any information that could reveal public author's identity.

(i) Did you run ablation studies to assess the impact of different components of your approach? [Yes] We include ablations isolating growth cycles, insertion mechanisms, optimizer interventions, and activation-function effects.


3. With respect to the code used to obtain your results...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results, including all dependencies (e.g., `requirements.txt` with explicit versions), random seeds, an instructive README with installation instructions, and execution commands (either in the supplemental material or as a URL)? [Yes] Code, dependencies, random seeds, README instructions, and execution commands are included in the supplemental material and in the anonymized linked repository.

(b) Did you include a minimal example to replicate results on a small subset of the experiments or on toy data? [Yes] A reduced example is included to check the main pipeline on a small subset of experiments.

(c) Did you ensure sufficient code quality and documentation so that someone else can execute and understand your code? [Yes] The code is documented and organized to allow reproduction and inspection of the main experiments.

(d) Did you include the raw results of running your experiments with the given code, data, and instructions? [No] Raw experimental outputs are not included with the code release due to size limitations.

(e) Did you include the code, additional data, and instructions needed to generate the figures and tables in your paper based on the raw results? [Yes] Plotting scripts and instructions are included to regenerate the paper figures and tables from raw results.

4. If you used existing assets (e.g., code, data, models)...

(a) Did you cite the creators of used assets? [Yes] We cite the creators of all datasets, algorithms, and software assets used.

(b) Did you discuss whether and how consent was obtained from people whose data you're using/curating if the license requires it? [N/A] We use standard public benchmark datasets and do not curate human-subject data requiring consent.

(c) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [Yes] We discuss that the datasets used are standard public benchmarks and do not contain personally identifiable information to our knowledge.

5. If you created/released new assets (e.g., code, data, models)...

(a) Did you mention the license of the new assets (e.g., as part of your code submission)? [Yes] The license for released code and assets is specified.

(b) Did you include the new assets either in the supplemental material or as a URL (to, e.g., GitHub or Hugging Face)? [Yes] The released assets are included in the supplemental material and provided through an anonymized repository URL.

6. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [No] No crowdsourcing or human-subject experiments were conducted.

(b) Did you describe any potential participant risks, with links to institutional review board (IRB) approvals, if applicable? [No] No crowdsourcing or human-subject experiments were conducted.

(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [No] No crowdsourcing or human-subject experiments were conducted.

7. If you included theoretical results…

(a) Did you state the full set of assumptions of all theoretical results? [No] The paper does not present theoretical results.

(b) Did you include complete proofs of all theoretical results? [No] The paper does not present theoretical results.

## A Datasets, Benchmarks and Hyperparameters

### A.1 Datasets and Benchmarks

We evaluate structural adaptation under three data regimes: (i) stationary i.i.d. supervised learning, (ii) non-stationary single-head class-incremental streams, and (iii) continual-learning benchmarks designed to stress plasticity under repeated shift. In all cases, we use the standard train/test splits provided with each dataset unless otherwise specified, and the test set is used only for evaluation.

### A.1.1 Independently and identically distributed (i.i.d.)

MNIST. MNIST is treated as a stationary 10-class classification problem. We train on the full training set and evaluate on the standard MNIST test set. Each run uses a fixed budget of 5 GROW or PRUNE cycles, with 20 epochs per cycle. Winning-ticket retraining runs for 100 epochs to match the total cycle-training budget.

CIFAR-100. CIFAR-100 is treated as a stationary 100-class classification problem. We train on the full training set and evaluate on the standard CIFAR-100 test set. Each run uses 200 epochs total for dense training and winning-ticket retraining. Cycle training is divided into 5 GROW or PRUNE cycles, with 40 epochs per cycle. This setting does not use experience replay.

CIFAR-10. CIFAR-10 is treated as a stationary 10-class classification problem. We train and evaluate under the same protocol used for CIFAR-100. CIFAR-10 results are reported in App. B.6.

### A.1.2 Class-Incremental Continual Learning.

Class-incremental Split-MNIST and Split-FashionMNIST. MNIST and FashionMNIST each contain 10 classes. We construct a stream of $K = 5$ tasks by partitioning the classes into five disjoint class pairs. We use a single shared 10-way classifier head throughout training, with no task-id routing and no multi-head evaluation. At task $t \in \{1, \dots, K\}$, training uses only the two classes assigned to task $t$, while evaluation uses the cumulative test set containing all classes observed up to task $t$. Unless otherwise noted, the class-pair ordering is randomized per run using a fixed seed, and the same ordering is shared across methods within each seed.

Experience replay (TinyER). To avoid forgetting-dominated collapse in the single-head class-incremental setting, we optionally use experience replay. The replay buffer stores up to $M = 50$ examples per class, with a maximum total size of 200 examples. During training on tasks $t > 1$, each mini-batch mixes current-task samples with replay samples. After finishing task $t$, we add up to $M$ examples for each newly observed class. Unless explicitly stated as "no replay," Split-MNIST and Split-FashionMNIST use TinyER with a replay fraction of 0.5 and a fixed budget of 20 epochs per task (100 epochs total across 5 tasks), aligned with the structural-edit cycles.

Split-CIFAR100 (class-incremental). We also evaluate class-incremental Split-CIFAR100 as a sequential accumulation benchmark. CIFAR-100 is partitioned into a sequence of disjoint class-incremental tasks, and the learner is trained without task-identity information at test time. As in the other single-head stream settings, we use a shared classifier head throughout training and evaluate cumulatively over all classes observed so far. At each task boundary, the model either remains dense or undergoes a structural edit before continuing optimization on the new task. Results for this setting are reported in the main text continual-learning section.

### A.1.3 Plasticity-stressing continual-learning benchmarks.
Following Kumar et al. [23], we evaluate five supervised continual image-classification benchmarks spanning two shift types: input-distribution shift (Permuted MNIST, 5+1 CIFAR, and Continual ImageNet) and concept shift (Random Label MNIST and Random Label CIFAR). Across all settings, training proceeds as a sequence of tasks without task-identity signals: the model is never told when a task switch occurs. Within each

task, the learner receives mini-batches for a fixed duration and is updated incrementally with cross-entropy on the arriving batches. Summary hyperparameters are reported in Table 2.

Permuted MNIST. Permuted MNIST [16] is used as an input-shift benchmark. We first sample a fixed subset of 10,000 images from the MNIST training set. Each task is defined by drawing a new fixed random permutation over pixel indices and applying it to every image in the subset. The permutation remains constant within a task and is independent across tasks. Each task presents exactly one pass over its 10,000 permuted images in mini-batches of size 16, after which the next task begins with a new permutation. We train for 500 tasks in total.

Random Label MNIST. Random Label MNIST [28] is used as a concept-shift benchmark. We fix a subset of 1,200 MNIST images once, and for each task generate a fresh random label for every image in the subset. The inputs are unchanged across tasks, but the input-label mapping changes completely. To encourage memorization under an arbitrary target function, the model is trained for 400 epochs per task with batch size 16. After each task, a new independent random labeling is sampled. We run 50 tasks in sequence.

Random Label CIFAR. Random Label CIFAR follows the same protocol as Random Label MNIST, but uses images drawn from CIFAR-10. We again fix a subset of 1,200 images, reassign random labels independently for each task, and train for 400 epochs per task with batch size 16 over 50 tasks.

5+1 CIFAR. 5+1 CIFAR is an input-shift benchmark with alternating task difficulty. Tasks are constructed from CIFAR-100 and alternate between hard tasks containing 5 classes (2,500 images total, 500 per class) and easy tasks containing a single class (500 images). Classes do not repeat across the sequence. Each task lasts 780 parameter-update steps. With batch size 32, this corresponds to approximately 10 epochs on hard tasks and approximately 50 epochs on easy tasks. We report performance on the hard tasks only, since single-class tasks are near ceiling for all methods.

Continual ImageNet. Continual ImageNet [9, 36] is used as an input-shift benchmark. Each task is a binary classification problem between two distinct ImageNet classes. For every task, we draw 1,200 images total (600 per class) and downsample them to 32×32, following Dohare et al. [9], to reduce compute while preserving semantic variability. Classes do not repeat across tasks. We train for 10 epochs per task with batch size 100 and report task accuracy.

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Per-Task Data Size</th>
      <th>Batch</th>
      <th>Epochs</th>
      <th>Timesteps</th>
      <th># Tasks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Permuted MNIST</td>
      <td>10,000 images</td>
      <td>16</td>
      <td>1</td>
      <td>625</td>
      <td>500</td>
    </tr>
    <tr>
      <td>Random Label MNIST</td>
      <td>1,200 images</td>
      <td>16</td>
      <td>400</td>
      <td>30,000</td>
      <td>50</td>
    </tr>
    <tr>
      <td>Random Label CIFAR</td>
      <td>1,200 images</td>
      <td>16</td>
      <td>400</td>
      <td>30,000</td>
      <td>50</td>
    </tr>
    <tr>
      <td>5+1 CIFAR</td>
      <td>Hard: 2,500 images (5 classes, 500/class)</td>
      <td>32</td>
      <td>Hard: $\approx 10$</td>
      <td>780</td>
      <td>15</td>
    </tr>
    <tr>
      <td></td>
      <td>Easy: 500 images (1 class)</td>
      <td></td>
      <td>Easy: $\approx 50$</td>
      <td></td>
      <td>15</td>
    </tr>
    <tr>
      <td>Continual ImageNet</td>
      <td>1,200 images/task (600/class)</td>
      <td>100</td>
      <td>10</td>
      <td>120</td>
      <td>500</td>
    </tr>
  </tbody>
</table>

Table 2: Hyperparameters and schedule per benchmark. Timesteps denote parameter-update steps (i.e., mini-batches) within a task. For 5+1 CIFAR, a fixed timestep budget per task implies approximate epochs depending on data size.

Notes. (i) In 5+1 CIFAR, classes do not repeat across tasks; tasks alternate easy/hard. 780 timesteps $\approx$ 10 epochs on the hard set (since $2,500/32 \approx 78.125$ batches/epoch) and $\approx 50$ epochs on the easy set (since $500/32 \approx 15.625$). (ii) In Continual ImageNet, images are downsampled to 32×32 to reduce compute; classes do not repeat across tasks. (iii) Timesteps are computed as the number of mini-batches per task.

### A.2 Hyperparameters, Optimizers and Learning Rates

Within each dataset–architecture setting, we hold the optimizer choice and training schedule fixed across Dense, Grow, and Prune to avoid optimizer dynamics dominate comparisons when the effective parameterization changes over time. For each dataset–experiment setting, we sweep $\eta \in \{0.1, 0.01, 0.001, 0.0001\}$ for all relevant methods and use the best-performing learning rate when reporting results for that setting. The only method-specific difference is the mask update rule and the resulting active set of units (see App. A.4). Unless otherwise stated, we use ReLU activations throughout. All Conv2d and Linear layers are initialized with Kaiming uniform initialization.

Learning-rate schedule. For all experiments in Sections 4 and 5 we apply cosine annealing over the full training horizon with $T_{\text{max}} = N_{\text{epochs}}$ and $\eta_{\text{min}} = 0$, so the learning rate decays smoothly from its initial value to zero over 200 epochs.

### A.3 Performance Metrics: ACC, trajectory-average accuracy (TAA), and TAOA

We use the same metric names and evaluation conventions as in the main text. Our primary evaluation index is defined by Grow/Prune checkpoints: in class-incremental streams these checkpoints coincide with task boundaries, while in i.i.d. settings they simply mark successive structural-edit cycles.

Checkpoint indexing. Let $t \in \{1, \dots, T\}$ index evaluation checkpoints along training. In our implementation, checkpoints occur at the end of each Grow/Prune cycle. In class-incremental streams, this aligns $t$ with task time; in i.i.d. settings, $t$ indexes successive structural-edit events.

Class-incremental settings. When tasks exist, let $K_t$ be the number of tasks encountered up to checkpoint $t$, and let $\text{Acc}_{t,k}$ denote test accuracy on task $k \in \{1, \dots, K_t\}$ measured at checkpoint $t$. We define cumulative accuracy at checkpoint $t$ as

$$
\text{Acc}_t^{\text{cum}} = \frac{1}{K_t} \sum_{k=1}^{K_t} \text{Acc}_{t,k}. \tag{1}
$$

We then report

$$
\text{ACC} = \text{Acc}_T^{\text{cum}}, \tag{2}
$$

$$
\text{TAA} = \frac{1}{T} \sum_{t=1}^{T} \text{Acc}_t^{\text{cum}}. \tag{3}
$$

Thus, ACC summarizes end-of-stream performance, while TAA summarizes performance over the full trajectory.

I.i.d. settings. When there are no tasks, let $\text{Acc}_t$ be the test accuracy at checkpoint $t$. We report

$$
\text{ACC} = \text{Acc}_T, \tag{4}
$$

$$
\text{TAA} = \frac{1}{T} \sum_{t=1}^{T} \text{Acc}_t. \tag{5}
$$

In this case, ACC is the standard final test accuracy and TAA is the average test accuracy over the checkpoint-defined training trajectory.

Cycle vs. Winning-Ticket evaluation. Both *Cycle* and *Winning-Ticket* results report ACC and TAA using the definitions above. Cycle metrics are measured during the Grow/Prune procedure while the mask changes across checkpoints. Winning-Ticket metrics are measured after freezing the discovered mask, reinitializing the model, and retraining it from scratch under the same data stream and replay setting, following standard Lottery Ticket evaluation [13].

Total Average Online Accuracy (TAOA). On the continual-learning benchmarks, we additionally use Total Average Online Accuracy (TAOA), following prior online continual-learning work [4, 15, 23, 34]. Unlike ACC and TAA, which are checkpoint-based, TAOA aggregates online accuracy over all mini-batches seen so far. Let $B_{\leq T} = \sum_{i=1}^T M_i$ be the total number of processed mini-batches up to task $T$, and let $a_t$ denote online accuracy at global batch index $t$. We define

$$
\mathrm{TAOA}_{\leq T} = \frac{1}{B_{\leq T}} \sum_{t=0}^{B_{\leq T}-1} a_t. \tag{6}
$$

If all tasks have equal length $M_i \equiv M$, this reduces to

$$
\mathrm{TAOA}_{\leq T} = \frac{1}{MT} \sum_{t=0}^{MT-1} a_t. \tag{7}
$$

We use TAOA in Section 6 to capture how quickly the agent learns the current task (plasticity) and distinguish it from the checkpoint-based ACC and TAA reported in the rest of the main paper.

### A.4 Neural Network Architectures

All methods (Dense, Grow, Prune) share the same underlying parameterization and differ only in the binary unit masks that determine which hidden units are active during forward and backward passes. This ensures that comparisons isolate the effect of structural adaptation rather than changes in the base architecture.

Unit-wise masking. For a hidden layer $\ell$ with pre-activation

$$
z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}, \tag{8}
$$

and activation function $\phi(\cdot)$, we define

$$
h_{\mathrm{raw}}^{(\ell)} = \phi\left(z^{(\ell)}\right), \quad h^{(\ell)} = h_{\mathrm{raw}}^{(\ell)} \odot m^{(\ell)}, \tag{9}
$$

where $m^{(\ell)} \in \{0, 1\}^{d_\ell}$ is a unit-wise binary mask and $\odot$ denotes element-wise multiplication. Mask entries set to zero fully deactivate the corresponding hidden units, affecting both activations and gradients. Masks are stored as non-trainable buffers and are therefore serialized with the model state.

#### A.4.1 MLP for MNIST and FashionMNIST.
Our MLP backbone is a 3-layer fully connected network:

$$
\mathrm{fc1} :\ 784 \rightarrow H, \tag{10}
$$

$$
\mathrm{fc2} :\ H \rightarrow H, \tag{11}
$$

$$
\mathrm{fc3} :\ H \rightarrow 10, \tag{12}
$$

with two masked hidden layers (fc1, fc2) and an unmasked output layer (fc3). Unless otherwise noted, $H=256$.

Dense training corresponds to $m^{(1)}=1$ and $m^{(2)}=1$ throughout. Growth begins from a small active fraction (default 10%) and expands masks over time by flipping selected zeros to ones. Pruning starts fully active and removes units by setting mask entries to zero.

#### A.4.2 ConvNet for CIFAR-10 & CIFAR-100.
For both CIFAR-10 and CIFAR-100 we use a fully active convolutional trunk and a growable/prunable MLP head.

Convolutional trunk. The trunk is

$$
\text{conv1 }: 3 \to 32,\ k=3,\ p=1 \to \phi \to \text{maxpool}(2), \tag{13}
$$

$$
\text{conv2 }: 32 \to 64,\ k=3,\ p=1 \to \phi \to \text{maxpool}(2), \tag{14}
$$

followed by flattening to $8 \times 8 \times 64 = 4096$ features. No structural adaptation is applied in the convolutional trunk.

Masked fully connected head. The head has four fully connected layers:

$$
\text{fc1 }: 4096 \to H_1 \to \phi \to \odot m^{(1)}, \tag{15}
$$

$$
\text{fc2 }: H_1 \to H_2 \to \phi \to \odot m^{(2)}, \tag{16}
$$

$$
\text{fc3 }: H_2 \to H_3 \to \phi \to \odot m^{(3)}, \tag{17}
$$

$$
\text{fc4 }: H_3 \to C, \tag{18}
$$

where $C$ is the number of output classes. By default, $(H_1, H_2, H_3) = (512, 512, 256)$. In all cases, structural adaptation is confined to the fully connected head.

## A.5 Compactness and Sparsity

Our structural interventions operate through unit-wise binary masks applied to selected hidden layers. Therefore, we define compactness budgets in terms of active units and match these budgets across Dense, Grow, and Prune.

Layer compactness. For a masked layer $\ell$ with $d_\ell$ units and mask $m^{(\ell)} \in \{0,1\}^{d_\ell}$, the number of active units is

$$
a_\ell = \|m^{(\ell)}\|_0 = \sum_{j=1}^{d_\ell} m_j^{(\ell)}, \tag{19}
$$

and the layer compactness is

$$
c_\ell = \frac{a_\ell}{d_\ell} \in (0,1]. \tag{20}
$$

Global compactness. Architectural final compactness target is a global value $c \in (0,1]$ specifying the fraction of weights to keep or activate across growable and prunable layers. We implement this by allocating a target kept-weight budget across layers and converting that budget into integer unit targets. Thus, $c$ should be interpreted primarily as a budget in weight space rather than as equal unit fractions in every layer.

Matched sparsity budget (operational definition). A run at global compactness $c$ is budget-matched if the final masks satisfy the same per-layer integer unit targets $\{u_\ell^\star\}$ (up to rounding/reconciliation) across methods. This implies identical active-unit counts in each masked layer at the final architecture, and therefore matches the effective structured sparsity (and capacity) within the subnetwork.

## A.6 Structural Adaptation Procedure

Both Grow and Prune operate on the same masked backbone (App. A.4) and update masks over $T$ cycles. Each cycle consists of (i) selecting units to activate or deactivate, (ii) updating the masks and any associated bookkeeping, and (iii) training for a fixed budget before the next cycle.

### A.6.1 Grow.

Selection rule. At cycle $t$, for each masked layer $\ell$, we compute a score $s_j^{(\ell)}$ for each currently inactive unit $j$ with $m_j^{(\ell)} = 0$. Then, we activate the top-$n_{t,\ell}$ inactive units in that layer.

Activation-frequency heuristic. Our default growth heuristic scores an inactive unit by the fraction of post-activation values exceeding a threshold $\tau$ (i.e. $\tau=0.05$). Given a mini-batch $B$, we define

$$
s_{j}^{\text{Act}}=
\begin{cases}
\frac{1}{|B|}\sum_{x\in B}\mathbf{1}\{A_j(x)>\tau\}, & \text{fully connected}, \\
\frac{1}{|B|}\sum_{x\in B}\left(\frac{1}{HW}\sum_{u\in[H]\times[W]}\mathbf{1}\{A_j(x,u)>\tau\}\right), & \text{convolutional}.
\end{cases}
\tag{21}
$$

Here $A_j$ denotes the post-activation value of unit or channel $j$. Intuitively, $s_{j}^{\text{Act}}$ estimates how often a currently inactive unit is meaningfully active on typical training inputs.

### A.6.2 Prune.

Selection rule. PRUNE starts from a fully active network and removes units over $T$ cycles until reaching the target per-layer unit counts. At cycle $t$, for each layer $\ell$ with current active count $a_\ell$ and target $a_\ell^\star$, the remaining number to remove is

$$
q_{t,\ell}=a_\ell - a_\ell^\star.
\tag{22}
$$

We prune

$$
k_{t,\ell}=\min\left(q_{t,\ell},\left\lceil\frac{q_{t,\ell}}{\max(1,T-t)}\right\rceil,a_\ell\right)
\tag{23}
$$

units from the active set by selecting the lowest-scoring units.

Magnitude score. Our default pruning score is the mean absolute weight magnitude per unit. For a Linear layer,

$$
s_{j}^{\text{Mag}}=\frac{1}{d_{\text{in}}}\sum_{i=1}^{d_{\text{in}}}|W_{j,i}|,
\tag{24}
$$

and for a Conv2d layer,

$$
s_{j}^{\text{Mag}}=\frac{1}{C_{\text{in}}k_Hk_W}\sum_{c,u,v}|W_{j,c,u,v}|.
\tag{25}
$$

We prune the smallest-magnitude units among the currently active set.

IMP rewind. Our default pruning procedure uses IMP-style rewinding. After updating the mask, we rewind surviving parameter slices to their initialization (or stored rewind snapshot) before retraining, using the current mask to select the surviving rows, filters, and input columns as needed. This isolates the effect of subnetwork selection from continued fine-tuning dynamics.

## B Additional Experimental Studies and Ablations

### B.1 Compactness-resolved MLP winning-ticket results

Table 3 expands the main-text MLP summary by reporting each compactness budget separately. The same conclusion holds at the per-budget level: across IID MNIST, Split-MNIST with Tiny ER, and Split-Fashion with Tiny ER, GROW and PRUNE produce closely matched winning-ticket final accuracy and TAA, with no stable ordering across compactness levels. The $\Delta_{\text{WT-C}}$ column further shows that the relation between the structural-edit trajectory and the retrained ticket endpoint differs by dataset: in IID MNIST, retraining generally gives slightly lower endpoints than the cycle procedure, whereas in the class-incremental settings the retrained tickets often outperform the cycle endpoints, reflecting the instability of the online structural-edit trajectory under continual accumulation. Overall, these compactness-resolved results support the main-text interpretation that small MLP settings are not sufficient to expose a robust GROW–PRUNE asymmetry.

<table>
<thead>
<tr>
<th>Method</th>
<th colspan="3">20%</th>
<th colspan="3">30%</th>
<th colspan="3">40%</th>
<th colspan="3">50%</th>
</tr>
<tr>
<th></th>
<th>WT Final</th>
<th>WT TAA</th>
<th>Δ<sub>WT-C</sub></th>
<th>WT Final</th>
<th>WT TAA</th>
<th>Δ<sub>WT-C</sub></th>
<th>WT Final</th>
<th>WT TAA</th>
<th>Δ<sub>WT-C</sub></th>
<th>WT Final</th>
<th>WT TAA</th>
<th>Δ<sub>WT-C</sub></th>
</tr>
<tr>
<th colspan="13">MNIST (IID) Dense (100%): WT Final 95.98 ± 0.09, WT TAA 93.13 ± 0.06</th>
</tr>
</thead>
<tbody>
<tr>
<td>Grow</td>
<td>95.39 ± 0.13</td>
<td>92.94 ± 0.13</td>
<td>−1.16 ± 0.17</td>
<td>95.96 ± 0.07</td>
<td>93.71 ± 0.07</td>
<td>−0.57 ± 0.10</td>
<td>96.19 ± 0.08</td>
<td>94.12 ± 0.07</td>
<td>−0.38 ± 0.11</td>
<td>96.40 ± 0.07</td>
<td>94.40 ± 0.06</td>
<td>−0.22 ± 0.22</td>
</tr>
<tr>
<td>Prune</td>
<td>95.33 ± 0.12</td>
<td>92.87 ± 0.12</td>
<td>−0.63 ± 0.20</td>
<td>95.87 ± 0.14</td>
<td>93.66 ± 0.11</td>
<td>−0.58 ± 0.15</td>
<td>96.16 ± 0.13</td>
<td>94.05 ± 0.10</td>
<td>−0.62 ± 0.14</td>
<td>96.39 ± 0.09</td>
<td>94.39 ± 0.08</td>
<td>−0.60 ± 0.10</td>
</tr>
<tr>
<td colspan="13">Split-MNIST (+Tiny ER) Dense (100%): WT Final 84.36 ± 0.75, WT TAA 90.44 ± 0.77</td>
</tr>
<tr>
<td>Grow</td>
<td>84.53 ± 0.77</td>
<td>90.19 ± 0.62</td>
<td>11.15 ± 1.66</td>
<td>84.47 ± 0.77</td>
<td>91.00 ± 0.52</td>
<td>11.93 ± 1.16</td>
<td>84.88 ± 1.09</td>
<td>91.30 ± 0.66</td>
<td>12.21 ± 1.54</td>
<td>85.81 ± 0.49</td>
<td>91.41 ± 0.75</td>
<td>10.85 ± 1.48</td>
</tr>
<tr>
<td>Prune</td>
<td>84.53 ± 0.72</td>
<td>90.33 ± 0.48</td>
<td>8.52 ± 2.14</td>
<td>85.14 ± 1.17</td>
<td>90.70 ± 0.84</td>
<td>7.92 ± 1.61</td>
<td>85.69 ± 0.71</td>
<td>90.98 ± 0.51</td>
<td>8.47 ± 1.51</td>
<td>85.20 ± 0.79</td>
<td>90.91 ± 0.36</td>
<td>7.29 ± 0.98</td>
</tr>
<tr>
<td colspan="13">Split-Fashion (+Tiny ER) Dense (100%): WT Final 75.01 ± 1.58, WT TAA 84.34 ± 1.53</td>
</tr>
<tr>
<td>Grow</td>
<td>74.88 ± 1.07</td>
<td>83.34 ± 2.09</td>
<td>8.13 ± 1.07</td>
<td>74.96 ± 1.66</td>
<td>85.84 ± 1.34</td>
<td>7.21 ± 0.92</td>
<td>75.70 ± 1.11</td>
<td>86.02 ± 1.11</td>
<td>8.49 ± 1.02</td>
<td>75.11 ± 2.03</td>
<td>85.02 ± 2.16</td>
<td>7.19 ± 2.12</td>
</tr>
<tr>
<td>Prune</td>
<td>75.53 ± 1.57</td>
<td>86.32 ± 0.76</td>
<td>5.01 ± 0.92</td>
<td>75.70 ± 1.52</td>
<td>85.72 ± 1.67</td>
<td>5.74 ± 2.15</td>
<td>75.76 ± 1.08</td>
<td>84.08 ± 2.21</td>
<td>4.92 ± 0.97</td>
<td>75.00 ± 1.48</td>
<td>84.92 ± 3.04</td>
<td>5.71 ± 1.54</td>
</tr>
</tbody>
</table>

Table 3: Compactness-resolved winning-ticket performance for the small MLP benchmarks. Entries are mean ± 95% CI over 10 seeds. For each compactness budget, we report Winning-Ticket final accuracy (WT Final), Winning-Ticket trajectory-average accuracy (WT TAA), and Δ<sub>WT-C</sub> = Final<sub>WT</sub> − Final<sub>Cycle</sub>. Negative Δ<sub>WT-C</sub> values indicate that the structural-edit cycle reached a higher endpoint than the retrained ticket, while positive values indicate that the frozen mask retrained from scratch to a higher endpoint than the cycle procedure achieved. Dense is invariant to compactness and is therefore reported once per dataset header.

## B.2 CIFAR-100: Additional Performance Results

Table 4 makes the CIFAR-100 cycle-vs.-ticket dissociation explicit at each compactness. This Table represents the same values shown in Fig. 1. During the adaptive structural process, GRow consistently achieves higher Cycle-ACC than PRUNE, with the gap increasing toward higher compactness. However, this advantage does not translate into a comparably stronger winning-ticket architecture: after retraining from scratch, GRow and PRUNE are nearly tied at 20–30% compactness, and PRUNE is slightly stronger in Winning-Ticket ACC at 40–50%. The within-method deltas reinforce this interpretation. GRow shows increasingly negative ΔACC as compactness rises, indicating that its strong cycle endpoint depends substantially on the adaptive path used to reach the mask. By contrast, PRUNE exhibits positive or near-zero ΔACC across compactness, showing that its discovered subnetworks retrain at least as well as, and often better than, their cycle endpoints suggest. Thus, the appendix table supports the main-text conclusion that on CIFAR-100 the dominant separation is between procedure-level adaptation and final retrainable architecture quality.

Table 5 shows that the cycle-level separation is statistically robust, whereas the winning-ticket separation is much weaker. Across all compactness levels, GRow vs. PRUNE is highly significant for Cycle-ACC ($p < 0.001$ throughout), confirming that the adaptive training trajectories of the two methods are genuinely different. In contrast, the same comparison is not significant for Winning-Ticket ACC at any compactness level, indicating that once the final masks are frozen and retrained from scratch, the apparent advantage largely disappears. The comparisons against DENSE follow the same pattern: GRow differs strongly from DENSE during the cycle and often also under retraining, whereas PRUNE is much closer to DENSE during cycle training but separates clearly under winning-ticket evaluation. Overall, the ACC p-values reinforce the main claim of the paper: the strongest and most reliable GRow–PRUNE difference on CIFAR-100 lies in the adaptive structural process itself, not in the final retrainable sparse architecture.

Table 6 shows that the cycle-level separation in trajectory quality is statistically reliable at low and intermediate compactness, but largely disappears under winning-ticket retraining. For Cycle-TAA, GRow vs. PRUNE is significant at 20%, 30%, and 40% compactness, confirming that the two methods induce genuinely different adaptation dynamics during structural editing. At 50%, however, the difference vanishes, consistent with the raw means being nearly identical. In contrast, the winning-ticket TAA comparison between GRow and PRUNE is not significant at any compactness, indicating that the trajectory-level separation does not survive retraining of the final

<table>
  <thead>
    <tr>
      <th rowspan="2">Comp. (%)</th>
      <th rowspan="2">Method</th>
      <th colspan="2">Cycle Eval.</th>
      <th colspan="2">Winning-ticket Eval.</th>
      <th colspan="2">Ticket – Cycle</th>
    </tr>
    <tr>
      <th>ACC</th>
      <th>TAA</th>
      <th>ACC</th>
      <th>TAA</th>
      <th>ΔACC</th>
      <th>ΔTAA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>Dense</td>
      <td>49.660±0.315</td>
      <td>48.062±0.284</td>
      <td>49.660±0.315</td>
      <td>45.998±0.262</td>
      <td>0.000</td>
      <td>−2.065</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Grow</td>
      <td>52.316±0.317</td>
      <td>47.288±0.269</td>
      <td>52.055±0.324</td>
      <td>46.257±0.188</td>
      <td>−0.261</td>
      <td>−1.031</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Prune</td>
      <td>47.911±0.644</td>
      <td>48.519±0.348</td>
      <td>52.057±0.283</td>
      <td>46.298±0.162</td>
      <td>+4.146</td>
      <td>−2.221</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Grow</td>
      <td>53.312±0.247</td>
      <td>47.871±0.374</td>
      <td>51.626±0.232</td>
      <td>46.909±0.220</td>
      <td>−1.686</td>
      <td>−0.963</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Prune</td>
      <td>49.549±0.325</td>
      <td>48.732±0.358</td>
      <td>51.615±0.280</td>
      <td>46.880±0.193</td>
      <td>+2.066</td>
      <td>−1.852</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Grow</td>
      <td>53.588±0.313</td>
      <td>48.338±0.291</td>
      <td>50.791±0.338</td>
      <td>46.886±0.239</td>
      <td>−2.797</td>
      <td>−1.452</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Prune</td>
      <td>49.963±0.481</td>
      <td>48.869±0.251</td>
      <td>51.205±0.415</td>
      <td>47.195±0.234</td>
      <td>+1.242</td>
      <td>−1.675</td>
    </tr>
    <tr>
      <td>50</td>
      <td>Grow</td>
      <td>54.338±0.221</td>
      <td>48.990±0.245</td>
      <td>50.260±0.257</td>
      <td>46.776±0.174</td>
      <td>−4.078</td>
      <td>−2.213</td>
    </tr>
    <tr>
      <td>50</td>
      <td>Prune</td>
      <td>50.227±0.548</td>
      <td>48.997±0.240</td>
      <td>50.320±0.415</td>
      <td>46.749±0.295</td>
      <td>+0.093</td>
      <td>−2.247</td>
    </tr>
  </tbody>
</table>

Table 4: CIFAR-100 (i.i.d.) ConvNet, ReLU, 200 epochs (SGD, $\eta$=0.1). Mean ± CI95 over $n=10$ seeds. We report cycle evaluation metrics (measured during structural adaptation) and winning-ticket evaluation metrics (mask frozen, weights reinitialized, retrained from scratch). Deltas are computed within method and compactness as $\Delta=(\text{ticket}-\text{cycle})$ (%). Bold highlights the higher value between Grow and Prune within each compactness for each column.

<table>
  <thead>
    <tr>
      <th rowspan="2">Comparison</th>
      <th colspan="2">20%</th>
      <th colspan="2">30%</th>
      <th colspan="2">40%</th>
      <th colspan="2">50%</th>
    </tr>
    <tr>
      <th>Cycle</th>
      <th>WT</th>
      <th>Cycle</th>
      <th>WT</th>
      <th>Cycle</th>
      <th>WT</th>
      <th>Cycle</th>
      <th>WT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Grow vs Dense</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0038</td>
    </tr>
    <tr>
      <td>Grow vs Prune</td>
      <td>0.0000</td>
      <td>0.9917</td>
      <td>0.0000</td>
      <td>0.9461</td>
      <td>0.0000</td>
      <td>0.0980</td>
      <td>0.0000</td>
      <td>0.7846</td>
    </tr>
    <tr>
      <td>Dense vs Prune</td>
      <td>0.0001</td>
      <td>0.0000</td>
      <td>0.5862</td>
      <td>0.0000</td>
      <td>0.2514</td>
      <td>0.0000</td>
      <td>0.0615</td>
      <td>0.0108</td>
    </tr>
  </tbody>
</table>

Table 5: Welch two-sample t-test p-values for final accuracy (ACC), comparing methods across compactness levels for Cycle and Winning-Ticket (WT) evaluations. Bold indicates statistical significance at $p<0.05$.

masks. Comparisons against Dense follow the same general pattern: Prune separates reliably from Dense in both cycle and winning-ticket TAA, whereas Grow is less consistently distinct during the cycle but often differs under retraining. Overall, the TAA p-values reinforce the main claim that on CIFAR-100 the most robust Grow–Prune difference lies in adaptive training dynamics rather than in the final retrainable sparse architecture.

<table>
  <thead>
    <tr>
      <th rowspan="2">Comparison</th>
      <th colspan="2">20%</th>
      <th colspan="2">30%</th>
      <th colspan="2">40%</th>
      <th colspan="2">50%</th>
    </tr>
    <tr>
      <th>Cycle</th>
      <th>WT</th>
      <th>Cycle</th>
      <th>WT</th>
      <th>Cycle</th>
      <th>WT</th>
      <th>Cycle</th>
      <th>WT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Grow vs Dense</td>
      <td>0.0003</td>
      <td>0.0876</td>
      <td>0.3701</td>
      <td>0.0000</td>
      <td>0.1425</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <td>Grow vs Prune</td>
      <td>0.0000</td>
      <td>0.7154</td>
      <td>0.0014</td>
      <td>0.8277</td>
      <td>0.0059</td>
      <td>0.0517</td>
      <td>0.9637</td>
      <td>0.8601</td>
    </tr>
    <tr>
      <td>Dense vs Prune</td>
      <td>0.0340</td>
      <td>0.0437</td>
      <td>0.0040</td>
      <td>0.0000</td>
      <td>0.0001</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0004</td>
    </tr>
  </tbody>
</table>

Table 6: Welch two-sample t-test p-values for TAA, comparing methods across compactness levels for Cycle and Winning-Ticket (WT) evaluations. Bold indicates statistical significance at $p<0.05$.

## B.3 CIFAR-100: Additional Mechanistic Analyses

### B.3.1 Absolute cohort diagnostics.
Structural plasticity changes who participates in the computation and who receives credit during optimization. Section 4.2 uses cohort-level signals around structural edits in two complementary ways: absolute cohort statistics as sanity checks, and parity (ratio or

log-ratio) statistics for the main mechanistic claims. Both are computed from the same per-cycle cohort snapshots, but they answer different questions. Absolute diagnostics ask whether newborn, kept, or pruned units exhibit nontrivial forward participation and learning signal; parity diagnostics ask how those cohorts compare relative to one another.

Unit-level cohorts and time-points. For GRow, at cycle $t$ we define the newborn cohort
$$
\mathcal{N}_t = \{\text{units activated at the Grow step of cycle } t\},
$$
and the incumbent cohort
$$
\mathcal{O}_t = \{\text{units already active before that Grow step}\}.
$$

For PRUNE, we define the kept and pruned cohorts
$$
\mathcal{K}_t = \{\text{units kept by the Prune decision at cycle } t\},
$$
and the pruned cohort
$$
\mathcal{P}_t = \{\text{units removed by the Prune decision at cycle } t\}.
$$

We use three time-points: (i) Post, the immediate snapshot after a structural edit and before further training; (ii) Exit, the end-of-cycle checkpoint at which prune decisions are made; and (iii) End, the end of the subsequent training segment.

Measured signals. For a layer $\ell$ and a unit $j$, we measure:

- Activation rate $\text{act}(j)$: the fraction of post-activation values exceeding a small threshold $\tau$ on a single mini-batch.
- Per-unit gradient magnitude $\text{grad}(j)$: the mean absolute pre-activation gradient $|\partial\mathcal{L}/\partial z_j|$ on a single mini-batch, where $z_j$ denotes the unit pre-activation.

For convolutional layers, activation rates are averaged over $(B,H,W)$; for linear layers, over $B$. We aggregate within cohort by averaging over units and seeds, and in the main-paper figures we additionally average across layers unless otherwise stated.

Absolute diagnostics. For GRow, we report newborn absolute vitality through the post-growth cohort means
$$
A_{\text{grow,act}}^{\text{abs}}(t, \ell) = \mathbb{E}_{j\in\mathcal{N}_t(\ell)}[\text{act}(j)], \quad A_{\text{grow,grad}}^{\text{abs}}(t, \ell) = \mathbb{E}_{j\in\mathcal{N}_t(\ell)}[\text{grad}(j)].
$$

These quantities serve as sanity checks: they ask whether newborn units are active at birth and whether they receive nonzero learning signal. For PRUNE, we analogously report absolute contrasts such as $\mathbb{E}[\text{act}]$ for kept and pruned cohorts at exit, together with post-prune changes in survivor activation.

### B.3.2 Parity and log-parity diagnostics.
Absolute activation rates and gradient magnitudes are useful for ruling out degenerate cohorts, but they are not sufficient for mechanistic claims about relative allocation. Because these quantities drift over training with loss scale, optimizer state, and representation maturity, we convert them into parity (ratio) or log-parity (log-ratio) statistics when comparing cohorts at the same snapshot. These quantities directly ask whether one cohort receives more forward participation or learning signal than another. In particular, they let us test the main question of Sec. 4.2: whether newborn units receive a fair share of activation and, more importantly, backward credit relative to previously active units. Log-parity is especially convenient because multiplicative advantages and disadvantages are symmetric around zero: a factor-$a$ advantage and a factor-$a$ disadvantage appear as equal-magnitude quantities with opposite sign.

Grow: newborn vs. incumbent parity at birth. At the post-growth snapshot of cycle $t$, we compute cohort means for newborns and incumbents and form

$$
R_{\text{grow,act}}(t, \ell) = \frac{\mathbb{E}_{j \in \mathcal{N}_t(\ell)}[\text{act}(j)]}{\mathbb{E}_{j \in \mathcal{O}_t(\ell)}[\text{act}(j)] + \varepsilon}, \quad R_{\text{grow,grad}}(t, \ell) = \frac{\mathbb{E}_{j \in \mathcal{N}_t(\ell)}[\text{grad}(j)]}{\mathbb{E}_{j \in \mathcal{O}_t(\ell)}[\text{grad}(j)] + \varepsilon},
$$

where $\varepsilon$ is a small constant for numerical stability. We typically visualize log-parity,

$$
\Delta_{\text{grow,act}}(t, \ell) = \log R_{\text{grow,act}}(t, \ell), \quad \Delta_{\text{grow,grad}}(t, \ell) = \log R_{\text{grow,grad}}(t, \ell).
$$

Parity corresponds to $R = 1$ (equivalently, $\log R = 0$). Negative log-parity indicates that newborns are disadvantaged relative to incumbents, while positive values indicate an advantage. Using parity rather than absolute curves controls for global drift because numerator and denominator are measured at the same time-point.

Prune: kept vs. pruned parity and survivor stability. At prune exit we analogously compute

$$
R_{\text{prune,act}}(t, \ell) = \frac{\mathbb{E}_{j \in \mathcal{K}_t(\ell)}[\text{act}(j)]}{\mathbb{E}_{j \in \mathcal{P}_t(\ell)}[\text{act}(j)] + \varepsilon}, \quad \Delta_{\text{prune,act}}(t, \ell) = \log R_{\text{prune,act}}(t, \ell).
$$

For survivor stability, we compare post-prune and end-of-cycle activation either through an additive change, $\mathbb{E}[\text{act}^{\text{post}} - \text{act}^{\text{end}}]$, or through log-parity,

$$
\log\left(\frac{\mathbb{E}[\text{act}^{\text{post}}]}{\mathbb{E}[\text{act}^{\text{end}}] + \varepsilon}\right),
$$

depending on whether we want a directly additive notion of change or a symmetric around-zero baseline.

### B.3.3 Gradient parity as the primary mechanistic signal.
Figure 6 plots Cycle-TAA against event-local gradient parity, measured as the log-ratio between the average per-unit gradient magnitude of the reference cohort and its comparison cohort. Parity corresponds to 0: negative values indicate that the reference cohort receives less gradient per unit, while positive values indicate the opposite. Across all compactness levels, Grow and Prune form well-separated clusters along this axis. Grow concentrates at negative gradient parity, showing that newborn units are systematically gradient-starved relative to incumbents, whereas Prune concentrates at positive gradient parity, indicating that kept units receive stronger learning signal than pruned ones.

This separation provides a compact summary of the mechanism identified in Sec. 4.2. The central asymmetry is not simply whether units are active, but how learning signal is allocated after a structural edit. In this view, Prune preserves a mature gradient allocation profile and correspondingly stronger trajectory quality, whereas Grow introduces newborn units that remain under-trained because they receive systematically weaker backward credit.

### B.3.4 Activation parity as a sanity check.
Figure 7 provides a complementary sanity check against a simple dead-unit explanation for Grow. We plot event-local activation parity using the same log-ratio transform, so that parity again corresponds to 0. Under neutral allocation, Grow tends to occupy a mildly negative activation-parity regime, whereas Prune tends to occupy a positive one. Thus, newborn units are somewhat less active on average than incumbents, but not trivially silent.

The key point is that activation parity alone does not explain the outcome gap. Newborn units can participate in the forward pass and still fail to integrate effectively if they receive insufficient backward credit. For this reason, we use activation parity primarily as a diagnostic that rules out a dead units, while gradient parity remains the more informative mechanistic quantity for explaining the Grow–Prune separation.

![](./images/7466532944091754543_5.png)

Figure 6: Gradient parity as the primary mechanistic signal (CIFAR-100). Cycle-TAA vs. event-local gradient parity across compactness. The vertical dashed line marks parity (0). Grow occupies the negative-parity regime, indicating newborn gradient disadvantage, whereas Prune occupies the positive-parity regime, indicating that kept units receive stronger learning signal than pruned ones.

![](./images/7466532944091754543_6.png)

Figure 7: Activation parity as a sanity check (CIFAR-100). Cycle-TAA vs. event-local activation parity across compactness. The vertical dashed line marks parity (0). Activation parity shows that newborn units are not trivially inactive, but it does not account for the main performance separation as directly as gradient parity.

B.3.5 Parity geometry of structural edits. Figure 8 reveals a consistent geometric separation between Grow and Prune. Grow-birth points occupy the quadrant with positive activation parity but negative gradient parity, showing that newborn units participate in the forward pass while receiving substantially weaker per-unit gradients than previously active units. In contrast, Prune-exit points lie in the quadrant with positive activation and positive gradient parity, indicating that kept units are both more active and receive stronger learning signal than pruned ones at the exit checkpoint.

This visualization compactly summarizes the main mechanistic asymmetry from Sec. 4.2. Grow does not primarily fail because newborn units are inactive; rather, it introduces units that are forward-participating but backward-starved. Prune, by contrast, preferentially removes low-vitality units while preserving a survivor set that remains both active and learning-capable.

![](./images/7466532944091754543_7.png)

Figure 8: Parity geometry of structural edits. Each panel corresponds to a compactness target $c$. Points plot log activation parity (x-axis) against log gradient parity (y-axis), color-coded by cycle. Markers distinguish event type: Grow-birth (new vs. old) and Prune-exit (kept vs. pruned). Positive x-values indicate greater activation in the focal cohort, while negative y-values indicate reduced per-unit learning signal.

### B.4 Gradient-based Grow does not remove the newborn bottleneck

Our results suggest that newly inserted units are not simply inactive; rather, they are forward-active but receive weaker backward credit than incumbent units. A natural concern is that this effect may be induced by the default activation-based Grow heuristic: if new units are selected using activation statistics, then perhaps the method preferentially inserts units that are active but poorly aligned with the loss gradient. To test this alternative explanation, we repeat the CIFAR-100 Grow sweep under neutral allocation bias, replacing activation-based top-$k$ selection with gradient-based top-$k$ selection while keeping the training horizon, compactness schedule, initialization, and optimizer fixed.

Table 7 shows that gradient-based Grow does not produce a systematic performance im- provement. Cycle ACC and Cycle TAA are nearly unchanged across compactness levels, and the winning-ticket metrics remain similar or slightly worse under gradient-based selection. Thus, selecting growth locations by gradient magnitude is not sufficient to improve either the adaptive trajectory or the re-trainability of the final mask.

<table>
  <thead>
    <tr>
      <th>Comp. (%)</th>
      <th>Grow heuristic</th>
      <th>Cycle ACC</th>
      <th>Cycle TAA</th>
      <th>WT ACC</th>
      <th>WT TAA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20</td>
      <td>Activation</td>
      <td>52.32 ± 0.27</td>
      <td>47.29 ± 0.23</td>
      <td>52.05 ± 0.28</td>
      <td>52.05 ± 0.28</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Gradient</td>
      <td>52.32 ± 0.15</td>
      <td>47.38 ± 0.12</td>
      <td>51.72 ± 0.26</td>
      <td>51.72 ± 0.26</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Activation</td>
      <td>53.31 ± 0.21</td>
      <td>47.87 ± 0.32</td>
      <td>51.63 ± 0.20</td>
      <td>51.63 ± 0.20</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Gradient</td>
      <td>53.22 ± 0.28</td>
      <td>48.09 ± 0.16</td>
      <td>51.67 ± 0.26</td>
      <td>51.67 ± 0.26</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Activation</td>
      <td>53.59 ± 0.27</td>
      <td>48.34 ± 0.25</td>
      <td>50.79 ± 0.29</td>
      <td>50.79 ± 0.29</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Gradient</td>
      <td>53.78 ± 0.26</td>
      <td>48.35 ± 0.30</td>
      <td>50.89 ± 0.31</td>
      <td>50.89 ± 0.31</td>
    </tr>
    <tr>
      <td>50</td>
      <td>Activation</td>
      <td>54.34 ± 0.19</td>
      <td>48.99 ± 0.21</td>
      <td>50.26 ± 0.22</td>
      <td>50.26 ± 0.22</td>
    </tr>
    <tr>
      <td>50</td>
      <td>Gradient</td>
      <td>54.14 ± 0.12</td>
      <td>48.95 ± 0.18</td>
      <td>50.12 ± 0.38</td>
      <td>50.12 ± 0.38</td>
    </tr>
  </tbody>
</table>

Table 7: Grow heuristic ablation. We compare activation-based and gradient-based top-$k$ Grow under neutral allocation bias. This tests whether the newborn gradient bottleneck is simply caused by selecting growth locations using activation statistics. Cycle metrics evaluate the adaptive structural trajectory; WT metrics freeze the final mask and retrain it from scratch. Values are mean ± CI95 over seeds.

Figure 9 compares the birth-time parity diagnostics for activation-based and gradient-based Grow. Both heuristics produce newborn units with positive activation parity, indicating that the inserted units are already forward-active at birth. However, both heuristics also produce strongly negative gradient parity across compactness levels and Grow cycles. Therefore, gradient- based selection does not eliminate the backward-pass disadvantage: even when growth is driven by gradient scores, newborn units still enter the network with substantially weaker gradient magnitudes than incumbent units.

Figure 10 shows the corresponding post-insertion post-birth dynamics. Activation ratio remains near parity and can exceed parity in later cycles, especially at lower compactness. In contrast, gradient ratio remains far below parity for both heuristics, with newborn units receiving only a fraction of the gradient magnitude of incumbent units throughout the cycle. The gradient-based heuristic therefore changes the criterion used to choose where to Grow, but it does not resolve the subsequent optimization problem faced by the inserted units.

### B.5 CIFAR-100: Allocation-Bias Ablation

Because the main CIFAR-100 experiments use a neutral layer-allocation schedule, it is important to verify that the observed Grow behavior is not simply an artifact of how compactness is distributed across layers. Table 8 defines the bias scalars used to distribute the global kept-weight budget across masked fully connected layers. The neutral schedule allocates proportionally to layer weight mass, while the other schedules mildly favor early layers, late layers, or both ends of the head. We use the

![](./images/7466532944091754543_8.png)

Figure 9: Birth-time parity under activation- and gradient-based Grow. We compare the default activation-based top-$k$ Grow heuristic with a gradient-based top-$k$ variant under neutral allocation bias. Top row reports birth activation parity, $\log(\text{act}_{new}/\text{act}_{old})$, and bottom row reports birth gradient parity, $\log(\text{grad}_{new}/\text{grad}_{old})$. The dotted line denotes parity. Both heuristics produce forward-active newborn units, but both remain far below gradient parity, indicating that gradient-based selection does not remove the birth-time backward disadvantage.

![](./images/7466532944091754543_9.png)

Figure 10: Post-insertion ratio under activation- and gradient-based Grow. We report cycle-level ratios from the vitality logs. Top row shows activation ratio $\text{act}_{new}/\text{act}_{old}$, and bottom row shows gradient ratio, $\text{grad}_{new}/\text{grad}_{old}$. The dotted line denotes parity. Although newborn activation rates remain close to parity, newborn gradient magnitudes remain below parity for both heuristics. The bottleneck is not simply a consequence of selecting growth locations by activation statistics; newly inserted units remain backward-disadvantaged even when growth is selected using gradients.

neutral schedule in the main experiments because it is the least assumption-laden default and, as the ablation shows, no alternative biasing pattern yields a consistent advantage across compactness levels and evaluation modes.

Table 9 shows that the CIFAR-100 Grow results are only moderately sensitive to the layer-allocation schedule. Biasing the kept-weight budget toward the last hidden layer (FC3-Protect) tends to improve cycle metrics at 20-40% compactness, suggesting that emphasizing later layers can help short-horizon adaptation during the structural-edit process. However, these gains do not carry over to the retrained winning-ticket evaluation: FC3-Protect consistently exhibits larger negative $\Delta\text{ACC}$ and $\Delta\text{TAA}$, indicating that its stronger cycle performance depends more heavily on the adaptive path and yields weaker final subnetworks after retraining. By contrast, the more

<table>
<thead>
  <tr>
    <th>Schedule</th>
    <th>$b_{\text{FC1}}$</th>
    <th>$b_{\text{FC2}}$</th>
    <th>$b_{\text{FC3}}$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Neutral</td>
    <td>1.0</td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>FC1-Protect</td>
    <td>1.5</td>
    <td>1.5</td>
    <td>0.6</td>
  </tr>
  <tr>
    <td>FC3-Protect</td>
    <td>0.6</td>
    <td>0.6</td>
    <td>1.5</td>
  </tr>
  <tr>
    <td>Ends-Skewed</td>
    <td>1.2</td>
    <td>0.6</td>
    <td>1.2</td>
  </tr>
</tbody>
</table>

Table 8: Bias scalars used to distribute the global kept-weight budget across masked fully connected layers. The neutral schedule allocates proportionally to layer weight mass, while the remaining schedules mildly favor early layers, late layers, or both ends of the head.

balanced Ends-Skewed and FC1-Protect schedules often reduce the ticket-minus-cycle gap and improve winning-ticket metrics at higher compactness, especially at 40-50%, but without producing a uniformly dominant schedule across all regimes. Overall, this ablation suggests that allocation bias can modulate the trade-off between procedure-level adaptation and final architecture quality, but does not overturn the main conclusion of the paper: the dominant limitation of Grow lies in newborn integration dynamics rather than in modest changes to layer-wise compactness allocation.

<table>
<thead>
  <tr>
    <th rowspan="2">Comp. (%)</th>
    <th rowspan="2">Method</th>
    <th colspan="2">Cycle Eval.</th>
    <th colspan="2">Winning-ticket Eval.</th>
    <th colspan="2">Ticket – Cycle</th>
  </tr>
  <tr>
    <th>ACC</th>
    <th>TAA</th>
    <th>ACC</th>
    <th>TAA</th>
    <th>$\Delta$ACC</th>
    <th>$\Delta$TAA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>20</td>
    <td>Grow</td>
    <td>52.316±0.317</td>
    <td>47.288±0.269</td>
    <td>52.055±0.324</td>
    <td>46.257±0.188</td>
    <td>−0.261</td>
    <td>−1.031</td>
  </tr>
  <tr>
    <td>20</td>
    <td>Grow (FC3-Protect)</td>
    <td>53.016±0.393</td>
    <td>47.717±0.461</td>
    <td>51.287±0.314</td>
    <td>46.085±0.175</td>
    <td>−1.729</td>
    <td>−1.632</td>
  </tr>
  <tr>
    <td>20</td>
    <td>Grow (Ends-Skewed)</td>
    <td>51.926±0.153</td>
    <td>46.987±0.196</td>
    <td>51.757±0.259</td>
    <td>45.758±0.190</td>
    <td>−0.169</td>
    <td>−1.230</td>
  </tr>
  <tr>
    <td>20</td>
    <td>Grow (FC1-Protect)</td>
    <td>51.794±0.201</td>
    <td>46.830±0.247</td>
    <td>51.622±0.276</td>
    <td>45.721±0.140</td>
    <td>−0.172</td>
    <td>−1.109</td>
  </tr>
  <tr>
    <td>30</td>
    <td>Grow</td>
    <td>53.312±0.247</td>
    <td>47.871±0.374</td>
    <td>51.626±0.232</td>
    <td>46.909±0.220</td>
    <td>−1.686</td>
    <td>−0.963</td>
  </tr>
  <tr>
    <td>30</td>
    <td>Grow (FC3-Protect)</td>
    <td>53.564±0.345</td>
    <td>48.235±0.284</td>
    <td>50.232±0.290</td>
    <td>46.295±0.161</td>
    <td>−3.332</td>
    <td>−1.940</td>
  </tr>
  <tr>
    <td>30</td>
    <td>Grow (Ends-Skewed)</td>
    <td>52.819±0.220</td>
    <td>47.650±0.326</td>
    <td>52.076±0.257</td>
    <td>46.876±0.189</td>
    <td>−0.743</td>
    <td>−0.775</td>
  </tr>
  <tr>
    <td>30</td>
    <td>Grow (FC1-Protect)</td>
    <td>52.647±0.395</td>
    <td>47.469±0.172</td>
    <td>51.879±0.303</td>
    <td>46.648±0.194</td>
    <td>−0.768</td>
    <td>−0.821</td>
  </tr>
  <tr>
    <td>40</td>
    <td>Grow</td>
    <td>53.588±0.313</td>
    <td>48.338±0.291</td>
    <td>50.791±0.338</td>
    <td>46.886±0.239</td>
    <td>−2.797</td>
    <td>−1.452</td>
  </tr>
  <tr>
    <td>40</td>
    <td>Grow (FC3-Protect)</td>
    <td>53.761±0.283</td>
    <td>48.424±0.257</td>
    <td>49.550±0.369</td>
    <td>46.105±0.219</td>
    <td>−4.211</td>
    <td>−2.319</td>
  </tr>
  <tr>
    <td>40</td>
    <td>Grow (Ends-Skewed)</td>
    <td>53.409±0.257</td>
    <td>48.146±0.389</td>
    <td>51.670±0.311</td>
    <td>47.129±0.231</td>
    <td>−1.739</td>
    <td>−1.017</td>
  </tr>
  <tr>
    <td>40</td>
    <td>Grow (FC1-Protect)</td>
    <td>53.305±0.258</td>
    <td>48.191±0.101</td>
    <td>51.263±0.359</td>
    <td>46.795±0.187</td>
    <td>−2.042</td>
    <td>−1.396</td>
  </tr>
  <tr>
    <td>50</td>
    <td>Grow</td>
    <td>54.338±0.221</td>
    <td>48.990±0.245</td>
    <td>50.260±0.257</td>
    <td>46.776±0.174</td>
    <td>−4.078</td>
    <td>−2.213</td>
  </tr>
  <tr>
    <td>50</td>
    <td>Grow (FC3-Protect)</td>
    <td>54.206±0.259</td>
    <td>48.941±0.230</td>
    <td>49.415±0.258</td>
    <td>46.293±0.165</td>
    <td>−4.791</td>
    <td>−2.647</td>
  </tr>
  <tr>
    <td>50</td>
    <td>Grow (Ends-Skewed)</td>
    <td>53.759±0.225</td>
    <td>48.272±0.298</td>
    <td>50.983±0.337</td>
    <td>47.057±0.253</td>
    <td>−2.776</td>
    <td>−1.215</td>
  </tr>
  <tr>
    <td>50</td>
    <td>Grow (FC1-Protect)</td>
    <td>53.791±0.264</td>
    <td>48.340±0.246</td>
    <td>50.884±0.258</td>
    <td>46.887±0.210</td>
    <td>−2.907</td>
    <td>−1.453</td>
  </tr>
</tbody>
</table>

Table 9: CIFAR-100 (i.i.d.) ConvNet, ReLU, 200 epochs (SGD, $\eta$=0.1). Mean $\pm$ CI95 over $n=10$ seeds. We report cycle evaluation metrics (measured during structural adaptation) and winning-ticket evaluation metrics (mask frozen, weights reinitialized, retrained from scratch). Deltas are computed within method and compactness as $\Delta=$ (ticket $-$ cycle) (%). Bold highlights the highest value among the Grow bias schedules within each compactness for each column.

## B.6 CIFAR-10: Additional Results

We repeat the main ConvNet comparison on CIFAR-10 and observe the same qualitative separation between procedure-level learning dynamics and architecture-level ticket quality, although in a milder regime than CIFAR-100 (Fig. 11, Table 10).

CIFAR-10 reproduces the same qualitative distinction seen on CIFAR-100, but in a weaker regime. During the adaptive structural process, Grow consistently attains higher Cycle-ACC than Prune across compactness, while Prune maintains a clear advantage in Cycle-TAA. Thus, as on CIFAR-100, Grow appears stronger at the endpoint of the adaptive trajectory, whereas Prune is stronger in time-averaged trajectory quality. After retraining the discovered masks from

![](./images/7466532944091754543_10.png)

Figure 11: Cycle vs. Winning-Ticket performance on CIFAR-10 (SGD, $\eta$=0.1). Panels (a)-(d) show mean ± 95% CI with per-seed scatter across compactness for Cycle and Winning-Ticket ACC (a,b) and TAA (c,d). Panel (e) reports the per-seed gap $\Delta$ = ticket – cycle in final accuracy.

scratch, however, the two methods become nearly indistinguishable: Winning-Ticket ACC and TAA are almost tied across all compactness levels. The within-method deltas support the same interpretation. Grow exhibits consistently negative $\Delta$ACC and less negative $\Delta$TAA, indicating that its cycle advantage depends more strongly on the adaptive path. Prune, by contrast, shows much smaller ACC drops and systematically more negative $\Delta$TAA, reflecting stronger cycle-time learning but little corresponding advantage in the final retrained architecture. Overall, CIFAR-10 supports the same procedure-versus-architecture distinction as CIFAR-100, but with a smaller overall separation.

<table>
  <thead>
    <tr>
      <th rowspan="2">Comp. (%)</th>
      <th rowspan="2">Method</th>
      <th colspan="2">Cycle Eval.</th>
      <th colspan="2">Winning-ticket Eval.</th>
      <th colspan="2">Ticket – Cycle</th>
    </tr>
    <tr>
      <th>ACC</th>
      <th>TAA</th>
      <th>ACC</th>
      <th>TAA</th>
      <th>$\Delta$ACC</th>
      <th>$\Delta$TAA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>Dense</td>
      <td>83.739±0.158</td>
      <td>82.652±0.157</td>
      <td>83.739±0.158</td>
      <td>80.934±0.106</td>
      <td>0.000</td>
      <td>–1.717</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Grow</td>
      <td>83.426±0.166</td>
      <td>81.531±0.149</td>
      <td>83.011±0.215</td>
      <td>79.738±0.146</td>
      <td>–0.415</td>
      <td>–1.794</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Prune</td>
      <td>82.265±0.494</td>
      <td>82.158±0.315</td>
      <td>83.146±0.207</td>
      <td>79.754±0.133</td>
      <td>+0.881</td>
      <td>–2.403</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Grow</td>
      <td>83.683±0.233</td>
      <td>81.520±0.286</td>
      <td>83.018±0.189</td>
      <td>79.935±0.136</td>
      <td>–0.665</td>
      <td>–1.585</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Prune</td>
      <td>82.853±0.415</td>
      <td>82.257±0.207</td>
      <td>83.174±0.290</td>
      <td>80.044±0.231</td>
      <td>+0.321</td>
      <td>–2.213</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Grow</td>
      <td>83.732±0.262</td>
      <td>81.598±0.211</td>
      <td>82.869±0.237</td>
      <td>79.975±0.173</td>
      <td>–0.863</td>
      <td>–1.623</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Prune</td>
      <td>83.015±0.463</td>
      <td>82.301±0.295</td>
      <td>82.882±0.195</td>
      <td>80.026±0.139</td>
      <td>–0.133</td>
      <td>–2.275</td>
    </tr>
    <tr>
      <td>50</td>
      <td>Grow</td>
      <td>83.802±0.152</td>
      <td>81.726±0.255</td>
      <td>82.820±0.280</td>
      <td>80.082±0.214</td>
      <td>–0.982</td>
      <td>–1.644</td>
    </tr>
    <tr>
      <td>50</td>
      <td>Prune</td>
      <td>83.254±0.394</td>
      <td>82.487±0.217</td>
      <td>82.923±0.333</td>
      <td>80.053±0.253</td>
      <td>–0.331</td>
      <td>–2.435</td>
    </tr>
  </tbody>
</table>

Table 10: CIFAR-10 (i.i.d.) ConvNet, ReLU, 200 epochs (SGD, $\eta$=0.1). Mean ± CI95 over $n=10$ seeds. We report cycle evaluation metrics (measured during structural adaptation) and winning-ticket evaluation metrics (mask frozen, weights reinitialized, retrained from scratch). Deltas are computed within method and compactness as $\Delta=$ (ticket – cycle) (%). Bold highlights the higher value between Grow and Prune within each compactness for each column.

Table 11 shows that the Cycle-ACC separation between Grow and Prune is statistically reliable across all compactness levels, whereas the corresponding Winning-Ticket ACC difference is not significant at any compactness. During adaptive structural editing, Grow and Prune follow genuinely different optimization trajectories, but once the final masks are frozen and retrained from scratch, that separation largely disappears. Comparisons against Dense reinforce this view. Prune is consistently distinct from Dense in both cycle and winning-ticket ACC, while Grow differs strongly from Dense under winning-ticket retraining and only at low compactness during cycle training. Overall, the ACC p-values indicate that on CIFAR-10, as on CIFAR-100, the most reliable Grow–Prune difference lies in the adaptive process rather than in the final retrainable sparse architecture.

<table>
<thead>
<tr>
<th rowspan="2">Comparison</th>
<th colspan="2">20%</th>
<th colspan="2">30%</th>
<th colspan="2">40%</th>
<th colspan="2">50%</th>
</tr>
<tr>
<th>Cycle</th>
<th>WT</th>
<th>Cycle</th>
<th>WT</th>
<th>Cycle</th>
<th>WT</th>
<th>Cycle</th>
<th>WT</th>
</tr>
</thead>
<tbody>
<tr>
<td>Grow vs Dense</td>
<td>0.0063</td>
<td>0.0000</td>
<td>0.6589</td>
<td>0.0000</td>
<td>0.9594</td>
<td>0.0000</td>
<td>0.5234</td>
<td>0.0000</td>
</tr>
<tr>
<td>Grow vs Prune</td>
<td>0.0004</td>
<td>0.3196</td>
<td>0.0014</td>
<td>0.3234</td>
<td>0.0085</td>
<td>0.9248</td>
<td>0.0129</td>
<td>0.5993</td>
</tr>
<tr>
<td>Dense vs Prune</td>
<td>0.0001</td>
<td>0.0001</td>
<td>0.0008</td>
<td>0.0017</td>
<td>0.0064</td>
<td>0.0000</td>
<td>0.0242</td>
<td>0.0003</td>
</tr>
</tbody>
</table>

Table 11: Welch two-sample t-test p-values comparing methods across compactness levels for Cycle and winning-ticket (WT) ACC evaluations. Bold indicates statistical significance at $p < 0.05$.

Table 12 shows that the strongest and most consistent CIFAR-10 separation appears in TAA during the adaptive cycle. Across all compactness levels, Grow vs. Prune is significant for Cycle-TAA, confirming that the two methods induce systematically different trajectory-level learning dynamics. In contrast, the same comparison is not significant for Winning-Ticket TAA at any compactness, indicating that this trajectory-level separation does not survive retraining of the final masks. Comparisons to Dense are also informative: Grow differs significantly from Dense for both cycle and winning-ticket TAA at all compactness levels, whereas Prune is significantly different from Dense in winning-ticket TAA throughout and in cycle TAA except at 50% compactness. Taken together, the TAA p-values reinforce the same conclusion as the raw results: on CIFAR-10, the dominant Grow–Prune difference is a difference in adaptive trajectory quality, not in the final retrainable sparse architecture.

<table>
<thead>
<tr>
<th rowspan="2">Comparison</th>
<th colspan="2">20%</th>
<th colspan="2">30%</th>
<th colspan="2">40%</th>
<th colspan="2">50%</th>
</tr>
<tr>
<th>Cycle</th>
<th>WT</th>
<th>Cycle</th>
<th>WT</th>
<th>Cycle</th>
<th>WT</th>
<th>Cycle</th>
<th>WT</th>
</tr>
</thead>
<tbody>
<tr>
<td>Grow vs Dense</td>
<td>0.0000</td>
<td>0.0000</td>
<td>0.0000</td>
<td>0.0000</td>
<td>0.0000</td>
<td>0.0000</td>
<td>0.0000</td>
<td>0.0000</td>
</tr>
<tr>
<td>Grow vs Prune</td>
<td>0.0014</td>
<td>0.8494</td>
<td>0.0002</td>
<td>0.3713</td>
<td>0.0004</td>
<td>0.6069</td>
<td>0.0001</td>
<td>0.8436</td>
</tr>
<tr>
<td>Dense vs Prune</td>
<td>0.0071</td>
<td>0.0000</td>
<td>0.0032</td>
<td>0.0000</td>
<td>0.0327</td>
<td>0.0000</td>
<td>0.1831</td>
<td>0.0000</td>
</tr>
</tbody>
</table>

Table 12: Welch two-sample t-test p-values comparing methods across compactness levels for Cycle and winning-ticket (WT) TAA evaluations. Bold indicates statistical significance at $p < 0.05$.

### B.7 Growth-cycle stress test

We test the time-scale interpretation from Sec. 4.2 by fixing the total training horizon to 200 epochs and varying the number of growth cycles, $K \in \{5, 10, 20\}$. Although smaller $K$ means fewer growth events, it consistently yields higher Cycle-TAA. This is only superficially counterintuitive. Under a fixed total training horizon, increasing $K$ does not create extra learning time; it simply divides the same budget across more insertion events. As a result, as shown in Fig. 12, the model repeatedly pays the cost of integrating newborn units, but has less time after each event for those units to become useful. Smaller $K$ therefore improves trajectory quality by reducing how often training is pulled back into the low-integration regime.

Post-birth dynamics under time scarcity. Figure 13 shows a consistent within-cycle pattern: newborn units experience an early post-birth integration deficit followed by gradual recovery as the cycle progresses. Longer cycles reveal more of this recovery tail. For $K = 5$ (40 epochs per cycle), the ratio curve continues rising well into later ages, indicating that newborn integration remains incomplete for many epochs. For larger $K$, each event may be individually milder, but recovery is repeatedly interrupted because new growth events occur more often and each cycle is shorter. This provides a mechanistic explanation for the TAA drop with increasing $K$: the system spends a larger fraction of total training time in low-integration phases and has fewer opportunities to reach the late-cycle recovery regime.

![](./images/7466532944091754543_11.png)

Figure 12: Grow cycle stress test on CIFAR-100. Left: Cycle-TAA degrades monotonically with $K$ at all compactness levels, indicating worse time-averaged learning when growth events become more frequent. Right: Cycle-ACC is less affected than TAA, with only mild sensitivity at higher compactness.

![](./images/7466532944091754543_12.png)

Figure 13: Post-birth dynamics under time scarcity. Gradient ratio as a function of newborn age for $K \in \{5, 10, 20\}$ across compactness levels. In all cases, parity improves gradually with age, showing that newborn integration is slow and continues over many epochs. Shorter cycles truncate this recovery by reducing the time available before the next growth event.

## C Two-Speed and Moment Transplant Explanation

This appendix briefly documents the optimizer-side interventions from Sec. 5. We clarify the mechanism they were designed to target: newborn units may be disadvantaged not only because they receive weak learning signal, but also because their newly created pathway is slow to become trainable and their optimizer state is born cold. Unless otherwise noted, these ablations use the same CIFAR-100 i.i.d. setting as the main intervention study.

### C.1 Intervention intuition and terminology

A growth event activates newborn units in a layer $\ell$; these features are then read by the immediate downstream layer $\ell + 1$. We refer to the layer where units are activated as the *producer*, and the immediate downstream reader as the *consumer*. This distinction is useful because newborn disadvantage can arise both in the grown layer itself and in the downstream pathway that must learn to use the new features.

Two-Speed. Two-SPEED is a temporary timescale-control mechanism applied after each optimizer step to a selected parameter slice. Rather than scaling raw gradients, we apply *delta scaling*,

$$
W_{\text{slice}} \leftarrow W_{\text{old,slice}} + r\left(W_{\text{new,slice}} - W_{\text{old,slice}}\right), \tag{26}
$$

where $W_{\text{new,slice}}$ is the parameter value after the optimizer update and $r > 1$ is the intended effective multiplier. Intuitively, this intervention tries to let newborn-related parameters write faster early in life.

For completeness, we also swept Two-Speed hyperparameters over learning-rate multipliers $r \in \{2, 3, 5, 8, 10\}$ and warm-up window lengths $N \in \{782, 1564, 1955, 3910, 7820\}$. These sweeps

were useful for checking whether the weak default performance of Two-Speed was simply a poor hyperparameter choice. In practice, the differences across settings were neither statistically reliable nor consistent across compactness levels: smaller multipliers sometimes improved integration, longer windows sometimes helped at higher compactness, but no setting dominated robustly across the grid. Therefore, we retained $r = 5$ and $N = 1955$ as the default configuration. These values were representative, avoided overfitting the intervention to a single compactness regime, and did not materially change the qualitative conclusion that Two-Speed remained less reliable than Moment Transplant.

Moment Transplant. MOMENT TRANSPLANT addresses optimizer cold-start directly by copying optimizer buffers from a matched active donor unit into the newly activated slice at birth. Under adaptive optimizers, this gives newborn parameters a nontrivial initial optimizer state instead of forcing them to accumulate moment estimates from scratch. Conceptually, Two-Speed modifies the *write rate* of newborn updates, whereas Moment Transplant modifies the *initial optimizer state*; the two are therefore complementary.

### C.2 Summary of findings

Across compactness levels, Moment Transplant was the more reliable of the two optimizer-side interventions, whereas default Two-Speed was weaker and in some regimes mildly harmful. This pattern is consistent with the interpretation that optimizer cold-start is a real component of newborn disadvantage, but that simply accelerating updates is not, by itself, a robust fix under the default Adam-based setting. In other words, optimizer-side asymmetry matters, but it does not fully explain the broader newborn integration bottleneck highlighted in the main paper.

## D Activation-Control Benchmark Analysis

The activation-function intervention in Sec. 5 was motivated by a specific mechanistic question: whether improving activation-level trainability helps reduce the newborn integration disadvantage of growth, rather than merely improving performance through an unrelated change in insertion, selection, or optimizer-state handling. In the main text, Rand. Smooth-Leaky was introduced precisely as this kind of intervention, since it alters the gradient pathway through which newly added units attempt to integrate into a mature network while leaving the structural adaptation process unchanged.

Protocol. For each benchmark and method, we compare the same structural procedure under two activations: standard ReLU and Rand. Smooth-Leaky (RSL). We report the signed difference

$$
\Delta = \text{RSL} - \text{ReLU},
$$

so that positive values indicate improvement under Rand. Smooth-Leaky. The comparison is performed independently for each method, benchmark, and compactness level, and the reported values are then aggregated over the compactness levels shown in the figure. To avoid mixing activation effects with learning-rate choice, the best learning rate is selected separately within each activation condition before computing the delta. The full suite spans eight datasets: the five repeated-shift plasticity benchmarks, Split-CIFAR100 as the sequential-accumulation continual-learning setting, and the i.i.d. CIFAR-10 and CIFAR-100 controls.

Figure 14 asks whether the gains attributed to the activation intervention are concentrated in the regime where growth is expected to struggle most, namely settings in which newly added units must integrate into an already mature representation under limited adaptation time.

Rand. Smooth-Leaky hyperparameters.. For the continual-learning plasticity benchmarks (Fig. 5), we used the default Rand. Smooth-Leaky hyperparameters reported in Lillo and Cheney (2025). For the additional settings considered in this paper—Split-CIFAR100 and the i.i.d. CIFAR-10/CIFAR-100 controls—we did not assume those defaults would transfer directly. Instead, we ran a dedicated sweep over the Rand. Smooth-Leaky shape parameters, varying both $c$ and $p$ over

$$\{0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5\},$$

and combining them with the following lower/upper-bound pairs:

(0.01, 0.05), (0.01, 1.00), (0.125, 0.333), (0.40, 1.00), (0.50, 1.00),
(0.60, 0.80), (0.60, 1.00), (0.70, 1.00), (0.673, 2.673).

Table 13 summarizes the Rand. Smooth-Leaky settings used in each benchmark.

For Split-CIFAR100 and the i.i.d. CIFAR-10/CIFAR-100 controls, the sweep was performed using the Dense configuration only. We then fixed the best-performing Rand. Smooth-Leaky configuration selected under Dense and reused that same configuration for both Grow and Prune. This design avoids method-specific over-tuning of the activation function and ensures that differences between Dense, Grow, and Prune reflect the structural operators themselves rather than separate activation hyperparameter searches.

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>c</th>
      <th>p</th>
      <th>Lower</th>
      <th>Upper</th>
      <th>LR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Permuted MNIST</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.6</td>
      <td>0.001</td>
    </tr>
    <tr>
      <td>Random-Label MNIST</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>0.6</td>
      <td>0.001</td>
    </tr>
    <tr>
      <td>Random-Label CIFAR</td>
      <td>0.8</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.001</td>
    </tr>
    <tr>
      <td>5+1 CIFAR</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.673</td>
      <td>2.673</td>
      <td>0.001</td>
    </tr>
    <tr>
      <td>Continual ImageNet</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.001</td>
    </tr>
    <tr>
      <td>Split-CIFAR100</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.673</td>
      <td>2.673</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <td>CIFAR-10 (i.i.d.)</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.125</td>
      <td>0.333</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>CIFAR-100 (i.i.d.)</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.125</td>
      <td>0.333</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>

Table 13: Rand. Smooth-Leaky hyperparameters used in the activation-control analysis. For the main continual-learning plasticity benchmarks, settings are inherited from Lillo and Cheney (2025). For Split-CIFAR100 and the i.i.d. controls, we ran a dedicated sweep over $c$, $p$, and lower/upper bounds; Split-CIFAR100 and i.i.d. CIFAR-10/CIFAR-100 reports the selected best configuration.

Results. Rand. Smooth-Leaky is strongly regime-dependent and interacts differently with each structural operator. Across several plasticity-stressing benchmarks, the largest positive deltas are concentrated in Grow. This is clearest on Random-Label CIFAR, 5+1 CIFAR, Continual ImageNet, and i.i.d. CIFAR-100, where replacing ReLU with Rand. Smooth-Leaky substantially improves growth relative to its ReLU counterpart, often by a much larger margin than for Prune and, in several cases, also more than for Dense. In line with the main paper's explanation, these results show that when the primary bottleneck is not merely representation capacity but the ability of newly added units to quickly become trainable, altering the activation pathway can materially improve how useful growth becomes. The intervention is not only stabilizing structural growth events, but can also improve trainability more broadly in highly non-stationary or memorization-heavy regimes as suggested by Dense also showing substantial positive deltas under Rand. Smooth-Leaky.

Interpretation. This ablation study allows us to help distinguish two explanations for the main-text activation result. On one side, it is possible that that Rand. Smooth-Leaky is simply a generally stronger activation and therefore improves all methods in roughly the same way. On the other

![](./images/7466532944091754543_13.png)

Figure 14: Activation-control analysis across eight benchmarks. Each bar reports the signed delta $\Delta =$ RSL$-$ReLU, positive values indicate that replacing ReLU with Rand. Smooth-Leaky improves performance for that method and compactness. The central pattern is not a uniform lift across all methods, but a redistribution of benefit across structural regimes: Rand. Smooth- Leaky most strongly improves GRow in several of the harder repeated-shift and CIFAR-100 settings, provides broader trainability gains in some plasticity-stressing benchmarks, benefits PRUNE most clearly in Split-CIFAR100. Results support the interpretation that the activation intervention primarily acts on optimization compatibility and trainability, especially where growth is limited by rapid post-birth integration, rather than serving as a generic activation swap that helps all methods equally.

hand, Rand. Smooth-Leaky helps the regime in which the paper predicts a trainability bottleneck, namely the rapid integration of new capacity into a mature network. Our results are more consistent with the second interpretation, although not in a purely exclusive form. Rand. Smooth-Leaky acts as a regime-sensitive trainability intervention. Its gains are often largest where optimization is hardest for growth—that is, where newly added units must become useful quickly under continued shift or under more difficult feature-learning conditions.

### D.1 ReLU vs. Rand. Smooth-Leaky Newborn Integration

As we just discussed, Rand. Smooth-Leaky often improves GRow in regimes where rapid adaptation is difficult. Next, we ask whether this performance gain is accompanied by the mechanistic consequences predicted by our main analysis: improved newborn integration after a growth event. To test this, we compare ReLU and Rand. Smooth-Leaky in the controlled CIFAR-100 GRow setting using the same newborn-old parity diagnostics introduced in Sec. 4.2.

Interpretation. Figure 15 shows that Rand. Smooth-Leaky modestly increases birth activation parity across compactness levels. Therefore, the activation intervention should not be interpreted as making newborn units instantly equivalent to previously active units. Although gradient ratio remains below parity, the gap is substantially smaller under Rand. Smooth-Leaky. Thus, Fig. 16 shows that the activation change appears to improve the trainability of newborn units during early integration rather than eliminating the birth-time disadvantage itself. These diagnostics support the interpretation that Rand. Smooth-Leaky acts as an integration-side intervention. It does not simply make newborn units active at birth; instead, it helps them remain better coupled to the forward computation and backward credit-assignment pathway during the period in which newly added capacity must become useful.

35

![](./images/7466532944091754543_14.png)

Figure 15: Rand. Smooth-Leaky increases newborn forward participation at birth but does not eliminate the immediate gradient disadvantage. We compare ReLU and Rand. Smooth- Leaky in the CIFAR-100 Grow setting using event-aligned newborn-old log-parity at the birth snapshot. Positive activation parity indicates that newborn units are forward-active relative to previously active units, while negative gradient parity indicates reduced per- unit backward credit. Rand. Smooth-Leaky consistently raises activation parity, showing stronger forward participation at insertion. However, gradient parity remains strongly negative for both activations, indicating that the activation change does not by itself remove the birth-time credit-assignment bottleneck.

![](./images/7466532944091754543_15.png)

Figure 16: Rand. Smooth-Leaky improves post-birth dynamics. We compare activation and gradient ratios for newborn units after growth events in CIFAR-100 Grow. Parity corresponds to a ratio of 1. Under ReLU, newborn units often remain below activation parity and receive substantially weaker gradient signal than incumbents. Rand. Smooth-Leaky shifts activation ratios above or closer to parity and consistently increases gradient ratios across compactness levels. Although newborn gradients remain below parity, the smaller gap under Rand. Smooth-Leaky indicates improved early integration rather than complete removal of the newborn disadvantage.

## E Early-Task Plasticity Under Repeated Shift

We additionally evaluate *Early Task TAA*, which summarizes performance only over the initial portion of each task, isolate immediate post-shift plasticity rather than asking only how well a method performs after substantial within-task adaptation. Thus, measuring how quickly the learner becomes useful once a new shift arrives.

Across the repeated-shift benchmarks, figure 17 shows that methods based on structural growth remain highly sensitive to the amount of optimization available after each edit. When the post-shift horizon is extremely short, as in Permuted MNIST and Continual ImageNet, the fully dense baseline

retains a clear advantage in the early window, consistent with the idea that already-mature capacity is easier to exploit immediately than newly inserted capacity. In these regimes, repeated growth events are costly because newborn units must begin contributing before they have had enough time to integrate.

At the same time, the figure also shows that growth is not uniformly ineffective. On the concept-shift benchmarks Random-Label MNIST and Random-Label CIFAR, GROW+RAND. SMOOTH-LEAKY substantially improves over vanilla GROW and is the strongest growth-based variant, narrowing much of the gap to PRUNE. This is consistent with the main intervention study (see App. D) on activation-level trainability for improving newborn integration. The remaining interventions—TwOSPEED, MOMENT TRANSPLANT, GRADMAX, and NET2WIDER—yield smaller and less consistent gains in the early window.

The same time-scale dependence appears in the sequential-accumulation regime. On Split-CIFAR100, where structural events are separated by more optimization, growth-based methods become much more competitive in the early window, and GROW+RAND. SMOOTH-LEAKY is especially strong at lower compactness levels. This supports the interpretation that the main limitation of growth is not simply the act of adding capacity, but whether newly added units can be stabilized and integrated quickly enough to support adaptation before the next change.

![](./images/7466532944091754543_16.png)

Figure 17: Early-task plasticity across sequential-accumulation and repeated-shift benchmarks. We report Early Task TAA, defined as the average accuracy over only the initial portion of each task, emphasizing immediate post-shift adaptation rather than late within-task convergence. DENSE or PRUNE-based methods retain the strongest early-window performance when adaptation time is severely limited, whereas growth-based methods improve when insertion is made more integration-friendly. GROW+RAND. SMOOTH-LEAKY is the most consistently effective growth-family intervention, substantially improving over vanilla GROW. These results support the view that the utility of growth depends critically on how quickly newborn units can integrate after insertion.

<table>
  <thead>
    <tr>
      <th>Comp.</th>
      <th>Method</th>
      <th colspan="2">Random-Label CIFAR</th>
      <th colspan="2">5+1 CIFAR</th>
      <th colspan="2">Continual ImageNet</th>
      <th colspan="2">Permuted MNIST</th>
      <th colspan="2">Random-Label MNIST</th>
      <th colspan="2">Split-CIFAR100</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Avg.</th>
      <th>Early</th>
      <th>Avg.</th>
      <th>Early</th>
      <th>Avg.</th>
      <th>Early</th>
      <th>Avg.</th>
      <th>Early</th>
      <th>Avg.</th>
      <th>Early</th>
      <th>Avg.</th>
      <th>Early</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20%</td>
      <td>Dense</td>
      <td>46.3 ± 11.6</td>
      <td>17.3 ± 0.6</td>
      <td>13.8 ± 5.1</td>
      <td>26.1 ± 2.4</td>
      <td>73.3 ± 0.4</td>
      <td>68.2 ± 2.5</td>
      <td>80.6 ± 0.9</td>
      <td>90.9 ± 0.2</td>
      <td>39.2 ± 9.8</td>
      <td>16.8 ± 0.7</td>
      <td>15.8 ± 0.5</td>
      <td>39.9 ± 1.6</td>
    </tr>
    <tr>
      <td></td>
      <td>Prune</td>
      <td>97.6 ± 0.3</td>
      <td>19.6 ± 0.0</td>
      <td>73.1 ± 0.8</td>
      <td>24.1 ± 0.4</td>
      <td>89.9 ± 0.3</td>
      <td>50.7 ± 0.1</td>
      <td>85.9 ± 0.1</td>
      <td>12.1 ± 0.1</td>
      <td>94.7 ± 0.4</td>
      <td>19.6 ± 0.1</td>
      <td>21.3 ± 0.6</td>
      <td>55.1 ± 2.1</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow</td>
      <td>13.5 ± 0.6</td>
      <td>12.8 ± 0.4</td>
      <td>23.7 ± 2.6</td>
      <td>16.9 ± 0.8</td>
      <td>70.7 ± 1.1</td>
      <td>50.7 ± 0.1</td>
      <td>71.5 ± 0.1</td>
      <td>18.1 ± 0.1</td>
      <td>16.3 ± 1.0</td>
      <td>12.9 ± 0.2</td>
      <td>19.8 ± 1.2</td>
      <td>50.7 ± 2.4</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Rand. Smooth-Leaky</td>
      <td>94.4 ± 0.2</td>
      <td>19.6 ± 0.0</td>
      <td>63.6 ± 0.9</td>
      <td>21.6 ± 0.9</td>
      <td>81.5 ± 0.1</td>
      <td>50.9 ± 0.1</td>
      <td>75.1 ± 0.1</td>
      <td>16.2 ± 0.1</td>
      <td>72.1 ± 2.3</td>
      <td>19.5 ± 0.1</td>
      <td>25.0 ± 0.7</td>
      <td>57.7 ± 2.6</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+TwoSpeed</td>
      <td>13.8 ± 0.5</td>
      <td>12.8 ± 0.3</td>
      <td>20.8 ± 3.0</td>
      <td>16.4 ± 1.2</td>
      <td>67.3 ± 0.4</td>
      <td>50.7 ± 0.0</td>
      <td>71.5 ± 0.2</td>
      <td>18.3 ± 0.1</td>
      <td>16.1 ± 0.9</td>
      <td>12.9 ± 0.2</td>
      <td>19.9 ± 0.8</td>
      <td>52.2 ± 2.3</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Moment Transplant</td>
      <td>15.2 ± 1.3</td>
      <td>13.4 ± 0.5</td>
      <td>25.0 ± 2.0</td>
      <td>18.1 ± 1.0</td>
      <td>68.9 ± 2.2</td>
      <td>50.8 ± 0.1</td>
      <td>72.0 ± 0.1</td>
      <td>18.1 ± 0.1</td>
      <td>17.7 ± 0.5</td>
      <td>13.7 ± 0.1</td>
      <td>20.6 ± 1.0</td>
      <td>52.9 ± 1.9</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+GradMax</td>
      <td>13.7 ± 1.0</td>
      <td>12.6 ± 0.4</td>
      <td>24.4 ± 3.0</td>
      <td>16.4 ± 1.2</td>
      <td>71.6 ± 1.9</td>
      <td>50.7 ± 0.0</td>
      <td>71.5 ± 0.2</td>
      <td>18.2 ± 0.1</td>
      <td>16.2 ± 0.6</td>
      <td>12.9 ± 0.1</td>
      <td>20.1 ± 1.1</td>
      <td>51.9 ± 2.3</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Net2Wider</td>
      <td>13.1 ± 0.4</td>
      <td>12.5 ± 0.1</td>
      <td>23.0 ± 2.7</td>
      <td>17.1 ± 0.9</td>
      <td>70.7 ± 2.2</td>
      <td>50.8 ± 0.1</td>
      <td>71.3 ± 0.1</td>
      <td>18.1 ± 0.1</td>
      <td>14.8 ± 0.5</td>
      <td>12.4 ± 0.1</td>
      <td>15.6 ± 1.0</td>
      <td>51.5 ± 2.3</td>
    </tr>
    <tr>
      <td>30%</td>
      <td>Dense</td>
      <td>46.3 ± 11.6</td>
      <td>17.3 ± 0.6</td>
      <td>13.8 ± 5.1</td>
      <td>26.1 ± 2.4</td>
      <td>73.3 ± 0.4</td>
      <td>68.2 ± 2.5</td>
      <td>80.6 ± 0.9</td>
      <td>90.9 ± 0.2</td>
      <td>39.2 ± 9.8</td>
      <td>16.8 ± 0.7</td>
      <td>15.8 ± 0.5</td>
      <td>39.9 ± 1.6</td>
    </tr>
    <tr>
      <td></td>
      <td>Prune</td>
      <td>97.6 ± 0.3</td>
      <td>19.6 ± 0.0</td>
      <td>74.0 ± 0.8</td>
      <td>24.1 ± 0.5</td>
      <td>92.0 ± 0.2</td>
      <td>50.7 ± 0.1</td>
      <td>86.5 ± 0.1</td>
      <td>11.7 ± 0.1</td>
      <td>95.1 ± 0.4</td>
      <td>19.6 ± 0.0</td>
      <td>21.3 ± 0.6</td>
      <td>54.5 ± 2.3</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow</td>
      <td>14.9 ± 1.6</td>
      <td>12.8 ± 0.6</td>
      <td>29.4 ± 3.0</td>
      <td>17.7 ± 1.5</td>
      <td>71.9 ± 1.4</td>
      <td>50.7 ± 0.0</td>
      <td>73.3 ± 0.1</td>
      <td>18.2 ± 0.1</td>
      <td>19.4 ± 1.1</td>
      <td>13.3 ± 0.1</td>
      <td>21.1 ± 0.4</td>
      <td>49.4 ± 1.3</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Rand. Smooth-Leaky</td>
      <td>96.1 ± 0.1</td>
      <td>19.6 ± 0.0</td>
      <td>65.3 ± 1.0</td>
      <td>22.1 ± 0.8</td>
      <td>82.8 ± 0.1</td>
      <td>50.8 ± 0.1</td>
      <td>78.1 ± 0.0</td>
      <td>15.6 ± 0.0</td>
      <td>83.4 ± 1.2</td>
      <td>19.5 ± 0.0</td>
      <td>24.5 ± 0.6</td>
      <td>57.8 ± 2.7</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+TwoSpeed</td>
      <td>15.4 ± 1.5</td>
      <td>12.9 ± 0.2</td>
      <td>28.3 ± 1.2</td>
      <td>17.3 ± 0.7</td>
      <td>69.3 ± 0.1</td>
      <td>50.7 ± 0.1</td>
      <td>73.3 ± 0.2</td>
      <td>18.2 ± 0.1</td>
      <td>20.1 ± 1.1</td>
      <td>13.3 ± 0.1</td>
      <td>21.2 ± 0.6</td>
      <td>50.5 ± 1.4</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Moment Transplant</td>
      <td>19.5 ± 3.5</td>
      <td>13.9 ± 0.8</td>
      <td>35.1 ± 1.9</td>
      <td>17.5 ± 0.8</td>
      <td>72.2 ± 1.2</td>
      <td>50.7 ± 0.0</td>
      <td>74.0 ± 0.1</td>
      <td>18.2 ± 0.1</td>
      <td>20.3 ± 1.0</td>
      <td>13.5 ± 0.1</td>
      <td>22.1 ± 0.5</td>
      <td>55.6 ± 1.2</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+GradMax</td>
      <td>13.9 ± 1.2</td>
      <td>12.7 ± 0.3</td>
      <td>31.8 ± 1.6</td>
      <td>17.7 ± 0.5</td>
      <td>69.9 ± 0.2</td>
      <td>50.8 ± 0.1</td>
      <td>73.4 ± 0.2</td>
      <td>18.1 ± 0.1</td>
      <td>20.6 ± 1.1</td>
      <td>13.5 ± 0.2</td>
      <td>21.6 ± 0.8</td>
      <td>53.2 ± 1.5</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Net2Wider</td>
      <td>15.1 ± 1.1</td>
      <td>12.9 ± 0.2</td>
      <td>30.6 ± 2.1</td>
      <td>18.0 ± 0.7</td>
      <td>67.9 ± 0.2</td>
      <td>50.7 ± 0.1</td>
      <td>73.1 ± 0.2</td>
      <td>18.1 ± 0.1</td>
      <td>16.5 ± 0.7</td>
      <td>12.7 ± 0.1</td>
      <td>16.1 ± 0.8</td>
      <td>51.6 ± 1.8</td>
    </tr>
    <tr>
      <td>40%</td>
      <td>Dense</td>
      <td>46.3 ± 11.6</td>
      <td>17.3 ± 0.6</td>
      <td>13.8 ± 5.1</td>
      <td>26.1 ± 2.4</td>
      <td>73.3 ± 0.4</td>
      <td>68.2 ± 2.5</td>
      <td>80.6 ± 0.9</td>
      <td>90.9 ± 0.2</td>
      <td>39.2 ± 9.8</td>
      <td>16.8 ± 0.7</td>
      <td>15.8 ± 0.5</td>
      <td>39.9 ± 1.6</td>
    </tr>
    <tr>
      <td></td>
      <td>Prune</td>
      <td>97.8 ± 0.2</td>
      <td>19.6 ± 0.0</td>
      <td>74.8 ± 0.7</td>
      <td>24.4 ± 0.3</td>
      <td>93.1 ± 0.1</td>
      <td>50.8 ± 0.1</td>
      <td>86.9 ± 0.2</td>
      <td>11.8 ± 0.1</td>
      <td>95.3 ± 0.4</td>
      <td>19.6 ± 0.0</td>
      <td>21.9 ± 0.9</td>
      <td>55.1 ± 3.2</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow</td>
      <td>14.1 ± 0.4</td>
      <td>12.8 ± 0.3</td>
      <td>33.8 ± 3.0</td>
      <td>18.8 ± 0.9</td>
      <td>70.4 ± 0.1</td>
      <td>50.7 ± 0.1</td>
      <td>74.7 ± 0.1</td>
      <td>18.2 ± 0.1</td>
      <td>21.1 ± 1.4</td>
      <td>13.4 ± 0.1</td>
      <td>21.9 ± 0.7</td>
      <td>51.9 ± 2.4</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Rand. Smooth-Leaky</td>
      <td>96.8 ± 0.1</td>
      <td>19.6 ± 0.0</td>
      <td>65.0 ± 0.8</td>
      <td>22.4 ± 0.8</td>
      <td>83.5 ± 0.1</td>
      <td>50.7 ± 0.1</td>
      <td>80.0 ± 0.0</td>
      <td>15.4 ± 0.0</td>
      <td>84.1 ± 0.9</td>
      <td>19.5 ± 0.1</td>
      <td>24.2 ± 0.5</td>
      <td>45.9 ± 1.7</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+TwoSpeed</td>
      <td>14.6 ± 0.9</td>
      <td>12.9 ± 0.4</td>
      <td>30.2 ± 2.0</td>
      <td>17.7 ± 0.9</td>
      <td>70.6 ± 0.1</td>
      <td>50.7 ± 0.1</td>
      <td>75.0 ± 0.1</td>
      <td>18.2 ± 0.1</td>
      <td>21.0 ± 1.2</td>
      <td>13.4 ± 0.2</td>
      <td>21.2 ± 0.8</td>
      <td>51.1 ± 2.2</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Moment Transplant</td>
      <td>17.4 ± 4.0</td>
      <td>13.6 ± 0.7</td>
      <td>38.3 ± 4.4</td>
      <td>18.2 ± 1.7</td>
      <td>70.2 ± 1.7</td>
      <td>50.7 ± 0.1</td>
      <td>75.0 ± 0.1</td>
      <td>18.3 ± 0.1</td>
      <td>22.4 ± 0.8</td>
      <td>14.4 ± 0.1</td>
      <td>21.5 ± 0.6</td>
      <td>53.0 ± 2.2</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+GradMax</td>
      <td>13.6 ± 0.5</td>
      <td>12.5 ± 0.3</td>
      <td>30.0 ± 1.7</td>
      <td>17.0 ± 0.9</td>
      <td>70.5 ± 0.2</td>
      <td>50.6 ± 0.1</td>
      <td>74.6 ± 0.1</td>
      <td>18.3 ± 0.1</td>
      <td>22.2 ± 1.4</td>
      <td>13.5 ± 0.2</td>
      <td>21.9 ± 0.8</td>
      <td>52.3 ± 2.4</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Net2Wider</td>
      <td>18.0 ± 1.9</td>
      <td>13.4 ± 0.4</td>
      <td>31.8 ± 3.0</td>
      <td>16.9 ± 1.0</td>
      <td>68.5 ± 0.2</td>
      <td>50.7 ± 0.1</td>
      <td>74.5 ± 0.1</td>
      <td>18.1 ± 0.1</td>
      <td>17.5 ± 1.0</td>
      <td>12.7 ± 0.2</td>
      <td>16.2 ± 1.6</td>
      <td>54.0 ± 1.6</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>Dense</td>
      <td>46.3 ± 11.6</td>
      <td>17.3 ± 0.6</td>
      <td>13.8 ± 5.1</td>
      <td>26.1 ± 2.4</td>
      <td>73.3 ± 0.4</td>
      <td>68.2 ± 2.5</td>
      <td>80.6 ± 0.9</td>
      <td>90.9 ± 0.2</td>
      <td>39.2 ± 9.8</td>
      <td>16.8 ± 0.7</td>
      <td>15.8 ± 0.5</td>
      <td>39.9 ± 1.6</td>
    </tr>
    <tr>
      <td></td>
      <td>Prune</td>
      <td>97.9 ± 0.2</td>
      <td>19.6 ± 0.0</td>
      <td>74.9 ± 0.7</td>
      <td>24.1 ± 0.5</td>
      <td>93.8 ± 0.1</td>
      <td>50.7 ± 0.1</td>
      <td>87.2 ± 0.2</td>
      <td>11.7 ± 0.1</td>
      <td>95.5 ± 0.4</td>
      <td>19.6 ± 0.0</td>
      <td>21.9 ± 0.4</td>
      <td>57.8 ± 2.3</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow</td>
      <td>15.4 ± 1.6</td>
      <td>13.0 ± 0.3</td>
      <td>34.4 ± 2.6</td>
      <td>17.7 ± 0.9</td>
      <td>71.3 ± 0.2</td>
      <td>50.7 ± 0.1</td>
      <td>75.7 ± 0.0</td>
      <td>18.3 ± 0.1</td>
      <td>23.2 ± 1.6</td>
      <td>13.4 ± 0.2</td>
      <td>21.5 ± 0.6</td>
      <td>52.0 ± 2.3</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Rand. Smooth-Leaky</td>
      <td>97.1 ± 0.1</td>
      <td>19.6 ± 0.0</td>
      <td>67.9 ± 0.5</td>
      <td>22.6 ± 0.5</td>
      <td>84.2 ± 0.1</td>
      <td>50.7 ± 0.1</td>
      <td>81.3 ± 0.0</td>
      <td>15.2 ± 0.0</td>
      <td>85.9 ± 0.7</td>
      <td>19.6 ± 0.0</td>
      <td>24.7 ± 0.8</td>
      <td>46.9 ± 2.8</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+TwoSpeed</td>
      <td>15.8 ± 2.3</td>
      <td>13.0 ± 0.5</td>
      <td>29.3 ± 0.9</td>
      <td>12.5 ± 1.1</td>
      <td>71.2 ± 0.2</td>
      <td>50.6 ± 0.1</td>
      <td>75.6 ± 0.1</td>
      <td>18.4 ± 0.0</td>
      <td>23.6 ± 1.3</td>
      <td>13.6 ± 0.2</td>
      <td>21.2 ± 0.9</td>
      <td>51.3 ± 2.4</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Moment Transplant</td>
      <td>17.3 ± 2.4</td>
      <td>13.6 ± 0.6</td>
      <td>31.6 ± 0.5</td>
      <td>13.3 ± 0.7</td>
      <td>70.8 ± 0.2</td>
      <td>50.8 ± 0.1</td>
      <td>76.1 ± 0.1</td>
      <td>18.4 ± 0.0</td>
      <td>23.7 ± 1.4</td>
      <td>13.6 ± 0.1</td>
      <td>22.1 ± 0.6</td>
      <td>53.5 ± 2.0</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+GradMax</td>
      <td>13.8 ± 0.4</td>
      <td>12.7 ± 0.2</td>
      <td>33.7 ± 3.6</td>
      <td>18.4 ± 1.2</td>
      <td>71.5 ± 0.2</td>
      <td>50.6 ± 0.1</td>
      <td>75.5 ± 0.1</td>
      <td>18.3 ± 0.1</td>
      <td>24.3 ± 1.6</td>
      <td>13.6 ± 0.2</td>
      <td>21.8 ± 1.0</td>
      <td>51.0 ± 2.8</td>
    </tr>
    <tr>
      <td></td>
      <td>Grow+Net2Wider</td>
      <td>14.8 ± 1.7</td>
      <td>12.7 ± 0.3</td>
      <td>30.9 ± 2.7</td>
      <td>17.2 ± 0.6</td>
      <td>69.6 ± 0.2</td>
      <td>50.7 ± 0.1</td>
      <td>75.4 ± 0.1</td>
      <td>18.4 ± 0.1</td>
      <td>18.5 ± 1.2</td>
      <td>12.6 ± 0.2</td>
      <td>16.9 ± 1.1</td>
      <td>47.2 ± 2.7</td>
    </tr>
  </tbody>
</table>

Table 14: Continual-learning summary (cycle). For each dataset we report Avg. Acc. and Early Task TAA. Within each compactness block, the best value for each dataset-metric pair is shown in bold. Dense is shown as the full-capacity reference within each compactness block, while non-dense methods are compared at matched compactness. Dense uses a dataset-specific fixed learning rate; all other methods select the best learning rate by Avg. Acc. among {0.01, 0.001, 0.0001}.
