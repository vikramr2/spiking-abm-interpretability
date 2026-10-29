# Spiking-ABM Interpretability Spec

Status: planning. No code yet. The first plan is in [`iterations/000-plan.md`](iterations/000-plan.md), and changes are logged in [`CHANGELOG.md`](CHANGELOG.md).

Items marked **Proposal** are candidate approaches that have not been decided. **Open** items are unresolved. Numbers stay preliminary until they are reproduced through the repo CLI.

## 1. Goal and hypotheses

**Question.** Does modelling a small transformer as a spiking network, and then as an agent-based model (ABM) of spiking neurons, tell us anything about its mechanisms that standard mechanistic interpretability does not, and do the findings line up with what is already known?

**Why it could work.** Standard interpretability has ground truth on toy tasks: specific circuits are known to implement induction and modular addition. A spiking model gives three things an ANN does not:

- a **causal event log**, in which every spike has an upstream cause;
- **time within a forward pass**, as the network integrates over $T$ simulation steps;
- a **natural cost**, the spike count.

The sibling project [map-stdp](https://github.com/vikramr2/map-stdp) treats communities of a spiking network's flow graph as states, and description length as a cost. Here the same tools are pointed at a network whose functional modules are already known.

**Why it could fail.** A spiking network converted with a rate code computes the same function as the ANN, because spike counts approximate activations. Anything found in it is the ANN's circuit read out more noisily. Any positive result therefore has to come from what the spiking model does differently (timing, sparsity, winner-take-all attention), and every analysis runs on the ANN too (§6).

**Hypotheses.** Each one has a falsification criterion, fixed before the run.

1. **H1, universality.** A spiking transformer trained directly with surrogate gradients, rather than converted, forms the same induction circuit as the ANN: a previous-token head in layer 0 composed with an induction head in layer 1, with a phase change in the loss. *Falsified if* no head in the spiking model passes the induction-score and ablation criteria of iteration 000 on at least 3 of 5 seeds, while the ANN passes on at least 4 of 5.
2. **H2, communities are circuits.** Map-equation communities of the spike-transfer graph (§5.1) recover the known circuit better than the layer partition alone, and better than the same analysis on ANN activations. *Falsified if* the adjusted mutual information (AMI) with the circuit labels is not above the layer-only baseline, or is not above the ANN control.
3. **H3, temporal logit lens.** In the spiking model, the step at which the correct token's readout first crosses threshold orders the inputs by difficulty, and it moves earlier when the induction head is active. *Falsified if* the first-crossing step does not correlate with the ANN's per-token loss (Spearman $\rho \le 0.2$).
4. **H4, cost tracks description length.** This is secondary, and mirrors map-stdp's side hypothesis. The spike count per head tracks that head's contribution to the description length of the flow. *Falsified if* the rank correlation across heads is $\le 0.3$.

## 2. Models

There are four variants. Each one shares the architecture and the task of the one before.

| ID | Model | Role |
| --- | --- | --- |
| A | ANN, attention-only transformer | The reference. Its circuits are identified with standard methods (§5). |
| B | Model A converted to an SNN with a rate code | **Null control.** It should reproduce A's circuit trivially. Any "spiking-only" finding that also appears in B is a conversion artefact or a restatement of A. |
| C | Spiking transformer trained directly (snnTorch, surrogate gradients) | The main object of study (H1–H4). |
| D | Model C rebuilt in SuperNeuroABM | Event-level causality and heterogeneous neurons (§3.2). Built only if a question needs it. |

**Architecture (Proposal).** Attention-only, 2 layers, 4 heads per layer, $d_{model} = 64$, $d_{head} = 16$, context 64, vocabulary 64, with learned positional embeddings and no MLPs and no LayerNorm. Attention-only models with two layers are the smallest ones known to form induction heads (Olsson et al., 2022). Leaving out LayerNorm removes an operation that spiking neurons implement poorly.

### 2.1 Attention in spikes

This is the hard part. The three non-native operations, and how model C handles each:

| Operation | Problem | Handling in C |
| --- | --- | --- |
| $QK^\top$ | It multiplies two activity patterns; no synapse does that. | **Proposal:** spike-driven self-attention in the style of Spikformer (Zhou et al., 2023). $Q$, $K$ and $V$ are spike tensors, so $QK^\top$ is a count of coincidences, with a causal mask (Spikformer has none). |
| Softmax | It needs a global normalisation. | **Proposal:** no softmax, as in Spikformer, with a fixed scale $s$. **Alternative:** winner-take-all through an inhibitory pool, which is closer to hard attention and is a natural ABM component (§3.2). |
| Residual sum | It is fine in rates, but spikes saturate. | The residual stream is the membrane input, not spikes: each layer adds current to the next layer's LIF neurons. |

**Open:** whether to make the attention pattern itself spiking (a spike per query–key coincidence), or to keep it as a real-valued score that drives LIF neurons. The first is more faithful and has an event log for each attention edge. The second is easier to train.

## 3. Simulators

### 3.1 snnTorch (models B and C)

- **Training:** surrogate gradients (fast sigmoid) through `snn.Leaky` LIF neurons, over $T$ steps per forward pass. The default is $T = 8$, with a sweep over 4, 8 and 16.
- **Recording:** spikes and membrane potentials for every layer and every step, for analysis.
- **Conversion (model B):** rate coding with the weights of A and thresholds normalised per layer from the 99.9th percentile of activations. **Proposal.**

### 3.2 SuperNeuroABM (model D)

[SuperNeuroABM](https://github.com/ORNL/superneuroabm) (ORNL, BSD-3, `pip install superneuroabm`, built on SAGESim) treats each neuron and each synapse as an agent. Points that matter here:

- **Neurons:** built-in LIF and Izhikevich somas, with per-agent parameters (`overrides={"hyperparameters": ...}`). Custom somas, synapses and learning rules can be registered.
- **Synapses:** single exponential is the only built-in synapse.
- **Recording:** `get_spike_times` per soma. With `enable_internal_states_tracking=True` there is per-tick state, at about 1 GB per 10k neurons per 1M ticks.
- **Hardware:** it **needs a GPU** (CUDA or ROCm, through CuPy). MPI is needed only for multiple GPUs.
- **Network building:** there is no import path from PyTorch. The network is rebuilt from exported weights with `create_soma` and `create_synapse`, or with `create_from_lists`.
- **Not yet checked:** the README claims bounded, quantized and memristive STDP, but the functionality guide documents only exponential pair-wise STDP. Model D needs no plasticity, so this does not block anything.

What D adds over C:

1. **Per-spike causality.** Each spike's presynaptic causes can be recovered, which gives an exact attribution graph at the spike level, comparable to attribution graphs over transcoders in ANNs.
2. **Heterogeneous neurons.** For example, inhibitory fast-spiking Izhikevich pools for winner-take-all attention, and regular-spiking excitatory neurons elsewhere.
3. **Cheap interventions.** An agent can be clamped, silenced or delayed in isolation.

**Cost.** Model C has about $2 \times 64 \times 64 \times 3$ neurons for Q, K and V across 64 positions, plus coincidence detectors if §2.1 is made spiking. That is tens of thousands of agents, which fits one GPU. Coincidence detectors grow as $L^2 \cdot H$, so context 64 is about the ceiling.

**Open:** whether D answers any question that C's recorded spikes cannot. It is built only if M3 shows that one does (§9).

## 4. Tasks

| Task | Known mechanism | Use |
| --- | --- | --- |
| **T1, induction.** Random token sequences in which the second half repeats the first. Loss is measured on the repeated half. | A previous-token head, then an induction head through K-composition (Elhage et al., 2021; Olsson et al., 2022). | H1–H4; iteration 000. |
| **T2, modular addition** ($a + b \bmod p$, $p = 113$). | Fourier features, with logits built from trigonometric identities (Nanda et al., 2023). | Later. It needs an MLP, so the spiking MLP needs a design first. |
| **T3, toy superposition.** Sparse features compressed into fewer dimensions. | Superposition geometry (Elhage et al., 2022). | Later: does thresholding change superposition? The direction of the effect is **Open**. |

## 5. Analyses

The ANN methods are standard and are applied to A, B and C alike, with C's spike counts or membrane currents in place of activations.

- **Head scores** (Olsson et al., 2022). The previous-token score is the attention from position $t$ to $t - 1$. The induction score is the attention from the second occurrence of a token to the token after its first occurrence. In spiking models, attention is the coincidence count normalised per query.
- **Ablation:** zero or mean ablation of each head, measured as the change in loss on the repeated half.
- **Path patching:** for K-composition, patch the output of layer-0 heads into the keys of layer-1 heads only.
- **Logit lens:** decode the residual stream after each layer through the unembedding.

### 5.1 Spike-transfer graph and communities

These analyses are new to this project.

**Nodes** (**Proposal**) are (layer, head, position) units, about 512 of them. The neuron-level graph is kept for model D.

**Edges.** In snnTorch, all layers update within one simulation step, so a hop is a layer boundary, not a step. The weight from unit $j$ to unit $i$ is the covariance pairing count

$$
\kappa_{ij} = \sum_{t} \left( s_i(t) - \bar{s}_i \right) \left( s_j(t) - \bar{s}_j \right) \qquad \text{(1)}
$$

over steps $t$ and over a batch of sequences. Here $s_i(t)$ is the spike count of unit $i$ at step $t$, and $\bar{s}_i$ is its mean. Edges are kept only where there is a structural path from $j$ to $i$, through an attention edge or the residual stream. Negative values are clipped to 0. This is map-stdp's covariance count; in map-stdp the hop is one simulation step instead.

**Communities.** Run Infomap (the map equation) on the directed weighted graph. Score the partition with AMI against two labelings: the circuit labels from the ablations (previous-token, induction, other) and the layer labels.

**Controls:**

- the same graph built from ANN activation covariance (model A);
- model B;
- a degree-preserving shuffle of the graph.

**Temporal logit lens (H3).** For each target token, record the first step $t$ at which the readout neuron of the correct token leads every other readout neuron by a fixed margin. This is the analogue of map-stdp's race readout.

**Cost (H4).** The cost of a head is its total spike count per sequence. Its description-length contribution is the change in map-equation code length when the head's units are removed from the graph.

## 6. Controls and validity

- **Every spiking analysis also runs on model A and model B.** A finding counts as spiking-specific only if it appears in C and not in A or B.
- **Seeds:** 5 per model (0–4) for every reported number.
- **Fixed thresholds:** the H1–H4 falsification criteria are set before the run, and changes to them are logged in the CHANGELOG with a reason.
- **Conversion error is reported** for B (the gap in loss to A) so that artefacts of B are not read as findings.

## 7. Relation to known results

| Known result | Prediction here |
| --- | --- |
| Induction heads and the phase change (Olsson et al., 2022) | H1: they appear in C. |
| K-composition between layer-0 and layer-1 heads (Elhage et al., 2021) | Path patching in C shows the same composition. |
| Trained networks are more clusterable than random ones (Filan et al., 2021); imposed modularity aids interpretability (Liu, Gan and Tegmark, 2023) | H2: communities of the spike flow line up with the circuit. |
| Residual-stream predictions sharpen layer by layer (the logit lens) | H3: in spikes they also sharpen step by step within a layer. |
| Attention is retrieval in a modern Hopfield network (Ramsauer et al., 2020) | Winner-take-all attention (§2.1) is a hard-retrieval limit; compare C with and without it. |
| Transformers relate to hippocampal place and grid codes (Whittington, Warren and Behrens, 2022) | Out of scope for M1–M3; noted for interpretation. |

## 8. Environment

**Proposal:** Python 3.11, PyTorch (CPU is enough for models A–C at this size), snnTorch, TransformerLens (for model A's hooks and head scores), infomap, numpy and scipy. SuperNeuroABM and CuPy are added only for model D, on a GPU machine. All of these go in `environment.yml`, with versions pinned at M0.

## 9. Milestones

| Milestone | Content | Exit criterion |
| --- | --- | --- |
| M0 | Environment, repo layout, CLI, seeds, fixed-seed regression test | `pytest` passes; one training run of A reproduces bit for bit. |
| M1 | Model A on T1; circuit identification | The induction circuit is found on at least 4 of 5 seeds (iteration 000). |
| M2 | Models B and C on T1; head scores, ablation and patching | H1 is decided. |
| M3 | Spike-transfer graph, Infomap, temporal logit lens and cost | H2–H4 are decided. |
| M4 | Model D in SuperNeuroABM, only if M3 names a question that needs it | That question is answered with per-spike attribution. |
| M5 | T2 and T3 | Spiking MLP design; superposition under thresholding. |

## 10. Open questions

1. Make attention spiking (coincidence detectors) or keep it real-valued (§2.1)?
2. Softmax replacement: a fixed scale, or winner-take-all inhibition?
3. Node granularity for §5.1: (layer, head, position) units, or individual neurons?
4. Does model C train at all on T1 without LayerNorm and softmax? If not, the fallback is to put LayerNorm back as a real-valued operation and record it as an abstraction.
5. Which description length to use beyond the map equation (map-stdp SPEC §6 lists alternatives)?

## References

- Elhage, N., et al. (2021). A mathematical framework for transformer circuits. Transformer Circuits Thread.
- Elhage, N., et al. (2022). Toy models of superposition. Transformer Circuits Thread.
- Filan, D., et al. (2021). Clusterability in neural networks. arXiv:2103.03386.
- Liu, Z., Gan, E., & Tegmark, M. (2023). Seeing is believing: brain-inspired modular training for mechanistic interpretability. arXiv:2305.08746.
- Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability. ICLR.
- Olsson, C., et al. (2022). In-context learning and induction heads. Transformer Circuits Thread.
- Ramsauer, H., et al. (2020). Hopfield networks is all you need. arXiv:2008.02217.
- Whittington, J. C. R., Warren, J., & Behrens, T. E. J. (2022). Relating transformers to models and neural representations of the hippocampal formation. ICLR.
- Zhou, Z., et al. (2023). Spikformer: when spiking neural network meets transformer. ICLR.
- Zhu, R.-J., et al. (2023). SpikeGPT: generative pre-trained language model with spiking neural networks. arXiv:2302.13939.
