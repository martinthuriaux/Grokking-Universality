# Grokking Universality

**What does it mean for two neural networks to be "the same"?**

When a small transformer groks modular addition, it famously learns a Fourier-based algorithm: it embeds numbers as points on circles of various frequencies, adds them by rotation, and reads the answer off with trig identities ([Nanda et al., 2023](https://arxiv.org/abs/2301.05217)). But that result was established by reverse-engineering individual models. This project asks the population-level question: **across 100 independently trained models, what about the learned solution is universal, and what is an arbitrary accident of the random seed?**

The answer turns out to be quite neat, it turns out the *algorithm* is universal, the *frequency budget* is nearly universal, and the *choice of frequencies* is essentially a uniform random draw.

## Setup

- **Task:** modular addition, (a + b) mod 53, presented as sequences `[a, b, =]`.
- **Model:** 1-layer transformer (d_model 128, 4 heads, d_mlp 512, ReLU, no LayerNorm, no biases), trained with AdamW and strong weight decay until it groks.
- **Sweep:** 100 random seeds, each controlling both the train/test split and the weight initialisation. Every run saves its seed, weights, training curves, and detected key frequencies, so all results are exactly reproducible.

## The measurement: a causal key-frequency detector

The whole study rests on reliably identifying each model's key frequencies without hand-inspecting 100 networks, so the detector went through three iterations:

1. **v1 — spectral threshold:** flag frequencies whose embedding Fourier power exceeds a multiple of the noise floor. This detector was *very* sensitive to the threshold choice.
2. **v2 — power fraction:** flag frequencies carrying more than a fixed fraction of the total non-constant embedding power. More stable, but still purely correlational as it counts frequencies that are *present*, not frequencies that *matter*, ie harmonic frequencies.
3. **v3 — two-stage causal detector (used for all results):** v2 at a deliberately generous threshold proposes candidates; each candidate is then ablated from the model's logits in 2D Fourier space over all 53² inputs, and is retained only if removing it drops accuracy by more than 1 percentage point. As a sufficiency check, keeping *only* the retained set must preserve the model's performance.

The causal filter matters a lot, and is what Neel Nanda used in his own detector. v3 changed the frequency signature on 12 of the 100 models relative to v2, typically by rejecting spectrally prominent frequencies, such as harmonics of a true key frequency, whose ablation costs exactly nothing. 

## Results across 100 grokked models

| Hypothesis | Result |
|---|---|
| **H1 — The fourier structure is shared amongst all seeds** | Supported. Mean accuracy over all 2,809 input pairs: 99.8%. On average 92.0% of embedding Fourier power sits in each model's key frequencies, 90.1% of logit power lies on the k_a = k_b "sum" diagonal, and 89.1% sits specifically at the detected key frequencies. |
| **H2 — The chosen frequencies vary amongst seeds** | Strongly supported. 99 distinct frequency signatures among 100 models; only 1 of 4,950 model pairs matched exactly (0.02%); median pairwise Jaccard similarity 0, mean 0.077. |
| **H3 — The number of frequencies vary amonst seeds** | Supported. 67 models use three frequencies, 30 use four (97% in {3, 4}); mean 3.30, mode 3, sd 0.52. The mode is stable across a 6x range of the ablation threshold. |
| **H4 — Some frequencies are preferred over others** | Not supported. Permutation test against uniform selection: p = 0.45; after multiple-comparison correction, no individual frequency is significantly preferred or avoided. |
| **H5 — The implementation predicts training dynamics** | Not supported. Frequency count vs grokking epoch: Spearman rho = 0.156, p = 0.12; no clear relationship with final weight norm. |

## Repository structure

| File | Contents |
|---|---|
| `00_reproduce.ipynb` | Single-model reproduction: train one model to grok and reverse-engineer its Fourier algorithm, validating the pipeline against known ground truth. |
| `01_seed_sweep.ipynb` | The 100-seed sweep, plus the development and validation of detectors v1 → v3 and the regeneration of every run record with causal frequency data. |
| `02_analysis.ipynb` | The population-level tests of H1–H5, with statistics, figures, and conclusions. |

Each notebook runs top-to-bottom in Google Colab (a T4 GPU is sufficient); run records are saved to Drive under `grokking_universality/runs`, one file per seed containing the weights, curves, and frequency diagnostics.

## Conclusion

Taken together, the five hypotheses paint a three-level picture of what training actually determines.

**At the level of the algorithm, training is deterministic.** All 100 models, from 100 independent initialisations and data splits, learned the same Fourier-clock solution, which is verified not merely by spectral signatures but also causally: in every model, the detected frequency set is sufficient (keeping only those frequencies preserves accuracy) and necessary (ablating them destroys it). However, it's important to note that since a deepdive is not done into every single seed, we do not have *definitive* proof that every model follows the same algorithm, but we do have heavy indication.

**At the level of resources, training is tightly constrained.** Although the models were free in principle to use anywhere from one to twenty-six frequencies, 97% used exactly three or four. The frequency *budget* behaves like a property of the task-architecture-optimiser combination, not of the seed. Whatever trade-off governs it, redundancy for robust readout against the cost of maintaining more clocks under weight decay, it resolves to the same answer almost every time.

**At the level of implementation, training is a coin flip.** *Which* frequencies fill those three or four slots is statistically indistinguishable from uniform random selection (H4), the resulting signatures are almost never repeated and mostly don't even overlap (H2), and the particular draw has no detectable consequence for how fast the model groks or where it ends up in weight space (H5). The 26 available clocks are perfectly symmetric under the task, and nothing in the architecture or optimiser breaks that symmetry, only initialisation does.

The upshot for interpretability is a precise, quantified version of the universality claim: **these 100 networks are mechanistically identical at the level of the algorithm and mechanistically disjoint at the level of its parameters.** Any two of them have entirely different weight matrices, almost certainly different frequency sets, and often not a single Fourier clock in common, yet they are running the same computation. "Same model" is therefore only a meaningful notion at the algorithmic level of description: an interpretability finding of the form *"this network computes modular addition with clocks at frequencies 6, 8 and 19"* does not transfer even to a re-run of the identical training script, while the finding *"this network computes modular addition with 3–4 Fourier clocks"* transfers to every member of the population..

### Limitations

The causal ablations act on the output logits in Fourier space, so they establish which frequencies the learned *function* depends on; a fully internal verification (e.g. projecting frequencies out of the embeddings and activations) was performed on a handful of seeds rather than all 100. The 3-vs-4 frequency boundary is mildly sensitive to the ablation threshold, though the concentration on {3, 4} is not. And with 330 frequency selections spread over 26 frequencies, H4's test has limited power against small preferences, the claim is uniformity at the resolution of this sweep, not perfect symmetry.
