# FreeBuff v4 — Defense / Viva Question Bank with Model Answers

**45 questions in six groups**, ordered the way a panel typically probes:
architecture → training → inference & explainability → clinical safety →
evaluation → ethics & regulation. Each answer is written the way you should
deliver it verbally: a direct opening sentence, then the technical substance.

---

## Group A — Architecture & Model Design

**A1. Walk me through your architecture end-to-end.**
FreeBuff v4 is a hybrid dual-branch network. Branch A is an ImageNet-pretrained
ResNet50 with a CBAM attention block on top of `layer4`, global-average-pooled
and projected to a 256-d spatial embedding. Branch B is a handcrafted texture
descriptor: a 2048-d HOG vector reduced by PCA (fitted on the training split
only) to 288 dims and embedded by an MLP to 512 dims. The two embeddings are
concatenated (768-d) and passed through a small fusion head (512 → 256 → 4)
producing logits over Glioma, Meningioma, No_Tumor and Pituitary.

**A2. Why combine a deep CNN with handcrafted HOG features?**
Two complementary inductive biases. The CNN learns texture/shape features
end-to-end but needs data and can latch onto shortcuts. HOG encodes
gradient-orientation structure explicitly — a strong prior for tumour margins
and texture heterogeneity that is cheap to compute and stable across scanners.
Empirically, hybrid deep-plus-handcrafted models are more robust on small
medical datasets because the handcrafted branch supplies signal that the deep
branch would otherwise have to relearn.

**A3. What exactly does CBAM add over a plain ResNet50?**
CBAM refines features sequentially along the two axes that matter for
lesion detection. Channel attention (a shared MLP over GAP/GMP descriptors)
re-weights the 2048 channels so tumour-relevant filters dominate. Spatial
attention (a 7×7 conv over channel mean/max) produces a where-to-look map
that highlights lesion regions and suppresses background. It costs ~0.1 M
parameters, and its spatial map doubles as an attention visualisation.

**A4. Why ResNet50 and not a newer architecture like ConvNeXt or a ViT?**
Constraints, honestly weighed. ResNet50 gives the best accuracy-per-VRAM
point at 144×144 on a 4–6 GB card, has ImageNet weights that transfer
reliably to medical images, and its `layer4` feature maps are the ideal
Grad-CAM++ target. A ViT at this resolution would need heavy patch
downsampling and far more data to avoid overfitting; ConvNeXt gains are
modest and its pretrained variants are heavier. ResNet50+CBAM is the
engineering sweet spot, and I can defend it quantitatively.

**A5. Explain the tensor flow from a 144×144 input to logits.**
Input (B,3,144,144) → conv1+maxpool (B,64,36,36) → layer1 (B,256,36,36) →
layer2 (B,512,18,18) → layer3 (B,1024,9,9) → layer4 (B,2048,5,5) → CBAM
(same shape) → GAP (B,2048) → spatial MLP (B,256). In parallel, HOG 2048 →
PCA 288 → MLP (B,512). Concatenate (B,768) → fusion head → (B,4) logits.

**A6. How many parameters does your model have, and which parts dominate?**
About 27.07 M. The ResNet50 backbone dominates with ≈ 23.5 M; CBAM, the
spatial MLP, the HOG branch and the fusion head contribute the remaining
≈ 3.5 M. All are trainable; a `freeze_backbone` option exists for
transfer-learning ablations.

**A7. Why fuse by simple concatenation rather than cross-attention?**
Concatenation is a linear-time, parameter-free fusion that lets the next
linear layer learn any per-dimension weighting — at 768-d the interaction
space is small, so cross-attention would add parameters and overfitting risk
with no representational benefit. Gated or bilinear fusion are reasonable
upgrades if ablation shows the branches compete, but that is future work.

**A8. What happens at inference if the PCA artifact is missing?**
The system degrades gracefully instead of crashing. The engine validates the
PCA against the live HOG extractor dimension; on mismatch (for example the
legacy 6144-d artifact from the v1 pipeline) the HOG branch is disabled and
the model runs on the deep branch alone with a zero HOG embedding — logged,
never silent. This is deliberate: clinical availability beats a hard failure.

**A9. Why convert grayscale MRI to three channels?**
The ImageNet-pretrained conv1 expects 3-channel input. Replicating the slice
across channels preserves the pretrained filters and lets us normalise with
standard ImageNet statistics. Training conv1 from scratch for 1 channel would
discard that transfer learning, which is the single largest source of our
accuracy.

**A10. Your input is 144×144 — isn't that small for MRI?**
It is a deliberate VRAM/accuracy trade. At batch 32 with AMP on a 4 GB card,
144×144 fits comfortably while preserving lesion-scale detail after brain
cropping — the crop removes skull/sk background, so effective object
resolution is higher than the raw number suggests. Higher resolutions gave
marginal accuracy gains in exchange for batch-size and TTA-memory penalties.

---

## Group B — Training Pipeline

**B1. Why Focal Loss instead of cross-entropy?**
Focal Loss adds a modulating factor (1−p_t)^γ that down-weights easy examples
and concentrates gradient on hard, borderline cases — exactly the cases that
decide clinical error. With γ=2, an example with p_t=0.9 contributes ~1 % of
the gradient it would under CE, while a p_t=0.3 example keeps ~49 %. I also
combine it with inverse-frequency class weighting (α), so both the loss
shape and the class prior are handled.

**B2. Derive or explain the Focal Loss formula.**
FL(p_t) = −(1−p_t)^γ · log(p_t), where p_t is the probability of the true
class. I implement it through the CE expansion: CE = −log(p_t), so
FL = (1−exp(−CE))^γ · CE — numerically stable, and `alpha` class weighting
composes cleanly through `F.cross_entropy(..., weight=α)`.

**B3. Why two different learning rates?**
Transfer-learning orthodoxy. The backbone already encodes useful generic
features, so it adapts gently at 1e-4; the fusion head and HOG branch are
freshly initialised and converge faster at 1e-3. A single LR is a compromise
in both directions — it either destabilises the backbone or stalls the head.

**B4. Why AdamW over Adam or SGD?**
AdamW decouples weight decay from the gradient: decay is applied directly to
weights rather than added to gradients where it is scaled by the adaptive
per-parameter step size. The result is regularisation that behaves like true
L2 and better generalisation than Adam, with the adaptive-rate benefits SGD
lacks on heterogeneous branches.

**B5. Explain your cosine annealing schedule and its parameters.**
The LR follows η_t = η_min + ½(η_0 − η_min)(1 + cos(πt/T_max)) with T_max=60
epochs and η_min=1e-6. It decays smoothly without stage boundaries, letting
the model settle into sharper minima late in training; η_min keeps the LR
strictly positive for late refinements. ReduceLROnPlateau is implemented as
a fallback mode but cosine was the selected default.

**B6. What is AMP and why does it matter on your GPU?**
Automatic Mixed Precision runs eligible ops in fp16 (tensor cores) while
keeping sensitive accumulations in fp32. On the RTX 3050 it roughly halves
activation memory and gives ~1.6–2× throughput, which is what makes batch 32
practical during training. GradScaler handles fp16's small dynamic range:
it scales the loss up, checks gradients for inf/NaN, skips poisoned steps,
and rescales dynamically. BatchNorm statistics stay fp32 automatically.

**B7. Why is gradient clipping applied after `scaler.unscale_()`?**
Because the scaler multiplies gradients — clipping the scaled gradients
would compute a wrong, scale-dependent effective threshold. `unscale_()`
divides the gradients back to their true fp32 values first, then I clip to
max-norm 1.0, then `scaler.step()` applies the update. Order matters and
getting it wrong silently distorts training.

**B8. What is early stopping doing here, and why patience 12?**
Training halts when validation weighted F1 has not improved for 12
consecutive epochs, and the best-F1 checkpoint — not the last one — is
deployed. Patience 12 is wide enough to ride out cosine-schedule plateaus
where validation stalls before improving, but tight enough to protect GPU
budget.

**B9. Why is weighted F1 your model-selection metric rather than accuracy?**
Tumour datasets are class-imbalanced; accuracy is dominated by the majority
class and can hide complete failure on a minority class. Weighted F1
aggregates per-class precision/recall weighted by support, so a model that
sacrifices a rare class cannot look healthy. In a triage context, recall on
tumour classes is exactly what you must not sacrifice silently.

**B10. How is the PCA fitted without data leakage?**
Strictly on the training split. Validation/test HOG vectors are projected
with the already-fitted transformer. Fitting on all data would let test-set
variance inform the projection — a subtle leak that inflates reported
performance, and reviewers increasingly check for exactly this.

**B11. How do you ensure reproducibility?**
`set_all_seeds(42)` seeds Python `random`, NumPy, torch CPU and all CUDA
devices, and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`. Splits
use a fixed `random_state`; class weights are derived deterministically from
label counts; the PCA artifact and per-epoch history CSV are persisted.

**B12. What data augmentation do you use, and why none at validation?**
Training uses geometric flips/rotations and photometric transforms on the
PIL image before tensorization — MRI anatomy is flip-symmetric-ish and
scanner intensities vary, so both families add invariance for free.
Validation/test must measure the model on the *real* distribution, so they
use only deterministic resize+normalise.

---

## Group C — Inference & Explainability

**C1. What is TTA, and why does your policy use only 2 views?**
Test-Time Augmentation runs the model on several deterministic views of the
same input and averages the predictions, reducing variance near decision
boundaries — effectively a cheap ensemble.

The important caveat is that a TTA view is only useful if it stays inside the
distribution the model was trained on. An earlier policy here used 4 flips ×
4 rotations × 3 scales = 48 views, including vertical flips and 90/180/270°
rotations. Those are anatomically invalid for an axial brain MRI: no scanner
emits an upside-down brain, so the model's opinion of those views is noise
being averaged into the answer.

Measuring it on the full 1600-image held-out set (policy selected on a
held-out split of `data/Training`, then confirmed once on `data/Testing`):

| policy | views | test acc | test macro F1 |
|---|---|---|---|
| previous default | 8 | 0.8894 | 0.8888 |
| previous full grid | 48 | 0.8869 | 0.8859 |
| no TTA | 1 | 0.9513 | 0.9508 |
| hflip only (current) | 2 | **0.9556** | **0.9552** |

The 48-view grid was *worse than no TTA at all*. The shipped policy is
identity + horizontal mirror, which is the one transform justified by the
anatomy — a brain is roughly bilaterally symmetric. That change alone is
worth **6.6 accuracy points**, and it costs 2 forward passes rather than 48.

The general lesson: TTA is not free variance reduction, it is an assumption
that the augmentation preserves the label *and* the input distribution.

**C2. Why average probabilities rather than logits or majority vote?**
Probability averaging is scale-aware and robust: logits can swing widely and
a single extreme view can dominate their mean, while majority voting throws
away magnitude information. Mean softmax keeps every view contributing
proportionally to its confidence, then we renormalise.

**C3. Explain how your Grad-CAM++ is implemented and why that way.**
A forward-only hook on `layer4` captures the exact output tensor — no clone,
no detach, because the tensor must remain the same graph node the class score
flows through. I then compute d(score)/d(features) with
`torch.autograd.grad(score, features, retain_graph=True)`. I deliberately
avoid `register_full_backward_hook`: when a module's output is reshaped
downstream (view/clone/cat), that hook fires with stale gradient shapes and
produced CUDA illegal-memory-access crashes on this stack. autograd.grad is
exact, AMP-safe, and cheaper.

**C4. What does Grad-CAM++ compute mathematically?**
For target class c and layer feature map f_k with gradients g: channel
weights use second-order spatial statistics —
α_k = Σ_ij g²_ij / (Σ_ij g²_ij + 2·Σ_ij g²_ij) style normalisations — then
the heatmap is ReLU(Σ_k α_k · ReLU(Σ_ij g_ij·f_ij)), upsampled bilinearly to
input size and min-max normalised. Compared to Grad-CAM, the α-weights
handle multiple simultaneous lesion instances better instead of averaging
their gradients away.

**C5. Why must ReLU/Dropout be non-inplace for Grad-CAM to work?**
In-place operators overwrite tensor storage the autograd engine still needs
to evaluate grad-of-grad-free backward paths. Since Grad-CAM computes
d(score)/d(features) *after* the forward pass, any in-place op on the
score→features path raises "variables needed for gradient computation have
been modified by an inplace operation". Every activation in attention,
embeddings and the fusion head is constructed with inplace=False — the
network is Grad-CAM-safe by construction.

**C6. What are the safety guards around Grad-CAM++?**
Class index clamped into valid range (CUDA out-of-bounds protection);
explicit RuntimeError if features were captured under inference_mode (where
gradients are impossible); hooks removed in a finally block; contiguous
tensors enforced before CUDA element-wise ops; and a grey fallback overlay
on any internal failure so the clinical UI can never crash from explainability.

**C7. Why do you also implement Score-CAM and attention maps?**
Score-CAM is a gradient-free baseline — it masks the input with each
channel's upsampled activation and measures the score delta, softmax-weighted
across up to 64 channels. Cross-checking a gradient method against a
masking method guards against explanation artifacts; the CBAM spatial map
additionally shows what the attention module itself learned. `explain_maps`
dispatches between them, auto mode preferring Grad-CAM++ with Score-CAM
fallback.

**C8. Describe your sensitivity-first decision rule.**
Alongside the argmax I compute the aggregate tumour probability — the sum of
Glioma+Meningioma+Pituitary probabilities. If it exceeds 30 % while the argmax
says No_Tumor, the system surfaces the highest-probability tumour class plus
a review alert instead of a bare "no tumour". Crucially, raw probabilities
are always displayed unchanged — the rule changes what is surfaced, never
what is hidden.

**C9. What is the temperature parameter doing in your softmax?**
Logits are divided by τ before softmax. τ>1 softens overconfident
distributions, τ<1 sharpens. It defaults to 1.0 (identity); `fit_temperature`
fits it post-hoc on held-out logits with LBFGS to minimise NLL — the standard
calibration fix when ECE shows overconfidence. It changes no weights.

**C10. How does multi-modal input work with a single 2D classifier?**
The three preprocessed sequences are averaged pixel-wise into one composite
144×144×3 image — a lightweight fusion that lets one 2D classifier serve the
multi-modal protocol within VRAM limits. Missing sequences are replicated
from a present one with an explicit log line. A per-branch encoder with
cross-modal attention is the principled upgrade and is scoped as future work.

**C11. What are the limits of what your explanations prove?**
Grad-CAM++ shows *where* the model looked, not *whether* it reasoned
correctly — a plausible heatmap is not evidence of a correct decision. It is
a debugging and trust-verification aid: if attention falls on skull
artifacts or corners instead of the lesion, I know to distrust that
prediction regardless of confidence.

---

## Group D — Clinical Safety & the Dashboard

**D1. What happens when confidence drops below 70 %?**
The low-confidence safeguard fires: a red banner marks the finding as
provisional and mandates secondary radiologist review; the narrative gains a
hardened caution line; the PDF report carries the flag and strengthened
language. The result is still shown — hiding it would be dangerous — but it
is impossible to miss that it must not be acted on alone.

**D2. Why 70 %?**
Three-way reasoning. Empirically, per-class error rates rise sharply below
that confidence on my validation curves. Operationally, 70 % is a clean,
auditable threshold clinicians understand. Ethically, it errs toward
conservative — the cost of an unnecessary review is minutes; the cost of a
missed tumour flagged confidently-wrong is catastrophic.

**D3. Reliability bands — what are they for?**
Four bands (High ≥90, Moderate ≥75, Low ≥60, Very Low <60) map a number to
an operational posture: green confirmation with a verification reminder,
amber caution, caution-with-review, red. Clinicians think in postures, not
decimals; the bands also drive consistent colouring across dashboard, chart
and PDF.

**D4. How does the system handle missing sequences?**
Any subset of T1c/T2/FLAIR is accepted; missing modalities are replicated
from a present one and the substitution is logged and reported in the result
payload and the dashboard — never silent. If nothing usable arrives, the
analysis refuses to run rather than fabricate output.

**D5. What does the PDF report contain and why does that matter?**
Cover/patient block, primary finding with confidence and band, the full
4-class confidence spectrum, WHO staging block, Grad-CAM++ overlays, the
generated clinical reasoning, and the complete regulatory disclaimer. The
PDF is the auditable record: what the model said, how confident, what it
looked at, and what the limitations were at that moment — checksummed for
tamper-evidence.

**D6. Walk me through the staging logic.**
The classifier predicts a tumour *family*. The staging engine then reports
that family's *possible WHO grade range* with typical imaging features,
malignancy profile and management pathways — it never certifies a single
grade. Definitive WHO grading requires histopathology and molecular markers
(IDH, 1p/19q, MGMT); the report says so verbatim. This is the legally and
clinically defensible framing.

**D7. What prevents a user from over-trusting the system?**
Layered friction: persistent disclaimer banners, band colouring, the <70 %
safeguard, the tumour-probability alert, staging framed as ranges, and a
report that repeats the disclaimer. The design goal is that no single
rendering surface shows the model's output without its uncertainty attached.

**D8. How does the architecture serve the RTX 3050's constraints?**
AMP halves activation memory; batch 32 at 144×144 fits comfortably;
TTA memory is bounded by processing views sequentially rather than batching
them; Score-CAM subsamples channels; and Grad-CAM++ runs one image at a
time with hooks removed in finally. Everything degrades by flag, not by crash.

---

## Group E — Evaluation & Results

**E0. What does your system actually achieve?**
**0.9453 accuracy, 0.9431 macro F1**, on the 1,189 held-out images we verified
are absent from the training set.

Quote that number, not the 0.9550 the full test set gives — and say why
unprompted, because it is the strongest thing in the project:

| Evaluation set | Images | Accuracy | Macro F1 |
|---|---|---|---|
| **Uncontaminated** | **1,189** | **0.9453** | **0.9431** |
| Full `data/Testing` | 1,600 | 0.9550 | 0.9547 |

We audited our own benchmark and found **25.7% of `data/Testing` is a duplicate
or near-duplicate of something in `data/Training`** — including **301 of the
400 No_Tumor images (75%)**. Scoring on those rewards memorisation, so they are
excluded from the headline.

Per class on the uncontaminated set:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Glioma | 0.9942 | 0.8779 | 0.9324 | 393 |
| Meningioma | 0.8773 | 0.9630 | 0.9181 | 297 |
| No_Tumor | 0.8919 | **1.0000** | 0.9429 | 99 |
| Pituitary | 0.9728 | 0.9850 | 0.9789 | 400 |

Points worth making unprompted:

1. **No_Tumor recall is still 1.0000** after removing the contamination — no
   healthy scan was called a tumour. That is the direction you want errors
   *not* to go, and it survives the audit.
2. The residual error is **Glioma → Meningioma**, the clinically plausible
   confusion, not a random failure.
3. Our earlier reported **0.9828 No_Tumor F1 was the most inflated figure**:
   only 99 of its 400 test images were ever a real test.

**"So is your model worse than you thought?"** No — the *measurement* was
optimistic, the model is unchanged. We retrained from scratch on a
de-duplicated dataset and scored both models on the 1,189 images neither was
trained on: 0.9453 vs 0.9445 accuracy. Statistically equivalent. The baseline's
apparent lead on the full test set was memorisation, not capability.

Reproduce both figures with one command:

```bash
python audit_leakage.py --evaluate
```

**E1. Summarise your evaluation methodology.**
Stratified 60/20/20 split, model selection on validation weighted F1, and
the test set touched exactly once per run. On test: accuracy, per-class
precision/recall/F1, macro and weighted F1, one-vs-rest ROC-AUC with micro
average, confusion matrices (counts and normalized), confidence histogram,
and a misclassification ledger with Grad-CAM++ exports for the hardest
failures.

**E2. Where does the model fail, and why does that matter?**
The persistent confusions are glioma-vs-meningioma on dural-based gliomas
and small-meningioma-vs-pituitary on sellar lesions — anatomically plausible
errors a junior reader would also make. I surface them rather than hide
them: the misclassification ledger with Grad-CAM++ overlays shows whether
failures are lesion-driven (hard cases) or artifact-driven (preprocessing
problems), which demand different fixes.

**E3. How do you know your accuracy isn't leakage?**
Three controls: PCA fitted on training data only; augmentation and class
weights derived from the training split alone; and — most importantly —
patient-level splits would be the gold standard, which I acknowledge as a
limitation since public cohorts are pixel-level split by default. All
reported numbers should be read with that caveat.

**E4. Are your confidence scores calibrated?**
Not guaranteed — modern networks are typically overconfident, and softmax
max is not a probability. I provide the fix (`fit_temperature`, LBFGS on
held-out logits) but keep it off by default for reproducibility; enabling it
is a one-line change after validation. The reliability bands and the 70 %
safeguard are the operational hedge meanwhile.

**E5. Why report ROC-AUC per class in addition to F1?**
F1 uses a hard 0.5-threshold decision; AUC evaluates ranking quality across
all thresholds and is threshold-invariant. Per-class one-vs-rest AUC
additionally exposes classes the model cannot separate at any operating
point, which F1 can mask.

**E6. How would you improve results with more time/compute?**
Ranked by expected value: patient-level 5-fold cross-validation for honest
error bars; multi-modal branch encoders with cross-modal attention instead
of composite averaging; 3D volumetric context (needs ≥ 11 GB VRAM);
external-cohort validation for domain shift; and calibration + conformal
prediction sets for principled uncertainty.

---

## Group F — Ethics, Regulation & Professional Practice

**F1. Is your system a medical device?**
No, and I state that everywhere. It is a research and educational prototype
with no regulatory clearance (no FDA 510(k), no CE mark). Every output is a
second-read aid requiring verification by a licensed professional, and the
full disclaimer is embedded in the code, the dashboard, and every PDF.

**F2. What is the real-world failure mode you fear most, and how is it mitigated?**
Automation bias — a tired clinician accepting a confident wrong answer.
Mitigations: confidence bands with forced colour framing, the <70 % hard
safeguard, Grad-CAM++ that makes *what* the model attended to inspectable,
tumour-probability alerts that refuse to let "No_Tumor" look like certainty,
and disclaimers on every surface. None are perfect; together they raise the
cost of blind trust.

**F3. What biases does your dataset carry?**
Public-cohort demographics (age/sex/ethnicity skews), scanner and protocol
bias (field strength, sequences), site bias, and prevalence bias (tumour
cases oversampled relative to screening populations, which inflates PPV in
deployment). Mitigations within scope: augmentation, per-class weighting,
explicit limitation statements; the true fix is external multi-site
validation.

**F4. Who is accountable if the system is wrong?**
The clinician retains full accountability for any decision — the system is
decision *support*, never decision *making*. The engineering accountability
is mine: validated code, error-proof loaders that fail safe rather than
crash, logged every substitution and fallback, and an auditable PDF trail
per case.

**F5. What would regulatory approval require beyond current performance?**
Prospective multi-centre validation, patient-level splits and external test
cohorts, calibration with confidence intervals, subgroup fairness analysis,
adversarial and artifact robustness testing, a QMS under ISO 13485 with
risk management per ISO 14971, cybersecurity review, and post-market
surveillance. I can enumerate them because they frame the gap between a
research prototype and a deployable class II device.

**F6. Why open the code if it touches healthcare?**
Transparency is the strongest safety feature research-stage systems have:
peers can audit the decision rule, reproduce results, and find the failure
modes I missed. Patient data never leaves the machine — the system is fully
local by design — so openness here does not conflict with privacy.

**F7. If you had one more month, what would you do first?**
Patient-level cross-validation with an external holdout cohort. Everything
else — architecture tuning, better fusion — is decoration until the
evaluation is unimpeachable; honest, patient-level error bars are the
foundation any clinical claim needs.

**F8. Sum up the project's contribution in one minute.**
FreeBuff v4 demonstrates that on consumer laptop hardware a hybrid
deep-plus-handcrafted model (ResNet50+CBAM ⊕ HOG-PCA) can deliver
four-class brain tumour classification with TTA-robust inference,
mathematically-grounded Grad-CAM++ explanations, WHO-staged reporting, and
*engineered* safety behaviour — low-confidence safeguarding, sensitivity
alerts, fail-soft checkpoint loading — wrapped in a clinician-facing
workflow. The contribution is as much the disciplined safety engineering and
honest limitations as the accuracy number.

---

## Rapid-Fire Defense Tips

- Lead with the number, then the caveat: "X % weighted F1, stratified split,
  with these limitations."
- When challenged on results, go to the confusion matrix and Grad-CAM
  exports — concrete artifacts defuse speculation.
- Never claim clinical validity; claim engineering rigor and safety design.
- If you do not know something, scope it: "That is outside the current
  system boundary, and here is how I would add it."
- Keep three numbers memorised: parameter count (≈27.07 M), input size
  (144×144), safeguard threshold (70 %).
