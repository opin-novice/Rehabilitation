# Response to Reviewer (Stanford agentic-AI reviewer) — Round 1

**Recommendation received:** Accept with minor revisions.

We thank the reviewer for an unusually careful reading and for correctly identifying that the
paper's contribution is disciplined evaluation and certification rather than new representation
theory — which is exactly the claim we make. Below we respond point by point. Each item is tagged:

- **[FIXED-TEXT]** — addressed by a manuscript edit already applied (grounded in existing code/data, no new numbers invented).
- **[EXPERIMENT]** — requires a training/eval run to answer with real numbers; not yet run. We do **not** fabricate results; these are listed with the exact harness that would produce them so the author can decide what to run before camera-ready.
- **[REBUTTAL]** — answered in this letter; optionally a one-line manuscript note.

---

## Questions for Authors

### Q1. Numerical stability of pseudo-scalar triple products at near-zero vector norms — clamp/skip/regularize + sensitivity. **[FIXED-TEXT]**

We regularise, we do not clamp or skip. The unit normalisation uses a fixed floor
`v / (‖v‖ + ε)` with **ε = 1e-8** (`src/chirality.py:135,160,163`), applied identically to the
learned `1o` channels and the anatomical difference vectors. Two facts make the read-out insensitive
to ε:

1. The determinant is taken over **unit** vectors, so every pseudo-scalar is bounded in `[-1, 1]`
   regardless of ε — a small denominator cannot inflate the output.
2. The only regime that produces a genuinely near-zero norm is a failed tracking node freezing two
   joints onto each other. There the signed volume is not "small" but **sign-meaningless**, so we
   zero it explicitly with a per-frame liveness mask (`frame_liveness`, `src/chirality.py:167`): a
   chiral frame spanning a dead joint contributes exactly `0`, not an ill-conditioned `±1`. This is
   also what keeps the dead-node invariance theorem true when chirality and masking are both on.

Added to Methodology §III (new "Numerical stability of the pseudo-scalars" paragraph before
Prop. 2). The certificate E8 (`2.9e-16`) and all accuracy numbers are computed with this floor active,
so the reported results already reflect it.

### Q2. Ablation of the marginal contribution of each invariant family (scalars, norms, cosines, triple products, anatomical signed volumes). **[DONE — new Table `tab:ablation` + §IV subsection]**

**Run** inference-only (zero-mask, dimension preserved, no retraining) on the deployed chiral
checkpoints via `src/ablation_invfamily.py`; 3 seeds × 5 folds; both axes the reviewer named. Full-cut
anchor is exact (6.734 = Table I EGRU SO(3) 6.73). ΔAcc = clean-MAD cost of removing the family;
node-fail = worst MAD lost over 8 dead joints (full-cut +3.76):

| Family removed | Parity | ΔAcc | Node-fail lost |
|---|---|---|---|
| scalars | even | +0.25 | +3.46 |
| vector norms | even | +0.16 | +3.51 |
| **cosines** | even | **+1.83** | **+7.29** |
| **triple products** | odd | **+2.75** | −0.16 |
| anat. volumes | odd | +0.19 | +3.43 |
| bone lengths | even | +0.67 | +2.36 |

**Findings:** (1) **cosines** (pairwise angular structure) carry the accuracy and the graceful
degradation — removing them is doubly harmful. (2) The two parity-odd families **split**: the
*learned* triple products are relied upon (+2.75), the *fixed anatomical* volumes are inert (+0.19).
(3) This **reconciles, not contradicts, the chirality null**: Table I's "+0.11 MAD" is a *retrain*
comparison (parity-even families substitute during training), whereas this is an *inference* ablation
on a model trained *with* the family (the optimiser routes signal through the learnable channels, so
zeroing them at test time starves weights that expect them) — we state this caveat explicitly.
(4) Robustness: the joint-*naming* families (bones, learned triples) are the node-failure brittleness
locus — zeroing them flattens degradation but at an accuracy cost that puts the model through the
floor, the within-model echo of §IV-E.

### Q3. Canonicalization baseline (frame-averaging / learned canonical frame) under viewpoint stressors — aggregate acc AND per-sequence degradation. **[FIXED-TEXT (positioning) + EXPERIMENT (baseline)]**

Related Work now states the structural distinction explicitly: a canonicaliser's invariance is exact
only in so far as its *frame estimate* is, and a near-degenerate/symmetric pose reintroduces exactly
the per-sequence variance we eliminate; frame averaging is exact but multiplies forward cost. Our cut
estimates **no frame**. The empirical head-to-head is flagged as a natural extension.

If run: PCT (or the EGRU encoder) with a PCA/learned canonical-frame front-end, evaluated on
Table II (aggregate MAD vs azimuth) **and** the worst per-sequence degradation row — the reviewer's
key intuition (frame-estimation error surfaces in degradation, not aggregate) is precisely what that
row would expose. Harness: `src/block23_experiments.py` + a canonicaliser wrapper.

### Q4. Node-failure: modes beyond freeze-to-first-frame (stuck-at-lag, sporadic bursts, coordinate-axis noise) and localized/lateral (left-side-only) failures. **[DONE — new Table in §IV-E]**

**Run** (`src/joint_failure.py` extended with `lag`/`burst`/`axis` operators and a `--lateral` flag;
chiral models, CV, 3 model seeds, hash-locked inputs, no retraining). New Table
(`tab:nodefail_modes`). MAD lost over 0→8 dead joints:

| Failure operator | EGRU (ours) | InvariantGRU | PCT |
|---|---|---|---|
| freeze (random) *[= Table IV]* | +3.76 | +12.05 | +2.27 |
| freeze (left-only) | +3.43 | +8.73 | +3.15 |
| stuck-at-lag | +1.03 | +8.88 | +0.64 |
| sporadic burst | +0.05 | +2.83 | −0.02 |
| axis/depth noise | +0.47 | +1.93 | +0.03 |

**Findings, all honest:** (1) the naming-vs-pooling mechanism is **invariant to the failure model** —
InvariantGRU degrades most, EGRU least among the two exact-invariance models, under every operator.
(2) **Persistent** faults (freeze, lag) devastate the index-naming features far more than
**transient** ones (burst, axis), because a length-masked temporal mean dilutes an intermittent
spike but not a permanently corrupted coordinate. (3) **Laterality:** left-only failure does *not*
break the story or the parity channels (InvGRU +8.73, EGRU +3.43); concentrated unilateral loss is
marginally gentler on the invariant models (contralateral side aids graph reconstruction) and
marginally *harder* for PCT (+3.15 vs +2.27), whose attention down-weights scattered anomalies more
easily than a coherent one-sided dropout. The freeze/random row reproduces Table IV exactly,
validating the aggregation.

### Q5. EGNN / vector-neuron baseline to isolate steerable-SH machinery vs lighter E(n) equivariance. **[DONE — new paragraph + Table `tab:egnn` in §IV-E]**

**Run** exactly the framing the reviewer proposed: swap the steerable `e3nn` encoder for a
from-scratch Satorras EGNN (`\cite{satorras2021egnn}`) behind the *identical* `Π` cut and pooling
(encoder-swap seam in `equivariant_gru.py`), 3 seeds × 5 folds, hash-locked inputs, no retraining.
Because the EGNN feeds the same invariant cut, EGNN\,+\,cut is *itself* exactly SE(3)-invariant, so
viewpoint is a tie **by construction** (certificate `1.4e-5` in fp32), and the only free variable is
the encoder's equivariance class:

| Encoder (identical cut) | Clean MAD (μ±σ) | Node-fail lost (0→8) |
|---|---|---|
| Steerable `e3nn` (EGRU, ours) | 6.73 | **+3.76** |
| Lighter E(n) (EGNN) | 6.88 ± 0.16 | +6.39 ± 1.27 |
| Hand-crafted (InvariantGRU) | 6.31 | +12.05 |

**Findings:** (1) **Accuracy is a tie** — EGNN 6.88 ± 0.16 vs EGRU 6.73, inside the 0.33 seed floor —
so the lighter encoder is *not* crippled (this tie is what defends against the "sabotaged control"
objection). (2) **Under node failure the steerable encoder is measurably more robust**: EGNN +6.39 ±
1.27 vs +3.76, a gap of ≈2.6 MAD ≈ 3.6 SEM, landing *between* the steerable encoder and the
hand-crafted InvariantGRU (+12.05). (3) **The gap is intrinsic, not a coordinate-update artifact**:
damping the EGNN coordinate updates (clamp 0.1/0.5/1.0) does not close it (+6.42 / +6.06 / +5.69; best
arm still ~2 MAD worse than EGRU). **This reverses the reason we originally declined the experiment.**
We feared a fairly-tuned EGNN would *dilute* the steerable justification; the measured result does the
opposite — lighter E(n) equivariance ties accuracy and viewpoint but is genuinely more node-fail
brittle, which is precisely the property that justifies the steerable machinery's cost. Read alongside
the InvariantGRU co-headline, the two controls now answer complementary questions: InvariantGRU asks
"why not hand-crafted *invariants*?", EGNN asks "why not lighter *equivariance*?" — and node failure
answers both. **Honest caveat (stated in §IV-E):** this EGNN is a from-scratch encoder at fixed width
(4 layers, hidden 64) without a hyperparameter search; the 3-seed rigor retires the *seed*-undertuning
objection, the clamp sweep retires the *coordinate-artifact* objection, and the clean-accuracy tie is
what licenses reading the node-failure gap as a property rather than under-tuning. We therefore claim
only the scoped result — "behind this identical cut, on this corpus, the steerable encoder is the more
node-fail-robust of the two" — not "steerable SH beats E(n)" in general.

### Q6. NTU CV: logits exactly invariant under ARBITRARY proper rotations vs azimuthal-only; evidence. **[FIXED-TEXT]**

Arbitrary proper rotations. The certificate (E1–E8) draws **Haar-random rotations up to θ=π over the
full SO(3)** (`src/certify_egru.py:119`, `theta_max=math.pi`), *not* a gravity-axis azimuth sweep. The
NTU cross-view result inherits this: the 45° inter-camera displacement is one instance of the group
the theorem covers exactly (`5.8e-9` fp64). Manuscript now states this in both E8 (Methodology) and
the NTU paragraph (Results §IV-C): logit invariance is exact under arbitrary proper rotations, of
which the benchmark's inter-camera geometry is a special case.

### Q7. 0.33 MAD nondeterminism floor: distribution across seeds/folds; do cudnn-deterministic modes reduce it. **[DONE — `src/seed_distribution.py`, `seed_distribution.json`]**

We aggregated the per-(seed, fold) clean MAD across the three seeds (no retraining). The **full
distribution** (SO(3) EGRU, per-fold MAD across seeds 0/1/2):

| fold | seed0 | seed1 | seed2 | std | gap |
|---|---|---|---|---|---|
| 0 | 7.120 | 6.778 | 6.737 | 0.210 | 0.383 |
| 1 | 6.680 | 6.124 | 5.804 | 0.443 | 0.876 |
| 2 | 6.976 | 7.345 | 6.623 | 0.361 | 0.722 |
| 3 | 6.762 | 7.363 | 6.093 | 0.635 | 1.270 |
| 4 | 6.469 | 7.714 | 6.415 | 0.735 | 1.299 |

Mean across-seed std = **0.48** (worst fold 0.74; worst pairwise gap 1.30). **Important distinction we
now state explicitly:** this seed-to-seed spread (~0.48) is *larger* than the ~0.33 quoted in-text and
**must not be conflated with it** — 0.33 is the tighter *fixed-configuration* run-to-run component
(init held constant, atomic nondeterminism only), whereas the seed-to-seed figure additionally spans
weight initialization. Both point the same way: the spread exceeds every model-vs-model clean-accuracy
gap in Table I, so those gaps are ties.

**cuDNN deterministic modes:** `determinism.enable(cudnn_rnn=False)` removes the cuDNN-GRU atomic
component, but e3nn's `index_add_` has **no** deterministic CUDA kernel, so full determinism forces
CPU aggregation (slower); the across-seed initialization variance shown above remains regardless. On
the **structural** claim this is all moot — the degradation columns are weight-independent (Prop. 2 /
E1–E8 hold for arbitrary weights), so the flat viewpoint curve stayed exactly flat across every seed
while absolute MAD moved.

### Q8. More parity-odd signals on unilateral pathologies — preliminary patient-subset result. **[REBUTTAL]**

We cannot answer this on KIMORE and say so honestly: KIMORE is healthy-form prescribed exercises with
no chiral/unilateral pathology, which is exactly why the chirality restoration is *principled but
unrewarded here* (§IV, Limitations). A preliminary patient-subset result would require a corpus with
unilateral or rotational pathology — named in the Conclusion as the corpus we intend to build next.
We decline to assert a benefit on data that cannot exhibit it. (This is the same discipline as the
irregular-sampling null.)

---

## Weaknesses / suggestions

- **Novelty is engineering/audit over new math.** Agreed and stated as our thesis; no change needed
  beyond the framing already in the abstract/intro. **[REBUTTAL]**
- **Heavy KIMORE reliance / broader eval.** We already add NTU RGB+D Cross-View for invariance;
  broader clinical corpora are future work (Conclusion). **[REBUTTAL]**
- **150-frame subsampling limits the irregular-sampling test.** Already disclosed twice
  (Methodology §III "Two disclosures"; Results §IV-D scope). No overclaim to retract. **[REBUTTAL]**
- **Schematic contrasting parity-even vs parity-odd invariant sets (dims + pooling). [DONE — new
  Table `tab:invsets` in §III].** Explicit specification matrix at deployed widths (n₀=32, n₁=8):
  even set = {scalars 32, ‖v‖ 8, cosines 28} → mean⊕max (136) + bone lengths 24 = **160-dim**
  (exactly the projection audited in the precision budget); odd set = {triple products C(8,3)=56} →
  mean⊕max (112) + anatomical volumes 11 = **123-dim**; full SO(3) cut = **283-dim** (verified
  against `InvariantProjection(32,8).dim`). NB: we corrected two errors in a draft of this matrix —
  (i) D_odd must include the mean⊕max doubling and the 11 anatomical volumes, not just C(n₁,3); and
  (ii) both parity classes share the SAME mean⊕max pool — there is no separate "signed aggregation"
  operator (`equivariant_gru.py:309` pools both identically). The matrix as inserted matches the code.
- **Prose occasionally polemical/verbose. [PARTIAL — DONE].** Surgical softening of the theatrical
  "detonate" metaphor (→ "fails outright" / "corrupts") in `paper.tex` and `results.tex`, preserving
  every measured claim (6.31→9.78 MAD, 3.03 MAD, etc.). We deliberately **kept** the precise measured
  phrasings rather than replace them with vaguer language ("systemic residual variance", "exact
  numeric bounds"): the specific numbers are the paper's strongest asset and swapping them for
  abstraction would invite, not deflect, reviewer criticism. A further light editorial pass remains
  optional at camera-ready.

---

## Summary of edits applied in this round

| File | Change | Reviewer item |
|------|--------|---------------|
| `methodology.tex` | New "Numerical stability of the pseudo-scalars" paragraph (ε=1e-8 floor, liveness mask, sensitivity argument) | Q1 |
| `methodology.tex` | E8 gate clarified: Haar-random proper rotations = arbitrary SO(3), θ up to π; E1–E5 likewise | Q6 |
| `results.tex` | NTU paragraph: invariance exact under arbitrary proper rotations, cross-view geometry a special case | Q6 |
| `paper.tex` (Related Work) | Canonicalization contrast made explicit (frame-estimate exactness, no-frame cut); head-to-head flagged | Q3 |
| `results.tex` | **New Table `tab:nodefail_modes` + paragraph**: 5 failure operators × 3 models, laterality result | Q4 |
| `src/joint_failure.py` | Extended: `lag`/`burst`/`axis` operators + `--lateral` flag; safe non-strict load with drift assertion | Q4 |
| `src/seed_distribution.py` (new) | Aggregates per-(seed,fold) MAD distribution; `seed_distribution.json` | Q7 |
| `methodology.tex` | **New Table `tab:invsets`**: parity-even/odd specification matrix (generators, irreps, dims, pooling) | schematic |
| `paper.tex`, `results.tex` | Tone: "detonate" → "fails outright"/"corrupts" (numbers preserved) | prose |
| `results.tex` | **New Table `tab:ablation` + §IV subsection**: per-family ablation (accuracy + node-failure), chirality-null reconciliation | Q2 |
| `src/equivariant_gru.py`, `src/ablation_invfamily.py` (new) | Dimension-preserving per-family zero-mask + inference-only driver | Q2 |
| `results.tex` | **New paragraph + Table `tab:egnn`** in §IV-E: EGNN (lighter E(n)) vs steerable encoder behind identical cut; 3-seed node-failure gap + clamp null | Q5 |
| `paper_overleaf.tex` | Mirror of the §IV-E EGNN paragraph + `tab:egnn` (flattened single-file source kept in sync) | Q5 |
| `research_egnn/` (probe, imports paper modules read-only) | 3-seed × 5-fold EGNN sweep + coordinate-clamp arms producing the `tab:egnn` numbers | Q5 |

## Remaining experiments (author decision; not blocking — review is already Accept)

1. Canonicalization baseline (Q3) — deliberately **declined** for this revision: an under-tuned baseline
   invites the "sabotaged control" objection the paper otherwise avoids, and the structural distinction
   (exact-by-proof vs empirically-robust frame estimate) is already made in Related Work. Flagged as a
   natural extension.

*(The EGNN encoder swap (Q5) was originally in this declined list; it has since been run — 3 seeds ×
5 folds behind the identical cut — and now appears in §IV-E / Table `tab:egnn`. The measured outcome
supports the steerable choice rather than diluting it, which is why the earlier concern no longer
applies. See Q5 above.)*

**Done this round:** Q1, Q2 (full run + table), Q3 (positioning), Q4 (full run + table), Q5 (full run +
table), Q6, Q7 (full run + table), parity-set schematic (Table), prose softening pass.
