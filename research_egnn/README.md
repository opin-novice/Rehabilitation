# research_egnn — isolated baseline probe (code isolated; Q5 result now in the paper)

This directory is a **self-contained research sandbox**. Its **code** is isolated — it imports the
paper's modules read-only and writes only under `research_egnn/outputs/`, modifying none of the paper's
code or artifacts. Two reviewer-suggested experiments were probed here:

- **Q5 (EGNN vs steerable encoder)** — the 3-seed × 5-fold sweep below is now **reported in the paper**
  (Results §IV-E, Table `tab:egnn`) as the answer to reviewer Q5. Only these aggregate numbers enter
  the manuscript; no checkpoint or script from this directory is shipped or cited as paper code.
- **Q3 (PCA-canonicalization front-end)** — remains a **probe only**, not reported in the paper
  (declined as future work in the response letter).

## Questions probed
- **Q5** — does a lighter **E(n)-equivariant (EGNN)** encoder match the paper's steerable e3nn
  encoder on **accuracy** and **node-failure robustness**? (Viewpoint is a tie by construction — see
  below.)
- **Q3** — does a **PCA canonicalization front-end** feeding a non-equivariant backbone (PCT) leave
  **per-sequence** degradation under camera rotation that the paper's exact invariant cut avoids?

## Isolation guarantees (why this cannot hamper the existing research)
- **No file under `src/` or `outputs/cde_block2/` is edited, overwritten, or deleted.** All existing
  modules are *imported read-only*.
- The EGNN model is obtained by **subclassing** `SE3EquivariantGRU` and reassigning `self.encoder`
  (`egnn_model.py`) — no change to `src/equivariant_gru.py`.
- **Every** checkpoint / JSON this sandbox writes goes under `research_egnn/outputs/` only, enforced
  by an assertion in `sandbox_train.py`.
- The existing invariant cut, GRU, head, corruption operators, splits, and metrics are reused via
  import, exactly as the paper uses them, so the baseline is comparable rather than re-derived.

## Files
| File | Role |
|---|---|
| `egnn_encoder.py` | from-scratch Satorras EGNN encoder honouring the paper's `n_scalar+3·n_vec` contract |
| `egnn_model.py` | `EGNNRecurrence` = `SE3EquivariantGRU` with the encoder swapped |
| `canonicalize.py` | per-frame PCA principal-axis canonicalizer (deterministic signs) |
| `sandbox_train.py` | faithful copy of the paper's training recipe, model-agnostic; writes to `outputs/` |
| `sandbox_eval.py` | clean MAD + viewpoint sweep + node-failure + EGNN invariance certificate |

## Run
```
python research_egnn/sandbox_train.py --model egnn  --cv     # 1 seed x 5 folds
python research_egnn/sandbox_train.py --model canon --cv
python research_egnn/sandbox_eval.py                          # -> outputs/sandbox_results.json
```

## Note on scope
- The Q5 comparison was first probed at **triage** budget (1 seed × 5 folds) and then re-run at the
  paper's **3-seed × 5-fold** rigor plus a coordinate-clamp sweep (see `FINDINGS.md`); the **3-seed**
  numbers are the ones reported in the paper. The EGNN remains a from-scratch encoder at fixed width
  (4 layers, hidden 64) with no hyperparameter search — the paper states this caveat explicitly and
  claims only the scoped node-failure result, not "steerable SH beats E(n)" in general.
- Because the EGNN feeds the *same* invariant cut, EGNN + cut is *also* exactly SE(3)-invariant
  (certified in `sandbox_eval.py`), so the viewpoint axis is a tie by construction — the informative
  axes are accuracy and node-failure.
- Vector-Neuron arm intentionally omitted.
