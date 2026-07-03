## N. KIMORE fine-tuning — full 77-fold true LOSO (Table N)

| Condition | Mean rho | 95% CI (20-seed bootstrap) | Beats scratch |
|---|---|---|---|
| scratch | **0.836** | [0.785, 0.867] | — |
| masked_ft | 0.823 | [0.773, 0.854] | No |
| contrastive_ft | 0.816 | [0.762, 0.851] | No |
| contrastive_lp | 0.689 | [0.617, 0.738] | No |
| masked_lp | 0.679 | [0.612, 0.727] | No |

- Primary contrast contrastive_ft vs masked_ft: paired Wilcoxon p=0.805 (N=380 matched), Δabs-err=-0.012.
- Pairwise Wilcoxon (Holm-Bonferroni over 10 pairs): 6 significant.
- Sample-level pooled OOF, N=380; true leave-one-subject-out (77 folds).
