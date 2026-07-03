"""Experiment registry (Layer 4): single source of truth for the 5 conditions.

Maps each condition name to the train_loso.py invocation flags. Tables/figures
read from here so naming stays consistent across the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Condition:
    name: str
    init_ckpt_key: str      # "" | "contrastive" | "masked"  (resolved to a path at runtime)
    freeze_encoder: bool
    out_subdir: str


CONDITIONS: Dict[str, Condition] = {
    "scratch":        Condition("scratch",        "",            False, "ssl_results/A_scratch"),
    "contrastive_lp": Condition("contrastive_lp", "contrastive", True,  "ssl_results/B_contrastive_lp"),
    "contrastive_ft": Condition("contrastive_ft", "contrastive", False, "ssl_results/C_contrastive_ft"),
    "masked_lp":      Condition("masked_lp",      "masked",      True,  "ssl_results/D_masked_lp"),
    "masked_ft":      Condition("masked_ft",      "masked",      False, "ssl_results/E_masked_ft"),
}

# Primary head-to-head contrast for the paper (paired Wilcoxon).
PRIMARY_CONTRAST = ("contrastive_ft", "masked_ft")
