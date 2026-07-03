"""Shared configuration for the novelty-finding analysis package.

Single source of truth for paths, the model->outputs registry, and analysis
thresholds. Adding a new model or dataset should require editing ONLY this file
- that is what keeps the analyses scalable.
"""
from __future__ import annotations

import os
import sys

# -- Path bootstrap ----------------------------------------------------------
# This package lives in src/novelty/. The existing modules (constants.py,
# protocol_inflation.py, ...) live in src/ and import each other by bare name,
# expecting src/ on sys.path. Insert it so `from protocol_inflation import ...`
# resolves whether this package is run as a module or as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -- Core data locations -----------------------------------------------------
POOLED_DIR = os.path.join(PROJECT_ROOT, "KIMORE_pooled")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
NOVELTY_OUT = os.path.join(OUTPUTS_DIR, "novelty")
IRDS_RELIABILITY_CSV = os.path.join(OUTPUTS_DIR, "irds_eval", "irds_reliability.csv")
PAIRWISE_STATS_CSV = os.path.join(OUTPUTS_DIR, "sample_stats", "pairwise_sample_level.csv")

# -- Model registry (mirrors sample_level_stats.EXPERIMENTS / irds_eval.MODELS) --
RIDGE_NAME = "Ridge (stat. features)"
EXPERIMENTS: list[tuple[str, str]] = [
    (RIDGE_NAME,                   "outputs/ridge_baseline"),
    ("LSTM baseline",             "outputs/loso_lstm"),
    ("ST-GCN",                    "outputs/loso_stgcn"),
    ("GraphTransformer",          "outputs/loso_graph_transformer"),
    ("GraphTransformer (no bias)", "outputs/loso_graph_transformer_no_bias"),
    ("TCN",                       "outputs/loso_tcn"),
    ("SCT",                       "outputs/loso_sct"),
    ("Exp E (Transformer)",       "outputs/loso_multitask_uiprmd_d128"),
]

# Models treated as attention-based for the periodicity contrast (N5).
TCN_NAME = "TCN"
ATTENTION_MODELS = [
    "GraphTransformer",
    "GraphTransformer (no bias)",
    "SCT",
    "Exp E (Transformer)",
]

EXERCISE_NAMES = {
    0: "Trunk Lateral Flex.",
    1: "Trunk Forward Flex.",
    2: "Trunk Rotation",
    3: "Hip Abduction",
    4: "Hip Circumduction",
}
N_EXERCISES = 5

# -- Deployment-readiness thresholds (N7) ------------------------------------
KENDALL_W_MIN = 0.50   # minimum cross-exercise rank consistency to deploy
PRED_SD_MIN = 0.10     # below this => degenerate (variance collapse)
ALPHA = 0.05           # significance level

# -- Observed effect-size priors for the power analysis (N10) -----------------
# From the current study (see EXECUTION_PLAN.md problem #3 and the F5 analysis):
#   correlation between KIMORE rho and IRDS Kendall W across models ~ -0.393
#   IRDS Kendall W is unstable across embeddings: W = 0.480 +/- 0.124 (n=10 subj)
DISSOCIATION_R_OBS = -0.393
KENDALL_W_SD_AT_10 = 0.124
IRDS_N_SUBJECTS = 10


def resolve(path: str) -> str:
    """Resolve a registry path (e.g. 'outputs/...') against PROJECT_ROOT."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def ensure_out() -> str:
    """Create and return the novelty output directory."""
    os.makedirs(NOVELTY_OUT, exist_ok=True)
    return NOVELTY_OUT
