"""Novelty-finding analysis package for the rehabilitation-scoring benchmark.

Implements the publishable open problems catalogued in NOVELTY_OPPORTUNITIES.md,
as composable, read-only analyses over the artifacts already produced by the
main pipeline (outputs/). No retraining is required for the Paper-1 bundle.

Modules
-------
config                : single source of truth (paths, model registry, thresholds)
io_utils              : robust, gracefully-degrading loaders
protocol_decomposition: N1 - 2x2 leakage x stratification inflation decomposition
periodicity           : N5 - per-exercise periodicity vs TCN-minus-attention margin
power_analysis        : N10 - Monte-Carlo power for the dissociation claim
deployment_rubric     : N7 - composite deployment-readiness screen
run_all               : orchestrator CLI
"""

__all__ = [
    "config",
    "io_utils",
    "protocol_decomposition",
    "periodicity",
    "power_analysis",
    "deployment_rubric",
]
