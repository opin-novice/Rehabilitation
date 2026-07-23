#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
determinism.py
==============
PHASE 1 -- kill the +/-0.33 MAD "hardware noise floor".

The floor was never hardware. It was two nondeterministic CUDA kernels in our own forward pass:

  1. e3nn message aggregation via `Tensor.index_add_` on CUDA. index_add_ resolves write
     collisions with ATOMIC ADDS, whose order is scheduler-dependent, so fp32 summation
     order -- and therefore the result -- changes run to run. The skeleton graph is FIXED
     (25 joints, 24 bones, 48 directed edges), so there was never a reason to scatter at all:
     see equivariant_gru.EquivariantSkeletonEncoder, which now aggregates with a precomputed
     dense (J x E) incidence matrix. A matmul has a fixed reduction order. Deterministic, and
     faster at this size.

  2. cuDNN's fused GRU backward, which also accumulates with atomics.

Why it matters more than it sounds. Our whole effect size is
    per-exercise mean floor 8.25  -  model ~6.6  =  ~1.65 MAD.
A 0.33 MAD run-to-run spread is 20% OF THE ENTIRE SIGNAL, and it is wider than every
model-vs-model gap we report. Under that floor "PCT 6.465 vs EGRU 6.619" is not a tie we
measured, it is a comparison we could not perform. Reporting it as a property of the hardware
overstates our own uncertainty and invites the reviewer to discount the paper.

Usage -- call ONCE, before any CUDA context is created (i.e. before the first .to(device)):

    import determinism
    determinism.enable(seed=0)

Set strict=False to downgrade the "no deterministic kernel exists" error to a warning; use it
only to find out WHICH op is at fault, never for a reported number.
"""

import os
import random

# CUBLAS workspace must be configured BEFORE the CUDA context exists, hence before `import torch`
# has a chance to initialise one. Setting it here, at module import, is the only reliable moment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402


def enable(seed: int = 0, strict: bool = True, cudnn_rnn: bool = False) -> dict:
    """Make training bit-reproducible for a fixed seed. Returns the config actually applied.

    cudnn_rnn=False DISABLES the cuDNN fused RNN path (torch.backends.cudnn.enabled=False is too
    blunt -- it would also disable conv, which we do not use, so it costs us nothing but we scope
    it anyway). PyTorch then runs the GRU as unfused ATen ops, whose backward is deterministic.
    The cost is wall-clock on the recurrence; the benefit is that a seed means something.
    Set cudnn_rnn=True to keep the fast fused kernel and accept a small nondeterministic residual
    -- and if you do, you MUST re-measure the seed spread rather than assume it is zero.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False              # autotuner picks different kernels per run
    if not cudnn_rnn:
        # The fused cuDNN GRU backward uses atomics. Unfused ATen GRU does not.
        torch.backends.cudnn.enabled = False

    # This is the assertion, not the fix: it makes any REMAINING nondeterministic kernel raise
    # instead of silently contributing noise. If this throws, the traceback names the op -- that
    # is the point. Do not paper over it with warn_only for a number you intend to publish.
    torch.use_deterministic_algorithms(True, warn_only=not strict)

    cfg = {
        "seed": seed,
        "strict": strict,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    return cfg


def report(cfg: dict) -> str:
    return ("  determinism: seed=%d  strict=%s  cudnn(enabled=%s det=%s bench=%s)  cublas=%s"
            % (cfg["seed"], cfg["strict"], cfg["cudnn_enabled"], cfg["cudnn_deterministic"],
               cfg["cudnn_benchmark"], cfg["cublas_workspace"]))
