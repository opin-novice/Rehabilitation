"""Orchestrator CLI for the novelty-finding analyses (NOVELTY_OPPORTUNITIES.md).

Runs the Paper-1 bundle (no retraining) over existing outputs/ artifacts:
  decomp      N1  - 2x2 protocol-inflation decomposition
  periodicity N5  - periodicity vs TCN-minus-attention margin
  power       N10 - Monte-Carlo power / design guideline
  rubric      N7  - deployment-readiness screen
  all             - run everything

Usage:
  python src/novelty/run_all.py all
  python src/novelty/run_all.py decomp --seeds 20
  python -m novelty.run_all all          # if src/ is on PYTHONPATH
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import preprocessing_control  # src/preprocessing_control.py (M2 control)
from novelty import (
    config,
    deployment_rubric,
    periodicity,
    power_analysis,
    protocol_decomposition,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Novelty-finding analysis runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dec = sub.add_parser("decomp", help="N1 protocol-inflation 2x2 decomposition")
    p_dec.add_argument("--seeds", type=int, default=20)

    sub.add_parser("periodicity", help="N5 periodicity law")

    p_pow = sub.add_parser("power", help="N10 power analysis")
    p_pow.add_argument("--target", type=float, default=None)
    p_pow.add_argument("--sims", type=int, default=2000)

    sub.add_parser("rubric", help="N7 deployment-readiness rubric")

    p_pre = sub.add_parser("preproc", help="M2 preprocessing-vs-protocol control")
    p_pre.add_argument("--seeds", type=int, default=10)

    p_all = sub.add_parser("all", help="run every analysis")
    p_all.add_argument("--seeds", type=int, default=20)
    p_all.add_argument("--sims", type=int, default=2000)

    args = ap.parse_args()
    config.ensure_out()

    def banner(t: str) -> None:
        print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

    if args.cmd == "decomp":
        protocol_decomposition.run(seeds=args.seeds)
    elif args.cmd == "periodicity":
        periodicity.run()
    elif args.cmd == "power":
        power_analysis.run(target=args.target, sims=args.sims)
    elif args.cmd == "rubric":
        deployment_rubric.run()
    elif args.cmd == "preproc":
        preprocessing_control.run(seeds=args.seeds)
    elif args.cmd == "all":
        banner("N1  Protocol-inflation 2x2 decomposition")
        protocol_decomposition.run(seeds=args.seeds)
        banner("N5  Periodicity law")
        periodicity.run()
        banner("N10 Power analysis")
        power_analysis.run(sims=args.sims)
        banner("N7  Deployment-readiness rubric")
        deployment_rubric.run()
        banner("M2  Preprocessing-vs-protocol control")
        preprocessing_control.run(seeds=10)
        print(f"\nAll novelty outputs in {config.NOVELTY_OUT}")


if __name__ == "__main__":
    main()
