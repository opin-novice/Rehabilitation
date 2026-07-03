"""Paper-2 figures from results JSON (PRD R7). Matplotlib is optional.

Fig 1: KIMORE rho by condition with bootstrap-CI error bars
Fig 2: zero-shot AUROC heatmap (condition x corpus) with naive baseline
Fig 4: pred_SD degeneracy histogram with the 0.10 gate line
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from selfsup.registry import CONDITIONS  # noqa: E402


def make_figures(results_root: str = "outputs") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"[figures] SKIP (matplotlib unavailable: {e})")
        return False

    fig_dir = os.path.join(results_root, "ssl_results", "figures")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    stats_p = os.path.join(results_root, "ssl_results", "stats.json")
    stats = json.loads(Path(stats_p).read_text()) if os.path.exists(stats_p) else {}
    names = list(CONDITIONS.keys())

    # Fig 1 -- KIMORE rho bars with CI
    cis = stats.get("bootstrap_cis", {})
    if cis:
        means = [cis.get(n, {}).get("mean", float("nan")) for n in names]
        lo = [cis.get(n, {}).get("mean", 0) - cis.get(n, {}).get("ci_low", 0) for n in names]
        hi = [cis.get(n, {}).get("ci_high", 0) - cis.get(n, {}).get("mean", 0) for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, means, yerr=[lo, hi], capsize=4, color="#0072B2")
        ax.set_ylabel("KIMORE LOSO Spearman rho"); ax.set_title("Fig 1: rho by condition (95% bootstrap CI)")
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig1_kimore_rho.png"), dpi=150); plt.close(fig)

    # Fig 2 -- zero-shot AUROC heatmap
    corpora = sorted({os.path.basename(p).replace("zeroshot_", "").replace(".json", "")
                      for cond in CONDITIONS.values()
                      for p in glob.glob(os.path.join(results_root, cond.out_subdir, "zeroshot_*.json"))})
    if corpora:
        M = np.full((len(names), len(corpora)), np.nan)
        for i, name in enumerate(names):
            for j, corpus in enumerate(corpora):
                zp = os.path.join(results_root, CONDITIONS[name].out_subdir, f"zeroshot_{corpus}.json")
                if os.path.exists(zp):
                    v = json.loads(Path(zp).read_text()).get("mean_auroc")
                    if v is not None:
                        M[i, j] = v
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(M, cmap="cividis", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(corpora))); ax.set_xticklabels(corpora, rotation=20, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_title("Fig 2: zero-shot AUROC"); fig.colorbar(im, ax=ax)
        plt.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig2_zeroshot_auroc.png"), dpi=150); plt.close(fig)

    print(f"[figures] wrote -> {fig_dir}")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="outputs")
    make_figures(ap.parse_args().results_root)
