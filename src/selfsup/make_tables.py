"""Paper-2 tables from results JSON (PRD R7).

Table 1: KIMORE LOSO rho by condition + bootstrap CI + beats-scratch
Table 2: zero-shot AUROC per corpus x condition (+ naive baseline row)
Table 3: zero-shot rank-transfer (Spearman) per corpus x condition
Table 4: contrastive_ft vs masked_ft head-to-head (primary contrast)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from selfsup.registry import CONDITIONS  # noqa: E402


def _write(df: pd.DataFrame, out_dir: str, name: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    try:
        df.to_markdown(os.path.join(out_dir, name + ".md"), index=False)
    except Exception:
        Path(os.path.join(out_dir, name + ".md")).write_text(df.to_string(index=False))


def make_tables(results_root: str = "outputs") -> None:
    out_dir = os.path.join(results_root, "ssl_results", "tables")
    stats_p = os.path.join(results_root, "ssl_results", "stats.json")
    stats = json.loads(Path(stats_p).read_text()) if os.path.exists(stats_p) else {}

    # Table 1 -- KIMORE rho by condition
    cis = stats.get("bootstrap_cis", {})
    beats = stats.get("beats_scratch", {})
    rows = []
    for name in CONDITIONS:
        c = cis.get(name, {})
        rows.append({
            "Condition": name,
            "Mean rho": round(c.get("mean", float("nan")), 4),
            "95% CI": f"[{c.get('ci_low', float('nan')):.3f}, {c.get('ci_high', float('nan')):.3f}]",
            "Beats scratch": beats.get(name, ""),
        })
    _write(pd.DataFrame(rows), out_dir, "table1_kimore_rho")

    # Tables 2 & 3 -- zero-shot AUROC and rank per corpus x condition
    corpora = sorted({os.path.basename(p).replace("zeroshot_", "").replace(".json", "")
                      for cond in CONDITIONS.values()
                      for p in glob.glob(os.path.join(results_root, cond.out_subdir, "zeroshot_*.json"))})
    auroc_rows, rank_rows = [], []
    for corpus in corpora:
        a_row = {"Corpus": corpus}
        r_row = {"Corpus": corpus}
        naive = None
        for name, cond in CONDITIONS.items():
            zp = os.path.join(results_root, cond.out_subdir, f"zeroshot_{corpus}.json")
            if os.path.exists(zp):
                z = json.loads(Path(zp).read_text())
                a_row[name] = round(z["mean_auroc"], 4) if z.get("mean_auroc") is not None else None
                r_row[name] = round(z["mean_rank_spearman"], 4) if z.get("mean_rank_spearman") is not None else None
                naive = z.get("naive_auroc", naive)
        a_row["naive_baseline"] = round(naive, 4) if naive is not None else None
        auroc_rows.append(a_row); rank_rows.append(r_row)
    if auroc_rows:
        _write(pd.DataFrame(auroc_rows), out_dir, "table2_zeroshot_auroc")
        _write(pd.DataFrame(rank_rows), out_dir, "table3_zeroshot_rank")

    # Table 4 -- primary contrast
    pc = stats.get("primary_contrast", {})
    _write(pd.DataFrame([{
        "Contrast": f"{pc.get('a')} vs {pc.get('b')}",
        "Wilcoxon p": pc.get("p"),
        "Median rho diff": pc.get("median_diff"),
    }]), out_dir, "table4_primary_contrast")

    print(f"[tables] wrote -> {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="outputs")
    make_tables(ap.parse_args().results_root)
