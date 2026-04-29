#!/usr/bin/env python
"""Build NeurIPS-style results tables (subject, view) for the 10-class top10 NTU
experiments. Reads pooled metrics CSVs from Tangent_Vector/results and
Raw_Skeleton/results and writes markdown tables.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
TV = REPO / "Tangent_Vector" / "results"
RS = REPO / "Raw_Skeleton" / "results"

CV_MODES = [("subject", ""), ("view", "_xview")]

METHOD_ORDER = [
    ("Raw Skeleton", "PCA + k-NN",   RS / "pca_clf_metrics{sfx}.csv",      "KNN"),
    ("Raw Skeleton", "VAE + k-NN",   RS / "vae_clf_metrics{sfx}.csv",      None),
    ("Raw Skeleton", "TCN",          RS / "sequence_clf_metrics{sfx}.csv", "TCN"),
    ("Raw Skeleton", "LSTM",         RS / "sequence_clf_metrics{sfx}.csv", "LSTM"),
    ("Raw Skeleton", "Transformer",  RS / "sequence_clf_metrics{sfx}.csv", "TRANSFORMER"),
    ("Raw Skeleton", "ST-GCN",       RS / "sequence_clf_metrics{sfx}.csv", "STGCN"),
    ("Tangent Vector", "PCA + k-NN", TV / "pca_clf_metrics{sfx}.csv",      "KNN"),
    ("Tangent Vector", "TCN",        TV / "sequence_clf_metrics{sfx}.csv", "TCN"),
    ("Tangent Vector", "LSTM",       TV / "sequence_clf_metrics{sfx}.csv", "LSTM"),
    ("Tangent Vector", "Transformer", TV / "sequence_clf_metrics{sfx}.csv", "TRANSFORMER"),
    ("Tangent Vector", "ST-GCN",     TV / "sequence_clf_metrics{sfx}.csv", "STGCN"),
    ("Tangent Vector", "ES-VAE + k-NN (proposed)", TV / "esvae_clf_metrics{sfx}.csv", None),
]


def fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} ({lo:.3f}, {hi:.3f})"


def get_row(csv_path: Path, method_filter: str | None) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if method_filter is not None:
        df = df[df["method"].str.upper().str.replace("-", "").str.replace(" ", "")
                == method_filter.upper().replace("-", "").replace(" ", "")]
        if df.empty:
            return None
    return df.iloc[0].to_dict()


def build_table(suffix: str, title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append("| Input Representation | Method | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |")
    lines.append("|---|---|---|---|---|")
    last_repr = None
    best_f1 = -1.0
    rows_data = []
    for repr_name, method, path_tmpl, mfilter in METHOD_ORDER:
        path = Path(str(path_tmpl).format(sfx=suffix))
        row = get_row(path, mfilter)
        if row is None:
            rows_data.append((repr_name, method, None))
            continue
        rows_data.append((repr_name, method, row))
        if row["F1 (macro) mean"] > best_f1:
            best_f1 = row["F1 (macro) mean"]

    for repr_name, method, row in rows_data:
        repr_cell = repr_name if repr_name != last_repr else ""
        last_repr = repr_name
        if row is None:
            lines.append(f"| {repr_cell} | {method} | — | — | — |")
            continue
        f1 = fmt(row["F1 (macro) mean"], row["F1 (macro) ci_low"], row["F1 (macro) ci_high"])
        prec = fmt(row["Precision (macro) mean"], row["Precision (macro) ci_low"], row["Precision (macro) ci_high"])
        rec = fmt(row["Recall (macro) mean"], row["Recall (macro) ci_low"], row["Recall (macro) ci_high"])
        bold = abs(row["F1 (macro) mean"] - best_f1) < 1e-6
        method_disp = f"**{method}**" if bold else method
        f1_disp = f"**{f1}**" if bold else f1
        lines.append(f"| {repr_cell} | {method_disp} | {f1_disp} | {prec} | {rec} |")
    return "\n".join(lines)


def main():
    out = []
    out.append("# NTU 10-class top10 — Classification Results")
    out.append("")
    out.append("Pooled out-of-fold predictions. 95% CIs from subject-level bootstrap "
               "(2000 resamples).")
    out.append("")
    for mode, sfx in CV_MODES:
        title = {
            "subject": "Cross-Subject (Leave-5-Subjects-Out, 8 folds)",
            "view":    "Cross-View (Leave-One-Camera-Out, 3 folds)",
        }[mode]
        out.append(build_table(sfx, title))
        out.append("")

    md_path = REPO / "results_tables_top10.md"
    md_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(out))


if __name__ == "__main__":
    main()
