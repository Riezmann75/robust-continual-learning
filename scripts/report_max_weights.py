"""Summarize final-epoch chain concentration for the fixed-m sweep."""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("chain_random", "js", "chain_random_js_bs32"),
    ("chain_random", "infonce", "chain_random_bs32"),
    ("metamix_fixed_m", "js", "metamix_fixed_m_bs32"),
    ("metamix_fixed_m", "infonce", "metamix_fixed_m_infonce_bs32"),
]


def final_weight(m, variant, loss, historical, seed):
    if m == "0.5":
        directories = (ROOT / "experiments" / f"unrolled_{historical}" / f"seed{seed}",
                       ROOT / "results" / f"unrolled_mixing_seed{seed}" / historical)
    else:
        directories = (ROOT / "experiments/fixed_m_sweep_bs32" / f"m{m}" /
                       f"{variant}_{loss}" / f"seed{seed}",)
    for directory in directories:
        path = directory / "progress.json"
        if path.exists():
            epochs = json.loads(path.read_text())["epochs"]
            if epochs and epochs[-1]["updates"] == 15650:
                last = epochs[-1]
                if not math.isclose(last["m_mean"], float(m), abs_tol=1e-5):
                    raise ValueError(f"Unexpected m in {path}")
                return last["weight_max_mean"]
    raise ValueError(f"{variant}/{loss}, m={m}, seed {seed}: need final progress.json in "
                     + " or ".join(str(p) for p in directories))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-table", action="store_true")
    args = parser.parse_args()
    rows, errors = [], []
    for m in ("0.1", "0.3", "0.5", "0.7", "0.9"):
        cells = [m]
        for variant, loss, historical in METHODS:
            values = []
            for seed in (17, 23, 43):
                try:
                    values.append(final_weight(m, variant, loss, historical, seed))
                except ValueError as error:
                    errors.append(str(error))
            cells.append(f"{mean(values):.4f}" if len(values) == 3 else "--")
        rows.append(cells)
    print("m | Chain-aware JS | Chain-aware InfoNCE | MetaMix JS | MetaMix InfoNCE")
    for row in rows:
        print(" | ".join(row))
    if errors:
        sys.exit("\n".join(errors))
    if args.update_table:
        path = ROOT / "paper/cvpr/tables/diagnostics.tex"
        lines = [r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{lrrrr}",
                 r"\toprule", r" & \multicolumn{2}{c}{Chain-aware} & \multicolumn{2}{c}{MetaMix} \\",
                 r"$m$ & JS & InfoNCE & JS & InfoNCE \\", r"\midrule",
                 *[" & ".join(row) + r" \\" for row in rows], r"\bottomrule", r"\end{tabular}",
                 r"\caption{Final-epoch mean maximum chain weight, averaged equally over seeds 17, 23, and 43. "
                 r"We take the maximum across the three chains before averaging. "
                 r"Equal weights give $1/3$; the weights divide the augmented share $1-m$.}\label{tab:weights}",
                 r"\end{table}"]
        path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
