"""Report batch-32 CIFAR-10-C accuracy for the six methods and three seeds."""
import argparse
import json
from pathlib import Path
from statistics import mean

from summarize_corruption_categories import GROUPS, aggregate, table

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "Chain-aware JS": "chain_random_js_bs32",
    "Chain-aware InfoNCE": "chain_random_bs32",
    "MetaMix JS, fixed m": "metamix_fixed_m_bs32",
    "MetaMix InfoNCE, fixed m": "metamix_fixed_m_infonce_bs32",
    "MetaMix JS, learned m": "metamix_bs32",
    "MetaMix InfoNCE, learned m": "metamix_infonce_bs32",
}
START = "<!-- unrolled-corruptions-start -->"
END = "<!-- unrolled-corruptions-end -->"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    results = {}
    missing = []
    for label, method in METHODS.items():
        results[label] = []
        for seed in (17, 23, 43):
            local = ROOT / "experiments" / f"unrolled_{method}" / f"seed{seed}" / "evaluation.json"
            archive = ROOT / "results" / f"unrolled_mixing_seed{seed}" / method / "evaluation.json"
            path = local if local.is_file() else archive
            if path.is_file():
                results[label].append(aggregate(json.loads(path.read_text())))
            else:
                missing.append(f"{label}, seed {seed}")
    if missing:
        parser.exit(1, "Missing evaluations:\n" + "\n".join(missing) + "\n")

    sections = []
    for metric, title in (("class_il", "Class-IL"), ("task_il", "Task-IL")):
        rows = []
        for corruption in [c for group in GROUPS.values() for c in group]:
            cells = [f"{100 * mean(r[metric]['per_corruption'][corruption] for r in runs):.2f}"
                     for runs in results.values()]
            rows.append([corruption.replace("_", " ").title(), *cells])
        rows.append(["Overall", *[f"{100 * mean(r[metric]['overall'] for r in runs):.2f}"
                                  for runs in results.values()]])
        sections.append(f"#### {title}\n\n" + table(["Corruption", *METHODS], rows))
    report = "\n\n".join(sections)
    if args.update_readme:
        path = ROOT / "README.md"
        before, block = path.read_text().split(START)
        _, after = block.split(END)
        path.write_text(before + START + "\n\n" + report + "\n\n" + END + after)
    else:
        print(report)


if __name__ == "__main__":
    main()
