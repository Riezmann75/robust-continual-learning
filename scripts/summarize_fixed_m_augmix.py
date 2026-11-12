"""Print three-seed fixed-m and AugMix accuracy tables from final evaluations."""
import json
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "experiments"
SEEDS = (17, 23, 43)
METRICS = [(dataset, mode) for mode in ("class_il", "task_il")
           for dataset in ("clean", "corrupted_mean")]


def summarize(label, folders, per_seed):
    values = []
    missing = []
    for seed, candidates in zip(SEEDS, folders):
        path = next((p / "evaluation.json" for p in candidates
                     if (p / "evaluation.json").is_file()), None)
        if path is None:
            missing.append(str(seed))
            continue
        data = json.loads(path.read_text())
        row = [100 * data[group][metric] for group, metric in METRICS]
        values.append(row)
        if per_seed:
            print(" | ".join([label, str(seed), *(f"{x:.2f}" for x in row)]))
    if missing:
        print(f"{label} | Missing evaluation for seed {', '.join(missing)}")
    else:
        print(" | ".join([label, "Mean ± SD",
                          *(f"{mean(xs):.2f} ± {stdev(xs):.2f}" for xs in zip(*values))]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", action="store_true", help="Also print individual seed scores.")
    args = parser.parse_args()
    header = "Setting | Seeds | Clean class-IL | Corrupted class-IL | Clean task-IL | Corrupted task-IL"
    print(header)
    for variant, method in (("chain_random", "Chain-aware"), ("metamix_fixed_m", "MetaMix")):
        for loss in ("js", "infonce"):
            for m in ("0.1", "0.3", "0.5", "0.7", "0.9"):
                paths = []
                for seed in SEEDS:
                    if m == "0.5":
                        default = "infonce" if variant == "chain_random" else "js"
                        suffix = "" if loss == default else "_" + loss
                        slug = f"{variant}{suffix}_bs32"
                        paths.append([EXPS / f"unrolled_{slug}" / f"seed{seed}",
                                      ROOT / "results" / f"unrolled_mixing_seed{seed}" / slug])
                    else:
                        paths.append([EXPS / "fixed_m_sweep_bs32" / f"m{m}" /
                                      f"{variant}_{loss}" / f"seed{seed}"])
                summarize(f"{method}, {loss}, m={m}", paths, args.per_seed)
    print("\n" + header)
    for coefficient, folder in ((1, "derpp_augmix_js1_bs32"), (12, "derpp_augmix_bs32")):
        summarize(f"DER++ + AugMix, {coefficient}× JS",
                  [[EXPS / folder / f"seed{seed}"] for seed in SEEDS], args.per_seed)


if __name__ == "__main__":
    main()
