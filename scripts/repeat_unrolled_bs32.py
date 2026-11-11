"""Preview or run the four batch-32 replications for one seed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "Chain-aware (classifier InfoNCE)": "unrolled_chain_random_bs32",
    "Chain-aware (classifier JS)": "unrolled_chain_random_js_bs32",
    "MetaMix (learned m)": "unrolled_metamix_bs32",
    "MetaMix (fixed m=0.5)": "unrolled_metamix_fixed_m_bs32",
}
METRICS = [(dataset, mode) for dataset in ("clean", "corrupted_mean")
           for mode in ("class_il", "task_il")]


def read(path):
    return json.loads(path.read_text())


def run_on_gpu(device, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
    for command, folder in jobs:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "launcher.log"
        with path.open("ab") as log:
            start = log.tell()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            with path.open("rb") as log:
                log.seek(max(start, path.stat().st_size - 8192))
                tail = log.read().decode(errors="replace").splitlines()[-20:]
            print(f"GPU {device}: {folder.parent.name}/{folder.name} exited with "
                  f"status {result.returncode}. Log: {path}\n" + "\n".join(tail),
                  file=sys.stderr, flush=True)
            result.check_returncode()


def summarize(seed, folders):
    print(f"Accuracy (%) for seed {seed}")
    print("Method | Clean class-IL | Clean task-IL | Corrupted class-IL | Corrupted task-IL")
    for name, folder in folders.items():
        if not (folder / "complete.json").exists():
            print(f"{name} | pending")
            continue
        score = read(folder / "evaluation.json")
        cells = [f"{100 * score[dataset][mode]:.2f}" for dataset, mode in METRICS]
        print(" | ".join([name, *cells]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--gpus", type=int, choices=range(1, 5), default=4,
                        help="Number of visible GPUs to use, one experiment per GPU (default 4).")
    parser.add_argument("--data-root", type=Path, help="Override the local CIFAR dataset directory.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="Launch the experiments on the selected GPUs.")
    action.add_argument("--summary", action="store_true", help="Read completed results only.")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("Choose a seed in the uint32 range.")
    if args.seed == 17 and not args.summary:
        parser.error("Use --summary to read the original seed-17 results.")

    folders = {}
    jobs = []
    for name, config_name in CONFIGS.items():
        config = read(ROOT / "configs" / f"{config_name}.json")
        options = config["args"].copy()
        original = ROOT / options["output"]
        repeat = original.with_name(f"seed{args.seed}")
        folders[name] = repeat
        if args.seed == 17 and not (repeat / "complete.json").exists():
            folders[name] = ROOT / "results" / "unrolled_mixing_seed17" / config_name.removeprefix("unrolled_")
        if args.summary:
            continue
        dataset = ROOT / "experiments/metamix_cifar10/data"
        if args.data_root:
            dataset = args.data_root.resolve()
        elif (original / "config.json").exists():
            dataset = Path(read(original / "config.json")["root"])
        options.update(seed=args.seed, output=repeat, root=dataset)
        command = [sys.executable, str(ROOT / "reproduce" / config["runner"])]
        for key, value in options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        if not (repeat / "complete.json").exists():
            jobs.append((command, repeat))
    if jobs:
        count = min(len(jobs), args.gpus)
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = ([str(i) for i in range(count)] if mask is None else
                   [d.strip() for d in mask.split(",") if d.strip()][:count])
        if len(set(devices)) != count or "-1" in devices:
            parser.error(f"Need {count} distinct visible GPUs. Check CUDA_VISIBLE_DEVICES or reduce --gpus.")
        if args.run:
            import torch
            if torch.cuda.device_count() < count:
                parser.error(f"Requested {count} GPUs but PyTorch sees {torch.cuda.device_count()}.")
        for i, (command, _) in enumerate(jobs):
            print(f"CUDA_VISIBLE_DEVICES={devices[i % count]} {shlex.join(command)}", flush=True)
        if args.run:
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [pool.submit(run_on_gpu, device, jobs[i::count])
                           for i, device in enumerate(devices)]
                for future in futures:
                    future.result()
    if args.run or args.summary:
        summarize(args.seed, folders)


if __name__ == "__main__":
    main()
