"""List, train or evaluate actual ablations; missing runs are never replaced by random weights."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from src import ROOT, VARIANTS, load_profile, model_params, setup_runtime
from src.CVRPModel import CVRPModel
from src.checkpoint import load_model
from src.data import dataset_id, load_dataset, write_json
from src.evaluation import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("list", "train", "evaluate"), required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236])
    parser.add_argument("--problem_size", type=int, choices=(20, 50, 100), default=100)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validation_data", type=Path)
    parser.add_argument("--test_data", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON: runs=[{variant, seed, checkpoint}]; paths relative to manifest")
    parser.add_argument("--save_dir", type=Path, default=ROOT / "runs/ablations")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/ablation.json"))
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("variants and seeds must be unique")
    profile = load_profile(args.config)
    if args.mode == "list":
        for name in args.variants:
            count = sum(p.numel() for p in CVRPModel(**model_params(name, profile)).parameters())
            print(f"{name}: {count:,} trainable parameters (configuration only)")
        return
    if args.mode == "train":
        if not args.validation_data:
            parser.error("train mode requires --validation_data")
        runs = []
        for variant in args.variants:
            for seed in args.seeds:
                folder = args.save_dir / f"{variant}-cvrp{args.problem_size}-seed{seed}"
                command = [sys.executable, str(ROOT / "train.py"), "--variant", variant, "--seed", str(seed),
                           "--problem_size", str(args.problem_size), "--validation_data", str(args.validation_data.resolve()),
                           "--save_dir", str(folder.resolve()), "--device", args.device]
                if args.config:
                    command += ["--config", str(args.config.resolve())]
                subprocess.run(command, check=True)
                runs.append({"variant": variant, "seed": seed, "checkpoint": str((folder / "best.pt").resolve())})
                write_json(args.output, {"runs": runs})
        return
    if not args.manifest or not args.test_data:
        parser.error("evaluate mode requires --manifest and --test_data")
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))["runs"]
    index = {(r["variant"], r["seed"]): r for r in entries}
    if len(index) != len(entries):
        raise ValueError("Duplicate variant/seed entries in manifest")
    # Validate completeness before any evaluation starts.
    requested = []
    for variant in args.variants:
        for seed in args.seeds:
            if (variant, seed) not in index:
                raise ValueError(f"Missing trained run: {variant}, seed {seed}")
            path = Path(index[variant, seed]["checkpoint"])
            path = path if path.is_absolute() else args.manifest.parent / path
            if not path.is_file():
                raise FileNotFoundError(path)
            requested.append((variant, seed, path))
    device = setup_runtime(args.device)
    data = load_dataset(args.test_data, args.problem_size)
    rows = []
    for variant, seed, path in requested:
        model, checkpoint, metadata = load_model(path, device, variant)
        recorded_seed = checkpoint.get("training", {}).get("seed", metadata.get("training_seed"))
        if recorded_seed != seed:
            raise ValueError(f"Checkpoint training seed {recorded_seed} does not match manifest seed {seed}")
        result = evaluate_dataset(model, data, device, args.batch_size or profile["evaluation"]["batch_size"],
                                  profile["evaluation"]["augmentation"], args.episodes)
        rows.append({"variant": variant, "seed": seed, "checkpoint": str(path.resolve()), "result": result})
        print(f"{variant}, seed={seed}: cost={result['mean_cost']:.6f}")
    summaries = []
    for variant in args.variants:
        costs = [r["result"]["mean_cost"] for r in rows if r["variant"] == variant]
        summaries.append({"variant": variant, "training_runs": len(costs), "mean_cost": statistics.mean(costs),
                          "sample_std": statistics.stdev(costs) if len(costs) > 1 else None})
    write_json(args.output, {"dataset_id": dataset_id(data), "runs": rows, "summaries": summaries,
                             "note": "No reference costs supplied here; use test.py --reference for per-instance gaps"})


if __name__ == "__main__":
    main()
