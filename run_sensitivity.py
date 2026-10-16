"""Prepare, train and evaluate clustering/anchor-fraction sensitivity runs."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from copy import deepcopy
from src import ROOT, load_profile, model_params, setup_runtime
from src.checkpoint import load_model
from src.data import dataset_id, load_dataset, load_reference, write_json
from src.evaluation import evaluate_dataset

CASES = {"baseline": {}, "iterations-1": {"hacd_anchor_kmeans_iters": 1},
         "iterations-2": {"hacd_anchor_kmeans_iters": 2},
         "iterations-5": {"hacd_anchor_kmeans_iters": 5},
         "fraction-0.1": {"hacd_anchor_demand_ratio": 0.1, "hacd_anchor_peripheral_ratio": 0.1},
         "fraction-0.3": {"hacd_anchor_demand_ratio": 0.3, "hacd_anchor_peripheral_ratio": 0.3}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--problem_size", type=int, choices=(20, 50, 100), default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--validation_data", type=Path)
    parser.add_argument("--test_data", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--episodes", type=int, help="Evaluation subset; omitted evaluates the full fixed file")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs/sensitivity")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        parser.error("cases and seeds must be unique")
    base = load_profile(args.config)
    configurations = []
    for case in args.cases:
        profile = deepcopy(base)
        profile["model"].update(CASES[case])
        config_file = args.output_dir / "configs" / f"{case}.json"
        configurations.append((case, profile, config_file))
    if args.mode in ("prepare", "train"):
        for _, profile, path in configurations:
            write_json(path, profile)
        if args.mode == "prepare":
            print(f"Prepared {len(configurations)} configurations in {args.output_dir / 'configs'}")
            return
        if not args.validation_data:
            parser.error("train mode requires --validation_data")
        runs = []
        for case, profile, config_file in configurations:
            for seed in args.seeds:
                folder = args.output_dir / case / f"seed-{seed}"
                command = [sys.executable, str(ROOT / "train.py"), "--config", str(config_file.resolve()),
                           "--problem_size", str(args.problem_size), "--variant", "full", "--seed", str(seed),
                           "--validation_data", str(args.validation_data.resolve()), "--device", args.device,
                           "--save_dir", str(folder.resolve())]
                if args.epochs is not None:
                    command += ["--epochs", str(args.epochs)]
                subprocess.run(command, check=True)
                runs.append({"case": case, "seed": seed, "checkpoint": str((folder / "best.pt").resolve()),
                             "model_params": model_params("full", profile)})
                write_json(args.output_dir / "runs.json", {"runs": runs})
        return
    if not args.test_data:
        parser.error("evaluate mode requires --test_data")
    records = json.loads((args.output_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    index = {(r["case"], r["seed"]): r for r in records}
    if len(index) != len(records):
        raise ValueError("Duplicate sensitivity run")
    for case in args.cases:
        for seed in args.seeds:
            if (case, seed) not in index:
                raise ValueError(f"Missing trained run: {case}, seed {seed}")
            if not Path(index[case, seed]["checkpoint"]).is_file():
                raise FileNotFoundError(index[case, seed]["checkpoint"])
    data = load_dataset(args.test_data, args.problem_size)
    reference, source = load_reference(args.reference, data) if args.reference else (None, None)
    device = setup_runtime(args.device, tf32=base["tf32"])
    results = []
    for case in args.cases:
        for seed in args.seeds:
            record = index[case, seed]
            model, checkpoint, _ = load_model(record["checkpoint"], device, "full")
            if checkpoint.get("training", {}).get("seed") != seed or checkpoint["model_params"] != record["model_params"]:
                raise ValueError("Checkpoint seed/configuration differs from sensitivity manifest")
            result = evaluate_dataset(model, data, device, base["evaluation"]["batch_size"],
                                      base["evaluation"]["augmentation"], args.episodes)
            gap = None
            if reference is not None:
                gap = statistics.mean((cost / ref - 1) * 100 for cost, ref in
                                      zip(result["costs"], reference[:result["instances"]].tolist()))
            results.append({"case": case, "seed": seed, "mean_gap_percent": gap, "result": result})
    summaries = []
    for case in args.cases:
        selected = [r for r in results if r["case"] == case]
        values = [r["mean_gap_percent"] if reference is not None else r["result"]["mean_cost"] for r in selected]
        summaries.append({"case": case, "seeds": len(values), "metric": "gap_percent" if reference is not None else "cost",
                          "mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None})
    write_json(args.output_dir / "evaluation.json", {"dataset_id": dataset_id(data), "reference_source": source,
                                                     "runs": results, "summaries": summaries})
    print(f"Saved {args.output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
