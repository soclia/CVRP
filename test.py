"""Evaluate a named checkpoint against a fixed, validated test file."""
import argparse
from pathlib import Path
import statistics
import numpy as np
from src.checkpoint import load_model
from src import VARIANTS, hardware_info, load_profile, setup_runtime
from src.data import dataset_id, load_dataset, load_reference, sha256_file, write_json
from src.evaluation import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--problem_size", type=int, choices=(20, 50, 100))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--aug", type=int, choices=(1, 8))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup_batches", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected_variant", choices=tuple(VARIANTS))
    parser.add_argument("--reference", type=Path, help="JSON with exact dataset_id, source and per-instance costs")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    profile = load_profile(args.config)
    settings = profile["evaluation"]
    device = setup_runtime(args.device, args.seed, profile["tf32"])
    data = load_dataset(args.test_data, args.problem_size)
    reference, source = load_reference(args.reference, data) if args.reference else (None, None)
    model, _, metadata = load_model(args.checkpoint, device, args.expected_variant)
    repeats = args.repeats if args.repeats is not None else settings["repeats"]
    if repeats < 1:
        parser.error("repeats must be positive")
    results = []
    for i in range(repeats):
        result = evaluate_dataset(model, data, device,
                                  args.batch_size if args.batch_size is not None else settings["batch_size"],
                                  args.aug if args.aug is not None else settings["augmentation"], args.episodes,
                                  args.warmup_batches if args.warmup_batches is not None else settings["warmup_batches"])
        results.append(result)
        print(f"repeat {i + 1}/{repeats}: cost={result['mean_cost']:.6f}, seconds={result['elapsed_seconds']:.3f}")
    first = results[0]
    if any(not np.allclose(r["costs"], first["costs"], atol=2e-5, rtol=2e-6) for r in results[1:]):
        raise RuntimeError("Repeated greedy evaluations disagree")
    first.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256_file(args.checkpoint),
                 checkpoint_metadata=metadata, dataset=str(args.test_data.resolve()), dataset_id=dataset_id(data),
                 dataset_sha256=sha256_file(args.test_data), hardware=hardware_info(device),
                 repeat_seconds=[r["elapsed_seconds"] for r in results],
                 median_seconds=statistics.median(r["elapsed_seconds"] for r in results),
                 peak_allocated_mb=max((r["peak_allocated_mb"] or 0) for r in results) if device.type == "cuda" else None,
                 reference_source=source, mean_gap_percent=None)
    if reference is not None:
        costs = np.asarray(first["costs"], dtype=np.float64)
        ref = reference[:first["instances"]].numpy()
        first["gap_percent"] = ((costs / ref - 1) * 100).tolist()
        first["mean_gap_percent"] = float(np.mean(first["gap_percent"]))
    write_json(args.output, first)
    print(f"Saved {args.output.resolve()} | variant={metadata['variant']} | gap={first['mean_gap_percent']}")


if __name__ == "__main__":
    main()
