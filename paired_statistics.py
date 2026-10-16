"""Paired relative-gap inference on verified fixed-dataset evaluation outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from src.data import write_json


def paired_summary(ours, baseline, reference, bootstrap_repeats=10000, seed=20261014):
    ours, baseline, reference = (np.asarray(v, dtype=np.float64) for v in (ours, baseline, reference))
    if ours.ndim == 1:
        ours = ours[None]
    if baseline.ndim == 1:
        baseline = baseline[None]
    if ours.ndim != 2 or baseline.ndim != 2 or reference.ndim != 1:
        raise ValueError("Expected seed-by-instance costs and one reference vector")
    n = reference.size
    if n < 2 or ours.shape[1] != n or baseline.shape[1] != n:
        raise ValueError("Paired vectors require identical instance counts, at least two")
    if any(not np.isfinite(v).all() for v in (ours, baseline, reference)) or np.any(reference <= 0):
        raise ValueError("Costs must be finite and reference costs positive")
    if bootstrap_repeats < 2:
        raise ValueError("bootstrap_repeats must be at least two")
    # Fixed trained models; averaging seeds does not make seeds independent instances.
    delta = 100 * (ours.mean(axis=0) - baseline.mean(axis=0)) / reference
    if np.all(delta == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="approx")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_repeats)
    for start in range(0, bootstrap_repeats, 128):
        count = min(128, bootstrap_repeats - start)
        samples[start:start+count] = delta[rng.integers(0, n, size=(count, n))].mean(axis=1)
    return {"instances": n, "ours_models": len(ours), "baseline_models": len(baseline),
            "mean_difference_pp": float(delta.mean()), "confidence_interval_95_pp": np.quantile(samples, [.025, .975]).tolist(),
            "wilcoxon_statistic": statistic, "wilcoxon_p_value": p_value,
            "alternative": "two-sided", "zero_method": "wilcox", "wilcoxon_method": "approx",
            "confidence_method": "paired instance percentile bootstrap of the mean difference",
            "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": seed,
            "seed_aggregation": "mean prediction cost across the supplied fixed trained models, separately for each instance",
            "inference_scope": "conditional on the supplied trained models; interval resamples instances, not training runs"}


def read_outputs(paths):
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for record in records:
        if not record.get("all_tours_validated") or not record.get("dataset_id"):
            raise ValueError("Inputs must be route-validated test.py evaluation outputs")
    if len({r.get("checkpoint_sha256") for r in records}) != len(records):
        raise ValueError("Repeated checkpoint output cannot represent independent trained models")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=Path("results/paired-statistics.json"))
    args = parser.parse_args()
    ours, baseline = read_outputs(args.ours), read_outputs(args.baseline)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    dataset_ids = {r["dataset_id"] for r in ours + baseline} | {reference["dataset_id"]}
    protocols = {(r["instances"], r["problem_size"], r.get("augmentation"), r.get("starts"), r.get("local_search"))
                 for r in ours + baseline}
    if len(dataset_ids) != 1 or len(protocols) != 1:
        raise ValueError("Paired outputs must use the same full dataset order and decoding protocol")
    if any(r["instances"] != len(reference["costs"]) for r in ours + baseline):
        raise ValueError("Use complete dataset evaluations; subsets require an explicitly matched reference")
    result = paired_summary([r["costs"] for r in ours], [r["costs"] for r in baseline], reference["costs"],
                            args.bootstrap_repeats, args.seed)
    result.update(dataset_id=reference["dataset_id"], ours_files=list(map(str, args.ours)),
                  baseline_files=list(map(str, args.baseline)), reference_source=reference["source"],
                  ours_training_seeds=[r["checkpoint_metadata"].get("training_seed") for r in ours],
                  baseline_training_seeds=[r["checkpoint_metadata"].get("training_seed") for r in baseline])
    write_json(args.output, result)
    print(f"Difference={result['mean_difference_pp']:.6f} pp; 95% CI={result['confidence_interval_95_pp']}; p={result['wilcoxon_p_value']:.6g}")


if __name__ == "__main__":
    main()
