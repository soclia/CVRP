"""Create fixed synthetic datasets; these are not the cloud paper test set."""
import argparse
from pathlib import Path
import torch
from src import ROOT
from src.CVRProblemDef import get_random_problems
from src.checkpoint import save_checkpoint
from src.data import dataset_id, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem_size", type=int, choices=(20, 50, 100), default=100)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    output = args.output or ROOT / "data" / f"cvrp{args.problem_size}_{args.split}_seed{args.seed}.pt"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fixed dataset: {output}")
    generator = torch.Generator().manual_seed(args.seed)
    depot, nodes, demand = get_random_problems(args.episodes, args.problem_size, generator)
    data = dict(depot_xy=depot, node_xy=nodes, node_demand=demand,
                metadata={"seed": args.seed, "split": args.split, "origin": "local synthetic generator; not cloud paper data"})
    save_checkpoint(output, data)
    manifest = {"file": output.name, "dataset_id": dataset_id(data), "sha256": sha256_file(output),
                "instances": args.episodes, "problem_size": args.problem_size, **data["metadata"]}
    write_json(output.with_suffix(".json"), manifest)
    print(f"Saved {output} | dataset_id={manifest['dataset_id']}")


if __name__ == "__main__":
    main()
