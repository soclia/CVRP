"""CVRPLIB EUC_2D evaluation with explicit feasibility and fleet constraints."""
import argparse
from pathlib import Path
import re
import time
import numpy as np
import torch
import vrplib
from src.checkpoint import load_model
from src import ROOT, hardware_info, setup_runtime
from src.data import sha256_file, write_json
from src.evaluation import decode_candidates, route_cost, synchronize, validate_tour

BEST_KNOWN = {"A-n32-k5": 784, "A-n36-k5": 799, "A-n37-k5": 669, "A-n38-k5": 730,
              "A-n39-k5": 822, "A-n44-k6": 937, "A-n45-k6": 944, "A-n46-k7": 914,
              "A-n48-k7": 1073, "A-n63-k10": 1314, "A-n64-k9": 1401, "A-n69-k9": 1159,
              "B-n34-k5": 788, "B-n35-k5": 955, "B-n45-k6": 678, "B-n51-k7": 1032,
              "P-n50-k8": 631, "P-n51-k10": 741, "P-n70-k10": 827}


def integer_distances(coords):
    coords = np.asarray(coords, dtype=np.float64)
    return np.floor(np.linalg.norm(coords[:, None] - coords[None], axis=-1) + 0.5).astype(np.int64)


def intra_2opt(edge_weight, route):
    best = list(route)
    while True:
        improved = False
        for i in range(len(best) - 3):
            for j in range(i + 2, len(best) - 1):
                a, b, c, d = best[i], best[i + 1], best[j], best[j + 1]
                if edge_weight[a, c] + edge_weight[b, d] < edge_weight[a, b] + edge_weight[c, d]:
                    best[i + 1:j + 1] = reversed(best[i + 1:j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return best


def choose_candidate(tours, demands, capacity, edges, max_vehicles=None, local_search="none"):
    best, capacity_valid, fleet_valid, route_counts = None, 0, 0, []
    for tour in tours:
        try:
            routes = validate_tour(tour, demands, capacity, tolerance=0)
        except ValueError:
            continue
        capacity_valid += 1
        route_counts.append(len(routes))
        if max_vehicles is not None and len(routes) > max_vehicles:
            continue
        fleet_valid += 1
        if local_search == "2opt":
            routes = [intra_2opt(edges, r) for r in routes]
        cost = int(route_cost(routes, edges))
        if best is None or cost < best["cost"]:
            best = {"cost": cost, "routes": routes, "vehicles": len(routes)}
    diagnostics = {"candidates": len(tours), "capacity_valid_candidates": capacity_valid,
                   "fleet_valid_candidates": fleet_valid, "minimum_candidate_vehicles": min(route_counts) if route_counts else None}
    if best is None:
        return {"status": "no_feasible_candidate", "cost": None, "routes": None, "vehicles": None, **diagnostics}
    validate_tour([0] + [node for route in best["routes"] for node in route[1:]], demands, capacity, max_vehicles, tolerance=0)
    return {"status": "feasible", **best, **diagnostics}


def solve_instance(model, instance_data, device, aug=8, max_vehicles=None, local_search="none"):
    if instance_data.get("edge_weight_type") != "EUC_2D":
        raise ValueError("Only EUC_2D is supported; no silent metric conversion")
    coords = np.asarray(instance_data["node_coord"], dtype=np.float64)
    demand = np.asarray(instance_data["demand"], dtype=np.float64)
    depots = np.asarray(instance_data.get("depot", [0])).reshape(-1)
    if len(depots) != 1:
        raise ValueError("Exactly one depot is required")
    depot = int(depots[0])
    order = np.array([depot] + [i for i in range(len(coords)) if i != depot])
    coords, demand = coords[order], demand[order]
    capacity = float(instance_data["capacity"])
    if capacity <= 0 or demand[0] != 0 or np.any(demand[1:] <= 0) or np.any(demand[1:] > capacity):
        raise ValueError("Invalid demands/capacity")
    span = float(np.ptp(coords, axis=0).max())
    normalized = (coords - coords.min(axis=0)) / (span if span > 0 else 1)
    data = {"depot_xy": torch.tensor(normalized[None, :1], dtype=torch.float32),
            "node_xy": torch.tensor(normalized[None, 1:], dtype=torch.float32),
            "node_demand": torch.tensor(demand[None, 1:] / capacity, dtype=torch.float32)}
    edges = integer_distances(coords)
    synchronize(device)
    started = time.perf_counter()
    _, tours = decode_candidates(model, data, device, aug)
    tours = tours[0].cpu().tolist()
    synchronize(device)
    result = choose_candidate(tours, demand, capacity, edges, max_vehicles, local_search)
    result.update(elapsed_seconds=time.perf_counter() - started, problem_size=len(coords) - 1,
                  augmentation=aug, starts=len(coords) - 1, max_vehicles=max_vehicles,
                  local_search=local_search, metric="EUC_2D integer round-half-up", normalization="translation + shared maximum coordinate span")
    if result["routes"] is not None:
        result["routes_original_1based"] = [[int(order[i]) + 1 for i in route] for route in result["routes"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data/cvrplib_instances")
    parser.add_argument("--instances", nargs="+", choices=tuple(BEST_KNOWN))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aug", type=int, choices=(1, 8), default=8)
    parser.add_argument("--fleet", choices=("filename-k", "unlimited"), default="unlimited")
    parser.add_argument("--local_search", choices=("none", "2opt"), default="none")
    parser.add_argument("--output", type=Path, default=Path("results/cvrplib.json"))
    args = parser.parse_args()
    device = setup_runtime(args.device)
    model, _, metadata = load_model(args.checkpoint, device)
    rows = []
    for name in args.instances or BEST_KNOWN:
        path = args.data_dir / f"{name}.vrp"
        if not path.is_file():
            raise FileNotFoundError(path)
        instance = vrplib.read_instance(str(path))
        limit = int(re.search(r"-k(\d+)$", name).group(1)) if args.fleet == "filename-k" else None
        result = solve_instance(model, instance, device, args.aug, limit, args.local_search)
        result.update(name=name, instance_sha256=sha256_file(path), best_known=BEST_KNOWN[name], gap_percent=None)
        if result["status"] == "feasible":
            result["gap_percent"] = (result["cost"] / BEST_KNOWN[name] - 1) * 100
        rows.append(result)
        print(f"{name}: status={result['status']}, cost={result['cost']}, gap={result['gap_percent']}", flush=True)
    feasible = [row for row in rows if row["status"] == "feasible"]
    complete = len(feasible) == len(rows)
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
              "checkpoint_metadata": metadata, "hardware": hardware_info(device), "fleet_protocol": args.fleet,
              "feasible_instances": len(feasible), "requested_instances": len(rows), "complete_coverage": complete,
              "mean_gap_percent": float(np.mean([r["gap_percent"] for r in rows])) if complete else None,
              "conditional_mean_gap_percent": float(np.mean([r["gap_percent"] for r in feasible])) if feasible else None,
              "bks_source": "CVRPLIB classic A/B/P best-known values; https://vrp.galgos.inf.puc-rio.br/",
              "timing_scope": "H2D, encoding, construction, D2H, feasibility filtering, integer scoring and optional 2opt; excludes disk I/O and normalization",
              "instances": rows}
    write_json(args.output, report)
    print(f"Saved {args.output.resolve()} | coverage={len(feasible)}/{len(rows)}")


if __name__ == "__main__":
    main()
