"""Greedy inference and independent feasibility/cost verification."""
import time
import numpy as np
import torch
from .CVRPEnv import CVRPEnv
from .data import KEYS, validate_data


def split_tour(tour):
    tour = [int(i) for i in tour]
    if not tour or tour[0] != 0 or tour[-1] != 0:
        raise ValueError("Tour must begin and end at depot 0")
    routes, current = [], [0]
    for node in tour[1:]:
        if node == 0:
            if len(current) > 1:
                routes.append(current + [0])
            current = [0]
        else:
            current.append(node)
    return routes


def validate_tour(tour, demands, capacity=1.0, max_vehicles=None, tolerance=1e-5):
    demands = np.asarray(demands, dtype=np.float64)
    routes = split_tour(tour)
    visited = [i for route in routes for i in route[1:-1]]
    if sorted(visited) != list(range(1, len(demands))):
        raise ValueError("Every customer must appear exactly once and all indices must be valid")
    if any(demands[route[1:-1]].sum() > capacity + tolerance for route in routes):
        raise ValueError("Tour exceeds vehicle capacity")
    if max_vehicles is not None and len(routes) > max_vehicles:
        raise ValueError("Tour exceeds vehicle-count constraint")
    return routes


def route_cost(routes, edge_weight):
    return float(sum(edge_weight[a, b] for route in routes for a, b in zip(route, route[1:])))


@torch.inference_mode()
def decode_candidates(model, data, device, aug=8, pomo_size=None):
    b, n = validate_data(data)
    env = CVRPEnv(problem_size=n, pomo_size=pomo_size or n, device=device)
    env.use_problems(*(data[k] for k in KEYS))
    env.load_problems(b, aug_factor=aug)
    state, _, _ = env.reset()
    model.eval()
    model.pre_forward(state)
    state, reward, done = env.pre_step()
    for _ in range(2 * n + 2):
        selected, _ = model(state)
        state, reward, done = env.step(selected)
        if done:
            break
    else:
        raise RuntimeError("Decoder did not produce complete tours within the CVRP step bound")
    costs = -reward.reshape(aug, b, env.pomo_size).permute(1, 0, 2).reshape(b, -1)
    tours = env.selected_node_list.reshape(aug, b, env.pomo_size, -1).permute(1, 0, 2, 3).reshape(b, aug * env.pomo_size, -1)
    return costs, tours


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_dataset(model, data, device, batch_size=200, aug=8, episodes=None, warmup_batches=3):
    available, n = validate_data(data)
    episodes = available if episodes is None else episodes
    if batch_size < 1 or not 1 <= episodes <= available or warmup_batches < 0:
        raise ValueError("Invalid batch size, episode count or warmup count")
    selected_data = {k: data[k][:episodes] for k in KEYS}
    for _ in range(warmup_batches):
        decode_candidates(model, {k: selected_data[k][:min(batch_size, episodes)] for k in KEYS}, device, aug)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    all_costs, all_tours = [], []
    for offset in range(0, episodes, batch_size):
        batch = {k: selected_data[k][offset:offset + batch_size] for k in KEYS}
        costs, tours = decode_candidates(model, batch, device, aug)
        best_cost, best_index = costs.min(dim=1)
        best_tour = tours[torch.arange(costs.size(0), device=device), best_index]
        all_costs.extend(best_cost.cpu().tolist())
        all_tours.extend(best_tour.cpu().tolist())
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
    for i, tour in enumerate(all_tours):
        demand = np.concatenate(([0.0], selected_data["node_demand"][i].numpy()))
        routes = validate_tour(tour, demand)
        coords = torch.cat((selected_data["depot_xy"][i], selected_data["node_xy"][i])).double().numpy()
        distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        if not np.isclose(route_cost(routes, distance), all_costs[i], atol=2e-5, rtol=2e-6):
            raise RuntimeError("Independent route cost disagrees with decoder reward")
    return {"costs": all_costs, "tours": all_tours, "mean_cost": float(np.mean(all_costs)),
            "elapsed_seconds": elapsed, "peak_allocated_mb": peak_mb, "instances": episodes,
            "problem_size": n, "batch_size": batch_size, "augmentation": aug, "starts": n,
            "local_search": "none", "warmup_batches": warmup_batches, "all_tours_validated": True,
            "timing_scope": "H2D transfer, encoding, greedy construction, augmentation/start selection, D2H output; excludes disk I/O, warmup, CPU route verification"}
