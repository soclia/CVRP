"""REINFORCE or frozen-teacher training with fixed validation and epoch resume."""
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from .CVRPEnv import CVRPEnv
from .CVRPModel import CVRPModel, _get_encoding
from .CVRProblemDef import get_random_problems
from .checkpoint import read_checkpoint, save_checkpoint, initialize_model
from . import hardware_info, variant_name
from .data import KEYS, dataset_id, sha256_file, write_json
from .evaluation import decode_candidates, evaluate_dataset, synchronize


def teacher_guided_loss(student, teacher, batch, device, starts=4, hard_weight=0.2):
    """Distill a frozen teacher on fresh training instances, retaining student modules.

    Select the teacher's best ``starts`` of N greedy rollouts (no augmentation),
    then match its action distributions along those feasible trajectories. The
    loss mixes forward KL and selected-action negative log likelihood, averaged
    over active decisions per trajectory. Forced first moves and padding are
    excluded. This is an optional training recipe, separate from REINFORCE.
    """
    b, n = batch["node_demand"].shape
    if not 1 <= starts <= n or not 0 <= hard_weight <= 1:
        raise ValueError("Invalid teacher starts or hard-label weight")
    if batch.get("metadata", {}).get("split") in ("validation", "test"):
        raise ValueError("Teacher training cannot consume validation or test data")
    costs, tours = decode_candidates(teacher, batch, device, aug=1)
    best = costs.topk(starts, dim=1, largest=False).indices
    target = tours.gather(1, best[:, :, None].expand(-1, -1, tours.shape[-1])).clone()
    teacher_cost = costs.gather(1, best).mean().item()
    env = CVRPEnv(problem_size=n, pomo_size=starts, device=device)
    env.use_problems(*(batch[k] for k in KEYS))
    env.load_problems(b)
    reset, _, _ = env.reset()
    student.train()
    student.pre_forward(reset)
    with torch.no_grad():
        teacher.pre_forward(reset)
    state, _, done = env.pre_step()
    kl_sum = torch.zeros(b, starts, device=device)
    nll_sum, counts = torch.zeros_like(kl_sum), torch.zeros_like(kl_sum)
    feasible = torch.ones((), dtype=torch.bool, device=device)
    for step in range(target.shape[-1]):
        selected = target[:, :, step]
        if step >= 2:
            active = ~state.finished
            last = _get_encoding(student.encoded_nodes, state.current_node)
            inputs = (last, state.current_node, state.load, state.ninf_mask, state.visited_ninf_flag)
            if getattr(student, "decoder_checkpointing", False):
                inputs = (last, state.current_node, state.load.clone(), state.ninf_mask, state.visited_ninf_flag.clone())
                probs = activation_checkpoint(student.decoder, *inputs, use_reentrant=False, preserve_rng_state=False)
            else:
                probs = student.decoder(*inputs)
            with torch.no_grad():
                teacher_last = _get_encoding(teacher.encoded_nodes, state.current_node)
                teacher_probs = teacher.decoder(teacher_last, state.current_node, state.load,
                                                state.ninf_mask, state.visited_ninf_flag)
            log_probs = probs.clamp_min(1e-30).log()
            teacher_logs = teacher_probs.clamp_min(1e-30).log()
            kl_sum = kl_sum + (teacher_probs * (teacher_logs - log_probs)).sum(-1) * active
            nll_sum = nll_sum - log_probs.gather(-1, selected[:, :, None]).squeeze(-1) * active
            counts = counts + active
        feasible = feasible & torch.isfinite(state.ninf_mask.gather(-1, selected[:, :, None])).all()
        state, _, done = env.step(selected)
        if done:
            break
    if not done or not feasible:
        raise RuntimeError("Teacher trajectories must be feasible and complete")
    kl, nll = (kl_sum / counts.clamp_min(1)).mean(), (nll_sum / counts.clamp_min(1)).mean()
    loss = (1 - hard_weight) * kl + hard_weight * nll
    return loss, {"kl": kl.item(), "nll": nll.item(), "teacher_cost": teacher_cost}


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "data_generator": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["data_generator"])


class CVRPTrainer:
    def __init__(self, model_params, training, validation_data, device, save_dir, resume=None, warm_start=None, initialize_from=None,
                 teacher_checkpoint=None):
        self.params, self.training, self.device = model_params, dict(training), device
        training = self.training
        for key in ("validation_interval", "validation_screen_size"):
            if training.get(key, 1) < 1:
                raise ValueError(f"{key} must be positive")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = CVRPModel(**model_params).to(device)
        self.model.decoder_checkpointing = training.get("decoder_checkpoint", False)
        self.teacher = None
        if teacher_checkpoint is not None:
            from .checkpoint import load_model
            self.teacher, _, teacher_metadata = load_model(teacher_checkpoint, device, "pomo")
            self.teacher.requires_grad_(False)
            training.setdefault("teacher_starts", 4)
            training.setdefault("teacher_hard_weight", 0.2)
            training["teacher_metadata"] = {"path": str(Path(teacher_checkpoint).resolve()),
                                             "sha256": sha256_file(teacher_checkpoint),
                                             "model_params": teacher_metadata["model_params"]}
            training["objective"] = "teacher_forward_kl_and_trajectory_nll"
        elif training.get("objective") == "teacher_forward_kl_and_trajectory_nll":
            raise ValueError("Teacher-guided training requires --teacher_checkpoint")
        if sum(x is not None for x in (resume, warm_start, initialize_from)) > 1:
            raise ValueError("resume, warm_start and initialize_from are mutually exclusive")
        # Independent online-data stream, separate from initialization and action sampling.
        self.generator = torch.Generator().manual_seed(training["seed"] + 1000000007)
        n = training["problem_size"]
        self.env = CVRPEnv(problem_size=n, pomo_size=n, device=device, generator=self.generator)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=training["milestones"], gamma=0.1)
        self.validation = validation_data
        self.validation_id = dataset_id(validation_data)
        self.start_epoch, self.best_val_cost, self.history = 1, float("inf"), []
        if resume:
            c = read_checkpoint(resume)
            if c["model_params"] != model_params:
                raise ValueError("Resume architecture/semantics differ from saved model_params")
            ignored = ("epochs", "save_interval", "warm_start_metadata", "initialization_metadata")
            comparable = {k: v for k, v in training.items() if k not in ignored}
            previous = {k: v for k, v in c["training"].items() if k not in ignored}
            if comparable != previous or c["validation_id"] != self.validation_id:
                raise ValueError("Resume requires identical training configuration and fixed validation data")
            if c["hardware"] != hardware_info(device):
                raise ValueError("Exact resume requires the same hardware, PyTorch version and precision configuration")
            if Path(resume).resolve().parent != self.save_dir.resolve():
                raise ValueError("Resume must save to the checkpoint's directory so the previous best checkpoint is retained")
            if "warm_start_metadata" in c["training"]:
                self.training["warm_start_metadata"] = c["training"]["warm_start_metadata"]
            if "initialization_metadata" in c["training"]:
                self.training["initialization_metadata"] = c["training"]["initialization_metadata"]
            self.model.load_state_dict(c["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(c["optimizer_state_dict"])
            self.scheduler.load_state_dict(c["scheduler_state_dict"])
            self.start_epoch, self.best_val_cost, self.history = c["epoch"] + 1, c["best_val_cost"], c["history"]
            restore_rng(c["rng_state"], self.generator)
        elif warm_start:
            from .checkpoint import load_model
            model, _, metadata = load_model(warm_start, device, variant_name(model_params))
            if any(model.model_params[k] != model_params[k] for k in model_params if k in model.model_params):
                raise ValueError("Warm start has different semantics; use a matching configuration")
            self.model.load_state_dict(model.state_dict(), strict=True)
            self.training["warm_start_metadata"] = metadata
        elif initialize_from:
            self.training["initialization_metadata"] = initialize_model(self.model, initialize_from)

    def _loss(self, batch_size):
        if self.teacher is not None:
            batch = dict(zip(KEYS, get_random_problems(batch_size, self.env.problem_size, self.generator)))
            loss, metrics = teacher_guided_loss(self.model, self.teacher, batch, self.device,
                                               self.training["teacher_starts"], self.training["teacher_hard_weight"])
            return loss, torch.tensor(metrics["teacher_cost"], device=self.device)
        self.model.train()
        self.env.load_problems(batch_size)
        reset, _, _ = self.env.reset()
        self.model.pre_forward(reset)
        state, reward, done = self.env.pre_step()
        log_probability = torch.zeros(batch_size, self.env.pomo_size, device=self.device)
        for _ in range(2 * self.env.problem_size + 2):
            selected, probability = self.model(state)
            # Finished depot padding has probability one, independent of graph.
            probability = torch.where(state.finished, torch.ones_like(probability), probability)
            log_probability = log_probability + probability.log()
            state, reward, done = self.env.step(selected)
            if done:
                break
        else:
            raise RuntimeError("Training rollout exceeded step bound")
        advantage = reward - reward.mean(dim=1, keepdim=True)
        loss = (-advantage * log_probability).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite policy loss")
        return loss, -reward.max(dim=1).values.mean()

    def _train_epoch(self):
        processed, score_sum, loss_sum = 0, 0.0, 0.0
        while processed < self.training["episodes_per_epoch"]:
            effective = min(self.training["batch_size"], self.training["episodes_per_epoch"] - processed)
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            while accumulated < effective:
                micro = min(self.training["micro_batch_size"], effective - accumulated)
                loss, score = self._loss(micro)
                (loss * (micro / effective)).backward()
                loss_sum += loss.item() * micro
                score_sum += score.item() * micro
                accumulated += micro
            finite = [torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]
            if finite and not torch.stack(finite).all():
                raise RuntimeError("Nonfinite gradients")
            if self.training.get("max_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training["max_grad_norm"], error_if_nonfinite=True)
            self.optimizer.step()
            processed += effective
        return score_sum / processed, loss_sum / processed

    def run(self):
        if self.start_epoch > self.training["epochs"]:
            raise ValueError("No remaining epochs: --epochs must exceed checkpoint epoch")
        if self.start_epoch == 1 and any(k in self.training for k in ("warm_start_metadata", "initialization_metadata")):
            initial = evaluate_dataset(self.model, self.validation, self.device,
                                       self.training["validation_batch_size"], self.training["validation_aug"], warmup_batches=0)
            self.best_val_cost = initial["mean_cost"]
            save_checkpoint(self.save_dir / "best.pt", self._snapshot(0))
            write_json(self.save_dir / "initial-validation.json", {"mean_cost": self.best_val_cost,
                                                                   "validation_id": self.validation_id})
            print(f"initial validation={self.best_val_cost:.6f}; saved as epoch 0 candidate", flush=True)
        write_json(self.save_dir / "config.json", {"model_params": self.params, "training": self.training,
                                                   "hardware": hardware_info(self.device), "validation_id": self.validation_id})
        for epoch in range(self.start_epoch, self.training["epochs"] + 1):
            synchronize(self.device)
            started = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            score, loss = self._train_epoch()
            interval = self.training.get("validation_interval", 1)
            full_validation = epoch % interval == 0 or epoch == self.training["epochs"]
            screen_size = min(self.training.get("validation_screen_size", 128), len(self.validation["node_xy"]))
            validation = self.validation if full_validation else {k: self.validation[k][:screen_size] for k in KEYS}
            val = evaluate_dataset(self.model, validation, self.device,
                                   self.training["validation_batch_size"], self.training["validation_aug"], warmup_batches=0)
            self.scheduler.step()  # after epoch, saved state already contains this decay
            synchronize(self.device)
            cost_name = "teacher_cost" if self.teacher is not None else "train_cost"
            loss_name = "distillation_loss" if self.teacher is not None else "policy_loss"
            row = {"epoch": epoch, cost_name: score, loss_name: loss,
                   "validation_cost": val["mean_cost"] if full_validation else None,
                   "learning_rate": lr, "next_learning_rate": self.optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.perf_counter() - started}
            self.history.append(row)
            if interval > 1:
                row["screen_validation_cost"] = float(np.mean(val["costs"][:screen_size]))
            is_best = full_validation and val["mean_cost"] < self.best_val_cost
            if full_validation:
                self.best_val_cost = min(self.best_val_cost, val["mean_cost"])
            checkpoint = self._snapshot(epoch)
            save_checkpoint(self.save_dir / "latest.pt", checkpoint)
            if is_best:
                save_checkpoint(self.save_dir / "best.pt", checkpoint)
            if epoch % self.training["save_interval"] == 0 or epoch == self.training["epochs"]:
                save_checkpoint(self.save_dir / f"epoch-{epoch}.pt", checkpoint)
            write_json(self.save_dir / "history.json", self.history)
            val_label = "val" if full_validation else "screen (not used for selection)"
            print(f"epoch {epoch}: {cost_name}={score:.6f} loss={loss:.6f} {val_label}={val['mean_cost']:.6f} lr={lr:g}", flush=True)
            if (self.save_dir / "STOP_AFTER_EPOCH").exists():
                print("Stopped after saving this epoch; best and latest checkpoints retained.", flush=True)
                break
        return self.history

    def _snapshot(self, epoch):
        return {"format_version": 1, "epoch": epoch, "model_params": self.params,
                "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(), "rng_state": rng_state(self.generator),
                "training": self.training, "hardware": hardware_info(self.device), "history": self.history,
                "validation_id": self.validation_id, "best_val_cost": self.best_val_cost,
                "metadata": {"variant": variant_name(self.params), "role": "training_run"}}
