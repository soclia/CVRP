"""Train an explicit HACD/POMO variant with a recorded FP32 configuration."""
import argparse
from pathlib import Path
import warnings
from src import ROOT, VARIANTS, load_profile, model_params, setup_runtime
from src.data import load_dataset
from src.CVRPTrainer import CVRPTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem_size", type=int, choices=(20, 50, 100), default=100)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validation_data", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm_start", type=Path)
    parser.add_argument("--initialize_from", type=Path, help="Matching tensors for a new run, with target semantics recorded explicitly")
    parser.add_argument("--teacher_checkpoint", type=Path, help="Optional frozen POMO teacher; selects policy distillation instead of REINFORCE")
    parser.add_argument("--teacher_starts", type=int, default=4)
    parser.add_argument("--teacher_hard_weight", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--decoder_checkpoint", action=argparse.BooleanOptionalAction, default=None)
    for name in ("epochs", "episodes", "batch_size", "micro_batch_size", "validation_batch_size", "save_interval",
                 "validation_interval", "validation_screen_size"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    args = parser.parse_args()
    if sum(x is not None for x in (args.resume, args.warm_start, args.initialize_from)) > 1:
        parser.error("resume, warm_start and initialize_from are mutually exclusive")
    profile = load_profile(args.config)
    training = dict(profile["training"][str(args.problem_size)])
    training.setdefault("decoder_checkpoint", False)
    if args.decoder_checkpoint is not None:
        training["decoder_checkpoint"] = args.decoder_checkpoint
    training.update(profile["optimizer"], problem_size=args.problem_size, seed=args.seed,
                    episodes_per_epoch=profile["training"]["episodes_per_epoch"],
                    validation_batch_size=profile["evaluation"]["batch_size"],
                    validation_aug=profile["training"]["validation_aug"], save_interval=500)
    for key in ("epochs", "batch_size", "micro_batch_size", "validation_batch_size", "save_interval", "lr", "weight_decay"):
        if getattr(args, key) is not None:
            training[key] = getattr(args, key)
    if args.episodes is not None:
        training["episodes_per_epoch"] = args.episodes
    for key in ("validation_interval", "validation_screen_size"):
        value = getattr(args, key)
        if value is not None:
            if value < 1:
                parser.error(f"{key} must be positive")
            training[key] = value
    if any(training[k] < 1 for k in ("epochs", "batch_size", "micro_batch_size", "validation_batch_size", "save_interval", "episodes_per_epoch")):
        parser.error("All counts must be positive")
    if training["lr"] <= 0 or training["weight_decay"] < 0:
        parser.error("Invalid optimizer parameters")
    if args.max_grad_norm is not None:
        if args.max_grad_norm <= 0:
            parser.error("max_grad_norm must be positive")
        training["max_grad_norm"] = args.max_grad_norm
    if args.teacher_checkpoint is not None:
        if not 1 <= args.teacher_starts <= args.problem_size or not 0 <= args.teacher_hard_weight <= 1:
            parser.error("Invalid teacher starts or hard-label weight")
        training.update(teacher_starts=args.teacher_starts, teacher_hard_weight=args.teacher_hard_weight)
    device = setup_runtime(args.device, args.seed, profile["tf32"])
    validation = load_dataset(args.validation_data, args.problem_size)
    if validation.get("metadata", {}).get("split") == "test":
        raise ValueError("A dataset labelled test cannot be used for best-checkpoint selection")
    if len(validation["node_xy"]) != profile["training"]["validation_size"]:
        warnings.warn("Validation set size differs from the default 1,000 instances.")
    save_dir = args.save_dir or (args.resume.resolve().parent if args.resume else ROOT / "runs" / f"{args.variant}-cvrp{args.problem_size}-seed{args.seed}")
    if not args.resume and any(save_dir.glob("*.pt")):
        raise FileExistsError(f"Existing checkpoints in {save_dir}; use --resume or a fresh --save_dir")
    print(f"device={device}; variant={args.variant}; effective batch={training['batch_size']}; micro={training['micro_batch_size']}", flush=True)
    trainer = CVRPTrainer(model_params(args.variant, profile), training, validation, device, save_dir,
                         args.resume, args.warm_start, args.initialize_from, args.teacher_checkpoint)
    trainer.run()


if __name__ == "__main__":
    main()
