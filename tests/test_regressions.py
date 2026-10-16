"""Regressions for publication-critical failures, using small CPU models."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import torch
from src.checkpoint import load_model, save_checkpoint
from src import VARIANTS, model_params, setup_runtime, load_profile
from src.CVRPEnv import CVRPEnv
from src.CVRPModel import CVRPModel, AnchorGuidedContext
from src.CVRProblemDef import get_random_problems
from src.CVRPTrainer import CVRPTrainer, teacher_guided_loss
from src.data import dataset_id, load_dataset, load_reference, validate_data
from src.evaluation import evaluate_dataset, validate_tour
from run_cvrplib import choose_candidate, integer_distances, intra_2opt


def small_params(variant="full"):
    p = model_params(variant)
    p.update(embedding_dim=16, sqrt_embedding_dim=4.0, encoder_layer_num=1,
             head_num=2, qkv_dim=8, ff_hidden_dim=32, hacd_branch_hidden_dim=8)
    return p


def data(seed=42, count=3, n=20):
    tensors = get_random_problems(count, n, torch.Generator().manual_seed(seed))
    return dict(zip(("depot_xy", "node_xy", "node_demand"), tensors))


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        setup_runtime("cpu")

    def test_named_variants_override_profile_structural_switches(self):
        profile = load_profile()
        for flag in ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled",
                     "hacd_without_capacity_feature", "hacd_without_depot_residual", "hacd_without_fusion_gate"):
            profile["model"][flag] = True
        profile["model"]["hacd_gate_enabled"] = False
        for variant in VARIANTS:
            actual, expected = model_params(variant, profile), model_params(variant)
            for flag in ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled",
                         "hacd_without_capacity_feature", "hacd_without_depot_residual", "hacd_without_fusion_gate"):
                self.assertEqual(actual[flag], expected[flag], (variant, flag))

    def test_builtin_defaults_are_independent_and_need_no_config_files(self):
        with patch("builtins.open", side_effect=AssertionError("Unexpected config file access")):
            first = load_profile()
            first["model"]["embedding_dim"] = 7
            first["training"]["100"]["milestones"].append(1)
            second = load_profile()
            self.assertEqual(second["model"]["embedding_dim"], 128)
            self.assertEqual(second["training"]["100"]["milestones"], [8000, 8050])
            model = CVRPModel(**model_params("full"))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 1968798)

    def test_checkpoint_requires_explicit_head_configuration(self):
        p = model_params("pomo")
        p.update(head_num=4, qkv_dim=32)
        model = CVRPModel(**p)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, {"model_state_dict": model.state_dict()})
            with self.assertRaisesRegex(ValueError, "model_params"):
                load_model(path, torch.device("cpu"))
            incomplete = dict(p)
            incomplete.pop("head_num")
            save_checkpoint(path, {"model_state_dict": model.state_dict(), "model_params": incomplete})
            with self.assertRaisesRegex(ValueError, "head_num"):
                load_model(path, torch.device("cpu"))
            save_checkpoint(path, {"model_state_dict": model.state_dict(), "model_params": p})
            loaded, _, _ = load_model(path, torch.device("cpu"))
            self.assertEqual(loaded.model_params["head_num"], 4)

    def test_capacity_tolerance_cannot_create_nonfinite_probabilities(self):
        model = CVRPModel(**small_params()).eval()
        first = torch.tensor(1.0 - 2**-21)
        second = 1.0 - first + torch.tensor(1e-6)
        d = {"depot_xy": torch.zeros(1, 1, 2), "node_xy": torch.rand(1, 3, 2),
             "node_demand": torch.stack((first, second, torch.tensor(.1)))[None]}
        env = CVRPEnv(problem_size=3, pomo_size=1)
        env.use_problems(*d.values()); env.load_problems(1)
        reset, _, _ = env.reset(); model.pre_forward(reset)
        state, _, _ = env.pre_step()
        for node in (0, 1, 2):
            self.assertTrue(torch.isfinite(state.ninf_mask[0, 0, node]))
            state, _, _ = env.step(torch.tensor([[node]]))
        selected, _ = model(state)
        self.assertEqual(selected.item(), 0)
        last = model.encoded_nodes.gather(1, state.current_node[:, :, None].expand(-1, -1, 16))
        probability = model.decoder(last, state.current_node, state.load, state.ninf_mask, state.visited_ninf_flag)
        self.assertTrue(torch.isfinite(probability).all())
        probability[0, 0, 0].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_transfer_initialization_retains_source_and_target_configurations(self):
        p = small_params()
        original = CVRPModel(**p)
        original_params = dict(p, hacd_anchor_method="legacy_soft", hacd_local_k=8)
        training = dict(problem_size=20, seed=1234, epochs=1, episodes_per_epoch=2,
                        batch_size=2, micro_batch_size=2, lr=1e-4, weight_decay=1e-6,
                        milestones=[1], validation_batch_size=2, validation_aug=1, save_interval=1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "source.pt"
            save_checkpoint(source, {"model_params": original_params, "model_state_dict": original.state_dict(), "epoch": 80})
            trainer = CVRPTrainer(p, training, data(count=2), torch.device("cpu"), folder / "run", initialize_from=source)
            self.assertEqual(trainer.model.model_params["hacd_anchor_method"], "kmeans")
            self.assertEqual(trainer.training["initialization_metadata"]["source_model_params"]["hacd_anchor_method"], "legacy_soft")
            for key, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, trainer.model.state_dict()[key]))

    def test_fixed_dataset_uses_contents_not_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "holdout.pt"
            save_checkpoint(path, data())
            loaded = load_dataset(path, 20)
            self.assertEqual(dataset_id(loaded), dataset_id(data()))
            with self.assertRaises(ValueError):
                load_dataset(path, 50)
            with self.assertRaises(FileNotFoundError):
                load_dataset(Path(folder) / "missing.pt")

    def test_dataset_validation_and_exhaustion(self):
        invalid = data()
        invalid["node_demand"][0, 0] = 1.01
        with self.assertRaises(ValueError):
            validate_data(invalid)
        env = CVRPEnv(problem_size=20, pomo_size=20)
        env.use_problems(*data().values())
        env.load_problems(2)
        with self.assertRaises(ValueError):
            env.load_problems(2)

    def test_neutral_checkpoint_flags_and_strict_loading(self):
        p = model_params("param-matched")
        model = CVRPModel(**p)
        # Historical metadata omitted subflags; loader must not enable HACD.
        for flag in ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled"):
            p.pop(flag)
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = Path(folder) / "model.pt"
            save_checkpoint(path, {"model_params": p, "model_state_dict": model.state_dict()})
            loaded, _, metadata = load_model(path, torch.device("cpu"), "param-matched")
            self.assertEqual(metadata["parameters"], 1969504)
            self.assertFalse(loaded.model_params["hacd_hilo_enabled"])
            with self.assertRaises(ValueError):
                load_model(path, torch.device("cpu"), "full")
            state = dict(model.state_dict())
            state.pop(next(iter(state)))
            save_checkpoint(path, {"model_params": p, "model_state_dict": state})
            with self.assertRaises(RuntimeError):
                load_model(path, torch.device("cpu"))

    def test_inference_disables_grad_for_encoder_and_decoder(self):
        model = CVRPModel(**small_params())
        flags = []
        handle = model.decoder.register_forward_pre_hook(lambda *_: flags.append(torch.is_grad_enabled()))
        result = evaluate_dataset(model, data(count=2), torch.device("cpu"), 1, 8, warmup_batches=0)
        handle.remove()
        self.assertTrue(flags)
        self.assertFalse(any(flags))
        self.assertTrue(result["all_tours_validated"])
        self.assertEqual(len(result["costs"]), 2)

    def test_greedy_outputs_independent_of_batch_partition(self):
        model = CVRPModel(**small_params())
        first = evaluate_dataset(model, data(), torch.device("cpu"), 1, 8, warmup_batches=0)
        second = evaluate_dataset(model, data(), torch.device("cpu"), 3, 8, warmup_batches=0)
        np.testing.assert_allclose(first["costs"], second["costs"], atol=2e-5, rtol=2e-6)

    def test_route_validator_rejects_duplicate_missing_and_capacity(self):
        demand = [0, 0.6, 0.6]
        for tour in ([0, 1, 1, 0], [0, 1, 0], [0, 1, 2, 0], [1, 0, 2, 0]):
            with self.assertRaises(ValueError):
                validate_tour(tour, demand)
        validate_tour([0, 1, 0, 2, 0, 0], demand, max_vehicles=2)

    def test_no_fleet_relaxation(self):
        tours = [[0, 1, 0, 2, 0]]
        edges = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        rejected = choose_candidate(tours, [0, 1, 1], 1, edges, max_vehicles=1)
        self.assertEqual(rejected["status"], "no_feasible_candidate")
        self.assertIsNone(rejected["cost"])
        self.assertEqual(rejected["minimum_candidate_vehicles"], 2)
        accepted = choose_candidate(tours, [0, 1, 1], 1, edges, max_vehicles=2)
        self.assertEqual(accepted["cost"], 4)

    def test_integer_rounding_and_2opt_use_same_objective(self):
        edges = integer_distances([[0, 0], [2.5, 0]])
        self.assertEqual(edges[0, 1], 3)
        edges = integer_distances([[0, 0], [10, 10], [0, 10], [10, 0]])
        route = [0, 1, 2, 3, 0]
        improved = intra_2opt(edges, route)
        def cost(r):
            return sum(edges[a, b] for a, b in zip(r, r[1:]))
        self.assertLess(cost(improved), cost(route))
        self.assertEqual(sorted(improved), sorted(route))

    def test_kmeans_iterations_change_tokens(self):
        g = torch.Generator().manual_seed(18)
        xy = torch.rand(4, 20, 2, generator=g)
        embeddings = torch.rand(4, 20, 16, generator=g)
        radial = xy.square().sum(-1).sqrt()
        p = small_params()
        p["hacd_anchor_kmeans_iters"] = 1
        one = AnchorGuidedContext(**p)._cluster_anchor_tokens(embeddings, xy, radial, 4)
        p["hacd_anchor_kmeans_iters"] = 3
        three = AnchorGuidedContext(**p)._cluster_anchor_tokens(embeddings, xy, radial, 4)
        self.assertGreater((one - three).abs().max().item(), 0.001)

    def test_empty_cluster_repair_has_finite_tokens_and_gradients(self):
        embeddings = torch.rand(2, 20, 16, requires_grad=True)
        xy = torch.zeros(2, 20, 2)
        tokens = AnchorGuidedContext(**small_params())._cluster_anchor_tokens(embeddings, xy, torch.zeros(2, 20), 8)
        self.assertTrue(torch.isfinite(tokens).all())
        tokens.sum().backward()
        self.assertTrue((embeddings.grad > 0).all())

    def test_reference_bound_to_exact_dataset(self):
        d = data()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ref.json"
            ref = {"dataset_id": dataset_id(d), "source": "test oracle", "costs": [1, 2, 3]}
            path.write_text(json.dumps(ref), encoding="utf-8")
            costs, _ = load_reference(path, d)
            self.assertEqual(costs.tolist(), [1, 2, 3])
            with self.assertRaises(ValueError):
                load_reference(path, data(seed=41))

    def test_exact_resume_restores_scheduler_rng_optimizer(self):
        training = dict(problem_size=20, seed=1234, epochs=2, episodes_per_epoch=4,
                        batch_size=2, micro_batch_size=1, lr=1e-4, weight_decay=1e-6,
                        milestones=[1], validation_batch_size=2, validation_aug=1, save_interval=1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            setup_runtime("cpu", 1234)
            uninterrupted = CVRPTrainer(small_params(), training, data(count=2), torch.device("cpu"), folder / "continuous")
            uninterrupted.run()
            setup_runtime("cpu", 1234)
            first = CVRPTrainer(small_params(), dict(training, epochs=1), data(count=2), torch.device("cpu"), folder / "split")
            first.run()
            setup_runtime("cpu", 9999)  # checkpoint must restore, not use this RNG
            resumed = CVRPTrainer(small_params(), training, data(count=2), torch.device("cpu"), folder / "split", folder / "split/latest.pt")
            resumed.run()
            for key, tensor in uninterrupted.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[key]), key)
            self.assertEqual(uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict())
            self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], 1e-5)
            a, b = uninterrupted.optimizer.state_dict()["state"], resumed.optimizer.state_dict()["state"]
            for k in a:
                for field in a[k]:
                    if isinstance(a[k][field], torch.Tensor):
                        self.assertTrue(torch.equal(a[k][field], b[k][field]))

    def test_all_variants_roll_out_feasible_tours(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = evaluate_dataset(CVRPModel(**small_params(variant)), data(count=1), torch.device("cpu"), 1, 1, warmup_batches=0)
                self.assertTrue(result["all_tours_validated"])


    def test_decoder_checkpoint_preserves_rollouts_and_gradients(self):
        training = dict(problem_size=20, seed=1234, epochs=1, episodes_per_epoch=2,
                        batch_size=2, micro_batch_size=2, lr=1e-4, weight_decay=1e-6,
                        milestones=[800], validation_batch_size=2, validation_aug=1, save_interval=1)
        with tempfile.TemporaryDirectory() as folder:
            setup_runtime("cpu", 1234)
            plain = CVRPTrainer(small_params(), training, data(count=2), torch.device("cpu"), Path(folder)/"plain")
            setup_runtime("cpu", 1234)
            recomputed = CVRPTrainer(small_params(), dict(training, decoder_checkpoint=True), data(count=2),
                                    torch.device("cpu"), Path(folder)/"recomputed")
            setup_runtime("cpu", 987)
            loss_a, score_a = plain._loss(2)
            loss_a.backward()
            setup_runtime("cpu", 987)
            loss_b, score_b = recomputed._loss(2)
            loss_b.backward()
            self.assertTrue(torch.equal(plain.env.selected_node_list, recomputed.env.selected_node_list))
            self.assertTrue(torch.equal(score_a, score_b))
            self.assertTrue(torch.equal(loss_a, loss_b))
            for (name_a, p_a), (name_b, p_b) in zip(plain.model.named_parameters(), recomputed.model.named_parameters()):
                self.assertEqual(name_a, name_b)
                if p_a.grad is not None:
                    torch.testing.assert_close(p_a.grad, p_b.grad, rtol=1e-5, atol=1e-7, msg=name_a)

    def test_teacher_objective_zero_kl_for_identical_policies_and_rejects_holdouts(self):
        student = CVRPModel(**small_params())
        teacher = copy.deepcopy(student).requires_grad_(False)
        batch = data(count=2)
        loss, metrics = teacher_guided_loss(student, teacher, batch, torch.device("cpu"), starts=3)
        self.assertAlmostEqual(metrics["kl"], 0.0, places=6)
        self.assertGreater(metrics["nll"], 0)
        loss.backward()
        for prefix in ("encoder.layers", "encoder.anchor_context", "decoder.capacity_risk"):
            grads = [p.grad for name,p in student.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, prefix)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "validation or test"):
                teacher_guided_loss(student, teacher, dict(batch, metadata={"split": split}), torch.device("cpu"))

    def test_validation_screen_does_not_replace_full_set_best(self):
        training = dict(problem_size=20, seed=1234, epochs=2, episodes_per_epoch=2,
                        batch_size=2, micro_batch_size=2, lr=1e-4, weight_decay=1e-6,
                        milestones=[800], validation_batch_size=2, validation_aug=1, save_interval=1,
                        validation_interval=2, validation_screen_size=1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "source.pt"
            p = small_params()
            p["ff_hidden_dim"] = 512
            save_checkpoint(source, {"model_params": p, "model_state_dict": CVRPModel(**p).state_dict()})
            trainer = CVRPTrainer(p, training, data(count=2), torch.device("cpu"), folder / "run", initialize_from=source)
            # A very low screen cost must not displace the better full-set candidate.
            with patch.object(importlib.import_module("src.CVRPTrainer"), "evaluate_dataset", side_effect=[
                    {"mean_cost": 20., "costs": [20., 20.]},
                    {"mean_cost": 1., "costs": [1.]},
                    {"mean_cost": 30., "costs": [30., 30.]}]):
                trainer.run()
            _, saved, _ = load_model(folder / "run/best.pt", torch.device("cpu"), "full")
            self.assertEqual(saved["epoch"], 0)
            self.assertEqual(trainer.best_val_cost, 20.)
            self.assertIsNone(trainer.history[0]["validation_cost"])
            self.assertEqual(trainer.history[0]["screen_validation_cost"], 1.)

    def test_teacher_training_checkpoint_preserves_full_student_and_source_identity(self):
        training = dict(problem_size=20, seed=1234, epochs=1, episodes_per_epoch=2,
                        batch_size=2, micro_batch_size=2, lr=1e-4, weight_decay=1e-6,
                        milestones=[800], validation_batch_size=2, validation_aug=1, save_interval=1,
                        teacher_starts=2, teacher_hard_weight=.2, max_grad_norm=1.0,
                        decoder_checkpoint=True)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            p = small_params("pomo")
            p["ff_hidden_dim"] = 512
            student_params = small_params()
            student_params["ff_hidden_dim"] = 512
            teacher_path = folder / "pomo.pt"
            save_checkpoint(teacher_path, {"model_params": p, "model_state_dict": CVRPModel(**p).state_dict()})
            trainer = CVRPTrainer(student_params, training, data(count=2), torch.device("cpu"), folder / "student",
                                  teacher_checkpoint=teacher_path)
            teacher_state = copy.deepcopy(trainer.teacher.state_dict())
            trainer.run()
            loaded, saved, metadata = load_model(folder / "student/latest.pt", torch.device("cpu"), "full")
            self.assertTrue(all(loaded.model_params[k] for k in ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled")))
            self.assertEqual(metadata["variant"], "full")
            self.assertIn("sha256", saved["training"]["teacher_metadata"])
            self.assertIn("teacher_cost", saved["history"][0])
            self.assertNotIn("train_cost", saved["history"][0])
            for key, value in teacher_state.items():
                self.assertTrue(torch.equal(value, trainer.teacher.state_dict()[key]))
            self.assertTrue(all(p.grad is None for p in trainer.teacher.parameters()))
            resumed = CVRPTrainer(student_params, dict(training, epochs=2), data(count=2), torch.device("cpu"),
                                  folder / "student", resume=folder / "student/latest.pt", teacher_checkpoint=teacher_path)
            resumed.run()
            self.assertEqual(len(resumed.history), 2)
            altered = CVRPModel(**p).state_dict()
            save_checkpoint(teacher_path, {"model_params": p, "model_state_dict": altered})
            with self.assertRaisesRegex(ValueError, "identical training configuration"):
                CVRPTrainer(student_params, dict(training, epochs=3), data(count=2), torch.device("cpu"),
                            folder / "student", resume=folder / "student/latest.pt", teacher_checkpoint=teacher_path)

if __name__ == "__main__":
    unittest.main()
