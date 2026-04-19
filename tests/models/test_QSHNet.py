import unittest

import torch
import torch.nn as nn

from models.QSHNet import HypergraphLearner, Model, SpikeRouter
from utils.configs import get_configs


class TestQSHNet(unittest.TestCase):

    def test_spike_router_starts_as_identity_with_moderate_event_gate(self):
        router = SpikeRouter(d_model=8)
        obs = torch.randn(2, 3, 8)
        mask_d = torch.ones_like(obs)
        variable_incidence_matrix = torch.tensor([
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ])
        variable_indices_flattened = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
        ])

        obs_base, obs_event, route_state = router(
            obs, mask_d, variable_incidence_matrix, variable_indices_flattened
        )

        self.assertEqual(route_state["retain_gate"].shape, torch.Size((2, 3)))
        self.assertEqual(route_state["event_gate"].shape, torch.Size((2, 3)))
        self.assertEqual(route_state["route_logit"].shape, torch.Size((2, 3)))
        self.assertTrue(torch.allclose(obs_base, obs, atol=1e-6))
        self.assertTrue(torch.allclose(obs_event, torch.zeros_like(obs), atol=1e-6))
        self.assertGreater(route_state["retain_gate"].min().item(), 0.99)
        self.assertGreater(route_state["event_gate"].mean().item(), 0.001)
        self.assertLess(route_state["event_gate"].mean().item(), 0.01)

    def test_spike_router_event_gate_is_decoupled_from_route_logit(self):
        router = SpikeRouter(d_model=8)
        obs = torch.randn(2, 3, 8)
        mask_d = torch.ones_like(obs)
        variable_incidence_matrix = torch.tensor([
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ])
        variable_indices_flattened = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
        ])
        with torch.no_grad():
            router.membrane_proj.weight.fill_(1.0)
            router.membrane_proj.bias.fill_(5.0)

        _, _, route_state = router(
            obs, mask_d, variable_incidence_matrix, variable_indices_flattened
        )

        expected_event_gate = torch.full_like(
            route_state["event_gate"],
            torch.exp(router.event_log_scale).item() * 0.5,
        )
        self.assertFalse(torch.allclose(route_state["route_logit"], torch.zeros_like(route_state["route_logit"])))
        self.assertTrue(torch.allclose(route_state["event_gate"], expected_event_gate, atol=1e-6))

    def test_spike_router_emits_selection_weight_from_route_logit(self):
        router = SpikeRouter(d_model=8)
        obs = torch.randn(2, 3, 8)
        mask_d = torch.ones_like(obs)
        variable_incidence_matrix = torch.tensor([
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ])
        variable_indices_flattened = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
        ])
        with torch.no_grad():
            router.membrane_proj.weight.fill_(0.5)
            router.membrane_proj.bias.fill_(0.25)

        _, _, route_state = router(
            obs, mask_d, variable_incidence_matrix, variable_indices_flattened
        )

        expected_selection_weight = torch.sigmoid(route_state["route_logit"])
        self.assertIn("selection_weight", route_state)
        self.assertEqual(route_state["selection_weight"].shape, torch.Size((2, 3)))
        self.assertTrue(torch.allclose(route_state["selection_weight"], expected_selection_weight, atol=1e-6))
        self.assertGreaterEqual(route_state["selection_weight"].min().item(), 0.0)
        self.assertLessEqual(route_state["selection_weight"].max().item(), 1.0)

    def test_hypergraph_learner_uses_nodewise_conditioned_quaternion_gate(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        linear_out = torch.randn(2, 3, 8)
        event_gate = torch.zeros(2, 3)

        quat_gate = learner.compute_quaternion_gate(0, linear_out, event_gate)

        self.assertEqual(quat_gate.shape, torch.Size((2, 3, 1)))
        self.assertTrue(torch.all(quat_gate >= 0.0))
        self.assertTrue(torch.all(quat_gate <= 1.0))
        self.assertLess(quat_gate.mean().item(), 0.1)

    def test_quaternion_residual_is_bounded_relative_to_linear_path(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        linear_out = torch.ones(2, 3, 8)
        quat_out = torch.full((2, 3, 8), 20.0)
        alpha = torch.ones(2, 3, 1)

        bounded_residual = learner.bound_quaternion_residual(
            linear_out=linear_out,
            quat_out=quat_out,
            alpha=alpha,
        )

        residual_norm = bounded_residual.norm(dim=-1)
        linear_norm = linear_out.norm(dim=-1)

        self.assertTrue(torch.all(residual_norm <= learner.quat_residual_ratio_max * linear_norm + 1e-6))

    def test_event_delta_normalization_centers_each_hyperedge_feature_vector(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        event_delta = torch.randn(2, 3, 8) * 5.0 + 7.0

        normalized_delta = learner.normalize_event_delta(0, event_delta, target="temporal")

        self.assertEqual(normalized_delta.shape, torch.Size((2, 3, 8)))
        self.assertTrue(
            torch.allclose(
                normalized_delta.mean(dim=-1),
                torch.zeros(2, 3),
                atol=1e-5,
            )
        )

    def test_event_injection_adds_bounded_delta_to_both_hyperedge_paths(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        event_delta = torch.full((2, 3, 8), 0.5)
        main_state = torch.randn(2, 3, 8)
        event_scale = torch.tensor(0.1)

        temporal_injected, temporal_diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="temporal",
            return_diagnostics=True,
        )
        variable_injected, variable_diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="variable",
            return_diagnostics=True,
        )

        expected = main_state + 0.05
        self.assertTrue(torch.allclose(temporal_injected, expected, atol=1e-6))
        self.assertTrue(torch.allclose(variable_injected, expected, atol=1e-6))
        self.assertEqual(temporal_diag["clip_rate"], 0.0)
        self.assertEqual(variable_diag["clip_rate"], 0.0)

    def test_coupled_residual_ratio_max_initializes_to_fused_context_default(self):
        learner = HypergraphLearner(n_layers=2, d_model=8, n_heads=1, time_length=4)

        self.assertAlmostEqual(learner.coupled_residual_ratio_max, 0.20)

    def test_coupled_residual_bound_keeps_small_event_injection_unchanged(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        main_state = torch.ones(2, 3, 8)
        event_delta = torch.full((2, 3, 8), 0.1)
        event_scale = torch.tensor(0.1)

        bounded_state, diag = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="temporal",
            return_diagnostics=True,
        )

        expected = main_state + 0.01
        self.assertTrue(torch.allclose(bounded_state, expected, atol=1e-6))
        self.assertEqual(diag["clip_rate"], 0.0)

    def test_coupled_residual_bound_scales_down_large_fused_context_residual(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        main_state = torch.ones(2, 3, 16)
        coupled_residual = torch.full((2, 3, 16), 10.0)

        bounded_residual, diag = learner.bound_coupled_residual(
            main_state=main_state,
            coupled_residual=coupled_residual,
        )

        residual_norm = bounded_residual.norm(dim=-1)
        main_norm = main_state.norm(dim=-1)

        self.assertTrue(torch.all(residual_norm <= 0.20 * main_norm + 1e-6))
        self.assertGreater(diag["clip_rate"], 0.0)

    def test_adaptive_fused_cap_tightens_under_high_density_and_high_residual(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        main_state = torch.ones(2, 3, 16)
        coupled_residual = torch.full((2, 3, 16), 0.3)
        low_density = torch.full((2, 3, 1), learner.event_density_baseline)
        high_density = torch.ones(2, 3, 1)

        low_cap = learner.compute_adaptive_coupled_ratio_max(
            main_state=main_state,
            coupled_residual=coupled_residual,
            fused_route_density=low_density,
        )
        high_cap = learner.compute_adaptive_coupled_ratio_max(
            main_state=main_state,
            coupled_residual=coupled_residual,
            fused_route_density=high_density,
        )

        self.assertTrue(torch.all(low_cap <= learner.coupled_residual_ratio_max + 1e-6))
        self.assertTrue(torch.all(high_cap < low_cap))

    def test_adaptive_fused_cap_preserves_base_limit_under_low_risk(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        main_state = torch.ones(2, 3, 16)
        coupled_residual = torch.full((2, 3, 16), 0.01)
        baseline_density = torch.full((2, 3, 1), learner.event_density_baseline)

        adaptive_cap = learner.compute_adaptive_coupled_ratio_max(
            main_state=main_state,
            coupled_residual=coupled_residual,
            fused_route_density=baseline_density,
        )

        self.assertTrue(
            torch.allclose(
                adaptive_cap,
                torch.full_like(adaptive_cap, learner.coupled_residual_ratio_max),
                atol=1e-6,
            )
        )

    def test_fused_context_stabilization_keeps_temporal_path_and_bounds_variable_path(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        temporal_base = torch.ones(2, 3, 8)
        variable_base = torch.ones(2, 3, 8)
        temporal_context = temporal_base + 5.0
        variable_context = variable_base + 10.0
        fused_density = torch.ones(2, 3, 1)

        stabilized_temporal, stabilized_variable, diag = learner.stabilize_fused_context(
            temporal_context_base=temporal_base,
            variable_context_base=variable_base,
            temporal_context=temporal_context,
            variable_context=variable_context,
            fused_route_density=fused_density,
        )

        variable_residual_norm = (stabilized_variable - variable_base).norm(dim=-1)
        variable_base_norm = variable_base.norm(dim=-1)

        self.assertTrue(torch.allclose(stabilized_temporal, temporal_context))
        self.assertTrue(torch.all(variable_residual_norm <= 0.20 * variable_base_norm + 1e-6))
        self.assertGreater(diag["clip_rate"], 0.0)

    def test_event_scale_is_capped_without_disturbing_small_initial_value(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)

        initial_scale = learner.compute_event_scale(0).item()
        with torch.no_grad():
            learner.event_residual_scale[0].fill_(10.0)
        capped_scale = learner.compute_event_scale(0).item()

        self.assertGreater(initial_scale, 0.05)
        self.assertLess(initial_scale, 0.12)
        self.assertLessEqual(capped_scale, 0.12 + 1e-6)

    def test_event_scale_density_modulation_keeps_base_scale_at_baseline_density(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        base_scale = learner.compute_event_scale(0)
        route_density = torch.full((2, 3, 1), learner.event_density_baseline)

        modulated_scale = learner.modulate_event_scale(base_scale, route_density, target="variable")

        expected_scale = torch.ones_like(route_density) * base_scale
        self.assertTrue(torch.allclose(modulated_scale, expected_scale))

    def test_event_scale_density_modulation_reduces_scale_for_dense_routes_on_variable_path(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        base_scale = torch.tensor(0.1)
        route_density = torch.ones(2, 3, 1)

        modulated_scale = learner.modulate_event_scale(base_scale, route_density, target="variable")

        expected_scale = base_scale * (1.0 - learner.variable_event_density_penalty_max)
        self.assertTrue(torch.allclose(modulated_scale, torch.full_like(route_density, expected_scale)))

    def test_fused_route_density_uses_max_of_temporal_and_variable_paths(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        temporal_density = torch.full((2, 3, 1), 0.95)
        variable_density = torch.full((2, 3, 1), 0.55)

        fused_density = learner.summarize_fused_route_density(
            temporal_density,
            variable_density,
        )

        self.assertTrue(torch.allclose(fused_density, temporal_density))

    def test_event_scale_density_modulation_keeps_temporal_path_unchanged_even_for_dense_routes(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        base_scale = torch.tensor(0.1)
        route_density = torch.ones(2, 3, 1)

        modulated_scale = learner.modulate_event_scale(base_scale, route_density, target="temporal")

        self.assertTrue(torch.allclose(modulated_scale, torch.full_like(route_density, base_scale)))

    def test_spike_router_caps_retain_gate_drop_under_large_scale(self):
        router = SpikeRouter(d_model=8)
        obs = torch.randn(2, 3, 8)
        mask_d = torch.ones_like(obs)
        variable_incidence_matrix = torch.tensor([
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ])
        variable_indices_flattened = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
        ])

        with torch.no_grad():
            router.membrane_proj.weight.zero_()
            router.membrane_proj.bias.fill_(-10.0)
            router.retain_log_scale.fill_(5.0)

        _, _, route_state = router(
            obs, mask_d, variable_incidence_matrix, variable_indices_flattened
        )

        self.assertGreaterEqual(route_state["retain_gate"].min().item(), 0.9)
        self.assertLessEqual(route_state["retain_gate"].max().item(), 1.0)

    def test_route_diagnostics_include_selection_statistics(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        route_state = {
            "retain_gate": torch.tensor([[1.0, 0.95]]),
            "event_gate": torch.tensor([[0.01, 0.01]]),
            "route_logit": torch.tensor([[0.0, 1.0]]),
            "selection_weight": torch.tensor([[0.5, 0.75]]),
        }
        temporal_route_density = torch.tensor([[[0.5], [0.6]]])
        variable_route_density = torch.tensor([[[0.4], [0.7]]])

        route_diag = learner.summarize_route_diagnostics(
            route_state,
            temporal_route_density,
            variable_route_density,
        )

        self.assertIn("selection_mean", route_diag)
        self.assertIn("selection_std", route_diag)
        self.assertAlmostEqual(route_diag["selection_mean"], 0.625)
        self.assertGreater(route_diag["selection_std"], 0.0)

    def test_propagation_selection_factor_preserves_identity_at_zero_route(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        route_state = {
            "selection_weight": torch.full((2, 3), 0.5),
        }

        selection_factor = learner.compute_propagation_selection_factor(route_state)

        self.assertTrue(torch.allclose(selection_factor, torch.ones(2, 3, 1)))

    def test_hypergraph_learner_uses_residual_selection_for_node_to_hyperedge_messages(self):
        class FixedSpikeRouter(nn.Module):
            def __init__(self, obs_base, selection_weight):
                super().__init__()
                self.obs_base = obs_base
                self.selection_weight = selection_weight

            def forward(self, observation_nodes, mask_d, variable_incidence_matrix, variable_indices_flattened):
                route_logit = torch.logit(self.selection_weight.clamp(1e-6, 1 - 1e-6))
                return self.obs_base, torch.zeros_like(self.obs_base), {
                    "retain_gate": torch.ones_like(self.selection_weight),
                    "event_gate": torch.full_like(self.selection_weight, 0.005),
                    "route_logit": route_logit,
                    "selection_weight": self.selection_weight,
                }

        class CaptureAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_k = None

            def forward(self, q, k, mask=None):
                self.last_k = k.detach().clone()
                return torch.zeros_like(q)

        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        learner.hyperedge2hyperedge_layers = []

        obs_base = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
        selection_weight = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)
        learner.spike_select[0] = FixedSpikeRouter(obs_base, selection_weight)
        learner.node2temporal_hyperedge[0] = CaptureAttention()
        learner.node2variable_hyperedge[0] = CaptureAttention()

        observation_nodes = torch.randn(1, 3, 8)
        temporal_hyperedges = torch.randn(1, 2, 8)
        variable_hyperedges = torch.randn(1, 2, 8)
        time_indices_flattened = torch.tensor([[0, 1, 0]])
        variable_indices_flattened = torch.tensor([[0, 1, 0]])
        temporal_incidence_matrix = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        variable_incidence_matrix = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        x_y_mask_flattened = torch.ones(1, 3)
        x_y_mask = torch.ones(1, 2, 2)
        y_mask_L_flattened = torch.zeros(1, 3)

        learner(
            observation_nodes=observation_nodes,
            temporal_hyperedges=temporal_hyperedges,
            variable_hyperedges=variable_hyperedges,
            time_indices_flattened=time_indices_flattened,
            variable_indices_flattened=variable_indices_flattened,
            temporal_incidence_matrix=temporal_incidence_matrix,
            variable_incidence_matrix=variable_incidence_matrix,
            x_y_mask_flattened=x_y_mask_flattened,
            x_y_mask=x_y_mask,
            y_mask_L_flattened=y_mask_L_flattened,
        )

        expected_factor = 1.0 + learner.propagation_selection_strength * (
            selection_weight.unsqueeze(-1) - 0.5
        )
        expected_selected = obs_base * expected_factor

        self.assertTrue(
            torch.allclose(
                learner.node2temporal_hyperedge[0].last_k[..., -8:],
                expected_selected,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                learner.node2variable_hyperedge[0].last_k[..., -8:],
                expected_selected,
                atol=1e-6,
            )
        )

    def test_hypergraph_learner_initializes_event_residual_with_small_nonzero_scale(self):
        learner = HypergraphLearner(n_layers=2, d_model=8, n_heads=1, time_length=4)

        event_scales = [
            torch.sigmoid(scale.detach()).item()
            for scale in learner.event_residual_scale
        ]

        for event_scale in event_scales:
            self.assertGreater(event_scale, 0.05)
            self.assertLess(event_scale, 0.15)

    def test_model_forward_runs_with_default_configs(self):
        configs = get_configs(args=["--model_name", "QSHNet", "--model_id", "QSHNet"])
        model = Model(configs)

        x = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in)
        result_dict = model(**{"x": x, "exp_stage": "test"})

        self.assertEqual(result_dict["pred"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
        self.assertEqual(result_dict["true"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))

    def test_model_forward_records_route_and_quaternion_diagnostics(self):
        configs = get_configs(args=["--model_name", "QSHNet", "--model_id", "QSHNet"])
        model = Model(configs)

        x = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in)
        model(**{"x": x, "exp_stage": "test"})

        diagnostics = model.hypergraph_learner.latest_event_diagnostics[0]
        self.assertIn("route", diagnostics)
        self.assertIn("quaternion", diagnostics)
        self.assertIn("retain_mean", diagnostics["route"])
        self.assertIn("selection_mean", diagnostics["route"])
        self.assertIn("alpha_mean", diagnostics["quaternion"])
