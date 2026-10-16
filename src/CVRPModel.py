import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = _with_hacd_defaults(model_params)

        self.encoder = CVRP_Encoder(**self.model_params)
        self.decoder = CVRP_Decoder(**self.model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes, routing_anchors = self.encoder(depot_xy, node_xy_demand)
        # shape: (batch, problem+1, embedding), (batch, anchor, embedding) or None

        depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(node_demand.size(0), 1), device=node_demand.device)
        depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        self.decoder.set_kv(self.encoded_nodes)
        self.decoder.set_static_context(depot_node_xy, depot_node_demand, routing_anchors)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            device = state.BATCH_IDX.device
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, pomo_size), device=device)

        elif state.selected_count == 1:  # Second Move, POMO
            device = state.BATCH_IDX.device
            selected = torch.arange(start=1, end=pomo_size+1, device=device)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=device)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            inputs = (encoded_last_node, state.current_node, state.load,
                      state.ninf_mask, state.visited_ninf_flag)
            if self.training and getattr(self, 'decoder_checkpointing', False):
                # These buffers are mutated by subsequent environment steps.
                inputs = (encoded_last_node, state.current_node, state.load.clone(),
                          state.ninf_mask, state.visited_ninf_flag.clone())
                probs = checkpoint(self.decoder, *inputs, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                probs = self.decoder(*inputs)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _with_hacd_defaults(model_params):
    params = dict(model_params)
    # Preserve the behaviour of unmodified POMO configurations. HACD is opt-in
    # so official checkpoints can still be loaded without extra parameters.
    params.setdefault('hacd_enabled', False)
    params.setdefault('hacd_hilo_enabled', params['hacd_enabled'])
    params.setdefault('hacd_anchor_enabled', params['hacd_enabled'])
    params.setdefault('hacd_capacity_residual_enabled', params['hacd_enabled'])
    params.setdefault('hacd_local_k', 16)
    params.setdefault('hacd_anchor_demand_ratio', 0.20)
    params.setdefault('hacd_anchor_peripheral_ratio', 0.20)
    params.setdefault('hacd_cluster_anchor_num', None)
    params.setdefault('hacd_anchor_kmeans_iters', 3)
    params.setdefault('hacd_anchor_method', 'kmeans')
    params.setdefault('hacd_anchor_temperature', 0.10)
    params.setdefault('hacd_residual_beta', 2.0)
    params.setdefault('hacd_depot_beta', 2.0)
    params.setdefault('hacd_branch_hidden_dim', max(32, params['embedding_dim'] // 2))
    params.setdefault('hacd_fusion_init_bias', -2.0)
    params.setdefault('hacd_without_capacity_feature', False)
    params.setdefault('hacd_without_depot_residual', False)
    params.setdefault('hacd_without_fusion_gate', False)
    # Map inverse-logic flags if provided by ablation callers
    if not params.get('hacd_depot_residual_enabled', True):
        params['hacd_without_depot_residual'] = True
    if not params.get('hacd_gate_enabled', True):
        params['hacd_without_fusion_gate'] = True
    if not params.get('hacd_capacity_feature_enabled', True):
        params['hacd_without_capacity_feature'] = True
    params.setdefault('hacd_debug_enabled', False)
    if params['hacd_anchor_method'] not in ('kmeans', 'kmeans_soft_pool', 'legacy_soft'):
        raise ValueError('Unknown anchor method')
    if params['hacd_anchor_kmeans_iters'] < 1:
        raise ValueError('K-means iterations must be positive')
    return params


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem+1, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        if self.model_params['hacd_hilo_enabled']:
            self.embedding_route = nn.Linear(5, embedding_dim)
            self.layers = nn.ModuleList([HiLoEncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        else:
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(3, embedding_dim)
            self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

        if self.model_params['hacd_anchor_enabled']:
            self.anchor_context = AnchorGuidedContext(**model_params)
        else:
            self.anchor_context = None

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        node_xy = node_xy_demand[:, :, :2]
        node_demand = node_xy_demand[:, :, 2]
        depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_node_demand = torch.cat((
            torch.zeros(size=(node_demand.size(0), 1), device=node_demand.device),
            node_demand,
        ), dim=1)
        depot_distance = (depot_node_xy - depot_xy).pow(2).sum(dim=2).sqrt()

        if self.model_params['hacd_hilo_enabled']:
            depot_indicator = torch.zeros_like(depot_node_demand)
            depot_indicator[:, 0] = 1
            route_features = torch.stack((
                depot_node_xy[:, :, 0],
                depot_node_xy[:, :, 1],
                depot_indicator,
                depot_node_demand,
                depot_distance,
            ), dim=2)
            out = self.embedding_route(route_features)
            local_k = min(
                self.model_params['hacd_local_k'], max(1, depot_node_xy.size(1) - 1)
            )
            local_distance_matrix = torch.cdist(depot_node_xy, depot_node_xy, p=2)
            diagonal_mask = torch.eye(
                depot_node_xy.size(1), device=depot_node_xy.device, dtype=torch.bool
            )[None, :, :]
            local_distance_matrix = local_distance_matrix.masked_fill(diagonal_mask, float('inf'))
            # Stable index-order ties for deterministic geometric neighbourhoods.
            neighbor_index = torch.argsort(local_distance_matrix, dim=2, stable=True)[:, :, :local_k]
            local_distance = local_distance_matrix.gather(2, neighbor_index)
            for layer in self.layers:
                out = layer(
                    out,
                    depot_node_demand,
                    depot_distance,
                    neighbor_index,
                    local_distance,
                )
        else:
            embedded_depot = self.embedding_depot(depot_xy)
            embedded_node = self.embedding_node(node_xy_demand)
            out = torch.cat((embedded_depot, embedded_node), dim=1)
            for layer in self.layers:
                out = layer(out)

        routing_anchors = None
        if self.anchor_context is not None:
            out, routing_anchors = self.anchor_context(out, node_xy, depot_node_demand, depot_distance)

        return out, routing_anchors
        # shape: (batch, problem+1, embedding), (batch, anchor, embedding) or None


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem+1, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem+1, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem+1, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem+1, embedding)


class HiLoEncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_high = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_high = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_high = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.local_bias = nn.Linear(3, head_num, bias=False)
        self.high_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.global_pool = nn.Linear(2 * embedding_dim + 4, embedding_dim)
        self.low_project = nn.Linear(2 * embedding_dim + 2, embedding_dim)
        self.cross_gate = nn.Linear(2 * embedding_dim + 2, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(
        self,
        input1,
        depot_node_demand,
        depot_distance,
        neighbor_index,
        local_distance,
    ):
        # input1.shape: (batch, problem+1, embedding)
        batch_size, node_count, _ = input1.size()

        high_out = self._local_high_path(
            input1, depot_node_demand, neighbor_index, local_distance
        )

        route_stats = torch.stack((depot_node_demand, depot_distance), dim=2)
        pooled = torch.cat((
            input1.mean(dim=1),
            input1.max(dim=1).values,
            route_stats.mean(dim=1),
            route_stats.max(dim=1).values,
        ), dim=1)
        global_context = F.gelu(self.global_pool(pooled))
        global_context = global_context[:, None, :].expand(batch_size, node_count, -1)
        low_input = torch.cat((input1, global_context, route_stats), dim=2)
        low_out = F.gelu(self.low_project(low_input))

        gate_input = torch.cat((high_out, low_out, route_stats), dim=2)
        gate = torch.sigmoid(self.cross_gate(gate_input))
        fused = gate * high_out + (1 - gate) * low_out

        out1 = self.add_n_normalization_1(input1, fused)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3

    def _local_high_path(self, input1, depot_node_demand, neighbor_index, local_distance):
        batch_size, node_count, _ = input1.size()
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        local_k = neighbor_index.size(2)
        # shape: (batch, problem+1, local_k)

        q = reshape_by_heads(self.Wq_high(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk_high(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv_high(input1), head_num=head_num)
        # shape: (batch, head, problem+1, qkv_dim)

        # Index the K nearest entries directly. The previous implementation
        # materialised two (B,H,N,N,D) views before gathering, which defeated
        # the memory advantage of local attention.
        batch_index = torch.arange(batch_size, device=input1.device)[:, None, None, None]
        head_index = torch.arange(head_num, device=input1.device)[None, :, None, None]
        local_index = neighbor_index[:, None, :, :]
        k_local = k[batch_index, head_index, local_index, :]
        v_local = v[batch_index, head_index, local_index, :]

        score = (q[:, :, :, None, :] * k_local).sum(dim=4) / math.sqrt(qkv_dim)
        demand_i = depot_node_demand[:, :, None].expand(batch_size, node_count, local_k)
        demand_j = depot_node_demand.gather(dim=1, index=neighbor_index.reshape(batch_size, -1)) \
            .reshape(batch_size, node_count, local_k)
        bias_feature = torch.stack((local_distance, demand_i, demand_j), dim=3)
        local_bias = self.local_bias(bias_feature).permute(0, 3, 1, 2)
        score = score + local_bias

        weights = F.softmax(score, dim=3)
        out = (weights[:, :, :, :, None] * v_local).sum(dim=3)
        out = out.transpose(1, 2).reshape(batch_size, node_count, head_num * qkv_dim)
        return self.high_combine(out)


class AnchorGuidedContext(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.Wq_anchor = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_node = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_node = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_node = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_anchor = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_anchor = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.add_n_normalization = AddAndInstanceNormalization(**model_params)

    def forward(self, encoded_nodes, node_xy, depot_node_demand, depot_distance):
        anchors = self._build_anchor_tokens(encoded_nodes, node_xy, depot_node_demand, depot_distance)

        q_anchor = self.Wq_anchor(anchors)
        k_node = self.Wk_node(encoded_nodes)
        v_node = self.Wv_node(encoded_nodes)
        anchor_score = torch.matmul(q_anchor, k_node.transpose(1, 2)) / math.sqrt(encoded_nodes.size(2))
        updated_anchors = torch.matmul(F.softmax(anchor_score, dim=2), v_node)

        q_node = self.Wq_node(encoded_nodes)
        k_anchor = self.Wk_anchor(updated_anchors)
        v_anchor = self.Wv_anchor(updated_anchors)
        feedback_score = torch.matmul(q_node, k_anchor.transpose(1, 2)) / math.sqrt(encoded_nodes.size(2))
        feedback = torch.matmul(F.softmax(feedback_score, dim=2), v_anchor)
        enhanced_nodes = self.add_n_normalization(encoded_nodes, feedback)
        return enhanced_nodes, updated_anchors

    def _build_anchor_tokens(self, encoded_nodes, node_xy, depot_node_demand, depot_distance):
        problem_size = node_xy.size(1)
        customer_embeddings = encoded_nodes[:, 1:, :]
        customer_demand = depot_node_demand[:, 1:]
        customer_depot_distance = depot_distance[:, 1:]

        anchor_list = [encoded_nodes[:, 0:1, :]]

        demand_k = max(1, int(math.ceil(self.model_params['hacd_anchor_demand_ratio'] * problem_size)))
        demand_index = torch.argsort(customer_demand, dim=1, descending=True, stable=True)[:, :demand_k]
        anchor_list.append(_batched_mean_gather(customer_embeddings, demand_index))

        peripheral_k = max(1, int(math.ceil(self.model_params['hacd_anchor_peripheral_ratio'] * problem_size)))
        peripheral_index = torch.argsort(
            customer_depot_distance, dim=1, descending=True, stable=True
        )[:, :peripheral_k]
        anchor_list.append(_batched_mean_gather(customer_embeddings, peripheral_index))

        cluster_num = self.model_params['hacd_cluster_anchor_num']
        if cluster_num is None:
            cluster_num = 4 if problem_size <= 50 else 8
        if cluster_num > 0:
            anchor_list.append(self._cluster_anchor_tokens(customer_embeddings, node_xy, customer_depot_distance, cluster_num))

        return torch.cat(anchor_list, dim=1)
        # shape: (batch, anchor, embedding)

    def _cluster_anchor_tokens(self, customer_embeddings, node_xy, customer_depot_distance, cluster_num):
        batch_size, problem_size, _ = customer_embeddings.size()
        cluster_num = min(cluster_num, problem_size)
        batch_index = torch.arange(batch_size, device=node_xy.device)

        # Median radial customer, then farthest-first; ties use customer index.
        radial_order = torch.argsort(
            customer_depot_distance, dim=1, stable=True
        )
        first_index = radial_order[:, problem_size // 2]
        selected_index = [first_index]
        centers = [node_xy[batch_index, first_index]]

        for _ in range(1, cluster_num):
            center_tensor = torch.stack(centers, dim=1)
            min_distance = torch.cdist(node_xy, center_tensor, p=2).min(dim=2).values
            used = torch.stack(selected_index, dim=1)
            min_distance.scatter_(1, used, float('-inf'))
            next_index = min_distance.argmax(dim=1)
            selected_index.append(next_index)
            centers.append(node_xy[batch_index, next_index])

        center_tensor = torch.stack(centers, dim=1)
        if self.model_params['hacd_anchor_method'] != 'legacy_soft':
            # Lloyd assignment/update. Empty clusters take the farthest eligible
            # customer from preceding centres; singleton donors are preserved.
            for _ in range(self.model_params['hacd_anchor_kmeans_iters']):
                distance = torch.cdist(node_xy, center_tensor, p=2)
                assignment = distance.argmin(dim=2)  # centre-order ties
                counts = torch.zeros(batch_size, cluster_num, device=node_xy.device, dtype=node_xy.dtype)
                counts.scatter_add_(1, assignment, torch.ones_like(assignment, dtype=node_xy.dtype))
                for k in range(cluster_num):
                    empty = counts[:, k] == 0
                    donor_count = counts.gather(1, assignment)
                    preceding = distance[:, :, :max(1, k)].min(dim=2).values
                    eligible = donor_count > 1
                    pick = preceding.masked_fill(~eligible, float('-inf')).argmax(dim=1)
                    donor = assignment[batch_index, pick]
                    assignment[batch_index, pick] = torch.where(empty, k, donor)
                    counts[batch_index, donor] -= empty.to(counts.dtype)
                    counts[:, k] += empty.to(counts.dtype)
                sums = torch.zeros(batch_size, cluster_num, 2, device=node_xy.device, dtype=node_xy.dtype)
                sums.scatter_add_(1, assignment[:, :, None].expand(-1, -1, 2), node_xy)
                center_tensor = sums / counts[:, :, None]
            if self.model_params['hacd_anchor_method'] == 'kmeans':
                tokens = torch.zeros(batch_size, cluster_num, customer_embeddings.size(2),
                                     device=customer_embeddings.device, dtype=customer_embeddings.dtype)
                tokens.scatter_add_(1, assignment[:, :, None].expand_as(customer_embeddings), customer_embeddings)
                return tokens / counts[:, :, None]
            # Optional explicit temperature pooling after Lloyd updates.
            # It is never inferred automatically for old checkpoints.
        distance = torch.cdist(node_xy, center_tensor, p=2)
        temperature = max(float(self.model_params['hacd_anchor_temperature']), 1e-4)
        weights = F.softmax(-distance / temperature, dim=1)
        return torch.bmm(weights.transpose(1, 2), customer_embeddings)
        # shape: (batch, cluster_num, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        branch_hidden_dim = self.model_params['hacd_branch_hidden_dim']

        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.encoded_nodes = None
        self.depot_node_xy = None
        self.depot_node_demand = None
        self.depot_distance = None
        self.routing_anchors = None
        self.last_hacd_debug = {}

        if self.model_params['hacd_capacity_residual_enabled']:
            self.spatial_last = nn.Linear(embedding_dim, branch_hidden_dim, bias=False)
            self.spatial_candidate = nn.Linear(embedding_dim, branch_hidden_dim, bias=False)
            self.spatial_geometry = nn.Linear(3, branch_hidden_dim)
            self.spatial_output = nn.Linear(branch_hidden_dim, 1)
            self.capacity_risk = nn.Sequential(
                nn.Linear(6, branch_hidden_dim),
                nn.GELU(),
                nn.Linear(branch_hidden_dim, 1),
            )
            self.risk_gate = nn.Sequential(
                nn.Linear(3, branch_hidden_dim),
                nn.GELU(),
                nn.Linear(branch_hidden_dim, 1),
            )
            self.customer_fusion_gate = nn.Linear(4, 1)
            self.depot_return_score = nn.Sequential(
                nn.Linear(4, branch_hidden_dim),
                nn.GELU(),
                nn.Linear(branch_hidden_dim, 1),
            )
            self.depot_fusion_gate = nn.Linear(4, 1)
            nn.init.constant_(self.customer_fusion_gate.bias, self.model_params['hacd_fusion_init_bias'])
            nn.init.constant_(self.depot_fusion_gate.bias, self.model_params['hacd_fusion_init_bias'])

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.encoded_nodes = encoded_nodes
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)
        if self.model_params['hacd_capacity_residual_enabled']:
            self.candidate_spatial = self.spatial_candidate(encoded_nodes[:, 1:, :])

    def set_static_context(self, depot_node_xy, depot_node_demand, routing_anchors):
        self.depot_node_xy = depot_node_xy
        self.depot_node_demand = depot_node_demand
        depot_xy = depot_node_xy[:, [0], :]
        self.depot_distance = (depot_node_xy - depot_xy).pow(2).sum(dim=2).sqrt()
        self.routing_anchors = routing_anchors

    def forward(self, encoded_last_node, current_node, load, ninf_mask, visited_ninf_flag):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # current_node.shape: (batch, pomo)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem+1)

        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, pomo, embedding+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem+1)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem+1)

        if self.model_params['hacd_capacity_residual_enabled']:
            score_scaled = self._apply_capacity_residual(
                score_scaled, encoded_last_node, current_node, load, visited_ninf_flag
            )

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem+1)

        return probs

    def _apply_capacity_residual(self, base_score, encoded_last_node, current_node, load, visited_ninf_flag):
        batch_size, pomo_size, node_count = base_score.size()
        problem_size = node_count - 1
        embedding_dim = encoded_last_node.size(2)
        eps = 1e-6

        current_xy = _get_xy(self.depot_node_xy, current_node)
        candidate_xy = self.depot_node_xy[:, None, 1:, :]
        current_distance = (candidate_xy - current_xy[:, :, None, :]).pow(2).sum(dim=3).sqrt()

        depot_distance = self.depot_distance[:, None, 1:].expand(batch_size, pomo_size, problem_size)
        candidate_demand = self.depot_node_demand[:, None, 1:].expand(batch_size, pomo_size, problem_size)
        load_expanded = load[:, :, None].expand(batch_size, pomo_size, problem_size)

        geometry = torch.stack((current_distance, depot_distance, load_expanded), dim=3)
        spatial_hidden = (
            self.spatial_last(encoded_last_node)[:, :, None, :]
            + self.candidate_spatial[:, None, :, :]
            + self.spatial_geometry(geometry)
        )
        spatial_attraction = self.spatial_output(F.gelu(spatial_hidden)).squeeze(dim=3)

        unserved_mask = visited_ninf_flag[:, :, 1:] == 0
        customer_demand = self.depot_node_demand[:, 1:]
        customer_depot_distance = self.depot_distance[:, 1:]
        unserved_demand = customer_demand[:, None, :] * unserved_mask.float()
        unserved_count = unserved_mask.float().sum(dim=2)
        unserved_sum = unserved_demand.sum(dim=2)
        mean_unserved_demand = unserved_sum / unserved_count.clamp_min(1)

        after_sum = unserved_sum[:, :, None] - candidate_demand * unserved_mask.float()
        after_count = unserved_count[:, :, None] - unserved_mask.float()
        mean_after_demand = after_sum / after_count.clamp_min(1)

        risk_feature = torch.stack((
            candidate_demand,
            load_expanded,
            candidate_demand / (load_expanded.clamp_min(0) + eps),
            load_expanded - candidate_demand,
            mean_unserved_demand[:, :, None].expand(batch_size, pomo_size, problem_size),
            mean_after_demand,
        ), dim=3)
        capacity_risk = F.softplus(self.capacity_risk(risk_feature).squeeze(dim=3))

        gate_feature = torch.stack((
            load,
            unserved_count / max(1, problem_size),
            mean_unserved_demand,
        ), dim=2)
        gamma = torch.sigmoid(self.risk_gate(gate_feature).squeeze(dim=2))

        beta = self.model_params['hacd_residual_beta']
        if self.model_params.get('hacd_without_capacity_feature', False):
            capacity_residual = beta * torch.tanh(spatial_attraction)
        else:
            capacity_residual = beta * torch.tanh(spatial_attraction - gamma[:, :, None] * capacity_risk)

        unserved_depot_distance = customer_depot_distance[:, None, :] * unserved_mask.float()
        mean_unserved_depot_distance = unserved_depot_distance.sum(dim=2) / unserved_count.clamp_min(1)
        depot_feature = torch.stack((
            load,
            1 - unserved_count / max(1, problem_size),
            mean_unserved_demand,
            mean_unserved_depot_distance,
        ), dim=2)
        depot_beta = self.model_params['hacd_depot_beta']
        if self.model_params.get('hacd_without_depot_residual', False):
            depot_residual = torch.zeros_like(base_score[:, :, 0])
        else:
            depot_residual = depot_beta * torch.tanh(self.depot_return_score(depot_feature).squeeze(dim=2))
        action_residual = torch.cat((depot_residual[:, :, None], capacity_residual), dim=2)

        if self.model_params.get('hacd_without_fusion_gate', False):
            fusion_gate = torch.ones_like(action_residual)
        else:
            customer_fusion_feature = torch.stack((
                base_score[:, :, 1:], capacity_residual, load_expanded, candidate_demand
            ), dim=3)
            customer_gate = torch.sigmoid(
                self.customer_fusion_gate(customer_fusion_feature).squeeze(dim=3)
            )

            depot_fusion_feature = torch.stack((
                base_score[:, :, 0],
                depot_residual,
                load,
                unserved_count / max(1, problem_size),
            ), dim=2)
            depot_gate = torch.sigmoid(self.depot_fusion_gate(depot_fusion_feature).squeeze(dim=2))
            fusion_gate = torch.cat((depot_gate[:, :, None], customer_gate), dim=2)

        if self.model_params['hacd_debug_enabled']:
            self.last_hacd_debug = {
                'spatial_attraction': spatial_attraction.detach(),
                'capacity_risk': capacity_risk.detach(),
                'risk_gate': gamma.detach(),
                'capacity_residual': action_residual.detach(),
                'fusion_gate': fusion_gate.detach(),
                'depot_residual': depot_residual.detach(),
            }

        return base_score + fusion_gate * action_residual


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / math.sqrt(key_dim)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


def _batched_mean_gather(source, index):
    # source.shape: (batch, problem, embedding)
    # index.shape: (batch, selected)
    batch_size, _, embedding_dim = source.size()
    gather_index = index[:, :, None].expand(batch_size, index.size(1), embedding_dim)
    return source.gather(dim=1, index=gather_index).mean(dim=1, keepdim=True)


def _get_xy(depot_node_xy, node_index_to_pick):
    # depot_node_xy.shape: (batch, problem+1, 2)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size, pomo_size = node_index_to_pick.size()
    gather_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, 2)
    return depot_node_xy.gather(dim=1, index=gather_index)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
