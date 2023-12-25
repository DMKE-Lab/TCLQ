import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

# from .data import Stack
from .data_temporal import Stack


@R.register("model.GNN-QE")
class QueryExecutor(nn.Module, core.Configurable):
    """
    Query executor for answering multi-hop logical queries.

    Parameters:
        model (nn.Module): GNN model for node representation learning
        logic (str, optional): which fuzzy logic system to use, ``godel``, ``product`` or ``lukasiewicz``
        dropout_ratio (float, optional): ratio for traversal dropout
        num_mlp_layer (int, optional): number of MLP layers
    """

    stack_size = 2

    def __init__(self, model, logic="product", dropout_ratio=0.25, num_mlp_layer=2):
        super(QueryExecutor, self).__init__()
        self.model = RelationProjection(model, num_mlp_layer)
        self.symbolic_model = SymbolicTraversal()
        self.logic = logic
        self.dropout_ratio = dropout_ratio
        self.num_heads = 4

        self.intersection_operator = layers.SelfAttentionBlock(model.output_dim * 2, self.num_heads)
        self.union_operator = layers.SelfAttentionBlock(model.output_dim * 2, self.num_heads)
        self.difference_operator = layers.SelfAttentionBlock(model.output_dim, self.num_heads)

        self.mlp = layers.MLP(model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [1])

    def traversal_dropout(self, graph, h_prob, r_index, timestamp):
        """
        Dropout edges that can be directly traversed to create an incomplete graph.
        """
        sample, h_index = h_prob.nonzero().t()
        r_index = r_index[sample]
        any = -torch.ones_like(h_index)
        pattern = torch.stack([h_index, any, r_index], dim=-1)
        inverse_pattern = torch.stack([any, h_index, r_index ^ 1], dim=-1)
        pattern = torch.cat([pattern, inverse_pattern])
        edge_index = graph.match(pattern)[0]

        h_index, t_index = graph.edge_list.t()[:2]
        degree_h = h_index.bincount()
        degree_t = t_index.bincount()
        h_index, t_index = graph.edge_list[edge_index, :2].t()
        must_keep = (degree_h[h_index] <= 1) | (degree_t[t_index] <= 1)
        edge_index = edge_index[~must_keep]

        is_sampled = torch.rand(len(edge_index), device=self.device) <= self.dropout_ratio
        edge_index = edge_index[is_sampled]

        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def execute(self, graph, query, all_loss=None, metric=None):
        """Execute queries on the graph."""
        batch_size = len(query)
        self.stack = Stack(batch_size, self.stack_size, graph.num_node, device=self.device)
        self.symbolic_stack = Stack(batch_size, self.stack_size, graph.num_node, device=self.device)
        self.var = Stack(batch_size, query.shape[1], graph.num_node, device=self.device)
        self.symbolic_var = Stack(batch_size, query.shape[1], graph.num_node, device=self.device)
        # instruction pointer
        self.IP = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.IP_feature = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        all_sample = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        op = query[all_sample, self.IP]
        self.flag = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        while not op.is_stop().all():
            is_operand = op.is_operand()
            is_intersection = op.is_intersection()
            is_union = op.is_union()
            is_negation = op.is_negation()
            is_projection = op.is_projection()
            if is_operand.any():
                h_index = op[is_operand].get_operand()
                self.apply_operand(is_operand, h_index, graph.num_node)
            if is_intersection.any():
                self.apply_intersection(is_intersection)
            if is_union.any():
                self.apply_union(is_union)
            if is_negation.any():
                self.apply_negation(is_negation)
            if not (is_operand | is_negation | is_intersection | is_union).any() and is_projection.any():
                r_index = op[is_projection].get_operand()

                self.IP[is_projection] += 1
                op = query[all_sample, self.IP]
                is_start_time = op.is_start_time()
                if is_start_time.any():
                    start_time = op[is_projection].get_start_time()
                else:
                    raise ValueError("Is not start time!!")
                self.IP[is_start_time] += 1
                op = query[all_sample, self.IP]
                is_end_time = op.is_end_time()
                if is_end_time.any():
                    end_time = op[is_projection].get_end_time()
                else:
                    raise ValueError("Is not start time!!")

                self.apply_projection(is_projection, graph, r_index, start_time, end_time, all_loss=all_loss,
                                      metric=metric)
            op = query[all_sample, self.IP]

        if (self.stack.SP > 1).any():
            raise ValueError("More operands than expected")

    def forward(self, graph, query, all_loss=None, metric=None):
        self.execute(graph, query, all_loss=all_loss, metric=metric)
        node_feature = self.stack.pop_feature()
        t_prob = F.sigmoid(self.mlp(node_feature).squeeze(-1))
        t_prob = t_prob.t()
        t_logit = ((t_prob + 1e-10) / (1 - t_prob + 1e-10)).log()
        return t_logit

    def visualize(self, graph, full_graph, query):
        self.execute(graph, query)
        var_probs = self.var.stack
        easy_answers = self.symbolic_var.stack
        self.execute(full_graph, query)
        all_answers = self.symbolic_var.stack
        return var_probs, easy_answers, all_answers

    def apply_operand(self, mask, h_index, num_node):
        h_prob = functional.one_hot(h_index, num_node)
        self.stack.push(mask, h_prob)
        self.var.push(mask, h_prob)
        self.IP[mask] += 1
        self.flag[mask] = True


    def apply_intersection(self, mask):
        if (self.flag[mask] == True).any():
            y_node_feature, sym_y_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
            x_node_feature, sym_x_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        else:
            y_node_feature, sym_y_prob = self.stack.pop_feature(mask), self.symbolic_stack.pop(mask)
            x_node_feature, sym_x_prob = self.stack.pop_feature(mask), self.symbolic_stack.pop(mask)
        z_feature = self.conjunction_processer(x_node_feature, y_node_feature)
        self.stack.push_feature(mask, z_feature)
        self.var.push_feature(mask, z_feature)
        self.IP[mask] += 1
        self.IP_feature[mask] += 1
        self.flag[mask] = False

    def apply_union(self, mask):
        if (self.flag[mask] == True).any():
            y_node_feature, sym_y_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
            x_node_feature, sym_x_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        else:
            y_node_feature, sym_y_prob = self.stack.pop_feature(mask), self.symbolic_stack.pop(mask)
            x_node_feature, sym_x_prob = self.stack.pop_feature(mask), self.symbolic_stack.pop(mask)
        z_feature = self.disjunction_processer(x_node_feature, y_node_feature)
        self.stack.push_feature(mask, z_feature)
        self.var.push_feature(mask, z_feature)
        self.IP[mask] += 1
        self.IP_feature[mask] += 1
        self.flag[mask] = False

    def apply_negation(self, mask):
        if (self.flag[mask] == True).any():
            x_node_feature, sym_x_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        else:
            x_node_feature, sym_x_prob = self.stack.pop_feature(mask), self.symbolic_stack.pop(mask)
        y_node_feature = self.negation_processer(x_node_feature)
        self.stack.push_feature(mask, y_node_feature)
        self.var.push_feature(mask, y_node_feature)
        self.IP[mask] += 1
        self.IP_feature[mask] += 1
        self.flag[mask] = False

    def apply_projection(self, mask, graph, r_index, start_time, end_time, all_loss=None, metric=None):
        flag_one = self.flag & mask
        flag_two = (~flag_one) & mask
        if (flag_one == True).any():
            h_prob, sym_h_prob = self.stack.pop(flag_one), self.symbolic_stack.pop(flag_one)
            time_vocab = graph.tid_vocab
            time_vocab = torch.tensor(time_vocab, device=self.device).repeat(len(r_index), 1)
            mask_start = time_vocab >= start_time.unsqueeze(0).t()
            mask_end = time_vocab <= end_time.unsqueeze(0).t()
            mask_time = mask_start & mask_end
            time_interval = [time_vocab[i][mask_time[i]] for i in range(mask_time.shape[0])]

            max_len = 0
            for tmp in time_interval:
                max_len = max(max_len, len(tmp))
            for index, value in enumerate(time_interval):
                if len(value) < max_len:
                    tmp_ones = torch.ones(max_len - len(value), dtype=torch.long, device=self.device) * value[-1]
                    time_interval[index] = torch.cat((time_interval[index], tmp_ones))
            time_stamp = time_interval[0].unsqueeze(0).t()
            for i in range(1, len(time_interval)):
                time_stamp = torch.cat((time_stamp, time_interval[i].unsqueeze(0).t()), 1)
            t_flag = True
            t_num = 0
            for index, value in enumerate(time_stamp):
                h_prob_t = h_prob.clone()
                flag2index = [i for i, x in enumerate(flag_one[mask]) if x == True]
                node_feature = self.model(graph, h_prob_t, r_index[flag2index], value, all_loss=all_loss, metric=metric)
                if t_flag:
                    t_prob_result = torch.zeros_like(node_feature)
                    t_flag = False
                t_prob_result = t_prob_result + node_feature
                t_num = t_num + 1

            t_prob = t_prob_result / t_num
            print("t_prob: ")
            print(t_prob.shape)
            self.stack.push_feature(flag_one, t_prob)
            self.var.push_feature(flag_one, t_prob)
            self.IP[flag_one] += 1
            self.IP_feature[flag_one] += 1
            self.flag[flag_one] = False

        if (flag_two == True).any():
            h_prob, sym_h_prob = self.stack.pop(flag_two), self.symbolic_stack.pop(flag_two)

            time_vocab = graph.tid_vocab
            time_vocab = torch.tensor(time_vocab, device=self.device).repeat(len(r_index), 1)

            mask_start = time_vocab >= start_time.unsqueeze(0).t()
            mask_end = time_vocab <= end_time.unsqueeze(0).t()
            mask_time = mask_start & mask_end
            time_interval = [time_vocab[i][mask_time[i]] for i in range(mask_time.shape[0])]

            max_len = 0
            for tmp in time_interval:
                max_len = max(max_len, len(tmp))
            for index, value in enumerate(time_interval):
                if len(value) < max_len:
                    tmp_ones = torch.ones(max_len - len(value), dtype=torch.long, device=self.device) * value[-1]
                    time_interval[index] = torch.cat((time_interval[index], tmp_ones))
            time_stamp = time_interval[0].unsqueeze(0).t()
            for i in range(1, len(time_interval)):
                time_stamp = torch.cat((time_stamp, time_interval[i].unsqueeze(0).t()), 1)
            t_flag = True
            t_num = 0
            for index, value in enumerate(time_stamp):
                h_prob_t = h_prob.clone()
                flag2index = [i for i, x in enumerate(flag_two[mask]) if x == True]
                node_feature = self.model(graph, h_prob_t, r_index[flag2index], value, all_loss=all_loss, metric=metric)
                if t_flag:
                    t_prob_result = torch.zeros_like(node_feature)
                    t_flag = False
                t_prob_result = t_prob_result + node_feature
                t_num = t_num + 1

            t_prob = t_prob_result / t_num
            print("t_prob: ")
            print(t_prob.shape)
            self.stack.push_feature(flag_two, t_prob)
            self.var.push_feature(flag_two, t_prob)
            self.IP[flag_two] += 1
            self.IP_feature[flag_two] += 1
            self.flag[flag_two] = False

            h_prob, sym_h_prob = self.stack.pop_feature(flag_two), self.symbolic_stack.pop(flag_two)
            h_prob = h_prob.detach()
            flag2index = [i for i, x in enumerate(flag_two[mask]) if x == True]
            node_feature = self.model(graph, h_prob, r_index[flag2index], all_loss=all_loss, metric=metric)
            sym_t_prob = self.symbolic_model(graph, sym_h_prob, r_index[flag2index], all_loss=all_loss, metric=metric)

            self.stack.push_feature(flag_two, node_feature)
            self.var.push_feature(flag_two, node_feature)
            self.IP[flag_two] += 1
            self.IP_feature[flag_two] += 1
            self.flag[flag_two] = False

    def conjunction(self, x, y):
        if self.logic == "godel":
            return torch.min(x, y)
        elif self.logic == "product":
            return x * y
        elif self.logic == "lukasiewicz":
            return (x + y - 1).clamp(min=0)
        else:
            raise ValueError("Unknown fuzzy logic `%s`" % self.logic)

    def conjunction_processer(self, x_feature, y_feature):
        query = y_feature.transpose(0, 1)
        key = x_feature.transpose(0, 1)
        value = query * key
        # value = torch.cat((query, key), dim=2)
        output = self.intersection_operator(value)[0].transpose(0, 1)
        output = F.sigmoid(output)
        return output

    def disjunction(self, x, y):
        if self.logic == "godel":
            return torch.max(x, y)
        elif self.logic == "product":
            return x + y - x * y
        elif self.logic == "lukasiewicz":
            return (x + y).clamp(max=1)
        else:
            raise ValueError("Unknown fuzzy logic `%s`" % self.logic)

    def disjunction_processer(self, x_feature, y_feature):
        query = y_feature.transpose(0, 1)
        key = x_feature.transpose(0, 1)
        value = torch.cat((query, key), dim=2)
        output = self.union_operator(value)[0].transpose(0, 1)
        output = F.sigmoid(output)
        return output


    def negation(self, x):
        return 1 - x

    def negation_processer(self, x_feature):
        query = x_feature.transpose(0, 1)
        output = self.difference_operator(query)[0].transpose(0, 1)
        output = F.sigmoid(output)
        return output

    def calculate_results(self, feature):
        t_prob = F.sigmoid(feature)
        return t_prob

@R.register("model.RelationProjection")
class RelationProjection(nn.Module, core.Configurable):
    def __init__(self, model, num_mlp_layer=2):
        super(RelationProjection, self).__init__()
        self.model = model
        self.query = nn.Embedding(model.num_relation, model.input_dim)
        self.query_time = nn.Embedding(model.num_timestamp, model.input_dim)
        self.mlp = layers.MLP(model.output_dim * 2, [model.output_dim] * (num_mlp_layer - 1) + [1])

    def forward(self, graph, h_prob, r_index, timestamp, all_loss=None, metric=None):
        query = self.query(r_index)
        query_time = self.query_time(timestamp)
        graph = graph.clone()
        query_r_t = torch.cat((query, query_time), dim=-1)
        with graph.graph():
            graph.query = query
            graph.query_t = query_time
            graph.query_r_t = query_r_t

        input_r_t = torch.einsum("bn, bd -> nbd", h_prob, query_r_t)
        output = self.model(graph, input_r_t, all_loss=all_loss, metric=metric)
        output = F.sigmoid(self.mlp(output["node_feature"]).squeeze(-1))
        output["node_feature"] = F.sigmoid(output["node_feature"])
        return output["node_feature"]

@R.register("model.Symbolic")
class SymbolicTraversal(nn.Module, core.Configurable):

    def forward(self, graph, h_prob, r_index, timestamp, all_loss=None, metric=None, is_projection=False):
        batch_size = len(h_prob)
        any = -torch.ones_like(r_index)
        pattern = torch.stack([any, any, r_index, timestamp], dim=-1)
        edge_index, num_edges = graph.match(pattern, is_projection)
        num_nodes = graph.num_node.repeat(batch_size)
        graph = data.PackedGraph(graph.edge_list[edge_index], num_nodes=num_nodes, num_edges=num_edges)

        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:3], graph.edge_weight,
                                            (graph.num_node, graph.num_node))
        t_prob = functional.generalized_spmm(adjacency.t(), h_prob.view(-1, 1), sum="max").clamp(min=0)

        return t_prob.view_as(h_prob)
