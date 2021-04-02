import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_softmax import Sparsemax
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce


class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min', fill_value=fill)
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class PageRankScore(MessagePassing):
    def __init__(self, channels, k=10, alpha=0.1, **kwargs):
        super(PageRankScore, self).__init__(aggr='add', **kwargs)
        self.channels = channels
        self.k = k
        self.alpha = alpha
        self.gnn = GCN(channels, 1)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = GCN.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)

        x = self.gnn(x, edge_index, edge_weight)
        hidden = x
        for k in range(self.k):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = x * (1 - self.alpha)
            x = x + self.alpha * hidden

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class MVPool(torch.nn.Module):
    def __init__(self, in_channels, ratio, args, negative_slop=0.2):
        super(MVPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.hc = args.hop_connection
        self.h_hop = args.hop
        self.lamb = args.lamb
        self.negative_slop = negative_slop

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.weight = Parameter(torch.Tensor(1, in_channels))
        nn.init.xavier_uniform_(self.weight.data)
        self.view_att = Parameter(torch.Tensor(3, 3))
        nn.init.xavier_uniform_(self.view_att.data)
        self.view_bias = Parameter(torch.Tensor(3))
        nn.init.zeros_(self.view_bias.data)
        self.alpha = Parameter(torch.Tensor(1))
        nn.init.ones_(self.alpha.data)
        self.beta = Parameter(torch.Tensor(1))
        nn.init.ones_(self.beta.data)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_pagerank_score = PageRankScore(in_channels)

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        row, col = edge_index
        score1 = torch.sigmoid(self.alpha * torch.log(degree(row, num_nodes=x.size(0)) + 1e-16) + self.beta).view(-1, 1)
        x_score2 = (x * self.weight).sum(dim=-1)
        score2 = torch.sigmoid(x_score2 / self.weight.norm(p=2, dim=-1)).view(-1, 1)
        x_score3 = self.calc_pagerank_score(x, edge_index, edge_attr)
        score3 = torch.sigmoid(x_score3).view(-1, 1)

        score_cat = torch.cat([score1, score2, score3], dim=1)
        max_value, _ = torch.max(torch.abs(score_cat), dim=0)
        score_cat = score_cat / max_value
        score_weight = torch.sigmoid(torch.matmul(score_cat, self.view_att) + self.view_bias)
        score_weight = torch.softmax(score_weight, dim=1)
        score = torch.sigmoid(torch.sum(score_cat * score_weight, dim=1))
        # score = score2.view(-1)

        # Graph Pooling
        original_x = x
        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # Discard structure learning layer, directly return
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch, perm

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            if self.h_hop >= 2:
                hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
                for _ in range(self.h_hop - 1):
                    hop_data = self.neighbor_augment(hop_data)
                hop_edge_index = hop_data.edge_index
                hop_edge_attr = hop_data.edge_attr
                new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0))
                if self.hc is True:
                    return x, new_edge_index, new_edge_attr, batch, perm
            else:
                new_edge_index = induced_edge_index
                new_edge_attr = induced_edge_attr
                if self.hc is True:
                    return x, new_edge_index, new_edge_attr, batch, perm

            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()
        else:
            # Learning the possible edge weights between each pair of nodes in the pooled subgraph, relative slower.
            if edge_attr is None:
                induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
                                               device=induced_edge_index.device)
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            # Construct batch fully connected graph in block diagonal matirx format
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index

            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index

            adj[induced_row, induced_col] += induced_edge_attr * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()

        return x, new_edge_index, new_edge_attr, batch, perm
