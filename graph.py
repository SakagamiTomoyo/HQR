from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch


class Graph:
    def __init__(self, embed_dim, triple_list, n_entity, n_relation, node_count, edge_count, sample_limit):
        self.embed_dim = embed_dim
        self.triple_list = triple_list
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.node_count = node_count
        self.edge_count = edge_count
        self.triple_num = len(triple_list)
        self.sample_limit = sample_limit
        self.edge_index = [[], []]
        self.edge_type = []

        self._make_adj_lists()
        self._make_pyg_graph()

    def _make_adj_lists(self):
        self.adj_lists = defaultdict(lambda: defaultdict(list))
        self.inverse_adj_lists = defaultdict(lambda: defaultdict(list))
        for triple in self.triple_list:
            h, r, t = triple
            self.adj_lists[h][r].append(t)
            self.inverse_adj_lists[t][r].append(h)
            self.edge_index[0].append(h)
            self.edge_index[1].append(t)
            self.edge_type.append(r)

    def _make_pyg_graph(self):
        self.pyg_kg = Data(edge_index=torch.tensor(self.edge_index, dtype=torch.long)).cuda()
        self.pyg_kg.edge_type = torch.tensor(self.edge_type, dtype=torch.long).cuda()
        self.pyg_kg.n_entity = self.n_entity
        self.pyg_kg.n_relation = self.n_relation
        self.pyg_kg.node_count = self.node_count
        self.pyg_kg.edge_count = self.edge_count
        self.pyg_kg.edge_count = self.edge_count
        self.pyg_kg.in_degree = degree(self.pyg_kg.edge_index[0])
        self.pyg_kg.out_degree = degree(self.pyg_kg.edge_index[1])

    def _sample_neighbors(self):
        return
