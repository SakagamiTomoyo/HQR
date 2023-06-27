import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter as Param
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min, scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import inits
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import numpy as np


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class ProjectionAggregating(nn.Module):
    def __init__(self, dim, aggr='mlp'):
        super(ProjectionAggregating, self).__init__()

        self.dim = dim
        self.aggr = aggr

        self.attn_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.attn_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.attn_layer_1.weight)
        nn.init.xavier_uniform_(self.attn_layer_2.weight)

    def forward(self, query_emb, refer_embs, query_r=None, refer_r=None, start_embs=None):
        bias = refer_embs - (start_embs * refer_r)
        layer_act_1 = F.relu(self.attn_layer_1(
            torch.cat([query_r.unsqueeze(1).repeat(1, refer_r.shape[1], 1), refer_r], dim=-1)
        ))

        final_refer_emb = torch.sum(self.attn_layer_2(layer_act_1) * bias, dim=-2)

        ref_norm = final_refer_emb.norm(p=1, dim=-1).unsqueeze(1)
        query_norm = query_emb.norm(p=1, dim=-1).unsqueeze(1)
        const = 2.5

        final_refer_emb = final_refer_emb / (1e-9 + ref_norm / query_norm * const)

        final_query_emb = query_emb + final_refer_emb

        return final_query_emb


class IDN_GQE_DistMult(nn.Module):
    def __init__(self, graph, args):
        super(IDN_GQE_DistMult, self).__init__()
        self.id_neighbor_limit = args.id_neighbor_limit
        self.graph = graph
        self.embed_dim = args.embed_dim
        self.loss_f = args.loss_function
        self.adaptive = args.adaptive
        self.multi_neg = args.multi_neg_answers
        self.num_layers = args.num_layers
        self.mp_model = args.mp_model
        self.epsilon = 2.0
        self.gamma = Param(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )
        self.embedding_range = Param(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.embed_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(graph.n_entity + graph.n_relation + 1, self.embed_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(graph.n_relation + 1, self.embed_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.query_diameter = {('e', ('r',)): 1,
                               ('e', ('r', 'r')): 2,
                               ('e', ('r', 'r', 'r')): 3,
                               (('e', ('r',)), ('e', ('r',))): 1,
                               (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): 1,
                               ((('e', ('r',)), ('e', ('r',))), ('r',)): 2,
                               (('e', ('r', 'r')), ('e', ('r',))): 2,
                               (('e', ('r',)), ('e', ('r', 'n'))): [2, 2],
                               (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): [3, 2],
                               ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): [1, 1],
                               (('e', ('r', 'r')), ('e', ('r', 'n'))): [2, 3],
                               (('e', ('r', 'r', 'n')), ('e', ('r',))): [2, 3],
                               (('e', ('r',)), ('e', ('r',)), ('u',)): [1, 1],
                               ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): [1, 1],
                               ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): [1, 1],
                               ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): [1, 1]
                               }

        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()

        self.center_net = CenterIntersection(self.embed_dim)
        self.mlp_projection = ProjectionAggregating(self.embed_dim, aggr='bias')

        print("idn-gqe-distmult")

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):

        batch_size = len(chosen_queries)

        h = self.entity_embedding
        r = self.relation_embedding

        h = torch.cat([h, torch.tensor([[0] * self.embed_dim], requires_grad=False).cuda()])
        r = torch.cat([r, torch.tensor([[0] * self.embed_dim], requires_grad=False).cuda()])

        if query_name_dict[query_type] == '1p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            query_emb = h[anchors] * r[rel_0]

            h_target = h[torch.tensor([subgraph.p1_target], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p1_rel], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)

        elif query_name_dict[query_type] == '2p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            rel_1 = [query[1][1] for query in chosen_queries]
            query_emb = h[anchors] * r[rel_0]

            h_target = h[torch.tensor([subgraph.p2_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p2_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            query_emb = query_emb * r[rel_1]

            h_target = h[torch.tensor([subgraph.p2_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p2_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[torch.tensor([subgraph.p2_target1], dtype=torch.long).squeeze()]

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)

        elif query_name_dict[query_type] == '3p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            rel_1 = [query[1][1] for query in chosen_queries]
            rel_2 = [query[1][2] for query in chosen_queries]
            query_emb = h[anchors] * r[rel_0]

            h_target = h[torch.tensor([subgraph.p3_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p3_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            query_emb = query_emb * r[rel_1]

            h_target = h[torch.tensor([subgraph.p3_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p3_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[torch.tensor([subgraph.p3_target1], dtype=torch.long).squeeze()]

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            query_emb = query_emb * r[rel_2]

            h_target = h[torch.tensor([subgraph.p3_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p3_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[torch.tensor([subgraph.p3_target2], dtype=torch.long).squeeze()]

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)

        elif query_name_dict[query_type] == '2i':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[1][1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] * r[rel_0]
            subquery_2 = h[anchor_1] * r[rel_1]

            h_target = h[torch.tensor([subgraph.i2_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.i2_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)

            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.i2_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.i2_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[anchor_1].unsqueeze(1)

            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))

        elif query_name_dict[query_type] == '3i':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]
            anchor_2 = [q[2][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[1][1][0] for query in chosen_queries]
            rel_2 = [query[2][1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] * r[rel_0]
            subquery_2 = h[anchor_1] * r[rel_1]
            subquery_3 = h[anchor_2] * r[rel_2]

            h_target = h[torch.tensor([subgraph.i3_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.i3_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)

            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.i3_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.i3_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[anchor_1].unsqueeze(1)

            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.i3_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.i3_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[anchor_2].unsqueeze(1)

            subquery_3 = self.mlp_projection(subquery_3, h_target, r_query, r_target, refer_start)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2, subquery_3]))

        elif query_name_dict[query_type] == 'pi':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[0][1][1] for query in chosen_queries]
            rel_2 = [query[1][1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] * r[rel_0]

            h_target = h[torch.tensor([subgraph.pi_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.pi_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            subquery_1 = subquery_1 * r[rel_1]

            h_target = h[torch.tensor([subgraph.pi_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.pi_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[torch.tensor([subgraph.pi_target1], dtype=torch.long).squeeze()]
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            subquery_2 = h[anchor_1] * r[rel_2]

            h_target = h[torch.tensor([subgraph.pi_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.pi_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[anchor_1].unsqueeze(1)
            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))

        elif query_name_dict[query_type] == 'ip':
            anchor_0 = [q[0][0][0] for q in chosen_queries]
            anchor_1 = [q[0][1][0] for q in chosen_queries]

            rel_0 = [query[0][0][1][0] for query in chosen_queries]
            rel_1 = [query[0][1][1][0] for query in chosen_queries]
            rel_2 = [query[1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] * r[rel_0]
            subquery_2 = h[anchor_1] * r[rel_1]

            h_target = h[torch.tensor([subgraph.ip_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.ip_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.ip_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.ip_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[anchor_1].unsqueeze(1)
            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))

            query_emb = query_emb * r[rel_2]

            h_target = h[torch.tensor([subgraph.ip_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.ip_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[torch.tensor([subgraph.ip_target1], dtype=torch.long).squeeze()]
            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)

        elif query_name_dict[query_type] == '2u-DNF':
            anchors = [[query[0][0] for query in chosen_queries], [query[1][0] for query in chosen_queries]]
            rels = [[query[0][1][0] for query in chosen_queries], [query[1][1][0] for query in chosen_queries]]

            subquery_1 = h[anchors[0]] * r[rels[0]]
            subquery_2 = h[anchors[1]] * r[rels[1]]

            h_target = h[torch.tensor([subgraph.u2_target], dtype=torch.long).squeeze()[:, 0]]
            r_target = r[torch.tensor([subgraph.u2_rel], dtype=torch.long).squeeze()[:, 0]]
            r_query = r[rels[0]]
            refer_start = h[anchors[0]].unsqueeze(1)
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.u2_target], dtype=torch.long).squeeze()[:, 1]]
            r_target = r[torch.tensor([subgraph.u2_rel], dtype=torch.long).squeeze()[:, 1]]
            r_query = r[rels[1]]
            refer_start = h[anchors[1]].unsqueeze(1)
            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            query_emb = torch.cat([subquery_1.unsqueeze(1),
                                   subquery_2.unsqueeze(1)], dim=-2)

        elif query_name_dict[query_type] == 'up-DNF':
            anchors = [[query[0][0][0] for query in chosen_queries],
                       [query[0][1][0] for query in chosen_queries]]

            rel_1 = [[query[0][0][1][0] for query in chosen_queries], [query[0][1][1][0] for query in chosen_queries]]
            rel_2 = [query[1][0] for query in chosen_queries]

            subquery_1 = h[anchors[0]] * r[rel_1[0]]
            subquery_2 = h[anchors[1]] * r[rel_1[1]]

            h_target = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 0]]
            r_target = r[torch.tensor([subgraph.up_rel_1], dtype=torch.long).squeeze()[:, 0]]
            r_query = r[rel_1[0]]
            refer_start = h[anchors[0]].unsqueeze(1)
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 1]]
            r_target = r[torch.tensor([subgraph.up_rel_1], dtype=torch.long).squeeze()[:, 1]]
            r_query = r[rel_1[1]]
            refer_start = h[anchors[1]].unsqueeze(1)
            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            subquery_1 = subquery_1 * r[rel_2]
            subquery_2 = subquery_2 * r[rel_2]

            h_target = h[torch.tensor([subgraph.up_target_2], dtype=torch.long).squeeze()[:, 0]]
            r_target = r[torch.tensor([subgraph.up_rel_2], dtype=torch.long).squeeze()[:, 0]]
            r_query = r[rel_2]
            refer_start = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 0]]
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            h_target = h[torch.tensor([subgraph.up_target_2], dtype=torch.long).squeeze()[:, 1]]
            r_target = r[torch.tensor([subgraph.up_rel_2], dtype=torch.long).squeeze()[:, 1]]
            r_query = r[rel_2]
            refer_start = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 1]]
            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            query_emb = torch.cat([subquery_1.unsqueeze(1),
                                   subquery_2.unsqueeze(1)], dim=-2)

        if not evaluating:
            target_embeds = self.entity_embedding[chosen_answers]
            distance = query_emb * target_embeds
            scores = torch.sum(distance, dim=-1)

            neg_embeds = self.entity_embedding[torch.tensor(neg_answers)]
            neg_distance = query_emb.unsqueeze(1) * neg_embeds
            neg_scores = torch.sum(neg_distance, dim=-1)

            return scores, None, neg_scores

        else:
            if 'u' in query_name_dict[query_type]:
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_embedding[:self.graph.n_entity]
                for i in range(batch_size):
                    distance = torch.sum(target_embeds.unsqueeze(1) * query_emb[i], dim=-1)
                    scores[i] = torch.max(distance, dim=-1)[0]
                return scores

            scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
            target_embeds = self.entity_embedding[:self.graph.n_entity]
            for i in range(batch_size):
                distance = query_emb[i].t().unsqueeze(0) * target_embeds
                scores[i] = torch.sum(distance, dim=-1)
            return scores

    @staticmethod
    def relu_like_distance(query_embs, refer_embs, allowed_max_dist=15, debuff=0.3):
        distance = torch.norm(refer_embs - query_embs, p=1, dim=-1)
        return torch.where(distance > allowed_max_dist, distance * debuff, torch.zeros(distance.shape).cuda())

    def margin_loss(self, chosen_queries, chosen_answers, neg_answers, query_type, sample_limit, subgraph,
                    all_query_num,
                    margin=1, query_name_dict=None):
        affs, refer_affs, neg_affs = self.forward(
            chosen_queries, query_type, sample_limit, subgraph,
            chosen_answers, query_name_dict, neg_answers=neg_answers)

        negative_score = F.logsigmoid(-neg_affs).mean()
        positive_score = F.logsigmoid(affs).mean(dim=0)
        positive_sample_loss = - positive_score
        negative_sample_loss = - negative_score

        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss
