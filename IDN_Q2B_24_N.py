import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param


def Identity(x):
    return x


class BoxOffsetIntersection(nn.Module):
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class ProjectionAggregating(nn.Module):
    def __init__(self, dim, cen, aggr='mlp'):
        super(ProjectionAggregating, self).__init__()

        self.dim = dim
        self.aggr = aggr

        self.attn_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.attn_layer_2 = nn.Linear(self.dim, self.dim)

        self.offset_layer_1 = nn.Linear(self.dim * 4, self.dim)
        self.offset_layer_2 = nn.Linear(self.dim * 3, self.dim)
        self.offset_layer_3 = nn.Linear(self.dim, 1)

        nn.init.xavier_uniform_(self.attn_layer_1.weight)
        nn.init.xavier_uniform_(self.attn_layer_2.weight)

        nn.init.xavier_uniform_(self.offset_layer_1.weight)
        nn.init.xavier_uniform_(self.offset_layer_2.weight)
        nn.init.xavier_uniform_(self.offset_layer_3.weight)

    def forward(self, query_emb, offset_emb, refer_embs,
                query_r=None, refer_r=None, start_embs=None,
                query_r_offset=None, refer_r_offset=None):
        bias = refer_embs - (start_embs + refer_r)

        layer_act_1 = F.relu(self.attn_layer_1(
            torch.cat([refer_r,
                       query_r.unsqueeze(1).repeat(1, refer_r.shape[1], 1)], dim=-1
                      )
        ))

        final_refer_emb = torch.sum(self.attn_layer_2(layer_act_1) * bias, dim=-2)

        final_query_emb = query_emb + final_refer_emb

        final_offset_emb = offset_emb

        return final_query_emb, final_offset_emb


class IDN_Q2B_24(nn.Module):
    def __init__(self, graph, args):
        super(IDN_Q2B_24, self).__init__()

        self.graph = graph
        self.embed_dim = args.embed_dim
        self.loss_f = args.loss_function
        self.multi_neg = args.multi_neg_answers

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
        activation, cen = 'none', 0.02
        self.cen = cen
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus

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

        self.offset_embedding = nn.Parameter(torch.zeros(graph.n_relation + 1, self.embed_dim))
        nn.init.uniform_(
            tensor=self.offset_embedding,
            a=0.,
            b=self.embedding_range.item()
        )
        self.center_net = CenterIntersection(self.embed_dim)
        self.offset_net = BoxOffsetIntersection(self.embed_dim)
        self.mlp_projection = ProjectionAggregating(self.embed_dim, self.cen, aggr='bias')

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):
        batch_size = len(chosen_queries)

        h = self.entity_embedding
        r_center = self.relation_embedding
        r_offset = self.offset_embedding

        h = torch.cat([h, torch.tensor([[0] * self.embed_dim], requires_grad=False).cuda()])
        r_center = torch.cat([r_center, torch.tensor([[0] * self.embed_dim], requires_grad=False).cuda()])
        r_offset = torch.cat([r_offset, torch.tensor([[0] * self.embed_dim], requires_grad=False).cuda()])

        if query_name_dict[query_type] == '1p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            query_emb = h[anchors] + r_center[rel_0]
            offset_emb = self.func(r_offset[rel_0])

            h_target = h[torch.tensor([subgraph.p1_target], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.p1_rel], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.p1_rel], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

        elif query_name_dict[query_type] == '2p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            rel_1 = [query[1][1] for query in chosen_queries]
            query_emb = h[anchors] + r_center[rel_0]
            offset_emb = self.func(r_offset[rel_0])

            h_target = h[torch.tensor([subgraph.p2_target1], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.p2_rel1], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.p2_rel1], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

            query_emb = query_emb + r_center[rel_1]
            offset_emb = offset_emb + self.func(r_offset[rel_1])

            h_target = h[torch.tensor([subgraph.p2_target2], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.p2_rel2], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.p2_rel2], dtype=torch.long).squeeze()]
            r_query = r_center[rel_1]
            r_query_offset = r_offset[rel_1]
            refer_start = h[torch.tensor([subgraph.p2_target1], dtype=torch.long).squeeze()]

            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

        elif query_name_dict[query_type] == '3p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            rel_1 = [query[1][1] for query in chosen_queries]
            rel_2 = [query[1][2] for query in chosen_queries]

            query_emb = h[anchors] + r_center[rel_0]
            offset_emb = self.func(r_offset[rel_0])

            h_target = h[torch.tensor([subgraph.p3_target1], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.p3_rel1], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.p3_rel1], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

            query_emb = query_emb + r_center[rel_1]
            offset_emb = offset_emb + self.func(r_offset[rel_1])

            h_target = h[torch.tensor([subgraph.p3_target2], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.p3_rel2], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.p3_rel2], dtype=torch.long).squeeze()]
            r_query = r_center[rel_1]
            r_query_offset = r_offset[rel_1]
            refer_start = h[torch.tensor([subgraph.p3_target1], dtype=torch.long).squeeze()]

            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

            query_emb = query_emb + r_center[rel_2]
            offset_emb = offset_emb + self.func(r_offset[rel_2])

            h_target = h[torch.tensor([subgraph.p3_target3], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.p3_rel3], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.p3_rel3], dtype=torch.long).squeeze()]
            r_query = r_center[rel_2]
            r_query_offset = r_offset[rel_2]
            refer_start = h[torch.tensor([subgraph.p3_target2], dtype=torch.long).squeeze()]

            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

        elif query_name_dict[query_type] == '2i':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[1][1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] + r_center[rel_0]
            subquery_2 = h[anchor_1] + r_center[rel_1]

            offset_emb_1 = self.func(r_offset[rel_0])
            offset_emb_2 = self.func(r_offset[rel_1])

            h_target = h[torch.tensor([subgraph.i2_target1], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.i2_rel1], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.i2_rel1], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)

            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                                        h_target, r_query, r_target,
                                                        refer_start,
                                                        r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.i2_target2], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.i2_rel2], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.i2_rel2], dtype=torch.long).squeeze()]
            r_query = r_center[rel_1]
            r_query_offset = r_offset[rel_1]
            refer_start = h[anchor_1].unsqueeze(1)

            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                                         h_target, r_query, r_target,
                                                         refer_start,
                                                         r_query_offset, r_target_offset)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2]))

        elif query_name_dict[query_type] == '3i':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]
            anchor_2 = [q[2][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[1][1][0] for query in chosen_queries]
            rel_2 = [query[2][1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] + r_center[rel_0]
            subquery_2 = h[anchor_1] + r_center[rel_1]
            subquery_3 = h[anchor_2] + r_center[rel_2]

            offset_emb_1 = self.func(r_offset[rel_0])
            offset_emb_2 = self.func(r_offset[rel_1])
            offset_emb_3 = self.func(r_offset[rel_2])

            h_target = h[torch.tensor([subgraph.i3_target1], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.i3_rel1], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.i3_rel1], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)

            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.i3_target2], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.i3_rel2], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.i3_rel2], dtype=torch.long).squeeze()]
            r_query = r_center[rel_1]
            r_query_offset = r_offset[rel_1]
            refer_start = h[anchor_1].unsqueeze(1)

            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.i3_target3], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.i3_rel3], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.i3_rel3], dtype=torch.long).squeeze()]
            r_query = r_center[rel_2]
            r_query_offset = r_offset[rel_2]
            refer_start = h[anchor_2].unsqueeze(1)

            subquery_3, offset_emb_3 = self.mlp_projection(subquery_3, offset_emb_3,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2, subquery_3]))
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2, offset_emb_3]))

        elif query_name_dict[query_type] == 'pi':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[0][1][1] for query in chosen_queries]
            rel_2 = [query[1][1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] + r_center[rel_0]
            offset_emb_1 = self.func(r_offset[rel_0])

            h_target = h[torch.tensor([subgraph.pi_target1], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.pi_rel1], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.pi_rel1], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)
            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                             h_target, r_query, r_target, refer_start,
                                             r_query_offset, r_target_offset)

            subquery_1 = subquery_1 + r_center[rel_1]
            offset_emb_1 = offset_emb_1 + self.func(r_offset[rel_1])

            h_target = h[torch.tensor([subgraph.pi_target2], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.pi_rel2], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.pi_rel2], dtype=torch.long).squeeze()]
            r_query = r_center[rel_1]
            r_query_offset = r_offset[rel_1]
            refer_start = h[torch.tensor([subgraph.pi_target1], dtype=torch.long).squeeze()]
            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                             h_target, r_query, r_target, refer_start,
                                             r_query_offset, r_target_offset)

            subquery_2 = h[anchor_1] + r_center[rel_2]
            offset_emb_2 = self.func(r_offset[rel_2])

            h_target = h[torch.tensor([subgraph.pi_target3], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.pi_rel3], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.pi_rel3], dtype=torch.long).squeeze()]
            r_query = r_center[rel_2]
            r_query_offset = r_offset[rel_2]
            refer_start = h[anchor_1].unsqueeze(1)
            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                             h_target, r_query, r_target, refer_start,
                                             r_query_offset, r_target_offset)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2]))

        elif query_name_dict[query_type] == 'ip':
            anchor_0 = [q[0][0][0] for q in chosen_queries]
            anchor_1 = [q[0][1][0] for q in chosen_queries]

            rel_0 = [query[0][0][1][0] for query in chosen_queries]
            rel_1 = [query[0][1][1][0] for query in chosen_queries]
            rel_2 = [query[1][0] for query in chosen_queries]

            subquery_1 = h[anchor_0] + r_center[rel_0]
            subquery_2 = h[anchor_1] + r_center[rel_2]
            offset_emb_1 = self.func(r_offset[rel_0])
            offset_emb_2 = self.func(r_offset[rel_2])

            h_target = h[torch.tensor([subgraph.ip_target1], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.ip_rel1], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.ip_rel1], dtype=torch.long).squeeze()]
            r_query = r_center[rel_0]
            r_query_offset = r_offset[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)
            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.ip_target2], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.ip_rel2], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.ip_rel2], dtype=torch.long).squeeze()]
            r_query = r_center[rel_1]
            r_query_offset = r_offset[rel_1]
            refer_start = h[anchor_1].unsqueeze(1)
            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2]))

            query_emb = query_emb + r_center[rel_2]
            offset_emb = offset_emb + self.func(r_offset[rel_2])

            h_target = h[torch.tensor([subgraph.ip_target3], dtype=torch.long).squeeze()]
            r_target = r_center[torch.tensor([subgraph.ip_rel3], dtype=torch.long).squeeze()]
            r_target_offset = r_offset[torch.tensor([subgraph.ip_rel3], dtype=torch.long).squeeze()]
            r_query = r_center[rel_2]
            r_query_offset = r_offset[rel_2]
            refer_start = h[torch.tensor([subgraph.ip_target1], dtype=torch.long).squeeze()]
            query_emb, offset_emb = self.mlp_projection(query_emb, offset_emb,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

        elif query_name_dict[query_type] == '2u-DNF':

            anchors = [[query[0][0] for query in chosen_queries], [query[1][0] for query in chosen_queries]]
            rels = [[query[0][1][0] for query in chosen_queries], [query[1][1][0] for query in chosen_queries]]

            subquery_1 = h[anchors[0]] + r_center[rels[0]]
            subquery_2 = h[anchors[1]] + r_center[rels[1]]
            offset_emb_1 = self.func(r_offset[rels[0]])
            offset_emb_2 = self.func(r_offset[rels[1]])

            h_target = h[torch.tensor([subgraph.u2_target], dtype=torch.long).squeeze()[:, 0]]
            r_target = r_center[torch.tensor([subgraph.u2_rel], dtype=torch.long).squeeze()[:, 0]]
            r_target_offset = r_offset[torch.tensor([subgraph.u2_rel], dtype=torch.long).squeeze()[:, 0]]
            r_query = r_center[rels[0]]
            r_query_offset = r_offset[rels[0]]
            refer_start = h[anchors[0]].unsqueeze(1)
            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.u2_target], dtype=torch.long).squeeze()[:, 1]]
            r_target = r_center[torch.tensor([subgraph.u2_rel], dtype=torch.long).squeeze()[:, 1]]
            r_target_offset = r_offset[torch.tensor([subgraph.u2_rel], dtype=torch.long).squeeze()[:, 1]]
            r_query = r_center[rels[1]]
            r_query_offset = r_offset[rels[1]]
            refer_start = h[anchors[1]].unsqueeze(1)
            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            query_emb = torch.cat([subquery_1.unsqueeze(1), subquery_2.unsqueeze(1)], dim=-2)

            offset_emb = torch.cat([offset_emb_1.unsqueeze(1), offset_emb_2.unsqueeze(1)], dim=-2)

        elif query_name_dict[query_type] == 'up-DNF':
            anchors = [[query[0][0][0] for query in chosen_queries],
                       [query[0][1][0] for query in chosen_queries]]

            rel_1 = [[query[0][0][1][0] for query in chosen_queries], [query[0][1][1][0] for query in chosen_queries]]
            rel_2 = [query[1][0] for query in chosen_queries]

            subquery_1 = h[anchors[0]] + r_center[rel_1[0]]
            subquery_2 = h[anchors[1]] + r_center[rel_1[1]]
            offset_emb_1 = self.func(r_offset[rel_1[0]])
            offset_emb_2 = self.func(r_offset[rel_1[1]])

            h_target = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 0]]
            r_target = r_center[torch.tensor([subgraph.up_rel_1], dtype=torch.long).squeeze()[:, 0]]
            r_target_offset = r_offset[torch.tensor([subgraph.up_rel_1], dtype=torch.long).squeeze()[:, 0]]
            r_query = r_center[rel_1[0]]
            r_query_offset = r_offset[rel_1[0]]
            refer_start = h[anchors[0]].unsqueeze(1)
            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 1]]
            r_target = r_center[torch.tensor([subgraph.up_rel_1], dtype=torch.long).squeeze()[:, 1]]
            r_target_offset = r_offset[torch.tensor([subgraph.up_rel_1], dtype=torch.long).squeeze()[:, 1]]
            r_query = r_center[rel_1[1]]
            r_query_offset = r_offset[rel_1[1]]
            refer_start = h[anchors[1]].unsqueeze(1)
            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            subquery_1 = subquery_1 + r_center[rel_2]
            subquery_2 = subquery_2 + r_center[rel_2]
            offset_emb_1 = offset_emb_1 + self.func(r_offset[rel_2])
            offset_emb_2 = offset_emb_2 + self.func(r_offset[rel_2])

            h_target = h[torch.tensor([subgraph.up_target_2], dtype=torch.long).squeeze()[:, 0]]
            r_target = r_center[torch.tensor([subgraph.up_rel_2], dtype=torch.long).squeeze()[:, 0]]
            r_target_offset = r_offset[torch.tensor([subgraph.up_rel_2], dtype=torch.long).squeeze()[:, 0]]
            r_query = r_center[rel_2]
            r_query_offset = r_offset[rel_2]
            refer_start = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 0]]
            subquery_1, offset_emb_1 = self.mlp_projection(subquery_1, offset_emb_1,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            h_target = h[torch.tensor([subgraph.up_target_2], dtype=torch.long).squeeze()[:, 1]]
            r_target = r_center[torch.tensor([subgraph.up_rel_2], dtype=torch.long).squeeze()[:, 1]]
            r_target_offset = r_offset[torch.tensor([subgraph.up_rel_2], dtype=torch.long).squeeze()[:, 1]]
            r_query = r_center[rel_2]
            r_query_offset = r_offset[rel_2]
            refer_start = h[torch.tensor([subgraph.up_target_1], dtype=torch.long).squeeze()[:, 1]]
            subquery_2, offset_emb_2 = self.mlp_projection(subquery_2, offset_emb_2,
                                                           h_target, r_query, r_target,
                                                           refer_start,
                                                           r_query_offset, r_target_offset)

            query_emb = torch.cat([subquery_1.unsqueeze(1),
                                   subquery_2.unsqueeze(1)], dim=-2)

            offset_emb = torch.cat([offset_emb_1.unsqueeze(1), offset_emb_2.unsqueeze(1)], dim=-2)

        if not evaluating:
            if 'u' in query_name_dict[query_type]:
                raise Exception("union query can not be trained!")
            target_embeds = self.entity_embedding[chosen_answers]
            delta = (target_embeds - query_emb).abs()
            distance_out = F.relu(delta - offset_emb)
            distance_in = torch.min(delta, offset_emb)
            scores = self.gamma - torch.norm(distance_out, p=1, dim=-1) - \
                     self.cen * torch.norm(distance_in, p=1, dim=-1)

            neg_embeds = self.entity_embedding[torch.tensor(neg_answers)]
            neg_delta = (neg_embeds - query_emb.unsqueeze(1)).abs()
            neg_distance_out = F.relu(neg_delta - offset_emb.unsqueeze(1))
            neg_distance_in = torch.min(neg_delta, offset_emb.unsqueeze(1))
            neg_scores = self.gamma - torch.norm(neg_distance_out, p=1, dim=-1) - \
                         self.cen * torch.norm(neg_distance_in, p=1, dim=-1)

            return scores, neg_scores

        else:
            if 'u' not in query_name_dict[query_type]:
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_embedding[:self.graph.n_entity]
                for i in range(batch_size):
                    delta = (target_embeds - query_emb[i].t().unsqueeze(0)).abs()
                    distance_out = F.relu(delta - offset_emb[i].t().unsqueeze(0))
                    distance_in = torch.min(delta, offset_emb[i].t().unsqueeze(0))
                    scores[i] = self.gamma - torch.norm(distance_out, p=1, dim=-1) - \
                                self.cen * torch.norm(distance_in, p=1, dim=-1)
            else :
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_embedding[:self.graph.n_entity]
                for i in range(batch_size):
                    delta = (target_embeds.unsqueeze(1) - query_emb[i].unsqueeze(0)).abs()
                    distance_out = F.relu(delta - offset_emb[i].unsqueeze(0))
                    distance_in = torch.min(delta, offset_emb[i].unsqueeze(0))
                    temp_score = self.gamma - torch.norm(distance_out, p=1, dim=-1) - \
                                 self.cen * torch.norm(distance_in, p=1, dim=-1)
                    scores[i] = torch.max(temp_score, dim=1)[0]
            return scores

    def margin_loss(self, chosen_queries, chosen_answers, neg_answers, query_type, sample_limit, subgraph, all_query_num,
                    margin=1, query_name_dict=None):
        affs, neg_affs = self.forward(
            chosen_queries, query_type, sample_limit, subgraph,
            chosen_answers, query_name_dict, neg_answers=neg_answers)

        negative_score = F.logsigmoid(-neg_affs).mean()
        positive_score = F.logsigmoid(affs).mean(dim=0)

        positive_sample_loss = - positive_score
        negative_sample_loss = - negative_score
        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss


