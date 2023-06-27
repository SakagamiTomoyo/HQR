import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param


def Identity(x):
    return x


class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class ProjectionAggregating(nn.Module):
    def __init__(self, dim, projection_net, projection_regularizer, aggr='mlp'):
        super(ProjectionAggregating, self).__init__()

        self.dim = dim
        self.aggr = aggr
        hidden_dim, num_layers = 1600, 2
        self.projection_regular = projection_regularizer
        self.projection_net = projection_net
        self.bias_regularizer = Regularizer(1, 0.05, 1e9)

        self.attn_layer_1 = nn.Linear(self.dim * 2, self.dim * 2)
        self.attn_layer_2 = nn.Linear(self.dim * 2, self.dim * 2)

        nn.init.xavier_uniform_(self.attn_layer_1.weight)
        nn.init.xavier_uniform_(self.attn_layer_2.weight)

        self.transform_layer_1 = nn.Linear(self.dim * 2, self.dim * 2, bias=True)
        self.transform_layer_2 = nn.Linear(self.dim * 2, self.dim * 2, bias=True)

        nn.init.xavier_uniform_(self.transform_layer_1.weight)
        nn.init.xavier_uniform_(self.transform_layer_2.weight)

    def forward(self, query_emb, refer_embs, query_r=None, refer_r=None, start_embs=None):
        if start_embs.shape[1] == 1:
            start_embs = start_embs.repeat(1, refer_embs.shape[1], 1)

        bias = refer_embs - self.projection_net(start_embs, refer_r)

        layer_act_1 = F.relu(self.attn_layer_1(
            torch.cat([query_r.unsqueeze(1).repeat(1, refer_r.shape[1], 1), refer_r], dim=-1)
        ))

        final_refer_emb = torch.mean(self.attn_layer_2(layer_act_1) * bias, dim=-2)

        final_refer_emb = self.transform_layer_2(F.relu(self.transform_layer_1(final_refer_emb)))
        ref_norm = final_refer_emb.norm(p=1, dim=-1).unsqueeze(1)
        query_norm = query_emb.norm(p=1, dim=-1).unsqueeze(1)
        const = 5

        final_refer_emb = final_refer_emb / (1e-9 + ref_norm / query_norm * const)

        final_query_emb = query_emb + final_refer_emb

        final_query_emb = self.bias_regularizer(final_query_emb)

        return final_query_emb


class IDN_BetaE(nn.Module):
    def __init__(self, graph, args):
        super(IDN_BetaE, self).__init__()

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

        self.entity_embedding = nn.Parameter(torch.zeros(graph.n_entity, self.embed_dim * 2))  # alpha and beta
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)  # make sure the parameters of beta embeddings are positive
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)  # make sure the parameters of beta embeddings after relation projection are positive
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(graph.n_relation, self.embed_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        hidden_dim, num_layers = 1600, 2
        self.center_net = BetaIntersection(self.embed_dim)
        self.projection_net = BetaProjection(self.embed_dim * 2,
                                             self.embed_dim,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)
        self.mlp_projection = ProjectionAggregating(self.embed_dim, self.projection_net, self.projection_regularizer, aggr='bias')

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):
        batch_size = len(chosen_queries)

        h = self.entity_embedding
        r = self.relation_embedding

        # 可以设置成参数
        h = torch.cat([h, torch.tensor([[0] * self.embed_dim * 2], requires_grad=False).cuda()])
        r = torch.cat([r, torch.tensor([[0] * self.embed_dim], requires_grad=False).cuda()])
        h = self.entity_regularizer(h)

        if query_name_dict[query_type] == '1p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            query_emb = self.projection_net(h[anchors], r[rel_0])

            h_target = h[torch.tensor([subgraph.p1_target], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p1_rel], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            alpha_embedding, beta_embedding = torch.chunk(query_emb, 2, dim=-1)

        elif query_name_dict[query_type] == '2p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            rel_1 = [query[1][1] for query in chosen_queries]
            query_emb = self.projection_net(h[anchors], r[rel_0])

            h_target = h[torch.tensor([subgraph.p2_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p2_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            query_emb = self.projection_net(query_emb, r[rel_1])

            h_target = h[torch.tensor([subgraph.p2_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p2_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[torch.tensor([subgraph.p2_target1], dtype=torch.long).squeeze()]

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            alpha_embedding, beta_embedding = torch.chunk(query_emb, 2, dim=-1)

        elif query_name_dict[query_type] == '3p':
            anchors = [q[0] for q in chosen_queries]
            rel_0 = [query[1][0] for query in chosen_queries]
            rel_1 = [query[1][1] for query in chosen_queries]
            rel_2 = [query[1][2] for query in chosen_queries]
            query_emb = self.projection_net(h[anchors], r[rel_0])

            h_target = h[torch.tensor([subgraph.p3_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p3_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchors].unsqueeze(1)

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            query_emb = self.projection_net(query_emb, r[rel_1])

            h_target = h[torch.tensor([subgraph.p3_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p3_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[torch.tensor([subgraph.p3_target1], dtype=torch.long).squeeze()]

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            query_emb = self.projection_net(query_emb, r[rel_2])

            h_target = h[torch.tensor([subgraph.p3_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.p3_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[torch.tensor([subgraph.p3_target2], dtype=torch.long).squeeze()]

            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)
            alpha_embedding, beta_embedding = torch.chunk(query_emb, 2, dim=-1)

        elif query_name_dict[query_type] == '2i':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[1][1][0] for query in chosen_queries]

            subquery_1 = self.projection_net(h[anchor_0], r[rel_0])
            subquery_2 = self.projection_net(h[anchor_1], r[rel_1])

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
            alpha_embedding_1, beta_embedding_1 = torch.chunk(subquery_1, 2, dim=-1)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(subquery_2, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2]))

        elif query_name_dict[query_type] == '3i':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]
            anchor_2 = [q[2][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[1][1][0] for query in chosen_queries]
            rel_2 = [query[2][1][0] for query in chosen_queries]

            subquery_1 = self.projection_net(h[anchor_0], r[rel_0])
            subquery_2 = self.projection_net(h[anchor_1], r[rel_1])
            subquery_3 = self.projection_net(h[anchor_2], r[rel_2])

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

            alpha_embedding_1, beta_embedding_1 = torch.chunk(subquery_1, 2, dim=-1)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(subquery_2, 2, dim=-1)
            alpha_embedding_3, beta_embedding_3 = torch.chunk(subquery_3, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2, alpha_embedding_3]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2, beta_embedding_3]))

        elif query_name_dict[query_type] == 'pi':
            anchor_0 = [q[0][0] for q in chosen_queries]
            anchor_1 = [q[1][0] for q in chosen_queries]

            rel_0 = [query[0][1][0] for query in chosen_queries]
            rel_1 = [query[0][1][1] for query in chosen_queries]
            rel_2 = [query[1][1][0] for query in chosen_queries]

            subquery_1 = self.projection_net(h[anchor_0], r[rel_0])

            h_target = h[torch.tensor([subgraph.pi_target1], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.pi_rel1], dtype=torch.long).squeeze()]
            r_query = r[rel_0]
            refer_start = h[anchor_0].unsqueeze(1)
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            subquery_1 = self.projection_net(subquery_1, r[rel_1])

            h_target = h[torch.tensor([subgraph.pi_target2], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.pi_rel2], dtype=torch.long).squeeze()]
            r_query = r[rel_1]
            refer_start = h[torch.tensor([subgraph.pi_target1], dtype=torch.long).squeeze()]
            subquery_1 = self.mlp_projection(subquery_1, h_target, r_query, r_target, refer_start)

            subquery_2 = self.projection_net(h[anchor_1], r[rel_2])

            h_target = h[torch.tensor([subgraph.pi_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.pi_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[anchor_1].unsqueeze(1)
            subquery_2 = self.mlp_projection(subquery_2, h_target, r_query, r_target, refer_start)

            alpha_embedding_1, beta_embedding_1 = torch.chunk(subquery_1, 2, dim=-1)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(subquery_2, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2]))

        elif query_name_dict[query_type] == 'ip':
            anchor_0 = [q[0][0][0] for q in chosen_queries]
            anchor_1 = [q[0][1][0] for q in chosen_queries]

            rel_0 = [query[0][0][1][0] for query in chosen_queries]
            rel_1 = [query[0][1][1][0] for query in chosen_queries]
            rel_2 = [query[1][0] for query in chosen_queries]

            subquery_1 = self.projection_net(h[anchor_0], r[rel_0])
            subquery_2 = self.projection_net(h[anchor_1], r[rel_1])

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

            alpha_embedding_1, beta_embedding_1 = torch.chunk(subquery_1, 2, dim=-1)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(subquery_2, 2, dim=-1)
            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2]))
            query_emb = torch.cat([alpha_embedding, beta_embedding], dim=-1)

            query_emb = self.projection_net(query_emb, r[rel_2])

            h_target = h[torch.tensor([subgraph.ip_target3], dtype=torch.long).squeeze()]
            r_target = r[torch.tensor([subgraph.ip_rel3], dtype=torch.long).squeeze()]
            r_query = r[rel_2]
            refer_start = h[torch.tensor([subgraph.ip_target1], dtype=torch.long).squeeze()]
            query_emb = self.mlp_projection(query_emb, h_target, r_query, r_target, refer_start)

            alpha_embedding, beta_embedding = torch.chunk(query_emb, 2, dim=-1)

        elif query_name_dict[query_type] == '2u-DNF':
            anchors = [[query[0][0] for query in chosen_queries], [query[1][0] for query in chosen_queries]]
            rels = [[query[0][1][0] for query in chosen_queries], [query[1][1][0] for query in chosen_queries]]

            subquery_1 = self.projection_net(h[anchors[0]], r[rels[0]])
            subquery_2 = self.projection_net(h[anchors[1]], r[rels[1]])

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

            alpha_embedding_1, beta_embedding_1 = torch.chunk(subquery_1, 2, dim=-1)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(subquery_2, 2, dim=-1)

        elif query_name_dict[query_type] == 'up-DNF':
            anchors = [[query[0][0][0] for query in chosen_queries],
                       [query[0][1][0] for query in chosen_queries]]

            rel_1 = [[query[0][0][1][0] for query in chosen_queries], [query[0][1][1][0] for query in chosen_queries]]
            rel_2 = [query[1][0] for query in chosen_queries]

            subquery_1 = self.projection_net(h[anchors[0]], r[rel_1[0]])
            subquery_2 = self.projection_net(h[anchors[1]], r[rel_1[1]])

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

            subquery_1 = self.projection_net(subquery_1, r[rel_2])
            subquery_2 = self.projection_net(subquery_2, r[rel_2])

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

            alpha_embedding_1, beta_embedding_1 = torch.chunk(subquery_1, 2, dim=-1)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(subquery_2, 2, dim=-1)

        if not evaluating:
            if 'u' in query_name_dict[query_type]:
                raise Exception("union query can not be trained!")

            alpha_embeddings = torch.cat([alpha_embedding], dim=0).unsqueeze(1)
            beta_embeddings = torch.cat([beta_embedding], dim=0).unsqueeze(1)
            dists = torch.distributions.beta.Beta(alpha_embeddings, beta_embeddings)

            target_embeds = self.entity_regularizer(self.entity_embedding[chosen_answers]).unsqueeze(1)
            alpha_embeds, beta_embeds = torch.chunk(target_embeds, 2, dim=-1)
            entity_dist = torch.distributions.beta.Beta(alpha_embeds, beta_embeds)
            scores = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, dists), p=1, dim=-1)

            neg_embeds = self.entity_regularizer(self.entity_embedding[torch.tensor(neg_answers)])
            neg_alpha_embeds, neg_beta_embeds = torch.chunk(neg_embeds, 2, dim=-1)
            neg_entity_dist = torch.distributions.beta.Beta(neg_alpha_embeds, neg_beta_embeds)
            neg_scores = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(neg_entity_dist, dists), p=1, dim=-1)

            return scores, neg_scores

        else:
            if 'u' in query_name_dict[query_type]:
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_regularizer(self.entity_embedding[:self.graph.n_entity]).unsqueeze(0).unsqueeze(0)
                for i in range(batch_size):
                    alpha_embeddings = torch.stack([alpha_embedding_1[i], alpha_embedding_2[i]]).unsqueeze(0).unsqueeze(2)
                    beta_embeddings = torch.stack([beta_embedding_1[i], beta_embedding_2[i]]).unsqueeze(0).unsqueeze(2)
                    dists = torch.distributions.beta.Beta(alpha_embeddings, beta_embeddings)

                    alpha_embeds, beta_embeds = torch.chunk(target_embeds, 2, dim=-1)
                    entity_dist = torch.distributions.beta.Beta(alpha_embeds, beta_embeds)
                    temp = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, dists), p=1,
                                                     dim=-1)
                    scores[i] = torch.max(temp, dim=1)[0]

            else:
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_regularizer(self.entity_embedding[:self.graph.n_entity]).unsqueeze(0)
                for i in range(batch_size):
                    alpha_embeddings = torch.cat([alpha_embedding[i]], dim=0).unsqueeze(0).unsqueeze(0)
                    beta_embeddings = torch.cat([beta_embedding[i]], dim=0).unsqueeze(0).unsqueeze(0)
                    dists = torch.distributions.beta.Beta(alpha_embeddings, beta_embeddings)

                    alpha_embeds, beta_embeds = torch.chunk(target_embeds, 2, dim=-1)
                    entity_dist = torch.distributions.beta.Beta(alpha_embeds, beta_embeds)
                    scores[i] = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, dists), p=1,
                                                     dim=-1)
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
