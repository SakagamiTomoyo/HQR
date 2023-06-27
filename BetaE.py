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


class BetaE(nn.Module):
    def __init__(self, graph, args):
        super(BetaE, self).__init__()

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

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):
        batch_size = len(chosen_queries)

        h = self.entity_regularizer(self.entity_embedding)
        r = self.relation_embedding

        if query_name_dict[query_type] == '1p':
            entity_emb = h[[query[0] for query in chosen_queries]]
            relation_emb = r[[query[1][0] for query in chosen_queries]]
            embedding = self.projection_net(entity_emb, relation_emb)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        elif query_name_dict[query_type] == '2p':
            entity_emb = h[[query[0] for query in chosen_queries]]
            relation_emb_1 = r[[query[1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(entity_emb, relation_emb_1)
            relation_emb = r[[query[1][1] for query in chosen_queries]]
            embedding = self.projection_net(embedding_1, relation_emb)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        elif query_name_dict[query_type] == '3p':
            entity_emb = h[[query[0] for query in chosen_queries]]
            relation_emb_1 = r[[query[1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(entity_emb, relation_emb_1)
            relation_emb_2 = r[[query[1][1] for query in chosen_queries]]
            embedding_2 = self.projection_net(embedding_1, relation_emb_2)
            relation_emb = r[[query[1][2] for query in chosen_queries]]
            embedding = self.projection_net(embedding_2, relation_emb)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        elif query_name_dict[query_type] == '2i':
            anchor_1 = h[[query[0][0] for query in chosen_queries]]
            relation_emb_1 = r[[query[0][1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(anchor_1, relation_emb_1)
            alpha_embedding_1, beta_embedding_1 = torch.chunk(embedding_1, 2, dim=-1)

            anchor_2 = h[[query[1][0] for query in chosen_queries]]
            relation_emb_2 = r[[query[1][1][0] for query in chosen_queries]]
            embedding_2 = self.projection_net(anchor_2, relation_emb_2)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2]))

        elif query_name_dict[query_type] == '3i':
            anchor_1 = h[[query[0][0] for query in chosen_queries]]
            relation_emb_1 = r[[query[0][1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(anchor_1, relation_emb_1)
            alpha_embedding_1, beta_embedding_1 = torch.chunk(embedding_1, 2, dim=-1)

            anchor_2 = h[[query[1][0] for query in chosen_queries]]
            relation_emb_2 = r[[query[1][1][0] for query in chosen_queries]]
            embedding_2 = self.projection_net(anchor_2, relation_emb_2)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)

            anchor_3 = h[[query[2][0] for query in chosen_queries]]
            relation_emb_3 = r[[query[2][1][0] for query in chosen_queries]]
            embedding_3 = self.projection_net(anchor_3, relation_emb_3)
            alpha_embedding_3, beta_embedding_3 = torch.chunk(embedding_3, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2, alpha_embedding_3]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2, beta_embedding_3]))

        elif query_name_dict[query_type] == 'pi':
            anchor_1 = h[[query[0][0] for query in chosen_queries]]
            relation_emb_1 = r[[query[0][1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(anchor_1, relation_emb_1)
            relation_emb = r[[query[0][1][1] for query in chosen_queries]]
            embedding = self.projection_net(embedding_1, relation_emb)
            alpha_embedding_1, beta_embedding_1 = torch.chunk(embedding, 2, dim=-1)

            anchor_2 = h[[query[1][0] for query in chosen_queries]]
            relation_emb_2 = r[[query[1][1][0] for query in chosen_queries]]
            embedding_2 = self.projection_net(anchor_2, relation_emb_2)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2]))

        elif query_name_dict[query_type] == 'ip':
            anchor_1 = h[[query[0][0][0] for query in chosen_queries]]
            relation_emb_1 = r[[query[0][0][1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(anchor_1, relation_emb_1)
            alpha_embedding_1, beta_embedding_1 = torch.chunk(embedding_1, 2, dim=-1)

            anchor_2 = h[[query[0][1][0] for query in chosen_queries]]
            relation_emb_2 = r[[query[0][1][1][0] for query in chosen_queries]]
            embedding_2 = self.projection_net(anchor_2, relation_emb_2)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)

            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1, alpha_embedding_2]),
                                                              torch.stack([beta_embedding_1, beta_embedding_2]))

            embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            relation_emb = r[[query[1][0] for query in chosen_queries]]
            embedding = self.projection_net(embedding, relation_emb)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        elif query_name_dict[query_type] == '2u-DNF':
            anchor_1 = h[[query[0][0] for query in chosen_queries]]
            relation_emb_1 = r[[query[0][1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(anchor_1, relation_emb_1)
            alpha_embedding_1, beta_embedding_1 = torch.chunk(embedding_1, 2, dim=-1)

            anchor_2 = h[[query[1][0] for query in chosen_queries]]
            relation_emb_2 = r[[query[1][1][0] for query in chosen_queries]]
            embedding_2 = self.projection_net(anchor_2, relation_emb_2)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)

        elif query_name_dict[query_type] == 'up-DNF':
            relation_u = r[[query[1][0] for query in chosen_queries]]

            anchor_1 = h[[query[0][0][0] for query in chosen_queries]]
            relation_emb_1 = r[[query[0][0][1][0] for query in chosen_queries]]
            embedding_1 = self.projection_net(anchor_1, relation_emb_1)
            embedding_1 = self.projection_net(embedding_1, relation_u)
            alpha_embedding_1, beta_embedding_1 = torch.chunk(embedding_1, 2, dim=-1)

            anchor_2 = h[[query[0][1][0] for query in chosen_queries]]
            relation_emb_2 = r[[query[0][1][1][0] for query in chosen_queries]]
            embedding_2 = self.projection_net(anchor_2, relation_emb_2)
            embedding_2 = self.projection_net(embedding_2, relation_u)
            alpha_embedding_2, beta_embedding_2 = torch.chunk(embedding_2, 2, dim=-1)

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
