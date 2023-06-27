import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param


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


class DistMult(nn.Module):
    def __init__(self, graph, args):
        super(DistMult, self).__init__()

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

        self.entity_embedding = nn.Parameter(torch.zeros(graph.n_entity, self.embed_dim))
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

        self.center_net = CenterIntersection(self.embed_dim)

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):
        batch_size = len(chosen_queries)

        if query_name_dict[query_type] == '1p':
            query_emb = self.entity_embedding[[query[0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[1][0] for query in chosen_queries]]
        elif query_name_dict[query_type] == '2p':
            query_emb = self.entity_embedding[[query[0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[1][0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[1][1] for query in chosen_queries]]
        elif query_name_dict[query_type] == '3p':
            query_emb = self.entity_embedding[[query[0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[1][0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[1][1] for query in chosen_queries]] * \
                        self.relation_embedding[[query[1][2] for query in chosen_queries]]
        elif query_name_dict[query_type] == '2i':
            subquery_1 = self.entity_embedding[[query[0][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[0][1][0] for query in chosen_queries]]
            subquery_2 = self.entity_embedding[[query[1][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[1][1][0] for query in chosen_queries]]
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))
        elif query_name_dict[query_type] == '3i':
            subquery_1 = self.entity_embedding[[query[0][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[0][1][0] for query in chosen_queries]]
            subquery_2 = self.entity_embedding[[query[1][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[1][1][0] for query in chosen_queries]]
            subquery_3 = self.entity_embedding[[query[2][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[2][1][0] for query in chosen_queries]]
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2, subquery_3]))
        elif query_name_dict[query_type] == 'pi':
            subquery_1 = self.entity_embedding[[query[0][0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[0][1][0] for query in chosen_queries]] * \
                        self.relation_embedding[[query[0][1][1] for query in chosen_queries]]
            subquery_2 = self.entity_embedding[[query[1][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[1][1][0] for query in chosen_queries]]
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))
        elif query_name_dict[query_type] == 'ip':
            subquery_1 = self.entity_embedding[[query[0][0][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[0][0][1][0] for query in chosen_queries]]
            subquery_2 = self.entity_embedding[[query[0][1][0] for query in chosen_queries]] * \
                         self.relation_embedding[[query[0][1][1][0] for query in chosen_queries]]
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))
            # 需要重跑
            query_emb = query_emb * self.relation_embedding[[query[1][0] for query in chosen_queries]]

        elif query_name_dict[query_type] == '2u-DNF':
            center_emb_1 = self.entity_embedding[[query[0][0] for query in chosen_queries]] * \
                           self.relation_embedding[[query[0][1][0] for query in chosen_queries]]
            center_emb_2 = self.entity_embedding[[query[1][0] for query in chosen_queries]] * \
                           self.relation_embedding[[query[1][1][0] for query in chosen_queries]]

            query_emb = torch.cat([center_emb_1.unsqueeze(1),
                                    center_emb_2.unsqueeze(1)], dim=-2)

        elif query_name_dict[query_type] == 'up-DNF':
            center = self.relation_embedding[[query[1][0] for query in chosen_queries]]

            center_emb_1 = self.entity_embedding[[query[0][0][0] for query in chosen_queries]] * \
                           self.relation_embedding[[query[0][0][1][0] for query in chosen_queries]] * center
            center_emb_2 = self.entity_embedding[[query[0][1][0] for query in chosen_queries]] * \
                           self.relation_embedding[[query[0][1][1][0] for query in chosen_queries]] * center

            query_emb = torch.cat([center_emb_1.unsqueeze(1),
                                    center_emb_2.unsqueeze(1)], dim=-2)

        if not evaluating:
            target_embeds = self.entity_embedding[chosen_answers]
            distance = query_emb * target_embeds
            scores = torch.sum(distance, dim=-1)

            neg_embeds = self.entity_embedding[torch.tensor(neg_answers)]
            neg_distance = query_emb.unsqueeze(1) * neg_embeds
            neg_scores = torch.sum(neg_distance, dim=-1)

            return scores, neg_scores

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

