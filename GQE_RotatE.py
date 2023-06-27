import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param

pi = 3.14159265358979323846


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


class GQE_RotatE(nn.Module):
    def __init__(self, graph, args):
        super(GQE_RotatE, self).__init__()

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

        self.entity_embedding = nn.Parameter(torch.zeros(graph.n_entity, self.embed_dim * 2))
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

        self.center_net = CenterIntersection(self.embed_dim * 2)

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):
        batch_size = len(chosen_queries)

        phase_relation = self.relation_embedding / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if query_name_dict[query_type] == '1p':
            anc_re, anc_im = torch.chunk(self.entity_embedding[[query[0] for query in chosen_queries]], 2, dim=-1)
            rel_1_re, rel_1_im = re_relation[[query[1][0] for query in chosen_queries]], im_relation[
                [query[1][0] for query in chosen_queries]]
            query_emb = torch.cat([anc_re * rel_1_re - anc_im * rel_1_im,
                                   anc_re * rel_1_im + anc_im * rel_1_re], dim=-1)

        elif query_name_dict[query_type] == '2p':
            anc_re, anc_im = torch.chunk(self.entity_embedding[[query[0] for query in chosen_queries]], 2, dim=-1)
            rel_1_re, rel_1_im = re_relation[[query[1][0] for query in chosen_queries]], im_relation[
                [query[1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[1][1] for query in chosen_queries]], im_relation[
                [query[1][1] for query in chosen_queries]]

            var_1_re = anc_re * rel_1_re - anc_im * rel_1_im
            var_1_im = anc_re * rel_1_im + anc_im * rel_1_re

            var_2_re = var_1_re * rel_2_re - var_1_im * rel_2_im
            var_2_im = var_1_re * rel_2_im + var_1_im * rel_2_re

            query_emb = torch.cat([var_2_re,
                                   var_2_im], dim=-1)

        elif query_name_dict[query_type] == '3p':
            anc_re, anc_im = torch.chunk(self.entity_embedding[[query[0] for query in chosen_queries]], 2, dim=-1)
            rel_1_re, rel_1_im = re_relation[[query[1][0] for query in chosen_queries]], im_relation[
                [query[1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[1][1] for query in chosen_queries]], im_relation[
                [query[1][1] for query in chosen_queries]]
            rel_3_re, rel_3_im = re_relation[[query[1][2] for query in chosen_queries]], im_relation[
                [query[1][2] for query in chosen_queries]]

            var_1_re = anc_re * rel_1_re - anc_im * rel_1_im
            var_1_im = anc_re * rel_1_im + anc_im * rel_1_re

            var_2_re = var_1_re * rel_2_re - var_1_im * rel_2_im
            var_2_im = var_1_re * rel_2_im + var_1_im * rel_2_re

            var_3_re = var_2_re * rel_3_re - var_2_im * rel_3_im
            var_3_im = var_2_re * rel_3_im + var_2_im * rel_3_re

            query_emb = torch.cat([var_3_re,
                                   var_3_im], dim=-1)

        elif query_name_dict[query_type] == '2i':
            anc_1_re, anc_1_im = torch.chunk(self.entity_embedding[[query[0][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_2_re, anc_2_im = torch.chunk(self.entity_embedding[[query[1][0] for query in chosen_queries]], 2,
                                             dim=-1)
            rel_1_re, rel_1_im = re_relation[[query[0][1][0] for query in chosen_queries]], im_relation[
                [query[0][1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[1][1][0] for query in chosen_queries]], im_relation[
                [query[1][1][0] for query in chosen_queries]]

            subquery_1 = torch.cat([anc_1_re * rel_1_re - anc_1_im * rel_1_im,
                                    anc_1_re * rel_1_im + anc_1_im * rel_1_re], dim=-1)
            subquery_2 = torch.cat([anc_2_re * rel_2_re - anc_2_im * rel_2_im,
                                    anc_2_re * rel_2_im + anc_2_im * rel_2_re], dim=-1)
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))

        elif query_name_dict[query_type] == '3i':
            anc_1_re, anc_1_im = torch.chunk(self.entity_embedding[[query[0][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_2_re, anc_2_im = torch.chunk(self.entity_embedding[[query[1][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_3_re, anc_3_im = torch.chunk(self.entity_embedding[[query[2][0] for query in chosen_queries]], 2,
                                             dim=-1)
            rel_1_re, rel_1_im = re_relation[[query[0][1][0] for query in chosen_queries]], im_relation[
                [query[0][1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[1][1][0] for query in chosen_queries]], im_relation[
                [query[1][1][0] for query in chosen_queries]]
            rel_3_re, rel_3_im = re_relation[[query[2][1][0] for query in chosen_queries]], im_relation[
                [query[2][1][0] for query in chosen_queries]]

            subquery_1 = torch.cat([anc_1_re * rel_1_re - anc_1_im * rel_1_im,
                                    anc_1_re * rel_1_im + anc_1_im * rel_1_re], dim=-1)
            subquery_2 = torch.cat([anc_2_re * rel_2_re - anc_2_im * rel_2_im,
                                    anc_2_re * rel_2_im + anc_2_im * rel_2_re], dim=-1)
            subquery_3 = torch.cat([anc_3_re * rel_3_re - anc_3_im * rel_3_im,
                                    anc_3_re * rel_3_im + anc_3_im * rel_3_re], dim=-1)
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2, subquery_3]))

        elif query_name_dict[query_type] == 'pi':
            anc_1_re, anc_1_im = torch.chunk(self.entity_embedding[[query[0][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_2_re, anc_2_im = torch.chunk(self.entity_embedding[[query[1][0] for query in chosen_queries]], 2,
                                             dim=-1)

            rel_1_re, rel_1_im = re_relation[[query[0][1][0] for query in chosen_queries]], im_relation[
                [query[0][1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[0][1][1] for query in chosen_queries]], im_relation[
                [query[0][1][1] for query in chosen_queries]]

            rel_3_re, rel_3_im = re_relation[[query[1][1][0] for query in chosen_queries]], im_relation[
                [query[1][1][0] for query in chosen_queries]]

            var_1_re = anc_1_re * rel_1_re - anc_1_im * rel_1_im
            var_1_im = anc_1_re * rel_1_im + anc_1_im * rel_1_re

            var_2_re = var_1_re * rel_2_re - var_1_im * rel_2_im
            var_2_im = var_1_re * rel_2_im + var_1_im * rel_2_re

            var_3_re = anc_2_re * rel_3_re - anc_2_im * rel_3_im
            var_3_im = anc_2_re * rel_3_im + anc_2_im * rel_3_re

            subquery_1 = torch.cat([var_2_re,
                                    var_2_im], dim=-1)
            subquery_2 = torch.cat([var_3_re,
                                    var_3_im], dim=-1)
            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))

        elif query_name_dict[query_type] == 'ip':
            anc_1_re, anc_1_im = torch.chunk(self.entity_embedding[[query[0][0][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_2_re, anc_2_im = torch.chunk(self.entity_embedding[[query[0][1][0] for query in chosen_queries]], 2,
                                             dim=-1)

            rel_1_re, rel_1_im = re_relation[[query[0][0][1][0] for query in chosen_queries]], im_relation[
                [query[0][0][1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[0][1][1][0] for query in chosen_queries]], im_relation[
                [query[0][1][1][0] for query in chosen_queries]]

            rel_3_re, rel_3_im = re_relation[[query[1][0] for query in chosen_queries]], im_relation[
                [query[1][0] for query in chosen_queries]]

            subquery_1 = torch.cat([anc_1_re * rel_1_re - anc_1_im * rel_1_im,
                                    anc_1_re * rel_1_im + anc_1_im * rel_1_re], dim=-1)
            subquery_2 = torch.cat([anc_2_re * rel_2_re - anc_2_im * rel_2_im,
                                    anc_2_re * rel_2_im + anc_2_im * rel_2_re], dim=-1)

            query_emb = self.center_net(torch.stack([subquery_1, subquery_2]))

            var_re, var_im = torch.chunk(query_emb, 2, dim=-1)

            var_3_re = var_re * rel_3_re - var_im * rel_3_im
            var_3_im = var_re * rel_3_im + var_im * rel_3_re

            query_emb = torch.cat([var_3_re, var_3_im], dim=-1)

        elif query_name_dict[query_type] == '2u-DNF':
            anc_1_re, anc_1_im = torch.chunk(self.entity_embedding[[query[0][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_2_re, anc_2_im = torch.chunk(self.entity_embedding[[query[1][0] for query in chosen_queries]], 2,
                                             dim=-1)

            rel_1_re, rel_1_im = re_relation[[query[0][1][0] for query in chosen_queries]], im_relation[
                [query[0][1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[1][1][0] for query in chosen_queries]], im_relation[
                [query[1][1][0] for query in chosen_queries]]

            subquery_1 = torch.cat([anc_1_re * rel_1_re - anc_1_im * rel_1_im,
                                    anc_1_re * rel_1_im + anc_1_im * rel_1_re], dim=-1)
            subquery_2 = torch.cat([anc_2_re * rel_2_re - anc_2_im * rel_2_im,
                                    anc_2_re * rel_2_im + anc_2_im * rel_2_re], dim=-1)

            query_emb = torch.cat([subquery_1.unsqueeze(1),
                                   subquery_2.unsqueeze(1)], dim=-2)

        elif query_name_dict[query_type] == 'up-DNF':
            anc_1_re, anc_1_im = torch.chunk(self.entity_embedding[[query[0][0][0] for query in chosen_queries]], 2,
                                             dim=-1)
            anc_2_re, anc_2_im = torch.chunk(self.entity_embedding[[query[0][1][0] for query in chosen_queries]], 2,
                                             dim=-1)

            rel_1_re, rel_1_im = re_relation[[query[0][0][1][0] for query in chosen_queries]], im_relation[
                [query[0][0][1][0] for query in chosen_queries]]
            rel_2_re, rel_2_im = re_relation[[query[0][1][1][0] for query in chosen_queries]], im_relation[
                [query[0][1][1][0] for query in chosen_queries]]

            subquery_1_re, subquery_1_im = anc_1_re * rel_1_re - anc_1_im * rel_1_im, anc_1_re * rel_1_im + anc_1_im * rel_1_re
            subquery_2_re, subquery_2_im = anc_2_re * rel_2_re - anc_2_im * rel_2_im, anc_2_re * rel_2_im + anc_2_im * rel_2_re

            center_re, center_im = re_relation[[query[1][0] for query in chosen_queries]], im_relation[[query[1][0] for query in chosen_queries]]

            center_emb_1 = torch.cat([subquery_1_re * center_re - subquery_1_im * center_im,
                                      subquery_1_re * center_im + subquery_1_im * center_re], dim=-1)
            center_emb_2 = torch.cat([subquery_2_re * center_re - subquery_2_im * center_im,
                                      subquery_2_re * center_im + subquery_2_im * center_re], dim=-1)

            query_emb = torch.cat([center_emb_1.unsqueeze(1),
                                    center_emb_2.unsqueeze(1)], dim=-2)

        if not evaluating:
            target_embeds = self.entity_embedding[chosen_answers]
            target_re, target_im = torch.chunk(target_embeds, 2, dim=-1)
            query_re, query_im = torch.chunk(query_emb, 2, dim=-1)
            distance = [-target_re + query_re, -target_im + query_im]
            scores = self.gamma - torch.sum(torch.stack(distance, dim=0).norm(dim=0), dim=-1)

            neg_embeds = self.entity_embedding[torch.tensor(neg_answers)]
            neg_re, neg_im = torch.chunk(neg_embeds, 2, dim=-1)
            neg_distance = [-neg_re + query_re.unsqueeze(1), -neg_im + query_im.unsqueeze(1)]
            neg_scores = self.gamma - torch.sum(torch.stack(neg_distance, dim=0).norm(dim=0), dim=-1)

            return scores, neg_scores

        else:
            if 'u' in query_name_dict[query_type]:
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_embedding[:self.graph.n_entity]
                target_re, target_im = torch.chunk(target_embeds, 2, dim=-1)
                query_re, query_im = torch.chunk(query_emb, 2, dim=-1)
                for i in range(batch_size):
                    distance = [-target_re.unsqueeze(1) + query_re[i], -target_im.unsqueeze(1) + query_im[i]]
                    scores[i] = torch.max(self.gamma - torch.sum(torch.stack(distance).norm(dim=0), dim=-1), dim=-1)[0]
                return scores

            scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
            target_embeds = self.entity_embedding[:self.graph.n_entity]
            target_re, target_im = torch.chunk(target_embeds, 2, dim=-1)
            query_re, query_im = torch.chunk(query_emb, 2, dim=-1)
            for i in range(batch_size):
                distance = [-target_re + query_re[i].unsqueeze(0), -target_im + query_im[i].unsqueeze(0)]
                scores[i] = self.gamma - torch.sum(torch.stack(distance).norm(dim=0), dim=-1)
            return scores

    def margin_loss(self, chosen_queries, chosen_answers, neg_answers, query_type, sample_limit, subgraph,
                    all_query_num,
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

