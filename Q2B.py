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


class Q2B(nn.Module):
    def __init__(self, graph, args):
        super(Q2B, self).__init__()

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

        self.relation_embedding = nn.Parameter(torch.zeros(graph.n_relation, self.embed_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.offset_embedding = nn.Parameter(torch.zeros(graph.n_relation, self.embed_dim))
        nn.init.uniform_(
            tensor=self.offset_embedding,
            a=0.,
            b=self.embedding_range.item()
        )
        self.center_net = CenterIntersection(self.embed_dim)
        self.offset_net = BoxOffsetIntersection(self.embed_dim)

    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):
        batch_size = len(chosen_queries)

        h = self.entity_embedding
        r_center = self.relation_embedding
        r_offset = self.offset_embedding

        if query_name_dict[query_type] == '1p':
            center_emb = h[[query[0] for query in chosen_queries]] + r_center[[query[1][0] for query in chosen_queries]]
            offset_emb = self.func(r_offset[[query[1][0] for query in chosen_queries]])

        elif query_name_dict[query_type] == '2p':
            center_emb = h[[query[0] for query in chosen_queries]] + \
                         r_center[[query[1][0] for query in chosen_queries]] + \
                         r_center[[query[1][1] for query in chosen_queries]]
            offset_emb = self.func(r_offset[[query[1][0] for query in chosen_queries]]) + \
                         self.func(r_offset[[query[1][1] for query in chosen_queries]])

        elif query_name_dict[query_type] == '3p':
            center_emb = h[[query[0] for query in chosen_queries]] + \
                         r_center[[query[1][0] for query in chosen_queries]] + \
                         r_center[[query[1][1] for query in chosen_queries]] + \
                         r_center[[query[1][2] for query in chosen_queries]]
            offset_emb = self.func(r_offset[[query[1][0] for query in chosen_queries]]) + \
                         self.func(r_offset[[query[1][1] for query in chosen_queries]]) + \
                         self.func(r_offset[[query[1][2] for query in chosen_queries]])

        elif query_name_dict[query_type] == '2i':
            center_emb_1 = h[[query[0][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][0] for query in chosen_queries]]
            center_emb_2 = h[[query[1][0] for query in chosen_queries]] +\
                           r_center[[query[1][1][0] for query in chosen_queries]]
            center_emb = self.center_net(torch.stack([center_emb_1, center_emb_2]))

            offset_emb_1 = self.func(r_offset[[query[0][1][0] for query in chosen_queries]])
            offset_emb_2 = self.func(r_offset[[query[1][1][0] for query in chosen_queries]])
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2]))

        elif query_name_dict[query_type] == '3i':
            center_emb_1 = h[[query[0][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][0] for query in chosen_queries]]
            center_emb_2 = h[[query[1][0] for query in chosen_queries]] + \
                           r_center[[query[1][1][0] for query in chosen_queries]]
            center_emb_3 = h[[query[2][0] for query in chosen_queries]] + \
                           r_center[[query[2][1][0] for query in chosen_queries]]
            center_emb = self.center_net(torch.stack([center_emb_1, center_emb_2, center_emb_3]))

            offset_emb_1 = self.func(r_offset[[query[0][1][0] for query in chosen_queries]])
            offset_emb_2 = self.func(r_offset[[query[1][1][0] for query in chosen_queries]])
            offset_emb_3 = self.func(r_offset[[query[2][1][0] for query in chosen_queries]])
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2, offset_emb_3]))

        elif query_name_dict[query_type] == 'pi':
            center_emb_1 = h[[query[0][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][1] for query in chosen_queries]]
            center_emb_2 = h[[query[1][0] for query in chosen_queries]] + \
                           r_center[[query[1][1][0] for query in chosen_queries]]
            center_emb = self.center_net(torch.stack([center_emb_1, center_emb_2]))

            offset_emb_1 = self.func(r_offset[[query[0][1][0] for query in chosen_queries]]) + \
                           self.func(r_offset[[query[0][1][1] for query in chosen_queries]])
            offset_emb_2 = self.func(r_offset[[query[1][1][0] for query in chosen_queries]])
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2]))

        elif query_name_dict[query_type] == 'ip':
            center_emb_1 = h[[query[0][0][0] for query in chosen_queries]] + \
                           r_center[[query[0][0][1][0] for query in chosen_queries]]
            center_emb_2 = h[[query[0][1][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][1][0] for query in chosen_queries]]
            center_emb = self.center_net(torch.stack([center_emb_1, center_emb_2]))

            offset_emb_1 = self.func(r_offset[[query[0][0][1][0] for query in chosen_queries]])
            offset_emb_2 = self.func(r_offset[[query[0][1][1][0] for query in chosen_queries]])
            offset_emb = self.offset_net(torch.stack([offset_emb_1, offset_emb_2]))

            center_emb += r_center[[query[1][0] for query in chosen_queries]]
            offset_emb += self.func(r_offset[[query[1][0] for query in chosen_queries]])

        elif query_name_dict[query_type] == '2u-DNF':
            center_emb_1 = h[[query[0][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][0] for query in chosen_queries]]
            center_emb_2 = h[[query[1][0] for query in chosen_queries]] + \
                           r_center[[query[1][1][0] for query in chosen_queries]]

            offset_emb_1 = self.func(r_offset[[query[0][1][0] for query in chosen_queries]])
            offset_emb_2 = self.func(r_offset[[query[1][1][0] for query in chosen_queries]])

            center_emb = torch.cat([center_emb_1.unsqueeze(1),
                                    center_emb_2.unsqueeze(1)], dim=-2)

            offset_emb = torch.cat([offset_emb_1.unsqueeze(1),
                                    offset_emb_2.unsqueeze(1)], dim=-2)

        elif query_name_dict[query_type] == 'up-DNF':
            center = r_center[[query[1][0] for query in chosen_queries]]
            offset = self.func(r_offset[[query[1][0] for query in chosen_queries]])

            center_emb_1 = h[[query[0][0][0] for query in chosen_queries]] + \
                           r_center[[query[0][0][1][0] for query in chosen_queries]] + center
            center_emb_2 = h[[query[0][1][0] for query in chosen_queries]] + \
                           r_center[[query[0][1][1][0] for query in chosen_queries]] + center

            offset_emb_1 = self.func(r_offset[[query[0][0][1][0] for query in chosen_queries]]) + offset
            offset_emb_2 = self.func(r_offset[[query[0][1][1][0] for query in chosen_queries]]) + offset

            center_emb = torch.cat([center_emb_1.unsqueeze(1),
                                    center_emb_2.unsqueeze(1)], dim=-2)

            offset_emb = torch.cat([offset_emb_1.unsqueeze(1),
                                    offset_emb_2.unsqueeze(1)], dim=-2)

        if not evaluating:
            if 'u' in query_name_dict[query_type]:
                raise Exception("union query can not be trained!")
            target_embeds = self.entity_embedding[chosen_answers]
            delta = (target_embeds - center_emb).abs()
            distance_out = F.relu(delta - offset_emb)
            distance_in = torch.min(delta, offset_emb)
            scores = self.gamma - torch.norm(distance_out, p=1, dim=-1) - \
                     self.cen * torch.norm(distance_in, p=1, dim=-1)

            neg_embeds = self.entity_embedding[torch.tensor(neg_answers)]
            neg_delta = (neg_embeds - center_emb.unsqueeze(1)).abs()
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
                    delta = (target_embeds - center_emb[i].t().unsqueeze(0)).abs()
                    distance_out = F.relu(delta - offset_emb[i].t().unsqueeze(0))
                    distance_in = torch.min(delta, offset_emb[i].t().unsqueeze(0))
                    scores[i] = self.gamma - torch.norm(distance_out, p=1, dim=-1) - \
                                self.cen * torch.norm(distance_in, p=1, dim=-1)
            else :
                scores = torch.ones((batch_size, self.graph.n_entity), dtype=torch.float16).cuda()
                target_embeds = self.entity_embedding[:self.graph.n_entity]
                for i in range(batch_size):
                    delta = (target_embeds.unsqueeze(1) - center_emb[i].unsqueeze(0)).abs()
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


