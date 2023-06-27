import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param

from abc import ABC, abstractmethod
from typing import Tuple
from torch import optim
import math
from torch import Tensor


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


class ProjectionAggregating(nn.Module):
    def __init__(self, dim):
        super(ProjectionAggregating, self).__init__()

        self.dim = dim

        self.attn_layer_1 = nn.Linear(self.dim * 4, self.dim * 2)
        self.attn_layer_2 = nn.Linear(self.dim * 2, self.dim * 2)

        nn.init.xavier_uniform_(self.attn_layer_1.weight)
        nn.init.xavier_uniform_(self.attn_layer_2.weight)

    def forward(self, lhs, refer_embs, rel, refer_r=None, start_embs=None):
        query_emb = [(lhs[0] * rel[0] - lhs[1] * rel[1]), (lhs[0] * rel[1] + lhs[1] * rel[0])]

        refer_r = refer_r[:, :, :self.dim], refer_r[:, :, self.dim:]
        query_emb = torch.cat(query_emb, dim=-1)
        if not isinstance(start_embs, tuple):
            if len(start_embs.shape) == 2:
                start_embs = start_embs[:, :self.dim], start_embs[:, self.dim:]
            else:
                start_embs = start_embs[:, :, :self.dim], start_embs[:, :, self.dim:]

        if not len(start_embs[0].shape) == 3:
            start_embs_0 = start_embs[0].unsqueeze(1)
            start_embs_1 = start_embs[1].unsqueeze(1)
        else:
            start_embs_0 = start_embs[0]
            start_embs_1 = start_embs[1]

        var_re = (start_embs_0 * refer_r[0] - start_embs_1 * refer_r[1])
        var_im = (start_embs_0 * refer_r[1] + start_embs_1 * refer_r[0])

        query_r = torch.cat(rel, dim=-1).unsqueeze(1).repeat(1, refer_r[0].shape[1], 1)
        refer_r = torch.cat(refer_r, dim=-1)

        bias = refer_embs - torch.cat([var_re, var_im], dim=-1)

        layer_act_1 = F.relu(self.attn_layer_1(torch.cat([query_r, refer_r], dim=-1)))

        final_refer_emb = torch.sum(self.attn_layer_2(layer_act_1) * bias, dim=-2)

        ref_norm = final_refer_emb.norm(p=1, dim=-1).unsqueeze(1)
        query_norm = query_emb.norm(p=1, dim=-1).unsqueeze(1)
        const = 10 # training 10

        final_refer_emb = final_refer_emb / (1e-9 + ref_norm / query_norm * const)

        final_query_emb = query_emb + final_refer_emb

        return final_query_emb[:, :self.dim], final_query_emb[:, self.dim:]


cur_type = None


class IDN_CQD_CO_24(nn.Module):
    def __init__(self, graph, args):
        super(IDN_CQD_CO_24, self).__init__()
        self.graph = graph
        self.embed_dim = args.embed_dim
        self.loss_f = args.loss_function
        self.multi_neg = args.multi_neg_answers

        self.entity_embedding = nn.Embedding(
            graph.n_entity + graph.n_relation + 1 + 1, 2 * self.embed_dim, sparse=True)

        self.relation_embedding = nn.Embedding(
            graph.n_relation + 1 + 1, 2 * self.embed_dim, sparse=True)

        self.init_size = 0.001

        self.entity_embedding.weight.data *= self.init_size
        self.relation_embedding.weight.data *= self.init_size

        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(0.05)

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

        self.obj_guess_1 = None
        self.obj_guess_2 = None

        self.mlp_projection = ProjectionAggregating(self.embed_dim)

    # training, only 1p
    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):

        def scoring_fn_chain(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1, h_target_1, r_target_1, start_embs_1)
            score_2, factors_2 = self.score_emb(obj_guess_1, rel_2, obj_guess_2, h_target_2, r_target_2, start_embs_2)
            factors = [factors_1[2], factors_2[2]]

            atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))

            if query_name_dict[query_type] == '3p':
                score_3, factors_3 = self.score_emb(obj_guess_2, rel_3, obj_guess_3, h_target_3, r_target_3, start_embs_3)
                factors.append(factors_3[2])
                atoms = torch.cat((atoms, torch.sigmoid(score_3)), dim=1)

            guess_regularizer = self.regularizer(factors)
            t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                if query_name_dict[query_type] == '2p':
                    score_2 = self.forward_emb(obj_guess_1, rel_2, h_target_2, r_target_2, start_embs_2)
                    atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2), dim=-1))
                else:
                    score_3 = self.forward_emb(obj_guess_2, rel_3, h_target_3, r_target_3, start_embs_3)
                    atoms = torch.sigmoid(
                        torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))

                all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        def scoring_fn_inter():
            score_1 = self.forward_emb(lhs_1, rel_1, h_target, r_target_1, start_embs_1)
            score_2 = self.forward_emb(lhs_2, rel_2, h_target, r_target_2, start_embs_2)
            atoms = torch.stack((score_1, score_2), dim=-1)

            if disjunctive:
                atoms = torch.sigmoid(atoms)

            if query_name_dict[query_type] == '3i':
                score_3 = self.forward_emb(lhs_3, rel_3, h_target, r_target_3, start_embs_3)
                atoms = torch.cat((atoms, score_3.unsqueeze(-1)), dim=-1)

            if disjunctive:
                all_scores = self.batch_t_conorm(atoms, norm_type)
            else:
                all_scores = self.batch_t_norm(atoms, norm_type)

            return all_scores

        # 2u
        def scoring_fn_union():
            score_1 = self.forward_emb(lhs_1, rel_1, h_target_1, r_target_1, start_embs_1)
            score_2 = self.forward_emb(lhs_2, rel_2, h_target_2, r_target_2, start_embs_2)
            atoms = torch.stack((score_1, score_2), dim=-1)

            if disjunctive:
                atoms = torch.sigmoid(atoms)

            if query_name_dict[query_type] == '3i':
                score_3 = self.forward_emb(lhs_3, rel_3, h_target, r_target_3, start_embs_3)
                atoms = torch.cat((atoms, score_3.unsqueeze(-1)), dim=-1)

            if disjunctive:
                all_scores = self.batch_t_conorm(atoms, norm_type)
            else:
                all_scores = self.batch_t_norm(atoms, norm_type)

            return all_scores

        def scoring_fn_pi(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1, h_target_1, r_target_1, lhs_1)
            score_2, _ = self.score_emb(obj_guess_1, rel_2, obj_guess_2, h_target, r_target_2, h_target_1)
            score_3, factors_2 = self.score_emb(lhs_2, rel_3, obj_guess_2, h_target, r_target_3, lhs_2)
            factors = [factors_1[2], factors_2[2]]

            atoms = torch.sigmoid(
                torch.cat((score_1, score_2, score_3), dim=1))

            guess_regularizer = self.regularizer(factors)

            t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                score_2 = self.forward_emb(obj_guess_1, rel_2, h_target, r_target_2, h_target_1)
                score_3 = self.forward_emb(lhs_2, rel_3, h_target, r_target_3, lhs_2)
                atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2, score_3), dim=-1))

                t_norm = self.batch_t_norm(atoms, norm_type)

                all_scores = t_norm

            return t_norm, guess_regularizer, all_scores

        def scoring_fn_ip(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1, h_target_1, r_target_1, lhs_1)
            score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess_1, h_target_1, r_target_2, lhs_2)
            score_3, factors_2 = self.score_emb(obj_guess_1, rel_3,
                                                obj_guess_2, h_target, r_target_3, h_target_1)
            factors = [factors_1[2], factors_2[2]]
            guess_regularizer = self.regularizer(factors)

            if not disjunctive:
                atoms = torch.sigmoid(torch.cat((score_1, score_2, score_3), dim=1))
                t_norm = self.batch_t_norm(atoms, norm_type)
            else:
                disj_atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))
                t_conorm = self.batch_t_conorm(disj_atoms, norm_type).unsqueeze(1)

                conj_atoms = torch.cat((t_conorm, torch.sigmoid(score_3)), dim=1)
                t_norm = self.batch_t_norm(conj_atoms, norm_type)

            all_scores = None
            if score_all:
                score_3 = self.forward_emb(obj_guess_1, rel_3, h_target, r_target_3, h_target_1)
                if not disjunctive:
                    atoms = torch.sigmoid(
                        torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))
                else:
                    atoms = torch.stack((t_conorm.expand_as(score_3), torch.sigmoid(score_3)), dim=-1)

                all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        def scoring_fn_up(score_all=False, left=True):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1 if left else obj_guess_3,
                                                h_target_1, r_target_1, lhs_1)
            score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess_1 if left else obj_guess_3,
                                        h_target_2, r_target_2, lhs_2)
            score_3, factors_2 = self.score_emb(obj_guess_1 if left else obj_guess_3,
                                                rel_3,
                                                obj_guess_2 if left else obj_guess_4,
                                                h_target_3 if left else h_target_4,
                                                r_target_3 if left else r_target_4,
                                                h_target_1)
            factors = [factors_1[2], factors_2[2]]
            guess_regularizer = self.regularizer(factors)

            if not disjunctive:
                atoms = torch.sigmoid(torch.cat((score_1, score_2, score_3), dim=1))
                t_norm = self.batch_t_norm(atoms, norm_type)
            else:
                disj_atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))
                t_conorm = self.batch_t_conorm(disj_atoms, norm_type).unsqueeze(1)

                conj_atoms = torch.cat((t_conorm, torch.sigmoid(score_3)), dim=1)
                t_norm = self.batch_t_norm(conj_atoms, norm_type)

            all_scores = None
            if score_all:
                score_3 = self.forward_emb(obj_guess_1 if left else obj_guess_3,
                                           rel_3, h_target_3 if left else h_target_4,
                                           r_target_3 if left else r_target_4, h_target_1)
                if not disjunctive:
                    atoms = torch.sigmoid(
                        torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))
                else:
                    atoms = torch.stack((t_conorm.expand_as(score_3), torch.sigmoid(score_3)), dim=-1)

                all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        batch_size = len(chosen_queries)
        norm_type = 'prod'
        disjunctive = False

        global cur_type
        cur_type = query_name_dict[query_type]

        self.entity_embedding.weight[-1].data *= 0
        self.relation_embedding.weight[-1].data *= 0

        if not evaluating:
            self.entity_embedding.requires_grad_(requires_grad=True)
            self.relation_embedding.requires_grad_(requires_grad=True)

            anchor = torch.tensor([query[0] for query in chosen_queries]).cuda()
            rel_ind = torch.tensor([query[1][0] for query in chosen_queries]).cuda()

            head = self.entity_embedding(anchor)
            rel = self.relation_embedding(rel_ind)

            lhs = head[:, :self.embed_dim], head[:, self.embed_dim:]
            rel = rel[:, :self.embed_dim], rel[:, self.embed_dim:]

            ans_ind = torch.tensor(chosen_answers).cuda()
            tail = self.entity_embedding(ans_ind)
            rhs = tail[:, :self.embed_dim], tail[:, self.embed_dim:]

            to_score = self.entity_embedding.weight[:self.graph.n_entity]
            to_score = to_score[:, :self.embed_dim], to_score[:, self.embed_dim:]

            h_target = self.entity_embedding.weight[torch.tensor([subgraph.p1_target], dtype=torch.long).cuda().squeeze()]
            r_target = self.relation_embedding.weight[torch.tensor([subgraph.p1_rel], dtype=torch.long).cuda().squeeze()]
            refer_start = lhs

            final_a, final_b = self.mlp_projection(lhs, h_target, rel, r_target, refer_start)

            prediction = final_a @ to_score[0].transpose(0, 1) + \
                         final_b @ to_score[1].transpose(0, 1)
            factor = (
                        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                     )

            return prediction, factor, ans_ind

        else:
            self.mlp_projection.requires_grad_(requires_grad=False)
            self.entity_embedding.requires_grad_(requires_grad=False)
            self.relation_embedding.requires_grad_(requires_grad=False)

            if query_name_dict[query_type] == '1p':
                anchor = torch.tensor([query[0] for query in chosen_queries]).cuda()
                rel_ind = torch.tensor([query[1][0] for query in chosen_queries]).cuda()

                head = self.entity_embedding(anchor)
                rel = self.relation_embedding(rel_ind)

                lhs = head[:, :self.embed_dim], head[:, self.embed_dim:]
                rel = rel[:, :self.embed_dim], rel[:, self.embed_dim:]

                h_target = self.entity_embedding.weight[torch.tensor([subgraph.p1_target], dtype=torch.long).cuda().squeeze()]
                r_target = self.relation_embedding.weight[torch.tensor([subgraph.p1_rel], dtype=torch.long).cuda().squeeze()]
                refer_start = lhs

                final_a, final_b = self.mlp_projection(lhs, h_target, rel, r_target, refer_start)

                q = torch.cat([final_a, final_b], dim=1)

                scores = q @ self.entity_embedding.weight[:self.graph.n_entity].transpose(0, 1)

            elif query_name_dict[query_type] == '2p':
                lhs_1 = self.entity_embedding(torch.tensor([query[0] for query in chosen_queries]).cuda())
                rel_1 = self.relation_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                rel_2 = self.relation_embedding(torch.tensor([query[1][1] for query in chosen_queries]).cuda())

                obj_guess_1 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)
                obj_guess_2 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)
                optimizer = optim.Adam([obj_guess_1, obj_guess_2], lr=0.1)

                start_embs_1 = lhs_1
                h_target_1 = self.entity_embedding.weight[torch.tensor([subgraph.p2_target1], dtype=torch.long).cuda().squeeze()]
                r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.p2_rel1], dtype=torch.long).cuda().squeeze()]

                start_embs_2 = h_target_1
                h_target_2 = self.entity_embedding.weight[torch.tensor([subgraph.p2_target2], dtype=torch.long).cuda().squeeze()]
                r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.p2_rel2], dtype=torch.long).cuda().squeeze()]

                prev_loss_value = 1000
                loss_value = 999
                losses = []

                with torch.set_grad_enabled(True):
                    i = 0
                    while i < 1001 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        norm, regularizer, _ = scoring_fn_chain()
                        loss = -norm.mean() + regularizer

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        i += 1

                        loss_value = loss.item()
                        losses.append(loss_value)

                with torch.no_grad():
                    *_, scores = scoring_fn_chain(score_all=True)

            elif query_name_dict[query_type] == '3p':
                lhs_1 = self.entity_embedding(torch.tensor([query[0] for query in chosen_queries]).cuda())
                rel_1 = self.relation_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                rel_2 = self.relation_embedding(torch.tensor([query[1][1] for query in chosen_queries]).cuda())
                rel_3 = self.relation_embedding(torch.tensor([query[1][2] for query in chosen_queries]).cuda())

                obj_guess_1 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)
                obj_guess_2 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)
                obj_guess_3 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)

                optimizer = optim.Adam([obj_guess_1, obj_guess_2, obj_guess_3], lr=0.1)

                start_embs_1 = lhs_1
                h_target_1 = self.entity_embedding.weight[torch.tensor([subgraph.p3_target1], dtype=torch.long).cuda().squeeze()]
                r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.p3_rel1], dtype=torch.long).cuda().squeeze()]

                start_embs_2 = h_target_1
                h_target_2 = self.entity_embedding.weight[torch.tensor([subgraph.p3_target2], dtype=torch.long).cuda().squeeze()]
                r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.p3_rel2], dtype=torch.long).cuda().squeeze()]

                start_embs_3 = h_target_2
                h_target_3 = self.entity_embedding.weight[torch.tensor([subgraph.p3_target3], dtype=torch.long).cuda().squeeze()]
                r_target_3 = self.relation_embedding.weight[torch.tensor([subgraph.p3_rel3], dtype=torch.long).cuda().squeeze()]

                prev_loss_value = 1000
                loss_value = 999
                losses = []

                with torch.set_grad_enabled(True):
                    i = 0
                    while i < 1001 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        norm, regularizer, _ = scoring_fn_chain()
                        loss = -norm.mean() + regularizer

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        i += 1

                        loss_value = loss.item()
                        losses.append(loss_value)

                with torch.no_grad():
                    *_, scores = scoring_fn_chain(score_all=True)

            elif query_name_dict[query_type] in ['2i', '2u-DNF']:

                if query_name_dict[query_type] == '2u-DNF':
                    disjunctive = True

                    lhs_1 = self.entity_embedding(torch.tensor([query[0][0] for query in chosen_queries]).cuda())
                    lhs_2 = self.entity_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                    rel_1 = self.relation_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                    rel_2 = self.relation_embedding(torch.tensor([query[1][1][0] for query in chosen_queries]).cuda())

                    start_embs_1 = lhs_1
                    start_embs_2 = lhs_2

                    h_target_1 = self.entity_embedding.weight[torch.tensor([subgraph.u2_target], dtype=torch.long).cuda().squeeze()[:, 0]]
                    h_target_2 = self.entity_embedding.weight[torch.tensor([subgraph.u2_target], dtype=torch.long).cuda().squeeze()[:, 1]]
                    r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.u2_rel], dtype=torch.long).cuda().squeeze()[:, 0]]
                    r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.u2_rel], dtype=torch.long).cuda().squeeze()[:, 1]]

                    scores = scoring_fn_union()

                else:
                    lhs_1 = self.entity_embedding(torch.tensor([query[0][0] for query in chosen_queries]).cuda())
                    lhs_2 = self.entity_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                    rel_1 = self.relation_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                    rel_2 = self.relation_embedding(torch.tensor([query[1][1][0] for query in chosen_queries]).cuda())

                    start_embs_1 = lhs_1
                    start_embs_2 = lhs_2
                    h_target = self.entity_embedding.weight[torch.tensor([subgraph.i2_target1], dtype=torch.long).cuda().squeeze()]
                    r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.i2_rel1], dtype=torch.long).cuda().squeeze()]
                    r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.i2_rel2], dtype=torch.long).cuda().squeeze()]

                    scores = scoring_fn_inter()

            elif query_name_dict[query_type] == '3i':
                lhs_1 = self.entity_embedding(torch.tensor([query[0][0] for query in chosen_queries]).cuda())
                lhs_2 = self.entity_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                lhs_3 = self.entity_embedding(torch.tensor([query[2][0] for query in chosen_queries]).cuda())
                rel_1 = self.relation_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                rel_2 = self.relation_embedding(torch.tensor([query[1][1][0] for query in chosen_queries]).cuda())
                rel_3 = self.relation_embedding(torch.tensor([query[2][1][0] for query in chosen_queries]).cuda())

                start_embs_1 = lhs_1
                start_embs_2 = lhs_2
                start_embs_3 = lhs_3
                h_target = self.entity_embedding.weight[torch.tensor([subgraph.i3_target1], dtype=torch.long).cuda().squeeze()]
                r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.i3_rel1], dtype=torch.long).cuda().squeeze()]
                r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.i3_rel2], dtype=torch.long).cuda().squeeze()]
                r_target_3 = self.relation_embedding.weight[torch.tensor([subgraph.i3_rel3], dtype=torch.long).cuda().squeeze()]

                scores = scoring_fn_inter()

            elif query_name_dict[query_type] in ['ip', 'up-DNF']:
                if query_name_dict[query_type] == 'up-DNF':
                    disjunctive = True

                    lhs_1 = self.entity_embedding(torch.tensor([query[0][0][0] for query in chosen_queries]).cuda())
                    rel_1 = self.relation_embedding(
                        torch.tensor([query[0][0][1][0] for query in chosen_queries]).cuda())
                    lhs_2 = self.entity_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                    rel_2 = self.relation_embedding(
                        torch.tensor([query[0][1][1][0] for query in chosen_queries]).cuda())
                    rel_3 = self.relation_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())

                    obj_guess_1 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                               device=self.entity_embedding.weight.device, requires_grad=True)
                    obj_guess_2 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                               device=self.entity_embedding.weight.device, requires_grad=True)
                    obj_guess_3 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                               device=self.entity_embedding.weight.device, requires_grad=True)
                    obj_guess_4 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                               device=self.entity_embedding.weight.device, requires_grad=True)
                    optimizer = optim.Adam([obj_guess_1, obj_guess_2, obj_guess_3, obj_guess_4], lr=0.1)

                    h_target_1 = self.entity_embedding.weight[torch.tensor([subgraph.up_target_1], dtype=torch.long).cuda().squeeze()[:, 0]]
                    r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.up_rel_1], dtype=torch.long).cuda().squeeze()[:, 0]]

                    h_target_2 = self.entity_embedding.weight[torch.tensor([subgraph.up_target_1], dtype=torch.long).cuda().squeeze()[:, 1]]
                    r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.up_rel_1], dtype=torch.long).cuda().squeeze()[:, 1]]

                    h_target_3 = self.entity_embedding.weight[torch.tensor([subgraph.up_target_2], dtype=torch.long).cuda().squeeze()[:, 0]]
                    r_target_3 = self.relation_embedding.weight[torch.tensor([subgraph.up_rel_2], dtype=torch.long).cuda().squeeze()[:, 0]]

                    h_target_4 = self.entity_embedding.weight[torch.tensor([subgraph.up_target_2], dtype=torch.long).cuda().squeeze()[:, 1]]
                    r_target_4 = self.relation_embedding.weight[torch.tensor([subgraph.up_rel_2], dtype=torch.long).cuda().squeeze()[:, 1]]

                    prev_loss_value = 1000
                    loss_value = 999
                    losses = []

                    with torch.set_grad_enabled(True):
                        i = 0
                        while i < 1001 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                            prev_loss_value = loss_value

                            norm, regularizer, _ = scoring_fn_up(left=True)
                            loss = -norm.mean() + regularizer

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            i += 1

                            loss_value = loss.item()
                            losses.append(loss_value)

                        # if i != 1001:
                        #     print("Search converged early after {} iterations".format(i))

                    with torch.set_grad_enabled(True):
                        i = 0
                        while i < 1001 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                            prev_loss_value = loss_value

                            norm, regularizer, _ = scoring_fn_up(left=False)
                            loss = -norm.mean() + regularizer

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            i += 1

                            loss_value = loss.item()
                            losses.append(loss_value)

                        # if i != 1001:
                        #     print("Search converged early after {} iterations".format(i))

                    with torch.no_grad():
                        *_, scores_1 = scoring_fn_up(score_all=True, left=True)
                        *_, scores_2 = scoring_fn_up(score_all=True, left=False)
                        scores = torch.max(torch.stack([scores_1, scores_2]), dim=0)[0]

                else:
                    lhs_1 = self.entity_embedding(torch.tensor([query[0][0][0] for query in chosen_queries]).cuda())
                    rel_1 = self.relation_embedding(torch.tensor([query[0][0][1][0] for query in chosen_queries]).cuda())
                    lhs_2 = self.entity_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                    rel_2 = self.relation_embedding(torch.tensor([query[0][1][1][0] for query in chosen_queries]).cuda())
                    rel_3 = self.relation_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())

                    obj_guess_1 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                               device=self.entity_embedding.weight.device, requires_grad=True)
                    obj_guess_2 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                               device=self.entity_embedding.weight.device, requires_grad=True)
                    optimizer = optim.Adam([obj_guess_1, obj_guess_2], lr=0.1)

                    h_target_1 = self.entity_embedding.weight[torch.tensor([subgraph.ip_target1], dtype=torch.long).cuda().squeeze()]
                    r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.ip_rel1], dtype=torch.long).cuda().squeeze()]
                    r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.ip_rel2], dtype=torch.long).cuda().squeeze()]

                    h_target = self.entity_embedding.weight[torch.tensor([subgraph.ip_target3], dtype=torch.long).cuda().squeeze()]
                    r_target_3 = self.relation_embedding.weight[torch.tensor([subgraph.ip_rel3], dtype=torch.long).cuda().squeeze()]

                    prev_loss_value = 1000
                    loss_value = 999
                    losses = []

                    with torch.set_grad_enabled(True):
                        i = 0
                        while i < 1001 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                            prev_loss_value = loss_value

                            norm, regularizer, _ = scoring_fn_ip()
                            loss = -norm.mean() + regularizer

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            i += 1

                            loss_value = loss.item()
                            losses.append(loss_value)

                    with torch.no_grad():
                        *_, scores = scoring_fn_ip(score_all=True)

            elif query_name_dict[query_type] == 'pi':

                lhs_1 = self.entity_embedding(torch.tensor([query[0][0] for query in chosen_queries]).cuda())
                rel_1 = self.relation_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                rel_2 = self.relation_embedding(torch.tensor([query[0][1][1] for query in chosen_queries]).cuda())
                lhs_2 = self.entity_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                rel_3 = self.relation_embedding(torch.tensor([query[1][1][0] for query in chosen_queries]).cuda())

                obj_guess_1 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)
                obj_guess_2 = torch.normal(0, self.init_size, (batch_size, self.embed_dim * 2),
                                           device=self.entity_embedding.weight.device, requires_grad=True)
                optimizer = optim.Adam([obj_guess_1, obj_guess_2], lr=0.1)

                h_target_1 = self.entity_embedding.weight[torch.tensor([subgraph.pi_target1], dtype=torch.long).cuda().squeeze()]
                r_target_1 = self.relation_embedding.weight[torch.tensor([subgraph.pi_rel1], dtype=torch.long).cuda().squeeze()]

                h_target = self.entity_embedding.weight[torch.tensor([subgraph.pi_target2], dtype=torch.long).cuda().squeeze()]
                r_target_2 = self.relation_embedding.weight[torch.tensor([subgraph.pi_rel2], dtype=torch.long).cuda().squeeze()]
                r_target_3 = self.relation_embedding.weight[torch.tensor([subgraph.pi_rel3], dtype=torch.long).cuda().squeeze()]

                prev_loss_value = 1000
                loss_value = 999
                losses = []

                with torch.set_grad_enabled(True):
                    i = 0
                    while i < 1001 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        norm, regularizer, _ = scoring_fn_pi()
                        loss = -norm.mean() + regularizer

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        i += 1

                        loss_value = loss.item()
                        losses.append(loss_value)

                with torch.no_grad():
                    *_, scores = scoring_fn_pi(score_all=True)

            else:
                raise ValueError("unknown query type: " + str(query_type))

            return scores

    def score_emb(self, lhs_emb, rel_emb, rhs_emb, refer_embs=None, refer_r=None, start_embs=None):
        lhs = lhs_emb[:, :self.embed_dim], lhs_emb[:, self.embed_dim:]
        rel = rel_emb[:, :self.embed_dim], rel_emb[:, self.embed_dim:]
        rhs = rhs_emb[:, :self.embed_dim], rhs_emb[:, self.embed_dim:]

        final_a, final_b = self.mlp_projection(lhs, refer_embs, rel, refer_r, start_embs)

        return torch.sum(
            final_a * rhs[0] +
            final_b * rhs[1],
            1, keepdim=True), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               )

    def forward_emb(self, lhs, rel, refer_embs=None, refer_r=None, start_embs=None):
        lhs = lhs[:, :self.embed_dim], lhs[:, self.embed_dim:]
        rel = rel[:, :self.embed_dim], rel[:, self.embed_dim:]

        final_a, final_b = self.mlp_projection(lhs, refer_embs, rel, refer_r, start_embs)

        to_score = self.entity_embedding.weight[:self.graph.n_entity]
        to_score = to_score[:, :self.embed_dim], to_score[:, self.embed_dim:]
        return (final_a @ to_score[0].transpose(0, 1) +
                final_b @ to_score[1].transpose(0, 1))

    @staticmethod
    def batch_t_norm(atoms: Tensor, norm_type: str = 'min') -> Tensor:
        if norm_type == 'min':
            scores = torch.min(atoms, dim=-1)[0]
        elif norm_type == 'prod':
            scores = torch.prod(atoms, dim=-1)
        else:
            raise ValueError(f't_norm must be "min" or "prod", got {norm_type}')

        return scores

    @staticmethod
    def batch_t_conorm(atoms: Tensor, norm_type: str = 'max') -> Tensor:
        if norm_type == 'min':
            scores = torch.max(atoms, dim=-1)[0]
        elif norm_type == 'prod':
            scores = torch.sum(atoms, dim=-1) - torch.prod(atoms, dim=-1)
        else:
            raise ValueError(f't_conorm must be "min" or "prod", got {norm_type}')

        return scores

    def margin_loss(self, chosen_queries, chosen_answers, neg_answers, query_type, sample_limit, subgraph, all_query_num,
                    margin=1, query_name_dict=None):
        predictions, factors, truth = self.forward(
            chosen_queries, query_type, sample_limit, subgraph,
            chosen_answers, query_name_dict, neg_answers=neg_answers)

        is_fine_tune = query_name_dict[query_type] != '1p'

        l_fit = self.loss(predictions, truth)
        l_reg = self.regularizer.forward(factors)
        loss = l_fit + l_reg

        return loss
