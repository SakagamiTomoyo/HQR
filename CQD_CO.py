import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param

from torch import optim
from abc import ABC, abstractmethod
from typing import Tuple
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


class CQD_CO(nn.Module):
    def __init__(self, graph, args):
        super(CQD_CO, self).__init__()
        self.graph = graph
        self.embed_dim = args.embed_dim
        self.loss_f = args.loss_function
        self.multi_neg = args.multi_neg_answers

        self.entity_embedding = nn.Embedding(
            graph.n_entity + graph.n_relation + 1, 2 * self.embed_dim, sparse=True)

        self.relation_embedding = nn.Embedding(
            graph.n_relation + 1, 2 * self.embed_dim, sparse=True)

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

    # training, only 1p
    def forward(self, chosen_queries, query_type, sample_limit, subgraph, chosen_answers, query_name_dict,
                evaluating=False, neg_answers=None):

        def scoring_fn_chain(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
            score_2, factors_2 = self.score_emb(obj_guess_1, rel_2, obj_guess_2)
            factors = [factors_1[2], factors_2[2]]

            atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))

            if query_name_dict[query_type] == '3p':
                score_3, factors_3 = self.score_emb(obj_guess_2, rel_3, obj_guess_3)
                factors.append(factors_3[2])
                atoms = torch.cat((atoms, torch.sigmoid(score_3)), dim=1)

            guess_regularizer = self.regularizer(factors)
            t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                if query_name_dict[query_type] == '2p':
                    score_2 = self.forward_emb(obj_guess_1, rel_2)
                    atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2), dim=-1))
                else:
                    score_3 = self.forward_emb(obj_guess_2, rel_3)
                    atoms = torch.sigmoid(
                        torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))

                all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        def scoring_fn_inter():
            score_1 = self.forward_emb(lhs_1, rel_1)
            score_2 = self.forward_emb(lhs_2, rel_2)
            atoms = torch.stack((score_1, score_2), dim=-1)

            if disjunctive:
                atoms = torch.sigmoid(atoms)

            if query_name_dict[query_type] == '3i':
                score_3 = self.forward_emb(lhs_3, rel_3)
                atoms = torch.cat((atoms, score_3.unsqueeze(-1)), dim=-1)

            if disjunctive:
                all_scores = self.batch_t_conorm(atoms, norm_type)
            else:
                all_scores = self.batch_t_norm(atoms, norm_type)

            return all_scores

        def scoring_fn_pi(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
            score_2, _ = self.score_emb(obj_guess_1, rel_2, obj_guess_2)
            score_3, factors_2 = self.score_emb(lhs_2, rel_3, obj_guess_2)
            factors = [factors_1[2], factors_2[2]]

            atoms = torch.sigmoid(
                torch.cat((score_1, score_2, score_3), dim=1))

            guess_regularizer = self.regularizer(factors)

            t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                score_2 = self.forward_emb(obj_guess_1, rel_2)
                score_3 = self.forward_emb(lhs_2, rel_3)
                atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2, score_3), dim=-1))

                t_norm = self.batch_t_norm(atoms, norm_type)

                all_scores = t_norm

            return t_norm, guess_regularizer, all_scores

        def scoring_fn_ip(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
            score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess_1)
            score_3, factors_2 = self.score_emb(obj_guess_1, rel_3,
                                                obj_guess_2)
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
                score_3 = self.forward_emb(obj_guess_1, rel_3)
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
            prediction = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + \
                         (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
            factor = (
                        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                     )

            return prediction, factor, ans_ind

        else:
            self.entity_embedding.requires_grad_(requires_grad=False)
            self.relation_embedding.requires_grad_(requires_grad=False)

            if query_name_dict[query_type] == '1p':
                anchor = torch.tensor([query[0] for query in chosen_queries]).cuda()
                rel_ind = torch.tensor([query[1][0] for query in chosen_queries]).cuda()

                head = self.entity_embedding(anchor)
                rel = self.relation_embedding(rel_ind)

                lhs = head[:, :self.embed_dim], head[:, self.embed_dim:]
                rel = rel[:, :self.embed_dim], rel[:, self.embed_dim:]

                q = torch.cat([
                    lhs[0] * rel[0] - lhs[1] * rel[1],
                    lhs[0] * rel[1] + lhs[1] * rel[0]
                ], 1)

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

                    # if i != 1001:
                    #     print("Search converged early after {} iterations".format(i))

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

                    # if i != 1001:
                    #     print("Search converged early after {} iterations".format(i))

                with torch.no_grad():
                    *_, scores = scoring_fn_chain(score_all=True)

            elif query_name_dict[query_type] in ['2i', '2u-DNF']:
                if query_name_dict[query_type] == '2u-DNF':
                    disjunctive = True

                lhs_1 = self.entity_embedding(torch.tensor([query[0][0] for query in chosen_queries]).cuda())
                lhs_2 = self.entity_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                rel_1 = self.relation_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                rel_2 = self.relation_embedding(torch.tensor([query[1][1][0] for query in chosen_queries]).cuda())
                scores = scoring_fn_inter()

            elif query_name_dict[query_type] == '3i':
                lhs_1 = self.entity_embedding(torch.tensor([query[0][0] for query in chosen_queries]).cuda())
                lhs_2 = self.entity_embedding(torch.tensor([query[1][0] for query in chosen_queries]).cuda())
                lhs_3 = self.entity_embedding(torch.tensor([query[2][0] for query in chosen_queries]).cuda())
                rel_1 = self.relation_embedding(torch.tensor([query[0][1][0] for query in chosen_queries]).cuda())
                rel_2 = self.relation_embedding(torch.tensor([query[1][1][0] for query in chosen_queries]).cuda())
                rel_3 = self.relation_embedding(torch.tensor([query[2][1][0] for query in chosen_queries]).cuda())
                scores = scoring_fn_inter()

            elif query_name_dict[query_type] in ['ip', 'up-DNF']:
                if query_name_dict[query_type] == 'up-DNF':
                    disjunctive = True

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

                    # if i != 1001:
                    #     print("Search converged early after {} iterations".format(i))

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

                    #


                with torch.no_grad():
                    *_, scores = scoring_fn_pi(score_all=True)

            else:
                raise ValueError("unknown query type: " + str(query_type))

            return scores

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        lhs = lhs_emb[:, :self.embed_dim], lhs_emb[:, self.embed_dim:]
        rel = rel_emb[:, :self.embed_dim], rel_emb[:, self.embed_dim:]
        rhs = rhs_emb[:, :self.embed_dim], rhs_emb[:, self.embed_dim:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               )

    def forward_emb(self, lhs, rel):
        lhs = lhs[:, :self.embed_dim], lhs[:, self.embed_dim:]
        rel = rel[:, :self.embed_dim], rel[:, self.embed_dim:]

        to_score = self.entity_embedding.weight[:self.graph.n_entity]
        to_score = to_score[:, :self.embed_dim], to_score[:, self.embed_dim:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

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

        l_fit = self.loss(predictions, truth)
        l_reg = self.regularizer.forward(factors)
        loss = l_fit + l_reg

        return loss
