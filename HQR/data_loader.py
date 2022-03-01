from collections import defaultdict
from qaidn_2.graph import Graph
import os
import pickle
import numpy as np
from functools import reduce
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data, Batch, NeighborSampler
from torch_geometric.data.sampler import EdgeIndex


query_name_dict = {('e', ('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                    }


def load_graph(data_dir, embed_dim, sample_limit, train_queries, train_answers):
    node_count = defaultdict(int)   # int_int
    edge_count = defaultdict(int)   # int_int
    triple_list = []

    with open(data_dir + 'train.txt', 'r', encoding='utf-8') as graph_reader:
        for triple in graph_reader.readlines():
            h, r, t = list(map(lambda x: int(x), triple.strip().split('\t')))
            triple_list.append((h, r, t))
            node_count[h] += 1
            edge_count[r] += 1
            node_count[t] += 1
        print(len(triple_list))

    with open(data_dir + 'stats.txt') as f:
        entrel = f.readlines()
        n_entity = int(entrel[0].split(' ')[-1])
        n_relation = int(entrel[1].split(' ')[-1])

    graph = Graph(embed_dim, triple_list, n_entity, n_relation, node_count, edge_count, sample_limit)
    return graph


def load_data(args, tasks, all_tasks, name_query_dict):
    train_queries = pickle.load(open(os.path.join(args.data_dir, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_dir, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_dir, "valid-queries.pkl"), 'rb'))

    valid_answers = pickle.load(open(os.path.join(args.data_dir, "valid-answers.pkl"), 'rb'))

    valid_hard_answers = pickle.load(open(os.path.join(args.data_dir, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_dir, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_dir, "test-queries.pkl"), 'rb'))
    test_answers = pickle.load(open(os.path.join(args.data_dir, "test-answers.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_dir, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_dir, "test-easy-answers.pkl"), 'rb'))

    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return train_queries, train_answers, valid_queries, valid_easy_answers, valid_hard_answers, \
           test_queries, test_easy_answers, test_hard_answers, valid_answers, test_answers


def get_anchors(query_type, query):
    if query_type in [('e', ('r',)),
                      ('e', ('r', 'r')),
                      ('e', ('r', 'r', 'r'))]:
        return [query[0]]
    elif query_type in [(('e', ('r',)), ('e', ('r',))),
                        (('e', ('r', 'r')), ('e', ('r',))),
                        (('e', ('r',)), ('e', ('r', 'n'))),
                        (('e', ('r', 'r')), ('e', ('r', 'n'))),
                        (('e', ('r', 'r', 'n')), ('e', ('r',))),
                        (('e', ('r',)), ('e', ('r',)), ('u',))]:
        return [query[0][0], query[1][0]]
    elif query_type in [((('e', ('r',)), ('e', ('r',))), ('r',)),
                        ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),
                        ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
                        ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)),
                        ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r'))]:
        return [query[0][0][0], query[0][1][0]]
    elif query_type in [(('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
                        (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))]:
        return [query[0][0], query[1][0], query[2][0]]


def get_edge_index(query_type, query, num):
    if query_type == ('e', ('r',)):
        return ([[0], [num]],
                [query[1][0]],
                [[query[1][0]]])
    elif query_type == ('e', ('r', 'r')):
        return ([[0, num], [num, num + 1]],
                [query[1][0], query[1][1]],
                [[query[1][0]], [query[1][1]]])
    elif query_type == ('e', ('r', 'r', 'r')):
        return ([[0, num, num + 1], [num, num + 1, num + 2]],
                [query[1][0], query[1][1], query[1][2]],
                [[query[1][0]], [query[1][1]], [query[1][2]]])
    elif query_type == (('e', ('r',)), ('e', ('r',))):
        return ([[0, 1], [num, num]],
                [query[0][1][0], query[1][1][0]],
                [[query[0][1][0], query[1][1][0]]])
    elif query_type == (('e', ('r', 'r')), ('e', ('r',))):
        return ([[0, num, 1], [num, num + 1, num + 1]],
                [query[0][1][0], query[0][1][1], query[1][1][0]],
                [[query[0][1][0]], [query[0][1][1], query[1][1][0]]])
    elif query_type == (('e', ('r',)), ('e', ('r', 'n'))):
        return None
    elif query_type == (('e', ('r', 'r')), ('e', ('r', 'n'))):
        return None
    elif query_type == (('e', ('r', 'r', 'n')), ('e', ('r',))):
        return None
    elif query_type == (('e', ('r',)), ('e', ('r',)), ('u',)):
        return ([[0, 1], [num, num]],
                [query[0][1][0], query[1][1][0]],
                [[query[0][1][0], query[1][1][0]]])
    elif query_type == ((('e', ('r',)), ('e', ('r',))), ('r',)):
        return ([[0, 1, num], [num, num, num + 1]],
                [query[0][0][1][0], query[0][1][1][0], query[1][0]],
                [[query[0][0][1][0], query[0][1][1][0]], [query[1][0]]])
    elif query_type == ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)):
        return None
    elif query_type == ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)):
        return ([[0, 1, num], [num, num, num + 1]],
                [query[0][0][1][0], query[0][1][1][0], query[1][0]],
                [[query[0][0][1][0], query[0][1][1][0]], [query[1][0]]])
    elif query_type == ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)):
        return None
    elif query_type == ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')):
        return None
    elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',))):
        return ([[0, 1, 2], [num, num, num]],
                [query[0][1][0], query[1][1][0], query[2][1][0]],
                [[query[0][1][0], query[1][1][0], query[2][1][0]]])
    elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))):
        return None


def generate_id_neighbor(args, graph,
                         train_queries, valid_queries, test_queries,
                         train_answers, valid_answers, test_answers,
                         query_name_dict):
    dir = args.data_dir
    id_neighbor_num = args.id_neighbor_limit

    heads = np.array(graph.edge_index[0])
    rels = np.array(graph.edge_type)
    tails = np.array(graph.edge_index[1])
    
    generate_id_neighbor_helper(heads, rels, tails, id_neighbor_num, dir, train_queries, train_answers, query_name_dict,
                            'train_queries_idneighbor_new_3.pkl')
    generate_id_neighbor_helper(heads, rels, tails, id_neighbor_num, dir, valid_queries, train_answers, query_name_dict,
                            'valid_queries_idneighbor_new_3.pkl', False)
    generate_id_neighbor_helper(heads, rels, tails, id_neighbor_num, dir, test_queries, train_answers, query_name_dict,
                            'test_queries_idneighbor_new_3.pkl', False)


def generate_id_neighbor_helper(heads, rels, tails, id_neighbor_num, dir, queries, answers, query_name_dict, fname,
                                training=True):
    query_idn_dict = defaultdict(dict)

    for query_type in queries:
        count = 0
        count_min = 0
        count_1 = 0
        for query in queries[query_type]:
            if count % 1000 == 0:
                print(count, count_min, count_1)
            if query_name_dict[query_type] == '1p':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)[0]
                chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
                chosen_heads_index = np.random.choice(chosen_heads_index,
                                                      id_neighbor_num * 30
                                                      if chosen_heads_index.shape[0] > id_neighbor_num * 30
                                                      else chosen_heads_index.shape[0])
                chosen_tails = tails[chosen_heads_index]
                chosen_rels = rels[chosen_heads_index]

                for i in range(chosen_tails.shape[0]):
                    q_neighbor = (anchor, (chosen_rels[i], ))
                    ans = chosen_tails[i]
                    if training:
                        if q_neighbor == query or ans in answers[query]:
                            continue
                    temp_map[query].add((anchor, chosen_rels[i], ans))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if chosen_tails.shape[0] < (id_neighbor_num / 3):
                    count_min += 1

            elif query_name_dict[query_type] == '2p':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)[0]
                chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
                chosen_heads_index = np.random.choice(chosen_heads_index,
                                                      id_neighbor_num * 2
                                                      if chosen_heads_index.shape[0] > id_neighbor_num * 2
                                                      else chosen_heads_index.shape[0])
                chosen_tails_1 = tails[chosen_heads_index]
                chosen_rels_1 = rels[chosen_heads_index]

                for i in range(chosen_tails_1.shape[0]):
                    temp_rel = chosen_rels_1[i]
                    chosen_tails_1_index = np.squeeze(np.argwhere(heads == chosen_tails_1[i]).T, axis=0)
                    chosen_tails_1_index = np.random.choice(chosen_tails_1_index,
                                                            id_neighbor_num * 2
                                                            if chosen_tails_1_index.shape[0] > id_neighbor_num * 2
                                                            else chosen_tails_1_index.shape[0])
                    chosen_tails_2 = tails[chosen_tails_1_index]
                    chosen_rels_2 = rels[chosen_tails_1_index]
                    for j in range(chosen_tails_2.shape[0]):
                        q_neighbor = (anchor, (temp_rel, chosen_rels_2[j]))
                        ans = chosen_tails_2[j]
                        if training:
                            if q_neighbor == query or ans in answers[query]:
                                continue
                        temp_map[query].add((anchor, temp_rel, chosen_tails_1[i], chosen_rels_2[j], ans))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if len(temp_map[query]) < (id_neighbor_num / 3):
                    count_min += 1
            elif query_name_dict[query_type] == '3p':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)[0]
                chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
                chosen_heads_index = np.random.choice(chosen_heads_index,
                                                      id_neighbor_num * 2
                                                      if chosen_heads_index.shape[0] > id_neighbor_num * 2
                                                      else chosen_heads_index.shape[0])
                chosen_tails_1 = tails[chosen_heads_index]
                chosen_rels_1 = rels[chosen_heads_index]

                for i in range(chosen_tails_1.shape[0]):
                    temp_rel_1 = chosen_rels_1[i]
                    chosen_tails_1_index = np.squeeze(np.argwhere(heads == chosen_tails_1[i]).T, axis=0)
                    chosen_tails_1_index = np.random.choice(chosen_tails_1_index,
                                                            id_neighbor_num * 2
                                                            if chosen_tails_1_index.shape[0] > id_neighbor_num * 2
                                                            else chosen_tails_1_index.shape[0])
                    chosen_tails_2 = tails[chosen_tails_1_index]
                    chosen_rels_2 = rels[chosen_tails_1_index]
                    for j in range(chosen_tails_2.shape[0]):
                        temp_rel_2 = chosen_rels_2[j]
                        chosen_tails_2_index = np.squeeze(np.argwhere(heads == chosen_tails_2[j]).T, axis=0)
                        chosen_tails_2_index = np.random.choice(chosen_tails_2_index,
                                                                id_neighbor_num * 2
                                                                if chosen_tails_2_index.shape[0] > id_neighbor_num * 2
                                                                else chosen_tails_2_index.shape[0])
                        chosen_tails_3 = tails[chosen_tails_2_index]
                        chosen_rels_3 = rels[chosen_tails_2_index]
                        for k in range(chosen_tails_3.shape[0]):
                            q_neighbor = (anchor, (temp_rel_1, temp_rel_2, chosen_rels_3[k]))
                            ans = chosen_tails_3[k]
                            if training:
                                if q_neighbor == query or ans in answers[query]:
                                    continue
                            temp_map[query].add((anchor, temp_rel_1, chosen_tails_1[i],
                                                 temp_rel_2, chosen_tails_2[j],
                                                 chosen_rels_3[k], ans))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if len(temp_map[query]) < int(id_neighbor_num / 3):
                    count_min += 1

            elif query_name_dict[query_type] == '2i':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)
                chosen_heads_index_1 = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
                chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)

                chosen_tails_1 = tails[chosen_heads_index_1]
                chosen_rels_1 = rels[chosen_heads_index_1]

                chosen_tails_2 = tails[chosen_heads_index_2]
                chosen_rels_2 = rels[chosen_heads_index_2]

                ans = np.intersect1d(chosen_tails_1, chosen_tails_2)
                if training:
                    ans = np.setdiff1d(ans, np.array(list(answers[query])))

                for i in range(ans.shape[0]):
                    rel1_list = chosen_rels_1[chosen_tails_1 == ans[i]]
                    rel2_list = chosen_rels_2[chosen_tails_2 == ans[i]]
                    for rel1 in rel1_list:
                        for rel2 in rel2_list:
                            # q_neighbor = ((anchor[0], (rel1,)), (anchor[1], (rel2,)))
                            a = ans[i]
                            temp_map[query].add(((anchor[0], rel1, a), (anchor[1], rel2, a)))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if len(temp_map[query]) < int(id_neighbor_num / 3):
                    # print("list_2i", len(list_2i))
                    count_min += 1
            elif query_name_dict[query_type] == '3i':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)
                chosen_heads_index_1 = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
                chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)
                chosen_heads_index_3 = np.squeeze(np.argwhere(heads == anchor[2]).T, axis=0)

                chosen_tails_1 = tails[chosen_heads_index_1]
                chosen_rels_1 = rels[chosen_heads_index_1]

                chosen_tails_2 = tails[chosen_heads_index_2]
                chosen_rels_2 = rels[chosen_heads_index_2]

                chosen_tails_3 = tails[chosen_heads_index_3]
                chosen_rels_3 = rels[chosen_heads_index_3]

                ans = reduce(np.intersect1d, (chosen_tails_1, chosen_tails_2, chosen_tails_3))
                if training:
                    ans = np.setdiff1d(ans, np.array(list(answers[query])))

                for i in range(ans.shape[0]):
                    rel1_list = chosen_rels_1[chosen_tails_1 == ans[i]]
                    rel2_list = chosen_rels_2[chosen_tails_2 == ans[i]]
                    rel3_list = chosen_rels_3[chosen_tails_3 == ans[i]]

                    for rel1 in rel1_list:
                        for rel2 in rel2_list:
                            for rel3 in rel3_list:
                                # q_neighbor = ((anchor[0], (rel1,)), (anchor[1], (rel2,)), (anchor[2], (rel3,)))
                                # a = ans[i]
                                temp_map[query].add(((anchor[0], rel1, ans[i]),
                                                     (anchor[1], rel2, ans[i]),
                                                     (anchor[2], rel3, ans[i])))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if len(temp_map[query]) <= 3:
                    # print("list_3i", len(list_3i))
                    count_min += 1

                if len(temp_map[query]) == 1:
                    # print("list_3i", len(list_3i))
                    count_1 += 1

            elif query_name_dict[query_type] == 'ip':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)
                chosen_heads_index_1 = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
                chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)

                chosen_tails_1 = tails[chosen_heads_index_1]
                chosen_rels_1 = rels[chosen_heads_index_1]

                chosen_tails_2 = tails[chosen_heads_index_2]
                chosen_rels_2 = rels[chosen_heads_index_2]

                ans = np.intersect1d(chosen_tails_1, chosen_tails_2)

                for i in range(ans.shape[0]):
                    rel1_list = chosen_rels_1[chosen_tails_1 == ans[i]]
                    rel2_list = chosen_rels_2[chosen_tails_2 == ans[i]]
                    for rel1 in rel1_list:
                        for rel2 in rel2_list:
                            a = ans[i]
                            chosen_heads_index = np.squeeze(np.argwhere(heads == a).T, axis=0)
                            chosen_heads_index = np.random.choice(chosen_heads_index,
                                                                  id_neighbor_num
                                                                  if chosen_heads_index.shape[0] > id_neighbor_num
                                                                  else chosen_heads_index.shape[0])
                            chosen_tails = tails[chosen_heads_index]
                            chosen_rels = rels[chosen_heads_index]

                            for j in range(chosen_tails.shape[0]):
                                # q_neighbor = (((anchor[0], (rel1,)), (anchor[1], (rel2,))), (chosen_rels[j], ))
                                real_ans = chosen_tails[j]
                                # if q_neighbor == query and real_ans in answers[query]:
                                #     continue
                                temp_map[query].add((((anchor[0], rel1, a), (anchor[1], rel2, a)),
                                                     (a, chosen_rels[j], real_ans)))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if len(temp_map[query]) < int(id_neighbor_num / 3):
                    # print("list_2i", len(list_2i))
                    count_min += 1
            elif query_name_dict[query_type] == 'pi':
                temp_map = defaultdict(set)
                count += 1
                anchor = get_anchors(query_type, query)
                chosen_heads_index = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
                chosen_heads_index = np.random.choice(chosen_heads_index,
                                                      id_neighbor_num
                                                      if chosen_heads_index.shape[0] > id_neighbor_num
                                                      else chosen_heads_index.shape[0])
                chosen_tails = tails[chosen_heads_index]
                chosen_rels = rels[chosen_heads_index]

                for i in range(chosen_tails.shape[0]):
                    first_rel = chosen_rels[i]

                    chosen_heads_index_1 = np.squeeze(np.argwhere(heads == chosen_tails[i]).T, axis=0)
                    chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)

                    chosen_tails_1 = tails[chosen_heads_index_1]
                    chosen_rels_1 = rels[chosen_heads_index_1]

                    chosen_tails_2 = tails[chosen_heads_index_2]
                    chosen_rels_2 = rels[chosen_heads_index_2]

                    ans = np.intersect1d(chosen_tails_1, chosen_tails_2)

                    for j in range(ans.shape[0]):
                        rel1_list = chosen_rels_1[chosen_tails_1 == ans[j]]
                        rel2_list = chosen_rels_2[chosen_tails_2 == ans[j]]
                        for rel1 in rel1_list:
                            for rel2 in rel2_list:
                                # q_neighbor = ((anchor[0], (first_rel, rel1)), (anchor[1], (rel2,)))
                                a = ans[j]
                                # if q_neighbor == query and a in answers[query]:
                                #     continue
                                temp_map[query].add(((anchor[0], first_rel, chosen_tails[i], rel1, a), (anchor[1], rel2, a)))

                temp_map[query] = random.sample(temp_map[query],
                                                id_neighbor_num * 2
                                                if len(temp_map[query]) > id_neighbor_num * 2
                                                else len(temp_map[query]))

                query_idn_dict[query_type].update(temp_map)

                if chosen_tails.shape[0] < (id_neighbor_num / 3):
                    count_min += 1
            elif query_name_dict[query_type] == '2u-DNF' and 'con' not in fname:
                temp_map = defaultdict(dict)
                count += 1
                anchors = get_anchors(query_type, query)
                for ind in range(len(anchors)):
                    temp_map_inner = defaultdict(set)
                    key = query[ind]
                    anchor = anchors[ind]
                    chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
                    chosen_heads_index = np.random.choice(chosen_heads_index,
                                                          id_neighbor_num * 10
                                                          if chosen_heads_index.shape[0] > id_neighbor_num * 10
                                                          else chosen_heads_index.shape[0])
                    chosen_tails = tails[chosen_heads_index]
                    chosen_rels = rels[chosen_heads_index]

                    for i in range(chosen_tails.shape[0]):
                        q_neighbor = (anchor, (chosen_rels[i],))
                        ans = chosen_tails[i]
                        # if q_neighbor == query and ans in answers[query]:
                        #     continue
                        temp_map_inner[key].add((anchor, chosen_rels[i], ans))

                    temp_map_inner[key] = random.sample(temp_map_inner[key],
                                                        id_neighbor_num * 2
                                                        if len(temp_map_inner[key]) > id_neighbor_num * 2
                                                        else len(temp_map_inner[key]))

                    temp_map[query].update(temp_map_inner)

                query_idn_dict[query_type].update(temp_map)
            elif query_name_dict[query_type] == 'up-DNF' and 'con' not in fname:
                temp_map = defaultdict(dict)

                count += 1
                anchors = get_anchors(query_type, query)
                for ind in range(len(anchors)):
                    temp_map_inner = defaultdict(set)
                    key = (query[0][ind], query[1])
                    anchor = anchors[ind]
                    chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
                    chosen_heads_index = np.random.choice(chosen_heads_index,
                                                          id_neighbor_num
                                                          if chosen_heads_index.shape[0] > id_neighbor_num
                                                          else chosen_heads_index.shape[0])
                    chosen_tails_1 = tails[chosen_heads_index]
                    chosen_rels_1 = rels[chosen_heads_index]

                    for i in range(chosen_tails_1.shape[0]):
                        temp_rel = chosen_rels_1[i]
                        chosen_tails_1_index = np.squeeze(np.argwhere(heads == chosen_tails_1[i]).T, axis=0)
                        chosen_tails_1_index = np.random.choice(chosen_tails_1_index,
                                                                id_neighbor_num
                                                                if chosen_tails_1_index.shape[0] > id_neighbor_num
                                                                else chosen_tails_1_index.shape[0])
                        chosen_tails_2 = tails[chosen_tails_1_index]
                        chosen_rels_2 = rels[chosen_tails_1_index]
                        for j in range(chosen_tails_2.shape[0]):
                            q_neighbor = (anchor, (temp_rel, chosen_rels_2[j]))
                            ans = chosen_tails_2[j]
                            # if q_neighbor == query and ans in answers[query]:
                            #     continue
                            temp_map_inner[key].add((anchor, temp_rel, chosen_tails_1[i], chosen_rels_2[j], ans))

                    temp_map_inner[key] = random.sample(temp_map_inner[key],
                                                        id_neighbor_num * 2
                                                        if len(temp_map_inner[key]) > id_neighbor_num * 2
                                                        else len(temp_map_inner[key]))

                    temp_map[query].update(temp_map_inner)

                query_idn_dict[query_type].update(temp_map)
           

    pickle.dump(query_idn_dict, open(os.path.join(dir, fname), 'wb'))

def load_helper_neighbor(heads, rels, tails, id_neighbor_num, dir, queries, answers, query_name_dict, fname):
    query_idn_dict = defaultdict(dict)

    for query_type in queries:
        count = 0
        count_min = 0
        count_1 = 0
        for query in queries[query_type]:
            if count % 1000 == 0:
                print(count, count_min, count_1)
            if query_name_dict[query_type] == '2i' or query_name_dict[query_type] == '3i':
                temp_map = defaultdict(dict)
                count += 1
                anchor = get_anchors(query_type, query)

                temp_set = defaultdict(set)
                for a in anchor:
                    temp_set = defaultdict(set)
                    chosen_heads_index = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
                    chosen_heads_index = np.random.choice(chosen_heads_index,
                                                          id_neighbor_num * 10
                                                          if chosen_heads_index.shape[0] > id_neighbor_num * 10
                                                          else chosen_heads_index.shape[0])
                    chosen_tails = tails[chosen_heads_index]
                    chosen_rels = rels[chosen_heads_index]

                    for j in range(chosen_tails.shape[0]):
                        q_neighbor = (anchor, (chosen_rels[j],))
                        ans = chosen_tails[j]
                        if q_neighbor == query and ans in answers[query]:
                            continue
                        temp_set[a].add((chosen_rels[j], ans))

                    temp_map[query].update(temp_set)

                for k in anchor:
                    temp_map[query][k] = random.sample(temp_map[query][k],
                                                       id_neighbor_num * 2
                                                       if len(temp_map[query][k]) > id_neighbor_num * 2
                                                       else len(temp_map[query][k]))

                query_idn_dict[query_type].update(temp_map)

    pickle.dump(query_idn_dict, open(os.path.join(dir, fname), 'wb'))


graph_static = None
query_node_num = None
query_name = None
name_query = None

train_answers_static = None
valid_easy_answers_static = None
valid_hard_answers_static = None
test_easy_answers_static = None
test_hard_answers_static = None

train_idn_static = None
valid_idn_static = None
test_idn_static = None

train_helper_idn_static = None
valid_helper_idn_static = None
test_helper_idn_static = None


def set_graph(graph_in):
    global graph_static, query_node_num, query_name, name_query
    graph_static = graph_in
    query_node_num = {('e', ('r',)): [1, 1],
                      ('e', ('r', 'r')): [1, 2],
                      ('e', ('r', 'r', 'r')): [1, 3],
                      (('e', ('r',)), ('e', ('r',))): [2, 1],
                      (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): [3, 1],
                      ((('e', ('r',)), ('e', ('r',))), ('r',)): [2, 2],
                      (('e', ('r', 'r')), ('e', ('r',))): [2, 2],
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
    query_name = {('e', ('r',)): '1p',
                       ('e', ('r', 'r')): '2p',
                       ('e', ('r', 'r', 'r')): '3p',
                       (('e', ('r',)), ('e', ('r',))): '2i',
                       (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                       ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                       (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                       (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                       (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                       ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                       (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                       (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                       (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                       ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                       ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                       ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                       }
    name_query = {value: key for key, value in query_name.items()}


def set_answers(train_answers,
                valid_easy_answers, valid_hard_answers,
                test_easy_answers, test_hard_answers):
    global train_answers_static, valid_easy_answers_static, \
        valid_hard_answers_static, test_easy_answers_static, \
        test_hard_answers_static
    train_answers_static = train_answers
    valid_easy_answers_static = valid_easy_answers
    valid_hard_answers_static = valid_hard_answers
    test_easy_answers_static = test_easy_answers
    test_hard_answers_static = test_hard_answers


def set_idn(train_idn, valid_idn, test_idn):
    global train_idn_static, valid_idn_static, test_idn_static
    train_idn_static = train_idn
    valid_idn_static = valid_idn
    test_idn_static = test_idn


def set_helper_idn(train_helper_idn, valid_helper_idn, test_helper_idn):
    global train_helper_idn_static, valid_helper_idn_static, test_helper_idn_static
    train_helper_idn_static = train_helper_idn
    valid_helper_idn_static = valid_helper_idn
    test_helper_idn_static = test_helper_idn


class QueryDataset(Dataset):
    def __init__(self, queries, query_type, args, status):
        self.queries = list(queries)
        self.queries_num = len(queries)
        self.query_type = query_type
        self.sample_limit = args.sample_limit
        self.pre_sample = args.pre_sample
        self.sample_times = args.sample_times
        self.status = status
        self.query_neis_dict = {}
        self.anchors_neis_dict = {}
        self.multi_neg_ans = args.multi_neg_answers
        self.neg_num = 128
        self.del_ans_edge = args.del_ans_edge
        self.id_neighbor_limit = args.id_neighbor_limit
        self.model = args.model
        self.mp_model = args.mp_model
        self.id_neighbor_pre_sample = args.id_neighbor_pre_sample

        if self.status == 'train' and 'cqd' in self.model:
            self.triples = graph_static.triple_list
            random.shuffle(self.triples)
            self.cqd_counter = 0

        if args.pre_sample and args.sample_limit != [0]:
            self._sample_neighbors()
            self._check_dict_size()

    def __len__(self):
        return self.queries_num

    def __getitem__(self, index):
        return index

    def collate_fn(self, idx_list):
        if self.model == 'cqd-co' and self.status is 'train':
            if self.cqd_counter >= len(self.triples):
                self.cqd_counter = 0
                random.shuffle(self.triples)

            selected_triple = self.triples[self.cqd_counter: self.cqd_counter + len(idx_list)]
            self.cqd_counter += len(idx_list)
            chosen_queries = [(tri[0], (tri[1],)) for tri in selected_triple]
            chosen_answers = [tri[2] for tri in selected_triple]
            all_entity = set([i for i in range(graph_static.n_entity)])
            neg_answers = [random.choices(list(all_entity - train_answers_static[query]), k=self.neg_num)
                           for query in chosen_queries]

            return chosen_queries, chosen_answers, neg_answers, self.query_type, self.sample_limit, None, self.queries_num

        if 'idn-cqd' in self.model and self.status is 'train':
            if self.cqd_counter >= len(self.triples):
                self.cqd_counter = 0
                random.shuffle(self.triples)

            selected_triple = self.triples[self.cqd_counter: self.cqd_counter + len(idx_list)]
            self.cqd_counter += len(idx_list)
            chosen_queries = [(tri[0], (tri[1],)) for tri in selected_triple]
            chosen_answers = [tri[2] for tri in selected_triple]
            all_entity = set([i for i in range(graph_static.n_entity)])
            neg_answers = [random.choices(list(all_entity - train_answers_static[query]), k=self.neg_num)
                           for query in chosen_queries]

            subgraph = QueryDataset.get_subgraph(chosen_queries,
                                                 self.query_type,
                                                 self.sample_limit,
                                                 self.pre_sample,
                                                 self.query_neis_dict,
                                                 self.del_ans_edge,
                                                 self.sample_times,
                                                 self.id_neighbor_limit,
                                                 self.status,
                                                 self.mp_model,
                                                 self.id_neighbor_pre_sample)

            return chosen_queries, chosen_answers, neg_answers, self.query_type, self.sample_limit, subgraph, self.queries_num

        chosen_queries = [self.queries[i] for i in idx_list]
        if self.status is 'train':
            chosen_answers = [random.choice(list(train_answers_static[query])) for query in chosen_queries]
        elif self.status is 'val':
            chosen_answers = \
                [list(valid_easy_answers_static[query] | valid_hard_answers_static[query])
                 for query in chosen_queries]
        elif self.status is 'test':
            chosen_answers = \
                [list(test_easy_answers_static[query] | test_hard_answers_static[query])
                 for query in chosen_queries]
        else:
            chosen_answers = None

        all_entity = set([i for i in range(graph_static.n_entity)])

        if self.status is 'train':
            if self.multi_neg_ans:
                neg_answers = [random.choices(list(all_entity - train_answers_static[query]), k=self.neg_num)
                               for query in chosen_queries]
                # neg_answers = [torch.randperm(len(all_entity - train_answers_static[query])).numpy().tolist()[:10]
                #                for query in chosen_queries]
            else:
                neg_answers = [random.choice(list(all_entity - train_answers_static[query])) for query in chosen_queries]
        else:
            neg_answers = None

        if self.model not in ['gqe', 'gqe-n', 'q2b', 'q2b-n',
                              'betae', 'betae-n', 'cqd', 'cqd-co',
                              'gqe-distmult', 'gqe-complex', 'gqe-rotate', 'gqe-pairre', 'gqe-tsne']:
            subgraph = QueryDataset.get_subgraph(chosen_queries,
                                                 self.query_type,
                                                 self.sample_limit,
                                                 self.pre_sample,
                                                 self.query_neis_dict,
                                                 self.del_ans_edge,
                                                 self.sample_times,
                                                 self.id_neighbor_limit,
                                                 self.status,
                                                 self.mp_model,
                                                 self.id_neighbor_pre_sample)
        else:
            subgraph = None

        if self.status == 'train':
            return chosen_queries, chosen_answers, neg_answers, self.query_type, self.sample_limit, subgraph, self.queries_num
        else:
            return chosen_queries, chosen_answers, neg_answers, self.query_type, self.sample_limit, subgraph, self.queries_num

    def _sample_neighbors(self):
        n_anchors, n_vars = query_node_num[self.query_type]

        batch_size = 4096

        # 分组
        anchors = []
        anchors.extend(get_anchors(self.query_type, query) for query in self.queries)

        if self.status == 'val' or self.status == 'test' or not self.del_ans_edge:
            nei_sampler = [list(NeighborSampler(graph_static.pyg_kg.edge_index,
                                                node_idx=torch.tensor(anchors, dtype=torch.long),
                                                sizes=self.sample_limit,
                                                batch_size=n_anchors,
                                                shuffle=False))
                           for _ in range(self.sample_times)]

            for i, query in enumerate(self.queries):
                self.query_neis_dict[query] = [nei_sampler[j][i] for j in range(len(nei_sampler))]
            return

        edge_info = graph_static.pyg_kg.edge_index.cpu().numpy().T
        edge_info = np.insert(edge_info, 1, values=graph_static.pyg_kg.edge_type.cpu().numpy().T, axis=1)
        edge_info = np.apply_along_axis(lambda x: np.array((x[0], x[1], x[2]), dtype='i,i,i'), 1, edge_info)

        for i in range(0, len(anchors), batch_size):
            batch_up = len(self.queries) if i + batch_size > len(self.queries) else i + batch_size
            batch_queries = self.queries[i: batch_up]
            node_idx = torch.tensor(anchors[i: batch_up], dtype=torch.long)

            '''
            判断头尾和关系，使用结构数组(numpy)
            '''
            self._sample_helper(batch_queries, node_idx, edge_info, n_anchors)

    def _sample_helper(self, batch_queries, node_idx, edge_info, n_anchors):
        delete_triple_list = []
        if self.query_type == ('e', ('r',)):
            for q in batch_queries:
                for ans in train_answers_static[q]:
                    delete_triple_list.append((q[0], q[1][0], ans))
        elif self.query_type == (('e', ('r',)), ('e', ('r',))):
            for q in batch_queries:
                for ans in train_answers_static[q]:
                    delete_triple_list.append((q[0][0], q[0][1][0], ans))
                    delete_triple_list.append((q[1][0], q[1][1][0], ans))
        elif self.query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',))):
            for q in batch_queries:
                for ans in train_answers_static[q]:
                    delete_triple_list.append((q[0][0], q[0][1][0], ans))
                    delete_triple_list.append((q[1][0], q[1][1][0], ans))
                    delete_triple_list.append((q[2][0], q[2][1][0], ans))
        else:
            nei_sampler = [(list(NeighborSampler(torch.tensor(graph_static.pyg_kg.edge_index, dtype=torch.long),
                                                 node_idx=node_idx,
                                                 sizes=self.sample_limit,
                                                 batch_size=n_anchors,
                                                 shuffle=False))) for _ in range(self.sample_times)]
            for k, query in enumerate(batch_queries):
                self.query_neis_dict[query] = [nei_sampler[j][k] for j in range(len(nei_sampler))]
            return

        delete_triple_list = np.array(delete_triple_list, dtype='i,i,i')
        temp_edge_index = edge_info.copy()
        mask = np.in1d(temp_edge_index, delete_triple_list, invert=False)
        temp_edge_index[mask] = np.array((-1, -1, -1), dtype='i,i,i')
        temp_edge_index = temp_edge_index[:, np.newaxis]
        temp_edge_index = temp_edge_index.view('int').reshape(graph_static.triple_num, -1)
        temp_edge_index = np.delete(temp_edge_index, 1, axis=1).T

        nei_sampler = [(list(NeighborSampler(torch.tensor(temp_edge_index, dtype=torch.long),
                                             node_idx=node_idx,
                                             sizes=self.sample_limit,
                                             batch_size=n_anchors,
                                             shuffle=False)))
                       for _ in range(self.sample_times)]

        for k, query in enumerate(batch_queries):
            self.query_neis_dict[query] = [nei_sampler[j][k] for j in range(len(nei_sampler))]

    def _generate_graph_without_ans(self):
        pass

    def _check_dict_size(self):
        print(str(self.query_type), self.status, len(self.query_neis_dict))

    def get_subgraph(queries, query_type, sample_limit, pre_sample, query_neis_dict, del_ans_edges,
                     sample_times=1, id_neighbor_limit=10,
                     status='train', mp_model='basic', id_neighbor_pre_sample=False):
        n_anchors, n_vars = query_node_num[query_type]

        res = []
        for i, query in enumerate(queries):

            edge_data = Data(edge_index=torch.tensor([[0, 1], [1, 2]]).cuda())
            edge_data.num_nodes = 3

            if status == 'train' and mp_model != 'basic':
                neighbors = train_idn_static[query_type][query]
                # if 'i' in query_name_dict[query_type]:
                #     for k, v in neighbors.items():
                #         neighbors[k] = random.sample(v, id_neighbor_limit if len(v) > id_neighbor_limit else len(v))
                # else:
                neighbors = random.sample(neighbors,
                                              id_neighbor_limit if len(neighbors) > id_neighbor_limit else len(
                                                  neighbors))
            elif status == 'val' and mp_model != 'basic':
                neighbors = valid_idn_static[query_type][query]
                if 'u' in query_name_dict[query_type]:
                    for k, v in neighbors.items():
                        neighbors[k] = random.sample(v, id_neighbor_limit if len(v) > id_neighbor_limit else len(v))
                else:
                    neighbors = random.sample(neighbors,
                                              id_neighbor_limit if len(neighbors) > id_neighbor_limit else len(
                                              neighbors))
            elif status == 'test' and mp_model != 'basic':
                neighbors = test_idn_static[query_type][query]
                if 'u' in query_name_dict[query_type]:
                    for k, v in neighbors.items():
                        neighbors[k] = random.sample(v, id_neighbor_limit if len(v) > id_neighbor_limit else len(v))
                else:
                    neighbors = random.sample(neighbors,
                                              id_neighbor_limit if len(neighbors) > id_neighbor_limit else len(
                                              neighbors))
            else:
                neighbors = None

            if query_name_dict[query_type] is '1p':
                edge_data.p1_rel = []
                edge_data.p1_target = []
                edge_data.p1_rel.append([tup[1] for tup in neighbors] +
                                        [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p1_target.append([tup[2] for tup in neighbors] +
                                           [-1] * (id_neighbor_limit - len(neighbors)))
            elif query_name_dict[query_type] is '2p':
                edge_data.p2_rel1 = []
                edge_data.p2_rel2 = []
                edge_data.p2_target1 = []
                edge_data.p2_target2 = []
                edge_data.p2_rel1.append([tup[1] for tup in neighbors] +
                                        [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p2_target1.append([tup[2] for tup in neighbors] +
                                           [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p2_rel2.append([tup[3] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p2_target2.append([tup[4] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
            elif query_name_dict[query_type] is '3p':
                edge_data.p3_rel1 = []
                edge_data.p3_rel2 = []
                edge_data.p3_rel3 = []
                edge_data.p3_target1 = []
                edge_data.p3_target2 = []
                edge_data.p3_target3 = []
                edge_data.p3_rel1.append([tup[1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p3_target1.append([tup[2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p3_rel2.append([tup[3] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p3_target2.append([tup[4] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p3_rel3.append([tup[5] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.p3_target3.append([tup[6] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
            elif query_name_dict[query_type] is '2i':
                edge_data.i2_rel1 = []
                edge_data.i2_rel2 = []
                edge_data.i2_target1 = []
                edge_data.i2_target2 = []
                edge_data.i2_rel1.append([tup[0][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i2_target1.append([tup[0][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i2_rel2.append([tup[1][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i2_target2.append([tup[1][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
            elif query_name_dict[query_type] is '3i':
                edge_data.i3_rel1 = []
                edge_data.i3_rel2 = []
                edge_data.i3_rel3 = []
                edge_data.i3_target1 = []
                edge_data.i3_target2 = []
                edge_data.i3_target3 = []
                edge_data.i3_rel1.append([tup[0][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i3_target1.append([tup[0][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i3_rel2.append([tup[1][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i3_target2.append([tup[1][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i3_rel3.append([tup[2][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.i3_target3.append([tup[2][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
            elif query_name_dict[query_type] is 'ip':
                edge_data.ip_rel1 = []
                edge_data.ip_rel2 = []
                edge_data.ip_rel3 = []
                edge_data.ip_target1 = []
                edge_data.ip_target2 = []
                edge_data.ip_target3 = []
                edge_data.ip_rel1.append([tup[0][0][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.ip_target1.append([tup[0][0][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.ip_rel2.append([tup[0][1][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.ip_target2.append([tup[0][1][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.ip_rel3.append([tup[1][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.ip_target3.append([tup[1][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
            elif query_name_dict[query_type] is 'pi':
                edge_data.pi_rel1 = []
                edge_data.pi_rel2 = []
                edge_data.pi_rel3 = []
                edge_data.pi_target1 = []
                edge_data.pi_target2 = []
                edge_data.pi_target3 = []
                edge_data.pi_rel1.append([tup[0][1] for tup in neighbors] +
                                          [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.pi_target1.append([tup[0][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.pi_rel2.append([tup[0][3] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.pi_target2.append([tup[0][4] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.pi_rel3.append([tup[1][1] for tup in neighbors] +
                                         [-1] * (id_neighbor_limit - len(neighbors)))
                edge_data.pi_target3.append([tup[1][2] for tup in neighbors] +
                                            [-1] * (id_neighbor_limit - len(neighbors)))
            elif '2u' in query_name_dict[query_type]:
                edge_data.u2_rel = []
                edge_data.u2_target = []
                edge_data.u2_rel.append([tup[1] for tup in neighbors[query[0]]] +
                                        [-1] * (id_neighbor_limit - len(neighbors[query[0]])))
                edge_data.u2_rel.append([tup[1] for tup in neighbors[query[1]]] +
                                        [-1] * (id_neighbor_limit - len(neighbors[query[1]])))
                edge_data.u2_target.append([tup[2] for tup in neighbors[query[0]]] +
                                           [-1] * (id_neighbor_limit - len(neighbors[query[0]])))
                edge_data.u2_target.append([tup[2] for tup in neighbors[query[1]]] +
                                           [-1] * (id_neighbor_limit - len(neighbors[query[1]])))
            elif 'up' in query_name_dict[query_type]:
                edge_data.up_rel_1 = []
                edge_data.up_rel_2 = []
                edge_data.up_target_1 = []
                edge_data.up_target_2 = []

                key_0 = (query[0][0], query[1])
                key_1 = (query[0][1], query[1])

                edge_data.up_rel_1.append([tup[1] for tup in neighbors[key_0]] +
                                          [-1] * (id_neighbor_limit - len(neighbors[key_0])))
                edge_data.up_rel_1.append([tup[1] for tup in neighbors[key_1]] +
                                          [-1] * (id_neighbor_limit - len(neighbors[key_1])))
                edge_data.up_rel_2.append([tup[3] for tup in neighbors[key_0]] +
                                          [-1] * (id_neighbor_limit - len(neighbors[key_0])))
                edge_data.up_rel_2.append([tup[3] for tup in neighbors[key_1]] +
                                          [-1] * (id_neighbor_limit - len(neighbors[key_1])))

                edge_data.up_target_1.append([tup[2] for tup in neighbors[key_0]] +
                                             [-1] * (id_neighbor_limit - len(neighbors[key_0])))
                edge_data.up_target_1.append([tup[2] for tup in neighbors[key_1]] +
                                             [-1] * (id_neighbor_limit - len(neighbors[key_1])))
                edge_data.up_target_2.append([tup[4] for tup in neighbors[key_0]] +
                                             [-1] * (id_neighbor_limit - len(neighbors[key_0])))
                edge_data.up_target_2.append([tup[4] for tup in neighbors[key_1]] +
                                             [-1] * (id_neighbor_limit - len(neighbors[key_1])))

            res.append(edge_data)

        return Batch.from_data_list(res).cuda()


def make_data_iterator(data_loader):
    iterator = iter(data_loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            continue


def get_queries_iterator(queries, args, query_type, status):
    dataset = QueryDataset(queries, query_type, args, status)
    loader = DataLoader(dataset, args.batch_size, shuffle=True,
                        collate_fn=dataset.collate_fn)
    return make_data_iterator(loader)


def load_id_neighbor_online(query, status, query_type, id_neighbor_num, heads, rels, tails):
    global train_idn_static, valid_idn_static, test_idn_static

    temp_dict = defaultdict(set)

    if query_name_dict[query_type] == '1p':
        anchor = get_anchors(query_type, query)[0]
        chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
        chosen_heads_index = np.random.choice(chosen_heads_index,
                                                id_neighbor_num
                                                if chosen_heads_index.shape[0] > id_neighbor_num
                                                else chosen_heads_index.shape[0])
        chosen_tails = tails[chosen_heads_index]
        chosen_rels = rels[chosen_heads_index]

        for i in range(chosen_tails.shape[0]):
            q_neighbor = (anchor, (chosen_rels[i], ))
            ans = chosen_tails[i]
            if q_neighbor == query and ans in train_answers_static[query]:
                continue
            temp_dict[query].add((anchor, chosen_rels[i], ans))

    elif query_name_dict[query_type] == '2p':
        anchor = get_anchors(query_type, query)[0]
        chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
        chosen_heads_index = np.random.choice(chosen_heads_index,
                                              id_neighbor_num
                                              if chosen_heads_index.shape[0] > id_neighbor_num
                                              else chosen_heads_index.shape[0])
        chosen_tails_1 = tails[chosen_heads_index]
        chosen_rels_1 = rels[chosen_heads_index]

        for i in range(chosen_tails_1.shape[0]):
            temp_rel = chosen_rels_1[i]
            chosen_tails_1_index = np.squeeze(np.argwhere(heads == chosen_tails_1[i]).T, axis=0)
            chosen_tails_1_index = np.random.choice(chosen_tails_1_index,
                                                    1
                                                    if chosen_tails_1_index.shape[0] > 1
                                                    else chosen_tails_1_index.shape[0])
            chosen_tails_2 = tails[chosen_tails_1_index]
            chosen_rels_2 = rels[chosen_tails_1_index]
            for j in range(chosen_tails_2.shape[0]):
                q_neighbor = (anchor, (temp_rel, chosen_rels_2[j]))
                ans = chosen_tails_2[j]
                if q_neighbor == query and ans in train_answers_static[query]:
                    continue
                temp_dict[query].add((anchor, temp_rel, chosen_tails_1[i], chosen_rels_2[j], ans))

    elif query_name_dict[query_type] == '3p':
        anchor = get_anchors(query_type, query)[0]
        chosen_heads_index = np.squeeze(np.argwhere(heads == anchor).T, axis=0)
        chosen_heads_index = np.random.choice(chosen_heads_index,
                                              id_neighbor_num
                                              if chosen_heads_index.shape[0] > id_neighbor_num
                                              else chosen_heads_index.shape[0])
        chosen_tails_1 = tails[chosen_heads_index]
        chosen_rels_1 = rels[chosen_heads_index]

        for i in range(chosen_tails_1.shape[0]):
            temp_rel_1 = chosen_rels_1[i]
            chosen_tails_1_index = np.squeeze(np.argwhere(heads == chosen_tails_1[i]).T, axis=0)
            chosen_tails_1_index = np.random.choice(chosen_tails_1_index,
                                                    1
                                                    if chosen_tails_1_index.shape[0] > 1
                                                    else chosen_tails_1_index.shape[0])
            chosen_tails_2 = tails[chosen_tails_1_index]
            chosen_rels_2 = rels[chosen_tails_1_index]
            for j in range(chosen_tails_2.shape[0]):
                temp_rel_2 = chosen_rels_2[j]
                chosen_tails_2_index = np.squeeze(np.argwhere(heads == chosen_tails_2[j]).T, axis=0)
                chosen_tails_2_index = np.random.choice(chosen_tails_2_index,
                                                        1
                                                        if chosen_tails_2_index.shape[0] > 1
                                                        else chosen_tails_2_index.shape[0])
                chosen_tails_3 = tails[chosen_tails_2_index]
                chosen_rels_3 = rels[chosen_tails_2_index]
                for k in range(chosen_tails_3.shape[0]):
                    q_neighbor = (anchor, (temp_rel_1, temp_rel_2, chosen_rels_3[k]))
                    ans = chosen_tails_3[k]
                    if q_neighbor == query and ans in train_answers_static[query]:
                        continue
                    temp_dict[query].add((anchor, temp_rel_1, chosen_tails_1[i],
                                        temp_rel_2, chosen_tails_2[j],
                                        chosen_rels_3[k], ans))

    elif query_name_dict[query_type] == '2i':
        anchor = get_anchors(query_type, query)
        chosen_heads_index_1 = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
        chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)

        chosen_tails_1 = tails[chosen_heads_index_1]
        chosen_rels_1 = rels[chosen_heads_index_1]

        chosen_tails_2 = tails[chosen_heads_index_2]
        chosen_rels_2 = rels[chosen_heads_index_2]

        ans = np.intersect1d(chosen_tails_1, chosen_tails_2)

        for i in range(ans.shape[0]):
            rel1_list = chosen_rels_1[chosen_tails_1 == ans[i]]
            rel2_list = chosen_rels_2[chosen_tails_2 == ans[i]]
            for rel1 in rel1_list:
                for rel2 in rel2_list:
                    q_neighbor = ((anchor[0], (rel1,)), (anchor[1], (rel2,)))
                    a = ans[i]
                    if q_neighbor == query and a in train_answers_static[query]:
                        continue
                    temp_dict[query].add(((anchor[0], rel1, a), (anchor[1], rel2, a)))

        temp_dict[query] = random.sample(temp_dict[query],
                                         id_neighbor_num
                                         if len(temp_dict[query]) > id_neighbor_num
                                         else len(temp_dict[query]))

    elif query_name_dict[query_type] == '3i':
        anchor = get_anchors(query_type, query)
        chosen_heads_index_1 = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
        chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)
        chosen_heads_index_3 = np.squeeze(np.argwhere(heads == anchor[2]).T, axis=0)

        chosen_tails_1 = tails[chosen_heads_index_1]
        chosen_rels_1 = rels[chosen_heads_index_1]

        chosen_tails_2 = tails[chosen_heads_index_2]
        chosen_rels_2 = rels[chosen_heads_index_2]

        chosen_tails_3 = tails[chosen_heads_index_3]
        chosen_rels_3 = rels[chosen_heads_index_3]

        ans = reduce(np.intersect1d, (chosen_tails_1, chosen_tails_2, chosen_tails_3))

        for i in range(ans.shape[0]):
            rel1_list = chosen_rels_1[chosen_tails_1 == ans[i]]
            rel2_list = chosen_rels_2[chosen_tails_2 == ans[i]]
            rel3_list = chosen_rels_3[chosen_tails_3 == ans[i]]

            for rel1 in rel1_list:
                for rel2 in rel2_list:
                    for rel3 in rel3_list:
                        q_neighbor = ((anchor[0], (rel1,)), (anchor[1], (rel2,)), (anchor[2], (rel3,)))
                        a = ans[i]
                        if q_neighbor == query and a in train_answers_static[query]:
                            continue
                        temp_dict[query].add(((anchor[0], rel1, ans[i]),
                                            (anchor[1], rel2, ans[i]),
                                            (anchor[2], rel3, ans[i])))

        temp_dict[query] = random.sample(temp_dict[query],
                                         id_neighbor_num
                                         if len(temp_dict[query]) > id_neighbor_num
                                         else len(temp_dict[query]))

    elif query_name_dict[query_type] == 'ip':
        anchor = get_anchors(query_type, query)
        chosen_heads_index_1 = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
        chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)

        chosen_tails_1 = tails[chosen_heads_index_1]
        chosen_rels_1 = rels[chosen_heads_index_1]

        chosen_tails_2 = tails[chosen_heads_index_2]
        chosen_rels_2 = rels[chosen_heads_index_2]

        ans = np.intersect1d(chosen_tails_1, chosen_tails_2)

        for i in range(ans.shape[0]):
            rel1_list = chosen_rels_1[chosen_tails_1 == ans[i]]
            rel2_list = chosen_rels_2[chosen_tails_2 == ans[i]]
            for rel1 in rel1_list:
                for rel2 in rel2_list:
                    a = ans[i]
                    chosen_heads_index = np.squeeze(np.argwhere(heads == a).T, axis=0)
                    chosen_heads_index = np.random.choice(chosen_heads_index,
                                                          id_neighbor_num
                                                          if chosen_heads_index.shape[0] > id_neighbor_num
                                                          else chosen_heads_index.shape[0])
                    chosen_tails = tails[chosen_heads_index]
                    chosen_rels = rels[chosen_heads_index]

                    for j in range(chosen_tails.shape[0]):
                        q_neighbor = (((anchor[0], (rel1,)), (anchor[1], (rel2,))), (chosen_rels[j], ))
                        real_ans = chosen_tails[j]
                        if q_neighbor == query and real_ans in train_answers_static[query]:
                            continue
                        temp_dict[query].add((((anchor[0], rel1, a), (anchor[1], rel2, a)),
                                                (a, chosen_rels[j], real_ans)))

        temp_dict[query] = random.sample(temp_dict[query],
                                         id_neighbor_num
                                         if len(temp_dict[query]) > id_neighbor_num
                                         else len(temp_dict[query]))

    elif query_name_dict[query_type] == 'pi':
        anchor = get_anchors(query_type, query)
        chosen_heads_index = np.squeeze(np.argwhere(heads == anchor[0]).T, axis=0)
        chosen_heads_index = np.random.choice(chosen_heads_index,
                                              id_neighbor_num
                                              if chosen_heads_index.shape[0] > id_neighbor_num
                                              else chosen_heads_index.shape[0])
        chosen_tails = tails[chosen_heads_index]
        chosen_rels = rels[chosen_heads_index]

        for i in range(chosen_tails.shape[0]):
            first_rel = chosen_rels[i]

            chosen_heads_index_1 = np.squeeze(np.argwhere(heads == chosen_tails[i]).T, axis=0)
            chosen_heads_index_2 = np.squeeze(np.argwhere(heads == anchor[1]).T, axis=0)

            chosen_tails_1 = tails[chosen_heads_index_1]
            chosen_rels_1 = rels[chosen_heads_index_1]

            chosen_tails_2 = tails[chosen_heads_index_2]
            chosen_rels_2 = rels[chosen_heads_index_2]

            ans = np.intersect1d(chosen_tails_1, chosen_tails_2)

            for j in range(ans.shape[0]):
                rel1_list = chosen_rels_1[chosen_tails_1 == ans[j]]
                rel2_list = chosen_rels_2[chosen_tails_2 == ans[j]]
                for rel1 in rel1_list:
                    for rel2 in rel2_list:
                        q_neighbor = ((anchor[0], (first_rel, rel1)), (anchor[1], (rel2,)))
                        a = ans[j]
                        if q_neighbor == query and a in train_answers_static[query]:
                            continue
                        temp_dict[query].add(((anchor[0], first_rel, chosen_tails[i], rel1, a), (anchor[1], rel2, a)))

        temp_dict[query] = random.sample(temp_dict[query],
                                         id_neighbor_num
                                         if len(temp_dict[query]) > id_neighbor_num
                                         else len(temp_dict[query]))

    if status == 'train':
        train_idn_static = defaultdict(dict)
        train_idn_static[query_type] = temp_dict
    elif status == 'val':
        valid_idn_static = defaultdict(dict)
        valid_idn_static[query_type] = temp_dict
    else:
        test_idn_static = defaultdict(dict)
        test_idn_static[query_type] = temp_dict

