from collections import defaultdict
from graph import Graph
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
    train_queries = None
    train_answers = None
    test_answers = None
    valid_answers = None

    train_queries = pickle.load(open(os.path.join(args.data_dir, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_dir, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_dir, "valid-queries.pkl"), 'rb'))

    valid_hard_answers = pickle.load(open(os.path.join(args.data_dir, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_dir, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_dir, "test-queries.pkl"), 'rb'))
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
    # 最后一个是query的结果
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
                            'train_queries_idneighbor_new.pkl')
    generate_id_neighbor_helper(heads, rels, tails, id_neighbor_num, dir, valid_queries, train_answers, query_name_dict,
                            'valid_queries_idneighbor_new.pkl', False)
    generate_id_neighbor_helper(heads, rels, tails, id_neighbor_num, dir, test_queries, train_answers, query_name_dict,
                            'test_queries_idneighbor_new.pkl', False)

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

                # 生成邻居信息，类似Neighbor_Sampler
            elif query_name_dict[query_type] == '2p':
                # 先选出所有的2p路径，再随机挑选(耗时过长)
                # 剪枝
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
                    # print('list_2p', len(list_2p))
                    count_min += 1
            elif query_name_dict[query_type] == '3p':
                # 剪枝
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
                    # print('list_3p', len(list_3p))
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
                                real_ans = chosen_tails[j]
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
                                a = ans[j]
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
    # 可以考虑更改var的个数，考虑u与否是不一样的，暂时先按照少的考虑
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

    @staticmethod
    def get_subgraph_old(queries, query_type, sample_limit, pre_sample, query_neis_dict, del_ans_edges,
                         sample_times=1, id_neighbor_limit=10,
                         status='train', mp_model='basic', id_neighbor_pre_sample=False):
        n_anchors, n_vars = query_node_num[query_type]

        anchors = []
        anchors.extend(get_anchors(query_type, query) for query in queries)

        # pre_sample：可以把train中的所有anchor在读取query的时候先采样
        if not sample_limit == [0]:
            if pre_sample:
                nei_sampler = [query_neis_dict[q][random.randint(0, sample_times) - 1] for q in queries]
            else:
                if del_ans_edges:
                    delete_triple_list = []
                    if query_type == ('e', ('r',)):
                        for q in queries:
                            for ans in train_answers_static[q]:
                                delete_triple_list.append((q[0], q[1][0], ans))
                    elif query_type == (('e', ('r',)), ('e', ('r',))):
                        for q in queries:
                            for ans in train_answers_static[q]:
                                delete_triple_list.append((q[0][0], q[0][1][0], ans))
                                delete_triple_list.append((q[1][0], q[1][1][0], ans))
                    elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',))):
                        for q in queries:
                            for ans in train_answers_static[q]:
                                delete_triple_list.append((q[0][0], q[0][1][0], ans))
                                delete_triple_list.append((q[1][0], q[1][1][0], ans))
                                delete_triple_list.append((q[2][0], q[2][1][0], ans))
                    else:
                        nei_sampler = list(NeighborSampler(torch.tensor(graph_static.pyg_kg.edge_index, dtype=torch.long),
                                                           node_idx=torch.tensor(anchors, dtype=torch.long),
                                                           sizes=sample_limit,
                                                           batch_size=n_anchors,
                                                           shuffle=False))

                    edge_info = graph_static.pyg_kg.edge_index.cpu().numpy().T
                    edge_info = np.insert(edge_info, 1, values=graph_static.pyg_kg.edge_type.cpu().numpy().T, axis=1)
                    edge_info = np.apply_along_axis(lambda x: np.array((x[0], x[1], x[2]), dtype='i,i,i'), 1, edge_info)

                    if len(delete_triple_list) > 0:
                        delete_triple_list = np.array(delete_triple_list, dtype='i,i,i')
                        temp_edge_index = edge_info.copy()
                        mask = np.in1d(temp_edge_index, delete_triple_list, invert=False)
                        temp_edge_index[mask] = np.array((-1, -1, -1), dtype='i,i,i')
                        temp_edge_index = temp_edge_index[:, np.newaxis]
                        temp_edge_index = temp_edge_index.view('int').reshape(graph_static.triple_num, -1)
                        temp_edge_index = np.delete(temp_edge_index, 1, axis=1).T

                        nei_sampler = list(NeighborSampler(torch.tensor(temp_edge_index, dtype=torch.long),
                                                           node_idx=torch.tensor(anchors, dtype=torch.long),
                                                           sizes=sample_limit,
                                                           batch_size=n_anchors,
                                                           shuffle=False))
                else:
                    nei_sampler = list(NeighborSampler(graph_static.pyg_kg.edge_index, node_idx=torch.tensor(anchors, dtype=torch.long),
                                                       sizes=sample_limit, batch_size=n_anchors,
                                                       shuffle=False))

        else:
            nei_sampler = None

        def get_all_nei_id(query_type, idn_triple, node_id):
            for triple in idn_triple:
                if query_type == ('e', ('r',)):
                    variables = [triple[2]]
                elif query_type == ('e', ('r', 'r')):
                    variables = [triple[2], triple[4]]
                elif query_type == ('e', ('r', 'r', 'r')):
                    variables = [triple[2], triple[4], triple[6]]
                elif query_type == (('e', ('r',)), ('e', ('r',))):
                    variables = [triple[0][2]]
                elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',))):
                    variables = [triple[0][2]]
                elif query_type == (('e', ('r', 'r')), ('e', ('r',))):
                    variables = [triple[0][2], triple[0][4]]
                elif query_type == ((('e', ('r',)), ('e', ('r',))), ('r',)):
                    variables = [triple[0][0][2], triple[1][2]]
                elif query_type == (('e', ('r',)), ('e', ('r',)), ('u',)):
                    pass
                elif query_type == ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)):
                    pass
                else:
                    variables = []
                for var in variables:
                    if var not in node_id.numpy().tolist():
                        node_id = torch.cat([node_id, torch.tensor([var], dtype=torch.long)])

            return node_id

        def extend_nei_edge_index(node_id, idn_triple, edge_index, query, edge_id, query_type, head_id=0):
            edge_index = edge_index.numpy().tolist()
            node_id = node_id.numpy().tolist()
            for triple in idn_triple:
                if query_type == ('e', ('r',)):
                    edge_index[0].append(head_id)
                    edge_index[1].append(node_id.index(triple[2]))
                    edge_id.append(triple[1])
                elif query_type == ('e', ('r', 'r')):
                    edge_index[0].extend([0, node_id.index(triple[2])])
                    edge_index[1].extend([node_id.index(triple[2]), node_id.index(triple[4])])
                    edge_id.extend([triple[1], triple[3]])
                elif query_type == ('e', ('r', 'r', 'r')):
                    edge_index[0].extend([0, node_id.index(triple[2]), node_id.index(triple[4])])
                    edge_index[1].extend([node_id.index(triple[2]), node_id.index(triple[4]), node_id.index(triple[6])])
                    edge_id.extend([triple[1], triple[3], triple[5]])
                elif query_type == (('e', ('r',)), ('e', ('r',))):
                    edge_index[0].extend([0, 1])
                    edge_index[1].extend([node_id.index(triple[0][2])] * 2)
                    edge_id.extend([triple[0][1], triple[1][1]])
                elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',))):
                    edge_index[0].extend([0, 1, 2])
                    edge_index[1].extend([node_id.index(triple[0][2])] * 3)
                    edge_id.extend([triple[0][1], triple[1][1], triple[2][1]])
                elif query_type == (('e', ('r', 'r')), ('e', ('r',))):
                    variables = [node_id.index(triple[0][2]), node_id.index(triple[0][4])]
                    edge_index[0].extend([0, variables[0], 1])
                    edge_index[1].extend([variables[0], variables[1], variables[1]])
                    edge_id.extend([triple[0][1], triple[0][3], triple[1][1]])
                elif query_type == ((('e', ('r',)), ('e', ('r',))), ('r',)):
                    variables = [node_id.index(triple[0][0][2]), node_id.index(triple[1][2])]
                    edge_index[0].extend([0, 1, variables[0]])
                    edge_index[1].extend([variables[0], variables[0], variables[1]])
                    edge_id.extend([triple[0][0][1], triple[0][1][1], triple[1][1]])
                else:
                    variables = []
            return edge_index, edge_id

        def deal_sampler(nei_info, query, status):
            n_anchors, n_vars = query_node_num[query_type]

            if not id_neighbor_pre_sample:
                id_neighbor_num = id_neighbor_limit

                heads = np.array(graph_static.edge_index[0])
                rels = np.array(graph_static.edge_type)
                tails = np.array(graph_static.edge_index[1])

                load_id_neighbor_online(query, status, query_type, id_neighbor_num, heads, rels, tails)

            helper_triple = None

            if status == 'train' and mp_model != 'basic':
                idn_triple = train_idn_static[query_type][query]
                idn_triple = random.sample(idn_triple,
                                           id_neighbor_limit if len(idn_triple) > id_neighbor_limit else len(idn_triple))

            elif status == 'val' and mp_model != 'basic':
                idn_triple = valid_idn_static[query_type][query]
                if 'u' in query_name_dict[query_type]:
                    for k, v in idn_triple.items():
                        idn_triple[k] = random.sample(v, id_neighbor_limit if len(v) > id_neighbor_limit else len(v))
                else:
                    idn_triple = random.sample(idn_triple,
                                                id_neighbor_limit if len(idn_triple) > id_neighbor_limit else len(
                                                idn_triple))

            elif status == 'test' and mp_model != 'basic':
                idn_triple = test_idn_static[query_type][query]
                if 'u' in query_name_dict[query_type]:
                    for k, v in idn_triple.items():
                        idn_triple[k] = random.sample(v, id_neighbor_limit if len(v) > id_neighbor_limit else len(v))
                else:
                    idn_triple = random.sample(idn_triple,
                                                id_neighbor_limit if len(idn_triple) > id_neighbor_limit else len(
                                                idn_triple))

            else:
                idn_triple = None

            if nei_info is None:
                q_edge_index, var_id, var_node = get_edge_index(query_type, query, n_anchors)
                q_edge_index = torch.tensor(q_edge_index, dtype=torch.long)
                edge_id = var_id
                node_id = torch.tensor(get_anchors(query_type, query), dtype=torch.long)
                var_tensor = torch.tensor([random.choice(edges) + graph_static.n_entity for edges in var_node],
                                          dtype=torch.long)
                all_node_id = torch.cat([node_id, var_tensor])
                all_node_id_real = node_id.numpy().tolist() + var_node

                edge_data = Data(edge_index=q_edge_index)
                edge_data.edge_type = torch.tensor(edge_id, dtype=torch.long)
                edge_data.node_type = all_node_id
                edge_data.node_type_list = all_node_id_real

                edge_data.query_edge_index = q_edge_index
                edge_data.query_edge_type = torch.tensor(var_id, dtype=torch.long)
                edge_data.query_node_type = torch.cat([node_id[: n_anchors], var_tensor])

                edge_data.n_anchors = n_anchors
                edge_data.n_vars = n_vars
                edge_data.num_nodes = torch.max(q_edge_index).item() + 1
                edge_data.n_neis = edge_data.num_nodes - n_vars - n_anchors
                return edge_data
            else:
                n_anchors, node_id, edge_info = nei_info
                if len(sample_limit) != 1:
                    nei_edge_index = torch.cat([info.edge_index for info in edge_info[::-1]], 1)
                    edge_id = torch.cat([info.e_id for info in edge_info[::-1]]).numpy().tolist()
                    edge_id = list(map(lambda x: graph_static.edge_type[x], edge_id))
                else:
                    nei_edge_index = edge_info.edge_index
                    edge_id = edge_info.e_id.numpy().tolist()
                    edge_id = list(map(lambda x: graph_static.edge_type[x], edge_id))

                neiA_edge_index = nei_edge_index
                neiA_edge_type = torch.tensor(edge_id, dtype=torch.long)

                if idn_triple is not None and 'u' not in query_name_dict[query_type]:
                    # 1. get all node id
                    # 2. extend node_id
                    node_id = get_all_nei_id(query_type, idn_triple, node_id)
                    # 3. extend nei_edge_index
                    # 4. extend edge_id
                    nei_edge_index, edge_id = extend_nei_edge_index(node_id, idn_triple, nei_edge_index, query, edge_id, query_type)
                    nei_edge_index = torch.tensor(nei_edge_index, dtype=torch.long, requires_grad=False)

                neiB_edge_index = nei_edge_index[:, neiA_edge_index.shape[1]:]
                neiB_edge_type = torch.tensor(edge_id[len(neiA_edge_type):], dtype=torch.long)

                neiC_edge_index = None
                neiC_edge_type = None

                q_edge_index, var_id, var_node = get_edge_index(query_type, query, len(node_id))
                edge_index = torch.cat([torch.tensor(nei_edge_index, dtype=torch.long),
                                        torch.tensor(q_edge_index, dtype=torch.long)],
                                       1)
                nei_edge_id = torch.tensor(edge_id, dtype=torch.long)
                edge_id.extend(var_id)

                var_tensor = torch.tensor([random.choice(edges) + graph_static.n_entity for edges in var_node], dtype=torch.long)
                all_node_id = torch.cat([node_id, var_tensor])

                all_node_id_real = node_id.numpy().tolist() + var_node

                edge_data = Data(edge_index=edge_index)
                edge_data.edge_type = torch.tensor(edge_id, dtype=torch.long)
                edge_data.node_type = all_node_id
                edge_data.node_type_list = all_node_id_real

                edge_data.query_edge_index = torch.tensor(q_edge_index, dtype=torch.long)
                edge_data.query_edge_type = torch.tensor(var_id, dtype=torch.long)
                edge_data.query_node_type = torch.cat([node_id[: n_anchors], var_tensor])

                edge_data.nei_edge_index = nei_edge_index
                edge_data.nei_edge_type = nei_edge_id
                edge_data.nei_node_type = None

                edge_data.neiA_edge_index = neiA_edge_index
                edge_data.neiA_edge_type = neiA_edge_type
                edge_data.neiA_node_type = None

                edge_data.neiB_edge_index = neiB_edge_index
                edge_data.neiB_edge_type = neiB_edge_type
                edge_data.neiB_node_type = None

                edge_data.neiC_edge_index = neiC_edge_index
                edge_data.neiC_edge_type = neiC_edge_type
                edge_data.neiC_node_type = None

                edge_data.neiC_target_list = neiC_edge_index[1, :].cpu().numpy().tolist() \
                    if neiC_edge_index is not None else None

                step = len(var_id)
                B_list = neiB_edge_index[1, :].cpu().numpy().tolist()
                B_edges = neiB_edge_type.cpu().numpy().tolist()
                edge_data.neiB_target_aslist = [B_list[i] for i in range(step - 1, len(B_list), step)]
                edge_data.neiB_target_aslist.extend(
                    [-1] * (id_neighbor_limit - len(edge_data.neiB_target_aslist)))
                edge_data.neiB_nodes = []
                edge_data.neiB_edges = []
                for ii in range(step - 1):
                    edge_data.neiB_nodes.append([B_list[j] for j in range(ii, len(B_list), step)])
                    edge_data.neiB_nodes[-1].extend(
                        [-1] * (id_neighbor_limit - len(edge_data.neiB_nodes[-1])))
                for ii in range(step):
                    edge_data.neiB_edges.append([B_edges[j] for j in range(ii, len(B_edges), step)])
                    edge_data.neiB_edges[-1].extend(
                        [-1] * (id_neighbor_limit - len(edge_data.neiB_edges[-1])))

                if '2u' in query_name_dict[query_type]:
                    edge_data.u2_rel = []
                    edge_data.u2_target = []
                    edge_data.u2_rel.append([tup[1] for tup in idn_triple[query[0]]] +
                                            [-1] * (id_neighbor_limit - len(idn_triple[query[0]])))
                    edge_data.u2_rel.append([tup[1] for tup in idn_triple[query[1]]] +
                                            [-1] * (id_neighbor_limit - len(idn_triple[query[1]])))
                    edge_data.u2_target.append([tup[2] for tup in idn_triple[query[0]]] +
                                               [-1] * (id_neighbor_limit - len(idn_triple[query[0]])))
                    edge_data.u2_target.append([tup[2] for tup in idn_triple[query[1]]] +
                                               [-1] * (id_neighbor_limit - len(idn_triple[query[1]])))

                elif 'up' in query_name_dict[query_type]:
                    edge_data.up_rel_1 = []
                    edge_data.up_rel_2 = []
                    edge_data.up_target_1 = []
                    edge_data.up_target_2 = []

                    key_0 = (query[0][0], query[1])
                    key_1 = (query[0][1], query[1])

                    edge_data.up_rel_1.append([tup[1] for tup in idn_triple[key_0]] +
                                              [-1] * (id_neighbor_limit - len(idn_triple[key_0])))
                    edge_data.up_rel_1.append([tup[1] for tup in idn_triple[key_1]] +
                                              [-1] * (id_neighbor_limit - len(idn_triple[key_1])))
                    edge_data.up_rel_2.append([tup[3] for tup in idn_triple[key_0]] +
                                              [-1] * (id_neighbor_limit - len(idn_triple[key_0])))
                    edge_data.up_rel_2.append([tup[3] for tup in idn_triple[key_1]] +
                                              [-1] * (id_neighbor_limit - len(idn_triple[key_1])))

                    edge_data.up_target_1.append([tup[2] for tup in idn_triple[key_0]] +
                                                 [-1] * (id_neighbor_limit - len(idn_triple[key_0])))
                    edge_data.up_target_1.append([tup[2] for tup in idn_triple[key_1]] +
                                                 [-1] * (id_neighbor_limit - len(idn_triple[key_1])))
                    edge_data.up_target_2.append([tup[4] for tup in idn_triple[key_0]] +
                                                 [-1] * (id_neighbor_limit - len(idn_triple[key_0])))
                    edge_data.up_target_2.append([tup[4] for tup in idn_triple[key_1]] +
                                                 [-1] * (id_neighbor_limit - len(idn_triple[key_1])))

                edge_data.helper_edges = []
                edge_data.helper_targets = []

                edge_data.n_anchors = n_anchors
                edge_data.n_vars = n_vars
                edge_data.num_nodes = torch.max(edge_index).item() + 1
                edge_data.n_neis = edge_data.num_nodes - n_vars - n_anchors
                return edge_data

        res = []
        for i, query in enumerate(queries):
            res.append(deal_sampler(nei_sampler[i] if nei_sampler is not None else None, query, status=status))

        all_sub_graph = Batch.from_data_list(res).cuda()
        return all_sub_graph

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

