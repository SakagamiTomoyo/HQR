import pickle
import os
from collections import defaultdict
import logging
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import random
import umap
import numpy as np
import time
# from .data_loader import generate_id_neighbor as gin

def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging


def answers_with_query_type(data_dir):
    train_queries = pickle.load(open(os.path.join(data_dir, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(data_dir, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(data_dir, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(data_dir, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(data_dir, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(data_dir, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(data_dir, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(data_dir, "test-easy-answers.pkl"), 'rb'))

    train_answers_with_type = defaultdict(dict)
    valid_answers_with_type = defaultdict(dict)
    test_answers_with_type = defaultdict(dict)

    valid_answers = defaultdict(dict)
    test_answers = defaultdict(dict)

    for query_type in valid_queries:
        for query in valid_queries[query_type]:
            valid_answers[query] = valid_easy_answers[query] | valid_hard_answers[query]

    for query_type in test_queries:
        for query in test_queries[query_type]:
            test_answers[query] = test_easy_answers[query] | test_hard_answers[query]

    pickle.dump(valid_answers, open(os.path.join(data_dir, "valid_answers.pkl"), 'wb'))
    pickle.dump(test_answers, open(os.path.join(data_dir, "test_answers.pkl"), 'wb'))

    pass


def load_id_neighbor(data_dir):
    train_idn = None
    valid_idn = None
    test_idn = None

    train_idn = pickle.load(open(os.path.join(data_dir, "train_queries_idneighbor_new.pkl"), 'rb'))
    valid_idn = pickle.load(open(os.path.join(data_dir, "valid_queries_idneighbor_new.pkl"), 'rb'))
    test_idn = pickle.load(open(os.path.join(data_dir, "test_queries_idneighbor_new.pkl"), 'rb'))

    print('loaded')

    return train_idn, valid_idn, test_idn


def load_helper_neighbor(data_dir):
    # train_idn = None
    train_idn = pickle.load(open(os.path.join(data_dir, "train_helper.pkl"), 'rb'))
    valid_idn = pickle.load(open(os.path.join(data_dir, "valid_helper.pkl"), 'rb'))
    test_idn = pickle.load(open(os.path.join(data_dir, "test_helper.pkl"), 'rb'))
    return train_idn, valid_idn, test_idn


def imscatter2D(x, y, ax=None, zoom=1, color='y', text='', size=5):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        # plt.plot(x0, y0, color + '' ' .', markersize=size)
        plt.plot(x0, y0, color, markersize=size)
        plt.text(x0, y0, text)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


op = None


def draw(features, query_type, chosen_queries, easy_answers, hard_answers, entity_embedding):
    global op

    colors = ['g', 'b', 'r', 'm', 'k', 'c', 'y']

    # if op is None:
    #     print('build TSNE model and transform')
    #     tsne = TSNE(n_components=2, init='pca', perplexity=30)
    #     tsne.fit(entity_embedding.cpu())
    #     output = tsne.embedding_
    #     op = output
    # else:
    #     output = op

    # tsne = TSNE(n_components=2, init='pca', perplexity=30)
    # tsne.fit(entity_embedding.cpu())
    # output = tsne.embedding_

    print(chosen_queries[0])

    features = torch.cat(features).reshape(len(features), -1)
    entity_embedding = torch.cat([entity_embedding, features], dim=0)
    backgroundNum = 200
    n_entity = entity_embedding.shape[0]
    new_embedding = torch.cat([entity_embedding[list(hard_answers[chosen_queries[0]])], features], dim=0)
    new_embedding = torch.cat([entity_embedding[random.sample(range(0, n_entity), backgroundNum)], new_embedding], )

    # new_embedding = entity_embedding

    reducer = umap.UMAP(random_state=42)
    output = reducer.fit_transform(new_embedding.cpu())

    # 太慢了
    # Y = tsne.fit_transform(entity_embedding.cpu())

    # Y = tsne.fit_transform(entity_embedding[-3:].cpu())

    # Y = tsne.fit(torch.cat([entity_embedding[-3:], entity_embedding[list(hard_answers[chosen_queries[0]])]], dim=0).cpu())

    # ans_Y = tsne.fit_transform(list(hard_answers[chosen_queries[0]]))

    print('draw pic')

    fig, ax = plt.subplots()
    fig.set_size_inches(21.6, 14.4)
    plt.axis('off')

    # num, dim = entity_embedding.shape

    # for i in range(features.shape[0]):
    #     imscatter2D(Y[num - i - 1, 0], Y[num - i - 1, 1], ax, 1, colors[i], text=str(features.shape[0] - i))

    # imscatter2D(output[:, 0], output[:, 1], ax, 1, 'r.', text='')
    # imscatter2D(output[-3, 0], output[-3, 1], ax, 1, 'c.', text='', size=12)
    # imscatter2D(output[-2, 0], output[-2, 1], ax, 1, 'k.', text='', size=12)
    # imscatter2D(output[-1, 0], output[-1, 1], ax, 1, 'k*', text='', size=9)
    # imscatter2D(output[-2, 0], output[-2, 1], ax, 1, colors[-4], text='qnew',  size=15))
    # imscatter2D(output[list(hard_answers[chosen_queries[0]]), 0], output[list(hard_answers[chosen_queries[0]]), 1], ax,
    #             1, 'r.', text='', size=12)

    # 1p
    # imscatter2D(output[:backgroundNum, 0], output[:backgroundNum, 1], ax, 1, 'y.', text='')
    # imscatter2D(output[backgroundNum: -3, 0], output[backgroundNum: -3, 1], ax, 1, 'r*', text='', size=12)
    # imscatter2D(output[-3, 0], output[-3, 1], ax, 1, 'ch', text='', size=12)
    # imscatter2D(output[-2, 0], output[-2, 1], ax, 1, 'kv', text='', size=12)
    # imscatter2D(output[-1, 0], output[-1, 1], ax, 1, 'k^', text='', size=12)

    # 2p
    # imscatter2D(output[:backgroundNum, 0], output[:backgroundNum, 1], ax, 1, 'y.', text='')
    # imscatter2D(output[backgroundNum: -5, 0], output[backgroundNum: -5, 1], ax, 1, 'r*', text='', size=12)
    # imscatter2D(output[-5, 0], output[-5, 1], ax, 1, 'ch', text='', size=12)
    # imscatter2D(output[-4, 0], output[-4, 1], ax, 1, 'kv', text='', size=12)
    # imscatter2D(output[-3, 0], output[-3, 1], ax, 1, 'k^', text='', size=12)
    # imscatter2D(output[-2, 0], output[-2, 1], ax, 1, 'b<', text='', size=12)
    # imscatter2D(output[-1, 0], output[-1, 1], ax, 1, 'b>', text='', size=12)

    # 3p
    imscatter2D(output[:backgroundNum, 0], output[:backgroundNum, 1], ax, 1, 'y.', text='')
    imscatter2D(output[backgroundNum: -7, 0], output[backgroundNum: -7, 1], ax, 1, 'r*', text='', size=12)
    imscatter2D(output[-7, 0], output[-7, 1], ax, 1, 'ch', text='', size=12)
    imscatter2D(output[-6, 0], output[-6, 1], ax, 1, 'kv', text='', size=12)
    imscatter2D(output[-5, 0], output[-5, 1], ax, 1, 'k^', text='', size=12)
    imscatter2D(output[-4, 0], output[-4, 1], ax, 1, 'b<', text='', size=12)
    imscatter2D(output[-3, 0], output[-3, 1], ax, 1, 'b>', text='', size=12)
    imscatter2D(output[-2, 0], output[-2, 1], ax, 1, 'm+', text='', size=12)
    imscatter2D(output[-1, 0], output[-1, 1], ax, 1, 'mx', text='', size=12)

    # 2i
    # imscatter2D(output[:backgroundNum, 0], output[:backgroundNum, 1], ax, 1, 'y.', text='', color='y')
    # imscatter2D(output[backgroundNum: -7, 0], output[backgroundNum: -7, 1], ax, 1, 'r.', text='', size=12, color='r')
    # imscatter2D(output[-7, 0], output[-7, 1], ax, 1, 'c.', text='', size=12, color='c')
    # imscatter2D(output[-6, 0], output[-6, 1], ax, 1, 'c.', text='', size=12, color='c')
    # imscatter2D(output[-5, 0], output[-5, 1], ax, 1, 'k.', text='', size=12, color='k')
    # imscatter2D(output[-4, 0], output[-4, 1], ax, 1, 'k.', text='', size=12, color='k')
    # imscatter2D(output[-3, 0], output[-3, 1], ax, 1, 'k*', text='', size=9, color='k')
    # imscatter2D(output[-2, 0], output[-2, 1], ax, 1, 'k*', text='', size=9, color='k')
    # imscatter2D(output[-1, 0], output[-1, 1], ax, 1, 'b.', text='', size=12, color='#CD5C5C')

    # imscatter2D(output[:, 0], output[:, 1], ax, 1, 'b.', text='', size=3)

    # imscatter2D(output[-2, 0], output[-2, 1], ax, 1, colors[-4], text='qnew',  size=15))
    # imscatter2D(output[list(hard_answers[chosen_queries[0]]), 0], output[list(hard_answers[chosen_queries[0]]), 1], ax,
    #             1, 'r.', text='', size=12)

    print('show pic')
    plt.show()
    # plt.savefig(fname='figure.eps', format='eps')
    # print('input to get next pic')
    # input()


# hit@x and MRR
# 指标会先按照总的答案来算一次，然后分别按照简单和复杂答案再算一次(尚未实现)
def evaluation(model, valid_iterators, easy_answers, hard_answers, logger, iteration, status='train', query_name_dict=None):
    metric = {}

    for query_type in valid_iterators:

        query_count = 0
        mrr = h1 = h3 = h10 = 0

        for batch in valid_iterators[query_type]:   # score: batch_size * n_entity

            if 'train' in status:
                if query_count > 5000:
                    break

            chosen_queries, chosen_answers, nag_answers, query_type, sample_limit, subgraph, val_size = batch
            if query_count >= val_size:
                break

            batch_size = len(chosen_queries)

            query_count += batch_size
            score = model.forward(chosen_queries, query_type, sample_limit, subgraph, chosen_answers,
                                  query_name_dict, evaluating=True)

            if type(score) == tuple:
                entity_embedding = score[2]
                features = score[1]
                scores = score[3]
                score = score[0]

                print(chosen_queries[0])

                argsort = torch.argsort(scores[-1], dim=0, descending=True)
                ranking = argsort.clone().to(torch.float)
                ranking = ranking.scatter_(0, argsort, torch.arange(scores[-1].shape[0]).to(torch.float).cuda())
                hard_answer = hard_answers[chosen_queries[0]]
                easy_answer = easy_answers[chosen_queries[0]]
                num_hard = len(hard_answer)
                num_easy = len(easy_answer)
                assert len(hard_answer.intersection(easy_answer)) == 0
                cur_ranking = ranking[list(easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy

                answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                cur_ranking = cur_ranking - answer_list + 1
                cur_ranking = cur_ranking[masks]

                if cur_ranking.shape[0] != 1 or cur_ranking[0] != 1:
                    continue

                last = 1e9
                flag = False
                for i in range(len(scores)):
                    argsort = torch.argsort(scores[i], dim=0, descending=True)

                    ranking = argsort.clone().to(torch.float)
                    ranking = ranking.scatter_(0, argsort, torch.arange(scores[i].shape[0]).to(torch.float).cuda())

                    for idx, chosen_query in enumerate(chosen_queries):
                        if idx >= 1:
                            break

                        hard_answer = hard_answers[chosen_query]
                        easy_answer = easy_answers[chosen_query]
                        num_hard = len(hard_answer)
                        num_easy = len(easy_answer)
                        assert len(hard_answer.intersection(easy_answer)) == 0

                        cur_ranking = ranking[list(easy_answer) + list(hard_answer)]
                        cur_ranking, indices = torch.sort(cur_ranking)
                        masks = indices >= num_easy

                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                        cur_ranking = cur_ranking - answer_list + 1
                        cur_ranking = cur_ranking[masks]

                        print(cur_ranking)
                        if cur_ranking.item() >= last:
                            flag = True
                            break
                        else:
                            last = cur_ranking.item()

                        pass

                if flag:
                    continue

                draw(features, query_type, chosen_queries, easy_answers, hard_answers, entity_embedding)

                while True:
                    print('input exit to break')
                    mes = input()
                    if mes == 'exit':
                        break
                    else:
                        draw(features, query_type, chosen_queries, easy_answers, hard_answers, entity_embedding)

            elif type(score) == list:
                pass

            # 每个query中实体的排序结果(id)
            argsort = torch.argsort(score, dim=1, descending=True)

            ranking = argsort.clone().to(torch.float)
            ranking = ranking.scatter_(1, argsort, torch.arange(score.shape[1]).to(torch.float).repeat(argsort.shape[0],
                                                                                               1).cuda())

            if 'train' in status:
                for idx, chosen_answer in enumerate(chosen_answers):
                    if type(chosen_answer) is not list:
                        chosen_answer = [chosen_answer]

                    cur_ranking = ranking[idx, chosen_answer]
                    cur_ranking, indices = torch.sort(cur_ranking)

                    answer_list = torch.arange(len(chosen_answer)).to(torch.float).cuda()
                    cur_ranking = cur_ranking - answer_list + 1

                    mrr += torch.mean(1. / (cur_ranking)).item()
                    h1 += torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 += torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 += torch.mean((cur_ranking <= 10).to(torch.float)).item()

            else:
                for idx, chosen_query in enumerate(chosen_queries):
                    hard_answer = hard_answers[chosen_query]
                    easy_answer = easy_answers[chosen_query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0

                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy

                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    cur_ranking = cur_ranking - answer_list + 1
                    cur_ranking = cur_ranking[masks]

                    mrr += torch.mean(1. / (cur_ranking)).item()
                    h1 += torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 += torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 += torch.mean((cur_ranking <= 10).to(torch.float)).item()

        mrr /= query_count
        h1 /= query_count
        h3 /= query_count
        h10 /= query_count

        metric[query_type] = \
            [('mrr', mrr), ('h1', h1), ('h3', h3), ('h10', h10)]

        if iteration is not None:
            logger.info("query_type: {:s}; Iter: {:d}; hit@1: {:f}; hit@3: {:f}; hit@10: {:f}; MRR: {:f}".
                        format(str(query_type), iteration, h1, h3, h10, mrr))
        else:
            logger.info("query_type: {:s}; testing; hit@1: {:f}; hit@3: {:f}; hit@10: {:f}; MRR: {:f}".
                        format(str(query_type), h1, h3, h10, mrr))

    return metric


def load_id_neighbor_for_main(data_dir):
    # train_idn = pickle.load(open(os.path.join(data_dir, "train_queries_idneighbor.pkl"), 'rb'))
    # valid_idn = pickle.load(open(os.path.join(data_dir, "valid_queries_idneighbor.pkl"), 'rb'))
    # test_idn = pickle.load(open(os.path.join(data_dir, "test_queries_idneighbor.pkl"), 'rb'))

    # for triples in train_idn[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))].keys():
    #     for subtriple in triples:
    #         if subtriple not in train_idn[('e', ('r',))].keys():
    #             print(subtriple)
    #
    # for triples in train_idn[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))].keys():
    #     for subtriple in triples:
    #         if subtriple not in train_idn[('e', ('r',))].keys():
    #             print(subtriple)

    valid_queries = pickle.load(open(os.path.join(data_dir, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(data_dir, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(data_dir, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(data_dir, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(data_dir, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(data_dir, "test-easy-answers.pkl"), 'rb'))
    pass
    # return train_idn, valid_idn, test_idn


def generate_cqd_training_data(args, graph, train_q=None, train_a=None):

    sum = 0
    for q in train_q[('e', ('r', 'r'))]:
        sum += len(train_a[q])

    print(sum)

    pass


def test(data_dir):
    triple_list = []
    with open(data_dir + 'train.txt', 'r', encoding='utf-8') as graph_reader:
        for triple in graph_reader.readlines():
            h, r, t = list(map(lambda x: int(x), triple.strip().split('\t')))
            triple_list.append((h, r, t))

    train_idn = pickle.load(open(os.path.join(data_dir, "train_queries_idneighbor.pkl"), 'rb'))
    valid_idn = pickle.load(open(os.path.join(data_dir, "valid_queries_idneighbor.pkl"), 'rb'))
    test_idn = pickle.load(open(os.path.join(data_dir, "test_queries_idneighbor.pkl"), 'rb'))

    inKG = 0
    notinKG = 0

    for k, v in train_idn.items():
        if k == ('e', ('r',)):
            for query, neis in v.items():
                for nei in neis:
                    if nei in triple_list:
                        inKG += 1
                    else:
                        notinKG += 1
                    print(inKG, notinKG)
                    if (inKG + notinKG) % 500 == 0:
                        break
            print(inKG, notinKG)


if __name__ == '__main__':
    # answers_with_query_type('../FB15k-237-betae/')
    # load_id_neighbor_for_main('../FB15k-237-q2b/')
    # load_id_neighbor_for_main('../FB15k-237-betae/')

    # queries = pickle.load(open("../FB15k-237-q2b/train-queries.pkl", 'rb'))
    # answers = pickle.load(open("../FB15k-237-q2b/train-answers.pkl", 'rb'))

    # queries = pickle.load(open("../NELL-q2b/valid_queries_idneighbor_u.pkl", 'rb'))
    # answers = pickle.load(open("../NELL-q2b/test_queries_idneighbor_u.pkl", 'rb'))

    load_id_neighbor('../NELL-q2b/')

    # test('../FB15k-237-q2b/')

    pass
