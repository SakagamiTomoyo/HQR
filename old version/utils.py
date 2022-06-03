import pickle
import os
from collections import defaultdict
import logging
import torch


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


def load_id_neighbor(data_dir):
    train_idn = pickle.load(open(os.path.join(data_dir, "train_queries_idneighbor_new.pkl"), 'rb'))
    valid_idn = pickle.load(open(os.path.join(data_dir, "valid_queries_idneighbor_new.pkl"), 'rb'))
    test_idn = pickle.load(open(os.path.join(data_dir, "test_queries_idneighbor_new.pkl"), 'rb'))
    print('loaded')

    return train_idn, valid_idn, test_idn


# hit@x and MRR
def evaluation(model, valid_iterators, easy_answers, hard_answers, logger, iteration, status='train', query_name_dict=None):
    metric = {}

    for query_type in valid_iterators:
        query_count = 0
        mrr = h1 = h3 = h10 = 0

        for batch in valid_iterators[query_type]:   # score: batch_size * n_entity

            if status == 'train':
                if query_count > 5000:
                    break

            chosen_queries, chosen_answers, nag_answers, query_type, sample_limit, subgraph, val_size = batch
            if query_count >= val_size:
                break

            batch_size = len(chosen_queries)

            query_count += batch_size
            score = model.forward(chosen_queries, query_type, sample_limit, subgraph, chosen_answers,
                                  query_name_dict, evaluating=True)

            # 每个query中实体的排序结果(id)
            argsort = torch.argsort(score, dim=1, descending=True)

            ranking = argsort.clone().to(torch.float)
            ranking = ranking.scatter_(1, argsort, torch.arange(score.shape[1]).to(torch.float).repeat(argsort.shape[0],
                                                                                               1).cuda())

            if status == 'train':
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
