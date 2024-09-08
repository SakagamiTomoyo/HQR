import numpy as np
import torch
from sacred import Ingredient
import math
import time
from data_loader import get_queries_iterator, set_graph, set_answers, set_idn, set_helper_idn
from utils import evaluation
from torch import optim


train_ingredient = Ingredient('train')


def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1 - ema_alpha) * ema_loss + ema_alpha * loss
    return losses, ema_loss


@train_ingredient.capture
@torch.no_grad()
def run_eval(model, queries, easy_answers, hard_answers, valid_iterators, iteration, logger, batch_size=128, by_type=False,
             _run=None, status='train', query_name_dict=None):
    model.eval()
    metric = evaluation(model, valid_iterators, easy_answers, hard_answers, logger, iteration, status, query_name_dict)
    return metric


@train_ingredient.capture
def run_train(model, optimizer, args, logger,
              train_queries, train_answers,
              valid_queries, valid_easy_answers, valid_hard_answers,
              test_queries, test_easy_answers, test_hard_answers,
              train_idn, valid_idn, test_idn,
              train_helper_idn, valid_helper_idn, test_helper_idn,
              query_name_dict, name_query_dict, query_node_num_dict):

    best_metric_sum = 0
    new_lr = args.lr

    set_graph(model.graph)
    set_answers(train_answers,
                valid_easy_answers, valid_hard_answers,
                test_easy_answers, test_hard_answers)
    set_idn(train_idn, valid_idn, test_idn)
    if train_helper_idn is not None:
        set_helper_idn(train_helper_idn, valid_helper_idn, test_helper_idn)

    train_iterators = {}
    for query_type in train_queries:
        queries = train_queries[query_type]
        train_iterators[query_type] = get_queries_iterator(queries,
                                                           args,
                                                           query_type,
                                                           'train')

    valid_iterators = {}
    for query_type in valid_queries:
            queries = valid_queries[query_type]
            valid_iterators[query_type] = get_queries_iterator(queries,
                                                               args,
                                                               query_type,
                                                               'val')

    test_iterators = {}
    for query_type in test_queries:
            queries = test_queries[query_type]
            test_iterators[query_type] = get_queries_iterator(queries,
                                                              args,
                                                              query_type,
                                                              'test')

    ema_loss = None
    losses = []

    if args.continue_training:
        model.load_state_dict(torch.load(args.model_dir + 'model_best.pt')['net'])

    warm_up_steps = args.max_iter // 2
    current_learning_rate = args.lr

    for i in range(args.max_iter):
        loss = 0

        if i == int(args.max_iter / 2) and 'idn-cqd' in args.model:
            for param_group in optimizer.param_groups:
                if param_group['lr'] != 1e-1:
                    param_group['lr'] *= 0.2
                    print('adjust lr to ' + str(param_group['lr']))

        if i >= warm_up_steps and args.KGReasoning_setting:
            current_learning_rate = current_learning_rate / 5
            logger.info('Change learning_rate to %f at step %d' % (current_learning_rate, i))
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 1.5

        optimizer.zero_grad()
        if 'cqd' not in args.model:
            for query_type in train_queries:
                loss += run_batch(train_iterators[query_type], model, query_name_dict)
                # torch.cuda.empty_cache()
            losses, ema_loss = update_loss(loss.item() / len(train_queries), losses, ema_loss)

        else:
            loss = run_batch(train_iterators[('e', ('r',))], model, query_name_dict)
            losses, ema_loss = update_loss(loss.item(), losses, ema_loss)

        loss.backward()
        optimizer.step()

        if i % args.log_every == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}; loss: {:f}".
                        format(i, ema_loss, loss.item() / (len(train_queries)
                                            if 'cqd' not in args.model else 1)))

        if i % args.val_every == 0 and i != 0:
            if 'cqd' not in args.model:
                run_eval(model,
                         train_queries, train_answers, None,
                         train_iterators,
                         i, logger, batch_size=args.batch_size, status='train',
                         query_name_dict=query_name_dict)
                metrics = run_eval(model,
		                           valid_queries, valid_easy_answers, valid_hard_answers,
		                           valid_iterators,
		                           i, logger, batch_size=args.batch_size, status='val',
		                           query_name_dict=query_name_dict)
                run_eval(model,
                         test_queries, test_easy_answers, test_hard_answers,
                         test_iterators,
                         i, logger, batch_size=args.batch_size, status='test',
                         query_name_dict=query_name_dict)
            else:
                metrics = run_eval(model,
		                           valid_queries, valid_easy_answers, valid_hard_answers,
		                           valid_iterators,
		                           i, logger, batch_size=args.batch_size, status='val',
		                           query_name_dict=query_name_dict)

                if 'idn-cqd' in args.model:
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] != 1e-1:
                            param_group['lr'] *= 0.2
                            print('adjust lr to ' + param_group['lr'])

            all_metric_sum = 0
            for t in metrics:
                for j in range(len(metrics[t])):
                    all_metric_sum += metrics[t][j][1]

            if all_metric_sum > best_metric_sum:
                best_metric_sum = all_metric_sum
                torch.save({
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i
                }, args.model_dir + 'model_best.pt')

        if i % args.save_every == 0 and i != 0:
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i
            }, args.model_dir + 'model_latest.pt')

    run_eval(model,
             test_queries, test_easy_answers, test_hard_answers,
             test_iterators,
             None, logger,
             batch_size=args.batch_size,
             status='test',
             query_name_dict=query_name_dict)

    torch.save({
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, args.model_dir + 'model_final.pt')


def run_batch(queries_iterator, enc_dec, query_name_dict, hard_negatives=False, step=0):
    enc_dec.train()
    batch = next(queries_iterator)
    loss = enc_dec.margin_loss(*batch, query_name_dict=query_name_dict)
    return loss
