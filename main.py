from argparse import ArgumentParser
import torch
import data_loader as data_loader
import utils as utils
from GQE_N import GQE
from analysis.analysis import answer_cluster
from IDN_GQE_21_N import IDN_GQE_21 as IDN_GQE
from analysis.IDN_GQE_Mess import IDN_GQE_21 as IDN_GQE_Mess
from analysis.IDN_GQE_TSNE import IDN_GQE_TSNE
from analysis.GQE_TSNE import GQE_TSNE
from IDN_GQE_DistMult import IDN_GQE_DistMult
from IDN_GQE_RotatE import IDN_GQE_RotatE
from IDN_Q2B_24_N import IDN_Q2B_24 as IDN_Q2B
from analysis.IDN_Q2B_EX import IDN_Q2B_EX
from IDN_CQD_CO_24_N import IDN_CQD_CO_24 as IDN_CQD_CO
from IDN_BetaE import IDN_BetaE
from train_helper import train_ingredient, run_train
from torch import optim
from sacred import Experiment
from sacred.observers import MongoObserver

import time
import os

torch.cuda.set_device(0)

parser = ArgumentParser()

# task settings
# parser.add_argument('--tasks', type=str, default='1p.2p.3p.2i.3i.ip.pi.2u.up',
#                     help="tasks' types")

# q2b的析取query需要采样
parser.add_argument('--tasks', type=str, default='1p.2p.3p.2i.3i.ip.pi.2u.up',
                    help="tasks' types")

parser.add_argument("--data_dir", type=str, default="../FB15k-q2b/",
                    help="location of data")
parser.add_argument("--log_dir", type=str, default="./log/",
                    help="location of logging file")
parser.add_argument("--model_dir", type=str, default="./model/RotatE_15k_best_HQR_",
                    help="location of the best model and the latest model")
parser.add_argument("--continue_training", type=bool, default=True,
                    help="continue training from breakpoint")

# model construction settings
parser.add_argument("--model", type=str, default="idn-gqe-rotate")
parser.add_argument("--KGReasoning_setting", type=bool, default=False)
parser.add_argument("--cqd_fine_tune", type=bool, default=False)
parser.add_argument("--analysis", type=str, default=None)
parser.add_argument("--mp_model", type=str, default="")
parser.add_argument("--num_layers", type=int, default=3,
                    help="layers' num of GNN")
parser.add_argument("--shared_layers", default=False,
                    help="GNN layers share parameters or not")
parser.add_argument("--adaptive", default=True,
                    help="layers' num decided by size of query type")
parser.add_argument("--diameter_augment", type=int, default=0,
                    help="")
parser.add_argument("--readout", type=str, default="mp",
                    help="way to get the embedding of query")
parser.add_argument("--loss_function", type=str, default='distance',
                    help="loss function")
parser.add_argument("--scatter_op", type=str, default='mean')
parser.add_argument("--num_bases", type=int, default=0)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--aggregator", type=str, default='add')
# parser.add_argument("--active_function", type=str, default='relu')

# hyper parameters and optimizer settings

parser.add_argument("--embed_dim", type=int, default=400,
                    help="node and relations' embeddings dimensions")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument('-g', '--gamma', type=float, default=24,    # betaE == 60
                    help="margin in the loss")
parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size")
parser.add_argument("--max_iter", type=int, default=0,
                    help="max epoch")
parser.add_argument("--pre_sample", type=bool, default=0,
                    help="sample neighbors before training to save time")
parser.add_argument("--sample_limit", type=list, default=[1],
                    help="sample neighbors' num, "
                         "the last one in the list is the 1-hop neighbors")
parser.add_argument("--id_neighbor_limit", type=int, default=10)
parser.add_argument("--id_neighbor_pre_sample", type=bool, default=True)
parser.add_argument('--sample_times', type=int, default=1,
                    help="Effective only 'pre_sample' is True, "
                         "each query sample 'sample_time' times before used")
parser.add_argument("--opt", type=str, default="adam",
                    help="optimizer")
parser.add_argument("--dropout", type=float, default=0,
                    help="dropout")
parser.add_argument("--multi_neg_answers", type=bool, default=True,
                    help="")
parser.add_argument("--del_ans_edge", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=0)
# 去掉一部分信息有助于缓解过拟合

# print and log settings
parser.add_argument('--log_every', type=int, default=100,
                    help="log once every 'log_every'")
parser.add_argument("--val_every", type=int, default=10000,
                    help="validate once every 'val_every'")
parser.add_argument("--save_every", type=int, default=10000,
                    help="save model once every 'save_every'")

# others
parser.add_argument("--parallel", type=bool, default=False,
                    help="use DataParallel or not")
parser.add_argument("--depth", type=int, default=100001)
parser.add_argument("--max_burn_in", type=int, default=1000000)
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--cuda", default=True)
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--path_weight", type=float, default=0.01)
parser.add_argument('-evu', '--evaluate_union', type=str, default="DNF")

# print('qaidn2')
args = parser.parse_args()
query_node_num_dict = {('e', ('r',)): [1, 1],
                       ('e', ('r', 'r')): [1, 2],
                       ('e', ('r', 'r', 'r')): [1, 3],
                       (('e', ('r',)), ('e', ('r',))): [2, 1],
                       (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): [3, 1],
                       ((('e', ('r',)), ('e', ('r',))), ('r',)): [2, 2],
                       (('e', ('r', 'r')), ('e', ('r',))): [1, 1],
                       (('e', ('r',)), ('e', ('r', 'n'))): [1, 1],
                       (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): [1, 1],
                       ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): [1, 1],
                       (('e', ('r', 'r')), ('e', ('r', 'n'))): [1, 1],
                       (('e', ('r', 'r', 'n')), ('e', ('r',))): [1, 1],
                       (('e', ('r',)), ('e', ('r',)), ('u',)): [1, 1],
                       ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): [1, 1],
                       ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): [1, 1],
                       ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): [1, 1]
                       }
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

name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

print("Loading query data..")
train_queries, train_answers, \
        valid_queries, valid_easy_answers, valid_hard_answers, \
        test_queries, test_easy_answers, test_hard_answers, valid_answers, test_answers = data_loader.load_data(args, args.tasks.split('.'),
                                                             all_tasks, name_query_dict)

train_helper_idn, valid_helper_idn, test_helper_idn = None, None, None

print("Loading graph data..")
graph = data_loader.load_graph(args.data_dir, args.embed_dim, args.sample_limit, train_queries, train_answers)

if args.model == 'get-cqd-training-data':
    utils.generate_cqd_training_data(args, graph, train_queries, train_answers)
    exit()

if args.model == 'generate':
    id_neighbor = data_loader.generate_id_neighbor(args, graph,
                                                   train_queries, valid_queries, test_queries,
                                                   train_answers, valid_answers, test_answers,
                                                   query_name_dict)
    exit()
elif args.model == 'gqe':
    train_idn, valid_idn, test_idn = None, None, None
    model = GQE(graph, args).cuda()
elif args.model == 'gqe-tsne':
    train_idn, valid_idn, test_idn = None, None, None
    model = GQE_TSNE(graph, args).cuda()
elif args.model == 'idn-gqe':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_GQE(graph, args).cuda()
elif args.model == 'idn-gqe-mess':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_GQE_Mess(graph, args).cuda()
elif args.model == 'idn-gqe-tsne':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_GQE_TSNE(graph, args).cuda()
elif args.model == 'idn-gqe-distmult':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_GQE_DistMult(graph, args).cuda()
elif args.model == 'idn-gqe-rotate':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_GQE_RotatE(graph, args).cuda()
elif args.model == 'idn-q2b':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_Q2B(graph, args).cuda()
elif args.model == 'idn-q2b-ex':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_Q2B_EX(graph, args).cuda()
elif args.model == 'idn-cqd-co':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_CQD_CO(graph, args).cuda()
elif args.model == 'idn-betae':
    train_idn, valid_idn, test_idn = utils.load_id_neighbor(args.data_dir)
    model = IDN_BetaE(graph, args).cuda()
else:
    raise Exception("no such model!")

if args.parallel:
    model = torch.nn.DataParallel(model)

if args.analysis == 'answer_cluster':
    answer_cluster(model, graph,
                   train_queries, train_answers,
                   valid_queries, valid_answers, valid_easy_answers, valid_hard_answers,
                   test_queries, test_answers, test_easy_answers, test_hard_answers,
                   args)
    exit(0)

if args.opt == "sgd":
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0)
elif args.opt == "adam" and 'cqd' not in args.model:
    for p in model.parameters():
        if p.requires_grad:
            print(p.shape)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.weight_decay)


elif args.opt == "adagrad" or 'cqd' in args.model:
    optimizer = optim.Adagrad([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr)

    if args.model == 'idn-cqd-co':
        param_cqd = []
        param_mlp = []
        print("change lr of component")
        for param in optimizer.param_groups[0]['params']:
            if param is model.entity_embedding.weight or \
                    param is model.relation_embedding.weight:
                param_cqd.append(param)
            else:
                param_mlp.append(param)

        optimizer = optim.Adagrad([{'params': param_cqd}, {'params': param_mlp, 'lr': 1e-4}],
                                        lr=args.lr, weight_decay=args.weight_decay)

fname = time.strftime("%Y%m%d_%H%M%S", time.localtime())
log_file = (args.log_dir + fname + ".log").format(
        data=args.data_dir.strip().split("/")[-1],
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        model=args.model,
        readout=args.readout)

model_file = "model.pt"

logger = utils.setup_logging(log_file)

for k, v in vars(args).items():
    logger.info(str(k) + ':' + str(v))

ex = Experiment(ingredients=[train_ingredient])
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))
else:
    print('Running without Sacred observers')


@ex.config
def config():
    model = args.model
    lr = args.lr
    num_layers = args.num_layers
    shared_layers = args.shared_layers
    adaptive = args.adaptive
    readout = args.readout
    dropout = args.dropout
    weight_decay = args.weight_decay
    max_burn_in = args.max_burn_in
    num_basis = args.num_bases
    scatter_op = args.scatter_op
    opt = args.opt
    data_dir = args.data_dir
    path_weight = args.path_weight
    decoder = args.decoder


@ex.main
def main(data_dir, _run):
    run_train(model.module if args.parallel else model, optimizer, args, logger,
              train_queries, train_answers,
              valid_queries, valid_easy_answers, valid_hard_answers,
              test_queries, test_easy_answers, test_hard_answers,
              train_idn, valid_idn, test_idn,
              train_helper_idn, valid_helper_idn, test_helper_idn,
              query_name_dict, name_query_dict, query_node_num_dict)


ex.run()
print('end')
