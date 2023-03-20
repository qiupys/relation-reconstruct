import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='cuda visible device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--nparty', default=2, type=int, help='number of participants')
parser.add_argument('--ndim', default=16, type=int, help='dimension of embedding')
parser.add_argument('--repeat', default=10, type=int, help='repeat times')
parser.add_argument('--dataset', default='cora', type=str, help='dataset')
parser.add_argument('--type', default='GraphSAGE', type=str, help='GNN type')
args = parser.parse_args()

# environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import numpy as np
import torch

from models import Server, MLPWorker, GCNWorker, GSWorker, GATWorker
from utils import setupSeed, loadData, vflTrain, calEmbeds, calAUC, recEncEmbed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for partition in range(1, 10):
    par_results = []
    for rep in range(args.repeat):
        setupSeed(10 * rep)
        data, split_data, num_data, num_classes = loadData(name=args.dataset, partition=partition * 0.1,
                                                           device=device)
        assert args.type in ['GraphSAGE', 'GAT']
        if args.type == 'GraphSAGE':
            workers = [MLPWorker(num_node_features=len(split_data[0].x[0]), num_embed=args.ndim).to(device),
                       GSWorker(in_features=len(split_data[1].x[0]), out_features=args.ndim).to(device)]
        elif args.type == 'GAT':
            workers = [MLPWorker(num_node_features=len(split_data[0].x[0]), num_embed=args.ndim).to(device),
                       GATWorker(in_features=len(split_data[1].x[0]), out_features=args.ndim).to(device)]
        server = Server(num_party=args.nparty, num_embed=args.ndim, num_classes=num_classes).to(device)

        print('{} th, partition: {:.1}, dataset: {}, VFL training...'.format(rep, partition * 0.1, args.dataset))
        eval_results = []

        # training result
        for epoch in range(1, 301):
            train_acc, val_acc, test_acc = vflTrain(workers, server, split_data, data)
            log = 'Epoch: {:03d}, train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            if epoch % 100 == 0:
                print(log.format(epoch, train_acc, val_acc, test_acc))
                eval_results.append(train_acc)
                eval_results.append(val_acc)
                eval_results.append(test_acc)

        # attack evaluation
        embeds = calEmbeds(workers, split_data)
        v_top = server(embeds)
        v_adv = embeds[0].detach().clone()
        v_top = v_top.detach().clone()
        v_tar = embeds[1].detach().clone()

        # baseline
        print("Baseline evaluation...")
        start = time.clock()
        auc_base = calAUC(num_data, v_tar, data.edge_index)
        end = time.clock()
        print("Baseline AUC: {:.4}, time: {:.4}".format(auc_base, end - start))
        eval_results.append(auc_base)

        # attack-0
        print("Attack-0 evaluation...")
        start = time.clock()
        auc_0 = calAUC(num_data, v_adv, data.edge_index)
        end = time.clock()
        print("Attack-0 AUC: {:.4}, time: {:.4}".format(auc_0, end - start))
        eval_results.append(auc_0)

        # attack-1
        print("Attack-1 evaluation...")
        start = time.clock()
        auc_1 = calAUC(num_data, v_top, data.edge_index)
        end = time.clock()
        print("Attack-1 AUC: {:.4}, time: {:.4}".format(auc_1, end - start))
        eval_results.append(auc_1)

        # attack-2
        print("Attack-2 evaluation...")
        start = time.clock()
        ap_tar = recEncEmbed(v_adv, v_top, 2, num_data, args.ndim, server, device)
        auc_2 = calAUC(num_data, ap_tar, data.edge_index)
        end = time.clock()
        print("Attack-2 AUC: {:.4}, time: {:.4}".format(auc_2, end - start))
        eval_results.append(auc_2)
        par_results.append(eval_results)
    path = './results/{}/sensitivity/{}'.format(args.dataset, args.type)
    if not os.path.exists(path):
        os.mkdir(path)
    np.savez('./results/{}/sensitivity/{}/partition_{}'.format(args.dataset, args.type, partition),
             np.array(par_results))
