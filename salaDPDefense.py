import argparse
import os
import random
import time

from torch_geometric.data import Data

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='cuda visible device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--nparty', default=2, type=int, help='number of participants')
parser.add_argument('--ndim', default=16, type=int, help='dimension of embedding')
parser.add_argument('--repeat', default=10, type=int, help='repeat times')
parser.add_argument('--dataset', default='cora', type=str, help='dataset')
args = parser.parse_args()

# environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Amazon
from models import Server, GCNWorker, MLPWorker
from utils import setupSeed, vflTrain, calEmbeds, calAUC, recEncEmbed, addMasks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_random_edge(num_data, edge_index, multiplier):
    origin_edge_index = edge_index.cpu().numpy()
    x = np.random.randint(0, num_data - 1, int(multiplier * len(origin_edge_index[0])))
    y = np.random.randint(0, num_data - 1, int(multiplier * len(origin_edge_index[0])))
    olx = origin_edge_index[0].tolist()
    oly = origin_edge_index[1].tolist()
    for i in range(len(x)):
        if x[i] == y[i]:
            continue
        olx.append(x[i])
        oly.append(y[i])
    random_edge_index = torch.LongTensor(np.array([olx, oly]))
    return random_edge_index


def loadData(name, multiplier):
    assert name in ['cora', 'citeseer', 'computers', 'photo']
    dataset = Planetoid(root='./datasets', name=name, split='full') if name in ['cora', 'citeseer'] else Amazon(
        root='./datasets',
        name=name)
    data = dataset.get(0)
    num_data, num_edge, num_classes, num_node_features, num_features = len(data.x), len(
        data.edge_index[0]), dataset.num_classes, dataset.num_node_features, int(dataset.num_node_features * 0.5)

    # add train_mask, val_mask and test_mask
    data = addMasks(data, dataset.num_classes)
    split_data = [Data(x=data.x[:, :num_features], y=data.y).to(device),
                  Data(edge_index=add_random_edge(num_data=num_data, edge_index=data.edge_index,
                                                  multiplier=multiplier),
                       x=data.x[:, num_features:],
                       y=data.y).to(device)]
    return data.to(device), split_data, num_data, num_classes


for mp in range(1, 11):
    mp_results = []
    for rep in range(args.repeat):
        setupSeed(10 * rep)
        data, split_data, num_data, num_classes = loadData(name=args.dataset, multiplier=mp * 0.1)
        workers = [MLPWorker(num_node_features=len(split_data[0].x[0]), num_embed=args.ndim).to(device),
                   GCNWorker(in_features=len(split_data[1].x[0]), out_features=args.ndim).to(device)]
        server = Server(num_party=args.nparty, num_embed=args.ndim, num_classes=num_classes).to(device)

        print('{} th, multiplier: {}, dataset: {}, VFL training...'.format(rep, mp, args.dataset))
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
        mp_results.append(eval_results)
    path = './results/{}/salaDP/'.format(args.dataset)
    if not os.path.exists(path):
        os.mkdir(path)
    np.savez('./results/{}/salaDP/multiplier_{}'.format(args.dataset, mp), np.array(mp_results))
