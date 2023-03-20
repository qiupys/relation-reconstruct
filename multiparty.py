import argparse
import os
import random
import time

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='cuda visible device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--nparty', default=3, type=int, help='number of participants')
parser.add_argument('--ndim', default=16, type=int, help='dimension of embedding')
parser.add_argument('--repeat', default=10, type=int, help='repeat times')
parser.add_argument('--dataset', default='cora', type=str, help='dataset')
args = parser.parse_args()

# environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import numpy as np
import torch

from models import Server, MLPWorker, GCNWorker
from utils import setupSeed, loadData, vflTrain, calEmbeds, calAUC, recEncEmbed, addMasks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadData(name, partition, nparty, device):
    assert name in ['cora', 'citeseer', 'computers', 'photo']
    dataset = Planetoid(root='./datasets', name=name, split='full') if name in ['cora', 'citeseer'] else Amazon(
        root='./datasets',
        name=name)
    data = dataset.get(0)
    num_data, num_edge, num_classes, num_node_features, num_features = len(data.x), len(
        data.edge_index[0]), dataset.num_classes, dataset.num_node_features, int(dataset.num_node_features * partition)

    # add train_mask, val_mask and test_mask
    data = addMasks(data, dataset.num_classes)

    # split data feature
    split_data = []
    for i in range(nparty - 1):
        split_data.append(
            Data(edge_index=data.edge_index, x=data.x[:, i * num_features: (i + 1) * num_features], y=data.y).to(
                device))
    split_data.append(Data(edge_index=data.edge_index, x=data.x[:, (nparty - 1) * num_features:], y=data.y).to(
        device))
    return data.to(device), split_data, num_data, num_classes


def vflTrain(workers, server, split_data, data):
    params = [{'params': server.parameters()}]
    for worker in workers:
        params.append({'params': worker.parameters()})
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.001)

    embeds = calEmbeds(workers, split_data)
    preds = server(embeds)
    optimizer.zero_grad()
    F.nll_loss(preds[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    logits, accs = preds, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def recEncEmbed(adversary, target, num_party, num_data, num_embed, model, device):
    print("Recover encrypted embeddings...")
    latent = [adversary]
    for i in range(num_party - 1):
        latent.append(torch.zeros(size=[num_data, num_embed]).to(device).requires_grad_(True))
    optimizer = torch.optim.LBFGS(latent[1:], lr=0.01)
    iters = 0
    previous = 0
    while True:
        def closure():
            optimizer.zero_grad()
            preds = model(latent)
            loss = ((preds - target) ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure()
        if abs(loss - previous) < 1e-9 or iters > 100:
            break
        previous = loss
        iters += 1
    return torch.cat(latent[1:], 1)


mp_results = []
for rep in range(args.repeat):
    setupSeed(10 * rep)
    data, split_data, num_data, num_classes = loadData(name=args.dataset, partition=0.2, nparty=args.nparty,
                                                       device=device)
    workers = [MLPWorker(num_node_features=len(split_data[i].x[0]), num_embed=args.ndim).to(device) for i in
               range(args.nparty - 1)]
    workers.append(GCNWorker(in_features=len(split_data[args.nparty - 1].x[0]), out_features=args.ndim).to(device))
    server = Server(num_party=args.nparty, num_embed=args.ndim, num_classes=num_classes).to(device)

    print('{} th, nparty: {}, dataset: {}, VFL training...'.format(rep, args.nparty, args.dataset))
    eval_results = []
    best_val_acc = test_acc = 0

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

    # attack-2
    print("Attack-2 evaluation...")
    start = time.clock()
    ap_tar = recEncEmbed(v_adv, v_top, args.nparty, num_data, args.ndim, server, device)
    auc_2 = calAUC(num_data, ap_tar, data.edge_index)
    end = time.clock()
    print("Attack-2 AUC: {:.4}, time: {:.4}".format(auc_2, end - start))
    eval_results.append(auc_2)
    mp_results.append(eval_results)

path = './results/{}/multiparty/'.format(args.dataset)
if not os.path.exists(path):
    os.mkdir(path)
np.savez('./results/{}/multiparty/nparty_{}'.format(args.dataset, args.nparty), np.array(mp_results))
