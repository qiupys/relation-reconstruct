import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='cuda visible device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--nparty', default=2, type=int, help='number of participants')
parser.add_argument('--ndim', default=16, type=int, help='dimension of embedding')
parser.add_argument('--repeat', default=10, type=int, help='repeat times')
parser.add_argument('--dataset', default='computers', type=str, help='dataset')
args = parser.parse_args()

# environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import numpy as np
import torch

from models import MLPWorker, GCNWorker
from utils import setupSeed, loadData, vflTrain, calEmbeds, calAUC, recEncEmbed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Server(torch.nn.Module):
    def __init__(self, num_party, num_embed, num_classes, num_layer):
        super(Server, self).__init__()
        self.num_party = num_party
        self.num_embed = num_embed
        self.num_classes = num_classes
        self.num_layer = num_layer
        self.fc = self._make_layers()

    def forward(self, embeds):
        x = torch.cat(embeds, 1)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

    def _make_layers(self):
        num_in_features = self.num_party * self.num_embed
        if self.num_layer == 1:
            return torch.nn.Sequential(torch.nn.Linear(in_features=num_in_features, out_features=self.num_classes))
        elif self.num_layer == 2:
            return torch.nn.Sequential(torch.nn.Linear(in_features=num_in_features, out_features=128),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(in_features=128, out_features=self.num_classes))
        else:
            layers = [torch.nn.Linear(in_features=num_in_features, out_features=128),
                      torch.nn.ReLU(inplace=True)]
            for i in range(1, self.num_layer - 1):
                layers += [torch.nn.Linear(in_features=int(128 / pow(2, i - 1)), out_features=int(128 / pow(2, i))),
                           torch.nn.ReLU(inplace=True)]
            layers += [torch.nn.Linear(in_features=int(128 / pow(2, self.num_layer - 2)), out_features=num_classes)]
            return torch.nn.Sequential(*layers)


for nlayer in range(1, 6):
    results = []
    for rep in range(args.repeat):
        setupSeed(10 * rep)
        data, split_data, num_data, num_classes = loadData(name=args.dataset, partition=0.5, device=device)
        workers = [MLPWorker(num_node_features=len(split_data[0].x[0]), num_embed=args.ndim).to(device),
                   GCNWorker(in_features=len(split_data[1].x[0]), out_features=args.ndim).to(device)]
        server = Server(num_party=args.nparty, num_embed=args.ndim, num_classes=num_classes, num_layer=nlayer).to(
            device)

        print('{} th, num_layers: {}, dataset: {}, VFL training...'.format(rep, nlayer, args.dataset))
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
        results.append(eval_results)
    path = './results/{}/sensitivity/'.format(args.dataset)
    if not os.path.exists(path):
        os.mkdir(path)
    np.savez('./results/{}/sensitivity/multilayer_{}'.format(args.dataset, nlayer), np.array(results))
