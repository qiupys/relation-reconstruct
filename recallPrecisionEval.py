import os

import argparse
import networkx as nx
import torch
from torch_geometric.datasets import Planetoid, Amazon

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='cuda visible device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--nparty', default=2, type=int, help='number of participants')
parser.add_argument('--ndim', default=128, type=int, help='dimension of embedding')
parser.add_argument('--repeat', default=10, type=int, help='repeat times')
args = parser.parse_args()

# environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# name = input('Enter dataset name:\n')
dataset = ['cora', 'citeseer', 'computers', 'photo']
for name in dataset:
    assert name in ['cora', 'citeseer', 'computers', 'photo']
    dataset = Planetoid(root='./datasets', name=name) if name in ['cora', 'citeseer'] else Amazon(root='./datasets',
                                                                                                  name=name)
    data = dataset.get(0)

    edge_index, num_data = data.edge_index, len(data.x)

    tp = 0
    for i in range(len(edge_index[0])):
        u, v = edge_index[0][i], edge_index[1][i]
        if data.y[u] == data.y[v]:
            tp += 1

    G = nx.Graph()
    G.add_nodes_from(range(num_data))
    for i in range(len(edge_index[0])):
        G.add_edge(edge_index[0][i].item(), edge_index[1][i].item())
    adj = nx.to_numpy_matrix(G).tolist()

    fp = 0
    for i in range(len(data.x)):
        for j in range(i + 1, len(data.x)):
            if data.y[i] == data.y[j] and adj[i][j] == 0:
                fp += 1

    recall = tp / len(edge_index[0])
    precision = tp / (tp + fp)
    print("{}: recall {:.4} & precision {:.4}".format(name, recall, precision))
