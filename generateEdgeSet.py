import os
import numpy as np
import torch
from scipy.spatial import distance
from torch_geometric.datasets import Planetoid, Amazon

from utils import addMasks

# environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = ['cora']


def generate_edge(name, data, threshold):
    num_data = len(data)
    edge_index = [[], []]
    for i in range(num_data):
        for j in range(i + 1, num_data):
            dist = distance.hamming(data[i], data[j])
            if dist <= threshold:
                edge_index[0].append(i)
                edge_index[1].append(j)
    print("{} edges are generated.".format(len(edge_index[0])))
    path = './results/{}/fakeEdgeSet'.format(name)
    if not os.path.exists(path):
        os.mkdir(path)
    np.savez('./results/{}/fakeEdgeSet/generated_edges_{}'.format(name, int(threshold * 100)), np.array(edge_index))
    return torch.LongTensor(np.array(edge_index))


for name in datasets:
    for i in range(1, 10):
        assert name in ['cora', 'citeseer', 'computers', 'photo']
        dataset = Planetoid(root='./datasets', name=name, split='full') if name in ['cora', 'citeseer'] else Amazon(
            root='./datasets',
            name=name)
        data = dataset.get(0)
        # data = addMasks(data, dataset.num_classes)
        num_data = len(data.x)
        num_features = dataset.num_node_features
        edge_index = generate_edge(name, data.x[:, :int(num_features / 2)], threshold=i * 0.01)
