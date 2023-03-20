import os
import random

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from models import MLPWorker


def setupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def addMasks(data, num_classes, train_rate=7, val_rate=1, test_rate=2):
    train_mask, val_mask, test_mask = torch.zeros(len(data.x), dtype=torch.bool), torch.zeros(len(data.x),
                                                                                              dtype=torch.bool), torch.zeros(
        len(data.x), dtype=torch.bool)

    for c in range(num_classes):
        # idx = (data.y == c).nonzero().view(-1)
        idx = torch.nonzero(data.y == c, as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:idx.size(0) // 10 * train_rate]]
        train_mask[idx] = True

    # remaining = (~train_mask).nonzero().view(-1)
    remaining = torch.nonzero(~train_mask, as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    val_mask[remaining[:len(remaining) // (val_rate + test_rate) * val_rate]] = True
    test_mask[remaining[len(remaining) // (val_rate + test_rate) * val_rate:]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def loadMultiData(name, num_party, partition, device):
    assert name in ['cora', 'citeseer', 'computers', 'photo']
    dataset = Planetoid(root='./datasets', name=name, split='full') if name in ['cora', 'citeseer'] else Amazon(
        root='./datasets',
        name=name)
    data = dataset.get(0)
    num_data, num_edge, num_classes, num_node_features, num_features = len(data.x), len(
        data.edge_index[0]), dataset.num_classes, dataset.num_node_features, dataset.num_node_features // num_party

    # add train_mask, val_mask and test_mask
    data = addMasks(data, dataset.num_classes)

    # split data feature
    split_data = []
    for i in range(num_party - 1):
        split_data.append(Data(edge_index=data.edge_index[:, random.sample(range(num_edge), int(num_edge * partition))],
                               x=data.x[:, i * num_features:(i + 1) * num_features], y=data.y).to(device))
    split_data.append(Data(edge_index=data.edge_index[:, random.sample(range(num_edge), int(num_edge * partition))],
                           x=data.x[:, (num_party - 1) * num_features:], y=data.y).to(device))
    return data.to(device), split_data, num_data, num_classes


def loadData(name, partition, device):
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
    split_data = [Data(edge_index=data.edge_index, x=data.x[:, :num_features], y=data.y).to(device),
                  Data(edge_index=data.edge_index, x=data.x[:, num_features:], y=data.y).to(device)]
    return data.to(device), split_data, num_data, num_classes


def standard(data):
    data -= torch.mean(data, dim=1, keepdim=True)
    data /= torch.std(data, dim=1, keepdim=True)
    return data


def cosDist(data):
    val = data.detach().clone()
    val = standard(val)
    score = torch.mm(val, val.t())
    norm = torch.norm(val, dim=1, keepdim=True)
    score /= torch.mm(norm, norm.t())
    return 1 - score


def edge_index_to_adj(num_data, edge_index):
    G = nx.Graph()
    G.add_nodes_from(range(num_data))
    for i in range(len(edge_index[0])):
        G.add_edge(edge_index[0][i].item(), edge_index[1][i].item())
    adj = nx.to_numpy_matrix(G)
    np.fill_diagonal(adj, 1)
    return adj


def calAUC(num_data, embed, edge_index):
    print("AUC calculating...")
    adj = torch.ShortTensor(edge_index_to_adj(num_data, edge_index))
    ground_truth = adj.cpu().detach().numpy()
    score = (1 - cosDist(embed)).cpu().detach().numpy()
    auc = roc_auc_score(np.nan_to_num(ground_truth[np.triu_indices(num_data, 1)]),
                        np.nan_to_num(score[np.triu_indices(num_data, 1)]))
    return auc


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


def calEmbeds(workers, split_data):
    embeds = []
    for i in range(len(workers)):
        worker = workers[i]
        if isinstance(worker, MLPWorker):
            embeds.append(worker(split_data[i].x))
        else:
            embeds.append(workers[i](split_data[i].x, split_data[i].edge_index))
    return embeds


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
    result = torch.zeros(size=[num_data, num_embed]).to(device)
    for approximate in latent[1:]:
        result += approximate
    return result / (num_party - 1)


def expand(vec, ground_truth, device):
    num_data = len(vec)
    adj = torch.ShortTensor(edge_index_to_adj(num_data, ground_truth)).to(device)
    d = torch.sum(adj, dim=1, keepdim=True)
    val = vec - torch.mean(vec, dim=1, keepdim=True)
    val = val / torch.std(val, dim=1, keepdim=True)
    score = torch.mm(val, val.t())
    norm = torch.norm(val, dim=1, keepdim=True)
    score /= torch.mm(norm, norm.t())
    avg = torch.sum(score * adj, dim=1, keepdim=True) / d
    # avg = (torch.sum(score * adj) - num_data) / (torch.sum(d) - num_data)
    nadj = adj.detach().clone()
    eval = score - avg
    nadj = torch.where(eval >= 0, torch.ones_like(adj), nadj)
    nd = torch.sum(nadj, dim=1)
    return nadj.to(device)


def transAttack(vec, ground_truth, num_embed, device):
    print("Transformer training...")
    transformer = MLPWorker(num_node_features=len(vec[0]), num_embed=len(vec[0])).to(device)
    params = [{'params': transformer.parameters()}]
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.01)

    nodes, adj = buildGraph(len(vec), 0, ground_truth)
    # node_list = random.sample(range(len(vec)), 200)
    # adj = sampleGraph(node_list, ground_truth)
    adj = torch.FloatTensor(adj).to(device)
    adj = torch.matrix_power(adj, 4)
    # maxi = torch.max(adj, dim=1, keepdim=True)
    # adj /= torch.max(adj, dim=1)

    temp = vec[nodes]
    iter, previous = 0, 0
    while True:
        def closure():
            optimizer.zero_grad()
            trans = transformer(temp)
            val = trans - torch.mean(trans, dim=1, keepdim=True)
            val = val / torch.std(val, dim=1, keepdim=True)
            score = torch.mm(val, val.t())
            # norm = torch.norm(val, dim=1, keepdim=True)
            # score /= torch.mm(norm, norm.t())
            loss = torch.sum((adj - score) ** 2)
            # loss = torch.sum(torch.log((adj - torch.eye(len(vec)).to(device)) * score))
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure()
        if iter > 5000:
            break
        previous = loss
        iter += 1
    return transformer(vec)


def cptransAttack(vec, ground_truth, num_embed, device):
    print("Transformer training...")
    transformer = MLPWorker(num_node_features=len(vec[0]), num_embed=len(vec[0])).to(device)
    params = [{'params': transformer.parameters()}]
    optimizer = torch.optim.Adam(params, lr=0.001)

    node_list = random.sample(range(len(vec)), 100)
    # nodes, adj = buildGraph(len(vec), 0, ground_truth)
    adj = sampleGraph(node_list, ground_truth)
    adj = torch.ShortTensor(adj).to(device)
    temp = vec[node_list]
    iter, previous = 0, 0
    while True:
        def closure():
            optimizer.zero_grad()
            trans = transformer(temp)
            val = trans - torch.mean(trans, dim=1, keepdim=True)
            val = val / torch.std(val, dim=1, keepdim=True)
            score = torch.mm(val, val.t())
            norm = torch.norm(val, dim=1, keepdim=True)
            score /= torch.mm(norm, norm.t())
            loss = torch.sum((adj - score) ** 2)
            # loss = torch.sum(torch.log((adj - torch.eye(len(vec)).to(device)) * score))
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure()
        if iter > 3000:
            break
        previous = loss
        iter += 1
    return transformer(vec)


def buildGraph(num_data, nid, edge_index):
    num_edges = len(edge_index[0])
    G = nx.Graph()
    G.add_nodes_from(range(num_data))
    for i in range(len(edge_index[0])):
        G.add_edge(edge_index[0][i].item(), edge_index[1][i].item())
    degree_hist = nx.degree_histogram(G)

    sub = nx.ego_graph(G, n=nid, radius=4, undirected=True)
    # while len(sub) < 4:
    #     nid += 1
    #     sub = nx.ego_graph(G, n=nid, radius=1, undirected=True)
    sub_degree_hist = nx.degree_histogram(sub)
    adj = nx.to_numpy_matrix(sub)
    np.fill_diagonal(adj, 1)
    return list(sub.nodes()), adj


def sampleGraph(node_list, edge_index):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    for i in range(len(edge_index)):
        if edge_index[0][i] in node_list and edge_index[1][i] in node_list:
            G.add_edge(edge_index[0][i].item(), edge_index[1][i].item())
    degree_hist = nx.degree_histogram(G)
    adj = nx.to_numpy_matrix(G)
    np.fill_diagonal(adj, 1)
    return adj


def attentionAttack(vec, ground_truth, device):
    print("Attention training...")
    num_data = len(vec)
    num_dim = len(vec[0])
    adj = torch.ShortTensor(edge_index_to_adj(len(vec), ground_truth)).to(device)
    # adj = expand(vec, ground_truth, device)
    Wq = torch.randn([num_dim, 4]).to(device).requires_grad_(True)
    Wk = torch.randn([num_dim, 4]).to(device).requires_grad_(True)
    Wv = torch.randn([num_dim, 4]).to(device).requires_grad_(True)
    params = [{'params': Wq}, {'params': Wk}, {'params': Wv}]
    optimizer = torch.optim.Adam(params, lr=0.001)

    iter, previous = 0, 0
    while True:
        def closure():
            optimizer.zero_grad()
            q = torch.mm(vec, Wq)
            k = torch.mm(vec, Wk)
            v = torch.mm(vec, Wv)
            temp = torch.softmax(torch.mm(q, k.t()), dim=1)
            z = torch.mm(temp, v)
            val = z - torch.mean(z, dim=1, keepdim=True)
            val = val / torch.std(val, dim=1, keepdim=True)
            score = torch.mm(val, val.t())
            norm = torch.norm(val, dim=1, keepdim=True)
            score /= torch.mm(norm, norm.t())
            loss = torch.sum(((adj - score) * adj) ** 2)
            # loss = -torch.sum(
            #     (adj - torch.eye(len(vec)).to(device)) * torch.log((adj - torch.eye(len(vec)).to(device)) * score))
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure()
        if abs(loss - previous) < 1e-9 or iter > 5000:
            break
        previous = loss
        iter += 1
    q = torch.mm(vec, Wq)
    k = torch.mm(vec, Wk)
    v = torch.mm(vec, Wv)
    temp = torch.softmax(torch.mm(q, k.t()), dim=1)
    return torch.mm(temp, v)
