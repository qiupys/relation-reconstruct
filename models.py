import torch
from torch_geometric.nn import SGConv, SAGEConv, GATConv


class GCNWorker(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNWorker, self).__init__()
        self.conv = SGConv(in_channels=in_features, out_channels=out_features, K=2)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GSWorker(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GSWorker, self).__init__()
        self.gs1 = SAGEConv(in_channels=in_features, out_channels=128)
        self.gs2 = SAGEConv(in_channels=128, out_channels=out_features)

    def forward(self, x, edge_index):
        x = self.gs1(x, edge_index)
        x = torch.relu(x)
        return self.gs2(x, edge_index)


class GATWorker(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GATWorker, self).__init__()
        self.gat1 = GATConv(in_channels=in_features, out_channels=128, heads=4, concat=False)
        self.gat2 = GATConv(in_channels=128, out_channels=out_features, heads=4, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        # x = torch.relu(x)
        return self.gat2(x, edge_index)


class Server(torch.nn.Module):
    def __init__(self, num_party, num_embed, num_classes):
        super(Server, self).__init__()
        self.fc = torch.nn.Linear(in_features=num_party * num_embed, out_features=num_embed)
        self.fc2 = torch.nn.Linear(in_features=num_embed, out_features=num_classes)

    def forward(self, embeds):
        x = torch.cat(embeds, 1)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


class MLPWorker(torch.nn.Module):
    def __init__(self, num_node_features, num_embed):
        super(MLPWorker, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=num_node_features, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=num_embed)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)
