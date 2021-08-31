import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, Set2Set


class GCN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 output_dim=12,
                 node_hidden_dim=32,
                 num_step_prop=3,
                 num_step_set2set=3):
        super(GCN, self).__init__()
        self.num_step_prop = num_step_prop
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.conv = GCNConv(node_hidden_dim, node_hidden_dim, cached=False)
        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))

        for i in range(self.num_step_prop):
            out = F.relu(self.conv(out, data.edge_index))

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out
