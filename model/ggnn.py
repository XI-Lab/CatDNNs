import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv

from torch_scatter import scatter_add


class GGNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 num_edge_type=5,
                 output_dim=12,
                 node_hidden_dim=32,
                 num_step_prop=3):
        super(GGNN, self).__init__()
        self.num_step_prop = num_step_prop
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.edge_embed = nn.Embedding(num_edge_type, node_hidden_dim * node_hidden_dim)
        # self.edge_network = nn.Sequential(
        #     nn.Linear(num_edge_type, node_hidden_dim), nn.ReLU(),
        #     nn.Linear(node_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(node_hidden_dim, node_hidden_dim, self.edge_embed, aggr='mean', root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)
        self.i_network = nn.Sequential(nn.Linear(2 * node_hidden_dim, node_hidden_dim), nn.Sigmoid(),
                                       nn.Linear(node_hidden_dim, output_dim), nn.Sigmoid())
        self.j_network = nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim), nn.Sigmoid(),
                                       nn.Linear(node_hidden_dim, output_dim))

    def forward(self, data):
        out0 = F.relu(self.lin0(data.x))
        out = out0
        h = out.unsqueeze(0)

        for i in range(self.num_step_prop):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.i_network(torch.cat((out, out0), 1)) * self.j_network(out)
        out = scatter_add(out, data.batch, dim=0)

        return out
