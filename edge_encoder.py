import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree

class EdgeGINConv(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=True):
        super(EdgeGINConv, self).__init__(aggr='add')  # "Add" aggregation
        self.nn = nn
        self.initial_eps = eps
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        edge_weight = self.edge_encoder(edge_attr.unsqueeze(-1)).squeeze()
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.nn((1 + self.eps) * x + out)
    
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j