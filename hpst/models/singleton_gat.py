from hpst.utils.options import Options
import hpst.layers.heterogenous_point_set_attention as psa
from typing import Tuple
from torch import Tensor, nn
import torch
    
from torch_geometric.nn import knn
from torch_geometric.nn.models import GAT

class GATModel(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 num_layers,
                 num_classes,
                 n_neighbors=8):
        super(GATModel, self).__init__()
        self.n_neighbors = n_neighbors
        self.model_x = GAT(in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               out_channels=num_classes, v2=True, jk="last")
        
    def forward(self, p1):
        coord1, feat1, batch1 = p1

        index1 = knn(coord1, coord1, self.n_neighbors, batch_x=batch1, batch_y=batch1)

        out_x = self.model_x(feat1, index1)#, batch=batch1)

        return out_x

class GATInterface(nn.Module):
    def __init__(self, options, feature_dim, output_dim):
        super(GATInterface, self).__init__()

        self.model = GATModel(
            in_channels=feature_dim,
            num_classes=output_dim,
            num_layers=options.gnn_layers,
            n_neighbors=options.gnn_neighbors,
            hidden_channels=options.gnn_hidden_channels
        )

    def forward(self, p1):
        return self.model(p1)