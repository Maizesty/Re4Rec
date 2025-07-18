import torch
import sys

sys.path.append('/home/yssun/pytorch-fm/torchfm/')

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear
from itertools import combinations

class InnerProductNetworkLoop(torch.nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.num_fields = num_fields
    def forward(self, x):
        num_fields = self.num_fields
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)

class FwFM(torch.nn.Module):
  def __init__(self, field_dims, embed_dim):
    super().__init__()
    self.embedding = FeaturesEmbedding(field_dims, embed_dim)
    self.num_fields = len(field_dims)
    self.inner_product_layer = InnerProductNetworkLoop(self.num_fields)
    interact_dim = int(self.num_fields * (self.num_fields - 1) / 2)
    self.interaction_weight = torch.nn.Linear(interact_dim, 1)
    self.linear_weight_layer = torch.nn.Linear(self.num_fields * embed_dim, 1, bias=False)
    self.embed_output_dim = len(field_dims) * embed_dim
  def forward(self, x):
    embed_x = self.embedding(x)
    inner_product_vec = self.inner_product_layer(embed_x)
    poly2_part = self.interaction_weight(inner_product_vec)
    linear_part = self.linear_weight_layer(embed_x.view(-1, self.embed_output_dim))
    x = poly2_part + linear_part 
    return torch.sigmoid(x.squeeze(1))