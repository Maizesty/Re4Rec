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

class LorentzFM(torch.nn.Module):
  def __init__(self, field_dims, embed_dim):
    super().__init__()
    self.embedding = FeaturesEmbedding(field_dims, embed_dim)
    self.num_fields = len(field_dims)
    self.p , self.q = zip(*list(combinations(range(self.num_fields), 2)))
    self.inner_product_layer = InnerProductNetworkLoop(self.num_fields)
  def get_zeroth_components(self, feature_emb):
      '''
      compute the 0th component
      '''
      sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
      zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
      return zeroth_components # batch * field

  def triangle_pooling(self, inner_product, zeroth_components):
      '''
      T(u,v) = (1 - <u, v>L - u0 - v0) / (u0 * v0)
              = (1 + u0 * v0 - inner_product - u0 - v0) / (u0 * v0)
              = 1 + (1 - inner_product - u0 - v0) / (u0 * v0)
      '''

      u0, v0 = zeroth_components[:, self.p], zeroth_components[:, self.q]  # batch * (f(f-1)/2)
      score_tensor = 1 + torch.div(1 - inner_product - u0 - v0, u0 * v0) # batch * (f(f-1)/2)
      output = torch.sum(score_tensor, dim=1, keepdim=True) # batch * 1
      return output    
  def forward(self, x):
    embed_x = self.embedding(x)
    inner_product = self.inner_product_layer(embed_x)
    zeroth_components = self.get_zeroth_components(embed_x)
    return torch.sigmoid(self.triangle_pooling(inner_product, zeroth_components) )
    