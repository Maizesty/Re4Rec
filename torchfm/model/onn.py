import torch
import sys
import torch.nn as nn
import numpy as np

sys.path.append('/home/yssun/pytorch-fm/torchfm/')

from layer import FeaturesEmbedding, FeaturesLinear, InnerProductNetwork, \
    OuterProductNetwork, MultiLayerPerceptron
    

class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.offsets = torch.as_tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64))

        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
    def ffm_interaction(self,field_wise_emb_list):
        interaction = []
        num_fields = self.num_fields
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                v_ij = field_wise_emb_list[j - 1][:, i, :]
                v_ji = field_wise_emb_list[i][:, j, :]
                dot = torch.sum(v_ij * v_ji, dim=1, keepdim=True)
                interaction.append(dot)
        return torch.cat(interaction, dim=1)
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = x + self.offsets

        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        return self.ffm_interaction(xs)

    
class ONN(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = torch.as_tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64))
        input_dim = embed_dim * self.num_fields + int(self.num_fields * (self.num_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(input_dim, mlp_dims, dropout)
        
    def field_aware_interaction(self, field_aware_emb_list):
        interaction = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                v_ij = field_aware_emb_list[j - 1][:, i, :]
                v_ji = field_aware_emb_list[i][:, j, :]
                dot = torch.sum(v_ij * v_ji, dim=1, keepdim=True)
                interaction.append(dot)
        return torch.cat(interaction, dim=1)    
    
    def forward(self, x):
      x = x + self.offsets
      field_aware_emb_list = [self.embeddings[i](x) for i in range(self.num_fields)]
      diag_embedding = field_aware_emb_list[0].flatten(start_dim=1)
      ffm_out = self.field_aware_interaction(field_aware_emb_list[1:])
      dnn_input = torch.cat([diag_embedding, ffm_out], dim=1)
      y_pred = self.mlp(dnn_input)
      return torch.sigmoid(y_pred.squeeze(1))


class ONNV2(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        # self.offsets = torch.as_tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64))

        self.embedding = FeaturesEmbedding(field_dims, self.embed_dim * self.num_fields)
        input_dim = embed_dim * self.num_fields + int(self.num_fields * (self.num_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(input_dim, mlp_dims, dropout)
        self.diag_mask = torch.eye(self.num_fields).bool()
        self.triu_mask = torch.triu(torch.ones(self.num_fields, self.num_fields), 1).bool()
        self.interact_units = int(self.num_fields * (self.num_fields - 1) / 2)
    
    def forward(self, x):
      # x = x + self.offsets
      field_wise_emb = self.embedding(x).view(-1, self.num_fields, self.num_fields, self.embed_dim)
      diag_embedding = torch.masked_select(field_wise_emb, self.diag_mask.unsqueeze(-1)).view(-1, self.embed_dim * self.num_fields)
      ffm_out = self.ffm_interaction(field_wise_emb)
      dnn_input = torch.cat([diag_embedding, ffm_out], dim=1)
      y_pred = self.mlp(dnn_input)
      return torch.sigmoid(y_pred.squeeze(1))    
    
    def ffm_interaction(self, field_wise_emb):
        out = (field_wise_emb.transpose(1, 2) * field_wise_emb).sum(dim=-1)
        out = torch.masked_select(out, self.triu_mask).view(-1, self.interact_units)
        return out