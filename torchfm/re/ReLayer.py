import numpy as np
import torch
import torch.nn.functional as F


class ReFeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, prefix,output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.prefix = prefix
        self.prefix_offsets = self.offsets[:self.prefix]
        self.rest_offsets = self.offsets[self.prefix:]
    def forward(self, x):
        """
        :param x: tuple of tensor ``(prefix_index, rest_index)``
        prefix_index ``(prefix_field)``
        rest_index  ``(batch_szie,rest_field)``
        """
        prefix_index, rest_index = x
        prefix_index = prefix_index + prefix_index.new_tensor(self.prefix_offsets).unsqueeze(0)
        rest_index = rest_index + rest_index.new_tensor(self.rest_offsets).unsqueeze(0)
        prefix_sum = torch.sum(self.fc(prefix_index)) + self.bias
        rest_sum = torch.sum(self.fc(rest_index),dim=1)
        return prefix_sum + rest_sum


class ReFeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim,prefix):
        super().__init__()
        self.prefix = prefix
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.prefix_offsets = self.offsets[:self.prefix]
        self.rest_offsets = self.offsets[self.prefix:]
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: tuple of tensor ``(prefix_index, rest_index)``
        prefix_index ``(prefix_field)``
        rest_index  ``(batch_szie,rest_field)``
        """
        prefix_index, rest_index = x
        prefix_index = prefix_index + prefix_index.new_tensor(self.prefix_offsets).unsqueeze(0)
        rest_index = rest_index + rest_index.new_tensor(self.rest_offsets).unsqueeze(0)
        return (self.embedding(prefix_index),self.embedding(rest_index))


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class ReFactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: tuple of tensor ``(prefix_index, rest_index)``
        prefix_index ``(1,prefix_field,embed_dim)``
        rest_index  ``(batch_szie,rest_field,embed_dim)``
        """
        prefix_embed, rest_embed = x
        prefix_embed_sum = torch.sum(prefix_embed,dim=1)
        rest_embed_sum = torch.sum(rest_embed,dim=1)
        square_of_sum = (prefix_embed_sum + rest_embed_sum) ** 2
        # square_of_sum = torch.sum(x, dim=1) ** 2
        prefix_embed_square_sum = torch.sum(prefix_embed ** 2,dim=1)
        rest_embed_square_sum = torch.sum(rest_embed ** 2,dim=1)
        sum_of_square = prefix_embed_square_sum + rest_embed_square_sum
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class ReLinear(torch.nn.Module):
    def __init__(self, embed_dim, 
                            prefix_input_dim,
                            rest_input_dim,
                             use_bias=True):
        super().__init__()
        self.prefix_input_dim = prefix_input_dim
        self.rest_input_dim = rest_input_dim
        self.prefix_ll = torch.nn.Linear(prefix_input_dim, embed_dim,bias = use_bias)
        self.rest_ll = torch.nn.Linear(rest_input_dim, embed_dim,bias = False)
    def forward(self, x):
        prefix_embed, rest_embed = x
        return self.prefix_ll(prefix_embed.view(-1, self.prefix_input_dim )) + self.rest_ll(rest_embed.view(-1,self.rest_input_dim))

class ReMultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim,prefix_input_dim,rest_input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for idx,embed_dim in enumerate(embed_dims):
            if idx == 0:
                layers.append(ReLinear(prefix_input_dim= prefix_input_dim,rest_input_dim=rest_input_dim, embed_dim=embed_dim))
            else:
                layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)