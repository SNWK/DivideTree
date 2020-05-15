import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x100x4
        # adj : 16x1x100x100

        # twice message passing 1.128 each node 2.64 each node
        # 16 1 100 128
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        # matrix multiply??
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)
        # 16, 100, 128

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)
        # 16, 100, 64

        return output


class GraphAggregation(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        # print('input', input.shape)
        i = self.sigmoid_linear(input)
        # print("i",i.shape)
        j = self.tanh_linear(input)
        # print("j",j.shape)
        output = torch.sum(torch.mul(i,j), 1)
        # print("output",output.shape)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output
