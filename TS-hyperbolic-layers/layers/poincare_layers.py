#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poincare Layers
Created on Sun Dec 12 08:59:31 2021

@author: mehrdad
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import layers.hyp_layers as hyp_layers
from manifolds.poincare import PoincareBall
from layers.att_layers import SpecialSpmm


class AdjustableModule(nn.Module):

    def __init__(self, curvature):
        super(AdjustableModule, self).__init__()
        self.curvature = curvature
    
    def update_curvature(self, curvature):
        self.curvature = curvature


class MobiousAdd(AdjustableModule):
    """
    Poincare linear layer.
    """

    def __init__(self, features, curvature):
        super(MobiousAdd, self).__init__(curvature)
        self.manifold = PoincareBall
        self.features = features
        self.bias = nn.Parameter(torch.Tensor(features))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.bias, 0)

    def forward(self, x):
        """
            x: a batch of vectors in mobius/hyperbolic space.
        Returns:
            a batch of vectors in the hyperbolic space.
        
        """
        p_bias = self.bias
#        p_bias = self.manifold.euclidean2poincare(p_bias, c=self.curvature)
        p_bias = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(p_bias, self.curvature), c=self.curvature), c=self.curvature)
        return self.manifold.mobius_add(x, p_bias, c=self.curvature) 

    def extra_repr(self):
        return 'features={}, c={}'.format(self.out_features, self.curvature)


class Linear(AdjustableModule):
    """
    Poincare linear layer.
    """

    def __init__(self, in_features, out_features, curvature, dropout, use_bias):
        super(Linear, self).__init__(curvature)
        self.manifold = PoincareBall
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
#        init.xavier_normal_(self.weight, gain=math.sqrt(2))
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.use_bias:
            init.constant_(self.bias, 0)

    def forward(self, x):
        """
            x: a batch of vectors in mobius/hyperbolic space.
        Returns:
            a batch of vectors in the hyperbolic space.
        
        """
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, self.curvature)
        if self.use_bias:
            p_bias = self.manifold.euclidean2poincare(self.bias, c=self.curvature)
            # mobius addition gaurantuees that the result is within the poincare ball.
            res = self.manifold.mobius_add(res, p_bias, c=self.curvature) 
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.curvature
        )

class Act(AdjustableModule):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(Act, self).__init__(c_in)
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt1 = self.manifold.proj_tan0(xt, c=self.c_out)
        out = self.manifold.proj(self.manifold.expmap0(xt1, c=self.c_out), c=self.c_out)
        return out

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )



class GraphConvolution(AdjustableModule):
    """
    Simple Poincare GCN layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout, act, curvature = 1, use_bias=False):
        super(GraphConvolution, self).__init__(curvature)
        self.manifold = manifold

        self.linear = Linear(in_features, out_features, self.curvature, dropout, use_bias=use_bias)
        self.act = Act(self.manifold, self.curvature, self.curvature, act)

        self.in_features = in_features
        self.out_features = out_features

    def update_curvature(self, c):
        super(GraphConvolution, self).update_curvature(c)
        self.linear.update_curvature(c)

    def forward(self, input):
        """The input must be already in poincare ball.
        """
        x, adj = input

        hidden = self.linear(x)
        # Doing the convolution in Euclidean space and scaling it back to poincare.
        hidden = PoincareBall.poincare2euclidean(hidden, c=self.curvature)

        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        support = PoincareBall.euclidean2poincare(support, c=self.curvature)

        support = self.act(support)
        return support, adj


    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class SpGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, activation, curvature = 1, use_bias=False):
        super(SpGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias

#        self.linear = Linear(in_features, out_features, self.curvature, dropout, use_bias=False)
        self.linear = nn.Linear(in_features, out_features, use_bias)
        if use_bias:
            self.add_bias = MobiousAdd(out_features, self.curvature)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation    

    def update_curvature(self, c):
        super(SpGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()
        
        # Note that as per initial experiments, we had the mobius linear or the 
        # linear layer does not have much difference in the performance for 
        # the NC task on PUBMED datasets.
        # However, theoretically speaking, the  eulidean linear layer weight 
        # updates can cause the output vector to go outside the PincareBall.
        # Although we clamp such vector, that cause in adding noise (claming noise)
        # and hence information loss. 
        # our initial reports showed 4% drop in test acc for NC task on pubmed.
        h = self.linear(input)
        if self.use_bias:
            h = self.add_bias(h)
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        # note that Euclidean concat might make the vector to go outside the poincare ball. although we clamp such vectors
        # this cost about 1% for NC task on PUBMED dataset.
        edge_h = PoincareBall.concat(torch.stack([h[edge[0, :], :], h[edge[1, :], :]], dim=-2)).t()

        # edge: 2*D x E
#        h = PoincareBall.poincare2euclidean(h, c=torch.Tensor([1]), scale=1)
#        edge_h = PoincareBall.poincare2euclidean(edge_h, c=torch.Tensor([1]), scale=1)
#        h = PoincareBall.poincare2euclidean(h, c=self.curvature, scale=1)
#        edge_h = PoincareBall.poincare2euclidean(edge_h, c=self.curvature, scale=1)
        
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = self.act(h_prime)

        out = PoincareBall.euclidean2poincare(out, c=self.curvature, scale=10)
#        out = PoincareBall.euclidean2poincare(out, c=torch.Tensor([1.0]), )

        return out


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(AdjustableModule):
    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat, curvature, use_bias):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.attentions = [SpGraphAttentionLayer(input_dim,
                                                 output_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 activation=activation,
                                                 curvature=curvature,
                                                 use_bias=use_bias) for _ in range(nheads)]
        self.concat = concat

        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x, adj = input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        x = PoincareBall.poincare2euclidean(x, c=self.curvature)
        if self.concat:
            h = PoincareBall.concat(torch.stack([att(x, adj) for att in self.attentions], dim=-2))            
        else:
            h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        h = PoincareBall.euclidean2poincare(h, c=self.curvature)
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)