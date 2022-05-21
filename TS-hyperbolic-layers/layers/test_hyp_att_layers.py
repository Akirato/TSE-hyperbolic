#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperbolic Attention layer unittests.
@author: mehrdad khatir
"""
import torch
import unittest

from layers.hyp_att_layers import HypGraphSharedSelfAttentionLayerV0, HypGraphSharedSelfAttentionLayerV1

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.curvature = 1
        self.input_dim = 3
        self.output_dim = 3

    def test_HypGraphSharedSelfAttentionLayerV0(self):
        hyp_att = HypGraphSharedSelfAttentionLayerV0(self.input_dim, self.output_dim, self.curvature)
        t_in= torch.tensor([[1.0192, 0.0800, 0.1799], 
                            [1.1018, 0.2379, 0.3966],
                            [-0.8950,  0.3513, -0.2600]])
        edges = torch.IntTensor([[0, 1], [1, 1], [0, 2], [1,2], [2, 0]])
        self.assertEqual(torch.all(torch.isnan(hyp_att(t_in, edges))), False)

    def test_HypGraphSharedSelfAttentionLayerV1(self):
        hyp_att = HypGraphSharedSelfAttentionLayerV1(self.input_dim, self.output_dim, self.curvature)
        t_in= torch.tensor([[1.0192, 0.0800, 0.1799], 
                            [1.1018, 0.2379, 0.066],
                            [-0.8950,  0.3513, -0.2600]])
        edges = torch.IntTensor([[0, 1], [1, 1], [0, 2], [1,2], [2, 0]])
        # TODO(mehrdad): Figure out a way to test the correctness of the 
        # output values as well.
        self.assertEqual(torch.all(torch.isnan(hyp_att(t_in, edges))), False)

if __name__ == '__main__':
    unittest.main()