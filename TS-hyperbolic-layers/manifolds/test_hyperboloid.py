#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperboloid maniford unit tests.
@author: mehrdad khatir
"""
import torch
import unittest
from manifolds.poincare import PoincareBall
from manifolds.hyperboloid import Hyperboloid

class TestHyperboloid(unittest.TestCase):
    def setUp(self):
        self.curvature = 1
        
    def test_hyp_transformations(self):
        # point ins hyperbolic plane.
        x = torch.tensor([[1.0192, 0.0800, 0.1799], [1.1018, 0.2379, 0.3966]])
        # its poincare mapped vector.
        p = torch.tensor([[0.0396, 0.0891], [0.1132, 0.1887]])
        x_p = PoincareBall.from_hyperboloid(x)
        self.assertEqual(torch.all(torch.round(p * 10**4) == torch.round(x_p * 10**4)).item(), True)
        p_x = Hyperboloid.from_poincare(p)
        self.assertEqual(torch.all(torch.round(p_x * 10**4) == torch.round(x * 10**4)).item(), True)

    def test_concat(self):
        # Hyperbolic vectors
        t1 = torch.tensor([1.8386, 0.0764, 0.6253, 1.3818, 0.2722])
        t2 = torch.tensor([1.5641, 1.1810, 0.1374, 0.0321, 0.1777])
        res = Hyperboloid.from_poincare(PoincareBall.concat(PoincareBall.from_hyperboloid(torch.stack([t1, t2]))), c=1)
        expected = torch.tensor([1.6397, 0.0487, 0.3987, 0.8811, 0.1736, 0.8337, 0.0970, 0.0227, 0.1254])
        self.assertEqual(torch.all(torch.round(res * 10**4) == torch.round(expected * 10**4)).item(), True)
        # Note that the size of hyperbolic concat is 2N - 1 (time dim does not get concatnated).
        self.assertEqual(torch.numel(res), torch.numel(t1) + torch.numel(t2) - 1)
        
    def test_sqdist(self):
        # Hyperbolic vectors
        x = torch.tensor([[1.0192, 0.0800, 0.1799]])
        y = torch.tensor([[1.1018, 0.2379, 0.3966]])
        self.assertEqual(torch.round(Hyperboloid.sqdist(x,y, c=self.curvature)*10**4).item(), 648)
        
        

if __name__ == '__main__':
    unittest.main()