#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperboloid maniford unit tests.
@author: mehrdad khatir
"""
import torch
import unittest

from manifolds.poincare import PoincareBall

class TestPoincareBall(unittest.TestCase):
    def setUp(self):
        self.curvature = 1
    
    def test_concat_split(self):
        # poincare vectors right on the horocyle (norm == 1)
        v = torch.Tensor([[0.3607, 0.2972, 0.1777], [0.3607, 0.2972, 0.1777], [0.3607, 0.2972, 0.1777]])
        z = PoincareBall.concat(v)
        # checks that z is within the new poincare ball model.
        self.assertEqual(z.norm() < 1, True)
        v_ = PoincareBall.split(v=z, m=3)
        # checks if the concat is a reversible transformation.
        self.assertEqual(torch.all(v_==v), True)

        # test batch concat
        v1 = torch.Tensor([[0.3607, 0.2972, 0.1777], [0.3607, 0.2972, 0.027]])
        z1 = PoincareBall.concat(v1)
        v2 = torch.Tensor([[0.3607, 0.32, 0.17], [0.307, 0.22, 0.1]])
        z2 = PoincareBall.concat(v2)
        zz = (PoincareBall.concat(torch.stack([v1, v2])))
        self.assertEqual(torch.all(torch.stack([z1,z2])==zz), True)
        # test batch split
        vv = PoincareBall.split(v=zz, m=2)        
        self.assertAlmostEqual(torch.all(torch.round(torch.stack([v1,v2]) * 10**4) == torch.round(vv * 10**4)), True)

    def test_mul_vect(self):
        v = torch.Tensor([[0.3, 0.2, 0.17], [0.307, 0.22, 0.1], [0.3607, 0.2972, 0.1777]])
        y = torch.Tensor([[0.36, 0.29, 0.17], [0.37, 0.22, 0.1]])
        self.assertEqual(PoincareBall.mobius_matvec(m=v, x=y, c=1).shape, torch.Size([2, 3]))
        
        # batch_mat_mul
        self.assertEqual(PoincareBall.mobius_bmm(
                bm=torch.stack([v,v,v,v]), bx=torch.stack([y,y,y,y])).shape, 
                torch.Size([4, 2, 3]))
        
    def test_batch_scalar_mul(self):
        v = torch.Tensor([[0.3, 0.2, 0.17], [0.307, 0.22, 0.1], [0.3607, 0.2972, 0.1777]])
        a = torch.Tensor([1, 0.5, 0.25])
        expected = torch.stack([
                PoincareBall.mobius_mul(x=v[0], t=a[0]),
                PoincareBall.mobius_mul(x=v[1], t=a[1]),
                PoincareBall.mobius_mul(x=v[2], t=a[2])])
        result = PoincareBall.mobius_mul(x=v.transpose(0, 1), t=a, dim=0).transpose(0,1)
        self.assertAlmostEqual(torch.all(torch.round(expected * 10**4) == torch.round(result * 10**4)), True)
 
    def test_batch_mobius_add(self):
        x = torch.Tensor([[0.3, 0.2, 0.17], [0.307, 0.22, 0.1], [0.3607, 0.2972, 0.1777]])
        y = torch.Tensor([[0.3, 0.2, 0.17], [0.307, 0.22, 0.1], [0.3607, 0.2972, 0.1777]])
        a = torch.Tensor([1, 0.5, 0.25])
        expected = torch.stack([
                PoincareBall.mobius_mul(x=v[0], t=a[0]),
                PoincareBall.mobius_mul(x=v[1], t=a[1]),
                PoincareBall.mobius_mul(x=v[2], t=a[2])])
        result = PoincareBall.mobius_mul(x=v.transpose(0, 1), t=a, dim=0).transpose(0,1)
        self.assertAlmostEqual(torch.all(torch.round(expected * 10**4) == torch.round(result * 10**4)), True)
     
        
        

if __name__ == '__main__':
    unittest.main()
