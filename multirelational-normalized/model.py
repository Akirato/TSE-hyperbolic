import numpy as np
import torch
from manifolds.poincare import PoincareBall
from manifolds.ts_poincare import TSPoincareBall
#from ts_utils import *
import torch.nn.functional as F

class MuRP(torch.nn.Module):
    def __init__(self, d, dim, c):
        super(MuRP, self).__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]


        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)   
        u_e = self.manifold.proj_tan0(self.manifold.logmap0(u,self.c),c=self.c)
        u_W = u_e * Ru
        u_m = self.manifold.proj(self.manifold.expmap0(u_W,self.c),c=self.c)
        v_m = self.manifold.proj(self.manifold.mobius_add(v, rvh,self.c),c=self.c)
        sqdist = self.manifold.sqdist(u_m,v_m,self.c)

        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 
    
class MuRT(torch.nn.Module):
    def __init__(self, d, dim, c):
        super(MuRT, self).__init__()
        self.manifold = TSPoincareBall()
        self.c = c
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]


        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)   
        u_e = self.manifold.proj_tan0(self.manifold.logmap0(u,self.c),c=self.c)
        u_W = u_e * Ru
        u_m = self.manifold.proj(self.manifold.expmap0(u_W,self.c),c=self.c)
        v_m = self.manifold.proj(self.manifold.mobius_add(v, rvh,self.c),c=self.c)
        sqdist = self.manifold.sqdist(u_m,v_m,self.c)

        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 


class MuRE(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRE, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E.weight.data = self.E.weight.data.double()
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv.weight.data = self.rv.weight.data.double()
        self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()
       
    def forward(self, u_idx, r_idx, v_idx):
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        Ru = self.Wu[r_idx]
        rv = self.rv.weight[r_idx]
        u_size = u.size()
        
        u_W = u * Ru

        sqdist = torch.sum(torch.pow(u_W - (v+rv), 2), dim=-1)
        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 

