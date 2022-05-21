"""Hyperboloid manifold."""

import torch

from manifolds.base import Manifold
from manifolds.poincare import PoincareBall

from utils.math_utils import arcosh, cosh, sinh 


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """
    name = 'Hyperboloid'
    eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    min_norm = 1e-15
    max_norm = 1e6

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    @classmethod
    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    @classmethod
    def minkowski_norm(self, u, keepdim=True):
        dot = Hyperboloid.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=Hyperboloid.eps[u.dtype]))

    @classmethod
    def sqdist(self, x, y, c):
        K = 1. / c
        prod = Hyperboloid.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + Hyperboloid.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    @classmethod
    def proj(self, x, c = 1):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=Hyperboloid.eps[x.dtype]))
        return vals + mask * x

    @classmethod
    def proj_tan(self, u, x, c):
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=Hyperboloid.eps[x.dtype])
        return vals + mask * u

    @classmethod
    def proj_tan0(self, u, c):
        """
        Note: the expmap cannot work on raw Euclidean vector. So we have to 
        transform the Euclidean vector into the Tangent space first.
        
        Note: always the input of proj_tan0 is in the Euclidean space.
        """
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    @classmethod
    def expmap(self, u, x, c):
        """ It folds the tangent space on the manifold (expx : TxM -> M).
        """
        K = 1. / c
        sqrtK = K ** 0.5
        normu = Hyperboloid.minkowski_norm(u)
        normu = torch.clamp(normu, max=Hyperboloid.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=Hyperboloid.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return Hyperboloid.proj(result, c)
        
    @classmethod
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(Hyperboloid.minkowski_dot(x, y) + K, max=-Hyperboloid.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = Hyperboloid.minkowski_norm(u)
        normu = torch.clamp(normu, min=Hyperboloid.min_norm)
        dist = Hyperboloid.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return Hyperboloid.proj_tan(result, x, c)

    @classmethod
    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=Hyperboloid.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return Hyperboloid.proj(res, c)

    @classmethod
    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=Hyperboloid.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + Hyperboloid.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    @classmethod
    def mobius_add(self, x, y, c):
        u = Hyperboloid.logmap0(y, c)
        v = Hyperboloid.ptransp0(x, u, c)
        return Hyperboloid.expmap(v, x, c)

    @classmethod
    def mobius_matvec(self, m, x, c):
        """
        Transform the vector to Euclidean apply matmul and transform back to 
        Hyperbolic space.
        """
        u = Hyperboloid.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return Hyperboloid.expmap0(mu, c)

    @classmethod
    def ptransp(self, x, y, u, c):
        logxy = Hyperboloid.logmap(x, y, c)
        logyx = Hyperboloid.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = Hyperboloid.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return Hyperboloid.proj_tan(res, y, c)

    @classmethod
    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return Hyperboloid.proj_tan(res, x, c)

    @classmethod
    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)


    @classmethod
    def from_poincare(self, x, c=1, ideal=False):
        """Convert from Poincare ball model to hyperboloid model.
        
        Note: converting a point from poincare ball to hyperbolic is 
            reversible, i.e. p == to_poincare(from_poincare(p)).
            
        Args:
            x: torch.tensor of shape (..., dim)
            ideal: boolean. Should be True if the input vectors are ideal points, False otherwise
        Returns:
            torch.tensor of shape (..., dim+1)
        To do:
            Add some capping to make things numerically stable. This is only needed in the case ideal == False
        """
        if ideal:
            t = torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)
            return torch.cat((t, x), dim=-1)
        else:
            K = 1./ c
            sqrtK = K ** 0.5
            eucl_squared_norm = (x * x).sum(dim=-1, keepdim=True)
            return sqrtK * torch.cat((K + eucl_squared_norm, 2 * sqrtK * x), dim=-1) / (K - eucl_squared_norm).clamp_min(Hyperboloid.min_norm)

    @classmethod
    def concat(self, v, c):
        """
        Note that the output dimension is (input_dim-1) * n + 1
        """
        p = PoincareBall.from_hyperboloid(v, c)
        p = PoincareBall.concat(p)
        return Hyperboloid.from_poincare(p, c)
        