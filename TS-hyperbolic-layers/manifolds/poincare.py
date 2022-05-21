"""Poincare ball manifold."""

import torch
import scipy.special as sc

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """
    name = 'PoincareBall'
#    min_norm = 1e-4
    eps = {torch.float32: 4e-3, torch.float64: 1e-5}
#    max_norm = 1 - 1e-4
#    euclidean_max = 15
    # for float 64
    min_norm = 1e-5
    max_norm = 1 - 1e-5
    euclidean_max = 33

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    @classmethod
    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        val = sqrt_c * PoincareBall.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)
        dist = artanh(val) * 2 / sqrt_c
        return (dist ** 2).clamp(min=-PoincareBall.euclidean_max, max=PoincareBall.euclidean_max)

    @classmethod
    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(PoincareBall.min_norm)

    @classmethod
    def egrad2rgrad(self, p, dp, c):
        lambda_p = PoincareBall._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    @classmethod
    def proj(self, x, c):
        """Project points to Poincare ball with curvature c.
        Args:
            x: torch.Tensor of dim(M,N) with hyperbolic points
            c: manifold curvature.
        Returns:
            torch.Tensor with projected hyperbolic points.
        """
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), PoincareBall.min_norm)
        maxnorm = (1 - PoincareBall.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    @classmethod
    def proj_tan(self, u, p, c):
        return u

    @classmethod
    def proj_tan0(self, u, c):
        return u

    @classmethod
    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(PoincareBall.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * PoincareBall._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = PoincareBall.mobius_add(p, second_term, c)
        return gamma_1

    @classmethod
    def logmap(self, p1, p2, c):
        sub = PoincareBall.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(PoincareBall.min_norm)
        lam = PoincareBall._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    @classmethod
    def expmap0(self, u, c, dim: int = -1):
        """Exponential map taken at the origin of the Poincare ball with curvature c.
        Args:
            u: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
        Returns:
            torch.Tensor with tangent points shape (B, d)
        """
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=dim, p=2, keepdim=True), PoincareBall.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    @classmethod
    def logmap0(self, p, c, dim: int = -1):
        """Logarithmic map taken at the origin of the Poincare ball with curvature c.
        Args:
            y: torch.Tensor of size B x d with tangent points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
        Returns:
            torch.Tensor with hyperbolic points.
        """
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=dim, p=2, keepdim=True).clamp_min(PoincareBall.min_norm)
        y = (sqrt_c * p_norm).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)
        return p / (y * artanh(y))

    @classmethod
    def mobius_add(self, x, y, c=1, dim=-1):
        """
        """
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = (1 + 2 * c * xy + c ** 2 * x2 * y2).clamp_min(PoincareBall.min_norm)
        return (num / denom).clamp(min=-PoincareBall.euclidean_max, max=PoincareBall.euclidean_max)

    @classmethod
    def mobius_mul(self, x, t, dim=-1):
        """Performs scalar mobius multi for a batch of poincare vectors.
        
        t*x = tanh(t*arctanh(|x|)) * x/|x|
        
        Note: arctanh(x) is only defined for x < 1
        
        Args:
            x: Tesnor that contains mobius vectors.
            t: a tensor that contains the coefficients.
            dim: The dimension that the mobius scalar multiplication is to be
                performed.
        Returns:
            a mobius vector that contains the multipliciation result.
        """
        # Normx has 1 dimension less than x, it must have equal dimensions as t
        normx = x.norm(dim=dim, p=2, keepdim=True).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)
        t = t.unsqueeze(dim)
        if t.shape != normx.shape:
            raise ValueError('t.shape({}) != normx.shape({})'.format(t.shape, normx.shape))
        return torch.mul(torch.tanh(torch.mul(t, torch.atanh(normx))) / normx, x)

    @classmethod
    def mobius_matvec(self, m, x, c, dim: int = -1):
        """Calculates the mobius matrix multiplication.
        
        It is implemented based on Lemma 6 (eq. 27) of "Hyperbolic Neural 
        Network" by Octavian-Eugen Ganea.

        Args:
            m: matrix of dim (M,N).
            a: matix of dim (N,P).
            c: curvature of the poincare ball.
        Returns:
             the matrix product of two arrays of dim (M,P).
        """
        if m.dim() > 2 and dim != -1:
            raise RuntimeError(
                "broadcasted Mobius matvec is supported for the last dim only"
            )
        x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(PoincareBall.min_norm)
        sqrt_c = c ** 0.5 if c else PoincareBall.min_norm
        if dim != -1 or m.dim() == 2:
            mx = torch.tensordot(x, m, dims=([dim], [1]))
        else:
            mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
        mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)

        res_c = tanh(mx_norm / x_norm * artanh((sqrt_c * x_norm).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm))) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    @classmethod
    def mobius_bmm(self, bm, bx, c = 1):
        """Calculates the the mobius matrix multiplication for a batch of matrices.
        
        Args:
            bm: batch of matrices with dim(b, m, n) where b is the is the batch
                dimension.
            bx: batch of matrices with dim(b, m, p) where b is the is the batch
                dimension.
            c: curvature of the poincare ball.

        Returns:
             the matrix product of two arrays of dim (M,P).
        """
        if bm.shape[0] != bx.shape[0]:
            raise ValueError(
                    'The batch dimension of the bm {} does not match the batch'
                    'dimension of the bx {}'.format(bm.shape[0], bx.shape[0]))
        # TODO(mehrdad): optimize it using torch modules.
        batch_m = []
        for i in range(0, bm.shape[0]):
            batch_m.append(PoincareBall.mobius_matvec(bm[i], bx[i], c))
        return torch.stack(batch_m)

    @classmethod
    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    @classmethod
    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(PoincareBall.min_norm)

    @classmethod
    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = PoincareBall._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    @classmethod
    def ptransp(self, x, y, u, c):
        lambda_x = PoincareBall._lambda_x(x, c)
        lambda_y = PoincareBall._lambda_x(y, c)
        return PoincareBall._gyration(y, -x, u, c) * lambda_x / lambda_y

    @classmethod
    def ptransp_(self, x, y, u, c):
        lambda_x = PoincareBall._lambda_x(x, c)
        lambda_y = PoincareBall._lambda_x(y, c)
        return PoincareBall._gyration(y, -x, u, c) * lambda_x / lambda_y

    @classmethod
    def ptransp0(self, x, u, c):
        lambda_x = PoincareBall._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(PoincareBall.min_norm)

    @classmethod
    def from_hyperboloid(self, x, c=1, ideal=False):
        """Convert from hyperboloid model to Poincare ball model.
        
        Note: converting a point from hyperbolic to poincare ball, for a random
            vector is a not reversivle i.e. p != from_poincare(to_poincare(p)).
            p must be in hyperbolic plane to satisfy the reversiblity.
            
        Args:
            x: torch.tensor of shape (..., Minkowski_dim), where Minkowski_dim >= 3
            ideal: boolean. Should be True if the input vectors are ideal points, False otherwise
        Returns:
            torch.tensor of shape (..., Minkowski_dim - 1)
        """
        if ideal:
            return x[..., 1:] / (x[..., 0].unsqueeze(-1)).clamp_min(PoincareBall.min_norm)
        else:
            sqrtK = (1. / c) ** 0.5
            return sqrtK * x[..., 1:] / (sqrtK + x[..., 0].unsqueeze(-1)).clamp_min(PoincareBall.min_norm)
    
    @classmethod
    def distance(self, x, y, keepdim=True):
        """Hyperbolic distance on the Poincare ball with curvature c.
        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            y: torch.Tensor of size B x d with hyperbolic points
        Returns: torch,Tensor with hyperbolic distances, size B x 1
        """
        pairwise_norm = PoincareBall.mobius_add(-x, y).norm(dim=-1, p=2, keepdim=True)
        dist = 2.0 * torch.atanh(pairwise_norm.clamp(-1 + PoincareBall.min_norm, 1 - PoincareBall.max_norm))
        if not keepdim:
            dist = dist.squeeze(-1)
        return dist
    
    
    @classmethod
    def pairwise_distance(self, x, keepdim=False):
        """All pairs of hyperbolic distances (NxN matrix)."""
        return PoincareBall.distance(x.unsqueeze(-2), x.unsqueeze(-3), keepdim=keepdim)
    
    
    @classmethod
    def distance0(self, x, keepdim=True):
        """Computes hyperbolic distance between x and the origin."""
        x_norm = x.norm(dim=-1, p=2, keepdim=True)
        d = 2 * torch.atanh(x_norm.clamp(-1 + 1e-15, 1 - 1e-15))
        if not keepdim:
            d = d.squeeze(-1)
        return d


    @classmethod
    def mobius_midpoint_pair(self, x, y):
        """Computes hyperbolic midpoint beween x and y."""
        t1 = PoincareBall.mobius_add(-x, y)
        t2 = PoincareBall.mobius_mul(t1, 0.5)
        return PoincareBall.mobius_add(x, t2)

        
    @classmethod
    def _sq_gamma(self, v, r = 1):
        """Calculates Gamma factor for poincare ball.
        
        according to eq. 2.45 of "Gyrovector space" by Ungar.
        L  = 1/sqrt(1 - (distance0(v)/ball_radius)^2)
        
        args:
            v: vector of dime (N) in poincare ball.
            r: radius of poincare ball.
        returns:
            a scalar that corresponds to the gamma factor for a given vector
            in r-ball.
        """
        return 1/(1 - torch.pow(PoincareBall.distance0(v)/r, 2)).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)
        
        
    @classmethod
    def mobius_midpoint(self, v, a=None):
        """Calculates the Einstein Midpoint for a weighted list of vectors.
        
        Note taht the eq. is the generalization of eq. 3.134 of  
        "Gyrovector space" by Ungar.

        args:
            a: The list of co-efficients (weights) in which each co-efficient 
                belongs to the vector with same index.
            v: The M*N dimensional time-like vectors.
        returns:
            An N dimensional time-like vector.
        """
        if a is not None:
            # Calculates the weighted vectors
            # Note: attention weights are scalars, we probably can have non-mobius mul here.
            # assert len(a) == v.size(dim=0)
            w_v = PoincareBall.mobius_mul(v, torch.nn.functional.normalize(a, dim=-1))
        else:
            w_v = v
        # Calculates the gamma factors for all vectors        
        gamma_ws = torch.Tensor([PoincareBall._sq_gamma(w_v_j) for _, w_v_j in enumerate(w_v)])
        weights = (gamma_ws / (torch.sum(gamma_ws) - w_v.shape[0] / 2)).reshape(w_v.shape[0], 1)
        out = PoincareBall.mobius_mul(x=torch.sum(weights * v, dim=0), t=torch.tensor(0.5))
        # Generalized mobius midpoint
        return out
    
    @classmethod
    def poincare_attention_weight(self, q, k, beta, c):
        """Calculate an attention wight for a given query, and keys.
        
        a(q_i,k_j) = expmap(-Beta*distnace(q,k) - Constant)
        more details: "Hyperbolic attention network" eq. 2
        
        Note: Beta can be either set manually or learned from query vector.
        Note: Both vectors must already be in hyperbolic space. (no poincare, no klein)
    
        args:
            q: an N-dimensional query vector for a location i.
            k: keys for the memory locations (N-dimensional)
            c: 
        returns:
            a tesnor of dim (M, 1) where contains the attention weight based on 
            corresponding query and key vectors.
        """
        # assert q.size(dim=0) == k.size(dim=0)
        # TODO(): in tensorflow, the for loop is optimized in compile time.
        # either impletemtn it in tensorflow, or find a way to unroll the loop
        # for pytorch for GPU performance.
        return torch.stack([PoincareBall.expmap(beta*PoincareBall.distance(q[idx], k[idx]) - c) for idx in enumerate(q)])
    
    @classmethod
    def poincare_aggregation(self, q, k, v):
        """Calculates the poincare attention for the given query, key, and values.
    
        Note: if used in Graph attention mechansim, all the masked vectors need 
            to be filtered out from q,k,v before invoking this function.

        args:
            q: M*M*N dimensional matrix of queries, where M is the number of 
                locations to attend to.
            k: M*M*N dimensional matrix of keys, where M is the number of 
                locations to attend to.
            v: M*M*N dimensional matrix of values, where M is the number of 
                locations to attend to. Note that v is a matrix where each row 
                is a vector in poincate model.
        returns:
            The self-attention calculation in M*N dimensional matrix form .
            i_th row corresponds to the attention embeddings for a location i.
        """
        # assert q.size(dim=x) == k.size(dim=x) == v.size(dim=x) for all dims
        h = []
        for i, v_i in enumerate(v):
            h.append([PoincareBall._mobius_midpoint(PoincareBall.poincare_attention_weight(q[i], k[i]), v_i)])
        return torch.stack(h)
        
    @classmethod
    def concat(self, v, c = None):
        """Concatnates a matrix of dim (M, N) across the last dimension.
        
        Note: of the inputs are given as a batch, it assumes that dim (B, M, N)
            where B is the batch size.
        
        The concat operation is based on "Hyperbolic Neural Network++" paper.

        Args:
            v: a tensor of dim(M,N) where M is the number of vectors to be 
               concatnated, and N is the dimension of the vectors.
               
        Returns:
            A tensor of dim(M*N) (or dim(B, M*N in case of batch inputs)) in 
            the poincare ball of the same radius.
        """
        del c
        concat_dim = 1 if len(v.shape) == 3 else 0
        a = sc.beta(v.shape[concat_dim]*v.shape[-1]/2, 0.5) / sc.beta(v.shape[-1]/2, 0.5)
        # Note that the following multiplication should not be a mobius mul. 
        # It is there to normalize the raduis of the new PoincareBall to <= 1.
        return torch.cat(tensors=torch.unbind(v*a, dim=concat_dim), dim=concat_dim)

    @classmethod
    def split(self, v, m):
        """Splits a vector in poincare model to m vectors of the same size.
        
        The split happens along with the last dim.
        Please note that the N (vector dim) must be dividable to m
        
        The concat operation is based on "Hyperbolic Neural Network++" paper.

        Args:
            v: a tensor of dim(N = M*N`) where M is the number of vectors to be 
               split, and N is the dimension of the vector.
            m: the number of output vectors.

        Returns:
            A tensor of dim(M,N) in the poincare ball of the same radius.
        """
        split_dim = 1 if len(v.shape) == 2 else 0
        a = sc.beta(v.shape[-1]/(m*2), 0.5) / sc.beta(v.shape[-1]/2, 0.5)
        # Note that the following multiplication should not be a mobius mul. 
        # It is there to normalize the raduis of the new PoincareBall to <= 1.
        return torch.stack(torch.split(tensor=v*a, split_size_or_sections=int(v.shape[-1]/m), dim=-1), dim=split_dim)
    
    @classmethod
    def euclidean2poincare(self, x, c, scale = 1):
        """Approximates Euclidean vector to poincare ball across the last dimension.
        """
        p_c = c[0] * PoincareBall.max_norm
        norm = torch.norm(x, dim=-1).clamp(min=PoincareBall.min_norm, max=PoincareBall.euclidean_max) / scale
        x_exp = torch.exp(-norm).clamp(min=PoincareBall.min_norm, max=PoincareBall.euclidean_max)
        coef = - c  * (1 - x_exp) / (1 + x_exp)
        return ((coef.unsqueeze(-1) * x).t() / norm).t().clamp(min=-p_c, max=p_c)
  
    @classmethod
    def poincare2euclidean(self, p, c, scale = 1):
        """
        x = ln((1-p)/(1+p))
        """
        # Note that the ratio cannot be 0... it produces inf
        norm = torch.norm(p, dim=-1)
        lg = torch.log(((c - norm)/(c + norm)).clamp_min(PoincareBall.min_norm)).clamp(min=-PoincareBall.euclidean_max, max=PoincareBall.euclidean_max)
        coef = (-lg / norm.clamp_min(PoincareBall.min_norm)).clamp(min=-PoincareBall.euclidean_max, max=PoincareBall.euclidean_max)
        return scale * (coef.unsqueeze(-1) *  p).clamp(min=-PoincareBall.euclidean_max, max=PoincareBall.euclidean_max)
