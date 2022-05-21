"""Graph decoders."""
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear
from layers.poincare_layers import GraphConvolution as PGraphConvolution
from layers.hyp_att_layers import GraphAttentionLayer as HGraphAttentionLayer
from layers.poincare_layers import GraphAttentionLayer as PGraphAttentionLayer

from layers.hyp_layers import  HyperbolicGraphConvolution
from manifolds.poincare import PoincareBall


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True

class HGCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(HGCNDecoder, self).__init__(c)
        act = lambda x: x
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = HyperbolicGraphConvolution(
                manifold=self.manifold,
                in_features=args.dim, 
                out_features=args.n_classes, 
                dropout=args.dropout, 
                act=act, 
                c_in =self.c,
                c_out=self.c,
                use_att=args.use_att, 
                local_agg=args.local_agg,
                use_bias=args.bias)
        self.decode_adj = True

    def update_curvature(self, c):
        self.c = torch.Tensor([c])
        self.cls.update_curvature(self.c)

    def decode(self, x, adj):
        x = super(HGCNDecoder, self).decode(x, adj)
        return self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)



class PGCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(PGCNDecoder, self).__init__(c)
        act = lambda x: x
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)  # act=F.softmax
        self.decode_adj = True
        self.scale = torch.Tensor([args.scale])


    def update_curvature(self, c):
        self.c = torch.Tensor([c])
        self.cls.update_curvature(self.c)

    def decode(self, x, adj):
#        x = PoincareBall.euclidean2poincare(x, c=self.scale, scale=self.c)
#        x = x * 5 / torch.norm(x).clamp(1e-8)
#        x = x * 5
        x = super(PGCNDecoder, self).decode(x, adj)
#        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return x

#    def decode(self, x, adj):
#        x = super(GATDecoder, self).decode(x, adj)
#        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)



class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = GraphAttentionLayer(input_dim=args.dim, 
                                       output_dim=args.n_classes, 
                                       dropout=args.dropout, 
                                       activation=F.elu, 
                                       alpha=args.alpha, 
                                       nheads=1, 
                                       concat=True)
        self.decode_adj = True

    def decode(self, x, adj):
        x = super(GATDecoder, self).decode(x, adj)
        return x

#    def decode(self, x, adj):
#        x = super(GATDecoder, self).decode(x, adj)
#        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)


class HATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(HATDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = HGraphAttentionLayer(
                manifold=self.manifold,
                input_dim=args.dim, 
                output_dim=args.n_classes, 
                dropout=args.dropout, 
                activation=F.elu, 
                alpha=args.alpha, 
                nheads=1, 
                concat=True,  
                curvature=self.c, 
#                use_bias= False)
                use_bias= args.bias)
        self.decode_adj = True

    def decode(self, x, adj):
        x = super(HATDecoder, self).decode(x, adj)
        return self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)


class PGATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(PGATDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = GraphAttentionLayer(input_dim=args.dim, 
                                       output_dim=args.n_classes, 
                                       dropout=args.dropout, 
                                       activation=F.softmax, 
                                       alpha=args.alpha, 
                                       nheads=1, 
                                       concat=True)
        self.decode_adj = True
        self.scale = torch.Tensor([args.scale])

    def decode(self, x, adj):
        x = PoincareBall.euclidean2poincare(x, c=self.scale, scale=self.c)
#        x = x * 5 / torch.norm(x).clamp(1e-8)
        x = x * 5
        x = super(PGATDecoder, self).decode(x, adj)
#        x = PoincareBall.proj(PoincareBall.expmap0(PoincareBall.proj_tan0(x, self.c), c=self.c), c=self.c)
#        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
#        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return x
 

class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
#    'PGAT': GATDecoder,
    'PGAT': PGATDecoder,
    'PGCN': PGCNDecoder,
    # Note: if the manifold is Poincare, PGCNDecoder works better but for Hyperboloid, even the lindear does not work... so, our technicque only works if the space is
    # PoincareBall model.
#    'PGCN': LinearDecoder,
#    'HAT': LinearDecoder,
    'HAT': HATDecoder,
    'HGATV0': LinearDecoder,
    'HNN': LinearDecoder,
#    'HGCN': HGCNDecoder,
    'HGCN': HGCNDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}
