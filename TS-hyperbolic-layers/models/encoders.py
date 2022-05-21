"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn

import manifolds
from layers.att_layers import GraphAttentionLayer
from layers.hyp_att_layers import MultiHeadGraphAttentionLayer as MultiHeadHGAT
from layers.hyp_att_layers import GraphAttentionLayer as HATlayer
from layers.poincare_layers import GraphAttentionLayer as PGraphAttentionLayer

import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
from manifolds.hyperboloid import Hyperboloid
from manifolds.poincare import PoincareBall
import layers.poincare_layers as p_layer


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True

class PGCN(Encoder):
    """
    Poincare Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(PGCN, self).__init__(c)
        assert args.num_layers > 0
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True
        self.scale = torch.Tensor([args.scale])
        self.layer_norm = nn.LayerNorm(out_dim)

    def encode(self, x, adj):
        # convert the Euclidean embeddings to poincare embeddings
#        x = PoincareBall.euclidean2poincare(x, c=self.scale, scale=self.curvatures[0])
        x = super(PGCN, self).encode(x, adj)
#        x = self.layer_norm(x)
        x = PoincareBall.proj(PoincareBall.expmap0(PoincareBall.proj_tan0(x, self.c), c=self.c), c=self.c)
#        x = x * 1 / torch.norm(x).clamp(1e-8)

        return x

  

class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        """
        Args:
            x: a vector in Euclidean space.
        """
        # Performs Hyperboloid.proj_tan0 on Euclidean vector.
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
#        dims, acts = get_dim_act(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        # convert the Euclidean embeddings to poincare embeddings
#        x = PoincareBall.euclidean2poincare(x, c=1/self.curvatures[0], scale=self.curvatures[0])
        x = super(GAT, self).encode(x, adj)
#        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return x

class HAT(Encoder):
    """
    Hyperbolic Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(HAT, self).__init__(c)
        assert args.num_layers > 0
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    HATlayer(
                            manifold=self.manifold,
                            input_dim=in_dim, 
                            output_dim=out_dim, 
                            dropout=args.dropout, 
                            activation=act, 
                            alpha=args.alpha, 
                            nheads=args.n_heads, 
                            concat=concat, 
                            curvature=self.curvatures[i], 
                            use_bias=args.bias))

        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True

    def update_curvature(self, c):
#        super(PGAT, self).update_curvature(c)
        self.c = torch.Tensor([c])
        for idx, _ in enumerate(self.curvatures):
            self.curvatures[idx] = self.c
        for layer in self.layers:
            layer.update_curvature(self.c)

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]), c=self.curvatures[0])
#            x = PoincareBall.euclidean2poincare(x, c=self.curvatures[0])
        return super(HAT, self).encode(x_hyp, adj)


class PGAT(Encoder):
    """
    Poincare Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(PGAT, self).__init__(c)
        assert args.num_layers > 0
#        dims, acts = get_dim_act(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True
        self.scale = torch.Tensor([args.scale])
        self.layer_norm = nn.LayerNorm(out_dim * args.n_heads)



    def encode(self, x, adj):
        # convert the Euclidean embeddings to poincare embeddings
        x = PoincareBall.euclidean2poincare(x, c=self.scale, scale=self.curvatures[0])
        x = super(PGAT, self).encode(x, adj)
        x = PoincareBall.proj(PoincareBall.expmap0(PoincareBall.proj_tan0(x, self.c), c=self.c), c=self.c)
#        x = self.layer_norm(x)
#        x = x * 1 / torch.norm(x).clamp(1e-8)
        return x


class HGAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, attention_version, c, args):
        super(HGAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.manifold = getattr(manifolds, args.manifold)()
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
#            concat = True
            gat_layers.append(
                    MultiHeadHGAT(
                            manifold=self.manifold,
                            input_dim=in_dim, 
                            output_dim=out_dim, 
                            curvature=self.curvatures[i],
                            dropout=args.dropout, 
                            activation=act, 
                            alpha=args.alpha, 
                            nheads=args.n_heads, 
#                           concat=concat,
                            self_attention_version=attention_version))

        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = Hyperboloid.proj(Hyperboloid.expmap0(Hyperboloid.proj_tan0(x, c=1), c=1), c=1)
        return super(HGAT, self).encode(x_hyp, adj)


class HGATV0(HGAT):
    """
    Graph Attention Networks.
    """
    def __init__(self, c, args):
        super(HGATV0, self).__init__(attention_version='v0', c=c, args=args)


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
