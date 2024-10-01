import numpy as np
import networkx as nx
import os
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges, remove_self_loops, add_random_edge
from torch_geometric.utils import from_networkx, negative_sampling, to_networkx
from torch_geometric.nn import Node2Vec, SGConv, GCNConv, GAE, InnerProductDecoder, GraphSAGE

from torchvision.ops import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = torch.tensor(1e-15)

class WalkGAE(GAE):
    def __init__(self, encoder, in_channels, out_channels):
        super(WalkGAE, self).__init__(encoder=encoder, decoder=InnerProductDecoder())

        self.input_dim = in_channels
        self.embedding_dim = out_channels

    def forward(self, x, edge_index):
        z_conv = self.encode(x, edge_index)
        return z_conv

    def walk_loss(self, z, pos_rw, neg_rw, sig=True):

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = z[start].view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = z[rest.view(-1)].view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        if sig:
            pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        else:
            pos_loss = -torch.log(out + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = z[start].view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = z[rest.view(-1)].view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        if sig:
            neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        else:
            neg_loss = -torch.log(1 - out + EPS).mean()

        return pos_loss + neg_loss

class DiSeNE(GAE):
    def __init__(self, encoder, in_channels, hidden_channels, out_channels):
        super(DiSeNE, self).__init__(encoder=encoder, decoder=InnerProductDecoder())

        self.linear = nn.Linear(hidden_channels, out_channels)
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.embedding_dim = out_channels

    def forward(self, x, edge_index):
        z_conv = self.encode(x, edge_index)
        return self.linear(z_conv).relu()

    def walk_loss(self, z, pos_rw, neg_rw, sig=True):

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = z[start].view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = z[rest.view(-1)].view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        if sig:
            pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        else:
            pos_loss = -torch.log(out + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = z[start].view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = z[rest.view(-1)].view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        if sig:
            neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        else:
            neg_loss = -torch.log(1 - out + EPS).mean()

        return pos_loss + neg_loss

    def size_loss(self, h):
        axs = torch.arange(h.dim())        
        mask_size = torch.sum(h, axis=tuple(axs[1:]))
        mask_norm = mask_size / torch.sum(mask_size, axis=0)
        mask_ent = torch.sum(- mask_norm * torch.log(mask_norm + EPS), axis=0)
        return torch.log(torch.tensor(h.shape[0], dtype=torch.float32)).to(device) - torch.mean(mask_ent) 

    def orth_loss(self, h):
        Q = h.matmul(h.T)
        I = torch.eye(Q.shape[0]).to(device)
        return nn.MSELoss(reduction='mean')(Q, I)


class DINE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x):
        h = self.linear1(x).sigmoid()
        out = self.linear2(h)
        return out, h
    
    def recon_loss(self, z, z_hat):
        return nn.MSELoss(reduction='mean')(z_hat, z)

    def size_loss(self, h):
        axs = torch.arange(h.dim())        
        mask_size = torch.sum(h, axis=tuple(axs[1:]))
        mask_norm = mask_size / torch.sum(mask_size, axis=0)
        mask_ent = torch.sum(- mask_norm * torch.log(mask_norm + EPS), axis=0)
        return torch.log(torch.tensor(h.shape[0], dtype=torch.float32)).to(device) - torch.mean(mask_ent) 

    def orth_loss(self, h):
        Q = h.matmul(h.T)
        I = torch.eye(Q.shape[0]).to(device)
        return nn.MSELoss(reduction='mean')(Q/torch.norm(Q), I/torch.norm(I))

class SGCEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_k_hops):
        super().__init__()
        self.conv1 = SGConv(in_channels=in_channels,
                            out_channels=out_channels,
                            K=num_k_hops, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class MLPEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_k_hops):
        super().__init__()
        self.mlp = MLP(in_channels=in_channels, 
                       hidden_channels=[out_channels,]*num_k_hops)

    def forward(self, x, edge_index=None):
        x = self.mlp(x)
        return x

def train_dine_emb(z, channels, learning_rate=0.1, epochs=2000, return_model=False):

    in_channels, hidden_channels = channels

    model = DINE(in_channels, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    Z = torch.from_numpy(z).to(device)
    N = Z.shape[0]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        idx_shuffled = torch.randperm(N)
        Z_hat, H = model(Z[idx_shuffled] + 0.2*torch.randn(Z.shape).to(device))#torch.normal(torch.zeros_like(Z), torch.ones_like(Z)*0.2))
        
        Recon_loss = model.recon_loss(Z[idx_shuffled], Z_hat)
        P = (H * H.sum(axis=0))
        Size_loss = model.size_loss(H.T) 
        Orth_loss = model.orth_loss(P.T)
        loss = Recon_loss + 1.*Size_loss + 1.*Orth_loss  
        
        loss.backward()
        optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z_hat, H = model(Z)
    return H.detach().cpu().numpy()


def train_sgc_emb(graph_data, channels, learning_rate=0.01, epochs=50, k_hops=1, window_size=5, return_model=False):
    
    in_channels, out_channels = channels

    walker = Node2Vec(graph_data.edge_index, embedding_dim=0,
                        walk_length=20, context_size=window_size+1,
                        walks_per_node=10, num_negative_samples=1).to(device)
    
    model = WalkGAE(SGCEncoder(in_channels, out_channels, k_hops), in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()

        loader = walker.loader(batch_size=128, shuffle=True, num_workers=12)

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            Z = model(graph_data.x, graph_data.edge_index)
            Batch_loss = model.walk_loss(Z, pos_rw.to(device), neg_rw.to(device))
            Batch_loss.backward()
            optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z = model(graph_data.x, graph_data.edge_index)
    return Z.detach().cpu().numpy()


def train_mlp_emb(graph_data, channels, learning_rate=0.01, epochs=50, k_hops=1, window_size=5, return_model=False):
    
    in_channels, out_channels = channels

    walker = Node2Vec(graph_data.edge_index, embedding_dim=0,
                        walk_length=20, context_size=window_size+1,
                        walks_per_node=10, num_negative_samples=1).to(device)

    model = WalkGAE(MLPEncoder(in_channels, out_channels, k_hops), in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()

        loader = walker.loader(batch_size=128, shuffle=True, num_workers=12)

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            Z = model(graph_data.x, None)
            Batch_loss = model.walk_loss(Z, pos_rw.to(device), neg_rw.to(device))
            Batch_loss.backward()
            optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z = model(graph_data.x, None)
    return Z.detach().cpu().numpy()


def train_isgc_emb(graph_data, channels, learning_rate=0.01, epochs=50, k_hops=1, window_size=5, return_model=False):
    
    in_channels, hidden_channels, out_channels = channels

    walker = Node2Vec(graph_data.edge_index, embedding_dim=0,
                        walk_length=20, context_size=window_size+1,
                        walks_per_node=10, num_negative_samples=1).to(device)
    
    model = DiSeNE(SGCEncoder(in_channels, hidden_channels, k_hops), in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()

        loader = walker.loader(batch_size=128, shuffle=True, num_workers=12)

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            Z = model(graph_data.x, graph_data.edge_index)
            Batch_Recon_loss = model.walk_loss(Z, pos_rw.to(device), neg_rw.to(device))
            P = (Z * Z.sum(axis=0))/(torch.norm(Z * Z.sum(axis=0), p=2, dim=0)+EPS)
            Batch_Size_loss = model.size_loss(Z.T) 
            Batch_Orth_loss = model.orth_loss(P.T)
            Batch_loss = Batch_Recon_loss + Batch_Size_loss + Batch_Orth_loss
            Batch_loss.backward()
            optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z = model(graph_data.x, graph_data.edge_index)
    return Z.detach().cpu().numpy()

def train_imlp_emb(graph_data, channels, learning_rate=0.01, epochs=50, k_hops=1, window_size=5, return_model=False):
    
    in_channels, hidden_channels, out_channels = channels

    walker = Node2Vec(graph_data.edge_index, embedding_dim=0,
                        walk_length=20, context_size=window_size+1,
                        walks_per_node=10, num_negative_samples=1).to(device)
    
    model = DiSeNE(MLPEncoder(in_channels, hidden_channels, k_hops), in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()

        loader = walker.loader(batch_size=128, shuffle=True, num_workers=12)

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            Z = model(graph_data.x, None)
            Batch_Recon_loss = model.walk_loss(Z, pos_rw.to(device), neg_rw.to(device))
            P = (Z * Z.sum(axis=0))/(torch.norm(Z * Z.sum(axis=0), p=2, dim=0)+EPS)
            Batch_Size_loss = model.size_loss(Z.T) 
            Batch_Orth_loss = model.orth_loss(P.T)
            Batch_loss = Batch_Recon_loss + Batch_Size_loss + Batch_Orth_loss
            Batch_loss.backward()
            optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z = model(graph_data.x, None)
    return Z.detach().cpu().numpy()


def train_deepwalk_emb(graph, emb_dim, window_size=5, seed=42):

    from node2vec import Node2Vec
    from gensim.models import Word2Vec

    node_name = sorted(map(int, graph.nodes()))

    node2vec = Node2Vec(graph, dimensions=2, walk_length=20, num_walks=10, seed=seed)
    model = Word2Vec(node2vec.walks, vector_size=emb_dim, window=window_size, min_count=0, sg=1, workers=6, seed=seed)
    Z = np.array([model.wv[str(n)] for n in node_name])

    return Z


def train_infwalk_emb(graph, emb_dim, window_size=5):

    import scipy as sp

    def compute_pmi_inf(adj, rank_approx=None):
        lap, deg_sqrt = sp.sparse.csgraph.laplacian(adj, normed=True, return_diag=True)
        iden = sp.sparse.identity(adj.shape[0])
        vol = adj.sum()
        ss_probs_invsqrt = np.sqrt(vol) / deg_sqrt # inverse square root of stationary probabilities
        lap_pinv = np.linalg.pinv(lap.todense(), hermitian=True)
        return 1. + ss_probs_invsqrt[:,np.newaxis] * np.array(lap_pinv - iden) * ss_probs_invsqrt[np.newaxis,:]

    def compute_log_ramp(pmi_inf, T=10, thresh=np.finfo(float).eps):
        pmi_inf_trans = T * np.log(np.maximum(thresh, 1. + pmi_inf / T))
        return pmi_inf_trans

    def compute_mat_embed(mat, dims=128):
        w, v = sp.sparse.linalg.eigsh(mat, k=dims)
        return np.sqrt(np.abs(w))[np.newaxis,:] * v

    node_name = sorted(map(int, graph.nodes()))

    A = nx.to_scipy_sparse_matrix(graph, nodelist=node_name)
    pmi_inf = compute_pmi_inf(A)
    pmi_inf_trans = compute_log_ramp(pmi_inf, T=window_size)
    Z = compute_mat_embed(pmi_inf_trans, emb_dim)
    
    return Z


def train_gae_emb(graph_data, channels, learning_rate=0.01, epochs=200, return_model=False):

    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 2*out_channels)
            self.conv2 = GCNConv(2*out_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)
    
    in_channels, out_channels = channels

    model = GAE(GCNEncoder(in_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()
        Z = model(graph_data.x, graph_data.edge_index)
        Batch_loss = model.recon_loss(Z, graph_data.edge_index)
        Batch_loss.backward()
        optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z = model(graph_data.x, graph_data.edge_index)
    return Z.detach().cpu().numpy()

def train_sage_emb(graph_data, channels, learning_rate=0.01, epochs=50, k_hops=1, window_size=5, return_model=False):
    
    in_channels, out_channels = channels

    walker = Node2Vec(graph_data.edge_index, embedding_dim=0,
                        walk_length=20, context_size=window_size+1,
                        walks_per_node=10, num_negative_samples=1).to(device)
    
    model = WalkGAE(GraphSAGE(in_channels, out_channels, k_hops), in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()

        loader = walker.loader(batch_size=128, shuffle=True, num_workers=12)

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            Z = model(graph_data.x, graph_data.edge_index)
            Batch_loss = model.walk_loss(Z, pos_rw.to(device), neg_rw.to(device))
            Batch_loss.backward()
            optimizer.step()

    if return_model:
        return model

    model.eval()
    with torch.no_grad():
        Z = model(graph_data.x, graph_data.edge_index)
    return Z.detach().cpu().numpy()