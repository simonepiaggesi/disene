import numpy as np
import networkx as nx
import os
import pandas as pd

from functools import partial
from multiprocessing import Pool
from collections import Counter

THRESH = 1e-9

from scipy.spatial.distance import cdist, pdist
import scipy.sparse as sparse
from scipy.sparse import csgraph
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

import torch
import torch_geometric as tg
from torch_geometric.utils import train_test_split_edges, remove_self_loops, add_random_edge
from torch_geometric.utils import from_networkx, negative_sampling, to_networkx
import torch_geometric.transforms as T

from torch_geometric.datasets import ExplainerDataset, StochasticBlockModelDataset, AttributedGraphDataset
from torch_geometric.datasets.graph_generator import BAGraph, ERGraph
from torch_geometric.datasets.motif_generator import MotifGenerator, CustomMotif
from torch_geometric.data import Data
from itertools import combinations
import pickle

from sklearn.preprocessing import MultiLabelBinarizer

def multilabel_f1_score(y, predictions, average='micro'):
    number_of_labels = y.shape[1]
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = np.argsort(predictions, axis=1)

    # the true number of labels for each node
    num_labels = np.sum(y, axis=1)
    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-int(num):].tolist())

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(classes=range(number_of_labels)).fit_transform(pred_reshaped)
    f1 = f1_score(y, pred_transformed, average=average)
    return f1

def giant_comp(g, nodelist=None, edgelist=None):
    if nodelist is not None:
        subg = g.subgraph(nodelist)
        return list(sorted(nx.connected_components(subg), key=len, reverse=True)[0])
    elif edgelist is not None:
        subg = g.edge_subgraph(edgelist)
        return list(sorted(nx.connected_components(subg), key=len, reverse=True)[0])
    else:
        return list(sorted(nx.connected_components(g), key=len, reverse=True)[0])

def prepare_attributed_dataset(dataset_name):

    dataset = AttributedGraphDataset("../../data/", dataset_name)

    transform = T.Compose([
                T.LargestConnectedComponents(),
                T.ToUndirected(),
                T.NormalizeFeatures()])

    data = transform(dataset._data)
    graph = to_networkx(data, to_undirected=True, remove_self_loops=True)

    return data, graph

def prepare_protein_dataset(dataset_name):

    x = pd.read_csv('../../data/%s/features.csv'%(dataset_name), header=None).values
    edge_index = pd.read_csv('../../data/%s/edge_list.csv'%(dataset_name)).iloc[:,:2].astype(int).values
    y = pd.read_csv('../../data/%s/labels.csv'%(dataset_name), header=None).values

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index.T),
                y=torch.tensor(y, dtype=torch.float32))

    transform = T.Compose([
                T.LargestConnectedComponents(),
                T.ToUndirected()])

    data = transform(data)
    graph = to_networkx(data, node_attrs=["y"], to_undirected=True, remove_self_loops=True)

    return data, graph

def prepare_dataset(dataset_name, num_cliques=32, num_nodes=10):

    if dataset_name in ["ba_cliques", 'er_cliques']:
        
        clique_motif = nx.from_edgelist(combinations(range(num_nodes), 2), create_using=nx.Graph)
        CliqueMotif = CustomMotif(clique_motif)

        tg.seed_everything(42)
        if dataset_name == "ba_cliques":
            dataset = ExplainerDataset(
                    graph_generator=BAGraph(num_nodes=num_cliques*num_nodes, num_edges=5),
                    motif_generator=CliqueMotif,
                    num_motifs=num_cliques,
                    )
        elif dataset_name == "er_cliques":
            dataset = ExplainerDataset(
                graph_generator=ERGraph(num_nodes=num_cliques*num_nodes, edge_prob=0.05),
                motif_generator=CliqueMotif,
                num_motifs=num_cliques,
            )

        x = torch.eye(dataset[0].num_nodes)
        data = Data(x=x, edge_index=dataset[0].edge_index, edge_mask=dataset[0].edge_mask, node_mask=dataset[0].node_mask)

        #cliques identity label
        data.node_mask[data.node_mask==1] = torch.tensor([[1.*(cl+1),]*clique_motif.number_of_nodes() for cl in range(num_cliques)]).ravel()
        data.edge_mask[data.edge_mask==1] = torch.tensor([[1.*(cl+1),]*clique_motif.number_of_edges()*2 for cl in range(num_cliques)]).ravel()

        #add random edges and update edge_mask
        edge_index, added_edges = add_random_edge(data.edge_index, p=0.05, force_undirected=True)
        data.edge_index = edge_index
        data.edge_mask = torch.cat([data.edge_mask, torch.zeros(added_edges.shape[1])])

        data.node_mask = torch.as_tensor(data.node_mask, dtype=int)
        data.edge_mask = torch.as_tensor(data.edge_mask, dtype=int)

    if dataset_name in ["stochastic_block_model"]:

        dataset_name = '%s_%d_%d'%(dataset_name, num_cliques, num_nodes)

        unsorted_graph = nx.read_weighted_edgelist('../../data/%s.edgelist.gz'%(dataset_name))
        unsorted_graph = nx.relabel_nodes(unsorted_graph, int)
        graph = nx.Graph()
        graph.add_nodes_from(sorted(unsorted_graph.nodes()))
        graph.add_edges_from(sorted(unsorted_graph.edges()))

        import community as community_louvain
        com_dict = community_louvain.best_partition(graph, resolution=2, random_state=11)
        node_attrs = {i: int(com_dict[i]+1) for i in graph.nodes()}
        edge_attrs = {(i,j):int(com_dict[i]+1) if com_dict[i]==com_dict[j] else 0 for i,j in graph.edges()}
        nx.set_node_attributes(graph, node_attrs, 'node_mask')
        nx.set_edge_attributes(graph, edge_attrs, 'edge_mask')

        data = from_networkx(graph)
        data.x = torch.eye(graph.number_of_nodes())
        data.node_mask = torch.as_tensor(data.node_mask, dtype=int)
        data.edge_mask = torch.as_tensor(data.edge_mask, dtype=int)

    if dataset_name in ["ring_of_cliques"]:

        dataset_name = '%s_%d_%d'%(dataset_name, num_cliques, num_nodes)

        unsorted_graph = nx.read_weighted_edgelist('../../data/%s.edgelist.gz'%(dataset_name))
        unsorted_graph = nx.relabel_nodes(unsorted_graph, int)
        graph = nx.Graph()
        graph.add_nodes_from(sorted(unsorted_graph.nodes()))
        graph.add_edges_from(sorted(unsorted_graph.edges()))

        import community as community_louvain
        com_dict = community_louvain.best_partition(graph, resolution=2, random_state=11)
        node_attrs = {i: int(com_dict[i]+1) for i in graph.nodes()}
        edge_attrs = {(i,j):int(com_dict[i]+1) if com_dict[i]==com_dict[j] else 0 for i,j in graph.edges()}
        nx.set_node_attributes(graph, node_attrs, 'node_mask')
        nx.set_edge_attributes(graph, edge_attrs, 'edge_mask')

        data = from_networkx(graph)
        data.x = torch.eye(graph.number_of_nodes())

        #add random edges and update edge_mask
        edge_index, added_edges = add_random_edge(data.edge_index, p=0.1, force_undirected=True)
        data.edge_index = edge_index
        data.edge_mask = torch.cat([data.edge_mask, torch.zeros(added_edges.shape[1])])

        data.node_mask = torch.as_tensor(data.node_mask, dtype=int)
        data.edge_mask = torch.as_tensor(data.edge_mask, dtype=int)

    graph = to_networkx(data, node_attrs=["node_mask"],
                              edge_attrs=["edge_mask"],
                              to_undirected=True, remove_self_loops=True)

    return data, graph

def emb_mask_single_dim(edge_emb, edge_scores_all, max_dims, dim):

    mask = edge_emb[:, dim] - edge_scores_all[:, dim]
            
    return dim, mask

def emb_mask_parallel(edge_emb, gt_masks, edge_scores_all, workers=12):

    def weighted_f1_score(y_true, y_score):
        if y_score.sum()==0.:
            return 0.
        precision = np.sum(y_true*y_score)/np.sum(y_score)
        recall = np.sum(y_true*(y_score>0.).astype(int))/np.sum(y_true)
        if (precision==0.) or (recall==0.):
            return 0.
        return 2*precision*recall/(precision+recall)
    
    dimensions = edge_emb.shape[1]
    
    with Pool(processes=workers) as pool:  
        ilist = pool.map(partial(emb_mask_single_dim, edge_emb, edge_scores_all, dimensions), range(dimensions))
        pool.close()
        pool.join() 
            
    if len(ilist)>0:
        dims, emb_masks = zip(*ilist)

        dims = np.array(dims)
        emb_masks = np.array(emb_masks)

        real_masks = np.maximum(emb_masks, THRESH)
        bin_masks = (emb_masks>THRESH).astype(float)

        scores = {}
        scores['f1_weighted'] = cdist(gt_masks, real_masks, lambda u,v: weighted_f1_score(u,v))
        scores['jaccard_score'] = 1. - pdist(bin_masks, metric='jaccard')

        M = real_masks.copy()
        Z = M.sum(axis=1).reshape(-1,1)
        E = -np.sum((M/Z)*np.log(M/Z), axis=1)
        scores['dim_ent_weighted'] = E/np.log(M.shape[1])

        return dims, emb_masks, scores
    else:
        return np.array([]), np.array([]), np.array([])


def shap_metrics(shap_masks, gt_masks):

    def weighted_f1_score(y_true, y_score):
        if y_score.sum()==0.:
            return 0.
        precision = np.sum(y_true*y_score)/np.sum(y_score)
        recall = np.sum(y_true*(y_score>0.).astype(int))/np.sum(y_true)
        if (precision==0.) or (recall==0.):
            return 0.
        return 2*precision*recall/(precision+recall)
    
    scores = {}

    scores['f1_weighted'] = cdist(gt_masks, np.maximum(shap_masks, 0.), lambda u,v: weighted_f1_score(u,v))

    return shap_masks, scores

def compute_plausibility_score(masks, scores, dict_task):
    
    clique_labels = dict_task['gt_labels']
    y_train = dict_task['y_train']
    y_test = dict_task['y_test']
    
    idx_test = np.arange(y_test[y_test==1].shape[0]) + y_train[y_train==1].shape[0]

    local_scores = []

    for idx in idx_test:
        if clique_labels[idx]>0:
            gt_clique_label = clique_labels[idx] -1
            w = np.maximum(masks[:,idx], THRESH)
            s = scores['f1_weighted'][gt_clique_label]
            
            if w.sum()>0:
                local_scores.append(np.sum(w*s/w.sum()))
            else: 
                local_scores.append(0.)
            
    return np.mean(local_scores)