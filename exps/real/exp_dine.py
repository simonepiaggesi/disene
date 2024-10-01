import numpy as np
import networkx as nx
import os
import pandas as pd
import argparse
import pickle

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges, remove_self_loops
from torch_geometric.utils import from_networkx, negative_sampling, to_networkx
from torch_geometric.nn import Node2Vec, SGConv, GCNConv, GAE, InnerProductDecoder
from torch_geometric.datasets import AttributedGraphDataset

import sys
sys.path.append("../")

from utils import *
from models import train_dine_emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default="cora", type=str,
                  help='input dataset')
#"cora", "wiki", "facebook", "ppi"

parser.add_argument('--model', default="deepwalk", type=str,
                  help='input model')
#'deeepwalk', 'gae'

parser.add_argument('--runs', default=5, type=int,
                  help='number of training sessions')

#########################################################

def main():

    params = vars(parser.parse_args())

    DATASET = params['dataset']
    MODEL = params['model'].upper()

    data, graph = prepare_attributed_dataset(DATASET)
        
    import community as community_louvain
    com_dict = community_louvain.best_partition(graph, resolution=1, random_state=11)
    node_lbl = np.array([com_dict[i] for i in range(data.num_nodes)])
    
    for SEED in range(params['runs']):
        
        tg.seed_everything(42*SEED)

        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.00, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False)(data)              
        train_graph_discn = to_networkx(train_data, to_undirected=True, remove_self_loops=True)

        #giant component
        giant = sorted(nx.connected_components(train_graph_discn), key=len, reverse=True)[0]
        train_graph = train_graph_discn.subgraph(giant)

        #build geodesic matrix
        dist_file = '../../data/%s.%d.spath.pickle'%(DATASET, SEED)
        if not os.path.isfile(dist_file):
            from collections import OrderedDict
            spd = dict()
            for i, d in nx.all_pairs_shortest_path_length(train_graph):
                spd[i] = OrderedDict(sorted(d.items()))
            pickle.dump(spd, open(dist_file, 'wb'))
        spd = pickle.load(open(dist_file, 'rb'))
        
        for D_in in [int(2**x) for x in range(3, 10)]:
            for D_out in [128]:

                ####### Training

                EMB_FOLDER = '../../output/real/%s+dine-%s/'%(MODEL.lower(), 'featureless')
                IN_FOLDER = '../../output/real/%s-%s/'%(MODEL.lower(), 'featureless')

                os.makedirs(EMB_FOLDER, exist_ok=True)
                emb_file = EMB_FOLDER+'%s_%s_%ddims.DINE_%ddims.%d.npz'%(DATASET, MODEL, D_in, D_out, SEED)

                if not os.path.isfile(emb_file):
                    in_file = IN_FOLDER+'%s_%s_%ddims.%d.npz'%(DATASET, MODEL, D_in, SEED)
                    X = np.load(in_file, allow_pickle=True)['arr_0']
                    Z = train_dine_emb(X, (D_in, D_out))
                    np.savez_compressed(emb_file, Z)
                    print(DATASET, MODEL, D_in, D_out, SEED, 'dine embeddings saved')

                ####### Dimensional Metrics

                Z = np.load(emb_file, allow_pickle=True)['arr_0'] 
                    
                edge_idx = np.array(train_graph.edges)
                node_idx = np.unique(edge_idx)

                edge_Z = (Z[edge_idx][:,0] * Z[edge_idx][:,1])
                edge_lbl = np.array(['-'.join(map(str, sorted([li, lj]))) for li,lj in node_lbl[edge_idx]])
                lbl_masks = np.array([[lbl==cl for lbl in edge_lbl] for cl in np.unique(edge_lbl)]).astype(int)
                
                METRIC_FOLDER = EMB_FOLDER+'/linearshap_metrics/'
                mean_edge_Z = np.mean(edge_Z, axis=0).reshape(1,-1) #LinearShap

                os.makedirs(METRIC_FOLDER, exist_ok=True)

                scores_file = METRIC_FOLDER + '%s_%s_%ddims.DINE_%ddims.%d.scores.npz'%(DATASET, MODEL, D_in,  D_out, SEED)
                masks_file = METRIC_FOLDER + '%s_%s_%ddims.DINE_%ddims.%d.masks.npz'%(DATASET, MODEL, D_in,  D_out, SEED)
                if not os.path.isfile(scores_file):
                    dims, emb_masks, emb_scores = emb_mask_parallel(edge_Z, lbl_masks, mean_edge_Z)
                    emb_scores['edge_index'] = edge_idx
                    np.savez_compressed(scores_file, **{k:emb_scores[k] for k in emb_scores.keys()})
                    np.savez_compressed(masks_file, emb_masks)
                
                    print(DATASET, MODEL, D_in, D_out, SEED, 'dim. metrics saved')

                ####### Positional Metrics

                emb_masks = np.load(masks_file, allow_pickle=True)['arr_0'] 
                dims, edges = np.where(emb_masks>THRESH)

                spath_file = METRIC_FOLDER + '%s_%s_%ddims.DINE_%ddims.%d.spath.npz'%(DATASET, MODEL, D_in, D_out, SEED)
                if not os.path.isfile(spath_file):
                    S_dict = {'min':[], 'max':[], 'mean':[], 'sum':[]}
                    for d in np.unique(dims):
                        nodelist = np.unique(edge_idx[edges[dims==d]]) 
                        S = np.array([list(spd[j].values()) for j in nodelist]).T
                        S = 1/(1+S)
                        S_dict['min'].append(S.min(axis=1))
                        S_dict['max'].append(S.max(axis=1))
                        S_dict['mean'].append(S.mean(axis=1))
                        S_dict['sum'].append(S.sum(axis=1))
                    np.savez_compressed(spath_file, **{k:np.array(S_dict[k]) for k in S_dict.keys()})

                    print(DATASET, MODEL, D_in, D_out, SEED, 'pos. metrics saved')

if __name__ == '__main__':
    main()