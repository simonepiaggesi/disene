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
from models import train_deepwalk_emb, train_infwalk_emb, train_sgc_emb, train_sage_emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default="ring_of_cliques", type=str,
                  help='input dataset')
#"ring_of_cliques", "stochastic_block_model", "ba_cliques", "er_cliques", "tree_cliques", "tree_grids"

parser.add_argument('--model', default="deepwalk", type=str,
                  help='input model')
#'deepwalk', 'infwalk', 'gae', 'sage'

parser.add_argument('--runs', default=5, type=int,
                  help='number of training sessions')

#########################################################

def main():

    params = vars(parser.parse_args())

    num_nodes = 10
    num_cliques = 32

    DATASET = params['dataset']
    MODEL = params['model'].upper()

    data, graph = prepare_dataset(DATASET, num_cliques, num_nodes)
    DATASET = '%s_%d_%d'%(DATASET, num_cliques, num_nodes)

    for SEED in range(params['runs']):

        tg.seed_everything(42*SEED)

        train_data, _, test_data = T.RandomLinkSplit(num_val=0.0, num_test=0.1, is_undirected=True, key='edge_mask',
                  split_labels=True, add_negative_train_samples=True)(data)          
        train_data.pos_edge_mask = train_data.pos_edge_mask.add(-1)
        test_data.pos_edge_mask = test_data.pos_edge_mask.add(-1)

        train_graph_discn = to_networkx(train_data, to_undirected=True, 
                      node_attrs=['node_mask'], edge_attrs=['pos_edge_mask']) 

        clique_labels = torch.cat((train_data.pos_edge_mask, test_data.pos_edge_mask)).numpy()
        clique_edge_masks = np.array([[lbl==(cl+1) for lbl in clique_labels] for cl in range(num_cliques)]).astype(int)

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

        N = train_data.num_nodes
        train_data.x = torch.tensor(np.eye(N), dtype=torch.float32)
        train_data.x = train_data.x.to(device)
        train_data.edge_index = train_data.edge_index.to(device)
        
        for D_in in [int(2**x) for x in range(1, 10)]: 

            ####### Training

            EMB_FOLDER = '../../output/synth/%s-%s/'%(MODEL.lower(), 'featureless')
            os.makedirs(EMB_FOLDER, exist_ok=True)
            emb_file = EMB_FOLDER+'%s_%s_%ddims.%d.npz'%(DATASET, MODEL, D_in, SEED)

            if not os.path.isfile(emb_file):
                if MODEL == 'DEEPWALK':
                    Z = train_deepwalk_emb(train_graph_discn, D_in, seed=42*SEED)
                elif MODEL == 'INFWALK':
                    Z = train_infwalk_emb(train_graph_discn, D_in)
                elif MODEL == 'GAE':
                    Z = train_sgc_emb(train_data, (train_data.x.shape[1], D_in))
                elif MODEL == 'SAGE':
                    Z = train_sage_emb(train_data, (train_data.x.shape[1], D_in))

                np.savez_compressed(emb_file, Z)
                print(DATASET, MODEL, D_in, SEED, 'embeddings saved')

            ####### Dimensional Metrics

            Z = np.load(emb_file, allow_pickle=True)['arr_0'] 
                
            edge_idx = np.array(train_graph.edges)
            node_idx = np.unique(edge_idx)

            edge_Z = (Z[edge_idx][:,0] * Z[edge_idx][:,1])
            edge_lbl = np.array([d['pos_edge_mask'] for i,j,d in train_graph.edges(data=True)])
            lbl_masks = np.array([[lbl==cl for lbl in edge_lbl] for cl in np.unique(edge_lbl)]).astype(int)
            
            METRIC_FOLDER = EMB_FOLDER+'/linearshap_metrics/'
            mean_edge_Z = np.mean(edge_Z, axis=0).reshape(1,-1) #LinearShap

            os.makedirs(METRIC_FOLDER, exist_ok=True)

            scores_file = METRIC_FOLDER + '%s_%s_%ddims.%d.scores.npz'%(DATASET, MODEL, D_in, SEED)
            masks_file = METRIC_FOLDER + '%s_%s_%ddims.%d.masks.npz'%(DATASET, MODEL, D_in, SEED)
            if not os.path.isfile(scores_file):
                dims, emb_masks, emb_scores = emb_mask_parallel(edge_Z, lbl_masks, mean_edge_Z)
                emb_scores['edge_index'] = edge_idx
                np.savez_compressed(scores_file, **{k:emb_scores[k] for k in emb_scores.keys()})
                np.savez_compressed(masks_file, emb_masks)
            
                print(DATASET, MODEL, D_in, SEED, 'dim. metrics saved')

            ####### Positional Metrics

            emb_masks = np.load(masks_file, allow_pickle=True)['arr_0'] 
            dims, edges = np.where(emb_masks>THRESH)

            spath_file = METRIC_FOLDER + '%s_%s_%ddims.%d.spath.npz'%(DATASET, MODEL, D_in, SEED)
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

                print(DATASET, MODEL, D_in, SEED, 'pos. metrics saved')

            ####### Task Metrics / Logistic Regression 

            Z = np.load(emb_file, allow_pickle=True)['arr_0'] 
                    
            edge_idx_train = np.concatenate((train_data.pos_edge_mask_index.T.numpy(), train_data.neg_edge_mask_index.T.numpy()))
            Z_train = Z[edge_idx_train[:,0]]*Z[edge_idx_train[:,1]]
            y_train = np.concatenate((np.ones(train_data.pos_edge_mask_index.shape[1]), np.zeros(train_data.neg_edge_mask_index.shape[1])))

            edge_idx_test = np.concatenate((test_data.pos_edge_mask_index.T.numpy(), test_data.neg_edge_mask_index.T.numpy()))
            Z_test = Z[edge_idx_test[:,0]]*Z[edge_idx_test[:,1]]
            y_test = np.concatenate((np.ones(test_data.pos_edge_mask_index.shape[1]), np.zeros(test_data.neg_edge_mask_index.shape[1])))

            METRIC_FOLDER = EMB_FOLDER+'/shap_metrics/'
            os.makedirs(METRIC_FOLDER, exist_ok=True)

            task_file = METRIC_FOLDER + '%s_%s_%ddims.%d.linear_task.npz'%(DATASET, MODEL, D_in, SEED)

            if not os.path.isfile(task_file):

                from sklearn import linear_model, model_selection
                clf = linear_model.LogisticRegression(max_iter=1000, random_state=42*SEED)
                idx_shuf = np.random.permutation(np.arange(Z_train.shape[0]))
                clf.fit(Z_train[idx_shuf], y_train[idx_shuf])

                i_task = {'Z_train':Z_train.copy(), 'Z_test':Z_test.copy(), 'y_train':y_train.copy(), 'y_test':y_test.copy(),
                           'edge_idx_train': edge_idx_train.copy(), 'edge_idx_test':edge_idx_test.copy(),
                           'y_pred_train': clf.predict_proba(Z_train)[:,1], 'y_pred_test': clf.predict_proba(Z_test)[:,1],
                           'gt_masks':clique_edge_masks, 'gt_labels': clique_labels}

                import shap

                Z_true = np.concatenate((Z_train[y_train==1], Z_test[y_test==1]))
                idx_train = np.arange(Z_train[y_train==1].shape[0])
                idx_test = np.arange(Z_test[y_test==1].shape[0]) + idx_train.shape[0]

                explainer = shap.LinearExplainer(clf, Z_train[y_train==1])
                shap_values = explainer(Z_true)
                emb_masks = shap_values.values.T

                np.savez_compressed(task_file, **{k:i_task[k] for k in i_task.keys()})

                scores_file = METRIC_FOLDER + '%s_%s_%ddims.%d.linear_scores.npz'%(DATASET, MODEL, D_in, SEED)
                masks_file = METRIC_FOLDER + '%s_%s_%ddims.%d.linear_masks.npz'%(DATASET, MODEL, D_in, SEED)
                i_masks, i_scores = shap_metrics(emb_masks, clique_edge_masks)
                np.savez_compressed(scores_file, **{k:i_scores[k] for k in i_scores.keys()})
                np.savez_compressed(masks_file, i_masks)

                print(DATASET, MODEL, D_in, SEED, 'task metrics saved')

if __name__ == '__main__':
    main()