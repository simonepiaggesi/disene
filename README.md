# DiSeNE: Disentangled and self-explainable node representation learning
The code has been tested with Python 3.11.5.

## Repository structure 
### Python scripts
In `exps/real/` and `exps/synth/` there are python scripts to train embeddings on real-world ("cora", "wiki", "facebook", "ppi") and synthetic data ("ring_of_cliques", "stochastic_block_model", "ba_cliques", "er_cliques") respectively.

  ```bash
    python exp_disene.py --dataset \ (default 'cora' or 'ring_of_cliques')  
                           --model \ choose a model among 'isgc' (DiSe-GAE), 'imlp' (DiSe-FCAE)
                          --k-hops \ number of NN layers (default 1)
                     --window-size \ random walk window parameter (default 5)
                            --runs \ number of training experiments (default 5)
  ```

  ```bash
    python exp_baseline.py --dataset \ (default 'cora' or 'ring_of_cliques') 
                             --model \ choose a model among 'deepwalk', 'infwalk', 'gae', 'sage'
                              --runs \ number of training experiments (default 5)
  ```

  ```bash
    python exp_dine.py --dataset \ choose a dataset (default 'cora' or 'ring_of_cliques') 
                         --model \ choose a model among 'deepwalk', 'gae'
                          --runs \ number of training experiments (default 5)
  ```
The scripts save embeddings in the folders `output/real/` and `output/synth/`. In each of these folders there will be two subfolders:

-`linearshap_metrics/` contains numpy arrays to calculate Comprehensibility, Sparsity, Overlap Consistency, and Positional Coherence.

-`shap_metrics/` contains numpy arrays to calculate Plausibility (only for synthetic data).

### Jupyter notebooks
In `exps/` there are two notebooks to compute metrics and show results:

-`DiSeNE_cora.ipynb` contains an example code to show the results for 'cora'. 

-`DiSeNE_synth.ipynb` contains an example code to show the results for 'ring_of_cliques'. 
