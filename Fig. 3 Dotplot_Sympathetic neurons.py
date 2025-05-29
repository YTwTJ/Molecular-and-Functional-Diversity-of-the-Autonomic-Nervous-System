### Dotplot for neurotransmitter expression in sympathetic neurons
## Datasets available at Gene Expression Omnibus (GEO) database: SCG (GSE175421), SG (GSE231924), CG-SMG (GSE278457)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scanpy as sc
from anndata import AnnData
import random
import os

# Set a fixed seed value for reproducibility
seed_value = 42
# Set the seed for numpy
np.random.seed(seed_value)
# Set the seed for the Python standard library random module
random.seed(seed_value)
# Set the seed for pandas
pd.util.testing.N = seed_value
os.environ['PYTHONHASHSEED'] = str(seed_value)

def hex_to_rgb_normalized(hex_color):
    """Convert a hex color string to a normalized RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def generate_colormap(end_color, start_color="#ffffff"):
    """
    Generates a sequential colormap that transitions from white to the given end color.
    """
    colors = [hex_to_rgb_normalized(start_color), hex_to_rgb_normalized(end_color)]
    n_bins = [3]  # Discretizes the interpolation into bins
    cmap_name = 'custom_div_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

CM = generate_colormap("#0000FF", "#808080")
CMr = generate_colormap("#ff0000", "#808080")
CMg = generate_colormap("#00a651", "#808080")

## Save neurons from all cells
# Load SCG (GSE175421) dataset
adata1 = sc.read_csv('./GSM5333151_SCG_rep1.dge.txt', delimiter='\t')
adata1 = adata1.transpose()
adata1.obs['dataset'] = 'SCG1'
adata2 = sc.read_csv('./GSM5333152_SCG_rep2.dge.txt', delimiter='\t')
adata2 = adata2.transpose()
adata2.obs['dataset'] = 'SCG2'
adata3 = sc.read_csv('./GSM5333153_SCG_rep3.dge.txt', delimiter='\t')
adata3 = adata3.transpose()
adata3.obs['dataset'] = 'SCG3'
adata4 = sc.read_csv('./GSM5333154_SCG_rep4.dge.txt', delimiter='\t')
adata4 = adata4.transpose()
adata4.obs['dataset'] = 'SCG4'
adata5 = sc.read_csv('./GSM5333155_SCG_rep5.dge.txt', delimiter='\t')
adata5 = adata5.transpose()
adata5.obs['dataset'] = 'SCG5'
adata = adata1
for d in [adata2, adata3, adata4, adata5]:
    adata = adata.concatenate(d, join='inner', fill_value=0.)
mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)
adata = adata[adata.obs['percent.mito'] < 0.1, :]
adata.raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50)
if True:
    import harmonypy as hm
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'])
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony')
else:
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=['leiden'], legend_loc="on data")
clusters_to_remove = ['0','1','2','4','5','6','8','9','10','11']
# 0-11 SAVE 3,7
adata_filtered = adata[~adata.obs['leiden'].isin(clusters_to_remove)].copy()
# Save the data matrix to a CSV file
expression_data_df = pd.DataFrame(adata_filtered.raw.X.toarray() if hasattr(adata_filtered.raw.X, "toarray") else adata_filtered.raw.X, index=adata_filtered.raw.obs_names, columns=adata_filtered.raw.var_names)
expression_data_df.to_csv('SCG_neu_clu3_7.csv')

# Load SG (GSE231924) dataset
adata = sc.read_csv('./SG_olu.all.counts_cleared-16383cells.csv')
adata = adata.transpose()
adata.obs['dataset'] = 'SG'
mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)
adata = adata[adata.obs['percent.mito'] < 0.1, :]
adata.raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50)
if False:
    import harmonypy as hm
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'])
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony')
else:
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=['leiden'], legend_loc="on data")
clusters_to_remove = ['0','1','2','3','4','5','6','7','8','10','11','12','13','15','16','17']
# 0-17 SAVE 9,14
adata_filtered = adata[~adata.obs['leiden'].isin(clusters_to_remove)].copy()
# Save the data matrix to a CSV file
expression_data_df = pd.DataFrame(adata_filtered.raw.X, index=adata_filtered.raw.obs_names, columns=adata_filtered.raw.var_names)
expression_data_df.to_csv('SG_neu_clu9_14.csv')

# Load CG-SMG (GSE278457) dataset


## Load sympathetic neurons
adataSCG = sc.read_csv('./SCG_neu_clu3_7_4283cells.csv')
adataSCG.obs['dataset'] = 'SCG'
adataSG = sc.read_csv('./SG_neu_clu9_14_1107cells.csv')
adataSG.obs['dataset'] = 'SG'
adataCGSMG = sc.read_csv('./CGSMG_neu_clu7_10_1740cells.csv')
adataCGSMG.obs['dataset'] = 'CGSMG'

adata = adataSCG
for d in [adataSG, adataCGSMG]:
    adata = adata.concatenate(d, join='inner', fill_value=0.)

mito_genes = adata.var_names.str.startswith('mt-')

adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)


adata = adata[adata.obs['percent.mito'] < 0.1, :]
sc.pp.filter_cells(adata, min_genes=2000)
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

adata.raw = adata.copy()

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata, max_value=100)

sc.pp.pca(adata, n_comps=50)

if True:
    import harmonypy as hm
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'], kmeans_method='scipy_kmeans2', random_state=seed_value)
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony')
else:
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca')
sc.tl.umap(adata)

sc.tl.leiden(adata, resolution=1)
markers1 = ['Th','Npy','Gal','Chgb','Crispld1','Nrg1']
sc.pl.dotplot(adata, markers1, groupby='dataset', dendrogram=False)
