import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import scanpy as sc
from anndata import AnnData
import random
import os

def hex_to_rgb_normalized(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def generate_colormap(end_color, start_color="#ffffff"):
    colors = [hex_to_rgb_normalized(start_color), hex_to_rgb_normalized(end_color)]
    n_bins = [3]  # Discretizes the interpolation into bins
    cmap_name = 'custom_div_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

CM = generate_colormap("#0000FF", "#808080")
CMr = generate_colormap("#ff0000", "#808080")
CMg = generate_colormap("#00a651", "#808080")

# Load NA_Veerakumar GSE198709, PMID: 35650438
Veerakumar_adata = sc.read_csv('./NA_Kra_counts.csv', first_column_names=True).T
Veerakumar_adata.obs['dataset'] = 'NA_Veerakumar'
Veerakumar_adata.var_names_make_unique()
print("---NA_Veerakumar---")
print(Veerakumar_adata) # 203 cells × 24029 genes

# Load NA_Coverdell GSE202760, PMID: 35705034
Coverdell_adata = sc.read_csv('./NA_GSE202760_dge.tsv', delimiter='\t', first_column_names=True).T
Coverdell_adata.obs['dataset'] = 'NA_Coverdell'
Coverdell_adata.var_names_make_unique()
print("---NA_Coverdell---")
print(Coverdell_adata) # 283 cells × 28062 genes

# Load NA_Su GSE211538, PMID: 38987587
su_adata = sc.read_10x_h5('./NA_Su_filtered_feature_bc_matrix.h5')
su_adata.obs['dataset'] = 'NA_Su'
su_adata.var_names_make_unique()
print("---NA_Su---")
print(su_adata) # 10072 cells × 31053 genes

# Load DMV_Tao GSE172411, PMID: 34077742
DMV_adata = sc.read_csv('./DMV_GSE172411_dge.tsv', delimiter='\t', first_column_names=True).T
DMV_adata.obs['dataset'] = 'DMV_Tao'
DMV_adata.var_names_make_unique()
print("---DMV_Tao---")
print(DMV_adata) # 383 cells × 29779 genes 

adata = Veerakumar_adata
for d in [Coverdell_adata, su_adata, DMV_adata]:
    adata = adata.concatenate(d, join='inner', fill_value=0.)
print(adata) # 10941 cells × 20014 genes
del [Veerakumar_adata, Coverdell_adata, su_adata, DMV_adata]

mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)

adata = adata[adata.obs['percent.mito'] < 0.1, :]

print(adata) # 10941 cells × 20014 genes

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata.copy()

adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['percent.mito'])

sc.pp.scale(adata, max_value=10)

sc.pp.pca(adata, n_comps=50)

# Save all cells from parasympathetic datasets for easier access
adata.write('3_para_cells.h5ad', compression='gzip', compression_opts=4)

# Load the saved dataset
adata = sc.read_h5ad('3_para_cells.h5ad')
print(adata)

import harmonypy as hm
ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'], kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony') # can adjust n_pcs=40

sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

clusters_to_r = ['0','1','2','3','5','7','8','9','10','11','12','14','15','16','17','18','19','20']
adata_sym_neu = adata[~adata.obs['leiden'].isin(clusters_to_r)].copy()
adata_sym_neu.write('4_para_neu.h5ad', compression='gzip', compression_opts=4)

# Load the saved neuronal dataset
adata = sc.read_h5ad('4_para_neu.h5ad')
print(adata) # 1073 parasym neurons

ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'], kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony') # can adjust n_pcs=40

sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=['leiden'], legend_loc="on data")

