import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import scanpy as sc
from anndata import AnnData
import random
import os

# Load SCG_Mapps data; GSE175421
adata1 = sc.read_csv('./GSM5333151_SCG_rep1.dge.txt', delimiter='\t')
adata1 = adata1.transpose()
adata1.obs['dataset'] = 'SCG_Mapps'
print("---SCG_Mapps---")
print(adata1) # 48357 × 17360
adata2 = sc.read_csv('./GSM5333152_SCG_rep2.dge.txt', delimiter='\t')
adata2 = adata2.transpose()
adata2.obs['dataset'] = 'SCG_Mapps'
print(adata2) # 45703 × 18030
adata3 = sc.read_csv('./GSM5333153_SCG_rep3.dge.txt', delimiter='\t')
adata3 = adata3.transpose()
adata3.obs['dataset'] = 'SCG_Mapps'
print(adata3) # 69123 × 18354
adata4 = sc.read_csv('./GSM5333154_SCG_rep4.dge.txt', delimiter='\t')
adata4 = adata4.transpose()
adata4.obs['dataset'] = 'SCG_Mapps'
print(adata4) # 44463 × 18259
adata5 = sc.read_csv('./GSM5333155_SCG_rep5.dge.txt', delimiter='\t')
adata5 = adata5.transpose()
adata5.obs['dataset'] = 'SCG_Mapps'
print(adata5) # 50942 × 17501

adata10 = adata1
for d in [adata2, adata3, adata4, adata5]:
    adata10 = adata10.concatenate(d, join='inner', fill_value=0.)
del [adata1, adata2, adata3, adata4, adata5]

# Load SCG_Ziegler; GSE231766
data_dir6 = './SCG_GSM7300193/'
data_dir7 = './SCG_GSM7300194/'
data_dir8 = './SCG_GSM7300195/'
data_dir9 = './SCG_GSM7300196/'
adata6 = sc.read_10x_mtx(data_dir6, var_names='gene_symbols', cache=True)
adata6.obs['dataset'] = 'SCG_Ziegler'
print("---SCG_Ziegler---")
print(adata6) # 24601 cells × 53647
adata7 = sc.read_10x_mtx(data_dir7, var_names='gene_symbols', cache=True)
adata7.obs['dataset'] = 'SCG_Ziegler'
print(adata7) # 7423 cells × 53647
adata8 = sc.read_10x_mtx(data_dir8, var_names='gene_symbols', cache=True)
adata8.obs['dataset'] = 'SCG_Ziegler'
print(adata8) # 7354 cells
adata9 = sc.read_10x_mtx(data_dir9, var_names='gene_symbols', cache=True)
adata9.obs['dataset'] = 'SCG_Ziegler'
print(adata9) # 6271 cells

adata11 = adata10
for d in [adata6, adata7, adata8, adata9]:
    adata11 = adata11.concatenate(d, join='inner', fill_value=0.)
del [adata10, adata6, adata7, adata8, adata9]

# Load SG_Sharma; GSE231924
adataSG = sc.read_csv('./SG_olu.all.counts_cleared-16383cells.csv')
adataSG = adataSG.transpose()
adataSG.obs['dataset'] = 'SG_Sharma'
print("---SG_Sharma---")
print(adataSG) # 23347 × 19224

# Load CGSMG_Wang; GSE278457
data_dir1 = './CGSMG_filtered_feature_bc_matrix1/'
data_dir2 = './CGSMG_filtered_feature_bc_matrix2/'
adataT = sc.read_10x_mtx(data_dir1, var_names='gene_symbols', cache=True)
adataT.obs['dataset'] = 'CGSMG_Wang'
adataU = sc.read_10x_mtx(data_dir2, var_names='gene_symbols', cache=True)
adataU.obs['dataset'] = 'CGSMG_Wang'
print("---CGSMG_Wang---")
print(adataT) # 20733 cells × 32183 genes
print(adataU) # 17903 cells × 32183 genes

adata = adata11
for d in [adataSG, adataT, adataU]:
    adata = adata.concatenate(d, join='inner', fill_value=0.)
print(adata) # 366220 cells × 13874 genes
del [adata11, adataSG, adataT, adataU]

mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)

adata = adata[adata.obs['percent.mito'] < 0.1, :]
sc.pp.filter_cells(adata, min_counts=1500)
sc.pp.filter_cells(adata, max_counts=45000)

print(adata) # 60826 cells × 13874 genes

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata.copy()

adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['percent.mito'])

sc.pp.scale(adata, max_value=10)

sc.pp.pca(adata, n_comps=50)

# Save all cells from sympathetic datasets for easier access
adata.write('1_sym_cells.h5ad', compression='gzip', compression_opts=4)

# Load the saved dataset
adata = sc.read_h5ad('1_sym_cells.h5ad')
print(adata)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata.copy()

adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['percent.mito'])

sc.pp.scale(adata, max_value=10)

sc.pp.pca(adata, n_comps=50)

import harmonypy as hm
ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'], kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony')

sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=['leiden'], legend_loc="on data")

clusters_to_r = ['0','1','2','3','5','6','7','8','9','10','11','12','13','14','15','17','20','21','22','23','24','25']
adata_sym_neu = adata[~adata.obs['leiden'].isin(clusters_to_r)].copy()
adata_sym_neu.write('2_sym_neu.h5ad', compression='gzip', compression_opts=4)

# Load the saved neuronal dataset
adata = sc.read_h5ad('2_sym_neu.h5ad')
print(adata) # 6765 sym neurons

ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['dataset'], kmeans_method='scipy_kmeans2')
adata.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca_harmony')

sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=['leiden'], legend_loc="on data")
