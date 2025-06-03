### Dotplot for neurotransmitter expression in parasympathetic neurons
## Datasets available at Gene Expression Omnibus (GEO) database: NA (GSE202760, GSE198709, GSE211538), DMV (GSE172411)

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
# Set the seed for pan das
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

CM = generate_colormap("#0000FF", "#ffffff")
CMr = generate_colormap("#ff0000", "#808080")
CMg = generate_colormap("#00a651", "#808080")

## Save parasympathetic neurons from all cells
# Load NA_Cam (GSE202760) dataset
cam_adata = sc.read_csv('./GSE202760_dge.tsv', delimiter='\t', first_column_names=True).T
cam_adata.obs['dataset'] = 'NA_Cam'
cam_adata.var_names_make_unique()

# Load NA_Kra (GSE198709) dataset
kra_adata = sc.read_csv('./NA_Kra_counts.csv', first_column_names=True).T
kra_adata.obs['dataset'] = 'NA_Kra'
kra_adata.var_names_make_unique()

# Load NA_Sun (GSE211538) dataset
sun_adata = sc.read_10x_h5('./NA_Sun_filtered_feature_bc_matrix.h5')
sun_adata.obs['dataset'] = 'NA_Sun'
sun_adata.var_names_make_unique()
sun_adata.raw = sun_adata
mito_genes = sun_adata.var_names.str.startswith('mt-')
sun_adata.obs['percent.mito'] = (sun_adata[:, mito_genes].X.sum(axis=1) / sun_adata.X.sum(axis=1))
sun_adata.obs['n_genes'] = (sun_adata.X > 0).sum(axis=1)
sun_adata.obs['total_counts'] = sun_adata.X.sum(axis=1)
sun_adata = sun_adata[sun_adata.obs['percent.mito'] < 0.1, :]
sc.pp.filter_cells(sun_adata, min_counts=1500)
sc.pp.filter_cells(sun_adata, max_counts=45000)
sc.pp.normalize_total(sun_adata, target_sum=1e4)
sc.pp.log1p(sun_adata)
sc.pp.highly_variable_genes(sun_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sun_adata = sun_adata[:, sun_adata.var.highly_variable]
sc.pp.scale(sun_adata, max_value=10)
sc.pp.pca(sun_adata, n_comps=50)
sc.pp.neighbors(sun_adata, n_neighbors=20, use_rep='X_pca')
sc.tl.umap(sun_adata)
sc.tl.leiden(sun_adata, resolution=0.5)
sc.pl.umap(sun_adata, color=['leiden','dataset', 'Snap25', 'Th'], cmap=CMr, legend_loc="on data")
specific_cluster = sun_adata.raw[sun_adata.obs['leiden'] == '9']
df_cluster = pd.DataFrame(specific_cluster.X.toarray(), index=sun_adata[sun_adata.obs['leiden'] == '9'].obs_names, columns=specific_cluster.var_names)
print(df_cluster.shape)
df_cluster.to_csv('NA_Sun9.csv')
sun_adata = sc.read_csv('./NA_Sun9_205neurons.csv')
sun_adata.obs['dataset'] = 'NA_Sun'
sun_adata.var_names_make_unique()

# Load DMV (GSE172411) dataset
DMV_adata = sc.read_csv('./DMV_GSE172411_dge.tsv', delimiter='\t', first_column_names=True).T
DMV_adata.obs['dataset'] = 'DMV'
DMV_adata.var_names_make_unique()

adata = sun_adata
for d in [cam_adata, kra_adata, DMV_adata]:
    adata = adata.concatenate(d, join='inner', fill_value=0.)

mito_genes = adata.var_names.str.startswith('mt-')

adata.obs['percent.mito'] = (adata[:, mito_genes].X.sum(axis=1) / adata.X.sum(axis=1))
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)

adata = adata[adata.obs['percent.mito'] < 0.05, :]
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
mapping = {
    'NA_Sun': 'NA',
    'NA_Cam': 'NA',
    'NA_Kra': 'NA',
    'DMV': 'DMV',
}
adata.obs['datasetc'] = adata.obs['dataset'].map(mapping).astype('category')
custom_colors = ["#FF7F0E", "#00AEEF"]
adata.uns['datasetc_colors'] = custom_colors

markers1 = ['Chat','Nos1','Adcyap1','Calca','Vip']
sc.pl.dotplot(adata, markers1, groupby='datasetc', cmap=CM, dendrogram=False, vmax=1)
