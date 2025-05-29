### Dataset downloaded from: https://www.proteinatlas.org/humanproteome/tissue/data#hpa_tissues_rna.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

### Plot adrenergic receptor expression at various peripheral tissues
df = pd.read_csv('rna_tissue_hpa.tsv', sep='\t')
all_genes = ['Adra1a', 'Adra1b', 'Adra1d', 'Adra2a', 'Adra2b', 'Adra2c', 'Adrb1', 'Adrb2', 'Adrb3']
all_genes = [g.upper() for g in all_genes]
df = df[df['Gene name'].isin(all_genes)][['Gene name', 'Tissue', 'TPM']]
df.to_csv('intermediate.csv')
print(df.shape)
df = df.pivot(index='Tissue', columns='Gene name', values='TPM')

# Re-order tissue types
df = df.loc[['tonsil','salivary gland','thyroid gland','breast','heart muscle','lung','esophagus','stomach','liver','gallbladder','pancreas','spleen','adrenal gland','kidney','small intestine','colon','rectum','appendix','urinary bladder','ovary','fallopian tube','endometrium','cervix','placenta','prostate','seminal vesicle','epididymis','testis','skin','adipose tissue','bone marrow','lymph node','smooth muscle','duodenum','cerebral cortex','choroid plexus','skeletal muscle','tongue','parathyroid gland','thymus']] 

plt.figure(figsize=(6, 8))
plt.tight_layout()
sns.heatmap(df, cmap=cm.jet,vmin = 0, vmax = 20)
plt.show()


### Plot cholinergic receptor expression at various peripheral tissues
all_genes = ['Chrm1','Chrm2','Chrm3','Chrm4','Chrm5','Chrna4','Chrna5','Chrna7','Chrnb1']
# 'Chrm1','Chrm2','Chrm3','Chrna3','Chrna4','Chrna5','Chrna6','Chrna7','Chrnb1','Chrnb2','Chrnb3','Chrnb4'
all_genes = [g.upper() for g in all_genes]
df = df[df['Gene name'].isin(all_genes)][['Gene name', 'Tissue', 'TPM']]
df.to_csv('ChatReceptor.csv')
print(df.shape)
df = df.pivot(index='Tissue', columns='Gene name', values='TPM')

# Re-order tissue types
df = df.loc[['salivary gland','thyroid gland','heart muscle','lung','esophagus','stomach','liver','spleen','small intestine','urinary bladder','ovary','placenta','prostate','skin','bone marrow','lymph node','smooth muscle','duodenum','cerebral cortex','choroid plexus','skeletal muscle','tongue','parathyroid gland','thymus','tonsil','breast','gallbladder','pancreas','adrenal gland','kidney','colon','rectum','appendix','fallopian tube','endometrium','cervix','seminal vesicle','epididymis','testis','adipose tissue']] 

plt.figure(figsize=(6, 8))
plt.tight_layout()
sns.heatmap(df, cmap=cm.jet,vmin = 0, vmax = 20)
plt.show()


