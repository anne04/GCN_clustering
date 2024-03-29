import sys
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import cell2location

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text for PDFs
results_folder = 'results/lymph_nodes_analysis/'

# create paths and names to results folders for reference regression and cell2location models
ref_run_name = f'{results_folder}/reference_signatures'
run_name = f'{results_folder}/cell2location_map'
#adata_vis = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")

adata_h5 = sc.read_visium(path='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_cell2location/V1_Human_Lymph_Node/') #st.Read10X(path='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_cell2location/', count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
adata_vis = adata_h5


adata_vis.obs['sample'] = list(adata_vis.uns['spatial'].keys())[0]
adata_vis.var['SYMBOL'] = adata_vis.var_names
adata_vis.var.set_index('gene_ids', drop=True, inplace=True)
adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]

# remove MT genes for spatial mapping (keeping their counts in the object)
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]

'''
adata_ref = sc.read(
    f'./data/sc.h5ad',
    backup_url='https://cell2location.cog.sanger.ac.uk/paper/integrated_lymphoid_organ_scrna/RegressionNBV4Torch_57covariates_73260cells_10237genes/sc.h5ad'
)
'''
adata_ref = sc.read('/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_cell2location/sc.h5ad')
adata_ref.var['SYMBOL'] = adata_ref.var.index
# rename 'GeneID-2' as necessary for your data
adata_ref.var.set_index('GeneID-2', drop=True, inplace=True)

# delete unnecessary raw slot (to be removed in a future version of the tutorial)
del adata_ref.raw
from cell2location.utils.filtering import filter_genes
selected = filter_genes(adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)

# filter the object
adata_ref = adata_ref[:, selected].copy()
cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                        # 10X reaction / sample / batch
                        batch_key='Sample',
                        # cell type, covariate used for constructing signatures
                        labels_key='Subset',
                        # multiplicative technical effects (platform, 3' vs 5', donor effect)
                        categorical_covariate_keys=['Method']
                       )
from cell2location.models import RegressionModel
mod = RegressionModel(adata_ref)

# view anndata_setup as a sanity check
mod.view_anndata_setup()
mod.train(max_epochs=250, use_gpu=True)

#mod.plot_history(20)

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
)

# Save model
mod.save(f"{ref_run_name}", overwrite=True)

# Save anndata object with results
adata_file = f"{ref_run_name}/sc.h5ad"
adata_ref.write(adata_file)
adata_file
'''
adata_ref = mod.export_posterior(
    adata_ref, use_quantiles=True,
    # choose quantiles
    add_to_obsm=["q05","q50", "q95", "q0001"],
    sample_kwargs={'batch_size': 2500, 'use_gpu': True}
)

mod.plot_QC()
'''
'''
adata_file = f"{ref_run_name}/sc.h5ad"
adata_ref = sc.read_h5ad(adata_file)
mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", adata_ref)
'''


if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]


#__________________________________________________________________________________________
# find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")
# create and train the model
mod = cell2location.models.Cell2location(
    adata_vis, cell_state_df=inf_aver,
    # the expected average cell abundance: tissue-dependent
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=30,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection:
    detection_alpha=20
)
mod.view_anndata_setup()
mod.train(max_epochs=30000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
          use_gpu=True,
         )

#mod.plot_history(1000)
#plt.legend(labels=['full data training']);

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_vis = mod.export_posterior(
    adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
)

# Save model
mod.save(f"{run_name}", overwrite=True)

# mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

# Save anndata object with results
adata_file = f"{run_name}/sp.h5ad"
adata_vis.write(adata_file)
adata_file
'''
adata_file = f"{run_name}/sp.h5ad"
adata_vis = sc.read_h5ad(adata_file)
mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)
'''
from cell2location.utils import select_slide
slide = select_slide(adata_vis, 'V1_Human_Lymph_Node')
adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

# select up to 6 clusters
clust_labels = ['T_CD4+_naive', 'B_naive', 'FDC']
clust_col = ['' + str(i) for i in clust_labels] # in case column names differ from labels

slide = select_slide(adata_vis, 'V1_Human_Lymph_Node')
from collections import defaultdict
spot_vs_type = defaultdict(list)
spot_vs_type['barcode']=[]
spot_vs_type['T_CD8+_naive']=[]
spot_vs_type['B_naive']=[]
spot_vs_type['FDC']=[]
for i in range (0, len(slide)):
    spot_vs_type['barcode'].append(slide.obs['T_CD8+_naive'].index[i])
 ,  spot_vs_type['T_CD8+_naive'].append(slide.obs['T_CD8+_naive'][i])
    spot_vs_type['B_naive'].append(slide.obs['B_naive'][i])
    spot_vs_type['FDC'].append(slide.obs['FDC'][i])

import pandas as pd
spot_vs_type_dataframe = pd.DataFrame(spot_vs_type)
spot_vs_type_dataframe.to_csv('/cluster/home/t116508uhn/64630/spot_vs_type_dataframe_V1_HumanLympNode.csv', index=False)

'''
>>> spot_vs_type_dataframe
                 barcode  T_CD8+_naive    B_naive       FDC
0     AAACAAGTATCTCCCA-1      1.102267   5.888736  0.178185
1     AAACAATCTACTAGCA-1      4.777420   0.103569  0.384692
2     AAACACCAATAACTGC-1      0.717926   3.651664  0.162864
3     AAACAGAGCGACTCCT-1      0.384908   0.002634  0.895615
4     AAACAGCTTTCAGAAG-1      0.191829  19.128501  1.077212
...                  ...           ...        ...       ...
4030  TTGTTTCACATCCAGG-1      0.589134   5.925571  0.318266
4031  TTGTTTCATTAGTCTA-1      0.299134   3.412221  0.780267
4032  TTGTTTCCATACAACT-1      0.954557   6.124115  0.371099
4033  TTGTTTGTATTACACG-1      2.486489   0.813700  0.095774
4034  TTGTTTGTGTAAATTC-1      5.706200   2.034936  0.515592

[4035 rows x 4 columns]
'''

fig = plot_spatial(
    adata=slide,
    # labels to show on a plot
    color=clust_col, labels=clust_labels,
    show_img=True,
    # 'fast' (white background) or 'dark_background'
    style='fast',
    # limit color scale at 99.2% quantile of cell abundance
    max_color_quantile=0.992,
    # size of locations (adjust depending on figure size)
    circle_diameter=6,
    colorbar_position='right'
)


