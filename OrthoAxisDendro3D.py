from agglomerate.agglomerate_3d import Agglomerate3D
from data.data_loader import DataLoader
from metrics.metric_utils import spearmanr_connectivity
from typing import Optional, List
import numpy as np
import pandas as pd
import time
import scanpy as sc

sc.settings.verbosity = 3  # Please tell me everything all the time

P_VAL_ADJ_THRESH = 0.01
AVG_LOG_FC_THRESH = 2


class CTDataLoader(DataLoader):

    def __init__(self, species: str):
        super().__init__()
        species_data = sc.read(f'withcolors/{species}_ex_colors.h5ad')

        # Label each observation with its region and species
        species_data.obs['clusters'] = species_data.obs['clusters'].apply(lambda s: species[0].upper() + '_' + s)
        species_data.obs['subregion'] = species_data.obs['clusters'].apply(lambda s: s.split('.')[0])

        # Split different regions into separate AnnData-s
        species_data_region_split = [species_data[species_data.obs['subregion'] == sr] for sr in
                                     np.unique(species_data.obs['subregion'])]

        # Compute DEGs between different subregions to get region axis mask
        sc.tl.rank_genes_groups(species_data, groupby='subregion', method='wilcoxon')
        # Filter by adjusted p value and log fold change
        r_axis_name_mask = ((pd.DataFrame(species_data.uns['rank_genes_groups']['pvals_adj']) < P_VAL_ADJ_THRESH) &
                            (pd.DataFrame(species_data.uns['rank_genes_groups']['logfoldchanges']) > AVG_LOG_FC_THRESH))
        # Our current mask is actually for names sorted by their z-scores, so have to get back to the original ordering
        r_axis_filtered_names = pd.DataFrame(species_data.uns['rank_genes_groups']['names'])[r_axis_name_mask]
        # Essentially take union between DEGs of different regions
        r_axis_filtered_names = r_axis_filtered_names.to_numpy().flatten()
        # Now go through genes in their original order and check if they are in our list of genes
        self.r_axis_mask = species_data.var.index.isin(r_axis_filtered_names)

        # Compute DEGs between cell types within subregions to get cell type axis mask
        ct_degs_by_subregion = []
        # Iterate over regions in this species
        for sr in species_data_region_split:
            # Need to have at least one cell type in the region
            if len(np.unique(sr.obs['clusters'])) > 1:
                # Compute DEGs for cell types in this region
                sc.tl.rank_genes_groups(sr, groupby='clusters', method='wilcoxon')
                # Filter by adjusted p value and log fold change
                deg_names_mask = ((pd.DataFrame(sr.uns['rank_genes_groups']['pvals_adj']) < P_VAL_ADJ_THRESH) & (pd.DataFrame(sr.uns['rank_genes_groups']['logfoldchanges']) > AVG_LOG_FC_THRESH))
                # Get the names
                ct_degs_by_subregion.append(pd.DataFrame(sr.uns['rank_genes_groups']['names'])[deg_names_mask])

        # Construct mask of genes in original ordering
        # Essentially take union between DEGs of different regions
        ct_axis_filtered_names = np.concatenate([degs.to_numpy().flatten() for degs in ct_degs_by_subregion])
        # Get rid of nans
        ct_axis_filtered_names = ct_axis_filtered_names[~pd.isnull(ct_axis_filtered_names)]
        # Now go through genes in their original order and check if they are in our list of genes
        self.ct_axis_mask = species_data.var.index.isin(ct_axis_filtered_names)

        # Average transcriptomes within each cell type and put into data frame with cell types as rows and genes as cols
        ct_names = np.unique(species_data.obs['clusters'])
        ct_avg_data = [species_data[species_data.obs['clusters'] == ct].X.mean(axis=0) for ct in ct_names]
        self.data = pd.concat([pd.DataFrame(data, columns=species_data.var.index, index=[cluster_name])
                               for data, cluster_name in zip(ct_avg_data, np.unique(species_data.obs['clusters']))])
        # Divide each row by mean, as in Tosches et al, rename columns,
        # and transpose so that column labels are genes and rows are cell types
        # Divide each row by mean
        self.data = self.data.div(self.data.mean(axis=0).values, axis=1)

    def get_names(self) -> List[str]:
        return self.data.index.values

    def get_corresponding_region_names(self) -> List[str]:
        def get_region(ct_name: str):
            return ct_name.split('.')[0]

        return np.vectorize(get_region)(self.get_names())

    def __getitem__(self, item):
        return self.data.to_numpy()[item]

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    ct_data_loader = CTDataLoader('mouse')

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_avg',
        max_region_diff=1,
        region_dist_scale=.5,
        verbose=False,
        pbar=True,
        integrity_check=True
    )

    start = time.process_time()
    agglomerate.agglomerate(ct_data_loader)
    end = time.perf_counter()
    pd.options.display.width = 0
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    print(agglomerate.linkage_mat_readable)
    print(agglomerate.compute_tree_score('BME'))
    agglomerate.view_tree3d()
    print(f'Total time elapsed: {(end - start) / 10}s')
