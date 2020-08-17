from agglomerate.agglomerate_3d import Agglomerate3D
from data.data_loader import DataLoader
from metrics.metric_utils import spearmanr_connectivity
from typing import Optional, List
import numpy as np
import pandas as pd
import time


P_VAL_ADJ_THRESH = 0.01
AVG_LOG_FC_THRESH = 2


class CTDataLoader(DataLoader):

    def __init__(self, species: List[str], deg_species: Optional[List[str]] = None, orthologs: Optional[bool] = True,
                 preprocess: Optional[bool] = True):
        super().__init__()
        if deg_species is None:
            deg_species = species
        if orthologs:
            postfix = 'DEGs.csv'
        else:
            postfix = 'DEGs_allgenes.csv'

        species_degs = [pd.read_csv(f'TranscriptomeData/Nomi_{s}{postfix}', header=0, index_col=0) for s in deg_species]
        species_data = [pd.read_csv(f'TranscriptomeData/Nomi_{s}allaverage.csv', header=0, index_col=0) for s in
                        species]

        if preprocess:
            # Filter for adjusted p-value and logFC
            for i in range(len(species_degs)):
                species_degs[i] = species_degs[i].loc[(species_degs[i]['p_val_adj'] < P_VAL_ADJ_THRESH) &
                                                      (species_degs[i]['avg_logFC'] > AVG_LOG_FC_THRESH)]

            # Find the common DEGs
            common_genes = species_degs[0]['gene']
            for deg in species_degs:
                common_genes = np.intersect1d(common_genes, deg['gene'])

            # Filter for common DEGs
            # Divide each row by mean, as in Tosches et al, rename columns,
            # and transpose so that column labels are genes and rows are cell types
            for i in range(len(species_data)):
                # Filter for common DEGs
                species_data[i] = species_data[i].loc[species_data[i].index.isin(common_genes)]
                # Divide each row by mean
                species_data[i] = species_data[i].div(species_data[i].mean(axis=1).values, axis=0)
                # Rename columns with species prefix and transpose so cell types are rows and genes are columns
                species_data[i] = species_data[i].add_prefix(f'{species[i][0]}_'.upper()).transpose()

        # Concatenate all the data_ct
        self.data: pd.DataFrame = pd.concat(species_data)

    def get_names(self) -> List[str]:
        return self.data.index.values

    def get_corresponding_region_names(self) -> List[str]:
        def get_region(ct_name: str):
            return ct_name[0] + '_' + ct_name.split('.')[1]

        return np.vectorize(get_region)(self.get_names())

    def __getitem__(self, item):
        return self.data.to_numpy()[item]

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    ct_data_loader = CTDataLoader(['mouse'], ['mouse'], orthologs=False)

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_avg',
        max_region_diff=1,
        region_dist_scale=.765,
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
