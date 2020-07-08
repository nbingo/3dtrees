import pandas as pd
import numpy as np
from typing import List, Optional

P_VAL_ADJ_THRESH = 0.01
AVG_LOG_FC_THRESH = 2


def read_data(species: List[str], deg_species: Optional[List[str]] = None, preprocess: Optional[bool] = True):
    if deg_species is None:
        deg_species = species

    species_degs = [pd.read_csv(f'TranscriptomeData/Nomi_{s}DEGs.csv', header=0, index_col=0) for s in deg_species]
    species_data = [pd.read_csv(f'TranscriptomeData/Nomi_{s}allaverage.csv', header=0, index_col=0) for s in species]

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
            species_data[i] = species_data[i].loc[species_data[i].index.isin(common_genes)]
            species_data[i] = species_data[i].div(species_data[i].mean(axis=1).values, axis=0).add_prefix(f'{species[i][0]}_'.upper()).transpose()

    # Concatenate all the data
    return pd.concat(species_data)


def get_region(ct_name: str):
    return ct_name[0] + '_' + ct_name.split('.')[1]
