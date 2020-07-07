import pandas as pd
import numpy as np


def read_data(preprocess: bool = True):
    deg_chicken = pd.read_csv('TranscriptomeData/Nomi_chickenDEGs.csv', header=0, index_col=0)
    data_chicken = pd.read_csv('TranscriptomeData/Nomi_chickenallaverage.csv', header=0, index_col=0)

    deg_mouse = pd.read_csv('TranscriptomeData/Nomi_mouseDEGs.csv', header=0, index_col=0)
    data_mouse = pd.read_csv('TranscriptomeData/Nomi_mouseallaverage.csv', header=0, index_col=0)

    deg_human = pd.read_csv('TranscriptomeData/Nomi_humanDEGs.csv', header=0, index_col=0)
    data_human = pd.read_csv('TranscriptomeData/Nomi_humanallaverage.csv', header=0, index_col=0)

    if preprocess:
        # Find the common DEGs
        common_genes = np.intersect1d(deg_human['gene'], np.intersect1d(deg_mouse['gene'], deg_chicken['gene']))

        # Filter for common DEGs
        data_chicken = data_chicken.loc[data_chicken.index.isin(common_genes)]
        data_mouse = data_mouse.loc[data_mouse.index.isin(common_genes)]
        data_human = data_human.loc[data_human.index.isin(common_genes)]

        # Divide each row by mean, as in Tosches et al, rename columns,
        # and transpose so that column labels are genes and rows are cell types
        data_chicken = data_chicken.div(data_chicken.mean(axis=1).values, axis=0).add_prefix('C_').transpose()
        data_mouse = data_mouse.div(data_mouse.mean(axis=1).values, axis=0).add_prefix('M_').transpose()
        data_human = data_human.div(data_human.mean(axis=1).values, axis=0).add_prefix('H_').transpose()

    # Concatenate all the data
    # data_all = pd.concat([data_mouse, data_chicken, data_human])
    # data_all = pd.concat([data_mouse, data_chicken])
    data_all = pd.concat([data_mouse])
    # data_all = pd.concat([data_chicken])
    # data_all = pd.concat(([data_human]))

    return data_all


def get_region(ct_name: str):
    return ct_name[0] + '_' + ct_name.split('.')[1]
