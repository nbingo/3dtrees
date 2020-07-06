from Agglomerate3D import Agglomerate3D
from data_utils import read_data
from scipy.stats import spearmanr
import pandas as pd

if __name__ == '__main__':
    data = read_data()

    def spearmanr_connectivity(x, y):
        # data is assumed to be (n_variables, n_examples)
        rho, _ = spearmanr(x, y, axis=1)
        return 1 - rho

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_avg',
        verbose=True
    )

    agglomerate.agglomerate(data)
    pd.options.display.width = 0
    print(agglomerate.linkage_mat_readable)
