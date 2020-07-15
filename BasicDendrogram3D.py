from Agglomerate3D import Agglomerate3D
from data_utils import read_data
from scipy.stats import spearmanr
import pandas as pd
import time

if __name__ == '__main__':
    data = read_data(['mouse'], ['mouse'])

    def spearmanr_connectivity(x, y):
        # data is assumed to be (n_variables, n_examples)
        rho, _ = spearmanr(x, y, axis=1)
        return 1 - rho

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_avg',
        max_region_diff=1,
        verbose=False,
        integrity_check=True
    )

    start = time.process_time()
    agglomerate.agglomerate(data)
    end = time.perf_counter()
    pd.options.display.width = 0
    print(agglomerate.linkage_mat_readable)
    print(f'Total time elapsed: {(end - start) / 10}s')
