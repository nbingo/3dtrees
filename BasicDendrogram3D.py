from agglomerate.agglomerate_3d import Agglomerate3D
from data.data_utils import read_data
from scipy.stats import spearmanr
import pandas as pd
import time

if __name__ == '__main__':
    data = read_data(['chicken', 'mouse'], ['chicken', 'mouse'], orthologs=False)

    def spearmanr_connectivity(x, y):
        # data is assumed to be (n_variables, n_examples)
        rho, _ = spearmanr(x, y, axis=1)
        return 1 - rho

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_avg',
        max_region_diff=1,
        region_dist_scale=.8,
        verbose=False,
        pbar=True,
        integrity_check=True
    )

    start = time.process_time()
    agglomerate.agglomerate(data)
    end = time.perf_counter()
    pd.options.display.width = 0
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    print(agglomerate.linkage_mat_readable)
    print(agglomerate.compute_tree_score('BME'))
    agglomerate.view_tree3d()
    print(f'Total time elapsed: {(end - start) / 10}s')
