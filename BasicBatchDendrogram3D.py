from BatchAgglomerate3D import BatchAgglomerate3D
from data_utils import read_data
from metric_utils import *
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = read_data(['mouse'])
    tree_rank = 'ME'
    agglomerate = BatchAgglomerate3D(
        cell_type_affinity=[spearmanr_connectivity],
        linkage_cell=['complete', 'average'],
        linkage_region=['homolog_avg'],
        tree_rank=tree_rank,
        max_region_diff=[0, 1],
        region_dist_scale=np.arange(0.7, 1.2, 0.01),
        verbose=False
    )

    agglomerate.agglomerate(data)
    best_agglomerator, score = agglomerate.get_best_agglomerator()
    pd.options.display.width = 0
    print(best_agglomerator.linkage_mat_readable)
    print(best_agglomerator.region_dist_scale)
    print(f'{tree_rank} Score: {score}')
