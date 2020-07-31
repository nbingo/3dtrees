from BatchAgglomerate3D import BatchAgglomerate3D, TREE_SCORE_OPTIONS
from data_utils import read_data
from metric_utils import *
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = read_data(['mouse'])
    agglomerate = BatchAgglomerate3D(
        cell_type_affinity=[spearmanr_connectivity],
        linkage_cell=['complete'],
        linkage_region=['homolog_avg'],
        max_region_diff=[1],
        region_dist_scale=np.arange(0.5, 1.5, 0.005),
        verbose=False
    )

    agglomerate.agglomerate(data)
    best_agglomerators = agglomerate.get_best_agglomerators()
    pd.options.display.width = 0
    # for metric in TREE_SCORE_OPTIONS:
    for metric in ['BME']:
        print(f'Best {metric} score: {best_agglomerators[metric][0]}\n'
              f'Agglomerator: {best_agglomerators[metric][1]}\n'
              f'region_dist_scale: {best_agglomerators[metric][1].region_dist_scale}\n'
              f'Tree:\n{best_agglomerators[metric][1].linkage_mat_readable}\n\n')
        best_agglomerators[metric][1].view_tree3d()
