from agglomerate.batch_agglomerate_3d import BatchAgglomerate3D
from OrthoAxisDendro3D import CTDataLoader
from metrics.metric_utils import spearmanr_connectivity
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = CTDataLoader('mouse')
    agglomerate = BatchAgglomerate3D(
        linkage_cell=['complete'],
        linkage_region=['homolog_avg'],
        cell_type_affinity=[spearmanr_connectivity],
        max_region_diff=[1],
        region_dist_scale=np.arange(0.1, 3, 0.005),
        verbose=False
    )

    agglomerate.agglomerate(data)
    best_agglomerators = agglomerate.get_best_agglomerators()
    pd.options.display.width = 0
    bme_as = best_agglomerators['BME'][1]
    bme_as_scored = [bme_a.compute_tree_score('BME') for bme_a in bme_as]
    best_bme_a = bme_as[np.argmin(bme_as_scored)]
    print(best_bme_a.compute_tree_score('BME'))
    print(best_bme_a.region_dist_scale)
    print(best_bme_a.linkage_mat_readable)
    best_bme_a.view_tree3d()
