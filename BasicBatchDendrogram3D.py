from BatchAgglomerate3D import BatchAgglomerate3D
from data_utils import read_data
from metric_utils import *

if __name__ == '__main__':
    data = read_data(['mouse'], ['mouse'])

    agglomerate = BatchAgglomerate3D(
        cell_type_affinity=[spearmanr_connectivity],
        linkage_cell=['complete'],
        linkage_region=['homolog_avg'],
        tree_rank='BME',
        max_region_diff=[0, 1, 2],
        verbose=True
    )

    agglomerate.agglomerate(data)
    print(agglomerate.get_best_agglomerator().linkage_mat_readable)
