from typing import Callable, Optional, List
from Agglomerate3D import Agglomerate3D
from itertools import product
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm

TREE_SCORE_OPTIONS = ['ME, BME, MP']


class BatchAgglomerate3D:
    def __init__(self,
                 cell_type_affinity: List[Callable],
                 linkage_cell: List[str],
                 linkage_region: List[str],
                 tree_rank: List[str],
                 max_region_diff: Optional[List[int]] = None,
                 verbose: Optional[bool] = False,
                 integrity_check: Optional[bool] = True):
        if max_region_diff is None:
            max_region_diff = [0]
        self.cell_type_affinity = cell_type_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.tree_rank = tree_rank
        if tree_rank not in TREE_SCORE_OPTIONS:
            raise UserWarning(f'Tree scoring options must be one of: {TREE_SCORE_OPTIONS}')
        self.max_region_diff = max_region_diff
        self.verbose = verbose
        self.integrity_check = integrity_check
        self.pool = mp.Pool(mp.cpu_count())
        self.agglomerators: List[Agglomerate3D] = []
        self.pbar = \
            tqdm(total=np.product(list(map(len, [cell_type_affinity, linkage_cell, linkage_region, max_region_diff]))))

    @staticmethod
    def _agglomerate_func(cta, lc, lr, mrd, ic, data):
        agglomerate = Agglomerate3D(cta, lc, lr, mrd, verbose=False, pbar=False, integrity_check=ic)
        agglomerate.agglomerate(data)
        return agglomerate

    def _collect_agglomerators(self, result):
        self.agglomerators.append(result)
        self.pbar.update(1)

    def agglomerate(self, data: pd.DataFrame):
        for cta, lc, lr, mrd, ic \
                in product(self.cell_type_affinity, self.linkage_cell, self.linkage_region, self.max_region_diff):
            self.pool.apply_async(self._agglomerate_func,
                                  args=(cta, lc, lr, mrd, ic, data),
                                  callback=self._collect_agglomerators)
        self.pool.close()
        self.pool.join()
        self.pbar.close()

    def get_best_agglomerator(self):
        def score_func(a: Agglomerate3D):
            if self.tree_rank == 'MP':
                return a.compute_mp_score()
            elif self.tree_rank == 'ME':
                return a.compute_me_score()
            else:
                return a.compute_bme_score()

        scores = list(map(score_func, self.agglomerators))
        return self.agglomerators[int(np.argmin(scores))]