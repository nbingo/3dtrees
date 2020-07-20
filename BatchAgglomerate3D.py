from typing import Callable, Optional, List, Dict, Iterable, Tuple
from Agglomerate3D import Agglomerate3D
from itertools import product
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
# Need this to have correct namespace during parallelization. Yet another reason why python is dumb
from metric_utils import *

TREE_SCORE_OPTIONS = ['ME', 'BME', 'MP']


class BatchAgglomerate3D:
    def __init__(self,
                 cell_type_affinity: List[Callable],
                 linkage_cell: List[str],
                 linkage_region: List[str],
                 max_region_diff: Optional[List[int]] = None,
                 region_dist_scale: Optional[Iterable[float]] = None,
                 verbose: Optional[bool] = False,
                 integrity_check: Optional[bool] = True):
        # Can't have mutable types as default :(
        if region_dist_scale is None:
            region_dist_scale = [1]
        if max_region_diff is None:
            max_region_diff = [0]
        self.cell_type_affinity = cell_type_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.max_region_diff = max_region_diff
        self.region_dist_scale = region_dist_scale
        self.verbose = verbose
        self.integrity_check = integrity_check
        self.manager = mp.Manager()
        self.agglomerators: List[Agglomerate3D] = self.manager.list()
        self.agglomerators_lock = self.manager.Lock()
        self.tree_scores: Dict[str, List[float]] = self.manager.dict({metric: self.manager.list() for metric in TREE_SCORE_OPTIONS})
        self.tree_scores_lock = self.manager.Lock()
        self.pbar = \
            tqdm(total=np.product(list(map(len, [
                cell_type_affinity, linkage_cell, linkage_region, max_region_diff, region_dist_scale
            ]))))
        self.pbar_lock = self.manager.Lock()

    @staticmethod
    def _agglomerate_func(cta, lc, lr, mrd, rp, ic, data):
        agglomerate = Agglomerate3D(cta, lc, lr, mrd, rp, verbose=False, pbar=False, integrity_check=ic)
        agglomerate.agglomerate(data)
        return agglomerate

    def _collect_agglomerators(self, result):
        with self.agglomerators_lock:
            self.agglomerators.append(result)
        with self.pbar_lock:
            self.pbar.update(1)

    def agglomerate(self, data: pd.DataFrame):
        pool = mp.Pool(mp.cpu_count())
        for cta, lc, lr, mrd, rds in product(self.cell_type_affinity,
                                             self.linkage_cell,
                                             self.linkage_region,
                                             self.max_region_diff,
                                             self.region_dist_scale):
            if self.verbose:
                print(f'Starting agglomeration with {cta, lc, lr, mrd, rds, self.integrity_check}')
            pool.apply_async(self._agglomerate_func,
                             args=(cta, lc, lr, mrd, rds, self.integrity_check, data),
                             callback=self._collect_agglomerators)
        pool.close()
        pool.join()
        self.pbar.close()

    def _collect_scores(self, scores: List[float]):
        with self.tree_scores_lock:
            self.tree_scores['MP'].append(scores[0])
            self.tree_scores['ME'].append(scores[1])
            self.tree_scores['BME'].append(scores[2])
        with self.pbar_lock:
            self.pbar.update(1)

    @staticmethod
    def _score_func(a: Agglomerate3D) -> List[float]:
        return [a.compute_mp_score(), a.compute_me_score(), a.compute_bme_score()]

    def get_best_agglomerators(self) -> Dict[str, Tuple[float, Agglomerate3D]]:
        self.pbar = tqdm(total=len(self.agglomerators))

        pool = mp.Pool(mp.cpu_count())
        results = []
        for a in self.agglomerators:
            results.append(pool.apply_async(func=self._score_func,
                                            args=(a,),
                                            callback=self._collect_scores
                                            ))
        pool.close()
        pool.join()
        self.pbar.close()

        best_agglomerators: Dict[str, Tuple[float, Agglomerate3D]] = {
            metric: (np.min(self.tree_scores[metric]), self.agglomerators[int(np.argmin(self.tree_scores[metric]))])
            for metric in TREE_SCORE_OPTIONS
        }

        return best_agglomerators
