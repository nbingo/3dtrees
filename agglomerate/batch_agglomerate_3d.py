from typing import Callable, Optional, List, Dict, Iterable, Tuple
from agglomerate.agglomerate_3d import Agglomerate3D, TREE_SCORE_OPTIONS
from itertools import product
from data.data_loader import DataLoader
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm


class BatchAgglomerate3D:
    def __init__(self,
                 linkage_cell: List[str],
                 linkage_region: List[str],
                 cell_type_affinity: List[Callable],
                 region_affinity: Optional[List[Callable]] = None,
                 max_region_diff: Optional[List[int]] = None,
                 region_dist_scale: Optional[Iterable[float]] = None,
                 verbose: Optional[bool] = False,
                 integrity_check: Optional[bool] = True):
        # Can't have mutable types as default :(
        if region_affinity is None:
            region_affinity = [None]
        if region_dist_scale is None:
            region_dist_scale = [1]
        if max_region_diff is None:
            max_region_diff = [0]
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.cell_type_affinity = cell_type_affinity
        self.region_affinity = region_affinity
        self.max_region_diff = max_region_diff
        self.region_dist_scale = region_dist_scale
        self.verbose = verbose
        self.integrity_check = integrity_check
        self.agglomerators: List[Agglomerate3D] = []
        self.augmented_tree_scores: List[Dict[str, float]] = []
        self.tree_scores: Dict[str, List[float]] = {metric: [] for metric in TREE_SCORE_OPTIONS}
        self.pbar = \
            tqdm(total=np.product(list(map(len, [
                linkage_cell, linkage_region, cell_type_affinity, region_affinity, max_region_diff, region_dist_scale
            ]))))

    @staticmethod
    def _agglomerate_func(lc, lr, cta, ra, mrd, rds, ic, data):
        agglomerate = Agglomerate3D(linkage_cell=lc,
                                    linkage_region=lr,
                                    cell_type_affinity=cta,
                                    region_affinity=ra,
                                    max_region_diff=mrd,
                                    region_dist_scale=rds,
                                    verbose=False,
                                    pbar=False,
                                    integrity_check=ic
                                    )
        agglomerate.agglomerate(data)
        return agglomerate

    def _collect_agglomerators(self, result):
        self.agglomerators.append(result)
        self.pbar.update(1)

    def agglomerate(self, data_ct: DataLoader):
        pool = mp.Pool(mp.cpu_count())
        for lc, lr, cta, ra, mrd, rds in product(self.linkage_cell,
                                                 self.linkage_region,
                                                 self.cell_type_affinity,
                                                 self.region_affinity,
                                                 self.max_region_diff,
                                                 self.region_dist_scale):
            if self.verbose:
                print(f'Starting agglomeration with {lc, lr, cta, ra, mrd, rds, self.integrity_check}')
            pool.apply_async(self._agglomerate_func,
                             args=(lc, lr, cta, ra, mrd, rds, self.integrity_check, data_ct),
                             callback=self._collect_agglomerators)
        pool.close()
        pool.join()
        self.pbar.close()

    def _collect_augmented_scores(self, result):
        lc, lr, mrd, rds, scores = result
        for metric, score in zip(TREE_SCORE_OPTIONS, scores):
            self.augmented_tree_scores.append(
                {'linkage_cell': lc, 'linkage_region': lr, 'max_region_diff': mrd, 'region_dist_scale': rds,
                 'score metric': metric, 'score': score})
        self.pbar.update(1)

    @staticmethod
    def _augmented_score_func(a: Agglomerate3D) -> Tuple[str, str, int, float, List[float]]:
        return a.linkage_cell,   \
               a.linkage_region,  \
               a.max_region_diff,  \
               a.region_dist_scale, \
               [a.compute_tree_score(m) for m in TREE_SCORE_OPTIONS]

    def get_all_scores(self) -> pd.DataFrame:
        self._compute_tree_scores(func=self._augmented_score_func, callback=self._collect_augmented_scores)
        return pd.DataFrame(self.augmented_tree_scores)

    def _compute_tree_scores(self, func: Callable, callback: Callable):
        self.pbar = tqdm(total=len(self.agglomerators))
        pool = mp.Pool(mp.cpu_count())
        for a in self.agglomerators:
            pool.apply_async(func=func,
                             args=(a,),
                             callback=callback
                             )
        pool.close()
        pool.join()
        self.pbar.close()

    def _collect_basic_scores(self, scores: List[float]):
        for metric, score in zip(TREE_SCORE_OPTIONS, scores):
            self.tree_scores[metric].append(score)
        self.pbar.update(1)

    @staticmethod
    def _basic_score_func(a: Agglomerate3D) -> List[float]:
        return [a.compute_tree_score(m) for m in TREE_SCORE_OPTIONS]

    def get_best_agglomerators(self) -> Dict[str, Tuple[float, np.array]]:
        self._compute_tree_scores(func=self._basic_score_func, callback=self._collect_basic_scores)

        # best_agglomerators: Dict[str, Tuple[float, Agglomerate3D]] = {
        #     metric: (np.min(self.tree_scores[metric]), self.agglomerators[int(np.argmin(self.tree_scores[metric]))])
        #     for metric in TREE_SCORE_OPTIONS
        # }
        best_agglomerators: Dict[str, Tuple[float, Agglomerate3D]] = {
            metric: (
                np.min(self.tree_scores[metric]),
                np.unique(
                    np.array(self.agglomerators)[np.where(self.tree_scores[metric] == np.min(self.tree_scores[metric]))]
                )
            )
            for metric in TREE_SCORE_OPTIONS
        }

        return best_agglomerators
