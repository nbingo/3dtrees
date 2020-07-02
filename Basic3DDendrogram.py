from dataclasses import dataclass
from typing import List, Callable, Union, Dict
from queue import PriorityQueue
from data_utils import get_region
from itertools import combinations, product
import functools
import numpy as np
import pandas as pd


@dataclass
class CellType:
    id_num: int
    region: int
    _transcriptome: np.array = None

    @property
    def transcriptome(self) -> np.array:
        if len(self._transcriptome.shape) == 1:
            return self._transcriptome.reshape(1, -1)
        return self._transcriptome

    @property
    def num_original(self):
        return self.transcriptome.shape[0]


@dataclass
class Region:
    id_num: int
    cell_types: List[CellType] = None

    @property
    def transcriptomes(self) -> np.array:
        transcriptomes = np.zeros((len(self.cell_types), self.cell_types[0].transcriptome.shape[1]))
        for c in range(len(self.cell_types)):
            transcriptomes[c] = self.cell_types[c].transcriptome
        return transcriptomes


@functools.total_ordering
@dataclass
class Edge:
    dist: float
    endpt1: Union[CellType, Region]
    endpt2: Union[CellType, Region]

    def __eq__(self, other):
        return self.dist == other.dist

    def __lt__(self, other):
        return self.dist < other.dist

    def __gt__(self, other):
        return self.dist > other.dist


LINKAGE_CELL_OPTIONS   = ['single', 'complete', 'average']
LINKAGE_REGION_OPTIONS = ['single', 'complete', 'average']
class Agglomerate3D:
    def __init__(self, cell_type_affinity: Callable, region_affinity: Callable, linkage_cell: str, linkage_region: str):
        self.cell_type_affinity = cell_type_affinity
        self.region_affinity = region_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.linkage_mat = pd.DataFrame({'Is region': [], 'ID1': [], 'ID2': [], 'Distance': [], 'Num children': []})
        if linkage_cell not in LINKAGE_CELL_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for cell linkage. Must be one of {LINKAGE_CELL_OPTIONS}')
        if linkage_region not in LINKAGE_REGION_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for region linkage. Must be one of '
                              f'{LINKAGE_REGION_OPTIONS}')

    def _compute_ct_dist(self, ct1: CellType, ct2: CellType) -> np.float64:
        dists = np.zeros((ct1.num_original, ct2.num_original))
        # Compute distance matrix
        # essentially only useful if this is working on merged cell types
        # otherwise just produces a matrix containing one value
        for ct1_idx, ct2_idx in product(range(ct1.num_original), range(ct2.num_original)):
            dists[ct1_idx, ct2_idx] = self.cell_type_affinity(ct1.transcriptome[ct1_idx], ct2.transcriptome[ct2_idx])
        if self.linkage_cell == 'single':
            dist = dists.min()
        elif self.linkage_cell == 'complete':
            dist = dists.max()
        else:  # default to 'average'
            dist = dists.mean()

        return np.float64(dist)

    def _compute_region_dist(self, r1: Region, r2: Region) -> np.float64:
        ct_dists = np.zeros((len(r1.cell_types), len(r2.cell_types)))
        for r1_idx, r2_idx in product(range(len(r1.cell_types)), range(len(r2.cell_types))):
            ct_dists[r1_idx, r2_idx] = self._compute_ct_dist(r1.cell_types[r1_idx], r2.cell_types[r2_idx])

        if self.linkage_region == 'single':
            dist = ct_dists.min()
        elif self.linkage_region == 'complete':
            dist = ct_dists.max()
        else:   # default to 'average':
            dist = ct_dists.mean()

        return dist

    def agglomerate(self, data: pd.DataFrame) -> pd.DataFrame:
        ct_dists: PriorityQueue[Edge] = PriorityQueue()
        r_dists: PriorityQueue[Edge] = PriorityQueue()

        ct_names = data.index.values
        ct_regions = np.vectorize(get_region)(ct_names)
        r_names = np.unique(ct_regions)
        region_to_id: Dict[str, int] = {r_names[i]: i for i in range(r_names.shape[0])}

        # Building initial regions and cell types
        regions: Dict[int, Region] = {r: Region(r) for r in range(len(r_names))}
        data_plain = data.to_numpy()

        cell_types: Dict[int, CellType] = {}
        for c in range(len(ct_names)):
            r_id = region_to_id[ct_regions[c]]
            cell_types[c] = CellType(c, r_id, data_plain[c])
            regions[r_id].cell_types.append(cell_types[c])

        ct_id_idx = len(ct_names)
        r_id_idx = len(r_names)

        # repeat until we're left with one region and one cell type
        # not necessarily true evolutionarily, but same assumption as normal dendrogram
        while len(regions) > 1 or len(cell_types) > 1:
            # Compute distances of all possible edges between cell types in the same region
            for region in regions.values():
                for ct1, ct2 in combinations(region.cell_types, 2):
                    dist = self._compute_ct_dist(ct1, ct2)
                    # add the edge with the desired distance to the priority queue
                    ct_dists.put(Edge(dist, ct1, ct2))

            for r1, r2 in combinations(regions.values(), 2):
                # condition for merging regions
                # currently must have same number of cell types
                # so that when regions are merged, each cell type is assumed to have an exact homolog
                if len(r1.cell_types) != len(r2.cell_types):
                    continue

                dist = self._compute_region_dist(r1, r2)
                r_dists.put(Edge(dist, r1, r2))