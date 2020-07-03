from dataclasses import dataclass, field
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
    _transcriptome: np.array

    @property
    def transcriptome(self) -> np.array:
        if len(self._transcriptome.shape) == 1:
            return self._transcriptome.reshape(1, -1)
        return self._transcriptome

    # @transcriptome.setter
    # def transcriptome(self, t: np.array):
    #     self._transcriptome = t

    @property
    def num_original(self):
        return self.transcriptome.shape[0]


@dataclass
class Region:
    id_num: int
    cell_types: Dict[int, CellType] = field(default_factory=dict)

    @property
    def transcriptomes(self) -> np.array:
        # ugly -- should refactor something
        ct_list = list(self.cell_types.values())
        transcriptome_length = ct_list[0].transcriptome.shape[1]
        transcriptomes = np.zeros((len(self.cell_types), transcriptome_length))
        for c in range(len(self.cell_types)):
            transcriptomes[c] = ct_list[c].transcriptome
        return transcriptomes

    @property
    def num_original(self):
        return np.sum([ct.num_original for ct in self.cell_types.values()])


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


LINKAGE_CELL_OPTIONS = ['single', 'complete', 'average']
LINKAGE_REGION_OPTIONS = ['single', 'complete', 'average']


class Agglomerate3D:
    def __init__(self, cell_type_affinity: Callable, region_affinity: Callable, linkage_cell: str, linkage_region: str):
        self.cell_type_affinity = cell_type_affinity
        self.region_affinity = region_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.linkage_history: List[Dict[str, int]] = []
        self.regions: Dict[int, Region] = {}
        self.cell_types: Dict[int, CellType] = {}
        self.ct_id_idx: int = 0
        self.r_id_idx: int = 0
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
        r1_ct_list = list(r1.cell_types.values())
        r2_ct_list = list(r2.cell_types.values())
        for r1_idx, r2_idx in product(range(len(r1.cell_types)), range(len(r2.cell_types))):
            ct_dists[r1_idx, r2_idx] = self._compute_ct_dist(r1_ct_list[r1_idx], r2_ct_list[r2_idx])

        if self.linkage_region == 'single':
            dist = ct_dists.min()
        elif self.linkage_region == 'complete':
            dist = ct_dists.max()
        else:  # default to 'average':
            dist = ct_dists.mean()

        return dist

    def _merge_cell_types(self, ct1, ct2, ct_dist):
        # Create new cell type and assign to region
        self.cell_types[self.ct_id_idx] = CellType(self.ct_id_idx,
                                                   ct1.region,
                                                   np.stack((ct1.transcriptome, ct2.transcriptome)))
        self.regions[ct1.region].cell_types[self.ct_id_idx] = self.cell_types[self.ct_id_idx]
        # record merger in linkage history
        self.linkage_history.append({'Is region': False,
                                     'ID1': ct1.id_num,
                                     'ID2': ct2.id_num,
                                     'Distance': ct_dist,
                                     'Num original': ct1.num_original + ct2.num_original
                                     })
        # increment cell type counter
        self.ct_id_idx += 1
        # remove the old ones
        self.cell_types.pop(ct1.id_num)
        self.cell_types.pop(ct2.id_num)
        self.regions[ct1.region].cell_types.pop(ct1.id_num)
        self.regions[ct1.region].cell_types.pop(ct2.id_num)

        # return id of newly created cell type
        return self.ct_id_idx - 1  # yeah, this is ugly b/c python doesn't have ++ct_id_idx

    def _merge_regions(self, r1, r2, r_dist):
        r1_ct_list = list(r1.cell_types.values())
        r2_ct_list = list(r2.cell_types.values())
        # create new region
        self.regions[self.r_id_idx] = Region(self.r_id_idx)
        pairwise_r_ct_dists = np.zeros((len(r1.cell_types), len(r2.cell_types)))
        for r1_ct_idx, r2_ct_idx in product(range(len(r1_ct_list)), range(len(r2_ct_list))):
            pairwise_r_ct_dists[r1_ct_idx, r2_ct_list] = self._compute_ct_dist(r1_ct_list[r1_ct_idx],
                                                                               r2_ct_list[r2_ct_idx])
        # Continuously pair up cell types, merge them, add them to the new region, and delete them
        while np.prod(pairwise_r_ct_dists.shape) != 0:
            ct_merge1_idx, ct_merge2_idx = np.unravel_index(np.argmin(pairwise_r_ct_dists),
                                                            pairwise_r_ct_dists.shape)
            # create new cell type, delete old ones and remove from their regions
            new_ct_id = self._merge_cell_types(r1_ct_list[ct_merge1_idx], r2_ct_list[ct_merge2_idx],
                                               pairwise_r_ct_dists.min())

            # remove from the distance matrix
            pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge1_idx, axis=0)
            pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge2_idx, axis=1)

            # add to our new region
            self.regions[self.r_id_idx].cell_types[new_ct_id] = self.cell_types[new_ct_id]
        # make sure no cell types are leftover in the regions we're about to delete
        assert len(r1.cell_types) == 0 and len(r2.cell_types) == 0
        self.regions.pop(r1.id_num)
        self.regions.pop(r2.id_num)

        # record merger in linkage history
        self.linkage_history.append({'Is region': True,
                                     'ID1': r1.id_num,
                                     'ID2': r2.id_num,
                                     'Distance': r_dist,
                                     'Num original': r1.num_original + r2.num_original
                                     })

        self.r_id_idx += 1
        return self.r_id_idx - 1

    def agglomerate(self, data: pd.DataFrame) -> pd.DataFrame:
        ct_dists: PriorityQueue[Edge] = PriorityQueue()
        r_dists: PriorityQueue[Edge] = PriorityQueue()

        ct_names = data.index.values
        ct_regions = np.vectorize(get_region)(ct_names)
        r_names = np.unique(ct_regions)
        region_to_id: Dict[str, int] = {r_names[i]: i for i in range(r_names.shape[0])}

        # Building initial regions and cell types
        self.regions = {r: Region(r) for r in range(len(r_names))}
        data_plain = data.to_numpy()

        for c in range(len(ct_names)):
            r_id = region_to_id[ct_regions[c]]
            self.cell_types[c] = CellType(c, r_id, data_plain[c])
            self.regions[r_id].cell_types[c] = self.cell_types[c]

        self.ct_id_idx = len(ct_names)
        self.r_id_idx = len(r_names)

        # repeat until we're left with one region and one cell type
        # not necessarily true evolutionarily, but same assumption as normal dendrogram
        while len(self.regions) > 1 or len(self.cell_types) > 1:
            # Compute distances of all possible edges between cell types in the same region
            for region in self.regions.values():
                for ct1, ct2 in combinations(list(region.cell_types.values()), 2):
                    dist = self._compute_ct_dist(ct1, ct2)
                    # add the edge with the desired distance to the priority queue
                    ct_dists.put(Edge(dist, ct1, ct2))

            # compute distances between mergeable regions
            for r1, r2 in combinations(self.regions.values(), 2):
                # condition for merging regions
                # currently must have same number of cell types
                # so that when regions are merged, each cell type is assumed to have an exact homolog
                if len(r1.cell_types) != len(r2.cell_types):
                    continue

                dist = self._compute_region_dist(r1, r2)
                r_dists.put(Edge(dist, r1, r2))

            # Now go on to merge step!
            # Decide whether we're merging cell types or regions
            ct_edge = ct_dists.get()
            r_edge = r_dists.get()

            # we're merging cell types, which gets a slight preference if equal
            if ct_edge.dist <= r_edge.dist:
                ct1 = ct_edge.endpt1
                ct2 = ct_edge.endpt2

                # of course, assumed to be in the same region
                assert ct1.region == ct2.region

                self._merge_cell_types(ct1, ct2, ct_edge.dist)

            # we're merging regions
            else:
                # First, we have to match up homologous cell types
                # Just look for closest pairs and match them up
                r1 = r_edge.endpt1
                r2 = r_edge.endpt2
                self._merge_regions(r1, r2, r_edge.dist)

        return pd.DataFrame(self.linkage_history)
