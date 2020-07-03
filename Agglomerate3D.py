from dataclasses import dataclass, field
from typing import List, Callable, Union, Dict
from queue import PriorityQueue
from data_utils import get_region
from itertools import combinations, product
from abc import ABC, abstractmethod
import functools
import numpy as np
import pandas as pd


LINKAGE_CELL_OPTIONS = ['single', 'complete', 'average']
LINKAGE_REGION_OPTIONS = ['single', 'complete', 'average']


@dataclass
class Node(ABC):
    id_num: int

    @property
    @abstractmethod
    def num_original(self):
        pass


@dataclass
class CellType(Node):
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

    def __repr__(self):
        return f'{self.region}.{self.id_num}'


@dataclass
class Region(Node):
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

    def __repr__(self):
        return f'{self.id_num}{list(self.cell_types.values())}'


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


class Agglomerate3D:
    def __init__(self, cell_type_affinity: Callable, linkage_cell: str, linkage_region: str, verbose: bool = False):
        self.cell_type_affinity = cell_type_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.verbose = verbose
        self.linkage_history: List[Dict[str, int]] = []
        self.regions: Dict[int, Region] = {}
        self.cell_types: Dict[int, CellType] = {}
        self.ct_id_idx: int = 0
        self.r_id_idx: int = 0
        self.ct_names: List[str] = []
        self.r_names: List[str] = []
        if linkage_cell not in LINKAGE_CELL_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for cell linkage. Must be one of {LINKAGE_CELL_OPTIONS}')
        if linkage_region not in LINKAGE_REGION_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for region linkage. Must be one of '
                              f'{LINKAGE_REGION_OPTIONS}')

    @property
    def linkage_mat(self):
        return pd.DataFrame(self.linkage_history)

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

    def _merge_cell_types(self, ct1: CellType, ct2: CellType, ct_dist: float, region_id: int = None):
        # must be in same region if not being created into a new region
        if region_id is None:
            assert ct1.region == ct2.region
            region_id = ct1.region

        # Create new cell type and assign to region
        self.cell_types[self.ct_id_idx] = CellType(self.ct_id_idx,
                                                   region_id,
                                                   np.row_stack((ct1.transcriptome, ct2.transcriptome)))
        self.regions[region_id].cell_types[self.ct_id_idx] = self.cell_types[self.ct_id_idx]

        self._record_link(ct1, ct2, self.ct_id_idx, ct_dist)

        # remove the old ones
        self.cell_types.pop(ct1.id_num)
        self.cell_types.pop(ct2.id_num)
        self.regions[ct1.region].cell_types.pop(ct1.id_num)
        self.regions[ct2.region].cell_types.pop(ct2.id_num)

        if self.verbose:
            print(f'Merged cell types {ct1} and {ct2} with distance {ct_dist} '
                  f'to form cell type {self.cell_types[self.ct_id_idx]} with {ct1.num_original + ct2.num_original} '
                  f'original data points.\n'
                  f'New cell type dict: {self.cell_types}\n'
                  f'New region dict: {self.regions}\n')

        # increment cell type counter
        self.ct_id_idx += 1

        # return id of newly created cell type
        return self.ct_id_idx - 1  # yeah, this is ugly b/c python doesn't have ++ct_id_idx

    def _merge_regions(self, r1, r2, r_dist):
        r1_ct_list = list(r1.cell_types.values())
        r2_ct_list = list(r2.cell_types.values())

        if self.verbose:
            print(f'Merging regions {r1} and {r2} into new region {self.r_id_idx}\n{{')

        # create new region
        self.regions[self.r_id_idx] = Region(self.r_id_idx)
        pairwise_r_ct_dists = np.zeros((len(r1.cell_types), len(r2.cell_types)))
        for r1_ct_idx, r2_ct_idx in product(range(len(r1_ct_list)), range(len(r2_ct_list))):
            pairwise_r_ct_dists[r1_ct_idx, r2_ct_idx] = self._compute_ct_dist(r1_ct_list[r1_ct_idx],
                                                                               r2_ct_list[r2_ct_idx])
        # Continuously pair up cell types, merge them, add them to the new region, and delete them
        while np.prod(pairwise_r_ct_dists.shape) != 0:
            ct_merge1_idx, ct_merge2_idx = np.unravel_index(np.argmin(pairwise_r_ct_dists),
                                                            pairwise_r_ct_dists.shape)
            # create new cell type, delete old ones and remove from their regions
            new_ct_id = self._merge_cell_types(r1_ct_list[ct_merge1_idx], r2_ct_list[ct_merge2_idx],
                                               pairwise_r_ct_dists.min(), self.r_id_idx)

            # remove from the distance matrix
            pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge1_idx, axis=0)
            pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge2_idx, axis=1)

            # add to our new region
            self.regions[self.r_id_idx].cell_types[new_ct_id] = self.cell_types[new_ct_id]
        # make sure no cell types are leftover in the regions we're about to delete
        assert len(r1.cell_types) == 0 and len(r2.cell_types) == 0
        self.regions.pop(r1.id_num)
        self.regions.pop(r2.id_num)

        self._record_link(r1, r2, self.r_id_idx, r_dist)

        if self.verbose:
            print(f'Merged regions {r1} and {r2} with distance {r_dist} to form '
                  f'{self.regions[self.r_id_idx]} with {self.regions[self.r_id_idx].num_original} original data points.'
                  f'\nNew region dict: {self.regions}\n}}\n')

        self.r_id_idx += 1
        return self.r_id_idx - 1

    def _record_link(self, n1: Node, n2: Node, new_id: int, dist: float):
        # Must be recording the linkage of two things of the same type
        assert type(n1) is type(n2)

        # record merger in linkage history
        if type(n1) is Region:
            num_orig = self.regions[new_id].num_original
        else:
            num_orig = self.cell_types[new_id].num_original
        self.linkage_history.append({'Is region': isinstance(n1, Region),
                                     'ID1': n1.id_num,
                                     'ID2': n2.id_num,
                                     'new ID': new_id,
                                     'Distance': dist,
                                     'Num original': num_orig
                                     })

    @property
    def linkage_mat_readable(self):
        lm = self.linkage_mat.copy()
        id_to_ct = {i: self.ct_names[i] for i in range(len(self.ct_names))}
        id_to_r = {i: self.r_names[i] for i in range(len(self.r_names))}
        for i in lm.index:
            if lm.loc[i, 'Is region']:
                if lm.loc[i, 'ID1'] in id_to_r:
                    lm.loc[i, 'ID1'] = id_to_r[lm.loc[i, 'ID1']]
                if lm.loc[i, 'ID2'] in id_to_r:
                    lm.loc[i, 'ID2'] = id_to_r[lm.loc[i, 'ID2']]
            else:
                if lm.loc[i, 'ID1'] in id_to_ct:
                    lm.loc[i, 'ID1'] = id_to_ct[lm.loc[i, 'ID1']]
                if lm.loc[i, 'ID2'] in id_to_ct:
                    lm.loc[i, 'ID2'] = id_to_ct[lm.loc[i, 'ID2']]

        return lm

    def agglomerate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.ct_names = data.index.values
        ct_regions = np.vectorize(get_region)(self.ct_names)
        self.r_names = np.unique(ct_regions)
        region_to_id: Dict[str, int] = {self.r_names[i]: i for i in range(len(self.r_names))}

        # Building initial regions and cell types
        self.regions = {r: Region(r) for r in range(len(self.r_names))}
        data_plain = data.to_numpy()

        for c in range(len(self.ct_names)):
            r_id = region_to_id[ct_regions[c]]
            self.cell_types[c] = CellType(c, r_id, data_plain[c])
            self.regions[r_id].cell_types[c] = self.cell_types[c]

        self.ct_id_idx = len(self.ct_names)
        self.r_id_idx = len(self.r_names)

        # repeat until we're left with one region and one cell type
        # not necessarily true evolutionarily, but same assumption as normal dendrogram
        while len(self.regions) > 1 or len(self.cell_types) > 1:
            ct_dists: PriorityQueue[Edge] = PriorityQueue()
            r_dists: PriorityQueue[Edge] = PriorityQueue()

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
            ct_edge = ct_dists.get() if not ct_dists.empty() else None
            r_edge = r_dists.get() if not r_dists.empty() else None

            # both shouldn't be None
            assert not (ct_edge is None and r_edge is None)

            # we're merging cell types, which gets a slight preference if equal
            if ct_edge is not None and (ct_edge.dist <= r_edge.dist):
                ct1 = ct_edge.endpt1
                ct2 = ct_edge.endpt2

                self._merge_cell_types(ct1, ct2, ct_edge.dist)

            # we're merging regions
            elif r_edge is not None:
                # First, we have to match up homologous cell types
                # Just look for closest pairs and match them up
                r1 = r_edge.endpt1
                r2 = r_edge.endpt2
                self._merge_regions(r1, r2, r_edge.dist)

        return self.linkage_mat
