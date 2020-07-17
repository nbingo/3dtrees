from typing import List, Callable
from queue import PriorityQueue
from data_utils import get_region
from itertools import combinations, product
from data_types import *
from tqdm import tqdm
import numpy as np
import pandas as pd

LINKAGE_CELL_OPTIONS = ['single', 'complete', 'average']
LINKAGE_REGION_OPTIONS = ['single', 'complete', 'average', 'homolog_avg']


class Agglomerate3D:
    def __init__(self,
                 cell_type_affinity: Callable,
                 linkage_cell: str,
                 linkage_region: str,
                 max_region_diff: Optional[int] = 0,
                 region_dist_scale: Optional[float] = 1,
                 verbose: Optional[bool] = False,
                 pbar: Optional[bool] = False,
                 integrity_check: Optional[bool] = True):
        self.cell_type_affinity = cell_type_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.max_region_diff = max_region_diff
        self.region_dist_scale = region_dist_scale
        self.verbose = verbose
        self.integrity_check = integrity_check
        self.linkage_history: List[Dict[str, int]] = []
        self.regions: Dict[int, Region] = {}
        self.cell_types: Dict[int, CellType] = {}
        self.orig_cell_types: Dict[int, CellType] = {}
        self.ct_id_idx: int = 0
        self.r_id_idx: int = 0
        self.ct_names: List[str] = []
        self.r_names: List[str] = []
        self.pbar = tqdm() if pbar else None
        if linkage_cell not in LINKAGE_CELL_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for cell linkage. Must be one of {LINKAGE_CELL_OPTIONS}')
        if linkage_region not in LINKAGE_REGION_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for region linkage. Must be one of '
                              f'{LINKAGE_REGION_OPTIONS}')

    @property
    def linkage_mat(self):
        return pd.DataFrame(self.linkage_history)

    @property
    def linkage_tree(self):
        return Node.tree_from_link_mat(self.linkage_mat)

    def _assert_integrity(self):
        # Make sure all cell types belong to their corresponding region
        for ct_id in self.cell_types:
            assert self.cell_types[ct_id].id_num == ct_id, 'Cell type dict key-value mismatch'
            assert ct_id in self.regions[self.cell_types[ct_id].region].cell_types, 'Cell type not in indicated region.'
        for r in self.regions.values():
            for ct_id in r.cell_types:
                assert r.cell_types[ct_id].id_num == ct_id, 'Within region cell type dict key-value mismatch'
                assert ct_id in self.cell_types, 'Region has cell type that does not exist recorded cell types.'

    def _trace_all_root_leaf_paths(self) -> Dict[int, List[int]]:
        assert not self.linkage_mat.empty, 'Tried tracing empty tree.'

        paths: Dict[int, List[int]] = {}

        def dfs(node: Node, path: List[int]):
            # we are a leaf node
            if node.right is None:
                # by construction, either left and right are None, or neither is None
                assert node.left is None
                paths[node.id_num] = path.copy()
            else:
                # choose
                path.append(node.right.id_num)
                # explore
                dfs(node.right, path)
                # un-choose
                path.pop()

                # repeat for left
                path.append(node.left.id_num)
                dfs(node.left, path)
                path.pop()

        dfs(self.linkage_tree, [])
        return paths

    def _compute_orig_ct_path_dists(self):
        num_ct = len(self.orig_cell_types)
        dists = np.zeros((num_ct, num_ct))
        paths = self._trace_all_root_leaf_paths()
        for ct1_idx, ct2_idx in product(range(num_ct), range(num_ct)):
            ct1_path = paths[ct1_idx][::-1]
            ct2_path = paths[ct2_idx][::-1]
            while (len(ct1_path) > 0) and (len(ct2_path) > 0) and (ct1_path[-1] == ct2_path[-1]):
                ct1_path.pop()
                ct2_path.pop()
            dists[ct1_idx, ct2_idx] = len(ct1_path) + len(ct2_path)
        return dists

    def _compute_orig_ct_linkage_dists(self):
        num_ct = len(self.orig_cell_types)
        dists = np.zeros((num_ct, num_ct))
        for ct1_idx, ct2_idx in product(range(num_ct), range(num_ct)):
            dists[ct1_idx, ct2_idx] = self._compute_ct_dist(self.orig_cell_types[ct1_idx],
                                                            self.orig_cell_types[ct2_idx])
        return dists

    def compute_bme_score(self) -> float:
        path_dists = self._compute_orig_ct_path_dists()
        linkage_dists = self._compute_orig_ct_linkage_dists()
        normalized_dists = linkage_dists / (2 ** path_dists)
        return normalized_dists.sum()

    def compute_me_score(self) -> float:
        # Get only the rows that make sense to sum
        to_sum = self.linkage_mat.loc[self.linkage_mat['Is region'] == self.linkage_mat['In reg merge']]
        return to_sum['Distance'].to_numpy().sum()

    def compute_mp_score(self) -> float:
        to_sum = self.linkage_mat.loc[self.linkage_mat['Is region'] == self.linkage_mat['In reg merge']]
        return to_sum.shape[0]

    # noinspection PyArgumentList
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

    # noinspection PyArgumentList
    def _compute_region_dist(self, r1: Region, r2: Region) -> np.float64:
        ct_dists = np.zeros((len(r1.cell_types), len(r2.cell_types)))
        r1_ct_list = list(r1.cell_types.values())
        r2_ct_list = list(r2.cell_types.values())
        for r1_idx, r2_idx in product(range(r1.num_cell_types), range(r2.num_cell_types)):
            ct_dists[r1_idx, r2_idx] = self._compute_ct_dist(r1_ct_list[r1_idx], r2_ct_list[r2_idx])

        if self.linkage_region == 'single':
            dist = ct_dists.min()
        elif self.linkage_region == 'complete':
            dist = ct_dists.max()
        elif self.linkage_region == 'homolog_avg':
            dists = []
            for i in range(np.min(ct_dists.shape)):
                # Add the distance between the two closest cell types (can consider as homologs)
                dists.append(ct_dists.min())
                # Delete these two homologs from the distance matrix
                ct_min1_idx, ct_min2_idx = np.unravel_index(np.argmin(ct_dists), ct_dists.shape)
                ct_dists = np.delete(ct_dists, ct_min1_idx, axis=0)
                ct_dists = np.delete(ct_dists, ct_min2_idx, axis=1)
            dist = np.mean(dists)
        else:  # default to 'average':
            dist = ct_dists.mean()

        return dist * self.region_dist_scale

    def _merge_cell_types(self, ct1: CellType, ct2: CellType, ct_dist: float, region_id: Optional[int] = None):
        # must be in same region if not being created into a new region
        if region_id is None:
            assert ct1.region == ct2.region, \
                'Tried merging cell types from different regions without new target region.'
            region_id = ct1.region

        # Create new cell type and assign to region
        self.cell_types[self.ct_id_idx] = CellType(self.ct_id_idx,
                                                   region_id,
                                                   np.row_stack((ct1.transcriptome, ct2.transcriptome)))
        self.regions[region_id].cell_types[self.ct_id_idx] = self.cell_types[self.ct_id_idx]

        self._record_link(ct1, ct2, self.cell_types[self.ct_id_idx], ct_dist)

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

        num_ct_diff = np.abs(r1.num_cell_types - r2.num_cell_types)

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
            # noinspection PyArgumentList
            new_ct_id = self._merge_cell_types(r1_ct_list[ct_merge1_idx], r2_ct_list[ct_merge2_idx],
                                               pairwise_r_ct_dists.min(), self.r_id_idx)

            # remove from the distance matrix
            pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge1_idx, axis=0)
            r1_ct_list.pop(ct_merge1_idx)
            pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge2_idx, axis=1)
            r2_ct_list.pop(ct_merge2_idx)

            # add to our new region
            self.regions[self.r_id_idx].cell_types[new_ct_id] = self.cell_types[new_ct_id]
        # Should have at least one empty region
        assert r1.num_cell_types == 0 or r2.num_cell_types == 0, 'Both regions non-empty after primary merging.'
        # if there is a nonempty region, put the remainder of the cell types in the non-empty region into the new region
        if r1.num_cell_types > 0 or r2.num_cell_types > 0:
            r_leftover = r1 if r1.num_cell_types > 0 else r2
            for ct in r_leftover.cell_types.values():
                ct.region = self.r_id_idx
                self.regions[self.r_id_idx].cell_types[ct.id_num] = ct
            r_leftover.cell_types.clear()

        # make sure no cell types are leftover in the regions we're about to delete
        assert r1.num_cell_types == 0 and r2.num_cell_types == 0, 'Tried deleting non-empty regions.'
        self.regions.pop(r1.id_num)
        self.regions.pop(r2.id_num)

        self._record_link(r1, r2, self.regions[self.r_id_idx], r_dist, num_ct_diff)

        if self.verbose:
            print(f'Merged regions {r1} and {r2} with distance {r_dist} to form '
                  f'{self.regions[self.r_id_idx]} with {self.regions[self.r_id_idx].num_original} original data points.'
                  f'\nNew region dict: {self.regions}\n}}\n')

        self.r_id_idx += 1
        return self.r_id_idx - 1

    def _record_link(self, n1: Mergeable, n2: Mergeable, new_node: Mergeable, dist: float,
                     ct_num_diff: Optional[int] = None):
        # Must be recording the linkage of two things of the same type
        assert type(n1) is type(n2), 'Tried recording linkage of a cell type with a region.'

        if self.pbar is not None:
            self.pbar.update(1)

        # record merger in linkage history
        region_merger = isinstance(n1, Region) or (n1.region != n2.region)
        self.linkage_history.append({'Is region': isinstance(n1, Region),
                                     'ID1': n1.id_num,
                                     'ID2': n2.id_num,
                                     'new ID': new_node.id_num,
                                     'Distance': dist,
                                     'Num original': new_node.num_original,
                                     'In region': new_node.region,
                                     'In reg merge': region_merger,
                                     'Cell type num diff': ct_num_diff
                                     })

    @property
    def linkage_mat_readable(self):
        lm = self.linkage_mat.copy()
        id_to_ct = {i: self.ct_names[i] for i in range(len(self.ct_names))}
        id_to_r = {i: self.r_names[i] for i in range(len(self.r_names))}
        for i in lm.index:
            id_to_x = id_to_r if lm.loc[i, 'Is region'] else id_to_ct
            if lm.loc[i, 'ID1'] in id_to_x:
                lm.loc[i, 'ID1'] = id_to_x[lm.loc[i, 'ID1']]
            if lm.loc[i, 'ID2'] in id_to_x:
                lm.loc[i, 'ID2'] = id_to_x[lm.loc[i, 'ID2']]
            if lm.loc[i, 'In region'] in id_to_r:
                lm.loc[i, 'In region'] = id_to_r[lm.loc[i, 'In region']]

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
            self.orig_cell_types[c] = CellType(c, r_id, data_plain[c])
            self.regions[r_id].cell_types[c] = self.orig_cell_types[c]
        self.cell_types = self.orig_cell_types.copy()

        self.ct_id_idx = len(self.ct_names)
        self.r_id_idx = len(self.r_names)

        if self.pbar is not None:
            self.pbar.total = len(self.ct_names) + len(self.r_names) - 2

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

            # compute distances between merge-able regions
            for r1, r2 in combinations(self.regions.values(), 2):
                # condition for merging regions
                # regions can only differ by self.max_region_diff number of cell types
                if np.abs(r1.num_cell_types - r2.num_cell_types) > self.max_region_diff:
                    continue

                dist = self._compute_region_dist(r1, r2)
                r_dists.put(Edge(dist, r1, r2))

            # Now go on to merge step!
            # Decide whether we're merging cell types or regions
            ct_edge = ct_dists.get() if not ct_dists.empty() else None
            r_edge = r_dists.get() if not r_dists.empty() else None

            # both shouldn't be None
            assert not (ct_edge is None and r_edge is None), 'No cell types or regions to merge.'

            # we're merging cell types, which gets a slight preference if equal
            if ct_edge is not None and ((r_edge is None) or (ct_edge.dist <= r_edge.dist)):
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

            if self.integrity_check:
                self._assert_integrity()
        if self.pbar is not None:
            self.pbar.close()
        return self.linkage_mat
