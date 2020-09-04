from typing import List, Callable, Tuple, Optional, Dict, Union
from queue import PriorityQueue
from data.data_loader import DataLoader
from itertools import combinations, product
from data.data_types import Region, CellType, Edge, Mergeable, LINKAGE_CELL_OPTIONS, LINKAGE_REGION_OPTIONS
from tqdm import tqdm
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
# Need for 3D plotting, even though not used directly. Python is dumb
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

TREE_SCORE_OPTIONS = ['ME', 'BME', 'MP']


@functools.total_ordering
class Agglomerate3D:
    def __init__(self,
                 linkage_cell: str,
                 linkage_region: str,
                 cell_type_affinity: Callable,
                 region_affinity: Optional[Callable] = None,
                 max_region_diff: Optional[int] = 0,
                 region_dist_scale: Optional[float] = 1,
                 verbose: Optional[bool] = False,
                 pbar: Optional[bool] = False,
                 integrity_check: Optional[bool] = True):
        self.linkage_cell: str = linkage_cell
        self.linkage_region: str = linkage_region
        self.cell_type_affinity: Callable = cell_type_affinity
        self.region_affinity: Callable = region_affinity
        self.max_region_diff: int = max_region_diff
        self.region_dist_scale: float = region_dist_scale
        self.verbose: bool = verbose
        self.integrity_check: bool = integrity_check
        self.linkage_history: List[Dict[str, int]] = []
        self._linkage_mat: pd.DataFrame = pd.DataFrame()
        self.regions: Dict[int, Region] = {}
        self.cell_types: Dict[int, CellType] = {}
        self.orig_cell_types: Dict[int, CellType] = {}
        self._ct_id_idx: int = 0
        self._r_id_idx: int = 0
        self.ct_names: List[str] = []
        self.r_names: List[str] = []
        self._ct_axis_mask = None
        self._r_axis_mask = None
        self._pbar = tqdm() if pbar else None
        if linkage_cell not in LINKAGE_CELL_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for cell linkage. Must be one of {LINKAGE_CELL_OPTIONS}')
        if linkage_region not in LINKAGE_REGION_OPTIONS:
            raise UserWarning(f'Incorrect argument passed in for region linkage. Must be one of '
                              f'{LINKAGE_REGION_OPTIONS}')

    def __repr__(self):
        return f'Agglomerate3D<cell_type_affinity={self.cell_type_affinity}, ' \
               f'linkage_cell={self.linkage_cell}, ' \
               f'linkage_region={self.linkage_region}, ' \
               f'max_region_diff={self.max_region_diff}, ' \
               f'region_dist_scale={self.region_dist_scale}>'

    def __eq__(self, other):
        return len(self.linkage_mat.index) == len(other.linkage_mat.index)

    def __lt__(self, other):
        return len(self.linkage_mat.index) < len(other.linkage_mat.index)

    @property
    def linkage_mat(self) -> pd.DataFrame:
        if self._linkage_mat.empty:
            return pd.DataFrame(self.linkage_history)
        return self._linkage_mat

    def view_tree3d(self):
        lm = self.linkage_mat
        segments = []
        colors = []
        num_regions = lm['In region'].max() + 1
        colormap = cm.get_cmap('hsv')(np.linspace(0, 1, num_regions))
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        def find_ct_index_region(ct_id: int, index: int) -> Tuple[Union[int, None], Union[int, None]]:
            if np.isnan(ct_id):
                return np.nan, np.nan
            # Cutoff at where we currently are so we don't repeat rows
            lm_bound = lm.loc[:index - 1]
            no_reg = lm_bound[~lm_bound['Is region']]
            ct_row = no_reg[no_reg['New ID'] == ct_id]
            # one of the original cell types and at the end of a branch
            if ct_row.empty:
                return None, self.orig_cell_types[ct_id].region
            # not one of the original ones, so have to check which region it's in
            return ct_row.index[-1], ct_row['In region'].iat[-1]

        def segment_builder(level: int, index: int, root_pos: List[int]):
            offset = 2 ** (level - 1)  # subtract 1 to divide by 2 since it's only half the line

            # figure out where to recurse on
            ct1_id, ct2_id = lm.loc[index, ['ID1', 'ID2']]
            ct1_index, ct1_region = find_ct_index_region(ct1_id, index)
            ct2_index, ct2_region = find_ct_index_region(ct2_id, index)

            if lm.loc[index, 'In reg merge']:
                # We're drawing on the y-axis
                split_axis = 1

                # Find the region we're merging in
                region = lm.loc[index, 'In region']
                region_mat = lm[lm['In region'] == region]
                dist = region_mat[region_mat['Is region']]['Distance'].iat[0]
            else:
                # We're drawing on the x-axis
                split_axis = 0
                dist = lm.loc[index, 'Distance']

            # To have the correct order of recursion so region splits match up
            # Also the case in which a cell type is just transferred between regions
            if (np.isnan(ct2_region)) or (ct1_region < ct2_region):
                l_index = None if ct1_index == index else ct1_index
                l_id = ct1_id
                l_region = ct1_region
                r_index = ct2_index
                r_id = ct2_id
                r_region = ct2_region
            else:
                l_index = ct2_index
                l_id = ct2_id
                l_region = ct2_region
                r_index = ct1_index
                r_id = ct1_id
                r_region = ct1_region

            # horizontal x/y-axis bar
            # Start is the left side
            h_start = root_pos.copy()
            h_start[split_axis] -= offset
            # end is the right side
            h_end = root_pos.copy()
            h_end[split_axis] += offset
            segments.append([h_start, root_pos])
            colors.append(colormap[l_region])
            # Don't do if just transferring one cell type to another region
            if ~np.isnan(r_region):
                segments.append([root_pos, h_end])
                colors.append(colormap[r_region])

            # vertical z-axis bars
            v_left_end = h_start.copy()
            v_left_end[2] -= dist
            v_right_end = h_end.copy()
            v_right_end[2] -= dist
            segments.append([h_start, v_left_end])
            colors.append(colormap[l_region])
            # Don't do if just transferring one cell type to another region
            if ~np.isnan(r_region):
                segments.append([h_end, v_right_end])
                colors.append(colormap[r_region])

            # don't recurse if at leaf, but do label
            if l_index is None:
                label = self.ct_names[int(l_id)]
                ax.text(*v_left_end, label, 'z')
            else:
                segment_builder(level - 1, l_index, v_left_end)
            # Don't do if just transferring one cell type to another region
            if ~np.isnan(r_region):
                if r_index is None:
                    label = self.ct_names[int(r_id)]
                    ax.text(*v_right_end, label, 'z')
                else:
                    segment_builder(level - 1, r_index, v_right_end)

        # Create root pos z-pos as max of sum of region and ct distances
        top_root_pos = [0, 0, lm['Distance'].sum()]
        top_level = len(lm.index) - 1
        # Should only happen if our tree starts with a region merger, which must consist of two cell types
        if lm.loc[top_level, 'Is region']:
            segment_builder(top_level, top_level - 1, top_root_pos)
        else:
            segment_builder(top_level, top_level, top_root_pos)

        segments = np.array(segments)

        x = segments[:, :, 0].flatten()
        y = segments[:, :, 1].flatten()
        z = segments[:, :, 2].flatten()

        ax.set_zlim(z.min(), z.max())
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel('Cell type', fontsize=20)
        ax.set_ylabel('Region', fontsize=20)
        ax.set_zlabel('Distance', fontsize=20)
        ax.set(xticklabels=[], yticklabels=[])
        for line, color in zip(segments, colors):
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, lw=2)
        plt.show()

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
        # Get only cell types
        lm = self.linkage_mat[~self.linkage_mat['Is region']]
        # Reduce to only ID numbers in numpy
        lm = lm[['ID1', 'ID2', 'New ID']].to_numpy()
        # Aliases for numpy indices
        ids = [0, 1]
        new_id = 2

        def dfs(row: np.array, path: List[int]):
            for id_idx in ids:
                # If there's a child on the side we're looking at
                if ~np.isnan(row[id_idx]):
                    # Is it a leaf node
                    if row[id_idx] in self.orig_cell_types:
                        path.append(row[id_idx])
                        paths[row[id_idx]] = path.copy()
                        path.pop()
                    else:
                        # choose
                        path.append(row[id_idx])
                        # explore
                        dfs(lm[lm[:, new_id] == row[id_idx]].squeeze(), path)
                        # un-choose
                        path.pop()

        dfs(lm[-1], [lm[-1, new_id]])
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
            dists[ct1_idx, ct2_idx] = CellType.diff(self.orig_cell_types[ct1_idx], self.orig_cell_types[ct2_idx],
                                                    affinity=self.cell_type_affinity, linkage=self.linkage_cell,
                                                    mask=self._ct_axis_mask)
        return dists

    def _compute_bme_score(self) -> float:
        path_dists = self._compute_orig_ct_path_dists()
        linkage_dists = self._compute_orig_ct_linkage_dists()
        normalized_dists = linkage_dists / (2 ** path_dists)
        return normalized_dists.sum()

    def _compute_me_score(self) -> float:
        # Get only the rows that make sense to sum
        to_sum = self.linkage_mat.loc[self.linkage_mat['Is region'] == self.linkage_mat['In reg merge']]
        return to_sum['Distance'].to_numpy().sum()

    def _compute_mp_score(self) -> float:
        to_sum = self.linkage_mat.loc[self.linkage_mat['Is region'] == self.linkage_mat['In reg merge']]
        return to_sum.shape[0]

    def compute_tree_score(self, metric: str):
        if metric not in TREE_SCORE_OPTIONS:
            raise ValueError(f'metric must be one of: {TREE_SCORE_OPTIONS}.')
        if metric == 'ME':
            return self._compute_me_score()
        elif metric == 'MP':
            return self._compute_mp_score()
        elif metric == 'BME':
            return self._compute_bme_score()

    def _merge_cell_types(self, ct1: CellType, ct2: CellType, ct_dist: float, region_id: Optional[int] = None):
        # Create new cell type and assign to region
        new_ct = CellType.merge(ct1, ct2, self._ct_id_idx, region_id)
        self.cell_types[self._ct_id_idx] = new_ct
        self.regions[new_ct.region].cell_types[new_ct.id_num] = new_ct

        self._record_link(ct1, ct2, self.cell_types[self._ct_id_idx], ct_dist)

        # remove the old ones
        self.cell_types.pop(ct1.id_num)
        self.cell_types.pop(ct2.id_num)
        self.regions[ct1.region].cell_types.pop(ct1.id_num)
        self.regions[ct2.region].cell_types.pop(ct2.id_num)

        if self.verbose:
            print(f'Merged cell types {ct1} and {ct2} with distance {ct_dist} '
                  f'to form cell type {self.cell_types[self._ct_id_idx]} with {ct1.num_original + ct2.num_original} '
                  f'original data points.\n'
                  f'New cell type dict: {self.cell_types}\n'
                  f'New region dict: {self.regions}\n')

        # increment cell type counter
        self._ct_id_idx += 1

        # return id of newly created cell type
        return self._ct_id_idx - 1  # yeah, this is ugly b/c python doesn't have ++_ct_id_idx

    def _merge_regions(self, r1, r2, r_dist):
        r1_ct_list = list(r1.cell_types.values())
        r2_ct_list = list(r2.cell_types.values())

        if self.verbose:
            print(f'Merging regions {r1} and {r2} into new region {self._r_id_idx}\n{{')

        # create new region
        self.regions[self._r_id_idx] = Region(self._r_id_idx)
        pairwise_r_ct_dists = np.zeros((len(r1.cell_types), len(r2.cell_types)))
        for r1_ct_idx, r2_ct_idx in product(range(len(r1_ct_list)), range(len(r2_ct_list))):
            pairwise_r_ct_dists[r1_ct_idx, r2_ct_idx] = CellType.diff(r1_ct_list[r1_ct_idx], r2_ct_list[r2_ct_idx],
                                                                      affinity=self.cell_type_affinity,
                                                                      linkage=self.linkage_cell,
                                                                      mask=self._ct_axis_mask)

        # Find the cell types that have to be merged between the two regions
        cts_merge: List[Tuple[CellType, CellType]] = []
        dists: List[float] = []
        if self.linkage_region == 'homolog_mnn':
            # Nearest neighbors for the cell types from region 1
            r1_ct_nn = np.argmin(pairwise_r_ct_dists, axis=1)
            # Nearest neighbors for the cell types from region 2
            r2_ct_nn = np.argmin(pairwise_r_ct_dists, axis=0)
            # Only append distance if we find a mutual nearest neighbor
            for i in range(r1_ct_nn.shape[0]):
                if r2_ct_nn[r1_ct_nn[i]] == i:
                    dists.append(pairwise_r_ct_dists[i, r1_ct_nn[i]])
                    cts_merge.append((r1_ct_list[i], r2_ct_list[r1_ct_nn[i]]))
        # otherwise just do a greedy pairing
        else:
            while np.prod(pairwise_r_ct_dists.shape) != 0:
                ct_merge1_idx, ct_merge2_idx = np.unravel_index(np.argmin(pairwise_r_ct_dists),
                                                                pairwise_r_ct_dists.shape)

                # Append distance to dists and indices to index list
                # noinspection PyArgumentList
                dists.append(pairwise_r_ct_dists.min())
                cts_merge.append((r1_ct_list[ct_merge1_idx], r2_ct_list[ct_merge2_idx]))

                # remove from the distance matrix
                pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge1_idx, axis=0)
                r1_ct_list.pop(ct_merge1_idx)
                pairwise_r_ct_dists = np.delete(pairwise_r_ct_dists, ct_merge2_idx, axis=1)
                r2_ct_list.pop(ct_merge2_idx)

        assert len(dists) == len(cts_merge), 'Number distances not equal to number of cell type mergers.'
        num_ct_diff = r1.num_cell_types + r2.num_cell_types - (2 * len(cts_merge))
        # Continuously pair up cell types, merge them, add them to the new region, and delete them
        for dist, (ct1, ct2) in zip(dists, cts_merge):
            # create new cell type, delete old ones and remove from their regions
            # noinspection PyArgumentList
            new_ct_id = self._merge_cell_types(ct1, ct2, dist, self._r_id_idx)

            # add to our new region
            self.regions[self._r_id_idx].cell_types[new_ct_id] = self.cell_types[new_ct_id]
        # Should have at least one empty region if not doing mutual nearest neighbors
        if self.linkage_region != 'homolog_mnn':
            assert r1.num_cell_types == 0 or r2.num_cell_types == 0, 'Both regions non-empty after primary merging.'
        # if there is a nonempty region, put the remainder of the cell types in the non-empty region into the new region
        for r_leftover in (r1, r2):
            for ct in r_leftover.cell_types.values():
                # Essentially copy the cell type but into a new region and with a new ID
                new_ct = CellType(self._ct_id_idx, self._r_id_idx, ct.transcriptome)
                self.cell_types[new_ct.id_num] = new_ct
                self.regions[self._r_id_idx].cell_types[new_ct.id_num] = new_ct
                # Delete the old cell type
                self.cell_types.pop(ct.id_num)
                # Record the transfer
                self._record_ct_transfer(ct, new_ct)
                self._ct_id_idx += 1
            r_leftover.cell_types.clear()

        # make sure no cell types are leftover in the regions we're about to delete
        assert r1.num_cell_types == 0 and r2.num_cell_types == 0, 'Tried deleting non-empty regions.'
        self.regions.pop(r1.id_num)
        self.regions.pop(r2.id_num)

        self._record_link(r1, r2, self.regions[self._r_id_idx], r_dist, num_ct_diff)

        if self.verbose:
            print(f'Merged regions {r1} and {r2} with distance {r_dist} to form '
                  f'{self.regions[self._r_id_idx]} with '
                  f'{self.regions[self._r_id_idx].num_original} original data points.'
                  f'\nNew region dict: {self.regions}\n}}\n')

        self._r_id_idx += 1
        return self._r_id_idx - 1

    def _record_ct_transfer(self, ct_orig: CellType, ct_new: CellType):
        assert ct_orig.region != ct_new.region, 'Tried transferring cell type to the same region'
        self.linkage_history.append({'Is region': False,
                                     'ID1': ct_orig.id_num,
                                     'ID2': None,
                                     'New ID': ct_new.id_num,
                                     'Distance': None,
                                     'Num original': ct_new.num_original,
                                     'In region': ct_new.region,
                                     'In reg merge': True,
                                     'Cell type num diff': None
                                     })

    def _record_link(self, n1: Mergeable, n2: Mergeable, new_node: Mergeable, dist: float,
                     ct_num_diff: Optional[int] = None):
        # Must be recording the linkage of two things of the same type
        assert type(n1) == type(n2), 'Tried recording linkage of a cell type with a region.'

        if self._pbar is not None:
            self._pbar.update(1)

        # record merger in linkage history
        region_merger = isinstance(n1, Region) or (n1.region != n2.region)
        self.linkage_history.append({'Is region': isinstance(n1, Region),
                                     'ID1': n1.id_num,
                                     'ID2': n2.id_num,
                                     'New ID': new_node.id_num,
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
        cols = ['ID1', 'ID2', 'New ID']
        for i in lm.index:
            id_to_x = id_to_r if lm.loc[i, 'Is region'] else id_to_ct
            for col in cols:
                if lm.loc[i, col] in id_to_x:
                    lm.loc[i, col] = id_to_x[lm.loc[i, col]]
            if lm.loc[i, 'In region'] in id_to_r:
                lm.loc[i, 'In region'] = id_to_r[lm.loc[i, 'In region']]

        return lm

    def agglomerate(self, data_ct: DataLoader, data_r: Optional[DataLoader] = None) -> pd.DataFrame:
        self.ct_names = data_ct.get_names()
        ct_regions = data_ct.get_corresponding_region_names()

        # Building initial regions and cell types
        if data_r is None:
            self.r_names = np.unique(ct_regions)
            self.regions = {r: Region(r) for r in range(len(self.r_names))}
            self._ct_axis_mask = data_ct.ct_axis_mask
            self._r_axis_mask = data_ct.r_axis_mask
        else:
            self.r_names = data_r.get_names()
            self.regions = {r: Region(r, _transcriptome=data_r[r]) for r in range(len(self.r_names))}

        region_to_id: Dict[str, int] = {self.r_names[i]: i for i in range(len(self.r_names))}

        for c in range(len(data_ct)):
            r_id = region_to_id[ct_regions[c]]
            self.orig_cell_types[c] = CellType(c, r_id, data_ct[c])
            self.regions[r_id].cell_types[c] = self.orig_cell_types[c]
        self.cell_types = self.orig_cell_types.copy()

        self._ct_id_idx = len(self.ct_names)
        self._r_id_idx = len(self.r_names)

        if self._pbar is not None:
            self._pbar.total = len(self.ct_names) + len(self.r_names) - 2

        # repeat until we're left with one region and one cell type
        # not necessarily true evolutionarily, but same assumption as normal dendrogram
        while len(self.regions) > 1 or len(self.cell_types) > 1:
            ct_dists: PriorityQueue[Edge] = PriorityQueue()
            r_dists: PriorityQueue[Edge] = PriorityQueue()

            # Compute distances of all possible edges between cell types in the same region
            for region in self.regions.values():
                for ct1, ct2 in combinations(list(region.cell_types.values()), 2):
                    dist = CellType.diff(ct1, ct2,
                                         affinity=self.cell_type_affinity,
                                         linkage=self.linkage_cell,
                                         mask=self._ct_axis_mask)
                    # add the edge with the desired distance to the priority queue
                    ct_dists.put(Edge(dist, ct1, ct2))

            # compute distances between merge-able regions
            for r1, r2 in combinations(self.regions.values(), 2):
                # condition for merging regions
                # regions can only differ by self.max_region_diff number of cell types
                if np.abs(r1.num_cell_types - r2.num_cell_types) > self.max_region_diff:
                    continue

                dist, num_ct_diff = Region.diff(r1, r2, affinity=self.region_affinity, linkage=self.linkage_region,
                                                affinity2=self.cell_type_affinity, linkage2=self.linkage_cell,
                                                mask=self._r_axis_mask, mask2=self._ct_axis_mask)
                # If we're using region linkage homolog_mnn, then the number of cell types contained different may go up
                if num_ct_diff > self.max_region_diff:
                    continue
                r_dists.put(Edge(dist, r1, r2))

            # Now go on to merge step!
            # Decide whether we're merging cell types or regions
            ct_edge = ct_dists.get() if not ct_dists.empty() else None
            r_edge = r_dists.get() if not r_dists.empty() else None

            # both shouldn't be None
            assert not (ct_edge is None and r_edge is None), 'No cell types or regions to merge.'

            # we're merging cell types, which gets a slight preference if equal
            if ct_edge is not None and ((r_edge is None) or (ct_edge.dist <= r_edge.dist * self.region_dist_scale)):
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
        if self._pbar is not None:
            self._pbar.close()
        return self.linkage_mat
