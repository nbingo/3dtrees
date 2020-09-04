from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Callable, Sequence
from abc import ABC, abstractmethod
from itertools import product
import functools
import numpy as np

LINKAGE_CELL_OPTIONS = ['single', 'complete', 'average']
LINKAGE_REGION_OPTIONS = ['single', 'complete', 'average', 'homolog_avg', 'homolog_mnn']

@dataclass
class Mergeable(ABC):
    id_num: int

    @property
    @abstractmethod
    def num_original(self) -> int:
        pass

    @property
    @abstractmethod
    def region(self) -> int:
        pass

    @property
    @abstractmethod
    def transcriptome(self) -> np.array:
        pass

    # @classmethod
    # @abstractmethod
    # def merge(cls, m1: Mergeable, m2: Mergeable, dist: float, new_id: int, region_id: Optional[int] = None) -> Mergeable:
    #     pass

    # noinspection PyArgumentList
    @staticmethod
    def _pairwise_diff(lhs_transcriptome: np.array, rhs_transcriptome: np.array,
                       affinity: Callable, linkage: str) -> float:
        lhs_len = lhs_transcriptome.shape[0]
        rhs_len = rhs_transcriptome.shape[0]
        dists = np.zeros((lhs_len, rhs_len))
        # Compute distance matrix
        # essentially only useful if this is working on merged cell types
        # otherwise just produces a matrix containing one value
        for ct1_idx, ct2_idx in product(range(lhs_len), range(rhs_len)):
            dists[ct1_idx, ct2_idx] = affinity(lhs_transcriptome[ct1_idx], rhs_transcriptome[ct2_idx])
        if linkage == 'single':
            dist = dists.min()
        elif linkage == 'complete':
            dist = dists.max()
        else:  # default to 'average'
            dist = dists.mean()

        return dist

    @staticmethod
    @abstractmethod
    def diff(lhs: Mergeable, rhs: Mergeable, affinity: Callable, linkage: str,
             affinity2: Optional[Callable] = None, linkage2: Optional[str] = None,
             mask: Optional[Sequence] = None, mask2: Optional[Sequence] = None):
        pass


@dataclass
class CellType(Mergeable):
    _region: int
    _transcriptome: np.array

    @property
    def transcriptome(self) -> np.array:
        if len(self._transcriptome.shape) == 1:
            return self._transcriptome.reshape(1, -1)
        return self._transcriptome

    @property
    def num_original(self):
        return self.transcriptome.shape[0]

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, r: int):
        self._region = r

    @classmethod
    def merge(cls, m1: CellType, m2: CellType, new_id: int, region_id: Optional[int] = None) -> CellType:
        # must be in same region if not being created into a new region
        if region_id is None:
            assert m1.region == m2.region, \
                'Tried merging cell types from different regions without new target region.'
            region_id = m1.region
        return cls(new_id, region_id, np.row_stack((m1.transcriptome, m2.transcriptome)))

    @staticmethod
    def diff(lhs: CellType, rhs: CellType, affinity: Callable, linkage: str,
             affinity2: Optional[Callable] = None, linkage2: Optional[str] = None,
             mask: Optional[Sequence] = None, mask2: Optional[Sequence] = None):
        lt = lhs.transcriptome if mask is None else lhs.transcriptome[:, mask]
        rt = rhs.transcriptome if mask is None else rhs.transcriptome[:, mask]
        return CellType._pairwise_diff(lt, rt, affinity, linkage)

    def __repr__(self):
        return f'{self.region}.{self.id_num}'


@dataclass
class Region(Mergeable):
    cell_types: Optional[Dict[int, CellType]] = field(default_factory=dict)
    _transcriptome: Optional[np.array] = None

    @property
    def transcriptome(self) -> np.array:
        if self._transcriptome is None:
            raise ValueError(f'Transcriptome for region {self.id_num} never defined.')
        if len(self._transcriptome.shape) == 1:
            return self._transcriptome.reshape(1, -1)
        return self._transcriptome

    @property
    def child_transcriptomes(self) -> np.array:
        # ugly -- should refactor something
        ct_list = list(self.cell_types.values())
        transcriptome_length = ct_list[0].transcriptome.shape[1]
        transcriptomes = np.zeros((len(self.cell_types), transcriptome_length))
        for c in range(len(self.cell_types)):
            transcriptomes[c] = ct_list[c].transcriptome
        return transcriptomes

    @property
    def num_original(self):
        if self._transcriptome is None:
            return np.sum([ct.num_original for ct in self.cell_types.values()])
        else:
            return self.transcriptome.shape[0]

    @property
    def num_cell_types(self):
        return len(self.cell_types)

    @property
    def region(self):
        return self.id_num

    # @classmethod
    # def merge(cls, m1: Region, m2: Region, dist: float, new_id: int, region_id: Optional[int] = None) -> Region:
    #     pass

    # noinspection PyArgumentList
    @staticmethod
    def diff(lhs: Region, rhs: Region, affinity: Callable, linkage: str,
             affinity2: Optional[Callable] = None, linkage2: Optional[str] = None,
             mask: Optional[np.array] = None, mask2: Optional[Sequence] = None):
        """
        Compute the distance between two regions.
        :param lhs: The lhs region
        :param rhs: The rhs region
        :param affinity: Affinity for transcriptome comparisons for region distances
        :param linkage: Linkage for region distances
        :param affinity2: Affinity for transcriptome comparisons for cell types distances
        :param linkage2: Linkage for cell type distances
        :param mask: Region gene mask
        :param mask2: Cell type gene mask
        :return: dist, num_ct_diff
        """
        # Difference in number of cell types contained. Only really matters for homolog_mnn since it can change there
        num_ct_diff = np.abs(lhs.num_cell_types - rhs.num_cell_types)

        if (lhs._transcriptome is None) or (rhs._transcriptome is None):
            if (affinity2 is None) or (linkage2 is None):
                raise ValueError('Both affinity and linkage must be defined for cell types')
            # Cell type dists using cell type gene mask
            ct_dists = np.zeros((lhs.num_cell_types, rhs.num_cell_types))
            # Cell type dists using region gene mask
            r_ct_dists = np.zeros((lhs.num_cell_types, rhs.num_cell_types))
            r1_ct_list = list(lhs.cell_types.values())
            r2_ct_list = list(rhs.cell_types.values())
            for r1_idx, r2_idx in product(range(lhs.num_cell_types), range(rhs.num_cell_types)):
                # Use the cell type gene mask here because matching up sister cell types
                ct_dists[r1_idx, r2_idx] = CellType.diff(r1_ct_list[r1_idx], r2_ct_list[r2_idx],
                                                         affinity=affinity2, linkage=linkage2, mask=mask2)
                r_ct_dists[r1_idx, r2_idx] = CellType.diff(r1_ct_list[r1_idx], r2_ct_list[r2_idx],
                                                           affinity=affinity2, linkage=linkage2, mask=mask)

            if linkage == 'single':
                dist = r_ct_dists.min()
            elif linkage == 'complete':
                dist = r_ct_dists.max()
            elif linkage == 'homolog_avg':
                dists = []
                for i in range(np.min(ct_dists.shape)):
                    ct_min1_idx, ct_min2_idx = np.unravel_index(np.argmin(ct_dists), ct_dists.shape)
                    # Add the distance between the two closest cell types (can consider as homologs)
                    dists.append(r_ct_dists[ct_min1_idx, ct_min2_idx])
                    # Delete these two homologs from the distance matrix
                    ct_dists = np.delete(ct_dists, ct_min1_idx, axis=0)
                    ct_dists = np.delete(ct_dists, ct_min2_idx, axis=1)
                    r_ct_dists = np.delete(r_ct_dists, ct_min1_idx, axis=0)
                    r_ct_dists = np.delete(r_ct_dists, ct_min2_idx, axis=1)
                dist = np.mean(dists)
            elif linkage == 'homolog_mnn':
                dists = []
                # Nearest neighbors for the cell types from region 1
                r1_ct_nn = np.argmin(ct_dists, axis=1)
                # Nearest neighbors for the cell types from region 2
                r2_ct_nn = np.argmin(ct_dists, axis=0)
                # Only append distance if we find a mutual nearest neighbor
                for i in range(r1_ct_nn.shape[0]):
                    if r2_ct_nn[r1_ct_nn[i]] == i:
                        dists.append(r_ct_dists[i, r1_ct_nn[i]])
                num_ct_diff = lhs.num_cell_types + rhs.num_cell_types - (2 * len(dists))
                dist = np.mean(dists)
            else:  # default to 'average':
                dist = r_ct_dists.mean()
        else:
            dist = Region._pairwise_diff(lhs.transcriptome, rhs.transcriptome, affinity, linkage)
        return dist, num_ct_diff

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
