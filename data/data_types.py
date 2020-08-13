from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Callable
from abc import ABC, abstractmethod
from itertools import product
import functools
import numpy as np


@dataclass
class Mergeable(ABC):
    id_num: int

    @property
    @abstractmethod
    def num_original(self):
        pass

    @property
    @abstractmethod
    def region(self):
        pass

    @property
    @abstractmethod
    def transcriptome(self):
        pass

    # noinspection PyArgumentList
    @staticmethod
    def _pairwise_diff(lhs_transcriptome: np.array, rhs_transcriptome: np.array, affinity: Callable, linkage: str):
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
             affinity2: Optional[Callable] = None, linkage2: Optional[str] = None):
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

    @staticmethod
    def diff(lhs: CellType, rhs: CellType, affinity: Callable, linkage: str,
             affinity2: Optional[Callable] = None, linkage2: Optional[str] = None):
        return CellType._pairwise_diff(lhs.transcriptome, rhs.transcriptome, affinity, linkage)

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

    # noinspection PyArgumentList
    @staticmethod
    def diff(lhs: Region, rhs: Region, affinity: Callable, linkage: str,
             affinity2: Optional[Callable] = None, linkage2: Optional[str] = None):
        if (lhs._transcriptome is None) or (rhs._transcriptome is None):
            if (affinity2 is None) or (linkage2 is None):
                raise TypeError('Both affinity and linkage must be defined for cell types')
            ct_dists = np.zeros((lhs.num_cell_types, rhs.num_cell_types))
            r1_ct_list = list(lhs.cell_types.values())
            r2_ct_list = list(rhs.cell_types.values())
            for r1_idx, r2_idx in product(range(lhs.num_cell_types), range(rhs.num_cell_types)):
                ct_dists[r1_idx, r2_idx] = CellType.diff(r1_ct_list[r1_idx], r2_ct_list[r2_idx],
                                                         affinity=affinity2, linkage=linkage2)

            if linkage == 'single':
                dist = ct_dists.min()
            elif linkage == 'complete':
                dist = ct_dists.max()
            elif linkage == 'homolog_avg':
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

            return dist
        else:
            return Region._pairwise_diff(lhs.transcriptome, rhs.transcriptome, affinity, linkage)

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
