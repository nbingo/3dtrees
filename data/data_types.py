from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Callable, ClassVar
from abc import ABC, abstractmethod
from itertools import product
import functools
import numpy as np


@dataclass
class Mergeable(ABC):
    id_num: int
    affinity: ClassVar[Callable]
    linkage: ClassVar[str]

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

    @classmethod
    @abstractmethod
    def diff(cls, lhs: Mergeable, rhs: Mergeable):
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

    # noinspection PyArgumentList
    @classmethod
    def diff(cls, lhs: CellType, rhs: CellType):
        dists = np.zeros((lhs.num_original, rhs.num_original))
        # Compute distance matrix
        # essentially only useful if this is working on merged cell types
        # otherwise just produces a matrix containing one value
        for ct1_idx, ct2_idx in product(range(lhs.num_original), range(rhs.num_original)):
            dists[ct1_idx, ct2_idx] = cls.affinity(lhs.transcriptome[ct1_idx], rhs.transcriptome[ct2_idx])
        if cls.linkage == 'single':
            dist = dists.min()
        elif cls.linkage == 'complete':
            dist = dists.max()
        else:  # default to 'average'
            dist = dists.mean()

        return np.float64(dist)

    def __repr__(self):
        return f'{self.region}.{self.id_num}'


@dataclass
class Region(Mergeable):
    cell_types: Optional[Dict[int, CellType]] = field(default_factory=dict)

    @property
    def transcriptome(self) -> np.array:
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

    @property
    def num_cell_types(self):
        return len(self.cell_types)

    @property
    def region(self):
        return self.id_num

    # noinspection PyArgumentList
    @classmethod
    def diff(cls, lhs: Region, rhs: Region):
        ct_dists = np.zeros((lhs.num_cell_types, rhs.num_cell_types))
        r1_ct_list = list(lhs.cell_types.values())
        r2_ct_list = list(rhs.cell_types.values())
        for r1_idx, r2_idx in product(range(lhs.num_cell_types), range(rhs.num_cell_types)):
            ct_dists[r1_idx, r2_idx] = CellType.diff(r1_ct_list[r1_idx], r2_ct_list[r2_idx])

        if cls.linkage == 'single':
            dist = ct_dists.min()
        elif cls.linkage == 'complete':
            dist = ct_dists.max()
        elif cls.linkage == 'homolog_avg':
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
