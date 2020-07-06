from dataclasses import dataclass, field
from typing import Union, Dict
from abc import ABC, abstractmethod
import functools
import numpy as np


@dataclass
class Node(ABC):
    id_num: int

    @property
    @abstractmethod
    def num_original(self):
        pass

    @property
    @abstractmethod
    def region(self):
        pass


@dataclass
class CellType(Node):
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

    @property
    def num_cell_types(self):
        return len(self.cell_types)

    @property
    def region(self):
        return self.id_num

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
