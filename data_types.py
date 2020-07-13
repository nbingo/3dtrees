from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
import functools
import numpy as np
import pandas as pd


class Node:
    def __init__(self, id_num: int, parent=None, left=None, right=None):
        self.id_num = id_num
        self.parent = parent
        self.left = left
        self.right = right

    def __repr__(self):
        parent_id = self.parent.id_num if self.parent is not None else None
        left_id = self.left.id_num if self.left is not None else None
        right_id = self.right.id_num if self.right is not None else None
        return f'Node(id_num={self.id_num}, parent={parent_id}, left={left_id}, right={right_id})'

    @classmethod
    def tree_from_link_mat(cls, link_mat: pd.DataFrame):
        ct_link_mat = link_mat.loc[link_mat['Is region'] == False]

        def recursive_tree_builder(id_num: int, parent):
            row = ct_link_mat.loc[ct_link_mat['new ID'] == id_num]
            if row.empty:
                return None
            left_id = row.iloc[0]['ID1']
            right_id = row.iloc[0]['ID2']
            n = cls(id_num, parent)
            n.left = recursive_tree_builder(left_id, n)
            n.right = recursive_tree_builder(right_id, n)
            return n

        root_id = ct_link_mat.iloc[-1]['new ID']
        return recursive_tree_builder(root_id, None)


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

    def __repr__(self):
        return f'{self.region}.{self.id_num}'


@dataclass
class Region(Mergeable):
    cell_types: Optional[Dict[int, CellType]] = field(default_factory=dict)

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
