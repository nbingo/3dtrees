from dataclasses import dataclass
from typing import List, Callable, Union
from queue import PriorityQueue
from data_utils import get_region
import functools
import numpy as np
import pandas as pd


@dataclass
class CellType:
    id_num: int
    region: int
    transcriptome: np.array = None
    num_children: int = 0


@dataclass
class Region:
    id_num: int
    cell_types: List[CellType] = None

    @property
    def transcriptomes(self) -> np.array:
        transcriptomes = np.zeros((len(self.cell_types), self.cell_types[0].transcriptome.shape[-1]))
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


class Agglomerate3D:
    def __init__(self, cell_type_affinity: Callable, region_affinity: Callable, linkage_cell: str, linkage_region: str):
        self.cell_type_affinity = cell_type_affinity
        self.region_affinity = region_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.linkage_mat = pd.DataFrame({'Is region': [], 'ID1': [], 'ID2': [], 'Distance': [], 'Num children': []})

    def agglomerate(self, data: pd.DataFrame) -> pd.DataFrame:
        cell_types = []
        ct_dists = PriorityQueue()
        r_dists = PriorityQueue()

        ct_names = data.index.values
        ct_regions = np.vectorize(get_region)(ct_names)
        r_names = np.unique(ct_regions)
        region_to_id = {i: r_names[i] for i in range(r_names.shape[0])}

        regions = {r: Region(r) for r in range(len(r_names))}
