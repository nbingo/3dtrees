from dataclasses import dataclass
from typing import List, Callable
import numpy as np
import pandas as pd


@dataclass
class CellType:
    transcriptome: np.array
    region: str
    id_num: int
    num_children: int = 0


@dataclass
class Region:
    name: str
    cell_types: List[CellType]
    id_num: int

    def get_transcriptomes(self) -> np.array:
        transcriptomes = np.zeros((len(self.cell_types), self.cell_types[0].transcriptome.shape[0]))
        for c in range(len(self.cell_types)):
            transcriptomes[c] = self.cell_types[c].transcriptome
        return transcriptomes


class Agglomerate3D:
    def __init__(self, cell_type_affinity: Callable, region_affinity: Callable, linkage_cell: str, linkage_region: str):
        self.cell_type_affinity = cell_type_affinity
        self.region_affinity = region_affinity
        self.linkage_cell = linkage_cell
        self.linkage_region = linkage_region
        self.linkage_mat = pd.DataFrame({'Is region': [], 'ID1': [], 'ID2': [], 'Distance': [], 'Num children': []})

    def agglomerate(self, data: pd.DataFrame) -> pd.DataFrame:
        cell_types = []
        regions = []

