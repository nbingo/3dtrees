from dataclasses import dataclass
import numpy as np


@dataclass
class CellType:
    transcriptome: np.array
    region: str
