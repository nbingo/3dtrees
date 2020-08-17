from typing import List, Optional
from abc import abstractmethod
from collections.abc import Sequence
import numpy as np


class DataLoader(Sequence):

    def __init__(self, ct_axis_mask: Optional[np.array] = None, r_axis_mask: Optional[np.array] = None):
        self.ct_axis_mask = ct_axis_mask
        self.r_axis_mask = r_axis_mask

    @abstractmethod
    def get_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_corresponding_region_names(self) -> List[str]:
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass
