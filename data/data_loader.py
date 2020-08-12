from typing import List
from abc import abstractmethod
from collections.abc import Sequence


class DataLoader(Sequence):

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
