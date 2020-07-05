"""
Wrappers of cloud vendors APIs
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple


class LabelsGatherer(ABC):

    @abstractmethod
    def get_labels(self, image: Path, n: int) -> Tuple[List[str], List[float]]:
        """Get the top n tags from an image

        Args:
            image: route to the image to be labelled
            n: number of labels to retrieve

        Returns:
            labels: list with the top n labels
            scores: list with the top n scores
        """
        raise NotImplementedError
