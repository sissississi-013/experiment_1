from abc import ABC, abstractmethod
from validation_pipeline.schemas.dataset import DatasetPlan


class BaseDownloader(ABC):
    @abstractmethod
    def download(self, plan: DatasetPlan) -> str:
        """Download dataset and return local path to image directory."""
        pass
