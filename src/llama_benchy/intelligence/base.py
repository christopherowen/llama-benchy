from abc import ABC, abstractmethod

from ..config import BenchmarkConfig
from .models import IntelligencePluginResult


class IntelligencePlugin(ABC):
    name: str

    @abstractmethod
    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        raise NotImplementedError

