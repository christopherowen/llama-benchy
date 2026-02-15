from ..config import BenchmarkConfig
from .models import IntelligenceReport
from .registry import resolve_plugins


class IntelligenceRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self) -> IntelligenceReport:
        plugins = resolve_plugins(self.config.intelligence_plugins)
        report = IntelligenceReport()
        for plugin in plugins:
            report.plugins.append(plugin.run(self.config))
        return report

