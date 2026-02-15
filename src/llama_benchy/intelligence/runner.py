import time

from ..config import BenchmarkConfig
from .models import IntelligenceReport
from .registry import resolve_plugins


class IntelligenceRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self) -> IntelligenceReport:
        plugins = resolve_plugins(self.config.intelligence_plugins)
        report = IntelligenceReport()
        total = len(plugins)
        for idx, plugin in enumerate(plugins, start=1):
            print(f"[intelligence] Starting plugin {plugin.name} ({idx}/{total})...")
            started = time.perf_counter()
            result = plugin.run(self.config)
            result.duration_seconds = time.perf_counter() - started
            report.plugins.append(result)
            if result.success:
                print(f"[intelligence] Finished plugin {plugin.name} ({idx}/{total}) in {result.duration_seconds:.2f}s")
            else:
                print(f"[intelligence] Plugin {plugin.name} failed ({idx}/{total}) after {result.duration_seconds:.2f}s: {result.error}")
        return report

