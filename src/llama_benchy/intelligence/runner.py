import time
import os

from ..config import BenchmarkConfig
from .models import IntelligenceReport
from .registry import resolve_plugins


class IntelligenceRunner:
    def __init__(self, config: BenchmarkConfig, run_dir: str):
        self.config = config
        self.run_dir = run_dir

    def run(self) -> IntelligenceReport:
        plugins = resolve_plugins(self.config.intelligence_plugins)
        report = IntelligenceReport()
        total = len(plugins)
        artifacts_root = os.path.join(self.run_dir, "artifacts")
        logs_root = os.path.join(self.run_dir, "logs")
        os.makedirs(artifacts_root, exist_ok=True)
        os.makedirs(logs_root, exist_ok=True)
        for idx, plugin in enumerate(plugins, start=1):
            plugin_artifacts_dir = os.path.join(artifacts_root, plugin.name)
            plugin_log = os.path.join(logs_root, f"plugin-{plugin.name}.log")
            os.makedirs(plugin_artifacts_dir, exist_ok=True)
            print(f"[intelligence] Starting plugin {plugin.name} ({idx}/{total})...")
            print(f"[intelligence]   artifacts: {plugin_artifacts_dir}")
            print(f"[intelligence]   log: {plugin_log}")
            started = time.perf_counter()
            result = plugin.run(self.config, plugin_artifacts_dir, plugin_log)
            result.duration_seconds = time.perf_counter() - started
            report.plugins.append(result)
            if result.success:
                print(f"[intelligence] Finished plugin {plugin.name} ({idx}/{total}) in {result.duration_seconds:.2f}s")
            else:
                print(f"[intelligence] Plugin {plugin.name} failed ({idx}/{total}) after {result.duration_seconds:.2f}s: {result.error}")
        return report

