import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult


class Core6Plugin(IntelligencePlugin):
    name = "core6"

    def _extract_tasks(self, payload: Dict[str, Any]) -> List[IntelligenceTaskResult]:
        results = payload.get("results", {})
        tasks: List[IntelligenceTaskResult] = []
        for task_name, metrics in results.items():
            scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if not scalar_metrics:
                continue
            metric_name = next(iter(scalar_metrics.keys()))
            tasks.append(
                IntelligenceTaskResult(
                    name=task_name,
                    metric=metric_name,
                    value=float(scalar_metrics[metric_name]),
                    raw=metrics,
                )
            )
        return tasks

    def run(self, config: BenchmarkConfig) -> IntelligencePluginResult:
        tasks = "mmlu,arc_challenge,hellaswag,winogrande,gsm8k,truthfulqa_mc2"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "core6.json")
            model_args = f"model={config.served_model_name},base_url={config.base_url},api_key={config.api_key}"
            cmd = [
                "lm_eval",
                "--model",
                "local-chat-completions",
                "--model_args",
                model_args,
                "--tasks",
                tasks,
                "--output_path",
                output_path,
            ]

            env = os.environ.copy()
            if config.dataset_cache_dir:
                env["HF_HOME"] = config.dataset_cache_dir
                env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

            try:
                subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
                with open(output_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                task_results = self._extract_tasks(payload)
                summary = None
                if task_results:
                    summary = sum(t.value for t in task_results if t.value is not None) / len(task_results)
                return IntelligencePluginResult(
                    plugin=self.name,
                    success=True,
                    summary_metric=summary,
                    tasks=task_results,
                )
            except FileNotFoundError:
                return IntelligencePluginResult(
                    plugin=self.name,
                    success=False,
                    error="lm_eval is not installed. Install intelligence extras to enable core6.",
                )
            except subprocess.CalledProcessError as exc:
                err = exc.stderr.strip() if exc.stderr else str(exc)
                return IntelligencePluginResult(plugin=self.name, success=False, error=err)
            except Exception as exc:
                return IntelligencePluginResult(plugin=self.name, success=False, error=str(exc))

