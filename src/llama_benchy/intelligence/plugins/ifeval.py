import json
import os
import subprocess
import time
from typing import Any, Dict, List

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream


class IFEvalPlugin(IntelligencePlugin):
    name = "ifeval"

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

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        output_path = os.path.join(artifacts_dir, "ifeval.json")
        model_args = f"model={config.served_model_name},base_url={config.base_url},api_key={config.api_key}"
        cmd = [
            "lm_eval",
            "--model",
            "local-chat-completions",
            "--model_args",
            model_args,
            "--tasks",
            "ifeval",
            "--output_path",
            output_path,
        ]

        env = os.environ.copy()
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        try:
            started = time.perf_counter()
            run_command_capture_stream(cmd, env=env, prefix="[ifeval]", cwd=artifacts_dir, log_file=log_file)
            duration = time.perf_counter() - started
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            task_results = self._extract_tasks(payload)
            for task in task_results:
                task.duration_seconds = duration
                task.raw["artifact_output_path"] = output_path
                task.raw["log_file"] = log_file
            summary = None
            if task_results:
                summary = sum(t.value for t in task_results if t.value is not None) / len(task_results)
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=summary,
                artifacts={"output_path": output_path, "log_file": log_file},
                tasks=task_results,
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="lm_eval is not installed. Install intelligence extras to enable ifeval.",
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"output_path": output_path, "log_file": log_file},
                error=err,
            )
        except Exception as exc:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"output_path": output_path, "log_file": log_file},
                error=str(exc),
            )

