import json
import os
import subprocess
import time
from typing import Any, Dict

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream

CORE6_TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "mmlu": {"task": "mmlu", "requires_loglikelihood": True},
    "arc-c": {"task": "arc_challenge", "requires_loglikelihood": True},
    "hellaswag": {"task": "hellaswag", "requires_loglikelihood": True},
    "winogrande": {"task": "winogrande", "requires_loglikelihood": True},
    "gsm8k": {"task": "gsm8k", "requires_loglikelihood": False},
    "truthfulqa": {"task": "truthfulqa_mc2", "requires_loglikelihood": True},
}
CORE6_PLUGIN_NAMES = list(CORE6_TASK_SPECS.keys())


class CoreTaskPlugin(IntelligencePlugin):
    def __init__(self, plugin_name: str, lm_eval_task: str, requires_loglikelihood: bool):
        self.name = plugin_name
        self.lm_eval_task = lm_eval_task
        self.requires_loglikelihood = requires_loglikelihood

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        output_path = os.path.join(artifacts_dir, f"{self.name}.json")
        model_args = f"model={config.served_model_name},base_url={config.base_url},api_key={config.api_key}"
        model_backend = "local-completions" if self.requires_loglikelihood else "local-chat-completions"
        cmd = [
            "lm_eval",
            "--model",
            model_backend,
            "--model_args",
            model_args,
            "--tasks",
            self.lm_eval_task,
            "--output_path",
            output_path,
        ]

        env = os.environ.copy()
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        try:
            started = time.perf_counter()
            run_command_capture_stream(cmd, env=env, prefix=f"[{self.name}]", cwd=artifacts_dir, log_file=log_file)
            duration = time.perf_counter() - started
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            metrics: Dict[str, Any] = payload.get("results", {}).get(self.lm_eval_task, {})
            scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            metric_name = next(iter(scalar_metrics.keys()), "score")
            metric_value = float(scalar_metrics[metric_name]) if scalar_metrics else None
            task_result = IntelligenceTaskResult(
                name=self.name,
                metric=metric_name,
                value=metric_value,
                duration_seconds=duration,
                raw={
                    **metrics,
                    "artifact_output_path": output_path,
                    "log_file": log_file,
                },
            )
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=metric_value,
                artifacts={"output_path": output_path, "log_file": log_file},
                tasks=[task_result],
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error=f"lm_eval is not installed. Install intelligence extras to enable {self.name}.",
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


def get_core6_plugins() -> Dict[str, IntelligencePlugin]:
    return {
        name: CoreTaskPlugin(name, spec["task"], bool(spec["requires_loglikelihood"]))
        for name, spec in CORE6_TASK_SPECS.items()
    }

