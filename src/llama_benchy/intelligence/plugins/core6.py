import json
import os
import re
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


def _lm_eval_console_filter(text: str) -> bool:
    # Keep stderr/info/error context visible, but suppress noisy progress bars.
    if not text.strip():
        return False
    if "Requesting API:" in text:
        return False
    if "Building contexts for " in text and " on rank " in text:
        return False
    if "Generating " in text and " split:" in text:
        return False
    if "%|" in text and "| " in text:
        return False
    return True


def _lm_eval_progress_line(text: str) -> str | None:
    if "Building contexts for " in text and " on rank " in text:
        match = re.search(r"Building contexts for ([^ ]+) on rank", text)
        task = match.group(1) if match else "task"
        return f"Building contexts: {task}"
    if "Running loglikelihood requests" in text:
        return "Running loglikelihood requests..."
    if "Requesting API:" in text:
        match = re.search(r"(\d+)\s*/\s*(\d+).+?([\d.]+it/s)", text)
        if not match:
            return "Requesting API..."
        done, total, rate = match.groups()
        return f"Requesting API {done}/{total} ({rate})"
    return None


class CoreTaskPlugin(IntelligencePlugin):
    def __init__(self, plugin_name: str, lm_eval_task: str, requires_loglikelihood: bool):
        self.name = plugin_name
        self.lm_eval_task = lm_eval_task
        self.requires_loglikelihood = requires_loglikelihood

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        output_path = os.path.join(artifacts_dir, f"{self.name}.json")
        model_backend = "local-completions"
        base_url = config.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            if model_backend == "local-completions":
                base_url = f"{base_url}/completions"
            else:
                base_url = f"{base_url}/chat/completions"
        model_args_parts = [
            f"model={config.served_model_name}",
            f"base_url={base_url}",
            f"api_key={config.api_key}",
        ]
        if config.tokenizer:
            model_args_parts.append(f"tokenizer={config.tokenizer}")
        if config.max_concurrent:
            model_args_parts.append(f"num_concurrent={config.max_concurrent}")
        model_args = ",".join(model_args_parts)
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
        # Keep lm-eval logs readable by suppressing noisy HF progress bars.
        env.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        try:
            started = time.perf_counter()
            print(f"[{self.name}] Preparing datasets (progress bars suppressed).")
            run_command_capture_stream(
                cmd,
                env=env,
                prefix=f"[{self.name}]",
                cwd=artifacts_dir,
                log_file=log_file,
                console_filter=_lm_eval_console_filter,
                progress_line=_lm_eval_progress_line,
            )
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

