import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream


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
    def _format_eta(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

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
        done_i = int(done)
        total_i = int(total)
        rate_f = float(rate.replace("it/s", ""))
        remaining = max(0, total_i - done_i)
        eta = _format_eta(remaining / rate_f) if rate_f > 0 else "unknown"
        return f"Requesting API {done_i}/{total_i} ({rate}, ETA {eta})"
    return None


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
        base_url = config.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = f"{base_url}/completions"
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
            "local-completions",
            "--model_args",
            model_args,
            "--tasks",
            "ifeval",
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
            print("[ifeval] Preparing datasets (progress bars suppressed).")
            run_command_capture_stream(
                cmd,
                env=env,
                prefix="[ifeval]",
                cwd=artifacts_dir,
                log_file=log_file,
                console_filter=_lm_eval_console_filter,
                progress_line=_lm_eval_progress_line,
            )
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

