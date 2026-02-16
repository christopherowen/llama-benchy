import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream


class _TerminalBenchBasePlugin(IntelligencePlugin):
    tb_agent: str

    @staticmethod
    def _read_accuracy(results_path: Path) -> Optional[float]:
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        value = payload.get("accuracy")
        if isinstance(value, (float, int)):
            return float(value)
        return None

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        if not config.allow_code_exec:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error=f"{self.name} may execute generated code. Re-run with --allow-code-exec to enable this plugin.",
            )

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", config.api_key)
        env.setdefault("OPENAI_BASE_URL", config.base_url)
        env.setdefault("OPENAI_API_BASE", config.base_url)
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        dataset = env.get("TB_DATASET", "terminal-bench-core==0.1.1")
        run_id = f"{self.name}_{int(time.time())}"
        n_concurrent = env.get("TB_N_CONCURRENT", "1")

        cmd = [
            "tb",
            "run",
            "--agent",
            self.tb_agent,
            "--model",
            config.served_model_name,
            "--dataset",
            dataset,
            "--run-id",
            run_id,
            "--output-path",
            artifacts_dir,
            "--n-concurrent",
            n_concurrent,
            "--no-upload-results",
        ]

        task_id = env.get("TB_TASK_ID")
        if task_id:
            cmd.extend(["--task-id", task_id])
        n_tasks = env.get("TB_N_TASKS")
        if n_tasks:
            cmd.extend(["--n-tasks", n_tasks])

        results_path = Path(artifacts_dir) / run_id / "results.json"
        summary_cmd = [
            "tb",
            "runs",
            "summarize",
            "--run-id",
            run_id,
            "--runs-dir",
            artifacts_dir,
        ]

        try:
            started = time.perf_counter()
            run_command_capture_stream(
                cmd,
                env=env,
                prefix=f"[{self.name}]",
                cwd=artifacts_dir,
                log_file=log_file,
            )
            run_command_capture_stream(
                summary_cmd,
                env=env,
                prefix=f"[{self.name}:summary]",
                cwd=artifacts_dir,
                log_file=log_file,
            )
            duration = time.perf_counter() - started
            accuracy = self._read_accuracy(results_path)
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=accuracy,
                artifacts={
                    "artifact_dir": artifacts_dir,
                    "results_path": str(results_path),
                    "log_file": log_file,
                },
                tasks=[
                    IntelligenceTaskResult(
                        name=self.name,
                        metric="accuracy",
                        value=accuracy,
                        duration_seconds=duration,
                        raw={
                            "results_path": str(results_path),
                            "run_id": run_id,
                            "dataset": dataset,
                            "log_file": log_file,
                        },
                    )
                ],
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="terminal-bench is not installed. Install intelligence extras to enable this plugin.",
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"artifact_dir": artifacts_dir, "log_file": log_file},
                error=err,
            )
        except Exception as exc:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"artifact_dir": artifacts_dir, "log_file": log_file},
                error=str(exc),
            )


class TerminalBenchPlugin(_TerminalBenchBasePlugin):
    name = "terminal_bench"
    tb_agent = "terminus"


class AiderPlugin(_TerminalBenchBasePlugin):
    name = "aider"
    tb_agent = "aider"
