import os
import re
import subprocess
import time
from typing import Dict, Optional

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream


class EvalPlusPlugin(IntelligencePlugin):
    name = "evalplus"

    @staticmethod
    def _extract_pass_at_1(stdout: str) -> Optional[float]:
        match = re.search(r"pass@1[^0-9]*([0-9]*\.?[0-9]+)", stdout, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_saved_paths(stdout: str, artifacts_dir: str) -> Dict[str, str]:
        saved_paths: Dict[str, str] = {}
        for line in stdout.splitlines():
            match = re.search(r"saved to\s+(.+)$", line.strip(), flags=re.IGNORECASE)
            if not match:
                continue
            rel_path = match.group(1).strip()
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(artifacts_dir, rel_path)
            key = "raw_output_path" if ".raw.jsonl" in rel_path else "output_path"
            saved_paths[key] = abs_path
        return saved_paths

    def _run_dataset(
        self,
        dataset: str,
        config: BenchmarkConfig,
        env: dict[str, str],
        artifacts_dir: str,
        log_file: str,
    ) -> IntelligenceTaskResult:
        cmd = [
            "evalplus.evaluate",
            "--model",
            config.served_model_name,
            "--dataset",
            dataset,
            "--backend",
            "openai",
            "--base-url",
            config.base_url,
            "--greedy",
        ]
        started = time.perf_counter()
        completed = run_command_capture_stream(
            cmd,
            env=env,
            prefix=f"[evalplus:{dataset}]",
            cwd=artifacts_dir,
            log_file=log_file,
        )
        duration = time.perf_counter() - started
        value = self._extract_pass_at_1(completed.stdout)
        saved_paths = self._extract_saved_paths(completed.stdout, artifacts_dir)
        return IntelligenceTaskResult(
            name=f"{dataset}+",
            metric="pass@1",
            value=value,
            duration_seconds=duration,
            raw={"stdout": completed.stdout, "log_file": log_file, **saved_paths},
        )

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        if not config.allow_code_exec:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="evalplus requires code execution. Re-run with --allow-code-exec to enable this plugin.",
            )

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", config.api_key)
        # EvalPlus defaults to 4GB; use a safer default for mixed model+eval workloads.
        # Users can override by explicitly setting EVALPLUS_MAX_MEMORY_BYTES.
        env.setdefault("EVALPLUS_MAX_MEMORY_BYTES", str(1024 * 1024 * 1024))
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        try:
            humaneval = self._run_dataset("humaneval", config, env, artifacts_dir, log_file)
            mbpp = self._run_dataset("mbpp", config, env, artifacts_dir, log_file)
            values = [v for v in [humaneval.value, mbpp.value] if v is not None]
            summary = sum(values) / len(values) if values else None
            artifacts = {"log_file": log_file}
            for idx, task in enumerate([humaneval, mbpp]):
                if "output_path" in task.raw:
                    artifacts[f"{idx}_output_path"] = str(task.raw["output_path"])
                if "raw_output_path" in task.raw:
                    artifacts[f"{idx}_raw_output_path"] = str(task.raw["raw_output_path"])
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=summary,
                artifacts=artifacts,
                tasks=[humaneval, mbpp],
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="evalplus is not installed. Install intelligence extras to enable evalplus.",
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error=err,
            )
        except Exception as exc:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error=str(exc),
            )

