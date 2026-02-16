import os
import re
import subprocess
import time
import importlib
from typing import Optional

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream


class LiveCodeBenchPlugin(IntelligencePlugin):
    name = "livecodebench"

    @staticmethod
    def _extract_pass_at_1(stdout: str) -> Optional[float]:
        match = re.search(r"pass@1[^0-9]*([0-9]*\.?[0-9]+)", stdout, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        if not config.allow_code_exec:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="livecodebench may execute generated code. Re-run with --allow-code-exec to enable this plugin.",
            )

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", config.api_key)
        env.setdefault("OPENAI_KEY", config.api_key)
        env.setdefault("OPENAI_BASE_URL", config.base_url)
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        try:
            importlib.import_module("lcb_runner.runner.main")
        except Exception:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"artifact_dir": artifacts_dir, "log_file": log_file},
                error=(
                    "Installed LiveCodeBench wheel is incomplete (missing lcb_runner.runner.main). "
                    "Install a LiveCodeBench build that includes runner modules."
                ),
            )

        model_repr = config.served_model_name.replace("/", "_")
        bootstrap = (
            "from datetime import datetime;"
            "from lcb_runner.lm_styles import LanguageModel,LMStyle,LanguageModelStore;"
            f"m={config.served_model_name!r};"
            f"r={model_repr!r};"
            "LanguageModelStore.setdefault(m, LanguageModel(m, r, LMStyle.OpenAIChat, datetime(2025,1,1)));"
            "import lcb_runner.runner.main as main;"
            "main.main()"
        )
        cmd = [
            "python",
            "-c",
            bootstrap,
            "--model",
            config.served_model_name,
            "--scenario",
            "codegeneration",
            "--evaluate",
            "--release_version",
            "release_latest",
        ]
        try:
            started = time.perf_counter()
            completed = run_command_capture_stream(
                cmd,
                env=env,
                prefix="[livecodebench]",
                cwd=artifacts_dir,
                log_file=log_file,
            )
            duration = time.perf_counter() - started
            value = self._extract_pass_at_1(completed.stdout)
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=value,
                artifacts={"artifact_dir": artifacts_dir, "log_file": log_file},
                tasks=[
                    IntelligenceTaskResult(
                        name="livecodebench_codegeneration",
                        metric="pass@1",
                        value=value,
                        duration_seconds=duration,
                        raw={
                            "stdout": completed.stdout,
                            "artifact_dir": artifacts_dir,
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
                error="LiveCodeBench is not installed. Install intelligence extras or lcb_runner package to enable livecodebench.",
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

