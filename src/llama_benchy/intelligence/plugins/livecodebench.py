import os
import re
import subprocess
import time
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

    def run(self, config: BenchmarkConfig) -> IntelligencePluginResult:
        if not config.allow_code_exec:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                error="livecodebench may execute generated code. Re-run with --allow-code-exec to enable this plugin.",
            )

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", config.api_key)
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        cmd = [
            "python",
            "-m",
            "lcb_runner.runner.main",
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
            completed = run_command_capture_stream(cmd, env=env, prefix="[livecodebench]")
            duration = time.perf_counter() - started
            value = self._extract_pass_at_1(completed.stdout)
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=value,
                tasks=[
                    IntelligenceTaskResult(
                        name="livecodebench_codegeneration",
                        metric="pass@1",
                        value=value,
                        duration_seconds=duration,
                        raw={"stdout": completed.stdout},
                    )
                ],
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                error="LiveCodeBench is not installed. Install intelligence extras or lcb_runner package to enable livecodebench.",
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            return IntelligencePluginResult(plugin=self.name, success=False, error=err)
        except Exception as exc:
            return IntelligencePluginResult(plugin=self.name, success=False, error=str(exc))

