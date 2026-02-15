import os
import re
import subprocess
from typing import Optional

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult


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

    def _run_dataset(self, dataset: str, config: BenchmarkConfig, env: dict[str, str]) -> IntelligenceTaskResult:
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
        completed = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        value = self._extract_pass_at_1(completed.stdout)
        return IntelligenceTaskResult(
            name=f"{dataset}+",
            metric="pass@1",
            value=value,
            raw={"stdout": completed.stdout},
        )

    def run(self, config: BenchmarkConfig) -> IntelligencePluginResult:
        if not config.allow_code_exec:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                error="evalplus requires code execution. Re-run with --allow-code-exec to enable this plugin.",
            )

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", config.api_key)
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        try:
            humaneval = self._run_dataset("humaneval", config, env)
            mbpp = self._run_dataset("mbpp", config, env)
            values = [v for v in [humaneval.value, mbpp.value] if v is not None]
            summary = sum(values) / len(values) if values else None
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=summary,
                tasks=[humaneval, mbpp],
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                error="evalplus is not installed. Install intelligence extras to enable evalplus.",
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            return IntelligencePluginResult(plugin=self.name, success=False, error=err)
        except Exception as exc:
            return IntelligencePluginResult(plugin=self.name, success=False, error=str(exc))

