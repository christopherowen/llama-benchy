import json
import os
import subprocess
import time
from typing import Optional

from ...config import BenchmarkConfig
from ..base import IntelligencePlugin
from ..models import IntelligencePluginResult, IntelligenceTaskResult
from ..subprocess_utils import run_command_capture_stream


class SWEBenchVerifiedPlugin(IntelligencePlugin):
    name = "swebench_verified"

    @staticmethod
    def _build_predictions_filename(model_name: str) -> str:
        model_nickname = model_name.split("/")[-1]
        return f"{model_nickname}__SWE-bench_Verified__test.jsonl"

    @staticmethod
    def _build_report_filename(model_name: str, run_id: str) -> str:
        model_nickname = model_name.replace("/", "__")
        return f"{model_nickname}.{run_id}.json"

    @staticmethod
    def _extract_resolved_rate(report_payload: dict) -> Optional[float]:
        total = report_payload.get("total_instances")
        resolved = report_payload.get("resolved_instances")
        if not isinstance(total, int) or total <= 0:
            return None
        if not isinstance(resolved, int):
            return None
        return resolved / total

    def run(self, config: BenchmarkConfig, artifacts_dir: str, log_file: str) -> IntelligencePluginResult:
        if not config.allow_code_exec:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="swebench_verified may execute generated code. Re-run with --allow-code-exec to enable this plugin.",
            )

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", config.api_key)
        env.setdefault("OPENAI_BASE_URL", config.base_url)
        if config.dataset_cache_dir:
            env["HF_HOME"] = config.dataset_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.dataset_cache_dir, "datasets")

        predictions_dir = os.path.join(artifacts_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        run_id = f"llama_benchy_{int(time.time())}"

        inference_bootstrap = (
            "import swebench.inference.run_api as api;"
            f"m={config.served_model_name!r};"
            "api.MODEL_LIMITS.setdefault(m, 128000);"
            "api.MODEL_COST_PER_INPUT.setdefault(m, 0.0);"
            "api.MODEL_COST_PER_OUTPUT.setdefault(m, 0.0);"
            f"api.main(dataset_name_or_path='SWE-bench/SWE-bench_Verified', split='test', model_name_or_path=m, shard_id=None, num_shards=None, output_dir={predictions_dir!r}, model_args='temperature=0.2,top_p=0.95', max_cost=None)"
        )

        inference_cmd = ["python", "-c", inference_bootstrap]
        predictions_path = os.path.join(
            predictions_dir, self._build_predictions_filename(config.served_model_name)
        )
        eval_cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "SWE-bench/SWE-bench_Verified",
            "--split",
            "test",
            "--predictions_path",
            predictions_path,
            "--max_workers",
            "1",
            "--run_id",
            run_id,
        ]
        report_path = os.path.join(
            artifacts_dir, self._build_report_filename(config.served_model_name, run_id)
        )

        try:
            started = time.perf_counter()
            run_command_capture_stream(
                inference_cmd,
                env=env,
                prefix="[swebench_verified:inference]",
                cwd=artifacts_dir,
                log_file=log_file,
            )
            run_command_capture_stream(
                eval_cmd,
                env=env,
                prefix="[swebench_verified:evaluation]",
                cwd=artifacts_dir,
                log_file=log_file,
            )
            duration = time.perf_counter() - started

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            resolved_rate = self._extract_resolved_rate(report)
            task = IntelligenceTaskResult(
                name="swebench_verified",
                metric="resolved_rate",
                value=resolved_rate,
                duration_seconds=duration,
                raw={
                    "report_path": report_path,
                    "predictions_path": predictions_path,
                    "log_file": log_file,
                    **report,
                },
            )
            return IntelligencePluginResult(
                plugin=self.name,
                success=True,
                summary_metric=resolved_rate,
                artifacts={
                    "predictions_path": predictions_path,
                    "report_path": report_path,
                    "log_file": log_file,
                },
                tasks=[task],
            )
        except FileNotFoundError:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={"log_file": log_file},
                error="swebench is not installed. Install intelligence extras to enable swebench_verified.",
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={
                    "predictions_path": predictions_path,
                    "report_path": report_path,
                    "log_file": log_file,
                },
                error=err,
            )
        except Exception as exc:
            return IntelligencePluginResult(
                plugin=self.name,
                success=False,
                artifacts={
                    "predictions_path": predictions_path,
                    "report_path": report_path,
                    "log_file": log_file,
                },
                error=str(exc),
            )
