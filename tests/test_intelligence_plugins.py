import asyncio
import json
import tempfile

import pytest

from llama_benchy.config import BenchmarkConfig
from llama_benchy.intelligence.models import IntelligencePluginResult, IntelligenceReport, IntelligenceTaskResult
from llama_benchy.intelligence.plugins.evalplus import EvalPlusPlugin
from llama_benchy.intelligence.plugins.swebench_verified import SWEBenchVerifiedPlugin
from llama_benchy.intelligence.plugins.terminal_bench import AiderPlugin, TerminalBenchPlugin
from llama_benchy.intelligence.registry import resolve_plugins
from llama_benchy.results import BenchmarkResults


def _base_config(**overrides):
    data = {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model": "test-model",
        "served_model_name": "test-model",
        "tokenizer": None,
        "pp_counts": [16],
        "tg_counts": [8],
        "depths": [0],
        "num_runs": 1,
        "no_cache": False,
        "latency_mode": "none",
        "no_warmup": True,
        "adapt_prompt": False,
        "enable_prefix_caching": False,
        "book_url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "post_run_cmd": None,
        "concurrency_levels": [1],
        "save_result": None,
        "output_dir": None,
        "result_format": "json",
        "save_total_throughput_timeseries": False,
        "save_all_throughput_timeseries": False,
        "enable_intelligence": True,
        "intelligence_plugins": ["core6"],
        "allow_code_exec": False,
        "dataset_cache_dir": None,
    }
    data.update(overrides)
    return BenchmarkConfig(**data)


def test_registry_rejects_unknown_plugin():
    with pytest.raises(ValueError):
        resolve_plugins(["does-not-exist"])


def test_evalplus_requires_allow_code_exec():
    plugin = EvalPlusPlugin()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = plugin.run(
            _base_config(intelligence_plugins=["evalplus"], allow_code_exec=False),
            artifacts_dir=tmpdir,
            log_file=f"{tmpdir}/evalplus.log",
        )
    assert result.success is False
    assert result.error is not None
    assert "--allow-code-exec" in result.error


def test_swebench_verified_requires_allow_code_exec():
    plugin = SWEBenchVerifiedPlugin()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = plugin.run(
            _base_config(intelligence_plugins=["swebench_verified"], allow_code_exec=False),
            artifacts_dir=tmpdir,
            log_file=f"{tmpdir}/swebench_verified.log",
        )
    assert result.success is False
    assert result.error is not None
    assert "--allow-code-exec" in result.error


def test_registry_resolves_swebench_verified():
    plugins = resolve_plugins(["swebench_verified"])
    assert len(plugins) == 1
    assert plugins[0].name == "swebench_verified"


def test_terminal_bench_requires_allow_code_exec():
    plugin = TerminalBenchPlugin()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = plugin.run(
            _base_config(intelligence_plugins=["terminal_bench"], allow_code_exec=False),
            artifacts_dir=tmpdir,
            log_file=f"{tmpdir}/terminal_bench.log",
        )
    assert result.success is False
    assert result.error is not None
    assert "--allow-code-exec" in result.error


def test_aider_requires_allow_code_exec():
    plugin = AiderPlugin()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = plugin.run(
            _base_config(intelligence_plugins=["aider"], allow_code_exec=False),
            artifacts_dir=tmpdir,
            log_file=f"{tmpdir}/aider.log",
        )
    assert result.success is False
    assert result.error is not None
    assert "--allow-code-exec" in result.error


def test_registry_resolves_terminal_bench():
    plugins = resolve_plugins(["terminal_bench"])
    assert len(plugins) == 1
    assert plugins[0].name == "terminal_bench"


def test_registry_resolves_aider():
    plugins = resolve_plugins(["aider"])
    assert len(plugins) == 1
    assert plugins[0].name == "aider"


def test_registry_resolves_all_alias():
    plugins = resolve_plugins(["all"])
    names = [p.name for p in plugins]
    assert names == [
        "mmlu",
        "arc-c",
        "hellaswag",
        "winogrande",
        "gsm8k",
        "truthfulqa",
        "ifeval",
        "evalplus",
        "terminal_bench",
        "aider",
        "swebench_verified",
    ]


def test_registry_resolves_core6_alias_to_individual_tasks():
    plugins = resolve_plugins(["core6"])
    names = [p.name for p in plugins]
    assert names == ["mmlu", "arc-c", "hellaswag", "winogrande", "gsm8k", "truthfulqa"]


def test_intelligence_json_output(capsys):
    results = BenchmarkResults()
    results.intelligence = IntelligenceReport(
        plugins=[
            IntelligencePluginResult(
                plugin="core6",
                success=True,
                summary_metric=0.42,
                duration_seconds=1.23,
                tasks=[IntelligenceTaskResult(name="mmlu", metric="acc", value=0.42)],
            )
        ]
    )
    results.save_report(filename=None, format="json", concurrency=1)
    out = capsys.readouterr().out
    payload = json.loads(out[out.find("{") : out.rfind("}") + 1])
    assert "intelligence" in payload
    assert payload["intelligence"]["plugins"][0]["plugin"] == "core6"
    assert payload["intelligence"]["plugins"][0]["duration_seconds"] == 1.23


def test_main_routes_intelligence_mode(monkeypatch):
    import llama_benchy.__main__ as main_mod
    import llama_benchy.results as results_mod

    config = _base_config(
        intelligence_plugins=["core6"],
        output_dir="/tmp/llama-benchy-intelligence-test",
        save_result=None,
        result_format="json",
        concurrency_levels=[1],
    )
    monkeypatch.setattr(main_mod.BenchmarkConfig, "from_args", lambda: config)

    class FakeIRunner:
        def __init__(self, cfg, run_dir):
            self.cfg = cfg
            self.run_dir = run_dir

        def run(self):
            return IntelligenceReport(
                plugins=[IntelligencePluginResult(plugin="core6", success=True, tasks=[])]
            )

    called = {"saved": False}

    class FakeResults(BenchmarkResults):
        def save_report(self, filename, format, concurrency=1):
            called["saved"] = True

    monkeypatch.setattr(main_mod, "IntelligenceRunner", FakeIRunner)
    monkeypatch.setattr(results_mod, "BenchmarkResults", FakeResults)
    asyncio.run(main_mod.main_async())
    assert called["saved"] is True


def test_main_routes_default_mode(monkeypatch):
    import llama_benchy.__main__ as main_mod

    config = _base_config(enable_intelligence=False)
    monkeypatch.setattr(main_mod.BenchmarkConfig, "from_args", lambda: config)

    class FakeCorpus:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 123

    class FakePromptGen:
        def __init__(self, _corpus):
            pass

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

    called = {"ran": False}

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        async def run_suite(self):
            called["ran"] = True

    monkeypatch.setattr(main_mod, "TokenizedCorpus", FakeCorpus)
    monkeypatch.setattr(main_mod, "PromptGenerator", FakePromptGen)
    monkeypatch.setattr(main_mod, "LLMClient", FakeClient)
    monkeypatch.setattr(main_mod, "BenchmarkRunner", FakeRunner)
    asyncio.run(main_mod.main_async())
    assert called["ran"] is True


def test_main_handles_missing_output_dir_without_traceback(monkeypatch, capsys):
    import llama_benchy.__main__ as main_mod

    config = _base_config(output_dir=None, enable_intelligence=True)
    monkeypatch.setattr(main_mod.BenchmarkConfig, "from_args", lambda: config)

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "ERROR: --enable-intelligence requires --output-dir." in err
    assert "Example: llama-benchy" in err

