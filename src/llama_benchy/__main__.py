"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
import json
import os
from . import __version__
from .config import BenchmarkConfig
from .corpus import TokenizedCorpus
from .prompts import PromptGenerator
from .client import LLMClient
from .runner import BenchmarkRunner
from .intelligence.runner import IntelligenceRunner

async def main_async():
    # 1. Parse Configuration
    config = BenchmarkConfig.from_args()
    
    # 2. Print Header
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({__version__})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {config.model} at {config.base_url}")
    print(f"Concurrency levels: {config.concurrency_levels}")

    if config.enable_intelligence:
        # Intelligence plugins are opt-in and run independently from throughput benchmarks.
        if not config.intelligence_plugins:
            raise ValueError("--enable-intelligence requires --intelligence-plugins.")
        if not config.output_dir:
            raise ValueError("--enable-intelligence requires --output-dir.")

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in config.model)
        run_dir = os.path.join(os.path.abspath(config.output_dir), f"{timestamp}_{model_slug}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Intelligence run directory: {run_dir}")

        intelligence_runner = IntelligenceRunner(config, run_dir)
        report = intelligence_runner.run()
        from .results import BenchmarkResults
        results = BenchmarkResults()
        results.intelligence = report

        report_path = os.path.join(run_dir, "report.json")
        results.save_report(report_path, "json", max(config.concurrency_levels) if config.concurrency_levels else 1)

        manifest = {
            "version": __version__,
            "model": config.model,
            "base_url": config.base_url,
            "timestamp": timestamp,
            "run_dir": run_dir,
            "report_json": report_path,
            "logs_dir": os.path.join(run_dir, "logs"),
            "artifacts_dir": os.path.join(run_dir, "artifacts"),
            "plugins": [plugin.model_dump() for plugin in report.plugins],
        }
        manifest_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved intelligence report: {report_path}")
        print(f"Saved intelligence manifest: {manifest_path}")

        if config.result_format != "json":
            results.save_report(None, config.result_format, max(config.concurrency_levels) if config.concurrency_levels else 1)
        else:
            results.save_report(None, "json", max(config.concurrency_levels) if config.concurrency_levels else 1)
    else:
        # 3. Prepare Data
        corpus = TokenizedCorpus(config.book_url, config.tokenizer, config.model)
        print(f"Total tokens available in text corpus: {len(corpus)}")
        
        # 4. Initialize Components
        prompt_gen = PromptGenerator(corpus)
        client = LLMClient(config.base_url, config.api_key, config.served_model_name)
        runner = BenchmarkRunner(config, client, prompt_gen)
        
        # 5. Run Benchmark Suite
        await runner.run_suite()
    
    print(f"\nllama-benchy ({__version__})")
    print(f"date: {current_time} | latency mode: {config.latency_mode}")

def main():
    """Entry point for the CLI command."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
