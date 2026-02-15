"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
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
        intelligence_runner = IntelligenceRunner(config)
        report = intelligence_runner.run()
        from .results import BenchmarkResults
        results = BenchmarkResults()
        results.intelligence = report
        results.save_report(config.save_result, config.result_format, max(config.concurrency_levels) if config.concurrency_levels else 1)
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
