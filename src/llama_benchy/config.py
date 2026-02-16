from pydantic import BaseModel, Field
from typing import List, Optional
import argparse
import os
from ._version import __version__

class BenchmarkConfig(BaseModel):
    base_url: str = Field(..., description="OpenAI compatible endpoint URL")
    api_key: str = Field(..., description="API Key for the endpoint")
    model: str = Field(..., description="Model name to use for benchmarking")
    served_model_name: str = Field(..., description="Model name used in API calls (defaults to --model if not specified)")
    tokenizer: Optional[str] = Field(None, description="HuggingFace tokenizer name (defaults to model name)")
    pp_counts: List[int] = Field(..., description="List of prompt processing token counts")
    tg_counts: List[int] = Field(..., description="List of token generation counts")
    depths: List[int] = Field(..., description="List of context depths (previous conversation tokens)")
    num_runs: int = Field(..., description="Number of runs per test")
    no_cache: bool = Field(..., description="Ensure unique requests to avoid prefix caching")
    latency_mode: str = Field(..., description="Method to measure latency: 'api', 'generation', or 'none'")
    no_warmup: bool = Field(..., description="Skip warmup phase")
    adapt_prompt: bool = Field(..., description="Adapt prompt size based on warmup token usage delta")
    enable_prefix_caching: bool = Field(..., description="Enable prefix caching performance measurement")
    book_url: str = Field(..., description="URL of a book to use for text generation")
    post_run_cmd: Optional[str] = Field(None, description="Command to execute after each test run")
    concurrency_levels: List[int] = Field(..., description="List of concurrency levels")
    save_result: Optional[str] = Field(None, description="File to save results to")
    output_dir: Optional[str] = Field(None, description="Output directory for run artifacts")
    result_format: str = Field("md", description="Output format (md, json, csv)")
    save_total_throughput_timeseries: bool = Field(False, description="Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run.")
    save_all_throughput_timeseries: bool = Field(False, description="Save calculated throughput timeseries for EACH individual request.")
    enable_intelligence: bool = Field(False, description="Enable intelligence benchmark mode")
    intelligence_plugins: List[str] = Field(default_factory=list, description="Intelligence plugins to run")
    allow_code_exec: bool = Field(False, description="Allow code execution for intelligence plugins that require it")
    dataset_cache_dir: Optional[str] = Field(None, description="Optional dataset cache directory for intelligence plugins")

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="LLM Benchmark Script")
        parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
        parser.add_argument("--base-url", type=str, required=True, help="OpenAI compatible endpoint URL")
        parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key for the endpoint")
        parser.add_argument("--model", type=str, required=True, help="Model name to use for benchmarking")
        parser.add_argument("--served-model-name", type=str, default=None, help="Model name used in API calls (defaults to --model if not specified)")
        parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (defaults to model name)")
        parser.add_argument("--pp", type=int, nargs='+', required=False, default=[2048], help="List of prompt processing token counts - default: 2048")
        parser.add_argument("--tg", type=int, nargs='+', required=False, default=[32], help="List of token generation counts - default: 32")
        parser.add_argument("--depth", type=int, nargs='+', default=[0], help="List of context depths (previous conversation tokens) - default: 0")
        parser.add_argument("--runs", type=int, default=3, help="Number of runs per test - default: 3")
        parser.add_argument("--no-cache", action="store_true", help="Ensure unique requests to avoid prefix caching and send cache_prompt=false to the server")
        parser.add_argument("--post-run-cmd", type=str, default=None, help="Command to execute after each test run")
        parser.add_argument("--book-url", type=str, default="https://www.gutenberg.org/files/1661/1661-0.txt", help="URL of a book to use for text generation, defaults to Sherlock Holmes")
        parser.add_argument("--latency-mode", type=str, default="api", choices=["api", "generation", "none"], help="Method to measure latency: 'api' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement)")
        parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
        parser.add_argument("--adapt-prompt", action="store_true", default=True, help="Adapt prompt size based on warmup token usage delta (default: True)")
        parser.add_argument("--no-adapt-prompt", action="store_false", dest="adapt_prompt", help="Disable prompt size adaptation")
        parser.add_argument("--enable-prefix-caching", action="store_true", help="Enable prefix caching performance measurement")
        parser.add_argument("--concurrency", type=int, nargs='+', default=[1], help="List of concurrency levels (number of concurrent requests per test) - default: [1]")
        parser.add_argument("--save-result", type=str, help="File to save results to")
        parser.add_argument("--output-dir", type=str, default=None, help="Output directory for run artifacts")
        parser.add_argument("--format", type=str, default="md", choices=["md", "json", "csv"], help="Output format")
        parser.add_argument("--save-total-throughput-timeseries", action="store_true", help="Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run.")
        parser.add_argument("--save-all-throughput-timeseries", action="store_true", help="Save calculated throughput timeseries for EACH individual request.")
        parser.add_argument("--enable-intelligence", action="store_true", help="Enable intelligence benchmark plugins (opt-in, default off)")
        parser.add_argument(
            "--intelligence-plugins",
            type=str,
            nargs="+",
            default=[],
            choices=["all", "core6", "mmlu", "arc-c", "hellaswag", "winogrande", "gsm8k", "truthfulqa", "ifeval", "evalplus", "terminal_bench", "aider", "swebench_verified"],
            help="Intelligence plugins to run: all, core6, mmlu, arc-c, hellaswag, winogrande, gsm8k, truthfulqa, ifeval, evalplus, terminal_bench, aider, swebench_verified",
        )
        parser.add_argument("--allow-code-exec", action="store_true", help="Allow code execution for plugins that require it (e.g. evalplus)")
        parser.add_argument("--dataset-cache-dir", type=str, default=None, help="Dataset cache directory used by intelligence plugins")
          
        args = parser.parse_args()
        
        return cls(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            served_model_name=args.served_model_name if args.served_model_name else args.model,
            tokenizer=args.tokenizer,
            pp_counts=args.pp,
            tg_counts=args.tg,
            depths=args.depth,
            num_runs=args.runs,
            no_cache=args.no_cache,
            latency_mode=args.latency_mode,
            no_warmup=args.no_warmup,
            adapt_prompt=args.adapt_prompt,
            enable_prefix_caching=args.enable_prefix_caching,
            book_url=args.book_url,
            post_run_cmd=args.post_run_cmd,
            concurrency_levels=args.concurrency,
            save_result=args.save_result,
            output_dir=args.output_dir,
            result_format=args.format,
            save_total_throughput_timeseries=args.save_total_throughput_timeseries,
            save_all_throughput_timeseries=args.save_all_throughput_timeseries,
            enable_intelligence=args.enable_intelligence,
            intelligence_plugins=args.intelligence_plugins,
            allow_code_exec=args.allow_code_exec,
            dataset_cache_dir=args.dataset_cache_dir,
        )
