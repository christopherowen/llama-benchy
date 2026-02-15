from typing import Dict, List

from .base import IntelligencePlugin
from .plugins.core6 import Core6Plugin
from .plugins.evalplus import EvalPlusPlugin
from .plugins.ifeval import IFEvalPlugin
from .plugins.livecodebench import LiveCodeBenchPlugin


def get_plugin_registry() -> Dict[str, IntelligencePlugin]:
    return {
        "core6": Core6Plugin(),
        "ifeval": IFEvalPlugin(),
        "evalplus": EvalPlusPlugin(),
        "livecodebench": LiveCodeBenchPlugin(),
    }


def resolve_plugins(selected_plugins: List[str]) -> List[IntelligencePlugin]:
    registry = get_plugin_registry()
    if not selected_plugins:
        return []

    if "all" in selected_plugins:
        selected_plugins = list(registry.keys())

    plugins: List[IntelligencePlugin] = []
    for plugin_name in dict.fromkeys(selected_plugins):
        plugin = registry.get(plugin_name)
        if plugin is None:
            raise ValueError(f"Unknown intelligence plugin: {plugin_name}")
        plugins.append(plugin)
    return plugins

