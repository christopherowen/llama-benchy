from typing import Dict, List

from .base import IntelligencePlugin
from .plugins.core6 import CORE6_PLUGIN_NAMES, get_core6_plugins
from .plugins.evalplus import EvalPlusPlugin
from .plugins.ifeval import IFEvalPlugin
from .plugins.livecodebench import LiveCodeBenchPlugin


def get_plugin_registry() -> Dict[str, IntelligencePlugin]:
    plugins: Dict[str, IntelligencePlugin] = {}
    plugins.update(get_core6_plugins())
    plugins["ifeval"] = IFEvalPlugin()
    plugins["evalplus"] = EvalPlusPlugin()
    plugins["livecodebench"] = LiveCodeBenchPlugin()
    return plugins


def resolve_plugins(selected_plugins: List[str]) -> List[IntelligencePlugin]:
    registry = get_plugin_registry()
    if not selected_plugins:
        return []

    expanded_plugins: List[str] = []
    for plugin_name in selected_plugins:
        if plugin_name == "all":
            expanded_plugins.extend(list(registry.keys()))
        elif plugin_name == "core6":
            expanded_plugins.extend(CORE6_PLUGIN_NAMES)
        else:
            expanded_plugins.append(plugin_name)

    plugins: List[IntelligencePlugin] = []
    for plugin_name in dict.fromkeys(expanded_plugins):
        plugin = registry.get(plugin_name)
        if plugin is None:
            raise ValueError(f"Unknown intelligence plugin: {plugin_name}")
        plugins.append(plugin)
    return plugins

