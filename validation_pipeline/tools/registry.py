import yaml
import importlib
from pathlib import Path
from validation_pipeline.tools.base import BaseTool


WRAPPER_MAP = {
    "opencv_wrapper.LaplacianBlurTool": "validation_pipeline.tools.wrappers.opencv_wrapper.LaplacianBlurTool",
    "opencv_wrapper.HistogramExposureTool": "validation_pipeline.tools.wrappers.opencv_wrapper.HistogramExposureTool",
    "opencv_wrapper.PixelStatsTool": "validation_pipeline.tools.wrappers.opencv_wrapper.PixelStatsTool",
}


class ToolRegistry:
    def __init__(self, configs_dir: str):
        self.configs_dir = Path(configs_dir)
        self.configs: dict[str, dict] = {}
        self.instances: dict[str, BaseTool] = {}
        self._load_configs()

    def _load_configs(self):
        if not self.configs_dir.exists():
            return
        for yaml_file in self.configs_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                if config and "name" in config:
                    self.configs[config["name"]] = config

    def list_tools(self) -> list[dict]:
        return list(self.configs.values())

    def get_tool(self, name: str) -> BaseTool:
        if name in self.instances:
            return self.instances[name]
        if name not in self.configs:
            raise KeyError(f"Tool '{name}' not found in registry")
        config = self.configs[name]
        wrapper_class = self._resolve_wrapper(config["wrapper_class"])
        instance = wrapper_class(config.get("default_config", {}))
        self.instances[name] = instance
        return instance

    def _resolve_wrapper(self, class_path: str):
        full_path = WRAPPER_MAP.get(class_path, class_path)
        module_path, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def search_by_task(self, task_type: str) -> list[dict]:
        return [c for c in self.configs.values() if c.get("task_type") == task_type]
