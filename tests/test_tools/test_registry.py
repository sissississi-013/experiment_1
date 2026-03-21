from validation_pipeline.tools.registry import ToolRegistry, WRAPPER_MAP


def test_wrapper_map_contains_roboflow():
    assert "roboflow_wrapper.RoboflowObjectDetectionTool" in WRAPPER_MAP


def test_wrapper_map_contains_openai_vision():
    assert "openai_vision_wrapper.GPT4VisionTool" in WRAPPER_MAP


def test_registry_loads_configs(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_tool.yaml").write_text(
        "name: test_tool\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\ndefault_config: {}\n'
    )
    registry = ToolRegistry(str(config_dir))
    assert "test_tool" in registry.configs


def test_registry_instantiates_tool(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "blur.yaml").write_text(
        "name: laplacian_blur\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\ndefault_config: {}\n'
    )
    registry = ToolRegistry(str(config_dir))
    tool = registry.get_tool("laplacian_blur")
    assert tool.name == "laplacian_blur"


def test_registry_search_by_task(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "blur.yaml").write_text(
        "name: laplacian_blur\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\n'
    )
    registry = ToolRegistry(str(config_dir))
    results = registry.search_by_task("image_quality")
    assert len(results) == 1
