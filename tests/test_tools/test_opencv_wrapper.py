import numpy as np
from PIL import Image
from validation_pipeline.tools.wrappers.opencv_wrapper import (
    LaplacianBlurTool, HistogramExposureTool, PixelStatsTool,
)


def _sharp_image():
    return Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


def _blurry_image():
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    noise = np.random.randint(-5, 5, (100, 100, 3), dtype=np.int16)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def _black_image():
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))


def test_laplacian_sharp_vs_blurry():
    tool = LaplacianBlurTool(config={})
    sharp_score = tool.execute(_sharp_image())
    blurry_score = tool.execute(_blurry_image())
    assert sharp_score > blurry_score


def test_laplacian_normalize_returns_tool_result():
    tool = LaplacianBlurTool(config={})
    raw = tool.execute(_sharp_image())
    result = tool.normalize(raw)
    assert result.tool_name == "laplacian_blur"
    assert result.dimension == "blur"
    assert 0.0 <= result.score <= 1.0


def test_histogram_detects_dark_image():
    tool = HistogramExposureTool(config={})
    dark_score = tool.execute(_black_image())
    normal_score = tool.execute(_sharp_image())
    assert dark_score < normal_score


def test_pixel_stats_detects_solid_color():
    tool = PixelStatsTool(config={})
    black_score = tool.execute(_black_image())
    normal_score = tool.execute(_sharp_image())
    assert black_score < normal_score
