from validation_pipeline.schemas.dataset import DatasetPlan


def test_dataset_plan_schema():
    plan = DatasetPlan(
        source="coco",
        url="http://images.cocodataset.org/zips/val2017.zip",
        subset="val2017",
        category_filter="horse",
        max_images=50,
        download_path="/tmp/coco_horses",
    )
    assert plan.source == "coco"
    assert plan.category_filter == "horse"
    assert plan.max_images == 50


def test_dataset_plan_minimal():
    plan = DatasetPlan(
        source="url",
        url="https://example.com/images.zip",
        max_images=100,
        download_path="/tmp/images",
    )
    assert plan.subset is None
    assert plan.category_filter is None


import json
import zipfile
from unittest.mock import patch, MagicMock
from validation_pipeline.dataset.coco import COCODownloader
from validation_pipeline.dataset.url import URLDownloader


def test_url_downloader_zip(tmp_path):
    """URLDownloader extracts a zip archive."""
    zip_path = tmp_path / "imgs.zip"
    img_dir = tmp_path / "source"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"\xff\xd8fake_jpg_data")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img_dir / "a.jpg", "a.jpg")

    download_path = str(tmp_path / "output")
    plan = DatasetPlan(source="url", url=f"file://{zip_path}", max_images=10, download_path=download_path)

    downloader = URLDownloader()
    result_path = downloader.download(plan)
    assert (tmp_path / "output").exists()


def test_coco_downloader(tmp_path):
    """COCODownloader fetches images by category."""
    annotations = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "coco_url": "http://example.com/img1.jpg"},
            {"id": 2, "file_name": "img2.jpg", "coco_url": "http://example.com/img2.jpg"},
            {"id": 3, "file_name": "img3.jpg", "coco_url": "http://example.com/img3.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 18},
            {"id": 2, "image_id": 2, "category_id": 18},
            {"id": 3, "image_id": 3, "category_id": 1},
        ],
        "categories": [
            {"id": 18, "name": "horse"},
            {"id": 1, "name": "person"},
        ],
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(annotations))

    download_path = str(tmp_path / "output")
    plan = DatasetPlan(
        source="coco", url="http://images.cocodataset.org",
        subset="val2017", category_filter="horse",
        max_images=2, download_path=download_path,
    )

    fake_img = b"\xff\xd8fake_jpg"
    mock_resp = MagicMock()
    mock_resp.content = fake_img
    mock_resp.raise_for_status = MagicMock()

    downloader = COCODownloader(cache_dir=str(tmp_path / "cache"))
    with patch.object(downloader, "_get_annotations_path", return_value=str(ann_path)), \
         patch("requests.get", return_value=mock_resp):
        result_path = downloader.download(plan)

    from pathlib import Path
    imgs = list(Path(result_path).glob("*.jpg"))
    assert len(imgs) == 2


from unittest.mock import patch, MagicMock
from validation_pipeline.modules.dataset_resolver import resolve_dataset, download_dataset


def test_resolve_dataset():
    mock_plan = DatasetPlan(
        source="coco", url="http://images.cocodataset.org",
        subset="val2017", category_filter="horse",
        max_images=50, download_path="/tmp/coco_horses",
    )
    with patch("validation_pipeline.modules.dataset_resolver._call_llm", return_value=mock_plan):
        plan = resolve_dataset("50 horse images from COCO")
    assert plan.source == "coco"
    assert plan.category_filter == "horse"


def test_download_dataset_dispatches_coco(tmp_path):
    plan = DatasetPlan(
        source="coco", url="http://images.cocodataset.org",
        subset="val2017", category_filter="horse",
        max_images=5, download_path=str(tmp_path / "out"),
    )
    with patch("validation_pipeline.dataset.coco.COCODownloader.download", return_value=str(tmp_path / "out")):
        result = download_dataset(plan)
    assert result == str(tmp_path / "out")


def test_download_dataset_dispatches_url(tmp_path):
    plan = DatasetPlan(
        source="url", url="https://example.com/imgs.zip",
        max_images=10, download_path=str(tmp_path / "out"),
    )
    with patch("validation_pipeline.dataset.url.URLDownloader.download", return_value=str(tmp_path / "out")):
        result = download_dataset(plan)
    assert result == str(tmp_path / "out")


import pytest
from validation_pipeline.errors import LLMError, DatasetError


def test_resolve_dataset_raises_llm_error():
    with patch("validation_pipeline.modules.dataset_resolver._call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(LLMError) as exc_info:
            resolve_dataset("test")
    assert exc_info.value.module == "dataset_resolver"


def test_download_dataset_raises_dataset_error_unknown_source():
    plan = DatasetPlan(
        source="unknown", url="http://x.com",
        max_images=10, download_path="/tmp/test",
    )
    with pytest.raises(DatasetError) as exc_info:
        download_dataset(plan)
    assert exc_info.value.module == "dataset_resolver"
    assert "unknown" in str(exc_info.value)
