import json
import os
import random
import requests
from pathlib import Path
from validation_pipeline.schemas.dataset import DatasetPlan
from validation_pipeline.dataset.base import BaseDownloader

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


class COCODownloader(BaseDownloader):
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/validation-pipeline/coco"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_annotations_path(self, subset: str) -> str:
        ann_file = self.cache_dir / f"instances_{subset}.json"
        if not ann_file.exists():
            self._download_annotations(subset)
        return str(ann_file)

    def _download_annotations(self, subset: str):
        import zipfile, io
        resp = requests.get(COCO_ANNOTATIONS_URL, stream=True, timeout=300)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            target = f"annotations/instances_{subset}.json"
            for name in zf.namelist():
                if name == target:
                    data = zf.read(name)
                    (self.cache_dir / f"instances_{subset}.json").write_bytes(data)
                    return
        raise FileNotFoundError(f"Annotation file for {subset} not found in archive")

    def download(self, plan: DatasetPlan) -> str:
        subset = plan.subset or "val2017"
        ann_path = self._get_annotations_path(subset)
        with open(ann_path) as f:
            data = json.load(f)
        cat_map = {c["name"].lower(): c["id"] for c in data["categories"]}
        if plan.category_filter:
            cat_id = cat_map.get(plan.category_filter.lower())
            if cat_id is None:
                raise ValueError(f"Category '{plan.category_filter}' not found. Available: {list(cat_map.keys())}")
            target_img_ids = {a["image_id"] for a in data["annotations"] if a["category_id"] == cat_id}
        else:
            target_img_ids = {img["id"] for img in data["images"]}
        images = [img for img in data["images"] if img["id"] in target_img_ids]
        random.shuffle(images)  # Randomize so each run gets different images
        images = images[:plan.max_images]
        dest = Path(plan.download_path)
        dest.mkdir(parents=True, exist_ok=True)
        for img_info in images:
            url = img_info["coco_url"]
            filename = img_info["file_name"]
            out_path = dest / filename
            if not out_path.exists():
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
        return str(dest)
