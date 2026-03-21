import requests
from pathlib import Path
from validation_pipeline.schemas.dataset import DatasetPlan
from validation_pipeline.dataset.base import BaseDownloader


class HuggingFaceDownloader(BaseDownloader):
    def download(self, plan: DatasetPlan) -> str:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        dest = Path(plan.download_path)
        dest.mkdir(parents=True, exist_ok=True)
        repo_id = plan.url
        files = list_repo_files(repo_id, repo_type="dataset")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in files if Path(f).suffix.lower() in image_exts]
        if plan.subset:
            image_files = [f for f in image_files if plan.subset in f]
        image_files = image_files[:plan.max_images]
        for fname in image_files:
            local = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
            target = dest / Path(fname).name
            if not target.exists():
                target.write_bytes(Path(local).read_bytes())
        return str(dest)
