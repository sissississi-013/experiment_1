import zipfile
import tarfile
import requests
from pathlib import Path
from urllib.parse import urlparse
from validation_pipeline.schemas.dataset import DatasetPlan
from validation_pipeline.dataset.base import BaseDownloader


class URLDownloader(BaseDownloader):
    def download(self, plan: DatasetPlan) -> str:
        dest = Path(plan.download_path)
        dest.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(plan.url)
        if parsed.scheme == "file":
            archive_path = parsed.path
        else:
            archive_path = self._fetch(plan.url, dest)
        if archive_path and zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest)
        elif archive_path and tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(dest)
        images = sorted(dest.rglob("*"))
        images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        for img in images[plan.max_images:]:
            img.unlink()
        return str(dest)

    def _fetch(self, url: str, dest: Path) -> str:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        filename = Path(urlparse(url).path).name or "download"
        path = dest / filename
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(path)
