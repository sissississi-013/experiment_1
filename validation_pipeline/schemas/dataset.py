from pydantic import BaseModel


class DatasetPlan(BaseModel):
    source: str
    url: str
    subset: str | None = None
    category_filter: str | None = None
    max_images: int
    download_path: str
