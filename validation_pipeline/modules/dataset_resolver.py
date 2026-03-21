import instructor
from openai import OpenAI
from validation_pipeline.schemas.dataset import DatasetPlan
from validation_pipeline.config import PipelineConfig
from validation_pipeline.dataset.coco import COCODownloader
from validation_pipeline.dataset.huggingface import HuggingFaceDownloader
from validation_pipeline.dataset.url import URLDownloader


SYSTEM_PROMPT = """You are a dataset resolver. Given a user's description of what dataset they need,
produce a DatasetPlan with:
- source: "coco", "huggingface", or "url"
- url: the base URL or repo ID for the dataset
- subset: dataset subset if applicable (e.g., "val2017", "train2017")
- category_filter: object category to filter by (e.g., "horse", "car") or null
- max_images: how many images to download
- download_path: use "/tmp/validation-pipeline-data/<descriptive-name>"

For COCO datasets:
- url should be "http://images.cocodataset.org"
- subset should be one of: "train2017", "val2017"
- Prefer val2017 for smaller downloads unless the user asks for training data

For HuggingFace datasets:
- url should be the repo_id (e.g., "username/dataset-name")

For direct URLs:
- url should be the full download URL"""


def _call_llm(description: str, config: PipelineConfig | None = None) -> DatasetPlan:
    config = config or PipelineConfig()
    client = instructor.from_openai(OpenAI(api_key=config.openai_api_key))
    return client.chat.completions.create(
        model=config.llm_model,
        response_model=DatasetPlan,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        max_retries=config.max_retries,
    )


def resolve_dataset(description: str, config: PipelineConfig | None = None) -> DatasetPlan:
    return _call_llm(description, config)


DOWNLOADERS = {
    "coco": lambda: COCODownloader(),
    "huggingface": lambda: HuggingFaceDownloader(),
    "url": lambda: URLDownloader(),
}


def download_dataset(plan: DatasetPlan) -> str:
    factory = DOWNLOADERS.get(plan.source)
    if not factory:
        raise ValueError(f"Unknown dataset source: {plan.source}. Supported: {list(DOWNLOADERS.keys())}")
    downloader = factory()
    return downloader.download(plan)
