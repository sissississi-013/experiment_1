import instructor
from openai import OpenAI
from validation_pipeline.schemas.dataset import DatasetPlan
from validation_pipeline.config import PipelineConfig
from validation_pipeline.dataset.coco import COCODownloader
from validation_pipeline.dataset.huggingface import HuggingFaceDownloader
from validation_pipeline.dataset.url import URLDownloader


COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

SYSTEM_PROMPT = f"""You are a dataset resolver. Given a user's description of what dataset they need,
produce a DatasetPlan with:
- source: "coco", "huggingface", or "url"
- url: the base URL or repo ID for the dataset
- subset: dataset subset if applicable (e.g., "val2017", "train2017")
- category_filter: object category to filter by (e.g., "horse", "car") or null
- max_images: how many images to download
- download_path: use "/tmp/validation-pipeline-data/<descriptive-name>"

For COCO datasets:
- ONLY use source "coco" if the requested category is in this list: {COCO_CATEGORIES}
- If the category is NOT in the list above, use "huggingface" instead
- url should be "http://images.cocodataset.org"
- subset should be one of: "train2017", "val2017"
- Prefer val2017 for smaller downloads unless the user asks for training data

For HuggingFace datasets:
- Use this when the requested object/category is NOT in COCO's 80 categories
- Search for a relevant dataset on HuggingFace Hub
- url should be the repo_id (e.g., "username/dataset-name")
- Pick datasets with many downloads/likes when possible

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
    from validation_pipeline.errors import LLMError
    try:
        return _call_llm(description, config)
    except Exception as e:
        raise LLMError(
            f"Dataset resolution failed: {e}",
            module="dataset_resolver",
            context={"description": description},
        ) from e


DOWNLOADERS = {
    "coco": lambda: COCODownloader(),
    "huggingface": lambda: HuggingFaceDownloader(),
    "url": lambda: URLDownloader(),
}


def download_dataset(plan: DatasetPlan) -> str:
    from validation_pipeline.errors import DatasetError
    factory = DOWNLOADERS.get(plan.source)
    if not factory:
        raise DatasetError(
            f"Unknown dataset source: {plan.source}. Supported: {list(DOWNLOADERS.keys())}",
            module="dataset_resolver",
            context={"source": plan.source, "supported": list(DOWNLOADERS.keys())},
        )
    downloader = factory()
    try:
        return downloader.download(plan)
    except Exception as e:
        raise DatasetError(
            f"Dataset download failed: {e}",
            module="dataset_resolver",
            context={"source": plan.source, "url": plan.url},
        ) from e
