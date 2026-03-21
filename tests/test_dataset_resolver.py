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
