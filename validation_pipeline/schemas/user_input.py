from pydantic import BaseModel


class UserInput(BaseModel):
    dataset_path: str | None = None
    intent: str
    exemplar_good_paths: list[str] = []
    exemplar_bad_paths: list[str] = []
    dataset_description: str | None = None
