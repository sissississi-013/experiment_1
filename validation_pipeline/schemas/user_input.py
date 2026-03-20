from pydantic import BaseModel


class UserInput(BaseModel):
    dataset_path: str
    intent: str
    exemplar_good_paths: list[str] = []
    exemplar_bad_paths: list[str] = []
