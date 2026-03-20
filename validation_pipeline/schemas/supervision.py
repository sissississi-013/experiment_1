from pydantic import BaseModel


class SupervisionCheck(BaseModel):
    check_name: str
    passed: bool
    details: str


class Anomaly(BaseModel):
    severity: str
    description: str
    likely_cause: str
    suggested_action: str


class SupervisionReport(BaseModel):
    status: str = "passed"
    checks: list[SupervisionCheck] = []
    anomalies: list[Anomaly] = []
    recommendation: str = ""
