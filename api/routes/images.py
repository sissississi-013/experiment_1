from fastapi import APIRouter, Request

router = APIRouter(tags=["images"])

@router.get("/api/runs/{run_id}/images")
async def get_run_images(run_id: str, request: Request, verdict: str | None = None):
    store = request.app.state.store
    if not store:
        return []
    return store.get_run_images(run_id, verdict=verdict)

@router.get("/api/images")
async def query_images(request: Request, verdict: str | None = None, min_score: float | None = None, dimension: str | None = None, limit: int = 100):
    store = request.app.state.store
    if not store:
        return []
    return store.query_images(verdict=verdict, min_score=min_score, dimension=dimension, limit=limit)
