import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from validation_pipeline.config import PipelineConfig
from validation_pipeline.persistence.run_store import RunStore


def get_config() -> PipelineConfig:
    return PipelineConfig(openai_api_key=os.environ.get("OPENAI_API_KEY", ""))


def get_store() -> RunStore | None:
    db_url = os.environ.get("NEON_DATABASE_URL", "")
    if db_url:
        return RunStore(db_url)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = get_config()
    app.state.store = get_store()
    yield


app = FastAPI(title="Validation Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from api.routes.runs import router as runs_router
app.include_router(runs_router)

from api.routes.images import router as images_router
app.include_router(images_router)

from api.routes.export import router as export_router
app.include_router(export_router)

from api.routes.ws import router as ws_router
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/api/images/file")
async def serve_image(path: str):
    """Serve a local image file by its absolute path."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        from fastapi import HTTPException
        raise HTTPException(404, "Image not found")
    suffix = file_path.suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}
    return FileResponse(str(file_path), media_type=media_types.get(suffix, "image/jpeg"))
