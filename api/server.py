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
    allow_origins=["*"],
    allow_credentials=False,
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


from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import requests as http_requests

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}

# Known source URL patterns for proxy fallback
SOURCE_URL_PATTERNS = [
    # COCO val2017
    lambda p: f"http://images.cocodataset.org/val2017/{Path(p).name}" if "coco" in p.lower() or Path(p).name.startswith("0000") else None,
    # COCO train2017
    lambda p: f"http://images.cocodataset.org/train2017/{Path(p).name}" if "train" in p.lower() else None,
]


def _guess_source_url(path: str) -> str | None:
    for pattern in SOURCE_URL_PATTERNS:
        url = pattern(path)
        if url:
            return url
    return None


@app.get("/api/images/file")
async def serve_image(path: str):
    """Serve image from local disk, or proxy from source URL if local file is missing."""
    from fastapi import HTTPException
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    media_type = MEDIA_TYPES.get(suffix, "image/jpeg")

    # Try local file first
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path), media_type=media_type)

    # Local file missing — try to proxy from source
    source_url = _guess_source_url(path)
    if source_url:
        try:
            resp = http_requests.get(source_url, timeout=10, stream=True)
            resp.raise_for_status()
            return StreamingResponse(
                resp.iter_content(chunk_size=8192),
                media_type=media_type,
                headers={"Cache-Control": "public, max-age=86400"},
            )
        except Exception:
            pass

    raise HTTPException(404, "Image not found")
