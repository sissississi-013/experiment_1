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


@app.get("/api/health")
async def health():
    return {"status": "ok"}
