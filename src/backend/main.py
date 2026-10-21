import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.api.routes import router
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.rag.vectorstore import warmup
from backend.services.audit import init_db


BASE_DIR = Path(__file__).resolve().parents[2]

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Fail early with a clear error if the frontend directories are missing.
if not STATIC_DIR.is_dir():
    raise RuntimeError(f"Static directory does not exist: {STATIC_DIR}")

if not TEMPLATES_DIR.is_dir():
    raise RuntimeError(f"Templates directory does not exist: {TEMPLATES_DIR}")

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Warm up in the background so a slow or offline network can never
    # delay server startup. The first chat request retries if this fails.
    async def _warm():
        try:
            await asyncio.to_thread(warmup)
            logger.info("Vector store warm-up complete")
        except Exception:
            logger.exception("Vector store warm-up failed; will retry on first request")

    task = asyncio.create_task(_warm())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": settings.app_name},
    )


app.include_router(router)