"""Optional FastAPI integration for serving a pipeline over HTTP."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from conveyor.pipeline import Pipeline


def create_app(pipeline: Pipeline, prefix: str = "/pipeline") -> Any:
    """Create a FastAPI application that serves *pipeline*.

    Requires ``fastapi`` to be installed (optional dependency).
    """
    try:
        import fastapi
        from fastapi.exceptions import HTTPException
    except ImportError as e:
        raise ImportError(
            "fastapi is required for create_app(). "
            "Install it with: pip install conveyor[server]"
        ) from e

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        async with pipeline:
            yield

    app = fastapi.FastAPI(lifespan=lifespan)
    router = fastapi.APIRouter()

    @router.post("/submit")
    async def submit(payload: Any):
        if not pipeline.available_slots():
            raise HTTPException(status_code=429, detail="Too many requests")
        return await pipeline.submit(payload)

    @router.post("/bulk/submit")
    async def bulk_submit(payload: list):
        if not payload:
            return []
        if len(payload) > pipeline.available_slots():
            raise HTTPException(status_code=429, detail="Too many requests")
        return await asyncio.gather(*[pipeline.submit(item) for item in payload])

    @router.get("/report")
    async def report():
        return await pipeline.report()

    @router.post("/stop")
    async def stop():
        await pipeline.stop()

    @router.post("/start")
    async def start():
        await pipeline.start()

    app.include_router(router, prefix=prefix)
    return app
