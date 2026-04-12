"""Optional FastAPI integration for serving a pipeline over HTTP."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from conveyor.pipeline import Pipeline


def create_app(
    pipeline: Pipeline,
    in_model=None,
    out_model=None,
    prefix: str = "/pipeline",
) -> Any:
    """Create a FastAPI application that serves *pipeline*.

    *in_model* and *out_model* specify the Pydantic request / response types.
    When omitted they are read from ``pipeline.in_type`` / ``pipeline.out_type``.

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

    in_model = in_model or dict
    out_model = out_model or dict

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        async with pipeline:
            yield

    app = fastapi.FastAPI(lifespan=lifespan)
    router = fastapi.APIRouter()

    @router.post("/submit")
    async def submit(payload: in_model = fastapi.Body()) -> out_model:  # type: ignore[valid-type]
        if not pipeline.available_slots():
            raise HTTPException(status_code=429, detail="Too many requests")
        return await pipeline.submit(payload)

    @router.post("/bulk/submit")
    async def bulk_submit(payload: list[in_model] = fastapi.Body()) -> list[out_model]:  # type: ignore[valid-type]
        if not payload:
            return []

        if len(payload) > pipeline.available_slots():
            raise HTTPException(status_code=429, detail="Too many requests")

        return await asyncio.gather(*[pipeline.submit(item) for item in payload])

    @router.post("/bulk/submit_nowait")
    async def bulk_submit_nowait(payload: list[in_model] = fastapi.Body()) -> list[bool]:  # type: ignore[valid-type]
        if not payload:
            return []

        if len(payload) > pipeline.available_slots():
            raise HTTPException(status_code=429, detail="Too many requests")

        return await asyncio.gather(*[pipeline.submit_nowait(item) for item in payload])

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
