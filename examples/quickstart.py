"""Minimal example — no GPU or ML dependencies required.

This is a minimal example of 3-stage pipeline
- preprocess (CPU, 4 workers): processing time: 0.02s per item
- model (GPU, 1 worker, dynamic batching): processing time: 0.05s per batch
- postprocess (CPU, 4 workers): processing time: 0.02s per item

Run:
    pip install conveyor
    python examples/quickstart.py
"""

import asyncio
import time

from conveyor import (
    BatchConfig,
    BatchStage,
    Pipeline,
    Stage,
    StageConfig,
)

import logging 
logging.basicConfig(level=logging.INFO)

async def preprocess(data: str) -> str:
    await asyncio.sleep(0.02)
    return data.upper()

async def model_batch_infer(batch: list[str], progress=None) -> list[str]:
    steps = 10

    for step in range(steps):
        await asyncio.sleep(0.01) # simulate a batch of 4 requests

        if progress:
            progress(step + 1, steps)

    return [f"[result:{item}]" for item in batch]


def model_infer(data: str, progress=None) -> str:
    steps = 10

    for step in range(steps):
        time.sleep(0.01 / 4) # simulate a single request

        if progress:
            progress(step + 1, steps)

    return f"[result:{data}]"

async def postprocess(data: str) -> str:
    await asyncio.sleep(0.02)
    return f"done:{data}"


async def main():
    pipeline = Pipeline(
        stages=[
            Stage(preprocess, StageConfig(workers=4, stage_name="preprocess")),
            BatchStage(
                model_batch_infer,
                BatchConfig(max_batch_size=8, timeout_s=0.05),
                StageConfig(workers=1, stage_name="model"),
            ),
            Stage(postprocess, StageConfig(workers=4, stage_name="postprocess")),
        ]
    )

    pipeline_processing_time = 0

    async with pipeline:
        t0 = time.perf_counter()
        tasks = [pipeline.submit(f"request-{i}") for i in range(20)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        for r in results[:5]:
            print(r)

        pipeline_processing_time = elapsed
        print(f"... ({len(results)} total in {pipeline_processing_time:.3f}s, average time per request: {pipeline_processing_time / len(results):.3f}s)")

    del pipeline

    # test the sequential pipeline
    preprocess_fn = preprocess
    model_fn = model_infer
    postprocess_fn = postprocess

    sequential_processing_time = 0

    for i in range(20):
        t_start = time.perf_counter()
        result = await postprocess_fn(model_fn(await preprocess_fn(f"request-{i}")))
        end_time = time.perf_counter()
        sequential_processing_time += end_time - t_start

    print(f"Sequential processing time: {sequential_processing_time:.3f}s, average time per request: {sequential_processing_time / 20:.3f}s")

    print("--------------------------------")
    print(f"Pipeline processing time: {pipeline_processing_time:.3f}s, average time per request: {pipeline_processing_time / 20:.3f}s")
    print(f"Sequential processing time: {sequential_processing_time:.3f}s, average time per request: {sequential_processing_time / 20:.3f}s")
    print("--------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
