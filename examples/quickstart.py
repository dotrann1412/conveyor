"""Minimal example — no GPU or ML dependencies required.

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


async def preprocess(data: str) -> str:
    await asyncio.sleep(0.02)
    return data.upper()


async def model_infer(batch: list[str], progress=None) -> list[str]:
    steps = 10
    for step in range(steps):
        await asyncio.sleep(0.01)
        if progress:
            progress(step + 1, steps)
    return [f"[result:{item}]" for item in batch]


async def postprocess(data: str) -> str:
    await asyncio.sleep(0.02)
    return f"done:{data}"


async def main():
    pipeline = Pipeline(
        stages=[
            Stage(preprocess, StageConfig(workers=4, stage_name="preprocess")),
            BatchStage(
                model_infer,
                BatchConfig(max_batch_size=8, timeout_s=0.05),
                StageConfig(workers=1, stage_name="model"),
            ),
            Stage(postprocess, StageConfig(workers=4, stage_name="postprocess")),
        ]
    )

    async with pipeline:
        t0 = time.perf_counter()
        tasks = [pipeline.submit(f"request-{i}") for i in range(20)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        for r in results[:5]:
            print(r)

        print(f"... ({len(results)} total in {elapsed:.3f}s)")


if __name__ == "__main__":
    asyncio.run(main())
