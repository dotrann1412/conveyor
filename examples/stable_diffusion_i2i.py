"""Stable Diffusion image-to-image editing with simulated network I/O.

3-stage pipeline:
    download (CPU, simulated) → img2img edit (GPU with progress) → upload (CPU, simulated)

The download and upload stages simulate real network latency (1-3s each),
demonstrating how Conveyor hides I/O wait behind GPU work.

Run:
    pip install conveyor diffusers torch accelerate pillow
    python examples/stable_diffusion_i2i.py
"""

import asyncio
import io
import os
import uuid
import time
import random
import logging
import glob
import torch
from PIL import Image

from conveyor import (
    Pipeline,
    Stage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

async def download_image(request: dict) -> dict:
    """Simulate downloading the source image from a remote URL."""
    source = request["source"]

    latency = 1 + random.random() * 2
    logger.info(f"Downloading image from {source} (simulated {latency:.1f}s)")
    await asyncio.sleep(latency)

    if os.path.isfile(source):
        img = Image.open(source).convert("RGB")
    else:
        img = Image.new("RGB", (512, 512), color=(
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200),
        ))

    # in production, we can add some extra steps here like 
    # downloading lora and/or loading it into memory and wait

    request["image"] = img.resize((512, 512))
    return request


def make_img2img_stage(device_id: int):
    """Load a Stable Diffusion img2img pipeline on a specific GPU."""
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(f"cuda:{device_id}")

    async def edit(request: dict) -> dict:
        prompt = request["prompt"]
        init_image = request["image"]
        num_steps = request.get("num_steps", 30)
        strength = request.get("strength", 0.75)
        guidance_scale = request.get("guidance_scale", 7.5)

        loop = asyncio.get_event_loop()

        @torch.no_grad()
        def run():
            return pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        request["result_image"] = await loop.run_in_executor(None, run)
        del request["image"]
        return request

    return edit


async def upload_result(result: dict) -> dict:
    """Encode the edited image and simulate CDN upload."""
    img: Image.Image = result["result_image"]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result["image_bytes"] = buf.getvalue()
    result["size"] = len(result["image_bytes"])
    del result["result_image"]

    latency = 1 + random.random() * 2
    logger.info(f"Uploading {result['size']} bytes to CDN (simulated {latency:.1f}s)")
    await asyncio.sleep(latency)

    result["url"] = f"https://cdn.example.com/edited_{uuid.uuid4().hex}.png"
    logger.info(f"Upload complete: {result['url']}")
    return result

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DEVICE_IDS = [0]  # [0, 1] for two GPUs

PROMPT = "a person wearing stylish glasses, high quality, detailed face"

inputs = [
    {"source": path,  "prompt": PROMPT}
    for path in glob.glob("examples/images/*.jpg")
]


async def main():
    pipeline = Pipeline(
        stages=[
            Stage([download_image] * 4, queue_size_per_worker=4, stage_name="download"),
            Stage([make_img2img_stage(did) for did in DEVICE_IDS], queue_size_per_worker=4, stage_name="img2img"),
            Stage([upload_result] * 4, queue_size_per_worker=4, stage_name="upload"),
        ],
        name="stable-diffusion-i2i",
    )

    # --- Pipeline run ---
    async with pipeline:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[pipeline.submit(req) for req in inputs])
        pipeline_time = time.perf_counter() - t0

        for i, r in enumerate(results):
            print(f"[{i}] {r['prompt'][:50]}... → {r['url']} ({r['size']} bytes)")
            Image.open(io.BytesIO(r["image_bytes"])).save(f"i2i_output_{i}.png")

    print(f"\nPipeline: {pipeline_time:.2f}s total, {pipeline_time / len(inputs):.2f}s avg")

    # --- Sequential baseline ---
    download_fn = download_image
    edit_fn = make_img2img_stage(DEVICE_IDS[0])
    upload_fn = upload_result

    seq_time = 0.0
    for i, req in enumerate(inputs):
        t0 = time.perf_counter()
        r = await upload_fn(await edit_fn(await download_fn(req)))
        seq_time += time.perf_counter() - t0

    print(f"Sequential: {seq_time:.2f}s total, {seq_time / len(inputs):.2f}s avg")
    print(f"Speedup: {seq_time / pipeline_time:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
