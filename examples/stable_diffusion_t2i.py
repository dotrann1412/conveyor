"""Stable Diffusion text to image generation with progress tracking.

A 2-stage pipeline: denoise (GPU with progress) → decode + save (CPU)

Run:
    pip install conveyor diffusers torch accelerate pillow
    python examples/stable_diffusion_t2i.py
"""

import asyncio
import io
import uuid
import time

import torch
from PIL import Image

from conveyor import (
    Pipeline,
    ProgressReporter,
    Stage,
    StageConfig,
)

import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def make_diffusion_stage(device_id: int):
    """Load a Stable Diffusion pipeline on a specific GPU."""
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(f"cuda:{device_id}")

    async def generate(request: dict, progress: ProgressReporter) -> dict:
        prompts = request["prompt"]

        num_steps = request.setdefault("num_steps", 30)
        guidance_scale = request.setdefault("guidance_scale", 7.5)
        width = request.setdefault("width", 512)
        height = request.setdefault("height", 512)

        # support by diffusers
        def callback(pipe, step, timestep, kwargs):
            if progress:
                progress(step + 1, num_steps)
            return kwargs

        loop = asyncio.get_event_loop()

        @torch.no_grad()
        def run():
            return pipe(
                prompts,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                callback_on_step_end=callback,
            ).images[0]

        request["image"] = await loop.run_in_executor(None, run)
        return request

    return generate

# this step can be uploading the image to cdn and save the url to db
async def save_image(result: dict) -> dict:
    """Encode the generated image to PNG bytes."""
    img: Image.Image = result["image"]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result["image_bytes"] = buf.getvalue()
    result["size"] = len(result["image_bytes"])
    del result["image"]

    logger.info(f"Uploading image to CDN: {len(result['image_bytes'])} bytes")
    await asyncio.sleep(random.random() * 5) # simulate network latency

    logger.info(f"Uploaded image to CDN: {len(result['image_bytes'])} bytes") 
    result["url"] = f"https://cdn.example.com/image_{uuid.uuid4().hex}.png"

    return result

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DEVICE_IDS = [0]  # [0, 1] for two GPUs


async def main():
    pipeline = Pipeline(
        stages=[
            Stage.from_factory(
                fn_factory=make_diffusion_stage,
                device_ids=DEVICE_IDS,
                config=StageConfig(workers=1, stage_name="denoise"),
            ),
            Stage(save_image, StageConfig(workers=2, stage_name="save")),
        ]
    )

    inputs = [
        {"prompt": "a cat sitting on a rainbow, digital art"},
        {"prompt": "a futuristic city skyline at sunset, cyberpunk style"},
        {"prompt": "an astronaut riding a horse on mars, photorealistic"},
        {"prompt": "a cozy cabin in a snowy forest, warm lighting, oil painting"},
        {"prompt": "underwater coral reef with tropical fish, macro photography"},
        {"prompt": "a steampunk clocktower surrounded by airships, concept art"},
        {"prompt": "a japanese garden in autumn with a red bridge, watercolor"},
        {"prompt": "a wolf howling at a full moon on a mountain peak, dramatic lighting"},
        {"prompt": "a medieval castle on a floating island in the clouds, fantasy art"},
        {"prompt": "a neon-lit ramen shop on a rainy tokyo street at night, anime style"},
    ]

    pipeline_processing_time = 0

    async with pipeline:
        t_start = time.perf_counter()

        tasks = [
            pipeline.submit(input)
            for input in inputs
        ]

        results = await asyncio.gather(*tasks)

        for i, r in enumerate(results):
            print(f"Prompt: {r['prompt']}, Image size: {r['size']} bytes, URL: {r['url']}")
            Image.open(io.BytesIO(r["image_bytes"])).save(f"image_{i}.png")

        end_time = time.perf_counter()
        pipeline_processing_time = end_time - t_start
        print(f"Pipeline processing time: {pipeline_processing_time:.2f} seconds, average time per request: {pipeline_processing_time / len(inputs):.2f} seconds")

    del pipeline

    # test the sequential pipeline
    model_fn = make_diffusion_stage(DEVICE_IDS[0])
    save_fn = save_image

    sequential_processing_time = 0
    for i, input in enumerate(inputs):
        t_start = time.perf_counter()
        result = await save_fn(await model_fn(input, progress=None))
        end_time = time.perf_counter()
        sequential_processing_time += end_time - t_start
        print(f"Sequential processing time: {sequential_processing_time:.2f} seconds, average time per request: {sequential_processing_time / (i + 1):.2f} seconds")

    print(f"Sequential processing time: {sequential_processing_time:.2f} seconds, average time per request: {sequential_processing_time / len(inputs):.2f} seconds")

    print("--------------------------------")
    print(f"Pipeline processing time: {pipeline_processing_time:.2f} seconds, average time per request: {pipeline_processing_time / len(inputs):.2f} seconds")
    print(f"Sequential processing time: {sequential_processing_time:.2f} seconds, average time per request: {sequential_processing_time / len(inputs):.2f} seconds")
    print("--------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
