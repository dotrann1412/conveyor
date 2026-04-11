"""Stable Diffusion image generation with progress tracking.

Pipeline:  encode prompt (CPU) → denoise (GPU with progress) → decode + save (CPU)

Run:
    pip install conveyor diffusers torch accelerate pillow
    python examples/stable_diffusion.py
"""

import asyncio
import io

import torch
from PIL import Image

from conveyor import (
    BatchConfig,
    BatchStage,
    Pipeline,
    ProgressReporter,
    Stage,
    StageConfig,
)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

async def encode_prompt(request: dict) -> dict:
    """Validate and prepare the generation request."""
    request.setdefault("num_steps", 30)
    request.setdefault("guidance_scale", 7.5)
    request.setdefault("width", 512)
    request.setdefault("height", 512)
    return request


def make_diffusion_stage(device_id: int):
    """Load a Stable Diffusion pipeline on a specific GPU."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(f"cuda:{device_id}")

    async def generate(batch: list[dict], progress: ProgressReporter) -> list[dict]:
        prompts = [req["prompt"] for req in batch]
        num_steps = batch[0]["num_steps"]

        # support by diffusers
        def callback(pipe, step, timestep, kwargs):
            progress(step + 1, num_steps)
            return kwargs

        loop = asyncio.get_event_loop()

        @torch.no_grad()
        def run():
            return pipe(
                prompts,
                num_inference_steps=num_steps,
                guidance_scale=batch[0]["guidance_scale"],
                width=batch[0]["width"],
                height=batch[0]["height"],
                callback_on_step_end=callback,
            ).images

        images = await loop.run_in_executor(None, run)

        for req, img in zip(batch, images):
            req["image"] = img

        return batch

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
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DEVICE_IDS = [0]  # [0, 1] for two GPUs

pipeline = Pipeline(
    stages=[
        Stage(encode_prompt, StageConfig(workers=2, stage_name="encode")),
        BatchStage.from_factory(
            fn_factory=make_diffusion_stage,
            device_ids=DEVICE_IDS,
            batch_config=BatchConfig(max_batch_size=4, timeout_s=1.0),
            stage_config=StageConfig(stage_name="denoise"),
        ),
        Stage(save_image, StageConfig(workers=2, stage_name="save")),
    ]
)


async def main():
    async with pipeline:
        tasks = [
            pipeline.submit({"prompt": "a cat sitting on a rainbow"}),
            pipeline.submit({"prompt": "a futuristic city at sunset"}),
        ]

        results = await asyncio.gather(*tasks)

        for r in results:
            print(f"Prompt: {r['prompt']}, Image size: {r['size']} bytes")


if __name__ == "__main__":
    asyncio.run(main())
