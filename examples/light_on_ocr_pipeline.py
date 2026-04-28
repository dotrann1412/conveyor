'''
dependencies:
----
conveyor @ git+https://github.com/dotrann1412/conveyor@a5382d19b65477d813ef8c0972e5323e351536de
httpx==0.28.1
pillow==12.2.0
pypdfium2==5.7.1
tokenizers==0.22.2
torch==2.11.0
torchvision==0.26.0
transformers @ git+https://github.com/huggingface/transformers@4aba7167e328965caadcdfc6834b982037889f86
fastapi==0.136.1
'''

from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import torch
from typing import Any
from PIL import Image
import httpx
from io import BytesIO
import base64
import re
import logging

logging.basicConfig(level=logging.INFO)

def make_generation_stage(cuda_index: int | None = None):
    cuda_index = cuda_index or 0
    device = (
        "mps" if torch.backends.mps.is_available() 
        else f"cuda:{cuda_index}" if torch.cuda.is_available() 
        else "cpu"
    )

    dtype = torch.float32 if device == "mps" else torch.bfloat16

    model = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-2-1B-bbox", torch_dtype=dtype).to(device)
    processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B-bbox")

    @torch.no_grad()
    def generate(req: dict[str, Any]):
        # this can be in a separate stage, maybe the preprocess
        inputs = processor.apply_chat_template(
            req.pop('messages'),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(device=device, dtype=dtype) 
            if v.is_floating_point() 
            else v.to(device) 
            for k, v in inputs.items()
        }

        output_ids = model.generate(**inputs, max_new_tokens=1024)
        
        # this can be located in the post process phase
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

        return {
            **req,
            "output": output_text,
        }

    return generate

async def initialize_stage(req: dict[str, Any]):
    image_url = req.get("image_url") 
    assert image_url, "image_url is required"

    if image_url.startswith("http"):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url, follow_redirects=True)
            image = Image.open(BytesIO(response.content))

    else:
        pat = re.compile(r"data:image/(.*);base64,(.*)")
        if match := pat.match(image_url):
            image = Image.open(BytesIO(base64.b64decode(match.group(2))))
        else:
            image = Image.open(image_url)

    image = image.convert("RGB")
    max_w, max_h = 1540, 1540

    w, h = image.size
    scale_w, scale_h = w / max_w, h / max_h
    scale = max(scale_w, scale_h)
    nw, nh = int(w / scale), int(h / scale)
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_uri = base64.b64encode(buffered.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_uri}"
                    }
                }
            ]
        }
    ]

    return {"messages": messages, **req}

def finalize_stage(req: dict[str, Any]):
    # parse the text output here, 
    # and/or render it directly into the image and response
    return {
        "output": req["output"].strip()
    }

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, Body
from conveyor import Pipeline, Stage

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with Pipeline(
        stages=[
            Stage([initialize_stage] * 8, queue_size_per_worker=100),
            Stage([make_generation_stage()], queue_size_per_worker=100),
            Stage([finalize_stage], queue_size_per_worker=100),
        ]
    ) as app.state.pipeline:
        yield

def depends_pipeline(req: Request):
    return req.app.state.pipeline

app = FastAPI(lifespan=lifespan)

@app.post("/extract")
async def extract(
    pipeline: Pipeline = Depends(depends_pipeline), 
    image_url: str = Body(...)
):
    return await pipeline.submit(
        {
            "image_url": image_url,
        }
    )
