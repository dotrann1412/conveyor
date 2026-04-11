"""YOLO object detection served as a pipeline.

Pipeline:  preprocess (CPU, 4 workers) → detect (GPU batch, 1 per GPU) → postprocess (CPU, 4 workers)

Run:
    pip install conveyor[server] ultralytics pillow
    uvicorn examples.yolo_detection:app
"""

from io import BytesIO

import uvicorn

from conveyor import (
    BatchConfig,
    BatchStage,
    Pipeline,
    Stage,
    StageConfig,
)
from conveyor.server import create_app


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field
from PIL import Image 
import httpx

class InferenceRequest(BaseModel):
    image_url: str = Field(default="https://ultralytics.com/images/bus.jpg")
    
class InferenceResponseTmp(InferenceRequest):
    loaded_image: Image.Image
    detections: list[dict] | None = None

    class Config:
        arbitrary_types_allowed = True

class InferenceResponse(InferenceRequest):
    detections: list[dict] = Field(default_factory=list)

# fastapi does not allow to create app with runtime type annotations, so we need to use these hacky decorators
def hacky_decorator(func):
    async def wrapper(payload: dict):
        return await func(InferenceRequest(**payload))
    return wrapper

def reversed_hacky_decorator(func):
    async def wrapper(payload: InferenceResponseTmp):
        return (await func(payload)).model_dump(mode="json")
    return wrapper

# hmm, this is a bit of a hack to get the request type from the function signature
@hacky_decorator
async def preprocess(request: InferenceRequest) -> InferenceResponseTmp:
    """Decode image bytes and resize for YOLO input."""
    
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=None),
        follow_redirects=True,
    ) as client:
        response = await client.get(request.image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB").resize((640, 640))

        return InferenceResponseTmp(
            image_url=request.image_url,
            loaded_image=img,
            detections=None,
        )


def make_yolo_batch(device_id: int):
    """Factory: load a YOLO model on a specific GPU and return batch inference fn."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.to(f"cuda:{device_id}")

    async def detect(batch: list[InferenceResponseTmp]) -> list[InferenceResponseTmp]:
        images = [item.loaded_image for item in batch]
        results = model(images, verbose=False)
        for item, result in zip(batch, results):
            item.detections = [
                {
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                }
                for box in result.boxes
            ]
        return batch

    return detect

# f*cking hacky, but it works
@reversed_hacky_decorator
async def postprocess(item: InferenceResponseTmp) -> InferenceResponse:
    """Extract detection results."""
    return InferenceResponse(
        image_url=item.image_url,
        detections=item.detections or [],
    )


# ---------------------------------------------------------------------------
# Pipeline & app
# ---------------------------------------------------------------------------

DEVICE_IDS = [0]  # adjust for multi-GPU: [0, 1, 2, 3]

pipeline = Pipeline(
    stages=[
        Stage(preprocess, StageConfig(workers=4, stage_name="preprocess")),
        BatchStage.from_factory(
            fn_factory=make_yolo_batch,
            device_ids=DEVICE_IDS,
            batch_config=BatchConfig(max_batch_size=16, timeout_s=0.05),
            stage_config=StageConfig(stage_name="detect"),
        ),
        Stage(postprocess, StageConfig(workers=4, stage_name="postprocess")),
    ]
)

app = create_app(pipeline, prefix="/yolo")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# test the app
'''
curl -X POST http://localhost:8000/yolo/submit -H "Content-Type: application/json" --data-raw '{"image_url": "https://ultralytics.com/images/bus.jpg"}'
'''