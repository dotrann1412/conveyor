"""YOLO object detection served as a pipeline.

Pipeline:  preprocess (CPU, 4 workers) → detect (GPU batch, 1 per GPU) → postprocess (CPU, 4 workers)

Run:
    pip install conveyor[server] ultralytics pillow
    uvicorn examples.yolo_detection:app
"""

from io import BytesIO
import asyncio
import uvicorn

from conveyor import (
    BatchStage,
    Pipeline,
    Stage,
)
from conveyor.server import create_app
import logging

logging.basicConfig(level=logging.INFO)

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field
from PIL import Image 
import httpx

class InferenceRequest(BaseModel):
    image_url: str = Field(default="https://ultralytics.com/images/bus.jpg")
    
class InferenceResponseInter(InferenceRequest):
    loaded_image: Image.Image
    detections: list[dict] | None = None

    class Config:
        arbitrary_types_allowed = True

class InferenceResponse(InferenceRequest):
    detections: list[dict] = Field(default_factory=list)

async def preprocess(request: InferenceRequest) -> InferenceResponseInter:
    """Decode image bytes and resize for YOLO input."""
    
    loop = asyncio.get_event_loop()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=None),
        follow_redirects=True,
    ) as client:
        logger.info("Loading image from %s", request.image_url)
        response = await client.get(request.image_url)

        def load_image():
            return Image.open(BytesIO(response.content)).convert("RGB").resize((640, 640))

        img = await loop.run_in_executor(None, load_image)

        return InferenceResponseInter(
            image_url=request.image_url,
            loaded_image=img,
            detections=None,
        )


def make_yolo_batch(device_id: int):
    """Factory: load a YOLO model on a specific GPU and return batch inference fn."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.to(f"cuda:{device_id}")

    def detect(batch: list[InferenceResponseInter]) -> list[InferenceResponseInter]:
        logger.info("Detecting %d images", len(batch))
        images = [item.loaded_image for item in batch]
        
        logger.info("Detecting %d images", len(images))

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

async def postprocess(item: InferenceResponseInter) -> InferenceResponse:
    """Extract detection results."""
    logger.info("Postprocessing %d detections", len(item.detections or []))
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
        Stage([preprocess] * 8, queue_size_per_worker=4, stage_name="preprocess"),
        BatchStage(
            [make_yolo_batch(did) for did in DEVICE_IDS],
            worker_queue_size=4,
            max_batch_size=32,
            timeout_s=2,
            stage_name="detect",
        ),
        Stage([postprocess] * 4, queue_size_per_worker=4, stage_name="postprocess"),
    ],
    name="yolo-detection",
)

app = create_app(
    pipeline, 
    in_model=InferenceRequest, 
    out_model=InferenceResponse, 
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# test the app
'''
curl -X POST http://localhost:8000/yolo/submit -H "Content-Type: application/json" --data-raw '{"image_url": "https://ultralytics.com/images/bus.jpg"}'
'''