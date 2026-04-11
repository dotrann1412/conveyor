"""YOLO object detection served as a pipeline.

Pipeline:  preprocess (CPU, 4 workers) → detect (GPU batch, 1 per GPU) → postprocess (CPU, 4 workers)

Run:
    pip install conveyor[server] ultralytics pillow
    uvicorn examples.yolo_detection:app
"""

import asyncio
import io
from contextlib import asynccontextmanager

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

async def preprocess(raw_bytes: bytes) -> dict:
    """Decode image bytes and resize for YOLO input."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((640, 640))
    return {"image": img, "original_bytes": raw_bytes}


def make_yolo_batch(device_id: int):
    """Factory: load a YOLO model on a specific GPU and return batch inference fn."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.to(f"cuda:{device_id}")

    async def detect(batch: list[dict]) -> list[dict]:
        images = [item["image"] for item in batch]
        results = model(images, verbose=False)
        for item, result in zip(batch, results):
            item["detections"] = [
                {
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                }
                for box in result.boxes
            ]
        return batch

    return detect


async def postprocess(item: dict) -> list[dict]:
    """Extract detection results."""
    return item["detections"]


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
