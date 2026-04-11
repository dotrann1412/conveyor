"""Face recognition pipeline: detect faces → extract embeddings → match.

Pipeline:  detect (GPU) → embed (GPU batch) → match (CPU)

Run:
    pip install conveyor insightface onnxruntime-gpu numpy pillow
    python examples/face_recognition.py
"""

import asyncio
import numpy as np

from conveyor import (
    BatchConfig,
    BatchStage,
    Pipeline,
    Stage,
    StageConfig,
)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def make_face_detector(device_id: int):
    """Load face detection model on a specific GPU."""
    import insightface

    det = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=[("CUDAExecutionProvider", {"device_id": device_id})],
    )
    det.prepare(ctx_id=device_id, det_size=(640, 640))

    async def detect(image: np.ndarray) -> dict:
        faces = det.get(image)
        return {"image": image, "faces": faces}

    return detect


def make_embedding_extractor(device_id: int):
    """Batch-extract face embeddings on a specific GPU."""

    async def extract(batch: list[dict]) -> list[dict]:
        for item in batch:
            embeddings = []
            for face in item["faces"]:
                embeddings.append(face.embedding)
            item["embeddings"] = embeddings
        return batch

    return extract


KNOWN_DB: dict[str, np.ndarray] = {}  # name → embedding


async def match_faces(item: dict) -> dict:
    """Match extracted embeddings against a known database."""
    matches = []
    for emb in item.get("embeddings", []):
        best_name, best_score = "unknown", 0.0
        for name, known_emb in KNOWN_DB.items():
            score = float(np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb)))
            if score > best_score:
                best_name, best_score = name, score
        matches.append({"name": best_name, "score": best_score})
    return {"faces": len(item["faces"]), "matches": matches}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DEVICE_IDS = [0]

pipeline = Pipeline(
    stages=[
        Stage.from_factory(
            fn_factory=make_face_detector,
            device_ids=DEVICE_IDS,
            config=StageConfig(stage_name="detect"),
        ),
        BatchStage.from_factory(
            fn_factory=make_embedding_extractor,
            device_ids=DEVICE_IDS,
            batch_config=BatchConfig(max_batch_size=8, timeout_s=0.1),
            stage_config=StageConfig(stage_name="embed"),
        ),
        Stage(match_faces, StageConfig(workers=4, stage_name="match")),
    ]
)


async def main():
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    async with pipeline:
        result = await pipeline.submit(dummy_image)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
