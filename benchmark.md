# Benchmark: Stable Diffusion Text-to-Image

## Setup

- **Model**: `runwayml/stable-diffusion-v1-5` (float16)
- **GPU**: 1x NVIDIA RTX 4060
- **Pipeline**: 2 stages — `denoise` (GPU, 1 worker) → `save` (CPU, 2 workers, simulated CDN upload with 0-5s random latency)
- **Inference**: 30 denoising steps per image, 512x512, guidance scale 7.5
- **Requests**: 10 unique prompts submitted concurrently

## Results

| Mode | Total time | Avg per request | Speedup |
|---|---|---|---|
| **Conveyor pipeline** | **47.32s** | **4.73s** | **1.48x** |
| Sequential | 70.14s | 7.01s | 1.0x |

### Where the speedup comes from

In sequential mode, the GPU sits idle while each image is saved/uploaded (~0-5s per image). Over 10 requests, that idle time adds up to ~23s of wasted GPU time.

Conveyor overlaps the save stage with the next denoise — while image N is uploading to CDN, the GPU is already generating image N+1:

```
Sequential:
  [denoise A][save A][denoise B][save B][denoise C][save C] ...
                     ↑ GPU idle          ↑ GPU idle

Conveyor:
  [denoise A][denoise B][denoise C] ...
             [save A]   [save B]   [save C] ...
             ↑ GPU never waits for save
```

### Test prompts

| # | Prompt |
|---|---|
| 1 | a cat sitting on a rainbow, digital art |
| 2 | a futuristic city skyline at sunset, cyberpunk style |
| 3 | an astronaut riding a horse on mars, photorealistic |
| 4 | a cozy cabin in a snowy forest, warm lighting, oil painting |
| 5 | underwater coral reef with tropical fish, macro photography |
| 6 | a steampunk clocktower surrounded by airships, concept art |
| 7 | a japanese garden in autumn with a red bridge, watercolor |
| 8 | a wolf howling at a full moon on a mountain peak, dramatic lighting |
| 9 | a medieval castle on a floating island in the clouds, fantasy art |
| 10 | a neon-lit ramen shop on a rainy tokyo street at night, anime style |

### Reproduce

```bash
pip install conveyor diffusers torch accelerate pillow
python examples/stable_diffusion_t2i.py
```
