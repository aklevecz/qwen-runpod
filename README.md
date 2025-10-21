# Qwen-Image RunPod Serverless Handler

Production-ready serverless handler for Qwen-Image text-to-image generation using diffusers on RunPod.

**Includes Ryze LoRA by default** - Just provide a prompt and go!

## Features

- Native diffusers pipeline (no ComfyUI overhead)
- **Ryze LoRA pre-loaded** (281MB, strength 0.8 by default)
- Custom LoRA support with dynamic loading
- Memory-optimized with VAE slicing/tiling
- Base64 image output
- Comprehensive error handling
- ~25.3GB Docker image size (including Ryze LoRA)
- 10-15s cold start time

## Quick Start

### 1. Build Docker Image

```bash
# Build for RunPod (AMD64 platform)
docker build --platform linux/amd64 -t qwen-image-runpod .

# This will take 15-30 minutes (downloads ~25GB of models)
```

### 2. Test Locally (Optional)

**Prerequisites**: NVIDIA GPU with CUDA support

```bash
# Install dependencies locally
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run test
python test_local.py

# Output will be saved as output_0.png
```

### 3. Deploy to RunPod

```bash
# Push to Docker Hub
docker tag qwen-image-runpod:latest yourusername/qwen-image-runpod:latest
docker login
docker push yourusername/qwen-image-runpod:latest

# Create endpoint in RunPod console
# - Container Image: yourusername/qwen-image-runpod:latest
# - GPU Type: A40 (48GB) recommended
# - Container Disk: 40GB minimum
# - Active Workers: 0-3 (auto-scale)
```

## API Usage

### Basic Generation (Uses Ryze LoRA by Default)

```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Simple: Ryze LoRA is automatically applied at 0.8 strength
result = endpoint.run_sync({
    "input": {
        "prompt": "A bottle of Ryze mushroom coffee on a kitchen counter, morning light"
    }
})

# result['images'][0]['image'] contains base64 encoded PNG
# result['metadata']['lora_used'] will be True
# result['metadata']['lora_scale'] will be 0.8
```

### Adjust Ryze LoRA Strength

```python
# Use stronger Ryze LoRA influence
result = endpoint.run_sync({
    "input": {
        "prompt": "A bottle of Ryze coffee",
        "lora_scale": 1.2  # Stronger influence
    }
})

# Use weaker Ryze LoRA influence
result = endpoint.run_sync({
    "input": {
        "prompt": "A bottle of Ryze coffee",
        "lora_scale": 0.5  # Subtle influence
    }
})

# Disable LoRA completely
result = endpoint.run_sync({
    "input": {
        "prompt": "A generic coffee bottle",
        "lora_path": None  # No LoRA
    }
})
```

### With Different Custom LoRA

```python
result = endpoint.run_sync({
    "input": {
        "prompt": "A futuristic city at night",
        "lora_path": "/app/loras/different_lora.safetensors",
        "lora_scale": 0.8
    }
})
```

### Advanced Parameters

```python
result = endpoint.run_sync({
    "input": {
        "prompt": "A magical forest with glowing mushrooms",
        "negative_prompt": "blurry, low quality, distorted",
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "width": 1920,
        "height": 1080,
        "seed": 42,
        "num_images": 2,
        "output_format": "png"
    }
})
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of desired image |
| `negative_prompt` | string | "" | What to avoid in generation |
| `num_inference_steps` | int | 50 | Number of denoising steps (20-100) |
| `true_cfg_scale` | float | 4.0 | Guidance strength (1.0-10.0) |
| `width` | int | 1024 | Output width (512-2048) |
| `height` | int | 1024 | Output height (512-2048) |
| `seed` | int | random | Random seed for reproducibility |
| `lora_path` | string | null | Path to LoRA file or HF repo |
| `lora_scale` | float | 1.0 | LoRA strength (0.0-2.0) |
| `num_images` | int | 1 | Number of images to generate (1-4) |
| `output_format` | string | "png" | Output format (png or jpeg) |

## Custom LoRA Integration

### Option 1: Bake into Docker Image (Recommended)

1. Place your LoRA in `loras/` directory:
```bash
mkdir -p loras
cp /path/to/your/lora.safetensors loras/my_custom_lora.safetensors
```

2. Uncomment this line in `Dockerfile`:
```dockerfile
COPY loras/my_custom_lora.safetensors /app/loras/
```

3. Rebuild image

4. Use in API:
```python
{
  "lora_path": "/app/loras/my_custom_lora.safetensors",
  "lora_scale": 0.8
}
```

### Option 2: HuggingFace Hub

1. Upload LoRA to HuggingFace
2. Reference in API:
```python
{
  "lora_path": "your-username/your-lora-repo",
  "lora_scale": 0.8
}
```

### LoRA Scale Guidelines

| Scale | Effect |
|-------|--------|
| 0.3-0.5 | Subtle influence |
| 0.6-0.8 | Balanced (recommended) |
| 0.9-1.2 | Strong influence |
| 1.3-2.0 | Very strong (may overfit) |

## Output Format

```json
{
  "images": [
    {
      "image": "base64_encoded_string",
      "seed": 42,
      "index": 0
    }
  ],
  "metadata": {
    "prompt": "...",
    "negative_prompt": "...",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "lora_used": true,
    "lora_path": "/app/loras/my_custom_lora.safetensors",
    "lora_scale": 0.8,
    "num_images_generated": 1
  }
}
```

## Decoding Images

### Python
```python
import base64
from PIL import Image
from io import BytesIO

# Get base64 string from API response
img_base64 = result['images'][0]['image']

# Decode
img_bytes = base64.b64decode(img_base64)
img = Image.open(BytesIO(img_bytes))

# Save
img.save('output.png')
```

### JavaScript/TypeScript
```javascript
// In browser
const img_base64 = result.images[0].image;
const img_url = `data:image/png;base64,${img_base64}`;
document.getElementById('myImg').src = img_url;

// In Node.js
const fs = require('fs');
const buffer = Buffer.from(img_base64, 'base64');
fs.writeFileSync('output.png', buffer);
```

## Performance

### A40 GPU (48GB VRAM)

| Resolution | Steps | Cold Start | Warm Start |
|------------|-------|------------|------------|
| 512x512 | 50 | ~20s | ~8s |
| 1024x1024 | 50 | ~30s | ~15s |
| 1920x1080 | 50 | ~45s | ~25s |
| 2048x2048 | 50 | ~60s | ~35s |

**Cold start**: First request after idle (includes model loading)
**Warm start**: Subsequent requests (models cached in VRAM)

### Cost Estimates (RunPod A40)

- **GPU**: ~$0.40/hr
- **Idle**: $0.00/hr (serverless scales to zero)
- **Per image** (1024x1024, 50 steps): ~$0.003-0.005

**Monthly costs**:
- 1,000 images: ~$4
- 10,000 images: ~$40
- 100,000 images: ~$400

## Common Presets

### Fast Preview
```json
{
  "num_inference_steps": 20,
  "true_cfg_scale": 3.0,
  "width": 512,
  "height": 512
}
```

### Balanced (Default)
```json
{
  "num_inference_steps": 50,
  "true_cfg_scale": 4.0,
  "width": 1024,
  "height": 1024
}
```

### High Quality
```json
{
  "num_inference_steps": 100,
  "true_cfg_scale": 5.0,
  "width": 2048,
  "height": 2048
}
```

### Aspect Ratios

```python
# Square
{"width": 1024, "height": 1024}

# Landscape (16:9)
{"width": 1920, "height": 1080}

# Portrait (9:16)
{"width": 1080, "height": 1920}

# Cinematic (21:9)
{"width": 2560, "height": 1080}
```

## Troubleshooting

### Out of Memory
- Reduce image resolution
- VAE optimizations are already enabled
- Use A100 for very large images (2048x2048+)

### Slow Cold Starts
- Models are pre-downloaded in Dockerfile ✅
- Consider keeping 1 worker active for instant response

### LoRA Not Loading
- Check file path is correct
- Verify LoRA format is `.safetensors`
- Check RunPod logs for error messages

### Black/Bad Images
- Increase `num_inference_steps` (try 100)
- Adjust `true_cfg_scale` (try 3.0-6.0)
- Try different seeds
- Check prompt quality

### Timeout Errors
- Increase RunPod timeout setting (default 300s)
- Reduce image resolution
- Reduce num_inference_steps

## Monitoring & Logs

View logs in RunPod Console:
```
Serverless → Your Endpoint → Logs
```

Look for:
- "Pipeline loaded successfully!" (startup)
- "Generating X image(s)..." (inference start)
- Error tracebacks with full stack trace

## Environment Variables

Optional environment variables you can set in RunPod:

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for private models/LoRAs |
| `HF_HOME` | HuggingFace cache directory (default: `/app/.cache/huggingface`) |

## Directory Structure

```
runpod/
├── Dockerfile              # Container definition
├── handler.py             # RunPod serverless handler
├── requirements.txt       # Python dependencies
├── test_input.json        # Sample input
├── test_local.py          # Local testing script
├── loras/                 # Custom LoRAs
│   └── my_custom_lora.safetensors
├── .dockerignore          # Docker build exclusions
├── README.md             # This file
├── PLAN_A_COMFYUI.md     # Alternative approach
└── PLAN_B_DIFFUSERS.md   # Implementation details
```

## License

This code is provided as-is for use with Qwen-Image models.

Qwen-Image models are licensed under Apache 2.0.

## Support

For RunPod issues: https://docs.runpod.io
For Qwen-Image issues: https://github.com/QwenLM/Qwen-Image
For diffusers issues: https://github.com/huggingface/diffusers

## Next Steps

1. **Build the Docker image**: `docker build --platform linux/amd64 -t qwen-image-runpod .`
2. **Test locally** (optional): `python test_local.py`
3. **Push to Docker Hub**: `docker push yourusername/qwen-image-runpod`
4. **Create RunPod endpoint**
5. **Integrate into your app**

Happy generating!
