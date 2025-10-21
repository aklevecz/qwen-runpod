# ✅ RunPod Handler Setup Complete

## What's Configured

Your Ryze LoRA (`ryze-lora.safetensors`, 281MB) is now the **default** for all image generations.

## How It Works

### Simple Usage (Ryze LoRA Auto-Applied)
```python
# Just provide a prompt - Ryze LoRA @ 0.8 strength is automatically used
{
  "input": {
    "prompt": "A bottle of Ryze mushroom coffee"
  }
}
```

### Adjust Strength
```python
{
  "input": {
    "prompt": "A bottle of Ryze coffee",
    "lora_scale": 1.2  # 0.0-2.0
  }
}
```

### Disable LoRA
```python
{
  "input": {
    "prompt": "Generic coffee",
    "lora_path": null
  }
}
```

## Files Created

```
runpod/
├── Dockerfile                      # Includes Ryze LoRA
├── handler.py                      # Defaults to Ryze LoRA @ 0.8
├── requirements.txt                # All dependencies
├── test_input.json                 # Full test with Ryze LoRA
├── test_simple.json                # Minimal test (auto-uses Ryze LoRA)
├── test_local.py                   # Local testing script
├── loras/
│   └── ryze-lora.safetensors      # 281MB - Your trained LoRA
├── .dockerignore                   # Build optimization
├── README.md                       # Complete documentation
└── SETUP_COMPLETE.md              # This file

```

## Quick Test Locally (Optional)

```bash
cd runpod

# Install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test with simple prompt (auto-uses Ryze LoRA)
python test_local.py

# Output will be saved as output_0.png
```

## Build & Deploy

### 1. Build Docker Image
```bash
cd runpod
docker build --platform linux/amd64 -t ryze-qwen-image .

# Takes 15-30 minutes (downloads ~25GB models + 281MB LoRA)
```

### 2. Push to Docker Hub
```bash
docker tag ryze-qwen-image yourusername/ryze-qwen-image:latest
docker login
docker push yourusername/ryze-qwen-image:latest
```

### 3. Deploy to RunPod
- Go to RunPod Console → Serverless
- Create New Endpoint
- **Container Image**: `yourusername/ryze-qwen-image:latest`
- **GPU Type**: A40 (48GB VRAM) recommended
- **Container Disk**: 40GB minimum
- **Workers**: 0-3 (auto-scale)

### 4. Use the API
```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Simple - Ryze LoRA automatically applied
result = endpoint.run_sync({
    "input": {
        "prompt": "A bottle of Ryze mushroom coffee on a wooden table, natural lighting, product photography, highly detailed"
    }
})

# Decode and save
import base64
from PIL import Image
from io import BytesIO

img_data = base64.b64decode(result['images'][0]['image'])
img = Image.open(BytesIO(img_data))
img.save('ryze_output.png')
```

## Default Settings

| Parameter | Default Value |
|-----------|---------------|
| `lora_path` | `/app/loras/ryze-lora.safetensors` |
| `lora_scale` | `0.8` |
| `num_inference_steps` | `50` |
| `true_cfg_scale` | `4.0` |
| `width` | `1024` |
| `height` | `1024` |
| `output_format` | `png` |

## Performance Estimates (A40 GPU)

| Resolution | Steps | Cold Start | Warm Start | Cost/Image |
|------------|-------|------------|------------|------------|
| 512x512 | 50 | ~20s | ~8s | $0.002 |
| 1024x1024 | 50 | ~30s | ~15s | $0.003 |
| 1920x1080 | 50 | ~45s | ~25s | $0.005 |
| 2048x2048 | 50 | ~60s | ~35s | $0.007 |

## LoRA Strength Guide

| Scale | Effect | When to Use |
|-------|--------|-------------|
| 0.3-0.5 | Subtle | Slight Ryze branding hints |
| 0.6-0.8 | **Balanced** (default) | Natural Ryze product look |
| 0.9-1.2 | Strong | Prominent Ryze characteristics |
| 1.3-2.0 | Very Strong | Maximum Ryze style influence |

## Next Steps

1. ✅ **Ryze LoRA is configured as default**
2. 🔨 **Build the Docker image** (takes 15-30 min)
3. 🚀 **Deploy to RunPod**
4. 🎨 **Generate images with just a prompt!**

---

**Everything is set up and ready to build!** 🎉
