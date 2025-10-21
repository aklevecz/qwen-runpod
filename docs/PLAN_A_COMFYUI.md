# Plan A: RunPod ComfyUI + Qwen-Image Handler

## Overview
Run ComfyUI as a background service with Qwen-Image models, exposing it via RunPod serverless handler. This approach allows you to use visual workflows and easily swap models/LoRAs through the UI.

## Architecture
```
┌─────────────────────────────────────┐
│   RunPod Serverless Container       │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │   handler.py │─▶│  ComfyUI    │ │
│  │   (RunPod)   │  │  Server     │ │
│  │              │  │  :8188      │ │
│  └──────────────┘  └─────────────┘ │
│         │                  │        │
│         │                  ▼        │
│         │          ┌─────────────┐ │
│         │          │   Models    │ │
│         │          │  - Qwen     │ │
│         │          │  - LoRA     │ │
│         │          └─────────────┘ │
└─────────────────────────────────────┘
```

## Pros & Cons

### Pros
- ✅ Visual workflow editor for development and testing
- ✅ Easy to switch models/workflows without code changes
- ✅ Access to custom nodes ecosystem
- ✅ Can test workflows locally before deployment
- ✅ Built-in image preview and debugging tools
- ✅ LoRA/ControlNet support through UI

### Cons
- ❌ Larger Docker image (~35GB+ with models and ComfyUI)
- ❌ More complex: Must run web server + handler
- ❌ Slower cold starts: ComfyUI startup overhead (15-30s)
- ❌ Harder to debug: Two-layer abstraction (RunPod → ComfyUI)
- ❌ Workflow JSON complexity: Exported workflows are verbose
- ❌ Memory overhead from running web server

## Directory Structure
```
runpod/
├── Dockerfile              # Multi-stage build with ComfyUI + models
├── handler.py             # RunPod serverless handler
├── comfy_api.py           # ComfyUI API interaction helpers
├── requirements.txt       # Python dependencies
├── entrypoint.sh          # Startup script (launch ComfyUI)
├── workflows/             # Example workflows
│   └── qwen_txt2img.json
├── builder/               # Model download scripts
│   └── download_models.py
├── test_input.json        # Sample input for local testing
└── README.md             # Deployment instructions
```

## Models Required

### Qwen-Image Models
Download to `ComfyUI/models/` subdirectories:

| Model | Size | Location |
|-------|------|----------|
| `qwen_image_fp8_e4m3fn.safetensors` | 20.4GB | `diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~15GB | `text_encoders/` |
| `qwen_image_vae.safetensors` | ~300MB | `vae/` |

**Total: ~36GB**

### Custom LoRA
- Place your trained LoRA in `ComfyUI/models/loras/`
- Name it something descriptive (e.g., `my_custom_lora.safetensors`)
- Reference in workflow with LoRA Loader node

## Download Sources
- **HuggingFace**: `Comfy-Org/Qwen-Image_ComfyUI`
- **ModelScope**: `Comfy-Org/Qwen-Image_ComfyUI`

## Implementation

### 1. Dockerfile

```dockerfile
# Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# Install ComfyUI requirements
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK
RUN pip install runpod requests

# Create model directories
RUN mkdir -p /app/ComfyUI/models/diffusion_models && \
    mkdir -p /app/ComfyUI/models/text_encoders && \
    mkdir -p /app/ComfyUI/models/vae && \
    mkdir -p /app/ComfyUI/models/loras

# Download Qwen-Image models
# Option 1: Download during build (slower build, faster cold start)
RUN pip install huggingface_hub && \
    python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Comfy-Org/Qwen-Image_ComfyUI', \
                    filename='qwen_image_fp8_e4m3fn.safetensors', \
                    local_dir='/app/ComfyUI/models/diffusion_models'); \
    hf_hub_download(repo_id='Comfy-Org/Qwen-Image_ComfyUI', \
                    filename='qwen_2.5_vl_7b_fp8_scaled.safetensors', \
                    local_dir='/app/ComfyUI/models/text_encoders'); \
    hf_hub_download(repo_id='Comfy-Org/Qwen-Image_ComfyUI', \
                    filename='qwen_image_vae.safetensors', \
                    local_dir='/app/ComfyUI/models/vae')"

# Copy your custom LoRA (you'll need to place it in the build context)
COPY loras/my_custom_lora.safetensors /app/ComfyUI/models/loras/

# Copy handler and helper scripts
WORKDIR /app
COPY handler.py .
COPY comfy_api.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose ComfyUI port (not used in serverless, but useful for debugging)
EXPOSE 8188

# Start ComfyUI in background, then run handler
CMD ["/app/entrypoint.sh"]
```

### 2. entrypoint.sh

```bash
#!/bin/bash
set -e

echo "Starting ComfyUI server..."
cd /app/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to start..."
for i in {1..30}; do
    if curl -s http://localhost:8188 > /dev/null; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Start RunPod handler
echo "Starting RunPod handler..."
cd /app
python3 handler.py
```

### 3. comfy_api.py

```python
"""Helper functions for interacting with ComfyUI API"""
import requests
import time
import json
from typing import Dict, List, Optional

COMFY_URL = "http://127.0.0.1:8188"

def submit_workflow(workflow: Dict) -> str:
    """
    Submit a workflow to ComfyUI

    Args:
        workflow: ComfyUI workflow JSON (API format)

    Returns:
        prompt_id: Unique ID for this generation
    """
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow}
    )
    response.raise_for_status()

    data = response.json()
    return data['prompt_id']

def get_history(prompt_id: str) -> Dict:
    """Get execution history for a prompt"""
    response = requests.get(f"{COMFY_URL}/history/{prompt_id}")
    response.raise_for_status()
    return response.json()

def get_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """Download an image from ComfyUI output"""
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }
    response = requests.get(f"{COMFY_URL}/view", params=params)
    response.raise_for_status()
    return response.content

def wait_for_completion(prompt_id: str, timeout: int = 300) -> List[Dict]:
    """
    Wait for workflow to complete and return output images

    Args:
        prompt_id: ID from submit_workflow
        timeout: Maximum seconds to wait

    Returns:
        List of image data dicts with 'filename' and 'data' (bytes)
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        history = get_history(prompt_id)

        if prompt_id in history:
            execution = history[prompt_id]

            # Check if completed
            if 'outputs' in execution:
                images = []

                # Find all output images in the history
                for node_id, node_output in execution['outputs'].items():
                    if 'images' in node_output:
                        for img_info in node_output['images']:
                            img_data = get_image(
                                img_info['filename'],
                                img_info.get('subfolder', ''),
                                img_info.get('type', 'output')
                            )
                            images.append({
                                'filename': img_info['filename'],
                                'data': img_data
                            })

                return images

            # Check for errors
            if 'status' in execution:
                if execution['status'].get('completed') == False:
                    error_msg = execution.get('status', {}).get('messages', ['Unknown error'])
                    raise RuntimeError(f"ComfyUI execution failed: {error_msg}")

        time.sleep(1)

    raise TimeoutError(f"Workflow did not complete within {timeout} seconds")

def health_check() -> bool:
    """Check if ComfyUI is responsive"""
    try:
        response = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False
```

### 4. handler.py

```python
"""RunPod serverless handler for ComfyUI + Qwen-Image"""
import runpod
import base64
import json
from typing import Dict, Any
import comfy_api

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for ComfyUI workflow execution

    event['input'] = {
        'workflow': Dict,  # ComfyUI workflow JSON (required)
        'lora_strength': float,  # Optional: LoRA strength (default: 1.0)
        'lora_name': str,  # Optional: LoRA filename (default: 'my_custom_lora.safetensors')
    }

    Returns:
        {
            'images': [
                {
                    'filename': str,
                    'data': str (base64)
                }
            ]
        }
    """
    try:
        job_input = event.get('input', {})

        # Validate workflow exists
        if 'workflow' not in job_input:
            return {'error': 'Missing required field: workflow'}

        workflow = job_input['workflow']

        # Optional: Inject LoRA parameters into workflow
        # (You'll need to modify the workflow JSON structure based on your nodes)
        if 'lora_strength' in job_input or 'lora_name' in job_input:
            # Find LoRA loader node (example - adjust based on your workflow)
            for node_id, node_data in workflow.items():
                if node_data.get('class_type') == 'LoraLoader':
                    if 'lora_name' in job_input:
                        node_data['inputs']['lora_name'] = job_input['lora_name']
                    if 'lora_strength' in job_input:
                        node_data['inputs']['strength_model'] = job_input['lora_strength']
                        node_data['inputs']['strength_clip'] = job_input['lora_strength']

        # Submit workflow to ComfyUI
        prompt_id = comfy_api.submit_workflow(workflow)

        # Wait for completion
        images = comfy_api.wait_for_completion(prompt_id, timeout=300)

        # Convert images to base64
        result_images = []
        for img in images:
            img_base64 = base64.b64encode(img['data']).decode('utf-8')
            result_images.append({
                'filename': img['filename'],
                'data': img_base64,
                'type': 'base64'
            })

        return {
            'images': result_images,
            'prompt_id': prompt_id
        }

    except TimeoutError as e:
        return {'error': f'Timeout: {str(e)}'}
    except Exception as e:
        return {'error': f'Error: {str(e)}'}

# Health check endpoint
def health_handler(event):
    """Check if ComfyUI is running"""
    if comfy_api.health_check():
        return {'status': 'healthy'}
    else:
        return {'status': 'unhealthy', 'error': 'ComfyUI not responding'}

# Start RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "health": health_handler
    })
```

### 5. workflows/qwen_txt2img.json

You'll need to export this from ComfyUI UI after setting up your workflow:

1. Open ComfyUI
2. Load Qwen-Image nodes
3. Add LoRA Loader node
4. Configure your workflow
5. **Workflow > Export (API Format)**
6. Save the JSON

Example structure (simplified):
```json
{
  "1": {
    "class_type": "LoraLoader",
    "inputs": {
      "lora_name": "my_custom_lora.safetensors",
      "strength_model": 1.0,
      "strength_clip": 1.0,
      "model": ["2", 0],
      "clip": ["2", 1]
    }
  },
  "2": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "qwen_image_fp8_e4m3fn.safetensors"
    }
  },
  ...
}
```

### 6. test_input.json

```json
{
  "input": {
    "workflow": {
      "// Load your exported workflow JSON here": "..."
    },
    "lora_strength": 0.8,
    "lora_name": "my_custom_lora.safetensors"
  }
}
```

## Building & Testing

### Local Build
```bash
cd runpod

# Build for AMD64 (RunPod compatible)
docker build --platform linux/amd64 -t qwen-comfyui-runpod .

# This will take 30-60 minutes due to model downloads
```

### Local Testing (with GPU)
```bash
docker run --rm --gpus all \
  -p 8188:8188 \
  qwen-comfyui-runpod

# In another terminal, test the handler
curl -X POST http://localhost:8188/prompt \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Deploy to RunPod

1. **Push to Docker Hub**:
```bash
docker tag qwen-comfyui-runpod yourusername/qwen-comfyui-runpod:latest
docker push yourusername/qwen-comfyui-runpod:latest
```

2. **Create Serverless Endpoint**:
- Go to RunPod Console
- Select "Serverless"
- Click "New Endpoint"
- Container Image: `yourusername/qwen-comfyui-runpod:latest`
- GPU: A40 (48GB) or better
- Container Disk: 50GB minimum
- Active Workers: 0-3 (auto-scale)

3. **Test the Endpoint**:
```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

result = endpoint.run_sync({
    "input": {
        "workflow": { ... },  # Your workflow JSON
        "lora_strength": 0.8
    }
})

print(result)
```

## Using Your Custom LoRA

### Option 1: Bake into Docker Image
Place your LoRA in the `loras/` directory before building:
```bash
mkdir -p loras
cp /path/to/your/lora.safetensors loras/my_custom_lora.safetensors
docker build ...
```

### Option 2: Download at Runtime
Modify `entrypoint.sh` to download from a URL:
```bash
wget -O /app/ComfyUI/models/loras/my_custom_lora.safetensors \
  "https://your-storage-url.com/lora.safetensors"
```

### Option 3: Use RunPod Network Volume
- Upload LoRA to Network Volume
- Mount volume in endpoint settings
- Reference via path in workflow

## Workflow Development Tips

1. **Develop locally in ComfyUI**:
   - Run ComfyUI locally
   - Test your LoRA with different strengths
   - Export workflow when satisfied

2. **Test workflow JSON**:
   - Use `test_input.json` to verify structure
   - Check that LoRA node IDs match your modifications in handler

3. **Monitor generations**:
   - Check ComfyUI logs in container: `docker logs <container>`
   - Use RunPod logs to debug handler issues

## Performance Considerations

- **Cold Start**: ~30-45 seconds (ComfyUI boot + model load)
- **Warm Inference**: ~15-60 seconds (depends on steps)
- **GPU Memory**: ~24GB VRAM (A40 recommended)
- **Container Size**: ~36GB

## Estimated Costs (RunPod)

- **A40 GPU**: ~$0.40/hr
- **Storage**: $0.10/GB/month (~$3.60/month for image)
- **Typical generation**: $0.005-0.01 per image

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ComfyUI not starting | Check logs, increase startup wait time |
| LoRA not loading | Verify file path and name in workflow |
| Out of memory | Reduce image resolution or use A100 |
| Timeout errors | Increase timeout in `wait_for_completion` |
| Workflow errors | Test locally in ComfyUI first |
