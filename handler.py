"""
RunPod Serverless Handler for Qwen-Image with Custom LoRA Support
Uses native diffusers pipeline - clean and production-ready
"""
import runpod
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import base64
from io import BytesIO
from typing import Dict, Any, Optional
import os
import uuid
import tempfile
from pathlib import Path

# Import cloud storage for R2 uploads
import cloud_storage

# Global pipeline (loaded once on container start)
print("=" * 60)
print("Loading Qwen-Image pipeline...")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load the base pipeline
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=DTYPE,
    cache_dir="/app/.cache/huggingface"
)
pipe = pipe.to(DEVICE)

# Enable memory optimizations
if DEVICE == "cuda":
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

print(f"Pipeline loaded successfully on {DEVICE}!")
print("=" * 60)

# Track currently loaded LoRA
CURRENT_LORA = None
CURRENT_LORA_SCALE = None


def load_lora(lora_path: str, lora_scale: float = 1.0, weight_name: str = None):
    """
    Load a LoRA into the pipeline

    Args:
        lora_path: Path to LoRA file or HuggingFace model ID
        lora_scale: LoRA strength (0.0 to 2.0, typically 0.5-1.0)
        weight_name: Specific weight file name (for HuggingFace repos with multiple files)
    """
    global CURRENT_LORA, CURRENT_LORA_SCALE

    # Skip if same LoRA already loaded
    if CURRENT_LORA == lora_path and CURRENT_LORA_SCALE == lora_scale:
        print(f"LoRA already loaded: {lora_path} (scale: {lora_scale})")
        return

    # Unload previous LoRA if exists
    if CURRENT_LORA is not None:
        print(f"Unloading previous LoRA: {CURRENT_LORA}")
        pipe.unload_lora_weights()

    # Load new LoRA
    print(f"Loading LoRA: {lora_path} (scale: {lora_scale}, weight_name: {weight_name})")

    if os.path.exists(lora_path):
        # Local file
        pipe.load_lora_weights(lora_path)
    else:
        # HuggingFace repo
        if weight_name:
            pipe.load_lora_weights(lora_path, weight_name=weight_name)
        else:
            pipe.load_lora_weights(lora_path)

    # Set LoRA scale
    pipe.fuse_lora(lora_scale=lora_scale)

    CURRENT_LORA = lora_path
    CURRENT_LORA_SCALE = lora_scale
    print(f"LoRA loaded successfully!")


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    weight_name: str = None,
    num_images: int = 1
) -> list[Image.Image]:
    """
    Generate images using Qwen-Image pipeline

    Args:
        prompt: Text description of desired image
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps (default: 50)
        true_cfg_scale: Classifier-free guidance strength (default: 4.0)
        width: Output width in pixels
        height: Output height in pixels
        seed: Random seed for reproducibility
        lora_path: Path to LoRA file (local or HF repo)
        lora_scale: LoRA strength (0.0-2.0)
        weight_name: Specific weight file for HuggingFace repos
        num_images: Number of images to generate

    Returns:
        List of PIL Images
    """
    # Load LoRA if specified
    if lora_path:
        load_lora(lora_path, lora_scale, weight_name)

    # Set random seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Generate
    print(f"Generating {num_images} image(s)...")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Steps: {num_inference_steps}, CFG: {true_cfg_scale}")
    print(f"Size: {width}x{height}")

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        generator=generator,
        num_images_per_prompt=num_images
    )

    return output.images


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler

    Expected input format:
    {
        "input": {
            "prompt": "A magical forest...",  # Required
            "negative_prompt": "blurry, low quality",  # Optional
            "num_inference_steps": 50,  # Optional (default: 50)
            "true_cfg_scale": 4.0,  # Optional (default: 4.0)
            "width": 1024,  # Optional (default: 1024)
            "height": 1024,  # Optional (default: 1024)
            "seed": 42,  # Optional (random if not provided)
            "lora_path": "/app/loras/my_custom_lora.safetensors",  # Optional
            "lora_scale": 0.8,  # Optional (default: 1.0)
            "num_images": 1,  # Optional (default: 1)
            "output_format": "png"  # Optional (png or jpeg)
        }
    }

    Returns:
    {
        "images": [
            {
                "image": "base64_string",
                "seed": 42
            }
        ],
        "metadata": {
            "prompt": "...",
            "width": 1024,
            "height": 1024,
            "steps": 50,
            "cfg_scale": 4.0,
            "lora_used": true
        }
    }
    """
    try:
        job_input = event.get('input', {})

        # Validate required fields
        if 'prompt' not in job_input:
            return {'error': 'Missing required field: prompt'}

        # Extract parameters with defaults
        prompt = job_input['prompt']
        negative_prompt = job_input.get('negative_prompt', '')
        num_inference_steps = job_input.get('num_inference_steps', 50)
        true_cfg_scale = job_input.get('true_cfg_scale', 4.0)
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)
        seed = job_input.get('seed')
        # Default to Ryze LoRA from HuggingFace if no lora_path specified
        # If lora_path is explicitly set to None, it will be None (no LoRA)
        if 'lora_path' not in job_input:
            lora_path = 'aklevecz/ryze_v1'
        else:
            lora_path = job_input['lora_path']
        lora_scale = job_input.get('lora_scale', 0.8)
        weight_name = job_input.get('weight_name', 'ryze.safetensors')
        num_images = job_input.get('num_images', 1)
        output_format = job_input.get('output_format', 'png').upper()
        upload_to_r2 = job_input.get('upload_to_r2', True)  # Default to R2 upload

        # Validate parameters
        if num_images > 4:
            return {'error': 'num_images must be <= 4'}

        if not (512 <= width <= 2048) or not (512 <= height <= 2048):
            return {'error': 'Width and height must be between 512 and 2048'}

        # Generate images
        images = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            width=width,
            height=height,
            seed=seed,
            lora_path=lora_path,
            lora_scale=lora_scale,
            weight_name=weight_name,
            num_images=num_images
        )

        # Process images - upload to R2 or convert to base64
        result_images = []
        for i, img in enumerate(images):
            image_data = {
                'seed': seed if seed is not None else 'random',
                'index': i
            }

            if upload_to_r2:
                # Upload to R2
                try:
                    # Generate unique filename
                    file_ext = 'png' if output_format == 'PNG' else 'jpg'
                    unique_id = str(uuid.uuid4())
                    filename = f"{unique_id}.{file_ext}"

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                        img.save(tmp_file.name, format=output_format)
                        tmp_path = Path(tmp_file.name)

                    # Upload to R2
                    object_name = f"generations/{filename}"
                    upload_success = cloud_storage.upload_file(tmp_path, object_name)

                    # Clean up temp file
                    tmp_path.unlink()

                    if upload_success:
                        # Get public URL
                        public_url = cloud_storage.get_public_url(object_name)
                        image_data['url'] = public_url
                        image_data['uploaded'] = True
                    else:
                        # Fallback to base64 if upload fails
                        img_base64 = image_to_base64(img, format=output_format)
                        image_data['image'] = img_base64
                        image_data['uploaded'] = False
                        image_data['upload_error'] = 'R2 upload failed'

                except Exception as e:
                    # Fallback to base64 on error
                    print(f"R2 upload error: {e}")
                    img_base64 = image_to_base64(img, format=output_format)
                    image_data['image'] = img_base64
                    image_data['uploaded'] = False
                    image_data['upload_error'] = str(e)
            else:
                # Return base64 (original behavior)
                img_base64 = image_to_base64(img, format=output_format)
                image_data['image'] = img_base64

            result_images.append(image_data)

        # Return results
        return {
            'images': result_images,
            'metadata': {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'num_inference_steps': num_inference_steps,
                'true_cfg_scale': true_cfg_scale,
                'lora_used': lora_path is not None,
                'lora_path': lora_path,
                'lora_scale': lora_scale if lora_path else None,
                'num_images_generated': len(images)
            }
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in handler: {error_trace}")
        return {
            'error': str(e),
            'traceback': error_trace
        }


# Start RunPod serverless worker
if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
