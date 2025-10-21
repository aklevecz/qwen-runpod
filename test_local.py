"""
Local testing script for the handler
Run this before deploying to RunPod
"""
import json
import base64
from PIL import Image
from io import BytesIO
from handler import handler


def test_handler():
    """Test the handler locally"""
    # Load test input
    with open('test_input.json', 'r') as f:
        test_event = json.load(f)

    print("Testing handler with input:")
    print(json.dumps(test_event, indent=2))
    print("\n" + "="*60 + "\n")

    # Run handler
    result = handler(test_event)

    # Check for errors
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        if 'traceback' in result:
            print(result['traceback'])
        return

    # Save generated images
    print(f"✅ Generated {len(result['images'])} image(s)")
    print(f"\nMetadata:")
    print(json.dumps(result['metadata'], indent=2))

    for i, img_data in enumerate(result['images']):
        # Decode base64
        img_bytes = base64.b64decode(img_data['image'])
        img = Image.open(BytesIO(img_bytes))

        # Save
        output_path = f"output_{i}.png"
        img.save(output_path)
        print(f"\n✅ Saved image {i} to: {output_path}")
        print(f"   Size: {img.size}")
        print(f"   Seed: {img_data['seed']}")


if __name__ == "__main__":
    test_handler()
