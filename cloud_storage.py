import boto3
from botocore.client import Config
import os
from pathlib import Path

# Load environment variables from .env file when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

# --- Configuration ---
# These are read from environment variables. In Runpod, they should be set as secrets.
# Runpod automatically prefixes secrets with "RUNPOD_SECRET_".
def get_env_vars():
    """Get environment variables dynamically to ensure they're loaded after dotenv."""
    return {
        'ENDPOINT_URL': os.environ.get("RUNPOD_SECRET_R2_ENDPOINT_URL"),
        'ACCESS_KEY_ID': os.environ.get("RUNPOD_SECRET_R2_ACCESS_KEY_ID"),
        'SECRET_ACCESS_KEY': os.environ.get("RUNPOD_SECRET_R2_SECRET_ACCESS_KEY"),
        'BUCKET_NAME': os.environ.get("RUNPOD_SECRET_R2_BUCKET_NAME"),
        'PUBLIC_URL_BASE': os.environ.get("RUNPOD_SECRET_R2_PUBLIC_URL_BASE")
    }

def get_s3_client():
    """Initializes and returns a boto3 S3 client configured for R2."""
    env_vars = get_env_vars()
    if not all([env_vars['ENDPOINT_URL'], env_vars['ACCESS_KEY_ID'], env_vars['SECRET_ACCESS_KEY'], env_vars['BUCKET_NAME']]):
        raise ValueError("One or more R2 environment variables are not set.")

    s3_client = boto3.client(
        's3',
        endpoint_url=env_vars['ENDPOINT_URL'],
        aws_access_key_id=env_vars['ACCESS_KEY_ID'],
        aws_secret_access_key=env_vars['SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='auto' # R2 specific
    )
    return s3_client

def upload_file(local_path: Path, object_name: str) -> bool:
    """
    Uploads a file to the R2 bucket.

    Args:
        local_path: The path to the local file to upload.
        object_name: The desired key (path) for the object in the bucket.

    Returns:
        True if upload was successful, False otherwise.
    """
    env_vars = get_env_vars()
    print(f"Attempting to upload '{local_path}' to bucket '{env_vars['BUCKET_NAME']}' as '{object_name}'...")
    try:
        s3_client = get_s3_client()

        # Determine content type based on file extension
        content_type = 'application/octet-stream'
        if str(local_path).endswith('.mp4'):
            content_type = 'video/mp4'
        elif str(local_path).endswith('.webm'):
            content_type = 'video/webm'
        elif str(local_path).endswith('.mov'):
            content_type = 'video/quicktime'
        elif str(local_path).endswith('.png'):
            content_type = 'image/png'
        elif str(local_path).endswith(('.jpg', '.jpeg')):
            content_type = 'image/jpeg'
        elif str(local_path).endswith('.webp'):
            content_type = 'image/webp'

        # Upload with proper content type and cache control
        s3_client.upload_file(
            str(local_path),
            env_vars['BUCKET_NAME'],
            object_name,
            ExtraArgs={
                'ContentType': content_type,
                'CacheControl': 'public, max-age=31536000'
            }
        )
        print(f"Successfully uploaded to {object_name} with Content-Type: {content_type}")
        return True
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False

def download_file(object_name: str, local_path: Path) -> bool:
    """
    Downloads a file from the R2 bucket to a local path.

    Args:
        object_name: The key (path) of the object in the bucket.
        local_path: The local path where the file should be saved.

    Returns:
        True if download was successful, False otherwise.
    """
    env_vars = get_env_vars()
    print(f"Attempting to download '{object_name}' from bucket '{env_vars['BUCKET_NAME']}' to '{local_path}'...")

    # Create parent directories if they don't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3_client = get_s3_client()
        s3_client.download_file(env_vars['BUCKET_NAME'], object_name, str(local_path))
        print(f"Successfully downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def get_public_url(object_name: str) -> str:
    """
    Constructs the public URL for an object in the R2 bucket.

    Note: This assumes your R2 bucket is configured for public access.

    Args:
        object_name: The key (path) of the object in the bucket.

    Returns:
        The full public URL to the object.
    """
    env_vars = get_env_vars()
    if not env_vars['PUBLIC_URL_BASE']:
        raise ValueError("R2_PUBLIC_URL_BASE environment variable is not set.")

    return f"{env_vars['PUBLIC_URL_BASE']}/{object_name}"
