"""Utilities for Providers"""

import base64
import os
import uuid
from pathlib import Path


def store_image(b64_image: str) -> str:
    """Store an base64-encoded image and provide a URL to access it"""
    base_url = os.environ["BASE_URL"]
    image_id = str(uuid.uuid4())
    Path(f"/tmp/oa4a/images/{image_id}.png").write_bytes(base64.b64decode(b64_image))
    return f"{base_url}/images/{image_id}.png"
