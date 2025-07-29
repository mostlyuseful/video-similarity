"""
Configuration module for video-similarity application.
"""

import os
from pathlib import Path

# --- Configuration ---
CACHE_DIR = (
    Path(os.getenv("XDG_CACHE_DIR", Path.home() / ".cache")) / "video-similarity"
)
THUMBNAIL_DIR = CACHE_DIR / "thumbnails"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = CACHE_DIR / "video_cache.db"
