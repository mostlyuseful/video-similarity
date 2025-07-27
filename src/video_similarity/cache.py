"""
This module provides a thread-safe and process-safe SQLite cache for video metadata.
It stores video_id, dimensions, and bitrate along with file size and modification time
for validation.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .config import DB_PATH

# Lock for database initialization
_init_lock = threading.Lock()

# Global connection object for the current process
_global_conn: Optional[sqlite3.Connection] = None
_global_conn_lock = threading.Lock()


def _get_db_connection() -> sqlite3.Connection:
    """
    Get a database connection for the current process.
    Uses a single connection per process with appropriate isolation level
    for write-ahead logging (WAL) mode.
    """
    global _global_conn
    
    with _global_conn_lock:
        if _global_conn is None:
            # Enable WAL mode for better concurrency
            _global_conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False,  # Allow use from multiple threads
                isolation_level=None  # Let SQLite handle locking
            )
            # Configure WAL mode for better concurrency
            _global_conn.execute("PRAGMA journal_mode=WAL")
            _global_conn.execute("PRAGMA synchronous=NORMAL")
            _global_conn.commit()
        return _global_conn


def _init_database():
    """Initialize the database schema if it doesn't exist."""
    global _global_conn
    
    with _init_lock:
        conn = _get_db_connection()
        
        # Create the videos table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                path TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                bitrate REAL NOT NULL,
                file_size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                duration REAL NOT NULL DEFAULT 0,
                fps REAL NOT NULL DEFAULT 0
            )
        ''')
        
        # Create an index on video_id for faster lookups
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_video_id ON videos (video_id)
        ''')
        
        # Create an index on path for faster lookups
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_path ON videos (path)
        ''')
        
        # Add duration and fps columns if they don't exist
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN duration REAL NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN fps REAL NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        conn.commit()


def get_video_metadata(video_path: Path) -> Optional[dict]:
    """
    Retrieve video metadata from the cache.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video metadata if found and valid, None otherwise
    """
    if not video_path.exists():
        return None
        
    try:
        conn = _get_db_connection()
        
        # Get file stats for validation
        stat = video_path.stat()
        current_size = stat.st_size
        current_mtime = stat.st_mtime
        
        # Query the database
        cursor = conn.execute(
            '''
            SELECT video_id, width, height, bitrate, file_size, mtime, duration, fps
            FROM videos
            WHERE path = ?
            ''',
            (str(video_path),)
        )
        
        row = cursor.fetchone()
        if row is None:
            return None
            
        # Unpack the row
        video_id, width, height, bitrate, cached_size, cached_mtime, duration, fps = row
        
        # Validate that the file hasn't changed
        if current_size == cached_size and abs(current_mtime - cached_mtime) < 1.0:
            return {
                "video_id": video_id,
                "dimensions": f"{width}x{height}",
                "width": width,
                "height": height,
                "bitrate": bitrate,
                "duration": duration,
                "fps": fps,
            }
            
        # File has changed, so the cache is invalid
        return None
        
    except (sqlite3.Error, OSError) as e:
        # If there's a database error, return None to force a refresh
        print(f"Error reading from cache: {e}")
        return None


def save_video_metadata(video_path: Path, video_id: str, width: int, height: int, bitrate: float, duration: float, fps: float):
    """
    Save video metadata to the cache.
    
    Args:
        video_path: Path to the video file
        video_id: The video's unique ID
        width: Video width in pixels
        height: Video height in pixels
        bitrate: Video bitrate in bits per second
        duration: Video duration in seconds
        fps: Video frames per second
    """
    if not video_path.exists():
        return
        
    try:
        conn = _get_db_connection()
        
        # Get current file stats
        stat = video_path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
        
        # Insert or update the video record
        conn.execute(
            '''
            INSERT OR REPLACE INTO videos
            (path, video_id, width, height, bitrate, file_size, mtime, duration, fps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (str(video_path), video_id, width, height, bitrate, file_size, mtime, duration, fps)
        )
        
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"Error saving to cache: {e}")


# Initialize the database when the module is imported
_init_database()