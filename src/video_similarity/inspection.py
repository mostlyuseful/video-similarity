"""
This module implements a FastAPI application for inspecting and managing video similarity results.

It loads a JSON report (from czkawka-cli), computes unique video IDs via content hashing, and generates
JPEG thumbnails on demand with OpenCV. Thumbnails are cached under
XDG_CACHE_DIR/video-similarity/thumbnails/<video_id>/<index>.jpg to prevent redundant work.

The web interface provides:
- A homepage listing groups of similar videos with representative thumbnails.
- A detail view for each group showing metadata and a series of thumbnails per video.
- A selection mechanism to mark videos for deletion and an optional dry-run mode.
- Actual deletion of selected videos, or logging when run with --dry-run.
"""

import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import typer
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from tqdm import tqdm

from video_similarity.cache import get_video_metadata, save_video_metadata
from video_similarity.config import THUMBNAIL_DIR

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# --- State ---
# This would be loaded from a file in a real app
report_data = []
video_map = {}
video_locks = {}


def get_video_id(video_path: Path) -> str:
    """
    Generates a unique ID for a video based on its content.
    The video $ID is generated from sparse video contents: 10 10KB strips are
    read from the video using a well-defined random number generator and a
    SHA256 hash of the strips is computed
    """
    n_strips = 3
    num_bytes = 25*1024
    if not video_path.exists():
        return ""
    hasher = hashlib.sha256()
    file_size = video_path.stat().st_size
    rng = random.Random(42)
    for _ in range(n_strips):
        offset = rng.randint(0, max(0, file_size - num_bytes))
        with video_path.open("rb") as f:
            f.seek(offset)
            hasher.update(f.read(num_bytes))
    return hasher.hexdigest()


def generate_thumbnail(video_path: Path, video_id: str, thumb_index: int):
    """
    Generates all thumbnails for a video at once, if they don't exist.
    Uses a lock to prevent race conditions.
    """
    thumb_path = THUMBNAIL_DIR / video_id / f"{thumb_index}.jpg"
    if thumb_path.exists():
        return thumb_path

    if video_id not in video_locks:
        video_locks[video_id] = threading.Lock()

    with video_locks[video_id]:
        # Re-check if thumbnails were generated while waiting for the lock
        if thumb_path.exists():
            return thumb_path

        thumb_dir = THUMBNAIL_DIR / video_id
        thumb_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return None

        # Generate thumbnails
        for i in range(10):
            current_thumb_path = thumb_dir / f"{i}.jpg"
            if current_thumb_path.exists():
                continue

            frame_pos = np.linspace(25, frame_count - 25, num=10, dtype=int)[i]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            thumbnail_width, thumbnail_height = (320, 180) # 16:9 aspect ratio

            thumb = 255 * np.ones((thumbnail_height, thumbnail_width, 3), dtype=np.uint8)
            if ret:
                # First, resize the frame to a thumbnail size, keeping aspect ratio:
                # Resize frame to fit within thumbnail_size, keeping aspect ratio
                h, w = frame.shape[:2]
                scale = min(thumbnail_width / w, thumbnail_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Place resized image on a thumbnail canvas, centered
                y_off = (thumbnail_height - new_h) // 2
                x_off = (thumbnail_width - new_w) // 2
                thumb[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            else:
                thumb[:,:] = (128, 128, 128)  # Gray placeholder if frame read fails
            # Then, convert the frame to JPEG format and save it:
            cv2.imwrite(str(current_thumb_path), thumb)

        cap.release()

        if thumb_path.exists():
            return thumb_path
    return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html.j2",
        {"request": request, "report": report_data, "enumerate": enumerate},
    )


@app.get("/thumbnail/{video_id}/{thumb_index}")
async def get_thumbnail(video_id: str, thumb_index: int):
    if video_id not in video_map:
        return HTMLResponse(status_code=404)

    video_info = video_map[video_id]
    video_path = Path(video_info["path"])

    thumb_path = generate_thumbnail(video_path, video_id, thumb_index)

    if thumb_path and thumb_path.exists():
        return FileResponse(thumb_path)
    # Return a placeholder if thumbnail generation fails
    return HTMLResponse(status_code=404)


@app.post("/delete")
async def delete_videos(request: Request):
    form_data = await request.form()
    dry_run = "dry_run" in form_data

    paths_to_delete_from_form = {key for key in form_data if key != "dry_run"}
    files_to_delete = []

    for group in report_data:
        group_paths = {video["path"] for video in group}
        deletions_in_group = group_paths.intersection(paths_to_delete_from_form)

        if not deletions_in_group:
            continue

        if len(deletions_in_group) == len(group_paths):
            print(
                "WARNING: All videos in a group were selected for deletion. "
                "No action will be taken for this group:"
            )
            for path in sorted(list(group_paths)):
                print(f"  - {path}")
            continue

        for path_str in deletions_in_group:
            files_to_delete.append(Path(path_str))

    if dry_run:
        print("--- DRY RUN ---")
        if files_to_delete:
            print("Keeping all other files. Would delete:")
            for f in files_to_delete:
                print(f"  - {f}")
        else:
            print("No files marked for deletion.")
        print("-----------------")
    else:
        if files_to_delete:
            print("Deleting files:")
            for f in files_to_delete:
                try:
                    print(f"  - Deleting: {f}")
                    f.unlink()
                except OSError as e:
                    print(f"    Error deleting {f}: {e}")
        else:
            print("No files marked for deletion.")

    # In a real app, you might want to remove the group from the report
    # and save the report back to disk. For this example, we just redirect.
    return RedirectResponse("/", status_code=303)


@app.get("/group/{group_id}", response_class=HTMLResponse)
async def get_group_detail(request: Request, group_id: int):
    """
    Serve the group detail page for a specific group of similar videos.
    
    Args:
        request: The FastAPI request object
        group_id: The index of the group in the report_data list
        
    Returns:
        HTMLResponse: Rendered group detail page with video data
        
    Raises:
        HTTPException: If group_id is invalid or out of range
    """
    # Validate group_id
    if group_id < 0 or group_id >= len(report_data):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    
    # Get the group data
    group = report_data[group_id]
    group_name = f"Group {group_id}"
    
    # Prepare video data for the template
    videos = []
    for video_info in group:
        video_path = Path(video_info["path"])
        
        # Try to get metadata from cache first
        cached_data = get_video_metadata(video_path)
        if cached_data:
            videos.append({
                "path": str(video_path),
                "src": f"file://{video_path.absolute()}",  # Direct file URL for video playback
                "width": cached_data["width"],
                "height": cached_data["height"],
                "fps": 0,  # Not cached, will be 0
                "duration": 0,  # Not cached, will be 0
                "size": video_path.stat().st_size,
                "bitrate": cached_data["bitrate"],
                "id": video_info["id"],
                "dimensions": cached_data["dimensions"]
            })
            continue
            
        # Cache miss, get metadata from file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get file size
        file_size = video_path.stat().st_size  # Size in bytes
        
        # Try to get bitrate from video properties first
        bitrate_prop = cap.get(cv2.CAP_PROP_BITRATE)  # bits per second
        if bitrate_prop > 0:
            bitrate = bitrate_prop
        else:
            # Fall back to calculated bitrate from file size and duration
            bitrate = (file_size * 8) / duration if duration > 0 else 0  # bits per second
        
        cap.release()
        
        # Save to cache
        save_video_metadata(video_path, video_info["id"], width, height, bitrate)
        
        # Add video data to the list
        videos.append({
            "path": str(video_path),
            "src": f"file://{video_path.absolute()}",  # Direct file URL for video playback
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "size": file_size,
            "bitrate": bitrate,
            "id": video_info["id"],
            "dimensions": f"{width}x{height}"
        })
    
    # Render the template with the group data
    return templates.TemplateResponse(
        "group_detail.html.j2",
        {
            "request": request,
            "group_name": group_name,
            "videos": videos,
            "group_id": group_id,
            "enumerate": enumerate,
        }
    )


def generate_all_thumbnails(report_data):
    """
    Generate thumbnails for all videos in the report data.
    This is called after processing the report to ensure all thumbnails are ready.
    """
    for group in tqdm(report_data, desc="Generating thumbnails for groups"):
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for video_info in group:
                video_path = Path(video_info["path"])
                if video_path.exists():
                    # Check cache first
                    cached_data = get_video_metadata(video_path)
                    if cached_data:
                        video_id = cached_data["video_id"]
                    else:
                        # Cache miss, compute video ID
                        video_id = get_video_id(video_path)
                        # We don't have the full metadata here, so we can't save to cache yet
                        # The cache will be updated when the video detail page is loaded
                    video_info["id"] = video_id
                    futures.append(executor.submit(generate_thumbnail, video_path, video_id, 0))
            # Wait for all futures to complete
            for future in tqdm(futures, desc="Generating thumbnails", leave=False):
                future.result()

def main(
    report: Path = typer.Option(
        ...,
        help="Path to the czkawka JSON report.",
    ),
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """
    Run the video similarity inspection server.
    """
    global report_data, video_map
    if report.exists():
        with report.open("r") as f:
            raw_data = json.load(f)
            random.shuffle(raw_data)  # Shuffle the groups for random order
            for group in tqdm(raw_data, desc="Processing groups"):
                if len(report_data) >= 10:
                    typer.echo("Reached maximum number of groups to process. Stopping further processing.")
                    break

                processed_group = []
                # Process videos in the group using a thread pool
                with ThreadPoolExecutor(max_workers=10) as executor:
                    # Create a list of futures for video processing
                    futures = []
                    for video_info in group:
                        path = Path(video_info["path"])
                        if path.exists():
                            # Check cache first
                            cached_data = get_video_metadata(path)
                            if cached_data:
                                video_id = cached_data["video_id"]
                            else:
                                # Cache miss, compute video ID
                                future = executor.submit(get_video_id, path)
                                futures.append((future, video_info))
                                continue
                                
                            # Use cached video ID
                            video_info["id"] = video_id
                            video_map[video_id] = video_info
                            processed_group.append(video_info)
                    
                    # Process completed futures with progress bar
                    for future, video_info in tqdm(futures, desc="Computing video hashes", leave=False):
                        video_id = future.result()
                        video_info["id"] = video_id
                        video_map[video_id] = video_info
                        processed_group.append(video_info)
                        
                        # Get video metadata for caching
                        video_path = Path(video_info["path"])
                        cap = cv2.VideoCapture(str(video_path))
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            # Try to get bitrate from video properties first
                            bitrate_prop = cap.get(cv2.CAP_PROP_BITRATE)  # bits per second
                            if bitrate_prop > 0:
                                bitrate = bitrate_prop
                            else:
                                # Fall back to calculated bitrate from file size and duration
                                file_size = video_path.stat().st_size  # Size in bytes
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                duration = frame_count / fps if fps > 0 else 0
                                bitrate = (file_size * 8) / duration if duration > 0 else 0  # bits per second
                            cap.release()
                            
                            # Save to cache
                            save_video_metadata(video_path, video_id, width, height, bitrate)
                
                if processed_group and len(processed_group) > 1:
                    # Add dimensions and bitrate to each video in the processed group
                    for video_info in processed_group:
                        video_path = Path(video_info["path"])
                        cached_data = get_video_metadata(video_path)
                        if cached_data:
                            video_info["dimensions"] = f"{cached_data['width']}x{cached_data['height']}"
                            video_info["bitrate"] = cached_data["bitrate"]
                        else:
                            # Fallback if not in cache for some reason
                            cap = cv2.VideoCapture(str(video_path))
                            if cap.isOpened():
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                video_info["dimensions"] = f"{width}x{height}"
                                
                                bitrate_prop = cap.get(cv2.CAP_PROP_BITRATE)
                                if bitrate_prop > 0:
                                    video_info["bitrate"] = bitrate_prop
                                else:
                                    file_size = video_path.stat().st_size
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    duration = frame_count / fps if fps > 0 else 0
                                    video_info["bitrate"] = (file_size * 8) / duration if duration > 0 else 0
                                cap.release()
                            else:
                                video_info["dimensions"] = "N/A"
                                video_info["bitrate"] = 0
                    report_data.append(processed_group)
    else:
        typer.echo(f"Error: Report file {report} does not exist.", err=True)
        raise typer.Exit(1)
    
    generate_all_thumbnails(report_data)

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)