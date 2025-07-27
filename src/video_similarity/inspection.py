"""
This file will contain a FastAPI app for inspecting video similarity results:
- czkawka-cli is used to find similar videos and generates a JSON report (externally, not done in this tool)
- The FastAPI app will serve the provided report (example-czkawka-report.json for example):
    - Each group of similar videos will be displayed as 3 still jpeg thumbnails (near beginning, middle, near end) of the first video in the group
    - On clicking a group, the app will display the videos in that group:
        - each video will be displayed in a separate row as a horizontal series of around 20 thumbnails
        - the thumbnails are aligned with the videos in the group (table)
        - Clicking a checkbox next to a video will select it for KEEPING, other videos in the group will be marked for DELETING
    - The app will provide a button to confirm the selection and delete the unwanted videos
      - If the app is started with --dry-run, the deletion will not be performed, but the selected videos will be printed to the console
    - All thumbnails are generated on the fly using OpenCV and stored in a cache directory under XDG_CACHE_DIR
      - Thumbnails are saved as JPEG files under the cache directory: $XDG_CACHE_DIR/video-similarity/thumbnails/$ID/$NR.jpg
        - The video $ID is generated from sparse video contents: 100 1KB strips are read from the video using a well-defined random number generator and a SHA256 hash of the strips is computed
"""

import hashlib
import json
import os
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

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# --- Configuration ---
CACHE_DIR = Path(os.getenv("XDG_CACHE_DIR", Path.home() / ".cache")) / "video-similarity"
THUMBNAIL_DIR = CACHE_DIR / "thumbnails"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

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
                    video_id = video_info.get("id", get_video_id(video_path))
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
                if len(report_data) >= 1000:
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
                            # Submit the get_video_id task to the thread pool
                            future = executor.submit(get_video_id, path)
                            futures.append((future, video_info))
                    
                    # Process completed futures with progress bar
                    for future, video_info in tqdm(futures, desc="Computing video hashes", leave=False):
                        video_id = future.result()
                        video_info["id"] = video_id
                        video_map[video_id] = video_info
                        processed_group.append(video_info)
                
                if processed_group and len(processed_group) > 1:
                    report_data.append(processed_group)
    else:
        typer.echo(f"Error: Report file {report} does not exist.", err=True)
        raise typer.Exit(1)
    
    generate_all_thumbnails(report_data)

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)