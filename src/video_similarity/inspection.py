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
from pathlib import Path

import cv2
import typer
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

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


def get_video_id(video_path: Path) -> str:
    """
    Generates a unique ID for a video based on its content.
    The video $ID is generated from sparse video contents: 10 10KB strips are
    read from the video using a well-defined random number generator and a
    SHA256 hash of the strips is computed
    """
    n_strips = 3
    num_bytes = 100*1024
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
    thumb_path = THUMBNAIL_DIR / video_id / f"{thumb_index}.jpg"
    if thumb_path.exists():
        return thumb_path

    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return None

    if thumb_index < 3:  # For the main page view
        if thumb_index == 0:
            frame_pos = int(frame_count * 0.1)  # Near beginning
        elif thumb_index == 1:
            frame_pos = int(frame_count * 0.5)  # Middle
        else:
            frame_pos = int(frame_count * 0.9)  # Near end
    else:  # For the detail view, 20 thumbnails
        frame_pos = int(frame_count * (thumb_index / 20.0))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(str(thumb_path), frame)
        return thumb_path
    return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html.j2", {"request": request, "report": report_data}
    )


@app.get("/group/{group_index}", response_class=HTMLResponse)
async def read_group(request: Request, group_index: int):
    if 0 <= group_index < len(report_data):
        group_details = report_data[group_index]
    else:
        group_details = None
    return templates.TemplateResponse(
        "index.html.j2",
        {
            "request": request,
            "report": report_data,
            "group_details": group_details,
            "group_index": group_index,
        },
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
async def delete_videos(
    request: Request,
    group_index: int = Form(...),
    keep: str = Form(...),
    dry_run: bool = Form(False),
):
    group = report_data[group_index]
    files_to_delete = []
    for video in group:
        if video["path"] != keep:
            files_to_delete.append(Path(video["path"]))

    if dry_run:
        print("--- DRY RUN ---")
        print(f"Keeping: {keep}")
        for f in files_to_delete:
            print(f"Would delete: {f}")
        print("-----------------")
    else:
        for f in files_to_delete:
            try:
                print(f"Deleting: {f}")
                f.unlink()
            except OSError as e:
                print(f"Error deleting {f}: {e}")

    # In a real app, you might want to remove the group from the report
    # and save the report back to disk. For this example, we just redirect.
    return RedirectResponse("/", status_code=303)


def main(
    report_file: Path = typer.Option(
        Path(__file__).parent / "example-czkawka-report.json",
        help="Path to the czkawka JSON report.",
    ),
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """
    Run the video similarity inspection server.
    """
    global report_data, video_map
    if report_file.exists():
        with report_file.open("r") as f:
            raw_data = json.load(f)
            for group in raw_data:
                processed_group = []
                for video_info in group:
                    path = Path(video_info["path"])
                    if path.exists():
                        video_id = get_video_id(path)
                        video_info["id"] = video_id
                        video_map[video_id] = video_info
                        processed_group.append(video_info)
                if processed_group:
                    report_data.append(processed_group)
    else:
        typer.echo(f"Error: Report file {report_file} does not exist.", err=True)
        raise typer.Exit(1)

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)