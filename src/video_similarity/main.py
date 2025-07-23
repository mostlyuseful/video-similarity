import typer
from dataclasses import dataclass
from pathlib import Path
from .processing import extract_aggregated_features, compare_features
import itertools
from .output import generate_json_output_poc
from . import inspection

SIMILARITY_THRESHOLD = 0.5  # Configurable threshold

app = typer.Typer()


@dataclass
class VideoFile:
    path: Path
    is_valid: bool


@app.command()
def compare(
    videos: list[str] = typer.Argument(..., help="List of video file paths"),
    frame_skip: int = typer.Option(
        0, help="Number of frames to skip during feature extraction"
    ),
):
    """
    CLI entry point for video similarity comparison.

    Args:
        videos: List of video file paths to compare
    """
    if len(videos) < 2:
        typer.echo("Error: At least 2 videos are required for comparison.")
        raise typer.Exit(1)

    video_files = [Path(video) for video in videos]

    # Validate files exist
    for video_file in video_files:
        if not video_file.exists():
            typer.echo(f"Error: File not found: {video_file}")
            raise typer.Exit(1)

    # Extract features for all videos
    typer.echo("Extracting features from videos...")
    video_features = {}
    for video_file in video_files:
        features = extract_aggregated_features(str(video_file), frame_skip=frame_skip)
        if features is None:
            typer.echo(f"Warning: Could not extract features from {video_file}")
            continue
        video_features[video_file] = features

    if len(video_features) < 2:
        typer.echo("Error: Need at least 2 videos with valid features for comparison.")
        raise typer.Exit(1)

    # Compare all pairs
    typer.echo("Comparing videos...")
    results = []
    for video_a, video_b in itertools.combinations(video_features.keys(), 2):
        similarity = compare_features(video_features[video_a], video_features[video_b])

        results.append(
            {
                "video_a": str(video_a),
                "video_b": str(video_b),
                "similarity": float(similarity),
                "is_similar": similarity >= SIMILARITY_THRESHOLD,
            }
        )

        typer.echo(f"{video_a.name} vs {video_b.name}: {similarity:.3f}")

    # Generate output
    typer.echo("\nGenerating output...")
    output = generate_json_output_poc(results)
    typer.echo(output)


@app.command()
def inspect(
    report_file: Path = typer.Option(
        ...,
        "--report",
        "-r",
        help="Path to the czkawka JSON report.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int = typer.Option(8000, help="Port to run the server on."),
):
    """
    Run the video similarity inspection server.
    """
    inspection.main(report_file=report_file, host=host, port=port)


def main():
    """
    Main entry point for the CLI.
    """
    app()


if __name__ == "__main__":
    main()
