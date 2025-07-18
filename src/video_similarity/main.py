import typer
from dataclasses import dataclass
from pathlib import Path
from .processing import extract_aggregated_features, compare_features
import itertools
from .output import generate_json_output_poc

SIMILARITY_THRESHOLD = 0.5  # Configurable threshold


@dataclass
class VideoFile:
    path: Path
    is_valid: bool


def cli(videos: list[str] = typer.Argument(..., help="List of video file paths"), frame_skip: int = typer.Option(0, help="Number of frames to skip during feature extraction")):
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
        similarity = compare_features(
            video_features[video_a], 
            video_features[video_b]
        )
        
        results.append({
            "video_a": str(video_a),
            "video_b": str(video_b),
            "similarity": float(similarity),
            "is_similar": similarity >= SIMILARITY_THRESHOLD
        })
        
        typer.echo(f"{video_a.name} vs {video_b.name}: {similarity:.3f}")

    # Generate output
    typer.echo("\nGenerating output...")
    output = generate_json_output_poc(results)
    typer.echo(output)

def main():
    """
    Main entry point for the CLI.
    """
    typer.run(cli)


if __name__ == "__main__":
    main()
