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


def main(videos: list[str] = typer.Argument(..., help="Paths to video files")):
    """Video Similarity Tool"""
    
    # Validate video files
    video_files = []
    for video_path in videos:
        path = Path(video_path)
        if path.exists() and path.is_file():
            video_files.append(VideoFile(path, True))
        else:
            print(f"WARNING: File not found or is a directory: {video_path}")
            video_files.append(VideoFile(path, False))
    
    valid_videos = [vf for vf in video_files if vf.is_valid]

    if len(valid_videos) < 2:
        print("ERROR: Need at least 2 valid videos for comparison")
        return

    print(f"Found {len(valid_videos)} valid videos:")
    for v in valid_videos:
        print(f"- {v.path}")

    # Extract features for each valid video
    video_features = {}
    for video in valid_videos:
        print(f"Processing {video.path}...")
        features = extract_aggregated_features(str(video.path))
        if features is not None:
            video_features[str(video.path)] = features
            print(f"  Extracted {len(features)} descriptors")
        else:
            print(f"  WARNING: Skipping {video.path} - feature extraction failed")

    # Ensure we have at least 2 videos with features
    if len(video_features) < 2:
        print("ERROR: Need at least 2 videos with extracted features")
        return

    # Perform pairwise comparisons and collect significant matches
    print("\nPerforming pairwise comparisons:")
    significant_matches = []
    video_paths = list(video_features.keys())
    for path_a, path_b in itertools.combinations(video_paths, 2):
        score = compare_features(video_features[path_a], video_features[path_b])
        print(f"{path_a} vs {path_b} - Score: {score:.4f}")

        # Check if score meets threshold
        if score >= SIMILARITY_THRESHOLD:
            match_data = {
                "file_a": path_a,
                "file_b": path_b,
                "metrics": {"normalized_match_score": float(f"{score:.4f}")},
            }
            significant_matches.append(match_data)

    # Generate and print JSON output
    json_output = generate_json_output_poc(significant_matches)
    print("\nFinal JSON Output:")
    print(json_output)


if __name__ == "__main__":
    typer.run(main)

