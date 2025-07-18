from unittest.mock import patch
from video_similarity.main import parse_and_validate_args, main, VideoFile
import argparse
from pathlib import Path
import numpy as np


def test_parse_and_validate_args_with_valid_and_invalid_files(tmp_path):
    # Create valid files
    valid1 = tmp_path / "valid1.mp4"
    valid1.touch()
    valid2 = tmp_path / "valid2.mp4"
    valid2.touch()

    # Invalid file (doesn't exist)
    invalid = tmp_path / "invalid.mp4"

    # Create mock arguments
    mock_args = argparse.Namespace(videos=[str(valid1), str(invalid), str(valid2)])

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        video_files = parse_and_validate_args()

    assert len(video_files) == 3
    assert video_files[0].path == valid1 and video_files[0].is_valid
    assert video_files[1].path == invalid and not video_files[1].is_valid
    assert video_files[2].path == valid2 and video_files[2].is_valid


@patch("video_similarity.main.parse_and_validate_args")
@patch("video_similarity.main.extract_aggregated_features")
@patch("video_similarity.main.compare_features")
@patch("video_similarity.main.generate_json_output_poc")
def test_main_with_insufficient_valid_videos(
    mock_gen_json, mock_compare, mock_extract, mock_parse_args, capsys
):
    # Setup mock data with only one valid video
    mock_parse_args.return_value = [
        VideoFile(Path("valid1.mp4"), True),
        VideoFile(Path("invalid.mp4"), False),
    ]

    main()

    captured = capsys.readouterr()
    assert "ERROR: Need at least 2 valid videos for comparison" in captured.out
    mock_extract.assert_not_called()
    mock_compare.assert_not_called()
    mock_gen_json.assert_not_called()


@patch("video_similarity.main.parse_and_validate_args")
@patch("video_similarity.main.extract_aggregated_features")
@patch("video_similarity.main.compare_features")
@patch("video_similarity.main.generate_json_output_poc")
def test_main_with_feature_extraction_failure(
    mock_gen_json, mock_compare, mock_extract, mock_parse_args, capsys
):
    # Setup two valid videos but one feature extraction fails
    mock_parse_args.return_value = [
        VideoFile(Path("valid1.mp4"), True),
        VideoFile(Path("valid2.mp4"), True),
        VideoFile(Path("valid3.mp4"), True),
    ]

    # Make one feature extraction fail
    mock_extract.side_effect = [np.array([1, 2, 3]), None, None]

    main()

    captured = capsys.readouterr()
    assert "WARNING: Skipping valid2.mp4 - feature extraction failed" in captured.out
    assert "ERROR: Need at least 2 videos with extracted features" in captured.out
    mock_compare.assert_not_called()
    mock_gen_json.assert_not_called()


@patch("video_similarity.main.parse_and_validate_args")
@patch("video_similarity.main.extract_aggregated_features")
@patch("video_similarity.main.compare_features")
@patch("video_similarity.main.generate_json_output_poc")
def test_main_with_successful_comparison(
    mock_gen_json, mock_compare, mock_extract, mock_parse_args, capsys
):
    # Setup two valid videos with successful feature extraction
    mock_parse_args.return_value = [
        VideoFile(Path("valid1.mp4"), True),
        VideoFile(Path("valid2.mp4"), True),
    ]

    # Mock feature extraction
    mock_extract.side_effect = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    # Mock comparison to return a high score
    mock_compare.return_value = 0.85

    # Mock JSON output
    mock_gen_json.return_value = '{"matches": []}'

    main()

    captured = capsys.readouterr()
    assert "Processing valid1.mp4..." in captured.out
    assert "Processing valid2.mp4..." in captured.out
    assert "valid1.mp4 vs valid2.mp4 - Score: 0.8500" in captured.out
    assert "Final JSON Output:" in captured.out
    mock_compare.assert_called_once()
    mock_gen_json.assert_called_once()
