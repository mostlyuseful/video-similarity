from unittest.mock import patch
from video_similarity.main import cli
import numpy as np


def test_cli_with_insufficient_videos():
    """Test CLI with insufficient videos."""
    with patch('typer.echo') as mock_echo, \
         patch('typer.Exit') as mock_exit:
        
        # Test with only 1 video
        cli(["video1.mp4"])
        
        mock_echo.assert_called_with("Error: At least 2 videos are required for comparison.")
        mock_exit.assert_called_once()


def test_cli_with_nonexistent_file():
    """Test CLI with non-existent file."""
    with patch('typer.echo') as mock_echo, \
         patch('typer.Exit') as mock_exit, \
         patch('pathlib.Path.exists', return_value=False):
        
        cli(["nonexistent.mp4", "video2.mp4"])
        
        mock_echo.assert_called_with("Error: File not found: nonexistent.mp4")
        mock_exit.assert_called_once()


@patch("video_similarity.main.extract_aggregated_features")
@patch("video_similarity.main.compare_features")
@patch("video_similarity.main.generate_json_output_poc")
def test_cli_with_feature_extraction_failure(
    mock_gen_json, mock_compare, mock_extract
):
    """Test CLI when feature extraction fails for some videos."""
    with patch('typer.echo') as mock_echo, \
         patch('pathlib.Path.exists', return_value=True):
        
        # Mock feature extraction - one succeeds, one fails
        mock_extract.side_effect = [[np.array([1, 2, 3])], None]
        mock_compare.return_value = 0.85
        mock_gen_json.return_value = '{"matches": []}'
        
        cli(["video1.mp4", "video2.mp4"])
        
        # Should warn about failed extraction
        mock_echo.assert_any_call("Warning: Could not extract features from video2.mp4")
        mock_echo.assert_any_call("Error: Need at least 2 videos with valid features for comparison.")


@patch("video_similarity.main.extract_aggregated_features")
@patch("video_similarity.main.compare_features")
@patch("video_similarity.main.generate_json_output_poc")
def test_cli_successful_comparison(
    mock_gen_json, mock_compare, mock_extract
):
    """Test successful CLI execution."""
    with patch('typer.echo') as mock_echo, \
         patch('pathlib.Path.exists', return_value=True):
        
        # Mock successful feature extraction
        mock_extract.side_effect = [
            [np.array([1, 2, 3])],  # Scene-based features for video1
            [np.array([4, 5, 6])]   # Scene-based features for video2
        ]
        mock_compare.return_value = 0.85
        mock_gen_json.return_value = '{"matches": []}'
        
        cli(["video1.mp4", "video2.mp4"])
        
        # Verify all expected outputs
        mock_echo.assert_any_call("Extracting features from videos...")
        mock_echo.assert_any_call("Comparing videos...")
        mock_echo.assert_any_call("video1.mp4 vs video2.mp4: 0.850")
        mock_echo.assert_any_call("\nGenerating output...")
        mock_echo.assert_any_call('{"matches": []}')


@patch("video_similarity.main.extract_aggregated_features")
@patch("video_similarity.main.compare_features")
@patch("video_similarity.main.generate_json_output_poc")
def test_cli_with_legacy_features(
    mock_gen_json, mock_compare, mock_extract
):
    """Test CLI with legacy concatenated features."""
    with patch('typer.echo') as mock_echo, \
         patch('pathlib.Path.exists', return_value=True):
        
        # Mock legacy concatenated features (single ndarray)
        mock_extract.side_effect = [
            np.random.rand(100, 32).astype(np.uint8),  # Legacy format
            np.random.rand(80, 32).astype(np.uint8)    # Legacy format
        ]
        mock_compare.return_value = 0.75
        mock_gen_json.return_value = '{"matches": []}'
        
        cli(["video1.mp4", "video2.mp4"])
        
        # Should work with legacy features
        mock_echo.assert_any_call("video1.mp4 vs video2.mp4: 0.750")
        mock_gen_json.assert_called_once()
