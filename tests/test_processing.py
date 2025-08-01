from unittest.mock import patch, MagicMock
from video_similarity.processing import (
    extract_aggregated_features,
    compare_features,
)
import numpy as np


def test_extract_aggregated_features_success(video_file):
    features = extract_aggregated_features(video_file)
    assert features is not None
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(scene_features, np.ndarray) for scene_features in features)
    assert all(
        scene_features.shape[1] == 32 for scene_features in features
    )  # ORB descriptors are 32 elements


def test_extract_aggregated_features_corrupted(corrupted_video_file):
    features = extract_aggregated_features(corrupted_video_file)
    assert features is None


def test_extract_aggregated_features_invalid_format(invalid_video_format_file):
    features = extract_aggregated_features(invalid_video_format_file)
    assert features is None


@patch("scenedetect.video_manager.VideoManager", autospec=True)
@patch("scenedetect.scene_manager.SceneManager", autospec=True)
def test_extract_aggregated_features_no_scenes(mock_scene_manager, mock_video_manager):
    # Mock scene detection to return empty scene list
    mock_scene_manager.return_value.get_scene_list.return_value = []

    features = extract_aggregated_features("dummy_path.mp4")
    assert features is None


@patch("cv2.ORB_create")
def test_extract_aggregated_features_no_descriptors(mock_orb):
    # Mock ORB to return no descriptors
    mock_orb.return_value.detectAndCompute.return_value = ([], None)

    features = extract_aggregated_features("dummy_path.mp4")
    assert features is None


@patch("scenedetect.video_manager.VideoManager", autospec=True)
@patch("scenedetect.scene_manager.SceneManager", autospec=True)
def test_extract_aggregated_features_frame_read_failure(
    mock_scene_manager, mock_video_manager
):
    # Mock scene detection to return valid scene list
    mock_scene_manager.return_value.get_scene_list.return_value = [
        (MagicMock(), MagicMock())
    ]

    # Mock video to return None when reading frame
    mock_video = MagicMock()
    mock_video.read.return_value = None
    mock_video_manager.return_value = mock_video

    features = extract_aggregated_features("dummy_path.mp4")
    assert features is None


def test_compare_features_valid_descriptors():
    # Create two sets of scene-based descriptors
    descriptors_a = [
        np.random.rand(50, 32).astype(np.uint8),
        np.random.rand(30, 32).astype(np.uint8),
    ]
    descriptors_b = [
        np.random.rand(45, 32).astype(np.uint8),
        np.random.rand(35, 32).astype(np.uint8),
    ]

    score = compare_features(descriptors_a, descriptors_b)
    assert 0.0 <= score <= 1.0


def test_compare_features_empty_descriptor_a():
    descriptors_a = []
    descriptors_b = [np.random.rand(100, 32).astype(np.uint8)]

    score = compare_features(descriptors_a, descriptors_b)
    assert score == 0.0


def test_compare_features_empty_descriptor_b():
    descriptors_a = [np.random.rand(100, 32).astype(np.uint8)]
    descriptors_b = []

    score = compare_features(descriptors_a, descriptors_b)
    assert score == 0.0


def test_compare_features_both_empty():
    descriptors_a = []
    descriptors_b = []

    score = compare_features(descriptors_a, descriptors_b)
    assert score == 0.0


def test_compare_features_none_descriptor_a():
    descriptors_a = None
    descriptors_b = [np.random.rand(100, 32).astype(np.uint8)]

    score = compare_features(descriptors_a, descriptors_b)
    assert score == 0.0


def test_compare_features_none_descriptor_b():
    descriptors_a = [np.random.rand(100, 32).astype(np.uint8)]
    descriptors_b = None

    score = compare_features(descriptors_a, descriptors_b)
    assert score == 0.0


def test_compare_features_both_none():
    descriptors_a = None
    descriptors_b = None

    score = compare_features(descriptors_a, descriptors_b)
    assert score == 0.0


@patch(
    "scenedetect.frame_timecode.FrameTimecode.get_frames",
    side_effect=Exception("Timecode error"),
)
def test_extract_aggregated_features_timecode_exception(mock_get_frames, video_file):
    features = extract_aggregated_features(video_file)
    assert features is None
