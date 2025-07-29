import pytest
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock


# Fixture to provide paths to test video files
@pytest.fixture(scope="session")
def video_data_dir():
    return Path(__file__).parent.parent / "test" / "data"


@pytest.fixture(
    params=[
        "big_buck_bunny_1080p_h264.mov",
        "BigBuckBunny_320x180.mp4",
        "sintel-2048-surround.mp4",
    ]
)
def video_file(request, video_data_dir, tmp_path):
    """Parametrized fixture providing paths to test video files"""
    video_filename = request.param
    src_path = video_data_dir / video_filename
    dst_path = tmp_path / video_filename
    shutil.copy(src_path, dst_path)
    return str(dst_path)


# Fixture for corrupted video file
@pytest.fixture
def corrupted_video_file(tmp_path):
    path = tmp_path / "corrupted.mp4"
    with open(path, "wb") as f:
        f.write(b"INVALID VIDEO DATA" * 1000)
    return str(path)


# Fixture for invalid video format
@pytest.fixture
def invalid_video_format_file(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text("This is not a video file")
    return str(path)


# Fixture to mock video manager
@pytest.fixture
def mock_video_manager(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("scenedetect.video_manager.VideoManager", mock)
    return mock


# Fixture to mock ORB detector
@pytest.fixture
def mock_orb_detector(monkeypatch):
    mock = MagicMock()
    mock.return_value.detectAndCompute.return_value = (
        [MagicMock()],
        np.random.rand(100, 32).astype(np.uint8),
    )
    monkeypatch.setattr("cv2.ORB_create", mock)
    return mock
