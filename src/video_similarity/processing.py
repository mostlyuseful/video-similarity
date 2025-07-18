import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from typing import Optional


def extract_aggregated_features(video_path: str) -> np.ndarray | None:
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        # Detect scenes
        scene_manager.detect_scenes(video=video, show_progress=True)
        scene_list = scene_manager.get_scene_list()

        # If no scenes detected, return None
        if not scene_list:
            print(f"WARNING: No scenes detected in {video_path}")
            return None

        # Initialize ORB detector
        orb = cv2.ORB_create()
        all_descriptors = []

        # Process each scene
        for scene in scene_list:
            # Get start frame of scene
            start_frame = scene[0].get_frames()
            video.seek(start_frame)

            # Read frame
            frame = video.read()
            if not isinstance(frame, np.ndarray):
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect and compute features
            _, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        # Check if any descriptors were found
        if not all_descriptors:
            print(f"WARNING: No descriptors found in {video_path}")
            return None

        # Stack all descriptors vertically
        return np.vstack(all_descriptors)

    except Exception as e:
        print(f"ERROR processing {video_path}: {str(e)}")
        return None


def compare_features_poc(
    descriptors_a: Optional[np.ndarray], descriptors_b: Optional[np.ndarray]
) -> float:
    if descriptors_a is None or descriptors_b is None:
        return 0.0
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return 0.0

    # Initialize BFMatcher with create method
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)

    # Perform KNN matching
    matches = bf.knnMatch(descriptors_a, descriptors_b, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Handle zero descriptor cases
    min_features = min(len(descriptors_a), len(descriptors_b))
    if min_features == 0:
        return 0.0

    # Calculate normalized score
    return len(good_matches) / min_features
