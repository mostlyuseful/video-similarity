import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def extract_aggregated_features(video_path: str, frame_skip = 0) -> List[np.ndarray] | None:
    """
    Extract ORB features per scene from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        List of numpy arrays, each containing descriptors for a scene,
        or None if extraction fails
    """
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        # Detect scenes
        scene_manager.detect_scenes(video=video, show_progress=True, frame_skip=frame_skip)
        scene_list = scene_manager.get_scene_list()

        # If no scenes detected, return None
        if not scene_list:
            logger.warning(f"No scenes detected in {video_path}")
            return None

        # Initialize ORB detector
        orb = cv2.ORB_create()
        scene_features = []

        # Process each scene
        for scene_idx, scene in enumerate(scene_list):
            # Get start frame of scene
            start_frame = scene[0].get_frames()
            video.seek(start_frame)

            # Read frame
            frame = video.read()
            if not isinstance(frame, np.ndarray):
                logger.warning(f"Could not read frame for scene {scene_idx}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect and compute features
            _, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) > 0:
                scene_features.append(descriptors)
                logger.debug(f"Scene {scene_idx}: extracted {len(descriptors)} descriptors")
            else:
                logger.warning(f"No descriptors found for scene {scene_idx}")

        # Check if any descriptors were found
        if not scene_features:
            logger.warning(f"No descriptors found in any scene of {video_path}")
            return None

        logger.info(f"Extracted features from {len(scene_features)} scenes")
        return scene_features

    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")
        return None


def extract_aggregated_features_legacy(video_path: str) -> np.ndarray | None:
    """
    Legacy version that concatenates all features into a single ndarray.
    Maintained for backward compatibility.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Concatenated descriptors as a single numpy array, or None if extraction fails
    """
    scene_features = extract_aggregated_features(video_path)
    if scene_features is None:
        return None
    
    # Concatenate all scene features into a single array
    return np.vstack(scene_features) if scene_features else None


def compare_features(
    features_a: Optional[List[np.ndarray] | np.ndarray], 
    features_b: Optional[List[np.ndarray] | np.ndarray]
) -> float:
    """
    Compare features between two videos using scene-level matching.
    
    Args:
        features_a: Scene-based features (list of ndarrays) or legacy concatenated features
        features_b: Scene-based features (list of ndarrays) or legacy concatenated features
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if features_a is None or features_b is None:
        return 0.0
    
    # Handle legacy format (single concatenated array)
    if isinstance(features_a, np.ndarray) and isinstance(features_b, np.ndarray):
        return _compare_legacy_features(features_a, features_b)
    
    # Handle new scene-based format
    if isinstance(features_a, list) and isinstance(features_b, list):
        return _compare_scene_features(features_a, features_b)
    
    # Handle mixed formats
    logger.warning("Mixed feature formats detected, using legacy comparison")
    if isinstance(features_a, list):
        features_a = np.vstack(features_a) if features_a else None
    if isinstance(features_b, list):
        features_b = np.vstack(features_b) if features_b else None
    
    if features_a is None or features_b is None:
        return 0.0
    
    return _compare_legacy_features(features_a, features_b)


def _compare_legacy_features(descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> float:
    """Legacy comparison method for concatenated features."""
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return 0.0

    # Initialize BFMatcher
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)

    # Perform KNN matching
    matches = bf.knnMatch(descriptors_a, descriptors_b, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Calculate normalized score
    min_features = min(len(descriptors_a), len(descriptors_b))
    if min_features == 0:
        return 0.0

    return len(good_matches) / min_features


def _compare_scene_features(
    scenes_a: List[np.ndarray], 
    scenes_b: List[np.ndarray]
) -> float:
    """
    Compare features using scene-level matching with density-aware scoring.
    
    Args:
        scenes_a: List of descriptor arrays for video A
        scenes_b: List of descriptor arrays for video B
        
    Returns:
        Scene-based similarity score accounting for density differences
    """
    if not scenes_a or not scenes_b:
        return 0.0
    
    # Initialize BFMatcher
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    
    # Calculate total descriptors for density normalization
    total_descriptors_a = sum(len(scene) for scene in scenes_a)
    total_descriptors_b = sum(len(scene) for scene in scenes_b)
    
    if total_descriptors_a == 0 or total_descriptors_b == 0:
        return 0.0
    
    # Scene matching with best-first approach
    scene_matches = []
    
    # For each scene in video A, find best matching scene in video B
    for scene_a in scenes_a:
        if len(scene_a) == 0:
            continue
            
        best_scene_score = 0.0
        
        for scene_b in scenes_b:
            if len(scene_b) == 0:
                continue
                
            # Perform KNN matching between scenes
            matches = bf.knnMatch(scene_a, scene_b, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Calculate scene-level score
            min_scene_features = min(len(scene_a), len(scene_b))
            if min_scene_features > 0:
                scene_score = len(good_matches) / min_scene_features
                best_scene_score = max(best_scene_score, scene_score)
        
        scene_matches.append(best_scene_score)
    
    if not scene_matches:
        return 0.0
    
    # Calculate weighted similarity score
    # Weight by scene size to account for density differences
    scene_weights = [len(scene) / total_descriptors_a for scene in scenes_a if len(scene) > 0]
    
    if len(scene_weights) != len(scene_matches):
        scene_weights = [1.0 / len(scene_matches)] * len(scene_matches)
    
    weighted_score = sum(score * weight for score, weight in zip(scene_matches, scene_weights))
    
    # Apply density compensation factor
    density_ratio = min(total_descriptors_a, total_descriptors_b) / max(total_descriptors_a, total_descriptors_b)
    density_compensation = 0.5 + 0.5 * density_ratio
    
    final_score = weighted_score * density_compensation
    
    return max(0.0, min(1.0, final_score))
