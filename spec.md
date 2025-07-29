### **Project Specification: Video Similarity Detector**

#### **1. Overview**

The goal is to create a Python command-line program that analyzes a small set of video files to identify similarity. The program will handle videos with different resolutions, encodings, and minor color variations. The project will be developed in two phases:

*   **Deliverable A (POC):** Focuses on detecting **global similarity** between entire videos.
*   **Deliverable B (Full Feature):** Extends the POC to identify **matching subsegments** within videos, providing precise time ranges for the matches.

#### **2. General Requirements (Applicable to both Deliverables)**

These requirements form the foundation of the program and should be implemented in Deliverable A.

*   **2.1. User Interface:**
    *   A Command-Line Interface (CLI).
    *   The user will invoke the script by passing video file paths as arguments:
        ```bash
        python main.py /path/to/video1.mp4 /path/to/video2.mkv /path/to/video3.mov
        ```

*   **2.2. Comparison Scope:**
    *   The program will perform an all-pairs comparison. For N input videos, it will conduct N * (N-1) / 2 unique comparisons.

*   **2.3. Error Handling:**
    *   **Invalid File Paths:** If a path is invalid (does not exist, is a directory), print an error to the console and skip that file. The program should continue if at least two valid videos remain.
    *   **Video Processing Failure:** If a video file is corrupt or cannot be opened by OpenCV, treat it as an invalid path.
    *   **No Features Extracted:** If SceneDetect finds no keyframes in a valid video, print a warning and exclude that video from comparisons.
    *   **Exit Condition:** If fewer than two processable videos are available after initial validation, the program should exit with an informative message.

*   **2.4. Performance:**
    *   The primary focus is on correctness and functionality. Performance optimization is a secondary concern for these initial versions.

*   **2.5. Licensing:**
    *   To be determined at a later date.

---

### **3. Deliverable A: Proof of Concept (Global Video Similarity)**

**Objective:** Implement the core framework and a simplified matching logic to determine if two videos are globally similar (i.e., contain largely the same content, ignoring intros/outros/minor edits).

*   **3.1. Architecture & Algorithm:**
    1.  **Input Validation:** Process CLI arguments and validate video file paths as per section 2.3.
    2.  **Keyframe Extraction:** For each valid video, use `SceneDetect` (content-aware detection) to extract a list of keyframes. Store these keyframes in memory.
    3.  **Feature Aggregation:** For each video, iterate through its extracted keyframes and extract `ORB` feature descriptors. Pool all descriptors for a single video into one large "bag of features".
    4.  **Pairwise Comparison:** For each pair of videos:
        *   Use OpenCV's `BFMatcher` (Brute-Force Matcher) with `KNN` (k=2) to match the aggregated feature "bag" of Video A against that of Video B.
        *   Apply Lowe's ratio test to filter for "good" matches.
    5.  **Similarity Score Calculation:**
        *   Calculate a `normalized_match_score` for the pair:
            `score = (number of good matches) / min(total features in Video A, total features in Video B)`
        *   This score will be a float between 0.0 and 1.0.
    6.  **Thresholding:** A match is considered "significant" if `normalized_match_score >= 0.5`. This value should be easily configurable at the top of the script.

*   **3.2. Output:**
    *   **Console:** Print progress updates (e.g., "Processing video1.mp4..."), warnings, and the result of each comparison (e.g., "video1.mp4 vs video2.mp4 - Score: 0.85 (Match)").
    *   **JSON Output:** Generate a single JSON object to `stdout`. It will contain a `matches` array. Only pairs that meet the significance threshold are included.

*   **3.3. JSON Schema for Deliverable A:**
    ```json
    {
      "matches": [
        {
          "file_a": "/path/to/video_a.mkv",
          "file_b": "/path/to/dir/video_b.mp4",
          "metrics": {
            "normalized_match_score": 0.85
          }
        }
      ]
    }
    ```

---

### **4. Deliverable B: Full Feature Implementation (Subsegment Matching)**

**Objective:** Upgrade the POC to identify specific, time-aligned subsegments that are similar between videos. This replaces the global matching logic with a more sophisticated temporal alignment algorithm.

*   **4.1. Architecture & Algorithm (Upgrade):**
    *   The Keyframe Extraction and Feature Extraction steps from Deliverable A remain the same. The change is in the comparison logic.
    *   **New Comparison Logic:**
        1.  **Do not aggregate features.** Keep features associated with their original keyframe (and its timestamp).
        2.  For each pair of videos (Video A with M keyframes, Video B with N keyframes), create an **M x N similarity matrix**.
        3.  The value at `Matrix[i][j]` is the similarity score between `keyframe_A[i]` and `keyframe_B[j]`. This score is calculated using `ORB` + `BFMatcher` as in the POC, but on a frame-by-frame basis. The score should be normalized (e.g., number of good matches / min features in frames).
        4.  Implement a **Dynamic Programming algorithm** (functionally similar to Smith-Waterman or Dynamic Time Warping) on this similarity matrix to find the optimal alignment path. This path represents the longest, contiguous sequence of best-matching keyframes that maintain temporal order in both videos.
        5.  From the start and end points of this optimal path, extract the corresponding keyframe indices for both videos.

*   **4.2. Thresholding:**
    *   A subsegment match is considered "significant" and included in the JSON output only if **both** of the following conditions are met:
        1.  **Coverage:** The number of matched keyframes is `> 75%` of the total keyframes in the shorter of the two video segments being compared.
        2.  **Quality:** The `average similarity score` per matched keyframe pair (total path score / path length) is `> 0.7`.
    *   These threshold values should be easily configurable.

*   **4.3. Output (Upgrade):**
    *   **Console:** Should now report on subsegment findings or lack thereof.
    *   **JSON Output:** The JSON schema is updated to include time ranges and more detailed metrics.

*   **4.4. JSON Schema for Deliverable B:**
    ```json
    {
      "matches": [
        {
          "file_a": "/path/to/video_a.mkv",
          "file_b": "/path/to/dir/video_b.mp4",
          "time_range_a_seconds": [197.4, 601.0],
          "time_range_b_seconds": [50.2, 453.8],
          "time_range_a_hms": "00:03:17.400-00:10:01.000",
          "time_range_b_hms": "00:00:50.200-00:07:33.800",
          "metrics": {
            "total_similarity_score": 1234.56,
            "matched_keyframe_count": 50,
            "average_match_score": 0.82,
            "coverage_short_video": 0.91
          }
        }
      ]
    }
    ```

---

### **5. Testing Plan**

A small suite of test videos should be prepared to validate the implementation at both stages.

*   **5.1. Unit Tests:**
    *   Test file path validation logic.
    *   Test the image-to-image similarity function in isolation.
    *   For Deliverable B, test the dynamic programming algorithm with a small, pre-computed similarity matrix to ensure it finds the correct path.

*   **5.2. Integration & Acceptance Tests:**
    *   **Test Case 1 (Identical Videos):** Two identical video files.
        *   *A-Expected:* High similarity score (~1.0).
        *   *B-Expected:* A match covering the full duration of both videos.
    *   **Test Case 2 (Different Encoding/Resolution):** The same video content saved with different codecs or resolutions.
        *   *A-Expected:* High similarity score.
        *   *B-Expected:* A match covering the full duration.
    *   **Test Case 3 (Subset Video):** Video B is a 30-second clip taken from the middle of Video A.
        *   *A-Expected:* Moderate similarity score.
        *   *B-Expected:* A significant match with `time_range_b` covering the full clip and `time_range_a` corresponding to the correct segment in Video A.
    *   **Test Case 4 (Cropped Video):** Video B is the same as Video A but with significant black bars cropped.
        *   *Expected (A & B):* A strong match should still be found.
    *   **Test Case 5 (Unrelated Videos):** Two completely different videos.
        *   *Expected (A & B):* No entry in the JSON output `matches` array.
    *   **Test Case 6 (Invalid Input):** Run the script with a mix of valid paths, a non-existent path, and a path to a non-video file.
        *   *Expected:* The script should print errors for the invalid paths but successfully complete the comparison for the valid ones.

### **6. Dependencies & Environment**

*   **Language:** Python 3.9+
*   **Dependency Management:** `uv` will be used for environment and package management.
*   **Core Dependencies:** A `requirements.txt` file will be provided, including:
    *   `opencv-python`
    *   `pyscenedetect[opencv]`
    *   `numpy`
