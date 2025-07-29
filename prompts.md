### **Project Blueprint & Development Plan**

The development will proceed in two main phases, corresponding to the two deliverables. We will build the entire foundation for the POC first, ensuring a working, end-to-end program. Then, we will swap out the core matching logic and enhance the output for the full-featured version.

**Phase 1: Deliverable A - Proof of Concept (Global Similarity)**

1.  **Project Scaffolding:** Set up the project structure, dependencies, and a basic entry point.
2.  **Argument Parsing & Validation:** Implement the CLI to accept and validate video file paths.
3.  **Keyframe & Feature Extraction:** Create a core processing function to extract keyframes and aggregate their ORB features for a single video.
4.  **Pairwise Comparison (POC):** Implement the function to compare two sets of aggregated features and return a single similarity score.
5.  **Main Loop & Wiring (POC):** Write the main application logic to process all videos, run the all-pairs comparison, and print simple results to the console.
6.  **JSON Output (POC):** Add the final step to format significant matches into the specified JSON structure.

**Phase 2: Deliverable B - Full Feature (Subsegment Matching)**

7.  **Refactor Feature Extraction:** Modify the extraction process to keep features associated with individual keyframes instead of aggregating them.
8.  **Similarity Matrix:** Create a function to build a 2D similarity matrix by comparing every keyframe of Video A with every keyframe of Video B.
9.  **Alignment Algorithm:** Implement the core dynamic programming (Smith-Waterman) algorithm to find the optimal alignment path in the similarity matrix. This will be in a separate, testable module.
10. **Result Analysis:** Create a function to interpret the alignment path, calculating all the required metrics (time ranges, coverage, average score).
11. **Main Loop & Wiring (Full Feature):** Replace the POC's comparison logic in the main loop with the new, three-step subsegment matching process (matrix -> alignment -> analysis).
12. **JSON Output (Full Feature):** Update the JSON generation function to produce the final, more detailed schema with time ranges and advanced metrics.

---

### **Prompts for Code-Generation LLM**

Here are the sequential prompts to provide to a code-generation LLM.

### **Phase 1: Deliverable A - Proof of Concept**

#### **Prompt 1: Project Scaffolding**

```text
Create a Python project structure for a video similarity tool. The project should use `uv` for dependency management.

1.  Create a `pyproject.toml` file. Specify Python 3.9+ and add the following dependencies:
    - `opencv-python`
    - `scenedetect`
    - `numpy`

2.  Create a `src` directory.
3.  Inside `src`, create a directory named `video_similarity`.
4.  Inside `src/video_similarity`, create an empty `__init__.py` file.
5.  Inside `src/video_similarity`, create the main entry point file named `main.py`.
6.  In `main.py`, write a placeholder main function that prints "Video Similarity Tool" and is called using the standard `if __name__ == "__main__":` block.
```

---

#### **Prompt 2: Argument Parsing and Validation**

```text
Modify the `src/video_similarity/main.py` file from the previous step.

1.  Import the `argparse` and `pathlib` standard libraries.
2.  Define a `dataclasses.dataclass` named `VideoFile` to hold information about each video. It should have two fields: `path` (a `pathlib.Path` object) and `is_valid` (a boolean).
3.  Implement a function `parse_and_validate_args() -> list[VideoFile]`. This function will:
    - Use `argparse` to accept a variable number of positional arguments (`nargs='+'`) representing video file paths.
    - Loop through the provided paths. For each path:
        - Check if the file exists and is a file (not a directory).
        - If it is, create a `VideoFile` instance with `is_valid=True`.
        - If it is not, print a warning to the console (e.g., "WARNING: File not found or is a directory: [path]") and create a `VideoFile` instance with `is_valid=False`.
    - The function should return a list of `VideoFile` objects.
4.  In the `main` function, call `parse_and_validate_args()` and then filter the list to get only the valid videos.
5.  Add a check: if the number of valid videos is less than 2, print an error message and exit the program. Otherwise, print the list of valid video paths found.
```

---

#### **Prompt 3: Keyframe & Feature Extraction (POC)**

```text
Create a new file `src/video_similarity/processing.py`. This module will handle all video processing logic.

1.  Import `cv2`, `numpy`, and `scenedetect`. From `scenedetect` import `video_manager`, `scene_manager`, and `detectors.ContentDetector`.
2.  Create a function `extract_aggregated_features(video_path: str) -> np.ndarray | None`. This function will perform the core processing for the POC.
3.  Inside the function:
    - Use a `try...except` block to catch potential errors during video processing (e.g., corrupt files). If an error occurs, print a warning and return `None`.
    - Create a `VideoManager` with the `video_path`.
    - Create a `SceneManager` and add a `ContentDetector` to it.
    - Call `scene_manager.detect_scenes(video=video_manager, show_progress=True)`.
    - Get the list of scenes (as frame numbers or timecodes). If no scenes are detected, print a warning and return `None`.
    - Initialize an ORB detector: `orb = cv2.ORB_create()`.
    - Create an empty list to hold all feature descriptors.
    - Loop through the start frame of each detected scene. For each keyframe:
        - Set the `video_manager` position to that frame.
        - Read the frame.
        - Convert the frame to grayscale.
        - Detect features and compute descriptors using `orb.detectAndCompute()`.
        - If descriptors are found, add them to the list of all descriptors.
    - If no descriptors were found across all keyframes, return `None`.
    - Vertically stack all collected descriptors into a single NumPy array using `np.vstack()`.
    - Return the final stacked NumPy array of descriptors.
```

---

#### **Prompt 4: Pairwise Comparison (POC)**

```text
Add a new function to the `src/video_similarity/processing.py` file.

1.  The function signature should be `compare_features_poc(descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> float`.
2.  Inside the function:
    - Initialize a Brute-Force Matcher: `bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)`.
    - Use `bf.knnMatch(descriptors_a, descriptors_b, k=2)` to find the k-nearest neighbors.
    - Apply Lowe's ratio test: Iterate through the matches and keep only the "good" ones where `m.distance < 0.75 * n.distance`.
    - Calculate a normalized score: `score = len(good_matches) / min(len(descriptors_a), len(descriptors_b))`. Handle the edge case of zero descriptors to avoid division by zero (return 0.0).
    - Return the normalized score.
```

---

#### **Prompt 5: Main Loop & Wiring (POC)**

```text
Now, let's wire everything together in `src/video_similarity/main.py`.

1.  Import the new functions `extract_aggregated_features` and `compare_features_poc` from `processing.py`.
2.  Import `itertools`.
3.  In the `main` function, after validating the video paths:
    - Create an empty dictionary `video_features` to store the features for each video.
    - Loop through the valid `VideoFile` objects. For each one:
        - Print a message like "Processing [video_path]...".
        - Call `extract_aggregated_features()` with the video path.
        - If the result is not `None`, store it in the `video_features` dictionary with the path as the key.
    - Use `itertools.combinations(video_features.keys(), 2)` to get all unique pairs of video paths.
    - Loop through these pairs. For each `(path_a, path_b)` pair:
        - Retrieve their descriptors from the `video_features` dictionary.
        - Call `compare_features_poc()` with the two sets of descriptors.
        - Print the result to the console, e.g., "[path_a] vs [path_b] - Score: [score]".
```

---

#### **Prompt 6: JSON Output (POC)**

```text
Let's add the final JSON output generation for the POC.

1.  Create a new file `src/video_similarity/output.py`.
2.  In this new file, import `json`.
3.  Create a function `generate_json_output_poc(matches: list[dict]) -> str`. This function will take a list of match dictionaries and dump it into a nicely formatted JSON string. The schema should be `{"matches": [...]}`.
4.  Now, modify `src/video_similarity/main.py`:
    - Import `generate_json_output_poc` from `output.py`.
    - In the `main` function, create an empty list called `significant_matches` before the `itertools.combinations` loop.
    - Inside the loop, after calculating the similarity score, add a threshold check: `if score >= 0.5:`.
    - If the condition is met, create a dictionary for the match according to the POC schema and append it to the `significant_matches` list.
    - After the loop finishes, call `generate_json_output_poc()` with the `significant_matches` list.
    - Print the resulting JSON string to the console.
```

### **Phase 2: Deliverable B - Full Feature**

#### **Prompt 7: Refactor Feature Extraction for Subsegments**

```text
We will now begin upgrading to the full-featured version. Modify `src/video_similarity/processing.py`.

1.  Create a new `dataclasses.dataclass` at the top of the file named `KeyframeFeatures`. It should have two fields: `timestamp_sec` (float) and `descriptors` (np.ndarray).
2.  Rename the old `extract_aggregated_features` function to `extract_aggregated_features_poc` to preserve it for reference if needed, but we will no longer use it.
3.  Create a new function `extract_keyframe_features(video_path: str) -> list[KeyframeFeatures] | None`.
4.  This new function will be very similar to the old one, but with a key difference:
    - Instead of pooling all descriptors, it should create a list of `KeyframeFeatures` objects.
    - For each detected keyframe, it will extract its ORB descriptors.
    - If descriptors are found for a keyframe, it will create a `KeyframeFeatures` instance containing the keyframe's timestamp (in seconds) and its descriptors, and append it to a list.
    - The function will return the list of `KeyframeFeatures` objects. If no keyframes or features are found, it should return an empty list or `None` and print a warning.
```

---

#### **Prompt 8: Build the Similarity Matrix**

```text
In `src/video_similarity/processing.py`, add a function to build the similarity matrix.

1.  The function signature should be `build_similarity_matrix(features_a: list[KeyframeFeatures], features_b: list[KeyframeFeatures]) -> np.ndarray`.
2.  Inside the function:
    - Get the number of keyframes `M` from `features_a` and `N` from `features_b`.
    - Initialize an `M x N` NumPy array with zeros: `similarity_matrix = np.zeros((M, N))`.
    - Create a `BFMatcher` and set up the ratio test logic, just like in the `compare_features_poc` function.
    - Use nested loops to iterate through each keyframe `i` in `features_a` and each keyframe `j` in `features_b`.
    - In the inner loop, perform a feature comparison between `features_a[i].descriptors` and `features_b[j].descriptors`.
    - Calculate the normalized similarity score for this pair of keyframes.
    - Assign this score to `similarity_matrix[i, j]`.
    - Return the completed `similarity_matrix`.
```

---

#### **Prompt 9: The Alignment Algorithm**

```text
This is the most complex part. Create a new file `src/video_similarity/alignment.py`.

1.  Import `numpy`.
2.  Create a function `find_optimal_path(similarity_matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]`. This function will implement the Smith-Waterman algorithm for local sequence alignment.
3.  Inside the function:
    - Define gap penalties (e.g., `gap_penalty = -0.1`, but make it a parameter with a default).
    - Get the dimensions of the `similarity_matrix`.
    - Create a scoring matrix of the same size, initialized to zeros.
    - Create a traceback matrix of the same size to store movement directions (e.g., 0 for stop, 1 for diagonal, 2 for up, 3 for left).
    - Keep track of the max score found and its position (`max_score`, `max_pos`).
    - Use nested loops to fill the scoring matrix. For each cell `(i, j)`:
        - Calculate the score for a diagonal move: `score_matrix[i-1, j-1] + similarity_matrix[i, j]`.
        - Calculate the score for a move from above: `score_matrix[i-1, j] + gap_penalty`.
        - Calculate the score for a move from the left: `score_matrix[i, j-1] + gap_penalty`.
        - The cell's score is the maximum of these three values and 0 (for local alignment).
        - Record the direction of the move in the traceback matrix.
        - Update `max_score` and `max_pos` if a new maximum is found.
    - After filling the matrices, perform the traceback starting from `max_pos`.
        - Follow the directions in the traceback matrix until a score of 0 is reached.
        - Collect the `(i, j)` coordinates of the path.
    - The path will be in reverse order, so reverse it before returning.
    - Return the path (a list of coordinate tuples) and the `max_score`.
```

---

#### **Prompt 10: Result Analysis**

```text
Now, create a function to make sense of the alignment path. Add this to `src/video_similarity/processing.py`.

1.  Import the `KeyframeFeatures` dataclass and the `find_optimal_path` function.
2.  Create a new `dataclasses.dataclass` named `MatchResult` to hold the final, structured results. It should contain fields for `file_a`, `file_b`, time ranges (seconds and hms), and all the required metrics from the spec.
3.  Create a function `analyze_path(path: list, total_score: float, features_a: list[KeyframeFeatures], features_b: list[KeyframeFeatures]) -> dict`.
4.  Inside the function:
    - If the path is empty or too short (e.g., less than 3 keyframes), return an empty dictionary.
    - Get the start and end keyframe indices for both videos from the path (e.g., `start_idx_a = path[0][0]`, `end_idx_a = path[-1][0]`).
    - Extract the start and end timestamps in seconds from the `features_a` and `features_b` lists using these indices.
    - Calculate `matched_keyframe_count` (the length of the path).
    - Calculate `average_match_score` (`total_score / matched_keyframe_count`).
    - Calculate `coverage_short_video`. This is `matched_keyframe_count` divided by the length of the shorter of the two *full* keyframe lists (`features_a` or `features_b`).
    - Create a dictionary containing all the metrics: start/end times, scores, counts, etc. You can also include a helper to format the seconds into HH:MM:SS.ms strings.
    - Return this dictionary of results.
```

---

#### **Prompt 11: Main Loop & Wiring (Full Feature)**

```text
Time to replace the core logic in `src/video_similarity/main.py`.

1.  Import the new functions: `extract_keyframe_features`, `build_similarity_matrix`, `analyze_path`, and the new `find_optimal_path` from `alignment.py`.
2.  In the `main` function, the logic to get `video_features` must change.
    - It should now call `extract_keyframe_features` and store the list of `KeyframeFeatures` objects for each video.
3.  In the `itertools.combinations` loop:
    - Remove the call to `compare_features_poc`.
    - Add the new three-step process:
        a. `similarity_matrix = build_similarity_matrix(features_a, features_b)`
        b. `path, score = find_optimal_path(similarity_matrix)`
        c. `result_data = analyze_path(path, score, features_a, features_b)`
4.  After getting `result_data`, check if it's not empty. Then apply the new, more complex thresholds:
    - `if result_data['coverage_short_video'] >= 0.75 and result_data['average_match_score'] >= 0.7:`
    - If the conditions are met, append the `result_data` to your `significant_matches` list. Don't forget to add the file paths to this dictionary.
    - Also, print a summary of the found match to the console.
```

---

#### **Prompt 12: Final JSON Output (Full Feature)**

```text
Finally, update the JSON output to match the full feature specification. Modify `src/video_similarity/output.py`.

1.  Rename the old `generate_json_output_poc` function to avoid confusion.
2.  Create a new function `generate_json_output_full(matches: list[dict]) -> str`.
3.  This function will take the list of `result_data` dictionaries from the main loop.
4.  It will format each dictionary into the final, detailed JSON object structure specified in the "JSON Schema for Deliverable B", including all time ranges and metrics.
5.  It will then wrap all these objects in the top-level `{"matches": [...]}` structure and return the final JSON string.
6.  In `src/video_similarity/main.py`, update the final call to use this new `generate_json_output_full` function.
```
