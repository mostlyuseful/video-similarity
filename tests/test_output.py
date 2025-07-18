from video_similarity.output import generate_json_output_poc
import json


def test_generate_json_output_poc_empty_matches():
    output = generate_json_output_poc([])
    parsed = json.loads(output)
    assert parsed == {"matches": []}


def test_generate_json_output_poc_with_matches():
    matches = [
        {
            "file_a": "video1.mp4",
            "file_b": "video2.mp4",
            "metrics": {"normalized_match_score": 0.85},
        },
        {
            "file_a": "video1.mp4",
            "file_b": "video3.mp4",
            "metrics": {"normalized_match_score": 0.72},
        },
    ]

    output = generate_json_output_poc(matches)
    parsed = json.loads(output)
    assert parsed == {"matches": matches}
