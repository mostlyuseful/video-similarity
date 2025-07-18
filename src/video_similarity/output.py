import json


def generate_json_output_poc(matches: list[dict]) -> str:
    """
    Generate JSON output for POC matches
    :param matches: List of match dictionaries
    :return: Formatted JSON string
    """
    output = {"matches": matches}
    return json.dumps(output, indent=2)
