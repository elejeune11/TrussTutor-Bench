from types import SimpleNamespace

from llm_api.run_gemini_client import _usage_from_response


def test_usage_from_response_counts_thinking_in_output_tokens():
    """Output tokens are total minus prompt tokens, so thinking tokens are counted."""

    fake_usage = SimpleNamespace(prompt_token_count=1500, candidates_token_count=300, total_token_count=4800)
    fake_response = SimpleNamespace(usage_metadata=fake_usage)

    usage = _usage_from_response(fake_response)

    assert usage == {"tokens": 3300, "answer_tokens": 300, "input_tokens": 1500}


import json
from unittest.mock import patch

from llm_api.run_gemini_client import run_gemini, run_gemini_formative

TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC"
SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}


def make_fake_response():
    """
    Builds a fake Gemini response with a JSON answer and token counts.

    Returns:
        SimpleNamespace: The fake response.
    """
    return SimpleNamespace(
        text='{"chain_of_thought": "...", "answer": 42}',
        usage_metadata=SimpleNamespace(prompt_token_count=1500, candidates_token_count=300, total_token_count=4800),
    )


def run_with_mock(runner, payload, tmp_path, *model_args):
    """
    Runs a Gemini function on a payload with the Gemini model mocked.

    Args:
        runner (callable): run_gemini or run_gemini_formative.
        payload (dict): Payload written to a temporary file.
        tmp_path (Path): Folder for the payload file.
        *model_args: Optional model name passed to the runner.

    Returns:
        tuple: (result, mock of the GenerativeModel class).
    """
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(payload))
    with patch("llm_api.run_gemini_client.genai.GenerativeModel") as mock_model_cls:
        mock_model_cls.return_value.generate_content.return_value = make_fake_response()
        result = runner(str(payload_file), *model_args)
    return result, mock_model_cls


def test_run_gemini_defaults_to_pro(tmp_path):
    """Uses gemini-3.1-pro-preview when no model is given."""

    payload = {"benchmark_id": "T1", "user_prompt": "p", "image_base64": TINY_PNG_B64, "structured_schema": SCHEMA}

    result, mock_model_cls = run_with_mock(run_gemini, payload, tmp_path)

    mock_model_cls.assert_called_once_with("gemini-3.1-pro-preview")
    assert result["answer"] == 42
    assert result["_usage"]["tokens"] == 3300


def test_run_gemini_uses_the_model_it_is_given(tmp_path):
    """Uses the model name it is given."""

    payload = {"benchmark_id": "T1", "user_prompt": "p", "image_base64": TINY_PNG_B64, "structured_schema": SCHEMA}

    result, mock_model_cls = run_with_mock(run_gemini, payload, tmp_path, "gemini-3.8-flash")

    mock_model_cls.assert_called_once_with("gemini-3.8-flash")


def test_run_gemini_formative_defaults_to_pro(tmp_path):
    """Uses gemini-3.1-pro-preview when no model is given."""

    payload = {"benchmark_id": "T1", "system_prompt": "s", "base_image_base64": TINY_PNG_B64,
               "student_work": [{"type": "text", "content": "step 1"}], "structured_schema": SCHEMA}

    result, mock_model_cls = run_with_mock(run_gemini_formative, payload, tmp_path)

    mock_model_cls.assert_called_once_with("gemini-3.1-pro-preview")
    assert result["answer"] == 42


def test_run_gemini_formative_uses_the_model_it_is_given(tmp_path):
    """Uses the model name it is given."""

    payload = {"benchmark_id": "T1", "system_prompt": "s", "base_image_base64": TINY_PNG_B64,
               "student_work": [{"type": "text", "content": "step 1"}], "structured_schema": SCHEMA}

    result, mock_model_cls = run_with_mock(run_gemini_formative, payload, tmp_path, "gemini-3.8-flash")

    mock_model_cls.assert_called_once_with("gemini-3.8-flash")
