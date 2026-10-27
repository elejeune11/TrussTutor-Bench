from llm_api.run_minimax_client import (
    build_minimax_request,
    run_minimax,
    build_minimax_request_formative,
    run_minimax_formative,
    describe_schema_for_minimax,
)
from pathlib import Path
import json
import pytest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

payload_dict = {
    "benchmark_id": "TRUSS_0001",
    "user_prompt": "A statically determinate equilateral truss is supported by a pin at A and a roller at B. A point load of P 50.0 kN is applied at the apex (Node C). If the bottom chord length is L, determine the maximum compression force in the structure.",
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC",
    "structured_schema": {
        "type": "object",
        "properties": {
            "max_compression_magnitude": {
                "type": "number",
                "description": "The numerical value of the maximum compressive force."
            },
            "units": {
                "type": "string",
                "description": "The units of the force."
            },
            "controlling_members": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True
            }
        },
        "required": [
            "max_compression_magnitude",
            "units"
        ],
        "additionalProperties": False
    }
}

formative_payload_dict = {
    "system_prompt": "You are a helpful assistant that provides step-by-step solutions to engineering problems.",
    "base_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC",
    "student_work": [
    {"type": "text", "content": "Step 1: I calculated the reactions..."},
    {"type": "image", "content_base64": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC"},
    {"type": "text", "content": "Step 2: I summed moments about A..."}
    ],
    "structured_schema": payload_dict["structured_schema"],
}


def test_describe_schema_for_minimax_mentions_json_and_includes_schema():
    """The schema description asks for JSON and contains the schema."""

    description = describe_schema_for_minimax(payload_dict["structured_schema"])

    # MiniMax has no structured-output support at all (not even DeepSeek's
    # generic json_object mode) -- the expected structure must be described
    # entirely in the prompt text.
    assert "json" in description.lower()
    assert json.dumps(payload_dict["structured_schema"], indent=2) in description


def test_build_minimax_request():
    """Builds the baseline request with the model, token limit, thinking settings and messages."""

    llm_model = "MiniMax-M3"
    output_dict = build_minimax_request(llm_model, payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 5
    assert output_dict["model"] == "MiniMax-M3"
    assert output_dict["max_completion_tokens"] == 100000
    assert output_dict["thinking"] == {"type": "adaptive"}
    assert output_dict["reasoning_split"] is True
    assert output_dict["messages"][0]["role"] == "user"
    assert output_dict["messages"][0]["content"][0]["type"] == "text"
    assert payload_dict["user_prompt"] in output_dict["messages"][0]["content"][0]["text"]
    assert describe_schema_for_minimax(payload_dict["structured_schema"]) in output_dict["messages"][0]["content"][0]["text"]
    assert output_dict["messages"][0]["content"][1]["type"] == "image_url"
    assert output_dict["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{payload_dict['image_base64']}"


def test_build_minimax_request_formative():
    """Builds the formative request with the system prompt, the student work and the thinking settings."""

    llm_model = "MiniMax-M3"
    output_dict = build_minimax_request_formative(llm_model, formative_payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 5
    assert output_dict["max_completion_tokens"] == 100000
    assert output_dict["thinking"] == {"type": "adaptive"}
    assert output_dict["reasoning_split"] is True
    assert output_dict["messages"][0]["role"] == "system"
    assert formative_payload_dict["system_prompt"] in output_dict["messages"][0]["content"]
    assert describe_schema_for_minimax(formative_payload_dict["structured_schema"]) in output_dict["messages"][0]["content"]
    assert output_dict["messages"][1]["role"] == "user"
    assert output_dict["messages"][1]["content"][0]["type"] == "image_url"
    assert output_dict["messages"][1]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{formative_payload_dict['base_image_base64']}"
    assert output_dict["messages"][1]["content"][1]["type"] == "text"
    assert output_dict["messages"][1]["content"][1]["text"] == formative_payload_dict["student_work"][0]["content"]
    assert output_dict["messages"][1]["content"][2]["type"] == "image_url"
    assert output_dict["messages"][1]["content"][2]["image_url"]["url"] == f"data:image/png;base64,{formative_payload_dict['student_work'][1]['content_base64']}"
    assert output_dict["messages"][1]["content"][3]["type"] == "text"
    assert output_dict["messages"][1]["content"][3]["text"] == formative_payload_dict["student_work"][2]["content"]


async def test_run_minimax(tmp_path):
    """Returns the parsed JSON answer with token usage from a baseline call."""

    json_file = tmp_path/"minimax_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"text1": "value1", "text2": "value2"}))
    fake_usage = SimpleNamespace(completion_tokens = 123)
    finish_reason = "stop"
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = finish_reason)
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "MiniMax-M3"

    with patch("llm_api.run_minimax_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_minimax(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 123, "status": "stop"}
    }


async def test_run_minimax_extracts_json_from_prose_wrapped_content(tmp_path):
    """Extracts the JSON answer when the model writes prose before it."""

    json_file = tmp_path/"minimax_request.json"
    json_file.write_text(json.dumps(payload_dict))

    # Real observed MiniMax behavior (2026-09-23 smoke test): with no
    # structured-output enforcement, it narrates a full worked derivation
    # before the JSON answer instead of returning JSON alone.
    prose_wrapped_content = (
        "Looking at this problem, I need to analyze an equilateral "
        "triangular truss...\n\n**Step 1: Calculate Reactions**\n\n"
        "By symmetry, A_y = B_y = 25 kN.\n\n"
        '{"max_compression_magnitude": 28.87, "units": "kN"}'
    )
    fake_message = SimpleNamespace(content = prose_wrapped_content)
    fake_usage = SimpleNamespace(completion_tokens = 2687)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "MiniMax-M3"

    with patch("llm_api.run_minimax_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_minimax(llm_model, json_file)

    assert response["max_compression_magnitude"] == 28.87
    assert response["units"] == "kN"


async def test_run_minimax_sends_thinking_and_reasoning_split_via_extra_body(tmp_path):
    """Sends thinking and reasoning_split through extra_body, not as SDK arguments."""

    json_file = tmp_path/"minimax_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"ok": True}))
    fake_usage = SimpleNamespace(completion_tokens = 50)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "MiniMax-M3"

    with patch("llm_api.run_minimax_client.AsyncOpenAI") as mock_client:

        create_mock = AsyncMock(return_value = fake_response)
        mock_client.return_value.chat.completions.create = create_mock

        await run_minimax(llm_model, json_file)

    _, kwargs = create_mock.call_args
    assert "thinking" not in kwargs
    assert "reasoning_split" not in kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "adaptive"}, "reasoning_split": True}


async def test_run_minimax_raises_on_empty_content(tmp_path):
    """Raises RuntimeError when the response content is empty."""

    json_file = tmp_path/"minimax_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "")
    fake_usage = SimpleNamespace(completion_tokens = 20000)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "length")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "MiniMax-M3"

    with patch("llm_api.run_minimax_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="length"):
            await run_minimax(llm_model, json_file)


async def test_run_minimax_raises_on_malformed_json_content(tmp_path):
    """Raises RuntimeError when the response content is not valid JSON."""

    json_file = tmp_path/"minimax_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "```json\n{not valid json\n```")
    fake_usage = SimpleNamespace(completion_tokens = 500)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "MiniMax-M3"

    with patch("llm_api.run_minimax_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="Could not parse"):
            await run_minimax(llm_model, json_file)


async def test_run_minimax_formative(tmp_path):
    """Returns the parsed JSON answer with token usage from a formative call."""

    json_file = tmp_path/"minimax_formative_request.json"
    json_file.write_text(json.dumps(formative_payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"is_work_correct": True}))
    fake_usage = SimpleNamespace(completion_tokens = 200)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "MiniMax-M3"

    with patch("llm_api.run_minimax_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_minimax_formative(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "is_work_correct": True,
        "_usage": {"tokens": 200, "status": "stop"}
    }
