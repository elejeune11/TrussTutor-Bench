from llm_api.run_deepseek_client import (
    build_deepseek_request,
    run_deepseek,
    build_deepseek_request_formative,
    run_deepseek_formative,
    describe_schema_for_deepseek,
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


def test_describe_schema_for_deepseek_mentions_json_and_includes_schema():
    """The schema description asks for JSON and contains the schema."""

    description = describe_schema_for_deepseek(payload_dict["structured_schema"])

    # DeepSeek's json_object mode has no schema enforcement -- their own docs
    # require the word "json" and the desired structure to appear in the
    # prompt text itself.
    assert "json" in description.lower()
    assert json.dumps(payload_dict["structured_schema"], indent=2) in description


def test_build_deepseek_request():
    """Builds the baseline request with the model, token limit, reasoning effort, messages and JSON mode."""

    llm_model = "deepseek-flash"
    output_dict = build_deepseek_request(llm_model, payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 5
    assert output_dict["model"] == "deepseek-flash"
    assert output_dict["max_tokens"] == 60000
    assert output_dict["reasoning_effort"] == "high"
    assert output_dict["messages"][0]["role"] == "user"
    assert output_dict["messages"][0]["content"][0]["type"] == "text"
    assert payload_dict["user_prompt"] in output_dict["messages"][0]["content"][0]["text"]
    assert describe_schema_for_deepseek(payload_dict["structured_schema"]) in output_dict["messages"][0]["content"][0]["text"]
    assert output_dict["messages"][0]["content"][1]["type"] == "image_url"
    assert output_dict["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{payload_dict['image_base64']}"
    assert output_dict["response_format"] == {"type": "json_object"}


def test_build_deepseek_request_formative():
    """Builds the formative request with the system prompt, the student work and JSON mode."""

    llm_model = "deepseek-flash"
    output_dict = build_deepseek_request_formative(llm_model, formative_payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 5
    assert output_dict["max_tokens"] == 60000
    assert output_dict["reasoning_effort"] == "high"
    assert output_dict["messages"][0]["role"] == "system"
    assert formative_payload_dict["system_prompt"] in output_dict["messages"][0]["content"]
    assert describe_schema_for_deepseek(formative_payload_dict["structured_schema"]) in output_dict["messages"][0]["content"]
    assert output_dict["messages"][1]["role"] == "user"
    assert output_dict["messages"][1]["content"][0]["type"] == "image_url"
    assert output_dict["messages"][1]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{formative_payload_dict['base_image_base64']}"
    assert output_dict["messages"][1]["content"][1]["type"] == "text"
    assert output_dict["messages"][1]["content"][1]["text"] == formative_payload_dict["student_work"][0]["content"]
    assert output_dict["messages"][1]["content"][2]["type"] == "image_url"
    assert output_dict["messages"][1]["content"][2]["image_url"]["url"] == f"data:image/png;base64,{formative_payload_dict['student_work'][1]['content_base64']}"
    assert output_dict["messages"][1]["content"][3]["type"] == "text"
    assert output_dict["messages"][1]["content"][3]["text"] == formative_payload_dict["student_work"][2]["content"]
    assert output_dict["response_format"] == {"type": "json_object"}


async def test_run_deepseek(tmp_path):
    """Returns the parsed JSON answer with token usage from a baseline call."""

    json_file = tmp_path/"deepseek_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"text1": "value1", "text2": "value2"}))
    fake_usage = SimpleNamespace(completion_tokens = 123)
    finish_reason = "stop"
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = finish_reason)
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "deepseek-flash"

    with patch("llm_api.run_deepseek_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_deepseek(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 123, "status": "stop"}
    }


async def test_run_deepseek_raises_on_empty_content(tmp_path):
    """Raises RuntimeError when the response content is empty."""

    json_file = tmp_path/"deepseek_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "")
    fake_usage = SimpleNamespace(completion_tokens = 20000)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "length")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "deepseek-flash"

    with patch("llm_api.run_deepseek_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="length"):
            await run_deepseek(llm_model, json_file)


async def test_run_deepseek_raises_on_malformed_json_content(tmp_path):
    """Raises RuntimeError when the response content is not valid JSON."""

    json_file = tmp_path/"deepseek_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "```json\n{not valid json\n```")
    fake_usage = SimpleNamespace(completion_tokens = 500)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "deepseek-flash"

    with patch("llm_api.run_deepseek_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="Could not parse"):
            await run_deepseek(llm_model, json_file)


async def test_run_deepseek_formative(tmp_path):
    """Returns the parsed JSON answer with token usage from a formative call."""

    json_file = tmp_path/"deepseek_formative_request.json"
    json_file.write_text(json.dumps(formative_payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"is_work_correct": True}))
    fake_usage = SimpleNamespace(completion_tokens = 200)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "deepseek-flash"

    with patch("llm_api.run_deepseek_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_deepseek_formative(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "is_work_correct": True,
        "_usage": {"tokens": 200, "status": "stop"}
    }
