from llm_api.run_togetherai_client import build_togetherai_request, run_togetherai, build_togetherai_request_formative, run_togetherai_formative, clean_schema_for_togetherai, get_reasoning_effort, get_thinking_field
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

def test_build_togetherai_request():

    llm_model = "Kimi-K3"
    output_dict = build_togetherai_request(llm_model, payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 6
    assert output_dict["model"] == "Kimi-K3"
    assert output_dict["max_tokens"] == 20000
    assert output_dict["reasoning"] == {"enabled": True}
    assert output_dict["reasoning_effort"] == "medium"
    assert output_dict["messages"][0]["role"] == "user"
    assert output_dict["messages"][0]["content"][0]["type"] == "text"
    assert output_dict["messages"][0]["content"][0]["text"] == payload_dict["user_prompt"]
    assert output_dict["messages"][0]["content"][1]["type"] == "image_url"
    assert output_dict["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{payload_dict['image_base64']}"
    assert output_dict["response_format"]["type"] == "json_schema"
    assert output_dict["response_format"]["json_schema"]["name"] == "structured_response"
    assert output_dict["response_format"]["json_schema"]["schema"] == clean_schema_for_togetherai(payload_dict["structured_schema"])


def test_build_togetherai_request_formative():

    llm_model = "Kimi-K3"
    output_dict = build_togetherai_request_formative(llm_model, formative_payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 6
    assert output_dict["max_tokens"] == 20000
    assert output_dict["reasoning"] == {"enabled": True}
    assert output_dict["reasoning_effort"] == "medium"
    assert output_dict["messages"][0]["role"] == "system"
    assert output_dict["messages"][0]["content"] == formative_payload_dict["system_prompt"]
    assert output_dict["messages"][1]["role"] == "user"
    assert output_dict["messages"][1]["content"][0]["type"] == "image_url"
    assert output_dict["messages"][1]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{formative_payload_dict['base_image_base64']}"
    assert output_dict["messages"][1]["content"][1]["type"] == "text"
    assert output_dict["messages"][1]["content"][1]["text"] == formative_payload_dict["student_work"][0]["content"]
    assert output_dict["messages"][1]["content"][2]["type"] == "image_url"
    assert output_dict["messages"][1]["content"][2]["image_url"]["url"] == f"data:image/png;base64,{formative_payload_dict['student_work'][1]['content_base64']}"
    assert output_dict["messages"][1]["content"][3]["type"] == "text"
    assert output_dict["messages"][1]["content"][3]["text"] == formative_payload_dict["student_work"][2]["content"]
    assert output_dict["response_format"]["type"] == "json_schema"
    assert output_dict["response_format"]["json_schema"]["name"] == "structured_response"
    assert output_dict["response_format"]["json_schema"]["schema"] == clean_schema_for_togetherai(formative_payload_dict["structured_schema"])


async def test_run_togetherai(tmp_path):

    json_file = tmp_path/"togetherai_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"text1": "value1", "text2": "value2"}))
    fake_usage = SimpleNamespace(completion_tokens = 123)
    finish_reason = "stop"
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = finish_reason)
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "Kimi-K3"

    with patch("llm_api.run_togetherai_client.AsyncTogether") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_togetherai(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 123, "status": "stop"}
    }


async def test_run_togetherai_raises_on_empty_content(tmp_path):

    json_file = tmp_path/"togetherai_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "")
    fake_usage = SimpleNamespace(completion_tokens = 20000)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "length")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "Kimi-K3"

    with patch("llm_api.run_togetherai_client.AsyncTogether") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="length"):
            await run_togetherai(llm_model, json_file)


async def test_run_togetherai_raises_on_whitespace_only_content(tmp_path):

    json_file = tmp_path/"togetherai_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "  \n  ")
    fake_usage = SimpleNamespace(completion_tokens = 20000)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "length")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "Kimi-K3"

    with patch("llm_api.run_togetherai_client.AsyncTogether") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="length"):
            await run_togetherai(llm_model, json_file)


async def test_run_togetherai_raises_on_malformed_json_content(tmp_path):

    json_file = tmp_path/"togetherai_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_message = SimpleNamespace(content = "```json\n{\"text1\": \"value1\"}\n```")
    fake_usage = SimpleNamespace(completion_tokens = 20000)
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = "stop")
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "Kimi-K3"

    with patch("llm_api.run_togetherai_client.AsyncTogether") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        with pytest.raises(RuntimeError, match="Raw content"):
            await run_togetherai(llm_model, json_file)


async def test_run_togetherai_formative(tmp_path):

    json_file = tmp_path/"togetherai_request.json"
    json_file.write_text(json.dumps(formative_payload_dict))

    fake_message = SimpleNamespace(content = json.dumps({"text1": "value1", "text2": "value2"}))
    fake_usage = SimpleNamespace(completion_tokens = 123)
    finish_reason = "stop"
    fake_choice = SimpleNamespace(message = fake_message, finish_reason = finish_reason)
    fake_response = SimpleNamespace(choices = [fake_choice], usage = fake_usage)

    llm_model = "Kimi-K3"

    with patch("llm_api.run_togetherai_client.AsyncTogether") as mock_client:

        mock_client.return_value.chat.completions.create = AsyncMock(return_value = fake_response)

        response = await run_togetherai_formative(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 123, "status": "stop"}
    }


def test_clean_schema_for_togetherai_required_and_nullable():

    schema = {
        "type": "object",
        "properties": {
            "is_work_correct": {"type": "boolean"},
            "error_step_number": {
                "type": "integer",
                "nullable": True,
                "description": "Must be null if is_work_correct is true."
            },
            "controlling_members": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True
            }
        },
        "required": ["is_work_correct", "error_step_number"]
    }

    cleaned = clean_schema_for_togetherai(schema)

    assert cleaned["properties"]["error_step_number"]["type"] == ["integer", "null"]
    assert "nullable" not in cleaned["properties"]["error_step_number"]
    assert cleaned["required"] == ["is_work_correct", "error_step_number"]
    assert "controlling_members" not in cleaned["required"]
    assert "uniqueItems" not in cleaned["properties"]["controlling_members"]
    assert cleaned["additionalProperties"] is False


def test_get_reasoning_effort_uses_override_for_deepseek():

    assert get_reasoning_effort("deepseek-ai/DeepSeek-V4.1-Flash") == 50


def test_get_reasoning_effort_uses_override_for_glm():

    assert get_reasoning_effort("zai-org/GLM-5.3-Flash") == "high"


def test_get_reasoning_effort_defaults_to_medium_for_other_models():

    assert get_reasoning_effort("MiniMaxAI/MiniMax-M3") == "medium"


def test_get_thinking_field_uses_override_for_minimax():

    assert get_thinking_field("MiniMaxAI/MiniMax-M3") == {"type": "enabled"}


def test_get_thinking_field_defaults_to_none_for_other_models():

    assert get_thinking_field("deepseek-ai/DeepSeek-V4.1-Flash") is None


def test_build_togetherai_request_adds_thinking_field_for_minimax():

    output_dict = build_togetherai_request("MiniMaxAI/MiniMax-M3", payload_dict)

    assert output_dict["thinking"] == {"type": "enabled"}


def test_build_togetherai_request_omits_thinking_field_by_default():

    output_dict = build_togetherai_request("Kimi-K3", payload_dict)

    assert "thinking" not in output_dict