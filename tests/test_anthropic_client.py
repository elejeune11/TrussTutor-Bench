from llm_api.run_anthropic_client import build_claude_request, build_claude_request_formative
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from llm_api.run_anthropic_client import run_anthropic, run_anthropic_formative

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

cleaned_schema = {
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
            "items": {"type": "string"}
        }
    },
    "required": [
        "max_compression_magnitude",
        "units"
    ],
    "additionalProperties": False
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


def test_build_claude_request():

    llm_model = "claude-opus-5"
    output_dict = build_claude_request(llm_model,payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 4
    assert output_dict["model"] == "claude-opus-5"
    assert output_dict["max_tokens"] == 20000
    assert output_dict["messages"][0]["role"] == "user"
    assert output_dict["messages"][0]["content"][0]["type"] == "text"
    assert output_dict["messages"][0]["content"][0]["text"] == payload_dict["user_prompt"]
    assert output_dict["messages"][0]["content"][1]["type"] == "image"
    assert output_dict["messages"][0]["content"][1]["source"]["type"] == "base64"
    assert output_dict["messages"][0]["content"][1]["source"]["media_type"] == "image/png"
    assert output_dict["messages"][0]["content"][1]["source"]["data"] == payload_dict["image_base64"]
    assert output_dict["output_config"] == {"format": {"type": "json_schema", "schema": cleaned_schema}}


async def test_run_anthropic(tmp_path):

    json_file = tmp_path / "claude_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_thinking_block = SimpleNamespace(type = "thinking", thinking = "reasoning about the truss...")
    fake_block = SimpleNamespace(type = "text", text = json.dumps({"text1": "value1", "text2": "value2"}))
    fake_usage = SimpleNamespace(output_tokens = 42)
    fake_response = SimpleNamespace(content = [fake_thinking_block, fake_block], usage = fake_usage, stop_reason = "end_turn")

    llm_model = "claude-opus-5"

    with patch("llm_api.run_anthropic_client.AsyncAnthropic") as mock_client:

        mock_client.return_value.messages.create = AsyncMock(return_value=fake_response)

        response = await run_anthropic(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 42, "stop_reason": "end_turn"}
    }

def test_build_claude_request_formative():

    llm_model = "claude-opus-5"
    output_dict = build_claude_request_formative(llm_model,formative_payload_dict)

    assert isinstance(output_dict, dict)
    assert output_dict["system"] == formative_payload_dict["system_prompt"]
    assert len(output_dict) == 5
    assert output_dict["model"] == "claude-opus-5"
    assert output_dict["max_tokens"] == 20000
    assert output_dict["messages"][0]["role"] == "user"
    assert output_dict["messages"][0]["content"][1]["type"] == "text"
    assert output_dict["messages"][0]["content"][1]["text"] == formative_payload_dict["student_work"][0]["content"]
    assert output_dict["messages"][0]["content"][0]["type"] == "image"
    assert output_dict["messages"][0]["content"][0]["source"]["type"] == "base64"
    assert output_dict["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert output_dict["messages"][0]["content"][0]["source"]["data"] == formative_payload_dict["base_image_base64"]
    assert output_dict["output_config"] == {"format": {"type": "json_schema", "schema": cleaned_schema}}
    assert output_dict["messages"][0]["content"][2]["type"] == "image"
    assert output_dict["messages"][0]["content"][2]["source"]["type"] == "base64"
    assert output_dict["messages"][0]["content"][2]["source"]["media_type"] == "image/png"
    assert output_dict["messages"][0]["content"][2]["source"]["data"] == formative_payload_dict["student_work"][1]["content_base64"]
    assert output_dict["messages"][0]["content"][3]["type"] == "text"
    assert output_dict["messages"][0]["content"][3]["text"] == formative_payload_dict["student_work"][2]["content"]


async def test_run_anthropic_formative(tmp_path):

    json_file = tmp_path / "claude_formative_request.json"
    json_file.write_text(json.dumps(formative_payload_dict))

    fake_thinking_block = SimpleNamespace(type = "thinking", thinking = "reasoning about the student's work...")
    fake_block = SimpleNamespace(type = "text", text = json.dumps({"text1": "value1", "text2": "value2"}))
    fake_usage = SimpleNamespace(output_tokens = 42)
    fake_response = SimpleNamespace(content = [fake_thinking_block, fake_block], usage = fake_usage, stop_reason = "end_turn")

    llm_model = "claude-opus-5"

    with patch("llm_api.run_anthropic_client.AsyncAnthropic") as mock_client:

        mock_client.return_value.messages.create = AsyncMock(return_value=fake_response)

        response = await run_anthropic_formative(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 42, "stop_reason": "end_turn"}
    }