from llm_api.run_openai_client import build_openai_request, build_openai_request_formative
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from llm_api.run_openai_client import run_openai, run_openai_formative
from llm_api.run_openai_client import clean_schema_for_openai

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
            "type": ["array", "null"],
            "items": {"type": "string"}
        }
    },
    "required": [
        "max_compression_magnitude",
        "units",
        "controlling_members"
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


def test_build_openai_request():

    llm_model = "gpt-5.1"
    output_dict = build_openai_request(llm_model, payload_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 4
    assert output_dict["model"] == "gpt-5.1"
    assert output_dict["max_output_tokens"] == 20000
    assert output_dict["input"][0]["role"] == "user"
    assert output_dict["input"][0]["content"][0]["type"] == "input_text"
    assert output_dict["input"][0]["content"][0]["text"] == payload_dict["user_prompt"]
    assert output_dict["input"][0]["content"][1]["type"] == "input_image"
    assert output_dict["input"][0]["content"][1]["image_url"] == f"data:image/png;base64,{payload_dict['image_base64']}"
    assert output_dict["text"] == {
        "format": {
            "type": "json_schema",
            "name": "structured_response",
            "schema": cleaned_schema,
            "strict": True
        }
    }


async def test_run_openai(tmp_path):

    json_file = tmp_path / "openai_request.json"
    json_file.write_text(json.dumps(payload_dict))

    fake_reasoning_item = SimpleNamespace(type="reasoning")
    fake_output_text_block = SimpleNamespace(type="output_text", text=json.dumps({"text1": "value1", "text2": "value2"}))
    fake_message_item = SimpleNamespace(type="message", content=[fake_output_text_block])
    fake_usage = SimpleNamespace(output_tokens=42)
    fake_response = SimpleNamespace(output=[fake_reasoning_item, fake_message_item], usage=fake_usage, status="completed")

    llm_model = "gpt-5.1"

    with patch("llm_api.run_openai_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.responses.create = AsyncMock(return_value=fake_response)

        response = await run_openai(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 42, "status": "completed"}
    }

def test_build_openai_request_formative():

    llm_model = "gpt-5.1"
    output_dict = build_openai_request_formative(llm_model, formative_payload_dict)

    assert isinstance(output_dict, dict)
    assert output_dict["instructions"] == formative_payload_dict["system_prompt"]
    assert len(output_dict) == 5
    assert output_dict["model"] == "gpt-5.1"
    assert output_dict["max_output_tokens"] == 20000
    assert output_dict["input"][0]["role"] == "user"
    assert output_dict["input"][0]["content"][0]["type"] == "input_image"
    assert output_dict["input"][0]["content"][0]["image_url"] == f"data:image/png;base64,{formative_payload_dict['base_image_base64']}"
    assert output_dict["input"][0]["content"][1]["type"] == "input_text"
    assert output_dict["input"][0]["content"][1]["text"] == formative_payload_dict["student_work"][0]["content"]
    assert output_dict["text"] == {
        "format": {
            "type": "json_schema",
            "name": "structured_response",
            "schema": cleaned_schema,
            "strict": True
        }
    }
    assert output_dict["input"][0]["content"][2]["type"] == "input_image"
    assert output_dict["input"][0]["content"][2]["image_url"] == f"data:image/png;base64,{formative_payload_dict['student_work'][1]['content_base64']}"
    assert output_dict["input"][0]["content"][3]["type"] == "input_text"
    assert output_dict["input"][0]["content"][3]["text"] == formative_payload_dict["student_work"][2]["content"]


async def test_run_openai_formative(tmp_path):

    json_file = tmp_path / "openai_formative_request.json"
    json_file.write_text(json.dumps(formative_payload_dict))

    fake_reasoning_item = SimpleNamespace(type="reasoning")
    fake_output_text_block = SimpleNamespace(type="output_text", text=json.dumps({"text1": "value1", "text2": "value2"}))
    fake_message_item = SimpleNamespace(type="message", content=[fake_output_text_block])
    fake_usage = SimpleNamespace(output_tokens=42)
    fake_response = SimpleNamespace(output=[fake_reasoning_item, fake_message_item], usage=fake_usage, status="completed")

    llm_model = "gpt-5.1"

    with patch("llm_api.run_openai_client.AsyncOpenAI") as mock_client:

        mock_client.return_value.responses.create = AsyncMock(return_value=fake_response)

        response = await run_openai_formative(llm_model, json_file)

    assert isinstance(response, dict)
    assert response == {
        "text1": "value1",
        "text2": "value2",
        "_usage": {"tokens": 42, "status": "completed"}
    }


def test_clean_schema_for_openai_nested_object():

    schema = {
        "type": "object",
        "properties": {
            "member_forces": {
                "type": "object",
                "properties": {
                    "magnitude": {"type": "number"},
                    "state": {"type": "string"}
                },
                "required": ["magnitude", "state"]
            }
        },
        "required": ["member_forces"]
    }

    cleaned = clean_schema_for_openai(schema)

    assert cleaned["required"] == ["member_forces"]
    assert cleaned["additionalProperties"] is False
    nested = cleaned["properties"]["member_forces"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["magnitude", "state"]
    assert nested["properties"]["magnitude"]["type"] == "number"
    assert nested["properties"]["state"]["type"] == "string"


def test_clean_schema_for_openai_required_and_nullable():
    
    schema = {
        "type": "object",
        "properties": {
            "is_work_correct": {"type": "boolean"},
            "error_step_number": {
                "type": "integer",
                "nullable": True,
                "description": "Must be null if is_work_correct is true."
            }
        },
        "required": ["is_work_correct", "error_step_number"]
    }

    cleaned = clean_schema_for_openai(schema)

    assert cleaned["properties"]["error_step_number"]["type"] == ["integer", "null"]
    assert "nullable" not in cleaned["properties"]["error_step_number"]
    assert cleaned["required"] == ["is_work_correct", "error_step_number"]
