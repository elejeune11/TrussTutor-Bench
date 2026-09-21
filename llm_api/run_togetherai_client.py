import base64
from PIL import Image
import io
from pathlib import Path
import json
from together import AsyncTogether

TOGETHERAI_IMAGE_RAW_SIZE_LIMIT = 20_000_000
TOGETHERAI_IMAGE_MAX_DIMENSION = 2048
TOGETHERAI_MAX_TOKENS = 20000
TOGETHERAI_REASONING_ENABLED = True
TOGETHERAI_REASONING_EFFORT_DEFAULT = "medium"
TOGETHERAI_REASONING_EFFORT_OVERRIDES = {
    "deepseek-ai/DeepSeek-V4.1-Flash": 50,
    "zai-org/GLM-5.3-Flash": "high",
}


def get_reasoning_effort(llm_model: str):
    return TOGETHERAI_REASONING_EFFORT_OVERRIDES.get(llm_model, TOGETHERAI_REASONING_EFFORT_DEFAULT)


TOGETHERAI_THINKING_OVERRIDES = {
    "MiniMaxAI/MiniMax-M3": {"type": "enabled"},
}


def get_thinking_field(llm_model: str):
    return TOGETHERAI_THINKING_OVERRIDES.get(llm_model)


def prepare_image_for_togetherai(image_base64: str) -> str:
    
    """
    Returns a data-URI string ("data:image/<format>;base64,<data>") safe for
    together.ai's Chat Completions API, which takes the image as a single string.
    """
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    media_type = f"image/{img.format.lower()}"

    exceeds_raw_size = len(image_bytes) > TOGETHERAI_IMAGE_RAW_SIZE_LIMIT
    exceeds_max_dimension = max(img.size) > TOGETHERAI_IMAGE_MAX_DIMENSION

    if not exceeds_raw_size and not exceeds_max_dimension:
        return f"data:{media_type};base64,{image_base64}"

    img.thumbnail((TOGETHERAI_IMAGE_MAX_DIMENSION, TOGETHERAI_IMAGE_MAX_DIMENSION), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{resized_base64}"


def _union_type_with_null(prop_schema: dict) -> None:
    """Mutates prop_schema["type"] in place to also allow "null"."""
    original_type = prop_schema["type"]
    if isinstance(original_type, list):
        if "null" not in original_type:
            prop_schema["type"] = original_type + ["null"]
    else:
        prop_schema["type"] = [original_type, "null"]


def clean_schema_for_togetherai(schema: dict) -> dict:
    """
    Translates this project's "nullable" flag into a
    real type union, strips "uniqueItems" (not supported by every
    model's structured-output grammar compiler), and forces
    additionalProperties: false.
    """

    if not isinstance(schema, dict):
        return schema

    is_nullable = schema.get("nullable") is True

    cleaned = {}

    for key, value in schema.items():
        if key in ("nullable", "uniqueItems"):
            continue
        elif isinstance(value,dict):
            cleaned[key] = clean_schema_for_togetherai(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_for_togetherai(item) for item in value]
        else:
            cleaned[key] = value
        
    if "properties" in cleaned:
            cleaned["additionalProperties"] = False
        
    if is_nullable and "type" in cleaned:
        _union_type_with_null(cleaned)
        
    return cleaned

def build_togetherai_request(llm_model: str, payload: dict) -> dict:

    """
    Build a request dictionary for the together.ai API based on the
    provided payload.

    Args:
        llm_model (str): The name of the LLM model to use.
        payload (dict): A dictionary containing the benchmark ID,
        user prompt, image in base64 format, and structured schema.

    Returns:
        dict: A dictionary formatted for the together.ai Chat Completions API request.
    """

    image_data_uri = prepare_image_for_togetherai(payload["image_base64"])

    request_dict = {
        "model": llm_model,
        "max_tokens": TOGETHERAI_MAX_TOKENS,
        "reasoning": {"enabled": TOGETHERAI_REASONING_ENABLED},
        "reasoning_effort": get_reasoning_effort(llm_model),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": payload["user_prompt"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url" : image_data_uri}
                    }
                ]
            }
        ],
        "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": clean_schema_for_togetherai(payload["structured_schema"])
        }
            }
    }

    thinking_field = get_thinking_field(llm_model)
    if thinking_field is not None:
        request_dict["thinking"] = thinking_field

    return request_dict


async def run_togetherai(llm_model: str, json_file_path: Path) -> dict:

    """
    Run the together.ai API using the provided JSON file path.

    Args:
        llm_model (str): The name of the LLM model to use.
        json_file_path (Path): Path to the JSON file containing the request payload.

    Returns:
        dict: The response from together.ai.
    """

    # Load the request payload from the JSON file
    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    # Build the request payload for the together.ai API
    togetherai_request = build_togetherai_request(llm_model, payload)

    # Initialize the together.ai client
    async_client = AsyncTogether(timeout=300)

    # Send the request and get the response
    thinking_field = togetherai_request.pop("thinking", None)
    extra_body = {"thinking": thinking_field} if thinking_field is not None else None
    api_response = await async_client.chat.completions.create(**togetherai_request, extra_body=extra_body)

    # Extract usage and status from the response
    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

    # Extract the LLM text response
    message_item = api_response.choices[0].message
    text_block = message_item.content
    if not text_block or not text_block.strip():
        raise RuntimeError(
            f"Empty content in together.ai response for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage})"
        )
    try:
        model_response = json.loads(text_block)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse together.ai response content as JSON for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage}): {e}\nRaw content: {text_block!r}"
        ) from e

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "status": status
    }

    return model_response


def build_togetherai_request_formative(llm_model: str, formative_payload: dict) -> dict:
    """
    Build a request dictionary for the together.ai API based on the
    provided formative payload.

    Args:
        llm_model (str): The name of the LLM model to use.
        formative_payload (dict): A dictionary containing the system prompt.

    Returns:
        dict: A dictionary formatted for the together.ai Chat Completions API request.
    """

    base_image_data_uri = prepare_image_for_togetherai(formative_payload["base_image_base64"])

    content = [
        {
            "type": "image_url",
            "image_url": {"url": base_image_data_uri}
        }
    ]

    for block in formative_payload["student_work"]:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["content"]})
        elif block["type"] == "image":
            student_image_data_uri = prepare_image_for_togetherai(block["content_base64"])
            content.append({
                "type": "image_url",
                "image_url": {"url": student_image_data_uri}
            })

    request_dict = {
        "model": llm_model,
        "max_tokens": TOGETHERAI_MAX_TOKENS,
        "reasoning": {"enabled": TOGETHERAI_REASONING_ENABLED},
        "reasoning_effort": get_reasoning_effort(llm_model),
        "messages": [
            {
                "role": "system",
                "content": formative_payload["system_prompt"],
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": clean_schema_for_togetherai(formative_payload["structured_schema"])
        }
            }
    }

    thinking_field = get_thinking_field(llm_model)
    if thinking_field is not None:
        request_dict["thinking"] = thinking_field

    return request_dict


async def run_togetherai_formative(llm_model: str, json_file_path: Path) -> dict:

    """
    Run the together.ai API for a formative assessment using the provided JSON file path.

    Args:
        llm_model (str): The name of the LLM model to use.
        json_file_path (Path): Path to the JSON file containing the formative request payload.

    Returns:
        dict: The response from the together.ai API.
    """

    # Load the request payload from the JSON file
    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    # Build the request payload for the together.ai API
    togetherai_request = build_togetherai_request_formative(llm_model, formative_payload)

    # Initialize the together.ai client
    # Default SDK timeout is 60s, too short for slower reasoning models under load.
    async_client = AsyncTogether(timeout=300)

    # Send the request and get the response
    thinking_field = togetherai_request.pop("thinking", None)
    extra_body = {"thinking": thinking_field} if thinking_field is not None else None
    api_response = await async_client.chat.completions.create(**togetherai_request, extra_body=extra_body)

    # Extract usage and status from the response
    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

    # Extract the LLM text response
    message_item = api_response.choices[0].message
    text_block = message_item.content
    if not text_block or not text_block.strip():
        raise RuntimeError(
            f"Empty content in together.ai response for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage})"
        )
    try:
        model_response = json.loads(text_block)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse together.ai response content as JSON for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage}): {e}\nRaw content: {text_block!r}"
        ) from e

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "status": status
    }

    return model_response