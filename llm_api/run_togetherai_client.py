import base64
from PIL import Image
import io
import math
from pathlib import Path
import json
from together import AsyncTogether

TOGETHERAI_IMAGE_RAW_SIZE_LIMIT = 20_000_000
TOGETHERAI_IMAGE_MAX_DIMENSION = 2048
TOGETHERAI_MAX_TOKENS = 64000

# Together rejects requests whose decoded images total too many pixels (413 error),
# so formative images are scaled down together to stay under this budget.
TOGETHERAI_PIXEL_BUDGET = 20_000_000

TOGETHERAI_REASONING_ENABLED = True
TOGETHERAI_REASONING_EFFORT = "high"


def prepare_image_for_togetherai(image_base64: str, scale: float = 1.0) -> str:
    """
    Returns the image as a data URI, resized to JPEG if it exceeds the size or
    dimension limit, or if scale is below 1.0.

    Args:
        image_base64 (str): Base64-encoded image.
        scale (float): Linear scale applied when the image is resized.

    Returns:
        str: Data URI ("data:image/<format>;base64,<data>").
    """
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    media_type = f"image/{img.format.lower()}"

    exceeds_raw_size = len(image_bytes) > TOGETHERAI_IMAGE_RAW_SIZE_LIMIT
    exceeds_max_dimension = max(img.size) > TOGETHERAI_IMAGE_MAX_DIMENSION

    if not exceeds_raw_size and not exceeds_max_dimension and scale >= 1.0:
        return f"data:{media_type};base64,{image_base64}"

    target_dimension = int(min(max(img.size), TOGETHERAI_IMAGE_MAX_DIMENSION) * scale)
    img.thumbnail((target_dimension, target_dimension), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{resized_base64}"


def get_formative_image_scale(images_base64: list) -> float:
    """
    Returns the scale (at most 1.0) that makes all images together fit the pixel budget.

    Args:
        images_base64 (list): Base64-encoded images of one payload.

    Returns:
        float: Linear scale to apply to every image.
    """
    total_pixels = 0
    for image_base64 in images_base64:
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        capped = min(1.0, TOGETHERAI_IMAGE_MAX_DIMENSION / max(img.size))
        total_pixels += int(img.size[0] * capped) * int(img.size[1] * capped)
    return min(1.0, math.sqrt(TOGETHERAI_PIXEL_BUDGET / total_pixels))


def _union_type_with_null(prop_schema: dict) -> None:
    """
    Allows "null" in prop_schema["type"], in place.

    Args:
        prop_schema (dict): Schema with a "type" key.
    """
    original_type = prop_schema["type"]
    if isinstance(original_type, list):
        if "null" not in original_type:
            prop_schema["type"] = original_type + ["null"]
    else:
        prop_schema["type"] = [original_type, "null"]


def clean_schema_for_togetherai(schema: dict) -> dict:
    """
    Prepares a schema for Together's structured output: "nullable" becomes a
    type union with "null", "uniqueItems" is removed, additionalProperties is false.

    Args:
        schema (dict): JSON Schema.

    Returns:
        dict: Cleaned copy of the schema.
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
    Builds the Chat Completions request for a baseline payload.

    Args:
        llm_model (str): Model ID.
        payload (dict): Payload with the prompt, base64 image and schema.

    Returns:
        dict: Request for Together's Chat Completions API.
    """

    image_data_uri = prepare_image_for_togetherai(payload["image_base64"])

    request_dict = {
        "model": llm_model,
        "max_tokens": TOGETHERAI_MAX_TOKENS,
        "reasoning": {"enabled": TOGETHERAI_REASONING_ENABLED},
        "reasoning_effort": TOGETHERAI_REASONING_EFFORT,
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
                        "image_url": {"url": image_data_uri}
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

    return request_dict


async def run_togetherai(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a baseline payload to Together and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and finish status.
    """

    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    togetherai_request = build_togetherai_request(llm_model, payload)

    # The SDK's default 60s timeout is too short for reasoning models under load.
    async_client = AsyncTogether(timeout=600)

    api_response = await async_client.chat.completions.create(**togetherai_request)

    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

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
    Builds the Chat Completions request for a formative payload.

    Args:
        llm_model (str): Model ID.
        formative_payload (dict): Payload with the system prompt, base image,
            student work (text and image blocks) and schema.

    Returns:
        dict: Request for Together's Chat Completions API.
    """

    all_images = [formative_payload["base_image_base64"]] + [
        block["content_base64"] for block in formative_payload["student_work"] if block["type"] == "image"
    ]
    scale = get_formative_image_scale(all_images)

    base_image_data_uri = prepare_image_for_togetherai(formative_payload["base_image_base64"], scale)

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
            student_image_data_uri = prepare_image_for_togetherai(block["content_base64"], scale)
            content.append({
                "type": "image_url",
                "image_url": {"url": student_image_data_uri}
            })

    request_dict = {
        "model": llm_model,
        "max_tokens": TOGETHERAI_MAX_TOKENS,
        "reasoning": {"enabled": TOGETHERAI_REASONING_ENABLED},
        "reasoning_effort": TOGETHERAI_REASONING_EFFORT,
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

    return request_dict


async def run_togetherai_formative(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a formative payload to Together and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the formative payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and finish status.
    """

    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    togetherai_request = build_togetherai_request_formative(llm_model, formative_payload)

    # The SDK's default 60s timeout is too short for reasoning models under load.
    async_client = AsyncTogether(timeout=600)

    api_response = await async_client.chat.completions.create(**togetherai_request)

    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

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
