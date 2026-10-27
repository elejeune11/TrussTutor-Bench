import base64
from PIL import Image
import io
import os
from pathlib import Path
import json
from openai import AsyncOpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek limits image size (32 MiB) but not pixel dimensions, so only size is checked.
DEEPSEEK_IMAGE_RAW_SIZE_LIMIT = 32 * 1024 * 1024

# Reasoning tokens share this budget, so a low cap can end a response before the answer.
DEEPSEEK_MAX_TOKENS = 60000

# DeepSeek accepts "low", "high" or "max".
DEEPSEEK_REASONING_EFFORT = "high"


def prepare_image_for_deepseek(image_base64: str) -> str:
    """
    Returns the image as a data URI, re-encoded as JPEG if it exceeds DeepSeek's size limit.

    Args:
        image_base64 (str): Base64-encoded image.

    Returns:
        str: Data URI ("data:image/<format>;base64,<data>").
    """
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    media_type = f"image/{img.format.lower()}"

    if len(image_bytes) <= DEEPSEEK_IMAGE_RAW_SIZE_LIMIT:
        return f"data:{media_type};base64,{image_base64}"

    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{resized_base64}"


def describe_schema_for_deepseek(schema: dict) -> str:
    """
    Describes the expected JSON structure as prompt text, because DeepSeek's
    JSON mode does not enforce a schema.

    Args:
        schema (dict): JSON Schema of the expected output.

    Returns:
        str: Instruction text containing the schema.
    """
    return (
        "Respond with ONLY a single valid JSON object (no markdown code "
        "fences, no extra commentary) that matches this JSON Schema "
        f"exactly:\n{json.dumps(schema, indent=2)}"
    )


def build_deepseek_request(llm_model: str, payload: dict) -> dict:
    """
    Builds the Chat Completions request for a baseline payload.

    Args:
        llm_model (str): Model ID.
        payload (dict): Payload with the prompt, base64 image and schema.

    Returns:
        dict: Request for DeepSeek's Chat Completions API.
    """

    image_data_uri = prepare_image_for_deepseek(payload["image_base64"])
    schema_instructions = describe_schema_for_deepseek(payload["structured_schema"])

    request_dict = {
        "model": llm_model,
        "max_tokens": DEEPSEEK_MAX_TOKENS,
        "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{payload['user_prompt']}\n\n{schema_instructions}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }

    return request_dict


async def run_deepseek(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a baseline payload to DeepSeek and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and finish status.
    """

    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    deepseek_request = build_deepseek_request(llm_model, payload)

    # DeepSeek uses the OpenAI SDK with its own base URL.
    async_client = AsyncOpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL, timeout=600)

    api_response = await async_client.chat.completions.create(**deepseek_request)

    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

    message_item = api_response.choices[0].message
    text_block = message_item.content
    if not text_block or not text_block.strip():
        raise RuntimeError(
            f"Empty content in DeepSeek response for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage})"
        )
    try:
        model_response = json.loads(text_block)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse DeepSeek response content as JSON for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage}): {e}\nRaw content: {text_block!r}"
        ) from e

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "status": status
    }

    return model_response


def build_deepseek_request_formative(llm_model: str, formative_payload: dict) -> dict:
    """
    Builds the Chat Completions request for a formative payload.

    Args:
        llm_model (str): Model ID.
        formative_payload (dict): Payload with the system prompt, base image,
            student work (text and image blocks) and schema.

    Returns:
        dict: Request for DeepSeek's Chat Completions API.
    """

    base_image_data_uri = prepare_image_for_deepseek(formative_payload["base_image_base64"])

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
            student_image_data_uri = prepare_image_for_deepseek(block["content_base64"])
            content.append({
                "type": "image_url",
                "image_url": {"url": student_image_data_uri}
            })

    schema_instructions = describe_schema_for_deepseek(formative_payload["structured_schema"])
    system_content = f"{formative_payload['system_prompt']}\n\n{schema_instructions}"

    request_dict = {
        "model": llm_model,
        "max_tokens": DEEPSEEK_MAX_TOKENS,
        "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "response_format": {"type": "json_object"}
    }

    return request_dict


async def run_deepseek_formative(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a formative payload to DeepSeek and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the formative payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and finish status.
    """

    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    deepseek_request = build_deepseek_request_formative(llm_model, formative_payload)

    # DeepSeek uses the OpenAI SDK with its own base URL.
    async_client = AsyncOpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL, timeout=600)

    api_response = await async_client.chat.completions.create(**deepseek_request)

    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

    message_item = api_response.choices[0].message
    text_block = message_item.content
    if not text_block or not text_block.strip():
        raise RuntimeError(
            f"Empty content in DeepSeek response for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage})"
        )
    try:
        model_response = json.loads(text_block)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse DeepSeek response content as JSON for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage}): {e}\nRaw content: {text_block!r}"
        ) from e

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "status": status
    }

    return model_response
