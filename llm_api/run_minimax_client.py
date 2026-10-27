import base64
from PIL import Image
import io
import os
from pathlib import Path
import json
from openai import AsyncOpenAI

MINIMAX_BASE_URL = "https://api.minimax.io/v1"

# MiniMax documents no image limit, so these values are used as safe defaults.
MINIMAX_IMAGE_RAW_SIZE_LIMIT = 20_000_000
MINIMAX_IMAGE_MAX_DIMENSION = 2048

# Reasoning tokens share this budget. MiniMax has no reasoning-effort setting, so the cap is set high.
MINIMAX_MAX_TOKENS = 100000

# Thinking is "adaptive" or "disabled" (not "enabled").
# reasoning_split returns the reasoning separately from the answer.
MINIMAX_THINKING_TYPE = "adaptive"
MINIMAX_REASONING_SPLIT = True


def prepare_image_for_minimax(image_base64: str) -> str:
    """
    Returns the image as a data URI, resized to JPEG if it exceeds the size or dimension limit.

    Args:
        image_base64 (str): Base64-encoded image.

    Returns:
        str: Data URI ("data:image/<format>;base64,<data>").
    """
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    media_type = f"image/{img.format.lower()}"

    exceeds_raw_size = len(image_bytes) > MINIMAX_IMAGE_RAW_SIZE_LIMIT
    exceeds_max_dimension = max(img.size) > MINIMAX_IMAGE_MAX_DIMENSION

    if not exceeds_raw_size and not exceeds_max_dimension:
        return f"data:{media_type};base64,{image_base64}"

    img.thumbnail((MINIMAX_IMAGE_MAX_DIMENSION, MINIMAX_IMAGE_MAX_DIMENSION), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{resized_base64}"


def _extract_last_json_object(text: str) -> str:
    """
    Returns the last valid {...} block in the text, because MiniMax can write
    prose before its JSON answer. Returns the text unchanged if none parses.

    Args:
        text (str): Model response content.

    Returns:
        str: The JSON block, or the original text.
    """
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[i:end + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        end = text.rfind("}", 0, end)
    return text


def describe_schema_for_minimax(schema: dict) -> str:
    """
    Describes the expected JSON structure as prompt text, because MiniMax has
    no structured-output mode.

    Args:
        schema (dict): JSON Schema of the expected output.

    Returns:
        str: Instruction text containing the schema.
    """
    return (
        "Respond with ONLY a single valid JSON object (no markdown code "
        "fences, no extra commentary, no <think> tags in your final answer) "
        f"that matches this JSON Schema exactly:\n{json.dumps(schema, indent=2)}"
    )


def build_minimax_request(llm_model: str, payload: dict) -> dict:
    """
    Builds the Chat Completions request for a baseline payload.

    Args:
        llm_model (str): Model ID.
        payload (dict): Payload with the prompt, base64 image and schema.

    Returns:
        dict: Request for MiniMax's Chat Completions API.
    """

    image_data_uri = prepare_image_for_minimax(payload["image_base64"])
    schema_instructions = describe_schema_for_minimax(payload["structured_schema"])

    request_dict = {
        "model": llm_model,
        "max_completion_tokens": MINIMAX_MAX_TOKENS,
        "thinking": {"type": MINIMAX_THINKING_TYPE},
        "reasoning_split": MINIMAX_REASONING_SPLIT,
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
        ]
    }

    return request_dict


async def run_minimax(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a baseline payload to MiniMax and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and finish status.
    """

    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    minimax_request = build_minimax_request(llm_model, payload)

    # MiniMax uses the OpenAI SDK with its own base URL; the key is read from MINIMAX_API_KEY.
    async_client = AsyncOpenAI(api_key=os.environ.get("MINIMAX_API_KEY"), base_url=MINIMAX_BASE_URL, timeout=1800)

    # thinking and reasoning_split are not SDK arguments, so they go through extra_body.
    thinking_field = minimax_request.pop("thinking")
    reasoning_split = minimax_request.pop("reasoning_split")
    extra_body = {"thinking": thinking_field, "reasoning_split": reasoning_split}

    api_response = await async_client.chat.completions.create(**minimax_request, extra_body=extra_body)

    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

    message_item = api_response.choices[0].message
    text_block = message_item.content
    if not text_block or not text_block.strip():
        raise RuntimeError(
            f"Empty content in MiniMax response for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage})"
        )
    try:
        model_response = json.loads(text_block)
    except json.JSONDecodeError:
        try:
            model_response = json.loads(_extract_last_json_object(text_block))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Could not parse MiniMax response content as JSON for model {llm_model} "
                f"(finish_reason={status!r}, tokens={tokens_usage}): {e}\nRaw content: {text_block!r}"
            ) from e

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "status": status
    }

    return model_response


def build_minimax_request_formative(llm_model: str, formative_payload: dict) -> dict:
    """
    Builds the Chat Completions request for a formative payload.

    Args:
        llm_model (str): Model ID.
        formative_payload (dict): Payload with the system prompt, base image,
            student work (text and image blocks) and schema.

    Returns:
        dict: Request for MiniMax's Chat Completions API.
    """

    base_image_data_uri = prepare_image_for_minimax(formative_payload["base_image_base64"])

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
            student_image_data_uri = prepare_image_for_minimax(block["content_base64"])
            content.append({
                "type": "image_url",
                "image_url": {"url": student_image_data_uri}
            })

    schema_instructions = describe_schema_for_minimax(formative_payload["structured_schema"])
    system_content = f"{formative_payload['system_prompt']}\n\n{schema_instructions}"

    request_dict = {
        "model": llm_model,
        "max_completion_tokens": MINIMAX_MAX_TOKENS,
        "thinking": {"type": MINIMAX_THINKING_TYPE},
        "reasoning_split": MINIMAX_REASONING_SPLIT,
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": content
            }
        ]
    }

    return request_dict


async def run_minimax_formative(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a formative payload to MiniMax and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the formative payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and finish status.
    """

    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    minimax_request = build_minimax_request_formative(llm_model, formative_payload)

    # MiniMax uses the OpenAI SDK with its own base URL; the key is read from MINIMAX_API_KEY.
    async_client = AsyncOpenAI(api_key=os.environ.get("MINIMAX_API_KEY"), base_url=MINIMAX_BASE_URL, timeout=1800)

    # thinking and reasoning_split are not SDK arguments, so they go through extra_body.
    thinking_field = minimax_request.pop("thinking")
    reasoning_split = minimax_request.pop("reasoning_split")
    extra_body = {"thinking": thinking_field, "reasoning_split": reasoning_split}

    api_response = await async_client.chat.completions.create(**minimax_request, extra_body=extra_body)

    tokens_usage = api_response.usage.completion_tokens
    status = api_response.choices[0].finish_reason

    message_item = api_response.choices[0].message
    text_block = message_item.content
    if not text_block or not text_block.strip():
        raise RuntimeError(
            f"Empty content in MiniMax response for model {llm_model} "
            f"(finish_reason={status!r}, tokens={tokens_usage})"
        )
    try:
        model_response = json.loads(text_block)
    except json.JSONDecodeError:
        try:
            model_response = json.loads(_extract_last_json_object(text_block))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Could not parse MiniMax response content as JSON for model {llm_model} "
                f"(finish_reason={status!r}, tokens={tokens_usage}): {e}\nRaw content: {text_block!r}"
            ) from e

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "status": status
    }

    return model_response
