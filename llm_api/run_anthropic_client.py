import json
import base64
import io
from anthropic import AsyncAnthropic
from pathlib import Path
from PIL import Image

# Claude rejects images over 10 MB in base64, so raw files are kept under 7 MB.
CLAUDE_IMAGE_RAW_SIZE_LIMIT = 7_000_000
CLAUDE_IMAGE_MAX_DIMENSION = 1600

# Claude caps each image at 2000 px per side when a request has several images (all formative requests).
CLAUDE_MANY_IMAGE_MAX_DIMENSION = 2000

# Reasoning tokens share max_tokens, so a response can end before the JSON answer.
# Effort controls how much the model reasons.
CLAUDE_THINKING_EFFORT = "high"
CLAUDE_MAX_TOKENS = 64000


def prepare_image_for_claude(image_base64: str) -> tuple[str, str]:
    """
    Returns the image and its media type for Claude. The real format is
    detected, because Claude rejects a declared type that does not match the
    bytes. Images over the size or dimension limit are downscaled to JPEG.

    Args:
        image_base64 (str): Base64-encoded image.

    Returns:
        tuple: (base64 image, media type).
    """
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    media_type = f"image/{img.format.lower()}"

    exceeds_raw_size = len(image_bytes) > CLAUDE_IMAGE_RAW_SIZE_LIMIT
    exceeds_many_image_dimension = max(img.size) > CLAUDE_MANY_IMAGE_MAX_DIMENSION

    if not exceeds_raw_size and not exceeds_many_image_dimension:
        return image_base64, media_type

    img.thumbnail((CLAUDE_IMAGE_MAX_DIMENSION, CLAUDE_IMAGE_MAX_DIMENSION), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/jpeg"

def clean_schema_for_claude(schema: dict) -> dict:
    """
    Removes the keyword Claude's structured output rejects ("uniqueItems") and
    turns the non-standard "nullable" flag into a type union with "null".

    Args:
        schema (dict): JSON Schema.

    Returns:
        dict: Cleaned copy of the schema.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        if key in ("uniqueItems", "nullable"):
            continue
        if isinstance(value, dict):
            cleaned[key] = clean_schema_for_claude(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_for_claude(item) for item in value]
        else:
            cleaned[key] = value

    if schema.get("nullable") is True and "type" in cleaned:
        original_type = cleaned["type"]
        if isinstance(original_type, list):
            if "null" not in original_type:
                cleaned["type"] = original_type + ["null"]
        else:
            cleaned["type"] = [original_type, "null"]

    return cleaned

def _dedupe_repeated_subschemas(schema, defs: dict):
    """
    Recursive helper for dedupe_schema_for_claude.

    Args:
        schema: Schema or sub-schema.
        defs (dict): Shared definitions, filled in place.

    Returns:
        The schema with repeated property schemas replaced by $ref.
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            recursed = {
                prop_key: _dedupe_repeated_subschemas(prop_val, defs)
                for prop_key, prop_val in value.items()
            }

            groups = {}
            for prop_key, prop_val in recursed.items():
                canon = json.dumps(prop_val, sort_keys=True)
                groups.setdefault(canon, []).append(prop_key)

            resolved = {}
            for prop_keys in groups.values():
                if len(prop_keys) >= 2:
                    def_name = f"shared_{len(defs)}"
                    defs[def_name] = recursed[prop_keys[0]]
                    for prop_key in prop_keys:
                        resolved[prop_key] = {"$ref": f"#/$defs/{def_name}"}
                else:
                    prop_key = prop_keys[0]
                    resolved[prop_key] = recursed[prop_key]

            # Preserve the original property order.
            result[key] = {prop_key: resolved[prop_key] for prop_key in value.keys()}
        elif isinstance(value, dict):
            result[key] = _dedupe_repeated_subschemas(value, defs)
        elif isinstance(value, list):
            result[key] = [_dedupe_repeated_subschemas(item, defs) for item in value]
        else:
            result[key] = value
    return result

def dedupe_schema_for_claude(schema: dict) -> dict:
    """
    Replaces repeated identical sub-schemas with one shared $ref definition,
    which keeps large per-member schemas under Claude's compiled-grammar size limit.

    Args:
        schema (dict): JSON Schema.

    Returns:
        dict: Schema with a "$defs" section holding the shared definitions.
    """
    defs: dict = {}
    deduped = _dedupe_repeated_subschemas(schema, defs)
    if defs:
        deduped["$defs"] = defs
    return deduped

def build_claude_request(llm_model: str, payload: dict) -> dict:
    """
    Builds the Messages API request for a baseline payload.

    Args:
        llm_model (str): Model ID.
        payload (dict): Payload with the prompt, base64 image and schema.

    Returns:
        dict: Request for the Claude Messages API.
    """

    image_data, image_media_type = prepare_image_for_claude(payload["image_base64"])

    request_dict = {
        "model": llm_model,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "thinking": {
            "type": "adaptive"
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": payload["user_prompt"]
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data
                        }
                    }
                ]
            }
        ],
        "output_config": {
            "effort": CLAUDE_THINKING_EFFORT,
            "format": {
                "type": "json_schema",
                "schema": dedupe_schema_for_claude(clean_schema_for_claude(payload["structured_schema"]))
            }
        }
    }

    return request_dict

async def run_anthropic(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a baseline payload to Claude and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and stop reason.
    """

    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    claude_request = build_claude_request(llm_model, payload)

    client = AsyncAnthropic()

    # Streaming is required: the SDK rejects non-streaming requests with a max_tokens this large.
    async with client.messages.stream(**claude_request) as stream:
        response = await stream.get_final_message()

    tokens_usage = response.usage.output_tokens
    stop_reason = response.stop_reason

    # The text block is not always first, because a thinking block may come before it.
    text_block = next((block for block in response.content if block.type == "text"), None)
    if text_block is None:
        block_types = [block.type for block in response.content]
        raise RuntimeError(
            f"No text block in Claude response (stop_reason={stop_reason!r}, "
            f"block_types={block_types!r})"
        )
    model_response = json.loads(text_block.text)

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "stop_reason": stop_reason
    }

    return model_response

def build_claude_request_formative(llm_model: str, formative_payload: dict) -> dict:
    """
    Builds the Messages API request for a formative payload.

    Args:
        llm_model (str): Model ID.
        formative_payload (dict): Payload with the system prompt, base image,
            student work (text and image blocks) and schema.

    Returns:
        dict: Request for the Claude Messages API.
    """

    image_data, image_media_type = prepare_image_for_claude(formative_payload["base_image_base64"])

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_data
            }
        }
    ]

    for block in formative_payload["student_work"]:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["content"]})
        elif block["type"] == "image":
            student_image_data, student_image_media_type = prepare_image_for_claude(block["content_base64"])
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": student_image_media_type,
                    "data": student_image_data
                }
            })

    request_dict = {
        "model": llm_model,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "thinking": {
            "type": "adaptive"
        },
        "system": formative_payload["system_prompt"],
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "output_config": {
            "effort": CLAUDE_THINKING_EFFORT,
            "format": {
                "type": "json_schema",
                "schema": dedupe_schema_for_claude(clean_schema_for_claude(formative_payload["structured_schema"]))
            }
        }
    }

    return request_dict

async def run_anthropic_formative(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a formative payload to Claude and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the formative payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens and stop reason.
    """

    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    claude_request = build_claude_request_formative(llm_model, formative_payload)

    client = AsyncAnthropic()

    # Streaming is required: the SDK rejects non-streaming requests with a max_tokens this large.
    async with client.messages.stream(**claude_request) as stream:
        response = await stream.get_final_message()

    tokens_usage = response.usage.output_tokens
    stop_reason = response.stop_reason

    # The text block is not always first, because a thinking block may come before it.
    text_block = next((block for block in response.content if block.type == "text"), None)
    if text_block is None:
        block_types = [block.type for block in response.content]
        raise RuntimeError(
            f"No text block in Claude response (stop_reason={stop_reason!r}, "
            f"block_types={block_types!r})"
        )
    model_response = json.loads(text_block.text)

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "stop_reason": stop_reason
    }

    return model_response
