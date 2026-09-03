import json
import base64
import io
from anthropic import AsyncAnthropic
from pathlib import Path
from PIL import Image

# Claude rejects images whose base64 encoding exceeds 10 MB. Base64 inflates
# raw bytes by ~4/3, so stay comfortably under that limit on the raw side.
CLAUDE_IMAGE_RAW_SIZE_LIMIT = 7_000_000
CLAUDE_IMAGE_MAX_DIMENSION = 1600

# Claude separately caps any individual image at 2000px per side once a
# request contains more than one image ("many-image requests") -- formative
# requests always do (base image + student work), so this applies there even
# when the baseline requests (single image) never trip it. It's independent
# of the raw byte-size limit above: an efficiently-compressed large PNG (an
# auto-generated line drawing) can be well under 7MB while still exceeding
# 2000px on a side.
CLAUDE_MANY_IMAGE_MAX_DIMENSION = 2000

def prepare_image_for_claude(image_base64: str) -> tuple[str, str]:
    """
    Returns (image_base64, media_type) safe for Claude's API.

    Some source images are actually JPEG data saved with a .png extension
    (real photographed textbook pages) -- Claude validates the declared
    media_type against the real bytes and rejects a mismatch, so the true
    format is detected here rather than assumed. Separately, some of those
    same photographed pages are 10MB+ as (lossless) PNG; those are
    downscaled and re-encoded as JPEG so they fit under the 10MB limit.
    Images that are already small enough are passed through unchanged, to
    avoid needlessly re-encoding the auto-generated line-drawing schematics
    that make up most of the dataset (JPEG compression can visibly blur
    thin lines/text that PNG preserves losslessly).
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
    Recursively removes JSON Schema fields Claude's structured-output mode
    rejects, and translates fields it doesn't understand into its own
    conventions.

    "nullable": true is a Gemini/OpenAPI convention, not standard JSON
    Schema, and Claude doesn't recognize it.
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
    Recursive helper for dedupe_schema_for_claude. Walks the schema; inside
    every "properties" block, groups sibling sub-schemas by exact (deep)
    equality, and for any group with 2+ identical members, hoists one copy
    into `defs` and replaces each occurrence with a $ref to it.
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
    Collapses sibling properties that share a byte-identical sub-schema
    (e.g. a "member_forces" object with the same {magnitude, state} shape
    repeated once per truss member) into a single shared definition
    referenced via $ref, instead of N duplicated inline copies. Claude's
    structured-output schema compiler supports internal $ref/$defs, and
    some tasks' per-member-keyed schemas are large enough (30+ duplicated
    sub-schemas) to trip its 'compiled grammar is too large' error when
    left fully inlined.
    """
    defs: dict = {}
    deduped = _dedupe_repeated_subschemas(schema, defs)
    if defs:
        deduped["$defs"] = defs
    return deduped

def build_claude_request(llm_model: str, payload: dict) -> dict:

    """
    Build a request dictionary for the Claude API based on the provided payload.

    Args:
        llm_model (str): The name of the LLM model to use.
        payload (dict): A dictionary containing the benchmark ID, 
        user prompt, image in base64 format, and structured schema.

    Returns:
        dict: A dictionary formatted for the Claude API request.
    """
    
    image_data, image_media_type = prepare_image_for_claude(payload["image_base64"])

    request_dict = {
        "model": llm_model,
        "max_tokens": 20000,
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
            "format": {
                "type": "json_schema",
                "schema": dedupe_schema_for_claude(clean_schema_for_claude(payload["structured_schema"]))
            }
        }
    }

    return request_dict

async def run_anthropic(llm_model: str, json_file_path: Path) -> dict:

    """
    Run the Anthropic API using the provided JSON file path.

    Args:
        llm_model (str): The name of the LLM model to use.
        json_file_path (Path): Path to the JSON file containing the request payload.

    Returns:
        dict: The response from the Anthropic API.
    """

    # Load the request payload from the JSON file
    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    # Build the request for Claude API
    claude_request = build_claude_request(llm_model, payload)

    # Initialize the Anthropic client
    client = AsyncAnthropic()

    # Send the request and get the response
    response = await client.messages.create(**claude_request)

    # Extract usage and stop reason from the response
    tokens_usage = response.usage.output_tokens
    stop_reason = response.stop_reason

    # Extract the text block. Extended thinking is adaptive, so the text
    # answer isn't guaranteed to be at a fixed index (a ThinkingBlock may
    # come first). Search for the block whose type is "text" instead.
    text_block = next(block for block in response.content if block.type == "text")
    model_response = json.loads(text_block.text)

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "stop_reason": stop_reason
    }

    return model_response

def build_claude_request_formative(llm_model: str, formative_payload: dict) -> dict:
    """
    Build a request dictionary for the Claude API based on the provided formative payload.

    Args:
        llm_model (str): The name of the LLM model to use.
        formative_payload (dict): A dictionary containing the system prompt.

    Returns:
        dict: A dictionary formatted for the Claude API request.
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
        "max_tokens": 20000,
        "system": formative_payload["system_prompt"],
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": dedupe_schema_for_claude(clean_schema_for_claude(formative_payload["structured_schema"]))
            }
        }
    }

    return request_dict

async def run_anthropic_formative(llm_model: str, json_file_path: Path) -> dict:

    """
    Run the Anthropic API for a formative assessment using the provided JSON file path.

    Args:
        llm_model (str): The name of the LLM model to use.
        json_file_path (Path): Path to the JSON file containing the formative request payload.

    Returns:
        dict: The response from the Anthropic API.
    """

    # Load the formative request payload from the JSON file
    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    # Build the request for Claude API
    claude_request = build_claude_request_formative(llm_model, formative_payload)

    # Initialize the Anthropic client
    client = AsyncAnthropic()

    # Send the request and get the response
    response = await client.messages.create(**claude_request)

    # Extract usage and stop reason from the response
    tokens_usage = response.usage.output_tokens
    stop_reason = response.stop_reason

    # Extract the text block. Extended thinking is adaptive, so the text
    # answer isn't guaranteed to be at a fixed index (a ThinkingBlock may
    # come first). Search for the block whose type is "text" instead.
    text_block = next(block for block in response.content if block.type == "text")
    model_response = json.loads(text_block.text)

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "stop_reason": stop_reason
    }

    return model_response