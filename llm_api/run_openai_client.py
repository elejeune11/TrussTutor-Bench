import json
import base64
import io
from openai import AsyncOpenAI
from pathlib import Path
from PIL import Image

OPENAI_IMAGE_RAW_SIZE_LIMIT = 20_000_000
OPENAI_IMAGE_MAX_DIMENSION = 2048

# Reasoning tokens share this budget, so a low cap can end a response before the answer.
OPENAI_MAX_OUTPUT_TOKENS = 20000

# Reasoning effort sent with every request.
OPENAI_REASONING_EFFORT = "high"


def prepare_image_for_openai(image_base64: str) -> str:
    """
    Returns the image as a data URI, resized to JPEG if it exceeds OpenAI's
    size or dimension limit.

    Args:
        image_base64 (str): Base64-encoded image.

    Returns:
        str: Data URI ("data:image/<format>;base64,<data>").
    """
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    media_type = f"image/{img.format.lower()}"

    exceeds_raw_size = len(image_bytes) > OPENAI_IMAGE_RAW_SIZE_LIMIT
    exceeds_max_dimension = max(img.size) > OPENAI_IMAGE_MAX_DIMENSION

    if not exceeds_raw_size and not exceeds_max_dimension:
        return f"data:{media_type};base64,{image_base64}"

    img.thumbnail((OPENAI_IMAGE_MAX_DIMENSION, OPENAI_IMAGE_MAX_DIMENSION), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{resized_base64}"

def dedupe_schema_for_openai(schema: dict) -> dict:
    """
    Replaces repeated identical sub-schemas with one shared $ref definition.

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

def _dedupe_repeated_subschemas(schema, defs: dict):
    """
    Recursive helper for dedupe_schema_for_openai.

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

def clean_schema_for_openai(schema: dict) -> dict:
    """
    Prepares a schema for OpenAI strict mode: "nullable" becomes a type union
    with "null", every property is required, additionalProperties is false.

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
        if key in ("uniqueItems", "nullable"):
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_key: clean_schema_for_openai(prop_val)
                for prop_key, prop_val in value.items()
            }
        elif isinstance(value, dict):
            cleaned[key] = clean_schema_for_openai(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_for_openai(item) for item in value]
        else:
            cleaned[key] = value

    if "properties" in cleaned:
        original_required = set(schema.get("required", []))
        for prop_key, prop_schema in cleaned["properties"].items():
            if prop_key in original_required or not isinstance(prop_schema, dict):
                continue
            if "type" in prop_schema:
                _union_type_with_null(prop_schema)

        cleaned["required"] = list(cleaned["properties"].keys())
        cleaned["additionalProperties"] = False

    if is_nullable and "type" in cleaned:
        _union_type_with_null(cleaned)

    return cleaned

def build_openai_request(llm_model: str, payload: dict) -> dict:
    """
    Builds the Responses API request for a baseline payload.

    Args:
        llm_model (str): Model ID.
        payload (dict): Payload with the prompt, base64 image and schema.

    Returns:
        dict: Request for the OpenAI Responses API.
    """

    image_data_uri = prepare_image_for_openai(payload["image_base64"])

    request_dict = {
        "model": llm_model,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": payload["user_prompt"]
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_uri
                    }
                ]
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "structured_response",
                "schema": dedupe_schema_for_openai(clean_schema_for_openai(payload["structured_schema"])),
                "strict": True
            }
        }
    }

    return request_dict

async def run_openai(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a baseline payload to OpenAI and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens, input tokens and status.
    """

    with open(json_file_path, 'r') as f:
        payload = json.load(f)

    openai_request = build_openai_request(llm_model, payload)

    client = AsyncOpenAI()

    response = await client.responses.create(**openai_request)

    tokens_usage = response.usage.output_tokens
    status = response.status

    message_item = next(item for item in response.output if item.type == "message")
    text_block = next(block for block in message_item.content if block.type == "output_text")
    model_response = json.loads(text_block.text)

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "input_tokens": response.usage.input_tokens,
        "status": status
    }

    return model_response

def build_openai_request_formative(llm_model: str, formative_payload: dict) -> dict:
    """
    Builds the Responses API request for a formative payload.

    Args:
        llm_model (str): Model ID.
        formative_payload (dict): Payload with the system prompt, base image,
            student work (text and image blocks) and schema.

    Returns:
        dict: Request for the OpenAI Responses API.
    """

    base_image_data_uri = prepare_image_for_openai(formative_payload["base_image_base64"])

    content = [
        {
            "type": "input_image",
            "image_url": base_image_data_uri
        }
    ]

    for block in formative_payload["student_work"]:
        if block["type"] == "text":
            content.append({"type": "input_text", "text": block["content"]})
        elif block["type"] == "image":
            student_image_data_uri = prepare_image_for_openai(block["content_base64"])
            content.append({
                "type": "input_image",
                "image_url": student_image_data_uri
            })

    request_dict = {
        "model": llm_model,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        "instructions": formative_payload["system_prompt"],
        "input": [
            {
                "role": "user",
                "content": content
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "structured_response",
                "schema": dedupe_schema_for_openai(clean_schema_for_openai(formative_payload["structured_schema"])),
                "strict": True
            }
        }
    }

    return request_dict

async def run_openai_formative(llm_model: str, json_file_path: Path) -> dict:
    """
    Sends a formative payload to OpenAI and returns the parsed JSON answer.

    Args:
        llm_model (str): Model ID.
        json_file_path (Path): Path to the formative payload JSON file.

    Returns:
        dict: The answer, with a "_usage" entry holding tokens, input tokens and status.
    """

    with open(json_file_path, 'r') as f:
        formative_payload = json.load(f)

    openai_request = build_openai_request_formative(llm_model, formative_payload)

    client = AsyncOpenAI()

    response = await client.responses.create(**openai_request)

    tokens_usage = response.usage.output_tokens
    status = response.status

    message_item = next(item for item in response.output if item.type == "message")
    text_block = next(block for block in message_item.content if block.type == "output_text")
    model_response = json.loads(text_block.text)

    model_response["_usage"] = {
        "tokens": tokens_usage,
        "input_tokens": response.usage.input_tokens,
        "status": status
    }

    return model_response
