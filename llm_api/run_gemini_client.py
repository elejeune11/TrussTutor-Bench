import json
import base64
import os
import time
import copy
import google.generativeai as genai
from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable, ResourceExhausted


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Thinking models can exceed the SDK's default deadline under load,
# so use a long timeout and retry only on transient errors.
_REQUEST_TIMEOUT_S = 900
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = (10, 30, 90)
_RETRYABLE_EXCEPTIONS = (DeadlineExceeded, ServiceUnavailable, ResourceExhausted)


def _generate_with_retry(model, contents, config):
    """
    Calls generate_content, retrying on transient errors with increasing waits.

    Args:
        model (genai.GenerativeModel): Model to call.
        contents (list): Prompt text and image blobs.
        config (genai.GenerationConfig): Generation settings.

    Returns:
        The Gemini response.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return model.generate_content(
                contents=contents,
                generation_config=config,
                request_options={"timeout": _REQUEST_TIMEOUT_S},
            )
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt == _MAX_RETRIES:
                raise
            wait_s = _RETRY_BACKOFF_S[attempt]
            print(f"  [RETRY] {type(e).__name__} (attempt {attempt + 1}/{_MAX_RETRIES}), "
                  f"retrying in {wait_s}s...")
            time.sleep(wait_s)


def _usage_from_response(response) -> dict:
    """
    Reads token usage from a Gemini response. Output tokens (answer plus
    thinking) are total minus prompt, because the SDK has no thinking count.

    Args:
        response: Gemini response.

    Returns:
        dict: "tokens", "answer_tokens" and "input_tokens".
    """
    usage = response.usage_metadata
    return {
        "tokens": usage.total_token_count - usage.prompt_token_count,
        "answer_tokens": usage.candidates_token_count,
        "input_tokens": usage.prompt_token_count,
    }


def clean_schema_for_gemini(schema: dict) -> dict:
    """
    Removes the JSON Schema keywords the Gemini SDK rejects, at every level.

    Args:
        schema (dict): JSON Schema.

    Returns:
        dict: Cleaned copy of the schema.
    """
    if not isinstance(schema, dict):
        return schema

    # Keywords the Gemini schema converter rejects.
    unsupported_keys = {"additionalProperties", "uniqueItems"}

    cleaned = {}
    for key, value in schema.items():
        if key in unsupported_keys:
            continue
        if isinstance(value, dict):
            cleaned[key] = clean_schema_for_gemini(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_for_gemini(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value

    return cleaned


def inject_scratchpad_into_schema(original_schema: dict) -> dict:
    """
    Adds a required chain_of_thought field so the model reasons before answering.

    Args:
        original_schema (dict): JSON Schema of the expected answer.

    Returns:
        dict: Copy with the scratchpad added and unsupported keywords removed.
    """
    new_schema = copy.deepcopy(original_schema)

    scratchpad_prop = {
        "type": "string",
        "description": "Your step-by-step scratchpad. Formulate equilibrium equations, identify zero-force members, and calculate internal forces here BEFORE providing the final answers."
    }

    new_properties = {"chain_of_thought": scratchpad_prop}
    new_properties.update(new_schema.get("properties", {}))
    new_schema["properties"] = new_properties

    required_list = new_schema.get("required", [])
    if "chain_of_thought" not in required_list:
        new_schema["required"] = ["chain_of_thought"] + required_list

    return clean_schema_for_gemini(new_schema)


def run_gemini(payload_path: str, llm_model: str = "gemini-3.1-pro-preview"):
    """
    Sends a baseline payload to Gemini and returns the parsed JSON answer.

    Args:
        payload_path (str): Path to the payload JSON file.
        llm_model (str): Gemini model name.

    Returns:
        dict: The answer, with a "_usage" entry holding token counts.
    """
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    benchmark_id = payload["benchmark_id"]
    prompt = payload["user_prompt"]
    image_b64 = payload["image_base64"]
    schema = payload["structured_schema"]

    print(f"Executing {benchmark_id} via {llm_model}...")

    image_data = base64.b64decode(image_b64)
    image_blob = {
        "mime_type": "image/png",
        "data": image_data
    }

    model = genai.GenerativeModel(llm_model)

    enforced_schema = inject_scratchpad_into_schema(schema)

    config = genai.GenerationConfig(
        # Gemini 3 thinking models need temperature 1.0 to avoid reasoning loops.
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=enforced_schema
    )

    response = _generate_with_retry(model, [prompt, image_blob], config)

    result = json.loads(response.text)
    result["_usage"] = _usage_from_response(response)

    print("\n--- Model Output ---")
    print(json.dumps(result, indent=4))

    return result


def run_gemini_formative(payload_path: str, llm_model: str = "gemini-3.1-pro-preview"):
    """
    Sends a formative payload to Gemini and returns the parsed JSON answer.

    Args:
        payload_path (str): Path to the formative payload JSON file.
        llm_model (str): Gemini model name.

    Returns:
        dict: The answer, with a "_usage" entry holding token counts.
    """
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    benchmark_id = payload["benchmark_id"]
    posture = payload.get("posture_type", "unknown")
    system_prompt = payload["system_prompt"]
    base_image_b64 = payload["base_image_base64"]
    student_work = payload["student_work"]
    schema = payload["structured_schema"]

    print(f"Executing Formative Assessment: {benchmark_id} ({posture}) via {llm_model}...")

    base_image_data = base64.b64decode(base_image_b64)
    contents = [
        system_prompt,
        {"mime_type": "image/png", "data": base_image_data}
    ]

    for block in student_work:
        if block["type"] == "text":
            contents.append(block["content"])
        elif block["type"] == "image":
            img_data = base64.b64decode(block["content_base64"])
            contents.append({
                "mime_type": "image/png",
                "data": img_data
            })

    model = genai.GenerativeModel(llm_model)

    enforced_schema = inject_scratchpad_into_schema(schema)

    config = genai.GenerationConfig(
        # Gemini 3 thinking models need temperature 1.0 to avoid reasoning loops.
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=enforced_schema
    )

    response = _generate_with_retry(model, contents, config)

    result = json.loads(response.text)
    result["_usage"] = _usage_from_response(response)

    print("\n--- Model Output ---")
    print(json.dumps(result, indent=4))

    return result
