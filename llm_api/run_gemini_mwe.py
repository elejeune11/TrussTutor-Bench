import json
import base64
import os
import copy
import google.generativeai as genai


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def clean_schema_for_gemini(schema: dict) -> dict:
    """
    Recursively removes JSON Schema fields unsupported by the Gemini SDK Protobuf converter
    (e.g., 'additionalProperties', 'uniqueItems').
    """
    if not isinstance(schema, dict):
        return schema
        
    # Set of OpenAPI/JSON-Schema keywords that Gemini's SDK Schema object rejects
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
    Dynamically injects a chain-of-thought scratchpad into any JSON schema.
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
        
    # Strip the unsupported keys right before returning
    return clean_schema_for_gemini(new_schema)


def run_gemini_mwe(payload_path: str):
    """
    Loads a benchmark payload and executes a single Gemini 3.1 Pro API call 
    with strict schema enforcement and a dynamic scratchpad.
    """
    # 2. Load the compiled payload from disk
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    benchmark_id = payload["benchmark_id"]
    prompt = payload["user_prompt"]
    image_b64 = payload["image_base64"]
    schema = payload["structured_schema"]
    
    print(f"Executing {benchmark_id} via Gemini 3.1 Pro...")

    # 3. Format the image data for the Gemini SDK
    image_data = base64.b64decode(image_b64)
    image_blob = {
        "mime_type": "image/png", 
        "data": image_data
    }
    
    # 4. Initialize the latest Gemini 3.1 Pro preview model
    model = genai.GenerativeModel("gemini-3.1-pro-preview")
    
    # 5. Dynamically inject the scratchpad pattern
    enforced_schema = inject_scratchpad_into_schema(schema)
    
    # 6. Construct the generation configuration
    config = genai.GenerationConfig(
        # Keep temperature at 1.0 for Gemini 3 thinking models 
        # to prevent reasoning loops and degraded performance.
        temperature=1.0,  
        response_mime_type="application/json",
        response_schema=enforced_schema
    )
    
    # 7. Execute the multimodal request
    response = model.generate_content(
        contents=[prompt, image_blob],
        generation_config=config
    )
    
    # 8. Parse the guaranteed JSON output
    result = json.loads(response.text)
    
    print("\n--- Model Output ---")
    print(json.dumps(result, indent=4))
    
    return result


def run_gemini_mwe_formative(payload_path: str):
    """
    Loads a formative benchmark payload and executes a single Gemini API call.
    Constructs a multimodal prompt interleaving the base problem, system instructions,
    and the student's step-by-step text and Free Body Diagram images.
    """
    # 1. Load the compiled payload from disk
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    benchmark_id = payload["benchmark_id"]
    posture = payload.get("posture_type", "unknown")
    system_prompt = payload["system_prompt"]
    base_image_b64 = payload["base_image_base64"]
    student_work = payload["student_work"]
    schema = payload["structured_schema"]
    
    print(f"Executing Formative Assessment: {benchmark_id} ({posture}) via Gemini 3.1 Pro...")

    # 2. Build the contents array starting with the prompt and base image
    base_image_data = base64.b64decode(base_image_b64)
    contents = [
        system_prompt,
        {"mime_type": "image/png", "data": base_image_data}
    ]
    
    # 3. Sequentially append the student's interleaved text and FBD images
    for block in student_work:
        if block["type"] == "text":
            contents.append(block["content"])
        elif block["type"] == "image":
            img_data = base64.b64decode(block["content_base64"])
            contents.append({
                "mime_type": "image/png", 
                "data": img_data
            })
            
    # 4. Initialize the latest Gemini 3.1 Pro preview model
    model = genai.GenerativeModel("gemini-3.1-pro-preview")
    
    # 5. Dynamically inject the scratchpad pattern
    # (Forces the model to analyze the student's work step-by-step before voting)
    enforced_schema = inject_scratchpad_into_schema(schema)
    
    # 6. Construct the generation configuration
    config = genai.GenerationConfig(
        # Keep temperature at 1.0 for Gemini 3 thinking models 
        # to prevent reasoning loops and degraded performance.
        temperature=1.0,  
        response_mime_type="application/json",
        response_schema=enforced_schema
    )
    
    # 7. Execute the multimodal request
    response = model.generate_content(
        contents=contents,
        generation_config=config
    )
    
    # 8. Parse the guaranteed JSON output
    result = json.loads(response.text)
    
    print("\n--- Model Output ---")
    print(json.dumps(result, indent=4))
    
    return result