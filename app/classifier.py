import os
import re
import json
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
import logging

load_dotenv()

api_key = os.getenv("WATSONX_API_KEY")
service_url = os.getenv("WATSONX_SERVICE_URL")
project_id = os.getenv("WATSONX_PROJECT_ID")
model_id = os.getenv("WATSONX_MODEL_ID")

if not api_key or not service_url or not project_id:
    logging.error("API key, service URL, and project ID must be set as environment variables.")

# -----------------------------------
# Initialize Watsonx Model
# -----------------------------------
model = ModelInference(
    model_id=model_id,
    credentials={"api_key": api_key, "url": service_url},
    params={"decoding_method": "greedy", "max_new_tokens": 800},
    project_id=project_id
)

def extract_json_from_llm_output(llm_response: str):
    """
    Extracts and parses a JSON object from LLM output string.

    Args:
        llm_response (str): Raw string output from the LLM.

    Returns:
        dict | None: Parsed JSON dictionary if successful, otherwise None.
    """
    try:
        # Remove markdown or "**Answer:**" prefix
        cleaned_response = re.sub(r'^\*\*Answer:\*\*\s*', '', llm_response.strip())

        # Extract JSON block from between ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
        json_str = match.group(1) if match else cleaned_response
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from model output: {e}")
        return None
    except Exception as e:
        logging.error("Unexpected error during JSON extraction.")
        return None


def call_llm_image(encoded_image: str, prompt_text: str) -> str:
    """
    Calls the Watsonx foundation model with a base64-encoded image and prompt text.

    Args:
        encoded_image (str): Base64-encoded image string.
        prompt_text (str): The prompt text to send to the model.

    Returns:
        str: Model's textual response.
             Returns a dictionary with an "error" if no output.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
            ]
        }
    ]
    try:
        response = model.chat(messages=messages)
        logging.info(f"Model Response=== {response}")
        usage = response.get("usage", {})
        logging.info(f"Prompt tokens: {usage.get('prompt_tokens')}, Completion tokens: {usage.get('completion_tokens')}, Total: {usage.get('total_tokens')}")

        choices = response.get("choices")
        if choices and "message" in choices[0]:
            llm_output = choices[0]["message"]["content"]
            parsed_json = extract_json_from_llm_output(llm_output)
            logging.info(f"Parsed Output: {parsed_json}")
            return parsed_json
        else:
            logging.error("Model returned no choices.")
            return {"error": "Model returned no choices."}

    except Exception as e:
        logging.error(f"Error during model inference. : {str(e)}")
        return {"error": "Error during model inference. "}

def call_llm_text(prompt):
    """
    Sends a text prompt to the LLM and attempts to parse the response as JSON.

    Args:
        prompt (str): The input prompt to be sent to the LLM.

    Returns:
        dict: Parsed JSON response from the LLM if successful.
              Returns a dictionary with an "error" key if parsing fails.
    """
    output = model.generate_text(prompt=prompt)
    output = output.replace("```","")
    # Attempt to parse the output as JSON
    try:
        parsed_output = json.loads(output)
        logging.info(f"Parsed Output: {parsed_output}")
        return parsed_output
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing output as JSON: {e}")
        return {"error": "Invalid JSON output from model"}
