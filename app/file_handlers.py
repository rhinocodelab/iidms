import os
import pandas as pd
from utils import read_file
from prompts import prompt
from classifier import call_llm_image, call_llm_text
import logging

CATEGORIES_FILE = r"./existing_categories.txt"

def load_existing_categories(filepath: str = CATEGORIES_FILE):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        categories = [line.strip() for line in f if line.strip()]
    return categories

def add_category_if_new(new_category: str, filepath: str = CATEGORIES_FILE):
    """Add new category to file if it's not already present."""
    new_category = new_category.strip()

    # Skip invalid or placeholder categories
    if not new_category or new_category.lower() in {"unknown", "none", "n/a", "undefined"}:
        logging.debug(f"Skipped adding invalid category: '{new_category}'")
        return

    categories = load_existing_categories(filepath)
    if new_category not in categories:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(new_category + "\n")
        logging.info(f"New category added: {new_category}")
    else:
        logging.debug(f"Category already exists: {new_category}")

def handle_file(file_path):
    """ Handle file based on its extension and classify it """
    # Load categories from txt file
    existing_categories = load_existing_categories()
    categories_str = ", ".join(existing_categories)

    # Read the content
    content = read_file(file_path)

    if isinstance(content, dict) and content.get("file_type") == "zip":
        results = {}
        for extracted_file in content.get("extracted_files", []):
            logging.info(f"Processing extracted file: {extracted_file}")
            result = handle_file(extracted_file)
            results[extracted_file] = result
        return results

    if isinstance(content, dict) and content.get("file_type") == "image":
        updated_prompt = prompt.replace("{existing_categories}", categories_str).replace("{content}", "Image content")
        return call_llm_image(content.get("img"), updated_prompt)

    elif isinstance(content, list) and all(isinstance(i, dict) for i in content):
        content_str = str(content[:100000])
    elif isinstance(content, pd.DataFrame):
        content_str = content.head(100).to_string()
    elif isinstance(content, str):
        content_str = content[:100000]
    else:
        return {
            "document_type": "Unknown",
            "Tags": "",
            "summary": "",
            "reasoning": "Unsupported file type or failed to read content."
        }

    updated_prompt = prompt.replace("{existing_categories}", categories_str).replace("{content}", content_str)

    return call_llm_text(updated_prompt)