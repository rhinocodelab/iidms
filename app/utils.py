import os
import base64
import pandas as pd
import json
import yaml
from docx import Document
from PyPDF2 import PdfReader
import traceback
import zipfile
from lxml import etree
import subprocess
import logging

def extract_archive(file_path, extract_to_dir):
    """ Decompress any archive file using 7z CLI and return the list of extracted file paths """
    try:
        # Run the 7z command-line tool to extract the archive file
        subprocess.run(['7z', 'x', file_path, f'-o{extract_to_dir}', '-y'], check=True)
        # Return the paths of the extracted files
        extracted_files = [os.path.join(extract_to_dir, file) for file in os.listdir(extract_to_dir)]
        logging.info(f"Archive Extraction Successful: {extracted_files}")
        return extracted_files
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting archive {file_path}: {e}")
        traceback.print_exc()  # This gives the full traceback
        return []

def read_file(file_path):
    """ Read the content of a file based on its extension """
    
    file_ext = os.path.splitext(file_path)[-1].lower()
    logging.info(f"Reading file: {file_path}")
    
    # Handle CSV files
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    
    # Handle Excel files
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    
    # Handle JSON files
    elif file_ext == '.json':
        with open(file_path, 'r', encoding='UTF-8') as file:
            content = file.read()
            if not content:
                logging.warning(f"Warning: The file {file_path} is empty.")
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {file_path}: {e}")
                return None
    
    # Handle YAML files (.yaml or .yml)
    elif file_ext in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='UTF-8') as file:
            try:
                # Load YAML content
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                logging.error(f"Error decoding YAML from {file_path}: {e}")
                return None
    
    # Handle .odt files (OpenDocument Text)
    elif file_ext == '.odt':
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Extract the content.xml file
                content = zf.read("content.xml")

            # Parse the XML content
            tree = etree.fromstring(content)

            # Define namespaces dictionary (this is crucial for parsing ODT)
            namespaces = {
                'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
                'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
                'style': 'urn:oasis:names:tc:opendocument:xmlns:style:1.0',
                'fo': 'urn:xsl:formats',
                'svg': 'urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0',
                'xlink': 'http://www.w3.org/1999/xlink'
            }

            paragraphs = tree.xpath('//text:p', namespaces=namespaces)
            paragraph_texts = '\n'.join([paragraph.text for paragraph in paragraphs if paragraph.text])
            return paragraph_texts
        except Exception as e:
            logging.error(f"Error reading .odt file {file_path}: {e}")
            traceback.print_exc()  # This gives the full traceback
            return None
    
    # Handle DOCX files (text extraction)
    elif file_ext == '.docx':
        doc = Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    
    # Handle PDF files (text extraction)
    elif file_ext == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    # Handle image files (Returning file path as placeholder)
    elif file_ext in ['.png', '.jpg', '.jpeg']:
        with open(file_path,"rb") as img:
            encoded_img = base64.b64encode(img.read()).decode("utf-8")
        return {"file_type": "image", "img": encoded_img}

    # Handle code files (Returning file content)
    elif file_ext in ['.txt','.cpp', '.java', '.php', '.py', '.bat']:
        with open(file_path, 'r',  encoding='utf-8') as file:
            return file.read()

    # Handle archive files (extract and process each file inside)
    elif file_ext in ['.zip', '.7z', '.tar', '.gz', '.bz2', '.xz', '.rar']:
        extract_to_dir = os.path.splitext(file_path)[0]  # Use a folder named after the archive file
        os.makedirs(extract_to_dir, exist_ok=True)
        extracted_files = extract_archive(file_path, extract_to_dir)
        return {"extracted_files":extracted_files, "file_type":"zip"}
