# Document Classification & FileNet Upload API
This FastAPI application processes uploaded files (including archives), classifies documents using a custom LLM model, assigns a criticality level based on document type, and uploads processed files to IBM FileNet folders named after the document types.

### Features
- Supports single files and archive files (.zip, .7z, .tar, .gz, .bz2, .xz, .rar)
- Classifies document types using a language model (LLM) dynamically.
- Assigns criticality level from a configurable JSON mapping
- Automatically uploads classified files to FileNet folders corresponding to their document types

## Environment Setup
Create a .env file in the project app directory with the following variables:

```
WATSONX_API_KEY=<your_watsonx_api_key>
WATSONX_SERVICE_URL=<your_watsonx_service_url>
WATSONX_PROJECT_ID=<your_watsonx_project_id>
WATSONX_MODEL_ID=<your_watsonx_model_id>

DATACAP_URL=<your_datacap_url>
APPLICATION=<your_application_name>
PASSWORD=<your_password>
STATION=<your_station>
USER=<your_username>
JOB=<your_job_name>
```

## Installation
1. Clone the repository:
```
git clone https://github.ibm.com/Build-Lab-India/CyberCorp.git
cd CyberCorp
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  
On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Ensure you have a criticality_config.json file in the project app directory. This JSON file maps document types to their criticality levels. Example:
```
{
     "Aadhaar Card":"Public",
     "Non-Disclosure Agreement (NDA)": "Confidential",
}
```

## Running the API
Start the FastAPI server using uvicorn with hot reload enabled:
```
uvicorn main:app --reload
```
By default, the app will be accessible at: http://127.0.0.1:8000

