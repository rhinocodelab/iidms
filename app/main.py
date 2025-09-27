from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
import os
import json
import shutil
import subprocess
import requests
import xml.etree.ElementTree as ET
from file_handlers import handle_file, add_category_if_new
from db_integration import data_manager
from database import db
from typing import List, Dict
import logging
import asyncio
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

app = FastAPI(title="IDMS - Intelligent Document Management System", description="AI-powered document classification and FileNet upload system")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

config_file_path = "./criticality_config.json"

# System status cache
system_status = {
    "watsonx_ai": {"status": "unknown", "last_check": None, "error": None},
    "filenet": {"status": "unknown", "last_check": None, "error": None},
    "server": {"status": "running", "last_check": datetime.now(), "error": None}
}

@app.on_event("startup")
def startup_event():
    try:
        from file_handlers import load_existing_categories  # import if it's defined there
        categories = load_existing_categories()
        logger.info(f"Loaded {len(categories)} existing categories on startup.")
        
        # Start background task for system status checking
        asyncio.create_task(check_system_status_periodically())
    except Exception as e:
        logger.warning(f"Failed to load categories on startup: {e}")

async def check_watsonx_status():
    """Check WatsonX AI service status"""
    try:
        api_key = os.getenv("WATSONX_API_KEY")
        service_url = os.getenv("WATSONX_SERVICE_URL")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        
        if not all([api_key, service_url, project_id]):
            return {"status": "error", "error": "Missing WatsonX configuration"}
        
        # Simple health check - try to make a minimal request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # This is a simple connectivity check - try the correct WatsonX API endpoint
        response = requests.get(f"{service_url}/ml/v1-beta/projects/{project_id}", headers=headers, timeout=5)
        
        if response.status_code == 200:
            return {"status": "online", "error": None}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def check_filenet_status():
    """Check FileNet/DataCap connection status"""
    try:
        datacap_url = os.getenv("DATACAP_URL")
        application = os.getenv("APPLICATION")
        password = os.getenv("PASSWORD")
        station = os.getenv("STATION")
        user = os.getenv("USER")
        
        if not all([datacap_url, application, password, station, user]):
            return {"status": "error", "error": "Missing DataCap configuration"}
        
        # Try to make a simple connection test
        logon_payload = f"""
        <LogonProperties>
            <application>{application}</application>
            <password>{password}</password>
            <station>{station}</station>
            <user>{user}</user>
        </LogonProperties>
        """
        
        response = requests.post(
            f"{datacap_url}/Session/Logon",
            data=logon_payload,
            headers={"Content-Type": "application/xml"},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"status": "connected", "error": None}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def check_system_status():
    """Check all system components status"""
    global system_status
    
    # Check WatsonX AI
    watsonx_result = await check_watsonx_status()
    system_status["watsonx_ai"] = {
        **watsonx_result,
        "last_check": datetime.now()
    }
    
    # Check FileNet
    filenet_result = await check_filenet_status()
    system_status["filenet"] = {
        **filenet_result,
        "last_check": datetime.now()
    }
    
    # Server is always running if we're here
    system_status["server"] = {
        "status": "running",
        "last_check": datetime.now(),
        "error": None
    }
    
    logger.info(f"System status updated: WatsonX={watsonx_result['status']}, FileNet={filenet_result['status']}")
    return system_status

async def check_system_status_periodically():
    """Background task to check system status every 30 seconds"""
    while True:
        try:
            await check_system_status()
        except Exception as e:
            logger.error(f"Error checking system status: {e}")
        
        await asyncio.sleep(30)  # Check every 30 seconds

@app.get("/api/system-status")
async def get_system_status():
    """API endpoint to get current system status"""
    return system_status

# Database API endpoints
@app.get("/api/dashboard-metrics")
async def get_dashboard_metrics():
    """Get dashboard metrics from database"""
    try:
        metrics = data_manager.get_dashboard_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        return {"error": str(e)}

@app.get("/api/recent-documents")
async def get_recent_documents(limit: int = 10):
    """Get recent documents from database"""
    try:
        documents = data_manager.get_recent_documents(limit)
        return documents
    except Exception as e:
        logger.error(f"Error getting recent documents: {e}")
        return {"error": str(e)}

@app.get("/api/processing-logs")
async def get_processing_logs(limit: int = 20):
    """Get recent processing logs"""
    try:
        # This would need to be implemented in the database class
        return []
    except Exception as e:
        logger.error(f"Error getting processing logs: {e}")
        return {"error": str(e)}

@app.get("/api/error-logs")
async def get_error_logs(limit: int = 20):
    """Get recent error logs"""
    try:
        # This would need to be implemented in the database class
        return []
    except Exception as e:
        logger.error(f"Error getting error logs: {e}")
        return {"error": str(e)}

@app.get("/api/analytics")
async def get_analytics_data():
    """Get comprehensive analytics data for dashboard charts"""
    try:
        analytics_data = db.get_analytics_data()
        return analytics_data
    except Exception as e:
        logger.error(f"Error getting analytics data: {e}")
        return {"error": str(e)}

@app.get("/api/export/analytics/pdf")
async def export_analytics_pdf():
    """Export analytics data as PDF report"""
    try:
        # Get analytics data
        analytics_data = db.get_analytics_data()
        
        # Create analytics-temp directory if it doesn't exist
        analytics_temp_dir = "analytics-temp"
        os.makedirs(analytics_temp_dir, exist_ok=True)
        
        # Generate PDF filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"analytics_report_{timestamp}.pdf"
        pdf_path = os.path.join(analytics_temp_dir, pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("IDMS Analytics Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated on: {report_date}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Performance Indicators
        story.append(Paragraph("Key Performance Indicators", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        kpi_data = [
            ['Metric', 'Value'],
            ['Total Documents', str(analytics_data.get('total_documents', 0))],
            ['Processed Today', str(analytics_data.get('processed_today', 0))],
            ['Average Processing Time', f"{analytics_data.get('avg_processing_time', 0)}s"],
            ['Success Rate', f"{analytics_data.get('success_rate', 0)}%"],
            ['Error Rate', f"{analytics_data.get('error_rate', 0)}%"],
            ['Total Categories', str(analytics_data.get('total_categories', 0))],
            ['FileNet Uploads', str(analytics_data.get('filenet_uploads', 0))]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 20))
        
        # Document Types Distribution
        document_types = analytics_data.get('document_types', [])
        if document_types:
            story.append(Paragraph("Document Types Distribution", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            doc_type_data = [['Document Type', 'Count']]
            for doc_type in document_types[:10]:  # Limit to top 10
                doc_type_data.append([doc_type.get('name', 'Unknown'), str(doc_type.get('count', 0))])
            
            doc_type_table = Table(doc_type_data, colWidths=[3*inch, 2*inch])
            doc_type_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(doc_type_table)
            story.append(Spacer(1, 20))
        
        # Criticality Levels
        criticality_levels = analytics_data.get('criticality_levels', [])
        if criticality_levels:
            story.append(Paragraph("Criticality Levels Distribution", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            criticality_data = [['Criticality Level', 'Count']]
            for level in criticality_levels:
                criticality_data.append([level.get('name', 'Unknown'), str(level.get('count', 0))])
            
            criticality_table = Table(criticality_data, colWidths=[3*inch, 2*inch])
            criticality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(criticality_table)
            story.append(Spacer(1, 20))
        
        # File Types Distribution
        file_types = analytics_data.get('file_types', [])
        if file_types:
            story.append(Paragraph("File Types Distribution", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            file_type_data = [['File Type', 'Count']]
            for file_type in file_types[:10]:  # Limit to top 10
                file_type_data.append([file_type.get('name', 'Unknown'), str(file_type.get('count', 0))])
            
            file_type_table = Table(file_type_data, colWidths=[3*inch, 2*inch])
            file_type_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(file_type_table)
            story.append(Spacer(1, 20))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,  # Center alignment
            textColor=colors.grey
        )
        story.append(Paragraph("Generated by IDMS - Intelligent Document Management System", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Return the PDF file
        return FileResponse(
            path=pdf_path,
            filename=pdf_filename,
            media_type='application/pdf',
            headers={"Content-Disposition": f"inline; filename={pdf_filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error generating analytics PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")


def load_criticality_config(config_file_path: str) -> Dict:
    """
    Load criticality configuration from a JSON file.

    Args:
        config_file_path (str): Path to the criticality configuration JSON file.

    Returns:
        dict: Criticality configuration mapping.

    Raises:
        HTTPException: Raises 404 if file not found, 400 if JSON parsing fails.
    """
    try:
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
            logger.info(f"Loaded criticality config from {config_file_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Criticality configuration file not found at {config_file_path}")
        raise HTTPException(status_code=404, detail="Criticality configuration file not found.")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing criticality configuration file: {e}")
        raise HTTPException(status_code=400, detail="Error parsing criticality configuration file.")

def upload_to_filenet(image_path: str, document_type: str, confidentiality: str) -> None:
    """
    Upload the image to FileNet using the Java CLI.

    Args:
        image_path (str): Path to the image file on disk.
        document_type (str): The document type extracted from AI model output.

    Raises:
        subprocess.CalledProcessError: If the Java command fails.
    """
    command = [
        "java",
        "-jar",
        r"C:\Users\Administrator\Desktop\FileNetUpload.jar",
        image_path,
        document_type,
        confidentiality
    ]

    # result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # return result.stdout  # Return output for logging or further processing if needed
    logger.info(f"FileNet command====, {command}")
    return True

def save_uploaded_file(uploaded_file: UploadFile, dest_folder: str) -> str:
    """ Save uploaded file to disk and return full file path. """
    os.makedirs(dest_folder, exist_ok=True)
    file_path = os.path.join(dest_folder, uploaded_file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(uploaded_file.file.read())
    logger.info(f"Saved uploaded file to {file_path}")
    return file_path

def assign_criticality_and_upload(file_path: str, result: dict, criticality_config: dict) -> dict:
    """ Add criticality to result and upload file to FileNet. """
    if result is None:
        result = {
            "document_type": "Unknown",
            "criticality": "Unknown",
            "error": "File could not be processed."
        }
        return result
    
    doc_type = result.get("document_type", "Unknown")

    # ✅ Add category to existing_categories.txt if new
    add_category_if_new(doc_type)

    criticality = criticality_config.get(doc_type, "Unknown")
    result["criticality"] = criticality

    # FileNet upload attempt
    try:
        filenet_output = upload_to_filenet(file_path, doc_type, criticality)
        logger.info(f"FileNet upload successful for {file_path}: {filenet_output}")
        result['filenet_upload'] = "Success"
    except subprocess.CalledProcessError as e:
        logger.error(f"FileNet upload failed for {file_path}: {e.stderr}")
        result['filenet_upload'] = f"Failed: {e.stderr}"
    except Exception as e:
        logger.error(f"Unexpected error during FileNet upload for {file_path}: {str(e)}")
        result['filenet_upload'] = f"Failed: {str(e)}"
    
    return result

def process_archive(file_path: str, criticality_config: dict) -> Dict[str, dict]:
    """ Process an archive file: extract, classify, upload, cleanup. """
    results = {}
    extracted_folder = os.path.splitext(file_path)[0]
    extracted_results = handle_file(file_path)  # assuming returns dict {filename: result}

    if isinstance(extracted_results, dict):
        for extracted_file, result in extracted_results.items():
            # Update result with criticality and FileNet upload
            updated_result = assign_criticality_and_upload(extracted_file, result, criticality_config)
            results[extracted_file] = updated_result

        # Cleanup extracted folder
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
            logger.info(f"Removed extracted folder: {extracted_folder}")
    else:
        logger.warning(f"Unexpected extracted result format for archive {file_path}")
        results[os.path.basename(file_path)] = {"error": "Unexpected archive processing result format"}

    return results

def process_single_file(file_path: str, criticality_config: dict) -> dict:
    """ Process a non-archive single file. """
    processing_start_time = datetime.now()
    
    try:
        result = handle_file(file_path)
        result = assign_criticality_and_upload(file_path, result, criticality_config)
        
        # Save to database
        try:
            processing_end_time = datetime.now()
            document_id = data_manager.save_document_processing(file_path, result, processing_start_time, processing_end_time)
            result['document_id'] = document_id
            logger.info(f"Document saved to database with ID: {document_id}")
        except Exception as db_error:
            logger.error(f"Failed to save document to database: {db_error}")
            data_manager.log_error("database_error", str(db_error), "medium", context_data={"file_path": file_path})
        
        return result
    except Exception as e:
        processing_end_time = datetime.now()
        data_manager.log_error("processing_error", str(e), "high", context_data={"file_path": file_path})
        raise e

def process_uploaded_files(files: List[UploadFile]) -> Dict[str, dict]:
    criticality_config = load_criticality_config(config_file_path)
    results = {}
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in files:
        temp_file_path = save_uploaded_file(uploaded_file, temp_dir)
        logger.info(f"Processing uploaded file - {uploaded_file.filename}")

        if temp_file_path.lower().endswith(('.zip', '.7z', '.tar', '.gz', '.bz2', '.xz', '.rar')):
            archive_results = process_archive(temp_file_path, criticality_config)
            results.update(archive_results)
        else:
            single_result = process_single_file(temp_file_path, criticality_config)
            results[uploaded_file.filename] = single_result

        # Cleanup uploaded file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")

    return results

@app.post("/api/process_files/")
async def process_files(files: List[UploadFile] = File(...)) -> Dict[str, dict]:
    """
    FastAPI endpoint to process uploaded files and assign criticality based on configuration.

    Args:
        files (List[UploadFile]): List of files uploaded via the form.

    Returns:
        Dict[str, dict]: Classification results with criticality for each uploaded file.

    Raises:
        HTTPException: For unexpected server errors.
    """
    try:
        return process_uploaded_files(files)
    except HTTPException:
        raise  # Re-raise HTTPExceptions as is
    except Exception as e:
        logger.error(f"Unexpected error processing files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


LOGON_URL =  f"{os.getenv('DATACAP_URL')}/Session/Logon"
CREATE_BATCH_URL = f"{os.getenv('DATACAP_URL')}/Queue/CreateBatch"
UPLOAD_FILE_URL_TEMPLATE = "{}/Queue/UploadFile/watsonxai/{}".format(
        os.getenv("DATACAP_URL"),
        "{queueId}"
    )
RELEASE_BATCH_URL_TEMPLATE = "{}/Queue/ReleaseBatch/watsonxai/{}/finished".format(
        os.getenv("DATACAP_URL"),
        "{queueId}"
    )

APPLICATION = os.getenv("APPLICATION")
PASSWORD = os.getenv("PASSWORD")
STATION = os.getenv("STATION")
USER = os.getenv("USER")
JOB = os.getenv("JOB")


def extract_queue_id(xml_str: str) -> str:
    """
    Extracts batch id from the XML response string.
    Assumes XML contains <batchId>...</batchId> element.
    Adjust this according to your actual XML response.
    """
    try:
        root = ET.fromstring(xml_str)
        queue_id = root.find("queueId")
        if queue_id is not None:
            return queue_id.text
        else:
            raise ValueError("queueId not found in response.")
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML: {e}")

@app.post("/api/business/upload")
async def upload_business_document(file: UploadFile = File(...)):
    session = requests.Session()
    # 1. Logon
    logon_payload = f"""
    <LogonProperties>
        <application>{APPLICATION}</application>
        <password>{PASSWORD}</password>
        <station>{STATION}</station>
        <user>{USER}</user>
    </LogonProperties>
    """
    logon_headers = {
        "Content-Type": "application/xml"
    }
    logon_resp = session.post(LOGON_URL, data=logon_payload, headers=logon_headers)
    logger.info(f"LOGIN Response: {logon_resp}")
    if logon_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Logon failed: {logon_resp.text}")

    # 2. CreateBatch
    create_batch_payload = f"""
    <createBatchAttributes>
    <application>{APPLICATION}</application>
    <job>Navigator Job</job>
    <pageFile>
    <B id="">
        <V n="STATUS">0</V>
        <V n="TYPE">{APPLICATION}</V>
        <V n="Filename">{file.filename}</V>
     </B>
    </pageFile>
    </createBatchAttributes>
    """
    create_batch_headers = {"Content-Type": "application/xml"}
    create_batch_resp = session.post(CREATE_BATCH_URL, data=create_batch_payload, headers=create_batch_headers)
    logger.info(f"CREATE BATCH Response: {create_batch_resp}")
    if create_batch_resp.status_code != 201:
        raise HTTPException(status_code=500, detail=f"CreateBatch failed: {create_batch_resp.text}")
    try:
        queueId = extract_queue_id(create_batch_resp.text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. UploadFile
    files = {
        'file': (file.filename, await file.read(), file.content_type)
    }
    upload_url = UPLOAD_FILE_URL_TEMPLATE.format(queueId=queueId)
    upload_resp = session.post(upload_url, files=files)
    logger.info(f"UPLOAD FILE Response: {upload_resp}")
    if upload_resp.status_code != 201:
        raise HTTPException(status_code=500, detail=f"UploadFile failed: {upload_resp.text}")

    # 4. ReleaseBatch
    release_url = RELEASE_BATCH_URL_TEMPLATE.format(queueId=queueId)
    release_resp = session.put(release_url)
    logger.info(f"RELEASE BATCH Response: {release_resp}")
    if release_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"ReleaseBatch failed: {release_resp.text}")

    return {"message": "Business document processed successfully", "queueId": queueId}

# Frontend Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload_files", response_class=HTMLResponse)
async def upload_files_frontend(request: Request, files: List[UploadFile] = File(...)):
    """Handle file uploads from the frontend form"""
    try:
        results = process_uploaded_files(files)
        return templates.TemplateResponse("results.html", {
            "request": request,
            "results": results,
            "message": "Files processed successfully!"
        })
    except Exception as e:
        logger.error(f"Error processing files from frontend: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/business_upload", response_class=HTMLResponse)
async def business_upload_frontend(request: Request, file: UploadFile = File(...)):
    """Handle business document uploads from the frontend"""
    try:
        # Call the existing business upload logic
        session = requests.Session()
        
        # Logon
        logon_payload = f"""
        <LogonProperties>
            <application>{APPLICATION}</application>
            <password>{PASSWORD}</password>
            <station>{STATION}</station>
            <user>{USER}</user>
        </LogonProperties>
        """
        logon_headers = {"Content-Type": "application/xml"}
        logon_resp = session.post(LOGON_URL, data=logon_payload, headers=logon_headers)
        
        if logon_resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Logon failed: {logon_resp.text}")

        # CreateBatch
        create_batch_payload = f"""
        <createBatchAttributes>
        <application>{APPLICATION}</application>
        <job>Navigator Job</job>
        <pageFile>
        <B id="">
            <V n="STATUS">0</V>
            <V n="TYPE">{APPLICATION}</V>
            <V n="Filename">{file.filename}</V>
         </B>
        </pageFile>
        </createBatchAttributes>
        """
        create_batch_headers = {"Content-Type": "application/xml"}
        create_batch_resp = session.post(CREATE_BATCH_URL, data=create_batch_payload, headers=create_batch_headers)
        
        if create_batch_resp.status_code != 201:
            raise HTTPException(status_code=500, detail=f"CreateBatch failed: {create_batch_resp.text}")
            
        try:
            queueId = extract_queue_id(create_batch_resp.text)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # UploadFile
        files_data = {'file': (file.filename, await file.read(), file.content_type)}
        upload_url = UPLOAD_FILE_URL_TEMPLATE.format(queueId=queueId)
        upload_resp = session.post(upload_url, files=files_data)
        
        if upload_resp.status_code != 201:
            raise HTTPException(status_code=500, detail=f"UploadFile failed: {upload_resp.text}")

        # ReleaseBatch
        release_url = RELEASE_BATCH_URL_TEMPLATE.format(queueId=queueId)
        release_resp = session.put(release_url)
        
        if release_resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ReleaseBatch failed: {release_resp.text}")

        return templates.TemplateResponse("success.html", {
            "request": request,
            "message": "Business document processed successfully",
            "queueId": queueId,
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Error processing business upload from frontend: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Serve the admin dashboard page"""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Serve the analytics dashboard page"""
    return templates.TemplateResponse("analytics.html", {"request": request})