from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
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
from mfa_utils import MFAUtils
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import mimetypes
import random
import configparser
import warnings
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import Document
import secrets
import hashlib
from typing import Optional

app = FastAPI(title="IDMS - Intelligent Document Management System", description="AI-powered document classification and FileNet upload system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Session storage (in production, use Redis or database)
user_sessions = {}

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

# Suppress Google Cloud internal warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")

def setup_gcp_credentials(credentials_file: str = "ghostlayer.json") -> str:
    """Setup GCP credentials using service account JSON file."""
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file not found: {credentials_file}")
    
    # Set the environment variable for Google Cloud credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(credentials_file)
    
    # Suppress Google Cloud internal warnings
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    
    # Load and validate the credentials file
    with open(credentials_file, 'r') as f:
        credentials = json.load(f)
    
    # Validate required fields
    required_fields = ['type', 'project_id', 'private_key', 'client_email']
    for field in required_fields:
        if field not in credentials:
            raise KeyError(f"Missing required field '{field}' in credentials file")
    
    if credentials.get('type') != 'service_account':
        raise ValueError("Invalid credentials type. Expected 'service_account'")
    
    return credentials['project_id']

def load_processor_config(config_file: str = "ghostlayer_ocr.ini") -> dict:
    """Load processor configuration from INI file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'Ghostlayer' not in config:
        raise configparser.Error("Missing [Ghostlayer] section in configuration file")
    
    ghostlayer_section = config['Ghostlayer']
    
    return {
        'location': ghostlayer_section.get('Region', 'us').lower(),
        'processor_id': ghostlayer_section.get('ID', ''),
        'name': ghostlayer_section.get('Name', 'ghostlayer'),
        'endpoint': ghostlayer_section.get('Prediction_Endpoint', '')
    }

def load_document_identification_config(config_file: str = "document_identification.json") -> dict:
    """Load document identification configuration from JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Document identification config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def classify_document(document_text: str, config: dict) -> dict:
    """Classify document type based on keywords found in the text."""
    document_types = config.get('document_types', {})
    settings = config.get('classification_settings', {})
    
    case_sensitive = settings.get('case_sensitive', False)
    partial_match = settings.get('partial_match', True)
    fallback_type = settings.get('fallback_document_type', 'unknown')
    
    # Prepare text for matching
    search_text = document_text if case_sensitive else document_text.lower()
    
    classification_results = []
    
    for doc_type, doc_config in document_types.items():
        keywords = doc_config.get('keywords', [])
        confidence_threshold = doc_config.get('confidence_threshold', 0.3)
        
        # Debug logging for Aadhaar
        if doc_type == 'aadhaar':
            logger.info(f"Processing Aadhaar classification:")
            logger.info(f"  Keywords: {keywords}")
            logger.info(f"  Threshold: {confidence_threshold}")
            logger.info(f"  Search text: {search_text[:100]}...")
        
        # Count keyword matches
        matches_found = 0
        matched_keywords = []
        
        for keyword in keywords:
            search_keyword = keyword if case_sensitive else keyword.lower()
            
            if partial_match:
                # Partial match - keyword appears anywhere in text
                if search_keyword in search_text:
                    matches_found += 1
                    matched_keywords.append(keyword)
                    if doc_type == 'aadhaar':
                        logger.info(f"  ✅ Matched keyword: '{keyword}' -> '{search_keyword}'")
            else:
                # Exact match - keyword appears as whole word
                import re
                pattern = r'\b' + re.escape(search_keyword) + r'\b'
                if re.search(pattern, search_text):
                    matches_found += 1
                    matched_keywords.append(keyword)
                    if doc_type == 'aadhaar':
                        logger.info(f"  ✅ Matched keyword: '{keyword}' -> '{search_keyword}'")
        
        # Calculate confidence score
        total_keywords = len(keywords)
        confidence_score = matches_found / total_keywords if total_keywords > 0 else 0
        
        if doc_type == 'aadhaar':
            logger.info(f"  Aadhaar results: {matches_found}/{total_keywords} matches, confidence: {confidence_score:.3f}, threshold: {confidence_threshold}")
        
        if confidence_score >= confidence_threshold:
            classification_results.append({
                'document_type': doc_type,
                'name': doc_config.get('name', doc_type),
                'description': doc_config.get('description', ''),
                'confidence_score': round(confidence_score, 3),
                'matches_found': matches_found,
                'total_keywords': total_keywords,
                'matched_keywords': matched_keywords
            })
    
    # Sort by confidence score (highest first)
    classification_results.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # Return the best match or fallback
    if classification_results:
        best_match = classification_results[0]
        return {
            'document_type': best_match['document_type'],
            'document_name': best_match['name'],
            'description': best_match['description'],
            'confidence_score': best_match['confidence_score'],
            'classification_details': {
                'matches_found': best_match['matches_found'],
                'total_keywords': best_match['total_keywords'],
                'matched_keywords': best_match['matched_keywords']
            },
            'all_matches': classification_results
        }
    else:
        return {
            'document_type': fallback_type,
            'document_name': 'Unknown Document',
            'description': 'Document type could not be determined',
            'confidence_score': 0.0,
            'classification_details': {
                'matches_found': 0,
                'total_keywords': 0,
                'matched_keywords': []
            },
            'all_matches': []
        }

def detect_mime_type(file_path: str) -> str:
    """Detect MIME type of the file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Handle common cases
    if mime_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return 'application/pdf'
        elif ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        else:
            return 'application/octet-stream'
    
    return mime_type

async def process_document_with_ai(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
) -> dict:
    """Process a document using Google Cloud Document AI OCR processor."""
    
    logger.info(f"Starting document processing for: {file_path}")
    
    # Setup Client and Request
    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # Full resource name of the processor
    name = client.processor_path(project_id, location, processor_id)

    # Read the file content
    with open(file_path, "rb") as image:
        image_content = image.read()

    raw_document = documentai.RawDocument(
        content=image_content, mime_type=mime_type
    )

    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    # Call the API
    result = client.process_document(request=request)
    document = result.document

    # Extract OCR Results (Text and Layout)
    extracted_data = {
        "status": "success",
        "processor_type": "Document OCR",
        "file_path": file_path,
        "mime_type": mime_type,
        "full_document_text": document.text,
        "pages": [],
        "document_classification": {}
    }
    
    # Function to extract normalized coordinates from a BoundingPoly
    def get_coords(bounding_poly):
        if bounding_poly and bounding_poly.normalized_vertices:
            return [
                {"x": round(v.x, 4), "y": round(v.y, 4)} 
                for v in bounding_poly.normalized_vertices
            ]
        return None

    # Extract layout information from document.pages
    for i, page in enumerate(document.pages):
        page_data = {
            "page_number": i + 1,
            "blocks": [],
            "paragraphs": [],
            "tokens": []
        }

        # Extract Blocks
        for block in page.blocks:
            if block.layout.text_anchor.text_segments:
                start_index = block.layout.text_anchor.text_segments[0].start_index
                end_index = block.layout.text_anchor.text_segments[0].end_index
                page_data["blocks"].append({
                    "text": document.text[start_index:end_index],
                    "coordinates": get_coords(block.layout.bounding_poly)
                })
            
        # Extract Paragraphs
        for paragraph in page.paragraphs:
            if paragraph.layout.text_anchor.text_segments:
                start_index = paragraph.layout.text_anchor.text_segments[0].start_index
                end_index = paragraph.layout.text_anchor.text_segments[0].end_index
                page_data["paragraphs"].append({
                    "text": document.text[start_index:end_index],
                    "coordinates": get_coords(paragraph.layout.bounding_poly)
                })

        # Extract Tokens
        for token in page.tokens:
            if token.layout.text_anchor.text_segments:
                start_index = token.layout.text_anchor.text_segments[0].start_index
                end_index = token.layout.text_anchor.text_segments[0].end_index
                page_data["tokens"].append({
                    "text": document.text[start_index:end_index],
                    "coordinates": get_coords(token.layout.bounding_poly)
                })
            
        extracted_data["pages"].append(page_data)

    # Classify document type based on extracted text
    try:
        doc_config = load_document_identification_config()
        logger.info(f"Loaded document identification config with {len(doc_config.get('document_types', {}))} document types")
        
        # Debug: Log Aadhaar keywords
        aadhaar_config = doc_config.get('document_types', {}).get('aadhaar', {})
        if aadhaar_config:
            logger.info(f"Aadhaar keywords: {aadhaar_config.get('keywords', [])}")
            logger.info(f"Aadhaar threshold: {aadhaar_config.get('confidence_threshold', 0.3)}")
        
        # Debug: Log extracted text
        logger.info(f"Extracted text for classification: {document.text[:200]}...")
        
        classification_result = classify_document(document.text, doc_config)
        logger.info(f"Classification result: {classification_result}")
        extracted_data["document_classification"] = classification_result
        
        # Debug: Log the final result
        logger.info(f"Final classification: {classification_result.get('document_type', 'unknown')} - {classification_result.get('document_name', 'Unknown')} (confidence: {classification_result.get('confidence_score', 0.0)})")
    except Exception as e:
        # If classification fails, continue without it
        extracted_data["document_classification"] = {
            "document_type": "unknown",
            "document_name": "Unknown Document",
            "description": "Document classification failed",
            "confidence_score": 0.0,
            "classification_details": {
                "matches_found": 0,
                "total_keywords": 0,
                "matched_keywords": []
            },
            "all_matches": [],
            "error": str(e)
        }

    return extracted_data

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

def process_archive(file_path: str, criticality_config: dict, user_data: dict = None) -> Dict[str, dict]:
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

def process_single_file(file_path: str, criticality_config: dict, user_data: dict = None) -> dict:
    """ Process a non-archive single file. """
    processing_start_time = datetime.now()
    
    try:
        result = handle_file(file_path)
        result = assign_criticality_and_upload(file_path, result, criticality_config)
        
        # Save to database
        try:
            processing_end_time = datetime.now()
            if user_data:
                # Use new AI document classifications table for authenticated users
                document_id = data_manager.save_ai_document_processing(file_path, result, processing_start_time, processing_end_time, user_data)
            else:
                # Fallback to old documents table for non-authenticated users
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

def process_uploaded_files(files: List[UploadFile], user_data: dict = None) -> Dict[str, dict]:
    criticality_config = load_criticality_config(config_file_path)
    results = {}
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in files:
        temp_file_path = save_uploaded_file(uploaded_file, temp_dir)
        logger.info(f"Processing uploaded file - {uploaded_file.filename}")

        if temp_file_path.lower().endswith(('.zip', '.7z', '.tar', '.gz', '.bz2', '.xz', '.rar')):
            archive_results = process_archive(temp_file_path, criticality_config, user_data)
            results.update(archive_results)
        else:
            single_result = process_single_file(temp_file_path, criticality_config, user_data)
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
    """Redirect to login page"""
    return RedirectResponse(url="/login")

@app.get("/test-auth", response_class=HTMLResponse)
async def test_auth(request: Request):
    """Test authentication endpoint"""
    user = require_auth(request)
    if user:
        return HTMLResponse(f"<h1>Authenticated as: {user.get('username')} ({user.get('role')})</h1>")
    else:
        return HTMLResponse("<h1>Not authenticated</h1>")

@app.get("/debug-sessions")
async def debug_sessions():
    """Debug sessions endpoint"""
    return {
        "total_sessions": len(user_sessions),
        "sessions": list(user_sessions.keys())[:5]  # Show first 5 session keys
    }

@app.post("/api/init-database")
async def init_database():
    """Initialize database with new tables"""
    try:
        # Reinitialize database to create new tables
        db.init_database()
        return {"success": True, "message": "Database initialized successfully"}
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return {"success": False, "message": f"Database initialization failed: {str(e)}"}

@app.get("/api/init-database")
async def init_database_get():
    """Initialize database with new tables (GET method for easy browser access)"""
    try:
        # Reinitialize database to create new tables
        db.init_database()
        return {"success": True, "message": "Database initialized successfully"}
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return {"success": False, "message": f"Database initialization failed: {str(e)}"}

@app.get("/api/check-tables")
async def check_tables():
    """Check if the new GhostLayer table exists"""
    try:
        import sqlite3
        
        # Connect to database
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Check if user_ghostlayer_documents table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_ghostlayer_documents'
        """)
        user_ghostlayer_exists = cursor.fetchone() is not None
        
        # Check if old ghostlayer_documents table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ghostlayer_documents'
        """)
        old_ghostlayer_exists = cursor.fetchone() is not None
        
        # Get table info if it exists
        table_info = None
        if user_ghostlayer_exists:
            cursor.execute("PRAGMA table_info(user_ghostlayer_documents)")
            columns = cursor.fetchall()
            table_info = {
                "columns": [{"name": col[1], "type": col[2], "not_null": col[3], "default": col[4]} for col in columns]
            }
        
        # Get row count if table exists
        row_count = 0
        if user_ghostlayer_exists:
            cursor.execute("SELECT COUNT(*) FROM user_ghostlayer_documents")
            row_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "user_ghostlayer_documents_exists": user_ghostlayer_exists,
            "old_ghostlayer_documents_exists": old_ghostlayer_exists,
            "table_info": table_info,
            "row_count": row_count,
            "database_path": db.db_path
        }
        
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return {
            "error": str(e),
            "user_ghostlayer_documents_exists": False,
            "old_ghostlayer_documents_exists": False
        }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard page (requires authentication)"""
    logger.info(f"Dashboard accessed from: {request.headers.get('referer', 'direct')}")
    user = require_auth(request)
    if not user:
        logger.warning("No user found, redirecting to login")
        return RedirectResponse(url="/login")
    
    logger.info(f"User {user.get('username')} accessing dashboard")
    
    # Redirect admin users to admin console
    if user.get("role") == "admin":
        logger.info("Admin user, redirecting to admin console")
        return RedirectResponse(url="/admin-console")
    
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the upload page (requires authentication)"""
    user = require_auth(request)
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})

@app.post("/upload_files", response_class=HTMLResponse)
async def upload_files_frontend(request: Request, files: List[UploadFile] = File(...)):
    """Handle file uploads from the frontend form"""
    try:
        # Get user from session for AI Document Classification
        user_data = require_auth(request)
        if not user_data:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": "Authentication required for AI Document Classification"
            })
        
        results = process_uploaded_files(files, user_data)
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
    """Serve the admin dashboard page (requires admin authentication)"""
    user = require_auth(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin.html", {"request": request, "user": user})

@app.get("/admin-console", response_class=HTMLResponse)
async def admin_console(request: Request):
    """Serve the admin console page (requires admin authentication)"""
    user = require_auth(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_console.html", {"request": request, "user": user})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Serve the analytics dashboard page (requires authentication)"""
    user = require_auth(request)
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("analytics.html", {"request": request, "user": user})

@app.get("/ghostlayer-ai", response_class=HTMLResponse)
async def ghostlayer_page(request: Request):
    """Serve the GhostLayer AI page (requires authentication)"""
    user = require_auth(request)
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("ghostlayer.html", {"request": request, "user": user})

@app.get("/ghostlayer-ai/view", response_class=HTMLResponse)
async def ghostlayer_view_page(request: Request):
    """Serve the GhostLayer AI document viewer page (requires authentication)"""
    user = require_auth(request)
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("ghostlayer_view.html", {"request": request, "user": user})

# OpenCV function to create black mask redaction
def draw_text_coordinates(image, coordinates_data):
    """Create black mask redaction on image using OpenCV"""
    import cv2
    import numpy as np
    
    # Create a copy of the image
    masked_image = image.copy()
    height, width = image.shape[:2]
    
    logger.info(f"Image dimensions: {width}x{height}")
    
    # Black color for masking (BGR format)
    mask_color = (0, 0, 0)  # Black
    
    try:
        # Process each page
        for page_idx, page in enumerate(coordinates_data.get('pages', [])):
            logger.info(f"Processing page {page_idx + 1}")
            blocks = page.get('blocks', [])
            logger.info(f"Found {len(blocks)} blocks in page {page_idx + 1}")
            
            # Mask blocks (black rectangles)
            for block_idx, block in enumerate(blocks):
                if 'coordinates' in block and len(block['coordinates']) >= 4:
                    coords = block['coordinates']
                    logger.info(f"Block {block_idx + 1} has {len(coords)} coordinates")
                    
                    # Convert normalized coordinates to pixel coordinates
                    points = []
                    for coord in coords:
                        x = int(coord['x'] * width)
                        y = int(coord['y'] * height)
                        points.append([x, y])
                        logger.info(f"Converted coord: {coord['x']:.3f}, {coord['y']:.3f} -> {x}, {y}")
                    
                    # Convert to numpy array for polygon drawing
                    points = np.array(points, np.int32)
                    
                    # Create black mask polygon
                    cv2.fillPoly(masked_image, [points], mask_color)
                    
                    logger.info(f"Applied black mask for block {block_idx + 1}")
                else:
                    logger.warning(f"Block {block_idx + 1} has invalid coordinates: {block.get('coordinates', [])}")
            
            # Mask paragraphs (black rectangles)
            for paragraph in page.get('paragraphs', []):
                if 'coordinates' in paragraph and len(paragraph['coordinates']) >= 4:
                    coords = paragraph['coordinates']
                    points = []
                    for coord in coords:
                        x = int(coord['x'] * width)
                        y = int(coord['y'] * height)
                        points.append([x, y])
                    
                    points = np.array(points, np.int32)
                    cv2.fillPoly(masked_image, [points], mask_color)
            
            # Mask tokens (black rectangles)
            for token in page.get('tokens', []):
                if 'coordinates' in token and len(token['coordinates']) >= 4:
                    coords = token['coordinates']
                    points = []
                    for coord in coords:
                        x = int(coord['x'] * width)
                        y = int(coord['y'] * height)
                        points.append([x, y])
                    
                    points = np.array(points, np.int32)
                    cv2.fillPoly(masked_image, [points], mask_color)
    
    except Exception as e:
        logger.error(f"Error creating black mask: {e}")
        # Return original image if masking fails
        return image
    
    return masked_image

# GhostLayer AI API Routes
@app.get("/api/ghostlayer/documents")
async def get_ghostlayer_documents(
    request: Request,
    page: int = 1, 
    limit: int = 10, 
    search: str = "", 
    status: str = ""
):
    """Get user-specific GhostLayer documents with pagination and filtering"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        user_id = user_data['id']
        offset = (page - 1) * limit
        
        # Get documents from new user-specific table
        documents = db.get_user_ghostlayer_documents(user_id=user_id, limit=limit, offset=offset)
        
        # Apply search filter if provided
        if search:
            documents = [doc for doc in documents if search.lower() in doc['document_name'].lower()]
        
        # Apply status filter if provided
        if status:
            documents = [doc for doc in documents if doc['processing_status'] == status]
        
        # Get total count for pagination
        all_docs = db.get_user_ghostlayer_documents(user_id=user_id, limit=1000, offset=0)
        total_count = len(all_docs)
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        start_item = offset + 1
        end_item = min(offset + limit, total_count)
        
        pagination = {
            "page": page,
            "pages": total_pages,
            "limit": limit,
            "total": total_count,
            "start": start_item,
            "end": end_item
        }
        
        return {
            "documents": documents,
            "pagination": pagination
        }
    except Exception as e:
        logger.error(f"Error fetching GhostLayer documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.get("/api/ghostlayer/stats")
async def get_ghostlayer_stats():
    """Get GhostLayer documents statistics"""
    try:
        stats = db.get_ghostlayer_stats()
        return stats
    except Exception as e:
        logger.error(f"Error fetching GhostLayer stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")

@app.get("/api/ai-documents")
async def get_user_ai_documents(request: Request, page: int = 1, limit: int = 10):
    """Get user's AI document classifications"""
    try:
        current_user = require_auth(request)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        offset = (page - 1) * limit
        
        # Get user's AI document classifications
        documents = db.get_ai_document_classifications(
            user_id=current_user['id'], 
            limit=limit, 
            offset=offset
        )
        
        # Get total count for pagination
        all_docs = db.get_ai_document_classifications(
            user_id=current_user['id'], 
            limit=1000, 
            offset=0
        )
        total_count = len(all_docs)
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        start_item = offset + 1
        end_item = min(offset + limit, total_count)
        
        pagination = {
            "page": page,
            "pages": total_pages,
            "limit": limit,
            "total": total_count,
            "start": start_item,
            "end": end_item
        }
        
        return {
            "documents": documents,
            "pagination": pagination
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user AI documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.get("/api/user-ghostlayer-documents")
async def get_user_ghostlayer_documents(request: Request, page: int = 1, limit: int = 10):
    """Get user-specific GhostLayer documents with pagination"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        user_id = user_data['id']
        offset = (page - 1) * limit
        
        # Get documents for the current user using new table
        documents = db.get_user_ghostlayer_documents(user_id=user_id, limit=limit, offset=offset)
        
        # Get total count for pagination
        all_docs = db.get_user_ghostlayer_documents(user_id=user_id, limit=1000, offset=0)
        total_count = len(all_docs)
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        start_item = offset + 1
        end_item = min(offset + limit, total_count)
        
        pagination = {
            "page": page,
            "pages": total_pages,
            "limit": limit,
            "total": total_count,
            "start": start_item,
            "end": end_item
        }
        
        return {
            "documents": documents,
            "pagination": pagination
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user GhostLayer documents: {e}")
        # Return empty result instead of error for now
        return {
            "documents": [],
            "pagination": {
                "page": 1,
                "pages": 1,
                "limit": limit,
                "total": 0,
                "start": 0,
                "end": 0
            }
        }

@app.post("/api/ghostlayer/upload")
async def upload_ghostlayer_document(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form("")
):
    """Upload a document for GhostLayer AI processing"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file format
        file_extension = os.path.splitext(file.filename)[1].lower()
        allowed_formats = ['.jpg', '.jpeg', '.png']
        
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Only JPEG, PNG, and JPG files are allowed. Received: {file_extension}"
            )
        
        # Get file info
        file_size = file.size
        file_format = file_extension[1:] if file_extension else "unknown"
        
        # Determine document type if not provided
        if not document_type:
            document_type = "Photo"
        
        # Create upload directory if it doesn't exist
        upload_dir = "upload_ghostlayer_docs"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Insert document record with user tracking
        document_data = {
            "user_id": user_data['id'],
            "uploaded_by": user_data['username'],
            "document_name": file.filename,
            "document_type": document_type,
            "document_format": file_format,
            "document_size": file_size,
            "document_path": file_path,
            "processing_status": "pending"
        }
        
        # Insert document record with user tracking using new table
        document_id = db.insert_user_ghostlayer_document(document_data)
        
        logger.info(f"User GhostLayer document uploaded: {file.filename} -> {file_path} by user {user_data['username']}")
        
        return {
            "message": "Document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading GhostLayer document: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@app.post("/api/ghostlayer/identify")
async def identify_document(
    request: Request,
    file: UploadFile = File(...)
):
    """Identify document type using Google Cloud Document AI and keyword classification"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file format
        file_extension = os.path.splitext(file.filename)[1].lower()
        allowed_formats = ['.jpg', '.jpeg', '.png']
        
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Only JPEG, PNG, and JPG files are allowed. Received: {file_extension}"
            )
        
        # Get file info
        file_size = file.size
        file_format = file_extension[1:] if file_extension else "unknown"
        
        # Create upload directory for GhostLayer docs
        upload_dir = "upload_ghostlayer_docs"
        try:
            os.makedirs(upload_dir, exist_ok=True)
            logger.info(f"Upload directory created/verified: {os.path.abspath(upload_dir)}")
        except Exception as e:
            logger.error(f"Error creating upload directory {upload_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create upload directory: {str(e)}")
        
        # For identify endpoint, we need to find the existing file path from the database
        # First, get user info to find their recent uploads
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        user_id = user_data['id']
        
        # Find the most recent pending document for this user
        existing_docs = db.get_user_ghostlayer_documents(user_id=user_id, limit=10, offset=0)
        pending_docs = [doc for doc in existing_docs if doc['processing_status'] == 'pending']
        
        if not pending_docs:
            raise HTTPException(status_code=404, detail="No pending documents found for processing")
        
        # Use the most recent pending document
        target_doc = pending_docs[0]
        file_path = target_doc['document_path']
        logger.info(f"Using existing file path: {file_path}")
        
        # Verify the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Get file info from existing record
        file_size = target_doc['document_size']
        file_format = target_doc['document_format']
        
        # Detect MIME type
        mime_type = detect_mime_type(file_path)
        
        # Setup GCP credentials and config
        try:
            project_id = setup_gcp_credentials("ghostlayer.json")
            config = load_processor_config("ghostlayer_ocr.ini")
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        
        # Process document with Document AI
        try:
            ai_result = await process_document_with_ai(
                project_id,
                config['location'],
                config['processor_id'],
                file_path,
                mime_type
            )
        except Exception as e:
            logger.error(f"Document AI processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Document AI processing failed: {str(e)}")
        
        # Extract classification results
        classification = ai_result.get('document_classification', {})
        document_type = classification.get('document_type', 'unknown')
        document_name = classification.get('document_name', 'Unknown Document')
        confidence_score = classification.get('confidence_score', 0.0)
        
        # Debug: Log classification details
        logger.info(f"Classification details: {classification}")
        logger.info(f"Document type: {document_type}, Name: {document_name}, Confidence: {confidence_score}")
        
        # Save coordinates JSON file
        coordinates_json_path = None
        try:
            # Create coordinates directory
            coordinates_dir = "upload_ghostlayer_docs/coordinates"
            os.makedirs(coordinates_dir, exist_ok=True)
            
            # Generate coordinates JSON filename using the existing file path
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            coordinates_filename = f"{base_filename}.json"
            coordinates_json_path = os.path.join(coordinates_dir, coordinates_filename)
            
            # Extract and save coordinates data
            coordinates_data = {
                "document_info": {
                    "original_filename": target_doc['document_name'],
                    "saved_filename": os.path.basename(file_path),
                    "document_type": document_name,
                    "confidence_score": confidence_score,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "full_text": ai_result.get('full_document_text', ''),
                "pages": []
            }
            
            # Extract coordinates from each page
            for page in ai_result.get('pages', []):
                page_data = {
                    "page_number": page.get('page_number', 1),
                    "blocks": page.get('blocks', []),
                    "paragraphs": page.get('paragraphs', []),
                    "tokens": page.get('tokens', [])
                }
                coordinates_data["pages"].append(page_data)
            
            # Save coordinates JSON file
            with open(coordinates_json_path, 'w', encoding='utf-8') as f:
                json.dump(coordinates_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Coordinates JSON saved: {coordinates_json_path}")
            
        except Exception as e:
            logger.error(f"Error saving coordinates JSON: {e}")
            # Continue processing even if coordinates save fails
        
        # Save to database
        document_data = {
            "document_name": file.filename,
            "document_type": document_name,
            "document_format": file_format.upper(),
            "document_size": file_size,
            "document_path": file_path,
            "coordinates_json_path": coordinates_json_path,
            "processing_status": "completed",
            "ai_analysis_result": ai_result
        }
        
        # Update the existing record in user_ghostlayer_documents table
        # We already have the target document from above
        document_id = target_doc['id']
        logger.info(f"Updating document ID: {document_id}")
        
        # Update the existing record with OCR results
        update_data = {
            "document_type": document_type,
            "coordinates_json_path": coordinates_json_path,
            "processing_status": "completed",
            "ai_analysis_result": str(ai_result)
        }
        db.update_user_ghostlayer_document(document_id, update_data)
        logger.info(f"Updated existing user GhostLayer document {document_id} with OCR results")
        
        logger.info(f"Document identified: {target_doc['document_name']} -> {document_name} (confidence: {confidence_score})")
        
        return {
            "status": "success",
            "document_id": document_id,
            "original_filename": target_doc['document_name'],
            "saved_filename": os.path.basename(file_path),
            "file_path": file_path,
            "document_type": document_type,
            "document_name": document_name,
            "confidence_score": confidence_score,
            "full_document_text": ai_result.get('full_document_text', ''),
            "classification_details": classification.get('classification_details', {}),
            "matched_keywords": classification.get('classification_details', {}).get('matched_keywords', []),
            "all_classification_matches": classification.get('all_matches', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error identifying document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to identify document: {str(e)}")

@app.delete("/api/ghostlayer/delete/{document_id}")
async def delete_ghostlayer_document(request: Request, document_id: int):
    """Delete a user's GhostLayer document"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get document info from user-specific table
        document = db.get_user_ghostlayer_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Verify the document belongs to the current user
        if document['user_id'] != user_data['id']:
            raise HTTPException(status_code=403, detail="Access denied - document belongs to another user")
        
        # Delete main document file from filesystem
        try:
            if os.path.exists(document['document_path']):
                os.remove(document['document_path'])
                logger.info(f"Deleted main document file: {document['document_path']}")
        except Exception as e:
            logger.warning(f"Failed to delete main document file {document['document_path']}: {e}")
        
        # Delete coordinates JSON file if it exists
        coordinates_path = document.get('coordinates_json_path')
        if coordinates_path:
            try:
                if os.path.exists(coordinates_path):
                    os.remove(coordinates_path)
                    logger.info(f"Deleted coordinates JSON file: {coordinates_path}")
            except Exception as e:
                logger.warning(f"Failed to delete coordinates JSON file {coordinates_path}: {e}")
        
        # Also try to delete coordinates file based on document filename pattern
        # This handles cases where coordinates_json_path might be None or incorrect
        try:
            document_filename = document.get('document_name', '')
            if document_filename:
                # Extract base filename without extension
                base_name = os.path.splitext(document_filename)[0]
                # Try different possible coordinate file patterns
                coordinates_dir = "upload_ghostlayer_docs/coordinates"
                possible_patterns = [
                    f"{base_name}.json",
                    f"*_{base_name}.json",
                    f"{base_name}_*.json"
                ]
                
                for pattern in possible_patterns:
                    import glob
                    matching_files = glob.glob(os.path.join(coordinates_dir, pattern))
                    for file_path in matching_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted coordinates file by pattern: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete coordinates file by pattern: {e}")
        
        # Delete from database
        if not db.delete_user_ghostlayer_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found in database")
        
        logger.info(f"GhostLayer document deleted: {document['document_name']} (ID: {document_id})")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting GhostLayer document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/api/ghostlayer/cleanup-orphaned-files")
async def cleanup_orphaned_coordinate_files(request: Request):
    """Clean up orphaned coordinate files that don't have corresponding database records"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        import glob
        import os
        
        # Get all coordinate files
        coordinates_dir = "upload_ghostlayer_docs/coordinates"
        if not os.path.exists(coordinates_dir):
            return {"message": "No coordinates directory found", "cleaned_files": []}
        
        coordinate_files = glob.glob(os.path.join(coordinates_dir, "*.json"))
        
        # Get all valid coordinate paths from database
        user_id = user_data['id']
        user_docs = db.get_user_ghostlayer_documents(user_id=user_id, limit=1000, offset=0)
        valid_coordinate_paths = set()
        
        for doc in user_docs:
            if doc.get('coordinates_json_path'):
                valid_coordinate_paths.add(doc['coordinates_json_path'])
        
        # Find orphaned files
        orphaned_files = []
        for file_path in coordinate_files:
            if file_path not in valid_coordinate_paths:
                orphaned_files.append(file_path)
        
        # Delete orphaned files
        cleaned_files = []
        for file_path in orphaned_files:
            try:
                os.remove(file_path)
                cleaned_files.append(file_path)
                logger.info(f"Cleaned up orphaned coordinate file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete orphaned file {file_path}: {e}")
        
        return {
            "message": f"Cleanup completed. Removed {len(cleaned_files)} orphaned files",
            "cleaned_files": cleaned_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up orphaned files: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup orphaned files")

@app.get("/api/ghostlayer/download/{document_id}")
async def download_ghostlayer_document(request: Request, document_id: int):
    """Download a user's GhostLayer document"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        document = db.get_user_ghostlayer_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Verify the document belongs to the current user
        if document['user_id'] != user_data['id']:
            raise HTTPException(status_code=403, detail="Access denied - document belongs to another user")
        
        if not os.path.exists(document['document_path']):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=document['document_path'],
            filename=document['document_name'],
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading GhostLayer document: {e}")
        raise HTTPException(status_code=500, detail="Failed to download document")

@app.get("/api/ghostlayer/coordinates/{document_id}")
async def download_ghostlayer_coordinates(request: Request, document_id: int):
    """Download coordinates JSON file for a user's GhostLayer document"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        document = db.get_user_ghostlayer_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Verify the document belongs to the current user
        if document['user_id'] != user_data['id']:
            raise HTTPException(status_code=403, detail="Access denied - document belongs to another user")
        
        coordinates_path = document.get('coordinates_json_path')
        if not coordinates_path or not os.path.exists(coordinates_path):
            raise HTTPException(status_code=404, detail="Coordinates file not found")
        
        # Generate filename for download
        coordinates_filename = f"coordinates_{document['document_name']}.json"
        
        return FileResponse(
            path=coordinates_path,
            filename=coordinates_filename,
            media_type='application/json'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading coordinates for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download coordinates: {str(e)}")

@app.get("/api/ghostlayer/documents/{document_id}")
async def get_ghostlayer_document(request: Request, document_id: int):
    """Get a specific user's GhostLayer document by ID"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        document = db.get_user_ghostlayer_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Verify the document belongs to the current user
        if document['user_id'] != user_data['id']:
            raise HTTPException(status_code=403, detail="Access denied - document belongs to another user")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {str(e)}")

@app.get("/api/ghostlayer/view/{document_id}")
async def get_ghostlayer_marked_image(request: Request, document_id: int):
    """Get marked image with text coordinates using OpenCV for a user's document"""
    try:
        # Get user from session
        user_data = require_auth(request)
        if not user_data:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        import cv2
        import numpy as np
        
        document = db.get_user_ghostlayer_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Verify the document belongs to the current user
        if document['user_id'] != user_data['id']:
            raise HTTPException(status_code=403, detail="Access denied - document belongs to another user")
        
        # Check if coordinates file exists
        coordinates_path = document.get('coordinates_json_path')
        if not coordinates_path or not os.path.exists(coordinates_path):
            raise HTTPException(status_code=404, detail="Coordinates file not found")
        
        # Load original image
        image_path = document['document_path']
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Original image not found")
        
        # Load image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=500, detail="Failed to load image")
        
        # Load coordinates JSON
        with open(coordinates_path, 'r', encoding='utf-8') as f:
            coordinates_data = json.load(f)
        
        logger.info(f"Processing document {document_id} with coordinates data structure:")
        logger.info(f"Pages count: {len(coordinates_data.get('pages', []))}")
        
        # Debug: Log first block coordinates
        if coordinates_data.get('pages'):
            first_page = coordinates_data['pages'][0]
            blocks = first_page.get('blocks', [])
            if blocks:
                logger.info(f"First block coordinates: {blocks[0].get('coordinates', [])[:2]}")
        
        # Draw coordinates on image
        marked_image = draw_text_coordinates(image, coordinates_data)
        
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', marked_image)
        
        return Response(
            content=buffer.tobytes(),
            media_type='image/jpeg',
            headers={'Content-Disposition': f'inline; filename="marked_{document["document_name"]}"'}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating marked image for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate marked image: {str(e)}")

@app.get("/api/ghostlayer/test-config")
async def test_ghostlayer_config():
    """Test GhostLayer configuration and credentials"""
    try:
        # Test GCP credentials
        project_id = setup_gcp_credentials("ghostlayer.json")
        
        # Test processor config
        config = load_processor_config("ghostlayer_ocr.ini")
        
        # Test document identification config
        doc_config = load_document_identification_config("document_identification.json")
        
        return {
            "status": "success",
            "gcp_project_id": project_id,
            "processor_config": config,
            "document_types_count": len(doc_config.get('document_types', {})),
            "classification_settings": doc_config.get('classification_settings', {})
        }
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration test failed: {str(e)}")

# Authentication Functions
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from session token"""
    token = credentials.credentials
    if token not in user_sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    user_data = user_sessions[token]
    return user_data

def require_auth(request: Request):
    """Check if user is authenticated"""
    # Check for token in Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        if token in user_sessions:
            logger.info(f"User authenticated via Authorization header: {user_sessions[token]['username']}")
            return user_sessions[token]
    
    # Check for token in cookies (for web requests)
    token = request.cookies.get("session_token")
    if token and token in user_sessions:
        logger.info(f"User authenticated via cookie: {user_sessions[token]['username']}")
        return user_sessions[token]
    
    # Session not found - this is normal after server restart
    logger.info(f"No valid session found. Available sessions: {len(user_sessions)}")
    return None

# Authentication Routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    logger.info(f"Login page accessed from: {request.headers.get('referer', 'direct')}")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/auth/login")
async def login(request: Request):
    """User login endpoint"""
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        
        # Authenticate user
        logger.info(f"Attempting authentication for username: {username}")
        user = db.authenticate_user(username, password)
        if not user:
            logger.warning(f"Authentication failed for username: {username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        logger.info(f"User {username} authenticated successfully")
        
        # Check if MFA is enabled and setup is complete
        logger.info(f"MFA check for user {username}: is_mfa_enabled={user.get('is_mfa_enabled')}, mfa_setup_complete={user.get('mfa_setup_complete')}")
        
        if user.get('is_mfa_enabled') and user.get('mfa_setup_complete'):
            # MFA is enabled and setup is complete - require MFA verification
            logger.info(f"MFA verification required for user {username}")
            return JSONResponse({
                "message": "MFA verification required",
                "mfa_required": True,
                "user_id": user['id']
            })
        elif user.get('is_mfa_enabled') and not user.get('mfa_setup_complete'):
            # MFA is enabled but setup is not complete - require MFA setup
            logger.info(f"MFA setup required for user {username}")
            return JSONResponse({
                "message": "MFA setup required",
                "mfa_setup_required": True,
                "user_id": user['id']
            })
        else:
            # No MFA required - check if password change is required (exclude admin)
            if not user.get('has_changed_default_password') and user.get('role') != 'admin':
                # Password change required for non-admin users
                logger.info(f"Password change required for user {username}")
                return JSONResponse({
                    "message": "Password change required",
                    "password_change_required": True,
                    "user_id": user['id']
                })
            else:
                # Normal login - proceed with session creation
                # Generate session token
                token = secrets.token_urlsafe(32)
                user_sessions[token] = user
                
                logger.info(f"User {username} logged in successfully. Token: {token[:10]}...")
                logger.info(f"Total active sessions: {len(user_sessions)}")
                
                # Create response with cookie
                response = JSONResponse({
                    "message": "Login successful",
                    "token": token,
                    "user": user
                })
                
                # Set session cookie
                response.set_cookie(
                    key="session_token",
                    value=token,
                    httponly=True,
                    secure=False,  # Set to True in production with HTTPS
                    samesite="lax"
                )
                
                return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/auth/logout")
async def logout(request: Request):
    """User logout endpoint"""
    try:
        # Get token from request
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            if token in user_sessions:
                del user_sessions[token]
        
        # Also check cookie
        token = request.cookies.get("session_token")
        if token and token in user_sessions:
            del user_sessions[token]
        
        # Create response and clear cookie
        response = JSONResponse({"message": "Logout successful"})
        response.delete_cookie("session_token")
        
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

@app.get("/api/auth/me")
async def get_current_user_info(request: Request):
    """Get current user information"""
    user = require_auth(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {"user": user}

# MFA Routes
@app.post("/api/auth/mfa/setup")
async def setup_mfa(request: Request):
    """Setup MFA for a user"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        # Generate MFA secret
        mfa_secret = MFAUtils.generate_mfa_secret()
        
        # Generate QR code
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        qr_path = MFAUtils.generate_qr_code(user['email'], mfa_secret)
        
        # Extract just the filename from the full path
        qr_filename = os.path.basename(qr_path)
        
        # Store the secret temporarily (will be saved when user confirms setup)
        user_sessions[f"mfa_setup_{user_id}"] = {
            'secret': mfa_secret,
            'qr_path': qr_path,
            'expires': datetime.now().timestamp() + 300  # 5 minutes
        }
        
        return {
            "message": "MFA setup initiated",
            "qr_path": qr_filename,  # Return just the filename
            "secret": mfa_secret  # For testing purposes
        }
        
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        raise HTTPException(status_code=500, detail="MFA setup failed")

@app.post("/api/auth/mfa/verify-setup")
async def verify_mfa_setup(request: Request):
    """Verify MFA setup with TOTP code"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        totp_code = data.get("totp_code")
        
        if not user_id or not totp_code:
            raise HTTPException(status_code=400, detail="User ID and TOTP code required")
        
        # Get the temporary MFA setup data
        setup_data = user_sessions.get(f"mfa_setup_{user_id}")
        if not setup_data:
            raise HTTPException(status_code=400, detail="MFA setup session expired")
        
        # Verify the TOTP code
        if MFAUtils.verify_totp_code(setup_data['secret'], totp_code):
            # Save the MFA secret to database
            logger.info(f"Saving MFA setup for user {user_id}")
            db.setup_mfa(user_id, setup_data['secret'])
            
            # Verify the save worked
            user = db.get_user_by_id(user_id)
            logger.info(f"MFA save verification for user {user_id}: is_mfa_enabled={user.get('is_mfa_enabled')}, mfa_setup_complete={user.get('mfa_setup_complete')}")
            
            # Clean up QR code
            MFAUtils.cleanup_qr_code(setup_data['qr_path'])
            
            # Remove temporary session
            del user_sessions[f"mfa_setup_{user_id}"]
            
            return {"message": "MFA setup completed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid TOTP code")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(status_code=500, detail="MFA verification failed")

@app.post("/api/auth/mfa/verify")
async def verify_mfa(request: Request):
    """Verify MFA code during login"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        totp_code = data.get("totp_code")
        
        if not user_id or not totp_code:
            raise HTTPException(status_code=400, detail="User ID and TOTP code required")
        
        # Get user's MFA secret
        mfa_status = db.get_user_mfa_status(user_id)
        if not mfa_status or not mfa_status.get('mfa_secret'):
            raise HTTPException(status_code=400, detail="MFA not configured for user")
        
        # Verify the TOTP code
        if MFAUtils.verify_totp_code(mfa_status['mfa_secret'], totp_code):
            return {"message": "MFA verification successful"}
        else:
            raise HTTPException(status_code=400, detail="Invalid TOTP code")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(status_code=500, detail="MFA verification failed")

@app.post("/api/auth/mfa/disable")
async def disable_mfa(request: Request):
    """Disable MFA for a user (Admin only)"""
    try:
        current_user = require_auth(request)
        if not current_user or current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        data = await request.json()
        user_id = data.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        if db.disable_mfa(user_id):
            return {"message": "MFA disabled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to disable MFA")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA disable error: {e}")
        raise HTTPException(status_code=500, detail="MFA disable failed")

@app.post("/api/auth/mfa/complete-login")
async def complete_mfa_login(request: Request):
    """Complete login after MFA verification"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        totp_code = data.get("totp_code")
        
        if not user_id or not totp_code:
            raise HTTPException(status_code=400, detail="User ID and TOTP code required")
        
        # Get user's MFA secret
        mfa_status = db.get_user_mfa_status(user_id)
        logger.info(f"MFA status for user {user_id}: {mfa_status}")
        if not mfa_status or not mfa_status.get('mfa_secret'):
            logger.warning(f"MFA not configured for user {user_id}: {mfa_status}")
            raise HTTPException(status_code=400, detail="MFA not configured for user")
        
        # Verify the TOTP code
        if MFAUtils.verify_totp_code(mfa_status['mfa_secret'], totp_code):
            # Get user details
            user = db.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Generate session token
            token = secrets.token_urlsafe(32)
            user_sessions[token] = user
            
            logger.info(f"User {user['username']} completed MFA login successfully. Token: {token[:10]}...")
            logger.info(f"Total active sessions: {len(user_sessions)}")
            
            # Create response with cookie
            response = JSONResponse({
                "message": "MFA login successful",
                "token": token,
                "user": user
            })
            
            # Set session cookie
            response.set_cookie(
                key="session_token",
                value=token,
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite="lax"
            )
            
            return response
        else:
            raise HTTPException(status_code=400, detail="Invalid TOTP code")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA complete login error: {e}")
        raise HTTPException(status_code=500, detail="MFA login failed")

@app.get("/api/auth/mfa/qr/{filename}")
async def get_mfa_qr_code(filename: str):
    """Serve MFA QR code images"""
    try:
        qr_path = f"temp/{filename}"
        if os.path.exists(qr_path):
            return FileResponse(qr_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="QR code not found")
    except Exception as e:
        logger.error(f"Error serving QR code {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error serving QR code")

@app.post("/api/auth/mfa/cleanup")
async def cleanup_mfa_setup(request: Request):
    """Clean up MFA setup (remove QR code and reset MFA state)"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        
        logger.info(f"MFA cleanup called for user {user_id}")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        # Get user from database
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Clean up QR code file if it exists
        email = user.get('email', '')
        if email:
            qr_filename = f"mfa_qr_{email.replace('@', '_').replace('.', '_')}.png"
            qr_path = f"temp/{qr_filename}"
            if os.path.exists(qr_path):
                os.remove(qr_path)
                logger.info(f"Cleaned up QR code: {qr_path}")
        
        # Reset MFA state in database
        db.disable_mfa(user_id)
        
        return {"message": "MFA setup cleaned up successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA cleanup error: {e}")
        raise HTTPException(status_code=500, detail="MFA cleanup failed")

@app.post("/api/auth/change-password-initial")
async def change_password_initial(request: Request):
    """Change user password for first-time login (no authentication required)"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        new_password = data.get("new_password")
        
        if not user_id or not new_password:
            raise HTTPException(status_code=400, detail="User ID and new password required")
        
        if len(new_password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
        
        # Update password and mark as changed
        success = db.update_password_changed(user_id, new_password)
        
        if success:
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to change password")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(status_code=500, detail="Password change failed")

@app.post("/api/auth/change-password")
async def change_password(request: Request):
    """Change user password"""
    try:
        current_user = require_auth(request)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        data = await request.json()
        current_password = data.get("current_password")
        new_password = data.get("new_password")
        
        if not current_password or not new_password:
            raise HTTPException(status_code=400, detail="Current and new password required")
        
        # Verify current password
        import hashlib
        import sqlite3
        current_password_hash = hashlib.sha256(current_password.encode()).hexdigest()
        
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT password_hash FROM users WHERE id = ?
            """, (current_user['id'],))
            
            stored_hash = cursor.fetchone()
            if not stored_hash or stored_hash[0] != current_password_hash:
                raise HTTPException(status_code=400, detail="Current password is incorrect")
            
            # Update password
            new_password_hash = hashlib.sha256(new_password.encode()).hexdigest()
            cursor.execute("""
                UPDATE users 
                SET password_hash = ?, has_changed_default_password = 1, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (new_password_hash, current_user['id']))
            conn.commit()
            
            return {"message": "Password changed successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            raise HTTPException(status_code=500, detail="Failed to change password")
        finally:
            conn.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {e}")
        raise HTTPException(status_code=500, detail="Change password failed")

# User Management Routes (Admin only)
@app.get("/api/users")
async def get_users(request: Request):
    """Get all users (Admin only)"""
    current_user = require_auth(request)
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = db.get_all_users()
    # Filter out the current admin user from the list
    filtered_users = [user for user in users if user.get("id") != current_user.get("id")]
    return {"users": filtered_users}

@app.post("/api/users")
async def create_user(request: Request):
    """Create new user (Admin only)"""
    user = require_auth(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        data = await request.json()
        success = db.create_user(
            username=data.get("username"),
            password=data.get("password"),
            full_name=data.get("full_name"),
            email=data.get("email"),
            role=data.get("role", "analyst"),
            is_active=data.get("is_active", True),
            is_mfa_enabled=data.get("mfa_enabled", False),
            created_by=user.get("id")
        )
        
        if success:
            return {"message": "User created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create user")
            
    except Exception as e:
        logger.error(f"Create user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.put("/api/users/{user_id}")
async def update_user(user_id: int, request: Request):
    """Update user (Admin only)"""
    user = require_auth(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        data = await request.json()
        success = db.update_user(user_id, **data)
        
        if success:
            return {"success": True, "message": "User updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update user")
            
    except Exception as e:
        logger.error(f"Update user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, request: Request):
    """Delete user (Admin only)"""
    user = require_auth(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        success = db.delete_user(user_id)
        
        if success:
            return {"message": "User deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to delete user")
            
    except Exception as e:
        logger.error(f"Delete user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")