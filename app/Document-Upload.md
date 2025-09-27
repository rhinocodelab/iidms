# Document Upload - AI Classification Process

This document explains the complete flow that occurs behind the scenes when the "Process with AI Classification" button is clicked in the IDMS application.

## Overview

The IDMS (Intelligent Document Management System) provides an AI-powered document classification and FileNet integration system. When users upload documents through the web interface, the system automatically classifies them using IBM WatsonX AI and stores them in IBM FileNet with appropriate criticality levels.

## Complete Process Flow

### 1. Frontend Form Submission

- **Form Action**: The upload form submits to `/upload_files` endpoint with `POST` method
- **File Selection**: Users can select multiple files with supported extensions
- **JavaScript Enhancement**: 
  - Button becomes disabled during processing
  - Shows "Processing..." text with spinner animation
  - Files are sent to the backend for processing

### 2. Backend Processing (`/upload_files` endpoint)

The `upload_files_frontend` function in `main.py` handles the incoming request:

#### Step 1: File Processing
- Calls `process_uploaded_files(files)` function
- Creates temporary directory (`./temp`) to store uploaded files
- For each uploaded file:
  - Saves file to temporary location using `save_uploaded_file()`
  - Determines if it's an archive (zip, 7z, tar, etc.) or single file

#### Step 2: AI Classification Process
For each file, the system performs the following:

- **Configuration Loading**: Loads criticality configuration from `criticality_config.json`
- **File Handler**: Calls `handle_file()` from `file_handlers.py` which:
  - Reads file content using `utils.read_file()`
  - Loads existing categories from `existing_categories.txt`
  - Calls AI classifier based on file type:
    - **Images**: Uses `call_llm_image()` with WatsonX AI
    - **Text/Documents**: Uses `call_llm_text()` with WatsonX AI
  - Returns classification result with document type, tags, summary, and reasoning

#### Step 3: WatsonX AI Classification
The `classifier.py` module handles AI processing:

- **Model Initialization**: Uses IBM WatsonX AI foundation models
- **API Communication**: Sends content + prompt to WatsonX API
- **Response Processing**: Extracts JSON response with classification results
- **Structured Output**: Returns data including:
  - Document type classification
  - Relevant tags
  - Content summary
  - Classification reasoning

#### Step 4: Criticality Assignment
- **Mapping**: Maps classified document type to criticality level using `criticality_config.json`
- **Category Management**: Adds new categories to `existing_categories.txt` if they don't exist
- **Security Levels**: Assigns appropriate criticality levels (Public, Confidential, Restricted, Top Secret, Classified)

#### Step 5: FileNet Upload
- **Integration**: Attempts to upload file to IBM FileNet using Java CLI
- **Tool**: Uses `FileNetUpload.jar` with document type and criticality level
- **Status Tracking**: Logs success/failure of FileNet upload
- **Error Handling**: Captures and reports any upload failures

#### Step 6: Database Storage
- **Persistence**: Saves processing results to SQLite database via `data_manager.save_document_processing()`
- **Metadata Recording**: Records:
  - File metadata (name, size, type, checksum)
  - Processing results (document type, criticality, tags, summary)
  - Processing timestamps (start and end times)
  - FileNet upload status
  - Error logs if any issues occur

#### Step 7: Cleanup
- **Temporary Files**: Removes temporary files from `./temp` directory
- **Memory Management**: Cleans up processing artifacts
- **Response**: Returns results to frontend

### 3. Response Handling

- **Success Path**: Renders `results.html` template with processing results
- **Error Path**: Renders `error.html` template with error message
- **Results Display**: Shows classification details for each processed file including:
  - Document type
  - Criticality level
  - Tags and summary
  - Processing status
  - FileNet upload status

## Key Components

### WatsonX AI Integration
- **Model**: IBM's foundation model for document classification
- **API**: RESTful API communication with IBM WatsonX service
- **Configuration**: Environment variables for API key, service URL, project ID, and model ID

### FileNet Integration
- **Enterprise CMS**: IBM's enterprise content management system
- **Java CLI**: Uses `FileNetUpload.jar` for file uploads
- **Batch Processing**: Creates batches and manages upload queues

### Database System
- **SQLite**: Local database for storing processing history and analytics
- **Schema**: Includes tables for documents, processing logs, and error tracking
- **Analytics**: Supports dashboard metrics and reporting

### Configuration Management
- **Criticality Config**: Maps document types to security levels
- **Category Management**: Maintains list of known document types
- **Environment Variables**: Stores API keys and service configurations

## Supported File Types

The system supports a wide range of file formats:

### Documents
- **PDF**: Portable Document Format
- **DOCX/DOC**: Microsoft Word documents
- **ODT**: OpenDocument Text format

### Spreadsheets
- **XLSX/XLS**: Microsoft Excel spreadsheets
- **CSV**: Comma-separated values

### Images
- **PNG**: Portable Network Graphics
- **JPG/JPEG**: Joint Photographic Experts Group

### Archives
- **ZIP**: ZIP compressed archives
- **7Z**: 7-Zip compressed archives
- **TAR**: Tape Archive format
- **GZ/BZ2/XZ**: Various compression formats
- **RAR**: RAR compressed archives

### Code and Data
- **TXT**: Plain text files
- **JSON**: JavaScript Object Notation
- **YAML/YML**: YAML Ain't Markup Language

## Error Handling

The system includes comprehensive error handling:

- **File Processing Errors**: Catches and logs file reading/parsing errors
- **AI Classification Errors**: Handles WatsonX API failures gracefully
- **FileNet Upload Errors**: Captures and reports upload failures
- **Database Errors**: Manages database connection and query failures
- **System Errors**: General exception handling with detailed logging

## Performance Considerations

- **Async Processing**: Uses asynchronous operations where possible
- **Batch Processing**: Handles multiple files efficiently
- **Memory Management**: Cleans up temporary files and resources
- **Timeout Handling**: Implements appropriate timeouts for external API calls
- **Error Recovery**: Continues processing other files if one fails

## Security Features

- **Criticality Levels**: Automatic assignment based on document type
- **Access Control**: Integration with enterprise security systems
- **Audit Trail**: Complete logging of all processing activities
- **Data Validation**: Input validation and sanitization
- **Secure Storage**: Enterprise-grade FileNet storage

## Monitoring and Analytics

- **Processing Metrics**: Tracks processing times and success rates
- **Error Tracking**: Monitors and reports system errors
- **Usage Analytics**: Provides insights into document processing patterns
- **System Status**: Real-time monitoring of AI and FileNet services
- **Dashboard Integration**: Visual representation of system performance

## Business Document Upload to DataCap Process

### Overview

The "Upload to DataCap" button provides a direct path for business documents to be uploaded to IBM DataCap without AI classification. This process is designed for documents that already have a known type and require immediate processing through the DataCap workflow.

### Complete Process Flow

#### 1. Frontend Form Submission

- **Form Action**: The business upload form submits to `/business_upload` endpoint with `POST` method
- **File Selection**: Users select a single business document (PDF, DOCX, ODT, CSV, XLSX, XLS, PNG, JPG, JPEG)
- **Confidentiality Level**: Users choose from predefined confidentiality levels:
  - Public
  - Confidential
  - Restricted
  - Top Secret
  - Classified
- **JavaScript Enhancement**: 
  - Button becomes disabled during processing
  - Shows "Uploading..." text with spinner animation

#### 2. Backend Processing (`/business_upload` endpoint)

The `business_upload_frontend` function in `main.py` handles the business document upload:

#### Step 1: Session Initialization
- Creates a `requests.Session()` for maintaining connection state
- Prepares for DataCap API communication

#### Step 2: DataCap Logon
- **Endpoint**: `{DATACAP_URL}/Session/Logon`
- **Method**: POST with XML payload
- **Payload Structure**:
  ```xml
  <LogonProperties>
      <application>{APPLICATION}</application>
      <password>{PASSWORD}</password>
      <station>{STATION}</station>
      <user>{USER}</user>
  </LogonProperties>
  ```
- **Authentication**: Uses environment variables for credentials
- **Response**: Establishes authenticated session with DataCap

#### Step 3: Create Batch
- **Endpoint**: `{DATACAP_URL}/Queue/CreateBatch`
- **Method**: POST with XML payload
- **Payload Structure**:
  ```xml
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
  ```
- **Purpose**: Creates a new processing batch in DataCap
- **Response**: Returns XML with `queueId` for the created batch

#### Step 4: Extract Queue ID
- **XML Parsing**: Uses `extract_queue_id()` function to parse XML response
- **Extraction**: Finds `<queueId>` element in the response
- **Validation**: Ensures queue ID is successfully extracted
- **Error Handling**: Raises HTTPException if parsing fails

#### Step 5: Upload File
- **Endpoint**: `{DATACAP_URL}/Queue/UploadFile/watsonxai/{queueId}`
- **Method**: POST with multipart/form-data
- **File Data**: 
  - Filename: Original file name
  - Content: File binary data
  - Content-Type: MIME type of the file
- **Session**: Uses established authenticated session
- **Purpose**: Uploads the actual document file to the created batch

#### Step 6: Release Batch
- **Endpoint**: `{DATACAP_URL}/Queue/ReleaseBatch/watsonxai/{queueId}/finished`
- **Method**: PUT
- **Purpose**: Marks the batch as complete and ready for processing
- **Workflow**: Triggers the DataCap processing workflow

#### Step 7: Response Handling
- **Success Path**: Renders `success.html` template with:
  - Success message
  - Queue ID for tracking
  - Original filename
  - Selected confidentiality level
- **Error Path**: Renders `error.html` template with error details

### Key Differences from AI Classification Process

| Aspect | AI Classification | DataCap Upload |
|--------|------------------|----------------|
| **Files** | Multiple files supported | Single file only |
| **AI Processing** | WatsonX AI classification | No AI processing |
| **Criticality** | Auto-assigned based on AI | User-selected |
| **Storage** | FileNet via Java CLI | Direct DataCap upload |
| **Workflow** | Extract → Classify → Upload | Direct upload to queue |
| **Processing Time** | Longer (AI processing) | Faster (direct upload) |

### DataCap Integration Details

#### Environment Variables Required
- `DATACAP_URL`: Base URL of the DataCap service
- `APPLICATION`: DataCap application name
- `PASSWORD`: Authentication password
- `STATION`: Processing station identifier
- `USER`: Authenticated user
- `JOB`: Job type (Navigator Job)

#### DataCap Workflow
1. **Authentication**: Establishes secure session
2. **Batch Creation**: Creates processing batch with metadata
3. **File Upload**: Uploads document to specific batch
4. **Release**: Marks batch ready for DataCap processing
5. **Processing**: DataCap handles document processing workflow

#### Supported File Types for DataCap
- **Documents**: PDF, DOCX, DOC, ODT
- **Spreadsheets**: XLSX, XLS, CSV
- **Images**: PNG, JPG, JPEG

### Error Handling

The DataCap upload process includes comprehensive error handling:

- **Authentication Errors**: Failed logon attempts
- **Batch Creation Errors**: Issues creating processing batches
- **File Upload Errors**: Problems uploading files to DataCap
- **Release Errors**: Issues marking batches as complete
- **Network Errors**: Connection and timeout issues
- **XML Parsing Errors**: Malformed DataCap responses

### Security Features

- **Session Management**: Maintains authenticated session throughout process
- **Confidentiality Levels**: User-selectable security classifications
- **Secure Transmission**: HTTPS communication with DataCap
- **Credential Management**: Environment-based credential storage

### Monitoring and Tracking

- **Queue ID**: Unique identifier for tracking batch processing
- **Status Updates**: Real-time feedback on upload progress
- **Error Logging**: Comprehensive logging of all DataCap interactions
- **Audit Trail**: Complete record of business document uploads

### Use Cases

The DataCap upload process is ideal for:

- **Known Document Types**: Documents with established classifications
- **Immediate Processing**: Documents requiring fast DataCap workflow processing
- **Business Workflows**: Standard business document processing
- **High-Volume Uploads**: Single documents that don't require AI analysis
- **Confidential Documents**: Documents with specific confidentiality requirements

## Conclusion

The IDMS document upload and AI classification process provides a robust, enterprise-ready solution for automated document processing. It combines the power of IBM WatsonX AI for intelligent classification with IBM FileNet for secure enterprise storage, all while maintaining comprehensive logging, error handling, and analytics capabilities.

Additionally, the system offers a streamlined DataCap upload process for business documents that require direct processing without AI classification, providing organizations with flexible options for different document processing needs.

The system is designed to be scalable, maintainable, and user-friendly, providing organizations with an efficient way to process and manage their document workflows with both AI-powered intelligence and direct enterprise system integration.
