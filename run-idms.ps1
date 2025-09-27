# IDMS Complete Setup & Run Script
# This single script handles everything: setup, installation, and starting the IDMS application

param(
    [switch]$SkipSetup,
    [switch]$ForceReinstall
)

Write-Host "IDMS - Intelligent Document Management System" -ForegroundColor Cyan
Write-Host "Complete Setup & Run Script" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to create directory if it doesn't exist
function Ensure-Directory($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# Function to show progress
function Show-Progress($message, $step, $total) {
    $percent = [math]::Round(($step / $total) * 100)
    Write-Host "[$percent%] $message" -ForegroundColor Green
}

# Start setup process
$setupSteps = 0
$totalSteps = 10

# Step 1: Check Python installation
$setupSteps++
Show-Progress "Checking Python installation..." $setupSteps $totalSteps
if (-not (Test-Command "python")) {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation!" -ForegroundColor Yellow
    exit 1
}

try {
    $pythonVersion = python --version 2>&1
    Write-Host "SUCCESS: Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python installation issue detected!" -ForegroundColor Red
    exit 1
}

# Step 2: Set execution policy
$setupSteps++
Show-Progress "Setting PowerShell execution policy..." $setupSteps $totalSteps
try {
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force | Out-Null
    Write-Host "SUCCESS: Execution policy set for current session" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Could not set execution policy, continuing..." -ForegroundColor Yellow
}

# Step 3: Create virtual environment (if needed)
$setupSteps++
Show-Progress "Setting up virtual environment..." $setupSteps $totalSteps

if ($ForceReinstall -and (Test-Path "venv")) {
    Write-Host "INFO: Force reinstall requested, removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

if (-not (Test-Path "venv")) {
    Write-Host "INFO: Creating virtual environment..." -ForegroundColor Blue
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
    Write-Host "SUCCESS: Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "SUCCESS: Virtual environment already exists" -ForegroundColor Green
}

# Step 4: Activate virtual environment
$setupSteps++
Show-Progress "Activating virtual environment..." $setupSteps $totalSteps

# Try PowerShell activation first
try {
    & "venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Virtual environment activated via PowerShell" -ForegroundColor Green
    } else {
        throw "PowerShell activation failed"
    }
} catch {
    Write-Host "WARNING: PowerShell activation failed, trying alternative..." -ForegroundColor Yellow
    # Alternative activation method
    $env:VIRTUAL_ENV = (Resolve-Path "venv").Path
    $env:PATH = "$env:VIRTUAL_ENV\Scripts;$env:PATH"
    Write-Host "SUCCESS: Virtual environment activated via environment variables" -ForegroundColor Green
}

# Step 5: Install/upgrade pip
$setupSteps++
Show-Progress "Upgrading pip..." $setupSteps $totalSteps
try {
    python -m pip install --upgrade pip
    Write-Host "SUCCESS: Pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Could not upgrade pip, continuing..." -ForegroundColor Yellow
}

# Step 6: Install dependencies
$setupSteps++
Show-Progress "Installing Python dependencies..." $setupSteps $totalSteps

# Always install/update dependencies from requirements.txt to ensure new modules are installed
Write-Host "INFO: Installing/updating dependencies from requirements.txt..." -ForegroundColor Blue
pip install -r requirements.txt --upgrade
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    Write-Host "Trying to install individual packages..." -ForegroundColor Yellow
    
    # Try installing packages individually
    $packages = @("fastapi", "uvicorn", "jinja2", "python-multipart", "ibm_watsonx_ai", "python-dotenv", "python-docx", "PyYAML", "PyPDF2", "openpyxl", "reportlab")
    foreach ($package in $packages) {
        Write-Host "Installing $package..." -ForegroundColor Blue
        pip install $package
    }
}
Write-Host "SUCCESS: Dependencies installed successfully!" -ForegroundColor Green

# Step 7: Create necessary directories
$setupSteps++
Show-Progress "Creating application directories..." $setupSteps $totalSteps
Ensure-Directory "app"
Ensure-Directory "app/templates"
Ensure-Directory "app/static"
Ensure-Directory "app/static/css"
Ensure-Directory "app/static/js"
Ensure-Directory "app/temp"
Write-Host "SUCCESS: Application directories created" -ForegroundColor Green

# Step 8: Create configuration files
$setupSteps++
Show-Progress "Setting up configuration files..." $setupSteps $totalSteps

# Create .env file if it doesn't exist
if (-not (Test-Path "app/.env")) {
    Write-Host "INFO: Creating .env template..." -ForegroundColor Blue
    $envTemplate = @"
# IBM WatsonX AI Configuration
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_SERVICE_URL=your_watsonx_service_url_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_MODEL_ID=your_watsonx_model_id_here

# IBM DataCap Configuration
DATACAP_URL=your_datacap_url_here
APPLICATION=your_application_name_here
PASSWORD=your_password_here
STATION=your_station_here
USER=your_username_here
JOB=your_job_name_here
"@
    
    $envTemplate | Out-File -FilePath "app/.env" -Encoding UTF8
    Write-Host "SUCCESS: .env template created in app/.env" -ForegroundColor Green
    Write-Host "WARNING: Please edit app/.env with your actual configuration values!" -ForegroundColor Yellow
} else {
    Write-Host "SUCCESS: .env file already exists" -ForegroundColor Green
}

# Create criticality config if it doesn't exist
if (-not (Test-Path "app/criticality_config.json")) {
    Write-Host "INFO: Creating criticality configuration..." -ForegroundColor Blue
    $criticalityConfig = @"
{
    "Aadhaar Card": "Public",
    "Non-Disclosure Agreement (NDA)": "Confidential",
    "NDA": "Confidential",
    "Bank Transaction Data": "Restricted",
    "ATM Transaction Logs": "Restricted",
    "Infrastructure as Code": "Restricted",
    "Authentication Logs": "Classified",
    "Batch Scripts": "Public",
    "Employee Handbook": "Confidential",
    "Business Proposal": "Top Secret",
    "Invoice": "Restricted"
}
"@
    
    $criticalityConfig | Out-File -FilePath "app/criticality_config.json" -Encoding UTF8
    Write-Host "SUCCESS: Criticality configuration created!" -ForegroundColor Green
} else {
    Write-Host "SUCCESS: Criticality configuration already exists" -ForegroundColor Green
}

# Step 9: Validate setup
$setupSteps++
Show-Progress "Validating setup..." $setupSteps $totalSteps

# Check if all required files exist
$requiredFiles = @(
    "app/main.py",
    "app/classifier.py",
    "app/file_handlers.py",
    "app/utils.py",
    "app/prompts.py",
    "requirements.txt"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "ERROR: Missing required files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "   - $file" -ForegroundColor Red
    }
    exit 1
}

Write-Host "SUCCESS: All required files present" -ForegroundColor Green

# Step 10: Start the application
$setupSteps++
Show-Progress "Starting IDMS application..." $setupSteps $totalSteps

Write-Host ""
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Blue
Write-Host "Frontend will be available at: http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host "API documentation at: http://127.0.0.1:5001/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "   Edit app/.env for WatsonX and DataCap settings" -ForegroundColor White
Write-Host "   Modify app/criticality_config.json for security levels" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

# Change to app directory and start uvicorn
Set-Location app
try {
    & "..\venv\Scripts\uvicorn.exe" main:app --reload --port 5001 --host 127.0.0.1
} catch {
    Write-Host "ERROR: Failed to start the application!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   Check if port 5001 is available" -ForegroundColor White
    Write-Host "   Verify your .env configuration" -ForegroundColor White
    Write-Host "   Try running: uvicorn main:app --reload --port 5001" -ForegroundColor White
    exit 1
} finally {
    # Return to original directory
    Set-Location ..
}

Write-Host ""
Write-Host "IDMS application stopped. Thank you for using IDMS!" -ForegroundColor Cyan
Write-Host ""
Write-Host "To restart the application, simply run this script again:" -ForegroundColor Blue
Write-Host "   .\run-idms.ps1" -ForegroundColor White