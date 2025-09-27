@echo off
echo 🚀 IDMS - Intelligent Document Management System
echo Complete Setup ^& Run Script
echo ============================================================

REM Check if Python is installed
echo [10%%] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check 'Add Python to PATH' during installation!
    pause
    exit /b 1
)
echo ✅ Python found

REM Create virtual environment if it doesn't exist
echo [30%%] Setting up virtual environment...
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created!
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo [40%%] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✅ Virtual environment activated!

REM Upgrade pip
echo [50%%] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install dependencies
echo [60%%] Installing Python dependencies...
if not exist "venv\Lib\site-packages\fastapi" (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies!
        echo Trying individual package installation...
        pip install fastapi uvicorn jinja2 python-multipart ibm_watsonx_ai python-dotenv python-docx PyYAML PyPDF2 openpyxl
    )
    echo ✅ Dependencies installed successfully!
) else (
    echo ✅ Dependencies already installed
)

REM Create directories
echo [70%%] Creating application directories...
if not exist "app\templates" mkdir app\templates
if not exist "app\static\css" mkdir app\static\css
if not exist "app\static\js" mkdir app\static\js
if not exist "app\temp" mkdir app\temp
echo ✅ Application directories created

REM Create .env file if it doesn't exist
echo [80%%] Setting up configuration files...
if not exist "app\.env" (
    echo 📝 Creating .env template...
    (
        echo # IBM WatsonX AI Configuration
        echo WATSONX_API_KEY=your_watsonx_api_key_here
        echo WATSONX_SERVICE_URL=your_watsonx_service_url_here
        echo WATSONX_PROJECT_ID=your_watsonx_project_id_here
        echo WATSONX_MODEL_ID=your_watsonx_model_id_here
        echo.
        echo # IBM DataCap Configuration
        echo DATACAP_URL=your_datacap_url_here
        echo APPLICATION=your_application_name_here
        echo PASSWORD=your_password_here
        echo STATION=your_station_here
        echo USER=your_username_here
        echo JOB=your_job_name_here
    ) > "app\.env"
    echo ✅ .env template created
    echo ⚠️  Please edit app\.env with your actual configuration values!
) else (
    echo ✅ .env file already exists
)

REM Create criticality config if it doesn't exist
if not exist "app\criticality_config.json" (
    echo 📝 Creating criticality configuration...
    (
        echo {
        echo     "Aadhaar Card": "Public",
        echo     "Non-Disclosure Agreement ^(NDA^)": "Confidential",
        echo     "NDA": "Confidential",
        echo     "Bank Transaction Data": "Restricted",
        echo     "ATM Transaction Logs": "Restricted",
        echo     "Infrastructure as Code": "Restricted",
        echo     "Authentication Logs": "Classified",
        echo     "Batch Scripts": "Public",
        echo     "Employee Handbook": "Confidential",
        echo     "Business Proposal": "Top Secret",
        echo     "Invoice": "Restricted"
        echo }
    ) > "app\criticality_config.json"
    echo ✅ Criticality configuration created!
) else (
    echo ✅ Criticality configuration already exists
)

REM Validate setup
echo [90%%] Validating setup...
if not exist "app\main.py" (
    echo ❌ Missing app\main.py
    pause
    exit /b 1
)
if not exist "app\classifier.py" (
    echo ❌ Missing app\classifier.py
    pause
    exit /b 1
)
if not exist "app\file_handlers.py" (
    echo ❌ Missing app\file_handlers.py
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo ❌ Missing requirements.txt
    pause
    exit /b 1
)
echo ✅ All required files present

REM Start the application
echo [100%%] Starting IDMS application...
echo.
echo 🎉 Setup completed successfully!
echo.
echo 🌐 Starting FastAPI server...
echo 📱 Frontend will be available at: http://127.0.0.1:5001
echo 📚 API documentation at: http://127.0.0.1:5001/docs
echo.
echo 🔧 Configuration:
echo    • Edit app\.env for WatsonX and DataCap settings
echo    • Modify app\criticality_config.json for security levels
echo.
echo Press Ctrl+C to stop the server
echo ============================================================

REM Change to app directory and start uvicorn
cd app
uvicorn main:app --reload --port 5001 --host 127.0.0.1
if errorlevel 1 (
    echo ❌ Failed to start the application!
    echo.
    echo 🔧 Troubleshooting:
    echo    • Check if port 5001 is available
    echo    • Verify your .env configuration
    echo    • Try running: uvicorn main:app --reload --port 5001
    pause
    exit /b 1
)

cd ..
echo.
echo 👋 IDMS application stopped. Thank you for using IDMS!
echo.
echo 💡 To restart the application, simply run this script again:
echo    run-idms.bat
pause
