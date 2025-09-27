# 🚀 IDMS Quick Start Guide

## One-Click Setup & Run

### **Option 1: PowerShell (Recommended)**
```powershell
.\run-idms.ps1
```

### **Option 2: Batch File (Alternative)**
```cmd
run-idms.bat
```

## What the Script Does

The `run-idms.ps1` script automatically:

1. ✅ **Checks Python installation**
2. ✅ **Creates virtual environment** (if needed)
3. ✅ **Installs all dependencies**
4. ✅ **Creates configuration files**
5. ✅ **Validates setup**
6. ✅ **Starts the application**

## Access Your Application

Once the script completes:

- **🌐 Frontend UI:** http://127.0.0.1:5001
- **📚 API Documentation:** http://127.0.0.1:5001/docs

## Configuration

After first run, edit these files:

- **`app/.env`** - Add your WatsonX and DataCap credentials
- **`app/criticality_config.json`** - Customize document security levels

## Script Options

### PowerShell Script Options:
```powershell
# Normal run (setup + start)
.\run-idms.ps1

# Skip setup (if already configured)
.\run-idms.ps1 -SkipSetup

# Force reinstall dependencies
.\run-idms.ps1 -ForceReinstall
```

## Troubleshooting

### Python Not Found
- Install Python 3.8+ from https://python.org
- Make sure "Add Python to PATH" is checked during installation

### Port Already in Use
- Change port in the script: `--port 5002`
- Or stop other applications using port 5001

### Permission Issues
- Run PowerShell as Administrator
- Or use the batch file version: `run-idms.bat`

## Features

- **🎨 Beautiful Web Interface** - Modern, responsive design
- **🤖 AI Document Classification** - IBM WatsonX integration
- **🔒 Security Levels** - Automatic criticality assignment
- **📁 FileNet Integration** - Automatic upload to IBM FileNet
- **📊 Real-time Results** - Live processing status
- **📱 Mobile Friendly** - Works on all devices

## File Structure

```
idms-v1/
├── run-idms.ps1          # Main PowerShell script
├── run-idms.bat          # Batch file alternative
├── setup-idms.ps1        # Setup only (legacy)
├── start-idms.ps1        # Start only (legacy)
├── requirements.txt      # Python dependencies
└── app/
    ├── main.py           # FastAPI application
    ├── .env              # Configuration (created by script)
    ├── templates/        # HTML templates
    ├── static/           # CSS/JS files
    └── criticality_config.json  # Security levels
```

## Support

- **Documentation:** http://127.0.0.1:5001/docs
- **Issues:** Check console output for error messages
- **Configuration:** Edit `app/.env` for API credentials

---

**🎉 That's it! Your IDMS application is ready to use!**
