@echo off
echo ==============================================
echo AMINQT CatPaw Monitor Setup
echo ==============================================
echo.
echo Step 1: Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not installed or not in PATH
    pause & exit /b 1
)
echo OK: Python installed
echo.
echo Step 2: Installing required packages...
pip install psutil requests >nul 2>&1
if errorlevel 1 (
    echo WARNING: pip install failed, trying pip3...
    pip3 install psutil requests >nul 2>&1
    if errorlevel 1 (echo ERROR: Failed to install packages & pause & exit /b 1)
)
echo OK: Packages installed
echo.
echo Step 3: Creating Windows Task Scheduler job...
set /p createTask="Create Windows Task Scheduler job? (Y/N): "
if /i "%createTask%"=="Y" (
    python monitor_catpaw.py --create-task
    echo Register (Admin PowerShell): schtasks /create /xml "catpaw_monitor_task.xml" /tn "AMINQTMonitor"
)
echo.
echo Step 4: Testing monitor (one-time health check)...
python monitor_catpaw.py --once --config catpaw_monitor_config.json
echo.
echo ==============================================
echo SETUP COMPLETE
echo ==============================================
echo Commands:
echo   Start:    python monitor_catpaw.py
echo   Status:   python monitor_catpaw.py --status
echo   Check:    python monitor_catpaw.py --check
echo   Task XML: python monitor_catpaw.py --create-task
echo.
echo Config: catpaw_monitor_config.json
echo Log:    catpaw_monitor.log
pause
