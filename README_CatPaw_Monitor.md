# AMINQT CatPaw Process Monitor

> Adapted from EWM Robot project `README_CatPaw_Monitor.md`
> Project: A股图形因子量化交易系统 (AMINQT)

## Overview

Monitors AMINQT processes (FastAPI + Streamlit) and auto-restarts on failure.

| Component | Port | Health URL |
|-----------|------|------------|
| FastAPI (uvicorn) | 8000 | `http://localhost:8000/docs` |
| Streamlit | 8501 | `http://localhost:8501` |

## Quick Start

```batch
cd "D:\AMINQT\AMINQT CODES"
setup_catpaw_monitor.bat
```

Or manual:
```batch
pip install psutil requests
python monitor_catpaw.py
```

## Usage

| Command | Description |
|---------|-------------|
| `python monitor_catpaw.py` | Continuous monitoring |
| `python monitor_catpaw.py --once` | One-time health check |
| `python monitor_catpaw.py --status` | Status report |
| `python monitor_catpaw.py --check` | Comprehensive health check |
| `python monitor_catpaw.py --create-task` | Create Windows Task Scheduler XML |

## Configuration

Edit `catpaw_monitor_config.json` to customize processes, ports, restart limits.

## Windows Task Scheduler

```powershell
# As Administrator
schtasks /create /xml "catpaw_monitor_task.xml" /tn "AMINQTMonitor"
```

## Log Files

- `catpaw_monitor.log` — Monitor logs (10MB rotation, 3 backups)
- `catpaw_alerts.log` — Alert notifications

## Features

- Process monitoring (PID tracking)
- HTTP health checks
- Auto-restart with configurable delays
- Restart limits (per hour) to prevent crash loops
- Windows Task Scheduler integration
- Log rotation
