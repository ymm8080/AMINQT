@echo off
chcp 65001 >nul
cd /d "D:\AMINQT\AMINQT CODES"
set "PYW=pythonw"
where pythonw >nul 2>&1 || set "PYW=C:\Users\91454\AppData\Local\Programs\Python\Python312\pythonw.exe"
echo 正在启动看板守护进程 (无窗口后台运行)...
start "" "%PYW%" scripts\dashboard_watchdog.py
echo 已启动。浏览器访问: http://localhost:8501
echo 首次打开约需 10-20 秒。
timeout /t 4 /nobreak >nul
