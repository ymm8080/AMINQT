@echo off
chcp 65001 >nul
echo 正在停止看板 (守护进程 + streamlit)...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('pythonw.exe','python.exe') -and $_.CommandLine -match 'dashboard_watchdog|streamlit run app/streamlit_app' } | ForEach-Object { Write-Host ('已停止 PID ' + $_.ProcessId + ' : ' + $_.CommandLine); Stop-Process -Id $_.ProcessId -Force }"
echo 已停止。
echo (注意: 看板守护会在 10 分钟内自动重新拉起 — 这就是"常驻"的设计)
echo 若想彻底关闭: 打开"任务计划程序"禁用 AMINQT-Dashboard-HB, 并删除启动文件夹里的"启动看板"快捷方式。
pause
