@echo off
cd /d "D:\AMINQT\AMINQT CODES"
set PY=C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe
echo [%date% %time%] start gen
"%PY%" -u scripts\_gen_legacy_list.py 20260902
if errorlevel 1 goto :fail
echo [%date% %time%] start deliver
"%PY%" -u scripts\_deliver_legacy_list.py 20260902
if errorlevel 1 goto :fail
echo [%date% %time%] start ths_push
"%PY%" -u scripts\_ths_watchlist_push.py 20260902
if errorlevel 1 goto :fail
echo [%date% %time%] ALL DONE
exit /b 0
:fail
echo [%date% %time%] FAILED rc=%errorlevel%
exit /b 1
