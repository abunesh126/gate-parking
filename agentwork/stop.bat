@echo off
setlocal
cd /d "%~dp0"
if not exist smartpark.pid (
  echo SmartPark is not running.
  exit /b 0
)
set /p SMARTPARK_PID=<smartpark.pid
for /f "delims=0123456789" %%A in ("%SMARTPARK_PID%") do (
  echo Invalid SmartPark PID file.
  del smartpark.pid
  exit /b 1
)
taskkill /PID %SMARTPARK_PID% /T /F >nul 2>&1
if errorlevel 1 echo SmartPark process was already stopped.
del smartpark.pid
echo SmartPark stopped.
