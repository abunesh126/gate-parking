@echo off
title SmartPark
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  exit /b 1
)
if exist smartpark.pid (
  echo SmartPark may already be running. Run stop.bat first.
  exit /b 1
)
for /f "delims=" %%P in ('powershell -NoProfile -Command "$p = Start-Process -FilePath '%CD%\.venv\Scripts\python.exe' -ArgumentList '-m','smartpark.app' -WorkingDirectory '%CD%' -PassThru; $p.Id"') do echo %%P>smartpark.pid
echo SmartPark started at http://127.0.0.1:5000
