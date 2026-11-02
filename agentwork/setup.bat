@echo off
setlocal
cd /d "%~dp0"
set "BOOTSTRAP_PYTHON="
if exist ".venv-ml-validation\Scripts\python.exe" set "BOOTSTRAP_PYTHON=%CD%\.venv-ml-validation\Scripts\python.exe"
if not defined BOOTSTRAP_PYTHON (
  where py >nul 2>&1
  if not errorlevel 1 for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)"') do set "BOOTSTRAP_PYTHON=%%P"
)
if not defined BOOTSTRAP_PYTHON (
  echo Python 3.12 is required. Install Python 3.12 or create .venv-ml-validation first.
  exit /b 1
)
if not exist .venv\Scripts\python.exe "%BOOTSTRAP_PYTHON%" -m venv .venv
call .venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import flask, cv2, numpy, tensorflow; from smartpark.backend.database import ParkingDatabase; from smartpark.config import DATABASE_PATH; ParkingDatabase(DATABASE_PATH); print('SmartPark dependencies and database are ready.')"
echo SmartPark setup complete.
