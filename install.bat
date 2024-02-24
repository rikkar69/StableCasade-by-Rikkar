@echo off
echo Creating 'outputs' directory...
if not exist "outputs" mkdir outputs

echo Creating virtual environment...
python -m venv csd

echo Activating virtual environment...
call csd\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Setup completed successfully.
pause
