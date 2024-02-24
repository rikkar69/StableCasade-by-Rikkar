@echo off
echo Activating the virtual environment...
call csd\Scripts\activate

echo Starting the app...CTRL+Click the link to launch app
python app.py

echo Opening the web browser...
start http://localhost:7860/

echo Done.
