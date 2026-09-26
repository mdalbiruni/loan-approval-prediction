@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup complete.
echo Put loan_approval_dataset.csv inside the data folder.
pause
