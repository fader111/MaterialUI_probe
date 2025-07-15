@echo off
REM Start the backend server
start "Backend Server" cmd /k "uvicorn server.api:app --reload"

REM Start the frontend development server
start "Frontend Server" cmd /k "npm run dev"

@REM REM Exit the batch script
exit