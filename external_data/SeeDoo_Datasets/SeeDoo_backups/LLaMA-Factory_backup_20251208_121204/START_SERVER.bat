@echo off
echo ======================================================================
echo Starting Web Server for Video Review
echo ======================================================================
echo.
echo Server will start at: http://localhost:8000
echo.
echo Opening browser in 3 seconds...
echo Press Ctrl+C to stop the server when done
echo ======================================================================
echo.

timeout /t 3 /nobreak >nul
start http://localhost:8000/final_working_report.html

python -m http.server 8000
