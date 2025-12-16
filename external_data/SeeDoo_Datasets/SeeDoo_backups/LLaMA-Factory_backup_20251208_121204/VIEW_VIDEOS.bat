@echo off
echo ========================================
echo    VIDEO VIEWER - HTTP Server
echo ========================================
echo.
echo Starting HTTP server on port 8000...
echo.
echo This window must stay open for videos to play!
echo.
echo TO VIEW VIDEOS:
echo Open your browser and go to:
echo.
echo    http://localhost:8000/test_videos.html
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
python -m http.server 8000
pause
