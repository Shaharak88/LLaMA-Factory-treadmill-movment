@echo off
REM Generate Local Video Report - Windows Batch File
REM This script generates an HTML report with working video playback for Windows

echo ========================================
echo VIDEO REPORT GENERATOR
echo ========================================
echo.

REM Find the most recent metadata CSV in the data directory
for /f "delims=" %%i in ('dir /b /o-d data\*_metadata_*.csv 2^>nul') do (
    set "LATEST_CSV=%%i"
    goto :found
)

:notfound
echo ERROR: No metadata CSV files found in data\ directory
echo.
echo Please run complete_dataset_review.py first to extract metadata
pause
exit /b 1

:found
echo Found metadata: data\%LATEST_CSV%
echo.

REM Generate report using Python
python generate_local_video_report.py "data\%LATEST_CSV%"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate report
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.

REM Find the generated HTML file
for /f "delims=" %%i in ('dir /b /o-d *_local_report.html 2^>nul') do (
    set "REPORT_HTML=%%i"
    goto :open
)

:open
if defined REPORT_HTML (
    echo Opening report in browser...
    start "" "%REPORT_HTML%"
) else (
    echo Report generated! Look for *_local_report.html
)

echo.
pause
