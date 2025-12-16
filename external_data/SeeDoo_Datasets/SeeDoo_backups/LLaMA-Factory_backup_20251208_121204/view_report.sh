#!/bin/bash
# Helper script to view the report with working videos

REPORT_FILE="${1:-FINAL_fixed_report.html}"

echo "======================================================================"
echo "🎬 Starting local web server for video playback..."
echo "======================================================================"
echo "Report: $REPORT_FILE"
echo "Server: http://localhost:8000"
echo ""
echo "Opening browser in 2 seconds..."
echo "Press Ctrl+C to stop the server when done"
echo "======================================================================"
echo ""

# Wait 2 seconds then open browser
sleep 2 && explorer.exe "http://localhost:8000/$REPORT_FILE" &

# Start HTTP server
python3 -m http.server 8000
