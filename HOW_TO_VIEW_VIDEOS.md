# How to View Video Reports

## Quick Start (Easiest Method)

### Option 1: Double-Click Method (Windows)
1. Double-click `VIEW_VIDEOS.bat`
2. A terminal window will open showing "Starting HTTP server..."
3. Open your browser and go to: `http://localhost:8000/test_videos.html`
4. Videos should now play! ✅
5. **IMPORTANT:** Keep the terminal window open while viewing videos

### Option 2: Manual Method (Any OS)
```bash
# Step 1: Open terminal in this directory
cd /mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory

# Step 2: Start HTTP server
python -m http.server 8000

# Step 3: Open browser to:
http://localhost:8000/test_videos.html
```

## Why Do I Need an HTTP Server?

Modern browsers block local video file playback for security (CORS policy).
The HTTP server is a simple, standard solution that makes browsers trust the files.

## Available Reports

- **test_videos.html** - Simple test with 4 sample videos (recommended to test first)
- **video_report.html** - Full dataset report (if generated)
- **my_report.html** - Custom report (if generated)

To view any report, replace `test_videos.html` with the desired filename:
```
http://localhost:8000/your_report_name.html
```

## Troubleshooting

### Videos Won't Play
1. Make sure HTTP server is running (terminal window is open)
2. Check you're using `http://localhost:8000/...` not `file:///...`
3. Try refreshing the browser (Ctrl+F5)

### Port 8000 Already in Use
```bash
# Use a different port
python -m http.server 8001

# Then open:
http://localhost:8001/test_videos.html
```

### Server Won't Start
```bash
# Stop any existing servers
pkill -f "python.*http.server"

# Try again
python -m http.server 8000
```

## Testing Your Setup

1. Start the server
2. Open: `http://localhost:8000/test_videos.html`
3. You should see a green success message: "✅ Success! All 4 videos loaded and ready to play."
4. Click play on any video - it should play smoothly

If you see errors, check the instructions above.

## Creating New Reports

Use the all-in-one review tool:
```bash
python3 review_dataset.py DATASET_NAME --output my_new_report.html
```

Then view it:
```bash
# Server should already be running, just open:
http://localhost:8000/my_new_report.html
```
