#!/bin/bash
# Double-click this file to start the SPX Trading Dashboard
cd "$(dirname "$0")"

# Kill any existing instance
lsof -ti:5050 2>/dev/null | xargs kill -9 2>/dev/null

echo ""
echo "================================================"
echo "  SPX Trading Dashboard"
echo "  Starting server on http://localhost:5050"
echo "================================================"
echo ""

# Open in Chrome after 3 seconds (Safari blocks localhost on macOS)
(sleep 3 && open -a "Google Chrome" "http://127.0.0.1:5050") &

# Run Flask server
python3 app.py
