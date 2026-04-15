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

# Open browser after 3 seconds
(sleep 3 && open "http://localhost:5050") &

# Run Flask server
python3 app.py
