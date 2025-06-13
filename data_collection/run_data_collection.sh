#!/bin/bash

# Data Collection Runner Script
# Usage: ./run_data_collection.sh [sessions] [turns] [concurrent] [endpoint]

set -e

SESSIONS=${1:-5}
TURNS=${2:-3}
CONCURRENT=${3:-3}
ENDPOINT=${4:-"simple-chat"}
OUTPUT_DIR="data_collection_logs"

echo "🚀 AI Teacher-Student Data Collection Runner"
echo "============================================="
echo "📊 Sessions: $SESSIONS"
echo "🔄 Turns per session: $TURNS"
echo "⚡ Max concurrent: $CONCURRENT"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🔗 Endpoint: $ENDPOINT"
echo ""

# Check if server is running
echo "🔍 Checking if server is running..."
if ! curl -s http://localhost:8000/docs > /dev/null; then
    echo "❌ Server is not running on http://localhost:8000"
    echo "Please start the server first with:"
    echo "   uvicorn server.main:app --reload"
    exit 1
fi
echo "✅ Server is running!"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run data collection with concurrent sessions
echo "🎯 Starting batch data collection..."
python batch_data_collection.py \
    --sessions "$SESSIONS" \
    --turns "$TURNS" \
    --concurrent "$CONCURRENT" \
    --delay 1 \
    --output "$OUTPUT_DIR" \
    --endpoint "$ENDPOINT"

echo ""
echo "📋 Collection Summary:"
echo "   Log files saved to: $OUTPUT_DIR/"
echo "   Check batch_summary_*.json for detailed statistics"

# Optional: Show a quick summary
if command -v jq &> /dev/null; then
    LATEST_SUMMARY=$(ls -t "$OUTPUT_DIR"/batch_summary_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_SUMMARY" ]; then
        echo ""
        echo "📊 Quick Stats from latest run:"
        echo "   Total sessions: $(jq -r '.total_sessions' "$LATEST_SUMMARY")"
        echo "   Successful: $(jq -r '.successful_sessions' "$LATEST_SUMMARY")"
        echo "   Success rate: $(jq -r '.success_rate * 100 | floor')%"
        echo "   Total messages: $(jq -r '.total_messages' "$LATEST_SUMMARY")"
    fi
fi

echo ""
echo "🎉 Data collection completed!"