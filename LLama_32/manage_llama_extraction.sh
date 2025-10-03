#!/bin/bash
# Helper script to manage the Llama extraction background process

OUTPUT_DIR="${1:-./llama_output}"
PID_FILE="$OUTPUT_DIR/llama_extraction.pid"
LOG_FILE="$OUTPUT_DIR/llama_extraction_background.log"
EXTRACTION_LOG="$OUTPUT_DIR/llama_extraction.log"

show_usage() {
    echo "Usage: $0 [output_dir] [command]"
    echo ""
    echo "Commands:"
    echo "  status     - Check if process is running"
    echo "  logs       - Tail the logs (Ctrl+C to exit)"
    echo "  stop       - Stop the background process"
    echo "  checkpoints - List saved checkpoint files"
    echo "  stats      - Show extraction statistics"
    echo ""
    echo "Default output_dir: ./llama_output"
}

check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ Process not running (no PID file found)"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Process is running (PID: $PID)"
        echo ""
        echo "Running since:"
        ps -p "$PID" -o lstart=
        echo ""
        echo "CPU/Memory usage:"
        ps -p "$PID" -o pid,pcpu,pmem,etime,cmd
        return 0
    else
        echo "❌ Process not running (PID $PID is dead)"
        return 1
    fi
}

tail_logs() {
    if [ -f "$EXTRACTION_LOG" ]; then
        echo "📋 Tailing extraction logs (Ctrl+C to exit)..."
        echo ""
        tail -f "$EXTRACTION_LOG"
    elif [ -f "$LOG_FILE" ]; then
        echo "📋 Tailing background logs (Ctrl+C to exit)..."
        echo ""
        tail -f "$LOG_FILE"
    else
        echo "❌ No log files found"
    fi
}

stop_process() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ No PID file found"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping process (PID: $PID)..."
        kill "$PID"
        sleep 2

        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Process still running, forcing..."
            kill -9 "$PID"
        fi

        rm "$PID_FILE"
        echo "✅ Process stopped"
    else
        echo "❌ Process not running"
        rm "$PID_FILE"
    fi
}

show_checkpoints() {
    echo "📦 Checkpoint files:"
    echo ""
    if ls "$OUTPUT_DIR"/*.h5 2>/dev/null | head -n 1 > /dev/null; then
        ls -lh "$OUTPUT_DIR"/*.h5 | awk '{print $9, "  ", $5, "  ", $6, $7, $8}'
        echo ""
        echo "Total checkpoints: $(ls "$OUTPUT_DIR"/*.h5 2>/dev/null | wc -l)"
    else
        echo "No checkpoint files found yet"
    fi
}

show_stats() {
    echo "📊 Extraction Statistics:"
    echo ""

    if [ -f "$EXTRACTION_LOG" ]; then
        # Count successful extractions
        SUCCESS=$(grep -c "Successfully extracted embeddings" "$EXTRACTION_LOG" 2>/dev/null || echo "0")
        # Count failures
        FAILED=$(grep -c "Failed to process" "$EXTRACTION_LOG" 2>/dev/null || echo "0")
        # Get latest progress line
        LATEST_PROGRESS=$(grep "Progress:" "$EXTRACTION_LOG" 2>/dev/null | tail -1)
        # Get latest GPU memory
        LATEST_GPU=$(grep "GPU Memory:" "$EXTRACTION_LOG" 2>/dev/null | tail -1)

        echo "Successful extractions: $SUCCESS"
        echo "Failed extractions: $FAILED"
        echo ""
        if [ ! -z "$LATEST_PROGRESS" ]; then
            echo "Latest progress:"
            echo "$LATEST_PROGRESS"
            echo ""
        fi
        if [ ! -z "$LATEST_GPU" ]; then
            echo "Latest GPU status:"
            echo "$LATEST_GPU"
            echo ""
        fi
    else
        echo "No extraction log found yet"
    fi

    show_checkpoints
}

# Main
COMMAND="${2:-status}"

case "$COMMAND" in
    status)
        check_status
        ;;
    logs)
        tail_logs
        ;;
    stop)
        stop_process
        ;;
    checkpoints)
        show_checkpoints
        ;;
    stats)
        show_stats
        ;;
    help)
        show_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
