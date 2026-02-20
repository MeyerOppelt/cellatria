#!/bin/bash
# ============================================================================
# CellAtria Streamlit Launcher
# ============================================================================
# Quick launcher script for the Streamlit version of CellAtria
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENV_PATH="${ENV_PATH:-.}"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-path)
            ENV_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --address)
            ADDRESS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-path PATH    Path to directory containing .env file (default: current directory)"
            echo "  --port PORT        Server port (default: 8501)"
            echo "  --address ADDR     Server address (default: 0.0.0.0)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --env-path /path/to/projects"
            echo "  $0 --env-path . --port 8502"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert to absolute path if relative
ENV_PATH=$(cd "$ENV_PATH" 2>/dev/null && pwd || echo "$ENV_PATH")

# Print banner
echo ""
echo "============================================================================"
echo -e "${BLUE}🧬 CellAtria - Streamlit Version${NC}"
echo "============================================================================"
echo ""

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Error: Streamlit is not installed${NC}"
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_DIR="$SCRIPT_DIR/agent"
BASE_FILE="$AGENT_DIR/base_streamlit.py"

# Check if base_streamlit.py exists
if [ ! -f "$BASE_FILE" ]; then
    echo -e "${RED}❌ Error: base_streamlit.py not found at $BASE_FILE${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$ENV_PATH/.env" ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found at $ENV_PATH/.env${NC}"
    echo "Please ensure your environment configuration file exists."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Export environment variables
export CELLATRIA_ENV_PATH="$ENV_PATH"
export PYTHONPATH="$AGENT_DIR:$PYTHONPATH"

# Print configuration
echo -e "${GREEN}✅ Configuration:${NC}"
echo "   📁 Project Path: $ENV_PATH"
echo "   🌐 Server Address: $ADDRESS"
echo "   🔌 Port: $PORT"
echo "   📂 Agent Directory: $AGENT_DIR"
echo ""

# Launch Streamlit
echo -e "${BLUE}🚀 Launching CellAtria Streamlit interface...${NC}"
echo "============================================================================"
echo ""
echo -e "${GREEN}Access the application at:${NC}"
echo "   👉 http://localhost:$PORT"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run Streamlit
streamlit run "$BASE_FILE" \
    --server.address="$ADDRESS" \
    --server.port="$PORT" \
    --browser.gatherUsageStats=false \
    --server.headless=true \
    --theme.base="light" \
    --theme.primaryColor="#1976d2"

# Cleanup on exit
echo ""
echo -e "${GREEN}✅ CellAtria stopped${NC}"
