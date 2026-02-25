#!/usr/bin/env python3
# -------------------------------
# chatbot_streamlit.py - Streamlit Entry Point for CellAtria
# -------------------------------

import os
import sys
import argparse

# -------------------------------

HELP_TEXT = """
cellAtria - Agentic Triage of Regulated single-cell data Ingestion and Analysis
Version: 1.0.0

Usage:
  streamlit run agent/chatbot_streamlit.py -- --env_path /path/to/projects

Options:
  --env_path     Path to directory containing .env file (default: /data)
  --version, -v  Show version information
"""

# Parse arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--version", "-v", action="store_true")
parser.add_argument("--env_path", type=str, default="/data")
args, unknown = parser.parse_known_args()

if args.version:
    print(HELP_TEXT)
    sys.exit(0)

# Set environment variable for base_streamlit.py to access
os.environ["CELLATRIA_ENV_PATH"] = args.env_path

# -------------------------------
# Launch Streamlit with base_streamlit.py, inheriting the environment
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_streamlit.py")
os.execvp("streamlit", [
    "streamlit", "run", script,
    "--server.address=0.0.0.0",
    "--server.port=7860",
    "--browser.gatherUsageStats=false"
])