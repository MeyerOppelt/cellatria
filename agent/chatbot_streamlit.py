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

print("\n")
print("=" * 60)
print("✅ CellAtria Streamlit version initializing...")
print(f"📍 Environment path: {args.env_path}")
print("=" * 60)