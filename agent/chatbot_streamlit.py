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
  --env_path     Path to directory containing .env file (default: /mnt/work/projects)
  --version, -v  Show version information
"""

# Parse arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--version", "-v", action="store_true")
parser.add_argument("--env_path", type=str, default="/mnt/work/projects")
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
print("📍 The application will open in your browser automatically.")
print("   If not, copy and paste the URL shown below.")
print("=" * 60)
print("\n")

# -------------------------------
# Launch is handled by Streamlit's run command
# The actual app is in base_streamlit.py
# -------------------------------
