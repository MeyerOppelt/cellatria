# utils_streamlit.py
# -------------------------------

import os
import re
import json
import uuid
import subprocess
import threading
import time
import tempfile
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic  
from dotenv import load_dotenv
import google.genai as genai

# -------------------------------

base_path = os.path.dirname(os.path.abspath(__file__))

# -------------------------------

def get_llm_from_env(path_to_env):
    load_dotenv(dotenv_path=os.path.join(path_to_env, ".env"), override=True)
    provider = os.getenv("PROVIDER", "").strip()

    if provider == "Azure OpenAI":
        return AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature = os.getenv("AZURE_OPENAI_TEMPERATURE"),
            top_p = os.getenv("AZURE_OPENAI_TOP_P"),
            streaming=True
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True
        )
    elif provider == "Anthropic":
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=True
        )
    elif provider == "Google":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_name = os.getenv("GOOGLE_MODEL", "gemini-pro")
        return genai.GenerativeModel(model_name)
    elif provider == "Local":
        return ChatOpenAI(
            model_name=os.getenv("LOCAL_MODEL_NAME", "local-model"),
            openai_api_base=os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"),
            openai_api_key=os.getenv("LOCAL_API_KEY", "not-needed"),
            temperature=float(os.getenv("LOCAL_TEMPERATURE", "0.7")),
            streaming=True
        )
    else:
        supported = ["Azure OpenAI", "OpenAI", "Anthropic", "Google", "Local"]
        raise ValueError(
            f"Invalid or unsupported PROVIDER in .env: '{provider}'\n"
            f"Supported providers: {', '.join(supported)}"
        )

# -------------------------------

def store_metadata_json(metadata: dict, project_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
    try:
        if project_dir is None:
            return "❌ project_dir is required"
        if filename is None:
            filename = json_filename()
        os.makedirs(project_dir, exist_ok=True)
        out_path = os.path.join(project_dir, filename)
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return f"✅ Metadata saved to `{out_path}`"
    except Exception as e:
        return f"❌ Failed to save metadata: {str(e)}"

# -------------------------------

def store_metadata_csv(metadata: dict, project_dir: str, filename: Optional[str] = None) -> str:
    try:
        if filename is None:
            filename = csv_filename()
        os.makedirs(project_dir, exist_ok=True)
        df = pd.DataFrame(metadata)
        df.dropna(axis=1, how="all", inplace=True)
        df = df.loc[:, df.apply(lambda col: any(str(cell).strip() != "" for cell in col))]
        out_path = os.path.join(project_dir, filename)
        df.to_csv(out_path, index=False)
        return f"✅ Metadata saved to `{out_path}`"
    except Exception as e:
        return f"❌ Failed to save metadata: {str(e)}"

# -------------------------------

def csv_filename(filename: Optional[str] = None) -> str:
    if filename:
        if not filename.endswith(".csv"):
            filename += ".csv"
    else:
        uid = str(uuid.uuid4())[:8]
        filename = f"metadata_{uid}.csv"
    return filename

# -------------------------------

def json_filename(filename: Optional[str] = None) -> str:
    if filename:
        if not filename.endswith(".json"):
            filename += ".json"
    else:
        uid = str(uuid.uuid4())[:8]
        filename = f"metadata_{uid}.json"
    return filename

# -------------------------------

def txt_filename(filename: Optional[str] = None) -> str:
    if filename:
        if not filename.endswith(".txt"):
            filename += ".txt"
    else:
        uid = str(uuid.uuid4())[:8]
        filename = f"metadata_{uid}.txt"
    return filename

# -------------------------------

def convert_none(value):
    """Convert a string 'None' (case-insensitive) to Python None."""
    return None if isinstance(value, str) and value.strip().lower() == "none" else value

# -------------------------------

def parse_vars(vars_str):
    """Parse a comma-separated string into a list of clean column names."""
    if vars_str is None or vars_str.strip() == "":
        return []
    return [col.strip() for col in vars_str.split(",") if col.strip()]

# -------------------------------

# Configuration
LOG_PATH = "/tmp/cellatria_log.txt"

def log_status(message: str):
    """Append a message to the log file and session state if available"""
    import streamlit as st
    with open(LOG_PATH, "a") as f:
        f.write(f"{message}\n")
    try:
        if hasattr(st, 'session_state') and 'log_lines' in st.session_state:
            st.session_state.log_lines.append(message)
    except Exception:
        pass

def read_log():
    """Read the current log file contents"""
    try:
        with open(LOG_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "📁 No logs yet."

# -------------------------------

# Terminal Session
class TerminalSession:
    def __init__(self):
        self.process = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.lock = threading.Lock()
        self.latest_output = ""
        self.history = ""  
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        for line in self.process.stdout:
            with self.lock:
                self.latest_output += line

    def run_command(self, cmd):
        if self.process.poll() is not None:
            return "❌ Shell has exited."
        with self.lock:
            self.latest_output = ""
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        time.sleep(0.3)
        with self.lock:
            result = self.latest_output.strip()
        return result

# -------------------------------

# Strip ANSI codes for clean display
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def clean_ansi(text):
    """Remove ANSI escape codes from text"""
    return ansi_escape.sub('', text)

# -------------------------------

def export_chat_history(state_data):
    """Export chat history as JSON string"""
    return json.dumps(state_data, indent=2)

# -------------------------------

def export_llm_metadata(llm_meta):
    """Export LLM metadata as JSON string"""
    return json.dumps(llm_meta, indent=2)

# -------------------------------

# Initial message
initial_message = {
    "role": "assistant",
    "content": (
        "👋 Hello! I'm **cellAtria**, your assistant for analyzing single-cell RNA-seq (scRNA-seq) datasets.\n\n"
        "Here's what I can help you with:\n"
        "1. Extract structured metadata from scientific articles (PDF or URL).\n"
        "2. Store and organize metadata in structured project directories.\n"
        "3. Access public databases (currently support GEO) and fetch associated sample metadata.\n"
        "4. Download and organize scRNA-seq datasets.\n"        
        "5. Trigger CellExpress standardized single-cell data processing.\n\n"
        "To see a list of all available actions, type `'help'`.\n"
        "**Let's see how far we can fly together.** 🕊️\n\n"
        "To get started, let's set up your working directory. I can show you the current one or help create a new path. You can change your working directory anytime you wish.\n"
        f"📂 **Current Working Directory:** `{os.getcwd()}`"
    )
}

# -------------------------------

def get_llm_metadata(llm):
    """
    Return a provider-agnostic dict describing the LLM that was constructed.
    Works with Azure OpenAI, OpenAI, Anthropic, Google/Gemini, and local models.
    """
    meta = {
        "provider": None,
        "model": None,
        "version": None,
        "temperature": None,
        "top_p": None,
        "raw_class": llm.__class__.__name__,
        "raw_module": llm.__class__.__module__,
    }

    # Azure OpenAI
    if hasattr(llm, "deployment_name") or hasattr(llm, "azure_endpoint"):
        meta["provider"] = "azure-openai"
        meta["model"] = getattr(llm, "deployment_name", None)
        meta["version"] = getattr(llm, "openai_api_version", None)
        meta["temperature"] = getattr(llm, "temperature", None)
        meta["top_p"] = getattr(llm, "top_p", None)
        return meta

    # OpenAI (non-Azure)
    if hasattr(llm, "model_name"):
        meta["provider"] = "openai"
        meta["model"] = getattr(llm, "model_name", None)
        meta["temperature"] = getattr(llm, "temperature", None)
        meta["top_p"] = getattr(llm, "top_p", None)
        return meta

    # Anthropic
    if llm.__class__.__module__.startswith("langchain_anthropic") or llm.__class__.__name__.lower().startswith("chatanthropic"):
        meta["provider"] = "anthropic"
        model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        meta["model"] = model
        meta["temperature"] = getattr(llm, "temperature", None)
        return meta

    # Google / Gemini
    if llm.__class__.__module__.startswith("google") or "genai" in llm.__class__.__module__.lower():
        meta["provider"] = "google-genai"
        model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        meta["model"] = model
        return meta

    # Fallback
    meta["provider"] = "unknown"
    return meta

# -------------------------------

import tiktoken

def count_tokens(text: str, model_name="gpt-4"):
    """Count tokens in text using tiktoken"""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# -------------------------------
