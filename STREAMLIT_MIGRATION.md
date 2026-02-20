# CellAtria - Streamlit Migration Guide

## Overview

CellAtria has been successfully migrated from Gradio to Streamlit for the frontend interface. This document outlines the changes and how to use the new Streamlit-based application.

## What Changed

### Dependencies
- **Removed**: Gradio and related dependencies (gradio==6.6.0, gradio_client==2.1.0, ffmpy==1.0.0)
- **Kept**: Streamlit==1.54.0 (was already in requirements.txt)

### New Files Created
1. **`agent/base_streamlit.py`** - Main Streamlit application with full UI
2. **`agent/utils_streamlit.py`** - Streamlit-compatible utility functions
3. **`agent/chatbot_streamlit.py`** - Entry point for Streamlit version

### Modified Files
1. **`requirements.txt`** - Removed Gradio dependencies
2. **`Dockerfile`** - Updated to use Streamlit (port 8501) instead of Gradio (port 7860)
3. **`Dockerfile.base`** - Updated to install Streamlit instead of Gradio

### Original Files (Preserved)
The original Gradio files remain unchanged:
- `agent/base.py` - Original Gradio implementation
- `agent/chatbot.py` - Original Gradio entry point
- `agent/utils.py` - Original utilities with Gradio

## Running the Streamlit Version

### Local Development

#### Option 1: Direct Streamlit Run
```bash
cd /path/to/cellatria
streamlit run agent/base_streamlit.py --server.address=0.0.0.0 --server.port=8501
```

#### Option 2: Using the Entry Point
```bash
cd /path/to/cellatria
python agent/chatbot_streamlit.py --env_path /path/to/projects
```

Then in another terminal:
```bash
streamlit run agent/base_streamlit.py
```

### Docker

#### Build the Docker Image
```bash
docker build -t cellatria:streamlit .
```

#### Run the Container
```bash
docker run -p 8501:8501 -v /path/to/projects:/data cellatria:streamlit
```

#### Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

### Environment Setup

Make sure you have a `.env` file in your projects directory with the following structure:

```env
# LLM Provider Configuration
PROVIDER=Azure OpenAI  # Options: Azure OpenAI, OpenAI, Anthropic, Google, Local

# Azure OpenAI (if using Azure)
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_TEMPERATURE=0.7
AZURE_OPENAI_TOP_P=0.95

# OpenAI (if using OpenAI)
OPENAI_MODEL=gpt-4
OPENAI_API_KEY=your-api-key

# Anthropic (if using Anthropic)
ANTHROPIC_API_KEY=your-api-key

# Google (if using Google/Gemini)
GOOGLE_API_KEY=your-api-key
GOOGLE_MODEL=gemini-pro

# Local (if using local models)
LOCAL_MODEL_NAME=local-model
LOCAL_BASE_URL=http://localhost:1234/v1
LOCAL_API_KEY=not-needed
LOCAL_TEMPERATURE=0.7
```

## Key Features in Streamlit Version

### Main Interface
- **Chat Interface**: Native Streamlit chat interface with message history
- **Streaming Responses**: Real-time streaming of LLM responses
- **PDF Upload**: Upload research articles for metadata extraction
- **Sidebar Controls**: 
  - Clear chat history
  - Export chat history (JSON)
  - Export LLM metadata (JSON)
  - Live logs viewer with auto-refresh

### Expandable Panels
- **Terminal Panel**: Execute shell commands directly
- **File Browser**: Navigate and explore project directories
- **Agent Backend**: View agent execution details and tool usage

### Persistent Features
- Chat history persists across sessions (stored in `/tmp/cellatria_chat_history.json`)
- Thread ID management for conversation continuity
- Log file tracking (`/tmp/cellatria_log.txt`)

## Port Changes

| Version | Port | URL |
|---------|------|-----|
| Gradio (old) | 7860 | http://localhost:7860 |
| Streamlit (new) | 8501 | http://localhost:8501 |

## API Compatibility

The underlying LangGraph agent, tools, and CellExpress pipeline remain unchanged. Only the frontend has been migrated. All core functionality is preserved:

- ✅ Metadata extraction from PDFs
- ✅ GEO database access
- ✅ Dataset download and organization
- ✅ CellExpress pipeline integration
- ✅ Terminal access
- ✅ File system management
- ✅ Multi-provider LLM support

## Benefits of Streamlit

1. **Better Integration**: Native Python integration with data science workflows
2. **Simpler Deployment**: Easier to customize and deploy
3. **Active Development**: Streamlit has frequent updates and strong community support
4. **Better Widgets**: More intuitive widget system for interactive components
5. **Native State Management**: Built-in session state management
6. **Improved Performance**: Better handling of large data and streaming

## Troubleshooting

### Issue: Application won't start
**Solution**: Ensure your `.env` file is in the correct location (specified by `--env_path` or `/data` in Docker)

### Issue: Port 8501 already in use
**Solution**: 
```bash
# Find process using port 8501
lsof -i :8501

# Kill the process or use a different port
streamlit run agent/base_streamlit.py --server.port=8502
```

### Issue: Chat history not persisting
**Solution**: Check permissions on `/tmp` directory:
```bash
ls -la /tmp/cellatria_*
```

### Issue: LLM not responding
**Solution**: 
1. Check your `.env` configuration
2. Review logs in the sidebar
3. Verify API keys are valid

## Migration Notes for Developers

If you were using the Gradio version and want to migrate custom code:

### Gradio → Streamlit Component Mapping

| Gradio Component | Streamlit Equivalent |
|------------------|---------------------|
| `gr.Chatbot()` | `st.chat_message()` + `st.chat_input()` |
| `gr.Textbox()` | `st.text_input()` or `st.text_area()` |
| `gr.Button()` | `st.button()` |
| `gr.File()` | `st.file_uploader()` |
| `gr.Accordion()` | `st.expander()` |
| `gr.Row()` / `gr.Column()` | `st.columns()` |
| `gr.State()` | `st.session_state` |
| `gr.Dropdown()` | `st.selectbox()` |

### Key Code Changes

**Gradio (old):**
```python
import gradio as gr

with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    user_input = gr.Textbox()
    
app.launch(server_port=7860)
```

**Streamlit (new):**
```python
import streamlit as st

st.title("CellAtria")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Your message...")
```

## Backwards Compatibility

The original Gradio implementation is preserved in:
- `agent/base.py`
- `agent/chatbot.py`
- `agent/utils.py`

To use the original Gradio version (if needed):
1. Re-add Gradio to requirements.txt
2. Use the original `chatbot.py` entry point
3. Update Dockerfile to expose port 7860

## Support

For issues or questions:
- Open an issue on GitHub: https://github.com/AstraZeneca/cellatria
- Review the logs in `/tmp/cellatria_log.txt`
- Check the Live Logs panel in the Streamlit sidebar

## Version Information

- **CellAtria Version**: 1.0.0
- **Streamlit Version**: 1.54.0
- **Migration Date**: February 2026
- **Previous Frontend**: Gradio 6.6.0
- **Current Frontend**: Streamlit 1.54.0

---

**Note**: This migration maintains full compatibility with the CellExpress pipeline and all backend functionality. Only the user interface has changed.
