# base_streamlit.py
# -------------------------------

import os
import re
import uuid
import asyncio
import json
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import traceback
from toolkit import tools
from utils_streamlit import (
    get_llm_from_env, 
    base_path, 
    log_status, 
    read_log, 
    TerminalSession, 
    clean_ansi,
    initial_message,
    get_llm_metadata,
    export_chat_history,
    export_llm_metadata
)

# -------------------------------
# Page config must be first Streamlit command
st.set_page_config(
    page_title="CellAtria - Agentic AI for Single-Cell Research",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Streamlit
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1976d2 0%, #1565c0 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #f6f9fb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .log-viewer {
        background-color: #1435F3;
        color: white;
        font-family: monospace;
        font-size: 11px;
        padding: 1rem;
        border-radius: 5px;
        height: 300px;
        overflow-y: auto;
    }
    .terminal-output {
        background-color: black;
        color: white;
        font-family: monospace;
        font-size: 14px;
        padding: 0.5rem;
        border-radius: 5px;
        height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [initial_message]
    if 'llm_meta' not in st.session_state:
        st.session_state.llm_meta = None
    if 'terminal' not in st.session_state:
        st.session_state.terminal = TerminalSession()
    if 'backend_log' not in st.session_state:
        st.session_state.backend_log = []
    if 'log_lines' not in st.session_state:
        st.session_state.log_lines = []

# -------------------------------
def create_cellatria_graph(env_path):
    """Create the LangGraph agent"""
    env_file = os.path.join(env_path, ".env")
    if not os.path.isfile(env_file):
        raise FileNotFoundError(f"*** 🚨 .env file not found at: {env_file}")
    
    llm = get_llm_from_env(env_path)
    llm_meta = get_llm_metadata(llm)

    # Load system prompt
    with open(os.path.join(base_path, "system_prompts.md"), "r") as f:
        system_message = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("messages")
    ])

    llm_with_tools = llm.bind_tools(tools)
    chat_fn = prompt | llm_with_tools

    # LangGraph schema - use add_messages for proper message handling
    class AgentState(TypedDict):
        messages: Annotated[List, add_messages]

    # Create graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("tools", ToolNode(tools))
    
    def _coerce_content(content):
        """Force-resolve any lazy iterators or non-standard content to a plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    # Text content block — keep as-is if it has 'type'+'text', else stringify
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def chatbot_node(state: AgentState) -> dict:
        from langchain_core.messages import ToolMessage
        safe_messages = []
        for msg in state["messages"]:
            if hasattr(msg, "content") and not isinstance(msg.content, str):
                coerced = _coerce_content(msg.content)
                msg = msg.model_copy(update={"content": coerced})
            safe_messages.append(msg)
        return {"messages": chat_fn.invoke({"messages": safe_messages})}
    
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.set_entry_point("chatbot")
    
    graph = graph_builder.compile(checkpointer=MemorySaver())
    
    return graph, llm_meta

# -------------------------------
def load_or_create_thread_id():
    """Load existing thread ID or create new one"""
    thread_id_file = "/tmp/cellatria_thread_id.txt"
    try:
        if os.path.exists(thread_id_file):
            with open(thread_id_file, "r") as f:
                thread_id = f.read().strip()
                log_status(f"📂 Loaded existing thread ID: {thread_id[:8]}...")
        else:
            thread_id = str(uuid.uuid4())
            with open(thread_id_file, "w") as f:
                f.write(thread_id)
            log_status(f"🆕 Created new thread ID: {thread_id[:8]}...")
        return thread_id
    except Exception as e:
        log_status(f"⚠️ Error with thread ID file: {e}")
        return str(uuid.uuid4())

# -------------------------------
def normalize_chat_history(history):
    """Normalize chat history to ensure all messages are in dict format"""
    normalized = []
    needs_normalization = False
    
    for i, msg in enumerate(history):
        if isinstance(msg, dict):
            # Already in correct format
            if "role" in msg and "content" in msg:
                normalized.append(msg)
        elif isinstance(msg, HumanMessage):
            needs_normalization = True
            log_status(f"⚠️ Normalizing HumanMessage at index {i}")
            normalized.append({
                "role": "user",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content)
            })
        elif isinstance(msg, AIMessage):
            needs_normalization = True
            log_status(f"⚠️ Normalizing AIMessage at index {i}")
            normalized.append({
                "role": "assistant",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content)
            })
        else:
            log_status(f"⚠️ Skipping unknown message type at index {i}: {type(msg)}")
    
    if needs_normalization:
        log_status(f"✅ Normalized {len(normalized)} messages in chat history")
    
    return normalized

# -------------------------------
def load_chat_history():
    """Load chat history from JSON file"""
    history_file = "/tmp/cellatria_chat_history.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                data = json.load(f)
                file_thread_id = data.get("thread_id")
                if file_thread_id == st.session_state.thread_id:
                    history = data.get("history", [initial_message])
                    # Normalize to ensure all messages are dicts
                    history = normalize_chat_history(history)
                    log_status(f"📂 Loaded {len(history)-1} messages from file")
                    return history
                else:
                    log_status(f"⚠️ Thread ID mismatch")
    except Exception as e:
        log_status(f"⚠️ Could not load history: {str(e)}")
    
    return [initial_message]

# -------------------------------
def save_chat_history():
    """Save chat history to JSON file"""
    history_file = "/tmp/cellatria_chat_history.json"
    try:
        # Normalize history before saving to ensure all are dicts
        normalized_history = normalize_chat_history(st.session_state.chat_history)
        data = {
            "thread_id": st.session_state.thread_id,
            "history": normalized_history,
            "timestamp": str(uuid.uuid1())
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)
        log_status(f"💾 Saved {len(normalized_history)-1} messages")
    except Exception as e:
        log_status(f"⚠️ Error saving history: {str(e)}")

# -------------------------------
def clear_chat():
    """Clear chat history and start new session"""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = [initial_message]
    st.session_state.backend_log = []
    st.session_state.log_lines = []
    
    # Save new thread ID
    thread_id_file = "/tmp/cellatria_thread_id.txt"
    try:
        with open(thread_id_file, "w") as f:
            f.write(st.session_state.thread_id)
    except Exception as e:
        log_status(f"⚠️ Error saving thread ID: {e}")
    
    # Clear log file
    try:
        open("/tmp/cellatria_log.txt", "w").close()
    except Exception as e:
        log_status(f"⚠️ Error clearing log: {e}")
    
    log_status(f"🗑️ Chat cleared. New session: {st.session_state.thread_id[:8]}...")
    st.rerun()

# -------------------------------
async def process_user_input(user_input: str, graph, thread_id: str):
    """Process user input through the LangGraph agent with streaming"""
    
    def _log(msg: str):
        log_status(msg)
    
    _log("🟢 New interaction started.")
    _log(f"👤 User input: {user_input}")
    
    # Ensure chat history is normalized before processing
    st.session_state.chat_history = normalize_chat_history(st.session_state.chat_history)
    
    # Convert chat history to LangChain format
    messages = []
    first_user_seen = False
    for msg in st.session_state.chat_history:
        # Handle both dict format and Message objects
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        elif isinstance(msg, HumanMessage):
            role = "user"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, AIMessage):
            role = "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        else:
            log_status(f"⚠️ Skipping unknown message type: {type(msg)}")
            continue
        
        if role == "user":
            first_user_seen = True
            messages.append(HumanMessage(content=content))
        elif role == "assistant" and first_user_seen:
            messages.append(AIMessage(content=content))
    
    # Add new user message
    messages.append(HumanMessage(content=user_input))
    
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 1000
    }
    
    accumulated_text = ""
    backend_log = []
    got_llm_content = False
    after_tool = False  # Track whether we're in the post-tool LLM response
    
    # Create placeholder for streaming response
    message_placeholder = st.empty()
    message_placeholder.markdown("🤔 Thinking...")
    
    try:
        _log("🤖 Invoking agent...")
        
        async for event in graph.astream_events(
            {"messages": messages}, 
            config=config, 
            version="v2"
        ):
            kind = "unknown_event"
            try:
                kind = event.get("event", "unknown_event")
                
                # Capture streaming tokens from LLM
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Clear placeholder text before starting real content
                        if accumulated_text.startswith("🔧 Running tool:") or accumulated_text == "🤔 Thinking...":
                            accumulated_text = ""
                        accumulated_text += chunk.content
                        got_llm_content = True
                        after_tool = False
                        # Update placeholder with streaming text
                        message_placeholder.markdown(accumulated_text)
                
                # Log tool usage
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    backend_log.append(f"**Step:** `tools`\n**Tool:** `{tool_name}`")
                    _log(f"🔧 Tool started: {tool_name}")
                    accumulated_text = f"🔧 Running tool: {tool_name}..."
                    got_llm_content = False  # Reset so post-tool response is captured
                    message_placeholder.markdown(accumulated_text)
                
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    _log(f"✅ Tool completed: {tool_name}")
                    accumulated_text = "🤔 Thinking..."
                    after_tool = True
                    got_llm_content = False  # Reset so post-tool LLM response is captured
                    message_placeholder.markdown(accumulated_text)
                
                # Log chat model completion
                elif kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output:
                        meta = getattr(output, "response_metadata", {})
                        model = meta.get("model_name", "")
                        tokens = meta.get("token_usage", {})
                        if model or tokens:
                            summary = [
                                f"**Step:** `chatbot`",
                                f"**Model:** `{model}`" if model else "",
                                f"**Tokens:** {tokens.get('completion_tokens', '?')} completion, "
                                f"{tokens.get('prompt_tokens', '?')} prompt, "
                                f"{tokens.get('total_tokens', '?')} total" if tokens else ""
                            ]
                            backend_log.append("\n".join([s for s in summary if s]))
            except Exception as event_error:
                _log(f"⚠️ Error processing event {kind}: {str(event_error)}")
                _log(traceback.format_exc())
                # Continue processing other events
                continue
        
        # If no LLM content, or still showing a placeholder, fetch final state from graph
        if not got_llm_content or accumulated_text in ("🤔 Thinking...",) or accumulated_text.startswith("🔧 Running tool:"):
            log_status("⚠️ No LLM content from stream, fetching final state...")
            try:
                final_state = graph.get_state(config)
                if final_state and final_state.values.get("messages"):
                    # Scan backward for last AIMessage with non-empty text content
                    for msg in reversed(final_state.values["messages"]):
                        if isinstance(msg, AIMessage):
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if content and content.strip():
                                accumulated_text = content
                                _log(f"✅ Retrieved final response ({len(accumulated_text)} chars)")
                                message_placeholder.markdown(accumulated_text)
                                break
            except Exception as state_error:
                _log(f"❌ Error fetching final state: {str(state_error)}")
                _log(traceback.format_exc())
        
        _log("✅ Agent response received.")
        
    except Exception as e:
        _log(f"❌ Error during agent execution: {str(e)}")
        _log(f"Error type: {type(e).__name__}")
        _log(traceback.format_exc())
        
        # Check if error is related to Message object subscripting
        if "subscriptable" in str(e):
            _log("⚠️⚠️⚠️ SUBSCRIPTABLE ERROR DETECTED ⚠️⚠️⚠️")
            _log(f"Chat history length: {len(st.session_state.chat_history)}")
            _log("Chat history types:")
            for idx, msg in enumerate(st.session_state.chat_history):
                _log(f"  [{idx}] {type(msg).__name__}: {type(msg)}")
            _log("⚠️⚠️⚠️ END SUBSCRIPTABLE ERROR DEBUG ⚠️⚠️⚠️")
        
        # Only set error message if we don't have valid content
        if not accumulated_text or accumulated_text in ["🤔 Thinking...", "🔧 Running tool:"]:
            accumulated_text = f"There was an error processing your request.\n\nError: {str(e)}"
            message_placeholder.markdown(accumulated_text)
        backend_log.append(f"❌ Error: {str(e)}")
        backend_log.append(f"Error type: {type(e).__name__}")
    
    _log("🟣 Interaction complete.\n---")
    
    # Final validation - ensure we have content and it's a string
    if not accumulated_text or accumulated_text == "🤔 Thinking..." or accumulated_text.startswith("🔧 Running tool:"):
        accumulated_text = "I apologize, but I couldn't generate a response. Please check the logs for details."
        _log("⚠️ No accumulated text after stream completion")
        message_placeholder.markdown(accumulated_text)
    
    # Ensure accumulated_text is a string
    if not isinstance(accumulated_text, str):
        accumulated_text = str(accumulated_text)
    
    # Update session state with the final response (always append as dicts)
    st.session_state.chat_history.append({"role": "user", "content": str(user_input)})
    st.session_state.chat_history.append({"role": "assistant", "content": str(accumulated_text)})
    
    # Immediately normalize to ensure no Message objects sneak in
    st.session_state.chat_history = normalize_chat_history(st.session_state.chat_history)
    st.session_state.backend_log = backend_log
    
    # Save history
    save_chat_history()
    
    _log(f"✅ Updated chat history. Total messages: {len(st.session_state.chat_history)}")

# -------------------------------
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Welcome to CellAtria</h1>
        <h3>Agentic Triage of Regulated single-cell data Ingestion and Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize graph if not already done
    if st.session_state.graph is None:
        with st.spinner("🔄 Initializing CellAtria agent..."):
            try:
                env_path = os.getenv("CELLATRIA_ENV_PATH", "/mnt/work/projects")
                st.session_state.graph, st.session_state.llm_meta = create_cellatria_graph(env_path)
                st.session_state.thread_id = load_or_create_thread_id()
                loaded_history = load_chat_history()
                # Normalize loaded history to ensure consistency
                st.session_state.chat_history = normalize_chat_history(loaded_history)
                st.success("✅ CellAtria successfully initialized!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", width='stretch'):
            clear_chat()
        
        st.divider()
        
        # Live logs — auto-refreshes every 3 s by polling session_state.log_lines
        st.subheader("📊 Live Logs")

        @st.fragment(run_every=3)
        def _live_logs():
            log_text = "\n".join(st.session_state.get("log_lines", [])) or "📁 No logs yet."
            # Escape HTML special chars so log content renders as plain text
            escaped = log_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f"""
                <div id="log-box" style="
                    background:#0e1117;
                    color:#e0e0e0;
                    font-family:monospace;
                    font-size:11px;
                    padding:8px;
                    height:200px;
                    overflow-y:auto;
                    border-radius:4px;
                    white-space:pre-wrap;
                    word-break:break-all;
                ">{escaped}</div>
                <script>
                    (function() {{
                        var el = document.getElementById('log-box');
                        if (el) el.scrollTop = el.scrollHeight;
                    }})();
                </script>
                """,
                unsafe_allow_html=True,
            )

        _live_logs()
        
        st.divider()
        
        # Terminal Panel
        st.subheader("🖥️ Terminal")
        shell_cmd = st.text_input("Command:", key="shell_input", label_visibility="collapsed", placeholder="Enter shell command...")
        if st.button("▶ Execute", key="exec_terminal", width='stretch'):
            if shell_cmd:
                if 'terminal_output' not in st.session_state:
                    st.session_state.terminal_output = ""
                result = st.session_state.terminal.run_command(shell_cmd)
                st.session_state.terminal_output = clean_ansi(result)
        if 'terminal_output' in st.session_state and st.session_state.terminal_output:
            st.code(st.session_state.terminal_output, language="bash")
        
        st.divider()
        
        # File Browser
        st.subheader("📁 File Browser")
        if 'fb_path' not in st.session_state:
            st.session_state.fb_path = os.getcwd()
        fb_path = st.text_input("Path:", value=st.session_state.fb_path, key="fb_path_input", label_visibility="collapsed")
        if st.button("🔄 Browse", key="fb_refresh", width='stretch'):
            st.session_state.fb_path = fb_path
            if os.path.exists(fb_path):
                items = os.listdir(fb_path)
                subdirs = sorted([f for f in items if os.path.isdir(os.path.join(fb_path, f))])
                files = sorted([f for f in items if os.path.isfile(os.path.join(fb_path, f))])
                st.session_state.fb_subdirs = subdirs
                st.session_state.fb_files = files
                st.session_state.fb_error = None
            else:
                st.session_state.fb_error = "Directory does not exist"
        if 'fb_error' in st.session_state and st.session_state.fb_error:
            st.error(st.session_state.fb_error)
        if 'fb_subdirs' in st.session_state:
            if st.session_state.fb_subdirs:
                st.caption("📂 Subfolders")
                st.text("\n".join(st.session_state.fb_subdirs))
            if st.session_state.fb_files:
                st.caption("📄 Files")
                st.text("\n".join(st.session_state.fb_files))
        
        st.divider()
        
        # Agent Backend
        st.subheader("🔧 Agent Backend")
        if st.session_state.backend_log:
            for entry in st.session_state.backend_log:
                st.markdown(entry)
        else:
            st.caption("No agent activity yet.")
    
    # Main chat interface — wrapped in a fragment so sidebar interactions
    # don't cancel a running prompt.
    @st.fragment
    def _chat_area():
        st.subheader("💬 Chat with CellAtria Agent")

        # Ensure chat history is normalized before display
        st.session_state.chat_history = normalize_chat_history(st.session_state.chat_history)

        # All messages (history + live streaming) render in this container,
        # which sits above the input widgets.
        messages_container = st.container()

        with messages_container:
            for msg in st.session_state.chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                elif isinstance(msg, HumanMessage):
                    role = "user"
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                else:
                    log_status(f"⚠️ Skipping display of unknown message type: {type(msg)}")
                    continue

                with st.chat_message(role):
                    st.markdown(content)

        # PDF upload and chat input stay below the message history
        uploaded_pdf = st.file_uploader("📄 Attach a PDF (optional)", type=['pdf'], label_visibility="collapsed")
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_pdf.read())
            st.info(f"📄 PDF attached: `{uploaded_pdf.name}` — ask the agent to extract metadata from it.")
            pdf_key = f"pdf_{uploaded_pdf.name}"
            if pdf_key not in st.session_state:
                st.session_state[pdf_key] = True
                st.session_state.chat_history.append({"role": "user", "content": f"Uploaded PDF: {uploaded_pdf.name}"})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"📄 Received PDF: `{uploaded_pdf.name}`. I can extract metadata from it!"
                })
                st.session_state.chat_history = normalize_chat_history(st.session_state.chat_history)
                save_chat_history()

        user_input = st.chat_input("Give me a task...")

        if user_input:
            # New messages stream inside the same container so they appear
            # above — not below — the input widgets.
            with messages_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    asyncio.run(process_user_input(
                        user_input,
                        st.session_state.graph,
                        st.session_state.thread_id
                    ))

            # Rerun the fragment to move the response into the history list
            st.rerun(scope="fragment")

    _chat_area()

# -------------------------------
if __name__ == "__main__":
    # Clear log file on startup
    open("/tmp/cellatria_log.txt", "w").close()
    
    # Run the app
    main()
