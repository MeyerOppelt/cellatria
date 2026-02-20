# base.py
# -------------------------------

import os
import re
import uuid
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import traceback
import json, tempfile
import pprint
import gradio as gr
from toolkit import tools
from utils import (gr_css, get_llm_from_env, chatbot_theme, base_path, 
                    log_status, read_log, 
                    TerminalSession, clean_ansi, terminal_interface, 
                    fb_initial_refresh, fb_navigate_subdir, 
                    export_chat_history, initial_message, get_llm_metadata,
                    export_llm_metadata)

# -------------------------------
def create_cellatria(env_path):

    env_file = os.path.join(env_path, ".env")
    if not os.path.isfile(env_file):
        raise FileNotFoundError(f"*** 🚨 .env file not found at: {env_file}")
    llm = get_llm_from_env(env_path)

    # -------------------------------

    llm_meta = get_llm_metadata(llm)

    # -------------------------------
    # Define prompt template
    # Load the system message from file

    with open(os.path.join(base_path, "system_prompts.md"), "r") as f:
        system_message = f.read()

    # Defines the agent’s role and toolset
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("messages")
    ])

    # -------------------------------
    # Bind tools to model
    llm_with_tools = llm.bind_tools(tools)
    chat_fn = prompt | llm_with_tools

    # -------------------------------
    # LangGraph schema
    class AgentState(TypedDict):
        messages: Annotated[List, add_messages]

    # -------------------------------
    # Create the graph with schema
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("tools", ToolNode(tools))
    def chatbot_node(state: AgentState) -> dict:  # type: ignore[misc]
        return {"messages": chat_fn.invoke(state["messages"])}  # type: ignore[arg-type]
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.set_entry_point("chatbot")
    
    # Use MemorySaver for runtime, but persist to JSON file
    graph = graph_builder.compile(checkpointer=MemorySaver())
    
    # File-based persistence for chat history
    history_file = "/tmp/cellatria_chat_history.json"
    log_status(f"💾 Using JSON file for chat persistence: {history_file}")

    # -------------------------------
    # Chat Handler
    # Persistent thread ID stored in file
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
    except Exception as e:
        log_status(f"⚠️ Error with thread ID file: {e}")
        thread_id = str(uuid.uuid4())
    
    thread_state = {"id": thread_id}
    
    def load_history_from_file():
        """Load chat history from JSON file"""
        try:
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    data = json.load(f)
                    file_thread_id = data.get("thread_id")
                    # Validate thread ID matches
                    if file_thread_id == thread_state["id"]:
                        history = data.get("history", [initial_message])
                        log_status(f"📂 Loaded {len(history)-1} messages from file (thread: {file_thread_id[:8]}...)")
                        return history
                    else:
                        log_status(f"⚠️ Thread ID mismatch: file={file_thread_id[:8] if file_thread_id else 'None'}... current={thread_state['id'][:8]}...")
            else:
                log_status(f"ℹ️ History file does not exist yet: {history_file}")
        except Exception as e:
            log_status(f"⚠️ Could not load history from file: {str(e)}")
            log_status(traceback.format_exc())
        
        return [initial_message]
    
    def save_history_to_file(history):
        """Save chat history to JSON file"""
        try:
            if not history:
                log_status("⚠️ History is empty, not saving")
                return
            data = {
                "thread_id": thread_state["id"],
                "history": history,
                "timestamp": str(uuid.uuid1())
            }
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
            log_status(f"💾 Saved {len(history)-1} messages to file: {history_file}")
        except Exception as e:
            log_status(f"⚠️ Error saving history to file: {str(e)}")
            log_status(traceback.format_exc())
    
    def clear_chat():
        """Clear chat history and start new session"""
        thread_state["id"] = str(uuid.uuid4())
        # Save new thread ID to file
        try:
            with open(thread_id_file, "w") as f:
                f.write(thread_state["id"])
        except Exception as e:
            log_status(f"⚠️ Error saving thread ID: {e}")
        
        # Clear the log file
        try:
            open("/tmp/cellatria_log.txt", "w").close()
        except Exception as e:
            log_status(f"⚠️ Error clearing log: {e}")
        
        log_status(f"🗑️ Chat cleared. New session started: {thread_state['id'][:8]}...")
        return [initial_message], [initial_message], "", None, "Chat history cleared."
    
    def gr_block_fn(user_input, pdf_file, history):
        messages = []

        # Load history from file if client state is empty/reset
        if not history or history == [initial_message]:
            loaded_history = load_history_from_file()
            if len(loaded_history) > 1:  # More than just initial message
                history = loaded_history
                log_status(f"📥 Loaded {len(history)-1} messages from file")

        log_status("🟢 New interaction started.")

        # Convert history into LangChain format
        # Skip leading assistant messages (e.g. initial greeting) to ensure
        # the message list always starts with a HumanMessage
        first_user_seen = False
        for h in history:
            role = h["role"]
            content = h["content"]
            if role == "user":
                first_user_seen = True
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and first_user_seen:
                messages.append(AIMessage(content=content))

        # Append new user input
        if user_input:
            log_status(f"👤 User input: {user_input}")
            messages.append(HumanMessage(content=user_input))

        # Prepare config
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_state["id"]},
            "recursion_limit": 1000
        }

        backend_log = []
        accumulated_text = ""
        pdf_note = ""

        # Handle PDF upload
        if pdf_file:
            pdf_note = f"\n\n📄 Received PDF: `{pdf_file.name}`. \nI can extract metadata from it!"
            log_status("🟣 Interaction complete.\n---")
            yield (
                history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": pdf_note}],
                "",
                None,
                history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": pdf_note}],
                pdf_note
            )
            return

        # Immediately show user message with a placeholder for assistant response
        yield (
            history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": "🤔 Thinking..."}
            ],
            "",
            None,
            history,  # Don't update state yet
            ""
        )

        # Stream responses from LangGraph
        try:
            log_status("🤖 Invoking agent...")
            
            import asyncio
            
            # Helper to run async streaming
            async def process_stream():
                nonlocal accumulated_text, backend_log
                
                async for event in graph.astream_events(
                    {"messages": messages}, 
                    config=config, 
                    version="v2"
                ):
                    kind = event["event"]
                    
                    # Capture streaming tokens from the LLM
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            # If coming from tool execution, reset first
                            if accumulated_text.startswith("🔧 Running tool:"):
                                accumulated_text = ""
                            accumulated_text += chunk.content
                            # Yield streaming update
                            yield accumulated_text
                    
                    # Log tool usage and show which tool is running
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        backend_log.append(f"**Step:** `tools`\n**Tool:** `{tool_name}`")
                        log_status(f"🔧 Tool started: {tool_name}")
                        # Show tool running status
                        accumulated_text = f"🔧 Running tool: {tool_name}..."
                        # Yield status update
                        yield accumulated_text
                    
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
            
            # Run async generator and yield results to Gradio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create async generator
                async_gen = process_stream()
                
                # Process each item as it arrives
                while True:
                    try:
                        # Get next item from async generator
                        partial_text = loop.run_until_complete(async_gen.__anext__())
                        
                        # Yield the update to Gradio immediately
                        yield (
                            history + [
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": partial_text}
                            ],
                            "",  # Clear user input
                            None,  # Clear PDF upload
                            history + [
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": partial_text}
                            ],
                            "\n\n".join(backend_log)
                        )
                    except StopAsyncIteration:
                        # Generator is exhausted
                        break
                    except Exception as stream_error:
                        log_status(f"⚠️ Stream error: {str(stream_error)}")
                        break
            finally:
                loop.close()
            
            log_status("✅ Agent response received.")
            
        except Exception as e:
            log_status(f"❌ Error: {str(e)}")
            log_status(traceback.format_exc())
            accumulated_text = "There was an error processing your request."
            backend_log.append(f"❌ Error: {str(e)}")
            backend_log.append(traceback.format_exc())

        log_status("🟣 Interaction complete.\n---")

        # Ensure final state is yielded
        # Always yield final state, even if accumulated_text is empty (e.g., during tool calls)
        if not accumulated_text:
            accumulated_text = "Processing your request..."
        
        yield (
            history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": accumulated_text}
            ],
            "",
            None,
            history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": accumulated_text}
            ],
            "\n\n".join(backend_log)
        )

    # -------------------------------
    # Clear the log file when app starts
    open("/tmp/cellatria_log.txt", "w").close()  # clears the log file

    # -------------------------------
    # Load initial history from file (for page refreshes)
    initial_history = load_history_from_file()
    history_info = f"💾 Restored {len(initial_history)-1} previous message(s) from file" if len(initial_history) > 1 else ""
    
    # -------------------------------
    # Interface
    with gr.Blocks() as cellatria:
        gr.HTML("""
            <div style='text-align: center; margin-bottom: 0;'>
                <h1 style='margin-bottom: 0.2em;'>Welcome to cellAtria</h1>
                <h3 style='margin-top: 0.2em;'>Agentic Triage of Regulated single cell data Ingestion and Analysis</h3>
            </div>
        """)
        
        # Show history restore info if applicable
        if history_info:
            gr.Markdown(f"_{history_info}_", elem_id="history_info")
        
        gr.Image(
            value=os.path.join(base_path, "cellatria_header.png"),
            label="Welcome to cellAtria",
            interactive=False,
            show_label=False,
            container=True,
            height=125, 
        )

        # Chat Interface - load from file on startup
        chatbot = gr.Chatbot(
            value=initial_history,  # type: ignore[arg-type]
            label="cellAtria Agent",
            show_label=False,
            height=500,
            autoscroll=True,
            resizable=True
        )

        with gr.Row(equal_height=True, elem_id="fixed_top_row"):
            with gr.Column(scale=4):
                user_input = gr.Textbox(placeholder="Give me a task...", label="Your Prompt", lines=1)
            with gr.Column(scale=1):
                with gr.Row():  
                    pdf_upload = gr.File(file_types=[".pdf"], label=".pdf", show_label=True, interactive=True)
                    submit_btn = gr.Button("Submit Prompt/PDF", variant="primary")
            with gr.Column(scale=2):
                log_viewer = gr.Textbox(label="Live Logs", lines=12, interactive=False, elem_id="log_viewer_aes")
        
        # Clear chat button
        with gr.Row():
            with gr.Column(scale=9):
                pass  # Spacer
            with gr.Column(scale=1):
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

        # Hidden state to maintain chat memory - initialize with loaded history
        state = gr.State(initial_history)
  
        # --- Backend Panel ---
        with gr.Accordion("Agent Backend", open=False, elem_id="agent_backend_panel"):
            agent_backend_md = gr.Markdown("No agent activity yet.", elem_id="agent_backend_md")

        # Bind inputs to gr_block_fn (Gradio 6.x auto-detects generators for streaming)
        user_input.submit(
            fn=gr_block_fn,
            inputs=[user_input, pdf_upload, state],
            outputs=[chatbot, user_input, pdf_upload, state, agent_backend_md]
        ).then(
            fn=save_history_to_file,
            inputs=[state],
            outputs=[]
        )
        submit_btn.click(
            fn=gr_block_fn,
            inputs=[user_input, pdf_upload, state],
            outputs=[chatbot, user_input, pdf_upload, state, agent_backend_md]
        ).then(
            fn=save_history_to_file,
            inputs=[state],
            outputs=[]
        )
        
        # Load history from file on page load/refresh
        def load_on_startup():
            """Load history when page loads"""
            loaded = load_history_from_file()
            return loaded, loaded
        
        cellatria.load(
            fn=load_on_startup,
            inputs=[],
            outputs=[chatbot, state]
        )
        
        # Clear chat button
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, state, user_input, pdf_upload, agent_backend_md]
        )

        # --- Terminal Panel ---
        with gr.Accordion("Terminal Panel", open=False, elem_id="logs_terminal_panel"):
            with gr.Row(equal_height=True):
                shell_input = gr.Textbox(placeholder="Enter shell command (e.g., ls -la)", lines=1, label="Command")
            shell_output = gr.Textbox(label="Terminal Output", lines=10, interactive=False, elem_id="terminal_aes")

        # Timer Hook
        log_timer = gr.Timer(value=1.0, active=True)
        log_timer.tick(fn=read_log, inputs=[], outputs=[log_viewer] )

        # Bind inputs to gr_block_fn
        shell_input.submit(
            fn=terminal_interface,
            inputs=shell_input,
            outputs=[shell_output, shell_input]
        )

        # --- File Browser ---
        with gr.Accordion("File Browser", open=False, elem_id="logs_browser_panel"):

            fb_base_path = gr.Textbox(value=os.getcwd(), label="Directory Path")
            fb_current_path_md = gr.Markdown()
            fb_dir_dropdown = gr.Dropdown(label="Subdirectories", choices=[], interactive=True)
            fb_refresh_button = gr.Button("Refresh Directory", variant="primary")
            fb_file_display = gr.Textbox(label="Files in Directory", lines=10, interactive=False)

            fb_dir_dropdown.change(
                fn=fb_navigate_subdir,
                inputs=[fb_dir_dropdown, fb_base_path],
                outputs=[fb_dir_dropdown, fb_file_display, fb_base_path, fb_current_path_md]
            )

            fb_refresh_button.click(
                fn=fb_initial_refresh,
                inputs=fb_base_path,
                outputs=[fb_dir_dropdown, fb_file_display, fb_base_path, fb_current_path_md]
            )

        # --- History Panel ---
        with gr.Accordion("Export Chat / Model", open=False, elem_id="logs_history_panel"):
            with gr.Row():
                with gr.Column(scale=1):
                    export_btn = gr.Button("Download Chat", variant="secondary", elem_id="btn_export")
                    chat_file = gr.File(file_types=[".json"], label="Chat .json", interactive=False, show_label=True)
                with gr.Column(scale=1):
                    llm_meta_btn = gr.Button("Download LLM Metadata", variant="secondary", elem_id="btn_llm_meta")
                    llm_meta_file = gr.File(file_types=[".json"], label="LLM .json", interactive=False, show_label=True)

        # Bind inputs to gr_block_fn
        export_btn.click(
            fn=export_chat_history,
            inputs=[state],
            outputs=[chat_file]
        )

        llm_meta_btn.click(
            fn=lambda: export_llm_metadata(llm_meta),
            inputs=[],
            outputs=[llm_meta_file]
        )
    
    # Enable queuing for streaming support (required in Gradio 6.x)
    cellatria.queue()

    return graph, cellatria

# -------------------------------