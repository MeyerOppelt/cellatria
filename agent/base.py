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
    graph = graph_builder.compile(checkpointer=MemorySaver())

    # -------------------------------
    # Chat Handler
    # Persistent thread ID for LangGraph
    chat_thread_id = str(uuid.uuid4())
    
    def gr_block_fn(user_input, pdf_file, history):
        messages = []

        # Ensure initial message is included only once
        if not history:
            history = [initial_message]

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
            "configurable": {"thread_id": chat_thread_id},
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
                            accumulated_text += chunk.content
                            # Yield streaming update
                            yield accumulated_text
                    
                    # Log tool usage
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        backend_log.append(f"**Step:** `tools`\n**Tool:** `{tool_name}`")
                    
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
            
            # Run async generator and yield results
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_gen = process_stream()
                while True:
                    try:
                        partial_text = loop.run_until_complete(async_gen.__anext__())
                        # Yield the update to Gradio
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
        if accumulated_text:
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
    # Interface
    with gr.Blocks() as cellatria:
        gr.HTML("""
            <div style='text-align: center; margin-bottom: 0;'>
                <h1 style='margin-bottom: 0.2em;'>Welcome to cellAtria</h1>
                <h3 style='margin-top: 0.2em;'>Agentic Triage of Regulated single cell data Ingestion and Analysis</h3>
            </div>
        """)
        gr.Image(
            value=os.path.join(base_path, "cellatria_header.png"),
            label="Welcome to cellAtria",
            interactive=False,
            show_label=False,
            container=True,
            height=125, 
        )

        # Chat Interface
        chatbot = gr.Chatbot(
            value=[initial_message],  # type: ignore[arg-type]
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


        # Hidden state to maintain chat memory
        state = gr.State([initial_message])
  
        # --- Backend Panel ---
        with gr.Accordion("Agent Backend", open=False, elem_id="agent_backend_panel"):
            agent_backend_md = gr.Markdown("No agent activity yet.", elem_id="agent_backend_md")

        # Bind inputs to gr_block_fn
        user_input.submit(
            fn=gr_block_fn,
            inputs=[user_input, pdf_upload, state],
            outputs=[chatbot, user_input, pdf_upload, state, agent_backend_md]
        )
        submit_btn.click(
            fn=gr_block_fn,
            inputs=[user_input, pdf_upload, state],
            outputs=[chatbot, user_input, pdf_upload, state, agent_backend_md]
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

    return graph, cellatria

# -------------------------------