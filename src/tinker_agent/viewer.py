"""Streamlit trace viewer for agent executions."""

import ast
import asyncio
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue

import streamlit as st

# Page config
st.set_page_config(
    page_title="Tinker Agent",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compact CSS
st.markdown(
    """
<style>
    /* Reduce padding everywhere - extra top padding to avoid cutoff */
    .block-container {
        padding: 2rem 1rem 1rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Fix header cutoff */
    .stApp > header {
        background: transparent !important;
    }
    .main > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1rem 0.5rem !important;
    }
    
    /* Smaller headers */
    h1 { font-size: 1.5rem !important; margin: 0 0 0.5rem 0 !important; }
    h2 { font-size: 1.2rem !important; margin: 0.5rem 0 !important; }
    h3 { font-size: 1rem !important; margin: 0.3rem 0 !important; }
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        padding: 0.3rem 0.5rem !important;
    }
    .streamlit-expanderContent {
        padding: 0.5rem !important;
    }
    
    /* Smaller text */
    p, span, div { font-size: 0.9rem; }
    
    /* Compact buttons */
    .stButton > button {
        padding: 0.2rem 0.5rem !important;
        font-size: 0.8rem !important;
    }
    
    /* Event styling */
    .event-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        font-size: 0.85rem;
        color: #888;
        border-bottom: 1px solid #333;
        margin-bottom: 4px;
    }
    
    .event-content {
        padding: 4px 0 8px 0;
        font-size: 0.9rem;
    }
    
    /* Tool call styling */
    .tool-name {
        background: #1e3d1e;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    
    /* Message styling */
    .msg-assistant {
        background: #1a1a2e;
        padding: 8px 12px;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 4px 0;
    }
    
    /* Compact code blocks - no wrap */
    .stCodeBlock {
        max-height: 300px;
    }
    pre {
        font-size: 0.8rem !important;
        padding: 8px !important;
        white-space: pre !important;
        overflow-x: auto !important;
    }
    code {
        white-space: pre !important;
    }
    
    /* Tool input styling - no wrap */
    .tool-input {
        background: #0e1117;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 8px;
        font-family: monospace;
        font-size: 0.8rem;
        overflow-x: auto;
        white-space: pre;
        margin: 4px 0;
    }
    
    /* Reduce gaps */
    .stMarkdown { margin-bottom: 0 !important; }
    .element-container { margin-bottom: 0.3rem !important; }
    
    /* Divider */
    hr { margin: 0.5rem 0 !important; }
    
    /* JSON viewer compact - no wrap */
    .stJson { 
        font-size: 0.8rem !important;
    }
    .stJson pre {
        white-space: pre !important;
        overflow-x: auto !important;
    }
    
    /* Command output (stdout/stderr) styling */
    .stdout-output {
        background: #0d1117;
        border-left: 3px solid #58a6ff;
        padding: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .stderr-output {
        background: #1a0d0d;
        border-left: 3px solid #f85149;
        padding: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Generic tool call styling */
    .tool-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 0.85rem;
        flex-wrap: wrap;
    }
    .tool-name-badge {
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .tool-name-badge.bash { background: #238636; }
    .tool-name-badge.todo { background: #8957e5; }
    .tool-desc {
        color: #8b949e;
        font-style: italic;
    }
    .tool-meta {
        margin-left: auto;
        display: flex;
        gap: 6px;
        align-items: center;
    }
    .tool-badge {
        background: #30363d;
        color: #8b949e;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .tool-time {
        color: #6e7681;
        font-size: 0.8rem;
    }
    .tool-waiting {
        color: #d29922;
        font-size: 0.8rem;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Todo list styling */
    .todo-list {
        background: #161b22;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    .todo-item {
        padding: 4px 0;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .todo-icon {
        font-family: monospace;
        width: 16px;
        text-align: center;
    }
    .todo-item.completed { color: #3fb950; }
    .todo-item.in-progress { color: #58a6ff; }
    .todo-item.pending { color: #8b949e; }

    /* CLI output styling */
    .cli-output {
        background: #0d1117;
        border-left: 3px solid #30363d;
        padding: 8px 12px;
        margin: 4px 0;
        font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
        font-size: 0.8rem;
        line-height: 1.4;
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
    }
    .cli-output pre {
        margin: 0;
        padding: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: #c9d1d9;
    }
    .cli-output.stderr {
        border-left-color: #f85149;
        background: #1a0d0d;
    }
    .cli-output.stderr pre {
        color: #f85149;
    }
    .cli-output.empty {
        color: #3fb950;
        padding: 4px 12px;
        font-size: 0.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def find_runs_with_traces(runs_dir: Path) -> list[dict]:
    """Find all run directories that have a traces.json file."""
    runs = []
    if not runs_dir.exists():
        return runs

    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            trace_file = run_dir / "traces.json"
            if trace_file.exists():
                try:
                    with open(trace_file) as f:
                        traces = json.load(f)
                        trace_count = len(traces) if isinstance(traces, list) else 1
                except (json.JSONDecodeError, IOError):
                    trace_count = 0

                runs.append(
                    {
                        "name": run_dir.name,
                        "path": run_dir,
                        "trace_file": trace_file,
                        "trace_count": trace_count,
                    }
                )
    return runs


def load_traces(trace_file: Path) -> list[dict]:
    """Load traces from a JSON file."""
    try:
        with open(trace_file) as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data] if data else []
    except (json.JSONDecodeError, IOError):
        return []


def fmt_time(ts: float) -> str:
    """Format timestamp."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_timeout(ms: int) -> str:
    """Format timeout in human-readable form."""
    if ms >= 60000:
        mins = ms // 60000
        return f"{mins}m"
    return f"{ms // 1000}s"


def render_bash_tool_call(
    tool_input: dict,
    time_str: str,
    is_pending: bool = False,
    log_file: str | None = None,
):
    """Render a Bash tool call with clean, compact UI."""
    command = tool_input.get("command", "")
    description = tool_input.get("description", "")
    timeout = tool_input.get("timeout")
    run_in_bg = tool_input.get("run_in_background", False)

    # Build metadata badges
    badges = []
    if timeout:
        badges.append(f"⏱ {fmt_timeout(timeout)}")
    if run_in_bg:
        badges.append("⚡ bg")
    if log_file and is_pending:
        badges.append("📡 streaming")

    badge_html = " ".join(f"<span class='tool-badge'>{b}</span>" for b in badges)

    # Header with description and badges
    desc_html = f"<span class='tool-desc'>{description}</span>" if description else ""

    # Show waiting indicator if pending with timeout
    waiting_html = ""
    if is_pending and timeout:
        waiting_html = "<span class='tool-waiting'>⏳ running...</span>"

    st.markdown(
        f"<div class='tool-header'><span class='tool-name-badge bash'>Bash</span> {desc_html} {waiting_html}<span class='tool-meta'>{badge_html} <span class='tool-time'>{time_str}</span></span></div>",
        unsafe_allow_html=True,
    )

    # Command in a code block (strip tee wrapper if present for cleaner display)
    display_command = command
    if "| stdbuf -oL tee /tmp/tinker_bash_logs/" in command:
        # Extract original command from wrapper
        display_command = command.split(") 2>&1 | stdbuf")[0]
        if display_command.startswith("("):
            display_command = display_command[1:]
    st.code(display_command, language="bash")

    # Show live streaming output if pending and log file exists
    if is_pending and log_file:
        log_path = Path(log_file)
        if log_path.exists():
            try:
                log_content = log_path.read_text()
                if log_content:
                    # Show last N lines of output
                    lines = log_content.splitlines()
                    max_lines = 50
                    if len(lines) > max_lines:
                        display_lines = lines[-max_lines:]
                        truncated_msg = (
                            f"... [{len(lines) - max_lines} earlier lines]\n"
                        )
                    else:
                        display_lines = lines
                        truncated_msg = ""

                    st.markdown(
                        f"<div class='cli-output' style='border-left-color: #d29922;'><pre>{truncated_msg}{_escape_html(chr(10).join(display_lines))}</pre></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='cli-output empty' style='border-left-color: #d29922;'>⏳ Waiting for output...</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass


def render_todo_tool_call(tool_input: dict, time_str: str):
    """Render a TodoWrite tool call with clean checklist UI."""
    todos = tool_input.get("todos", [])

    # Count by status
    in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
    completed = sum(1 for t in todos if t.get("status") == "completed")
    pending = sum(1 for t in todos if t.get("status") == "pending")

    # Summary badge
    summary = f"{completed}✓ {in_progress}→ {pending}○"

    st.markdown(
        f"<div class='tool-header'><span class='tool-name-badge todo'>Todo</span> <span class='tool-desc'>{len(todos)} items</span><span class='tool-meta'><span class='tool-badge'>{summary}</span> <span class='tool-time'>{time_str}</span></span></div>",
        unsafe_allow_html=True,
    )

    # Render todos as compact list
    todo_html = "<div class='todo-list'>"
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")
        icon = {"completed": "✓", "in_progress": "→", "pending": "○"}.get(status, "○")
        status_class = status.replace("_", "-")
        todo_html += f"<div class='todo-item {status_class}'><span class='todo-icon'>{icon}</span> {content}</div>"
    todo_html += "</div>"

    st.markdown(todo_html, unsafe_allow_html=True)


def parse_tool_result(result: str) -> dict:
    """Parse tool result, extracting stdout/stderr if present."""
    if not result:
        return {"type": "empty"}

    parsed = None

    # Try to parse as JSON first
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        pass

    # If JSON failed, try Python literal (handles single quotes from repr)
    if parsed is None:
        try:
            parsed = ast.literal_eval(result)
        except (ValueError, SyntaxError):
            pass

    # Process parsed dict
    if isinstance(parsed, dict):
        # Bash/shell output format
        if "stdout" in parsed or "stderr" in parsed:
            return {
                "type": "shell",
                "stdout": parsed.get("stdout", ""),
                "stderr": parsed.get("stderr", ""),
                "exit_code": parsed.get("exitCode") or parsed.get("exit_code"),
            }
        # TodoWrite result
        if "oldTodos" in parsed or "newTodos" in parsed:
            return {"type": "todo_result", "data": parsed}
        # File content
        if "content" in parsed and len(parsed) <= 3:
            return {"type": "file", "content": parsed.get("content", "")}
        # Generic JSON
        return {"type": "json", "data": parsed}

    # Plain text
    return {"type": "text", "content": result}


def render_tool_result(result: str, tool_name: str, is_error: bool, time_str: str):
    """Render tool result in a clean, CLI-like format."""
    parsed = parse_tool_result(result)
    result_type = parsed.get("type", "text")

    # Shell output (Bash, etc.)
    if result_type == "shell":
        stdout = parsed.get("stdout", "")
        stderr = parsed.get("stderr", "")
        exit_code = parsed.get("exit_code")

        if stdout:
            st.markdown(
                f"<div class='cli-output'><pre>{_escape_html(stdout[:5000])}</pre></div>",
                unsafe_allow_html=True,
            )
        if stderr:
            st.markdown(
                f"<div class='cli-output stderr'><pre>{_escape_html(stderr[:2000])}</pre></div>",
                unsafe_allow_html=True,
            )
        if not stdout and not stderr:
            st.markdown(
                "<div class='cli-output empty'>✓ (no output)</div>",
                unsafe_allow_html=True,
            )

    # TodoWrite result - just show success, the call already showed the todos
    elif result_type == "todo_result":
        pass  # Don't show anything, the tool call already rendered the todos

    # File content
    elif result_type == "file":
        content = parsed.get("content", "")
        if content:
            st.code(
                content[:3000] + ("..." if len(content) > 3000 else ""), language=None
            )

    # Generic JSON
    elif result_type == "json":
        data = parsed.get("data", {})
        # For small JSON, show inline
        json_str = json.dumps(data, indent=2)
        if len(json_str) < 500:
            st.code(json_str, language="json")
        else:
            with st.expander("Output", expanded=False):
                st.code(json_str, language="json")

    # Plain text
    elif result_type == "text":
        content = parsed.get("content", "")
        if content:
            # Check if it looks like CLI output
            if "\n" in content or len(content) > 100:
                st.markdown(
                    f"<div class='cli-output'><pre>{_escape_html(content[:3000])}</pre></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<code>{_escape_html(content)}</code>", unsafe_allow_html=True
                )

    # Empty
    elif result_type == "empty":
        st.markdown("<div class='cli-output empty'>✓</div>", unsafe_allow_html=True)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_event(
    event: dict,
    completed_tool_ids: set | None = None,
    trace_ended: bool = True,
    stream_log_files: dict | None = None,
):
    """Render a single event compactly."""
    etype = event.get("type", "unknown")
    ts = event.get("timestamp", 0)
    data = event.get("data", {})
    completed_tool_ids = completed_tool_ids or set()
    stream_log_files = stream_log_files or {}

    icons = {
        "message": "💬",
        "tool_call": "🔧",
        "tool_result": "📋",
        "thinking": "🧠",
        "error": "❌",
        "stop": "⏹",
        "result": "✅",
    }
    icon = icons.get(etype, "•")
    time_str = fmt_time(ts) if ts else ""

    if etype == "message":
        content = data.get("content", "")
        st.markdown(
            f"<div class='event-header'>{icon} <b>Assistant</b> <span>{time_str}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='msg-assistant'>{content}</div>", unsafe_allow_html=True
        )

    elif etype == "tool_call":
        tool_name = data.get("tool_name", "unknown")
        tool_input = data.get("tool_input", {})
        tool_id = data.get("tool_id", "")

        # Check if this tool call is still pending (no result yet)
        is_pending = tool_id not in completed_tool_ids and not trace_ended

        # Get streaming log file if available (match by original command)
        log_file = None
        if tool_name.lower() in {"bash", "shell", "execute", "run"}:
            command = tool_input.get("command", "")
            # Extract original command if wrapped with tee
            original_cmd = command
            if "| stdbuf -oL tee /tmp/tinker_bash_logs/" in command:
                original_cmd = command.split(") 2>&1 | stdbuf")[0]
                if original_cmd.startswith("("):
                    original_cmd = original_cmd[1:]
            log_file = stream_log_files.get(original_cmd)

        # Special rendering for specific tools
        if tool_name.lower() in {"bash", "shell", "execute", "run"}:
            render_bash_tool_call(
                tool_input, time_str, is_pending=is_pending, log_file=log_file
            )
        elif tool_name.lower() in {"todowrite", "todo_write", "todo"}:
            render_todo_tool_call(tool_input, time_str)
        # Single param and short value -> inline
        elif len(tool_input) == 1:
            key, val = list(tool_input.items())[0]
            val_str = json.dumps(val) if not isinstance(val, str) else val
            if len(val_str) < 200:
                st.markdown(
                    f"<div class='event-header'>{icon} <span class='tool-name'>{tool_name}</span> <code>{key}={val_str}</code> <span>{time_str}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='event-header'>{icon} <span class='tool-name'>{tool_name}</span> <span>{time_str}</span></div>",
                    unsafe_allow_html=True,
                )
                st.code(json.dumps(tool_input, indent=2), language="json")
        else:
            input_str = json.dumps(tool_input, indent=2)
            st.markdown(
                f"<div class='event-header'>{icon} <span class='tool-name'>{tool_name}</span> <span>{time_str}</span></div>",
                unsafe_allow_html=True,
            )
            if len(input_str) < 1000:
                st.code(input_str, language="json")
            else:
                with st.expander("Input", expanded=False):
                    st.code(input_str, language="json")

    elif etype == "tool_result":
        result = data.get("result", "")
        is_error = data.get("is_error", False)
        tool_name = data.get("tool_name", "")

        # Use the new clean renderer
        render_tool_result(result, tool_name, is_error, time_str)

    elif etype == "thinking":
        content = data.get("content", "")
        with st.expander(f"{icon} Thinking ({time_str})", expanded=False):
            st.markdown(content[:2000] + ("..." if len(content) > 2000 else ""))

    elif etype == "error":
        st.error(f"{icon} {data.get('message', 'Error')}")

    elif etype == "stop":
        st.caption(f"{icon} Stop: {data.get('reason', '')}")

    elif etype == "result":
        # Skip - the result content is already shown in the final assistant message
        pass


def render_trace(trace: dict):
    """Render a complete trace."""
    trace_id = trace.get("id", "?")
    prompt = trace.get("prompt", "")
    model = trace.get("model", "?")
    started = trace.get("started_at", "")
    ended = trace.get("ended_at")
    events = trace.get("events", [])

    # Status
    status = (
        "✅"
        if ended and not trace.get("error")
        else "❌"
        if trace.get("error")
        else "🔄"
    )

    # Header row
    cols = st.columns([4, 2, 2])
    with cols[0]:
        st.markdown(f"### {status} `{trace_id}` — {model}")
    with cols[1]:
        st.caption(f"Started: {started[:19] if started else '?'}")
    with cols[2]:
        st.caption(f"Events: {len(events)}")

    # Prompt
    st.markdown(f"**📝 Prompt:** {prompt}")

    # Build set of tool_call IDs that have received results
    completed_tool_ids = set()
    for event in events:
        if event.get("type") == "tool_result":
            tool_id = event.get("data", {}).get("tool_id")
            if tool_id:
                completed_tool_ids.add(tool_id)

    # Build mapping of original_command -> log_file from bash_stream_start events
    # We match by command since tool_id from can_use_tool doesn't match SDK's tool_use_id
    stream_log_files = {}
    for event in events:
        if event.get("type") == "bash_stream_start":
            data = event.get("data", {})
            original_command = data.get("original_command")
            log_file = data.get("log_file")
            if original_command and log_file:
                stream_log_files[original_command] = log_file

    # Events
    for event in events:
        render_event(
            event,
            completed_tool_ids=completed_tool_ids,
            trace_ended=ended is not None,
            stream_log_files=stream_log_files,
        )

    # Final error only (result is already shown as an event)
    if trace.get("error"):
        st.error(f"❌ {trace['error']}")

    # Auto-scroll: track event count and scroll when new events appear
    event_count = len(events)
    prev_count = st.session_state.get("prev_event_count", 0)
    is_live = not ended  # Still running

    if event_count > prev_count or (is_live and event_count > 0):
        st.session_state.prev_event_count = event_count
        # Inject scroll script via components
        import streamlit.components.v1 as components

        components.html(
            f"""
            <script>
                console.log('[Trace Viewer] Attempting scroll, events: {event_count}');
                (function() {{
                    function scrollToBottom() {{
                        const parent = window.parent;
                        if (!parent) {{
                            console.log('[Trace Viewer] No parent window');
                            return;
                        }}
                        // Find the scrollable main area
                        const doc = parent.document;
                        const selectors = [
                            '[data-testid="stAppViewContainer"]',
                            '[data-testid="stMain"]',
                            'section.stMain',
                            'section.main',
                            '.main',
                            '.block-container'
                        ];
                        for (const sel of selectors) {{
                            const el = doc.querySelector(sel);
                            if (el) {{
                                console.log('[Trace Viewer] Found:', sel, 'scrollHeight:', el.scrollHeight, 'clientHeight:', el.clientHeight);
                                if (el.scrollHeight > el.clientHeight) {{
                                    el.scrollTop = el.scrollHeight;
                                    console.log('[Trace Viewer] Scrolled via', sel);
                                    return;
                                }}
                            }}
                        }}
                        // Try scrolling body and html
                        doc.body.scrollTop = doc.body.scrollHeight;
                        doc.documentElement.scrollTop = doc.documentElement.scrollHeight;
                        parent.scrollTo(0, doc.documentElement.scrollHeight);
                        console.log('[Trace Viewer] Used body/window scroll');
                    }}
                    setTimeout(scrollToBottom, 150);
                }})();
            </script>
            """,
            height=1,
        )


def setup_run_directory(project_root: Path) -> Path:
    """Create a new run directory with uv project setup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = project_root / "runs" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Copy .env
    env_file = project_root / ".env"
    if env_file.exists():
        shutil.copy(env_file, runs_dir / ".env")

    # Setup uv project
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(runs_dir / ".venv")
    env["UV_NO_WORKSPACE"] = "1"

    subprocess.run(
        ["uv", "init", "--no-workspace"],
        cwd=runs_dir,
        env=env,
        check=True,
        capture_output=True,
    )

    # Update pyproject.toml
    pyproject = runs_dir / "pyproject.toml"
    content = pyproject.read_text()
    content = content.replace(
        "dependencies = []",
        'dependencies = [\n    "tinker",\n    "tinker-cookbook",\n    "wandb",\n    "python-dotenv",\n]',
    )
    content += '\n[tool.uv.sources]\ntinker-cookbook = { git = "https://github.com/thinking-machines-lab/tinker-cookbook.git" }\n'
    pyproject.write_text(content)

    subprocess.run(
        ["uv", "sync"], cwd=runs_dir, env=env, check=True, capture_output=True
    )

    return runs_dir


def run_agent_thread(prompt: str, runs_dir: Path, status_queue: Queue):
    """Run the agent in a background thread."""
    import sys

    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from tinker_agent.agent import run_agent, Config

    trace_path = runs_dir / "traces.json"
    config = Config(
        prompt=prompt,
        cwd=str(runs_dir),
        trace_path=str(trace_path),
    )

    try:
        status_queue.put({"status": "running"})
        asyncio.run(run_agent(config))
        status_queue.put({"status": "done"})
    except Exception as e:
        status_queue.put({"status": "error", "error": str(e)})


def main():
    # Find project root
    cwd = Path.cwd()
    project_root = (
        cwd
        if (cwd / "runs").exists()
        else cwd.parent
        if (cwd.parent / "runs").exists()
        else cwd
    )
    runs_dir = project_root / "runs"

    # Session state
    if "selected_run" not in st.session_state:
        st.session_state.selected_run = None
    if "trace_idx" not in st.session_state:
        st.session_state.trace_idx = 0
    if "agent_running" not in st.session_state:
        st.session_state.agent_running = False
    if "status_queue" not in st.session_state:
        st.session_state.status_queue = Queue()

    # Check for status updates from agent thread
    if st.session_state.agent_running:
        try:
            while not st.session_state.status_queue.empty():
                status = st.session_state.status_queue.get_nowait()
                if status.get("status") in ("done", "error"):
                    st.session_state.agent_running = False
                    if status.get("status") == "error":
                        st.error(f"Agent error: {status.get('error', 'Unknown')}")
        except:
            pass

    # Sidebar with sessions list (always visible)
    with st.sidebar:
        st.markdown("## 🔍 Sessions")

        # New session button (only show if a session is selected)
        if st.session_state.selected_run:
            if st.button("➕ New Session", use_container_width=True, type="primary"):
                st.session_state.selected_run = None
                st.session_state.agent_running = False
                st.session_state.trace_idx = 0
                st.rerun()
            st.divider()

        # Runs list
        runs = find_runs_with_traces(runs_dir)

        if not runs:
            st.caption("No sessions yet")
        else:
            for run in runs:
                name = run["name"]
                is_sel = st.session_state.selected_run == name
                if st.button(
                    f"{'▶' if is_sel else '○'} {name} ({run['trace_count']})",
                    key=f"r_{name}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.selected_run = name
                    st.session_state.trace_idx = 0
                    st.session_state.prev_event_count = 0
                    st.rerun()

    # Main area
    if not st.session_state.selected_run and not st.session_state.agent_running:
        # Centered chat interface
        st.markdown(
            """
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 60vh;
                text-align: center;
            }
            .chat-title {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }
            .chat-subtitle {
                color: #8b949e;
                margin-bottom: 2rem;
            }
        </style>
        <div class="chat-container">
            <div class="chat-title">🔧 Tinker Agent</div>
            <div class="chat-subtitle">Fine-tune language models with Tinker</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Chat input
        prompt = st.chat_input("What post-training task do you need help with?")

        if prompt:
            # Setup run directory
            with st.spinner("Setting up environment..."):
                new_run_dir = setup_run_directory(project_root)
                st.session_state.selected_run = new_run_dir.name

            # Start agent in background
            st.session_state.agent_running = True
            thread = threading.Thread(
                target=run_agent_thread,
                args=(prompt, new_run_dir, st.session_state.status_queue),
                daemon=True,
            )
            thread.start()
            st.rerun()

    else:
        # Agent is running or run is selected - show trace
        if st.session_state.selected_run:
            # Status indicator
            if st.session_state.agent_running:
                st.markdown(
                    '<span style="color: #d29922; animation: pulse 1s infinite;">● Running...</span>',
                    unsafe_allow_html=True,
                )

            trace_file = runs_dir / st.session_state.selected_run / "traces.json"

            if trace_file.exists():
                traces = load_traces(trace_file)

                if traces:
                    # Trace selector for multiple traces
                    if len(traces) > 1:
                        idx = st.selectbox(
                            "Trace",
                            range(len(traces)),
                            index=min(st.session_state.trace_idx, len(traces) - 1),
                            format_func=lambda i: f"{traces[i].get('id', '?')}: {traces[i].get('prompt', '')[:40]}...",
                            label_visibility="collapsed",
                        )
                        if idx != st.session_state.trace_idx:
                            st.session_state.prev_event_count = 0
                        st.session_state.trace_idx = idx
                    else:
                        idx = 0

                    render_trace(traces[idx])
                else:
                    if st.session_state.agent_running:
                        st.info("⏳ Agent starting...")
                    else:
                        st.warning("Empty trace file")
            else:
                if st.session_state.agent_running:
                    st.info("⏳ Agent starting...")
                else:
                    st.error("Trace file not found")

    # Auto-refresh
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=500, key="trace_refresh")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
