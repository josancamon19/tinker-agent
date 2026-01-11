"""Streamlit trace viewer for agent executions."""

import ast
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

# Page config
st.set_page_config(
    page_title="Tinker Agent Viewer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compact CSS (keeping the same styling)
st.markdown(
    """
<style>
    .block-container {
        padding: 2rem 1rem 1rem 1rem !important;
        max-width: 100% !important;
    }

    .stApp > header {
        background: transparent !important;
    }
    .main > div:first-child {
        padding-top: 0 !important;
    }

    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1rem 0.5rem !important;
    }

    h1 { font-size: 1.5rem !important; margin: 0 0 0.5rem 0 !important; }
    h2 { font-size: 1.2rem !important; margin: 0.5rem 0 !important; }
    h3 { font-size: 1rem !important; margin: 0.3rem 0 !important; }

    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        padding: 0.3rem 0.5rem !important;
    }
    .streamlit-expanderContent {
        padding: 0.5rem !important;
    }

    p, span, div { font-size: 0.9rem; }

    .stButton > button {
        padding: 0.2rem 0.5rem !important;
        font-size: 0.8rem !important;
    }

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

    .tool-name {
        background: #1e3d1e;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }

    .msg-assistant {
        background: #1a1a2e;
        padding: 8px 12px;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 4px 0;
    }

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

    .stMarkdown { margin-bottom: 0 !important; }
    .element-container { margin-bottom: 0.3rem !important; }

    hr { margin: 0.5rem 0 !important; }

    .stJson {
        font-size: 0.8rem !important;
    }
    .stJson pre {
        white-space: pre !important;
        overflow-x: auto !important;
    }

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
                runs.append(
                    {
                        "name": run_dir.name,
                        "path": run_dir,
                        "trace_file": trace_file,
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


def fmt_relative_time(dt: datetime) -> str:
    """Format relative time (e.g., '5 minutes ago', 'yesterday')."""
    now = datetime.now()
    diff = now - dt

    seconds = diff.total_seconds()

    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 172800:  # Less than 2 days
        return "yesterday"
    else:
        days = int(seconds / 86400)
        return f"{days}d ago"


def parse_run_name(name: str) -> datetime | None:
    """Parse run directory name (YYYYMMDD_HHMMSS) to datetime."""
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def fmt_timeout(ms: int) -> str:
    """Format timeout in human-readable form."""
    if ms >= 60000:
        mins = ms // 60000
        return f"{mins}m"
    return f"{ms // 1000}s"


def extract_log_file_from_command(command: str) -> str | None:
    """Extract log file path directly from a tee-wrapped command."""
    # Pattern: (cmd) 2>&1 | stdbuf -oL tee /path/to/log
    if "| stdbuf -oL tee " in command:
        parts = command.split("| stdbuf -oL tee ")
        if len(parts) >= 2:
            # The log path is after "tee ", may have trailing content
            log_path = parts[-1].strip().split()[0] if parts[-1].strip() else None
            return log_path
    return None


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

    # Try to extract log file path directly from command if not provided
    if not log_file:
        log_file = extract_log_file_from_command(command)

    badges = []
    if timeout:
        badges.append(f"⏱ {fmt_timeout(timeout)}")
    if run_in_bg:
        badges.append("⚡ bg")
    if log_file and is_pending:
        badges.append("📡 streaming")

    badge_html = " ".join(f"<span class='tool-badge'>{b}</span>" for b in badges)
    desc_html = f"<span class='tool-desc'>{description}</span>" if description else ""

    # Show waiting indicator if pending
    waiting_html = ""
    if is_pending and timeout:
        waiting_html = (
            "<span style='color: #d29922; margin-left: 8px;'>⏳ running...</span>"
        )

    st.markdown(
        f"<div class='tool-header'><span class='tool-name-badge bash'>Bash</span> {desc_html} {waiting_html}<span class='tool-meta'>{badge_html} <span class='tool-time'>{time_str}</span></span></div>",
        unsafe_allow_html=True,
    )

    # Strip tee wrapper for cleaner display
    display_command = command
    if "| stdbuf -oL tee" in command and ") 2>&1 |" in command:
        display_command = command.split(") 2>&1 |")[0]
        if display_command.startswith("("):
            display_command = display_command[1:]
    st.code(display_command, language="bash")

    # Show streaming output if pending and log file exists
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

    in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
    completed = sum(1 for t in todos if t.get("status") == "completed")
    pending = sum(1 for t in todos if t.get("status") == "pending")

    summary = f"{completed}✓ {in_progress}→ {pending}○"

    st.markdown(
        f"<div class='tool-header'><span class='tool-name-badge todo'>Todo</span> <span class='tool-desc'>{len(todos)} items</span><span class='tool-meta'><span class='tool-badge'>{summary}</span> <span class='tool-time'>{time_str}</span></span></div>",
        unsafe_allow_html=True,
    )

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

    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        pass

    if parsed is None:
        try:
            parsed = ast.literal_eval(result)
        except (ValueError, SyntaxError):
            pass

    if isinstance(parsed, dict):
        if "stdout" in parsed or "stderr" in parsed:
            return {
                "type": "shell",
                "stdout": parsed.get("stdout", ""),
                "stderr": parsed.get("stderr", ""),
                "exit_code": parsed.get("exitCode") or parsed.get("exit_code"),
            }
        if "oldTodos" in parsed or "newTodos" in parsed:
            return {"type": "todo_result", "data": parsed}
        if "content" in parsed and len(parsed) <= 3:
            return {"type": "file", "content": parsed.get("content", "")}
        return {"type": "json", "data": parsed}

    return {"type": "text", "content": result}


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_tool_result(
    result: str,
    tool_name: str,
    is_error: bool,
    time_str: str,
    log_file: str | None = None,
):
    """Render tool result in a clean, CLI-like format."""
    parsed = parse_tool_result(result)
    result_type = parsed.get("type", "text")

    if result_type == "shell":
        stdout = parsed.get("stdout", "")
        stderr = parsed.get("stderr", "")

        # For large outputs, prefer showing log file if available
        if log_file and (len(stdout) > 5000 or not stdout):
            log_path = Path(log_file)
            if log_path.exists():
                try:
                    log_content = log_path.read_text()
                    if log_content:
                        lines = log_content.splitlines()
                        total_lines = len(lines)
                        # Show last 100 lines for completed commands
                        max_lines = 100
                        if total_lines > max_lines:
                            display_lines = lines[-max_lines:]
                            truncated_msg = f"... [{total_lines - max_lines} earlier lines - view full log: {log_file}]\n"
                        else:
                            display_lines = lines
                            truncated_msg = ""

                        st.markdown(
                            f"<div class='cli-output'><pre>{truncated_msg}{_escape_html(chr(10).join(display_lines))}</pre></div>",
                            unsafe_allow_html=True,
                        )
                        # Show stderr separately if present
                        if stderr:
                            st.markdown(
                                f"<div class='cli-output stderr'><pre>{_escape_html(stderr[:2000])}</pre></div>",
                                unsafe_allow_html=True,
                            )
                        return  # Skip default stdout handling
                except Exception:
                    pass  # Fall back to stdout handling

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

    elif result_type == "todo_result":
        pass  # Don't show anything

    elif result_type == "file":
        content = parsed.get("content", "")
        if content:
            st.code(
                content[:3000] + ("..." if len(content) > 3000 else ""), language=None
            )

    elif result_type == "json":
        data = parsed.get("data", {})
        json_str = json.dumps(data, indent=2)
        if len(json_str) < 500:
            st.code(json_str, language="json")
        else:
            with st.expander("Output", expanded=False):
                st.code(json_str, language="json")

    elif result_type == "text":
        content = parsed.get("content", "")
        if content:
            if "\n" in content or len(content) > 100:
                st.markdown(
                    f"<div class='cli-output'><pre>{_escape_html(content[:3000])}</pre></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<code>{_escape_html(content)}</code>", unsafe_allow_html=True
                )

    elif result_type == "empty":
        st.markdown("<div class='cli-output empty'>✓</div>", unsafe_allow_html=True)


def render_event(
    event: dict,
    completed_tool_ids: set | None = None,
    trace_ended: bool = True,
    stream_log_files: dict | None = None,
    tool_id_to_log_file: dict | None = None,
):
    """Render a single event compactly."""
    etype = event.get("type", "unknown")
    ts = event.get("timestamp", 0)
    data = event.get("data", {})
    completed_tool_ids = completed_tool_ids or set()
    stream_log_files = stream_log_files or {}
    tool_id_to_log_file = tool_id_to_log_file or {}

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

        # Get streaming log file - first try direct extraction from command, then fall back to stream_log_files
        log_file = None
        if tool_name.lower() in {"bash", "shell", "execute", "run"}:
            command = tool_input.get("command", "")
            # Try to extract log file path directly from tee-wrapped command
            log_file = extract_log_file_from_command(command)

            # Fall back to stream_log_files mapping if direct extraction failed
            if not log_file:
                original_cmd = command
                if ") 2>&1 | stdbuf -oL tee " in command:
                    original_cmd = command.split(") 2>&1 | stdbuf -oL tee ")[0]
                    if original_cmd.startswith("("):
                        original_cmd = original_cmd[1:]
                log_file = stream_log_files.get(original_cmd)

        if tool_name.lower() in {"bash", "shell", "execute", "run"}:
            render_bash_tool_call(
                tool_input, time_str, is_pending=is_pending, log_file=log_file
            )
        elif tool_name.lower() in {"todowrite", "todo_write", "todo"}:
            render_todo_tool_call(tool_input, time_str)
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
        tool_id = data.get("tool_id", "")

        # Get log file for this tool result
        log_file = tool_id_to_log_file.get(tool_id)

        render_tool_result(result, tool_name, is_error, time_str, log_file=log_file)

    elif etype == "thinking":
        content = data.get("content", "")
        with st.expander(f"{icon} Thinking ({time_str})", expanded=False):
            st.markdown(content[:2000] + ("..." if len(content) > 2000 else ""))

    elif etype == "error":
        st.error(f"{icon} {data.get('message', 'Error')}")

    elif etype == "stop":
        st.caption(f"{icon} Stop: {data.get('reason', '')}")

    elif etype == "result":
        pass  # Skip


def render_trace(trace: dict):
    """Render a complete trace."""
    trace_id = trace.get("id", "?")
    prompt = trace.get("prompt", "")
    model = trace.get("model", "?")
    started = trace.get("started_at", "")
    ended = trace.get("ended_at")
    events = trace.get("events", [])

    status = (
        "✅"
        if ended and not trace.get("error")
        else "❌"
        if trace.get("error")
        else "🔄"
    )

    cols = st.columns([4, 2, 2])
    with cols[0]:
        st.markdown(f"### {status} `{trace_id}` — {model}")
    with cols[1]:
        st.caption(f"Started: {started[:19] if started else '?'}")
    with cols[2]:
        st.caption(f"Events: {len(events)}")

    st.markdown(f"**📝 Prompt:** {prompt}")

    # Build set of tool_call IDs that have received results
    completed_tool_ids = set()
    for event in events:
        if event.get("type") == "tool_result":
            tool_id = event.get("data", {}).get("tool_id")
            if tool_id:
                completed_tool_ids.add(tool_id)

    # Build mapping of original_command -> log_file from bash_stream_start events
    stream_log_files = {}
    for event in events:
        if event.get("type") == "bash_stream_start":
            data = event.get("data", {})
            original_command = data.get("original_command")
            log_file = data.get("log_file")
            if original_command and log_file:
                stream_log_files[original_command] = log_file

    # Build mapping of tool_id -> log_file for tool_result rendering
    tool_id_to_log_file = {}
    for event in events:
        if event.get("type") == "tool_call":
            data = event.get("data", {})
            tool_name = data.get("tool_name", "")
            tool_id = data.get("tool_id", "")
            tool_input = data.get("tool_input", {})

            if tool_name.lower() in {"bash", "shell", "execute", "run"} and tool_id:
                command = tool_input.get("command", "")
                # Extract log file directly from command
                log_file = extract_log_file_from_command(command)
                if log_file:
                    tool_id_to_log_file[tool_id] = log_file

    # Events
    for event in events:
        render_event(
            event,
            completed_tool_ids=completed_tool_ids,
            trace_ended=ended is not None,
            stream_log_files=stream_log_files,
            tool_id_to_log_file=tool_id_to_log_file,
        )

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


def main():
    """Main viewer application."""
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
    if "prev_event_count" not in st.session_state:
        st.session_state.prev_event_count = 0

    # Sidebar with sessions list
    with st.sidebar:
        st.markdown("## 🔍 Sessions")

        runs = find_runs_with_traces(runs_dir)

        if not runs:
            st.caption("No sessions yet")
        else:
            for run in runs:
                name = run["name"]
                is_sel = st.session_state.selected_run == name

                # Parse timestamp and format as relative time
                dt = parse_run_name(name)
                display_name = fmt_relative_time(dt) if dt else name

                if st.button(
                    f"{'▶' if is_sel else '○'} {display_name}",
                    key=f"r_{name}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.selected_run = name
                    st.session_state.trace_idx = 0
                    st.session_state.prev_event_count = 0
                    st.rerun()

    # Main area
    if not st.session_state.selected_run:
        st.markdown(
            """
        <div style="text-align: center; padding: 4rem 0;">
            <h1>🔧 Tinker Agent Viewer</h1>
            <p style="color: #8b949e;">Select a session from the sidebar to view traces</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        trace_file = runs_dir / st.session_state.selected_run / "traces.json"

        if trace_file.exists():
            traces = load_traces(trace_file)

            if traces:
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
                st.warning("Empty trace file")
        else:
            st.error("Trace file not found")

    # Auto-refresh
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=2000, key="trace_refresh")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
