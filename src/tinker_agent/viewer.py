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


def render_bash_tool_call(tool_input: dict, time_str: str):
    """Render a Bash tool call with clean, compact UI."""
    command = tool_input.get("command", "")
    description = tool_input.get("description", "")
    timeout = tool_input.get("timeout")
    run_in_bg = tool_input.get("run_in_background", False)

    badges = []
    if timeout:
        badges.append(f"⏱ {fmt_timeout(timeout)}")
    if run_in_bg:
        badges.append("⚡ bg")

    badge_html = " ".join(f"<span class='tool-badge'>{b}</span>" for b in badges)
    desc_html = f"<span class='tool-desc'>{description}</span>" if description else ""

    st.markdown(
        f"<div class='tool-header'><span class='tool-name-badge bash'>Bash</span> {desc_html} <span class='tool-meta'>{badge_html} <span class='tool-time'>{time_str}</span></span></div>",
        unsafe_allow_html=True,
    )

    # Strip tee wrapper for cleaner display
    display_command = command
    if "| stdbuf -oL tee /tmp/tinker_bash_logs/" in command:
        display_command = command.split(") 2>&1 | stdbuf")[0]
        if display_command.startswith("("):
            display_command = display_command[1:]
    st.code(display_command, language="bash")


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


def render_tool_result(result: str, tool_name: str, is_error: bool, time_str: str):
    """Render tool result in a clean, CLI-like format."""
    parsed = parse_tool_result(result)
    result_type = parsed.get("type", "text")

    if result_type == "shell":
        stdout = parsed.get("stdout", "")
        stderr = parsed.get("stderr", "")

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


def render_event(event: dict):
    """Render a single event compactly."""
    etype = event.get("type", "unknown")
    ts = event.get("timestamp", 0)
    data = event.get("data", {})

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

        if tool_name.lower() in {"bash", "shell", "execute", "run"}:
            render_bash_tool_call(tool_input, time_str)
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

    for event in events:
        render_event(event)

    if trace.get("error"):
        st.error(f"❌ {trace['error']}")


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
                if st.button(
                    f"{'▶' if is_sel else '○'} {name} ({run['trace_count']})",
                    key=f"r_{name}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.selected_run = name
                    st.session_state.trace_idx = 0
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
