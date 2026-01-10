"""Streamlit trace viewer for agent executions."""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

# Page config
st.set_page_config(
    page_title="Trace Viewer",
    page_icon="🔍",
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

    # Build metadata badges
    badges = []
    if timeout:
        badges.append(f"⏱ {fmt_timeout(timeout)}")
    if run_in_bg:
        badges.append("⚡ bg")

    badge_html = " ".join(f"<span class='tool-badge'>{b}</span>" for b in badges)

    # Header with description and badges
    desc_html = f"<span class='tool-desc'>{description}</span>" if description else ""

    st.markdown(
        f"""<div class='tool-header'>
            <span class='tool-name-badge bash'>Bash</span>
            {desc_html}
            <span class='tool-meta'>{badge_html} <span class='tool-time'>{time_str}</span></span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Command in a code block
    st.code(command, language="bash")


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
        f"""<div class='tool-header'>
            <span class='tool-name-badge todo'>Todo</span>
            <span class='tool-desc'>{len(todos)} items</span>
            <span class='tool-meta'><span class='tool-badge'>{summary}</span> <span class='tool-time'>{time_str}</span></span>
        </div>""",
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

        # Special rendering for specific tools
        if tool_name.lower() in {"bash", "shell", "execute", "run"}:
            render_bash_tool_call(tool_input, time_str)
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

        # Command tools (Bash, Shell, etc.) show output expanded
        is_command_tool = tool_name.lower() in {"bash", "shell", "execute", "run"}

        if is_command_tool:
            # Show command output prominently
            icon_display = "🔴" if is_error else "📺"
            label = "stderr" if is_error else "stdout"
            st.markdown(
                f"<div class='event-header'>{icon_display} <b>{label}</b> <span>{time_str}</span></div>",
                unsafe_allow_html=True,
            )
            # Show first 5000 chars expanded for command output
            display_result = result[:5000] + (
                "\n... [truncated]" if len(result) > 5000 else ""
            )
            st.code(display_result, language="bash")
        else:
            # Other tools: show collapsed
            st.markdown(
                f"<div class='event-header'>{icon} Result {'❌' if is_error else ''} <span>{time_str}</span></div>",
                unsafe_allow_html=True,
            )
            with st.expander("Output", expanded=False):
                st.code(
                    result[:3000] + ("..." if len(result) > 3000 else ""), language=None
                )

    elif etype == "thinking":
        content = data.get("content", "")
        with st.expander(f"{icon} Thinking ({time_str})", expanded=False):
            st.markdown(content[:2000] + ("..." if len(content) > 2000 else ""))

    elif etype == "error":
        st.error(f"{icon} {data.get('message', 'Error')}")

    elif etype == "stop":
        st.caption(f"{icon} Stop: {data.get('reason', '')}")

    elif etype == "result":
        st.success(f"{icon} {data.get('content', '')[:500]}")


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

    # Events
    for event in events:
        render_event(event)

    # Final result/error
    if trace.get("result"):
        st.success(f"✅ {trace['result'][:500]}")
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

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Traces")
        st.caption("Auto-refresh: 500ms")

        st.divider()

        # Runs list
        runs = find_runs_with_traces(runs_dir)

        if not runs:
            st.caption(f"No traces in {runs_dir}")
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
                    st.session_state.prev_event_count = 0  # Reset scroll tracking
                    st.rerun()

    # Main area
    if st.session_state.selected_run:
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
                        st.session_state.prev_event_count = 0  # Reset scroll tracking
                    st.session_state.trace_idx = idx
                else:
                    idx = 0

                render_trace(traces[idx])
            else:
                st.warning("Empty trace file")
        else:
            st.error("Trace file not found")
    else:
        st.markdown("### 👈 Select a run")
        st.caption("Auto-refreshes to show live execution")

    # Auto-refresh using streamlit-autorefresh (more reliable than JS/meta refresh)
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=500, key="trace_refresh")
    except ImportError:
        # Fallback: manual refresh hint
        st.caption(
            "Install `streamlit-autorefresh` for auto-refresh: `pip install streamlit-autorefresh`"
        )


if __name__ == "__main__":
    main()
