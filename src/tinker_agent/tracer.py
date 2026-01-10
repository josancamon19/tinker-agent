"""Simple JSONL tracer for agent executions."""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """A single event in a trace."""

    type: str  # "message", "tool_call", "tool_result", "thinking", "error", "stop"
    timestamp: float
    data: dict[str, Any]


@dataclass
class Trace:
    """A complete trace of an agent run."""

    id: str
    started_at: str
    prompt: str
    model: str
    events: list[TraceEvent] = field(default_factory=list)
    ended_at: str | None = None
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects and writes traces to JSONL format."""

    def __init__(self, output_path: str | Path = "traces.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_trace: Trace | None = None

    def start_trace(
        self, prompt: str, model: str = "unknown", metadata: dict | None = None
    ) -> str:
        """Start a new trace. Returns the trace ID."""
        trace_id = str(uuid.uuid4())[:8]
        self.current_trace = Trace(
            id=trace_id,
            started_at=datetime.now().isoformat(),
            prompt=prompt,
            model=model,
            metadata=metadata or {},
        )
        self._write_trace()
        return trace_id

    def add_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Add an event to the current trace."""
        if not self.current_trace:
            return

        event = TraceEvent(type=event_type, timestamp=time.time(), data=data)
        self.current_trace.events.append(event)
        self._write_trace()

    def log_message(self, role: str, content: str) -> None:
        """Log a message event."""
        self.add_event("message", {"role": role, "content": content})

    def log_tool_call(
        self, tool_name: str, tool_input: dict, tool_id: str = ""
    ) -> None:
        """Log a tool call event."""
        self.add_event(
            "tool_call",
            {"tool_name": tool_name, "tool_input": tool_input, "tool_id": tool_id},
        )

    def log_tool_result(
        self, tool_id: str, result: Any, is_error: bool = False
    ) -> None:
        """Log a tool result event."""
        # Truncate large results for storage
        result_str = str(result)
        if len(result_str) > 10000:
            result_str = result_str[:10000] + "\n... [truncated]"

        self.add_event(
            "tool_result",
            {"tool_id": tool_id, "result": result_str, "is_error": is_error},
        )

    def log_thinking(self, thinking: str) -> None:
        """Log a thinking block."""
        self.add_event("thinking", {"content": thinking})

    def log_error(self, error: str) -> None:
        """Log an error."""
        self.add_event("error", {"message": error})

    def end_trace(self, result: str | None = None, error: str | None = None) -> None:
        """End the current trace."""
        if not self.current_trace:
            return

        self.current_trace.ended_at = datetime.now().isoformat()
        self.current_trace.result = result
        self.current_trace.error = error
        self._write_trace()
        self.current_trace = None

    def _write_trace(self) -> None:
        """Write/update the current trace to the JSONL file."""
        if not self.current_trace:
            return

        # Read existing traces
        traces: dict[str, dict] = {}
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trace = json.loads(line)
                        traces[trace["id"]] = trace

        # Update current trace
        trace_dict = asdict(self.current_trace)
        # Convert TraceEvent dataclasses to dicts
        trace_dict["events"] = [asdict(e) for e in self.current_trace.events]
        traces[self.current_trace.id] = trace_dict

        # Write all traces back (keeps file consistent)
        with open(self.output_path, "w") as f:
            for trace in traces.values():
                f.write(json.dumps(trace) + "\n")

    def record_message(self, message: Any, verbose: bool = False) -> None:
        """Record a message from the claude_agent_sdk stream."""
        if not self.current_trace:
            return

        # Handle final result
        if hasattr(message, "result") and message.result:
            self.add_event("result", {"content": message.result})
            return

        # Handle content blocks
        if hasattr(message, "content"):
            for block in message.content:
                block_type = getattr(block, "type", None)

                # Text content
                if hasattr(block, "text") and not block_type:
                    self.log_message("assistant", block.text)

                # Thinking block
                elif block_type == "thinking" or hasattr(block, "thinking"):
                    thinking = getattr(block, "thinking", None) or getattr(
                        block, "text", ""
                    )
                    if thinking:
                        self.log_thinking(thinking)

                # Tool use block
                elif block_type == "tool_use" or hasattr(block, "name"):
                    tool_name = getattr(block, "name", "unknown")
                    tool_input = getattr(block, "input", {})
                    tool_id = getattr(block, "id", "")
                    self.log_tool_call(tool_name, tool_input, tool_id)

                # Tool result block
                elif block_type == "tool_result":
                    tool_use_id = getattr(block, "tool_use_id", "")
                    content = getattr(block, "content", "")
                    is_error = getattr(block, "is_error", False)

                    if isinstance(content, list):
                        content = "\n".join(getattr(c, "text", str(c)) for c in content)

                    self.log_tool_result(tool_use_id, content, is_error)

        # Handle error messages
        if hasattr(message, "error") and message.error:
            self.log_error(str(message.error))

        # Handle stop reason
        if hasattr(message, "stop_reason") and message.stop_reason:
            self.add_event("stop", {"reason": message.stop_reason})
