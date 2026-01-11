"""Tests for the agent module."""

import json
from unittest.mock import patch
import pytest

from tinker_agent.agent import (
    _create_validate_on_stop_hook,
    _create_log_tool_output_hook,
    MAX_VALIDATION_RETRIES,
)
from tinker_agent.eval import ValidationResult
import tinker_agent.agent as agent_module


# Test session ID for isolation
TEST_SESSION_ID = "test-session-123"


@pytest.fixture(autouse=True)
def reset_agent_globals():
    """Reset global state before each test to prevent test pollution."""
    agent_module._session_state.clear()
    agent_module._session_state[TEST_SESSION_ID] = {
        "tool_output_chars": 0,
        "validation_retries": 0,
    }
    yield
    agent_module._session_state.clear()


def get_session_retry_count():
    """Helper to get retry count from session state."""
    return agent_module._session_state.get(TEST_SESSION_ID, {}).get("validation_retries", 0)


def get_session_tool_output_chars():
    """Helper to get tool output chars from session state."""
    return agent_module._session_state.get(TEST_SESSION_ID, {}).get("tool_output_chars", 0)


class TestValidateOnStopHook:
    """Tests for the validate_on_stop hook function."""

    @pytest.fixture
    def validate_on_stop(self):
        """Create a validate_on_stop hook for testing."""
        return _create_validate_on_stop_hook(TEST_SESSION_ID)

    @pytest.fixture
    def mock_input_data(self):
        """Mock input data for the Stop hook."""
        return {"hook_event_name": "Stop"}

    @pytest.fixture
    def valid_results_dir(self, tmp_path):
        """Create a valid results directory structure."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        # Create valid base_model.jsonl (40% accuracy)
        base_rows = [
            {
                "index": i,
                "question": f"Q{i}",
                "ground_truth": f"A{i}",
                "completion": f"C{i}",
                "extracted_answer": f"A{i}" if i < 4 else "Wrong",
                "correct": i < 4,
            }
            for i in range(10)
        ]
        (results_dir / "base_model.jsonl").write_text(
            "\n".join(json.dumps(r) for r in base_rows)
        )

        # Create valid trained_model.jsonl (70% accuracy - better than base)
        trained_rows = [
            {
                "index": i,
                "question": f"Q{i}",
                "ground_truth": f"A{i}",
                "completion": f"C{i}",
                "extracted_answer": f"A{i}" if i < 7 else "Wrong",
                "correct": i < 7,
            }
            for i in range(10)
        ]
        (results_dir / "trained_model.jsonl").write_text(
            "\n".join(json.dumps(r) for r in trained_rows)
        )

        # Create valid summary.json
        summary = {
            "tinker_run_id": "run_123",
            "log_path": "results/logs/run_123",
            "wandb_url": "https://wandb.ai/project/run_123",
            "baseline_score": 0.4,
            "trained_score": 0.7,
        }
        (results_dir / "summary.json").write_text(json.dumps(summary))

        return tmp_path

    @pytest.fixture
    def invalid_results_dir(self, tmp_path):
        """Create an invalid results directory (missing files)."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Missing everything - validation will fail
        return tmp_path

    @pytest.fixture
    def trained_not_better_dir(self, tmp_path):
        """Create results where trained model doesn't outperform base."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        # Base model: 60% accuracy
        base_rows = [
            {
                "index": i,
                "question": f"Q{i}",
                "ground_truth": f"A{i}",
                "completion": f"C{i}",
                "extracted_answer": f"A{i}" if i < 6 else "Wrong",
                "correct": i < 6,
            }
            for i in range(10)
        ]
        (results_dir / "base_model.jsonl").write_text(
            "\n".join(json.dumps(r) for r in base_rows)
        )

        # Trained model: 50% accuracy - WORSE than base
        trained_rows = [
            {
                "index": i,
                "question": f"Q{i}",
                "ground_truth": f"A{i}",
                "completion": f"C{i}",
                "extracted_answer": f"A{i}" if i < 5 else "Wrong",
                "correct": i < 5,
            }
            for i in range(10)
        ]
        (results_dir / "trained_model.jsonl").write_text(
            "\n".join(json.dumps(r) for r in trained_rows)
        )

        summary = {
            "tinker_run_id": "run_123",
            "log_path": "results/logs/run_123",
            "wandb_url": "https://wandb.ai/project/run_123",
            "baseline_score": 0.6,
            "trained_score": 0.5,
        }
        (results_dir / "summary.json").write_text(json.dumps(summary))

        return tmp_path

    @pytest.mark.asyncio
    async def test_hook_allows_stop_when_valid(
        self, validate_on_stop, mock_input_data, valid_results_dir, monkeypatch
    ):
        """Hook should return empty dict when validation passes."""
        monkeypatch.chdir(valid_results_dir)

        result = await validate_on_stop(mock_input_data, "tool_123", {})

        assert result == {}

    @pytest.mark.asyncio
    async def test_hook_continues_when_missing_files(
        self, validate_on_stop, mock_input_data, invalid_results_dir, monkeypatch
    ):
        """Hook should return continue decision when files are missing."""
        monkeypatch.chdir(invalid_results_dir)

        result = await validate_on_stop(mock_input_data, "tool_123", {})

        assert "hookSpecificOutput" in result
        output = result["hookSpecificOutput"]
        assert output["hookEventName"] == "Stop"
        assert output["decision"] == "continue"
        assert "Validation failed" in output["updatedStopReason"]

    @pytest.mark.asyncio
    async def test_hook_continues_when_trained_not_better(
        self, validate_on_stop, mock_input_data, trained_not_better_dir, monkeypatch
    ):
        """Hook should return continue decision when trained model isn't better."""
        monkeypatch.chdir(trained_not_better_dir)

        result = await validate_on_stop(mock_input_data, "tool_123", {})

        assert "hookSpecificOutput" in result
        output = result["hookSpecificOutput"]
        assert output["decision"] == "continue"
        assert "does not outperform" in output["updatedStopReason"]

    @pytest.mark.asyncio
    async def test_hook_includes_error_summary(
        self, validate_on_stop, mock_input_data, invalid_results_dir, monkeypatch
    ):
        """Hook should include error details in the stop reason."""
        monkeypatch.chdir(invalid_results_dir)

        result = await validate_on_stop(mock_input_data, "tool_123", {})

        output = result["hookSpecificOutput"]
        # Should mention at least one missing file
        assert any(
            term in output["updatedStopReason"]
            for term in ["base_model", "trained_model", "summary", "logs", "Missing"]
        )

    @pytest.mark.asyncio
    async def test_hook_limits_errors_to_three(
        self, validate_on_stop, mock_input_data, tmp_path, monkeypatch
    ):
        """Hook should only include first 3 errors in summary."""
        # Create results dir with many errors
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # All files missing = 4+ errors, but hook should only show 3
        monkeypatch.chdir(tmp_path)

        result = await validate_on_stop(mock_input_data, "tool_123", {})

        output = result["hookSpecificOutput"]
        # Count semicolons (error separators) - should be at most 2 (for 3 errors)
        error_count = output["updatedStopReason"].count(";")
        assert error_count <= 2


class TestValidateOnStopCallsEval:
    """Tests to verify validate_on_stop actually calls eval.validate_results."""

    @pytest.fixture
    def validate_on_stop(self):
        """Create a validate_on_stop hook for testing."""
        return _create_validate_on_stop_hook(TEST_SESSION_ID)

    @pytest.fixture
    def mock_input_data(self):
        """Mock input data for the Stop hook."""
        return {"hook_event_name": "Stop"}

    @pytest.mark.asyncio
    async def test_validate_results_is_called(self, validate_on_stop, mock_input_data):
        """Verify that validate_results from eval.py is actually called."""
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=True, errors=[], base_score=0.4, trained_score=0.7
            )

            await validate_on_stop(mock_input_data, "tool_123", {})

            # Assert validate_results was called exactly once
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_results_called_with_path(self, validate_on_stop, mock_input_data):
        """Verify validate_results is called with a Path object."""
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=True, errors=[], base_score=0.5, trained_score=0.8
            )

            await validate_on_stop(mock_input_data, "tool_123", {})

            # Get the actual call arguments - should be a Path
            call_args = mock_validate.call_args
            from pathlib import Path
            assert isinstance(call_args[0][0], Path)

    @pytest.mark.asyncio
    async def test_hook_uses_validation_result_valid_flag(self, mock_input_data):
        """Verify hook respects the valid flag from ValidationResult."""
        # Test with valid=True
        validate_on_stop = _create_validate_on_stop_hook(TEST_SESSION_ID + "-valid")
        agent_module._session_state[TEST_SESSION_ID + "-valid"] = {
            "tool_output_chars": 0,
            "validation_retries": 0,
        }
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=True, errors=[], base_score=0.4, trained_score=0.7
            )
            result = await validate_on_stop(mock_input_data, "tool_123", {})
            assert result == {}  # Should allow stop

        # Test with valid=False (new session to reset retry count)
        validate_on_stop2 = _create_validate_on_stop_hook(TEST_SESSION_ID + "-invalid")
        agent_module._session_state[TEST_SESSION_ID + "-invalid"] = {
            "tool_output_chars": 0,
            "validation_retries": 0,
        }
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=False,
                errors=["Something went wrong"],
                base_score=0.4,
                trained_score=0.3,
            )
            result = await validate_on_stop2(mock_input_data, "tool_123", {})
            assert "hookSpecificOutput" in result  # Should continue
            assert result["hookSpecificOutput"]["decision"] == "continue"

    @pytest.mark.asyncio
    async def test_hook_passes_errors_from_validation(self, validate_on_stop, mock_input_data):
        """Verify hook includes error messages from ValidationResult in stop reason."""
        specific_error = (
            "Trained model (45.00%) does not outperform base model (50.00%)"
        )
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=False,
                errors=[specific_error],
                base_score=0.5,
                trained_score=0.45,
            )

            result = await validate_on_stop(mock_input_data, "tool_123", {})

            assert specific_error in result["hookSpecificOutput"]["updatedStopReason"]

    @pytest.mark.asyncio
    async def test_hook_handles_multiple_validation_errors(self, validate_on_stop, mock_input_data):
        """Verify hook handles multiple errors from ValidationResult."""
        errors = [
            "Missing required path: base_model.jsonl",
            "Missing required path: trained_model.jsonl",
            "Missing required path: summary.json",
            "Missing required path: logs/",
        ]
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=False, errors=errors, base_score=None, trained_score=None
            )

            result = await validate_on_stop(mock_input_data, "tool_123", {})

            # Should include first 3 errors (semicolon-separated)
            stop_reason = result["hookSpecificOutput"]["updatedStopReason"]
            assert "base_model.jsonl" in stop_reason
            assert "trained_model.jsonl" in stop_reason
            assert "summary.json" in stop_reason
            # 4th error might be truncated
            assert stop_reason.count(";") <= 2  # max 3 errors = max 2 semicolons


class TestValidateOnStopRetryLogic:
    """Tests for the retry logic in validate_on_stop."""

    @pytest.fixture
    def validate_on_stop(self):
        """Create a validate_on_stop hook for testing."""
        return _create_validate_on_stop_hook(TEST_SESSION_ID)

    @pytest.fixture
    def mock_input_data(self):
        """Mock input data for the Stop hook."""
        return {"hook_event_name": "Stop"}

    @pytest.mark.asyncio
    async def test_first_failure_returns_continue(self, validate_on_stop, mock_input_data):
        """First validation failure should return continue decision."""
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=False,
                errors=["Something failed"],
                base_score=None,
                trained_score=None,
            )

            result = await validate_on_stop(mock_input_data, "tool_123", {})

            assert "hookSpecificOutput" in result
            assert result["hookSpecificOutput"]["decision"] == "continue"
            assert get_session_retry_count() == 1

    @pytest.mark.asyncio
    async def test_second_failure_allows_stop(self, validate_on_stop, mock_input_data):
        """After MAX_VALIDATION_RETRIES (2), hook should allow stop."""
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=False,
                errors=["Something failed"],
                base_score=None,
                trained_score=None,
            )

            # First failure - should continue
            result1 = await validate_on_stop(mock_input_data, "tool_123", {})
            assert "hookSpecificOutput" in result1
            assert result1["hookSpecificOutput"]["decision"] == "continue"
            assert get_session_retry_count() == 1

            # Second failure - should allow stop (max retries reached)
            result2 = await validate_on_stop(mock_input_data, "tool_123", {})
            assert result2 == {}  # Empty dict = allow stop
            assert get_session_retry_count() == MAX_VALIDATION_RETRIES

    @pytest.mark.asyncio
    async def test_success_resets_nothing_just_allows_stop(self, validate_on_stop, mock_input_data):
        """Successful validation should just allow stop without affecting retry count."""
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=True, errors=[], base_score=0.4, trained_score=0.7
            )

            result = await validate_on_stop(mock_input_data, "tool_123", {})

            assert result == {}
            # Retry count should still be 0 (success doesn't increment)
            assert get_session_retry_count() == 0

    @pytest.mark.asyncio
    async def test_failure_then_success_allows_stop(self, validate_on_stop, mock_input_data):
        """If validation fails once then succeeds, should allow stop."""
        with patch("tinker_agent.agent.validate_results") as mock_validate:
            # First call fails
            mock_validate.return_value = ValidationResult(
                valid=False,
                errors=["Something failed"],
                base_score=None,
                trained_score=None,
            )
            result1 = await validate_on_stop(mock_input_data, "tool_123", {})
            assert "hookSpecificOutput" in result1
            assert get_session_retry_count() == 1

            # Second call succeeds
            mock_validate.return_value = ValidationResult(
                valid=True, errors=[], base_score=0.4, trained_score=0.7
            )
            result2 = await validate_on_stop(mock_input_data, "tool_123", {})
            assert result2 == {}  # Allow stop
            # Retry count stays at 1 (success doesn't reset or increment)
            assert get_session_retry_count() == 1

    @pytest.mark.asyncio
    async def test_max_validation_retries_constant(self):
        """Verify MAX_VALIDATION_RETRIES is set to expected value."""
        assert MAX_VALIDATION_RETRIES == 2


class TestLogToolOutputHook:
    """Tests for the log_tool_output PostToolUse hook."""

    @pytest.fixture
    def log_tool_output(self):
        """Create a log_tool_output hook for testing."""
        return _create_log_tool_output_hook(TEST_SESSION_ID)

    @pytest.mark.asyncio
    async def test_log_tool_output_returns_empty_dict(self, log_tool_output):
        """PostToolUse hook should always return empty dict (pass-through)."""
        input_data = {
            "tool_name": "Read",
            "tool_result": "some file contents",
        }

        result = await log_tool_output(input_data, "tool_123", {})

        assert result == {}

    @pytest.mark.asyncio
    async def test_log_tool_output_tracks_cumulative_chars(self, log_tool_output):
        """Hook should track cumulative character count."""
        input_data_1 = {"tool_name": "Read", "tool_result": "a" * 100}
        input_data_2 = {"tool_name": "Write", "tool_result": "b" * 200}

        await log_tool_output(input_data_1, "tool_1", {})
        assert get_session_tool_output_chars() == 100

        await log_tool_output(input_data_2, "tool_2", {})
        assert get_session_tool_output_chars() == 300

    @pytest.mark.asyncio
    async def test_log_tool_output_handles_missing_fields(self, log_tool_output):
        """Hook should handle missing tool_name or tool_result gracefully."""
        input_data = {}  # Missing both fields

        # Should not raise
        result = await log_tool_output(input_data, "tool_123", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_log_tool_output_handles_large_output(self, log_tool_output, capsys):
        """Hook should handle large outputs and print warning."""
        large_result = "x" * 60000  # > 50000 chars
        input_data = {"tool_name": "Read", "tool_result": large_result}

        await log_tool_output(input_data, "tool_123", {})

        # Check that it printed (we can't easily check rich output, but ensure no crash)
        assert get_session_tool_output_chars() == 60000


class TestHookSignatures:
    """Tests to verify hook functions have correct signatures for claude_agent_sdk."""

    def test_validate_on_stop_is_async(self):
        """validate_on_stop must be an async function for the SDK."""
        import asyncio

        validate_on_stop = _create_validate_on_stop_hook("test-sig")
        assert asyncio.iscoroutinefunction(validate_on_stop)

    def test_log_tool_output_is_async(self):
        """log_tool_output must be an async function for the SDK."""
        import asyncio

        log_tool_output = _create_log_tool_output_hook("test-sig")
        assert asyncio.iscoroutinefunction(log_tool_output)

    def test_validate_on_stop_accepts_three_args(self):
        """Hook must accept (input_data, tool_use_id, context)."""
        import inspect

        validate_on_stop = _create_validate_on_stop_hook("test-sig")
        sig = inspect.signature(validate_on_stop)
        params = list(sig.parameters.keys())
        assert len(params) == 3
        assert params == ["input_data", "tool_use_id", "context"]

    def test_log_tool_output_accepts_three_args(self):
        """Hook must accept (input_data, tool_use_id, context)."""
        import inspect

        log_tool_output = _create_log_tool_output_hook("test-sig")
        sig = inspect.signature(log_tool_output)
        params = list(sig.parameters.keys())
        assert len(params) == 3
        assert params == ["input_data", "tool_use_id", "context"]
