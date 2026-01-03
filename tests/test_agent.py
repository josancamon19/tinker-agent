"""Tests for the agent module."""

import json
import pytest

from tinker_agent.agent import validate_on_stop


class TestValidateOnStopHook:
    """Tests for the validate_on_stop hook function."""

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
        self, mock_input_data, valid_results_dir, monkeypatch
    ):
        """Hook should return empty dict when validation passes."""
        monkeypatch.chdir(valid_results_dir)

        result = await validate_on_stop(mock_input_data, "tool_123", {})

        assert result == {}

    @pytest.mark.asyncio
    async def test_hook_continues_when_missing_files(
        self, mock_input_data, invalid_results_dir, monkeypatch
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
        self, mock_input_data, trained_not_better_dir, monkeypatch
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
        self, mock_input_data, invalid_results_dir, monkeypatch
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
    async def test_hook_limits_errors_to_three(self, mock_input_data, tmp_path, monkeypatch):
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

