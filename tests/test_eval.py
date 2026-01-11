"""Tests for the eval module."""

import json
import pytest
from pathlib import Path

from tinker_agent.eval import (
    validate_jsonl_row_rl,
    validate_jsonl_row_sft,
    validate_jsonl_file,
    validate_results,
    detect_task_type_from_row,
    REQUIRED_FIELDS_RL,
    REQUIRED_FIELDS_SFT,
)


class TestDetectTaskType:
    """Tests for task type detection."""

    def test_detects_rl_task(self):
        """Should detect RL task from row with 'correct' and 'question'."""
        row = {"question": "Q1", "correct": True, "other": "data"}
        assert detect_task_type_from_row(row) == "rl"

    def test_detects_sft_task(self):
        """Should detect SFT task from row with 'nll' and 'prompt'."""
        row = {"prompt": "P1", "nll": 1.5, "other": "data"}
        assert detect_task_type_from_row(row) == "sft"

    def test_returns_none_for_unknown(self):
        """Should return None for unknown schema."""
        row = {"foo": "bar"}
        assert detect_task_type_from_row(row) is None


class TestValidateJsonlRowRL:
    """Tests for validate_jsonl_row_rl function."""

    def test_valid_row(self):
        """A row with all required fields should pass."""
        row = {
            "index": 0,
            "question": "What is 2+2?",
            "ground_truth": "4",
            "completion": "The answer is 4.",
            "extracted_answer": "4",
            "correct": True,
        }
        errors = validate_jsonl_row_rl(row, 0)
        assert errors == []

    def test_missing_fields(self):
        """A row missing fields should report them."""
        row = {"index": 0, "question": "What is 2+2?"}
        errors = validate_jsonl_row_rl(row, 0)
        assert len(errors) == 1
        assert "missing fields" in errors[0]

    def test_correct_must_be_boolean(self):
        """The 'correct' field must be a boolean."""
        row = {
            "index": 0,
            "question": "What is 2+2?",
            "ground_truth": "4",
            "completion": "The answer is 4.",
            "extracted_answer": "4",
            "correct": "yes",  # Wrong type
        }
        errors = validate_jsonl_row_rl(row, 0)
        assert len(errors) == 1
        assert "must be boolean" in errors[0]

    def test_correct_false_is_valid(self):
        """correct=False is a valid boolean value."""
        row = {
            "index": 0,
            "question": "What is 2+2?",
            "ground_truth": "4",
            "completion": "The answer is 5.",
            "extracted_answer": "5",
            "correct": False,
        }
        errors = validate_jsonl_row_rl(row, 0)
        assert errors == []


class TestValidateJsonlRowSFT:
    """Tests for validate_jsonl_row_sft function."""

    def test_valid_row(self):
        """A row with all required fields should pass."""
        row = {
            "index": 0,
            "prompt": "Complete this:",
            "completion": "Done!",
            "nll": 1.234,
        }
        errors = validate_jsonl_row_sft(row, 0)
        assert errors == []

    def test_missing_fields(self):
        """A row missing fields should report them."""
        row = {"index": 0, "prompt": "Complete this:"}
        errors = validate_jsonl_row_sft(row, 0)
        assert len(errors) == 1
        assert "missing fields" in errors[0]

    def test_nll_must_be_numeric(self):
        """The 'nll' field must be numeric."""
        row = {
            "index": 0,
            "prompt": "Complete this:",
            "completion": "Done!",
            "nll": "not a number",
        }
        errors = validate_jsonl_row_sft(row, 0)
        assert len(errors) == 1
        assert "must be numeric" in errors[0]

    def test_nll_int_is_valid(self):
        """Integer NLL is valid."""
        row = {
            "index": 0,
            "prompt": "Complete this:",
            "completion": "Done!",
            "nll": 2,
        }
        errors = validate_jsonl_row_sft(row, 0)
        assert errors == []


class TestValidateJsonlFile:
    """Tests for validate_jsonl_file function."""

    def test_file_not_found(self, tmp_path):
        """Missing file should return error."""
        errors, score, task_type = validate_jsonl_file(tmp_path / "missing.jsonl")
        assert len(errors) == 1
        assert "File not found" in errors[0]
        assert score is None

    def test_empty_file(self, tmp_path):
        """Empty file should return error."""
        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")
        errors, score, task_type = validate_jsonl_file(file_path)
        assert any("empty" in e.lower() for e in errors)
        assert score is None

    def test_valid_rl_file(self, tmp_path):
        """Valid RL JSONL file should return accuracy score."""
        file_path = tmp_path / "valid.jsonl"
        rows = [
            {
                "index": 0,
                "question": "Q1",
                "ground_truth": "A1",
                "completion": "C1",
                "extracted_answer": "A1",
                "correct": True,
            },
            {
                "index": 1,
                "question": "Q2",
                "ground_truth": "A2",
                "completion": "C2",
                "extracted_answer": "A2",
                "correct": True,
            },
            {
                "index": 2,
                "question": "Q3",
                "ground_truth": "A3",
                "completion": "C3",
                "extracted_answer": "X",
                "correct": False,
            },
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows))

        errors, score, task_type = validate_jsonl_file(file_path)
        assert errors == []
        assert score == pytest.approx(2 / 3)
        assert task_type == "rl"

    def test_valid_sft_file(self, tmp_path):
        """Valid SFT JSONL file should return mean NLL."""
        file_path = tmp_path / "valid_sft.jsonl"
        rows = [
            {"index": 0, "prompt": "P1", "completion": "C1", "nll": 1.0},
            {"index": 1, "prompt": "P2", "completion": "C2", "nll": 2.0},
            {"index": 2, "prompt": "P3", "completion": "C3", "nll": 3.0},
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows))

        errors, score, task_type = validate_jsonl_file(file_path)
        assert errors == []
        assert score == pytest.approx(2.0)  # mean of 1, 2, 3
        assert task_type == "sft"

    def test_invalid_json_line(self, tmp_path):
        """Invalid JSON should be caught."""
        file_path = tmp_path / "invalid.jsonl"
        file_path.write_text('{"valid": true}\nnot valid json\n{"also": "valid"}')

        errors, score, task_type = validate_jsonl_file(file_path)
        assert any("invalid JSON" in e for e in errors)

    def test_100_percent_accuracy(self, tmp_path):
        """All correct should give 100% accuracy."""
        file_path = tmp_path / "perfect.jsonl"
        rows = [
            {
                "index": i,
                "question": f"Q{i}",
                "ground_truth": "A",
                "completion": "C",
                "extracted_answer": "A",
                "correct": True,
            }
            for i in range(5)
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows))

        errors, score, task_type = validate_jsonl_file(file_path)
        assert errors == []
        assert score == 1.0

    def test_0_percent_accuracy(self, tmp_path):
        """All incorrect should give 0% accuracy."""
        file_path = tmp_path / "zero.jsonl"
        rows = [
            {
                "index": i,
                "question": f"Q{i}",
                "ground_truth": "A",
                "completion": "C",
                "extracted_answer": "X",
                "correct": False,
            }
            for i in range(5)
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows))

        errors, score, task_type = validate_jsonl_file(file_path)
        assert errors == []
        assert score == 0.0


class TestValidateResults:
    """Tests for validate_results function."""

    def _create_valid_rl_jsonl(self, path: Path, accuracy: float, count: int = 10):
        """Helper to create a valid RL JSONL file with given accuracy."""
        correct_count = int(count * accuracy)
        rows = []
        for i in range(count):
            rows.append(
                {
                    "index": i,
                    "question": f"Question {i}",
                    "ground_truth": f"Answer {i}",
                    "completion": f"Completion {i}",
                    "extracted_answer": f"Answer {i}" if i < correct_count else "Wrong",
                    "correct": i < correct_count,
                }
            )
        path.write_text("\n".join(json.dumps(r) for r in rows))

    def _create_valid_sft_jsonl(self, path: Path, mean_nll: float, count: int = 10):
        """Helper to create a valid SFT JSONL file with given mean NLL."""
        rows = []
        for i in range(count):
            rows.append(
                {
                    "index": i,
                    "prompt": f"Prompt {i}",
                    "completion": f"Completion {i}",
                    "nll": mean_nll,  # All same NLL for simplicity
                }
            )
        path.write_text("\n".join(json.dumps(r) for r in rows))

    def _create_valid_summary_rl(self, path: Path):
        """Helper to create a valid summary.json for RL task."""
        summary = {
            "task_type": "rl",
            "tinker_run_id": "run_123",
            "checkpoint_path": "checkpoints/run_123",
            "log_path": "results/logs/run_123",
            "wandb_url": "https://wandb.ai/project/run_123",
            "baseline_score": 0.4,
            "trained_score": 0.7,
        }
        path.write_text(json.dumps(summary))

    def _create_valid_summary_sft(self, path: Path):
        """Helper to create a valid summary.json for SFT task."""
        summary = {
            "task_type": "sft",
            "tinker_run_id": "run_123",
            "checkpoint_path": "checkpoints/run_123",
            "log_path": "results/logs/run_123",
            "wandb_url": "https://wandb.ai/project/run_123",
            "baseline_nll": 2.5,
            "trained_nll": 1.8,
        }
        path.write_text(json.dumps(summary))

    def test_missing_results_dir(self, tmp_path):
        """Missing results directory should fail."""
        result = validate_results(tmp_path / "nonexistent")
        assert not result.valid
        assert any("not found" in e.lower() for e in result.errors)

    def test_missing_all_files(self, tmp_path):
        """Missing all required files should report each."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        result = validate_results(results_dir)
        assert not result.valid
        assert any("base_model.jsonl" in e for e in result.errors)
        assert any("trained_model.jsonl" in e for e in result.errors)
        assert any("summary.json" in e for e in result.errors)
        assert any("logs" in e for e in result.errors)

    def test_valid_rl_results(self, tmp_path):
        """Complete valid RL results should pass."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_rl_jsonl(results_dir / "base_model.jsonl", accuracy=0.4)
        self._create_valid_rl_jsonl(results_dir / "trained_model.jsonl", accuracy=0.7)
        self._create_valid_summary_rl(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert result.valid, f"Errors: {result.errors}"
        assert result.task_type == "rl"
        assert result.base_score == pytest.approx(0.4)
        assert result.trained_score == pytest.approx(0.7)

    def test_valid_sft_results(self, tmp_path):
        """Complete valid SFT results should pass."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_sft_jsonl(results_dir / "base_model.jsonl", mean_nll=2.5)
        self._create_valid_sft_jsonl(results_dir / "trained_model.jsonl", mean_nll=1.8)
        self._create_valid_summary_sft(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert result.valid, f"Errors: {result.errors}"
        assert result.task_type == "sft"
        assert result.base_nll == pytest.approx(2.5)
        assert result.trained_nll == pytest.approx(1.8)

    def test_trained_not_better_than_base_rl(self, tmp_path):
        """Trained model must outperform base model for RL."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_rl_jsonl(results_dir / "base_model.jsonl", accuracy=0.6)
        self._create_valid_rl_jsonl(
            results_dir / "trained_model.jsonl", accuracy=0.5
        )  # Worse!
        self._create_valid_summary_rl(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("does not outperform" in e for e in result.errors)

    def test_trained_not_better_than_base_sft(self, tmp_path):
        """Trained model must have lower NLL for SFT."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_sft_jsonl(results_dir / "base_model.jsonl", mean_nll=2.0)
        self._create_valid_sft_jsonl(
            results_dir / "trained_model.jsonl", mean_nll=2.5
        )  # Worse!
        self._create_valid_summary_sft(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("not better" in e for e in result.errors)

    def test_trained_equal_to_base_fails_rl(self, tmp_path):
        """Trained model equal to base should fail (must be strictly better)."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_rl_jsonl(results_dir / "base_model.jsonl", accuracy=0.5)
        self._create_valid_rl_jsonl(
            results_dir / "trained_model.jsonl", accuracy=0.5
        )  # Same
        self._create_valid_summary_rl(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("does not outperform" in e for e in result.errors)

    def test_summary_missing_keys(self, tmp_path):
        """Summary with missing keys should fail."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_rl_jsonl(results_dir / "base_model.jsonl", accuracy=0.4)
        self._create_valid_rl_jsonl(results_dir / "trained_model.jsonl", accuracy=0.7)

        # Incomplete summary
        (results_dir / "summary.json").write_text(json.dumps({"tinker_run_id": "123"}))

        result = validate_results(results_dir)
        assert not result.valid
        assert any("missing keys" in e for e in result.errors)

    def test_summary_invalid_json(self, tmp_path):
        """Invalid JSON in summary should fail."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_rl_jsonl(results_dir / "base_model.jsonl", accuracy=0.4)
        self._create_valid_rl_jsonl(results_dir / "trained_model.jsonl", accuracy=0.7)
        (results_dir / "summary.json").write_text("not json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("invalid JSON" in e for e in result.errors)


class TestRequiredFields:
    """Tests for the REQUIRED_FIELDS constants."""

    def test_required_fields_rl_defined(self):
        """All expected RL fields should be in REQUIRED_FIELDS_RL."""
        expected = {
            "index",
            "question",
            "ground_truth",
            "completion",
            "extracted_answer",
            "correct",
        }
        assert REQUIRED_FIELDS_RL == expected

    def test_required_fields_sft_defined(self):
        """All expected SFT fields should be in REQUIRED_FIELDS_SFT."""
        expected = {
            "index",
            "prompt",
            "completion",
            "nll",
        }
        assert REQUIRED_FIELDS_SFT == expected
