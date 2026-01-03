"""Tests for the eval module."""

import json
import pytest
from pathlib import Path

from tinker_agent.eval import (
    validate_jsonl_row,
    validate_jsonl_file,
    validate_results,
    REQUIRED_FIELDS,
)


class TestValidateJsonlRow:
    """Tests for validate_jsonl_row function."""

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
        errors = validate_jsonl_row(row, 0)
        assert errors == []

    def test_missing_fields(self):
        """A row missing fields should report them."""
        row = {"index": 0, "question": "What is 2+2?"}
        errors = validate_jsonl_row(row, 0)
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
        errors = validate_jsonl_row(row, 0)
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
        errors = validate_jsonl_row(row, 0)
        assert errors == []


class TestValidateJsonlFile:
    """Tests for validate_jsonl_file function."""

    def test_file_not_found(self, tmp_path):
        """Missing file should return error."""
        errors, score = validate_jsonl_file(tmp_path / "missing.jsonl")
        assert len(errors) == 1
        assert "File not found" in errors[0]
        assert score is None

    def test_empty_file(self, tmp_path):
        """Empty file should return error."""
        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")
        errors, score = validate_jsonl_file(file_path)
        assert any("empty" in e.lower() for e in errors)
        assert score is None

    def test_valid_file(self, tmp_path):
        """Valid JSONL file should return accuracy score."""
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

        errors, score = validate_jsonl_file(file_path)
        assert errors == []
        assert score == pytest.approx(2 / 3)

    def test_invalid_json_line(self, tmp_path):
        """Invalid JSON should be caught."""
        file_path = tmp_path / "invalid.jsonl"
        file_path.write_text('{"valid": true}\nnot valid json\n{"also": "valid"}')

        errors, score = validate_jsonl_file(file_path)
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

        errors, score = validate_jsonl_file(file_path)
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

        errors, score = validate_jsonl_file(file_path)
        assert errors == []
        assert score == 0.0


class TestValidateResults:
    """Tests for validate_results function."""

    def _create_valid_jsonl(self, path: Path, accuracy: float, count: int = 10):
        """Helper to create a valid JSONL file with given accuracy."""
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

    def _create_valid_summary(self, path: Path):
        """Helper to create a valid summary.json."""
        summary = {
            "tinker_run_id": "run_123",
            "log_path": "results/logs/run_123",
            "wandb_url": "https://wandb.ai/project/run_123",
            "baseline_score": 0.4,
            "trained_score": 0.7,
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

    def test_valid_results(self, tmp_path):
        """Complete valid results should pass."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_jsonl(results_dir / "base_model.jsonl", accuracy=0.4)
        self._create_valid_jsonl(results_dir / "trained_model.jsonl", accuracy=0.7)
        self._create_valid_summary(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert result.valid, f"Errors: {result.errors}"
        assert result.base_score == pytest.approx(0.4)
        assert result.trained_score == pytest.approx(0.7)

    def test_trained_not_better_than_base(self, tmp_path):
        """Trained model must outperform base model."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_jsonl(results_dir / "base_model.jsonl", accuracy=0.6)
        self._create_valid_jsonl(
            results_dir / "trained_model.jsonl", accuracy=0.5
        )  # Worse!
        self._create_valid_summary(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("does not outperform" in e for e in result.errors)

    def test_trained_equal_to_base_fails(self, tmp_path):
        """Trained model equal to base should fail (must be strictly better)."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_jsonl(results_dir / "base_model.jsonl", accuracy=0.5)
        self._create_valid_jsonl(
            results_dir / "trained_model.jsonl", accuracy=0.5
        )  # Same
        self._create_valid_summary(results_dir / "summary.json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("does not outperform" in e for e in result.errors)

    def test_summary_missing_keys(self, tmp_path):
        """Summary with missing keys should fail."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        self._create_valid_jsonl(results_dir / "base_model.jsonl", accuracy=0.4)
        self._create_valid_jsonl(results_dir / "trained_model.jsonl", accuracy=0.7)

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

        self._create_valid_jsonl(results_dir / "base_model.jsonl", accuracy=0.4)
        self._create_valid_jsonl(results_dir / "trained_model.jsonl", accuracy=0.7)
        (results_dir / "summary.json").write_text("not json")

        result = validate_results(results_dir)
        assert not result.valid
        assert any("invalid JSON" in e for e in result.errors)


class TestRequiredFields:
    """Tests for the REQUIRED_FIELDS constant."""

    def test_required_fields_defined(self):
        """All expected fields should be in REQUIRED_FIELDS."""
        expected = {
            "index",
            "question",
            "ground_truth",
            "completion",
            "extracted_answer",
            "correct",
        }
        assert REQUIRED_FIELDS == expected
