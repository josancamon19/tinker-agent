"""Validation functions for post-training results."""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    base_score: float | None = None
    trained_score: float | None = None


REQUIRED_FIELDS = {
    "index",
    "question",
    "ground_truth",
    "completion",
    "extracted_answer",
    "correct",
}
# TODO: make sure things are runnable, setup env + run stuff


def validate_jsonl_row(row: dict, row_idx: int) -> list[str]:
    """Validate a single JSONL row has all required fields."""
    errors = []
    missing = REQUIRED_FIELDS - set(row.keys())
    if missing:
        errors.append(f"Row {row_idx}: missing fields {missing}")
    if "correct" in row and not isinstance(row["correct"], bool):
        errors.append(
            f"Row {row_idx}: 'correct' must be boolean, got {type(row['correct']).__name__}"
        )
    return errors


def validate_jsonl_file(path: Path) -> tuple[list[str], float | None]:
    """Validate JSONL file and return errors + accuracy score."""
    errors = []
    if not path.exists():
        return [f"File not found: {path}"], None

    rows = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)
                    errors.extend(validate_jsonl_row(row, i))
                except json.JSONDecodeError as e:
                    errors.append(f"Row {i}: invalid JSON - {e}")
    except Exception as e:
        return [f"Error reading {path}: {e}"], None

    if not rows:
        errors.append(f"File is empty: {path}")
        return errors, None

    # Calculate accuracy
    correct = sum(1 for r in rows if r.get("correct", False))
    accuracy = correct / len(rows) if rows else 0.0

    return errors, accuracy


def validate_results(results_dir: str | Path = "results") -> ValidationResult:
    """
    Validate post-training results directory.

    Checks:
    1. Required files exist (base_model.jsonl, trained_model.jsonl, summary.json, logs/)
    2. JSONL files have correct schema
    3. Trained model outperforms base model
    """
    results_dir = Path(results_dir)
    errors = []

    # Check directory exists
    if not results_dir.exists():
        return ValidationResult(
            valid=False, errors=[f"Results directory not found: {results_dir}"]
        )

    # Check required paths
    required_paths = {
        "base_model.jsonl": results_dir / "base_model.jsonl",
        "trained_model.jsonl": results_dir / "trained_model.jsonl",
        "summary.json": results_dir / "summary.json",
        "logs/": results_dir / "logs",
    }

    for name, path in required_paths.items():
        if not path.exists():
            errors.append(f"Missing required path: {name}")

    # Validate JSONL files
    base_errors, base_score = validate_jsonl_file(required_paths["base_model.jsonl"])
    trained_errors, trained_score = validate_jsonl_file(
        required_paths["trained_model.jsonl"]
    )

    errors.extend([f"base_model.jsonl: {e}" for e in base_errors])
    errors.extend([f"trained_model.jsonl: {e}" for e in trained_errors])

    # Check trained model outperforms base
    if base_score is not None and trained_score is not None:
        if trained_score <= base_score:
            errors.append(
                f"Trained model ({trained_score:.2%}) does not outperform base model ({base_score:.2%})"
            )

    # Validate summary.json schema
    summary_path = required_paths["summary.json"]
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            required_keys = {
                "tinker_run_id",
                "log_path",
                "wandb_url",
                "baseline_score",
                "trained_score",
            }
            missing_keys = required_keys - set(summary.keys())
            if missing_keys:
                errors.append(f"summary.json missing keys: {missing_keys}")
        except json.JSONDecodeError as e:
            errors.append(f"summary.json is invalid JSON: {e}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        base_score=base_score,
        trained_score=trained_score,
    )


if __name__ == "__main__":
    result = validate_results()
    if result.valid:
        print("✓ Validation passed")
        print(f"  Base model:    {result.base_score:.2%}")
        print(f"  Trained model: {result.trained_score:.2%}")
    else:
        print("✗ Validation failed:")
        for error in result.errors:
            print(f"  - {error}")
