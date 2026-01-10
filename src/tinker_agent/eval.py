"""Validation functions for post-training results."""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    task_type: str | None = None  # "rl" or "sft"
    # RL metrics (accuracy-based)
    base_score: float | None = None
    trained_score: float | None = None
    # SFT metrics (loss-based)
    base_nll: float | None = None
    trained_nll: float | None = None


# RL tasks use accuracy-based evaluation
REQUIRED_FIELDS_RL = {
    "index",
    "question",
    "ground_truth",
    "completion",
    "extracted_answer",
    "correct",
}

# SFT tasks use loss-based evaluation
REQUIRED_FIELDS_SFT = {
    "index",
    "prompt",
    "completion",
    "nll",
}


def detect_task_type_from_row(row: dict) -> str | None:
    """Detect task type from JSONL row structure."""
    if "correct" in row and "question" in row:
        return "rl"
    elif "nll" in row and "prompt" in row:
        return "sft"
    return None


def validate_jsonl_row_rl(row: dict, row_idx: int) -> list[str]:
    """Validate a single JSONL row for RL (accuracy-based) evaluation."""
    errors = []
    missing = REQUIRED_FIELDS_RL - set(row.keys())
    if missing:
        errors.append(f"Row {row_idx}: missing fields {missing}")
    if "correct" in row and not isinstance(row["correct"], bool):
        errors.append(
            f"Row {row_idx}: 'correct' must be boolean, got {type(row['correct']).__name__}"
        )
    return errors


def validate_jsonl_row_sft(row: dict, row_idx: int) -> list[str]:
    """Validate a single JSONL row for SFT (loss-based) evaluation."""
    errors = []
    missing = REQUIRED_FIELDS_SFT - set(row.keys())
    if missing:
        errors.append(f"Row {row_idx}: missing fields {missing}")
    if "nll" in row and not isinstance(row["nll"], (int, float)):
        errors.append(
            f"Row {row_idx}: 'nll' must be numeric, got {type(row['nll']).__name__}"
        )
    return errors


def validate_jsonl_file(
    path: Path, task_type: str | None = None
) -> tuple[list[str], float | None, str | None]:
    """
    Validate JSONL file and return errors + metric + detected task type.

    For RL: returns accuracy (higher is better)
    For SFT: returns mean NLL (lower is better)
    """
    errors = []
    if not path.exists():
        return [f"File not found: {path}"], None, None

    rows = []
    detected_type = task_type

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)

                    # Detect task type from first row if not specified
                    if detected_type is None and i == 0:
                        detected_type = detect_task_type_from_row(row)

                    # Validate based on task type
                    if detected_type == "rl":
                        errors.extend(validate_jsonl_row_rl(row, i))
                    elif detected_type == "sft":
                        errors.extend(validate_jsonl_row_sft(row, i))
                    else:
                        errors.append(f"Row {i}: cannot determine task type from schema")

                except json.JSONDecodeError as e:
                    errors.append(f"Row {i}: invalid JSON - {e}")
    except Exception as e:
        return [f"Error reading {path}: {e}"], None, None

    if not rows:
        errors.append(f"File is empty: {path}")
        return errors, None, detected_type

    # Calculate metric based on task type
    if detected_type == "rl":
        # Accuracy: higher is better
        correct = sum(1 for r in rows if r.get("correct", False))
        metric = correct / len(rows) if rows else 0.0
    elif detected_type == "sft":
        # Mean NLL: lower is better
        nlls = [r.get("nll", 0) for r in rows if isinstance(r.get("nll"), (int, float))]
        metric = sum(nlls) / len(nlls) if nlls else None
    else:
        metric = None

    return errors, metric, detected_type


def validate_results(results_dir: str | Path = "results") -> ValidationResult:
    """
    Validate post-training results directory.

    Checks:
    1. Required files exist (base_model.jsonl, trained_model.jsonl, summary.json, logs/)
    2. JSONL files have correct schema (RL or SFT)
    3. Trained model outperforms base model:
       - RL: trained_score > base_score (accuracy)
       - SFT: trained_nll < base_nll (loss)
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

    # Try to detect task type from summary.json first
    task_type = None
    summary_path = required_paths["summary.json"]
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            task_type = summary.get("task_type")
        except json.JSONDecodeError:
            pass

    # Validate JSONL files
    base_errors, base_metric, detected_type = validate_jsonl_file(
        required_paths["base_model.jsonl"], task_type
    )
    trained_errors, trained_metric, _ = validate_jsonl_file(
        required_paths["trained_model.jsonl"], task_type or detected_type
    )

    # Use detected type if not specified in summary
    task_type = task_type or detected_type

    errors.extend([f"base_model.jsonl: {e}" for e in base_errors])
    errors.extend([f"trained_model.jsonl: {e}" for e in trained_errors])

    # Check trained model outperforms base based on task type
    base_score, trained_score = None, None
    base_nll, trained_nll = None, None

    if base_metric is not None and trained_metric is not None:
        if task_type == "rl":
            base_score, trained_score = base_metric, trained_metric
            if trained_score <= base_score:
                errors.append(
                    f"Trained model ({trained_score:.2%}) does not outperform base model ({base_score:.2%})"
                )
        elif task_type == "sft":
            base_nll, trained_nll = base_metric, trained_metric
            if trained_nll >= base_nll:
                errors.append(
                    f"Trained model NLL ({trained_nll:.4f}) is not better than base model ({base_nll:.4f})"
                )

    # Validate summary.json schema
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)

            # Common required keys
            required_keys = {
                "task_type",
                "tinker_run_id",
                "log_path",
                "wandb_url",
            }

            # Task-specific required keys
            if task_type == "rl":
                required_keys.update({"baseline_score", "trained_score"})
            elif task_type == "sft":
                required_keys.update({"baseline_nll", "trained_nll"})

            missing_keys = required_keys - set(summary.keys())
            if missing_keys:
                errors.append(f"summary.json missing keys: {missing_keys}")
        except json.JSONDecodeError as e:
            errors.append(f"summary.json is invalid JSON: {e}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        task_type=task_type,
        base_score=base_score,
        trained_score=trained_score,
        base_nll=base_nll,
        trained_nll=trained_nll,
    )


if __name__ == "__main__":
    result = validate_results()
    if result.valid:
        print("✓ Validation passed")
        if result.task_type == "rl":
            print(f"  Task type:     RL (accuracy-based)")
            print(f"  Base model:    {result.base_score:.2%}")
            print(f"  Trained model: {result.trained_score:.2%}")
            improvement = (result.trained_score - result.base_score) / result.base_score * 100
            print(f"  Improvement:   +{improvement:.1f}%")
        elif result.task_type == "sft":
            print(f"  Task type:     SFT (loss-based)")
            print(f"  Base NLL:      {result.base_nll:.4f}")
            print(f"  Trained NLL:   {result.trained_nll:.4f}")
            reduction = (result.base_nll - result.trained_nll) / result.base_nll * 100
            print(f"  NLL reduction: {reduction:.1f}%")
    else:
        print("✗ Validation failed:")
        for error in result.errors:
            print(f"  - {error}")
