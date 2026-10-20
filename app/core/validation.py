"""Validation utilities for Precedent V1 (FR-5, FR-6, FR-9)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Type, TypeVar, get_type_hints
from pydantic import BaseModel, ValidationError

from app.schemas import ClaimVerificationResult
from app.core.errors import ModelOutputValidationError


T = TypeVar("T", bound=BaseModel)


def validate_structured_output(
    raw: Union[str, Dict[str, Any]],
    model_class: Type[T],
    context: str = "unknown",
) -> T:
    """
    Validate structured output from LLM against a Pydantic model.
    Accepts either a JSON string or a dictionary.
    Raises ModelOutputValidationError on failure.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ModelOutputValidationError(
                f"Invalid JSON in {context}: {e}",
                details={"raw": raw[:500], "error": str(e)},
            )
    else:
        data = raw

    try:
        return model_class(**data)
    except ValidationError as e:
        raise ModelOutputValidationError(
            f"Validation failed for {context}: {e}",
            details={"raw": str(raw)[:500], "errors": e.errors()},
        )


def validate_required_fields(
    data: Dict[str, Any],
    required_fields: List[str],
    context: str = "unknown",
) -> None:
    """
    Validate that all required fields are present in data.
    Raises ModelOutputValidationError if any are missing.
    """
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ModelOutputValidationError(
            f"Missing required fields in {context}: {missing}",
            details={"missing_fields": missing, "provided_fields": list(data.keys())},
        )


def validate_confidence_range(
    value: Any,
    field_name: str,
    context: str = "unknown",
) -> float:
    """
    Validate that a confidence value is a float in [0.0, 1.0].
    Returns the validated float value.
    Raises ModelOutputValidationError if invalid.
    """
    try:
        conf = float(value)
    except (TypeError, ValueError):
        raise ModelOutputValidationError(
            f"{field_name} must be a number in {context}",
            details={"field": field_name, "value": value, "type": type(value).__name__},
        )

    if not (0.0 <= conf <= 1.0):
        raise ModelOutputValidationError(
            f"{field_name} must be in range [0.0, 1.0] in {context}",
            details={"field": field_name, "value": conf},
        )

    return conf


def validate_enum_value(
    value: Any,
    allowed_values: List[str],
    field_name: str,
    context: str = "unknown",
) -> str:
    """
    Validate that a value is one of the allowed enum values.
    Returns the validated string value.
    Raises ModelOutputValidationError if invalid.
    """
    if value not in allowed_values:
        raise ModelOutputValidationError(
            f"{field_name} must be one of {allowed_values} in {context}",
            details={"field": field_name, "value": value, "allowed": allowed_values},
        )
    return str(value)


def validate_non_empty_string(
    value: Any,
    field_name: str,
    context: str = "unknown",
    max_length: Optional[int] = None,
) -> str:
    """
    Validate that a value is a non-empty string.
    Returns the validated string.
    Raises ModelOutputValidationError if invalid.
    """
    if not isinstance(value, str):
        raise ModelOutputValidationError(
            f"{field_name} must be a string in {context}",
            details={"field": field_name, "type": type(value).__name__},
        )

    if not value.strip():
        raise ModelOutputValidationError(
            f"{field_name} must not be empty in {context}",
            details={"field": field_name},
        )

    if max_length is not None and len(value) > max_length:
        raise ModelOutputValidationError(
            f"{field_name} exceeds maximum length of {max_length} in {context}",
            details={"field": field_name, "length": len(value), "max": max_length},
        )

    return value


def validate_list_of_strings(
    value: Any,
    field_name: str,
    context: str = "unknown",
    min_length: int = 0,
    max_length: Optional[int] = None,
) -> List[str]:
    """
    Validate that a value is a list of strings.
    Returns the validated list.
    Raises ModelOutputValidationError if invalid.
    """
    if not isinstance(value, list):
        raise ModelOutputValidationError(
            f"{field_name} must be a list in {context}",
            details={"field": field_name, "type": type(value).__name__},
        )

    if len(value) < min_length:
        raise ModelOutputValidationError(
            f"{field_name} must have at least {min_length} items in {context}",
            details={"field": field_name, "length": len(value)},
        )

    if max_length is not None and len(value) > max_length:
        raise ModelOutputValidationError(
            f"{field_name} must have at most {max_length} items in {context}",
            details={"field": field_name, "length": len(value)},
        )

    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ModelOutputValidationError(
                f"{field_name}[{i}] must be a string in {context}",
                details={"field": field_name, "index": i, "type": type(item).__name__},
            )

    return value


def validate_claim_verification_result(
    raw: Union[str, Dict[str, Any]],
    context: str = "claim_verification",
) -> ClaimVerificationResult:
    """
    Convenience function to validate a ClaimVerificationResult.
    """
    return validate_structured_output(raw, ClaimVerificationResult, context)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract JSON object from text that may contain extra content.
    Returns None if no valid JSON object found.
    """
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end+1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try progressively smaller substrings
        for i in range(end, start, -1):
            try:
                return json.loads(text[start:i+1])
            except json.JSONDecodeError:
                continue
    return None