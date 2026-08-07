from __future__ import annotations


DIGIT_THRESHOLDS = {
    "0": 0.55,
    "1": 0.25,
    "2": 0.20,
    "3": 0.45,
    "4": 0.20,
    "5": 0.20,
    "6": 0.20,
    "7": 0.20,
    "8": 0.20,
    "9": 0.45,
}


def threshold_for_digit(
    digit: str,
) -> float:
    return DIGIT_THRESHOLDS.get(
        digit,
        1.0,
    )


def digit_is_trusted(
    digit: str,
    confidence: float,
) -> bool:
    if digit not in DIGIT_THRESHOLDS:
        return False

    return (
        confidence
        >= threshold_for_digit(
            digit
        )
    )
