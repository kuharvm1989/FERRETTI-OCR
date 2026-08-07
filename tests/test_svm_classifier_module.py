from __future__ import annotations

import sys
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


from app.ocr.svm_digit_classifier import (
    classify_digit,
)


CONFIRMED_DIR = (
    PROJECT_DIR
    / "dataset"
    / "confirmed"
)


def main() -> None:
    total = 0
    correct = 0

    for digit in "0123456789":
        folder = (
            CONFIRMED_DIR
            / digit
        )

        for path in sorted(
            folder.glob("*.png")
        ):
            image = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE,
            )

            prediction = classify_digit(
                image
            )

            total += 1

            if prediction.digit == digit:
                correct += 1

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print(
        f"Samples:  {total}"
    )

    print(
        f"Correct:  {correct}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()
