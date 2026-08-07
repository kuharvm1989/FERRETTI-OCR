from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


from training.train_digit_svm import (
    create_hog,
    extract_features,
)


CELLS_DIR = (
    PROJECT_DIR
    / "debug"
    / "training_sheet"
    / "cells_preview"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "digit_svm.yml"
)

MIN_INK_PIXELS = 35


def get_true_digit(
    file_path: Path,
) -> int:
    # digit_7_14.png -> 7
    parts = file_path.stem.split("_")

    return int(
        parts[1]
    )


def has_digit(
    image: np.ndarray,
) -> bool:
    return int(
        np.sum(
            image < 200
        )
    ) >= MIN_INK_PIXELS


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено модель: {MODEL_PATH}"
        )

    if not CELLS_DIR.exists():
        raise FileNotFoundError(
            f"Не знайдено клітинки: {CELLS_DIR}"
        )

    svm = cv2.ml.SVM_load(
        str(MODEL_PATH)
    )

    hog = create_hog()

    total = 0
    correct = 0

    per_digit_total = Counter()
    per_digit_correct = Counter()
    confusion = Counter()

    print()
    print("=" * 62)
    print(
        "FERRETTI OCR — EXTERNAL SVM VALIDATION"
    )
    print("=" * 62)
    print()

    files = sorted(
        CELLS_DIR.glob(
            "digit_*.png"
        )
    )

    for image_path in files:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            continue

        if not has_digit(
            image
        ):
            continue

        true_digit = get_true_digit(
            image_path
        )

        features = extract_features(
            image,
            hog,
        ).reshape(
            1,
            -1,
        )

        _result, prediction = svm.predict(
            features
        )

        predicted_digit = int(
            prediction[
                0,
                0,
            ]
        )

        total += 1

        per_digit_total[
            true_digit
        ] += 1

        if predicted_digit == true_digit:
            correct += 1

            per_digit_correct[
                true_digit
            ] += 1
        else:
            confusion[
                (
                    true_digit,
                    predicted_digit,
                )
            ] += 1

    accuracy = (
        correct / total * 100.0
        if total
        else 0.0
    )

    print(
        f"Перевірено:    {total}"
    )

    print(
        f"Правильно:     {correct}"
    )

    print(
        f"ТОЧНІСТЬ:      {accuracy:.2f}%"
    )

    print()
    print(
        "ТОЧНІСТЬ ПО ЦИФРАХ"
    )

    print("-" * 62)

    for digit in range(10):
        digit_total = (
            per_digit_total[
                digit
            ]
        )

        if digit_total == 0:
            print(
                f"{digit}: немає даних"
            )
            continue

        digit_correct = (
            per_digit_correct[
                digit
            ]
        )

        value = (
            digit_correct
            / digit_total
            * 100.0
        )

        print(
            f"{digit}: "
            f"{digit_correct}/{digit_total} "
            f"({value:.1f}%)"
        )

    print()
    print(
        "ПОМИЛКИ"
    )

    print("-" * 62)

    if not confusion:
        print(
            "Помилок немає."
        )
    else:
        for (
            true_digit,
            predicted_digit,
        ), count in confusion.most_common():
            print(
                f"{true_digit} -> "
                f"{predicted_digit}: "
                f"{count}"
            )

    print()
    print("=" * 62)


if __name__ == "__main__":
    main()
