from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


from digit_confidence import (
    classify_digit_with_confidence,
)


CELLS_DIR = (
    PROJECT_DIR
    / "debug"
    / "training_sheet"
    / "cells_preview"
)

MIN_INK_PIXELS = 35

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]


def true_digit_from_name(
    path: Path,
) -> str:
    return path.stem.split("_")[1]


def main() -> None:
    results = defaultdict(list)

    for path in sorted(
        CELLS_DIR.glob("digit_*.png")
    ):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            continue

        if int(
            (image < 200).sum()
        ) < MIN_INK_PIXELS:
            continue

        true_digit = (
            true_digit_from_name(path)
        )

        prediction = (
            classify_digit_with_confidence(
                image
            )
        )

        if prediction.digit not in "0123456789":
            continue

        results[
            prediction.digit
        ].append({
            "true": true_digit,
            "confidence": prediction.confidence,
            "correct": (
                prediction.digit
                == true_digit
            ),
        })

    print()
    print("=" * 76)
    print(
        "FERRETTI OCR — CONFIDENCE BY PREDICTED DIGIT"
    )
    print("=" * 76)

    for digit in "0123456789":
        items = results[digit]

        print()
        print(
            f"ЦИФРА {digit} "
            f"(передбачено {len(items)} разів)"
        )

        print("-" * 76)

        if not items:
            print("Немає даних.")
            continue

        for threshold in THRESHOLDS:
            accepted = [
                item
                for item in items
                if item["confidence"]
                >= threshold
            ]

            if not accepted:
                print(
                    f">= {threshold * 100:2.0f}%: "
                    f"0"
                )
                continue

            correct = sum(
                1
                for item in accepted
                if item["correct"]
            )

            accuracy = (
                correct
                / len(accepted)
                * 100.0
            )

            coverage = (
                len(accepted)
                / len(items)
                * 100.0
            )

            print(
                f">= {threshold * 100:2.0f}%: "
                f"{correct:2d}/"
                f"{len(accepted):2d} "
                f"правильно "
                f"({accuracy:5.1f}%), "
                f"покриття "
                f"{coverage:5.1f}%"
            )

    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
