from __future__ import annotations

import sys
from collections import Counter
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


def true_digit_from_name(
    path: Path,
) -> str:
    return path.stem.split(
        "_"
    )[1]


def main() -> None:
    results = []

    total = 0
    correct = 0

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

        is_correct = (
            prediction.digit
            == true_digit
        )

        total += 1

        if is_correct:
            correct += 1

        results.append({
            "file": path.name,
            "true": true_digit,
            "pred": prediction.digit,
            "confidence": (
                prediction.confidence
            ),
            "correct": is_correct,
            "neighbor": (
                prediction.neighbor_digit
            ),
            "agreement": (
                prediction.agreement
            ),
        })

    print()
    print("=" * 72)
    print(
        "FERRETTI OCR — CONFIDENCE DIAGNOSTICS"
    )
    print("=" * 72)

    print()
    print(
        f"Всього:       {total}"
    )

    print(
        f"Правильно:    {correct}"
    )

    if total:
        print(
            f"Точність:     "
            f"{correct / total * 100:.2f}%"
        )

    correct_scores = [
        item["confidence"]
        for item in results
        if item["correct"]
    ]

    wrong_scores = [
        item["confidence"]
        for item in results
        if not item["correct"]
    ]

    print()
    print(
        "CONFIDENCE — ПРАВИЛЬНІ"
    )
    print("-" * 72)

    if correct_scores:
        print(
            f"Кількість: {len(correct_scores)}"
        )
        print(
            f"MIN:       "
            f"{min(correct_scores) * 100:.1f}%"
        )
        print(
            f"AVG:       "
            f"{sum(correct_scores) / len(correct_scores) * 100:.1f}%"
        )
        print(
            f"MAX:       "
            f"{max(correct_scores) * 100:.1f}%"
        )

    print()
    print(
        "CONFIDENCE — ПОМИЛКОВІ"
    )
    print("-" * 72)

    if wrong_scores:
        print(
            f"Кількість: {len(wrong_scores)}"
        )
        print(
            f"MIN:       "
            f"{min(wrong_scores) * 100:.1f}%"
        )
        print(
            f"AVG:       "
            f"{sum(wrong_scores) / len(wrong_scores) * 100:.1f}%"
        )
        print(
            f"MAX:       "
            f"{max(wrong_scores) * 100:.1f}%"
        )

    print()
    print(
        "10 НАЙВИЩИХ ПОМИЛКОВИХ SCORE"
    )
    print("-" * 72)

    wrong_sorted = sorted(
        (
            item
            for item in results
            if not item["correct"]
        ),
        key=lambda item: (
            item["confidence"]
        ),
        reverse=True,
    )

    for item in wrong_sorted[:10]:
        print(
            f"{item['true']} -> "
            f"{item['pred']}   "
            f"{item['confidence'] * 100:5.1f}%   "
            f"KNN={item['neighbor']}   "
            f"agree={item['agreement']}"
        )

    print()
    print(
        "10 НАЙНИЖЧИХ ПРАВИЛЬНИХ SCORE"
    )
    print("-" * 72)

    correct_sorted = sorted(
        (
            item
            for item in results
            if item["correct"]
        ),
        key=lambda item: (
            item["confidence"]
        ),
    )

    for item in correct_sorted[:10]:
        print(
            f"{item['true']} -> "
            f"{item['pred']}   "
            f"{item['confidence'] * 100:5.1f}%   "
            f"KNN={item['neighbor']}   "
            f"agree={item['agreement']}"
        )

    print()
    print(
        "ТОЧНІСТЬ ПРИ РІЗНИХ ПОРОГАХ"
    )
    print("-" * 72)

    for threshold in (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ):
        accepted = [
            item
            for item in results
            if item["confidence"]
            >= threshold
        ]

        if not accepted:
            print(
                f">= {threshold * 100:.0f}%: "
                f"0 прикладів"
            )
            continue

        accepted_correct = sum(
            1
            for item in accepted
            if item["correct"]
        )

        accepted_accuracy = (
            accepted_correct
            / len(accepted)
            * 100.0
        )

        coverage = (
            len(accepted)
            / len(results)
            * 100.0
        )

        print(
            f">= {threshold * 100:.0f}%: "
            f"{accepted_correct}/"
            f"{len(accepted)} правильно "
            f"({accepted_accuracy:.1f}%), "
            f"покриття {coverage:.1f}%"
        )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
