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

from digit_classifier import classify_digit


CONFIRMED_DIR = (
    PROJECT_DIR
    / "dataset"
    / "confirmed"
)


def main() -> None:
    total = 0
    correct = 0

    per_digit_total = Counter()
    per_digit_correct = Counter()
    confusion = Counter()

    print()
    print("=" * 58)
    print(
        "FERRETTI OCR — DIGIT CLASSIFIER TEST"
    )
    print("=" * 58)
    print()

    for true_digit in "0123456789":
        digit_dir = (
            CONFIRMED_DIR
            / true_digit
        )

        if not digit_dir.exists():
            continue

        for image_path in sorted(
            digit_dir.glob("*.png")
        ):
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                print(
                    f"ERROR: {image_path.name}"
                )
                continue

            prediction = classify_digit(
                image
            )

            predicted_digit = (
                prediction.digit
            )

            total += 1

            per_digit_total[
                true_digit
            ] += 1

            is_correct = (
                predicted_digit
                == true_digit
            )

            if is_correct:
                correct += 1

                per_digit_correct[
                    true_digit
                ] += 1
            else:
                confusion[
                    (
                        true_digit,
                        predicted_digit or "<empty>",
                    )
                ] += 1

            result_text = (
                "PASS"
                if is_correct
                else "FAIL"
            )

            print(
                f"{image_path.name:<42} "
                f"true={true_digit} "
                f"pred={predicted_digit or '-'} "
                f"conf={prediction.confidence:.1%} "
                f"{result_text}"
            )

    print()
    print("-" * 58)

    print(
        f"Всього:       {total}"
    )

    print(
        f"Правильно:    {correct}"
    )

    accuracy = (
        correct / total * 100.0
        if total > 0
        else 0.0
    )

    print(
        f"Точність:     {accuracy:.2f}%"
    )

    print()
    print(
        "ТОЧНІСТЬ ПО ЦИФРАХ"
    )
    print("-" * 58)

    for digit in "0123456789":
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

        digit_accuracy = (
            digit_correct
            / digit_total
            * 100.0
        )

        print(
            f"{digit}: "
            f"{digit_correct}/{digit_total} "
            f"({digit_accuracy:.1f}%)"
        )

    print()
    print(
        "ПОМИЛКИ"
    )
    print("-" * 58)

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
    print("=" * 58)


if __name__ == "__main__":
    main()
