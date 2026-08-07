from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BASE_DIR
    / "dataset"
    / "dataset_manifest.csv"
)


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(
            "Не знайдено dataset_manifest.csv"
        )
        return

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        rows = list(reader)

    confirmed_rows = [
        row
        for row in rows
        if row.get(
            "status"
        ) == "CONFIRMED"
    ]

    total_with_ocr = 0
    correct = 0

    confusion = Counter()
    per_digit_total = Counter()
    per_digit_correct = Counter()

    for row in confirmed_rows:
        true_digit = (
            row.get(
                "digit",
                ""
            ).strip()
        )

        ocr_value = (
            row.get(
                "ocr_value",
                ""
            ).strip()
        )

        if not true_digit:
            continue

        if not ocr_value:
            continue

        total_with_ocr += 1

        per_digit_total[
            true_digit
        ] += 1

        if ocr_value == true_digit:
            correct += 1

            per_digit_correct[
                true_digit
            ] += 1
        else:
            confusion[
                (
                    true_digit,
                    ocr_value,
                )
            ] += 1

    print()
    print("=" * 52)
    print(
        "FERRETTI OCR — ACCURACY REPORT"
    )
    print("=" * 52)
    print()

    print(
        f"Підтверджених зразків: "
        f"{len(confirmed_rows)}"
    )

    print(
        f"З OCR-значенням:        "
        f"{total_with_ocr}"
    )

    if total_with_ocr == 0:
        print()
        print(
            "Поки що у manifest немає "
            "збережених ocr_value."
        )
        print(
            "Це нормально для перших "
            "імпортованих зразків."
        )
        return

    accuracy = (
        correct
        / total_with_ocr
        * 100.0
    )

    print(
        f"Правильно:              "
        f"{correct}"
    )

    print(
        f"Точність:               "
        f"{accuracy:.2f}%"
    )

    print()
    print(
        "ТОЧНІСТЬ ПО ЦИФРАХ"
    )
    print("-" * 52)

    for digit in "0123456789":
        total = per_digit_total[
            digit
        ]

        correct_digit = (
            per_digit_correct[
                digit
            ]
        )

        if total == 0:
            print(
                f"{digit}: немає даних"
            )
            continue

        digit_accuracy = (
            correct_digit
            / total
            * 100.0
        )

        print(
            f"{digit}: "
            f"{correct_digit}/{total} "
            f"({digit_accuracy:.1f}%)"
        )

    print()
    print(
        "НАЙЧАСТІШІ ПОМИЛКИ"
    )
    print("-" * 52)

    if not confusion:
        print(
            "Помилок немає."
        )
    else:
        for (
            true_digit,
            ocr_value,
        ), count in confusion.most_common(
            20
        ):
            print(
                f"{true_digit} -> "
                f"{ocr_value}: "
                f"{count}"
            )

    print()
    print("=" * 52)


if __name__ == "__main__":
    main()
