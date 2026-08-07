from __future__ import annotations

import csv
from collections import Counter

from app.config.project_paths import (
    DATASET_MANIFEST_PATH,
    LEARNING_IMPORT_REPORT_PATH,
)

def main() -> None:
    manual_counts = Counter()

    total_manual = 0

    if DATASET_MANIFEST_PATH.exists():
        with DATASET_MANIFEST_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file,
                delimiter=";",
            )

            for row in reader:
                sample_id = (
                    row.get(
                        "id",
                        ""
                    )
                    .strip()
                )

                digit = (
                    row.get(
                        "digit",
                        ""
                    )
                    .strip()
                )

                status = (
                    row.get(
                        "status",
                        ""
                    )
                    .strip()
                )

                if (
                    sample_id.startswith(
                        "manual_"
                    )
                    and status == "CONFIRMED"
                    and digit in "0123456789"
                ):
                    manual_counts[
                        digit
                    ] += 1

                    total_manual += 1

    imported_rows = 0
    warnings = 0

    if LEARNING_IMPORT_REPORT_PATH.exists():
        with LEARNING_IMPORT_REPORT_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file,
                delimiter=";",
            )

            for row in reader:
                status = (
                    row.get(
                        "status",
                        ""
                    )
                    .strip()
                )

                if status == "IMPORTED":
                    imported_rows += 1

                elif status == "WARNING":
                    warnings += 1

    print()
    print("=" * 58)
    print(
        "FERRETTI OCR — LEARNING STATISTICS"
    )
    print("=" * 58)

    print()
    print(
        f"Ручних підтверджених цифр: "
        f"{total_manual}"
    )

    print(
        f"Імпортованих підтверджень:  "
        f"{imported_rows}"
    )

    print(
        f"WARNING:                    "
        f"{warnings}"
    )

    print()
    print(
        "РУЧНІ ПРИКЛАДИ ПО ЦИФРАХ"
    )

    print("-" * 58)

    for digit in "0123456789":
        print(
            f"{digit}: "
            f"{manual_counts[digit]}"
        )

    print()
    print(
        "РЕКОМЕНДАЦІЯ"
    )

    print("-" * 58)

    if total_manual < 25:
        print(
            "Поки рано перенавчати модель."
        )

        print(
            "Накопичте хоча б 25 нових "
            "ручних цифр."
        )

    elif total_manual < 100:
        print(
            "Можна зробити тестове "
            "перенавчання."
        )

        print(
            "Для стабільного ефекту краще "
            "накопичити 100+ нових цифр."
        )

    else:
        print(
            "Рекомендовано перенавчити SVM."
        )

    print()
    print("=" * 58)


if __name__ == "__main__":
    main()
