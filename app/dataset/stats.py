from __future__ import annotations

from app.config.project_paths import (
    DATASET_DIR,
    DATASET_MANIFEST_PATH,
    DATASET_CONFIRMED_DIR,
    DATASET_PENDING_DIR,
    DATASET_REJECTED_DIR,
)

def count_png_files(
    folder: Path,
) -> int:
    if not folder.exists():
        return 0

    return len(
        list(
            folder.glob("*.png")
        )
    )


def main() -> None:
    print()
    print("=" * 46)
    print("FERRETTI OCR — DATASET STATISTICS")
    print("=" * 46)
    print()

    counts: dict[str, int] = {}

    for digit in "0123456789":
        folder = (
            DATASET_CONFIRMED_DIR
            / digit
        )

        count = count_png_files(
            folder
        )

        counts[digit] = count

        print(
            f"Цифра {digit}: {count:5d}"
        )

    total_confirmed = sum(
        counts.values()
    )

    pending = count_png_files(
        DATASET_PENDING_DIR
    )

    rejected = count_png_files(
        DATASET_REJECTED_DIR
    )

    print()
    print("-" * 46)

    print(
        f"Підтверджено всього: {total_confirmed}"
    )

    print(
        f"Очікують перевірки:   {pending}"
    )

    print(
        f"Відхилено:            {rejected}"
    )

    print("-" * 46)

    if total_confirmed == 0:
        print()
        print(
            "Поки що немає підтверджених "
            "прикладів."
        )
        return

    non_zero_counts = [
        count
        for count in counts.values()
        if count > 0
    ]

    if non_zero_counts:
        minimum = min(
            non_zero_counts
        )

        maximum = max(
            non_zero_counts
        )

        print()
        print(
            f"Найменший клас: {minimum}"
        )

        print(
            f"Найбільший клас: {maximum}"
        )

        if minimum > 0:
            imbalance = (
                maximum / minimum
            )

            print(
                f"Перекіс max/min: "
                f"{imbalance:.2f}×"
            )

    missing_digits = [
        digit
        for digit, count in counts.items()
        if count == 0
    ]

    if missing_digits:
        print()
        print(
            "Ще немає прикладів для цифр: "
            + ", ".join(
                missing_digits
            )
        )

    print()
    print("=" * 46)


if __name__ == "__main__":
    main()
