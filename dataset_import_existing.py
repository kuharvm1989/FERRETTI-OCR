from __future__ import annotations

import hashlib
from pathlib import Path

from dataset_manifest import (
    add_sample,
)


BASE_DIR = Path(__file__).resolve().parent

CONFIRMED_DIR = (
    BASE_DIR
    / "dataset"
    / "confirmed"
)


def file_hash(
    file_path: Path,
) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as source:
        while True:
            chunk = source.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def main() -> None:
    added = 0
    skipped = 0

    for digit in "0123456789":
        digit_dir = (
            CONFIRMED_DIR
            / digit
        )

        if not digit_dir.exists():
            continue

        for image_path in sorted(
            digit_dir.glob("*.png")
        ):
            digest = file_hash(
                image_path
            )

            sample_id = digest[:16]

            relative_path = (
                image_path
                .relative_to(
                    BASE_DIR
                )
                .as_posix()
            )

            before = (
                sample_id
            )

            add_sample(
                sample_id=sample_id,
                image_path=relative_path,
                status="CONFIRMED",
                digit=digit,
            )

            # add_sample не повертає bool,
            # тому просто рахуємо всі знайдені файли.
            # Повторний запуск безпечний:
            # дублікати не додадуться.
            added += 1

    print()
    print(
        "Імпорт завершено."
    )

    print(
        f"Опрацьовано файлів: {added}"
    )

    print()
    print(
        "Тепер перевірте:"
    )

    print(
        "python dataset_manifest.py"
    )


if __name__ == "__main__":
    main()
