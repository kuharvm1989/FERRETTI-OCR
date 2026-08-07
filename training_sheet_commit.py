from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import cv2

from dataset_manifest import (
    add_sample,
    load_rows,
)


BASE_DIR = Path(__file__).resolve().parent

SOURCE_PDF = (
    BASE_DIR
    / "input"
    / "TRAINING_02.PDF"
)

CELLS_DIR = (
    BASE_DIR
    / "debug"
    / "training_sheet"
    / "cells_preview"
)

DATASET_DIR = (
    BASE_DIR
    / "dataset"
)

CONFIRMED_DIR = (
    DATASET_DIR
    / "confirmed"
)


# Мінімальна кількість чорних пікселів,
# щоб клітинка вважалася заповненою.
MIN_INK_PIXELS = 35


def file_hash(
    file_path: Path,
) -> str:
    hasher = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as source:
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


def parse_file_name(
    file_path: Path,
) -> tuple[str, int]:
    """
    Очікується:
    digit_2_09.png

    Повертає:
    digit = "2"
    column = 9
    """

    parts = file_path.stem.split(
        "_"
    )

    if len(parts) != 3:
        raise ValueError(
            f"Невідома назва: "
            f"{file_path.name}"
        )

    if parts[0] != "digit":
        raise ValueError(
            f"Невідома назва: "
            f"{file_path.name}"
        )

    digit = parts[1]
    column = int(
        parts[2]
    )

    if digit not in "0123456789":
        raise ValueError(
            f"Некоректна цифра: "
            f"{file_path.name}"
        )

    return (
        digit,
        column,
    )


def count_ink_pixels(
    file_path: Path,
) -> int:
    image = cv2.imread(
        str(file_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return 0

    # Після нашого очищення:
    # цифра чорна, фон білий.
    mask = image < 200

    return int(
        mask.sum()
    )


def main() -> None:
    print()
    print("=" * 58)
    print(
        "FERRETTI OCR — TRAINING SHEET COMMIT"
    )
    print("=" * 58)
    print()

    if not CELLS_DIR.exists():
        print(
            "Не знайдено:"
        )
        print(
            CELLS_DIR
        )
        return

    existing_rows = load_rows()

    known_ids = {
        row.get(
            "id",
            ""
        )
        for row in existing_rows
    }

    imported = 0
    empty = 0
    duplicates = 0
    errors = 0

    per_digit = {
        str(digit): 0
        for digit in range(10)
    }

    files = sorted(
        CELLS_DIR.glob(
            "digit_*.png"
        )
    )

    for source_path in files:
        try:
            (
                digit,
                column,
            ) = parse_file_name(
                source_path
            )
        except Exception as error:
            print(
                f"ERROR {source_path.name}: "
                f"{error}"
            )

            errors += 1
            continue

        ink_pixels = count_ink_pixels(
            source_path
        )

        if ink_pixels < MIN_INK_PIXELS:
            empty += 1
            continue

        digest = file_hash(
            source_path
        )

        sample_id = (
            "training_"
            + digest[:16]
        )

        if sample_id in known_ids:
            duplicates += 1
            continue

        destination_dir = (
            CONFIRMED_DIR
            / digit
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_name = (
            f"training_"
            f"{SOURCE_PDF.stem}_"
            f"d{digit}_"
            f"c{column:02d}_"
            f"{digest[:10]}.png"
        )

        destination_path = (
            destination_dir
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        relative_path = (
            destination_path
            .relative_to(
                BASE_DIR
            )
            .as_posix()
        )

        add_sample(
            sample_id=sample_id,
            image_path=relative_path,
            digit=digit,
            status="CONFIRMED",
            source_pdf=(
                SOURCE_PDF.name
            ),
            row=digit,
            column=str(
                column
            ),
            day="",
            ocr_value="",
            ocr_confidence="",
        )

        known_ids.add(
            sample_id
        )

        imported += 1

        per_digit[
            digit
        ] += 1

    print(
        f"Знайдено клітинок:      "
        f"{len(files)}"
    )

    print(
        f"Імпортовано:            "
        f"{imported}"
    )

    print(
        f"Порожніх пропущено:     "
        f"{empty}"
    )

    print(
        f"Дублікатів пропущено:   "
        f"{duplicates}"
    )

    print(
        f"Помилок:                "
        f"{errors}"
    )

    print()
    print(
        "ДОДАНО ПО ЦИФРАХ"
    )

    print("-" * 58)

    for digit in "0123456789":
        print(
            f"{digit}: "
            f"{per_digit[digit]}"
        )

    print()
    print("=" * 58)


if __name__ == "__main__":
    main()
