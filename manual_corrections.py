from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = (
    BASE_DIR
    / "dataset"
    / "manual_corrections.csv"
)


FIELDNAMES = [
    "created_at",
    "row",
    "column",
    "day",
    "ocr_value",
    "confirmed_value",
    "source_image",
]


def save_manual_correction(
    *,
    row: int,
    column: int,
    day: str,
    ocr_value: str,
    confirmed_value: str,
    source_image: str,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = OUTPUT_PATH.exists()

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            delimiter=";",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "row": row,
            "column": column,
            "day": day,
            "ocr_value": ocr_value,
            "confirmed_value": confirmed_value,
            "source_image": source_image,
        })
