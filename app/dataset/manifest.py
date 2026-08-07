from __future__ import annotations
from app.config.project_paths import (
    DATASET_DIR,
    DATASET_MANIFEST_PATH,
)


import csv
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

DATASET_MANIFEST_PATH = (
    DATASET_DIR
    / "dataset_manifest.csv"
)


FIELDNAMES = [
    "id",
    "image_path",
    "digit",
    "status",
    "source_pdf",
    "row",
    "column",
    "day",
    "ocr_value",
    "ocr_confidence",
    "created_at",
    "confirmed_at",
]


def ensure_manifest() -> None:
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATASET_MANIFEST_PATH.exists():
        return

    with DATASET_MANIFEST_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            delimiter=";",
        )

        writer.writeheader()


def load_rows() -> list[dict[str, str]]:
    ensure_manifest()

    with DATASET_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        return list(reader)


def save_rows(
    rows: list[dict[str, str]],
) -> None:
    ensure_manifest()

    with DATASET_MANIFEST_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            delimiter=";",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def add_sample(
    *,
    sample_id: str,
    image_path: str,
    status: str = "PENDING",
    digit: str = "",
    source_pdf: str = "",
    row: str = "",
    column: str = "",
    day: str = "",
    ocr_value: str = "",
    ocr_confidence: str = "",
) -> None:
    rows = load_rows()

    existing_ids = {
        row_data.get(
            "id",
            ""
        )
        for row_data in rows
    }

    if sample_id in existing_ids:
        return

    rows.append({
        "id": sample_id,
        "image_path": image_path,
        "digit": digit,
        "status": status,
        "source_pdf": source_pdf,
        "row": row,
        "column": column,
        "day": day,
        "ocr_value": ocr_value,
        "ocr_confidence": ocr_confidence,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "confirmed_at": "",
    })

    save_rows(
        rows
    )


def confirm_sample(
    sample_id: str,
    digit: str,
    image_path: str | None = None,
) -> bool:
    rows = load_rows()

    updated = False

    for row_data in rows:
        if row_data.get(
            "id"
        ) != sample_id:
            continue

        row_data[
            "digit"
        ] = digit

        row_data[
            "status"
        ] = "CONFIRMED"

        row_data[
            "confirmed_at"
        ] = datetime.now().isoformat(
            timespec="seconds"
        )

        if image_path is not None:
            row_data[
                "image_path"
            ] = image_path

        updated = True
        break

    if updated:
        save_rows(
            rows
        )

    return updated


def reject_sample(
    sample_id: str,
    image_path: str | None = None,
) -> bool:
    rows = load_rows()

    updated = False

    for row_data in rows:
        if row_data.get(
            "id"
        ) != sample_id:
            continue

        row_data[
            "digit"
        ] = ""

        row_data[
            "status"
        ] = "REJECTED"

        row_data[
            "confirmed_at"
        ] = datetime.now().isoformat(
            timespec="seconds"
        )

        if image_path is not None:
            row_data[
                "image_path"
            ] = image_path

        updated = True
        break

    if updated:
        save_rows(
            rows
        )

    return updated


def main() -> None:
    ensure_manifest()

    rows = load_rows()

    confirmed = sum(
        1
        for row_data in rows
        if row_data.get(
            "status"
        ) == "CONFIRMED"
    )

    pending = sum(
        1
        for row_data in rows
        if row_data.get(
            "status"
        ) == "PENDING"
    )

    rejected = sum(
        1
        for row_data in rows
        if row_data.get(
            "status"
        ) == "REJECTED"
    )

    print()
    print(
        "FERRETTI OCR — DATASET MANIFEST"
    )

    print(
        f"Записів:       {len(rows)}"
    )

    print(
        f"Підтверджено:  {confirmed}"
    )

    print(
        f"Очікують:      {pending}"
    )

    print(
        f"Відхилено:     {rejected}"
    )

    print()
    print(
        DATASET_MANIFEST_PATH
    )


if __name__ == "__main__":
    main()
