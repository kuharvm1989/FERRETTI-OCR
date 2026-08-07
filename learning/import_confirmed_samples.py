from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


from digit_splitter import split_digits
from ocr_engine_v2 import build_blue_mask
from dataset_manifest import (
    add_sample,
    load_rows,
)


CORRECTIONS_PATH = (
    PROJECT_DIR
    / "dataset"
    / "manual_corrections.csv"
)

CONFIRMED_DIR = (
    PROJECT_DIR
    / "dataset"
    / "confirmed"
)

REPORT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "learning_import_report.csv"
)


REPORT_FIELDS = [
    "source_image",
    "ocr_value",
    "confirmed_value",
    "status",
    "details",
]


def file_hash(
    image: bytes,
) -> str:
    return hashlib.sha256(
        image
    ).hexdigest()


def load_reported_keys() -> set[str]:
    if not REPORT_PATH.exists():
        return set()

    result = set()

    with REPORT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            source = row.get(
                "source_image",
                "",
            )

            confirmed = row.get(
                "confirmed_value",
                "",
            )

            status = row.get(
                "status",
                "",
            )

            if (
                source
                and confirmed
                and status == "IMPORTED"
            ):
                result.add(
                    f"{source}|{confirmed}"
                )

    return result


def append_report(
    *,
    source_image: str,
    ocr_value: str,
    confirmed_value: str,
    status: str,
    details: str,
) -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = REPORT_PATH.exists()

    with REPORT_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=REPORT_FIELDS,
            delimiter=";",
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            "source_image": source_image,
            "ocr_value": ocr_value,
            "confirmed_value": confirmed_value,
            "status": status,
            "details": details,
        })


def build_mask(
    image,
):
    mask = build_blue_mask(
        image
    )

    if int(
        cv2.countNonZero(mask)
    ) >= 20:
        return mask

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )[1]


def main() -> None:
    if not CORRECTIONS_PATH.exists():
        print(
            "Не знайдено manual_corrections.csv"
        )
        return

    already_imported = (
        load_reported_keys()
    )

    manifest_rows = load_rows()

    known_ids = {
        row.get("id", "")
        for row in manifest_rows
    }

    with CORRECTIONS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        corrections = list(
            csv.DictReader(
                csv_file,
                delimiter=";",
            )
        )

    imported_rows = 0
    imported_digits = 0
    skipped = 0
    warnings = 0

    print()
    print("=" * 64)
    print(
        "FERRETTI OCR — LEARNING IMPORT"
    )
    print("=" * 64)
    print()

    for correction in corrections:
        source_image_text = (
            correction.get(
                "source_image",
                "",
            ).strip()
        )

        ocr_value = (
            correction.get(
                "ocr_value",
                "",
            ).strip()
        )

        confirmed_value = (
            correction.get(
                "confirmed_value",
                "",
            ).strip()
        )

        unique_key = (
            f"{source_image_text}|"
            f"{confirmed_value}"
        )

        if unique_key in already_imported:
            skipped += 1
            continue

        if (
            not confirmed_value
            or not confirmed_value.isdigit()
            or len(confirmed_value) > 2
        ):
            append_report(
                source_image=source_image_text,
                ocr_value=ocr_value,
                confirmed_value=confirmed_value,
                status="WARNING",
                details=(
                    "Некоректне підтверджене "
                    "значення."
                ),
            )

            warnings += 1
            continue

        source_path = Path(
            source_image_text
        )

        if not source_path.exists():
            append_report(
                source_image=source_image_text,
                ocr_value=ocr_value,
                confirmed_value=confirmed_value,
                status="WARNING",
                details=(
                    "Файл клітинки не знайдено."
                ),
            )

            warnings += 1
            continue

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            append_report(
                source_image=source_image_text,
                ocr_value=ocr_value,
                confirmed_value=confirmed_value,
                status="WARNING",
                details=(
                    "Не вдалося відкрити "
                    "зображення."
                ),
            )

            warnings += 1
            continue

        mask = build_mask(
            image
        )

        segments = split_digits(
            mask
        )

        if len(segments) != len(
            confirmed_value
        ):
            append_report(
                source_image=source_image_text,
                ocr_value=ocr_value,
                confirmed_value=confirmed_value,
                status="WARNING",
                details=(
                    f"Сегментів: {len(segments)}, "
                    f"цифр: {len(confirmed_value)}."
                ),
            )

            warnings += 1
            continue

        row_imported = 0

        for index, (
            segment,
            digit,
        ) in enumerate(
            zip(
                segments,
                confirmed_value,
            ),
            start=1,
        ):
            ok, encoded = cv2.imencode(
                ".png",
                segment.image,
            )

            if not ok:
                continue

            digest = file_hash(
                encoded.tobytes()
            )

            sample_id = (
                "manual_"
                + digest[:16]
            )

            if sample_id in known_ids:
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
                f"manual_"
                f"{source_path.stem}_"
                f"d{index}_"
                f"{digest[:10]}.png"
            )

            destination_path = (
                destination_dir
                / destination_name
            )

            cv2.imwrite(
                str(destination_path),
                segment.image,
            )

            relative_path = (
                destination_path
                .relative_to(
                    PROJECT_DIR
                )
                .as_posix()
            )

            add_sample(
                sample_id=sample_id,
                image_path=relative_path,
                digit=digit,
                status="CONFIRMED",
                source_pdf="",
                row=correction.get(
                    "row",
                    "",
                ),
                column=correction.get(
                    "column",
                    "",
                ),
                day=correction.get(
                    "day",
                    "",
                ),
                ocr_value=ocr_value,
                ocr_confidence="",
            )

            known_ids.add(
                sample_id
            )

            imported_digits += 1
            row_imported += 1

        append_report(
            source_image=source_image_text,
            ocr_value=ocr_value,
            confirmed_value=confirmed_value,
            status="IMPORTED",
            details=(
                f"Додано цифр: "
                f"{row_imported}"
            ),
        )

        already_imported.add(
            unique_key
        )

        imported_rows += 1

    print(
        f"Підтверджень імпортовано: "
        f"{imported_rows}"
    )

    print(
        f"Нових цифр додано:        "
        f"{imported_digits}"
    )

    print(
        f"Вже оброблено:            "
        f"{skipped}"
    )

    print(
        f"Потребують уваги:         "
        f"{warnings}"
    )

    print()
    print(
        "Звіт:"
    )

    print(
        REPORT_PATH
    )

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
