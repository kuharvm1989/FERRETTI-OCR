from __future__ import annotations

import csv
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from dataset_manifest import add_sample
from ocr_engine_v2 import (
    configure_tesseract,
    tesseract_candidate,
)


BASE_DIR = Path(__file__).resolve().parent

SEGMENTS_DIR = (
    BASE_DIR
    / "debug"
    / "ocr_engine_v2"
    / "segments"
)

DATASET_DIR = BASE_DIR / "dataset"
PENDING_DIR = DATASET_DIR / "pending"

OLD_MANIFEST_PATH = (
    DATASET_DIR
    / "pending_manifest.csv"
)


SEGMENT_PATTERN = re.compile(
    r"r(?P<row>\d+)_"
    r"c(?P<column>\d+)_"
    r"(?P<day>[A-Z]+)_"
    r"digit_(?P<digit_index>\d+)"
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

            hasher.update(chunk)

    return hasher.hexdigest()


def existing_hashes() -> set[str]:
    if not OLD_MANIFEST_PATH.exists():
        return set()

    result: set[str] = set()

    with OLD_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            digest = (
                row.get(
                    "sha256",
                    ""
                )
                .strip()
            )

            if digest:
                result.add(digest)

    return result


def ensure_old_manifest() -> None:
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PENDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OLD_MANIFEST_PATH.exists():
        return

    with OLD_MANIFEST_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(
            csv_file,
            delimiter=";",
        )

        writer.writerow([
            "id",
            "file_name",
            "source_file",
            "sha256",
            "created_at",
            "status",
            "confirmed_digit",
        ])


def parse_segment_name(
    file_path: Path,
) -> dict[str, str]:
    match = SEGMENT_PATTERN.search(
        file_path.stem
    )

    if match is None:
        return {
            "row": "",
            "column": "",
            "day": "",
            "digit_index": "",
        }

    return {
        "row": match.group("row"),
        "column": match.group("column"),
        "day": match.group("day"),
        "digit_index": match.group(
            "digit_index"
        ),
    }


def recognize_segment(
    image_path: Path,
) -> tuple[str, str]:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return "", ""

    candidate = tesseract_candidate(
        image,
        psm=10,
        method="dataset_collector",
    )

    if candidate is None:
        return "", ""

    confidence = max(
        0.0,
        min(
            100.0,
            candidate.confidence,
        ),
    )

    return (
        candidate.value,
        f"{confidence:.2f}",
    )


def collect_segments() -> None:
    ensure_old_manifest()

    if not SEGMENTS_DIR.exists():
        print()
        print(
            "Не знайдено папку сегментів:"
        )
        print(SEGMENTS_DIR)
        print()
        print(
            "Спочатку виконайте "
            "«Розпізнати цифри» "
            "у FERRETTI OCR."
        )
        return

    image_files = sorted(
        SEGMENTS_DIR.glob(
            "*.png"
        )
    )

    if not image_files:
        print()
        print(
            "У папці segments немає PNG."
        )
        return

    configure_tesseract()

    known_hashes = existing_hashes()

    added = 0
    skipped = 0
    recognized = 0

    with OLD_MANIFEST_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(
            csv_file,
            delimiter=";",
        )

        for source_path in image_files:
            digest = file_hash(
                source_path
            )

            if digest in known_hashes:
                skipped += 1
                continue

            sample_id = digest[:12]

            destination_name = (
                f"{source_path.stem}_"
                f"{sample_id}.png"
            )

            destination_path = (
                PENDING_DIR
                / destination_name
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

            metadata = parse_segment_name(
                source_path
            )

            (
                ocr_value,
                ocr_confidence,
            ) = recognize_segment(
                source_path
            )

            if ocr_value:
                recognized += 1

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
                status="PENDING",
                row=metadata["row"],
                column=metadata["column"],
                day=metadata["day"],
                ocr_value=ocr_value,
                ocr_confidence=ocr_confidence,
            )

            writer.writerow([
                sample_id,
                destination_name,
                str(source_path),
                digest,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                "PENDING",
                "",
            ])

            known_hashes.add(
                digest
            )

            added += 1

    print()
    print("=" * 50)
    print(
        "FERRETTI OCR — DATASET COLLECTOR"
    )
    print("=" * 50)

    print()
    print(
        f"Додано нових цифр:     {added}"
    )

    print(
        f"OCR зміг розпізнати:   {recognized}"
    )

    print(
        f"Пропущено дублікатів:  {skipped}"
    )

    print()
    print(
        f"Pending:"
    )
    print(PENDING_DIR)

    print()
    print(
        "Dataset manifest:"
    )

    print(
        DATASET_DIR
        / "dataset_manifest.csv"
    )

    print()
    print("=" * 50)


if __name__ == "__main__":
    collect_segments()
