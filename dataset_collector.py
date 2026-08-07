from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SEGMENTS_DIR = (
    BASE_DIR
    / "debug"
    / "ocr_engine_v2"
    / "segments"
)

DATASET_DIR = BASE_DIR / "dataset"
PENDING_DIR = DATASET_DIR / "pending"

MANIFEST_PATH = (
    DATASET_DIR
    / "pending_manifest.csv"
)


def file_hash(
    file_path: Path,
) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def existing_hashes() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()

    result: set[str] = set()

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            digest = row.get(
                "sha256",
                "",
            ).strip()

            if digest:
                result.add(digest)

    return result


def ensure_manifest() -> None:
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PENDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MANIFEST_PATH.exists():
        return

    with MANIFEST_PATH.open(
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


def collect_segments() -> None:
    ensure_manifest()

    if not SEGMENTS_DIR.exists():
        print(
            "Не знайдено папку сегментів:"
        )
        print(
            SEGMENTS_DIR
        )
        print()
        print(
            "Спочатку запустіть OCR "
            "для заповненого листа."
        )
        return

    known_hashes = existing_hashes()

    image_files = sorted(
        list(
            SEGMENTS_DIR.glob("*.png")
        )
    )

    if not image_files:
        print(
            "У папці segments немає PNG."
        )
        return

    added = 0
    skipped = 0

    with MANIFEST_PATH.open(
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

            short_hash = digest[:12]

            destination_name = (
                f"{source_path.stem}_"
                f"{short_hash}.png"
            )

            destination_path = (
                PENDING_DIR
                / destination_name
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

            writer.writerow([
                short_hash,
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
    print(
        f"Додано нових цифр: {added}"
    )

    print(
        f"Пропущено дублікатів: {skipped}"
    )

    print()
    print(
        f"Pending:"
    )
    print(
        PENDING_DIR
    )

    print()
    print(
        f"Manifest:"
    )
    print(
        MANIFEST_PATH
    )


if __name__ == "__main__":
    collect_segments()
