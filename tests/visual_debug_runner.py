from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


from app.ocr.digit_splitter import (
    detect_digit_ranges,
    split_digits,
)

from ocr_engine_v2 import (
    build_blue_mask,
)


SOURCE_DIR = (
    PROJECT_DIR
    / "debug"
    / "ocr_engine_v2"
    / "source_cells"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "tests"
    / "outputs"
    / "splitter"
)


def save_image(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(path),
        image,
    ):
        raise RuntimeError(
            f"Не вдалося зберегти: {path}"
        )


def create_projection_image(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Створює графік вертикальної проєкції чорнила.
    Чим вища колонка — тим більше пікселів рукопису
    знаходиться у відповідній X-координаті.
    """
    height, width = mask.shape[:2]

    projection = np.count_nonzero(
        mask,
        axis=0,
    ).astype(np.float32)

    canvas_height = 300

    canvas = np.full(
        (
            canvas_height,
            width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    maximum = float(
        projection.max(
            initial=1
        )
    )

    for x, value in enumerate(
        projection
    ):
        bar_height = int(
            round(
                (
                    value
                    / maximum
                )
                * (
                    canvas_height
                    - 20
                )
            )
        )

        cv2.line(
            canvas,
            (
                x,
                canvas_height - 1,
            ),
            (
                x,
                canvas_height
                - 1
                - bar_height,
            ),
            (
                0,
                0,
                0,
            ),
            1,
        )

    return canvas


def create_split_overlay(
    original: np.ndarray,
    ranges: list[
        tuple[int, int]
    ],
) -> np.ndarray:
    overlay = original.copy()

    image_height = overlay.shape[0]

    for index, (
        x1,
        x2,
    ) in enumerate(
        ranges,
        start=1,
    ):
        cv2.rectangle(
            overlay,
            (
                x1,
                1,
            ),
            (
                x2,
                image_height - 2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

        cv2.putText(
            overlay,
            str(index),
            (
                x1 + 3,
                20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    return overlay


def process_file(
    source_path: Path,
) -> dict:
    image = cv2.imread(
        str(source_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Не вдалося відкрити: {source_path}"
        )

    sample_output_dir = (
        OUTPUT_DIR
        / source_path.stem
    )

    sample_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        sample_output_dir
        / "01_original.png",
        image,
    )

    mask = build_blue_mask(
        image
    )

    save_image(
        sample_output_dir
        / "02_blue_mask.png",
        mask,
    )

    projection_image = (
        create_projection_image(
            mask
        )
    )

    save_image(
        sample_output_dir
        / "03_projection.png",
        projection_image,
    )

    ranges = detect_digit_ranges(
        mask
    )

    overlay = create_split_overlay(
        image,
        ranges,
    )

    save_image(
        sample_output_dir
        / "04_split_overlay.png",
        overlay,
    )

    segments = split_digits(
        mask
    )

    segment_data = []

    for segment in segments:
        file_name = (
            f"05_digit_"
            f"{segment.index}.png"
        )

        save_image(
            sample_output_dir
            / file_name,
            segment.image,
        )

        segment_data.append({
            "index": segment.index,
            "x1": segment.x1,
            "x2": segment.x2,
            "width": (
                segment.x2
                - segment.x1
            ),
            "file": file_name,
        })

    report = {
        "source_file": (
            source_path.name
        ),
        "source_path": str(
            source_path
        ),
        "image_width": int(
            image.shape[1]
        ),
        "image_height": int(
            image.shape[0]
        ),
        "detected_digits": len(
            segments
        ),
        "ranges": [
            {
                "x1": int(x1),
                "x2": int(x2),
                "width": int(
                    x2 - x1
                ),
            }
            for x1, x2 in ranges
        ],
        "segments": segment_data,
    }

    report_path = (
        sample_output_dir
        / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


def main() -> None:
    print()
    print("=" * 58)
    print(
        "FERRETTI OCR — VISUAL DIGIT SPLITTER DEBUG"
    )
    print("=" * 58)
    print()

    if not SOURCE_DIR.exists():
        print(
            "Не знайдено:"
        )
        print(
            SOURCE_DIR
        )
        print()
        print(
            "Спочатку виконайте "
            "«Розпізнати цифри» "
            "у FERRETTI OCR."
        )
        return

    source_files = sorted(
        SOURCE_DIR.glob(
            "*.png"
        )
    )

    if not source_files:
        print(
            "У source_cells немає PNG."
        )
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    one_digit = 0
    two_digits = 0
    failed = 0

    summary = []

    for source_path in source_files:
        try:
            report = process_file(
                source_path
            )
        except Exception as error:
            print(
                f"ERROR: "
                f"{source_path.name}: "
                f"{error}"
            )

            failed += 1
            continue

        detected = int(
            report[
                "detected_digits"
            ]
        )

        total += 1

        if detected == 1:
            one_digit += 1

        elif detected == 2:
            two_digits += 1

        else:
            failed += 1

        summary.append({
            "file": (
                source_path.name
            ),
            "digits": detected,
        })

        print(
            f"{source_path.name:<30} "
            f"→ {detected} digit(s)"
        )

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                "total": total,
                "one_digit": one_digit,
                "two_digits": two_digits,
                "failed": failed,
                "samples": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("-" * 58)

    print(
        f"Оброблено клітинок:   {total}"
    )

    print(
        f"1 цифра:              {one_digit}"
    )

    print(
        f"2 цифри:              {two_digits}"
    )

    print(
        f"Не визначено:         {failed}"
    )

    print()
    print(
        "Результати:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print("=" * 58)


if __name__ == "__main__":
    main()
