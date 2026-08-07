from __future__ import annotations

import json
from pathlib import Path

import cv2

from core.grid_detector import (
    detect_horizontal_grid_lines,
    draw_grid_detection_preview,
)


BASE_DIR = Path(__file__).resolve().parent

CONFIG_PATH = (
    BASE_DIR
    / "config"
    / "form_v2.json"
)

NORMALIZED_IMAGE_PATH = (
    BASE_DIR
    / "debug"
    / "page_001_normalized.png"
)

PREVIEW_PATH = (
    BASE_DIR
    / "debug"
    / "grid_detection_preview.png"
)

MASK_PATH = (
    BASE_DIR
    / "debug"
    / "horizontal_lines_mask.png"
)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено конфігурацію:\n"
            f"{CONFIG_PATH}"
        )

    if not NORMALIZED_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено вирівняну сторінку:\n"
            f"{NORMALIZED_IMAGE_PATH}\n\n"
            "Спочатку відкрийте PDF у програмі "
            "та натисніть «Знайти мітки та вирівняти»."
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    grid_bounds = config["table_grid"]

    normalized_image = cv2.imread(
        str(NORMALIZED_IMAGE_PATH)
    )

    if normalized_image is None:
        raise RuntimeError(
            "OpenCV не зміг відкрити "
            "вирівняну сторінку."
        )

    result = detect_horizontal_grid_lines(
        normalized_image,
        grid_bounds,
    )

    preview = draw_grid_detection_preview(
        normalized_image,
        result,
        grid_bounds,
    )

    if not cv2.imwrite(
        str(PREVIEW_PATH),
        preview,
    ):
        raise RuntimeError(
            "Не вдалося зберегти preview."
        )

    if not cv2.imwrite(
        str(MASK_PATH),
        result.detection_mask,
    ):
        raise RuntimeError(
            "Не вдалося зберегти маску."
        )

    print(
        f"Знайдено горизонтальних ліній: "
        f"{len(result.horizontal_lines)}"
    )

    print(
        f"Виявлено рядків таблиці: "
        f"{len(result.row_ranges)}"
    )

    print("\nКоординати ліній:")

    for index, coordinate in enumerate(
        result.horizontal_lines,
        start=1,
    ):
        print(
            f"{index:02d}: y={coordinate}"
        )

    print(
        f"\nPreview:\n{PREVIEW_PATH}"
    )

    print(
        f"\nМаска горизонтальних ліній:\n"
        f"{MASK_PATH}"
    )


if __name__ == "__main__":
    main()
