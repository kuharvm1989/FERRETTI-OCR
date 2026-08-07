from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "debug"
    / "page_001_300dpi.png"
)

CONFIG_PATH = (
    BASE_DIR
    / "config"
    / "form_v1.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "debug"
    / "cells"
)

GRID_PREVIEW_PATH = (
    BASE_DIR
    / "debug"
    / "cells_grid_preview.png"
)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено конфігурацію: {CONFIG_PATH}"
        )

    return json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def build_boundaries(
    start: int,
    end: int,
    count: int,
) -> list[int]:
    """
    Рівномірно ділить діапазон і повертає
    координати всіх меж, включно з крайніми.
    """
    return [
        round(value)
        for value in np.linspace(
            start,
            end,
            count + 1,
        )
    ]


def crop_cell(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """
    Вирізає клітинку без самих ліній таблиці.
    """
    horizontal_margin = 7
    vertical_margin = 7

    inner_x1 = x1 + horizontal_margin
    inner_y1 = y1 + vertical_margin
    inner_x2 = x2 - horizontal_margin
    inner_y2 = y2 - vertical_margin

    if (
        inner_x2 <= inner_x1
        or inner_y2 <= inner_y1
    ):
        raise ValueError(
            "Клітинка замала після відступів."
        )

    return image[
        inner_y1:inner_y2,
        inner_x1:inner_x2,
    ]


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено PNG: {IMAGE_PATH}"
        )

    config = load_config()

    grid = config["data_grid"]

    x1 = int(grid["x1"])
    y1 = int(grid["y1"])
    x2 = int(grid["x2"])
    y2 = int(grid["y2"])

    day_columns = int(
        config["day_columns"]
    )

    operation_rows = int(
        config["operation_rows"]
    )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            "OpenCV не зміг відкрити сторінку."
        )

    image_height, image_width = (
        image.shape[:2]
    )

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError(
            "Координати сітки виходять "
            "за межі зображення."
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_boundaries = build_boundaries(
        x1,
        x2,
        day_columns,
    )

    y_boundaries = build_boundaries(
        y1,
        y2,
        operation_rows,
    )

    day_names = [
        "MON",
        "TUE",
        "WED",
        "THU",
        "FRI",
        "SAT",
    ]

    preview = image.copy()

    # Межі колонок.
    for x in x_boundaries:
        cv2.line(
            preview,
            (x, y1),
            (x, y2),
            (0, 0, 255),
            3,
        )

    # Межі рядків.
    for y in y_boundaries:
        cv2.line(
            preview,
            (x1, y),
            (x2, y),
            (255, 0, 0),
            2,
        )

    cell_count = 0

    for row_index in range(
        operation_rows
    ):
        row_folder = (
            OUTPUT_DIR
            / f"row_{row_index + 1:02d}"
        )

        row_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        cell_y1 = y_boundaries[row_index]
        cell_y2 = y_boundaries[row_index + 1]

        for column_index in range(
            day_columns
        ):
            cell_x1 = x_boundaries[column_index]
            cell_x2 = x_boundaries[column_index + 1]

            cell = crop_cell(
                image,
                cell_x1,
                cell_y1,
                cell_x2,
                cell_y2,
            )

            file_name = (
                f"r{row_index + 1:02d}_"
                f"c{column_index + 1}_"
                f"{day_names[column_index]}.png"
            )

            output_path = (
                row_folder / file_name
            )

            if not cv2.imwrite(
                str(output_path),
                cell,
            ):
                raise RuntimeError(
                    f"Не вдалося зберегти "
                    f"{output_path}"
                )

            cell_count += 1

    if not cv2.imwrite(
        str(GRID_PREVIEW_PATH),
        preview,
    ):
        raise RuntimeError(
            "Не вдалося зберегти "
            "попередній перегляд сітки."
        )

    grid_width = x2 - x1
    grid_height = y2 - y1

    print(
        f"Робоча область: "
        f"{grid_width} × {grid_height} px"
    )

    print(
        f"Колонок: {day_columns}"
    )

    print(
        f"Рядків: {operation_rows}"
    )

    print(
        f"Створено клітинок: {cell_count}"
    )

    print(
        f"Орієнтовний розмір клітинки: "
        f"{grid_width / day_columns:.1f} × "
        f"{grid_height / operation_rows:.1f} px"
    )

    print(
        f"Перегляд сітки: "
        f"{GRID_PREVIEW_PATH}"
    )

    print(
        f"Клітинки: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
