from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ocr.cell_geometry import (
    CellGeometry,
    Rect,
    load_geometry_settings,
)
from core.grid_detector import (
    GridDetectionResult,
    detect_horizontal_grid_lines,
)


@dataclass(frozen=True)
class CellResult:
    row: int
    column: int
    day: str

    x1: int
    y1: int
    x2: int
    y2: int

    detection_x1: int
    detection_y1: int
    detection_x2: int
    detection_y2: int

    ocr_x1: int
    ocr_y1: int
    ocr_x2: int
    ocr_y2: int

    ink_ratio: float
    largest_component_area: int
    is_filled: bool
    image_path: Path | None = None


@dataclass(frozen=True)
class CellAnalysisResult:
    cells: list[CellResult]
    detected_rows: int
    total_cells: int
    filled_cells: int
    horizontal_lines: list[int]
    grid_detection: GridDetectionResult


DAY_NAMES = (
    "MON",
    "TUE",
    "WED",
    "THU",
    "FRI",
    "SAT",
)


def load_form_config(
    config_path: Path,
) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Не знайдено конфігурацію шаблону: {config_path}"
        )

    try:
        config = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"Не вдалося прочитати конфігурацію: {error}"
        ) from error

    required = (
        "table_grid",
        "day_columns",
    )

    missing = [
        key
        for key in required
        if key not in config
    ]

    if missing:
        raise ValueError(
            "У конфігурації відсутні поля: "
            + ", ".join(missing)
        )

    return config


def build_boundaries(
    start: int,
    end: int,
    count: int,
) -> list[int]:
    if count <= 0:
        raise ValueError(
            "Кількість колонок має бути більшою за нуль."
        )

    return [
        int(round(value))
        for value in np.linspace(
            start,
            end,
            count + 1,
        )
    ]


def build_blue_ink_mask(
    cell_bgr: np.ndarray,
) -> np.ndarray:
    """
    Виділяє лише синє або синьо-фіолетове чорнило.

    На відміну від попередньої версії, не використовує
    надто м'який поріг для блідих пікселів, через який
    чорні та сірі залишки ліній таблиці помилково
    вважалися рукописом.
    """
    if cell_bgr.size == 0:
        raise ValueError(
            "Отримано порожнє зображення клітинки."
        )

    hsv = cv2.cvtColor(
        cell_bgr,
        cv2.COLOR_BGR2HSV,
    )

    blue, green, red = cv2.split(
        cell_bgr
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    blue_i = blue.astype(
        np.int16
    )
    green_i = green.astype(
        np.int16
    )
    red_i = red.astype(
        np.int16
    )

    # Синя/синьо-фіолетова ручка:
    # - достатня насиченість;
    # - синій канал помітно сильніший за зелений;
    # - піксель не білий і не майже чорний.
    strict_blue = (
        (hue >= 90)
        & (hue <= 178)
        & (saturation >= 38)
        & (value >= 35)
        & (value <= 250)
        & ((blue_i - green_i) >= 8)
        & (
            (blue_i - red_i) >= -35
        )
    )

    mask = np.where(
        strict_blue,
        255,
        0,
    ).astype(np.uint8)

    # Легке з'єднання розірваних штрихів без
    # сильного потовщення рукопису.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    image_height, image_width = (
        mask.shape[:2]
    )

    for component_index in range(
        1,
        component_count,
    ):
        x = int(
            stats[
                component_index,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            stats[
                component_index,
                cv2.CC_STAT_TOP,
            ]
        )
        width = int(
            stats[
                component_index,
                cv2.CC_STAT_WIDTH,
            ]
        )
        height = int(
            stats[
                component_index,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        area = int(
            stats[
                component_index,
                cv2.CC_STAT_AREA,
            ]
        )

        touches_top_or_bottom = (
            y <= 1
            or y + height
            >= image_height - 1
        )

        touches_left_or_right = (
            x <= 1
            or x + width
            >= image_width - 1
        )

        line_like = (
            width >= 18
            and height <= 4
            and width >= height * 5
        )

        tiny_border_noise = (
            area < 35
            and (
                touches_top_or_bottom
                or touches_left_or_right
            )
        )

        if (
            area >= 10
            and width >= 2
            and height >= 3
            and not line_like
            and not tiny_border_noise
        ):
            cleaned[
                labels == component_index
            ] = 255

    return cleaned

def measure_blue_ink(
    blue_mask: np.ndarray,
) -> tuple[float, int]:
    total_pixels = int(
        blue_mask.size
    )

    if total_pixels == 0:
        return 0.0, 0

    ink_pixels = int(
        cv2.countNonZero(
            blue_mask
        )
    )

    ink_ratio = (
        ink_pixels / total_pixels
    )

    component_count, _labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            blue_mask,
            connectivity=8,
        )
    )

    largest_component_area = 0

    if component_count > 1:
        largest_component_area = int(
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ].max()
        )

    return (
        ink_ratio,
        largest_component_area,
    )


def is_cell_filled(
    ink_ratio: float,
    largest_component_area: int,
) -> bool:
    return (
        ink_ratio >= 0.0012
        and largest_component_area >= 10
    ) or largest_component_area >= 24


def analyze_cells(
    normalized_bgr: np.ndarray,
    config_path: Path,
    output_dir: Path,
    preview_path: Path,
    geometry_config_path: Path,
    *,
    save_all_cells: bool = False,
) -> CellAnalysisResult:
    config = load_form_config(
        config_path
    )

    geometry_settings = (
        load_geometry_settings(
            geometry_config_path
        )
    )

    geometry = CellGeometry(
        geometry_settings
    )

    grid = config["table_grid"]

    x1 = int(grid["x1"])
    y1 = int(grid["y1"])
    x2 = int(grid["x2"])
    y2 = int(grid["y2"])

    day_columns = int(
        config["day_columns"]
    )

    image_height, image_width = (
        normalized_bgr.shape[:2]
    )

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError(
            "Координати table_grid виходять "
            "за межі вирівняного зображення."
        )

    if day_columns != len(DAY_NAMES):
        raise ValueError(
            f"Шаблон має {day_columns} колонок, "
            "але підтримується 6 днів."
        )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_detection = (
        detect_horizontal_grid_lines(
            normalized_bgr,
            grid,
        )
    )

    row_ranges = (
        grid_detection.row_ranges
    )

    x_boundaries = build_boundaries(
        x1,
        x2,
        day_columns,
    )

    preview = normalized_bgr.copy()
    results: list[CellResult] = []

    for row_index, (
        cell_y1,
        cell_y2,
    ) in enumerate(row_ranges):
        for column_index in range(
            day_columns
        ):
            cell_x1 = (
                x_boundaries[
                    column_index
                ]
            )

            cell_x2 = (
                x_boundaries[
                    column_index + 1
                ]
            )

            cell_rect = Rect(
                x1=cell_x1,
                y1=cell_y1,
                x2=cell_x2,
                y2=cell_y2,
            )

            detection_rect = (
                geometry.detection_rect(
                    cell_rect
                ).clamp(
                    image_width,
                    image_height,
                )
            )

            ocr_rect = (
                geometry.ocr_rect(
                    cell_rect
                ).clamp(
                    image_width,
                    image_height,
                )
            )

            cell_bgr = normalized_bgr[
                detection_rect.y1:
                detection_rect.y2,
                detection_rect.x1:
                detection_rect.x2,
            ]

            blue_mask = build_blue_ink_mask(
                cell_bgr
            )

            (
                ink_ratio,
                largest_component_area,
            ) = measure_blue_ink(
                blue_mask
            )

            filled = is_cell_filled(
                ink_ratio,
                largest_component_area,
            )

            image_path: Path | None = None

            if filled or save_all_cells:
                state_folder = (
                    output_dir
                    / (
                        "filled"
                        if filled
                        else "empty"
                    )
                )

                state_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                file_name = (
                    f"r{row_index + 1:02d}_"
                    f"c{column_index + 1}_"
                    f"{DAY_NAMES[column_index]}.png"
                )

                image_path = (
                    state_folder
                    / file_name
                )

                ocr_image = cv2.bitwise_not(
                    blue_mask
                )

                if not cv2.imwrite(
                    str(image_path),
                    ocr_image,
                ):
                    raise RuntimeError(
                        f"Не вдалося зберегти {image_path}"
                    )

            color = (
                (0, 180, 0)
                if filled
                else (170, 170, 170)
            )

            thickness = (
                4
                if filled
                else 1
            )

            cv2.rectangle(
                preview,
                (
                    cell_rect.x1,
                    cell_rect.y1,
                ),
                (
                    cell_rect.x2,
                    cell_rect.y2,
                ),
                color,
                thickness,
            )

            if (
                filled
                and geometry_settings.debug_draw_ocr_zone
            ):
                cv2.rectangle(
                    preview,
                    (
                        ocr_rect.x1,
                        ocr_rect.y1,
                    ),
                    (
                        ocr_rect.x2,
                        ocr_rect.y2,
                    ),
                    (255, 180, 0),
                    2,
                )

            if filled:
                cv2.putText(
                    preview,
                    (
                        f"{row_index + 1}/"
                        f"{column_index + 1}"
                    ),
                    (
                        cell_rect.x1 + 8,
                        cell_rect.y1 + 24,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            results.append(
                CellResult(
                    row=row_index + 1,
                    column=column_index + 1,
                    day=DAY_NAMES[
                        column_index
                    ],
                    x1=cell_rect.x1,
                    y1=cell_rect.y1,
                    x2=cell_rect.x2,
                    y2=cell_rect.y2,
                    detection_x1=(
                        detection_rect.x1
                    ),
                    detection_y1=(
                        detection_rect.y1
                    ),
                    detection_x2=(
                        detection_rect.x2
                    ),
                    detection_y2=(
                        detection_rect.y2
                    ),
                    ocr_x1=ocr_rect.x1,
                    ocr_y1=ocr_rect.y1,
                    ocr_x2=ocr_rect.x2,
                    ocr_y2=ocr_rect.y2,
                    ink_ratio=ink_ratio,
                    largest_component_area=(
                        largest_component_area
                    ),
                    is_filled=filled,
                    image_path=image_path,
                )
            )

    if not cv2.imwrite(
        str(preview_path),
        preview,
    ):
        raise RuntimeError(
            f"Не вдалося зберегти preview: {preview_path}"
        )

    filled_count = sum(
        1
        for result in results
        if result.is_filled
    )

    return CellAnalysisResult(
        cells=results,
        detected_rows=len(
            row_ranges
        ),
        total_cells=len(
            results
        ),
        filled_cells=filled_count,
        horizontal_lines=(
            grid_detection.horizontal_lines
        ),
        grid_detection=grid_detection,
    )
