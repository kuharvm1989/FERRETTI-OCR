from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GridDetectionResult:
    horizontal_lines: list[int]
    row_ranges: list[tuple[int, int]]
    detection_mask: np.ndarray
    used_fallback: bool = False


def group_nearby_coordinates(
    coordinates: list[int],
    maximum_gap: int = 4,
) -> list[int]:
    if not coordinates:
        return []

    sorted_coordinates = sorted(coordinates)
    groups: list[list[int]] = [[sorted_coordinates[0]]]

    for coordinate in sorted_coordinates[1:]:
        if coordinate - groups[-1][-1] <= maximum_gap:
            groups[-1].append(coordinate)
        else:
            groups.append([coordinate])

    return [
        int(round(sum(group) / len(group)))
        for group in groups
    ]


def remove_lines_too_close(
    coordinates: list[int],
    minimum_distance: int = 12,
) -> list[int]:
    if not coordinates:
        return []

    result = [coordinates[0]]

    for coordinate in coordinates[1:]:
        if coordinate - result[-1] >= minimum_distance:
            result.append(coordinate)
        else:
            result[-1] = int(round((result[-1] + coordinate) / 2))

    return result


def build_row_ranges(
    horizontal_lines: list[int],
    minimum_row_height: int = 20,
    maximum_row_height: int = 160,
) -> list[tuple[int, int]]:
    row_ranges: list[tuple[int, int]] = []

    for first_line, second_line in zip(
        horizontal_lines,
        horizontal_lines[1:],
    ):
        row_height = second_line - first_line
        if minimum_row_height <= row_height <= maximum_row_height:
            row_ranges.append((first_line, second_line))

    return row_ranges


def detect_horizontal_grid_lines(
    normalized_image: np.ndarray,
    grid_bounds: dict[str, int],
    minimum_line_coverage: float = 0.55,
) -> GridDetectionResult:
    if normalized_image is None:
        raise ValueError("Нормалізоване зображення відсутнє.")

    x1 = int(grid_bounds["x1"])
    y1 = int(grid_bounds["y1"])
    x2 = int(grid_bounds["x2"])
    y2 = int(grid_bounds["y2"])

    image_height, image_width = normalized_image.shape[:2]
    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError(
            "Координати grid_bounds виходять за межі зображення."
        )

    grid_crop = normalized_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(grid_crop, cv2.COLOR_BGR2GRAY)

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    crop_width = binary.shape[1]
    kernel_width = max(80, int(crop_width * 0.35))

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )

    horizontal_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    horizontal_mask = cv2.dilate(
        horizontal_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)),
        iterations=1,
    )

    row_projection = np.count_nonzero(horizontal_mask, axis=1)
    minimum_pixels = int(crop_width * minimum_line_coverage)

    candidate_coordinates = np.where(
        row_projection >= minimum_pixels
    )[0].tolist()

    grouped_local_lines = group_nearby_coordinates(
        candidate_coordinates,
        maximum_gap=4,
    )
    grouped_local_lines = remove_lines_too_close(
        grouped_local_lines,
        minimum_distance=12,
    )

    horizontal_lines = [y1 + coordinate for coordinate in grouped_local_lines]
    row_ranges = build_row_ranges(
        horizontal_lines,
        minimum_row_height=20,
        maximum_row_height=160,
    )

    if len(horizontal_lines) < 2 or not row_ranges:
        raise RuntimeError(
            "Не вдалося автоматично визначити горизонтальні рядки таблиці."
        )

    return GridDetectionResult(
        horizontal_lines=horizontal_lines,
        row_ranges=row_ranges,
        detection_mask=horizontal_mask,
        used_fallback=False,
    )


def draw_grid_detection_preview(
    normalized_image: np.ndarray,
    result: GridDetectionResult,
    grid_bounds: dict[str, int],
) -> np.ndarray:
    preview = normalized_image.copy()
    x1 = int(grid_bounds["x1"])
    x2 = int(grid_bounds["x2"])

    for index, y_coordinate in enumerate(result.horizontal_lines, start=1):
        cv2.line(
            preview,
            (x1, y_coordinate),
            (x2, y_coordinate),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            preview,
            str(index),
            (x1 + 10, max(30, y_coordinate - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return preview
