from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

from app.config.project_paths import (
    INPUT_DIR,
    DEBUG_DIR,
)


BASE_DIR = Path(__file__).resolve().parent

INPUT_PDF = (
    INPUT_DIR
    / "TRAINING_03.PDF"
)

DEBUG_DIR = (
    BASE_DIR
    / "debug"
    / "training_sheet"
)

PAGE_IMAGE_PATH = (
    DEBUG_DIR
    / "01_page.png"
)

ALIGNED_PATH = (
    DEBUG_DIR
    / "02_aligned.png"
)

PREVIEW_PATH = (
    DEBUG_DIR
    / "03_grid_preview.png"
)

CELLS_DIR = (
    DEBUG_DIR
    / "cells_preview"
)


# Нормалізований A4 при 300 dpi.
PAGE_WIDTH = 2480
PAGE_HEIGHT = 3508

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


# Координати сітки з training_sheet_generator.py.
GRID_LEFT_MM = 17.0
GRID_RIGHT_MM = 198.0
GRID_TOP_MM = 38.0
GRID_BOTTOM_MM = 278.0

LABEL_WIDTH_MM = 10.0

ROWS = 10
COLUMNS = 20


def mm_to_x(
    value_mm: float,
) -> int:
    return int(
        round(
            value_mm
            / A4_WIDTH_MM
            * PAGE_WIDTH
        )
    )


def mm_to_y(
    value_mm: float,
) -> int:
    return int(
        round(
            value_mm
            / A4_HEIGHT_MM
            * PAGE_HEIGHT
        )
    )


def render_pdf() -> np.ndarray:
    if not INPUT_PDF.exists():
        raise FileNotFoundError(
            f"Не знайдено PDF:\n{INPUT_PDF}"
        )

    document = fitz.open(
        INPUT_PDF
    )

    if document.page_count < 1:
        document.close()

        raise RuntimeError(
            "PDF не містить сторінок."
        )

    page = document[0]

    matrix = fitz.Matrix(
        300 / 72,
        300 / 72,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )

    document.close()

    if pixmap.n == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )
    elif pixmap.n == 4:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR,
        )

    return image


def detect_markers(
    image: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Пробує кілька стандартних словників ArUco.
    Потрібні ID 0,1,2,3.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    dictionary_ids = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_6X6_50,
    ]

    for dictionary_id in dictionary_ids:
        dictionary = (
            cv2.aruco.getPredefinedDictionary(
                dictionary_id
            )
        )

        parameters = (
            cv2.aruco.DetectorParameters()
        )

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

        corners, ids, _rejected = (
            detector.detectMarkers(
                gray
            )
        )

        if ids is None:
            continue

        detected: dict[
            int,
            np.ndarray,
        ] = {}

        for marker_corners, marker_id in zip(
            corners,
            ids.flatten(),
        ):
            detected[
                int(marker_id)
            ] = np.asarray(
                marker_corners,
                dtype=np.float32,
            ).reshape(
                4,
                2,
            )

        if all(
            marker_id in detected
            for marker_id in (
                0,
                1,
                2,
                3,
            )
        ):
            return detected

    raise RuntimeError(
        "Не вдалося знайти всі "
        "ArUco ID 0,1,2,3."
    )


def inner_marker_point(
    detected: dict[
        int,
        np.ndarray,
    ],
    marker_id: int,
) -> np.ndarray:
    """
    OpenCV:
    0 = top-left
    1 = top-right
    2 = bottom-right
    3 = bottom-left
    """

    corner_index = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }[marker_id]

    return detected[
        marker_id
    ][
        corner_index
    ]


def align_page(
    image: np.ndarray,
    detected: dict[
        int,
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Не розтягуємо прямокутник міток
    на весь A4.

    Ми знаємо реальні координати
    внутрішніх кутів міток на шаблоні
    і будуємо homography у систему A4.
    """

    source = np.float32([
        inner_marker_point(
            detected,
            0,
        ),
        inner_marker_point(
            detected,
            1,
        ),
        inner_marker_point(
            detected,
            2,
        ),
        inner_marker_point(
            detected,
            3,
        ),
    ])

    # Мітки 10 мм:
    # ID0: x=4..14, y=4..14
    # ID1: x=196..206, y=4..14
    # ID2: x=196..206, y=283..293
    # ID3: x=4..14, y=283..293
    #
    # Внутрішні кути:
    destination = np.float32([
        [
            mm_to_x(14),
            mm_to_y(14),
        ],
        [
            mm_to_x(196),
            mm_to_y(14),
        ],
        [
            mm_to_x(196),
            mm_to_y(283),
        ],
        [
            mm_to_x(14),
            mm_to_y(283),
        ],
    ])

    transform = (
        cv2.getPerspectiveTransform(
            source,
            destination,
        )
    )

    aligned = cv2.warpPerspective(
        image,
        transform,
        (
            PAGE_WIDTH,
            PAGE_HEIGHT,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    return aligned


def build_boundaries(
    start: int,
    end: int,
    count: int,
) -> list[int]:
    return [
        int(
            round(value)
        )
        for value in np.linspace(
            start,
            end,
            count + 1,
        )
    ]


def create_grid_preview(
    aligned: np.ndarray,
) -> int:
    preview = aligned.copy()

    cells_left = mm_to_x(
        GRID_LEFT_MM
        + LABEL_WIDTH_MM
    )

    cells_right = mm_to_x(
        GRID_RIGHT_MM
    )

    grid_top = mm_to_y(
        GRID_TOP_MM
    )

    grid_bottom = mm_to_y(
        GRID_BOTTOM_MM
    )

    x_boundaries = build_boundaries(
        cells_left,
        cells_right,
        COLUMNS,
    )

    y_boundaries = build_boundaries(
        grid_top,
        grid_bottom,
        ROWS,
    )

    CELLS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for digit in range(
        ROWS
    ):
        for column in range(
            COLUMNS
        ):
            x1 = x_boundaries[
                column
            ]

            x2 = x_boundaries[
                column + 1
            ]

            y1 = y_boundaries[
                digit
            ]

            y2 = y_boundaries[
                digit + 1
            ]

            # Повна клітинка.
            cv2.rectangle(
                preview,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    0,
                    180,
                    0,
                ),
                2,
            )

            cv2.putText(
                preview,
                f"{digit}:{column + 1}",
                (
                    x1 + 3,
                    y1 + 17,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (
                    0,
                    0,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            # Для тесту вирізаємо внутрішню
            # область без ліній сітки.
            cell_width = x2 - x1
            cell_height = y2 - y1

            margin_x = max(
                12,
                int(
                    round(
                        cell_width * 0.10
                    )
                ),
            )

            margin_y = max(
                14,
                int(
                    round(
                        cell_height * 0.05
                    )
                ),
            )

            inner_x1 = x1 + margin_x
            inner_x2 = x2 - margin_x

            inner_y1 = y1 + margin_y
            inner_y2 = y2 - margin_y

            cell = aligned[
                inner_y1:inner_y2,
                inner_x1:inner_x2,
            ]

            hsv = cv2.cvtColor(
                cell,
                cv2.COLOR_BGR2HSV,
            )

            blue, green, red = cv2.split(
                cell
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

            blue_mask = (
                (hue >= 90)
                & (hue <= 178)
                & (saturation >= 35)
                & (value >= 35)
                & (value <= 250)
                & ((blue_i - green_i) >= 8)
                & ((blue_i - red_i) >= -35)
            )

            mask = np.where(
                blue_mask,
                255,
                0,
            ).astype(
                np.uint8
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (3, 3),
                ),
                iterations=1,
            )

            component_count, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8,
                )
            )

            cleaned_mask = np.zeros_like(
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

                touches_top = (
                    y <= 2
                )

                touches_bottom = (
                    y + height
                    >= image_height - 2
                )

                touches_left = (
                    x <= 2
                )

                touches_right = (
                    x + width
                    >= image_width - 2
                )

                long_horizontal = (
                    width >= image_width * 0.35
                    and height <= 8
                )

                border_noise = (
                    (
                        touches_top
                        or touches_bottom
                        or touches_left
                        or touches_right
                    )
                    and area < 180
                )

                if (
                    area >= 12
                    and not long_horizontal
                    and not border_noise
                ):
                    cleaned_mask[
                        labels == component_index
                    ] = 255

            cell = cv2.bitwise_not(
                cleaned_mask
            )

            file_name = (
                f"digit_{digit}_"
                f"{column + 1:02d}.png"
            )

            cv2.imwrite(
                str(
                    CELLS_DIR
                    / file_name
                ),
                cell,
            )

            count += 1

    cv2.imwrite(
        str(PREVIEW_PATH),
        preview,
    )

    return count


def main() -> None:
    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 58)
    print(
        "FERRETTI OCR — TRAINING SHEET IMPORT TEST"
    )
    print("=" * 58)

    print()
    print(
        "1. Читаємо PDF..."
    )

    image = render_pdf()

    cv2.imwrite(
        str(PAGE_IMAGE_PATH),
        image,
    )

    print(
        f"   {image.shape[1]} x "
        f"{image.shape[0]} px"
    )

    print(
        "2. Шукаємо ArUco..."
    )

    detected = detect_markers(
        image
    )

    print(
        "   Знайдено ID:",
        sorted(
            detected.keys()
        ),
    )

    print(
        "3. Вирівнюємо A4..."
    )

    aligned = align_page(
        image,
        detected,
    )

    cv2.imwrite(
        str(ALIGNED_PATH),
        aligned,
    )

    print(
        "4. Будуємо сітку 10 x 20..."
    )

    count = create_grid_preview(
        aligned
    )

    print()
    print(
        f"Знайдено клітинок: {count}"
    )

    print()
    print(
        "Preview:"
    )

    print(
        PREVIEW_PATH
    )

    print()
    print(
        "Тестові клітинки:"
    )

    print(
        CELLS_DIR
    )

    print()
    print(
        "ВАЖЛИВО:"
    )

    print(
        "Цей режим ще НЕ додає "
        "нічого у dataset."
    )

    print()
    print("=" * 58)


if __name__ == "__main__":
    main()
