from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

INPUT_IMAGE = (
    BASE_DIR
    / "debug"
    / "page_001_300dpi.png"
)

OUTPUT_IMAGE = (
    BASE_DIR
    / "debug"
    / "page_001_300dpi_deskewed.png"
)

PREVIEW_IMAGE = (
    BASE_DIR
    / "debug"
    / "deskew_preview.png"
)


def estimate_skew_angle(
    image: np.ndarray,
) -> float:
    """
    Визначає кут нахилу за довгими горизонтальними
    лініями таблиці.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    edges = cv2.Canny(
        blurred,
        50,
        150,
        apertureSize=3,
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800,
        threshold=180,
        minLineLength=500,
        maxLineGap=30,
    )

    if lines is None:
        raise RuntimeError(
            "Не вдалося знайти лінії для вирівнювання."
        )

    angles: list[float] = []

    for line in lines:
        coordinates = np.asarray(line).reshape(-1)

    if coordinates.size != 4:
        continue

    x1, y1, x2, y2 = map(
        int,
        coordinates,
    )

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = np.degrees(
            np.arctan2(dy, dx)
        )

        # Беремо тільки майже горизонтальні лінії.
        if -5.0 <= angle <= 5.0:
            angles.append(angle)

    if not angles:
        raise RuntimeError(
            "Не знайдено достатньо горизонтальних ліній."
        )

    return float(
        np.median(angles)
    )


def rotate_image(
    image: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Повертає зображення без обрізання країв.
    """
    height, width = image.shape[:2]

    center = (
        width / 2,
        height / 2,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos_value = abs(
        rotation_matrix[0, 0]
    )

    sin_value = abs(
        rotation_matrix[0, 1]
    )

    new_width = int(
        height * sin_value +
        width * cos_value
    )

    new_height = int(
        height * cos_value +
        width * sin_value
    )

    rotation_matrix[0, 2] += (
        new_width / 2 - center[0]
    )

    rotation_matrix[1, 2] += (
        new_height / 2 - center[1]
    )

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def main() -> None:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Не знайдено: {INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:
        raise RuntimeError(
            "OpenCV не зміг відкрити зображення."
        )

    angle = estimate_skew_angle(
        image
    )

    print(
        f"Виявлений нахил: {angle:.4f}°"
    )

    # Для випрямлення повертаємо у протилежний бік.
    corrected = rotate_image(
        image,
        angle,
    )

    if not cv2.imwrite(
        str(OUTPUT_IMAGE),
        corrected,
    ):
        raise RuntimeError(
            "Не вдалося зберегти вирівняне зображення."
        )

    preview = corrected.copy()

    height, width = preview.shape[:2]

    cv2.line(
        preview,
        (0, height // 2),
        (width, height // 2),
        (0, 0, 255),
        4,
    )

    if not cv2.imwrite(
        str(PREVIEW_IMAGE),
        preview,
    ):
        raise RuntimeError(
            "Не вдалося зберегти preview."
        )

    print(
        f"Створено: {OUTPUT_IMAGE}"
    )

    print(
        f"Створено: {PREVIEW_IMAGE}"
    )

    print(
        f"Новий розмір: "
        f"{corrected.shape[1]} × "
        f"{corrected.shape[0]} px"
    )


if __name__ == "__main__":
    main()
