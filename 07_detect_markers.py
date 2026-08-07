from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "debug"
    / "page_001_300dpi.png"
)

PREVIEW_PATH = (
    BASE_DIR
    / "debug"
    / "aruco_detection_preview.png"
)

EXPECTED_IDS = {0, 1, 2, 3}


def detect_markers(
    image: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters,
    )

    corners, ids, _rejected = (
        detector.detectMarkers(image)
    )

    return corners, ids


def marker_center(
    marker_corners: np.ndarray,
) -> tuple[int, int]:
    points = marker_corners.reshape(4, 2)

    center = points.mean(axis=0)

    return (
        int(round(center[0])),
        int(round(center[1])),
    )


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено PNG: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(
            "OpenCV не зміг відкрити PNG."
        )

    corners, ids = detect_markers(image)

    preview = image.copy()

    if ids is None or len(ids) == 0:
        cv2.imwrite(
            str(PREVIEW_PATH),
            preview,
        )

        raise RuntimeError(
            "Не знайдено жодної ArUco-мітки."
        )

    ids_flat = ids.flatten().astype(int)

    cv2.aruco.drawDetectedMarkers(
        preview,
        corners,
        ids,
        borderColor=(0, 255, 0),
    )

    detected: dict[int, tuple[int, int]] = {}

    for marker_id, marker_corners in zip(
        ids_flat,
        corners,
    ):
        center_x, center_y = marker_center(
            marker_corners
        )

        detected[int(marker_id)] = (
            center_x,
            center_y,
        )

        cv2.circle(
            preview,
            (center_x, center_y),
            12,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            preview,
            f"ID {marker_id}",
            (center_x + 18, center_y - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    detected_ids = set(detected)
    missing_ids = EXPECTED_IDS - detected_ids
    unexpected_ids = detected_ids - EXPECTED_IDS

    if not cv2.imwrite(
        str(PREVIEW_PATH),
        preview,
    ):
        raise RuntimeError(
            "Не вдалося зберегти preview."
        )

    print(
        f"Розмір сторінки: "
        f"{image.shape[1]} × {image.shape[0]} px"
    )

    print(
        f"Знайдено міток: {len(ids_flat)}"
    )

    for marker_id in sorted(detected):
        center = detected[marker_id]

        print(
            f"ID {marker_id}: "
            f"центр x={center[0]}, y={center[1]}"
        )

    if missing_ids:
        print(
            "Відсутні мітки: "
            + ", ".join(
                map(str, sorted(missing_ids))
            )
        )

    if unexpected_ids:
        print(
            "Неочікувані ID: "
            + ", ".join(
                map(str, sorted(unexpected_ids))
            )
        )

    print(f"Preview: {PREVIEW_PATH}")

    if missing_ids:
        raise RuntimeError(
            "Розпізнано не всі мітки 0, 1, 2, 3."
        )

    print(
        "УСПІХ: усі чотири мітки "
        "розпізнано."
    )


if __name__ == "__main__":
    main()
