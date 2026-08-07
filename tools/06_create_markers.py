from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "markers"

MARKER_IDS = (0, 1, 2, 3)
MARKER_SIZE_PX = 400
WHITE_BORDER_PX = 70


def create_marker(marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    marker = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        MARKER_SIZE_PX,
    )

    return cv2.copyMakeBorder(
        marker,
        WHITE_BORDER_PX,
        WHITE_BORDER_PX,
        WHITE_BORDER_PX,
        WHITE_BORDER_PX,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for marker_id in MARKER_IDS:
        marker = create_marker(marker_id)

        output_path = (
            OUTPUT_DIR /
            f"aruco_{marker_id}.png"
        )

        if not cv2.imwrite(
            str(output_path),
            marker,
        ):
            raise RuntimeError(
                f"Не вдалося створити {output_path}"
            )

        print(f"Створено: {output_path}")

    print("\nПризначення:")
    print("0 — верхній лівий")
    print("1 — верхній правий")
    print("2 — нижній правий")
    print("3 — нижній лівий")


if __name__ == "__main__":
    main()
