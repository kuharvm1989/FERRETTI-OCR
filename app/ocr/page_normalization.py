from __future__ import annotations

import json

import cv2
import numpy as np

from app.config.project_paths import (
    FORM_V2_CONFIG_PATH,
)


ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50

EXPECTED_MARKERS = {0, 1, 2, 3}

NORMALIZED_HEIGHT = 3508
MIN_NORMALIZED_WIDTH = 600
MAX_NORMALIZED_WIDTH = 3000


def detect_aruco_markers(
    image_bgr: np.ndarray,
) -> dict[int, dict[str, object]]:
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            ARUCO_DICTIONARY_ID
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

    if ids is None or len(ids) == 0:
        raise RuntimeError(
            "На сторінці не знайдено "
            "ArUco-міток."
        )

    result: dict[
        int,
        dict[str, object],
    ] = {}

    for (
        marker_id_array,
        marker_corners,
    ) in zip(
        ids,
        corners,
    ):
        marker_id = int(
            np.asarray(
                marker_id_array
            ).reshape(-1)[0]
        )

        points = np.asarray(
            marker_corners
        ).reshape(
            4,
            2,
        )

        center_array = points.mean(
            axis=0
        )

        center = (
            int(
                round(
                    center_array[0]
                )
            ),
            int(
                round(
                    center_array[1]
                )
            ),
        )

        result[marker_id] = {
            "corners": points,
            "center": center,
        }

    return result


def euclidean_distance(
    first_point: np.ndarray,
    second_point: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(
                first_point,
                dtype=np.float32,
            )
            - np.asarray(
                second_point,
                dtype=np.float32,
            )
        )
    )


def get_inner_marker_corner(
    detected: dict[
        int,
        dict[str, object],
    ],
    marker_id: int,
) -> np.ndarray:
    points = np.asarray(
        detected[
            marker_id
        ]["corners"],
        dtype=np.float32,
    ).reshape(
        4,
        2,
    )

    inner_corner_index = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }[marker_id]

    return points[
        inner_corner_index
    ]


def load_saved_marker_aspect_ratio(
) -> float | None:
    if not FORM_V2_CONFIG_PATH.exists():
        return None

    try:
        config = json.loads(
            FORM_V2_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        ratio = float(
            config.get(
                "marker_aspect_ratio",
                0,
            )
        )

        if 0.15 <= ratio <= 2.0:
            return ratio

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None

    return None


def normalize_page_by_markers(
    image_bgr: np.ndarray,
    detected: dict[
        int,
        dict[str, object],
    ],
) -> np.ndarray:
    for marker_id in EXPECTED_MARKERS:
        if marker_id not in detected:
            raise ValueError(
                (
                    "Немає мітки "
                    f"ID {marker_id}."
                )
            )

    top_left = (
        get_inner_marker_corner(
            detected,
            0,
        )
    )

    top_right = (
        get_inner_marker_corner(
            detected,
            1,
        )
    )

    bottom_right = (
        get_inner_marker_corner(
            detected,
            2,
        )
    )

    bottom_left = (
        get_inner_marker_corner(
            detected,
            3,
        )
    )

    top_width = euclidean_distance(
        top_left,
        top_right,
    )

    bottom_width = (
        euclidean_distance(
            bottom_left,
            bottom_right,
        )
    )

    left_height = (
        euclidean_distance(
            top_left,
            bottom_left,
        )
    )

    right_height = (
        euclidean_distance(
            top_right,
            bottom_right,
        )
    )

    measured_width = (
        top_width
        + bottom_width
    ) / 2.0

    measured_height = (
        left_height
        + right_height
    ) / 2.0

    if (
        measured_width < 100
        or measured_height < 100
    ):
        raise RuntimeError(
            (
                "Відстань між "
                "ArUco-мітками "
                "занадто мала."
            )
        )

    measured_ratio = (
        measured_width
        / measured_height
    )

    saved_ratio = (
        load_saved_marker_aspect_ratio()
    )

    aspect_ratio = (
        saved_ratio
        if saved_ratio is not None
        else measured_ratio
    )

    output_height = (
        NORMALIZED_HEIGHT
    )

    output_width = int(
        round(
            output_height
            * aspect_ratio
        )
    )

    output_width = max(
        MIN_NORMALIZED_WIDTH,
        min(
            MAX_NORMALIZED_WIDTH,
            output_width,
        ),
    )

    source_points = np.float32(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ]
    )

    destination_points = np.float32(
        [
            [0, 0],
            [
                output_width - 1,
                0,
            ],
            [
                output_width - 1,
                output_height - 1,
            ],
            [
                0,
                output_height - 1,
            ],
        ]
    )

    transform = (
        cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )
    )

    return cv2.warpPerspective(
        image_bgr,
        transform,
        (
            output_width,
            output_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=(
            cv2.BORDER_CONSTANT
        ),
        borderValue=(
            255,
            255,
            255,
        ),
    )
