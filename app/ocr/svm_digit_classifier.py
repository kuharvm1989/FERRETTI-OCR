from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from app.config.project_paths import (
    DIGIT_SVM_MODEL_PATH,
)

import cv2
import numpy as np



IMAGE_SIZE = 64


@dataclass(frozen=True)
class SvmDigitPrediction:
    digit: str
    confidence: float
    method: str


_model = None
_hog = None


def create_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )


def prepare_image(
    image: np.ndarray,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError(
            "Порожнє зображення цифри."
        )

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    # Очікуємо чорну цифру на білому фоні.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    ink = cv2.bitwise_not(
        binary
    )

    points = cv2.findNonZero(
        ink
    )

    if points is None:
        return np.zeros(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            dtype=np.uint8,
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    crop = ink[
        y:y + height,
        x:x + width,
    ]

    padding = max(
        4,
        int(
            round(
                max(
                    width,
                    height,
                )
                * 0.15
            )
        ),
    )

    side = max(
        width + 2 * padding,
        height + 2 * padding,
    )

    canvas = np.zeros(
        (
            side,
            side,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        side - width
    ) // 2

    offset_y = (
        side - height
    ) // 2

    canvas[
        offset_y:
        offset_y + height,
        offset_x:
        offset_x + width,
    ] = crop

    return cv2.resize(
        canvas,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )


def extract_features(
    image: np.ndarray,
) -> np.ndarray:
    global _hog

    if _hog is None:
        _hog = create_hog()

    prepared = prepare_image(
        image
    )

    features = _hog.compute(
        prepared
    )

    return (
        features
        .reshape(1, -1)
        .astype(np.float32)
    )


def load_model():
    global _model

    if _model is not None:
        return _model

    if not DIGIT_SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено FERRETTI SVM:\n"
            f"{DIGIT_SVM_MODEL_PATH}"
        )

    _model = cv2.ml.SVM_load(
        str(DIGIT_SVM_MODEL_PATH)
    )

    return _model


def classify_digit(
    image: np.ndarray,
) -> SvmDigitPrediction:
    model = load_model()

    features = extract_features(
        image
    )

    _result, prediction = model.predict(
        features
    )

    digit = str(
        int(
            prediction[
                0,
                0,
            ]
        )
    )

    # OpenCV C-SVC не повертає калібровану
    # ймовірність для multiclass.
    # До окремого етапу калібрування не будемо
    # вигадувати фальшиву confidence.
    return SvmDigitPrediction(
        digit=digit,
        confidence=0.0,
        method="ferretti_svm_v1",
    )
