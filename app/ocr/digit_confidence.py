from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ocr.svm_digit_classifier import (
    classify_digit,
    extract_features,
)


BASE_DIR = Path(__file__).resolve().parent

CONFIRMED_DIR = (
    BASE_DIR
    / "dataset"
    / "confirmed"
)

K_NEIGHBORS = 7


@dataclass(frozen=True)
class ConfidentDigitPrediction:
    digit: str
    confidence: float
    svm_digit: str
    neighbor_digit: str
    agreement: bool


_reference_features: np.ndarray | None = None
_reference_labels: np.ndarray | None = None


def load_reference_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    global _reference_features
    global _reference_labels

    if (
        _reference_features is not None
        and _reference_labels is not None
    ):
        return (
            _reference_features,
            _reference_labels,
        )

    features = []
    labels = []

    for digit in "0123456789":
        folder = (
            CONFIRMED_DIR
            / digit
        )

        if not folder.exists():
            continue

        for image_path in folder.glob(
            "*.png"
        ):
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                continue

            feature = extract_features(
                image
            ).reshape(-1)

            features.append(
                feature
            )

            labels.append(
                int(digit)
            )

    if not features:
        raise RuntimeError(
            "Dataset confirmed порожній."
        )

    _reference_features = np.asarray(
        features,
        dtype=np.float32,
    )

    _reference_labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    return (
        _reference_features,
        _reference_labels,
    )


def neighbor_confidence(
    image: np.ndarray,
    predicted_digit: str,
) -> tuple[
    float,
    str,
]:
    features, labels = (
        load_reference_dataset()
    )

    query = extract_features(
        image
    ).reshape(-1)

    distances = np.linalg.norm(
        features - query,
        axis=1,
    )

    predicted_label = int(
        predicted_digit
    )

    # ----------------------------------
    # 1. KNN-голосування
    # ----------------------------------

    k = min(
        K_NEIGHBORS,
        len(distances),
    )

    nearest_indices = np.argsort(
        distances
    )[:k]

    class_weights: dict[
        int,
        float,
    ] = {}

    for index in nearest_indices:
        label = int(
            labels[index]
        )

        distance = float(
            distances[index]
        )

        weight = (
            1.0
            / (
                distance
                + 1e-6
            )
        )

        class_weights[label] = (
            class_weights.get(
                label,
                0.0,
            )
            + weight
        )

    if not class_weights:
        return (
            0.0,
            "",
        )

    neighbor_digit_int = max(
        class_weights,
        key=class_weights.get,
    )

    total_weight = sum(
        class_weights.values()
    )

    predicted_weight = (
        class_weights.get(
            predicted_label,
            0.0,
        )
    )

    vote_score = (
        predicted_weight
        / total_weight
        if total_weight > 0
        else 0.0
    )

    # ----------------------------------
    # 2. Відстань до свого класу
    #    та найближчого чужого
    # ----------------------------------

    same_mask = (
        labels
        == predicted_label
    )

    other_mask = (
        labels
        != predicted_label
    )

    same_distances = distances[
        same_mask
    ]

    other_distances = distances[
        other_mask
    ]

    if (
        len(same_distances) == 0
        or len(other_distances) == 0
    ):
        return (
            0.0,
            str(
                neighbor_digit_int
            ),
        )

    # Беремо не один випадковий nearest,
    # а середнє трьох найближчих.
    same_k = min(
        3,
        len(same_distances),
    )

    other_k = min(
        3,
        len(other_distances),
    )

    same_distance = float(
        np.mean(
            np.partition(
                same_distances,
                same_k - 1,
            )[:same_k]
        )
    )

    other_distance = float(
        np.mean(
            np.partition(
                other_distances,
                other_k - 1,
            )[:other_k]
        )
    )

    # Якщо чужий клас значно далі,
    # separation наближається до 1.
    separation_score = (
        other_distance
        / (
            same_distance
            + other_distance
            + 1e-6
        )
    )

    # 0.5 = межа між класами.
    # Перетворюємо 0.5...1.0 у 0...1.
    separation_score = (
        separation_score
        - 0.5
    ) * 2.0

    separation_score = float(
        max(
            0.0,
            min(
                1.0,
                separation_score,
            ),
        )
    )

    # ----------------------------------
    # 3. Фінальний score
    # ----------------------------------

    confidence = (
        0.45 * vote_score
        + 0.55 * separation_score
    )

    # Якщо KNN і SVM не погоджуються,
    # це дуже сильний сигнал ризику.
    if (
        neighbor_digit_int
        != predicted_label
    ):
        confidence *= 0.35

    return (
        float(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )
        ),
        str(
            neighbor_digit_int
        ),
    )


def classify_digit_with_confidence(
    image: np.ndarray,
) -> ConfidentDigitPrediction:
    svm_prediction = classify_digit(
        image
    )

    digit = svm_prediction.digit

    if digit not in "0123456789":
        return ConfidentDigitPrediction(
            digit="",
            confidence=0.0,
            svm_digit="",
            neighbor_digit="",
            agreement=False,
        )

    (
        confidence,
        neighbor_digit,
    ) = neighbor_confidence(
        image,
        digit,
    )

    agreement = (
        digit
        == neighbor_digit
    )

    # Якщо SVM і найближчі реальні
    # приклади не погоджуються —
    # різко знижуємо довіру.


    return ConfidentDigitPrediction(
        digit=digit,
        confidence=confidence,
        svm_digit=digit,
        neighbor_digit=neighbor_digit,
        agreement=agreement,
    )
