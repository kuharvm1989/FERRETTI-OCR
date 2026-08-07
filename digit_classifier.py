from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract


TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


@dataclass(frozen=True)
class DigitPrediction:
    digit: str
    confidence: float
    method: str


def configure_tesseract() -> None:
    if TESSERACT_PATH.exists():
        pytesseract.pytesseract.tesseract_cmd = str(
            TESSERACT_PATH
        )


def prepare_digit(
    image: np.ndarray,
    output_size: int = 128,
) -> np.ndarray:
    """
    Нормалізує одну цифру, зберігаючи її пропорції.
    Очікується чорна цифра на білому фоні.
    """
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

    # Переконуємося, що фон білий.
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(
            gray
        )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    ink_mask = cv2.bitwise_not(
        binary
    )

    points = cv2.findNonZero(
        ink_mask
    )

    if points is None:
        return np.full(
            (
                output_size,
                output_size,
            ),
            255,
            dtype=np.uint8,
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    crop = binary[
        y:y + height,
        x:x + width,
    ]

    padding = max(
        8,
        int(
            round(
                max(
                    width,
                    height,
                )
                * 0.18
            )
        ),
    )

    side = max(
        width + 2 * padding,
        height + 2 * padding,
    )

    canvas = np.full(
        (
            side,
            side,
        ),
        255,
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
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_AREA,
    )


def classify_with_tesseract(
    image: np.ndarray,
) -> DigitPrediction:
    configure_tesseract()

    prepared = prepare_digit(
        image
    )

    best_digit = ""
    best_confidence = 0.0

    for psm in (
        10,
        8,
        13,
    ):
        data = pytesseract.image_to_data(
            prepared,
            config=(
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist="
                "0123456789"
            ),
            output_type=(
                pytesseract.Output.DICT
            ),
        )

        texts = data.get(
            "text",
            [],
        )

        confidences = data.get(
            "conf",
            [],
        )

        for text, confidence_text in zip(
            texts,
            confidences,
        ):
            value = str(
                text
            ).strip()

            if len(value) != 1:
                continue

            if value not in "0123456789":
                continue

            try:
                confidence = float(
                    confidence_text
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if confidence > best_confidence:
                best_digit = value
                best_confidence = confidence

    return DigitPrediction(
        digit=best_digit,
        confidence=max(
            0.0,
            min(
                1.0,
                best_confidence / 100.0,
            ),
        ),
        method="tesseract_single_digit",
    )


def classify_digit(
    image: np.ndarray,
) -> DigitPrediction:
    """
    Єдина точка входу для класифікації цифри.

    Пізніше саме тут підключимо FERRETTI CNN
    замість Tesseract.
    """
    return classify_with_tesseract(
        image
    )
