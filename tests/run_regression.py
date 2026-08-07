from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


from digit_splitter import split_digits

from ocr_engine_v2 import build_blue_mask

from app.ocr.svm_digit_classifier import (
    classify_digit,
)


REGRESSION_DIR = (
    PROJECT_DIR
    / "tests"
    / "regression"
)

IMAGES_DIR = (
    REGRESSION_DIR
    / "images"
)

EXPECTED_PATH = (
    REGRESSION_DIR
    / "expected.csv"
)

FAILURES_DIR = (
    REGRESSION_DIR
    / "failures"
)


def load_expected() -> list[tuple[str, str]]:
    if not EXPECTED_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено:\n{EXPECTED_PATH}"
        )

    rows: list[tuple[str, str]] = []

    with EXPECTED_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        sample = csv_file.read(
            2048
        )

        csv_file.seek(0)

        delimiter = (
            ";"
            if sample.count(";")
            > sample.count(",")
            else ","
        )

        reader = csv.DictReader(
            csv_file,
            delimiter=delimiter,
        )

        if (
            reader.fieldnames is None
            or "filename"
            not in reader.fieldnames
            or "expected"
            not in reader.fieldnames
        ):
            raise RuntimeError(
                "expected.csv повинен мати "
                "колонки filename та expected."
            )

        for row in reader:
            filename = (
                row.get(
                    "filename",
                    ""
                )
                .strip()
            )

            expected = (
                row.get(
                    "expected",
                    ""
                )
                .strip()
            )

            if filename:
                rows.append(
                    (
                        filename,
                        expected,
                    )
                )

    return rows


def fallback_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Резервний режим для вже очищених
    чорно-білих regression-зображень.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    _threshold, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    return mask


def create_digit_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Для виробничої кольорової клітинки
    використовує нашу синю маску.

    Якщо синього чорнила немає,
    підтримує також очищені тестові PNG.
    """

    if len(image.shape) == 3:
        mask = build_blue_mask(
            image
        )

        ink_pixels = int(
            np.count_nonzero(
                mask
            )
        )

        if ink_pixels >= 20:
            return mask

    return fallback_mask(
        image
    )


def recognize_image(
    image: np.ndarray,
) -> tuple[
    str,
    np.ndarray,
    list,
]:
    mask = create_digit_mask(
        image
    )

    segments = split_digits(
        mask
    )

    if not segments:
        return (
            "",
            mask,
            [],
        )

    digits: list[str] = []

    for segment in segments:
        prediction = classify_digit(
            segment.image
        )

        if prediction.digit not in (
            "0123456789"
        ):
            return (
                "",
                mask,
                segments,
            )

        digits.append(
            prediction.digit
        )

    value = "".join(
        digits
    )

    return (
        value,
        mask,
        segments,
    )


def save_failure_debug(
    filename: str,
    image: np.ndarray,
    mask: np.ndarray,
    segments: list,
) -> None:
    stem = Path(
        filename
    ).stem

    output_dir = (
        FAILURES_DIR
        / stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_original.png"
        ),
        image,
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_mask.png"
        ),
        mask,
    )

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        cv2.imwrite(
            str(
                output_dir
                / (
                    f"03_digit_"
                    f"{index}.png"
                )
            ),
            segment.image,
        )


def main() -> None:
    print()
    print("=" * 64)
    print(
        "FERRETTI OCR — REGRESSION TEST"
    )
    print("=" * 64)
    print()

    expected_rows = load_expected()

    if not expected_rows:
        print(
            "У expected.csv немає тестів."
        )
        return

    total = 0
    passed = 0
    failed = 0

    for (
        filename,
        expected,
    ) in expected_rows:
        image_path = (
            IMAGES_DIR
            / filename
        )

        total += 1

        if not image_path.exists():
            failed += 1

            print(
                f"[FAIL] "
                f"{filename:<28} "
                f"файл не знайдено"
            )

            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            failed += 1

            print(
                f"[FAIL] "
                f"{filename:<28} "
                f"не вдалося відкрити"
            )

            continue

        try:
            (
                actual,
                mask,
                segments,
            ) = recognize_image(
                image
            )
        except Exception as error:
            failed += 1

            print(
                f"[ERROR] "
                f"{filename:<27} "
                f"{error}"
            )

            continue

        is_correct = (
            actual == expected
        )

        if is_correct:
            passed += 1

            print(
                f"[PASS] "
                f"{filename:<28} "
                f"{actual}"
            )
        else:
            failed += 1

            print(
                f"[FAIL] "
                f"{filename:<28} "
                f"expected={expected:<4} "
                f"actual={actual or '-':<4} "
                f"segments={len(segments)}"
            )

            save_failure_debug(
                filename,
                image,
                mask,
                segments,
            )

    accuracy = (
        passed
        / total
        * 100.0
        if total
        else 0.0
    )

    print()
    print("-" * 64)

    print(
        f"Всього тестів:  {total}"
    )

    print(
        f"PASS:           {passed}"
    )

    print(
        f"FAIL:           {failed}"
    )

    print(
        f"Точність:       {accuracy:.2f}%"
    )

    print("-" * 64)

    if failed == 0:
        print()
        print(
            "REGRESSION: PASS"
        )
    else:
        print()
        print(
            "REGRESSION: FAIL"
        )

        print(
            "Debug помилок:"
        )

        print(
            FAILURES_DIR
        )

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
