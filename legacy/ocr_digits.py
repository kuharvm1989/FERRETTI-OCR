from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from cell_parser import CellAnalysisResult, CellResult
from digit_segmenter import segment_characters


TESSERACT_DEFAULT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


@dataclass(frozen=True)
class DigitOcrResult:
    row: int
    column: int
    day: str
    value: str
    confidence: float
    status: str
    source_path: Path | None
    prepared_path: Path | None


@dataclass(frozen=True)
class DigitOcrBatchResult:
    items: list[DigitOcrResult]
    recognized_count: int
    review_count: int
    empty_count: int


def configure_tesseract() -> None:
    if TESSERACT_DEFAULT_PATH.exists():
        pytesseract.pytesseract.tesseract_cmd = str(
            TESSERACT_DEFAULT_PATH
        )

    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        raise RuntimeError(
            "Tesseract OCR не знайдено. Перевірте, що він "
            "встановлений у C:\\Program Files\\Tesseract-OCR."
        ) from error


def crop_cell_from_page(
    normalized_bgr: np.ndarray,
    cell: CellResult,
) -> np.ndarray:
    """
    Використовує готову OCR-зону з CellGeometry.
    Жодних додаткових відсоткових обрізань тут немає.
    """
    image_height, image_width = (
        normalized_bgr.shape[:2]
    )

    x1 = max(
        0,
        min(
            cell.ocr_x1,
            image_width - 1,
        ),
    )
    y1 = max(
        0,
        min(
            cell.ocr_y1,
            image_height - 1,
        ),
    )
    x2 = max(
        x1 + 1,
        min(
            cell.ocr_x2,
            image_width,
        ),
    )
    y2 = max(
        y1 + 1,
        min(
            cell.ocr_y2,
            image_height,
        ),
    )

    return normalized_bgr[
        y1:y2,
        x1:x2,
    ].copy()


def build_blue_ink_mask(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Виділяє сині та синьо-фіолетові штрихи ручки.
    Чорні/сірі лінії таблиці мають низьку насиченість
    і переважно відсікаються.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        raise ValueError("Отримано порожню клітинку.")

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(cell_bgr)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Основний діапазон для синьої/фіолетової ручки.
    hsv_mask = (
        (hue >= 88)
        & (hue <= 178)
        & (saturation >= 28)
        & (value <= 250)
    )

    # Додатковий колірний критерій для блідих синіх штрихів.
    blue_advantage = (
        blue.astype(np.int16)
        - green.astype(np.int16)
    )

    red_advantage = (
        red.astype(np.int16)
        - green.astype(np.int16)
    )

    color_mask = (
        (saturation >= 20)
        & (value <= 252)
        & (
            (blue_advantage >= 7)
            | (
                (blue_advantage >= 2)
                & (red_advantage >= 5)
            )
        )
    )

    mask = np.where(
        hsv_mask | color_mask,
        255,
        0,
    ).astype(np.uint8)

    # Прибираємо дрібний сканерний шум і з'єднуємо штрихи.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2),
        ),
        iterations=1,
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

    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(mask)

    for component_index in range(
        1,
        component_count,
    ):
        area = int(
            stats[
                component_index,
                cv2.CC_STAT_AREA,
            ]
        )

        component_width = int(
            stats[
                component_index,
                cv2.CC_STAT_WIDTH,
            ]
        )

        component_height = int(
            stats[
                component_index,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        if (
            area >= 8
            and component_width >= 2
            and component_height >= 2
        ):
            cleaned[
                labels == component_index
            ] = 255

    return cleaned


def remove_horizontal_artifacts(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Видаляє тонкі довгі горизонтальні штрихи,
    які зазвичай є залишками ліній таблиці або сканерним шумом.
    Короткі рукописні горизонтальні частини цифр зберігаються.
    """
    if mask is None or mask.size == 0:
        return mask

    result = mask.copy()

    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            result,
            connectivity=8,
        )
    )

    image_width = result.shape[1]

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

        is_long_and_thin = (
            width >= max(
                22,
                int(
                    image_width * 0.18
                ),
            )
            and height <= 5
            and width >= height * 6
        )

        is_small_edge_dash = (
            area <= 45
            and height <= 5
            and (
                x <= 5
                or x + width
                >= image_width - 5
            )
        )

        if (
            is_long_and_thin
            or is_small_edge_dash
        ):
            result[
                labels == component_index
            ] = 0

    return result


def keep_relevant_components(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Залишає компоненти, схожі на частини рукописних цифр.
    Дрібний шум і дуже низькі плями відкидаються.
    """
    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(mask)

    if component_count <= 1:
        return cleaned

    component_areas = [
        int(
            stats[
                index,
                cv2.CC_STAT_AREA,
            ]
        )
        for index in range(
            1,
            component_count,
        )
    ]

    largest_area = max(
        component_areas,
        default=0,
    )

    for component_index in range(
        1,
        component_count,
    ):
        area = int(
            stats[
                component_index,
                cv2.CC_STAT_AREA,
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

        relative_area = (
            area / largest_area
            if largest_area > 0
            else 0.0
        )

        if (
            area >= 8
            and width >= 2
            and height >= 4
            and (
                relative_area >= 0.05
                or area >= 18
            )
        ):
            cleaned[
                labels == component_index
            ] = 255

    return cleaned


def crop_to_ink(
    mask: np.ndarray,
    padding: int = 10,
) -> np.ndarray | None:
    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return None

    x, y, width, height = (
        cv2.boundingRect(
            points
        )
    )

    if width < 3 or height < 4:
        return None

    x1 = max(
        0,
        x - padding,
    )
    y1 = max(
        0,
        y - padding,
    )
    x2 = min(
        mask.shape[1],
        x + width + padding,
    )
    y2 = min(
        mask.shape[0],
        y + height + padding,
    )

    return mask[
        y1:y2,
        x1:x2,
    ]


def resize_preserving_aspect(
    binary_mask: np.ndarray,
    target_height: int = 170,
) -> np.ndarray:
    """
    Збільшує зображення однаково по обох осях,
    не змінюючи пропорції почерку.
    """
    height, _width = (
        binary_mask.shape[:2]
    )

    scale = max(
        2.0,
        target_height
        / max(
            1,
            height,
        ),
    )

    return cv2.resize(
        binary_mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )


def prepare_digit_image(
    cell_bgr: np.ndarray,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
]:
    """
    Повертає:
    1) чорні цифри на білому фоні для OCR;
    2) очищену білу маску рукопису на чорному фоні
       для сегментації окремих символів.
    """
    mask = build_blue_ink_mask(
        cell_bgr
    )

    mask = remove_horizontal_artifacts(
        mask
    )

    mask = keep_relevant_components(
        mask
    )

    cropped = crop_to_ink(
        mask,
        padding=8,
    )

    if cropped is None:
        return None, None

    enlarged_mask = (
        resize_preserving_aspect(
            cropped,
            target_height=170,
        )
    )

    # Не з'єднуємо цифри морфологічною дилатацією.
    # Лише дуже легке закриття розривів усередині одного штриха.
    enlarged_mask = cv2.morphologyEx(
        enlarged_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2),
        ),
        iterations=1,
    )

    prepared = cv2.bitwise_not(
        enlarged_mask
    )

    prepared = cv2.copyMakeBorder(
        prepared,
        28,
        28,
        28,
        28,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    segmentation_mask = (
        cv2.copyMakeBorder(
            enlarged_mask,
            28,
            28,
            28,
            28,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    )

    return (
        prepared,
        segmentation_mask,
    )


def normalize_digit_text(
    text: str,
) -> str:
    value = str(
        text or ""
    ).strip()

    substitutions = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }

    for source, target in (
        substitutions.items()
    ):
        value = value.replace(
            source,
            target,
        )

    value = re.sub(
        r"[^0-9]",
        "",
        value,
    )

    return value


def tesseract_single_pass(
    prepared: np.ndarray,
    page_segmentation_mode: int,
) -> tuple[str, float]:
    configuration = (
        f"--oem 3 --psm "
        f"{page_segmentation_mode} "
        "-c tessedit_char_whitelist="
        "0123456789"
    )

    data = pytesseract.image_to_data(
        prepared,
        config=configuration,
        output_type=(
            pytesseract.Output.DICT
        ),
    )

    best_value = ""
    best_confidence = -1.0

    for text, confidence_text in zip(
        data.get(
            "text",
            [],
        ),
        data.get(
            "conf",
            [],
        ),
    ):
        value = normalize_digit_text(
            text
        )

        try:
            confidence = float(
                confidence_text
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = -1.0

        if (
            value
            and confidence
            > best_confidence
        ):
            best_value = value
            best_confidence = confidence

    return (
        best_value,
        best_confidence,
    )


def run_segmented_tesseract(
    segmentation_mask: np.ndarray,
) -> tuple[str, float]:
    segments = segment_characters(
        segmentation_mask
    )

    if not segments or len(segments) > 2:
        return "", -1.0

    values: list[str] = []
    confidences: list[float] = []

    for segment in segments:
        value, confidence = (
            tesseract_single_pass(
                segment.image,
                page_segmentation_mode=10,
            )
        )

        if len(value) != 1:
            return "", -1.0

        values.append(
            value
        )

        confidences.append(
            confidence
        )

    return (
        "".join(
            values
        ),
        min(
            confidences,
            default=-1.0,
        ),
    )



def run_tesseract(
    prepared: np.ndarray,
    segmentation_mask: np.ndarray,
) -> tuple[str, float]:
    """
    Порівнює два незалежні методи:
    1) OCR цілого числа;
    2) сегментація та OCR кожної цифри окремо.
    """
    candidates: list[
        tuple[str, float, str]
    ] = []

    for page_segmentation_mode in (
        7,
        8,
        13,
    ):
        value, confidence = (
            tesseract_single_pass(
                prepared,
                page_segmentation_mode,
            )
        )

        if value:
            candidates.append(
                (
                    value,
                    confidence,
                    "whole",
                )
            )

    segmented_value, segmented_confidence = (
        run_segmented_tesseract(
            segmentation_mask
        )
    )

    if segmented_value:
        candidates.append(
            (
                segmented_value,
                segmented_confidence,
                "segmented",
            )
        )

    if not candidates:
        return "", -1.0

    # Якщо два методи погодилися — додаємо бонус довіри.
    grouped: dict[
        str,
        list[tuple[float, str]],
    ] = {}

    for value, confidence, source in candidates:
        grouped.setdefault(
            value,
            [],
        ).append(
            (
                confidence,
                source,
            )
        )

    scored_candidates: list[
        tuple[float, str, float]
    ] = []

    for value, observations in (
        grouped.items()
    ):
        best_confidence = max(
            confidence
            for confidence, _source
            in observations
        )

        source_count = len(
            {
                source
                for _confidence, source
                in observations
            }
        )

        agreement_bonus = (
            18.0
            if source_count >= 2
            else 0.0
        )

        # Невелика перевага сегментованого результату,
        # коли він складається з 1–2 цифр.
        segmented_bonus = (
            5.0
            if any(
                source == "segmented"
                for _confidence, source
                in observations
            )
            else 0.0
        )

        score = (
            best_confidence
            + agreement_bonus
            + segmented_bonus
        )

        scored_candidates.append(
            (
                score,
                value,
                best_confidence,
            )
        )

    scored_candidates.sort(
        reverse=True
    )

    _score, best_value, best_confidence = (
        scored_candidates[0]
    )

    return (
        best_value,
        best_confidence,
    )



def recognize_single_cell(
    normalized_bgr: np.ndarray,
    cell: CellResult,
    source_dir: Path,
    prepared_dir: Path,
) -> DigitOcrResult:
    cell_bgr = crop_cell_from_page(
        normalized_bgr,
        cell,
    )

    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_name = (
        f"r{cell.row:02d}_"
        f"c{cell.column}_"
        f"{cell.day}.png"
    )

    source_path = (
        source_dir / file_name
    )

    cv2.imwrite(
        str(source_path),
        cell_bgr,
    )

    prepared, segmentation_mask = (
        prepare_digit_image(
            cell_bgr
        )
    )

    if (
        prepared is None
        or segmentation_mask is None
    ):
        return DigitOcrResult(
            row=cell.row,
            column=cell.column,
            day=cell.day,
            value="",
            confidence=0.0,
            status="ПЕРЕВІРИТИ",
            source_path=source_path,
            prepared_path=None,
        )

    prepared_path = (
        prepared_dir / file_name
    )

    cv2.imwrite(
        str(prepared_path),
        prepared,
    )

    value, confidence_percent = (
        run_tesseract(
            prepared,
            segmentation_mask,
        )
    )

    confidence = (
        max(
            0.0,
            min(
                100.0,
                confidence_percent,
            ),
        )
        / 100.0
        if confidence_percent >= 0
        else 0.0
    )

    status = (
        "OK"
        if (
            value
            and confidence >= 0.75
        )
        else "ПЕРЕВІРИТИ"
    )

    return DigitOcrResult(
        row=cell.row,
        column=cell.column,
        day=cell.day,
        value=value,
        confidence=confidence,
        status=status,
        source_path=source_path,
        prepared_path=prepared_path,
    )


def recognize_filled_cells(
    normalized_bgr: np.ndarray,
    analysis: CellAnalysisResult,
    debug_dir: Path,
) -> DigitOcrBatchResult:
    configure_tesseract()

    source_dir = (
        debug_dir / "source_cells"
    )

    prepared_dir = (
        debug_dir / "prepared_digits"
    )

    items: list[DigitOcrResult] = []

    for cell in analysis.cells:
        if not cell.is_filled:
            continue

        items.append(
            recognize_single_cell(
                normalized_bgr,
                cell,
                source_dir,
                prepared_dir,
            )
        )

    recognized_count = sum(
        1
        for item in items
        if item.value
    )

    review_count = sum(
        1
        for item in items
        if item.status
        == "ПЕРЕВІРИТИ"
    )

    empty_count = sum(
        1
        for item in items
        if not item.value
    )

    return DigitOcrBatchResult(
        items=items,
        recognized_count=recognized_count,
        review_count=review_count,
        empty_count=empty_count,
    )


def export_ocr_results_csv(
    batch: DigitOcrBatchResult,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(
            csv_file,
            delimiter=";",
        )

        writer.writerow([
            "Рядок",
            "Колонка",
            "День",
            "Кількість",
            "Впевненість OCR",
            "Статус",
            "Оригінальна клітинка",
            "Підготовлене зображення",
        ])

        for item in batch.items:
            writer.writerow([
                item.row,
                item.column,
                item.day,
                item.value,
                f"{item.confidence:.4f}",
                item.status,
                str(
                    item.source_path
                    or ""
                ),
                str(
                    item.prepared_path
                    or ""
                ),
            ])
