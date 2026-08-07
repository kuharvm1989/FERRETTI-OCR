from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from cell_parser import CellAnalysisResult, CellResult


TESSERACT_DEFAULT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


@dataclass(frozen=True)
class OcrCandidate:
    value: str
    confidence: float
    method: str


@dataclass(frozen=True)
class DigitOcrResult:
    row: int
    column: int
    day: str
    value: str
    confidence: float
    status: str
    method: str
    source_path: Path | None
    prepared_path: Path | None
    segment_paths: tuple[Path, ...]


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
            "Tesseract OCR не знайдено. Перевірте встановлення у "
            r"C:\Program Files\Tesseract-OCR"
        ) from error


def crop_ocr_zone(
    normalized_bgr: np.ndarray,
    cell: CellResult,
) -> np.ndarray:
    image_height, image_width = normalized_bgr.shape[:2]

    x1 = max(0, min(cell.ocr_x1, image_width - 1))
    y1 = max(0, min(cell.ocr_y1, image_height - 1))
    x2 = max(x1 + 1, min(cell.ocr_x2, image_width))
    y2 = max(y1 + 1, min(cell.ocr_y2, image_height))

    return normalized_bgr[
        y1:y2,
        x1:x2,
    ].copy()


def build_blue_mask(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Строга маска синього/синьо-фіолетового рукопису.
    Чорні та сірі лінії таблиці не повинні потрапляти.
    """
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    blue, green, red = cv2.split(
        image_bgr
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    blue_i = blue.astype(np.int16)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)

    mask_bool = (
        (hue >= 90)
        & (hue <= 178)
        & (saturation >= 38)
        & (value >= 35)
        & (value <= 250)
        & ((blue_i - green_i) >= 8)
        & ((blue_i - red_i) >= -35)
    )

    mask = np.where(
        mask_bool,
        255,
        0,
    ).astype(np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    return remove_noise_components(mask)


def remove_noise_components(
    mask: np.ndarray,
) -> np.ndarray:
    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(mask)
    image_height, image_width = mask.shape[:2]

    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])

        touches_border = (
            x <= 1
            or y <= 1
            or x + width >= image_width - 1
            or y + height >= image_height - 1
        )

        line_like = (
            width >= 18
            and height <= 4
            and width >= height * 5
        )

        tiny_border_noise = (
            touches_border
            and area < 35
        )

        if (
            area >= 10
            and width >= 2
            and height >= 3
            and not line_like
            and not tiny_border_noise
        ):
            cleaned[
                labels == component_index
            ] = 255

    return cleaned


def crop_mask_to_content(
    mask: np.ndarray,
    padding: int = 10,
) -> np.ndarray | None:
    points = cv2.findNonZero(mask)

    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)

    if width < 3 or height < 4:
        return None

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(mask.shape[1], x + width + padding)
    y2 = min(mask.shape[0], y + height + padding)

    return mask[
        y1:y2,
        x1:x2,
    ]


def prepare_whole_number(
    mask: np.ndarray,
) -> np.ndarray | None:
    cropped = crop_mask_to_content(
        mask,
        padding=12,
    )

    if cropped is None:
        return None

    target_height = 170
    scale = max(
        2.0,
        target_height / max(1, cropped.shape[0]),
    )

    enlarged = cv2.resize(
        cropped,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    prepared = cv2.bitwise_not(enlarged)

    return cv2.copyMakeBorder(
        prepared,
        35,
        35,
        35,
        35,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def active_ranges(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    projection = np.count_nonzero(
        mask,
        axis=0,
    )

    ranges: list[tuple[int, int]] = []
    start: int | None = None

    for index, count in enumerate(projection):
        active = count > 0

        if active and start is None:
            start = index

        if not active and start is not None:
            ranges.append((start, index))
            start = None

    if start is not None:
        ranges.append((start, len(projection)))

    return [
        (start, end)
        for start, end in ranges
        if end - start >= 3
    ]


def merge_tiny_gaps(
    ranges: list[tuple[int, int]],
    gap_limit: int,
) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged = [ranges[0]]

    for current_start, current_end in ranges[1:]:
        previous_start, previous_end = merged[-1]

        if current_start - previous_end <= gap_limit:
            merged[-1] = (
                previous_start,
                current_end,
            )
        else:
            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

    return merged


def find_valley_split(
    mask: np.ndarray,
) -> int | None:
    height, width = mask.shape[:2]

    if width < 18 or height < 10:
        return None

    projection = np.count_nonzero(
        mask,
        axis=0,
    ).astype(np.float32)

    kernel_size = max(
        3,
        int(round(width * 0.04)),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed = cv2.GaussianBlur(
        projection.reshape(1, -1),
        (kernel_size, 1),
        0,
    ).reshape(-1)

    left_limit = max(
        3,
        int(round(width * 0.25)),
    )

    right_limit = min(
        width - 3,
        int(round(width * 0.75)),
    )

    if right_limit <= left_limit:
        return None

    search = smoothed[
        left_limit:right_limit
    ]

    cut = left_limit + int(
        np.argmin(search)
    )

    left_ink = int(
        np.count_nonzero(mask[:, :cut])
    )

    right_ink = int(
        np.count_nonzero(mask[:, cut:])
    )

    total_ink = left_ink + right_ink

    if total_ink <= 0:
        return None

    if min(left_ink, right_ink) / total_ink < 0.12:
        return None

    neighborhood = max(
        4,
        width // 9,
    )

    left_peak = float(
        smoothed[
            max(0, cut - neighborhood):cut
        ].max(initial=0)
    )

    right_peak = float(
        smoothed[
            cut + 1:min(width, cut + neighborhood)
        ].max(initial=0)
    )

    local_peak = max(
        left_peak,
        right_peak,
        1.0,
    )

    if float(smoothed[cut]) / local_peak > 0.78:
        return None

    return cut


def segment_ranges(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    cropped = crop_mask_to_content(
        mask,
        padding=0,
    )

    if cropped is None:
        return []

    points = cv2.findNonZero(mask)
    assert points is not None
    offset_x, _offset_y, width, height = cv2.boundingRect(points)

    ranges = active_ranges(cropped)

    ranges = merge_tiny_gaps(
        ranges,
        gap_limit=max(
            1,
            int(round(width * 0.02)),
        ),
    )

    if len(ranges) == 2:
        return [
            (
                offset_x + start,
                offset_x + end,
            )
            for start, end in ranges
        ]

    if len(ranges) == 1:
        aspect_ratio = width / max(1, height)

        if aspect_ratio >= 0.72:
            cut = find_valley_split(
                cropped
            )

            if cut is not None:
                return [
                    (
                        offset_x,
                        offset_x + cut,
                    ),
                    (
                        offset_x + cut,
                        offset_x + width,
                    ),
                ]

        start, end = ranges[0]

        return [
            (
                offset_x + start,
                offset_x + end,
            )
        ]

    if len(ranges) > 2:
        scored: list[
            tuple[int, int, int]
        ] = []

        for start, end in ranges:
            ink = int(
                np.count_nonzero(
                    cropped[:, start:end]
                )
            )

            scored.append(
                (
                    ink,
                    start,
                    end,
                )
            )

        selected = sorted(
            scored,
            reverse=True,
        )[:2]

        selected_ranges = sorted(
            (
                start,
                end,
            )
            for _ink, start, end in selected
        )

        return [
            (
                offset_x + start,
                offset_x + end,
            )
            for start, end in selected_ranges
        ]

    return []


def prepare_character(
    mask: np.ndarray,
    x1: int,
    x2: int,
) -> np.ndarray:
    character_mask = mask[
        :,
        x1:x2,
    ]

    points = cv2.findNonZero(
        character_mask
    )

    if points is None:
        return np.full(
            (230, 230),
            255,
            dtype=np.uint8,
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    crop = character_mask[
        y:y + height,
        x:x + width,
    ]

    padding = max(
        8,
        int(
            round(
                max(width, height) * 0.15
            )
        ),
    )

    side = max(
        height + 2 * padding,
        width + 2 * padding,
    )

    canvas = np.zeros(
        (side, side),
        dtype=np.uint8,
    )

    offset_x = (side - width) // 2
    offset_y = (side - height) // 2

    canvas[
        offset_y:offset_y + height,
        offset_x:offset_x + width,
    ] = crop

    enlarged = cv2.resize(
        canvas,
        (180, 180),
        interpolation=cv2.INTER_NEAREST,
    )

    return cv2.copyMakeBorder(
        cv2.bitwise_not(enlarged),
        25,
        25,
        25,
        25,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def normalize_text(
    text: str,
) -> str:
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

    value = str(text or "").strip()

    for source, target in substitutions.items():
        value = value.replace(source, target)

    return "".join(
        character
        for character in value
        if character.isdigit()
    )


def tesseract_candidate(
    image: np.ndarray,
    psm: int,
    method: str,
) -> OcrCandidate | None:
    data = pytesseract.image_to_data(
        image,
        config=(
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=0123456789"
        ),
        output_type=pytesseract.Output.DICT,
    )

    best: OcrCandidate | None = None

    for text, confidence_text in zip(
        data.get("text", []),
        data.get("conf", []),
    ):
        value = normalize_text(text)

        try:
            confidence = float(confidence_text)
        except (TypeError, ValueError):
            confidence = -1.0

        if not value:
            continue

        candidate = OcrCandidate(
            value=value,
            confidence=confidence,
            method=method,
        )

        if (
            best is None
            or candidate.confidence > best.confidence
        ):
            best = candidate

    return best


def recognize_whole(
    prepared: np.ndarray,
) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []

    for psm in (7, 8, 13):
        candidate = tesseract_candidate(
            prepared,
            psm=psm,
            method=f"whole_psm_{psm}",
        )

        if candidate is not None:
            candidates.append(candidate)

    return candidates


def recognize_segments(
    mask: np.ndarray,
    segment_dir: Path,
    stem: str,
) -> tuple[
    OcrCandidate | None,
    tuple[Path, ...],
]:
    ranges = segment_ranges(mask)

    if not ranges or len(ranges) > 2:
        return None, ()

    values: list[str] = []
    confidences: list[float] = []
    paths: list[Path] = []

    segment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, (x1, x2) in enumerate(
        ranges,
        start=1,
    ):
        image = prepare_character(
            mask,
            x1,
            x2,
        )

        path = (
            segment_dir
            / f"{stem}_digit_{index}.png"
        )

        cv2.imwrite(
            str(path),
            image,
        )

        paths.append(path)

        candidate = tesseract_candidate(
            image,
            psm=10,
            method="segmented",
        )

        if (
            candidate is None
            or len(candidate.value) != 1
        ):
            return None, tuple(paths)

        values.append(
            candidate.value
        )

        confidences.append(
            candidate.confidence
        )

    return (
        OcrCandidate(
            value="".join(values),
            confidence=min(
                confidences,
                default=-1.0,
            ),
            method="segmented",
        ),
        tuple(paths),
    )


def select_best_candidate(
    candidates: list[OcrCandidate],
) -> OcrCandidate | None:
    if not candidates:
        return None

    grouped: dict[
        str,
        list[OcrCandidate],
    ] = {}

    for candidate in candidates:
        grouped.setdefault(
            candidate.value,
            [],
        ).append(candidate)

    scored: list[
        tuple[float, OcrCandidate]
    ] = []

    for value, group in grouped.items():
        best = max(
            group,
            key=lambda item: item.confidence,
        )

        methods = {
            candidate.method
            for candidate in group
        }

        agreement_bonus = (
            18.0
            if len(methods) >= 2
            else 0.0
        )

        segmented_bonus = (
            6.0
            if "segmented" in methods
            else 0.0
        )

        length_bonus = (
            3.0
            if len(value) <= 2
            else -20.0
        )

        score = (
            best.confidence
            + agreement_bonus
            + segmented_bonus
            + length_bonus
        )

        scored.append(
            (
                score,
                best,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored[0][1]


def recognize_cell(
    normalized_bgr: np.ndarray,
    cell: CellResult,
    output_dir: Path,
) -> DigitOcrResult:
    source_dir = output_dir / "source_cells"
    prepared_dir = output_dir / "prepared_numbers"
    segment_dir = output_dir / "segments"

    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"r{cell.row:02d}_"
        f"c{cell.column}_"
        f"{cell.day}"
    )

    source = crop_ocr_zone(
        normalized_bgr,
        cell,
    )

    source_path = (
        source_dir
        / f"{stem}.png"
    )

    cv2.imwrite(
        str(source_path),
        source,
    )

    mask = build_blue_mask(
        source
    )

    prepared = prepare_whole_number(
        mask
    )

    if prepared is None:
        return DigitOcrResult(
            row=cell.row,
            column=cell.column,
            day=cell.day,
            value="",
            confidence=0.0,
            status="ПЕРЕВІРИТИ",
            method="none",
            source_path=source_path,
            prepared_path=None,
            segment_paths=(),
        )

    prepared_path = (
        prepared_dir
        / f"{stem}.png"
    )

    cv2.imwrite(
        str(prepared_path),
        prepared,
    )

    candidates = recognize_whole(
        prepared
    )

    (
        segmented_candidate,
        segment_paths,
    ) = recognize_segments(
        mask,
        segment_dir,
        stem,
    )

    if segmented_candidate is not None:
        candidates.append(
            segmented_candidate
        )

    best = select_best_candidate(
        candidates
    )

    if best is None:
        value = ""
        confidence = 0.0
        method = "none"
    else:
        value = best.value
        confidence = max(
            0.0,
            min(100.0, best.confidence),
        ) / 100.0
        method = best.method

    status = (
        "OK"
        if value and confidence >= 0.80
        else "ПЕРЕВІРИТИ"
    )

    return DigitOcrResult(
        row=cell.row,
        column=cell.column,
        day=cell.day,
        value=value,
        confidence=confidence,
        status=status,
        method=method,
        source_path=source_path,
        prepared_path=prepared_path,
        segment_paths=segment_paths,
    )


def recognize_filled_cells(
    normalized_bgr: np.ndarray,
    analysis: CellAnalysisResult,
    debug_dir: Path,
) -> DigitOcrBatchResult:
    configure_tesseract()

    items = [
        recognize_cell(
            normalized_bgr,
            cell,
            debug_dir,
        )
        for cell in analysis.cells
        if cell.is_filled
    ]

    recognized_count = sum(
        1
        for item in items
        if item.value
    )

    review_count = sum(
        1
        for item in items
        if item.status == "ПЕРЕВІРИТИ"
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
            "Метод",
            "Оригінальна клітинка",
            "Підготовлене число",
            "Сегменти",
        ])

        for item in batch.items:
            writer.writerow([
                item.row,
                item.column,
                item.day,
                item.value,
                f"{item.confidence:.4f}",
                item.status,
                item.method,
                str(item.source_path or ""),
                str(item.prepared_path or ""),
                " | ".join(
                    str(path)
                    for path in item.segment_paths
                ),
            ])
