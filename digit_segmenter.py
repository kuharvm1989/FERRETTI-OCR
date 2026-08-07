from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CharacterSegment:
    x1: int
    x2: int
    image: np.ndarray


def _crop_to_content(
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return mask, 0

    x, y, width, height = cv2.boundingRect(
        points
    )

    return (
        mask[
            y:y + height,
            x:x + width,
        ],
        x,
    )


def _active_ranges(
    projection: np.ndarray,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(
        projection
    ):
        active = value > 0

        if active and start is None:
            start = index

        if not active and start is not None:
            ranges.append(
                (
                    start,
                    index,
                )
            )
            start = None

    if start is not None:
        ranges.append(
            (
                start,
                len(projection),
            )
        )

    return ranges


def _merge_small_gaps(
    ranges: list[tuple[int, int]],
    maximum_gap: int,
) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged = [
        ranges[0]
    ]

    for current_start, current_end in ranges[1:]:
        previous_start, previous_end = (
            merged[-1]
        )

        if (
            current_start
            - previous_end
            <= maximum_gap
        ):
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


def _find_valley_cut(
    mask: np.ndarray,
) -> int | None:
    """
    Шукає найменш заповнену вертикальну позицію
    в центральній частині широкого рукописного числа.

    Працює навіть тоді, коли дві цифри торкаються
    одним або кількома пікселями.
    """
    height, width = mask.shape[:2]

    if width < 18 or height < 10:
        return None

    projection = np.count_nonzero(
        mask,
        axis=0,
    ).astype(np.float32)

    # Згладжуємо локальні зубці штриха.
    kernel_size = max(
        3,
        int(
            round(
                width * 0.035
            )
        )
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed = cv2.GaussianBlur(
        projection.reshape(1, -1),
        (
            kernel_size,
            1,
        ),
        0,
    ).reshape(-1)

    search_start = max(
        3,
        int(
            round(
                width * 0.28
            )
        ),
    )

    search_end = min(
        width - 3,
        int(
            round(
                width * 0.72
            )
        ),
    )

    if search_end <= search_start:
        return None

    local = smoothed[
        search_start:search_end
    ]

    candidate = (
        search_start
        + int(
            np.argmin(
                local
            )
        )
    )

    left_ink = int(
        np.count_nonzero(
            mask[:, :candidate]
        )
    )

    right_ink = int(
        np.count_nonzero(
            mask[:, candidate:]
        )
    )

    total_ink = (
        left_ink
        + right_ink
    )

    if total_ink <= 0:
        return None

    smaller_part_ratio = (
        min(
            left_ink,
            right_ink,
        )
        / total_ink
    )

    # Обидві частини мають містити суттєву кількість чорнила.
    if smaller_part_ratio < 0.13:
        return None

    valley_height = float(
        smoothed[
            candidate
        ]
    )

    local_peak = max(
        float(
            smoothed[
                max(
                    0,
                    candidate
                    - max(
                        4,
                        width // 8,
                    )
                ):
                candidate
            ].max(
                initial=0
            )
        ),
        float(
            smoothed[
                candidate + 1:
                min(
                    width,
                    candidate
                    + max(
                        5,
                        width // 8,
                    )
                )
            ].max(
                initial=0
            )
        ),
        1.0,
    )

    # Навіть у дотичних цифр долина має бути нижчою за сусідні штрихи.
    if valley_height / local_peak > 0.72:
        return None

    return candidate


def find_character_ranges(
    mask: np.ndarray,
    maximum_characters: int = 2,
) -> list[tuple[int, int]]:
    """
    Повертає діапазони окремих цифр.

    Алгоритм:
    1. використовує реальні порожні проміжки;
    2. якщо число є одним широким компонентом,
       шукає центральну долину та розділяє дотичні цифри.
    """
    if mask is None or mask.size == 0:
        return []

    content, offset_x = _crop_to_content(
        mask
    )

    if content.size == 0:
        return []

    height, width = content.shape[:2]

    projection = np.count_nonzero(
        content,
        axis=0,
    )

    ranges = _active_ranges(
        projection
    )

    maximum_gap = max(
        1,
        int(
            round(
                width * 0.025
            )
        ),
    )

    ranges = _merge_small_gaps(
        ranges,
        maximum_gap,
    )

    ranges = [
        (
            start,
            end,
        )
        for start, end in ranges
        if end - start >= 3
    ]

    if len(ranges) == 2:
        return [
            (
                offset_x + start,
                offset_x + end,
            )
            for start, end in ranges
        ]

    if len(ranges) > 2:
        # Залишаємо дві найбільші групи за кількістю чорнила.
        scored = []

        for start, end in ranges:
            ink = int(
                np.count_nonzero(
                    content[:, start:end]
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
        )[:maximum_characters]

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

    # Один компонент: розділяємо лише якщо він досить широкий.
    aspect_ratio = (
        width / max(
            1,
            height,
        )
    )

    if (
        len(ranges) == 1
        and aspect_ratio >= 0.72
    ):
        cut = _find_valley_cut(
            content
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

    if len(ranges) == 1:
        start, end = ranges[0]

        return [
            (
                offset_x + start,
                offset_x + end,
            )
        ]

    return []


def prepare_character_image(
    mask: np.ndarray,
    x1: int,
    x2: int,
    output_size: int = 180,
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
            (
                output_size + 50,
                output_size + 50,
            ),
            255,
            dtype=np.uint8,
        )

    local_x, local_y, width, height = (
        cv2.boundingRect(
            points
        )
    )

    crop = character_mask[
        local_y:local_y + height,
        local_x:local_x + width,
    ]

    padding = max(
        6,
        int(
            round(
                max(
                    width,
                    height,
                )
                * 0.12
            )
        ),
    )

    canvas_height = (
        height
        + 2 * padding
    )

    canvas_width = (
        width
        + 2 * padding
    )

    side = max(
        canvas_height,
        canvas_width,
    )

    square = np.zeros(
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

    square[
        offset_y:
        offset_y + height,
        offset_x:
        offset_x + width,
    ] = crop

    enlarged = cv2.resize(
        square,
        (
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    prepared = cv2.bitwise_not(
        enlarged
    )

    return cv2.copyMakeBorder(
        prepared,
        25,
        25,
        25,
        25,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def segment_characters(
    mask: np.ndarray,
) -> list[CharacterSegment]:
    ranges = find_character_ranges(
        mask
    )

    return [
        CharacterSegment(
            x1=x1,
            x2=x2,
            image=prepare_character_image(
                mask,
                x1,
                x2,
            ),
        )
        for x1, x2 in ranges
    ]
