from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DigitSegment:
    index: int
    x1: int
    x2: int
    image: np.ndarray


def crop_to_content(
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    points = cv2.findNonZero(mask)

    if points is None:
        return mask, 0

    x, y, width, height = cv2.boundingRect(
        points
    )

    cropped = mask[
        y:y + height,
        x:x + width,
    ]

    return cropped, x


def find_active_ranges(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    projection = np.count_nonzero(
        mask,
        axis=0,
    )

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


def merge_small_gaps(
    ranges: list[tuple[int, int]],
    max_gap: int,
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

        gap = (
            current_start
            - previous_end
        )

        if gap <= max_gap:
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
        int(
            round(
                width * 0.04
            )
        ),
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
                width * 0.25
            )
        ),
    )

    search_end = min(
        width - 3,
        int(
            round(
                width * 0.75
            )
        ),
    )

    if search_end <= search_start:
        return None

    local_projection = smoothed[
        search_start:search_end
    ]

    cut = (
        search_start
        + int(
            np.argmin(
                local_projection
            )
        )
    )

    left_ink = int(
        np.count_nonzero(
            mask[:, :cut]
        )
    )

    right_ink = int(
        np.count_nonzero(
            mask[:, cut:]
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

    if smaller_part_ratio < 0.12:
        return None

    neighborhood = max(
        4,
        width // 9,
    )

    left_peak = float(
        smoothed[
            max(
                0,
                cut - neighborhood,
            ):
            cut
        ].max(
            initial=0
        )
    )

    right_peak = float(
        smoothed[
            cut + 1:
            min(
                width,
                cut + neighborhood,
            )
        ].max(
            initial=0
        )
    )

    local_peak = max(
        left_peak,
        right_peak,
        1.0,
    )

    valley_ratio = (
        float(
            smoothed[
                cut
            ]
        )
        / local_peak
    )

    if valley_ratio > 0.78:
        return None

    return cut


def detect_digit_ranges(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    content, offset_x = crop_to_content(
        mask
    )

    if content.size == 0:
        return []

    height, width = content.shape[:2]

    raw_ranges = find_active_ranges(
        content
    )

    max_gap = max(
        1,
        int(
            round(
                width * 0.02
            )
        ),
    )

    ranges = merge_small_gaps(
        raw_ranges,
        max_gap,
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

    if len(ranges) == 1:
        aspect_ratio = (
            width
            / max(
                1,
                height,
            )
        )

        if aspect_ratio >= 0.72:
            cut = find_valley_split(
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

        start, end = ranges[0]

        return [
            (
                offset_x + start,
                offset_x + end,
            )
        ]

    if len(ranges) > 2:
        scored_ranges = []

        for start, end in ranges:
            ink = int(
                np.count_nonzero(
                    content[
                        :,
                        start:end,
                    ]
                )
            )

            scored_ranges.append(
                (
                    ink,
                    start,
                    end,
                )
            )

        selected = sorted(
            scored_ranges,
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


def normalize_digit_image(
    mask: np.ndarray,
    x1: int,
    x2: int,
    target_size: int = 64,
) -> np.ndarray:
    digit_mask = mask[
        :,
        x1:x2,
    ]

    points = cv2.findNonZero(
        digit_mask
    )

    if points is None:
        return np.full(
            (
                target_size,
                target_size,
            ),
            255,
            dtype=np.uint8,
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    crop = digit_mask[
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

    resized = cv2.resize(
        canvas,
        (
            target_size,
            target_size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.bitwise_not(
        resized
    )


def split_digits(
    mask: np.ndarray,
) -> list[DigitSegment]:
    ranges = detect_digit_ranges(
        mask
    )

    segments: list[
        DigitSegment
    ] = []

    for index, (
        x1,
        x2,
    ) in enumerate(
        ranges,
        start=1,
    ):
        image = normalize_digit_image(
            mask,
            x1,
            x2,
        )

        segments.append(
            DigitSegment(
                index=index,
                x1=x1,
                x2=x2,
                image=image,
            )
        )

    return segments
