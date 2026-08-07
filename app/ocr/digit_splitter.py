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


def find_major_components(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int, int]]:
    """
    Повертає суттєві зв'язані компоненти рукопису:
    (x, y, width, height, area).

    Дрібні відірвані точки/шум ігноруються.
    """
    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    total_ink = int(
        np.count_nonzero(mask)
    )

    if total_ink <= 0:
        return []

    image_height, image_width = (
        mask.shape[:2]
    )

    minimum_area = max(
        12,
        int(
            round(
                total_ink * 0.07
            )
        ),
    )

    minimum_height = max(
        5,
        int(
            round(
                image_height * 0.25
            )
        ),
    )

    components = []

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

        if area < minimum_area:
            continue

        if height < minimum_height:
            continue

        components.append(
            (
                x,
                y,
                width,
                height,
                area,
            )
        )

    components.sort(
        key=lambda item: item[0]
    )

    return components

def should_accept_split(
    content: np.ndarray,
    cut: int,
) -> bool:
    """
    Перевіряє, чи обидві частини після valley split
    достатньо великі, щоб вважатися окремими цифрами.
    """

    if cut <= 0 or cut >= content.shape[1]:
        return False

    left = content[:, :cut]
    right = content[:, cut:]

    left_pixels = int(
        np.count_nonzero(left)
    )

    right_pixels = int(
        np.count_nonzero(right)
    )

    total_pixels = (
        left_pixels
        + right_pixels
    )

    if total_pixels <= 0:
        return False

    left_ratio = (
        left_pixels
        / total_pixels
    )

    right_ratio = (
        right_pixels
        / total_pixels
    )

    # Обидві частини мають містити
    # суттєву частину всього рукопису.
    if (
        left_ratio < 0.18
        or right_ratio < 0.18
    ):
        return False

    left_points = cv2.findNonZero(
        left
    )

    right_points = cv2.findNonZero(
        right
    )

    if (
        left_points is None
        or right_points is None
    ):
        return False

    (
        _lx,
        _ly,
        left_width,
        left_height,
    ) = cv2.boundingRect(
        left_points
    )

    (
        _rx,
        _ry,
        right_width,
        right_height,
    ) = cv2.boundingRect(
        right_points
    )

    image_height = content.shape[0]

    minimum_height = max(
        8,
        int(
            round(
                image_height * 0.45
            )
        ),
    )

    if (
        left_height < minimum_height
        or right_height < minimum_height
    ):
        return False

    if (
        left_width < 3
        or right_width < 3
    ):
        return False

    return True

def detect_digit_ranges(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    content, offset_x = crop_to_content(
        mask
    )

    if content.size == 0:
        return []

    height, width = content.shape[:2]

    major_components = (
        find_major_components(
            content
        )
    )

    # -------------------------------------------------
    # ВАРІАНТ 1:
    # Є дві суттєві незалежні компоненти.
    #
    # Це наш основний доказ того, що в клітинці
    # написано дві цифри.
    # -------------------------------------------------
    if len(major_components) == 2:
        (
            x1,
            _y1,
            w1,
            h1,
            area1,
        ) = major_components[0]

        (
            x2,
            _y2,
            w2,
            h2,
            area2,
        ) = major_components[1]

        total_area = (
            area1
            + area2
        )

        smaller_area_ratio = (
            min(
                area1,
                area2,
            )
            / max(
                1,
                total_area,
            )
        )

        smaller_height_ratio = (
            min(
                h1,
                h2,
            )
            / max(
                1,
                max(
                    h1,
                    h2,
                ),
            )
        )

    # Для двох справжніх цифр обидві компоненти
    # повинні містити суттєву частину рукопису.
    #
    # Маленький відірваний штрих у 6, 5, 2 тощо
    # не повинен створювати другу цифру.
        two_real_digits = (
            smaller_area_ratio >= 0.18
            and smaller_height_ratio >= 0.55
        )

        if two_real_digits:
            end1 = x1 + w1

            split_x = int(
                round(
                    (
                        end1
                        + x2
                    )
                    / 2
                )
            )

            return [
                (
                    offset_x,
                    offset_x + split_x,
                ),
                (
                    offset_x + split_x,
                    offset_x + width,
                ),
            ]

    # Друга компонента занадто мала:
    # трактуємо весь рукопис як одну цифру.
        return [
            (
                offset_x,
                offset_x + width,
            )
        ]

    # -------------------------------------------------
    # ВАРІАНТ 2:
    # Є одна суттєва компонента.
    #
    # За замовчуванням це ОДНА цифра.
    # Саме це повинно виправити наші 5 і 2.
    # -------------------------------------------------
    if len(major_components) == 1:
        aspect_ratio = (
            width
            / max(
                1,
                height,
            )
        )

        # Valley split дозволяємо тільки для
        # дуже широкої суцільної компоненти.
        #
        # Це запасний випадок для двох цифр,
        # які реально торкнулися одна одної.
        if aspect_ratio >= 1.20:
            cut = find_valley_split(
                content
            )

            if (
                cut is not None
                and should_accept_split(
                    content,
                    cut,
                )
            ):
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

    # -------------------------------------------------
    # ВАРІАНТ 3:
    # Компонент більше двох.
    #
    # Це може бути рукопис із відірваними штрихами.
    # Тут використовуємо вертикальну проєкцію
    # як резервний механізм.
    # -------------------------------------------------
    raw_ranges = find_active_ranges(
        content
    )

    max_gap = max(
        1,
        int(
            round(
                width * 0.06
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

    # Якщо після всіх перевірок немає
    # переконливого доказу двох цифр,
    # безпечніше вважати запис однією цифрою.
    if width > 0:
        return [
            (
                offset_x,
                offset_x + width,
            )
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
