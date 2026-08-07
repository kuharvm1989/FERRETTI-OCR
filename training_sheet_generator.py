from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


BASE_DIR = Path(__file__).resolve().parent

MARKERS_DIR = BASE_DIR / "markers"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_PDF = (
    OUTPUT_DIR
    / "FERRETTI_DIGIT_TRAINING_SHEET.pdf"
)


# A4 у пунктах PDF.
MM = 72.0 / 25.4

PAGE_WIDTH = 210 * MM
PAGE_HEIGHT = 297 * MM


def mm(value: float) -> float:
    return value * MM


def add_marker(
    page: fitz.Page,
    marker_id: int,
    x_mm: float,
    y_mm: float,
    size_mm: float = 10,
) -> None:
    marker_path = (
        MARKERS_DIR
        / f"aruco_{marker_id}.png"
    )

    if not marker_path.exists():
        raise FileNotFoundError(
            f"Не знайдено ArUco marker: "
            f"{marker_path}"
        )

    rect = fitz.Rect(
        mm(x_mm),
        mm(y_mm),
        mm(x_mm + size_mm),
        mm(y_mm + size_mm),
    )

    page.insert_image(
        rect,
        filename=str(marker_path),
    )


def draw_training_grid(
    page: fitz.Page,
) -> None:
    """
    10 рядків:
    0 ... 9

    У кожному рядку — 20 клітинок.
    """

    grid_left_mm = 17
    grid_right_mm = 198

    grid_top_mm = 38
    grid_bottom_mm = 278

    digit_label_width_mm = 10

    columns = 20
    rows = 10

    cells_left_mm = (
        grid_left_mm
        + digit_label_width_mm
    )

    cell_width_mm = (
        grid_right_mm
        - cells_left_mm
    ) / columns

    row_height_mm = (
        grid_bottom_mm
        - grid_top_mm
    ) / rows

    # Вертикальна межа між номером цифри
    # та клітинками.
    page.draw_line(
        fitz.Point(
            mm(cells_left_mm),
            mm(grid_top_mm),
        ),
        fitz.Point(
            mm(cells_left_mm),
            mm(grid_bottom_mm),
        ),
        width=0.7,
    )

    # Горизонтальні лінії.
    for row in range(
        rows + 1
    ):
        y = (
            grid_top_mm
            + row * row_height_mm
        )

        page.draw_line(
            fitz.Point(
                mm(grid_left_mm),
                mm(y),
            ),
            fitz.Point(
                mm(grid_right_mm),
                mm(y),
            ),
            width=0.7,
        )

    # Вертикальні лінії клітинок.
    for column in range(
        columns + 1
    ):
        x = (
            cells_left_mm
            + column * cell_width_mm
        )

        page.draw_line(
            fitz.Point(
                mm(x),
                mm(grid_top_mm),
            ),
            fitz.Point(
                mm(x),
                mm(grid_bottom_mm),
            ),
            width=0.45,
        )

    # Ліва і права межі.
    page.draw_line(
        fitz.Point(
            mm(grid_left_mm),
            mm(grid_top_mm),
        ),
        fitz.Point(
            mm(grid_left_mm),
            mm(grid_bottom_mm),
        ),
        width=0.7,
    )

    page.draw_line(
        fitz.Point(
            mm(grid_right_mm),
            mm(grid_top_mm),
        ),
        fitz.Point(
            mm(grid_right_mm),
            mm(grid_bottom_mm),
        ),
        width=0.7,
    )

    # Підписи 0–9.
    for digit in range(
        10
    ):
        row_top = (
            grid_top_mm
            + digit * row_height_mm
        )

        row_bottom = (
            row_top
            + row_height_mm
        )

        label_rect = fitz.Rect(
            mm(grid_left_mm),
            mm(row_top),
            mm(cells_left_mm),
            mm(row_bottom),
        )

        page.insert_textbox(
            label_rect,
            str(digit),
            fontsize=18,
            fontname="helv",
            align=fitz.TEXT_ALIGN_CENTER,
        )


def create_training_sheet() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = fitz.open()

    page = document.new_page(
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
    )

    # Заголовок.
    title_rect = fitz.Rect(
        mm(25),
        mm(8),
        mm(185),
        mm(17),
    )

    page.insert_textbox(
        title_rect,
        "FERRETTI OCR - HANDWRITTEN DIGIT TRAINING",
        fontsize=13,
        fontname="helv",
        align=fitz.TEXT_ALIGN_CENTER,
    )

    # Поля метаданих.
    page.insert_text(
        fitz.Point(
            mm(24),
            mm(24),
        ),
        "Employee:",
        fontsize=9,
    )

    page.draw_line(
        fitz.Point(
            mm(44),
            mm(25),
        ),
        fitz.Point(
            mm(115),
            mm(25),
        ),
        width=0.5,
    )

    page.insert_text(
        fitz.Point(
            mm(125),
            mm(24),
        ),
        "Date:",
        fontsize=9,
    )

    page.draw_line(
        fitz.Point(
            mm(138),
            mm(25),
        ),
        fitz.Point(
            mm(177),
            mm(25),
        ),
        width=0.5,
    )

    instruction_rect = fitz.Rect(
        mm(24),
        mm(28),
        mm(185),
        mm(35),
    )

    page.insert_textbox(
        instruction_rect,
        (
            "Write the digit shown at the left "
            "once in every cell of the row."
        ),
        fontsize=8,
        align=fitz.TEXT_ALIGN_CENTER,
    )

    # ArUco:
    # ID0 верх-ліво
    # ID1 верх-право
    # ID2 низ-право
    # ID3 низ-ліво
    add_marker(
        page,
        0,
        4,
        4,
    )

    add_marker(
        page,
        1,
        196,
        4,
    )

    add_marker(
        page,
        2,
        196,
        283,
    )

    add_marker(
        page,
        3,
        4,
        283,
    )

    draw_training_grid(
        page
    )

    document.save(
        OUTPUT_PDF,
        garbage=4,
        deflate=True,
    )

    document.close()

    print()
    print(
        "Навчальний лист створено:"
    )

    print(
        OUTPUT_PDF
    )

    print()
    print(
        "Кількість клітинок: 200"
    )

    print(
        "20 прикладів кожної цифри 0-9"
    )


if __name__ == "__main__":
    create_training_sheet()
