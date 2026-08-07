from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "input" / "FORM_V2_TEST.PDF"
OUTPUT_DIR = BASE_DIR / "debug"

DPI = 300


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено тестовий PDF: {PDF_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(PDF_PATH)

    try:
        print(f"Файл: {PDF_PATH.name}")
        print(f"Сторінок: {document.page_count}")

        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            pixmap = page.get_pixmap(
                dpi=DPI,
                alpha=False,
            )

            output_path = (
                OUTPUT_DIR /
                f"page_{page_index + 1:03d}_{DPI}dpi.png"
            )

            pixmap.save(output_path)

            print(
                f"Створено: {output_path}\n"
                f"Розмір: {pixmap.width} × {pixmap.height} px"
            )

    finally:
        document.close()


if __name__ == "__main__":
    main()
