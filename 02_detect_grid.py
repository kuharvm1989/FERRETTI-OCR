from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "debug" / "page_001_300dpi.png"
OUTPUT_DIR = BASE_DIR / "debug"

# Орієнтовна область таблиці для FORM-V1.
# Поки що це стартові значення — після тесту за потреби підкоригуємо.
TABLE_X1 = 210
TABLE_Y1 = 760
TABLE_X2 = 2260
TABLE_Y2 = 3200


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено зображення: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(
            "OpenCV не зміг відкрити PNG."
        )

    height, width = image.shape[:2]

    print(f"Розмір сторінки: {width} × {height}")

    x1 = max(0, min(TABLE_X1, width - 1))
    y1 = max(0, min(TABLE_Y1, height - 1))
    x2 = max(x1 + 1, min(TABLE_X2, width))
    y2 = max(y1 + 1, min(TABLE_Y2, height))

    table = image[y1:y2, x1:x2]

    output_path = OUTPUT_DIR / "table_crop.png"

    cv2.imwrite(str(output_path), table)

    preview = image.copy()

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        8,
    )

    preview_path = OUTPUT_DIR / "table_preview.png"

    cv2.imwrite(str(preview_path), preview)

    print(f"Створено: {output_path}")
    print(f"Створено: {preview_path}")
    print(
        f"Область таблиці: "
        f"x={x1}:{x2}, y={y1}:{y2}"
    )


if __name__ == "__main__":
    main()
