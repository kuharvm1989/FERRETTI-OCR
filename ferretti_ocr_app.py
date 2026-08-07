from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageTk

from cell_parser import CellAnalysisResult, analyze_cells
from app.ocr.ocr_engine import (
    DigitOcrBatchResult,
    export_ocr_results_csv,
    recognize_filled_cells,
)
from app.dataset.manual_corrections import (
    save_manual_correction,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
DEBUG_DIR = BASE_DIR / "debug"
CONFIG_DIR = BASE_DIR / "config"
FORM_V2_CONFIG_PATH = CONFIG_DIR / "form_v2.json"
CELL_GEOMETRY_CONFIG_PATH = CONFIG_DIR / "cell_geometry.json"
CELL_OUTPUT_DIR = DEBUG_DIR / "cells_v2"
CELL_PREVIEW_PATH = DEBUG_DIR / "cells_detection_preview.png"
OCR_OUTPUT_DIR = DEBUG_DIR / "ocr_engine_v2"
OCR_RESULTS_CSV = DEBUG_DIR / "ocr_results.csv"

APP_TITLE = "FERRETTI Production OCR"
RENDER_DPI = 300

ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50
EXPECTED_MARKERS = {0, 1, 2, 3}

# Нормалізоване зображення має фіксовану висоту,
# а ширина обчислюється пропорційно геометрії між ArUco-мітками.
# Це усуває непропорційне розтягування рукописних цифр.
NORMALIZED_HEIGHT = 3508
MIN_NORMALIZED_WIDTH = 600
MAX_NORMALIZED_WIDTH = 3000

PREVIEW_MAX_WIDTH = 560
PREVIEW_MAX_HEIGHT = 720


class FerrettiOcrApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1250x900")
        self.root.minsize(1000, 700)

        self.pdf_path: Path | None = None
        self.pdf_document: pymupdf.Document | None = None
        self.page_index = 0

        self.original_bgr: np.ndarray | None = None
        self.normalized_bgr: np.ndarray | None = None
        self.last_cell_analysis: CellAnalysisResult | None = None
        self.last_ocr_batch: DigitOcrBatchResult | None = None

        self.original_photo: ImageTk.PhotoImage | None = None
        self.normalized_photo: ImageTk.PhotoImage | None = None
        self.table_calibration_active = False
        self.table_calibration_points: list[tuple[int, int]] = []
        self.normalized_preview_scale = 1.0
        self.normalized_preview_offset_x = 0
        self.normalized_preview_offset_y = 0

        self.file_var = tk.StringVar(
            value="PDF не вибрано"
        )

        self.page_var = tk.StringVar(
            value="Сторінка: —"
        )

        self.marker_var = tk.StringVar(
            value="Мітки: —"
        )

        self.status_var = tk.StringVar(
            value="Виберіть PDF для початку роботи."
        )

        self.build_interface()

    def build_interface(self) -> None:
        toolbar = ttk.Frame(
            self.root,
            padding=8,
        )
        toolbar.pack(
            fill=tk.X,
        )

        ttk.Button(
            toolbar,
            text="Відкрити PDF",
            command=self.select_pdf,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Калібрувати таблицю",
            command=self.start_table_calibration,
        ).pack(
            side=tk.LEFT,
            padx=12,
        )

        ttk.Button(
            toolbar,
            text="Зберегти калібрування",
            command=self.save_table_calibration,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        self.previous_button = ttk.Button(
            toolbar,
            text="← Попередня сторінка",
            command=self.previous_page,
            state=tk.DISABLED,
        )
        self.previous_button.pack(
            side=tk.LEFT,
            padx=4,
        )

        self.next_button = ttk.Button(
            toolbar,
            text="Наступна сторінка →",
            command=self.next_page,
            state=tk.DISABLED,
        )
        self.next_button.pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Знайти мітки та вирівняти",
            command=self.detect_and_normalize,
        ).pack(
            side=tk.LEFT,
            padx=12,
        )

        ttk.Button(
            toolbar,
            text="Зберегти вирівняну сторінку",
            command=self.save_normalized_page,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Знайти заповнені клітинки",
            command=self.detect_filled_cells,
        ).pack(
            side=tk.LEFT,
            padx=12,
        )

        ttk.Button(
            toolbar,
            text="Розпізнати цифри",
            command=self.recognize_digits,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        info_frame = ttk.Frame(
            self.root,
            padding=(12, 4),
        )
        info_frame.pack(
            fill=tk.X,
        )

        ttk.Label(
            info_frame,
            textvariable=self.file_var,
        ).pack(
            side=tk.LEFT,
            padx=(0, 30),
        )

        ttk.Label(
            info_frame,
            textvariable=self.page_var,
        ).pack(
            side=tk.LEFT,
            padx=(0, 30),
        )

        ttk.Label(
            info_frame,
            textvariable=self.marker_var,
        ).pack(
            side=tk.LEFT,
        )

        preview_container = ttk.Frame(
            self.root,
            padding=8,
        )
        preview_container.pack(
            fill=tk.BOTH,
            expand=True,
        )

        original_frame = ttk.LabelFrame(
            preview_container,
            text="Оригінальна сторінка",
            padding=6,
        )

        normalized_frame = ttk.LabelFrame(
            preview_container,
            text="Вирівняна сторінка",
            padding=6,
        )

        original_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 4),
        )

        normalized_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(4, 0),
        )

        self.original_canvas = tk.Canvas(
            original_frame,
            background="#d8d8d8",
            highlightthickness=0,
        )
        self.original_canvas.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.normalized_canvas = tk.Canvas(
            normalized_frame,
            background="#d8d8d8",
            highlightthickness=0,
        )
        self.normalized_canvas.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.normalized_canvas.bind(
            "<Button-1>",
            self.on_normalized_canvas_click,
        )

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=6,
        )
        status_bar.pack(
            fill=tk.X,
            side=tk.BOTTOM,
        )

    def edit_ocr_result(
        self,
        event=None,
    ) -> None:
        selection = self.results_tree.selection()

        if not selection:
            return

        item_id = selection[0]

        values = self.results_tree.item(
            item_id,
            "values",
        )

        if not values:
            return

        try:
            row_number = int(
                values[0]
            )

            column_number = int(
                values[1]
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            return

        current_value = ""

        if len(values) >= 4:
            current_value = str(
                values[3]
            )

        editor = tk.Toplevel(
            self.root
        )

        editor.title(
            "Перевірка OCR"
        )

        editor.geometry(
            "460x520"
        )

        editor.resizable(
            False,
            False,
        )

        editor.transient(
            self.root
        )

        editor.grab_set()

        title = tk.Label(
            editor,
            text=(
                f"Рядок {row_number}, "
                f"колонка {column_number}"
            ),
            font=(
                "Arial",
                14,
                "bold",
            ),
        )

        title.pack(
            pady=10
        )

        image_label = tk.Label(
            editor,
            bg="white",
            width=400,
            height=300,
        )

        image_label.pack(
            padx=20,
            pady=10,
        )

        image_path = (
            DEBUG_DIR
            / "ocr_engine_v2"
            / "source_cells"
            / (
                f"r{row_number:02d}_"
                f"c{column_number}_"
            )
        )

        matching_files = list(
            image_path.parent.glob(
                image_path.name
                + "*.png"
            )
        )

        photo = None

        if matching_files:
            try:
                image = Image.open(
                    matching_files[0]
                )

                image.thumbnail(
                    (
                        380,
                        280,
                    )
                )

                photo = ImageTk.PhotoImage(
                    image
                )

                image_label.configure(
                    image=photo,
                    text="",
                )

                image_label.image = photo

            except Exception:
                image_label.configure(
                    text=(
                        "Не вдалося "
                        "відкрити зображення"
                    )
                )
        else:
            image_label.configure(
                text=(
                    "Зображення клітинки "
                    "не знайдено"
                )
            )

        tk.Label(
            editor,
            text="Значення:",
            font=(
                "Arial",
                12,
            ),
        ).pack(
            pady=5
        )

        value_var = tk.StringVar(
            value=current_value
        )

        value_entry = tk.Entry(
            editor,
            textvariable=value_var,
            font=(
                "Arial",
                22,
                "bold",
            ),
            justify="center",
            width=8,
        )

        value_entry.pack(
            pady=8
        )
    
        value_entry.focus_set()

        def save_value() -> None:
            new_value = (
                value_var.get()
                .strip()
            )

            if (
                new_value
                and not new_value.isdigit()
            ):
                messagebox.showerror(
                    "Помилка",
                    "Допускаються тільки цифри.",
                    parent=editor,
                )
                return

            new_values = list(
                values
            )

            if len(new_values) >= 4:
                new_values[3] = new_value

            if len(new_values) >= 6:
                new_values[5] = "ПІДТВЕРДЖЕНО"

            self.results_tree.item(
                item_id,
                values=new_values,
            )

            day_value = ""

            if len(values) >= 3:
                day_value = str(
                    values[2]
                )

            source_image_path = ""

            if matching_files:
                source_image_path = str(
                    matching_files[0]
                )

            save_manual_correction(
                row=row_number,
                column=column_number,
                day=day_value,
                ocr_value=current_value,
                confirmed_value=new_value,
                source_image=source_image_path,
            )

            editor.destroy()

        save_button = tk.Button(
            editor,
            text="Зберегти",
            command=save_value,
            font=(
                "Arial",
                12,
                "bold",
            ),
            width=14,
        )

        save_button.pack(
            pady=15
        )

        editor.bind(
            "<Return>",
            lambda _event: save_value(),
        )

    def select_pdf(self) -> None:
        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected = filedialog.askopenfilename(
            title="Виберіть PDF із листами виробітку",
            initialdir=str(INPUT_DIR),
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
        )

        if not selected:
            return

        self.close_pdf()

        try:
            document = pymupdf.open(selected)
        except Exception as error:
            messagebox.showerror(
                "Помилка відкриття PDF",
                str(error),
            )
            return

        if document.page_count == 0:
            document.close()

            messagebox.showerror(
                "Порожній PDF",
                "Документ не містить сторінок.",
            )
            return

        self.pdf_document = document
        self.pdf_path = Path(selected)
        self.page_index = 0

        self.file_var.set(
            f"Файл: {self.pdf_path.name}"
        )

        self.render_current_page()

    def close_pdf(self) -> None:
        if self.pdf_document is not None:
            self.pdf_document.close()

        self.pdf_document = None
        self.pdf_path = None
        self.original_bgr = None
        self.normalized_bgr = None

    def render_current_page(self) -> None:
        if self.pdf_document is None:
            return

        try:
            page = self.pdf_document.load_page(
                self.page_index
            )

            pixmap = page.get_pixmap(
                dpi=RENDER_DPI,
                alpha=False,
            )

            image_array = np.frombuffer(
                pixmap.samples,
                dtype=np.uint8,
            ).reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )

            if pixmap.n == 4:
                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGBA2BGR,
                )
            else:
                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR,
                )

        except Exception as error:
            messagebox.showerror(
                "Помилка відображення сторінки",
                str(error),
            )
            return

        self.original_bgr = image_bgr
        self.normalized_bgr = None
        self.last_cell_analysis = None
        self.last_ocr_batch = None

        self.page_var.set(
            f"Сторінка: {self.page_index + 1} "
            f"із {self.pdf_document.page_count}"
        )

        self.marker_var.set(
            "Мітки: не перевірено"
        )

        self.status_var.set(
            "Сторінку завантажено. "
            "Натисніть «Знайти мітки та вирівняти»."
        )

        self.show_bgr_image(
            self.original_canvas,
            self.original_bgr,
            target="original",
        )

        self.normalized_canvas.delete("all")

        self.update_navigation_buttons()

    def previous_page(self) -> None:
        if (
            self.pdf_document is None
            or self.page_index <= 0
        ):
            return

        self.page_index -= 1
        self.render_current_page()

    def next_page(self) -> None:
        if self.pdf_document is None:
            return

        if (
            self.page_index >=
            self.pdf_document.page_count - 1
        ):
            return

        self.page_index += 1
        self.render_current_page()

    def update_navigation_buttons(self) -> None:
        if self.pdf_document is None:
            self.previous_button.configure(
                state=tk.DISABLED
            )
            self.next_button.configure(
                state=tk.DISABLED
            )
            return

        self.previous_button.configure(
            state=(
                tk.NORMAL
                if self.page_index > 0
                else tk.DISABLED
            )
        )

        self.next_button.configure(
            state=(
                tk.NORMAL
                if self.page_index <
                self.pdf_document.page_count - 1
                else tk.DISABLED
            )
        )

    def detect_and_normalize(self) -> None:
        if self.original_bgr is None:
            messagebox.showwarning(
                "Немає сторінки",
                "Спочатку відкрийте PDF.",
            )
            return

        try:
            detected = detect_aruco_markers(
                self.original_bgr
            )
        except Exception as error:
            messagebox.showerror(
                "Помилка пошуку міток",
                str(error),
            )
            return

        detected_ids = set(detected)
        missing_ids = EXPECTED_MARKERS - detected_ids

        preview = self.original_bgr.copy()

        for marker_id, marker_data in detected.items():
            corners = marker_data["corners"].astype(
                np.int32
            )

            center = marker_data["center"]

            cv2.polylines(
                preview,
                [corners],
                True,
                (0, 255, 0),
                5,
            )

            cv2.circle(
                preview,
                center,
                12,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                preview,
                f"ID {marker_id}",
                (
                    center[0] + 15,
                    center[1] - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        self.show_bgr_image(
            self.original_canvas,
            preview,
            target="original",
        )

        if missing_ids:
            self.marker_var.set(
                "Мітки: знайдено "
                f"{len(detected_ids)} із 4"
            )

            raise_message = (
                "Не знайдено мітки: "
                + ", ".join(
                    str(value)
                    for value in sorted(missing_ids)
                )
            )

            messagebox.showerror(
                "Не всі мітки знайдено",
                raise_message,
            )

            self.status_var.set(
                raise_message
            )
            return

        try:
            normalized = normalize_page_by_markers(
                self.original_bgr,
                detected,
            )
        except Exception as error:
            messagebox.showerror(
                "Помилка вирівнювання",
                str(error),
            )
            return

        self.normalized_bgr = normalized

        self.marker_var.set(
            "Мітки: 4 із 4"
        )

        normalized_height, normalized_width = (
            self.normalized_bgr.shape[:2]
        )

        self.status_var.set(
            "Сторінку вирівняно без зміни пропорцій: "
            f"{normalized_width} × {normalized_height} px."
        )

        self.show_bgr_image(
            self.normalized_canvas,
            self.normalized_bgr,
            target="normalized",
        )

        self.save_debug_results(
            preview,
            normalized,
        )

    def save_debug_results(
        self,
        marker_preview: np.ndarray,
        normalized: np.ndarray,
    ) -> None:
        DEBUG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        page_number = self.page_index + 1

        marker_path = (
            DEBUG_DIR /
            f"page_{page_number:03d}_markers.png"
        )

        normalized_path = (
            DEBUG_DIR /
            f"page_{page_number:03d}_normalized.png"
        )

        cv2.imwrite(
            str(marker_path),
            marker_preview,
        )

        cv2.imwrite(
            str(normalized_path),
            normalized,
        )

    def start_table_calibration(self) -> None:
        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                "Спочатку знайдіть мітки "
                "та вирівняйте сторінку.",
            )
            return

        self.table_calibration_points.clear()
        self.table_calibration_active = True

        self.status_var.set(
            "Калібрування таблиці: клацніть "
            "верхній лівий кут першої клітинки ПН."
        )

        self.draw_table_calibration()


    def on_normalized_canvas_click(
        self,
        event: tk.Event,
    ) -> None:
        if (
            not self.table_calibration_active
            or self.normalized_bgr is None
        ):
            return

        canvas_x = self.normalized_canvas.canvasx(
            event.x
        )

        canvas_y = self.normalized_canvas.canvasy(
            event.y
        )

        image_x = (
            canvas_x -
            self.normalized_preview_offset_x
        )

        image_y = (
            canvas_y -
            self.normalized_preview_offset_y
        )

        if self.normalized_preview_scale <= 0:
            return

        original_x = round(
            image_x /
            self.normalized_preview_scale
        )

        original_y = round(
            image_y /
            self.normalized_preview_scale
        )

        height, width = self.normalized_bgr.shape[:2]

        if not (
            0 <= original_x < width
            and 0 <= original_y < height
        ):
            return

        self.table_calibration_points.append(
            (original_x, original_y)
        )

        if len(self.table_calibration_points) == 1:
            self.status_var.set(
                "Перша точка збережена. "
                "Клацніть нижній правий кут "
                "останньої клітинки СБ."
            )

        elif len(self.table_calibration_points) >= 2:
            self.table_calibration_points = (
                self.table_calibration_points[:2]
            )

            self.table_calibration_active = False

            first = self.table_calibration_points[0]
            second = self.table_calibration_points[1]

            self.status_var.set(
                f"Калібрування готове: "
                f"({first[0]}, {first[1]}) → "
                f"({second[0]}, {second[1]})."
            )

        self.draw_table_calibration()


    def draw_table_calibration(self) -> None:
        self.normalized_canvas.delete(
            "table_calibration"
        )

        if not self.table_calibration_points:
            return

        display_points = []

        for index, point in enumerate(
            self.table_calibration_points,
            start=1,
        ):
            display_x = (
                point[0] *
                self.normalized_preview_scale +
                self.normalized_preview_offset_x
            )

            display_y = (
                point[1] *
                self.normalized_preview_scale +
                self.normalized_preview_offset_y
            )

            display_points.append(
                (display_x, display_y)
            )

            radius = 6

            self.normalized_canvas.create_oval(
                display_x - radius,
                display_y - radius,
                display_x + radius,
                display_y + radius,
                outline="red",
                fill="yellow",
                width=2,
                tags="table_calibration",
            )

            self.normalized_canvas.create_text(
                display_x + 12,
                display_y - 12,
                text=str(index),
                fill="red",
                font=("Arial", 12, "bold"),
                tags="table_calibration",
            )

        if len(display_points) == 2:
            first = display_points[0]
            second = display_points[1]

            self.normalized_canvas.create_rectangle(
                first[0],
                first[1],
                second[0],
                second[1],
                outline="red",
                width=3,
                tags="table_calibration",
            )

    def save_table_calibration(self) -> None:
        if len(self.table_calibration_points) != 2:
            messagebox.showwarning(
                "Калібрування не завершено",
                "Потрібно встановити дві точки.",
            )
            return

        first = self.table_calibration_points[0]
        second = self.table_calibration_points[1]

        x1 = min(first[0], second[0])
        y1 = min(first[1], second[1])
        x2 = max(first[0], second[0])
        y2 = max(first[1], second[1])

        config_dir = BASE_DIR / "config"
        config_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_path = (
            config_dir /
            "form_v2.json"
        )

        import json

        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                "Спочатку знайдіть мітки "
                "та вирівняйте сторінку.",
            )
            return

        normalized_height, normalized_width = (
            self.normalized_bgr.shape[:2]
        )

        config = {
            "template_version": "FORM-V2",
            "normalized_width": normalized_width,
            "normalized_height": normalized_height,
            "marker_aspect_ratio": (
                normalized_width / normalized_height
            ),
            "table_grid": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "day_columns": 6,
        }

        config_path.write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        messagebox.showinfo(
            "Калібрування збережено",
            f"Файл:\n{config_path}",
    )

    def detect_filled_cells(self) -> None:
        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                "Спочатку знайдіть ArUco-мітки "
                "та вирівняйте сторінку.",
            )
            return

        if not FORM_V2_CONFIG_PATH.exists():
            messagebox.showwarning(
                "Немає калібрування FORM-V2",
                f"Не знайдено файл:\n{FORM_V2_CONFIG_PATH}\n\n"
                "Виконайте калібрування таблиці та збережіть його.",
            )
            return

        self.status_var.set(
            "Аналіз клітинок: визначаємо заповнені поля..."
        )
        self.root.update_idletasks()

        try:
            analysis = analyze_cells(
                normalized_bgr=self.normalized_bgr,
                config_path=FORM_V2_CONFIG_PATH,
                output_dir=CELL_OUTPUT_DIR,
                preview_path=CELL_PREVIEW_PATH,
                geometry_config_path=CELL_GEOMETRY_CONFIG_PATH,
                save_all_cells=False,
            )
        except Exception as error:
            messagebox.showerror(
                "Помилка аналізу клітинок",
                str(error),
            )
            self.status_var.set(
                "Не вдалося проаналізувати клітинки."
            )
            return

        self.last_cell_analysis = analysis
        self.last_ocr_batch = None

        preview = cv2.imread(str(CELL_PREVIEW_PATH))
        if preview is not None:
            self.show_bgr_image(
                self.normalized_canvas,
                preview,
                target="normalized",
            )

        self.status_var.set(
            f"Аналіз завершено: виявлено рядків {analysis.detected_rows}; "
            f"заповнених клітинок {analysis.filled_cells} "
            f"із {analysis.total_cells}."
        )

        messagebox.showinfo(
            "Аналіз клітинок завершено",
            f"Виявлено рядків: {analysis.detected_rows}\n"
            f"Усього клітинок: {analysis.total_cells}\n"
            f"Ймовірно заповнених: {analysis.filled_cells}\n\n"
            f"Preview:\n{CELL_PREVIEW_PATH}\n\n"
            f"Вирізані клітинки:\n{CELL_OUTPUT_DIR / 'filled'}",
        )

    def recognize_digits(self) -> None:
        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                "Спочатку знайдіть мітки "
                "та вирівняйте сторінку.",
            )
            return

        if self.last_cell_analysis is None:
            messagebox.showwarning(
                "Клітинки ще не проаналізовано",
                "Спочатку натисніть "
                "«Знайти заповнені клітинки».",
            )
            return

        self.status_var.set(
            "OCR: виділяємо синє чорнило "
            "та розпізнаємо цифри..."
        )

        self.root.update_idletasks()

        try:
            batch = recognize_filled_cells(
                self.normalized_bgr,
                self.last_cell_analysis,
                OCR_OUTPUT_DIR,
            )

            export_ocr_results_csv(
                batch,
                OCR_RESULTS_CSV,
            )
        except Exception as error:
            messagebox.showerror(
                "Помилка OCR",
                str(error),
            )

            self.status_var.set(
                "Не вдалося розпізнати цифри."
            )
            return

        self.last_ocr_batch = batch

        self.status_var.set(
            f"OCR завершено: розпізнано "
            f"{batch.recognized_count}; "
            f"потребують перевірки "
            f"{batch.review_count}."
        )

        self.show_ocr_results_window(
            batch
        )

        messagebox.showinfo(
            "OCR завершено",
            f"Заповнених клітинок: "
            f"{len(batch.items)}\n"
            f"Розпізнано значень: "
            f"{batch.recognized_count}\n"
            f"Потребують перевірки: "
            f"{batch.review_count}\n"
            f"Без значення: "
            f"{batch.empty_count}\n\n"
            f"CSV:\n{OCR_RESULTS_CSV}",
        )


    def show_ocr_results_window(
        self,
        batch: DigitOcrBatchResult,
    ) -> None:
        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Результати OCR"
        )

        window.geometry(
            "760x560"
        )

        frame = ttk.Frame(
            window,
            padding=8,
        )

        frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        columns = (
            "row",
            "column",
            "day",
            "value",
            "confidence",
            "status",
        )

        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
        )

        headings = {
            "row": "Рядок",
            "column": "Колонка",
            "day": "День",
            "value": "Кількість",
            "confidence": "Впевненість",
            "status": "Статус",
        }

        widths = {
            "row": 70,
            "column": 80,
            "day": 80,
            "value": 100,
            "confidence": 110,
            "status": 130,
        }

        for column in columns:
            tree.heading(
                column,
                text=headings[column],
            )

            tree.column(
                column,
                width=widths[column],
                anchor=tk.CENTER,
            )

        scrollbar = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=tree.yview,
        )

        tree.configure(
            yscrollcommand=scrollbar.set
        )

        tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        tree.tag_configure(
            "review",
            background="#fce5cd",
        )

        self.results_tree = tree

        tree.bind(
            "<Double-1>",
            self.edit_ocr_result,
        )

        for item in batch.items:
            tree.insert(
                "",
                tk.END,
                values=(
                    item.row,
                    item.column,
                    item.day,
                    item.value,
                    f"{item.confidence:.0%}",
                    item.status,
                ),
                tags=(
                    ("review",)
                    if item.status
                    == "ПЕРЕВІРИТИ"
                    else ()
                ),
            )


    def save_normalized_page(self) -> None:
        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                "Спочатку знайдіть мітки "
                "та виконайте вирівнювання.",
            )
            return

        default_name = (
            f"normalized_page_"
            f"{self.page_index + 1:03d}.png"
        )

        target = filedialog.asksaveasfilename(
            title="Зберегти вирівняну сторінку",
            initialdir=str(DEBUG_DIR),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
            ],
        )

        if not target:
            return

        success = cv2.imwrite(
            target,
            self.normalized_bgr,
        )

        if not success:
            messagebox.showerror(
                "Помилка",
                "Не вдалося зберегти PNG.",
            )
            return

        messagebox.showinfo(
            "Сторінку збережено",
            target,
        )

    def show_bgr_image(
        self,
        canvas: tk.Canvas,
        image_bgr: np.ndarray,
        target: str,
    ) -> None:
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        preview = resize_for_preview(
            pil_image,
            PREVIEW_MAX_WIDTH,
            PREVIEW_MAX_HEIGHT,
        )

        photo = ImageTk.PhotoImage(
            preview
        )

        canvas.delete("all")

        canvas.update_idletasks()

        canvas_width = max(
            canvas.winfo_width(),
            PREVIEW_MAX_WIDTH,
        )

        canvas_height = max(
            canvas.winfo_height(),
            PREVIEW_MAX_HEIGHT,
        )

        x = canvas_width // 2
        y = canvas_height // 2

        canvas.create_image(
            x,
            y,
            image=photo,
            anchor=tk.CENTER,
        )

        if target == "normalized":
            self.normalized_preview_scale = (
                preview.width / image_bgr.shape[1]
            )

            self.normalized_preview_offset_x = (
                x - preview.width / 2
            )

            self.normalized_preview_offset_y = (
                y - preview.height / 2
            )

            self.draw_table_calibration()

        if target == "original":
            self.original_photo = photo
        else:
            self.normalized_photo = photo


def detect_aruco_markers(
    image_bgr: np.ndarray,
) -> dict[int, dict[str, object]]:
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTIONARY_ID
    )

    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters,
    )

    corners, ids, _rejected = (
        detector.detectMarkers(gray)
    )

    if ids is None or len(ids) == 0:
        raise RuntimeError(
            "На сторінці не знайдено ArUco-міток."
        )

    result: dict[int, dict[str, object]] = {}

    for marker_id_array, marker_corners in zip(
        ids,
        corners,
    ):
        marker_id = int(
            np.asarray(marker_id_array).reshape(-1)[0]
        )

        points = np.asarray(
            marker_corners
        ).reshape(4, 2)

        center_array = points.mean(
            axis=0
        )

        center = (
            int(round(center_array[0])),
            int(round(center_array[1])),
        )

        result[marker_id] = {
            "corners": points,
            "center": center,
        }

    return result


def euclidean_distance(
    first_point: np.ndarray,
    second_point: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first_point, dtype=np.float32)
            - np.asarray(second_point, dtype=np.float32)
        )
    )


def get_inner_marker_corner(
    detected: dict[int, dict[str, object]],
    marker_id: int,
) -> np.ndarray:
    """
    Повертає кут ArUco-мітки, спрямований усередину таблиці.

    Порядок кутів OpenCV:
    0 — верхній лівий;
    1 — верхній правий;
    2 — нижній правий;
    3 — нижній лівий.
    """
    points = np.asarray(
        detected[marker_id]["corners"],
        dtype=np.float32,
    ).reshape(4, 2)

    inner_corner_index = {
        0: 2,  # верхня ліва мітка → нижній правий кут
        1: 3,  # верхня права мітка → нижній лівий кут
        2: 0,  # нижня права мітка → верхній лівий кут
        3: 1,  # нижня ліва мітка → верхній правий кут
    }[marker_id]

    return points[inner_corner_index]


def load_saved_marker_aspect_ratio() -> float | None:
    """
    Після першого правильного калібрування використовуємо
    зафіксоване співвідношення сторін для всіх наступних сканів.
    """
    if not FORM_V2_CONFIG_PATH.exists():
        return None

    try:
        import json

        config = json.loads(
            FORM_V2_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        ratio = float(
            config.get(
                "marker_aspect_ratio",
                0,
            )
        )

        if 0.15 <= ratio <= 2.0:
            return ratio

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None

    return None


def normalize_page_by_markers(
    image_bgr: np.ndarray,
    detected: dict[int, dict[str, object]],
) -> np.ndarray:
    for marker_id in EXPECTED_MARKERS:
        if marker_id not in detected:
            raise ValueError(
                f"Немає мітки ID {marker_id}."
            )

    # Використовуємо внутрішні кути міток, а не їх центри.
    # Так робоча область визначається точніше.
    top_left = get_inner_marker_corner(
        detected,
        0,
    )
    top_right = get_inner_marker_corner(
        detected,
        1,
    )
    bottom_right = get_inner_marker_corner(
        detected,
        2,
    )
    bottom_left = get_inner_marker_corner(
        detected,
        3,
    )

    top_width = euclidean_distance(
        top_left,
        top_right,
    )
    bottom_width = euclidean_distance(
        bottom_left,
        bottom_right,
    )
    left_height = euclidean_distance(
        top_left,
        bottom_left,
    )
    right_height = euclidean_distance(
        top_right,
        bottom_right,
    )

    measured_width = (
        top_width + bottom_width
    ) / 2.0

    measured_height = (
        left_height + right_height
    ) / 2.0

    if measured_width < 100 or measured_height < 100:
        raise RuntimeError(
            "Відстань між ArUco-мітками занадто мала."
        )

    measured_ratio = (
        measured_width / measured_height
    )

    saved_ratio = (
        load_saved_marker_aspect_ratio()
    )

    aspect_ratio = (
        saved_ratio
        if saved_ratio is not None
        else measured_ratio
    )

    output_height = NORMALIZED_HEIGHT

    output_width = int(
        round(
            output_height
            * aspect_ratio
        )
    )

    output_width = max(
        MIN_NORMALIZED_WIDTH,
        min(
            MAX_NORMALIZED_WIDTH,
            output_width,
        ),
    )

    source_points = np.float32([
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ])

    destination_points = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [
            output_width - 1,
            output_height - 1,
        ],
        [0, output_height - 1],
    ])

    transform = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    normalized = cv2.warpPerspective(
        image_bgr,
        transform,
        (
            output_width,
            output_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return normalized


def resize_for_preview(
    image: Image.Image,
    max_width: int,
    max_height: int,
) -> Image.Image:
    width, height = image.size

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    new_width = max(
        1,
        round(width * scale),
    )

    new_height = max(
        1,
        round(height * scale),
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def main() -> None:
    root = tk.Tk()
    app = FerrettiOcrApp(root)

    def on_close() -> None:
        app.close_pdf()
        root.destroy()

    root.protocol(
        "WM_DELETE_WINDOW",
        on_close,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
