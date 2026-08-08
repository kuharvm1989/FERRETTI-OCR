from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from app.services.ocr_service import OCRService

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageTk

from app.ocr.cell_parser import CellAnalysisResult, analyze_cells
from app.ocr.ocr_engine import (
    DigitOcrBatchResult,
    export_ocr_results_csv,
    recognize_filled_cells,
)
from app.gui.controllers.table_calibration import (
    TableCalibrationController,
)
from app.gui.dialogs.ocr_results_dialog import (
    OcrResultsDialog,
)
from app.gui.controllers.pdf_controller import (
    PdfController,
)
from app.services.ocr_pipeline import (
    OCRPipeline,
)
from app.ocr.page_normalization import (
    EXPECTED_MARKERS,
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

        self.table_calibration = (
            TableCalibrationController(
                canvas=self.normalized_canvas,
                status_var=self.status_var,
            )
        )
        self.pdf_controller = PdfController(
            file_var=self.file_var,
            page_var=self.page_var,
            marker_var=self.marker_var,
            status_var=self.status_var,
            previous_button=self.previous_button,
            next_button=self.next_button,
            render_dpi=RENDER_DPI,
            show_original_callback=(
                lambda image: self.show_bgr_image(
                    self.original_canvas,
                    image,
                    target="original",
                )
            ),
            clear_normalized_callback=(
                lambda: self.normalized_canvas.delete(
                    "all"
                )
            ),
            reset_analysis_callback=(
                self.reset_page_analysis
            ),
        )
        self.ocr_pipeline = OCRPipeline()
        
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

    def reset_page_analysis(
        self,
    ) -> None:
        self.normalized_bgr = None
        self.last_cell_analysis = None
        self.last_ocr_batch = None

    def select_pdf(self) -> None:
        self.pdf_controller.select_pdf()
        self.sync_pdf_state()

    def close_pdf(self) -> None:
        self.pdf_controller.close_pdf()
        self.sync_pdf_state()

    def render_current_page(self) -> None:
        self.pdf_controller.render_current_page()
        self.sync_pdf_state()

    def previous_page(self) -> None:
        self.pdf_controller.previous_page()
        self.sync_pdf_state()

    def next_page(self) -> None:
        self.pdf_controller.next_page()
        self.sync_pdf_state()

    def update_navigation_buttons(
        self,
    ) -> None:
        self.pdf_controller.update_navigation_buttons()

    def sync_pdf_state(
        self,
    ) -> None:
        self.pdf_document = (
            self.pdf_controller.pdf_document
        )

        self.pdf_path = (
            self.pdf_controller.pdf_path
        )

        self.page_index = (
            self.pdf_controller.page_index
        )

        self.original_bgr = (
            self.pdf_controller.original_bgr
        )

    def detect_and_normalize(
        self,
    ) -> None:
        if self.original_bgr is None:
            messagebox.showwarning(
                "Немає сторінки",
                "Спочатку відкрийте PDF.",
            )
            return

        detected = self._detect_markers()

        if detected is None:
            return

        preview = self._build_marker_preview(
            detected
        )

        self.show_bgr_image(
            self.original_canvas,
            preview,
            target="original",
        )

        if not self._validate_markers(
            detected
        ):
            return

        normalized = self._normalize_page(
            detected
        )

        if normalized is None:
            return

        self._apply_normalized_page(
            preview,
            normalized,
        )

    def _detect_markers(
        self,
    ):
        try:
            return self.ocr_pipeline.detect_markers(
                self.original_bgr
            )

        except Exception as error:
            messagebox.showerror(
                "Помилка пошуку міток",
                str(error),
            )
            return None


    def _build_marker_preview(
        self,
        detected,
    ) -> np.ndarray:
        preview = self.original_bgr.copy()

        for marker_id, marker_data in (
            detected.items()
        ):
            corners = (
                marker_data["corners"]
                .astype(np.int32)
            )

            center = marker_data[
                "center"
            ]

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

        return preview


    def _validate_markers(
        self,
        detected,
    ) -> bool:
        missing_ids = (
            self.ocr_pipeline.validate_markers(
                detected,
                EXPECTED_MARKERS,
            )
        )

        detected_ids = set(
            detected
        )

        if not missing_ids:
            return True

        self.marker_var.set(
            (
                "Мітки: знайдено "
                f"{len(detected_ids)} із 4"
            )
        )

        message = (
            "Не знайдено мітки: "
            + ", ".join(
                str(value)
                for value in sorted(
                    missing_ids
                )
            )
        )

        messagebox.showerror(
            "Не всі мітки знайдено",
            message,
        )

        self.status_var.set(
            message
        )

        return False


    def _normalize_page(
        self,
        detected,
    ):
        try:
            return self.ocr_pipeline.normalize_page(
                self.original_bgr,
                detected,
            )

        except Exception as error:
            messagebox.showerror(
                "Помилка вирівнювання",
                str(error),
            )
            return None


    def _apply_normalized_page(
        self,
        preview: np.ndarray,
        normalized: np.ndarray,
    ) -> None:
        self.normalized_bgr = normalized
    
        self.marker_var.set(
            "Мітки: 4 із 4"
        )

        (
            normalized_height,
            normalized_width,
        ) = normalized.shape[:2]

        self.status_var.set(
            (
                "Сторінку вирівняно "
                "без зміни пропорцій: "
                f"{normalized_width} × "
                f"{normalized_height} px."
            )
        )

        self.show_bgr_image(
            self.normalized_canvas,
            normalized,
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

    def start_table_calibration(
        self,
    ) -> None:
        self.table_calibration.update_view_state(
            normalized_bgr=self.normalized_bgr,
            preview_scale=(
                self.normalized_preview_scale
            ),
            preview_offset_x=(
                self.normalized_preview_offset_x
            ),
            preview_offset_y=(
                self.normalized_preview_offset_y
            ),
        )

        self.table_calibration.start()


    def on_normalized_canvas_click(
        self,
        event: tk.Event,
    ) -> None:
        self.table_calibration.update_view_state(
            normalized_bgr=self.normalized_bgr,
            preview_scale=(
                self.normalized_preview_scale
            ),
            preview_offset_x=(
                self.normalized_preview_offset_x
            ),
            preview_offset_y=(
                self.normalized_preview_offset_y
            ),
        )

        self.table_calibration.on_canvas_click(
            event
        )


    def draw_table_calibration(
        self,
    ) -> None:
        self.table_calibration.update_view_state(
            normalized_bgr=self.normalized_bgr,
            preview_scale=(
                self.normalized_preview_scale
            ),
            preview_offset_x=(
                self.normalized_preview_offset_x
            ),
            preview_offset_y=(
                self.normalized_preview_offset_y
            ),
        )

        self.table_calibration.draw()

    def save_table_calibration(
        self,
    ) -> None:
        self.table_calibration.update_view_state(
            normalized_bgr=self.normalized_bgr,
            preview_scale=(
                self.normalized_preview_scale
            ),
            preview_offset_x=(
                self.normalized_preview_offset_x
            ),
            preview_offset_y=(
                self.normalized_preview_offset_y
            ),
        )

        self.table_calibration.save()

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
            analysis = self.ocr_pipeline.analyze_cells(
                self.normalized_bgr,
                form_config_path=FORM_V2_CONFIG_PATH,
                output_dir=CELL_OUTPUT_DIR,
                preview_path=CELL_PREVIEW_PATH,
                geometry_config_path=CELL_GEOMETRY_CONFIG_PATH,
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

        OcrResultsDialog(
            self.root,
            batch,
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

