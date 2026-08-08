from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pymupdf

from app.config.project_paths import INPUT_DIR


class PdfController:
    def __init__(
        self,
        *,
        file_var: tk.StringVar,
        page_var: tk.StringVar,
        marker_var: tk.StringVar,
        status_var: tk.StringVar,
        previous_button,
        next_button,
        render_dpi: int,
        show_original_callback,
        clear_normalized_callback,
        reset_analysis_callback,
    ) -> None:
        self.file_var = file_var
        self.page_var = page_var
        self.marker_var = marker_var
        self.status_var = status_var

        self.previous_button = previous_button
        self.next_button = next_button

        self.render_dpi = render_dpi

        self.show_original_callback = (
            show_original_callback
        )

        self.clear_normalized_callback = (
            clear_normalized_callback
        )

        self.reset_analysis_callback = (
            reset_analysis_callback
        )

        self.pdf_path: Path | None = None
        self.pdf_document: pymupdf.Document | None = None
        self.page_index = 0
        self.original_bgr: np.ndarray | None = None

    def select_pdf(
        self,
    ) -> None:
        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected = filedialog.askopenfilename(
            title=(
                "Виберіть PDF із листами "
                "виробітку"
            ),
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
            document = pymupdf.open(
                selected
            )

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
                (
                    "Документ не містить "
                    "сторінок."
                ),
            )
            return

        self.pdf_document = document
        self.pdf_path = Path(selected)
        self.page_index = 0

        self.file_var.set(
            f"Файл: {self.pdf_path.name}"
        )

        self.render_current_page()

    def close_pdf(
        self,
    ) -> None:
        if self.pdf_document is not None:
            self.pdf_document.close()

        self.pdf_document = None
        self.pdf_path = None
        self.original_bgr = None

    def render_current_page(
        self,
    ) -> None:
        if self.pdf_document is None:
            return

        try:
            page = self.pdf_document.load_page(
                self.page_index
            )

            pixmap = page.get_pixmap(
                dpi=self.render_dpi,
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

        self.reset_analysis_callback()

        self.page_var.set(
            (
                f"Сторінка: "
                f"{self.page_index + 1} "
                f"із "
                f"{self.pdf_document.page_count}"
            )
        )

        self.marker_var.set(
            "Мітки: не перевірено"
        )

        self.status_var.set(
            (
                "Сторінку завантажено. "
                "Натисніть «Знайти мітки "
                "та вирівняти»."
            )
        )

        self.show_original_callback(
            image_bgr
        )

        self.clear_normalized_callback()

        self.update_navigation_buttons()

    def previous_page(
        self,
    ) -> None:
        if (
            self.pdf_document is None
            or self.page_index <= 0
        ):
            return

        self.page_index -= 1
        self.render_current_page()

    def next_page(
        self,
    ) -> None:
        if self.pdf_document is None:
            return

        if (
            self.page_index
            >= self.pdf_document.page_count - 1
        ):
            return

        self.page_index += 1
        self.render_current_page()

    def update_navigation_buttons(
        self,
    ) -> None:
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
                if self.page_index
                < self.pdf_document.page_count - 1
                else tk.DISABLED
            )
        )
