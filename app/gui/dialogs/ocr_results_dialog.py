from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from app.config.project_paths import DEBUG_DIR
from app.dataset.manual_corrections import (
    save_manual_correction,
)


class OcrResultsDialog:
    def __init__(
        self,
        parent: tk.Misc,
        batch,
    ) -> None:
        self.parent = parent
        self.batch = batch
        self.results_tree: ttk.Treeview | None = None

        self.window = tk.Toplevel(
            parent
        )

        self.window.title(
            "Результати OCR"
        )

        self.window.geometry(
            "760x560"
        )

        self.build_interface()

    def build_interface(
        self,
    ) -> None:
        frame = ttk.Frame(
            self.window,
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

        for item in self.batch.items:
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

    def edit_ocr_result(
        self,
        event=None,
    ) -> None:
        if self.results_tree is None:
            return

        selection = (
            self.results_tree.selection()
        )

        if not selection:
            return

        item_id = selection[0]

        values = (
            self.results_tree.item(
                item_id,
                "values",
            )
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
            self.window
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
            self.window
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
                new_values[5] = (
                    "ПІДТВЕРДЖЕНО"
                )

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
            lambda _event: (
                save_value()
            ),
        )
