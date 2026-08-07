from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "debug" / "page_001_300dpi.png"
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "form_v1.json"

WINDOW_TITLE = "FERRETTI Production OCR"
CANVAS_WIDTH = 1050
CANVAS_HEIGHT = 760


class ProductionOcrApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1200x900")
        self.root.minsize(900, 650)

        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.current_image_path: Path | None = None

        self.scale = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0

        # Координати в оригінальному зображенні.
        self.calibration_points: list[tuple[int, int]] = []

        self.status_var = tk.StringVar(
            value="Відкрийте PNG сторінки для калібрування."
        )

        self.build_interface()

        if DEFAULT_IMAGE.exists():
            self.load_image(DEFAULT_IMAGE)

    def build_interface(self) -> None:
        toolbar = ttk.Frame(
            self.root,
            padding=8,
        )
        toolbar.pack(
            side=tk.TOP,
            fill=tk.X,
        )

        ttk.Button(
            toolbar,
            text="Відкрити PNG",
            command=self.select_image,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Почати калібрування",
            command=self.start_calibration,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Скасувати точки",
            command=self.clear_calibration,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Зберегти FORM-V1",
            command=self.save_calibration,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(
            toolbar,
            text="Завантажити FORM-V1",
            command=self.load_calibration,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )

        instruction_frame = ttk.LabelFrame(
            self.root,
            text="Калібрування",
            padding=8,
        )
        instruction_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=8,
            pady=(0, 8),
        )

        instruction_text = (
            "Після натискання «Почати калібрування» поставте дві точки:\n"
            "1 — верхній лівий кут першої комірки ПН для першої операції;\n"
            "2 — нижній правий кут останньої комірки СБ для останньої операції."
        )

        ttk.Label(
            instruction_frame,
            text=instruction_text,
            justify=tk.LEFT,
        ).pack(
            anchor=tk.W,
        )

        canvas_frame = ttk.Frame(
            self.root,
            padding=8,
        )
        canvas_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            background="#d9d9d9",
            highlightthickness=1,
            highlightbackground="#888888",
            cursor="crosshair",
        )

        horizontal_scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient=tk.HORIZONTAL,
            command=self.canvas.xview,
        )

        vertical_scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient=tk.VERTICAL,
            command=self.canvas.yview,
        )

        self.canvas.configure(
            xscrollcommand=horizontal_scrollbar.set,
            yscrollcommand=vertical_scrollbar.set,
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vertical_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        horizontal_scrollbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        canvas_frame.rowconfigure(
            0,
            weight=1,
        )

        canvas_frame.columnconfigure(
            0,
            weight=1,
        )

        self.canvas.bind(
            "<Button-1>",
            self.on_canvas_click,
        )

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=5,
        )
        status_bar.pack(
            side=tk.BOTTOM,
            fill=tk.X,
        )

        self.calibration_active = False

    def select_image(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Виберіть PNG сторінки",
            initialdir=str(BASE_DIR / "debug"),
            filetypes=[
                ("PNG images", "*.png"),
                ("JPEG images", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if not selected_file:
            return

        self.load_image(
            Path(selected_file)
        )

    def load_image(self, image_path: Path) -> None:
        try:
            image = Image.open(image_path)
            image.load()
        except Exception as error:
            messagebox.showerror(
                "Помилка",
                f"Не вдалося відкрити зображення:\n{error}",
            )
            return

        self.original_image = image.convert("RGB")
        self.current_image_path = image_path
        self.calibration_points.clear()
        self.calibration_active = False

        self.render_image()

        width, height = self.original_image.size

        self.status_var.set(
            f"Відкрито: {image_path.name} | "
            f"Оригінальний розмір: {width} × {height} px"
        )

    def render_image(self) -> None:
        if self.original_image is None:
            return

        self.root.update_idletasks()

        available_width = max(
            self.canvas.winfo_width() - 20,
            200,
        )
        available_height = max(
            self.canvas.winfo_height() - 20,
            200,
        )

        original_width, original_height = (
            self.original_image.size
        )

        self.scale = min(
            available_width / original_width,
            available_height / original_height,
            1.0,
        )

        display_width = max(
            1,
            round(original_width * self.scale),
        )
        display_height = max(
            1,
            round(original_height * self.scale),
        )

        self.display_image = self.original_image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )

        self.photo_image = ImageTk.PhotoImage(
            self.display_image
        )

        self.canvas.delete("all")

        self.image_offset_x = 10
        self.image_offset_y = 10

        self.canvas.create_image(
            self.image_offset_x,
            self.image_offset_y,
            anchor=tk.NW,
            image=self.photo_image,
            tags="page_image",
        )

        self.canvas.configure(
            scrollregion=(
                0,
                0,
                display_width + 20,
                display_height + 20,
            )
        )

        self.draw_calibration()

    def start_calibration(self) -> None:
        if self.original_image is None:
            messagebox.showwarning(
                "Немає зображення",
                "Спочатку відкрийте PNG сторінки.",
            )
            return

        self.calibration_points.clear()
        self.calibration_active = True
        self.draw_calibration()

        self.status_var.set(
            "Калібрування: клацніть верхній лівий кут "
            "першої комірки ПН."
        )

    def clear_calibration(self) -> None:
        self.calibration_points.clear()
        self.calibration_active = False
        self.draw_calibration()

        self.status_var.set(
            "Точки калібрування очищено."
        )

    def on_canvas_click(
        self,
        event: tk.Event,
    ) -> None:
        if (
            not self.calibration_active
            or self.original_image is None
        ):
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        image_x = canvas_x - self.image_offset_x
        image_y = canvas_y - self.image_offset_y

        display_width = round(
            self.original_image.width * self.scale
        )
        display_height = round(
            self.original_image.height * self.scale
        )

        if not (
            0 <= image_x <= display_width
            and 0 <= image_y <= display_height
        ):
            return

        original_x = round(
            image_x / self.scale
        )
        original_y = round(
            image_y / self.scale
        )

        self.calibration_points.append(
            (original_x, original_y)
        )

        if len(self.calibration_points) == 1:
            self.status_var.set(
                "Перша точка збережена. "
                "Тепер клацніть нижній правий кут "
                "останньої комірки СБ."
            )

        elif len(self.calibration_points) >= 2:
            self.calibration_points = (
                self.calibration_points[:2]
            )
            self.calibration_active = False

            first_point = self.calibration_points[0]
            second_point = self.calibration_points[1]

            self.status_var.set(
                "Калібрування завершено: "
                f"({first_point[0]}, {first_point[1]}) → "
                f"({second_point[0]}, {second_point[1]}). "
                "Натисніть «Зберегти FORM-V1»."
            )

        self.draw_calibration()

    def draw_calibration(self) -> None:
        self.canvas.delete("calibration")

        if not self.calibration_points:
            return

        display_points: list[tuple[float, float]] = []

        for index, point in enumerate(
            self.calibration_points,
            start=1,
        ):
            display_x = (
                point[0] * self.scale
                + self.image_offset_x
            )
            display_y = (
                point[1] * self.scale
                + self.image_offset_y
            )

            display_points.append(
                (display_x, display_y)
            )

            radius = 7

            self.canvas.create_oval(
                display_x - radius,
                display_y - radius,
                display_x + radius,
                display_y + radius,
                outline="red",
                fill="yellow",
                width=3,
                tags="calibration",
            )

            self.canvas.create_text(
                display_x + 12,
                display_y - 12,
                text=str(index),
                fill="red",
                font=("Arial", 14, "bold"),
                tags="calibration",
            )

        if len(display_points) == 2:
            first = display_points[0]
            second = display_points[1]

            self.canvas.create_rectangle(
                first[0],
                first[1],
                second[0],
                second[1],
                outline="red",
                width=3,
                tags="calibration",
            )

    def save_calibration(self) -> None:
        if self.original_image is None:
            messagebox.showwarning(
                "Немає зображення",
                "Спочатку відкрийте PNG.",
            )
            return

        if len(self.calibration_points) != 2:
            messagebox.showwarning(
                "Калібрування не завершене",
                "Потрібно встановити дві точки.",
            )
            return

        first_point = self.calibration_points[0]
        second_point = self.calibration_points[1]

        x1 = min(
            first_point[0],
            second_point[0],
        )
        y1 = min(
            first_point[1],
            second_point[1],
        )
        x2 = max(
            first_point[0],
            second_point[0],
        )
        y2 = max(
            first_point[1],
            second_point[1],
        )

        config = {
            "template_version": "FORM-V1",
            "source_dpi": 300,
            "source_width": self.original_image.width,
            "source_height": self.original_image.height,
            "data_grid": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "day_columns": 6,
            "description": (
                "Робоча область клітинок ПН–СБ "
                "від першої до останньої операції."
            ),
        }

        CONFIG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            CONFIG_PATH.write_text(
                json.dumps(
                    config,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as error:
            messagebox.showerror(
                "Помилка збереження",
                str(error),
            )
            return

        messagebox.showinfo(
            "Калібрування збережено",
            f"Файл:\n{CONFIG_PATH}",
        )

        self.status_var.set(
            f"FORM-V1 збережено: {CONFIG_PATH.name}"
        )

    def load_calibration(self) -> None:
        if not CONFIG_PATH.exists():
            messagebox.showwarning(
                "Конфігурацію не знайдено",
                f"Файл відсутній:\n{CONFIG_PATH}",
            )
            return

        if self.original_image is None:
            messagebox.showwarning(
                "Немає зображення",
                "Спочатку відкрийте PNG сторінки.",
            )
            return

        try:
            config = json.loads(
                CONFIG_PATH.read_text(
                    encoding="utf-8"
                )
            )

            grid = config["data_grid"]

            self.calibration_points = [
                (
                    int(grid["x1"]),
                    int(grid["y1"]),
                ),
                (
                    int(grid["x2"]),
                    int(grid["y2"]),
                ),
            ]

        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            messagebox.showerror(
                "Помилка конфігурації",
                str(error),
            )
            return

        self.calibration_active = False
        self.draw_calibration()

        self.status_var.set(
            "Калібрування FORM-V1 завантажено."
        )


def main() -> None:
    root = tk.Tk()
    ProductionOcrApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
