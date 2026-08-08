from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox

from app.config.project_paths import (
    FORM_V2_CONFIG_PATH,
)


class TableCalibrationController:
    def __init__(
        self,
        *,
        canvas: tk.Canvas,
        status_var: tk.StringVar,
    ) -> None:
        self.canvas = canvas
        self.status_var = status_var

        self.normalized_bgr = None

        self.points: list[
            tuple[int, int]
        ] = []

        self.active = False

        self.preview_scale = 1.0
        self.preview_offset_x = 0
        self.preview_offset_y = 0

    def update_view_state(
        self,
        *,
        normalized_bgr,
        preview_scale: float,
        preview_offset_x: int,
        preview_offset_y: int,
    ) -> None:
        self.normalized_bgr = (
            normalized_bgr
        )

        self.preview_scale = (
            preview_scale
        )

        self.preview_offset_x = (
            preview_offset_x
        )

        self.preview_offset_y = (
            preview_offset_y
        )

    def start(
        self,
    ) -> None:
        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                (
                    "Спочатку знайдіть мітки "
                    "та вирівняйте сторінку."
                ),
            )
            return

        self.points.clear()
        self.active = True

        self.status_var.set(
            (
                "Калібрування таблиці: "
                "клацніть верхній лівий кут "
                "першої клітинки ПН."
            )
        )

        self.draw()

    def on_canvas_click(
        self,
        event: tk.Event,
    ) -> None:
        if (
            not self.active
            or self.normalized_bgr
            is None
        ):
            return

        canvas_x = self.canvas.canvasx(
            event.x
        )

        canvas_y = self.canvas.canvasy(
            event.y
        )

        image_x = (
            canvas_x
            - self.preview_offset_x
        )

        image_y = (
            canvas_y
            - self.preview_offset_y
        )

        if self.preview_scale <= 0:
            return

        original_x = round(
            image_x
            / self.preview_scale
        )

        original_y = round(
            image_y
            / self.preview_scale
        )

        height, width = (
            self.normalized_bgr.shape[:2]
        )

        if not (
            0 <= original_x < width
            and 0 <= original_y < height
        ):
            return

        self.points.append(
            (
                original_x,
                original_y,
            )
        )

        if len(self.points) == 1:
            self.status_var.set(
                (
                    "Перша точка збережена. "
                    "Клацніть нижній правий кут "
                    "останньої клітинки СБ."
                )
            )

        elif len(self.points) >= 2:
            self.points = (
                self.points[:2]
            )

            self.active = False

            first = self.points[0]
            second = self.points[1]

            self.status_var.set(
                (
                    "Калібрування готове: "
                    f"({first[0]}, {first[1]}) → "
                    f"({second[0]}, {second[1]})."
                )
            )

        self.draw()

    def draw(
        self,
    ) -> None:
        self.canvas.delete(
            "table_calibration"
        )

        if not self.points:
            return

        display_points = []

        for index, point in enumerate(
            self.points,
            start=1,
        ):
            display_x = (
                point[0]
                * self.preview_scale
                + self.preview_offset_x
            )

            display_y = (
                point[1]
                * self.preview_scale
                + self.preview_offset_y
            )

            display_points.append(
                (
                    display_x,
                    display_y,
                )
            )

            radius = 6

            self.canvas.create_oval(
                display_x - radius,
                display_y - radius,
                display_x + radius,
                display_y + radius,
                outline="red",
                fill="yellow",
                width=2,
                tags="table_calibration",
            )

            self.canvas.create_text(
                display_x + 12,
                display_y - 12,
                text=str(index),
                fill="red",
                font=(
                    "Arial",
                    12,
                    "bold",
                ),
                tags="table_calibration",
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
                tags="table_calibration",
            )

    def save(
        self,
    ) -> None:
        if len(self.points) != 2:
            messagebox.showwarning(
                "Калібрування не завершено",
                (
                    "Потрібно встановити "
                    "дві точки."
                ),
            )
            return

        if self.normalized_bgr is None:
            messagebox.showwarning(
                "Немає вирівняної сторінки",
                (
                    "Спочатку знайдіть мітки "
                    "та вирівняйте сторінку."
                ),
            )
            return

        first = self.points[0]
        second = self.points[1]

        x1 = min(
            first[0],
            second[0],
        )

        y1 = min(
            first[1],
            second[1],
        )

        x2 = max(
            first[0],
            second[0],
        )

        y2 = max(
            first[1],
            second[1],
        )

        (
            normalized_height,
            normalized_width,
        ) = self.normalized_bgr.shape[:2]

        config = {
            "template_version":
                "FORM-V2",
            "normalized_width":
                normalized_width,
            "normalized_height":
                normalized_height,
            "marker_aspect_ratio":
                (
                    normalized_width
                    / normalized_height
                ),
            "table_grid": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "day_columns": 6,
        }

        FORM_V2_CONFIG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        FORM_V2_CONFIG_PATH.write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.status_var.set(
            "Калібрування таблиці збережено."
        )
