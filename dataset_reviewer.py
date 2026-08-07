from __future__ import annotations

import csv
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk
from dataset_manifest import (
    confirm_sample,
    reject_sample,
)


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
PENDING_DIR = DATASET_DIR / "pending"
CONFIRMED_DIR = DATASET_DIR / "confirmed"
REJECTED_DIR = DATASET_DIR / "rejected"

MANIFEST_PATH = (
    DATASET_DIR
    / "pending_manifest.csv"
)


class DatasetReviewer:
    def __init__(
        self,
        root: tk.Tk,
    ) -> None:
        self.root = root

        self.root.title(
            "FERRETTI OCR — Dataset Reviewer"
        )

        self.root.geometry(
            "520x620"
        )

        self.root.resizable(
            False,
            False,
        )

        self.image_files: list[Path] = []
        self.current_index = 0
        self.current_photo = None

        self.status_var = tk.StringVar(
            value=""
        )

        self.build_interface()

        self.root.bind(
            "<Key>",
            self.on_key_press,
        )

        self.reload_files()


    def build_interface(
        self,
    ) -> None:
        title = tk.Label(
            self.root,
            text="ПІДТВЕРДЖЕННЯ ЦИФР",
            font=(
                "Arial",
                18,
                "bold",
            ),
        )

        title.pack(
            pady=12
        )

        self.counter_label = tk.Label(
            self.root,
            text="",
            font=(
                "Arial",
                11,
            ),
        )

        self.counter_label.pack()

        self.image_label = tk.Label(
            self.root,
            bg="white",
            width=420,
            height=360,
        )

        self.image_label.pack(
            padx=20,
            pady=15,
        )

        instruction = tk.Label(
            self.root,
            text=(
                "Натисніть 0–9 — підтвердити цифру\n"
                "R — відхилити\n"
                "S — пропустити"
            ),
            font=(
                "Arial",
                13,
            ),
            justify="center",
        )

        instruction.pack(
            pady=10
        )

        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=(
                "Arial",
                10,
            ),
        )

        status_label.pack(
            pady=8
        )


    def reload_files(
        self,
    ) -> None:
        PENDING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.image_files = sorted(
            PENDING_DIR.glob("*.png")
        )

        self.current_index = 0

        if not self.image_files:
            self.show_empty_state()
            return

        self.show_current_image()


    def show_empty_state(
        self,
    ) -> None:
        self.image_label.configure(
            image="",
            text="Немає цифр для перевірки",
            font=(
                "Arial",
                16,
            ),
        )

        self.counter_label.configure(
            text="0 / 0"
        )

        self.status_var.set(
            "dataset\\pending порожня"
        )


    def show_current_image(
        self,
    ) -> None:
        if not self.image_files:
            self.show_empty_state()
            return

        if self.current_index >= len(
            self.image_files
        ):
            self.reload_files()
            return

        image_path = (
            self.image_files[
                self.current_index
            ]
        )

        image = Image.open(
            image_path
        ).convert(
            "L"
        )

        image.thumbnail(
            (
                380,
                320,
            ),
            Image.Resampling.NEAREST,
        )

        canvas = Image.new(
            "L",
            (
                420,
                360,
            ),
            color=255,
        )

        x = (
            canvas.width
            - image.width
        ) // 2

        y = (
            canvas.height
            - image.height
        ) // 2

        canvas.paste(
            image,
            (
                x,
                y,
            ),
        )

        self.current_photo = (
            ImageTk.PhotoImage(
                canvas
            )
        )

        self.image_label.configure(
            image=self.current_photo,
            text="",
        )

        self.counter_label.configure(
            text=(
                f"{self.current_index + 1} / "
                f"{len(self.image_files)}"
            )
        )

        self.status_var.set(
            image_path.name
        )


    def on_key_press(
        self,
        event: tk.Event,
    ) -> None:
        key = str(
            event.char
        ).lower()

        if key in "0123456789":
            self.confirm_digit(
                key
            )
            return

        if key == "r":
            self.reject_current()
            return

        if key == "s":
            self.skip_current()
            return


    def current_file(
        self,
    ) -> Path | None:
        if not self.image_files:
            return None

        if self.current_index >= len(
            self.image_files
        ):
            return None

        return self.image_files[
            self.current_index
        ]


    def confirm_digit(
        self,
        digit: str,
    ) -> None:
        source_path = (
            self.current_file()
        )

        if source_path is None:
            return

        destination_dir = (
            CONFIRMED_DIR
            / digit
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_path = (
            destination_dir
            / source_path.name
        )

        shutil.move(
            str(source_path),
            str(destination_path),
        )

        relative_path = (
            destination_path
            .relative_to(BASE_DIR)
            .as_posix()
        )

        confirm_sample(
            source_path.stem.split("_")[-1],
            digit,
            image_path=relative_path,
        )

        self.update_manifest(
            source_path.name,
            status="CONFIRMED",
            confirmed_digit=digit,
        )

        self.after_action(
            f"Підтверджено: {digit}"
        )


    def reject_current(
        self,
    ) -> None:
        source_path = (
            self.current_file()
        )

        if source_path is None:
            return

        REJECTED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_path = (
            REJECTED_DIR
            / source_path.name
        )

        shutil.move(
            str(source_path),
            str(destination_path),
        )

        relative_path = (
            destination_path
            .relative_to(BASE_DIR)
            .as_posix()
        )

        reject_sample(
            source_path.stem.split("_")[-1],
            image_path=relative_path,
        )

        self.update_manifest(
            source_path.name,
            status="REJECTED",
            confirmed_digit="",
        )

        self.after_action(
            "Відхилено"
        )


    def skip_current(
        self,
    ) -> None:
        if not self.image_files:
            return

        self.current_index += 1

        if self.current_index >= len(
            self.image_files
        ):
            self.current_index = 0

        self.show_current_image()


    def after_action(
        self,
        message: str,
    ) -> None:
        self.status_var.set(
            message
        )

        self.image_files = sorted(
            PENDING_DIR.glob("*.png")
        )

        if not self.image_files:
            self.show_empty_state()

            messagebox.showinfo(
                "Готово",
                "Усі цифри перевірені.",
            )

            return

        if self.current_index >= len(
            self.image_files
        ):
            self.current_index = 0

        self.show_current_image()


    def update_manifest(
        self,
        file_name: str,
        *,
        status: str,
        confirmed_digit: str,
    ) -> None:
        if not MANIFEST_PATH.exists():
            return

        with MANIFEST_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file,
                delimiter=";",
            )

            fieldnames = (
                reader.fieldnames
            )

            rows = list(
                reader
            )

        if not fieldnames:
            return

        for row in rows:
            if (
                row.get(
                    "file_name"
                )
                == file_name
            ):
                row[
                    "status"
                ] = status

                row[
                    "confirmed_digit"
                ] = confirmed_digit

        with MANIFEST_PATH.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                delimiter=";",
            )

            writer.writeheader()
            writer.writerows(
                rows
            )


def main() -> None:
    root = tk.Tk()

    DatasetReviewer(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
