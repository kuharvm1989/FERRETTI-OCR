from __future__ import annotations


class OCRPipeline:
    def __init__(
        self,
        app,
    ) -> None:
        self.app = app

    def detect_and_normalize(
        self,
    ) -> None:
        self.app.detect_and_normalize()

    def detect_filled_cells(
        self,
    ) -> None:
        self.app.detect_filled_cells()

    def recognize_digits(
        self,
    ) -> None:
        self.app.recognize_digits()
