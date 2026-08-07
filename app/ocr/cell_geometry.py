from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def clamp(
        self,
        image_width: int,
        image_height: int,
    ) -> "Rect":
        x1 = max(0, min(self.x1, image_width - 1))
        y1 = max(0, min(self.y1, image_height - 1))
        x2 = max(x1 + 1, min(self.x2, image_width))
        y2 = max(y1 + 1, min(self.y2, image_height))

        return Rect(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )


@dataclass(frozen=True)
class GeometrySettings:
    detection_margin_left: int = 6
    detection_margin_right: int = 6
    detection_margin_top: int = 5
    detection_margin_bottom: int = 5

    ocr_margin_left: int = 8
    ocr_margin_right: int = 8
    ocr_margin_top: int = 7
    ocr_margin_bottom: int = 7

    debug_draw_ocr_zone: bool = True


def load_geometry_settings(
    config_path: Path,
) -> GeometrySettings:
    defaults = GeometrySettings()

    if not config_path.exists():
        return defaults

    try:
        data = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"Не вдалося прочитати geometry config: {error}"
        ) from error

    return GeometrySettings(
        detection_margin_left=int(
            data.get(
                "detection_margin_left",
                defaults.detection_margin_left,
            )
        ),
        detection_margin_right=int(
            data.get(
                "detection_margin_right",
                defaults.detection_margin_right,
            )
        ),
        detection_margin_top=int(
            data.get(
                "detection_margin_top",
                defaults.detection_margin_top,
            )
        ),
        detection_margin_bottom=int(
            data.get(
                "detection_margin_bottom",
                defaults.detection_margin_bottom,
            )
        ),
        ocr_margin_left=int(
            data.get(
                "ocr_margin_left",
                defaults.ocr_margin_left,
            )
        ),
        ocr_margin_right=int(
            data.get(
                "ocr_margin_right",
                defaults.ocr_margin_right,
            )
        ),
        ocr_margin_top=int(
            data.get(
                "ocr_margin_top",
                defaults.ocr_margin_top,
            )
        ),
        ocr_margin_bottom=int(
            data.get(
                "ocr_margin_bottom",
                defaults.ocr_margin_bottom,
            )
        ),
        debug_draw_ocr_zone=bool(
            data.get(
                "debug_draw_ocr_zone",
                defaults.debug_draw_ocr_zone,
            )
        ),
    )


class CellGeometry:
    def __init__(
        self,
        settings: GeometrySettings,
    ) -> None:
        self.settings = settings

    @staticmethod
    def _inset(
        rect: Rect,
        *,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> Rect:
        result = Rect(
            x1=rect.x1 + max(0, left),
            y1=rect.y1 + max(0, top),
            x2=rect.x2 - max(0, right),
            y2=rect.y2 - max(0, bottom),
        )

        if result.width <= 0 or result.height <= 0:
            raise ValueError(
                "Відступи геометрії більші за розмір клітинки."
            )

        return result

    def detection_rect(
        self,
        cell_rect: Rect,
    ) -> Rect:
        return self._inset(
            cell_rect,
            left=self.settings.detection_margin_left,
            right=self.settings.detection_margin_right,
            top=self.settings.detection_margin_top,
            bottom=self.settings.detection_margin_bottom,
        )

    def ocr_rect(
        self,
        cell_rect: Rect,
    ) -> Rect:
        return self._inset(
            cell_rect,
            left=self.settings.ocr_margin_left,
            right=self.settings.ocr_margin_right,
            top=self.settings.ocr_margin_top,
            bottom=self.settings.ocr_margin_bottom,
        )
