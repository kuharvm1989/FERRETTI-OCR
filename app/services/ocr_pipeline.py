from __future__ import annotations

from app.ocr.page_normalization import (
    detect_aruco_markers,
    normalize_page_by_markers,
)
from app.services.ocr_service import (
    OCRService,
)


class OCRPipeline:
    def detect_markers(
        self,
        image_bgr,
    ):
        return detect_aruco_markers(
            image_bgr
        )

    def validate_markers(
        self,
        detected,
        expected_markers,
    ) -> set[int]:
        detected_ids = set(
            detected
        )

        return (
            expected_markers
            - detected_ids
        )

    def normalize_page(
        self,
        image_bgr,
        detected,
    ):
        return normalize_page_by_markers(
            image_bgr,
            detected,
        )

    def analyze_cells(
        self,
        normalized_bgr,
        *,
        form_config_path,
        output_dir,
        preview_path,
        geometry_config_path,
    ):
        return OCRService.analyze_page(
            normalized_bgr,
            form_config_path,
            output_dir,
            preview_path,
            geometry_config_path,
            save_all_cells=False,
        )
