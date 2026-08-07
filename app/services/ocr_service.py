from __future__ import annotations

from app.ocr.cell_parser import analyze_cells


class OCRService:
    """Фасад для OCR-операцій, які не повинні жити в GUI."""

    @staticmethod
    def analyze_page(
        normalized_bgr,
        form_config_path,
        output_dir,
        preview_path,
        geometry_config_path,
        *,
        save_all_cells=False,
    ):
        return analyze_cells(
            normalized_bgr,
            form_config_path,
            output_dir,
            preview_path,
            geometry_config_path,
            save_all_cells=save_all_cells,
        )
