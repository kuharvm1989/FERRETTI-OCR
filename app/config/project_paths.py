from __future__ import annotations

from pathlib import Path


# Корінь усього проєкту:
# C:\FERRETTI\ProductionOCR
PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# --------------------------------------------------
# ОСНОВНІ ПАПКИ
# --------------------------------------------------

APP_DIR = (
    PROJECT_DIR
    / "app"
)

CONFIG_DIR = (
    PROJECT_DIR
    / "config"
)

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
)

DEBUG_DIR = (
    PROJECT_DIR
    / "debug"
)

INPUT_DIR = (
    PROJECT_DIR
    / "input"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
)

MODELS_DIR = (
    PROJECT_DIR
    / "models"
)

MARKERS_DIR = (
    PROJECT_DIR
    / "markers"
)

TESTS_DIR = (
    PROJECT_DIR
    / "tests"
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

DATASET_CONFIRMED_DIR = (
    DATASET_DIR
    / "confirmed"
)

DATASET_PENDING_DIR = (
    DATASET_DIR
    / "pending"
)

DATASET_REJECTED_DIR = (
    DATASET_DIR
    / "rejected"
)

DATASET_MANIFEST_PATH = (
    DATASET_DIR
    / "dataset_manifest.csv"
)

MANUAL_CORRECTIONS_PATH = (
    DATASET_DIR
    / "manual_corrections.csv"
)

LEARNING_IMPORT_REPORT_PATH = (
    DATASET_DIR
    / "learning_import_report.csv"
)

LEARNING_STATE_PATH = (
    DATASET_DIR
    / "learning_state.json"
)


# --------------------------------------------------
# MODELS
# --------------------------------------------------

DIGIT_SVM_MODEL_PATH = (
    MODELS_DIR
    / "digit_svm.yml"
)

MODEL_BACKUPS_DIR = (
    MODELS_DIR
    / "backups"
)


# --------------------------------------------------
# OCR DEBUG
# --------------------------------------------------

OCR_DEBUG_DIR = (
    DEBUG_DIR
    / "ocr_engine_v2"
)

OCR_SOURCE_CELLS_DIR = (
    OCR_DEBUG_DIR
    / "source_cells"
)

OCR_SEGMENTS_DIR = (
    OCR_DEBUG_DIR
    / "segments"
)


# --------------------------------------------------
# CONFIG FILES
# --------------------------------------------------

FORM_V1_CONFIG_PATH = (
    CONFIG_DIR
    / "form_v1.json"
)

FORM_V2_CONFIG_PATH = (
    CONFIG_DIR
    / "form_v2.json"
)

CELL_GEOMETRY_CONFIG_PATH = (
    CONFIG_DIR
    / "cell_geometry.json"
)
