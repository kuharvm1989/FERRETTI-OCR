from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_DIR
    / "dataset"
    / "dataset_manifest.csv"
)

STATE_PATH = (
    PROJECT_DIR
    / "dataset"
    / "learning_state.json"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "digit_svm.yml"
)

BACKUP_DIR = (
    PROJECT_DIR
    / "models"
    / "backups"
)

TRAIN_SCRIPT = (
    PROJECT_DIR
    / "training"
    / "train_digit_svm.py"
)

REGRESSION_SCRIPT = (
    PROJECT_DIR
    / "tests"
    / "run_regression.py"
)


RETRAIN_THRESHOLD = 25


def count_manual_samples() -> int:
    if not MANIFEST_PATH.exists():
        return 0

    count = 0

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            sample_id = (
                row.get(
                    "id",
                    ""
                )
                .strip()
            )

            status = (
                row.get(
                    "status",
                    ""
                )
                .strip()
            )

            if (
                sample_id.startswith(
                    "manual_"
                )
                and status == "CONFIRMED"
            ):
                count += 1

    return count


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "manual_count_at_last_retrain": 0,
            "last_retrain_at": "",
        }

    try:
        return json.loads(
            STATE_PATH.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return {
            "manual_count_at_last_retrain": 0,
            "last_retrain_at": "",
        }


def save_state(
    manual_count: int,
) -> None:
    state = {
        "manual_count_at_last_retrain":
            manual_count,
        "last_retrain_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    STATE_PATH.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_backup() -> Path | None:
    if not MODEL_PATH.exists():
        return None

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_DIR
        / f"digit_svm_{timestamp}.yml"
    )

    shutil.copy2(
        MODEL_PATH,
        backup_path,
    )

    return backup_path


def run_script(
    script_path: Path,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(
            PROJECT_DIR
        ),
        text=True,
        capture_output=True,
    )


def restore_backup(
    backup_path: Path | None,
) -> None:
    if (
        backup_path is None
        or not backup_path.exists()
    ):
        return

    shutil.copy2(
        backup_path,
        MODEL_PATH,
    )


def main() -> None:
    print()
    print("=" * 64)
    print(
        "FERRETTI OCR — SAFE RETRAIN"
    )
    print("=" * 64)

    manual_count = (
        count_manual_samples()
    )

    state = load_state()

    previous_count = int(
        state.get(
            "manual_count_at_last_retrain",
            0,
        )
    )

    new_samples = max(
        0,
        manual_count
        - previous_count,
    )

    print()
    print(
        f"Ручних цифр у dataset:     "
        f"{manual_count}"
    )

    print(
        f"Було при останньому train: "
        f"{previous_count}"
    )

    print(
        f"Нових після train:         "
        f"{new_samples}"
    )

    print(
        f"Поріг перенавчання:        "
        f"{RETRAIN_THRESHOLD}"
    )

    if new_samples < RETRAIN_THRESHOLD:
        remaining = (
            RETRAIN_THRESHOLD
            - new_samples
        )

        print()
        print(
            "Перенавчання поки "
            "не потрібне."
        )

        print(
            f"Потрібно ще приблизно "
            f"{remaining} нових цифр."
        )

        print()
        print("=" * 64)
        return

    print()
    print(
        "Поріг досягнуто."
    )

    print(
        "Створюємо backup моделі..."
    )

    backup_path = create_backup()

    if backup_path is not None:
        print(
            backup_path
        )
    else:
        print(
            "Попередньої моделі немає."
        )

    print()
    print(
        "Навчаємо нову модель..."
    )

    training = run_script(
        TRAIN_SCRIPT
    )

    print(
        training.stdout
    )

    if training.returncode != 0:
        print(
            training.stderr
        )

        print()
        print(
            "ПОМИЛКА НАВЧАННЯ."
        )

        restore_backup(
            backup_path
        )

        print(
            "Стару модель відновлено."
        )

        return

    if not MODEL_PATH.exists():
        print(
            "Нова модель не створена."
        )

        restore_backup(
            backup_path
        )

        return

    print()
    print(
        "Запускаємо regression..."
    )

    regression = run_script(
        REGRESSION_SCRIPT
    )

    print(
        regression.stdout
    )

    regression_passed = (
        "REGRESSION: PASS"
        in regression.stdout
    )

    if not regression_passed:
        print()
        print(
            "REGRESSION FAILED."
        )

        print(
            "Нова модель відхилена."
        )

        restore_backup(
            backup_path
        )

        print(
            "Попередню модель "
            "автоматично відновлено."
        )

        return

    save_state(
        manual_count
    )

    print()
    print("=" * 64)

    print(
        "НОВА МОДЕЛЬ ПРИЙНЯТА."
    )

    print(
        "Regression: PASS"
    )

    print(
        f"Зафіксовано ручних цифр: "
        f"{manual_count}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()
