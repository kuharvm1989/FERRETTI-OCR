from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "confirmed"
)

MODEL_DIR = (
    PROJECT_DIR
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "digit_svm.yml"
)

IMAGE_SIZE = 64
RANDOM_SEED = 42
TEST_RATIO = 0.20


def prepare_image(
    image: np.ndarray,
) -> np.ndarray:
    if image is None:
        raise ValueError(
            "Не вдалося прочитати зображення."
        )

    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    # Чорна цифра на білому фоні.
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    ink = cv2.bitwise_not(
        binary
    )

    points = cv2.findNonZero(
        ink
    )

    if points is None:
        return np.zeros(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            dtype=np.uint8,
        )

    x, y, w, h = cv2.boundingRect(
        points
    )

    crop = ink[
        y:y + h,
        x:x + w,
    ]

    padding = max(
        4,
        int(
            round(
                max(w, h) * 0.15
            )
        ),
    )

    side = max(
        w + 2 * padding,
        h + 2 * padding,
    )

    canvas = np.zeros(
        (
            side,
            side,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        side - w
    ) // 2

    offset_y = (
        side - h
    ) // 2

    canvas[
        offset_y:
        offset_y + h,
        offset_x:
        offset_x + w,
    ] = crop

    resized = cv2.resize(
        canvas,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def create_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )


def extract_features(
    image: np.ndarray,
    hog: cv2.HOGDescriptor,
) -> np.ndarray:
    prepared = prepare_image(
        image
    )

    features = hog.compute(
        prepared
    )

    return features.reshape(
        -1
    ).astype(
        np.float32
    )


def load_dataset():
    samples = []

    for digit in "0123456789":
        folder = (
            DATASET_DIR
            / digit
        )

        if not folder.exists():
            continue

        for image_path in sorted(
            folder.glob("*.png")
        ):
            samples.append(
                (
                    image_path,
                    int(digit),
                )
            )

    return samples


def split_dataset(
    samples,
):
    random.seed(
        RANDOM_SEED
    )

    per_class = {
        digit: []
        for digit in range(10)
    }

    for sample in samples:
        per_class[
            sample[1]
        ].append(sample)

    train = []
    test = []

    for digit, items in (
        per_class.items()
    ):
        random.shuffle(
            items
        )

        if len(items) < 2:
            train.extend(
                items
            )
            continue

        test_count = max(
            1,
            int(
                round(
                    len(items)
                    * TEST_RATIO
                )
            ),
        )

        test.extend(
            items[:test_count]
        )

        train.extend(
            items[test_count:]
        )

    random.shuffle(
        train
    )

    random.shuffle(
        test
    )

    return train, test


def build_matrix(
    samples,
    hog,
):
    features = []
    labels = []

    for image_path, digit in samples:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            continue

        features.append(
            extract_features(
                image,
                hog,
            )
        )

        labels.append(
            digit
        )

    return (
        np.asarray(
            features,
            dtype=np.float32,
        ),
        np.asarray(
            labels,
            dtype=np.int32,
        ),
    )


def main() -> None:
    samples = load_dataset()

    print()
    print("=" * 60)
    print(
        "FERRETTI OCR — SVM DIGIT TRAINING"
    )
    print("=" * 60)

    print()
    print(
        f"Всього зразків: {len(samples)}"
    )

    counts = Counter(
        label
        for _path, label in samples
    )

    for digit in range(10):
        print(
            f"{digit}: {counts[digit]}"
        )

    if len(samples) < 50:
        print()
        print(
            "Недостатньо даних для тестового "
            "навчання."
        )
        return

    train_samples, test_samples = (
        split_dataset(
            samples
        )
    )

    print()
    print(
        f"Train: {len(train_samples)}"
    )

    print(
        f"Test:  {len(test_samples)}"
    )

    hog = create_hog()

    x_train, y_train = build_matrix(
        train_samples,
        hog,
    )

    x_test, y_test = build_matrix(
        test_samples,
        hog,
    )

    svm = cv2.ml.SVM_create()

    svm.setType(
        cv2.ml.SVM_C_SVC
    )

    svm.setKernel(
        cv2.ml.SVM_RBF
    )

    svm.setC(
        10.0
    )

    svm.setGamma(
        0.01
    )

    print()
    print(
        "Навчання..."
    )

    svm.train(
        x_train,
        cv2.ml.ROW_SAMPLE,
        y_train,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    svm.save(
        str(MODEL_PATH)
    )

    print(
        f"Модель: {MODEL_PATH}"
    )

    if len(x_test) == 0:
        return

    _, predictions = svm.predict(
        x_test
    )

    predictions = (
        predictions
        .reshape(-1)
        .astype(np.int32)
    )

    correct = int(
        np.sum(
            predictions
            == y_test
        )
    )

    total = len(
        y_test
    )

    accuracy = (
        correct
        / total
        * 100.0
    )

    print()
    print(
        f"Правильно: {correct}/{total}"
    )

    print(
        f"TEST ACCURACY: "
        f"{accuracy:.2f}%"
    )

    print()
    print(
        "ПОМИЛКИ"
    )

    print("-" * 60)

    errors = Counter()

    for true_digit, predicted in zip(
        y_test,
        predictions,
    ):
        if true_digit != predicted:
            errors[
                (
                    int(true_digit),
                    int(predicted),
                )
            ] += 1

    if not errors:
        print(
            "Помилок немає."
        )
    else:
        for (
            true_digit,
            predicted,
        ), count in errors.most_common():
            print(
                f"{true_digit} -> "
                f"{predicted}: "
                f"{count}"
            )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
