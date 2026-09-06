import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATASET_DIR, NUM_WORKERS
from dataset import FruitDataset, discover_samples
from model import FruitRipenessModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the held-out test split."
    )

    # Automatically locate the project root:
    # C:\fruit_checking\src\evaluate.py
    # -> C:\fruit_checking
    project_root = Path(__file__).resolve().parents[1]

    # Automatically use:
    # C:\fruit_checking\models\best_model.pth
    default_checkpoint = project_root / "models" / "best_model.pth"

    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint),
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load trained model checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    fruit_to_index = checkpoint["fruit_to_index"]
    ripeness_to_index = checkpoint["ripeness_to_index"]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Only the TEST split is used here.
    # No validation data is used.
    # ---------------------------------------------------------
    samples_by_split, _, problems = discover_samples(
        DATASET_DIR,
        ("test",),
    )

    if problems:
        print(
            "Dataset warnings:",
            *problems[:10],
            sep="\n  ",
        )

    test_samples = samples_by_split["test"]

    print(f"Test samples: {len(test_samples)}")

    # Create test dataset
    dataset = FruitDataset(
        test_samples,
        fruit_to_index,
        ripeness_to_index,
        False,
        checkpoint["image_size"],
    )

    # Create test DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Recreate model architecture
    model = FruitRipenessModel(
        len(ripeness_to_index),
        pretrained=False,
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    # Store predictions
    fruit_true = []
    fruit_pred = []

    ripeness_true = []
    ripeness_pred = []

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------
    print("\nEvaluating test set...")

    with torch.no_grad():
        for batch in loader:

            # Extract image features using EfficientNetV2-S
            features = model.encode_image(batch["image"])

            # Fruit prediction
            fruit_logits = model.fruit_head(features)
            fruit_predictions = fruit_logits.argmax(1)

            # Ripeness prediction
            ripeness_logits = model.ripeness_head(features)
            ripeness_predictions = ripeness_logits.argmax(1)

            # Store ground truth
            fruit_true.extend(
                batch["fruit"].tolist()
            )

            ripeness_true.extend(
                batch["ripeness"].tolist()
            )

            # Store predictions
            fruit_pred.extend(
                fruit_predictions.tolist()
            )

            ripeness_pred.extend(
                ripeness_predictions.tolist()
            )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    fruit_labels = [
        name
        for name, _ in sorted(
            fruit_to_index.items(),
            key=lambda item: item[1],
        )
    ]

    ripeness_labels = [
        name
        for name, _ in sorted(
            ripeness_to_index.items(),
            key=lambda item: item[1],
        )
    ]

    # =========================================================
    # FRUIT RESULTS
    # =========================================================
    fruit_accuracy = accuracy_score(
        fruit_true,
        fruit_pred,
    )

    print("\n" + "=" * 60)
    print("FRUIT CLASSIFICATION")
    print("=" * 60)

    print(
        f"Fruit accuracy: {fruit_accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            fruit_true,
            fruit_pred,
            labels=list(range(len(fruit_labels))),
            target_names=fruit_labels,
            zero_division=0,
        )
    )

    print("Fruit confusion matrix:")

    print(
        confusion_matrix(
            fruit_true,
            fruit_pred,
            labels=list(range(len(fruit_labels))),
        )
    )

    # =========================================================
    # RIPENESS RESULTS
    # =========================================================
    ripeness_accuracy = accuracy_score(
        ripeness_true,
        ripeness_pred,
    )

    print("\n" + "=" * 60)
    print("RIPENESS CLASSIFICATION")
    print("=" * 60)

    print(
        f"Ripeness accuracy: {ripeness_accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            ripeness_true,
            ripeness_pred,
            labels=list(range(len(ripeness_labels))),
            target_names=ripeness_labels,
            zero_division=0,
        )
    )

    print("Ripeness confusion matrix:")

    print(
        confusion_matrix(
            ripeness_true,
            ripeness_pred,
            labels=list(range(len(ripeness_labels))),
        )
    )

    # =========================================================
    # REGRESSION STATUS
    # =========================================================
    print("\n" + "=" * 60)
    print("REGRESSION")
    print("=" * 60)

    if not checkpoint.get("regression_available", False):
        print(
            "Regression metrics unavailable:"
            " checkpoint was trained without linked "
            "sensor targets."
        )
    else:
        print(
            "Regression metrics are available in this checkpoint."
        )

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    print(
        f"Test samples       : {len(test_samples)}"
    )

    print(
        f"Fruit accuracy     : {fruit_accuracy:.4f}"
    )

    print(
        f"Ripeness accuracy  : {ripeness_accuracy:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()