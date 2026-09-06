from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATASET_DIR, NUM_WORKERS
from dataset import FruitDataset, discover_samples
from model import FruitRipenessModel


def main() -> None:

    # =========================================================
    # PATHS
    # =========================================================

    # confidence.py is located directly inside:
    # C:\fruit_checking\confidence.py
    #
    # Therefore .parent gives:
    # C:\fruit_checking
    project_root = Path(__file__).resolve().parent

    checkpoint_path = (
        project_root
        / "models"
        / "best_model.pth"
    )

    results_dir = (
        project_root
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("=" * 70)
    print("DETAILED MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    # =========================================================
    # LOAD CHECKPOINT
    # =========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    fruit_to_index = checkpoint[
        "fruit_to_index"
    ]

    ripeness_to_index = checkpoint[
        "ripeness_to_index"
    ]

    # =========================================================
    # LOAD ONLY TEST DATA
    # =========================================================

    # IMPORTANT:
    # Only TEST data is used here.
    #
    # No validation folder is used.
    # No train data is used.
    samples_by_split, _, problems = discover_samples(
        DATASET_DIR,
        ("test",),
    )

    if problems:
        print(
            "\nDataset warnings:",
            *problems[:10],
            sep="\n  ",
        )

    test_samples = samples_by_split["test"]

    print(
        f"\nTest samples: {len(test_samples)}"
    )

    if len(test_samples) == 0:
        raise RuntimeError(
            "No test images were found. "
            "Please check the dataset/test folder structure."
        )

    dataset = FruitDataset(
        test_samples,
        fruit_to_index,
        ripeness_to_index,
        False,
        checkpoint["image_size"],
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # =========================================================
    # CREATE MODEL
    # =========================================================

    model = FruitRipenessModel(
        len(ripeness_to_index),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    # =========================================================
    # LABELS
    # =========================================================

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

    print(
        f"Fruit classes    : {fruit_labels}"
    )

    print(
        f"Ripeness classes : {ripeness_labels}"
    )

    # =========================================================
    # STORAGE
    # =========================================================

    fruit_true = []
    fruit_pred = []
    fruit_confidence = []

    ripeness_true = []
    ripeness_pred = []
    ripeness_confidence = []

    image_paths = []

    # =========================================================
    # EVALUATE
    # =========================================================

    print("\nEvaluating test set...")

    with torch.no_grad():

        for batch in loader:

            # -------------------------------------------------
            # IMAGE FEATURES
            # -------------------------------------------------

            features = model.encode_image(
                batch["image"]
            )

            # -------------------------------------------------
            # FRUIT CLASSIFICATION
            # -------------------------------------------------

            fruit_logits = model.fruit_head(
                features
            )

            fruit_probabilities = torch.softmax(
                fruit_logits,
                dim=1,
            )

            fruit_predictions = (
                fruit_probabilities.argmax(
                    dim=1
                )
            )

            fruit_confidences = (
                fruit_probabilities.max(
                    dim=1
                ).values
            )

            # -------------------------------------------------
            # RIPENESS CLASSIFICATION
            # -------------------------------------------------

            ripeness_logits = (
                model.ripeness_head(
                    features
                )
            )

            ripeness_probabilities = (
                torch.softmax(
                    ripeness_logits,
                    dim=1,
                )
            )

            ripeness_predictions = (
                ripeness_probabilities.argmax(
                    dim=1
                )
            )

            ripeness_confidences = (
                ripeness_probabilities.max(
                    dim=1
                ).values
            )

            # -------------------------------------------------
            # STORE RESULTS
            # -------------------------------------------------

            fruit_true.extend(
                batch["fruit"].tolist()
            )

            fruit_pred.extend(
                fruit_predictions.tolist()
            )

            fruit_confidence.extend(
                (
                    fruit_confidences * 100
                ).tolist()
            )

            ripeness_true.extend(
                batch["ripeness"].tolist()
            )

            ripeness_pred.extend(
                ripeness_predictions.tolist()
            )

            ripeness_confidence.extend(
                (
                    ripeness_confidences * 100
                ).tolist()
            )

            image_paths.extend(
                batch["path"]
            )

    # =========================================================
    # ACCURACY
    # =========================================================

    fruit_accuracy = accuracy_score(
        fruit_true,
        fruit_pred,
    )

    ripeness_accuracy = accuracy_score(
        ripeness_true,
        ripeness_pred,
    )

    # =========================================================
    # PRINT FRUIT RESULTS
    # =========================================================

    print("\n")
    print("=" * 70)
    print("FRUIT CLASSIFICATION")
    print("=" * 70)

    print(
        f"Accuracy: {fruit_accuracy * 100:.2f}%"
    )

    fruit_report = classification_report(
        fruit_true,
        fruit_pred,
        labels=list(
            range(len(fruit_labels))
        ),
        target_names=fruit_labels,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(fruit_report)

    # =========================================================
    # FRUIT CONFUSION MATRIX
    # =========================================================

    fruit_cm = confusion_matrix(
        fruit_true,
        fruit_pred,
        labels=list(
            range(len(fruit_labels))
        ),
    )

    print("Confusion Matrix:")
    print(fruit_cm)

    # Save fruit confusion matrix
    fruit_cm_path = (
        results_dir
        / "fruit_confusion_matrix.png"
    )

    fig, ax = plt.subplots()

    display = ConfusionMatrixDisplay(
        confusion_matrix=fruit_cm,
        display_labels=fruit_labels,
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Fruit Classification Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        fruit_cm_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================
    # RIPENESS RESULTS
    # =========================================================

    print("\n")
    print("=" * 70)
    print("RIPENESS CLASSIFICATION")
    print("=" * 70)

    print(
        f"Accuracy: {ripeness_accuracy * 100:.2f}%"
    )

    ripeness_report = classification_report(
        ripeness_true,
        ripeness_pred,
        labels=list(
            range(len(ripeness_labels))
        ),
        target_names=ripeness_labels,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(ripeness_report)

    # =========================================================
    # RIPENESS CONFUSION MATRIX
    # =========================================================

    ripeness_cm = confusion_matrix(
        ripeness_true,
        ripeness_pred,
        labels=list(
            range(len(ripeness_labels))
        ),
    )

    print("Confusion Matrix:")
    print(ripeness_cm)

    # Save ripeness confusion matrix
    ripeness_cm_path = (
        results_dir
        / "ripeness_confusion_matrix.png"
    )

    fig, ax = plt.subplots()

    display = ConfusionMatrixDisplay(
        confusion_matrix=ripeness_cm,
        display_labels=ripeness_labels,
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Ripeness Classification Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        ripeness_cm_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================
    # PER-IMAGE CSV
    # =========================================================

    fruit_index_to_name = {
        index: name
        for name, index
        in fruit_to_index.items()
    }

    ripeness_index_to_name = {
        index: name
        for name, index
        in ripeness_to_index.items()
    }

    rows = []

    for i in range(
        len(image_paths)
    ):

        fruit_actual = fruit_index_to_name[
            fruit_true[i]
        ]

        fruit_prediction = fruit_index_to_name[
            fruit_pred[i]
        ]

        ripeness_actual = (
            ripeness_index_to_name[
                ripeness_true[i]
            ]
        )

        ripeness_prediction = (
            ripeness_index_to_name[
                ripeness_pred[i]
            ]
        )

        rows.append(
            {
                "image": image_paths[i],

                "fruit_actual":
                    fruit_actual,

                "fruit_predicted":
                    fruit_prediction,

                "fruit_confidence_percent":
                    round(
                        fruit_confidence[i],
                        2,
                    ),

                "fruit_correct":
                    fruit_actual
                    == fruit_prediction,

                "ripeness_actual":
                    ripeness_actual,

                "ripeness_predicted":
                    ripeness_prediction,

                "ripeness_confidence_percent":
                    round(
                        ripeness_confidence[i],
                        2,
                    ),

                "ripeness_correct":
                    ripeness_actual
                    == ripeness_prediction,
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    # =========================================================
    # SAVE ALL TEST PREDICTIONS
    # =========================================================

    csv_path = (
        results_dir
        / "test_predictions.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # =========================================================
    # INCORRECT PREDICTIONS
    # =========================================================

    incorrect_fruit = results_df[
        results_df["fruit_correct"] == False
    ]

    incorrect_ripeness = results_df[
        results_df["ripeness_correct"] == False
    ]

    print("\n")
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Incorrect fruit predictions:"
        f" {len(incorrect_fruit)}"
    )

    print(
        f"Incorrect ripeness predictions:"
        f" {len(incorrect_ripeness)}"
    )

    # =========================================================
    # SAVE INCORRECT PREDICTIONS
    # =========================================================

    incorrect_fruit_path = (
        results_dir
        / "incorrect_fruit_predictions.csv"
    )

    incorrect_ripeness_path = (
        results_dir
        / "incorrect_ripeness_predictions.csv"
    )

    incorrect_fruit.to_csv(
        incorrect_fruit_path,
        index=False,
    )

    incorrect_ripeness.to_csv(
        incorrect_ripeness_path,
        index=False,
    )

    # =========================================================
    # LOW-CONFIDENCE PREDICTIONS
    # =========================================================

    low_confidence = results_df[
        results_df[
            "ripeness_confidence_percent"
        ] < 80
    ]

    print(
        f"Ripeness predictions below 80% confidence:"
        f" {len(low_confidence)}"
    )

    # =========================================================
    # SAVE LOW-CONFIDENCE PREDICTIONS
    # =========================================================

    low_confidence_path = (
        results_dir
        / "low_confidence_predictions.csv"
    )

    low_confidence.to_csv(
        low_confidence_path,
        index=False,
    )

    # =========================================================
    # CONFIDENCE SUMMARY
    # =========================================================

    average_fruit_confidence = (
        results_df[
            "fruit_confidence_percent"
        ].mean()
    )

    average_ripeness_confidence = (
        results_df[
            "ripeness_confidence_percent"
        ].mean()
    )

    print("\n")
    print("=" * 70)
    print("CONFIDENCE SUMMARY")
    print("=" * 70)

    print(
        f"Average fruit confidence:"
        f" {average_fruit_confidence:.2f}%"
    )

    print(
        f"Average ripeness confidence:"
        f" {average_ripeness_confidence:.2f}%"
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Test samples      : {len(test_samples)}"
    )

    print(
        f"Fruit accuracy    : "
        f"{fruit_accuracy * 100:.2f}%"
    )

    print(
        f"Ripeness accuracy : "
        f"{ripeness_accuracy * 100:.2f}%"
    )

    print(
        f"Wrong fruit       : "
        f"{len(incorrect_fruit)}"
    )

    print(
        f"Wrong ripeness    : "
        f"{len(incorrect_ripeness)}"
    )

    print(
        f"Low confidence    : "
        f"{len(low_confidence)}"
    )

    print("\nFiles saved:")

    print(
        f"  {fruit_cm_path}"
    )

    print(
        f"  {ripeness_cm_path}"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {incorrect_fruit_path}"
    )

    print(
        f"  {incorrect_ripeness_path}"
    )

    print(
        f"  {low_confidence_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()