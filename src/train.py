import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    FINE_TUNE_LEARNING_RATE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    STAGE_1_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import FruitDataset, discover_samples
from model import FruitRipenessModel
from utils import seed_everything


def build_mappings(
    samples_by_split: dict,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Build mappings for fruit and ripeness labels.

    Example:

        banana -> 0
        mango  -> 1

        overripe -> 0
        ripe     -> 1
        unripe   -> 2
    """

    fruit_names = sorted(
        {
            sample.fruit_name
            for samples in samples_by_split.values()
            for sample in samples
        }
    )

    ripeness_names = sorted(
        {
            sample.ripeness_name
            for samples in samples_by_split.values()
            for sample in samples
        }
    )

    if not fruit_names:
        raise RuntimeError(
            "No fruit classes were found in the training dataset."
        )

    if not ripeness_names:
        raise RuntimeError(
            "No ripeness classes were found in the training dataset."
        )

    fruit_to_index = {
        name: index
        for index, name in enumerate(fruit_names)
    }

    ripeness_to_index = {
        name: index
        for index, name in enumerate(ripeness_names)
    }

    return fruit_to_index, ripeness_to_index


def _run_classification_epoch(
    model,
    loader,
    optimizer,
    fruit_loss,
    ripeness_loss,
    device,
    training: bool,
):
    """
    Run one training epoch.

    Only image features are used in this phase.

    Temperature, humidity and days_remaining are intentionally
    NOT used yet because the current phase is image-only training.
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_fruit = 0
    correct_ripeness = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        fruit_targets = batch["fruit"].to(device)
        ripeness_targets = batch["ripeness"].to(device)

        with torch.set_grad_enabled(training):

            # ---------------------------------------------------------
            # EfficientNetV2-S extracts image features
            # ---------------------------------------------------------
            features = model.encode_image(images)

            # ---------------------------------------------------------
            # Fruit classification:
            # banana / mango
            # ---------------------------------------------------------
            fruit_logits = model.fruit_head(features)

            # ---------------------------------------------------------
            # Ripeness classification:
            # unripe / ripe / overripe
            # ---------------------------------------------------------
            ripeness_logits = model.ripeness_head(features)

            # ---------------------------------------------------------
            # Combined classification loss
            # ---------------------------------------------------------
            fruit_loss_value = fruit_loss(
                fruit_logits,
                fruit_targets,
            )

            ripeness_loss_value = ripeness_loss(
                ripeness_logits,
                ripeness_targets,
            )

            loss = (
                fruit_loss_value
                + ripeness_loss_value
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        correct_fruit += (
            fruit_logits.argmax(dim=1)
            == fruit_targets
        ).sum().item()

        correct_ripeness += (
            ripeness_logits.argmax(dim=1)
            == ripeness_targets
        ).sum().item()

        total += batch_size

    if total == 0:
        raise RuntimeError(
            "The training dataset contains no valid images."
        )

    return {
        "loss": total_loss / total,
        "fruit_accuracy": correct_fruit / total,
        "ripeness_accuracy": correct_ripeness / total,
    }


def save_checkpoint(
    model,
    fruit_to_index,
    ripeness_to_index,
    training_loss,
):
    """
    Save the current best training checkpoint.

    Regression remains disabled in this phase.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state": model.state_dict(),

        "fruit_to_index": fruit_to_index,

        "ripeness_to_index": ripeness_to_index,

        "image_size": IMAGE_SIZE,

        # Regression is NOT trained yet.
        "regression_available": False,

        "training_loss": training_loss,

        "normalization": {
            "mean": [
                0.485,
                0.456,
                0.406,
            ],
            "std": [
                0.229,
                0.224,
                0.225,
            ],
        },
    }

    checkpoint_path = MODEL_DIR / "best_model.pth"

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def train_model() -> None:
    seed_everything(RANDOM_SEED)

    print("=" * 60)
    print("FRUIT RIPENESS MODEL TRAINING")
    print("=" * 60)

    print()
    print("Dataset:", DATASET_DIR)
    print("Training splits: train ONLY")
    print("Validation split: NOT USED")
    print("Test split: RESERVED FOR FINAL EVALUATION")
    print()

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # We explicitly request ONLY train and test.
    #
    # The dataset.py implementation also ignores valid completely.
    # -------------------------------------------------------------
    samples_by_split, counts, problems = discover_samples(
        DATASET_DIR,
        ("train", "test"),
    )

    # -------------------------------------------------------------
    # Show dataset warnings, if any.
    # -------------------------------------------------------------
    if problems:
        print("Dataset warnings:")

        for problem in problems[:20]:
            print(f"  - {problem}")

        if len(problems) > 20:
            print(
                f"  ... and {len(problems) - 20} more warnings"
            )

        print()

    # -------------------------------------------------------------
    # TRAIN is mandatory.
    # -------------------------------------------------------------
    if not samples_by_split["train"]:
        raise RuntimeError(
            "No training images were found."
        )

    # -------------------------------------------------------------
    # TEST is also mandatory because you want a train/test setup.
    #
    # We DO NOT use test during training.
    # It will be used later by evaluate.py.
    # -------------------------------------------------------------
    if not samples_by_split["test"]:
        raise RuntimeError(
            "No test images were found."
        )

    # -------------------------------------------------------------
    # Build class mappings using train + test labels.
    #
    # This does NOT train on test images.
    # It only makes sure both splits use the same label mapping.
    # -------------------------------------------------------------
    fruit_to_index, ripeness_to_index = build_mappings(
        samples_by_split
    )

    print("Fruit classes:")
    for name, index in fruit_to_index.items():
        print(f"  {index}: {name}")

    print()

    print("Ripeness classes:")
    for name, index in ripeness_to_index.items():
        print(f"  {index}: {name}")

    print()

    # -------------------------------------------------------------
    # Print image counts.
    # -------------------------------------------------------------
    print("Dataset counts:")

    for key, count in sorted(counts.items()):
        fruit_name, split, ripeness_name = key

        print(
            f"  {fruit_name:8s} | "
            f"{split:5s} | "
            f"{ripeness_name:10s} | "
            f"{count}"
        )

    print()

    print(
        f"Training images: "
        f"{len(samples_by_split['train'])}"
    )

    print(
        f"Test images: "
        f"{len(samples_by_split['test'])}"
    )

    print()

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Only TRAIN images are placed into the DataLoader.
    #
    # TEST images are completely excluded from training.
    # -------------------------------------------------------------
    train_dataset = FruitDataset(
        samples=samples_by_split["train"],
        fruit_to_index=fruit_to_index,
        ripeness_to_index=ripeness_to_index,
        training=True,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------
    # Select device.
    # -------------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()

    # -------------------------------------------------------------
    # Load pretrained EfficientNetV2-S.
    # -------------------------------------------------------------
    model = FruitRipenessModel(
        num_ripeness_classes=len(
            ripeness_to_index
        ),
        pretrained=True,
    ).to(device)

    print(
        "Loaded pretrained EfficientNetV2-S."
    )

    print()

    # -------------------------------------------------------------
    # STAGE 1
    #
    # Freeze EfficientNet backbone.
    #
    # Only:
    #   fruit_head
    #   ripeness_head
    #
    # are trained initially.
    # -------------------------------------------------------------
    model.freeze_backbone(True)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=2,
            factor=0.2,
        )
    )

    # -------------------------------------------------------------
    # Classification losses.
    # -------------------------------------------------------------
    fruit_loss = nn.CrossEntropyLoss()

    ripeness_loss = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # Track best TRAINING loss.
    #
    # Since you explicitly don't want validation, we cannot use
    # validation loss for checkpoint selection.
    #
    # Test is NEVER used for checkpoint selection.
    # -------------------------------------------------------------
    best_loss = float("inf")

    best_epoch = 0

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    for epoch in range(NUM_EPOCHS):

        # ---------------------------------------------------------
        # STAGE 2
        #
        # After STAGE_1_EPOCHS, unfreeze EfficientNet and fine-tune
        # the complete model with a smaller learning rate.
        # ---------------------------------------------------------
        if epoch == STAGE_1_EPOCHS:

            print()
            print(
                "Unfreezing EfficientNetV2-S backbone "
                "for fine-tuning..."
            )
            print()

            model.freeze_backbone(False)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=FINE_TUNE_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=2,
                    factor=0.2,
                )
            )

        # ---------------------------------------------------------
        # Train one epoch.
        # ---------------------------------------------------------
        train_metrics = _run_classification_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            fruit_loss=fruit_loss,
            ripeness_loss=ripeness_loss,
            device=device,
            training=True,
        )

        # ---------------------------------------------------------
        # Update learning-rate scheduler.
        # ---------------------------------------------------------
        scheduler.step(
            train_metrics["loss"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ---------------------------------------------------------
        # Print metrics.
        # ---------------------------------------------------------
        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"loss={train_metrics['loss']:.4f} "
            f"fruit_acc={train_metrics['fruit_accuracy']:.4f} "
            f"ripeness_acc={train_metrics['ripeness_accuracy']:.4f} "
            f"lr={current_lr:.2e}"
        )

        # ---------------------------------------------------------
        # Save best TRAINING checkpoint.
        #
        # Again:
        #   NO validation
        #   NO test evaluation
        #   NO test-based model selection
        # ---------------------------------------------------------
        if train_metrics["loss"] < best_loss:

            best_loss = train_metrics["loss"]
            best_epoch = epoch + 1

            checkpoint_path = save_checkpoint(
                model=model,
                fruit_to_index=fruit_to_index,
                ripeness_to_index=ripeness_to_index,
                training_loss=best_loss,
            )

            print(
                f"  -> Best training checkpoint saved "
                f"(epoch {best_epoch})"
            )

    # -------------------------------------------------------------
    # Training finished.
    # -------------------------------------------------------------
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best training loss: {best_loss:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Checkpoint:",
        MODEL_DIR / "best_model.pth",
    )

    print()

    print(
        "IMPORTANT: Test images were NOT used during training."
    )

    print(
        "Run evaluate.py next to measure performance on the "
        "held-out test set."
    )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train EfficientNetV2-S for fruit "
            "and ripeness classification."
        )
    )

    parser.parse_args()

    train_model()


if __name__ == "__main__":
    main()