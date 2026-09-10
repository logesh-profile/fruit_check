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
    Use fixed class mappings so training and prediction
    always use the same class indices.

    Fruit:
        banana -> 0
        mango  -> 1

    Ripeness:
        overripe -> 0
        ripe     -> 1
        unripe   -> 2
    """

    # ---------------------------------------------------------
    # FIXED FRUIT MAPPING
    # ---------------------------------------------------------
    fruit_to_index = {
        "banana": 0,
        "mango": 1,
    }

    # ---------------------------------------------------------
    # FIXED RIPENESS MAPPING
    # ---------------------------------------------------------
    ripeness_to_index = {
        "overripe": 0,
        "ripe": 1,
        "unripe": 2,
    }

    # ---------------------------------------------------------
    # FIND CLASSES PRESENT IN DATASET
    # ---------------------------------------------------------
    found_fruits = {
        sample.fruit_name
        for samples in samples_by_split.values()
        for sample in samples
    }

    found_ripeness = {
        sample.ripeness_name
        for samples in samples_by_split.values()
        for sample in samples
        if sample.ripeness_name is not None
    }

    # ---------------------------------------------------------
    # CHECK REQUIRED FRUIT CLASSES
    # ---------------------------------------------------------
    missing_fruits = (
        set(fruit_to_index)
        - found_fruits
    )

    if missing_fruits:
        raise RuntimeError(
            "Missing required fruit classes: "
            f"{sorted(missing_fruits)}"
        )

    # ---------------------------------------------------------
    # CHECK REQUIRED RIPENESS CLASSES
    # ---------------------------------------------------------
    missing_ripeness = (
        set(ripeness_to_index)
        - found_ripeness
    )

    if missing_ripeness:
        raise RuntimeError(
            "Missing required ripeness classes: "
            f"{sorted(missing_ripeness)}"
        )

    return (
        fruit_to_index,
        ripeness_to_index,
    )


def freeze_all_backbone(model):
    """
    Freeze the complete EfficientNetV2-S backbone.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_last_backbone_blocks(
    model,
    num_blocks: int = 2,
):
    """
    Unfreeze only the final EfficientNetV2-S feature blocks.
    """

    freeze_all_backbone(model)

    backbone_blocks = list(
        model.features.children()
    )

    if len(backbone_blocks) < num_blocks:
        raise RuntimeError(
            "EfficientNet backbone has fewer "
            "blocks than requested."
        )

    for block in backbone_blocks[
        -num_blocks:
    ]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def _run_classification_epoch(
    model,
    loader,
    optimizer,
    fruit_loss,
    ripeness_loss,
    device,
    training: bool,
    scaler=None,
):
    """
    Run one image-classification epoch.

    Fruit classification:
        banana / mango

    Ripeness classification:
        unripe / ripe / overripe
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    correct_fruit = 0
    correct_ripeness = 0

    total_samples = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        fruit_targets = batch["fruit"].to(
            device,
            non_blocking=True,
        )

        ripeness_targets = batch["ripeness"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):

                # -------------------------------------------------
                # Extract image features
                # -------------------------------------------------
                features = model.encode_image(
                    images
                )

                # -------------------------------------------------
                # Fruit classification
                # -------------------------------------------------
                fruit_logits = model.fruit_head(
                    features
                )

                # -------------------------------------------------
                # Ripeness classification
                # -------------------------------------------------
                ripeness_logits = model.ripeness_head(
                    features
                )

                # -------------------------------------------------
                # Losses computation
                # -------------------------------------------------
                fruit_loss_value = fruit_loss(
                    fruit_logits,
                    fruit_targets,
                )

                ripeness_loss_value = ripeness_loss(
                    ripeness_logits,
                    ripeness_targets,
                )

                # -------------------------------------------------
                # Combined loss
                # -------------------------------------------------
                loss = (
                    fruit_loss_value
                    + ripeness_loss_value
                )

            # -----------------------------------------------------
            # Backpropagation
            # -----------------------------------------------------
            if training:

                if scaler is not None:

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss.backward()

                    optimizer.step()

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------
        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        # ---------------------------------------------------------
        # Fruit accuracy
        # ---------------------------------------------------------
        fruit_predictions = (
            fruit_logits.argmax(dim=1)
        )

        correct_fruit += (
            fruit_predictions
            == fruit_targets
        ).sum().item()

        # ---------------------------------------------------------
        # Ripeness accuracy
        # ---------------------------------------------------------
        ripeness_predictions = (
            ripeness_logits.argmax(dim=1)
        )

        correct_ripeness += (
            ripeness_predictions
            == ripeness_targets
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "The training dataset contains "
            "no valid images."
        )

    return {
        "loss": total_loss / total_samples,
        "fruit_accuracy": (
            correct_fruit / total_samples
        ),
        "ripeness_accuracy": (
            correct_ripeness / total_samples
        ),
    }


def save_checkpoint(
    model,
    fruit_to_index,
    ripeness_to_index,
    training_loss,
):
    """
    Save the best image-classification checkpoint.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state": model.state_dict(),

        "fruit_to_index": fruit_to_index,

        "ripeness_to_index": ripeness_to_index,

        "num_fruit_classes": len(
            fruit_to_index
        ),

        "num_ripeness_classes": len(
            ripeness_to_index
        ),

        "image_size": IMAGE_SIZE,

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

    checkpoint_path = (
        MODEL_DIR / "best_model.pth"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def train_model() -> None:

    seed_everything(
        RANDOM_SEED
    )

    print("=" * 60)
    print("FRUIT RIPENESS MODEL TRAINING")
    print("=" * 60)

    print()

    print(
        "Dataset:",
        DATASET_DIR,
    )

    print(
        "Training splits: train ONLY"
    )

    print(
        "Validation split: NOT USED"
    )

    print(
        "Test split: RESERVED FOR FINAL EVALUATION"
    )

    print()

    # ---------------------------------------------------------
    # DISCOVER ONLY TRAIN + TEST
    # ---------------------------------------------------------
    samples_by_split, counts, problems = (
        discover_samples(
            DATASET_DIR,
            ("train", "test"),
        )
    )

    # ---------------------------------------------------------
    # DATASET WARNINGS
    # ---------------------------------------------------------
    if problems:

        print("Dataset warnings:")

        for problem in problems[:20]:

            print(
                f"  - {problem}"
            )

        if len(problems) > 20:

            print(
                f"  ... and "
                f"{len(problems) - 20} "
                f"more warnings"
            )

        print()

    # ---------------------------------------------------------
    # TRAINING DATA REQUIRED
    # ---------------------------------------------------------
    if not samples_by_split["train"]:

        raise RuntimeError(
            "No training images were found."
        )

    # ---------------------------------------------------------
    # TEST DATA REQUIRED
    # ---------------------------------------------------------
    if not samples_by_split["test"]:

        raise RuntimeError(
            "No test images were found."
        )

    # ---------------------------------------------------------
    # BUILD FIXED MAPPINGS
    # ---------------------------------------------------------
    (
        fruit_to_index,
        ripeness_to_index,
    ) = build_mappings(
        samples_by_split
    )

    print("Fruit classes:")

    for name, index in (
        fruit_to_index.items()
    ):

        print(
            f"  {index}: {name}"
        )

    print()

    print("Ripeness classes:")

    for name, index in (
        ripeness_to_index.items()
    ):

        print(
            f"  {index}: {name}"
        )

    print()

    # ---------------------------------------------------------
    # DATASET COUNTS
    # ---------------------------------------------------------
    print("Dataset counts:")

    for key, count in sorted(
        counts.items()
    ):

        fruit_name, split, ripeness_name = key

        ripeness_display = (
            "none"
            if ripeness_name is None
            else ripeness_name
        )

        print(
            f"  {fruit_name:8s} | "
            f"{split:5s} | "
            f"{ripeness_display:10s} | "
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

    # ---------------------------------------------------------
    # TRAIN DATASET
    # ---------------------------------------------------------
    train_dataset = FruitDataset(
        samples=samples_by_split["train"],
        fruit_to_index=fruit_to_index,
        ripeness_to_index=ripeness_to_index,
        training=True,
        image_size=IMAGE_SIZE,
    )

    # ---------------------------------------------------------
    # TRAIN DATALOADER
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    # ---------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB",
        )

    print()

    # ---------------------------------------------------------
    # LOAD PRETRAINED EFFICIENTNETV2-S
    # ---------------------------------------------------------
    model = FruitRipenessModel(
        num_ripeness_classes=len(
            ripeness_to_index
        ),
        num_fruit_classes=len(
            fruit_to_index
        ),
        pretrained=True,
    ).to(device)

    print(
        "Loaded pretrained EfficientNetV2-S."
    )

    print()

    # ---------------------------------------------------------
    # STAGE 1
    #
    # Freeze EfficientNet backbone.
    # Train classification heads.
    # ---------------------------------------------------------
    freeze_all_backbone(
        model
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
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

    # ---------------------------------------------------------
    # LOSSES
    # ---------------------------------------------------------
    fruit_loss = nn.CrossEntropyLoss()

    ripeness_loss = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # CUDA AMP
    # ---------------------------------------------------------
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    # ---------------------------------------------------------
    # BEST TRAINING LOSS
    # ---------------------------------------------------------
    best_loss = float("inf")

    best_epoch = 0

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    print()

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Initial frozen epochs: "
        f"{STAGE_1_EPOCHS}"
    )

    print(
        "Fine-tuning: last 2 EfficientNet blocks"
    )

    print(
        f"Mixed precision: "
        f"{'ON' if device.type == 'cuda' else 'OFF'}"
    )

    print()

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(
        NUM_EPOCHS
    ):

        # -----------------------------------------------------
        # STAGE 2
        #
        # Unfreeze final 2 EfficientNet blocks.
        # -----------------------------------------------------
        if epoch == STAGE_1_EPOCHS:

            print()

            print(
                "Fine-tuning final 2 "
                "EfficientNetV2-S blocks..."
            )

            print()

            unfreeze_last_backbone_blocks(
                model,
                num_blocks=2,
            )

            trainable_parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]

            optimizer = torch.optim.AdamW(
                trainable_parameters,
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

        # -----------------------------------------------------
        # RUN TRAINING EPOCH
        # -----------------------------------------------------
        train_metrics = (
            _run_classification_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                fruit_loss=fruit_loss,
                ripeness_loss=ripeness_loss,
                device=device,
                training=True,
                scaler=scaler,
            )
        )

        # -----------------------------------------------------
        # SCHEDULER
        # -----------------------------------------------------
        scheduler.step(
            train_metrics["loss"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # -----------------------------------------------------
        # PRINT METRICS
        # -----------------------------------------------------
        print(
            f"Epoch "
            f"[{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"loss={train_metrics['loss']:.4f} "
            f"fruit_acc={train_metrics['fruit_accuracy']:.4f} "
            f"ripeness_acc={train_metrics['ripeness_accuracy']:.4f} "
            f"lr={current_lr:.2e}"
        )

        # -----------------------------------------------------
        # SAVE BEST TRAINING CHECKPOINT
        # -----------------------------------------------------
        if (
            train_metrics["loss"]
            < best_loss
        ):

            best_loss = (
                train_metrics["loss"]
            )

            best_epoch = (
                epoch + 1
            )

            checkpoint_path = (
                save_checkpoint(
                    model=model,
                    fruit_to_index=fruit_to_index,
                    ripeness_to_index=ripeness_to_index,
                    training_loss=best_loss,
                )
            )

            print(
                f"  -> Best training "
                f"checkpoint saved "
                f"(epoch {best_epoch})"
            )

    # ---------------------------------------------------------
    # TRAINING COMPLETE
    # ---------------------------------------------------------
    print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best training loss: "
        f"{best_loss:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "Checkpoint:",
        MODEL_DIR / "best_model.pth",
    )

    print()

    print(
        "IMPORTANT: "
        "Test images were NOT used during training."
    )

    print(
        "The test set is reserved for final evaluation."
    )

    print()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train EfficientNetV2-S for "
            "fruit and ripeness classification."
        )
    )

    parser.parse_args()

    train_model()


if __name__ == "__main__":
    main()