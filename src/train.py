import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    STAGE_1_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import FruitDataset, discover_samples, inspect_csv
from model import FruitRipenessModel
from utils import seed_everything


def build_mappings(samples_by_split: dict) -> tuple[dict[str, int], dict[str, int]]:
    fruit_names = sorted({sample.fruit_name for samples in samples_by_split.values() for sample in samples})
    ripeness_names = sorted({sample.ripeness_name for samples in samples_by_split.values() for sample in samples})
    if not fruit_names or not ripeness_names:
        raise RuntimeError("No valid image samples were found.")
    return (
        {name: index for index, name in enumerate(fruit_names)},
        {name: index for index, name in enumerate(ripeness_names)},
    )


def _run_classification_epoch(model, loader, optimizer, fruit_loss, ripeness_loss, device, training):
    model.train(training)
    total_loss = 0.0
    correct_fruit = 0
    correct_ripeness = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        fruit_targets = batch["fruit"].to(device)
        ripeness_targets = batch["ripeness"].to(device)
        with torch.set_grad_enabled(training):
            features = model.encode_image(images)
            fruit_logits = model.fruit_head(features)
            ripeness_logits = model.ripeness_head(features)
            loss = fruit_loss(fruit_logits, fruit_targets) + ripeness_loss(ripeness_logits, ripeness_targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct_fruit += (fruit_logits.argmax(1) == fruit_targets).sum().item()
        correct_ripeness += (ripeness_logits.argmax(1) == ripeness_targets).sum().item()
        total += images.size(0)
    if total == 0:
        raise RuntimeError("A dataset split contains no valid images.")
    return {
        "loss": total_loss / total,
        "fruit_accuracy": correct_fruit / total,
        "ripeness_accuracy": correct_ripeness / total,
    }


def train_model() -> None:
    seed_everything(RANDOM_SEED)
    metadata = inspect_csv()
    samples_by_split, _, problems = discover_samples(DATASET_DIR, ("train", "test"))
    if problems:
        print("Dataset warnings:", *problems[:10], sep="\n  ")
    if any(not samples_by_split[split] for split in ("train", "test")):
        raise RuntimeError("Train and test image splits are required; images were not moved or modified.")
    fruit_to_index, ripeness_to_index = build_mappings(samples_by_split)
    print("Detected ripeness classes:", list(ripeness_to_index))
    print("CSV columns:", metadata["columns"])
    if metadata["image_column"] is None or metadata["group_column"] is None:
        print("Regression disabled: CSV has no reliable image link and physical-fruit group key.")
    datasets = {
        "train": FruitDataset(samples_by_split["train"], fruit_to_index, ripeness_to_index, True, IMAGE_SIZE)
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitRipenessModel(len(ripeness_to_index), pretrained=True).to(device)
    model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.2)
    fruit_loss = nn.CrossEntropyLoss()
    ripeness_loss = nn.CrossEntropyLoss()
    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        if epoch == STAGE_1_EPOCHS:
            model.freeze_backbone(False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE / 10, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.2)
        train_metrics = _run_classification_epoch(model, loaders["train"], optimizer, fruit_loss, ripeness_loss, device, True)
        scheduler.step(train_metrics["loss"])
        print(f"epoch={epoch + 1} train={train_metrics}")
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fruit_to_index": fruit_to_index,
                    "ripeness_to_index": ripeness_to_index,
                    "image_size": IMAGE_SIZE,
                    "regression_available": False,
                    "training_loss": best_loss,
                    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                },
                MODEL_DIR / "best_model.pth",
            )
    print("Best checkpoint written to", MODEL_DIR / "best_model.pth")


def main() -> None:
    argparse.ArgumentParser(description="Train dynamic fruit and ripeness classifiers.").parse_args()
    train_model()


if __name__ == "__main__":
    main()
