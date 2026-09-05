import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATASET_DIR, NUM_WORKERS
from dataset import FruitDataset, discover_samples
from model import FruitRipenessModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the held-out test split.")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    fruit_to_index = checkpoint["fruit_to_index"]
    ripeness_to_index = checkpoint["ripeness_to_index"]
    samples_by_split, _, problems = discover_samples(DATASET_DIR, ("test",))
    if problems:
        print("Dataset warnings:", *problems[:10], sep="\n  ")
    dataset = FruitDataset(samples_by_split["test"], fruit_to_index, ripeness_to_index, False, checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = FruitRipenessModel(len(ripeness_to_index), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    fruit_true, fruit_pred, ripeness_true, ripeness_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            features = model.encode_image(batch["image"])
            fruit_true.extend(batch["fruit"].tolist())
            fruit_pred.extend(model.fruit_head(features).argmax(1).tolist())
            ripeness_true.extend(batch["ripeness"].tolist())
            ripeness_pred.extend(model.ripeness_head(features).argmax(1).tolist())
    fruit_labels = [name for name, _ in sorted(fruit_to_index.items(), key=lambda item: item[1])]
    ripeness_labels = [name for name, _ in sorted(ripeness_to_index.items(), key=lambda item: item[1])]
    print("Fruit accuracy:", accuracy_score(fruit_true, fruit_pred))
    print(classification_report(fruit_true, fruit_pred, labels=list(range(len(fruit_labels))), target_names=fruit_labels, zero_division=0))
    print("Fruit confusion matrix:\n", confusion_matrix(fruit_true, fruit_pred, labels=list(range(len(fruit_labels)))))
    print("Ripeness accuracy:", accuracy_score(ripeness_true, ripeness_pred))
    print(classification_report(ripeness_true, ripeness_pred, labels=list(range(len(ripeness_labels))), target_names=ripeness_labels, zero_division=0))
    print("Ripeness confusion matrix:\n", confusion_matrix(ripeness_true, ripeness_pred, labels=list(range(len(ripeness_labels)))))
    if not checkpoint.get("regression_available", False):
        print("Regression metrics unavailable: checkpoint was trained without linked sensor targets.")


if __name__ == "__main__":
    main()
