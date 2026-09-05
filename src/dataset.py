import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from config import CSV_PATH, DATASET_DIR, FRUIT_NAMES, IMAGE_EXTENSIONS, SPLITS


@dataclass(frozen=True)
class Sample:
    image_path: Path
    fruit_name: str
    ripeness_name: str
    temperature: float | None = None
    humidity: float | None = None
    days_remaining: float | None = None


def normalise_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def inspect_csv(csv_path: Path = CSV_PATH) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    missing = {column: sum(not (row.get(column) or "").strip() for row in rows) for column in columns}
    numeric = {}
    for column in columns:
        values = []
        invalid = 0
        for row in rows:
            try:
                values.append(float((row.get(column) or "").strip()))
            except (TypeError, ValueError):
                invalid += 1
        if values:
            numeric[column] = {"invalid": invalid, "min": min(values), "max": max(values)}
    lowered = {column.lower(): column for column in columns}
    return {
        "columns": columns,
        "rows": len(rows),
        "missing": missing,
        "numeric": numeric,
        "fruit_values": sorted({row.get(lowered.get("fruit", ""), "") for row in rows}),
        "stage_values": sorted({row.get(lowered.get("stage", ""), "") for row in rows}),
        "duplicate_rows": len(rows) - len({tuple(row.items()) for row in rows}),
        "image_column": next((column for column in columns if "image" in column.lower() or "path" in column.lower()), None),
        "group_column": next((column for column in columns if any(term in column.lower() for term in ("fruit_id", "fruit id", "sample_id", "sample id", "observation", "image_id", "date", "day"))), None),
    }


def _candidate_split_dirs(fruit_dir: Path, split: str) -> list[Path]:
    legacy_names = {"train": "Training", "test": "Test", "valid": "Valid"}
    return [path for path in (fruit_dir / split, fruit_dir / legacy_names[split]) if path.exists()]


def discover_samples(dataset_dir: Path = DATASET_DIR, requested_splits: tuple[str, ...] = SPLITS) -> tuple[dict[str, list[Sample]], Counter, list[str]]:
    samples_by_split = {split: [] for split in SPLITS}
    counts = Counter()
    problems = []
    for fruit_name in FRUIT_NAMES:
        fruit_dir = dataset_dir / fruit_name
        if not fruit_dir.exists():
            problems.append(f"Missing fruit directory: {fruit_dir}")
            continue
        for split in requested_splits:
            split_dirs = _candidate_split_dirs(fruit_dir, split)
            if not split_dirs:
                legacy_name = {"train": "Training", "test": "Test", "valid": "Valid"}[split]
                nested_classes = [path for path in fruit_dir.iterdir() if path.is_dir() and (path / legacy_name).exists()]
                if nested_classes:
                    for class_dir in sorted(nested_classes):
                        class_name = normalise_label(class_dir.name)
                        for image_path in sorted((class_dir / legacy_name).rglob("*")):
                            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                                continue
                            try:
                                with Image.open(image_path) as image:
                                    image.verify()
                            except (UnidentifiedImageError, OSError):
                                problems.append(f"Unreadable image: {image_path}")
                                continue
                            samples_by_split[split].append(Sample(image_path, fruit_name, class_name))
                            counts[(fruit_name, split, class_name)] += 1
                    continue
                problems.append(f"Missing {split} directory for {fruit_name}: expected {fruit_dir / split}")
                continue
            split_dir = split_dirs[0]
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                class_name = normalise_label(class_dir.name)
                for image_path in sorted(class_dir.rglob("*")):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except (UnidentifiedImageError, OSError):
                        problems.append(f"Unreadable image: {image_path}")
                        continue
                    samples_by_split[split].append(Sample(image_path, fruit_name, class_name))
                    counts[(fruit_name, split, class_name)] += 1
    return samples_by_split, counts, problems


def scan_images(dataset_dir: Path = DATASET_DIR) -> tuple[list[tuple[Path, str, str]], Counter, list[str]]:
    samples_by_split, counts, problems = discover_samples(dataset_dir)
    flattened = [(sample.image_path, sample.fruit_name, sample.ripeness_name) for samples in samples_by_split.values() for sample in samples]
    return flattened, counts, problems


class FruitDataset(Dataset):
    def __init__(self, samples: list[Sample], fruit_to_index: dict[str, int], ripeness_to_index: dict[str, int], training: bool, image_size: int):
        self.samples = samples
        self.fruit_to_index = fruit_to_index
        self.ripeness_to_index = ripeness_to_index
        self.transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)) if training else transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip() if training else transforms.Lambda(lambda image: image),
            transforms.RandomRotation(8) if training else transforms.Lambda(lambda image: image),
            transforms.ColorJitter(brightness=0.12, saturation=0.12) if training else transforms.Lambda(lambda image: image),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return {
            "image": image_tensor,
            "fruit": self.fruit_to_index[sample.fruit_name],
            "ripeness": self.ripeness_to_index[sample.ripeness_name],
            "temperature": float("nan") if sample.temperature is None else sample.temperature,
            "humidity": float("nan") if sample.humidity is None else sample.humidity,
            "days_remaining": float("nan") if sample.days_remaining is None else sample.days_remaining,
            "path": str(sample.image_path),
        }


def validate_number(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid {field}: {value!r}")
    return parsed
