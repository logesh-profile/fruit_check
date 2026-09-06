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
    """Convert labels such as 'Over-Ripe', 'over_ripe' to 'over ripe'."""
    return " ".join(
        value.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def inspect_csv(csv_path: Path = CSV_PATH) -> dict:
    """
    Inspect the sensor CSV.

    This is kept for the future temperature/humidity/regression phase.
    It is NOT used to train the current image-only model.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")

    missing = {
        column: sum(
            not (row.get(column) or "").strip()
            for row in rows
        )
        for column in columns
    }

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
            numeric[column] = {
                "invalid": invalid,
                "min": min(values),
                "max": max(values),
            }

    lowered = {
        column.lower(): column
        for column in columns
    }

    return {
        "columns": columns,
        "rows": len(rows),
        "missing": missing,
        "numeric": numeric,
        "fruit_values": sorted(
            {
                row.get(lowered.get("fruit", ""), "")
                for row in rows
            }
        ),
        "stage_values": sorted(
            {
                row.get(lowered.get("stage", ""), "")
                for row in rows
            }
        ),
        "duplicate_rows": len(rows)
        - len({tuple(row.items()) for row in rows}),
        "image_column": next(
            (
                column
                for column in columns
                if "image" in column.lower()
                or "path" in column.lower()
            ),
            None,
        ),
        "group_column": next(
            (
                column
                for column in columns
                if any(
                    term in column.lower()
                    for term in (
                        "fruit_id",
                        "fruit id",
                        "sample_id",
                        "sample id",
                        "observation",
                        "image_id",
                        "date",
                        "day",
                    )
                )
            ),
            None,
        ),
    }


# ---------------------------------------------------------------------
# DATASET STRUCTURE HANDLING
# ---------------------------------------------------------------------
#
# This project intentionally uses ONLY:
#
#     train
#     test
#
# The following are supported:
#
# 1. fruit/train/class/image.jpg
# 2. fruit/test/class/image.jpg
#
# 3. fruit/Training/class/image.jpg
# 4. fruit/Test/class/image.jpg
#
# 5. fruit/class/train/image.jpg
# 6. fruit/class/test/image.jpg
#
# 7. fruit/class/Training/image.jpg
# 8. fruit/class/Test/image.jpg
#
# IMPORTANT:
# valid / Valid folders are NEVER scanned.
# ---------------------------------------------------------------------


SPLIT_NAMES = {
    "train": ("train", "Training"),
    "test": ("test", "Test"),
}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _verify_image(
    image_path: Path,
    problems: list[str],
) -> bool:
    """Check whether an image can actually be opened."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        problems.append(f"Unreadable image: {image_path}")
        return False


def _add_images(
    samples_by_split: dict[str, list[Sample]],
    counts: Counter,
    problems: list[str],
    image_dir: Path,
    fruit_name: str,
    class_name: str,
    split: str,
) -> None:
    """Add all valid images from one directory."""

    class_name = normalise_label(class_name)

    for image_path in sorted(image_dir.rglob("*")):
        if not _is_image(image_path):
            continue

        if not _verify_image(image_path, problems):
            continue

        samples_by_split[split].append(
            Sample(
                image_path=image_path,
                fruit_name=fruit_name,
                ripeness_name=class_name,
            )
        )

        counts[(fruit_name, split, class_name)] += 1


def _find_direct_split_dirs(
    fruit_dir: Path,
    split: str,
) -> list[Path]:
    """
    Find:

        fruit/train
        fruit/Training
        fruit/test
        fruit/Test

    Only train/test are considered.
    """

    return [
        fruit_dir / name
        for name in SPLIT_NAMES[split]
        if (fruit_dir / name).is_dir()
    ]


def _find_nested_class_split_dirs(
    fruit_dir: Path,
    split: str,
) -> list[tuple[Path, Path]]:
    """
    Find structures such as:

        fruit/
            ripe/
                train/
            unripe/
                train/
            overripe/
                test/

    Returns:
        [(class_directory, split_directory), ...]
    """

    results = []

    for class_dir in sorted(fruit_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        # Never treat these as ripeness classes.
        if normalise_label(class_dir.name) in {
            "train",
            "test",
            "valid",
            "training",
            "testing",
            "validation",
        }:
            continue

        for split_name in SPLIT_NAMES[split]:
            split_dir = class_dir / split_name

            if split_dir.is_dir():
                results.append((class_dir, split_dir))

    return results


def discover_samples(
    dataset_dir: Path = DATASET_DIR,
    requested_splits: tuple[str, ...] = SPLITS,
) -> tuple[dict[str, list[Sample]], Counter, list[str]]:
    """
    Discover image samples from ONLY train and test.

    Validation/valid folders are intentionally ignored.
    No files are moved, deleted, renamed, or modified.
    """

    # Force this function to work only with train/test.
    allowed_splits = ("train", "test")

    requested_splits = tuple(
        split
        for split in requested_splits
        if split in allowed_splits
    )

    samples_by_split = {
        split: []
        for split in allowed_splits
    }

    counts = Counter()
    problems = []

    for fruit_name in FRUIT_NAMES:
        fruit_dir = dataset_dir / fruit_name

        if not fruit_dir.is_dir():
            problems.append(
                f"Missing fruit directory: {fruit_dir}"
            )
            continue

        for split in requested_splits:

            # ---------------------------------------------------------
            # STRUCTURE 1:
            #
            # fruit/
            #     train/
            #         ripe/
            #         unripe/
            #         overripe/
            #
            # ---------------------------------------------------------

            direct_split_dirs = _find_direct_split_dirs(
                fruit_dir,
                split,
            )

            if direct_split_dirs:

                for split_dir in direct_split_dirs:

                    for class_dir in sorted(
                        path
                        for path in split_dir.iterdir()
                        if path.is_dir()
                    ):
                        class_name = normalise_label(
                            class_dir.name
                        )

                        # Extra safety: never read validation folders.
                        if class_name in {
                            "valid",
                            "validation",
                            "training",
                            "testing",
                        }:
                            continue

                        _add_images(
                            samples_by_split,
                            counts,
                            problems,
                            class_dir,
                            fruit_name,
                            class_name,
                            split,
                        )

                continue

            # ---------------------------------------------------------
            # STRUCTURE 2:
            #
            # fruit/
            #     ripe/
            #         train/
            #         test/
            #     unripe/
            #         train/
            #         test/
            #     overripe/
            #         train/
            #         test/
            #
            # ---------------------------------------------------------

            nested_split_dirs = _find_nested_class_split_dirs(
                fruit_dir,
                split,
            )

            if nested_split_dirs:

                for class_dir, split_dir in nested_split_dirs:

                    _add_images(
                        samples_by_split,
                        counts,
                        problems,
                        split_dir,
                        fruit_name,
                        class_dir.name,
                        split,
                    )

                continue

            # ---------------------------------------------------------
            # No train/test found for this fruit.
            #
            # We deliberately DO NOT look for:
            #
            #     valid/
            #     Valid/
            #     validation/
            #
            # ---------------------------------------------------------

            problems.append(
                f"Missing {split} images for {fruit_name}: "
                f"no supported train/test structure found in "
                f"{fruit_dir}"
            )

    return samples_by_split, counts, problems


def scan_images(
    dataset_dir: Path = DATASET_DIR,
) -> tuple[
    list[tuple[Path, str, str]],
    Counter,
    list[str],
]:
    """
    Scan ONLY train and test images.

    The valid folder is completely ignored.
    """

    samples_by_split, counts, problems = discover_samples(
        dataset_dir,
        ("train", "test"),
    )

    flattened = [
        (
            sample.image_path,
            sample.fruit_name,
            sample.ripeness_name,
        )
        for split in ("train", "test")
        for sample in samples_by_split[split]
    ]

    return flattened, counts, problems


# ---------------------------------------------------------------------
# PYTORCH DATASET
# ---------------------------------------------------------------------


class FruitDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        fruit_to_index: dict[str, int],
        ripeness_to_index: dict[str, int],
        training: bool,
        image_size: int,
    ):
        self.samples = samples
        self.fruit_to_index = fruit_to_index
        self.ripeness_to_index = ripeness_to_index

        if training:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size + 32, image_size + 32)
                    ),
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.85, 1.0),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.12,
                        saturation=0.12,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size + 32, image_size + 32)
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image_tensor = self.transform(
                image.convert("RGB")
            )

        return {
            "image": image_tensor,
            "fruit": self.fruit_to_index[
                sample.fruit_name
            ],
            "ripeness": self.ripeness_to_index[
                sample.ripeness_name
            ],
            "temperature": (
                float("nan")
                if sample.temperature is None
                else sample.temperature
            ),
            "humidity": (
                float("nan")
                if sample.humidity is None
                else sample.humidity
            ),
            "days_remaining": (
                float("nan")
                if sample.days_remaining is None
                else sample.days_remaining
            ),
            "path": str(sample.image_path),
        }


def validate_number(
    value: str,
    field: str,
) -> float:
    """Validate a numeric CSV value."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field}: {value!r}"
        ) from error

    if not math.isfinite(parsed):
        raise ValueError(
            f"Invalid {field}: {value!r}"
        )

    return parsed