import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATASET_DIR, IMAGE_EXTENSIONS


# ============================================================
# LABELS
# ============================================================

FRUIT_NAMES = (
    "banana",
    "mango",
    "no_fruit",
)

RIPENESS_NAMES = (
    "unripe",
    "ripe",
    "overripe",
)


# ============================================================
# SAMPLE
# ============================================================

@dataclass(frozen=True)
class Sample:
    image_path: Path
    fruit_name: str
    ripeness_name: str | None = None

    # Kept for future MLP/regression integration.
    temperature: float | None = None
    humidity: float | None = None
    days_remaining: float | None = None


# ============================================================
# LABEL NORMALISATION
# ============================================================

def normalise_label(value: str) -> str:
    """
    Convert labels such as:

        Over-Ripe
        over_ripe
        OVER RIPE

    into:

        over ripe
    """

    return " ".join(
        value.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def normalise_ripeness_label(value: str) -> str:
    """
    Convert folder names to our standard ripeness labels.
    """

    label = normalise_label(value)

    aliases = {
        "unripe": "unripe",
        "un ripe": "unripe",

        "ripe": "ripe",

        "overripe": "overripe",
        "over ripe": "overripe",
    }

    if label not in aliases:
        raise ValueError(
            f"Unknown ripeness class: {value!r}"
        )

    return aliases[label]


# ============================================================
# IMAGE HELPERS
# ============================================================

def _is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _verify_image(
    image_path: Path,
    problems: list[str],
) -> bool:
    """
    Verify that the image can actually be opened.
    """

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except (
        UnidentifiedImageError,
        OSError,
    ):
        problems.append(
            f"Unreadable image: {image_path}"
        )
        return False


# ============================================================
# SPLIT HANDLING
# ============================================================

TRAIN_NAMES = (
    "train",
    "Training",
)

TEST_NAMES = (
    "test",
    "Test",
)


def _find_split_dirs(
    directory: Path,
    split: str,
) -> list[Path]:
    """
    Find train/test folders using either capitalization.

    Example:

        train
        Training

        test
        Test
    """

    if split == "train":
        names = TRAIN_NAMES

    elif split == "test":
        names = TEST_NAMES

    else:
        raise ValueError(
            f"Unsupported split: {split}"
        )

    return [
        directory / name
        for name in names
        if (directory / name).is_dir()
    ]


# ============================================================
# ADD IMAGES
# ============================================================

def _add_images(
    samples_by_split: dict[str, list[Sample]],
    counts: Counter,
    problems: list[str],
    image_dir: Path,
    fruit_name: str,
    ripeness_name: str | None,
    split: str,
) -> None:
    """
    Add all images from a directory.

    For no_fruit:

        ripeness_name = None

    For banana/mango:

        ripeness_name = unripe / ripe / overripe
    """

    if ripeness_name is not None:
        ripeness_name = normalise_ripeness_label(
            ripeness_name
        )

    for image_path in sorted(
        image_dir.rglob("*")
    ):

        if not _is_image(image_path):
            continue

        if not _verify_image(
            image_path,
            problems,
        ):
            continue

        sample = Sample(
            image_path=image_path,
            fruit_name=fruit_name,
            ripeness_name=ripeness_name,
        )

        samples_by_split[split].append(sample)

        counts[
            (
                fruit_name,
                split,
                ripeness_name,
            )
        ] += 1


# ============================================================
# FRUIT DATASET DISCOVERY
# ============================================================

def _discover_fruit(
    fruit_dir: Path,
    fruit_name: str,
    samples_by_split: dict[str, list[Sample]],
    counts: Counter,
    problems: list[str],
) -> None:
    """
    Discover banana/mango images.

    Supports BOTH:

    Structure A:

        banana/
            train/
                ripe/
                unripe/
                overripe/
            test/
                ripe/
                unripe/
                overripe/

    Structure B:

        mango/
            ripe/
                Training/
                Test/
            unripe/
                Training/
                Test/
            overripe/
                train/
                test/
    """

    for split in ("train", "test"):

        # --------------------------------------------------------
        # STRUCTURE A
        #
        # fruit/train/class
        # fruit/test/class
        # --------------------------------------------------------

        direct_split_dirs = _find_split_dirs(
            fruit_dir,
            split,
        )

        if direct_split_dirs:

            for split_dir in direct_split_dirs:

                for class_dir in sorted(
                    split_dir.iterdir()
                ):

                    if not class_dir.is_dir():
                        continue

                    class_name = normalise_ripeness_label(
                        class_dir.name
                    )

                    _add_images(
                        samples_by_split=samples_by_split,
                        counts=counts,
                        problems=problems,
                        image_dir=class_dir,
                        fruit_name=fruit_name,
                        ripeness_name=class_name,
                        split=split,
                    )

            continue

        # --------------------------------------------------------
        # STRUCTURE B
        #
        # fruit/class/train
        # fruit/class/test
        # --------------------------------------------------------

        found_nested = False

        for class_dir in sorted(
            fruit_dir.iterdir()
        ):

            if not class_dir.is_dir():
                continue

            # Don't treat train/test as classes.
            if class_dir.name.lower() in {
                "train",
                "training",
                "test",
                "testing",
                "valid",
                "validation",
            }:
                continue

            try:
                ripeness_name = normalise_ripeness_label(
                    class_dir.name
                )
            except ValueError:
                continue

            nested_split_dirs = _find_split_dirs(
                class_dir,
                split,
            )

            for split_dir in nested_split_dirs:

                found_nested = True

                _add_images(
                    samples_by_split=samples_by_split,
                    counts=counts,
                    problems=problems,
                    image_dir=split_dir,
                    fruit_name=fruit_name,
                    ripeness_name=ripeness_name,
                    split=split,
                )

        if not found_nested:

            problems.append(
                f"Missing {split} images for "
                f"{fruit_name}: {fruit_dir}"
            )


# ============================================================
# NO-FRUIT DATASET DISCOVERY
# ============================================================

def _discover_no_fruit(
    no_fruit_dir: Path,
    samples_by_split: dict[str, list[Sample]],
    counts: Counter,
    problems: list[str],
) -> None:
    """
    Discover no_fruit images.

    Expected structure:

        no_fruit/
            train/
            test/

    no_fruit does NOT have a ripeness label.
    """

    if not no_fruit_dir.is_dir():

        problems.append(
            f"Missing no_fruit directory: "
            f"{no_fruit_dir}"
        )

        return

    for split in ("train", "test"):

        split_dirs = _find_split_dirs(
            no_fruit_dir,
            split,
        )

        if not split_dirs:

            problems.append(
                f"Missing {split} images for "
                f"no_fruit: {no_fruit_dir}"
            )

            continue

        for split_dir in split_dirs:

            _add_images(
                samples_by_split=samples_by_split,
                counts=counts,
                problems=problems,
                image_dir=split_dir,
                fruit_name="no_fruit",
                ripeness_name=None,
                split=split,
            )


# ============================================================
# DISCOVER ALL SAMPLES
# ============================================================

def discover_samples(
    dataset_dir: Path = DATASET_DIR,
    requested_splits: tuple[str, ...] = (
        "train",
        "test",
    ),
) -> tuple[
    dict[str, list[Sample]],
    Counter,
    list[str],
]:
    """
    Discover the complete image dataset.

    ONLY train and test are used.

    valid / Valid / validation folders
    are completely ignored.
    """

    allowed_splits = {
        "train",
        "test",
    }

    requested_splits = tuple(
        split
        for split in requested_splits
        if split in allowed_splits
    )

    samples_by_split = {
        "train": [],
        "test": [],
    }

    counts = Counter()
    problems: list[str] = []

    # ------------------------------------------------------------
    # BANANA
    # ------------------------------------------------------------

    if "train" in requested_splits or "test" in requested_splits:

        banana_dir = dataset_dir / "banana"

        if banana_dir.is_dir():

            _discover_fruit(
                banana_dir,
                "banana",
                samples_by_split,
                counts,
                problems,
            )

        else:

            problems.append(
                f"Missing fruit directory: "
                f"{banana_dir}"
            )

    # ------------------------------------------------------------
    # MANGO
    # ------------------------------------------------------------

    if "train" in requested_splits or "test" in requested_splits:

        mango_dir = dataset_dir / "mango"

        if mango_dir.is_dir():

            _discover_fruit(
                mango_dir,
                "mango",
                samples_by_split,
                counts,
                problems,
            )

        else:

            problems.append(
                f"Missing fruit directory: "
                f"{mango_dir}"
            )

    # ------------------------------------------------------------
    # NO FRUIT
    # ------------------------------------------------------------

    if "train" in requested_splits or "test" in requested_splits:

        no_fruit_dir = dataset_dir / "no_fruit"

        _discover_no_fruit(
            no_fruit_dir,
            samples_by_split,
            counts,
            problems,
        )

    # ------------------------------------------------------------
    # Remove samples from splits that weren't requested.
    # ------------------------------------------------------------

    for split in ("train", "test"):

        if split not in requested_splits:
            samples_by_split[split] = []

    return (
        samples_by_split,
        counts,
        problems,
    )


# ============================================================
# FLAT IMAGE SCANNER
# ============================================================

def scan_images(
    dataset_dir: Path = DATASET_DIR,
) -> tuple[
    list[tuple[Path, str, str | None]],
    Counter,
    list[str],
]:
    """
    Scan all train/test images.

    Returns:

        image_path
        fruit_name
        ripeness_name

    no_fruit has:

        ripeness_name = None
    """

    samples_by_split, counts, problems = (
        discover_samples(
            dataset_dir,
            ("train", "test"),
        )
    )

    flattened = []

    for split in ("train", "test"):

        for sample in samples_by_split[split]:

            flattened.append(
                (
                    sample.image_path,
                    sample.fruit_name,
                    sample.ripeness_name,
                )
            )

    return (
        flattened,
        counts,
        problems,
    )


# ============================================================
# PYTORCH DATASET
# ============================================================

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

        self.fruit_to_index = (
            fruit_to_index
        )

        self.ripeness_to_index = (
            ripeness_to_index
        )

        # --------------------------------------------------------
        # TRAINING TRANSFORMS
        # --------------------------------------------------------

        if training:

            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size + 32,
                            image_size + 32,
                        )
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

        # --------------------------------------------------------
        # TEST TRANSFORMS
        # --------------------------------------------------------

        else:

            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size + 32,
                            image_size + 32,
                        )
                    ),

                    transforms.CenterCrop(
                        image_size
                    ),

                    transforms.ToTensor(),

                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        sample = self.samples[index]

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        with Image.open(
            sample.image_path
        ) as image:

            image_tensor = self.transform(
                image.convert("RGB")
            )

        # --------------------------------------------------------
        # FRUIT LABEL
        # --------------------------------------------------------

        fruit_index = self.fruit_to_index[
            sample.fruit_name
        ]

        # --------------------------------------------------------
        # RIPENESS LABEL
        #
        # no_fruit has no ripeness.
        #
        # We use -1 so train.py can ignore this target.
        # --------------------------------------------------------

        if sample.ripeness_name is None:

            ripeness_index = -1

        else:

            ripeness_index = (
                self.ripeness_to_index[
                    sample.ripeness_name
                ]
            )

        # --------------------------------------------------------
        # RETURN
        # --------------------------------------------------------

        return {
            "image": image_tensor,

            "fruit": fruit_index,

            "ripeness": ripeness_index,

            "temperature": (
                float("nan")
                if sample.temperature is None
                else float(sample.temperature)
            ),

            "humidity": (
                float("nan")
                if sample.humidity is None
                else float(sample.humidity)
            ),

            "days_remaining": (
                float("nan")
                if sample.days_remaining is None
                else float(sample.days_remaining)
            ),

            "path": str(
                sample.image_path
            ),
        }


# ============================================================
# NUMBER VALIDATION
# ============================================================

def validate_number(
    value: str,
    field: str,
) -> float:
    """
    Validate a numeric value.
    """

    try:

        parsed = float(value)

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            f"Invalid {field}: {value!r}"
        ) from error

    if not math.isfinite(parsed):

        raise ValueError(
            f"Invalid {field}: {value!r}"
        )

    return parsed