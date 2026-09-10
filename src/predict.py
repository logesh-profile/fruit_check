import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import FruitRipenessModel

# =========================================================
# HARDCODED IMAGE PATH SETTING
# Replace this path with the exact image you want to test
# =========================================================
DEFAULT_IMAGE_PATH = r"C:\fruit_checking\dataset\banana_unripe1.jpg"


def predict_image(image_path: Path) -> None:

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "models" / "best_model.pth"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    print(f"Loading model: {checkpoint_path}")
    print(f"Testing image: {image_path}")

    # ---------------------------------------------------------
    # LOAD CHECKPOINT
    # ---------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    fruit_to_index = checkpoint["fruit_to_index"]
    ripeness_to_index = checkpoint["ripeness_to_index"]

    num_fruit_classes = checkpoint.get(
        "num_fruit_classes",
        len(fruit_to_index),
    )

    num_ripeness_classes = checkpoint.get(
        "num_ripeness_classes",
        len(ripeness_to_index),
    )

    # ---------------------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------------------
    model = FruitRipenessModel(
        num_ripeness_classes=num_ripeness_classes,
        num_fruit_classes=num_fruit_classes,
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    # ---------------------------------------------------------
    # IMAGE TRANSFORMATION
    # ---------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    checkpoint["image_size"],
                    checkpoint["image_size"],
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint["normalization"]["mean"],
                checkpoint["normalization"]["std"],
            ),
        ]
    )

    # ---------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------
    with Image.open(image_path) as image:
        image_tensor = transform(
            image.convert("RGB")
        ).unsqueeze(0)

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------
    with torch.no_grad():

        image_features = model.encode_image(
            image_tensor
        )

        # -----------------------------------------------------
        # FRUIT PREDICTION
        # -----------------------------------------------------
        fruit_logits = model.fruit_head(
            image_features
        )

        fruit_probabilities = torch.softmax(
            fruit_logits,
            dim=1,
        )

        fruit_index = fruit_logits.argmax(
            dim=1
        ).item()

        fruit_name = next(
            name
            for name, index in fruit_to_index.items()
            if index == fruit_index
        )

        fruit_confidence = (
            fruit_probabilities[
                0,
                fruit_index
            ].item()
            * 100
        )

        # -----------------------------------------------------
        # RIPENESS PREDICTION
        # -----------------------------------------------------
        ripeness_logits = model.ripeness_head(
            image_features
        )

        ripeness_probabilities = torch.softmax(
            ripeness_logits,
            dim=1,
        )

        ripeness_index = ripeness_logits.argmax(
            dim=1
        ).item()

        ripeness_name = next(
            name
            for name, index in ripeness_to_index.items()
            if index == ripeness_index
        )

        ripeness_confidence = (
            ripeness_probabilities[
                0,
                ripeness_index
            ].item()
            * 100
        )

    # ---------------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------------
    print()
    print("=" * 55)
    print("FRUIT RIPENESS PREDICTION")
    print("=" * 55)

    print(
        f"Fruit Type          : {fruit_name.title()}"
    )

    print(
        f"Fruit Confidence    : {fruit_confidence:.2f}%"
    )

    print(
        f"Ripeness Stage      : {ripeness_name.title()}"
    )

    print(
        f"Ripeness Confidence : {ripeness_confidence:.2f}%"
    )

    print()
    print(
        "Predicted Time      : unavailable"
    )

    print(
        "Reason              : MLP prediction will be connected later."
    )

    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fruit and ripeness prediction on a single image."
    )

    # Made --image_path optional with DEFAULT_IMAGE_PATH as fallback
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the image file to run prediction on.",
    )

    args = parser.parse_args()

    predict_image(Path(args.image_path))


if __name__ == "__main__":
    main()