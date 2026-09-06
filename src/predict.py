from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_PATH
from model import FruitRipenessModel


def main() -> None:
    # =========================================================
    # IMAGE TO TEST
    # =========================================================
    # Change this path to the image you want to test.
    image_path = Path(
        r"C:\fruit_checking\dataset\overripe_mango1.webp"
    )

    # =========================================================
    # CHECKPOINT
    # =========================================================
    # Automatically uses:
    # C:\fruit_checking\models\best_model.pth
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

    # =========================================================
    # LOAD CHECKPOINT
    # =========================================================
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    fruit_to_index = checkpoint["fruit_to_index"]
    ripeness_to_index = checkpoint["ripeness_to_index"]

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
    # IMAGE TRANSFORMATION
    # =========================================================
    transform = transforms.Compose([
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
    ])

    # =========================================================
    # LOAD IMAGE
    # =========================================================
    with Image.open(image_path) as image:
        image_tensor = transform(
            image.convert("RGB")
        ).unsqueeze(0)

    # =========================================================
    # PREDICTION
    # =========================================================
    with torch.no_grad():

        features = model.encode_image(
            image_tensor
        )

        fruit_logits = model.fruit_head(
            features
        )

        ripeness_logits = model.ripeness_head(
            features
        )

        fruit_probabilities = torch.softmax(
            fruit_logits,
            dim=1,
        )

        ripeness_probabilities = torch.softmax(
            ripeness_logits,
            dim=1,
        )

        fruit_index = fruit_logits.argmax(
            dim=1
        ).item()

        ripeness_index = ripeness_logits.argmax(
            dim=1
        ).item()

    # =========================================================
    # CONVERT INDEX TO NAME
    # =========================================================
    fruit_name = next(
        name
        for name, index in fruit_to_index.items()
        if index == fruit_index
    )

    ripeness_name = next(
        name
        for name, index in ripeness_to_index.items()
        if index == ripeness_index
    )

    fruit_confidence = (
        fruit_probabilities[0, fruit_index].item()
        * 100
    )

    ripeness_confidence = (
        ripeness_probabilities[
            0,
            ripeness_index
        ].item()
        * 100
    )

    # =========================================================
    # DISPLAY RESULT
    # =========================================================
    print()
    print("=" * 50)
    print("FRUIT RIPENESS PREDICTION")
    print("=" * 50)

    print(
        f"Fruit Type       : {fruit_name.title()}"
    )

    print(
        f"Fruit Confidence : {fruit_confidence:.2f}%"
    )

    print(
        f"Ripeness Stage   : {ripeness_name.title()}"
    )

    print(
        f"Ripeness Confidence : {ripeness_confidence:.2f}%"
    )

    print()
    print(
        "Predicted Time   : unavailable"
    )

    print(
        "Reason           : Regression was not trained "
        "with linked temperature/humidity targets."
    )

    print("=" * 50)


if __name__ == "__main__":
    main()