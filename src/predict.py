import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_PATH, RANGE_HALF_WIDTH
from model import FruitRipenessModel
from utils import prediction_range


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fruit type and ripeness from an image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--humidity", required=True, type=float)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    args = parser.parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    values = torch.tensor([args.temperature, args.humidity])
    if not torch.isfinite(values).all():
        raise ValueError("Temperature and humidity must be finite numbers.")
    if not 0 <= args.humidity <= 100:
        raise ValueError("Humidity must be between 0 and 100.")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    fruit_to_index = checkpoint["fruit_to_index"]
    ripeness_to_index = checkpoint["ripeness_to_index"]
    model = FruitRipenessModel(len(ripeness_to_index), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization"]["mean"], checkpoint["normalization"]["std"]),
    ])
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        fruit_logits = model.fruit_head(features)
        ripeness_logits = model.ripeness_head(features)
    fruit_name = next(name for name, index in fruit_to_index.items() if index == fruit_logits.argmax(1).item())
    ripeness_name = next(name for name, index in ripeness_to_index.items() if index == ripeness_logits.argmax(1).item())
    print("========================================")
    print("FRUIT RIPENESS PREDICTION")
    print(f"Fruit Type       : {fruit_name.title()}")
    print(f"Ripeness Stage   : {ripeness_name.title()}")
    print(f"Temperature      : {args.temperature:.1f} °C")
    print(f"Humidity         : {args.humidity:.1f} %")
    if not checkpoint.get("regression_available", False):
        print("Predicted Time   : unavailable; regression was not trained from linked targets")
        print("Expected Range   : unavailable")
    else:
        with torch.no_grad():
            output = model(image_tensor, torch.tensor([args.temperature]), torch.tensor([args.humidity]))
        days = max(0.0, float(output["days_remaining"].item()))
        print(f"Predicted Time   : {days:.2f} days")
        print(f"Expected Range   : {prediction_range(days, RANGE_HALF_WIDTH)}")
    print("========================================")


if __name__ == "__main__":
    main()
