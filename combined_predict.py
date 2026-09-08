from pathlib import Path
import joblib
import torch
from PIL import Image
from torchvision import transforms

from src.model import FruitRipenessModel


# =========================================================
# SETTINGS
# =========================================================

# Change this to your test image
image_path = Path("C:/Users/ShanthiRamesh/Desktop/fruit_check/dataset/mango1.webp")
# =========================================================
# PROJECT PATHS
# =========================================================

project_root = Path(__file__).resolve().parent

checkpoint_path = (
    project_root / "models" / "best_model.pth"
)

mlp_path = (
    project_root / "mlp_model.pkl"
)


# =========================================================
# CHECK FILES
# =========================================================

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Image model not found: {project_root / "models" / "best_model.pth"}"
    )

if not mlp_path.exists():
    raise FileNotFoundError(
        f"MLP model not found: {mlp_path}"
    )


# =========================================================
# LOAD IMAGE MODEL
# =========================================================

print("Loading image model...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=torch.device("cpu"),
    weights_only=False,
)

fruit_to_index = checkpoint["fruit_to_index"]
ripeness_to_index = checkpoint["ripeness_to_index"]

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
# IMAGE PREDICTION
# =========================================================

with Image.open(image_path) as image:

    image_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0)


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
# GET PREDICTED NAMES
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
    fruit_probabilities[
        0,
        fruit_index
    ].item()
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
# LOAD MLP MODEL
# =========================================================

print("Loading MLP model...")

mlp_model = joblib.load(
    mlp_path
)


# =========================================================
# USER INPUT
# =========================================================

print()
days_remaining = float(
    input("Enter days remaining: ")
)


# =========================================================
# MLP PREDICTION
# =========================================================

mlp_input = {
    "Fruit": [fruit_name],
    "Days remaining": [days_remaining],
}

mlp_prediction = mlp_model.predict(
    mlp_input
)[0]


# =========================================================
# FINAL RESULT
# =========================================================

print()
print("=" * 50)
print("FRUIT RIPENESS PREDICTION")
print("=" * 50)

print(
    f"Fruit Type          : {fruit_name.title()}"
)

print(
    f"Fruit Confidence    : {fruit_confidence:.2f}%"
)

print(
    f"Image Ripeness      : {ripeness_name.title()}"
)

print(
    f"Ripeness Confidence : {ripeness_confidence:.2f}%"
)

print(
    f"Days Remaining      : {days_remaining}"
)

print(
    f"MLP Predicted Stage : {mlp_prediction}"
)

print("=" * 50)