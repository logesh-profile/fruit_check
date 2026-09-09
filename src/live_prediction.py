import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

# ---------------------------------------------------------
# Allow importing project files when running this script
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import FruitRipenessModel
from days_model import DaysRemainingMLP


# =========================================================
# CONFIGURATION
# =========================================================

EFFICIENTNET_MODEL_PATH = (
    PROJECT_ROOT / "models" / "best_model.pth"
)

MLP_MODEL_PATH = (
    PROJECT_ROOT / "models" / "days_model.pth"
)

CAMERA_INDEX = 0

IMAGE_SIZE = 384

# ---------------------------------------------------------
# Temporary sensor values
#
# Later these will come from Raspberry Pi sensors.
# ---------------------------------------------------------

TEMPERATURE = 28.0
HUMIDITY = 70.0

# ---------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------

FRUIT_CONFIDENCE_THRESHOLD = 0.70

RIPENESS_CONFIDENCE_THRESHOLD = 0.60


# =========================================================
# IMAGE TRANSFORM
# =========================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# =========================================================
# LOAD EFFICIENTNET
# =========================================================

def load_efficientnet():
    print("Loading EfficientNet model...")

    if not EFFICIENTNET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"EfficientNet model not found:\n"
            f"{EFFICIENTNET_MODEL_PATH}"
        )

    checkpoint = torch.load(
        EFFICIENTNET_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    fruit_to_index = checkpoint[
        "fruit_to_index"
    ]

    ripeness_to_index = checkpoint[
        "ripeness_to_index"
    ]

    num_fruit_classes = checkpoint.get(
        "num_fruit_classes",
        len(fruit_to_index),
    )

    num_ripeness_classes = checkpoint.get(
        "num_ripeness_classes",
        len(ripeness_to_index),
    )

    model = FruitRipenessModel(
        num_ripeness_classes=num_ripeness_classes,
        num_fruit_classes=num_fruit_classes,
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print("EfficientNet loaded.")

    print(
        "Fruit classes:",
        fruit_to_index,
    )

    print(
        "Ripeness classes:",
        ripeness_to_index,
    )

    return (
        model,
        checkpoint,
        fruit_to_index,
        ripeness_to_index,
    )


# =========================================================
# LOAD MLP
# =========================================================

def load_mlp():
    print("Loading MLP model...")

    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MLP model not found:\n"
            f"{MLP_MODEL_PATH}"
        )

    checkpoint = torch.load(
        MLP_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = DaysRemainingMLP(
        input_dim=checkpoint["input_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Your train_days_model.py does NOT save:
    #
    # checkpoint["scaler"]
    #
    # Instead it saves these four values directly.
    # -----------------------------------------------------

    temperature_mean = float(
        checkpoint["temperature_mean"]
    )

    temperature_scale = float(
        checkpoint["temperature_scale"]
    )

    humidity_mean = float(
        checkpoint["humidity_mean"]
    )

    humidity_scale = float(
        checkpoint["humidity_scale"]
    )

    fruit_to_index = checkpoint[
        "fruit_to_index"
    ]

    stage_to_index = checkpoint[
        "stage_to_index"
    ]

    best_test_mae = float(
        checkpoint["best_test_mae"]
    )

    print("MLP loaded.")

    print(
        "MLP fruit mapping:",
        fruit_to_index,
    )

    print(
        "MLP stage mapping:",
        stage_to_index,
    )

    return (
        model,
        checkpoint,
        fruit_to_index,
        stage_to_index,
        temperature_mean,
        temperature_scale,
        humidity_mean,
        humidity_scale,
        best_test_mae,
    )


# =========================================================
# EFFICIENTNET PREDICTION
# =========================================================

def predict_image(
    model,
    frame,
    fruit_to_index,
    ripeness_to_index,
):
    """
    Predict fruit and ripeness from one camera frame.
    """

    # -----------------------------------------------------
    # OpenCV uses BGR.
    # EfficientNet expects RGB.
    # -----------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image_tensor = IMAGE_TRANSFORM(
        rgb
    )

    image_tensor = image_tensor.unsqueeze(0)

    # -----------------------------------------------------
    # Dummy sensor tensors.
    #
    # EfficientNet model contains a regression branch,
    # but we do NOT use its regression output.
    #
    # The separate MLP handles days remaining.
    # -----------------------------------------------------

    temperature_tensor = torch.tensor(
        [TEMPERATURE],
        dtype=torch.float32,
    )

    humidity_tensor = torch.tensor(
        [HUMIDITY],
        dtype=torch.float32,
    )

    with torch.no_grad():

        outputs = model(
            image_tensor,
            temperature_tensor,
            humidity_tensor,
        )

        fruit_probabilities = torch.softmax(
            outputs["fruit_logits"],
            dim=1,
        )

        ripeness_probabilities = torch.softmax(
            outputs["ripeness_logits"],
            dim=1,
        )

    # -----------------------------------------------------
    # Fruit prediction
    # -----------------------------------------------------

    fruit_confidence, fruit_index = (
        torch.max(
            fruit_probabilities,
            dim=1,
        )
    )

    fruit_confidence = float(
        fruit_confidence.item()
    )

    fruit_index = int(
        fruit_index.item()
    )

    index_to_fruit = {
        index: fruit
        for fruit, index in fruit_to_index.items()
    }

    fruit_name = index_to_fruit.get(
        fruit_index,
        "unknown",
    )

    # -----------------------------------------------------
    # Unknown / low-confidence rejection
    # -----------------------------------------------------

    if (
        fruit_name == "no_fruit"
        or fruit_confidence
        < FRUIT_CONFIDENCE_THRESHOLD
    ):
        return {
            "fruit": "No Fruit",
            "fruit_confidence": fruit_confidence,
            "ripeness": None,
            "ripeness_confidence": 0.0,
        }

    # -----------------------------------------------------
    # Ripeness prediction
    # -----------------------------------------------------

    ripeness_confidence, ripeness_index = (
        torch.max(
            ripeness_probabilities,
            dim=1,
        )
    )

    ripeness_confidence = float(
        ripeness_confidence.item()
    )

    ripeness_index = int(
        ripeness_index.item()
    )

    index_to_ripeness = {
        index: ripeness
        for ripeness, index
        in ripeness_to_index.items()
    }

    ripeness_name = index_to_ripeness.get(
        ripeness_index,
        "unknown",
    )

    if (
        ripeness_confidence
        < RIPENESS_CONFIDENCE_THRESHOLD
    ):
        ripeness_name = "uncertain"

    return {
        "fruit": fruit_name,
        "fruit_confidence": fruit_confidence,
        "ripeness": ripeness_name,
        "ripeness_confidence": ripeness_confidence,
    }


# =========================================================
# CREATE MLP INPUT
# =========================================================

def create_mlp_input(
    fruit,
    stage,
    temperature,
    humidity,
    fruit_to_index,
    stage_to_index,
    temperature_mean,
    temperature_scale,
    humidity_mean,
    humidity_scale,
):
    """
    Create exactly the same 7 features used during MLP
    training.

    Features:

        Banana
        Mango
        Unripe
        Ripe
        Overripe
        Temperature normalized
        Humidity normalized
    """

    # -----------------------------------------------------
    # Normalize names
    # -----------------------------------------------------

    fruit = (
        fruit
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    stage = (
        stage
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    # -----------------------------------------------------
    # Check mappings
    # -----------------------------------------------------

    if fruit not in fruit_to_index:
        raise ValueError(
            f"Fruit '{fruit}' not found in MLP mapping."
        )

    if stage not in stage_to_index:
        raise ValueError(
            f"Stage '{stage}' not found in MLP mapping."
        )

    # -----------------------------------------------------
    # Fruit one-hot
    # -----------------------------------------------------

    fruit_features = [
        0.0
    ] * len(fruit_to_index)

    fruit_features[
        fruit_to_index[fruit]
    ] = 1.0

    # -----------------------------------------------------
    # Stage one-hot
    # -----------------------------------------------------

    stage_features = [
        0.0
    ] * len(stage_to_index)

    stage_features[
        stage_to_index[stage]
    ] = 1.0

    # -----------------------------------------------------
    # Temperature normalization
    # -----------------------------------------------------

    temperature_normalized = (
        temperature
        - temperature_mean
    ) / temperature_scale

    # -----------------------------------------------------
    # Humidity normalization
    # -----------------------------------------------------

    humidity_normalized = (
        humidity
        - humidity_mean
    ) / humidity_scale

    # -----------------------------------------------------
    # Final feature vector
    # -----------------------------------------------------

    features = (
        fruit_features
        + stage_features
        + [
            temperature_normalized,
            humidity_normalized,
        ]
    )

    return torch.tensor(
        [features],
        dtype=torch.float32,
    )


# =========================================================
# MLP PREDICTION
# =========================================================

def predict_days_remaining(
    model,
    fruit,
    stage,
    temperature,
    humidity,
    fruit_to_index,
    stage_to_index,
    temperature_mean,
    temperature_scale,
    humidity_mean,
    humidity_scale,
):
    """
    Predict days remaining using the separately trained MLP.
    """

    input_tensor = create_mlp_input(
        fruit=fruit,
        stage=stage,
        temperature=temperature,
        humidity=humidity,
        fruit_to_index=fruit_to_index,
        stage_to_index=stage_to_index,
        temperature_mean=temperature_mean,
        temperature_scale=temperature_scale,
        humidity_mean=humidity_mean,
        humidity_scale=humidity_scale,
    )

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

    days = float(
        prediction.item()
    )

    # -----------------------------------------------------
    # Prevent negative days.
    # -----------------------------------------------------

    days = max(
        0.0,
        days,
    )

    return days


# =========================================================
# DRAW TEXT
# =========================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2,
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("LIVE FRUIT RIPENESS + DAYS PREDICTION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load EfficientNet
    # -----------------------------------------------------

    (
        efficientnet,
        efficientnet_checkpoint,
        fruit_to_index,
        ripeness_to_index,
    ) = load_efficientnet()

    # -----------------------------------------------------
    # Load MLP
    # -----------------------------------------------------

    (
        mlp,
        mlp_checkpoint,
        mlp_fruit_to_index,
        mlp_stage_to_index,
        temperature_mean,
        temperature_scale,
        humidity_mean,
        humidity_scale,
        best_test_mae,
    ) = load_mlp()

    # -----------------------------------------------------
    # Open camera
    # -----------------------------------------------------

    print()
    print(
        f"Opening camera index {CAMERA_INDEX}..."
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open camera.\n"
            "Check CAMERA_INDEX and camera permissions."
        )

    # -----------------------------------------------------
    # Optional camera resolution
    # -----------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    print("Camera started.")
    print()
    print("Press Q to quit.")
    print("=" * 60)

    # -----------------------------------------------------
    # Main camera loop
    # -----------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read frame from camera."
            )

            break

        # -------------------------------------------------
        # EfficientNet prediction
        # -------------------------------------------------

        try:

            result = predict_image(
                model=efficientnet,
                frame=frame,
                fruit_to_index=fruit_to_index,
                ripeness_to_index=ripeness_to_index,
            )

        except Exception as error:

            print(
                f"Prediction error: {error}"
            )

            draw_text(
                frame,
                "Prediction Error",
                (20, 40),
                0.9,
                2,
            )

            cv2.imshow(
                "Fruit Ripeness Detection",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            continue

        # -------------------------------------------------
        # Read result
        # -------------------------------------------------

        fruit = result["fruit"]

        fruit_confidence = result[
            "fruit_confidence"
        ]

        ripeness = result[
            "ripeness"
        ]

        ripeness_confidence = result[
            "ripeness_confidence"
        ]

        # -------------------------------------------------
        # Display fruit confidence
        # -------------------------------------------------

        draw_text(
            frame,
            f"Fruit: {fruit.title()}",
            (20, 40),
            0.8,
            2,
        )

        draw_text(
            frame,
            f"Confidence: "
            f"{fruit_confidence * 100:.1f}%",
            (20, 75),
            0.65,
            2,
        )

        # -------------------------------------------------
        # No fruit / unknown
        # -------------------------------------------------

        if (
            fruit == "No Fruit"
            or ripeness is None
        ):

            draw_text(
                frame,
                "Place Banana or Mango in view",
                (20, 120),
                0.65,
                2,
            )

            draw_text(
                frame,
                f"Temperature: "
                f"{TEMPERATURE:.1f} C",
                (20, 160),
                0.60,
                2,
            )

            draw_text(
                frame,
                f"Humidity: "
                f"{HUMIDITY:.1f} %",
                (20, 195),
                0.60,
                2,
            )

        # -------------------------------------------------
        # Known fruit
        # -------------------------------------------------

        else:

            draw_text(
                frame,
                f"Stage: {ripeness.title()}",
                (20, 120),
                0.8,
                2,
            )

            draw_text(
                frame,
                f"Stage Confidence: "
                f"{ripeness_confidence * 100:.1f}%",
                (20, 155),
                0.60,
                2,
            )

            draw_text(
                frame,
                f"Temperature: "
                f"{TEMPERATURE:.1f} C",
                (20, 195),
                0.60,
                2,
            )

            draw_text(
                frame,
                f"Humidity: "
                f"{HUMIDITY:.1f} %",
                (20, 230),
                0.60,
                2,
            )

            # ---------------------------------------------
            # Days remaining
            #
            # Only run the MLP when:
            #
            #   Fruit = Banana/Mango
            #   Stage = valid ripeness stage
            # ---------------------------------------------

            if (
                fruit.lower()
                in mlp_fruit_to_index
                and ripeness.lower()
                in mlp_stage_to_index
            ):

                try:

                    days = predict_days_remaining(
                        model=mlp,
                        fruit=fruit,
                        stage=ripeness,
                        temperature=TEMPERATURE,
                        humidity=HUMIDITY,
                        fruit_to_index=mlp_fruit_to_index,
                        stage_to_index=mlp_stage_to_index,
                        temperature_mean=temperature_mean,
                        temperature_scale=temperature_scale,
                        humidity_mean=humidity_mean,
                        humidity_scale=humidity_scale,
                    )

                    # -------------------------------------
                    # Display days
                    # -------------------------------------

                    if days <= 0.5:

                        days_text = (
                            "Ready / Fully Ripe"
                        )

                    elif days < 1.0:

                        days_text = (
                            f"{days:.1f} day remaining"
                        )

                    else:

                        days_text = (
                            f"{days:.1f} days remaining"
                        )

                    draw_text(
                        frame,
                        days_text,
                        (20, 275),
                        0.8,
                        2,
                    )

                    draw_text(
                        frame,
                        f"MLP Test MAE: "
                        f"+/-{best_test_mae:.2f} days",
                        (20, 310),
                        0.55,
                        1,
                    )

                except Exception as error:

                    draw_text(
                        frame,
                        "Days prediction unavailable",
                        (20, 275),
                        0.60,
                        2,
                    )

                    print(
                        f"MLP prediction error: "
                        f"{error}"
                    )

            else:

                draw_text(
                    frame,
                    "MLP mapping mismatch",
                    (20, 275),
                    0.60,
                    2,
                )

        # -------------------------------------------------
        # Camera instructions
        # -------------------------------------------------

        draw_text(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            0.55,
            1,
        )

        # -------------------------------------------------
        # Show frame
        # -------------------------------------------------

        cv2.imshow(
            "Fruit Ripeness Detection",
            frame,
        )

        # -------------------------------------------------
        # Keyboard
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("Camera stopped.")
    print("Program finished.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()