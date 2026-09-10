from pathlib import Path
import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from model import FruitRipenessModel
from days_model import DaysRemainingMLP


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_text(value: str) -> str:
    """Normalize class names for dictionary lookup."""
    return value.strip().lower().replace("_", " ").replace("-", " ")


def format_days_range(days: float) -> str:
    """
    Convert a numerical prediction into a simple range.

    Examples:
        2.1  -> 2-3 days
        2.8  -> 2-3 days
        3.2  -> 3-4 days
        7.4  -> 7-8 days
        10.8 -> 10-11 days
    """

    if days <= 0.5:
        return "0 days (Fully Ripe)"

    lower = int(days)
    upper = lower + 1

    return f"{lower}-{upper} days"


def create_mlp_input(
    fruit_name: str,
    stage_name: str,
    temperature: float,
    humidity: float,
    days_checkpoint: dict,
) -> torch.Tensor:
    """
    Create the exact 7-feature input expected by DaysRemainingMLP.

    Features:
        Fruit one-hot
        Stage one-hot
        Normalized temperature
        Normalized humidity
    """

    fruit = normalize_text(fruit_name)
    stage = normalize_text(stage_name)

    fruit_to_index = days_checkpoint["fruit_to_index"]
    stage_to_index = days_checkpoint["stage_to_index"]

    if fruit not in fruit_to_index:
        raise ValueError(
            f"Fruit '{fruit}' is not present in the Days Remaining model."
        )

    if stage not in stage_to_index:
        raise ValueError(
            f"Stage '{stage}' is not present in the Days Remaining model."
        )

    # -----------------------------------------------------
    # Fruit one-hot
    # -----------------------------------------------------

    fruit_features = [0.0] * len(fruit_to_index)
    fruit_features[fruit_to_index[fruit]] = 1.0

    # -----------------------------------------------------
    # Ripeness stage one-hot
    # -----------------------------------------------------

    stage_features = [0.0] * len(stage_to_index)
    stage_features[stage_to_index[stage]] = 1.0

    # -----------------------------------------------------
    # Normalize temperature
    # -----------------------------------------------------

    temp_scale = days_checkpoint["temperature_scale"]

    if temp_scale == 0:
        temp_scale = 1.0

    temp_norm = (
        temperature - days_checkpoint["temperature_mean"]
    ) / temp_scale

    # -----------------------------------------------------
    # Normalize humidity
    # -----------------------------------------------------

    hum_scale = days_checkpoint["humidity_scale"]

    if hum_scale == 0:
        hum_scale = 1.0

    hum_norm = (
        humidity - days_checkpoint["humidity_mean"]
    ) / hum_scale

    # -----------------------------------------------------
    # Final 7-feature vector
    # -----------------------------------------------------

    features = (
        fruit_features
        + stage_features
        + [temp_norm, hum_norm]
    )

    return torch.tensor(
        [features],
        dtype=torch.float32,
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # =====================================================
    # PROJECT PATHS
    # =====================================================

    project_root = Path(__file__).resolve().parents[1]

    image_checkpoint_path = (
        project_root / "models" / "best_model.pth"
    )

    days_checkpoint_path = (
        project_root / "models" / "days_model.pth"
    )

    if not image_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Image model checkpoint not found:\n"
            f"{image_checkpoint_path}"
        )

    if not days_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Days model checkpoint not found:\n"
            f"{days_checkpoint_path}"
        )

    # =====================================================
    # SENSOR INPUT
    # =====================================================

    print("=" * 60)
    print("ENVIRONMENT SENSOR DATA ENTRY")
    print("=" * 60)

    try:

        temp_input = float(
            input(
                "Enter Ambient Temperature (°C) "
                "[Default 28.0]: "
            ) or "28.0"
        )

        hum_input = float(
            input(
                "Enter Relative Humidity (%) "
                "[Default 65.0]: "
            ) or "65.0"
        )

    except ValueError:

        print(
            "\nInvalid input!"
            "\nUsing default values:"
            "\nTemperature = 28.0°C"
            "\nHumidity    = 65.0%"
        )

        temp_input = 28.0
        hum_input = 65.0

    # Basic sensor validation

    if not (-20 <= temp_input <= 60):
        print(
            "\nWarning: Temperature is outside the normal "
            "expected range."
        )

    if not (0 <= hum_input <= 100):
        print(
            "\nWarning: Humidity must normally be between "
            "0 and 100%."
        )

    print(
        f"\nActive Sensors -> "
        f"Temp: {temp_input:.1f}°C | "
        f"Humidity: {hum_input:.1f}%\n"
    )

    # =====================================================
    # LOAD EFFICIENTNET CHECKPOINT
    # =====================================================

    print(
        f"Loading EfficientNet checkpoint: "
        f"{image_checkpoint_path.name}"
    )

    image_checkpoint = torch.load(
        image_checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    # -----------------------------------------------------
    # Read mappings saved during training
    # -----------------------------------------------------

    fruit_to_index = image_checkpoint["fruit_to_index"]
    ripeness_to_index = image_checkpoint["ripeness_to_index"]

    num_fruit_classes = image_checkpoint.get(
        "num_fruit_classes",
        len(fruit_to_index),
    )

    num_ripeness_classes = image_checkpoint.get(
        "num_ripeness_classes",
        len(ripeness_to_index),
    )

    # -----------------------------------------------------
    # Create EfficientNet model
    # -----------------------------------------------------

    classifier = FruitRipenessModel(
        num_ripeness_classes=num_ripeness_classes,
        num_fruit_classes=num_fruit_classes,
        pretrained=False,
    )

    classifier.load_state_dict(
        image_checkpoint["model_state"]
    )

    classifier.eval()

    # -----------------------------------------------------
    # Reverse mappings
    # -----------------------------------------------------

    index_to_fruit = {
        v: k
        for k, v in fruit_to_index.items()
    }

    index_to_ripeness = {
        v: k
        for k, v in ripeness_to_index.items()
    }

    # =====================================================
    # IMAGE PREPROCESSING
    # =====================================================

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    image_checkpoint["image_size"],
                    image_checkpoint["image_size"],
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                image_checkpoint["normalization"]["mean"],
                image_checkpoint["normalization"]["std"],
            ),
        ]
    )

    # =====================================================
    # LOAD DAYS REMAINING MLP
    # =====================================================

    print(
        f"Loading Days Remaining model: "
        f"{days_checkpoint_path.name}"
    )

    days_checkpoint = torch.load(
        days_checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    days_model = DaysRemainingMLP(
        input_dim=days_checkpoint["input_dim"]
    )

    days_model.load_state_dict(
        days_checkpoint["model_state"]
    )

    days_model.eval()

    print("Days Remaining MLP loaded successfully.")

    # =====================================================
    # LOAD YOLO DETECTOR
    # =====================================================

    print("Loading YOLOv8 detector...")

    yolo_model = YOLO("yolov8n.pt")

    # COCO IDs:
    # 46 = banana
    # 47 = apple
    # 49 = orange

    ALLOWED_YOLO_CLASSES = [
        46,
        47,
        49,
    ]

    # -----------------------------------------------------
    # YOLO detection confidence
    # -----------------------------------------------------

    YOLO_CONFIDENCE = 0.40

    # -----------------------------------------------------
    # EfficientNet fruit confidence
    # -----------------------------------------------------

    FRUIT_CONFIDENCE = 0.65

    # Ripeness confidence is kept separately.
    # We do NOT use it to reject the fruit because your
    # working code only used fruit confidence for rejection.

    # =====================================================
    # OPEN CAMERA
    # =====================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "Error: Could not open camera."
        )

        return

    print("\n" + "=" * 60)
    print("STARTING REAL-TIME FRUIT PIPELINE")
    print("=" * 60)

    print(
        "YOLO      -> Object detection"
    )

    print(
        "EfficientNet -> Fruit + Ripeness"
    )

    print(
        "MLP       -> Days Remaining"
    )

    print(
        "\nPress 'q' to quit.\n"
    )

    # =====================================================
    # IMPORTANT:
    # We do NOT create [ [temp] ] tensors here.
    #
    # The working FruitRipenessModel forward() expects
    # temperature and humidity in shape [batch].
    #
    # However, we are using encode_image(), fruit_head()
    # and ripeness_head() directly, so sensor tensors are
    # not required for the EfficientNet prediction.
    #
    # Temperature/humidity are used by the separate MLP.
    # =====================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            print(
                "Error: Could not read frame."
            )
            break

        # =================================================
        # STAGE 1
        # YOLO OBJECT DETECTION
        # =================================================

        results = yolo_model(
            frame,
            verbose=False,
            conf=YOLO_CONFIDENCE,
            classes=ALLOWED_YOLO_CLASSES,
        )[0]

        # =================================================
        # PROCESS EVERY DETECTED OBJECT
        # =================================================

        for box in results.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            # -------------------------------------------------
            # Clamp coordinates to frame
            # -------------------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2,
            )

            y2 = min(
                frame.shape[0],
                y2,
            )

            # -------------------------------------------------
            # Crop detected object
            # -------------------------------------------------

            crop = frame[
                y1:y2,
                x1:x2,
            ]

            if crop.size == 0:
                continue

            # -------------------------------------------------
            # OpenCV BGR -> RGB
            # -------------------------------------------------

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB,
            )

            # -------------------------------------------------
            # Convert to PIL
            # -------------------------------------------------

            pil_crop = Image.fromarray(
                crop_rgb
            )

            # -------------------------------------------------
            # EXACT SAME PREPROCESSING AS WORKING CODE
            # -------------------------------------------------

            image_tensor = transform(
                pil_crop
            ).unsqueeze(0)

            # =================================================
            # STAGE 2
            # EFFICIENTNET FRUIT + RIPENESS
            # =================================================

            with torch.no_grad():

                # -------------------------------------------------
                # IMPORTANT:
                # Use the exact working prediction method.
                # -------------------------------------------------

                features = classifier.encode_image(
                    image_tensor
                )

                # -------------------------------------------------
                # FRUIT PREDICTION
                # -------------------------------------------------

                fruit_logits = classifier.fruit_head(
                    features
                )

                fruit_probs = torch.softmax(
                    fruit_logits,
                    dim=1,
                )

                fruit_idx = fruit_logits.argmax(
                    dim=1
                ).item()

                fruit_name = index_to_fruit[
                    fruit_idx
                ]

                fruit_conf = fruit_probs[
                    0,
                    fruit_idx,
                ].item()

                # -------------------------------------------------
                # RIPENESS PREDICTION
                # -------------------------------------------------

                ripeness_logits = classifier.ripeness_head(
                    features
                )

                ripeness_probs = torch.softmax(
                    ripeness_logits,
                    dim=1,
                )

                ripeness_idx = ripeness_logits.argmax(
                    dim=1
                ).item()

                ripeness_name = index_to_ripeness[
                    ripeness_idx
                ]

                ripeness_conf = ripeness_probs[
                    0,
                    ripeness_idx,
                ].item()

            # =================================================
            # UNKNOWN / LOW-CONFIDENCE HANDLING
            # =================================================

            if fruit_conf < FRUIT_CONFIDENCE:

                label = (
                    f"Unknown Object "
                    f"({fruit_conf * 100:.0f}%)"
                )

                color = (
                    0,
                    0,
                    255,
                )

            else:

                # =================================================
                # STAGE 3
                # DAYS REMAINING MLP
                # =================================================

                try:

                    mlp_input = create_mlp_input(
                        fruit_name=fruit_name,
                        stage_name=ripeness_name,
                        temperature=temp_input,
                        humidity=hum_input,
                        days_checkpoint=days_checkpoint,
                    )

                    with torch.no_grad():

                        prediction = days_model(
                            mlp_input
                        )

                    raw_days = float(
                        prediction.item()
                    )

                    # Prevent negative predictions

                    raw_days = max(
                        0.0,
                        raw_days,
                    )

                    days_range = format_days_range(
                        raw_days
                    )

                    # -------------------------------------------------
                    # Console output
                    # -------------------------------------------------

                    print("-" * 60)

                    print(
                        f"Fruit Detected : "
                        f"{normalize_text(fruit_name).title()} "
                        f"({fruit_conf * 100:.2f}%)"
                    )

                    print(
                        f"Ripeness Stage : "
                        f"{normalize_text(ripeness_name).title()} "
                        f"({ripeness_conf * 100:.2f}%)"
                    )

                    print(
                        f"Temperature    : "
                        f"{temp_input:.1f} °C"
                    )

                    print(
                        f"Humidity       : "
                        f"{hum_input:.1f} %"
                    )

                    print(
                        f"Raw Prediction : "
                        f"{raw_days:.2f} days"
                    )

                    print(
                        f"Days Remaining : "
                        f"{days_range}"
                    )

                    print("-" * 60)

                    # -------------------------------------------------
                    # Camera label
                    # -------------------------------------------------

                    label = (
                        f"{fruit_name.title()} | "
                        f"{ripeness_name.title()} | "
                        f"{days_range}"
                    )

                    color = (
                        0,
                        255,
                        0,
                    )

                except Exception as error:

                    print(
                        f"MLP prediction error: {error}"
                    )

                    label = (
                        f"{fruit_name.title()} | "
                        f"{ripeness_name.title()}"
                    )

                    color = (
                        0,
                        165,
                        255,
                    )

            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # -------------------------------------------------
            # Label position
            # -------------------------------------------------

            label_y1 = max(
                0,
                y1 - 30,
            )

            label_width = max(
                150,
                len(label) * 11,
            )

            label_x2 = min(
                frame.shape[1],
                x1 + label_width,
            )

            cv2.rectangle(
                frame,
                (x1, label_y1),
                (label_x2, y1),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    max(20, y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        # =====================================================
        # DISPLAY CAMERA
        # =====================================================

        cv2.imshow(
            "Fruit & Ripeness Detector",
            frame,
        )

        # =====================================================
        # QUIT
        # =====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()

    print("\nReal-time detection stopped.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()