from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from model import FruitRipenessModel


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "models" / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print(f"Loading EfficientNet checkpoint: {checkpoint_path}")

    # ---------------------------------------------------------
    # 1. LOAD EFFICIENTNET MODEL
    # ---------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    fruit_to_index = checkpoint["fruit_to_index"]
    ripeness_to_index = checkpoint["ripeness_to_index"]

    num_fruit_classes = checkpoint.get("num_fruit_classes", len(fruit_to_index))
    num_ripeness_classes = checkpoint.get("num_ripeness_classes", len(ripeness_to_index))

    classifier = FruitRipenessModel(
        num_ripeness_classes=num_ripeness_classes,
        num_fruit_classes=num_fruit_classes,
        pretrained=False,
    )
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()

    index_to_fruit = {v: k for k, v in fruit_to_index.items()}
    index_to_ripeness = {v: k for k, v in ripeness_to_index.items()}

    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            checkpoint["normalization"]["mean"],
            checkpoint["normalization"]["std"],
        ),
    ])

    # ---------------------------------------------------------
    # 2. LOAD YOLO DETECTOR & DEFINE TARGET CLASSES
    # COCO class IDs: 46 = banana, 47 = apple, 49 = orange
    # ---------------------------------------------------------
    print("Loading YOLOv8 detector...")
    yolo_model = YOLO("yolov8n.pt")

    # Target COCO class IDs for generic fruit detection
    ALLOWED_YOLO_CLASSES = [46, 47, 49]  # Banana, Apple, Orange

    # Set minimum confidence threshold (drops weak/unknown predictions)
    CONFIDENCE_THRESHOLD = 0.65

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nStarting Real-Time Detection...")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------------------------
        # STAGE 1: DETECT OBJECTS WITH CLASS FILTERING
        # classes=ALLOWED_YOLO_CLASSES ensures persons, chairs, etc. are IGNORED
        # ---------------------------------------------------------
        results = yolo_model(frame, verbose=False, conf=0.40, classes=ALLOWED_YOLO_CLASSES)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Crop ROI
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            image_tensor = transform(pil_crop).unsqueeze(0)

            # ---------------------------------------------------------
            # STAGE 2: CLASSIFY CROPPED FRUIT
            # ---------------------------------------------------------
            with torch.no_grad():
                features = classifier.encode_image(image_tensor)

                # Fruit prediction
                fruit_logits = classifier.fruit_head(features)
                fruit_probs = torch.softmax(fruit_logits, dim=1)
                fruit_idx = fruit_logits.argmax(dim=1).item()
                fruit_name = index_to_fruit[fruit_idx]
                fruit_conf = fruit_probs[0, fruit_idx].item()

                # Ripeness prediction
                ripeness_logits = classifier.ripeness_head(features)
                ripeness_probs = torch.softmax(ripeness_logits, dim=1)
                ripeness_idx = ripeness_logits.argmax(dim=1).item()
                ripeness_name = index_to_ripeness[ripeness_idx]
                ripeness_conf = ripeness_probs[0, ripeness_idx].item()

            # FLAW FIX: If classifier confidence is lower than threshold, mark as Unknown
            if fruit_conf < CONFIDENCE_THRESHOLD:
                label = "Unknown Object"
                color = (0, 0, 255)  # Red for unknown
            else:
                label = f"{fruit_name.title()} ({fruit_conf * 100:.0f}%) | {ripeness_name.title()} ({ripeness_conf * 100:.0f}%)"
                color = (0, 255, 0)  # Green for valid fruits

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 11, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        cv2.imshow("Fruit & Ripeness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()