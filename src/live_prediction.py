import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from days_model import DaysRemainingMLP
from model import FruitRipenessModel


def main() -> None:
    device = torch.device("cpu")
    print("=" * 60)
    print("FRUIT RIPENESS AI SYSTEM — YOLOV8 HYBRID INFERENCE")
    print("=" * 60)

    # 1. LOAD YOLOV8 DETECTOR
    # Pretrained YOLOv8n contains COCO classes (e.g. 'banana' = class 46)
    print("Loading YOLOv8 Object Detector...")
    yolo_model = YOLO("yolov8n.pt")

    # 2. LOAD EFFICIENTNET MODEL
    models_dir = PROJECT_ROOT / "models"
    img_checkpoint_path = models_dir / "best_model.pth"
    mlp_checkpoint_path = models_dir / "days_model.pth"

    if not img_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {img_checkpoint_path}")

    print(f"Loading EfficientNet Model: {img_checkpoint_path.name}")
    img_checkpoint = torch.load(img_checkpoint_path, map_location=device, weights_only=False)

    fruit_to_index = img_checkpoint["fruit_to_index"]
    ripeness_to_index = img_checkpoint["ripeness_to_index"]

    index_to_fruit = {index: name for name, index in fruit_to_index.items()}
    index_to_ripeness = {index: name for name, index in ripeness_to_index.items()}

    img_model = FruitRipenessModel(
        num_ripeness_classes=len(ripeness_to_index),
        num_fruit_classes=len(fruit_to_index),
        pretrained=False,
    )
    img_model.load_state_dict(img_checkpoint["model_state"])
    img_model.eval()

    # 3. LOAD DAYS REMAINING MLP MODEL
    print(f"Loading Days Model: {mlp_checkpoint_path.name}")
    mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location=device, weights_only=False)

    mlp_model = DaysRemainingMLP(input_dim=mlp_checkpoint["input_dim"])
    mlp_model.load_state_dict(mlp_checkpoint["model_state"])
    mlp_model.eval()

    temp_mean = mlp_checkpoint["temperature_mean"]
    temp_scale = mlp_checkpoint["temperature_scale"]
    hum_mean = mlp_checkpoint["humidity_mean"]
    hum_scale = mlp_checkpoint["humidity_scale"]

    # Environmental Sensor Values
    current_temp = 28.0
    current_hum = 70.0

    # 4. PREPROCESSING PIPELINE FOR CROPS
    image_size = img_checkpoint.get("image_size", 384)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=img_checkpoint["normalization"]["mean"],
            std=img_checkpoint["normalization"]["std"]
        )
    ])

    # 5. LIVE CAMERA LOOP
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera index 0 could not be opened.")

    print("\nStarting camera feed... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 Object Detection on full frame
        yolo_results = yolo_model(frame, verbose=False)[0]
        boxes = yolo_results.boxes

        fruit_detected_in_frame = False

        if len(boxes) > 0:
            for box in boxes:
                # Extract coordinates and bounding metadata
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Get predicted class label from YOLO
                yolo_label = yolo_model.names[cls_id].lower()

                # Filter target classes (e.g., banana, apple, orange, or custom trained mango)
                if yolo_label in ["banana", "apple", "orange"] and confidence > 0.40:
                    fruit_detected_in_frame = True

                    # Extract Bounding Box Crop
                    crop_bgr = frame[y1:y2, x1:x2]
                    if crop_bgr.size == 0:
                        continue

                    # Convert BGR to RGB for PyTorch Processing
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(crop_rgb)
                    input_tensor = transform(pil_image).unsqueeze(0).to(device)

                    # EfficientNet Inference
                    with torch.no_grad():
                        features = img_model.encode_image(input_tensor)

                        # Predict Fruit Type
                        fruit_logits = img_model.fruit_head(features)
                        fruit_idx = fruit_logits.argmax(dim=1).item()
                        fruit_name = index_to_fruit.get(fruit_idx, "banana")

                        # Predict Ripeness Stage
                        ripeness_logits = img_model.ripeness_head(features)
                        ripeness_probs = torch.softmax(ripeness_logits, dim=1)
                        ripeness_idx = ripeness_logits.argmax(dim=1).item()
                        ripeness_name = index_to_ripeness.get(ripeness_idx, "ripe")
                        ripeness_conf = ripeness_probs[0, ripeness_idx].item() * 100

                        # Calculate Days Remaining with MLP
                        fruit_oh = [1.0, 0.0] if fruit_name == "banana" else [0.0, 1.0]
                        stage_oh = [0.0, 0.0, 0.0]
                        if ripeness_name == "unripe":
                            stage_oh[0] = 1.0
                        elif ripeness_name == "ripe":
                            stage_oh[1] = 1.0
                        elif ripeness_name == "overripe":
                            stage_oh[2] = 1.0

                        norm_temp = (current_temp - temp_mean) / temp_scale
                        norm_hum = (current_hum - hum_mean) / hum_scale

                        mlp_in = torch.tensor([fruit_oh + stage_oh + [norm_temp, norm_hum]], dtype=torch.float32)
                        days_remaining = max(0.0, mlp_model(mlp_in).item())

                    # Draw Bounding Box & Labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_str = f"{fruit_name.title()} | {ripeness_name.title()} ({ripeness_conf:.0f}%)"
                    days_str = f"Days Left: {days_remaining:.1f}"

                    cv2.putText(frame, label_str, (x1, max(y1 - 25, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, days_str, (x1, max(y1 - 5, 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if not fruit_detected_in_frame:
            cv2.putText(frame, "Status: Searching for Fruit...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Fruit Ripeness AI System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()