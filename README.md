# Fruit Ripeness Prediction

This project defines a PyTorch/TorchVision EfficientNetV2-S model for Mango and Banana fruit classification and dynamically discovered ripeness classification. With the current folders, the ripeness vocabulary is discovered from the available folders rather than fixed to five classes. Temperature and humidity are reserved for the sensor branch used by a future linked regression dataset.

## Current dataset finding

The supplied CSV has 1,000 complete rows and these exact columns: `Fruit`, `Stage`, `Temp avg`, `Humidity avg`, and `Days remaining`. It has no image path, fruit ID, sample ID, date, or day field. Sensor rows therefore cannot be assigned to image observations, and leakage-safe grouped splitting is impossible.

The supplied Banana image tree has `train`, `valid`, and `test` folders with `overripe`, `ripe`, and `unripe`. The Mango tree uses a different legacy layout with `Training` and `Test` nested below class folders and no validation folder. Training uses only `train`/`Training`; the existing `test`/`Test` folders are reserved for `evaluate.py`. No folders or images are moved by this project. Any discovered folder is treated as a class after normalization, so future `early_ripe` and `partially_ripe` folders are picked up automatically.

Classification training uses the existing explicit image splits and does not fabricate sensor values. The supplied CSV has no image path or physical-fruit group key, so days-remaining regression is disabled and no scientifically valid regression checkpoint can be created. Add an `image_path` column and a physical `fruit_id` column, with actual temperature, humidity, and `Days remaining` values for each linked observation, before implementing the regression training branch.

## Structure

`src/config.py` contains paths and configuration. `src/dataset.py` preserves explicit splits, discovers classes, validates images, and defines ImageNet preprocessing. `src/model.py` defines EfficientNetV2-S, fruit and dynamic ripeness heads, the sensor MLP, and regression head. `src/train.py`, `src/evaluate.py`, and `src/predict.py` are the command-line entry points.

## Installation

Windows training computer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Inspect the supplied data with:

```powershell
python src/train.py --report-only
```

Classification training is `python src/train.py`; it uses only training folders and saves the best training-loss checkpoint. Evaluation is `python src/evaluate.py`, which uses only test folders. Prediction is:

```powershell
python src/predict.py --image path/to/image.jpg --temperature 26.4 --humidity 68
```

## Raspberry Pi 5

Training may run on another computer. Copy `src/`, `models/best_model.pth`, and the required runtime packages to the Pi. Prediction uses CPU-safe `map_location="cpu"`, `model.eval()`, and `torch.no_grad()`, so CUDA is not required. Camera support can later replace the image-file open step with a PIL-compatible camera frame. DHT22 support can later replace the two CLI numbers; no hardware dependency is included now.

## Model and output

The model is EfficientNetV2-S with an image feature vector, a 2-class fruit head, a dynamically sized ripeness head, a two-input temperature/humidity MLP, and a fused regression head for days remaining. The current training path trains only the classification heads because the CSV is not linked to images; its checkpoint records `regression_available: false` and prediction reports time-to-ripe as unavailable.

The displayed time range is only a configurable presentation approximation: `4.32` becomes `4-5 days` using a half-day interval. It is not ground truth. Ethylene is not implemented or used.

## Limitations

THE CODE CAN BE MADE EXECUTABLE, BUT THE AVAILABLE DATA IS NOT SUFFICIENT TO SCIENTIFICALLY TRAIN THE DAYS-REMAINING REGRESSION MODEL. The required additional data is an image-to-CSV link, a physical fruit ID for grouped splitting, and actual sensor and days-remaining targets for each image observation. No claim is made that the supplied temperature/humidity values have a scientifically valid relationship to ripening until that linked dataset exists.