import argparse
from pathlib import Path

import torch

from days_model import DaysRemainingMLP


MODEL_PATH = Path("models/days_model.pth")


def normalize_text(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
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

    return model, checkpoint


def create_input(
    fruit: str,
    stage: str,
    temperature: float,
    humidity: float,
    checkpoint: dict,
) -> torch.Tensor:

    fruit = normalize_text(fruit)
    stage = normalize_text(stage)

    fruit_to_index = checkpoint["fruit_to_index"]
    stage_to_index = checkpoint["stage_to_index"]

    # ---------------------------------------------------------
    # Check fruit
    # ---------------------------------------------------------

    if fruit not in fruit_to_index:
        raise ValueError(
            f"Unknown fruit '{fruit}'. "
            f"Available fruits: "
            f"{list(fruit_to_index.keys())}"
        )

    # ---------------------------------------------------------
    # Check stage
    # ---------------------------------------------------------

    if stage not in stage_to_index:
        raise ValueError(
            f"Unknown stage '{stage}'. "
            f"Available stages: "
            f"{list(stage_to_index.keys())}"
        )

    # ---------------------------------------------------------
    # Validate sensor values
    # ---------------------------------------------------------

    if not torch.isfinite(
        torch.tensor(
            [temperature, humidity],
            dtype=torch.float32,
        )
    ).all():
        raise ValueError(
            "Temperature and humidity must be finite."
        )

    if not 0 <= humidity <= 100:
        raise ValueError(
            "Humidity must be between 0 and 100."
        )

    # ---------------------------------------------------------
    # Fruit one-hot
    # ---------------------------------------------------------

    fruit_features = [
        0.0
    ] * len(fruit_to_index)

    fruit_features[
        fruit_to_index[fruit]
    ] = 1.0

    # ---------------------------------------------------------
    # Stage one-hot
    # ---------------------------------------------------------

    stage_features = [
        0.0
    ] * len(stage_to_index)

    stage_features[
        stage_to_index[stage]
    ] = 1.0

    # ---------------------------------------------------------
    # Normalize temperature
    # ---------------------------------------------------------

    temperature_normalized = (
        temperature
        - checkpoint["temperature_mean"]
    ) / checkpoint["temperature_scale"]

    # ---------------------------------------------------------
    # Normalize humidity
    # ---------------------------------------------------------

    humidity_normalized = (
        humidity
        - checkpoint["humidity_mean"]
    ) / checkpoint["humidity_scale"]

    # ---------------------------------------------------------
    # Combine all features
    # ---------------------------------------------------------

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


def prediction_range(
    days: float,
    mae: float,
) -> str:

    if days <= 0.5:
        return "0 days"

    lower = max(
        0.0,
        days - mae,
    )

    upper = days + mae

    return (
        f"{lower:.1f}-{upper:.1f} days"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Test the trained fruit ripening "
            "days prediction MLP."
        )
    )

    parser.add_argument(
        "--fruit",
        required=True,
        help="Fruit name, e.g. banana or mango.",
    )

    parser.add_argument(
        "--stage",
        required=True,
        help=(
            "Ripeness stage, e.g. "
            "unripe, ripe or overripe."
        ),
    )

    parser.add_argument(
        "--temperature",
        required=True,
        type=float,
        help="Temperature in Celsius.",
    )

    parser.add_argument(
        "--humidity",
        required=True,
        type=float,
        help="Relative humidity in percentage.",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to trained MLP checkpoint.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    model, checkpoint = load_model(
        model_path
    )

    input_tensor = create_input(
        fruit=args.fruit,
        stage=args.stage,
        temperature=args.temperature,
        humidity=args.humidity,
        checkpoint=checkpoint,
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

    days = max(
        0.0,
        float(prediction.item()),
    )

    mae = float(
        checkpoint["best_test_mae"]
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print()
    print("=" * 50)
    print("FRUIT RIPENING PREDICTION")
    print("=" * 50)

    print(
        f"Fruit       : "
        f"{normalize_text(args.fruit).title()}"
    )

    print(
        f"Stage       : "
        f"{normalize_text(args.stage).title()}"
    )

    print(
        f"Temperature : "
        f"{args.temperature:.1f} °C"
    )

    print(
        f"Humidity    : "
        f"{args.humidity:.1f} %"
    )

    print("-" * 50)

    print(
        f"Predicted   : "
        f"{days:.2f} days"
    )

    print(
        f"Expected    : "
        f"{prediction_range(days, mae)}"
    )

    print(
        f"Model MAE   : "
        f"±{mae:.2f} days"
    )

    print("=" * 50)
    print()


if __name__ == "__main__":
    main()