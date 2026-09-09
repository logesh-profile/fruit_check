import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from days_model import DaysRemainingMLP
from days_dataset import load_days_data


RANDOM_SEED = 42
TEST_SIZE = 0.20
MODEL_PATH = Path("models/days_model.pth")


def create_features(df, fruit_to_index, stage_to_index):
    num_samples = len(df)

    fruit_features = np.zeros(
        (num_samples, len(fruit_to_index)),
        dtype=np.float32,
    )

    stage_features = np.zeros(
        (num_samples, len(stage_to_index)),
        dtype=np.float32,
    )

    for i, fruit in enumerate(df["Fruit"]):
        fruit_features[i, fruit_to_index[fruit]] = 1.0

    for i, stage in enumerate(df["Stage"]):
        stage_features[i, stage_to_index[stage]] = 1.0

    temperature = df["Temp avg"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    humidity = df["Humidity avg"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    return np.concatenate(
        [
            fruit_features,
            stage_features,
            temperature,
            humidity,
        ],
        axis=1,
    ).astype(np.float32)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate trained fruit ripening days MLP."
    )

    parser.add_argument(
        "--csv",
        default="dataset/prediction_datas.csv",
        help="Path to prediction CSV.",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to trained MLP model.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FRUIT RIPENING DAYS MLP EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    csv_path = Path(args.csv)

    print(f"\nLoading dataset: {csv_path}")

    df = load_days_data(csv_path)

    print(f"Total samples: {len(df)}")

    # --------------------------------------------------
    # Create mappings
    # --------------------------------------------------

    fruit_values = sorted(df["Fruit"].unique())
    stage_values = sorted(df["Stage"].unique())

    fruit_to_index = {
        fruit: index
        for index, fruit in enumerate(fruit_values)
    }

    stage_to_index = {
        stage: index
        for index, stage in enumerate(stage_values)
    }

    print("\nFruit mapping:")
    print(fruit_to_index)

    print("\nStage mapping:")
    print(stage_to_index)

    # --------------------------------------------------
    # Create feature matrix
    # --------------------------------------------------

    X = create_features(
        df,
        fruit_to_index,
        stage_to_index,
    )

    y = df["Days remaining"].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------
    # Same train/test split as training
    # --------------------------------------------------

    stratify_labels = (
        df["Fruit"] + "_" + df["Stage"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    model_path = Path(args.model)

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

    # --------------------------------------------------
    # Apply saved normalization
    # --------------------------------------------------

    X_test[:, -2] = (
        X_test[:, -2]
        - checkpoint["temperature_mean"]
    ) / checkpoint["temperature_scale"]

    X_test[:, -1] = (
        X_test[:, -1]
        - checkpoint["humidity_mean"]
    ) / checkpoint["humidity_scale"]

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():
        predictions = model(test_tensor).numpy()

    # Prevent negative predictions
    predictions = np.maximum(
        predictions,
        0.0,
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTest samples : {len(y_test)}")
    print(f"MAE          : {mae:.3f} days")
    print(f"RMSE         : {rmse:.3f} days")
    print(f"R²           : {r2:.3f}")

    # --------------------------------------------------
    # Show sample predictions
    # --------------------------------------------------

    results = pd.DataFrame(
        {
            "Actual Days": y_test,
            "Predicted Days": predictions,
            "Error": np.abs(
                y_test - predictions
            ),
        }
    )

    print("\nSample predictions:")
    print(
        results.head(20).to_string(
            index=False,
            formatters={
                "Actual Days": "{:.2f}".format,
                "Predicted Days": "{:.2f}".format,
                "Error": "{:.2f}".format,
            },
        )
    )

    # --------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------

    output_path = Path(
        "results/days_model_predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nPrediction results saved to:\n"
        f"{output_path}"
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()