import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from days_dataset import load_days_data
from days_model import DaysRemainingMLP


RANDOM_SEED = 42

TEST_SIZE = 0.20

BATCH_SIZE = 32

NUM_EPOCHS = 300

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 25

MODEL_PATH = Path("models/days_model.pth")


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_features(
    df,
    fruit_to_index,
    stage_to_index,
):
    """
    Convert Fruit and Stage to one-hot features.

    Current categories:
        Fruit:
            banana
            mango

        Stage:
            unripe
            ripe
            overripe

    Together with:
        temperature
        humidity

    Total input features = 7.
    """

    num_samples = len(df)

    num_fruits = len(fruit_to_index)
    num_stages = len(stage_to_index)

    fruit_features = np.zeros(
        (num_samples, num_fruits),
        dtype=np.float32,
    )

    stage_features = np.zeros(
        (num_samples, num_stages),
        dtype=np.float32,
    )

    for row_index, fruit in enumerate(df["Fruit"]):
        fruit_features[
            row_index,
            fruit_to_index[fruit],
        ] = 1.0

    for row_index, stage in enumerate(df["Stage"]):
        stage_features[
            row_index,
            stage_to_index[stage],
        ] = 1.0

    temperature = df["Temp avg"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    humidity = df["Humidity avg"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    return (
        fruit_features,
        stage_features,
        temperature,
        humidity,
    )


def build_feature_matrix(
    fruit_features,
    stage_features,
    temperature,
    humidity,
):
    return np.concatenate(
        [
            fruit_features,
            stage_features,
            temperature,
            humidity,
        ],
        axis=1,
    ).astype(np.float32)


def evaluate_model(
    model,
    loader,
    device,
):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for features, target in loader:

            features = features.to(device)

            output = model(features)

            predictions.extend(
                output.cpu().numpy()
            )

            targets.extend(
                target.numpy()
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    targets = np.asarray(
        targets,
        dtype=np.float32,
    )

    mae = mean_absolute_error(
        targets,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions,
        )
    )

    r2 = r2_score(
        targets,
        predictions,
    )

    return mae, rmse, r2


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the MLP model for predicting "
            "days remaining to fully ripe."
        )
    )

    parser.add_argument(
        "--csv",
        default="../dataset/prediction_datas.csv",
        help="Path to the prediction CSV.",
    )

    parser.add_argument(
        "--output",
        default=str(MODEL_PATH),
        help="Path for saving the trained MLP.",
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 60)
    print("FRUIT RIPENING DAYS MLP TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load CSV
    # ---------------------------------------------------------

    csv_path = Path(args.csv)

    print(f"\nLoading dataset: {csv_path}")

    df = load_days_data(csv_path)

    print(f"Total valid samples: {len(df)}")

    # ---------------------------------------------------------
    # 2. Create mappings
    # ---------------------------------------------------------

    fruit_values = sorted(
        df["Fruit"].unique()
    )

    stage_values = sorted(
        df["Stage"].unique()
    )

    fruit_to_index = {
        fruit: index
        for index, fruit in enumerate(
            fruit_values
        )
    }

    stage_to_index = {
        stage: index
        for index, stage in enumerate(
            stage_values
        )
    }

    print("\nFruit categories:")
    print(fruit_to_index)

    print("\nStage categories:")
    print(stage_to_index)

    # ---------------------------------------------------------
    # 3. Create features
    # ---------------------------------------------------------

    (
        fruit_features,
        stage_features,
        temperature,
        humidity,
    ) = create_features(
        df,
        fruit_to_index,
        stage_to_index,
    )

    X = build_feature_matrix(
        fruit_features,
        stage_features,
        temperature,
        humidity,
    )

    y = df[
        "Days remaining"
    ].to_numpy(
        dtype=np.float32
    )

    print(
        f"\nFeature matrix shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # ---------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------

    # Preserve Fruit + Stage combinations
    # in both train and test sets.

    stratify_labels = (
        df["Fruit"]
        + "_"
        + df["Stage"]
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        stratify_train,
        stratify_test,
    ) = train_test_split(
        X,
        y,
        stratify_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # ---------------------------------------------------------
    # 5. Normalize ONLY temperature + humidity
    # ---------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_train[:, -2:]
    )

    X_train[:, -2:] = scaler.transform(
        X_train[:, -2:]
    )

    X_test[:, -2:] = scaler.transform(
        X_test[:, -2:]
    )

    # ---------------------------------------------------------
    # 6. Convert to PyTorch tensors
    # ---------------------------------------------------------

    train_features = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    train_targets = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    test_features = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    test_targets = torch.tensor(
        y_test,
        dtype=torch.float32,
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_features,
        train_targets,
    )

    test_dataset = torch.utils.data.TensorDataset(
        test_features,
        test_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # 7. Create model
    # ---------------------------------------------------------

    input_dim = X_train.shape[1]

    print(
        f"\nMLP input features: {input_dim}"
    )

    model = DaysRemainingMLP(
        input_dim=input_dim
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    print(
        f"Training device: {device}"
    )

    # ---------------------------------------------------------
    # 8. Loss + optimizer
    # ---------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=8,
        factor=0.5,
    )

    # ---------------------------------------------------------
    # 9. Training
    # ---------------------------------------------------------

    best_test_mae = float("inf")

    patience_counter = 0

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nStarting training...\n")

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        sample_count = 0

        for features, targets in train_loader:

            features = features.to(device)

            targets = targets.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                features
            )

            loss = criterion(
                predictions,
                targets,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * features.size(0)
            )

            sample_count += features.size(0)

        train_loss = (
            running_loss
            / sample_count
        )

        test_mae, test_rmse, test_r2 = (
            evaluate_model(
                model,
                test_loader,
                device,
            )
        )

        scheduler.step(test_mae)

        if (
            test_mae
            < best_test_mae
        ):

            best_test_mae = test_mae

            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),

                    "input_dim": input_dim,

                    "fruit_to_index":
                        fruit_to_index,

                    "stage_to_index":
                        stage_to_index,

                    "temperature_mean":
                        scaler.mean_[0],

                    "temperature_scale":
                        scaler.scale_[0],

                    "humidity_mean":
                        scaler.mean_[1],

                    "humidity_scale":
                        scaler.scale_[1],

                    "best_test_mae":
                        best_test_mae,

                    "best_test_rmse":
                        test_rmse,

                    "best_test_r2":
                        test_r2,
                },
                output_path,
            )

        else:

            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or patience_counter == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test MAE: {test_mae:.4f} days | "
                f"Test RMSE: {test_rmse:.4f} days | "
                f"Test R²: {test_r2:.4f}"
            )

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ---------------------------------------------------------
    # 10. Final evaluation
    # ---------------------------------------------------------

    checkpoint = torch.load(
        output_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    final_mae, final_rmse, final_r2 = (
        evaluate_model(
            model,
            test_loader,
            device,
        )
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Final Test MAE  : {final_mae:.3f} days"
    )

    print(
        f"Final Test RMSE : {final_rmse:.3f} days"
    )

    print(
        f"Final Test R²   : {final_r2:.3f}"
    )

    print(
        f"\nBest model saved to:"
        f"\n{output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()