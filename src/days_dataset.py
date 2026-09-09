from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "Fruit",
    "Stage",
    "Temp avg",
    "Humidity avg",
    "Days remaining",
}


# ============================================================
# LOAD AND VALIDATE CSV
# ============================================================

def load_days_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the days-remaining dataset and validate its structure.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Days prediction CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("CSV contains no data.")

    # Keep only the columns required by this model.
    df = df[
        [
            "Fruit",
            "Stage",
            "Temp avg",
            "Humidity avg",
            "Days remaining",
        ]
    ].copy()

    # Remove completely empty rows.
    df.dropna(how="all", inplace=True)

    # Validate numerical columns.
    numerical_columns = [
        "Temp avg",
        "Humidity avg",
        "Days remaining",
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove rows with invalid numerical values.
    df.dropna(
        subset=[
            "Fruit",
            "Stage",
            "Temp avg",
            "Humidity avg",
            "Days remaining",
        ],
        inplace=True,
    )

    if df.empty:
        raise ValueError(
            "No valid rows remain after cleaning the CSV."
        )

    # Make category strings consistent.
    df["Fruit"] = (
        df["Fruit"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Stage"] = (
        df["Stage"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
    )

    # Check for invalid numerical values.
    if not np.isfinite(
        df[
            [
                "Temp avg",
                "Humidity avg",
                "Days remaining",
            ]
        ].to_numpy()
    ).all():
        raise ValueError(
            "CSV contains NaN or infinite numerical values."
        )

    # Humidity must be physically meaningful.
    if ((df["Humidity avg"] < 0) | (df["Humidity avg"] > 100)).any():
        raise ValueError(
            "Humidity values must be between 0 and 100."
        )

    # Days remaining cannot be negative.
    if (df["Days remaining"] < 0).any():
        raise ValueError(
            "Days remaining cannot be negative."
        )

    return df.reset_index(drop=True)


# ============================================================
# CATEGORY MAPPINGS
# ============================================================

def create_mappings(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Create numerical mappings for Fruit and Stage.
    """

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

    return fruit_to_index, stage_to_index


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    fruit_to_index: dict[str, int],
    stage_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert dataframe into numerical feature matrix X
    and regression target y.
    """

    unknown_fruits = set(df["Fruit"]) - set(fruit_to_index)

    if unknown_fruits:
        raise ValueError(
            f"Unknown fruit categories: {sorted(unknown_fruits)}"
        )

    unknown_stages = set(df["Stage"]) - set(stage_to_index)

    if unknown_stages:
        raise ValueError(
            f"Unknown stage categories: {sorted(unknown_stages)}"
        )

    fruit_encoded = df["Fruit"].map(fruit_to_index)
    stage_encoded = df["Stage"].map(stage_to_index)

    # Feature order is IMPORTANT.
    #
    # [Fruit, Stage, Temperature, Humidity]
    #
    X = np.column_stack(
        [
            fruit_encoded.to_numpy(dtype=np.float32),
            stage_encoded.to_numpy(dtype=np.float32),
            df["Temp avg"].to_numpy(dtype=np.float32),
            df["Humidity avg"].to_numpy(dtype=np.float32),
        ]
    )

    y = df["Days remaining"].to_numpy(
        dtype=np.float32
    )

    return X, y


# ============================================================
# PYTORCH DATASET
# ============================================================

class DaysRemainingDataset(Dataset):
    """
    PyTorch Dataset for days-remaining regression.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:

        if len(features) != len(targets):
            raise ValueError(
                "Features and targets must contain "
                "the same number of samples."
            )

        if len(features) == 0:
            raise ValueError(
                "Dataset cannot be empty."
            )

        self.features = torch.tensor(
            features,
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.features[index],
            self.targets[index],
        )


# ============================================================
# FEATURE SCALING
# ============================================================

def fit_scaler(
    features: np.ndarray,
) -> StandardScaler:
    """
    Fit a StandardScaler on training features only.
    """

    scaler = StandardScaler()

    scaler.fit(features)

    return scaler


def transform_features(
    features: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply an already-fitted scaler.
    """

    transformed = scaler.transform(features)

    return transformed.astype(np.float32)