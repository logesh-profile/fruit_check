import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_range(days: float, half_width: float = 0.5) -> str:
    if days <= 0.5:
        return "0 days"
    lower = max(0, int(np.floor(days - half_width)))
    upper = max(lower, int(np.ceil(days + half_width)))
    return f"{lower}-{upper} days"