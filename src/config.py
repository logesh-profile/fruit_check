from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
CSV_PATH = DATASET_DIR / "prediction_datas.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_EPOCHS = 30
STAGE_1_EPOCHS = 5
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
REGRESSION_LOSS_WEIGHT = 0.5
EARLY_STOPPING_PATIENCE = 7
RANDOM_SEED = 42
RANGE_HALF_WIDTH = 0.5
SPLITS = ("train", "valid", "test")
FRUIT_NAMES = ("banana", "mango")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
