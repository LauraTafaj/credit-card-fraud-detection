import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

DATA_FILE    = os.path.join(DATA_DIR,   "creditcard.csv")
SCALER_FILE  = os.path.join(MODELS_DIR, "scaler.joblib")
MODELS_FILE  = os.path.join(MODELS_DIR, "fraud_models.joblib")
ANN_FILE     = os.path.join(MODELS_DIR, "ann_model.joblib")
RESULTS_FILE = os.path.join(MODELS_DIR, "model_results.joblib")

RANDOM_STATE     = 42
TEST_SIZE        = 0.30
COLUMNS_TO_SCALE = ["Time", "Amount"]
TARGET_COLUMN    = "Class"


def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
