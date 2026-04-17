from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATASET_PATH = DATA_DIR / "employee_performance_dataset.csv"
MODEL_PATH = MODELS_DIR / "employee_performance_model.pkl"
METRICS_PATH = OUTPUTS_DIR / "metrics_summary.json"
PREDICTIONS_PATH = OUTPUTS_DIR / "sample_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
PERFORMANCE_DISTRIBUTION_PATH = OUTPUTS_DIR / "performance_distribution.png"
NUMERIC_CORRELATION_PATH = OUTPUTS_DIR / "numeric_correlation_heatmap.png"
