import pandas as pd
import joblib

try:
    from . import config
    from . import data_preprocessing as dp
except ImportError:
    import config
    import data_preprocessing as dp


def load_artifacts(model_name="Random Forest"):
    scaler = joblib.load(config.SCALER_FILE)

    if model_name == "ANN":
        model = joblib.load(config.ANN_FILE)
    else:
        sklearn_models = joblib.load(config.MODELS_FILE)
        if model_name not in sklearn_models:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Available: {list(sklearn_models.keys())} + ['ANN']")
        model = sklearn_models[model_name]

    return model, scaler, model_name


def _prepare(transaction, scaler):
    if isinstance(transaction, dict):
        row = pd.DataFrame([transaction])
    else:
        row = transaction.copy()
    row, _ = dp.scale_features(row, scaler=scaler)
    return row


def predict_transaction(transaction, model_name="Random Forest"):
    model, scaler, model_name = load_artifacts(model_name)
    row   = _prepare(transaction, scaler)
    proba = float(model.predict_proba(row)[:, 1][0])
    label = int(model.predict(row)[0])

    return {
        "model":             model_name,
        "prediction":        "FRAUD" if label == 1 else "NORMAL",
        "fraud_probability": round(proba, 4),
    }


def _demo():
    print("Fraud prediction demo")
    print("=" * 40)

    df = pd.read_csv(config.DATA_FILE)
    fraud_row  = df[df[config.TARGET_COLUMN] == 1].iloc[0].drop(config.TARGET_COLUMN).to_dict()
    normal_row = df[df[config.TARGET_COLUMN] == 0].iloc[0].drop(config.TARGET_COLUMN).to_dict()

    for label, row in [("a KNOWN FRAUD row", fraud_row), ("a KNOWN NORMAL row", normal_row)]:
        result = predict_transaction(row, model_name="Random Forest")
        print(f"\nPredicting on {label}:")
        print(f"   -> {result['prediction']} (fraud probability = {result['fraud_probability']})")


if __name__ == "__main__":
    _demo()
