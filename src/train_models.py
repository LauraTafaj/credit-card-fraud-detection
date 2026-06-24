import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from . import config
    from . import data_preprocessing as dp
except ImportError:
    import config
    import data_preprocessing as dp


def build_sklearn_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
    }


def train_sklearn_models(X_train, y_train):
    print("\n[5] Training the scikit-learn models")
    models = build_sklearn_models()
    for name, model in models.items():
        print(f"    - training {name} ...")
        model.fit(X_train, y_train)
    print("    Done.")
    return models


def train_ann(X_train, y_train):
    print("\n[6] Training the Artificial Neural Network (ANN)")
    ann = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=50,
        random_state=config.RANDOM_STATE,
        verbose=False,
    )
    ann.fit(X_train, y_train)
    print("    Done.")
    return ann


def save_models(sklearn_models, scaler, ann=None):
    print("\n[7] Saving models to the models/ folder")
    config.ensure_directories()
    joblib.dump(sklearn_models, config.MODELS_FILE)
    joblib.dump(scaler, config.SCALER_FILE)
    if ann is not None:
        joblib.dump(ann, config.ANN_FILE)
    print("    Saved scaler, scikit-learn models" +
          (" and ANN." if ann is not None else "."))


def run_training(train_ann_model=True):
    X_train, X_test, y_train, y_test, scaler, feature_names = dp.preprocess()
    sklearn_models = train_sklearn_models(X_train, y_train)

    ann = None
    if train_ann_model:
        ann = train_ann(X_train, y_train)

    save_models(sklearn_models, scaler, ann)
    return sklearn_models, ann, scaler, X_test, y_test, feature_names


if __name__ == "__main__":
    run_training()
