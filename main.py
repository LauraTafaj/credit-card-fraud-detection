import pandas as pd

from src import config
from src import train_models
from src import evaluate_models


def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - FULL PIPELINE")
    print("=" * 60)

    config.ensure_directories()

    sklearn_models, ann, scaler, X_test, y_test, feature_names = \
        train_models.run_training(train_ann_model=True)

    models_dict = dict(sklearn_models)
    if ann is not None:
        models_dict["ANN"] = ann

    results, table = evaluate_models.run_evaluation(models_dict, X_test, y_test)

    print("\n[11] Example prediction using the saved Random Forest model")
    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test.values
    fraud_example = test_df[test_df[config.TARGET_COLUMN] == 1]
    if len(fraud_example) > 0:
        row   = fraud_example.iloc[0].drop(config.TARGET_COLUMN).to_dict()
        proba = sklearn_models["Random Forest"].predict_proba(pd.DataFrame([row]))[:, 1][0]
        print(f"    A real fraud transaction was scored with fraud probability = {proba:.4f}")

    best = table.iloc[0]
    print("\n" + "=" * 60)
    print("PIPELINE FINISHED")
    print(f"Best model by F1-score: {best['Model']} "
          f"(F1 = {best['F1-score']}, ROC-AUC = {best['ROC-AUC']})")
    print("Outputs saved to the 'outputs/' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
