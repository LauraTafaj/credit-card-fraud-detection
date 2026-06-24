import os
import pandas as pd
import streamlit as st

from src import config
from src import predict as predict_module


st.set_page_config(page_title="Credit Card Fraud Detection",
                   page_icon=":material/security:", layout="centered")

st.title("Credit Card Fraud Detection")
st.caption("Bachelor diploma project — Logistic Regression, Decision Tree, "
           "Random Forest and an Artificial Neural Network (ANN).")


def models_are_trained():
    return os.path.exists(config.MODELS_FILE) and os.path.exists(config.SCALER_FILE)


@st.cache_data
def load_dataset():
    return pd.read_csv(config.DATA_FILE)


def show_result(result, true_label=None):
    prob = result["fraud_probability"]
    if result["prediction"] == "FRAUD":
        st.error(f"### Prediction: FRAUD\nFraud probability: **{prob:.2%}**",
                 icon=":material/warning:")
    else:
        st.success(f"### Prediction: NORMAL\nFraud probability: **{prob:.2%}**",
                   icon=":material/check_circle:")

    st.progress(min(max(prob, 0.0), 1.0))

    if true_label is not None:
        actual = "FRAUD" if int(true_label) == 1 else "NORMAL"
        st.info(f"Actual label in the dataset: **{actual}**")


if not models_are_trained():
    st.warning(
        "No trained models found.\n\n"
        "Run **`python main.py`** once after placing `creditcard.csv` in the `data/` folder."
    )
    st.stop()


st.sidebar.header("Settings")
available_models = ["Random Forest", "Logistic Regression", "Decision Tree"]
if os.path.exists(config.ANN_FILE):
    available_models.append("ANN")

model_name = st.sidebar.selectbox(
    "Model used for prediction", available_models, index=0,
    help="Random Forest usually gives the best F1-score."
)

mode = st.sidebar.radio(
    "How do you want to enter a transaction?",
    ["Sample from dataset", "Upload a CSV file", "Manual entry"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: run `python main.py` again any time you retrain the models.")


if mode == "Sample from dataset":
    st.subheader("Check a real transaction from the dataset")

    df = load_dataset()

    col1, col2 = st.columns(2)
    with col1:
        only_fraud = st.checkbox("Only show fraud transactions", value=False)
    with col2:
        if st.button("Pick a random transaction", icon=":material/casino:"):
            st.session_state["row_idx"] = (
                df[df[config.TARGET_COLUMN] == 1].sample(1).index[0]
                if only_fraud else df.sample(1).index[0]
            )

    if "row_idx" not in st.session_state:
        st.session_state["row_idx"] = int(df.index[0])

    idx = st.number_input("Row number", min_value=0, max_value=len(df) - 1,
                          value=int(st.session_state["row_idx"]), step=1)

    row        = df.loc[idx]
    true_label = row[config.TARGET_COLUMN]
    transaction = row.drop(config.TARGET_COLUMN).to_dict()

    with st.expander("See this transaction's values"):
        st.dataframe(pd.DataFrame([transaction]).T.rename(columns={0: "value"}))

    if st.button("Predict", type="primary"):
        result = predict_module.predict_transaction(transaction, model_name=model_name)
        show_result(result, true_label=true_label)


elif mode == "Upload a CSV file":
    st.subheader("Predict on many transactions at once")
    st.write("Upload a CSV with the same columns as the dataset "
             "(`Time`, `V1`–`V28`, `Amount`). A `Class` column is optional.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(data):,}** transactions.")

        if st.button("Predict all", type="primary"):
            features = data.drop(columns=[config.TARGET_COLUMN], errors="ignore")

            model, scaler, _ = predict_module.load_artifacts(model_name)
            scaled, _ = predict_module.dp.scale_features(features, scaler=scaler)

            proba = model.predict_proba(scaled)[:, 1]
            preds = model.predict(scaled)

            out = data.copy()
            out["fraud_probability"] = proba.round(4)
            out["prediction"] = ["FRAUD" if p == 1 else "NORMAL" for p in preds]

            n_fraud = int((preds == 1).sum())
            st.success(f"Done. Flagged **{n_fraud}** transaction(s) as fraud out of {len(out):,}.")
            st.dataframe(out.head(50))

            st.download_button(
                "Download full results as CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                icon=":material/download:",
            )


else:
    st.subheader("Enter a transaction by hand")
    st.write("The `V1`–`V28` features come from a PCA transformation and are hard to set "
             "manually. You can leave them at 0 and adjust `Amount` and `Time`.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
    with col2:
        time = st.number_input("Time (seconds since first transaction)",
                               min_value=0.0, value=50000.0, step=1000.0)

    v_values = {f"V{i}": 0.0 for i in range(1, 29)}
    with st.expander("Advanced: set the V1–V28 features"):
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0,
                                                    format="%.4f", key=f"v{i}")

    if st.button("Predict", type="primary"):
        transaction = {"Time": time, **v_values, "Amount": amount}
        result = predict_module.predict_transaction(transaction, model_name=model_name)
        show_result(result)
