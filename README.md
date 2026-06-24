# Credit Card Fraud Detection System Using Machine Learning

A Bachelor diploma project that detects fraudulent credit-card transactions using
machine learning. It trains and compares four models — **Logistic Regression,
Decision Tree, Random Forest, and an Artificial Neural Network (ANN)** — on the
well-known Kaggle credit-card dataset, and reports their performance using
Accuracy, Precision, Recall, F1-score and ROC-AUC.


---

## 0. Quick start on Windows (easiest)

If you are on Windows and just want it to run, you can skip the manual commands:

1. Make sure Python is installed (https://www.python.org/downloads/ — tick
   **"Add python.exe to PATH"** during install).
2. Put `creditcard.csv` (from Kaggle) into the `data\` folder.
3. Double-click **`setup.bat`** once (creates the environment and installs everything).
4. Double-click **`run.bat`** (trains the models and opens the web app).

If a `python` command ever opens the Microsoft Store instead of running, use
**`py`** instead of `python` — the batch files above already do this for you.

---

## 1. Project structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv          # the dataset (download from Kaggle, see below)
│
├── models/                     # created automatically when you train
│   ├── scaler.joblib           # the fitted StandardScaler
│   ├── fraud_models.joblib     # Logistic Regression, Decision Tree, Random Forest
│   ├── ann_model.keras         # the trained neural network
│   └── model_results.joblib    # (optional) saved metric values
│
├── outputs/                    # created automatically when you evaluate
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── f1_comparison.png
│   └── model_comparison.csv    # the metrics table
│
├── src/
│   ├── config.py               # all paths and settings in one place
│   ├── data_preprocessing.py   # load, scale, split, SMOTE
│   ├── train_models.py         # build and train the four models
│   ├── evaluate_models.py      # metrics, confusion matrices, ROC curves
│   └── predict.py              # classify a new transaction (prediction module)
│
├── explore_data.py             # exploratory data analysis (EDA) script
├── main.py                     # runs the whole pipeline end-to-end
├── app.py                      # optional Streamlit web interface (UI)
├── requirements.txt            # Python dependencies
└── README.md                   # this file
```

### What each file does

| File | Purpose |
|------|---------|
| `src/config.py` | Stores every path and setting (test size, random seed, columns to scale). Change something once here and the whole project follows. |
| `src/data_preprocessing.py` | Loads the CSV, checks for missing values and imbalance, scales `Time`/`Amount`, splits into train/test, and balances the training set with SMOTE. |
| `src/train_models.py` | Creates the four models with simple hyper-parameters, trains them on the balanced data, and saves them. |
| `src/evaluate_models.py` | Computes Accuracy, Precision, Recall, F1, ROC-AUC; draws confusion matrices, ROC curves and an F1 bar chart; writes the comparison CSV. |
| `src/predict.py` | The fraud-prediction module: loads a saved model and classifies a new transaction as `FRAUD` or `NORMAL`. |
| `explore_data.py` | Stand-alone data exploration (class balance, amount distribution, top correlated features). |
| `main.py` | Convenience script that runs preprocessing → training → evaluation → an example prediction in one go. |
| `app.py` | Optional **Streamlit web interface (UI)** for checking transactions in the browser. |

---

## 2. The dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle
(European cardholders, September 2013):

- **284,807** transactions over two days
- **492** of them are fraud — only **0.172%** of all rows (highly imbalanced)
- **30** input features:
  - `V1`–`V28`: anonymised features produced by a **PCA** transformation
  - `Time`: seconds elapsed since the first transaction
  - `Amount`: the transaction amount
- `Class`: the label — **1 = fraud**, **0 = normal**
- **No missing values**

Download it from
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
and place `creditcard.csv` inside the `data/` folder.

---

## 3. How to run

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. install the dependencies
pip install -r requirements.txt

# 3. (optional) explore the data first
python explore_data.py

# 4. run the full pipeline: preprocess -> train -> evaluate
python main.py

# 5. try a single prediction with the saved model (command line)
python -m src.predict

# 6. OR open the simple web interface (after main.py has been run)
streamlit run app.py
```

### The web interface (UI)

`app.py` is a small **Streamlit** web app. After you have run `python main.py`
once (so the models exist), start it with:

```bash
streamlit run app.py
```

It opens in your browser (usually `http://localhost:8501`) and lets you:

- **pick which model** to use (Random Forest, Logistic Regression, Decision Tree, ANN),
- **check a real transaction** taken from the dataset (and compare with its true label),
- **upload a CSV** of many transactions and download the predictions, or
- **type a transaction by hand** and get an instant FRAUD / NORMAL result with the
  fraud probability.

The command-line script (`python -m src.predict`) and the web app do exactly the
same prediction underneath — the UI is just a friendlier front end.

After `main.py` finishes, open the `outputs/` folder to see the metrics table
(`model_comparison.csv`) and all the figures.

---

## 4. Method, in short

1. **Preprocessing** — scale `Time` and `Amount` with `StandardScaler`
   (the `V` features are already PCA outputs); split 70% / 30% with
   stratification so both sets keep the same fraud ratio.
2. **Class imbalance** — apply **SMOTE** to the *training set only*, which
   creates synthetic fraud examples so the models stop ignoring the rare class.
   The test set is left untouched for an honest evaluation.
3. **Models** — Logistic Regression, Decision Tree, Random Forest, and a small
   Keras ANN (256 → 128 → 64 → 1).
4. **Evaluation** — because the data is so imbalanced, plain accuracy is
   misleading; the project focuses on **Precision, Recall, F1-score and
   ROC-AUC**, and shows confusion matrices and ROC curves for every model.

---

## 5. Notes

- Re-running `main.py` overwrites the saved models and figures.
- The random seed is fixed (`42`) so results are reproducible.
- If you do not want to use the neural network, you can skip installing
  TensorFlow — the other three models will still train and the ANN is simply
  skipped.
