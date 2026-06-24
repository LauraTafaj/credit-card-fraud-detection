"""
explore_data.py
===============
A standalone script that explores the dataset BEFORE any modelling. Running it
answers the questions every thesis "Dataset Analysis" section needs:

    * How big is the dataset and what columns does it have?
    * Are there missing values?
    * How imbalanced are the classes?
    * What do 'Time' and 'Amount' look like?
    * Which features are most correlated with fraud?

It saves two figures into the outputs/ folder:
    - class_distribution.png
    - amount_distribution.png

Run it with:   python explore_data.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config          # works when run from the project root folder


def main():
    import pandas as pd

    config.ensure_directories()
    print("Loading data ...")
    df = pd.read_csv(config.DATA_FILE)

    # ----- 1. Shape and columns -------------------------------------------
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Columns:", ", ".join(df.columns))

    # ----- 2. Missing values ----------------------------------------------
    print(f"\nMissing values in the whole dataset: {df.isnull().sum().sum()}")

    # ----- 3. Class balance -----------------------------------------------
    counts = df[config.TARGET_COLUMN].value_counts()
    n_fraud = int(counts.get(1, 0))
    n_total = len(df)
    print(f"\nNormal (0): {counts.get(0, 0):,}")
    print(f"Fraud  (1): {n_fraud:,}  ({100 * n_fraud / n_total:.3f}% of all rows)")

    plt.figure(figsize=(5, 4))
    plt.bar(["Normal (0)", "Fraud (1)"], [counts.get(0, 0), n_fraud],
            color=["#4C72B0", "#C44E52"])
    plt.ylabel("Number of transactions")
    plt.title("Class distribution (note the huge imbalance)")
    for i, v in enumerate([counts.get(0, 0), n_fraud]):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.tight_layout()
    p1 = os.path.join(config.OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # ----- 4. Amount summary ----------------------------------------------
    print("\n'Amount' summary:")
    print(df["Amount"].describe().to_string())

    plt.figure(figsize=(6, 4))
    # log scale because most amounts are small but a few are very large
    plt.hist(df[df[config.TARGET_COLUMN] == 0]["Amount"], bins=50,
             alpha=0.6, label="Normal", color="#4C72B0", log=True)
    plt.hist(df[df[config.TARGET_COLUMN] == 1]["Amount"], bins=50,
             alpha=0.6, label="Fraud", color="#C44E52", log=True)
    plt.xlabel("Transaction amount")
    plt.ylabel("Count (log scale)")
    plt.title("Transaction amount: fraud vs normal")
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(config.OUTPUTS_DIR, "amount_distribution.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # ----- 5. Features most correlated with fraud -------------------------
    corr = df.corr(numeric_only=True)[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    top = corr.abs().sort_values(ascending=False).head(8)
    print("\nTop 8 features most correlated with the fraud label:")
    for feat in top.index:
        print(f"   {feat:>8}: {corr[feat]:+.3f}")

    print("\nExploration finished. See the outputs/ folder for the figures.")


if __name__ == "__main__":
    main()
