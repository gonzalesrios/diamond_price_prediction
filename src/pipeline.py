"""MACHINE LEARNING TRAINING PIPELINE MODULE

This file packages our entire data preprocessing and model into one
Scikit-Learn Pipeline. It prevents data leakage, scales numbers,
encodes text, and trains our best DecisionTreeRegressor model.
"""

from pathlib import Path
import joblib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
# Import our cleaning function from File 1 (src/data_cleaning.py)
from src.data_cleaning import clean_diamond_data


def build_pipeline() -> Pipeline:
    """Builds and returns the full Scikit-Learn Pipeline."""

    # -------------------------------------------------------------------------
    # 1. ORGANIZE OUR FEATURE TYPES
    # -------------------------------------------------------------------------
    # Continuous numbers that need scaling:
    numerical_features = ["carat", "depth", "table", "x", "y", "z"]

    # Text categories that need to be converted to numbers:
    categorical_features = ["cut", "color", "clarity"]

    # -------------------------------------------------------------------------
    # 2. DEFINE ORDINAL RANKINGS (WORST TO BEST)
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS:
    # Unlike random categories (like red, blue, green), diamond grades have an
    # agreed-upon industry hierarchy. Higher grades mean higher quality, which
    # directly corresponds to higher market value:
    #   Cut:     Fair (0) is lowest, Ideal (4) is highest.
    #   Color:   J (0) has faint yellow, D (6) is completely colorless.
    #   Clarity: I1 (0) has visible flaws, IF (7) is internally flawless.
    cut_rank = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_rank = ["J", "I", "H", "G", "F", "E", "D"]
    clarity_rank = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    # -------------------------------------------------------------------------
    # 3. SET UP PREPROCESSING WITH ColumnTransformer
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS:
    # - StandardScaler: Centers numbers around 0 with a standard deviation of 1
    #   using z = (x - mean) / std. This stops large numbers like table (~60)
    #   from overpowering small numbers like carat (0.2 - 2.0).
    # - OrdinalEncoder: Converts the text words into our ranked numbers.
    # - ColumnTransformer: Applies both transformations at the exact same time.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_scaler", StandardScaler(), numerical_features),
            (
                "cat_encoder",
                OrdinalEncoder(categories=[cut_rank, color_rank, clarity_rank]),
                categorical_features,
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # 4. ASSEMBLE PIPELINE WITH OUR CHAMPION MODEL
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS FOR CHOOSING DECISION TREE:
    # In our assignments, Linear, Ridge, and Lasso regression hit an error floor
    # of ~$1,122 - $1,205 because diamond price scales non-linearly (curved).
    # The Decision Tree Regressor easily models non-linear steps.
    # We set min_samples_leaf=10 (found via GridSearchCV) to stop the tree from
    # memorizing noisy individual diamonds (overfitting).
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                DecisionTreeRegressor(min_samples_leaf=10, random_state=432),
            ),
        ]
    )

    return model_pipeline


def main():
    # -------------------------------------------------------------------------
    # STEP A: LOAD & CLEAN DATA
    # -------------------------------------------------------------------------
    print("Loading raw diamond data...")
    raw_df = sns.load_dataset("diamonds")

    # Clean impossible 0s using our logic from data_cleaning.py:
    clean_df = clean_diamond_data(raw_df)
    print(f"Data cleaned. Rows remaining: {len(clean_df):,}")

    # -------------------------------------------------------------------------
    # STEP B: TRAIN / TEST SPLIT (PREVENT DATA LEAKAGE)
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS:
    # Separate the target we want to predict ('price') from our input features:
    X = clean_df.drop(columns="price")
    y = clean_df["price"]

    # Split 80% for studying (train) and 20% for the final exam (test).
    # random_state=432 ensures the exact same split happens every time.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=432
    )

    # -------------------------------------------------------------------------
    # STEP C: TRAIN THE PIPELINE
    # -------------------------------------------------------------------------
    print("Building and training pipeline on X_train...")
    pipeline = build_pipeline()

    # The pipeline scales, encodes, and trains the tree in one clean command:
    pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # STEP D: EVALUATE ON UNSEEN TEST DATA
    # -------------------------------------------------------------------------
    print("Evaluating predictions on unseen test data...")
    y_pred = pipeline.predict(X_test)

    # RMSE tells us how many dollars off our predictions are on average:
    test_rmse = root_mean_squared_error(y_test, y_pred)
    # MAE gives the direct median dollar difference:
    test_mae = mean_absolute_error(y_test, y_pred)

    print("\n==========================================")
    print("        MODEL EVALUATION RESULTS          ")
    print("==========================================")
    print(f"Test RMSE: ${test_rmse:,.2f}  (Average dollar error)")
    print(f"Test MAE:  ${test_mae:,.2f}  (Mean absolute error)")

    # -------------------------------------------------------------------------
    # STEP E: SAVE MODEL ARTIFACT TO DISK
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS:
    # By saving the fitted pipeline with joblib, another developer can load it
    # and predict prices on new customer diamonds with a single function call!
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    export_path = models_dir / "diamond_pricing_pipeline.joblib"

    joblib.dump(pipeline, export_path)
    print(f"\nModel pipeline successfully saved to: {export_path}")


if __name__ == "__main__":
    main()
