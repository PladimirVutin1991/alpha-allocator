import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

INPUT_PATH = 'data/ml_dataset.csv'
OUTPUT_PATH = 'data/model_predictions.csv'
MODEL_DIR = 'models'

TEST_START_DATE = '2024-01-01' 

def train_and_optimize():
    print("--- 1. Loading Data ---")
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    exclude_cols = ['Target_Return_Beat', 'Ticker', 'Future_Ret']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Splitting data at {TEST_START_DATE}...")
    train_data = df[df.index < TEST_START_DATE]
    test_data = df[df.index >= TEST_START_DATE]

    X_train = train_data[feature_cols]
    y_train = train_data['Target_Return_Beat']
    X_test = test_data[feature_cols]
    y_test = test_data['Target_Return_Beat']

    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    
    # ---- SANITY CHECKS ----
    print(f"DEBUG: Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(f"DEBUG: NaN values in X_train: {X_train.isnull().sum().sum()}")
    assert X_train.isnull().sum().sum() == 0, "Error: NaNs found in training data!"

    scaler = StandardScaler()

    # --- 2. MODEL: GRADIENT BOOSTING ---
    # Gradient Boosting (like XGBoost)
    print("--- 2. Training Gradient Boosting Model ---")
    
    gb = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=200,          
        early_stopping=True,   
        l2_regularization=1.0 
    )
    
    # We optimize for Precision (Win Rate)
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8],
        'min_samples_leaf': [20, 50]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        estimator=gb, param_distributions=param_dist, n_iter=10,
        scoring='precision', cv=tscv, verbose=1, random_state=42
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"✅ Best Params: {search.best_params_}")

    # --- 3. EVALUATION (TOP 10%) ---
    # NOTE: For this study, we define the 'Top Decile' dynamically based on the 
    # test set distribution to simulate a consistently active strategy (approx. 
    # 10% of capital deployed daily). In a production environment, this threshold 
    # would be fixed based on the training set (e.g., Score > 0.65).
    print("--- 3. Evaluating with 'Top 10% Conviction' ---")
    git init
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Dataframe to sort predictions by confidence
    eval_df = pd.DataFrame({
        'Target': y_test,
        'Confidence': y_prob
    }, index=X_test.index)
    
    # We only buy the Top 10% of stocks every day
    cutoff = eval_df['Confidence'].quantile(0.90)
    print(f"Top 10% Confidence Threshold: {cutoff:.4f}")
    
    eval_df['Predicted_Signal'] = (eval_df['Confidence'] >= cutoff).astype(int)
    
    # Metrics on the Test Set
    y_true = eval_df['Target']
    y_pred = eval_df['Predicted_Signal']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)

    print(f"Global Accuracy: {acc:.2%}")
    print(f"Precision (Win Rate of Top 10% Picks): {prec:.2%} (Target > 50%)")
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(best_model, f'{MODEL_DIR}/gb_model.pkl')

    # For Dashboard
    output_df = test_data.copy()
    output_df['Target'] = y_test
    output_df['Predicted_Signal'] = y_pred
    output_df['Confidence_Score'] = y_prob
    
    cols = ['Ticker', 'Target', 'Predicted_Signal', 'Confidence_Score'] + [c for c in output_df.columns if 'SEC_' in c]
    output_df[cols].to_csv(OUTPUT_PATH)
    print(f"\n✅ Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_optimize()