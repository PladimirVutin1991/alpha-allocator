import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    if not os.path.exists('data/master_dataset.csv'):
        print("Error: Dataset not found.")
        return

    df = pd.read_csv('data/master_dataset.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Temporal Split
    train_data = df[df.index.year < 2024]
    test_data = df[df.index.year == 2024]
    
    exclude_cols = ['Target', 'Ticker', 'Price']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train = train_data[feature_cols]
    y_train = train_data['Target']
    X_test = test_data[feature_cols]
    y_test = test_data['Target']
    
    print(f"Training on {len(X_train)} samples")
    
    # --- MODEL TUNING ---
    # class_weight='balanced': Fixes the issue where model is lazy and just says "Sell"
    # n_estimators=200: More trees = smoother predictions
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=6, 
        min_samples_leaf=10, 
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy (2024): {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))
    
    # Save Predictions
    test_data = test_data.copy()
    test_data['Predicted_Signal'] = y_pred
    test_data['Confidence_Score'] = y_prob
    test_data.to_csv('data/model_predictions.csv')
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(rf, 'models/random_forest.pkl')

if __name__ == "__main__":
    train_model()