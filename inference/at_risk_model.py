import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(".")
from processing.features import load_and_engineer


def build_atrisk_model():
    """
    Builds a Logistic Regression model that predicts
    whether a student is at risk of poor performance.
    
    Features used → engagement_score, completion_velocity,
                     avg_duration, total_sessions, avg_quiz_score
    Target        → risk_flag (1=at risk, 0=not at risk)
    """
    
    df, sessions, modules = load_and_engineer()
    
    # Features and target
    feature_cols = [
        "engagement_score","completion_velocity",
        "avg_duration","total_sessions","completion_rate"
    ]
    
    X = df[feature_cols].fillna(0)
    y = df["risk_flag"]
    
    print(f"Dataset: {len(df)} students")
    print(f"At-risk : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Safe    : {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    
    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features — logistic regression needs scaled inputs
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n── Model Performance ──────────────────")
    print(f"   Accuracy : {accuracy*100:.1f}%")
    print(f"\n── Classification Report ──────────────")
    print(classification_report(y_test, y_pred,
                                 target_names=["Safe","At-Risk"]))
    
    # Feature importance
    print("── Feature Importance ─────────────────")
    for feat, coef in sorted(
        zip(feature_cols, model.coef_[0]),
        key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"   {feat:25} : {coef:+.4f}")
    
    # Save results
    report = f"""At-Risk Student Prediction Model
================================
Algorithm  : Logistic Regression
Students   : {len(df)}
At-risk    : {y.sum()} ({y.mean()*100:.1f}%)
Accuracy   : {accuracy*100:.1f}%

Feature Importance:
{chr(10).join([f"  {f}: {c:+.4f}" for f,c in zip(feature_cols, model.coef_[0])])}
"""
    with open("inference/atrisk_results.txt", "w") as f:
        f.write(report)
    
    print("\n At-risk model complete")
    return model, scaler, df


if __name__ == "__main__":
    model, scaler, df = build_atrisk_model()
