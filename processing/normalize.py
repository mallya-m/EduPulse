import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sqlite3
import sys
sys.path.append(".")

def normalize_features():
    """
    Takes raw session data and applies two types of normalization:
    
    1. MinMaxScaler  → scales values to 0-1 range
       Used for: engagement score components
       Why: keeps proportional relationships
    
    2. StandardScaler → scales to mean=0, std=1
       Used for: features going into ML model
       Why: logistic regression needs standardized inputs
    """
    
    conn     = sqlite3.connect("database/eduPulse.db")
    sessions = pd.read_sql("SELECT * FROM sessions", conn)
    conn.close()
    
    print(f"Normalizing {len(sessions)} session records...")
    
    # Columns to normalize
    numeric_cols = ["duration_minutes","videos_watched","quiz_score","login_hour"]
    
    # MinMax normalization — results in values between 0 and 1
    minmax   = MinMaxScaler()
    mm_data  = minmax.fit_transform(sessions[numeric_cols])
    mm_df    = pd.DataFrame(mm_data,
                            columns=[f"{c}_minmax" for c in numeric_cols])
    
    # Standard normalization — results in mean=0, std=1
    standard = StandardScaler()
    std_data = standard.fit_transform(sessions[numeric_cols])
    std_df   = pd.DataFrame(std_data,
                             columns=[f"{c}_standard" for c in numeric_cols])
    
    # Combine
    result = pd.concat([sessions[numeric_cols], mm_df, std_df], axis=1)
    
    print("\n── MinMax Normalized (0-1 range) ──────────────")
    print(mm_df.describe().round(3).to_string())
    
    print("\n── Standard Normalized (mean=0, std=1) ────────")
    print(std_df.describe().round(3).to_string())
    
    print("\nNormalization complete")
    print("   MinMaxScaler  → used in engagement score calculation")
    print("   StandardScaler → used in at-risk logistic regression model")
    
    return result, minmax, standard


if __name__ == "__main__":
    result, mm, std = normalize_features()
    print("\nSample normalized values:")
    print(result.head(3).to_string())
