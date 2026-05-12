import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_engineer():
    """
    Loads raw data from SQLite and creates new features
    that will make our statistical analysis more meaningful.
    
    New features we create:
    - engagement_score   : overall measure of how engaged a student is
    - completion_velocity: how efficiently a student completes modules
    - risk_flag          : 1 if student shows signs of struggling
    """
    
    conn     = sqlite3.connect("database/eduPulse.db")
    students = pd.read_sql("SELECT * FROM students", conn)
    sessions = pd.read_sql("SELECT * FROM sessions", conn)
    modules  = pd.read_sql("SELECT * FROM modules",  conn)
    conn.close()
    
    print(f"Loaded {len(students)} students, {len(sessions)} sessions")
    
    # ── Per-student aggregation ──────────────────────────────
    # Group all sessions by student to get one row per student
    student_stats = sessions.groupby("student_id").agg(
        total_sessions    = ("session_id",       "count"),
        avg_duration      = ("duration_minutes", "mean"),
        total_duration    = ("duration_minutes", "sum"),
        avg_quiz_score    = ("quiz_score",        "mean"),
        total_videos      = ("videos_watched",    "sum"),
        completion_rate   = ("completed",         "mean"),
        unique_modules    = ("module_id",         "nunique"),
        avg_login_hour    = ("login_hour",        "mean")
    ).reset_index()
    
    # ── Engagement Score ─────────────────────────────────────
    # A composite score combining multiple behavioral signals
    # Each component is normalized 0-1 so they contribute equally
    
    scaler = MinMaxScaler()
    
    cols_to_scale = [
        "total_sessions","avg_duration","avg_quiz_score",
        "total_videos","completion_rate","unique_modules"
    ]
    
    scaled = scaler.fit_transform(student_stats[cols_to_scale])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in cols_to_scale])
    
    # Weighted average — quiz score and completion matter most
    student_stats["engagement_score"] = (
        scaled_df["total_sessions_scaled"]  * 0.15 +
        scaled_df["avg_duration_scaled"]    * 0.20 +
        scaled_df["avg_quiz_score_scaled"]  * 0.30 +
        scaled_df["total_videos_scaled"]    * 0.10 +
        scaled_df["completion_rate_scaled"] * 0.20 +
        scaled_df["unique_modules_scaled"]  * 0.05
    ).round(4)
    
    # ── Completion Velocity ──────────────────────────────────
    # How much a student completes per hour of study
    # High velocity = efficient learner
    student_stats["completion_velocity"] = (
        student_stats["completion_rate"] /
        (student_stats["avg_duration"] / 60 + 0.001)
    ).round(4)
    
    # ── Risk Flag ────────────────────────────────────────────
    # A student is flagged as at-risk if they show warning signs
    # This is our target variable for the prediction model
    student_stats["risk_flag"] = (
        (student_stats["avg_quiz_score"]  < 50) |
        (student_stats["completion_rate"] < 0.3) |
        (student_stats["total_sessions"]  < 5)
    ).astype(int)
    
    # ── Merge with student demographics ─────────────────────
    final = students.merge(student_stats, on="student_id", how="inner")
    
    # ── Encode categorical columns for modeling ──────────────
    # Machine learning models need numbers, not text
    final["gender_encoded"]  = final["gender"].map(
        {"Male":0, "Female":1, "Other":2}
    )
    final["device_encoded"]  = final["device_type"].map(
        {"Mobile":0, "Desktop":1, "Tablet":2}
    )
    
    print(f" Feature engineering complete")
    print(f"   Total students with features : {len(final)}")
    print(f"   At-risk students flagged     : {final['risk_flag'].sum()}")
    print(f"   Engagement score range       : {final['engagement_score'].min():.3f} — {final['engagement_score'].max():.3f}")
    print(f"\nNew columns added:")
    new_cols = ["engagement_score","completion_velocity","risk_flag",
                "gender_encoded","device_encoded"]
    for c in new_cols:
        print(f"   • {c}")
    
    return final, sessions, modules


if __name__ == "__main__":
    df, sessions, modules = load_and_engineer()
    print("\nSample rows:")
    print(df[["student_id","name","engagement_score",
              "completion_velocity","risk_flag"]].head(5).to_string(index=False))
