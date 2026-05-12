import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)
fake.seed_instance(42)


def generate_students(filepath="data_gen/raw/studentInfo.csv"):
    df = pd.read_csv(filepath)
    df = df[["id_student","gender","age_band","highest_education","final_result"]].copy()
    df.columns = ["real_id","gender","age_band","education","final_result"]
    df["gender"] = df["gender"].map({"M":"Male","F":"Female"}).fillna("Other")
    age_map = {"0-35":25,"35-55":45,"55<=":60}
    df["age"] = df["age_band"].map(age_map).fillna(25).astype(int)
    cities  = ["Bengaluru","Mumbai","Delhi","Chennai","Hyderabad","Pune","Kolkata","Jaipur"]
    devices = ["Mobile","Desktop","Tablet"]
    df["location"]    = [random.choice(cities)  for _ in range(len(df))]
    df["device_type"] = [random.choice(devices) for _ in range(len(df))]
    df["name"]        = [fake.name()             for _ in range(len(df))]
    df["joined_date"] = [
        (datetime.now() - timedelta(days=random.randint(100,900))).strftime("%Y-%m-%d")
        for _ in range(len(df))
    ]
    result = df[["name","age","gender","location","joined_date","device_type"]].head(500).reset_index(drop=True)
    print(f" Loaded {len(result)} real students from OULAD dataset")
    return [tuple(row) for row in result.values]


def generate_modules():
    modules = [
        ("Python Basics",               "Programming",  "Beginner",     20, 5),
        ("Data Science 101",            "Data Science", "Beginner",     25, 6),
        ("Machine Learning",            "AI/ML",        "Intermediate", 30, 8),
        ("Statistics Fundamentals",     "Mathematics",  "Beginner",     18, 5),
        ("Deep Learning",               "AI/ML",        "Advanced",     35,10),
        ("Data Visualization",          "Data Science", "Intermediate", 22, 6),
        ("SQL and Databases",           "Programming",  "Beginner",     15, 4),
        ("Web Development",             "Programming",  "Intermediate", 28, 7),
        ("Cloud Computing",             "DevOps",       "Intermediate", 20, 5),
        ("Natural Language Processing", "AI/ML",        "Advanced",     32, 9),
    ]
    print(f" Generated {len(modules)} modules")
    return modules


def generate_sessions(filepath="data_gen/raw/studentVle.csv", num_students=500, num_modules=10):
    try:
        # Read only first 200000 rows to avoid memory issues
        df_vle = pd.read_csv(filepath, nrows=200000)
        print(f"   VLE data loaded: {len(df_vle)} records (capped at 200k)")

        df_sessions = df_vle.groupby(["id_student","date"]).agg(
            total_clicks=("sum_click","sum")
        ).reset_index()

        real_ids   = df_sessions["id_student"].unique()
        id_mapping = {rid: (i % num_students)+1 for i, rid in enumerate(real_ids)}
        df_sessions["student_id"] = df_sessions["id_student"].map(id_mapping)

        df_sessions["duration_minutes"] = (df_sessions["total_clicks"] * 1.5).clip(5,180).astype(int)
        df_sessions["module_id"]        = np.random.randint(1, num_modules+1, size=len(df_sessions))
        df_sessions["videos_watched"]   = (df_sessions["duration_minutes"] / 15).clip(0,10).astype(int)

        base   = df_sessions["total_clicks"].clip(0,100)
        noise  = np.random.normal(0, 15, len(df_sessions))
        df_sessions["quiz_score"] = (base * 0.5 + 40 + noise).clip(0,100).round(2)

        df_sessions["completed"]   = (df_sessions["duration_minutes"] > 60).astype(int)
        
        # Simple randint for login_hour — no probability weights needed
        df_sessions["login_hour"]  = np.random.randint(0, 24, size=len(df_sessions))
        
        df_sessions["device_type"] = np.random.choice(
            ["Mobile","Desktop","Tablet"], size=len(df_sessions)
        )

        base_date = datetime(2024,1,1)
        df_sessions["session_date"] = df_sessions["date"].apply(
            lambda d: (base_date + timedelta(days=int(abs(d)))).strftime("%Y-%m-%d")
        )

        df_sessions = df_sessions.head(8000)

        final = df_sessions[[
            "student_id","module_id","session_date","duration_minutes",
            "videos_watched","quiz_score","completed","login_hour","device_type"
        ]]

        print(f"Generated {len(final)} sessions from real OULAD data")
        return [tuple(row) for row in final.values]

    except FileNotFoundError:
        print("   VLE file not found — using synthetic sessions")
        sessions = []
        for _ in range(8000):
            student_id = random.randint(1, num_students)
            module_id  = random.randint(1, num_modules)
            days_ago   = random.randint(1, 365)
            date_str   = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            duration   = random.randint(5, 180)
            videos     = min(int(duration/15) + random.randint(0,2), 10)
            score      = round(max(0, min(100, random.gauss(65,20))), 2)
            completed  = 1 if duration > 60 and random.random() > 0.3 else 0
            login_hour = random.randint(0, 23)
            device     = random.choice(["Mobile","Desktop","Tablet"])
            sessions.append((student_id, module_id, date_str, duration,
                             videos, score, completed, login_hour, device))
        print(f" Generated {len(sessions)} synthetic sessions")
        return sessions


if __name__ == "__main__":
    s = generate_students()
    m = generate_modules()
    sess = generate_sessions()
    print(f"\nFinal — Students:{len(s)} Modules:{len(m)} Sessions:{len(sess)}")
