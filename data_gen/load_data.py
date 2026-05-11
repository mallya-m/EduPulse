import sqlite3
import sys

# This imports the functions we just wrote in generate_data.py
sys.path.append(".")
from data_gen.generate_data import generate_students, generate_modules, generate_sessions


def load_all_data():
    """
    Connects to the database and inserts all generated data.
    """
    # Connect to the database Moni already created
    conn   = sqlite3.connect("database/eduPulse.db")
    cursor = conn.cursor()
    
    print("Connected to database\n")
    
    # ── Load Students ──────────────────────────────
    students = generate_students(200)
    
    # executemany inserts all rows in one go — much faster than a loop
    # The ? marks are placeholders — Python fills them in safely
    # This prevents something called SQL injection (a security issue)
    cursor.executemany("""
        INSERT INTO students (name, age, gender, location, joined_date, device_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, students)
    
    print(f" Inserted {len(students)} students")
    
    # ── Load Modules ───────────────────────────────
    modules = generate_modules()
    
    cursor.executemany("""
        INSERT INTO modules (module_name, category, difficulty, total_videos, total_quizzes)
        VALUES (?, ?, ?, ?, ?)
    """, modules)
    
    print(f" Inserted {len(modules)} modules")
    
    # ── Load Sessions ──────────────────────────────
    sessions = generate_sessions(200, 10, 5000)
    
    cursor.executemany("""
        INSERT INTO sessions (
            student_id, module_id, session_date, duration_minutes,
            videos_watched, quiz_score, completed, login_hour, device_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sessions)
    
    print(f" Inserted {len(sessions)} sessions")
    
    # Save everything to the database file
    conn.commit()
    
    # ── Verify the data is actually there ──────────
    print("\n── Verification ──────────────────────────")
    
    cursor.execute("SELECT COUNT(*) FROM students")
    print(f"   Students  in DB : {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM modules")
    print(f"   Modules   in DB : {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM sessions")
    print(f"   Sessions  in DB : {cursor.fetchone()[0]}")
    
    # Show a sample student so we can see the data looks real
    print("\n── Sample Student ────────────────────────")
    cursor.execute("SELECT * FROM students LIMIT 1")
    cols = [desc[0] for desc in cursor.description]
    row  = cursor.fetchone()
    for col, val in zip(cols, row):
        print(f"   {col:15} : {val}")
    
    conn.close()
    print("\n Database loaded successfully")


if __name__ == "__main__":
    load_all_data()
