import sqlite3
import sys
sys.path.append(".")
from data_gen.generate_data import generate_students, generate_modules, generate_sessions


def load_all_data():
    conn   = sqlite3.connect("database/eduPulse.db")
    cursor = conn.cursor()
    
    # Clear existing data so we don't get duplicates on re-runs
    cursor.execute("DELETE FROM sessions")
    cursor.execute("DELETE FROM modules")
    cursor.execute("DELETE FROM students")
    cursor.execute("DELETE FROM sqlite_sequence")
    conn.commit()
    print(" Old data cleared\n")
    
    students = generate_students()
    cursor.executemany("""
        INSERT INTO students (name, age, gender, location, joined_date, device_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, students)
    print(f" Inserted {len(students)} students")
    conn.commit()
    
    modules = generate_modules()
    cursor.executemany("""
        INSERT INTO modules (module_name, category, difficulty, total_videos, total_quizzes)
        VALUES (?, ?, ?, ?, ?)
    """, modules)
    print(f" Inserted {len(modules)} modules")
    conn.commit()
    
    sessions = generate_sessions()
    cursor.executemany("""
        INSERT INTO sessions (
            student_id, module_id, session_date, duration_minutes,
            videos_watched, quiz_score, completed, login_hour, device_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sessions)
    print(f" Inserted {len(sessions)} sessions")
    conn.commit()
    
    print("\n── Verification ──────────────────────────")
    cursor.execute("SELECT COUNT(*) FROM students")
    print(f"   Students  in DB : {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM modules")
    print(f"   Modules   in DB : {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM sessions")
    print(f"   Sessions  in DB : {cursor.fetchone()[0]}")
    
    print("\n── Sample Student ────────────────────────")
    cursor.execute("SELECT * FROM students LIMIT 1")
    cols = [d[0] for d in cursor.description]
    row  = cursor.fetchone()
    for c, v in zip(cols, row):
        print(f"   {c:15} : {v}")
    
    conn.close()
    print("\n Database loaded with real OULAD data")


if __name__ == "__main__":
    load_all_data()
