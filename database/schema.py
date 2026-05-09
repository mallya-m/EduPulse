import sqlite3
import os

# sqlite3 is built into Python — no installation needed
# It creates a database as a single .db file on your computer

def create_database():
    """
    This function creates the EduPulse database and all its tables.
    If the database already exists, it just connects to it.
    """
    
    # This creates (or connects to) a file called eduPulse.db
    # inside the database/ folder
    db_path = "database/eduPulse.db"
    conn = sqlite3.connect(db_path)
    
    # A cursor is like a pen — you use it to write SQL commands
    cursor = conn.cursor()
    
    # ─────────────────────────────────────────────
    # TABLE 1: students
    # Stores one row per student
    # ─────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            age          INTEGER NOT NULL,
            gender       TEXT    NOT NULL,
            location     TEXT    NOT NULL,
            joined_date  TEXT    NOT NULL,
            device_type  TEXT    NOT NULL
        )
    """)
    # INTEGER PRIMARY KEY AUTOINCREMENT means:
    #   each student gets a unique ID number (1, 2, 3...)
    #   automatically — you don't have to set it manually
    # NOT NULL means the field cannot be empty
    
    # ─────────────────────────────────────────────
    # TABLE 2: modules
    # Stores the courses available on the platform
    # ─────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS modules (
            module_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name  TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            difficulty   TEXT    NOT NULL,
            total_videos INTEGER NOT NULL,
            total_quizzes INTEGER NOT NULL
        )
    """)
    
    # ─────────────────────────────────────────────
    # TABLE 3: sessions
    # Stores every study session — one row per login
    # This is the most important table
    # ─────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id        INTEGER NOT NULL,
            module_id         INTEGER NOT NULL,
            session_date      TEXT    NOT NULL,
            duration_minutes  INTEGER NOT NULL,
            videos_watched    INTEGER NOT NULL,
            quiz_score        REAL    NOT NULL,
            completed         INTEGER NOT NULL,
            login_hour        INTEGER NOT NULL,
            device_type       TEXT    NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (module_id)  REFERENCES modules(module_id)
        )
    """)
    # REAL means a decimal number (like 87.5 for a quiz score)
    # completed is 0 or 1 (0 = not finished, 1 = finished)
    # login_hour is 0-23 (what hour of day they logged in)
    # FOREIGN KEY means student_id here must match a real student in students table
    #   this prevents orphan data — you can't have a session for a student that doesn't exist
    
    # Save all changes to the file
    conn.commit()
    
    # Close the connection when done
    conn.close()
    
    print(f" Database created at: {db_path}")
    print(" Tables created: students, modules, sessions")
    return db_path


def verify_database():
    """
    This function opens the database and checks what tables exist.
    Use it to confirm everything was created correctly.
    """
    conn = sqlite3.connect("database/eduPulse.db")
    cursor = conn.cursor()
    
    # This SQL command asks: what tables exist in this database?
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("\n Tables found in database:")
    for table in tables:
        table_name = table[0]
        # Count how many columns each table has
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"    {table_name} — {len(columns)} columns")
        for col in columns:
            # col[1] is the column name, col[2] is the data type
            print(f"      • {col[1]} ({col[2]})")
    
    conn.close()


# When Python runs this file directly, execute these two functions
if __name__ == "__main__":
    create_database()
    verify_database()
