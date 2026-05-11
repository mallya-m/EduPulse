import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta

# Faker() creates a fake data generator object
# We tell it to use Indian locale for realistic Indian names and cities
fake = Faker("en_IN")

# random.seed makes sure we get the same data every time we run this
# Without this, every run gives different data which causes inconsistency
random.seed(42)
fake.seed_instance(42)


def generate_students(num_students=200):
    """
    Creates a list of fake students.
    Each student is a tuple matching the students table columns.
    """
    students = []
    
    # Possible values to pick from randomly
    genders      = ["Male", "Female", "Other"]
    device_types = ["Mobile", "Desktop", "Tablet"]
    cities       = ["Bengaluru", "Mumbai", "Delhi", "Chennai", 
                    "Hyderabad", "Pune", "Kolkata", "Jaipur"]
    
    for i in range(num_students):
        # fake.name() generates a random realistic name
        name = fake.name()
        
        # random.randint(a, b) picks a random integer between a and b
        age = random.randint(18, 35)
        
        # random.choice picks one item randomly from a list
        gender   = random.choice(genders)
        location = random.choice(cities)
        device   = random.choice(device_types)
        
        # Generate a random date in the past 2 years as joined_date
        days_ago   = random.randint(30, 730)
        joined     = datetime.now() - timedelta(days=days_ago)
        joined_str = joined.strftime("%Y-%m-%d")
        
        # Each student is stored as a tuple — one row in the table
        students.append((name, age, gender, location, joined_str, device))
    
    print(f" Generated {len(students)} students")
    return students


def generate_modules():
    """
    Creates the list of course modules available on the platform.
    We hardcode these because we want specific realistic course names.
    """
    modules = [
        # (module_name, category, difficulty, total_videos, total_quizzes)
        ("Python Basics",            "Programming",   "Beginner",      20, 5),
        ("Data Science 101",         "Data Science",  "Beginner",      25, 6),
        ("Machine Learning",         "AI/ML",         "Intermediate",  30, 8),
        ("Statistics Fundamentals",  "Mathematics",   "Beginner",      18, 5),
        ("Deep Learning",            "AI/ML",         "Advanced",      35, 10),
        ("Data Visualization",       "Data Science",  "Intermediate",  22, 6),
        ("SQL and Databases",        "Programming",   "Beginner",      15, 4),
        ("Web Development",          "Programming",   "Intermediate",  28, 7),
        ("Cloud Computing",          "DevOps",        "Intermediate",  20, 5),
        ("Natural Language Processing", "AI/ML",      "Advanced",      32, 9),
    ]
    
    print(f" Generated {len(modules)} modules")
    return modules


def generate_sessions(num_students=200, num_modules=10, num_sessions=5000):
    """
    Creates fake study sessions.
    Each session = one student studying one module on one day.
    
    This is the most important table — it has all the behavioral data
    that we will run statistics on later.
    """
    sessions = []
    device_types = ["Mobile", "Desktop", "Tablet"]
    
    for i in range(num_sessions):
        # Pick a random student and module for this session
        # student IDs go from 1 to num_students (SQLite starts at 1)
        student_id = random.randint(1, num_students)
        module_id  = random.randint(1, num_modules)
        
        # Random session date in the past year
        days_ago     = random.randint(1, 365)
        session_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Duration between 5 and 180 minutes
        duration = random.randint(5, 180)
        
        # Videos watched — more time usually means more videos
        # min() makes sure we don't exceed 10 videos per session
        videos = min(int(duration / 15) + random.randint(0, 2), 10)
        
        # Quiz score between 0 and 100
        # We use a slight positive skew — most students score okay
        # random.gauss gives a bell curve centered at 65, spread of 20
        quiz_score = max(0, min(100, random.gauss(65, 20)))
        quiz_score = round(quiz_score, 2)
        
        # Completed: students who studied longer are more likely to finish
        # This creates a realistic relationship between duration and completion
        completed = 1 if duration > 60 and random.random() > 0.3 else 0
        
        # Login hour: 0-23 (what time of day they logged in)
        # Students tend to study morning or evening — weighted distribution
        login_hour = random.choices(
            range(24),
            # Higher weights for morning (8-10) and evening (19-22)
            weights=[1,1,1,1,1,1,1,2,4,4,3,2,2,2,2,2,3,3,4,5,5,4,2,1]
        )[0]
        
        device = random.choice(device_types)
        
        sessions.append((
            student_id, module_id, session_date, duration,
            videos, quiz_score, completed, login_hour, device
        ))
    
    print(f" Generated {len(sessions)} sessions")
    return sessions


if __name__ == "__main__":
    print("Starting data generation...\n")
    
    students = generate_students(200)
    modules  = generate_modules()
    sessions = generate_sessions(200, 10, 5000)
    
    print("\n All data generated successfully")
    print(f"   Students : {len(students)}")
    print(f"   Modules  : {len(modules)}")
    print(f"   Sessions : {len(sessions)}")
