import sqlite3
import os
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "app.db")
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
def init_db():
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    conn = get_db()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        row_count INTEGER,
        uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        dataset_id INTEGER,
        reviewer_name TEXT,
        review_text TEXT NOT NULL,
        rating REAL,
        prediction TEXT,           -- 'Genuine' or 'Fake'
        confidence REAL,
        sentiment TEXT,            -- Positive / Negative / Neutral
        spam_score REAL,
        is_duplicate INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    );
    """)

    conn.commit()
    cur.execute("SELECT COUNT(*) AS c FROM users WHERE username = 'admin'")
    if cur.fetchone()["c"] == 0:
        from werkzeug.security import generate_password_hash
        cur.execute(
            "INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, 1)",
            ("admin", "admin@example.com", generate_password_hash("admin123")),
        )
        conn.commit()
    conn.close()

def log_activity(user_id, action, details=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO activity_log (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details),
    )
    conn.commit()
    conn.close()
