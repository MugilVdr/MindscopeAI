from database.db_service import get_connection


def _ensure_column(cursor, table_name, column_name, column_definition):
    columns = {
        row[1]
        for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def ensure_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        username TEXT UNIQUE,
        email TEXT,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_text TEXT,
        face_emotion TEXT,
        mental_state TEXT,
        insight TEXT,
        text_confidence REAL,
        face_confidence REAL,
        support_level TEXT,
        input_source TEXT,
        urgency_score REAL,
        triage_level TEXT,
        triage_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    _ensure_column(cursor, "emotion_logs", "text_confidence", "REAL")
    _ensure_column(cursor, "emotion_logs", "face_confidence", "REAL")
    _ensure_column(cursor, "emotion_logs", "support_level", "TEXT")
    _ensure_column(cursor, "emotion_logs", "input_source", "TEXT")
    _ensure_column(cursor, "emotion_logs", "user_id", "INTEGER")
    _ensure_column(cursor, "emotion_logs", "urgency_score", "REAL")
    _ensure_column(cursor, "emotion_logs", "triage_level", "TEXT")
    _ensure_column(cursor, "emotion_logs", "triage_reason", "TEXT")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    ensure_database()
    print("Database and tables created successfully")
