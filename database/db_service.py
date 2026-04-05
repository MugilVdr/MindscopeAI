import hashlib
import sqlite3
from hmac import compare_digest
from pathlib import Path
from secrets import token_hex


DB_NAME = str(Path(__file__).resolve().parent.parent / "mindscope.db")
PBKDF2_ITERATIONS = 120000


def get_connection():
    return sqlite3.connect(DB_NAME)


def _hash_password(password, salt=None):
    salt = salt or token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def _verify_password(password, stored_password):
    if not stored_password:
        return False

    if not stored_password.startswith("pbkdf2_sha256$"):
        return compare_digest(password, stored_password)

    _, iterations, salt, digest = stored_password.split("$", 3)
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        int(iterations)
    ).hex()
    return compare_digest(candidate, digest)


def register_user(name, username, email, password):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        password_hash = _hash_password(password)
        cursor.execute("""
        INSERT INTO users (name, username, email, password)
        VALUES (?, ?, ?, ?)
        """, (name, username, email, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, name, password FROM users
    WHERE username=?
    """, (username,))
    user = cursor.fetchone()

    if not user or not _verify_password(password, user[2]):
        conn.close()
        return None

    if not user[2].startswith("pbkdf2_sha256$"):
        cursor.execute(
            "UPDATE users SET password=? WHERE id=?",
            (_hash_password(password), user[0])
        )
        conn.commit()

    conn.close()
    return user[:2]


def save_prediction(
    user_id,
    text,
    face_emotion,
    mental_state,
    insight,
    text_confidence=None,
    face_confidence=None,
    support_level=None,
    input_source=None,
    urgency_score=None,
    triage_level=None,
    triage_reason=None,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO emotion_logs (
        user_id, user_text, face_emotion, mental_state, insight,
        text_confidence, face_confidence, support_level, input_source,
        urgency_score, triage_level, triage_reason
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        text,
        face_emotion,
        mental_state,
        insight,
        text_confidence,
        face_confidence,
        support_level,
        input_source,
        urgency_score,
        triage_level,
        triage_reason,
    ))
    conn.commit()
    conn.close()


def get_user_history(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        user_text,
        face_emotion,
        mental_state,
        insight,
        text_confidence,
        face_confidence,
        support_level,
        input_source,
        urgency_score,
        triage_level,
        triage_reason,
        created_at
    FROM emotion_logs
    WHERE user_id=?
    ORDER BY created_at DESC, id DESC
    """, (user_id,))
    data = cursor.fetchall()
    conn.close()
    return data


def get_emotion_counts(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT mental_state, COUNT(*)
    FROM emotion_logs
    WHERE user_id=?
    GROUP BY mental_state
    """, (user_id,))
    data = cursor.fetchall()
    conn.close()
    return data


def get_latest_emotion(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT mental_state
    FROM emotion_logs
    WHERE user_id=?
    ORDER BY created_at DESC, id DESC
    LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_recent_predictions(user_id, limit=10):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        mental_state,
        text_confidence,
        face_emotion,
        face_confidence,
        support_level,
        urgency_score,
        triage_level,
        created_at
    FROM emotion_logs
    WHERE user_id=?
    ORDER BY created_at DESC, id DESC
    LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_trend_points(user_id, limit=30):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        created_at,
        mental_state,
        text_confidence,
        support_level,
        urgency_score,
        triage_level
    FROM emotion_logs
    WHERE user_id=?
    ORDER BY created_at DESC, id DESC
    LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return list(reversed(rows))
