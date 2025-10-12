# utils/db.py
import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()  # reads .env at project root

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "mechanical_inventory"),
}

def get_conn():
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            class_name VARCHAR(255) UNIQUE,
            qty INT NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(512),
            class_name VARCHAR(255),
            qty INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

# keep your existing imports and DB_CONFIG (use mech_user + its password in .env)

def update_inventory(class_name: str, delta: int, filename: str = "upload", score_avg=None):
    """Add delta to item_count for the given category_name (class_name)."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        # inventory update uses your schema names
        cur.execute(
            """
            INSERT INTO inventory (category_name, item_count)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE item_count = item_count + VALUES(item_count)
            """,
            (class_name, delta),
        )

        # detection log uses your schema names
        cur.execute(
            """
            INSERT INTO detection_log (image_filename, category_name, count_detected, score_avg)
            VALUES (%s, %s, %s, %s)
            """,
            (filename, class_name, delta, score_avg),
        )

        conn.commit()
    finally:
        cur.close()
        conn.close()


def fetch_inventory():
    """Return rows with keys class_name, qty so templates don’t change."""
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT 
                category_name AS class_name, 
                item_count    AS qty
            FROM inventory
            ORDER BY category_name
            """
        )
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

