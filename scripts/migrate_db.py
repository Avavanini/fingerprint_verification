"""
migrate_db.py — Add minutiae_template_str column to existing SQLite database.

Run this script once to migrate an existing database to support hybrid matching.
New databases created after this change will have the column automatically.

Usage:
    python scripts/migrate_db.py
"""

import os
import sys
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'fingerprints.db'


def migrate():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. No migration needed.")
        print("The column will be created automatically when the app starts.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(user_templates)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'minutiae_template_str' in columns:
        print("Column 'minutiae_template_str' already exists. No migration needed.")
        conn.close()
        return

    print("Adding 'minutiae_template_str' column to 'user_templates' table...")
    cursor.execute("ALTER TABLE user_templates ADD COLUMN minutiae_template_str TEXT")
    conn.commit()
    conn.close()

    print("Migration complete. Existing users will use embedding-only matching")
    print("until they re-enroll with the updated system.")


if __name__ == "__main__":
    migrate()
