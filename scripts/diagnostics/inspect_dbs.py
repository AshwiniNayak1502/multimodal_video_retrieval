"""Inspection script for SQLite databases."""

import sqlite3
from pathlib import Path

def inspect_db(db_path: Path):
    print(f"\n==================== {db_path} ====================")
    if not db_path.is_file():
        print("File does not exist!")
        return
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        tname = t[0]
        cols = c.execute(f"PRAGMA table_info({tname})").fetchall()
        cnt = c.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        print(f"\nTable: {tname} (rows: {cnt})")
        print(f"Columns: {[col[1] for col in cols]}")
        sample = c.execute(f"SELECT * FROM {tname} LIMIT 3").fetchall()
        for s in sample:
            print("  ", s)
    conn.close()

if __name__ == "__main__":
    inspect_db(Path("data/app.db"))
    inspect_db(Path("module2/output/vector_db/metadata.db"))
