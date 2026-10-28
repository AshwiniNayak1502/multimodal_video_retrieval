import sqlite3

conn = sqlite3.connect('module2/output/vector_db/metadata.db')
c = conn.cursor()
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables in metadata.db:", tables)
for t in tables:
    count = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    cols = [col[1] for col in c.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"Table {t} ({count} rows): {cols}")
conn.close()
