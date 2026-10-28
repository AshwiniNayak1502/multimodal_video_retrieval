import sqlite3

conn = sqlite3.connect('module2/output/vector_db/metadata.db')
c = conn.cursor()
vid_id = 'vid_d9b1180af485c6cd'

print(f"=== Visual vectors for {vid_id} ===")
vis_rows = c.execute("SELECT vector_id, frame_id, timestamp_seconds, timestamp_formatted, frame_path FROM visual_vectors WHERE video_id = ?", (vid_id,)).fetchall()
print(f"Count: {len(vis_rows)}")
for r in vis_rows[:10]:
    print(" ", r)

print(f"\n=== Text vectors for {vid_id} ===")
txt_rows = c.execute("SELECT vector_id, segment_uid, start_timestamp, end_timestamp, start_formatted, end_formatted, transcript_text FROM text_vectors WHERE video_id = ?", (vid_id,)).fetchall()
print(f"Count: {len(txt_rows)}")
for r in txt_rows[:10]:
    print(" ", r)

conn.close()
