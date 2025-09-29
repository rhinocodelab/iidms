import sqlite3

# Check what tables exist
conn = sqlite3.connect('app/idms.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print('=== EXISTING TABLES ===')
for table in tables:
    print(table[0])

conn.close()
