import sys
import os
sys.path.append('app')

from database import IDMSDatabase

# Initialize the database
db = IDMSDatabase('app/idms.db')
db.init_database()
print('Database initialized successfully')

# Check if the new table was created
import sqlite3
conn = sqlite3.connect('app/idms.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_ghostlayer_documents'")
table_exists = cursor.fetchone()

if table_exists:
    print('✅ user_ghostlayer_documents table created successfully')
else:
    print('❌ user_ghostlayer_documents table not found')

conn.close()
