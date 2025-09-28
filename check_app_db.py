#!/usr/bin/env python3
"""
Check app/idms.db database contents
"""

import sqlite3
import os

def check_app_database():
    db_path = "app/idms.db"
    
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in database.")
            return
        
        print(f"Tables in {db_path}:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check if users table exists
        if ('users',) in tables:
            print("\nUsers table found. Checking users...")
            cursor.execute("SELECT id, username, email, role, is_active FROM users")
            users = cursor.fetchall()
            
            if users:
                print(f"Found {len(users)} users:")
                for user in users:
                    print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Role: {user[3]}, Active: {user[4]}")
            else:
                print("No users found in users table.")
        else:
            print("\nUsers table does not exist.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_app_database()
