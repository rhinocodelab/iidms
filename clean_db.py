#!/usr/bin/env python3
"""
Simple database cleanup script
Run with: python clean_db.py
"""

import sqlite3
import hashlib
import os

def clean_database():
    db_path = "idms.db"
    
    if not os.path.exists(db_path):
        print("Database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("Users table does not exist. Database is already clean.")
            return
        
        # Get current user count
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        print(f"Found {total_users} users in database")
        
        if total_users == 0:
            print("Database is already empty.")
            return
        
        # Delete all users except admin (ID = 1)
        cursor.execute("DELETE FROM users WHERE id != 1")
        deleted = cursor.rowcount
        print(f"Deleted {deleted} users")
        
        # Reset admin user if it exists
        cursor.execute("SELECT id FROM users WHERE id = 1")
        if cursor.fetchone():
            admin_password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
            cursor.execute("""
                UPDATE users SET 
                    password_hash = ?,
                    has_changed_default_password = 0,
                    is_mfa_enabled = 0,
                    mfa_secret = NULL,
                    mfa_setup_complete = 0,
                    last_login = NULL,
                    email = 'admin@idmsdemo.com'
                WHERE id = 1
            """, (admin_password_hash,))
            print("Admin user reset to default state")
        else:
            print("No admin user found to reset")
        
        conn.commit()
        print("Database cleaned successfully!")
        print("Admin credentials: admin@idmsdemo.com / admin123")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    clean_database()
