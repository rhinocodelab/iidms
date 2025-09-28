#!/usr/bin/env python3
"""
Remove specific users from app/idms.db
Removes user1, user2, user3 from the users table
"""

import sqlite3
import os

def remove_specific_users():
    db_path = "app/idms.db"
    
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get current users
        cursor.execute("SELECT id, username, email, role FROM users")
        users = cursor.fetchall()
        
        print("Current users in database:")
        for user in users:
            print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Role: {user[3]}")
        
        # Remove user1, user2, user3
        users_to_remove = ['user1', 'user2', 'user3']
        removed_count = 0
        
        print(f"\nRemoving users: {users_to_remove}")
        
        for username in users_to_remove:
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            if cursor.rowcount > 0:
                print(f"Removed user: {username}")
                removed_count += 1
            else:
                print(f"User {username} not found")
        
        conn.commit()
        print(f"\nRemoved {removed_count} users successfully!")
        
        # Show remaining users
        cursor.execute("SELECT id, username, email, role FROM users")
        remaining_users = cursor.fetchall()
        
        if remaining_users:
            print(f"\nRemaining users ({len(remaining_users)}):")
            for user in remaining_users:
                print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Role: {user[3]}")
        else:
            print("\nNo users remaining in database.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    remove_specific_users()
