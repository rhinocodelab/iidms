#!/usr/bin/env python3
"""
Database cleanup script to remove all users except admin
This script will:
1. Keep only the admin user (ID: 1)
2. Remove all other users from the database
3. Reset admin user to default state
"""

import sqlite3
import os
import sys
from pathlib import Path

def clean_user_database():
    """Clean the user database, keeping only admin user"""
    
    # Database path
    db_path = "idms.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found!")
        print("Make sure you're running this script from the project root directory.")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 Checking current users...")
        
        # Get current user count
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        print(f"📊 Total users in database: {total_users}")
        
        # Get admin user info
        cursor.execute("SELECT id, username, email, role FROM users WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        
        if not admin_user:
            print("❌ Admin user not found! Cannot proceed with cleanup.")
            return False
        
        print(f"👤 Admin user found: {admin_user[1]} ({admin_user[3]})")
        
        # Confirm deletion
        if total_users > 1:
            print(f"\n⚠️  WARNING: This will delete {total_users - 1} users!")
            print("Only the admin user will remain.")
            
            confirm = input("\n🤔 Are you sure you want to proceed? (yes/no): ").lower().strip()
            if confirm not in ['yes', 'y']:
                print("❌ Operation cancelled.")
                return False
        
        # Delete all users except admin
        print("\n🧹 Cleaning user database...")
        cursor.execute("DELETE FROM users WHERE id != 1")
        deleted_count = cursor.rowcount
        
        # Reset admin user to default state
        print("🔄 Resetting admin user to default state...")
        cursor.execute("""
            UPDATE users SET 
                password_hash = ?,
                has_changed_default_password = 0,
                is_mfa_enabled = 0,
                mfa_secret = NULL,
                mfa_setup_complete = 0,
                last_login = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (hashlib.sha256('admin123'.encode()).hexdigest(),))
        
        # Commit changes
        conn.commit()
        
        print(f"✅ Database cleanup completed!")
        print(f"🗑️  Deleted {deleted_count} users")
        print(f"👤 Admin user reset to default state")
        print(f"🔑 Admin credentials: admin@idmsdemo.com / admin123")
        print(f"🔒 MFA disabled, password change required on first login")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during database cleanup: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main function"""
    print("🧹 IDMS User Database Cleanup Tool")
    print("=" * 40)
    
    # Import hashlib for password hashing
    import hashlib
    
    # Check if we're in the right directory
    if not os.path.exists("idms.db"):
        print("❌ Database file not found!")
        print("Please run this script from the project root directory (where idms.db is located)")
        sys.exit(1)
    
    # Run cleanup
    success = clean_user_database()
    
    if success:
        print("\n🎉 Database cleanup completed successfully!")
        print("You can now start the application with a clean user database.")
    else:
        print("\n💥 Database cleanup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
