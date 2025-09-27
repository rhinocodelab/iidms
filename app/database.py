"""
IDMS Database Models and Schema
SQLite database for storing document processing information, user data, and system metrics
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class IDMSDatabase:
    def __init__(self, db_path: str = "idms.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create all tables
        self.create_documents_table(cursor)
        self.create_processing_logs_table(cursor)
        self.create_document_categories_table(cursor)
        self.create_criticality_levels_table(cursor)
        self.create_filenet_uploads_table(cursor)
        self.create_system_metrics_table(cursor)
        self.create_user_sessions_table(cursor)
        self.create_error_logs_table(cursor)
        self.create_configuration_table(cursor)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def create_documents_table(self, cursor):
        """Documents table - stores information about processed documents"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                mime_type TEXT,
                document_type TEXT NOT NULL,
                criticality_level TEXT NOT NULL,
                file_path TEXT,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processing_timestamp DATETIME,
                processing_duration REAL,
                ai_confidence_score REAL,
                tags TEXT, -- JSON array of tags
                summary TEXT,
                reasoning TEXT,
                is_archive BOOLEAN DEFAULT 0,
                parent_archive_id INTEGER,
                checksum TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_archive_id) REFERENCES documents(id)
            )
        """)
    
    def create_processing_logs_table(self, cursor):
        """Processing logs table - tracks all processing activities"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                processing_step TEXT NOT NULL,
                status TEXT NOT NULL, -- 'started', 'completed', 'failed'
                start_time DATETIME,
                end_time DATETIME,
                duration REAL,
                details TEXT, -- JSON object with step details
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
    
    def create_document_categories_table(self, cursor):
        """Document categories table - tracks discovered document types"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL,
                description TEXT,
                first_discovered DATETIME DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 1,
                is_custom BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def create_criticality_levels_table(self, cursor):
        """Criticality levels table - manages security classification levels"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS criticality_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level_name TEXT UNIQUE NOT NULL,
                description TEXT,
                priority INTEGER NOT NULL, -- 1=Public, 5=Top Secret
                color_code TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def create_filenet_uploads_table(self, cursor):
        """FileNet uploads table - tracks FileNet upload attempts and results"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filenet_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                upload_type TEXT NOT NULL, -- 'classification', 'business'
                queue_id TEXT,
                upload_status TEXT NOT NULL, -- 'success', 'failed', 'pending'
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                completion_timestamp DATETIME,
                filenet_path TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
    
    def create_system_metrics_table(self, cursor):
        """System metrics table - stores performance and usage statistics"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                additional_data TEXT -- JSON object with extra metrics
            )
        """)
    
    def create_user_sessions_table(self, cursor):
        """User sessions table - tracks user interactions and sessions"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                pages_visited TEXT, -- JSON array of visited pages
                actions_performed TEXT, -- JSON array of actions
                documents_uploaded INTEGER DEFAULT 0,
                total_file_size INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def create_error_logs_table(self, cursor):
        """Error logs table - tracks system errors and exceptions"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                stack_trace TEXT,
                context_data TEXT, -- JSON object with error context
                severity TEXT NOT NULL, -- 'low', 'medium', 'high', 'critical'
                resolved BOOLEAN DEFAULT 0,
                resolution_notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME
            )
        """)
    
    def create_configuration_table(self, cursor):
        """Configuration table - stores system configuration and settings"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                config_type TEXT NOT NULL, -- 'string', 'integer', 'boolean', 'json'
                description TEXT,
                is_sensitive BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    # Document Operations
    def insert_document(self, document_data: Dict) -> int:
        """Insert a new document record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents (
                filename, original_filename, file_size, file_type, mime_type,
                document_type, criticality_level, file_path, processing_timestamp,
                processing_duration, ai_confidence_score, tags, summary, reasoning,
                is_archive, parent_archive_id, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document_data['filename'],
            document_data['original_filename'],
            document_data['file_size'],
            document_data['file_type'],
            document_data.get('mime_type'),
            document_data['document_type'],
            document_data['criticality_level'],
            document_data.get('file_path'),
            document_data.get('processing_timestamp'),
            document_data.get('processing_duration'),
            document_data.get('ai_confidence_score'),
            json.dumps(document_data.get('tags', [])),
            document_data.get('summary'),
            document_data.get('reasoning'),
            document_data.get('is_archive', 0),
            document_data.get('parent_archive_id'),
            document_data.get('checksum')
        ))
        
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Document inserted with ID: {document_id}")
        return document_id
    
    def get_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get documents with pagination"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM documents 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        documents = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return documents
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Get a specific document by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    # Processing Logs Operations
    def insert_processing_log(self, log_data: Dict) -> int:
        """Insert a processing log entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO processing_logs (
                document_id, processing_step, status, start_time, end_time,
                duration, details, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_data['document_id'],
            log_data['processing_step'],
            log_data['status'],
            log_data.get('start_time'),
            log_data.get('end_time'),
            log_data.get('duration'),
            json.dumps(log_data.get('details', {})),
            log_data.get('error_message')
        ))
        
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id
    
    # FileNet Upload Operations
    def insert_filenet_upload(self, upload_data: Dict) -> int:
        """Insert a FileNet upload record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO filenet_uploads (
                document_id, upload_type, queue_id, upload_status,
                upload_timestamp, completion_timestamp, filenet_path,
                error_message, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            upload_data['document_id'],
            upload_data['upload_type'],
            upload_data.get('queue_id'),
            upload_data['upload_status'],
            upload_data.get('upload_timestamp'),
            upload_data.get('completion_timestamp'),
            upload_data.get('filenet_path'),
            upload_data.get('error_message'),
            upload_data.get('retry_count', 0)
        ))
        
        upload_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return upload_id
    
    # System Metrics Operations
    def insert_system_metric(self, metric_data: Dict) -> int:
        """Insert a system metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics (
                metric_name, metric_value, metric_unit, additional_data
            ) VALUES (?, ?, ?, ?)
        """, (
            metric_data['metric_name'],
            metric_data['metric_value'],
            metric_data.get('metric_unit'),
            json.dumps(metric_data.get('additional_data', {}))
        ))
        
        metric_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return metric_id
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Document statistics
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_documents = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_timestamp IS NOT NULL")
        processed_documents = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(processing_duration) FROM documents WHERE processing_duration IS NOT NULL")
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # FileNet upload statistics
        cursor.execute("SELECT COUNT(*) FROM filenet_uploads WHERE upload_status = 'success'")
        successful_uploads = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM filenet_uploads")
        total_uploads = cursor.fetchone()[0]
        
        success_rate = (successful_uploads / total_uploads * 100) if total_uploads > 0 else 0
        
        # Document categories
        cursor.execute("SELECT COUNT(*) FROM document_categories")
        total_categories = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_documents': total_documents,
            'processed_documents': processed_documents,
            'avg_processing_time': round(avg_processing_time, 2),
            'successful_uploads': successful_uploads,
            'total_uploads': total_uploads,
            'success_rate': round(success_rate, 1),
            'total_categories': total_categories
        }
    
    def get_analytics_data(self) -> Dict:
        """Get comprehensive analytics data for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total documents processed today
        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE DATE(upload_timestamp) = DATE('now')
        """)
        processed_today = cursor.fetchone()[0]
        
        # Error rate calculation
        cursor.execute("SELECT COUNT(*) FROM processing_logs WHERE status = 'failed'")
        failed_processing = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM processing_logs")
        total_processing = cursor.fetchone()[0]
        
        error_rate = (failed_processing / total_processing * 100) if total_processing > 0 else 0
        
        # Document types distribution
        cursor.execute("""
            SELECT document_type, COUNT(*) as count 
            FROM documents 
            WHERE document_type IS NOT NULL AND document_type != 'Unknown'
            GROUP BY document_type 
            ORDER BY count DESC
        """)
        document_types = cursor.fetchall()
        
        # Criticality levels distribution
        cursor.execute("""
            SELECT criticality_level, COUNT(*) as count 
            FROM documents 
            WHERE criticality_level IS NOT NULL AND criticality_level != 'Unknown'
            GROUP BY criticality_level 
            ORDER BY count DESC
        """)
        criticality_levels = cursor.fetchall()
        
        # Processing trends (last 30 days)
        cursor.execute("""
            SELECT DATE(upload_timestamp) as date, COUNT(*) as count
            FROM documents 
            WHERE upload_timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(upload_timestamp)
            ORDER BY date ASC
        """)
        processing_trends = cursor.fetchall()
        
        # File types distribution
        cursor.execute("""
            SELECT file_type, COUNT(*) as count 
            FROM documents 
            WHERE file_type IS NOT NULL AND file_type != ''
            GROUP BY file_type 
            ORDER BY count DESC
        """)
        file_types = cursor.fetchall()
        
        conn.close()
        
        return {
            'processed_today': processed_today,
            'error_rate': round(error_rate, 1),
            'document_types': [{'name': row[0], 'count': row[1]} for row in document_types],
            'criticality_levels': [{'name': row[0], 'count': row[1]} for row in criticality_levels],
            'processing_trends': [{'date': row[0], 'count': row[1]} for row in processing_trends],
            'file_types': [{'name': row[0], 'count': row[1]} for row in file_types]
        }
    
    # Error Logging
    def insert_error_log(self, error_data: Dict) -> int:
        """Insert an error log entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO error_logs (
                error_type, error_message, stack_trace, context_data,
                severity, resolution_notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            error_data['error_type'],
            error_data['error_message'],
            error_data.get('stack_trace'),
            json.dumps(error_data.get('context_data', {})),
            error_data.get('severity', 'medium'),
            error_data.get('resolution_notes')
        ))
        
        error_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return error_id

    # Configuration Management
    def set_config(self, key: str, value: str, config_type: str = 'string', description: str = None):
        """Set a configuration value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO configuration (
                config_key, config_value, config_type, description, updated_at
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (key, value, config_type, description))
        
        conn.commit()
        conn.close()
    
    def get_config(self, key: str) -> Optional[str]:
        """Get a configuration value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT config_value FROM configuration WHERE config_key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None

# Global database instance
db = IDMSDatabase()
