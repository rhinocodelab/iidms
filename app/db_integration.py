"""
Database Integration Module
Integrates the SQLite database with the existing IDMS application
"""

import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional
from database import db
import logging

logger = logging.getLogger(__name__)

class IDMSDataManager:
    """Manages data persistence for the IDMS application"""
    
    def __init__(self):
        self.db = db
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def save_document_processing(self, file_path: str, processing_result: Dict, 
                               processing_start_time: datetime, processing_end_time: datetime) -> int:
        """Save document processing information to database"""
        
        # Calculate file information
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Determine file type and MIME type
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.zip': 'application/zip',
            '.7z': 'application/x-7z-compressed',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml'
        }
        
        mime_type = mime_type_map.get(file_ext, 'application/octet-stream')
        
        # Calculate processing duration
        processing_duration = (processing_end_time - processing_start_time).total_seconds()
        
        # Prepare document data
        document_data = {
            'filename': filename,
            'original_filename': filename,
            'file_size': file_size,
            'file_type': file_ext,
            'mime_type': mime_type,
            'document_type': processing_result.get('document_type', 'Unknown'),
            'criticality_level': processing_result.get('criticality', 'Unknown'),
            'file_path': file_path,
            'processing_timestamp': processing_start_time.isoformat(),
            'processing_duration': processing_duration,
            'ai_confidence_score': processing_result.get('confidence_score'),
            'tags': processing_result.get('Tags', '').split(', ') if processing_result.get('Tags') else [],
            'summary': processing_result.get('summary', ''),
            'reasoning': processing_result.get('reasoning', ''),
            'is_archive': file_ext in ['.zip', '.7z', '.tar', '.gz', '.bz2', '.xz', '.rar'],
            'checksum': self.calculate_file_checksum(file_path) if file_path else ''
        }
        
        # Insert document record
        document_id = self.db.insert_document(document_data)
        
        # Log processing steps
        self.log_processing_step(document_id, 'file_upload', 'completed', 
                               processing_start_time, processing_start_time, 0)
        
        self.log_processing_step(document_id, 'ai_classification', 
                               'completed' if processing_result.get('document_type') else 'failed',
                               processing_start_time, processing_end_time, processing_duration,
                               {'document_type': processing_result.get('document_type'),
                                'confidence': processing_result.get('confidence_score')})
        
        # Log FileNet upload if attempted
        if 'filenet_upload' in processing_result:
            filenet_status = 'success' if processing_result['filenet_upload'] == 'Success' else 'failed'
            self.log_filenet_upload(document_id, 'classification', filenet_status,
                                  processing_result.get('filenet_upload'))
        
        return document_id
    
    def save_ai_document_processing(self, file_path: str, processing_result: Dict, 
                                  processing_start_time: datetime, processing_end_time: datetime, 
                                  user_data: Dict) -> int:
        """Save AI document processing information to ai_document_classifications table"""
        
        # Calculate file information
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Determine file type and MIME type
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.zip': 'application/zip',
            '.7z': 'application/x-7z-compressed',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml'
        }
        
        mime_type = mime_type_map.get(file_ext, 'application/octet-stream')
        
        # Calculate processing duration
        processing_duration = (processing_end_time - processing_start_time).total_seconds()
        
        # Prepare document data for ai_document_classifications table
        document_data = {
            'user_id': user_data['id'],
            'uploaded_by': user_data['username'],
            'filename': filename,
            'original_filename': filename,
            'file_size': file_size,
            'file_type': file_ext,
            'mime_type': mime_type,
            'document_type': processing_result.get('document_type', 'Unknown'),
            'criticality_level': processing_result.get('criticality', 'Unknown'),
            'file_path': file_path,
            'processing_timestamp': processing_start_time.isoformat(),
            'processing_duration': processing_duration,
            'ai_confidence_score': processing_result.get('confidence_score'),
            'tags': processing_result.get('Tags', '').split(', ') if processing_result.get('Tags') else [],
            'summary': processing_result.get('summary', ''),
            'reasoning': processing_result.get('reasoning', ''),
            'processing_status': 'completed' if processing_result.get('document_type') else 'failed',
            'ai_analysis_result': processing_result,
            'filenet_upload_status': 'success' if processing_result.get('filenet_upload') == 'Success' else 'pending',
            'filenet_document_id': processing_result.get('filenet_document_id', ''),
            'error_message': processing_result.get('error_message', '')
        }
        
        # Insert document record into ai_document_classifications table
        document_id = self.db.insert_ai_document_classification(document_data)
        
        # Log processing steps
        self.log_processing_step(document_id, 'file_upload', 'completed', 
                               processing_start_time, processing_start_time, 0)
        
        self.log_processing_step(document_id, 'ai_classification', 
                               'completed' if processing_result.get('document_type') else 'failed',
                               processing_start_time, processing_end_time, processing_duration,
                               {'document_type': processing_result.get('document_type'),
                                'confidence': processing_result.get('confidence_score')})
        
        # Log FileNet upload if attempted
        if 'filenet_upload' in processing_result:
            filenet_status = 'success' if processing_result['filenet_upload'] == 'Success' else 'failed'
            self.log_filenet_upload(document_id, 'classification', filenet_status,
                                  processing_result.get('filenet_upload'))
        
        return document_id
    
    def log_processing_step(self, document_id: int, step: str, status: str,
                          start_time: datetime, end_time: datetime, duration: float,
                          details: Dict = None, error_message: str = None):
        """Log a processing step"""
        
        log_data = {
            'document_id': document_id,
            'processing_step': step,
            'status': status,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': duration,
            'details': details or {},
            'error_message': error_message
        }
        
        self.db.insert_processing_log(log_data)
    
    def log_filenet_upload(self, document_id: int, upload_type: str, status: str,
                          result_message: str = None, queue_id: str = None):
        """Log FileNet upload attempt"""
        
        upload_data = {
            'document_id': document_id,
            'upload_type': upload_type,
            'queue_id': queue_id,
            'upload_status': status,
            'upload_timestamp': datetime.now().isoformat(),
            'error_message': result_message if status == 'failed' else None
        }
        
        if status == 'success':
            upload_data['completion_timestamp'] = datetime.now().isoformat()
        
        self.db.insert_filenet_upload(upload_data)
    
    def log_system_metric(self, metric_name: str, value: float, unit: str = None, additional_data: Dict = None):
        """Log a system metric"""
        
        metric_data = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'additional_data': additional_data or {}
        }
        
        self.db.insert_system_metric(metric_data)
    
    def log_error(self, error_type: str, error_message: str, severity: str = 'medium',
                 stack_trace: str = None, context_data: Dict = None):
        """Log a system error"""
        
        error_data = {
            'error_type': error_type,
            'error_message': error_message,
            'severity': severity,
            'stack_trace': stack_trace,
            'context_data': context_data or {}
        }
        
        self.db.insert_error_log(error_data)
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict]:
        """Get recently processed documents"""
        return self.db.get_documents(limit=limit)
    
    def get_document_statistics(self) -> Dict:
        """Get document processing statistics"""
        return self.db.get_system_stats()
    
    def get_processing_errors(self, limit: int = 50) -> List[Dict]:
        """Get recent processing errors"""
        # This would need to be implemented in the database class
        # For now, return empty list
        return []
    
    def update_document_category_usage(self, category_name: str):
        """Update usage count for a document category"""
        # This would need to be implemented in the database class
        pass
    
    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for dashboard display"""
        stats = self.db.get_system_stats()
        
        # Add additional calculated metrics
        dashboard_metrics = {
            'total_documents': stats['total_documents'],
            'processed_documents': stats['processed_documents'],
            'success_rate': stats['success_rate'],
            'avg_processing_time': stats['avg_processing_time'],
            'total_categories': stats['total_categories'],
            'filenet_uploads': stats['successful_uploads'],
            'processing_errors': 0,  # Would need to calculate from error logs
            'system_uptime': 0,  # Would need to track from system metrics
            'last_24h_documents': 0,  # Would need to query with date filter
            'last_24h_errors': 0  # Would need to query with date filter
        }
        
        return dashboard_metrics

# Global data manager instance
data_manager = IDMSDataManager()
