"""
MFA (Multi-Factor Authentication) utilities for IDMS
Handles TOTP generation, QR code creation, and MFA validation
"""

import secrets
import pyotp
import qrcode
import io
import base64
import os
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MFAUtils:
    """Utility class for MFA operations"""
    
    @staticmethod
    def generate_mfa_secret() -> str:
        """Generate a new TOTP secret key"""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_qr_code(user_email: str, secret: str, app_name: str = "IDMS") -> str:
        """
        Generate QR code for MFA setup
        Returns the file path of the generated QR code image
        """
        try:
            # Create TOTP URI
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user_email,
                issuer_name=app_name
            )
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Save to temp directory
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            qr_filename = f"mfa_qr_{user_email.replace('@', '_').replace('.', '_')}.png"
            qr_path = os.path.join(temp_dir, qr_filename)
            
            img.save(qr_path)
            logger.info(f"QR code generated for {user_email}: {qr_path}")
            
            return qr_path
            
        except Exception as e:
            logger.error(f"Error generating QR code for {user_email}: {e}")
            raise
    
    @staticmethod
    def verify_totp_code(secret: str, code: str) -> bool:
        """
        Verify a TOTP code against the secret
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)  # Allow 1 window (30 seconds) tolerance
        except Exception as e:
            logger.error(f"Error verifying TOTP code: {e}")
            return False
    
    @staticmethod
    def cleanup_qr_code(qr_path: str) -> bool:
        """
        Remove QR code file after use
        """
        try:
            if os.path.exists(qr_path):
                os.remove(qr_path)
                logger.info(f"QR code cleaned up: {qr_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up QR code {qr_path}: {e}")
            return False
    
    @staticmethod
    def get_current_totp_code(secret: str) -> str:
        """
        Get current TOTP code for testing purposes
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.now()
        except Exception as e:
            logger.error(f"Error getting current TOTP code: {e}")
            return ""
