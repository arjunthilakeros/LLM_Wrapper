"""
S3 Storage Module for Terminal Chatbot
Handles file uploads and downloads to S3-compatible storage.
"""

import os
import io
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, BinaryIO
from urllib.parse import urljoin

try:
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

from logger import get_logger
from exceptions import ChatbotError, FileProcessingError

logger = get_logger()

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")
BUCKET_NAME = os.getenv("S3_BUCKET")
S3_BASE_URL = f"{S3_ENDPOINT}/{BUCKET_NAME}/LLM/"
PRE_SIGNED_URL_EXPIRATION = int(os.getenv("S3_URL_EXPIRATION", "86400"))
S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() == "true"


class StorageError(ChatbotError):
    """Storage-specific errors."""
    pass


class S3Storage:
    """S3-compatible storage manager."""

    _instance: Optional['S3Storage'] = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(
        cls,
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket_name: str = None,
        region: str = "us-east-1"
    ) -> 'S3Storage':
        """Initialize S3 client."""
        if not S3_AVAILABLE:
            raise StorageError(
                "S3 support requires boto3. Install with: pip install boto3"
            )

        if cls._client is not None:
            return cls._instance

        resolved_endpoint = endpoint_url or S3_ENDPOINT
        resolved_access_key = access_key or ACCESS_KEY
        resolved_secret_key = secret_key or SECRET_KEY

        if not all([resolved_endpoint, resolved_access_key, resolved_secret_key]):
            raise StorageError(
                "S3 credentials required: set S3_ENDPOINT, S3_ACCESS_KEY, and S3_SECRET_KEY environment variables"
            )

        try:
            cls._client = boto3.client(
                's3',
                endpoint_url=resolved_endpoint,
                aws_access_key_id=resolved_access_key,
                aws_secret_access_key=resolved_secret_key,
                region_name='us-east-1', # Many compatible S3s default to this
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'},
                    request_checksum_calculation='when_required',
                    response_checksum_validation='when_required'
                )
            )
            cls._bucket = bucket_name or BUCKET_NAME
            cls._base_path = "LLM/"

            logger.info(f"S3 storage initialized: {resolved_endpoint}")
            
            # Ensure instance is created
            if cls._instance is None:
                cls._instance = cls()
                
            return cls._instance

        except Exception as e:
            logger.error(f"Failed to initialize S3: {e}", exc_info=True)
            raise StorageError(f"S3 initialization failed: {e}")

    @classmethod
    def get_instance(cls) -> 'S3Storage':
        """Get storage instance."""
        if cls._client is None:
            if S3_ENABLED:
                cls.initialize()
            else:
                raise StorageError("S3 storage is disabled")
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """Check if S3 is available and enabled."""
        return S3_AVAILABLE and S3_ENABLED

    def _get_s3_key(self, user_id: str, filename: str, folder: str = "uploads") -> str:
        """Generate S3 key for a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
        return f"{self._base_path}{folder}/{user_id}/{timestamp}_{safe_filename}"

    def upload_file(
        self,
        file_path: str,
        user_id: str,
        folder: str = "uploads",
        metadata: Dict = None
    ) -> Dict:
        """Upload a file to S3."""
        path = Path(file_path)
        if not path.exists():
            raise FileProcessingError(f"File not found: {file_path}")

        s3_key = self._get_s3_key(user_id, path.name, folder)

        try:
            # Calculate file hash
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # Upload
            extra_args = {
                'ContentType': self._guess_content_type(path.name),
                'Metadata': {
                    'original_filename': path.name,
                    'user_id': user_id,
                    'uploaded_at': datetime.now().isoformat(),
                    'file_hash': file_hash,
                    **(metadata or {})
                }
            }

            self._client.upload_file(
                str(path),
                self._bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            logger.info(f"Uploaded file to S3: {s3_key}")

            return {
                "success": True,
                "key": s3_key,
                "bucket": self._bucket,
                "url": self.get_url(s3_key),
                "size": path.stat().st_size,
                "hash": file_hash
            }

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}", exc_info=True)
            raise StorageError(f"Upload failed: {e}")

    def upload_bytes(
        self,
        data: bytes,
        filename: str,
        user_id: str,
        folder: str = "uploads",
        content_type: str = None,
        metadata: Dict = None
    ) -> Dict:
        """Upload bytes directly to S3."""
        s3_key = self._get_s3_key(user_id, filename, folder)
        file_hash = hashlib.md5(data).hexdigest()

        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=data,
                ContentType=content_type or self._guess_content_type(filename),
                Metadata={
                    'original_filename': filename,
                    'user_id': user_id,
                    'uploaded_at': datetime.now().isoformat(),
                    'file_hash': file_hash,
                    **(metadata or {})
                }
            )

            logger.info(f"Uploaded bytes to S3: {s3_key}")

            return {
                "success": True,
                "key": s3_key,
                "bucket": self._bucket,
                "url": self.get_url(s3_key),
                "size": len(data),
                "hash": file_hash
            }

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}", exc_info=True)
            raise StorageError(f"Upload failed: {e}")

    def download_file(self, s3_key: str, destination: str) -> Dict:
        """Download a file from S3."""
        try:
            self._client.download_file(self._bucket, s3_key, destination)
            logger.info(f"Downloaded file from S3: {s3_key}")
            return {
                "success": True,
                "key": s3_key,
                "destination": destination
            }
        except ClientError as e:
            logger.error(f"S3 download failed: {e}", exc_info=True)
            raise StorageError(f"Download failed: {e}")

    def download_bytes(self, s3_key: str) -> bytes:
        """Download file content as bytes."""
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"S3 download failed: {e}", exc_info=True)
            raise StorageError(f"Download failed: {e}")

    def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3."""
        try:
            self._client.delete_object(Bucket=self._bucket, Key=s3_key)
            logger.info(f"Deleted file from S3: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}", exc_info=True)
            raise StorageError(f"Delete failed: {e}")

    def get_url(self, s3_key: str, expiration: int = None) -> str:
        """Get a pre-signed URL for a file."""
        try:
            url = self._client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self._bucket, 'Key': s3_key},
                ExpiresIn=expiration or PRE_SIGNED_URL_EXPIRATION
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate URL: {e}", exc_info=True)
            raise StorageError(f"URL generation failed: {e}")

    def list_files(
        self,
        user_id: str,
        folder: str = "uploads",
        limit: int = 100
    ) -> list:
        """List files for a user."""
        prefix = f"{self._base_path}{folder}/{user_id}/"
        try:
            response = self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
                MaxKeys=limit
            )
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'url': self.get_url(obj['Key'])
                })
            return files
        except ClientError as e:
            logger.error(f"S3 list failed: {e}", exc_info=True)
            raise StorageError(f"List failed: {e}")

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def get_file_info(self, s3_key: str) -> Optional[Dict]:
        """Get file metadata."""
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return {
                'key': s3_key,
                'size': response['ContentLength'],
                'content_type': response['ContentType'],
                'last_modified': response['LastModified'].isoformat(),
                'metadata': response.get('Metadata', {})
            }
        except ClientError:
            return None

    def _guess_content_type(self, filename: str) -> str:
        """Guess content type from filename."""
        ext = Path(filename).suffix.lower()
        content_types = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
            '.py': 'text/x-python',
            '.js': 'application/javascript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        return content_types.get(ext, 'application/octet-stream')

    def health_check(self) -> Dict:
        """Check S3 health."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return {
                "status": "healthy",
                "bucket": self._bucket,
                "endpoint": S3_ENDPOINT
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def get_storage() -> S3Storage:
    """Get S3 storage instance (required for production)."""
    if not S3_ENABLED:
        raise StorageError("S3_ENABLED must be true for production")
    if not S3_AVAILABLE:
        raise StorageError("S3 unavailable. Install boto3: pip install boto3")
    return S3Storage.get_instance()
