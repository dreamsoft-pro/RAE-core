"""E2E encryption helpers for memory synchronization."""

import base64
import json
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class E2EEncryption:
    """End-to-end encryption for memory synchronization.

    Uses Fernet (symmetric encryption) with key derivation.
    """

    def __init__(self, key: bytes | None = None, password: str | None = None):
        """Initialize E2E encryption.

        Args:
            key: Optional encryption key (32 bytes, base64-encoded)
            password: Optional password for key derivation
        """
        if key:
            self.key = key
        elif password:
            self.key = self._derive_key_from_password(password)
        else:
            # Generate new key
            self.key = Fernet.generate_key()

        self.fernet = Fernet(self.key)

    def _derive_key_from_password(
        self,
        password: str,
        salt: bytes | None = None,
    ) -> bytes:
        """Derive encryption key from password using PBKDF2.

        Args:
            password: Password string
            salt: Optional salt (generates new if not provided)

        Returns:
            Derived key
        """
        if salt is None:
            salt = b"rae-memory-sync-salt"  # Fixed salt for deterministic key

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )

        key_material = kdf.derive(password.encode())
        return base64.urlsafe_b64encode(key_material)

    def encrypt(self, data: Any) -> str:
        """Encrypt data.

        Args:
            data: Data to encrypt (will be JSON-serialized)

        Returns:
            Encrypted data as base64 string
        """
        # Serialize data to JSON
        json_data = json.dumps(data, default=str)
        json_bytes = json_data.encode()

        # Encrypt
        encrypted_bytes = self.fernet.encrypt(json_bytes)

        # Return as base64 string
        return base64.b64encode(encrypted_bytes).decode()

    def decrypt(self, encrypted_data: str) -> Any:
        """Decrypt data.

        Args:
            encrypted_data: Encrypted data as base64 string

        Returns:
            Decrypted data (JSON-deserialized)
        """
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode())

        # Decrypt
        decrypted_bytes = self.fernet.decrypt(encrypted_bytes)

        # Deserialize JSON
        json_data = decrypted_bytes.decode()
        return json.loads(json_data)

    def encrypt_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Encrypt sensitive fields in memory record.

        Args:
            memory: Memory record dictionary

        Returns:
            Memory record with encrypted fields
        """
        encrypted_memory = memory.copy()

        # Fields to encrypt
        sensitive_fields = ["content", "metadata"]

        for field in sensitive_fields:
            if field in encrypted_memory:
                encrypted_memory[field] = self.encrypt(encrypted_memory[field])
                encrypted_memory[f"{field}_encrypted"] = True

        return encrypted_memory

    def decrypt_memory(self, encrypted_memory: dict[str, Any]) -> dict[str, Any]:
        """Decrypt sensitive fields in memory record.

        Args:
            encrypted_memory: Memory record with encrypted fields

        Returns:
            Memory record with decrypted fields
        """
        memory = encrypted_memory.copy()

        # Fields to decrypt
        sensitive_fields = ["content", "metadata"]

        for field in sensitive_fields:
            if memory.get(f"{field}_encrypted"):
                memory[field] = self.decrypt(memory[field])
                del memory[f"{field}_encrypted"]

        return memory

    def get_key(self) -> str:
        """Get encryption key as base64 string.

        Returns:
            Base64-encoded encryption key
        """
        return self.key.decode()

    @staticmethod
    def generate_key() -> str:
        """Generate new encryption key.

        Returns:
            Base64-encoded encryption key
        """
        return Fernet.generate_key().decode()


def encrypt_batch(
    memories: list[dict[str, Any]],
    encryption: E2EEncryption,
) -> list[dict[str, Any]]:
    """Encrypt a batch of memories.

    Args:
        memories: List of memory records
        encryption: E2EEncryption instance

    Returns:
        List of encrypted memory records
    """
    return [encryption.encrypt_memory(mem) for mem in memories]


def decrypt_batch(
    encrypted_memories: list[dict[str, Any]],
    encryption: E2EEncryption,
) -> list[dict[str, Any]]:
    """Decrypt a batch of memories.

    Args:
        encrypted_memories: List of encrypted memory records
        encryption: E2EEncryption instance

    Returns:
        List of decrypted memory records
    """
    return [encryption.decrypt_memory(mem) for mem in encrypted_memories]
