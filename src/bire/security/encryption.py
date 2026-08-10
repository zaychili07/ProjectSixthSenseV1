from __future__ import annotations

import os
from cryptography.fernet import Fernet

from pathlib import Path

from cryptography.fernet import Fernet


def generate_encryption_key() -> str:
    """
    Generate a secure encryption key.

    IMPORTANT:
    Do not commit this key to GitHub.
    Store it as an environment variable.
    """
    return Fernet.generate_key().decode()


def get_encryption_key(
    env_var: str = "BIRE_ENCRYPTION_KEY",
) -> str:
    """
    Load encryption key from environment variable.
    """
    key = os.getenv(env_var)

    if key is None:
        raise EnvironmentError(
            f"Missing environment variable: {env_var}. "
            "Set BIRE_ENCRYPTION_KEY before encrypting/decrypting files."
        )

    return key


def get_fernet(
    env_var: str = "BIRE_ENCRYPTION_KEY",
) -> Fernet:
    """
    Create Fernet encryption object from environment key.
    """
    key = get_encryption_key(env_var)
    return Fernet(key.encode())


def encrypt_file(
    input_path: str | Path,
    output_path: str | Path,
    env_var: str = "BIRE_ENCRYPTION_KEY",
) -> Path:
    """
    Encrypt a file and save encrypted output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    fernet = get_fernet(env_var)

    raw_data = input_path.read_bytes()
    encrypted_data = fernet.encrypt(raw_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encrypted_data)

    return output_path


def decrypt_file(
    input_path: str | Path,
    output_path: str | Path,
    env_var: str = "BIRE_ENCRYPTION_KEY",
) -> Path:
    """
    Decrypt a file and save decrypted output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    fernet = get_fernet(env_var)

    encrypted_data = input_path.read_bytes()
    decrypted_data = fernet.decrypt(encrypted_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decrypted_data)

    return output_path