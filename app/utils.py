import os

def sanitize_path(path: str, base_dir: str) -> str:
    """
    Sanitizes and validates that a path is contained within a base directory.
    Prevents directory traversal attacks.
    """
    real_base = os.path.realpath(base_dir)
    real_path = os.path.realpath(path)
    
    if not real_path.startswith(real_base):
        raise ValueError(f"Path traversal detected: {path} is not in {base_dir}")
    
    return real_path

def validate_extension(path: str, allowed_extensions: list[str]) -> bool:
    """
    Validates that the file has an allowed extension.
    """
    _, ext = os.path.splitext(path)
    return ext.lower() in allowed_extensions
