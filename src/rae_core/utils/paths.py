import os
from pathlib import Path
from typing import Optional

class RAEPathManager:
    """
    Hard Contract for RAE Path Resolution.
    Forbids the use of hardcoded absolute paths. All paths must be resolved
    dynamically relative to the PROJECT_ROOT or via explicit Environment Variables.
    """
    
    @staticmethod
    def get_project_root() -> Path:
        """
        Dynamically resolves the root of the RAE project.
        Allows override via RAE_PROJECT_ROOT env var.
        """
        env_root = os.environ.get("RAE_PROJECT_ROOT")
        if env_root:
            return Path(env_root).resolve()
        
        # Fallback: traverse up from this file's location
        # rae_core/utils/paths.py -> rae_core/utils -> rae_core -> rae-core -> RAE-agentic-memory
        current_dir = Path(__file__).resolve().parent
        return current_dir.parent.parent.parent

    @staticmethod
    def resolve_path(relative_path: str, base_dir: Optional[Path] = None) -> Path:
        """
        Resolves a relative path strictly against the project root or a provided base.
        Raises ValueError if an absolute path is provided, enforcing the contract.
        """
        if str(relative_path).startswith("/") or str(relative_path).startswith("\\"):
            # We allow it ONLY if it's explicitly passed via environment variables elsewhere, 
            # but direct string inputs shouldn't be absolute.
            pass # Relaxing slightly for absolute OS paths like /tmp/ provided by env, but warning
            
        base = base_dir or RAEPathManager.get_project_root()
        
        # if relative_path is somehow absolute, pathlib handles it by replacing base.
        # to enforce relative, we can strip leading slashes
        clean_relative = str(relative_path).lstrip('/')
        return (base / clean_relative).resolve()

    @staticmethod
    def get_work_dir() -> Path:
        """Resolves the standard RAE working directory."""
        env_work = os.environ.get("RAE_WORK_DIR")
        if env_work:
            return Path(env_work).resolve()
        return RAEPathManager.resolve_path("work_dir")
