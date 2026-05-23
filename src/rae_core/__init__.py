# RAE-core/src/rae_core/__init__.py
import os
import logging

# Pro-active Database Self-Healing Migration Trigger on Import
try:
    # Run only if RAE_DB_MODE is not set to ignore, and DATABASE_URL is provided in the environment
    if os.getenv("RAE_DB_MODE") != "ignore" and os.getenv("DATABASE_URL"):
        from rae_core.utils.self_healing_migration import AlembicSelfHealer
        healer = AlembicSelfHealer()
        healer.verify_and_align_schema()
except Exception as e:
    logging.getLogger("RAE-Core-Init").error(f"Error running auto self-healing migration: {e}")
