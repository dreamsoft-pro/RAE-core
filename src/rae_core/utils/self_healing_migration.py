# RAE-core/src/rae_core/utils/self_healing_migration.py
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import os

logger = logging.getLogger("RAE-Core-SelfHealer")

class AlembicSelfHealer:
    """Proactively verifies database schema compatibility on startup and auto-executes hot-fixes for drifts."""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://rae:rae_pass@localhost:5432/rae_memory")
        self.engine = create_engine(self.db_url)

    def verify_and_align_schema(self):
        """Checks for missing core columns in the 'memories' table and dynamically ALTERS schema if needed."""
        logger.info("Starting database self-healing verification cycle...")
        required_columns = {
            "session_id": "UUID",
            "project": "VARCHAR(255)",
            "source": "VARCHAR(255)",
            "ttl": "INTEGER",
            "strength": "DOUBLE PRECISION"
        }
        
        try:
            with self.engine.connect() as conn:
                # 1. Fetch current column names of 'memories' table
                res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'memories'"))
                existing_columns = {row[0] for row in res}
                
                if not existing_columns:
                    logger.warning("Table 'memories' does not exist yet. Deferring to standard migration flow.")
                    return
                
                # 2. Check for missing columns and run ALTER TABLE commands
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        logger.warning(f"Drift detected: missing column '{col_name}' in 'memories'. Aligning...")
                        alter_query = f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}"
                        conn.execute(text(alter_query))
                        logger.info(f"Successfully aligned column '{col_name}'!")
                        
            logger.info("Database self-healing alignment completed successfully.")
        except Exception as e:
            logger.error(f"Failed to automatically align database schema: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    healer = AlembicSelfHealer()
    healer.verify_and_align_schema()
