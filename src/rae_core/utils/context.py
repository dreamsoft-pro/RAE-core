# rae_core/utils/context.py
import os
import logging

class RAEContextLocator:
    """Intelligently detects project and tenant context based on environment and filesystem."""
    
    TENANT_MAP = {
        "screenwatcher_project": "66435998-b1d9-5521-9481-55a9fd10e014",
        "dreamsoft_factory": "53717286-fe94-4c8f-baf9-c4d2758eb672",
        "billboard-splitter": "67694908-0b76-58a9-979d-3db20071e34a",
        "RAE-agentic-memory": "00000000-0000-0000-0000-000000000000",
        "RAE-Suite": "00000000-0000-0000-0000-000000000000"
    }

    @staticmethod
    def get_current_tenant_id():
        # 1. Sprawdzamy zmienną środowiskową
        env_id = os.getenv("RAE_TENANT_ID")
        if env_id: return env_id
        
        # 2. Sprawdzamy RAE_PROJECT_NAME
        proj_name = os.getenv("RAE_PROJECT_NAME")
        if proj_name in RAEContextLocator.TENANT_MAP:
            return RAEContextLocator.TENANT_MAP[proj_name]

        # 3. Inteligentna detekcja po ścieżce
        cwd = os.getcwd()
        for folder, uuid in RAEContextLocator.TENANT_MAP.items():
            if folder in cwd:
                return uuid
        
        # 4. FALLBACK: W środowisku produkcyjnym/dev używamy Dreamsoft jako domyślny tenant
        # zamiast pustego UUID, aby nie tracić logów audytowych.
        return RAEContextLocator.TENANT_MAP["dreamsoft_factory"]

    @staticmethod
    def get_project_name():
        env_name = os.getenv("RAE_PROJECT_NAME")
        if env_name: return env_name

        cwd = os.getcwd()
        for folder in RAEContextLocator.TENANT_MAP.keys():
            if folder in cwd:
                return folder
        
        return "unnamed_production_module"
