import os
import logging
from pathlib import Path

class ContractManager:
    """Loads and enforces binary hard contracts."""
    
    def __init__(self, contracts_path=None):
        if contracts_path is None:
            project_root = Path(os.environ.get('RAE_PROJECT_ROOT', Path(__file__).resolve().parent.parent.parent.parent))
            self.contracts_path = str(project_root / "contracts")
        else:
            self.contracts_path = contracts_path
        self.rules = {}
        self._load_contracts()

    def _load_contracts(self):
        if not os.path.exists(self.contracts_path):
            return
        for file in os.listdir(self.contracts_path):
            if file.endswith(".bin"):
                with open(os.path.join(self.contracts_path, file), "r") as f:
                    self.rules[file] = f.read()

    def verify_operation(self, op_name, impact_level, info_class="internal"):
        # Przykład twardej walidacji na podstawie SECURITY_ISO.bin
        if info_class in ["restricted", "critical"] and impact_level != "high":
            return False, f"CRITICAL DATA REQUIRE HIGH IMPACT AUDIT (Violation of SECURITY_ISO.bin)"
        
        # Sprawdzenie polityki Zero-Warning z AGENT_PROTOCOL.bin
        # (Logika może być rozbudowana o sprawdzanie flag systemowych)
        
        return True, "OK"

    def get_bootstrap_summary(self):
        """Returns a concise summary for Agent System Prompt injection."""
        summary = "--- MANDATORY HARD CONTRACTS LOADED ---\n"
        for name, content in self.rules.items():
            summary += f"[{name}]:\n{content}\n"
        return summary
