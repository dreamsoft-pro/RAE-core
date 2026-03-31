import os
from enum import Enum
from typing import List, Dict, Optional
from pathlib import Path

class MissionState(Enum):
    STRATEGY = "STRATEGY"           # CEO Planning
    RECON = "RECONNAISSANCE"        # Phoenix Analysis
    LAB = "LAB_VALIDATION"          # Prototyping in Lab
    EXECUTION = "HIVE_EXECUTION"    # OpenClaw / Hive Work
    AUDIT = "QUALITY_TRIBUNAL"      # Quality Check
    COMPLETED = "COMPLETED"

class MissionMode(Enum):
    REFACTOR = "REFACTOR"
    CREATE = "CREATE"

class MissionContract:
    """
    Hard Contract: Mission Protocol Enforcer.
    Prevents execution jumping and ensures Lab validation before Mass Production.
    """
    
    PROTOCOLS = {
        MissionMode.REFACTOR: [
            MissionState.STRATEGY, 
            MissionState.RECON, 
            MissionState.LAB, 
            MissionState.EXECUTION, 
            MissionState.AUDIT,
            MissionState.COMPLETED
        ],
        MissionMode.CREATE: [
            MissionState.STRATEGY, 
            MissionState.LAB, 
            MissionState.EXECUTION, 
            MissionState.AUDIT,
            MissionState.COMPLETED
        ]
    }

    @staticmethod
    def validate_action(current_state: MissionState, action_target: MissionState, mode: MissionMode) -> bool:
        """
        Enforces the sequential flow of the RAE Suite.
        """
        protocol = MissionContract.PROTOCOLS.get(mode, [])
        if not protocol:
            return False
            
        try:
            curr_idx = protocol.index(current_state)
            target_idx = protocol.index(action_target)
            
            # Allow staying in same state or moving to exactly next state
            return target_idx <= curr_idx + 1
        except ValueError:
            return False

    @staticmethod
    def pre_flight_check(action: str, mission_context: Dict) -> (bool, str):
        """
        Hard-wired guard for autonomic tools.
        """
        state = MissionState(mission_context.get("state", "STRATEGY"))
        mode = MissionMode(mission_context.get("mode", "REFACTOR"))
        
        if action in ["openclaw_write", "hive_mass_convert"]:
            if state != MissionState.LAB:
                return False, f"CRITICAL PROTOCOL ERROR: {action} is forbidden in state {state}. Must complete LAB_VALIDATION first."
        
        return True, "OK"
