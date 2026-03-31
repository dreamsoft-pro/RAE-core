"""
RAE Ingest Policy Selector.
Stage 3: Choosing the right tools for the job.
"""

from typing import Dict, Any
from .interfaces import IPolicySelector, ContentSignature, IngestAudit

class IngestPolicySelector(IPolicySelector):
    """
    Decides the ingestion policy based on the content signature.
    Uses dominance rules instead of weights.
    """
    
    def select_policy(self, signature: ContentSignature) -> tuple[str, IngestAudit]:
        struct = signature.struct
        dist = signature.dist
        stab = signature.stab
        
        # Dominance Rules
        if signature.struct.get("mode") == "OPERATIONAL_FALLBACK":
            policy = "POLICY_FALLBACK"
        elif stab["conflict"]:
            policy = "POLICY_MIXED_SAFE"
        elif struct["mode"] in ["LINEAR_LOG_LIKE", "MACHINE_TELEMETRY_LIKE"]:
            # If repeatability is high or it's telemetry, use stream policy
            if dist["repeatability_score"] > 0.4 or struct["mode"] == "MACHINE_TELEMETRY_LIKE":
                policy = "POLICY_LOG_STREAM"
            else:
                policy = "POLICY_MIXED_SAFE"
        elif struct["mode"] == "LIST_PROCEDURE_LIKE":
            policy = "POLICY_PROCEDURE_DOC"
        elif struct["mode"] == "TECHNICAL_SPEC_LIKE":
            policy = "POLICY_TECHNICAL_FORMAL"
        elif struct["mode"] == "TABLE_RECORD_LIKE":
            policy = "POLICY_DATA_TABLE"
        else:
            policy = "POLICY_PROSE_TEXT"
            
        audit = IngestAudit(
            stage="policy",
            action="policy_selection",
            trace={
                "policy_selected": policy,
                "reasoning": f"mode={struct['mode']}, conflict={stab['conflict']}"
            }
        )
        
        return policy, audit
