"""
RAE Universal Segmenter (Stage 4).
Content-aware atomic chunking with AFE (Automatic Feature Extraction).
Eliminates hardcoding by utilizing math_controller.yaml patterns.
"""

import re
from typing import Any, Dict, List, Optional
import structlog
from .interfaces import ISegmenter, ContentSignature, IngestChunk, IngestAudit

logger = structlog.get_logger(__name__)

class IngestSegmenter(ISegmenter):
    """
    Splits text into chunks by identifying atomic units and aggregating them coherently.
    Performs autonomous feature extraction based on configured rules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        ingest_cfg = self.config.get("ingest_params", {})
        
        # Pull limits and patterns from config (Zero Hardcoding)
        self.target_size = ingest_cfg.get("target_chunk_size", 1500)
        self.hard_limit = ingest_cfg.get("hard_limit", self.target_size * 3)
        self.patterns = ingest_cfg.get("boundary_patterns", {})
        self.extraction_rules = ingest_cfg.get("extraction_rules", {})

    async def segment(self, text: str, policy: str, signature: ContentSignature) -> tuple[List[IngestChunk], IngestAudit]:
        # 1. Atomic Splitting
        if policy == "POLICY_LOG_STREAM":
            chunks = self._segment_log_entries(text)
        elif policy == "POLICY_PROCEDURE_DOC":
            chunks = self._segment_atomic_units(text, self.patterns.get("procedure_step"))
        elif policy == "POLICY_TECHNICAL_FORMAL":
            chunks = self._segment_atomic_units(text, r'^[A-Z0-9]{3,}:') 
        else:
            chunks = self._segment_default(text)
            
        # 2. Feature Extraction (AFE)
        for chunk in chunks:
            self._apply_afe(chunk)
            
        audit = IngestAudit(
            stage="segment",
            action="atomic_splitting_with_afe",
            trace={
                "policy": policy,
                "chunk_count": len(chunks),
                "extracted_keys": list(set(k for c in chunks for k in c.metadata.get("extracted_features", {}).keys()))
            }
        )
        
        return chunks, audit

    def _apply_afe(self, chunk: IngestChunk):
        """
        Extracts metrics and identifiers using configuration-driven regex rules.
        """
        features = {}
        for rule_name, rule in self.extraction_rules.items():
            pattern = rule.get("pattern")
            mapping = rule.get("mapping", {})
            if not pattern: continue
            
            # Find all matches in this chunk
            matches = re.finditer(pattern, chunk.content)
            for match in matches:
                for group_idx, target_key in mapping.items():
                    try:
                        val = match.group(int(group_idx))
                        # Type conversion
                        try:
                            if '.' in val: val = float(val)
                            else: val = int(val)
                        except ValueError: pass
                        features[target_key] = val
                    except (IndexError, ValueError): continue
        
        if features:
            if "extracted_features" not in chunk.metadata:
                chunk.metadata["extracted_features"] = {}
            chunk.metadata["extracted_features"].update(features)

    def _segment_log_entries(self, text: str) -> List[IngestChunk]:
        """Groups log entries by timestamps, ensuring entries are never split."""
        lines = text.split('\n')
        ts_pattern = self.patterns.get("log_timestamp", r'\d{2,4}[-:/]\d{2}[-:/]\d{2,4}')
        
        atoms = []
        current_atom = []
        for line in lines:
            if not line.strip(): continue
            if re.match(ts_pattern, line.strip()):
                if current_atom:
                    atoms.append("\n".join(current_atom))
                current_atom = [line]
            else:
                if current_atom:
                    current_atom.append(line)
                else:
                    current_atom = [line]
        if current_atom:
            atoms.append("\n".join(current_atom))

        return self._aggregate_atoms(atoms, "log_block")

    def _segment_atomic_units(self, text: str, boundary_pattern: str) -> List[IngestChunk]:
        """Splits by markers (Steps, Labels) keeping markers at start of atoms."""
        if not boundary_pattern:
            return self._segment_default(text)
            
        parts = re.split(f'({boundary_pattern})', text, flags=re.MULTILINE)
        atoms = []
        if parts[0].strip():
            atoms.append(parts[0].strip())
            
        for i in range(1, len(parts), 2):
            marker = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            atom = (marker + content).strip()
            if atom:
                atoms.append(atom)

        return self._aggregate_atoms(atoms, "atomic_unit")

    def _aggregate_atoms(self, atoms: List[str], unit_type: str) -> List[IngestChunk]:
        """Greedy aggregation of atoms into chunks without splitting atoms."""
        chunks = []
        current_group = []
        current_size = 0
        chunk_offset = 0
        
        for atom in atoms:
            atom_size = len(atom)
            
            # Oversized atom handling
            if atom_size > self.hard_limit:
                if current_group:
                    chunks.append(self._create_chunk(current_group, unit_type, chunk_offset))
                    chunk_offset += current_size + (len(current_group) * 2)
                    current_group = []
                    current_size = 0
                
                # Split huge atom by target_size characters
                for i in range(0, atom_size, self.target_size):
                    sub_content = atom[i:i + self.target_size]
                    chunks.append(IngestChunk(
                        content=sub_content,
                        metadata={"type": unit_type, "oversized_atom": True},
                        offset=chunk_offset + i,
                        length=len(sub_content)
                    ))
                chunk_offset += atom_size
                continue

            # Standard greedy aggregation
            if current_size + atom_size > self.target_size and current_group:
                chunks.append(self._create_chunk(current_group, unit_type, chunk_offset))
                chunk_offset += current_size + (len(current_group) * 2)
                current_group = [atom]
                current_size = atom_size
            else:
                current_group.append(atom)
                current_size += atom_size
            
        if current_group:
            chunks.append(self._create_chunk(current_group, unit_type, chunk_offset))
            
        return chunks

    def _create_chunk(self, atoms: List[str], unit_type: str, offset: int) -> IngestChunk:
        content = "\n\n".join(atoms)
        return IngestChunk(
            content=content,
            metadata={"type": unit_type, "atomic_count": len(atoms)},
            offset=offset,
            length=len(content)
        )

    def _segment_default(self, text: str) -> List[IngestChunk]:
        """Fallback to paragraph-based segmentation."""
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paras: 
            return [IngestChunk(content=text, metadata={"type": "raw"}, offset=0, length=len(text))]
        return self._aggregate_atoms(paras, "prose_block")
