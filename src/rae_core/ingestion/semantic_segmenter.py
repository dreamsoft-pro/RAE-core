"""
RAE Semantic Recursive Segmenter (Integrated with RAE's NativeEmbeddingProvider)
Author: Gemini Agent (in research mode)
Date: 2026-02-18
Branch: research/segmenter-overhaul

Principle: This segmenter combines hierarchical structural splitting with
           semantic splitting based on dynamic, percentile-based thresholding,
           powered by RAE's internal NativeEmbeddingProvider (ONNX models).
           It adheres to RAE's core principles of being lightweight, local-first,
           auditable, and fully integrated with the RAE ecosystem.
"""
import numpy as np
import re
from typing import List, Optional, Any
from rae_core.ingestion.interfaces import IngestChunk, IngestAudit

# Import RAE's NativeEmbeddingProvider
from rae_core.embedding.native import NativeEmbeddingProvider

# --- Helper Functions ---

def _calculate_similarity_scores(embeddings: np.ndarray) -> List[float]:
    """Calculates cosine similarity between consecutive embeddings."""
    if embeddings.shape[0] < 2:
        return []
    
    similarities = []
    for i in range(embeddings.shape[0] - 1):
        embedding1 = embeddings[i]
        embedding2 = embeddings[i+1]
        # dot product is enough since vectors are normalized
        similarity = np.dot(embedding1, embedding2)
        similarities.append(similarity)
    return similarities

# --- The Advanced Segmenter Class ---

class SemanticRecursiveSegmenter:
    """
    An advanced segmenter that combines hierarchical structural splitting with
    semantic splitting based on dynamic, percentile-based thresholding.
    """

    def __init__(
        self,
        structural_separators: Optional[List[str]] = None,
        semantic_separators: Optional[List[str]] = None,
        chunk_size: int = 1000,
        semantic_threshold_percentile: int = 90,
        embedding_model_path: str = "/app/models/all-MiniLM-L6-v2/model.onnx", # Use RAE's ONNX model
        embedding_tokenizer_path: str = "/app/models/all-MiniLM-L6-v2/tokenizer.json"
    ):
        self._structural_separators = structural_separators or ["\n\n", "\n", " ", ""]
        self._chunk_size = chunk_size
        self._semantic_separators = semantic_separators or [". ", "? ", "! ", "\n"]
        self.semantic_threshold_percentile = semantic_threshold_percentile
        
        # Instantiate RAE's NativeEmbeddingProvider
        self._embedder = NativeEmbeddingProvider(
            model_path=embedding_model_path,
            tokenizer_path=embedding_tokenizer_path
        ) 

    async def _semantic_split(self, text: str) -> List[str]:
        """Performs the semantic splitting using RAE's NativeEmbeddingProvider."""
        sentences = re.split(f"({'|'.join(map(re.escape, self._semantic_separators))})", text)
        
        processed_sentences = []
        i = 0
        while i < len(sentences):
            sentence_part = sentences[i].strip()
            if not sentence_part:
                i += 1
                continue
            
            delimiter = ""
            if i + 1 < len(sentences) and sentences[i+1].strip() in self._semantic_separators:
                delimiter = sentences[i+1].strip()
                i += 1
            
            processed_sentences.append(sentence_part + delimiter)
            i += 1

        if len(processed_sentences) <= 1:
            return [text] if text.strip() else []
        
        # Generate embeddings using RAE's NativeEmbeddingProvider (Now properly awaited)
        embeddings = await self._embedder.embed_batch(processed_sentences)
        
        # Convert embeddings to numpy array for similarity calculation
        embeddings_np = np.array(embeddings)
        
        similarities = _calculate_similarity_scores(embeddings_np)

        if not similarities:
            return [text] if text.strip() else []

        dynamic_threshold = np.percentile(similarities, self.semantic_threshold_percentile)
        
        final_semantic_chunks = []
        current_chunk_sentences = [processed_sentences[0]]
        
        for i, sentence in enumerate(processed_sentences[1:]):
            # Ensure we don't exceed similarities bounds
            if i < len(similarities) and similarities[i] < dynamic_threshold:
                final_semantic_chunks.append("".join(current_chunk_sentences))
                current_chunk_sentences = []
            
            current_chunk_sentences.append(sentence)
        
        if current_chunk_sentences:
            final_semantic_chunks.append("".join(current_chunk_sentences))
        
        return final_semantic_chunks


    async def segment(self, text: str, policy: str = "default", signature: Any = None) -> tuple[List[IngestChunk], IngestAudit]:
        """
        Segments a given text using a two-stage hierarchical and semantic process.
        Returns a list of IngestChunk objects and an IngestAudit.
        """
        # --- Stage 1: Hierarchical Structural Splitting ---
        structural_splits = self._perform_structural_split(text, self._structural_separators)

        # --- Stage 2: Semantic Refinement for Oversized Structural Chunks ---
        final_chunks = []
        for split in structural_splits:
            if len(split) <= self._chunk_size:
                if split.strip():
                    final_chunks.append(split)
            else:
                semantic_chunks = await self._semantic_split(split)
                final_chunks.extend(semantic_chunks)

        # Create IngestChunk objects and an audit trail
        ingest_chunks = []
        offset = 0
        for chunk_content in final_chunks:
            chunk = IngestChunk(
                content=chunk_content,
                metadata={"policy": policy, "source_signature": signature.to_dict() if signature else None}, # Convert signature to dict
                offset=offset,
                length=len(chunk_content)
            )
            ingest_chunks.append(chunk)
            offset += len(chunk_content) # No extra offset for separator if not explicitly adding it

        audit = IngestAudit(
            stage="segment",
            action="hybrid_segmentation",
            trace={"chunk_count": len(ingest_chunks), "policy": policy, "signature": signature.to_dict() if signature else None} # Convert signature to dict
        )
        
        return ingest_chunks, audit

    def _perform_structural_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Helper for recursive structural splitting.
        """
        if not separators or len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        separator = None
        for s in separators:
            if s == "": # Character split fallback marker
                separator = ""
                break
            if s in text:
                separator = s
                break
        
        if separator is None: 
            return [text] if text.strip() else []

        # For character-level split, we take fixed-size slices
        if separator == "":
            return [text[i:i+self._chunk_size] for i in range(0, len(text), self._chunk_size)]

        splits = text.split(separator)
        structural_chunks = []
        current_aggregated_part = []
        current_length = 0

        for part in splits:
            if not part.strip():
                continue
            
            part_with_separator_len = len(part) + (len(separator) if current_aggregated_part else 0)
            
            if part_with_separator_len > self._chunk_size or \
               (current_length + part_with_separator_len > self._chunk_size and current_aggregated_part):
                
                if current_aggregated_part:
                    structural_chunks.append(separator.join(current_aggregated_part))
                
                sub_chunks = self._perform_structural_split(part, separators[1:])
                structural_chunks.extend(sub_chunks)
                
                current_aggregated_part = []
                current_length = 0
            else:
                current_aggregated_part.append(part)
                current_length += part_with_separator_len
        
        if current_aggregated_part:
            structural_chunks.append(separator.join(current_aggregated_part))

        return structural_chunks

