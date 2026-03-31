"""
RAE Chunking Utilities.
Semantic and procedural splitting for industrial documents.
"""

import re
from typing import List

class ProceduralChunker:
    """
    Chunks text while respecting procedural boundaries and sentence integrity.
    Designed for OneNote-like content and industrial instructions.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks, trying to break at:
        1. Double newlines (paragraphs)
        2. Procedural markers (Step 1, Krok 1, etc.)
        3. Sentence boundaries
        """
        if not text:
            return []
            
        # 1. Split into logical blocks
        # We look for markers like "Step X", "Krok X", "1.", "2." at the beginning of lines
        # Use regex that works with literal newlines or escaped ones
        blocks = re.split(r'\n\s*(?=(?:Step|Krok|Kolejno)\s*\d+[:.]|\d+\.\s+[A-Z])', text)
        
        refined_blocks = []
        for block in blocks:
            if len(block) <= self.chunk_size:
                refined_blocks.append(block.strip())
            else:
                # Block too big, split by double newlines
                paras = block.split("\n\n")
                current_para_chunk = ""
                for para in paras:
                    if len(current_para_chunk) + len(para) < self.chunk_size:
                        current_para_chunk += para + "\n\n"
                    else:
                        if current_para_chunk:
                            refined_blocks.append(current_para_chunk.strip())
                        current_para_chunk = para + "\n\n"
                        
                        # If a single paragraph is still too big, split by sentences
                        if len(current_para_chunk) > self.chunk_size:
                            # Sentence splitting (naive but effective for Polish/English)
                            sentences = re.split(r'(?<=[.!?])\s+', current_para_chunk)
                            current_sent_chunk = ""
                            for sent in sentences:
                                if len(current_sent_chunk) + len(sent) < self.chunk_size:
                                    current_sent_chunk += sent + " "
                                else:
                                    if current_sent_chunk:
                                        refined_blocks.append(current_sent_chunk.strip())
                                    current_sent_chunk = sent + " "
                            current_para_chunk = current_sent_chunk
                
                if current_para_chunk:
                    refined_blocks.append(current_para_chunk.strip())
                    
        # 2. Merge small blocks to avoid fragmentation
        final_chunks = []
        temp_chunk = ""
        for b in refined_blocks:
            if not b: continue
            if len(temp_chunk) + len(b) < self.chunk_size:
                temp_chunk += b + "\n\n"
            else:
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
                temp_chunk = b + "\n\n"
        
        if temp_chunk:
            final_chunks.append(temp_chunk.strip())
            
        return final_chunks
