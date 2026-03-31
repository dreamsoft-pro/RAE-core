"""Service for converting between float and fixed-point integer vectors."""

import math
from typing import Final, Sequence

# Precision configuration for Q16.16 (or similar)
# We use a scale that preserves enough precision for dot products in 64-bit space.
# 2^16 = 65536. Standard for DSP.
SCALE_FACTOR: Final[int] = 65536  # 2^16
MAX_SAFE_INT: Final[int] = 9223372036854775807  # 2^63 - 1 (Python handles larger, but for strictness)


def quantize_vector(vector: Sequence[float]) -> list[int]:
    """Convert a float vector to a fixed-point integer vector.
    
    Args:
        vector: List of floating point numbers.
        
    Returns:
        List of integers scaled by SCALE_FACTOR.
        
    Raises:
        ValueError: If vector contains NaN or Infinity.
    """
    if not vector:
        return []

    result = []
    for val in vector:
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value in vector: {val}")
        
        # Round to nearest integer deterministically
        # Python's round() rounds to nearest even for .5, which is good.
        scaled = val * SCALE_FACTOR
        
        # Handle overflow for typical embedding range (-1.0 to 1.0) is trivial.
        # But for general vectors, we might clamp. Embeddings are usually normalized.
        # We assume normalized or reasonable range.
        
        result.append(int(round(scaled)))
    
    return result


def dequantize_vector(vector: Sequence[int]) -> list[float]:
    """Convert a fixed-point integer vector back to float.
    
    Args:
        vector: List of scaled integers.
        
    Returns:
        List of floats.
    """
    if not vector:
        return []
    
    return [val / SCALE_FACTOR for val in vector]


def dot_product_fixed(vec_a: Sequence[int], vec_b: Sequence[int]) -> int:
    """Compute dot product of two fixed-point vectors.
    
    Result is scaled by SCALE_FACTOR^2.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimension mismatch")
    
    # Python handles arbitrary precision integers, so overflow isn't an issue like in C.
    # But for strict determinism we rely on integer math properties.
    total = 0
    for a, b in zip(vec_a, vec_b):
        total += a * b
        
    return total


def cosine_similarity_fixed(vec_a: Sequence[int], vec_b: Sequence[int]) -> float:
    """Compute cosine similarity from fixed-point vectors.
    
    Returns a float (0.0 to 1.0) because the final metric is usually needed as float.
    However, the intermediate steps are integer-based.
    
    Sim(A, B) = (A . B) / (||A|| * ||B||)
    """
    dot = dot_product_fixed(vec_a, vec_b)
    
    norm_a_sq = dot_product_fixed(vec_a, vec_a)
    norm_b_sq = dot_product_fixed(vec_b, vec_b)
    
    if norm_a_sq == 0 or norm_b_sq == 0:
        return 0.0
        
    # Sqrt requires floating point, but it's done at the very end.
    # We can keep it deterministic by using math.sqrt on the integer result.
    # The error introduced here is minimal compared to doing everything in float.
    
    return dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
