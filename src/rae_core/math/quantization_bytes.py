"""Service for converting between float and fixed-point integer vectors (Deterministyczna Wersja)."""

import math
import struct
from typing import Final, Sequence, cast

# Stała skalowania (Precision configuration for Q16.16)
# 2^16 = 65536. Standard dla przetwarzania sygnałów, wystarczający dla osadzeń wektorowych.
SCALE_FACTOR: Final[int] = 65536


def quantize_vector_bytes(vector: Sequence[float]) -> bytes:
    """Konwersja wektora float na spakowany ciąg bajtów int32 (Big Endian).
    
    Format: >i (signed 32-bit big-endian integer).
    Zapewnia to przenośność i determinizm niezależnie od architektury CPU.
    """
    if not vector:
        return b""

    # Alokacja bufora: 4 bajty na każdą liczbę
    packed_data = bytearray(len(vector) * 4)
    
    offset = 0
    fmt = ">i"  # Big-endian signed int
    pack = struct.pack_into
    
    for val in vector:
        if not math.isfinite(val):
            # W środowisku produkcyjnym logujemy błąd, tutaj rzucamy wyjątek dla bezpieczeństwa
            raise ValueError(f"Non-finite value in vector: {val}")
        
        # Round half to even (standard Python round)
        scaled_val = int(round(val * SCALE_FACTOR))
        
        # Clamp to 32-bit signed range to prevent overflow errors in pack
        # Embeddingi normalnie są w zakresie -1..1, więc po skalowaniu mieszczą się bez problemu.
        # Ale dla bezpieczeństwa:
        if scaled_val > 2147483647:
            scaled_val = 2147483647
        elif scaled_val < -2147483648:
            scaled_val = -2147483648
            
        pack(fmt, packed_data, offset, scaled_val)
        offset += 4
        
    return bytes(packed_data)


def dequantize_vector_bytes(data: bytes) -> list[float]:
    """Konwersja spakowanych bajtów int32 z powrotem na listę float.
    
    Args:
        data: Ciąg bajtów reprezentujący wektor.
        
    Returns:
        Lista liczb zmiennoprzecinkowych.
    """
    if not data:
        return []
        
    count = len(data) // 4
    fmt = f">{count}i"
    
    integers = struct.unpack(fmt, data)
    
    return [val / SCALE_FACTOR for val in integers]


def dot_product_bytes(vec_a_bytes: bytes, vec_b_bytes: bytes) -> int:
    """Obliczenie iloczynu skalarnego bezpośrednio na bajtach (int32).
    
    Args:
        vec_a_bytes: Wektor A (spakowany int32).
        vec_b_bytes: Wektor B (spakowany int32).
        
    Returns:
        Wynik iloczynu skalarnego (int64/arbitrary precision python int).
        Skalowany przez SCALE_FACTOR^2.
    """
    len_a = len(vec_a_bytes)
    len_b = len(vec_b_bytes)
    
    if len_a != len_b:
        raise ValueError(f"Vector dimension mismatch: {len_a} vs {len_b} bytes")
        
    count = len_a // 4
    fmt = f">{count}i"
    
    ints_a = struct.unpack(fmt, vec_a_bytes)
    ints_b = struct.unpack(fmt, vec_b_bytes)
    
    total = 0
    # Python 3 integers support arbitrary precision, so intermediate overflow is not an issue.
    # This loop is slower than numpy but purely Python and deterministic.
    for a, b in zip(ints_a, ints_b):
        total += a * b
        
    return total


def cosine_similarity_bytes(vec_a_bytes: bytes, vec_b_bytes: bytes) -> float:
    """Obliczenie podobieństwa kosinusowego na bajtach.
    
    Returns:
        Wartość float z zakresu [-1.0, 1.0].
    """
    dot = dot_product_bytes(vec_a_bytes, vec_b_bytes)
    
    norm_a_sq = dot_product_bytes(vec_a_bytes, vec_a_bytes)
    norm_b_sq = dot_product_bytes(vec_b_bytes, vec_b_bytes)
    
    if norm_a_sq == 0 or norm_b_sq == 0:
        return 0.0
        
    # Sqrt na końcu operacji (nieuniknione przejście na float).
    # Ale wejście do sqrt jest deterministyczne (int).
    return dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
