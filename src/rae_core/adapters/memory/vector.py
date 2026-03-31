"""In-Memory vector store adapter for RAE-core.

Deprecated: Use rae_core.adapters.memory.storage.InMemoryStorage which
implements both IMemoryStorage and IVectorStore with deterministic
fixed-point arithmetic (System 87.0).
"""

from typing import Any
from uuid import UUID

from rae_core.adapters.memory.storage import InMemoryStorage
from rae_core.interfaces.vector import IVectorStore

# Inherit from InMemoryStorage which implements IVectorStore
class InMemoryVectorStore(InMemoryStorage):
    """Legacy wrapper for InMemoryStorage.
    
    This class ensures backward compatibility while using the new
    deterministic 'System 87.0' engine (Fixed-Point, Arenas, Bloom Filters).
    """
    
    def __init__(self) -> None:
        super().__init__()
