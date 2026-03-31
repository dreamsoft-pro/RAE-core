"""Native Embedding Provider using ONNX Runtime.

This provider runs embedding models locally using ONNX Runtime, ensuring
consistent, high-performance vector generation across platforms (Linux, Windows, Mobile).
"""

from pathlib import Path
from typing import cast

import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError:
    ort = None
    Tokenizer = None

from rae_core.interfaces.embedding import IEmbeddingProvider


class NativeEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider using local ONNX models."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        model_name: str = "nomic-embed-text-v1.5",
        max_length: int = 8192,
        normalize: bool = True,
        vector_name: str = "dense",
        use_gpu: bool = False,
    ):
        """Initialize ONNX provider.

        Args:
            model_path: Path to .onnx model file.
            tokenizer_path: Path to tokenizer.json file.
            model_name: Name of the model.
            max_length: Maximum sequence length.
            normalize: Whether to L2-normalize vectors.
            vector_name: Name of the vector space this provider serves.
            use_gpu: Whether to attempt using GPU (CUDA).
        """
        if ort is None or Tokenizer is None:
            raise ImportError(
                "onnxruntime and tokenizers are required. "
                "Install with: pip install onnxruntime tokenizers"
            )

        self.model_path = str(model_path)
        self.tokenizer_path = str(tokenizer_path)
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.vector_name = vector_name

        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        # Enable truncation but NOT fixed-length padding
        self.tokenizer.enable_truncation(max_length=max_length)
        # Dynamic padding: pad to the longest sequence in the batch
        self.tokenizer.enable_padding()

        # Load ONNX Model
        # Use CUDA if requested and available, else CPU
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            # Only add CUDA if it actually works (check_device fails if driver is missing)
            try:
                temp_session = ort.InferenceSession(
                    self.model_path, providers=["CUDAExecutionProvider"]
                )
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                del temp_session
            except Exception:
                pass

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[
            0
        ].name  # Usually 'last_hidden_state'

        # Determine dimension dynamically by running a dummy inference
        try:
            # Create a dummy input (batch_size=1, seq_len=1)
            dummy_ids = np.array([[1]], dtype=np.int64)
            dummy_mask = np.array([[1]], dtype=np.int64)
            dummy_type = np.array([[0]], dtype=np.int64)
            inputs = {"input_ids": dummy_ids, "attention_mask": dummy_mask}

            # Check for token_type_ids
            input_names = [i.name for i in self.session.get_inputs()]
            if "token_type_ids" in input_names:
                inputs["token_type_ids"] = dummy_type

            outputs = self.session.run(None, inputs)
            # outputs[0] is (batch, seq, dim)
            self._dimension = outputs[0].shape[-1]
        except Exception as e:
            raise RuntimeError(f"Failed to inspect ONNX model dimension: {e}")

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return int(self._dimension)

    def _mean_pooling(
        self, last_hidden_state: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Perform Mean Pooling on last hidden state."""
        # last_hidden_state: (batch, seq, dim)
        # attention_mask: (batch, seq)

        # Expand mask to (batch, seq, dim)
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(
            last_hidden_state.dtype
        )

        # Sum embeddings (ignoring padding)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)

        # Sum mask (count of tokens)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # Avoid div by zero

        return cast(np.ndarray, sum_embeddings / sum_mask)

    def _normalize_l2(self, vectors: np.ndarray) -> np.ndarray:
        """Perform L2 normalization."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return cast(np.ndarray, vectors / np.clip(norms, a_min=1e-9, a_max=None))

    async def embed_text(
        self, text: str, task_type: str = "search_document"
    ) -> list[float]:
        """Embed a single text string."""
        vectors = await self.embed_batch([text], task_type=task_type)
        return vectors[0]

    async def embed_batch(
        self, texts: list[str], task_type: str = "search_document"
    ) -> list[list[float]]:
        """Embed a batch of texts."""
        # Preprocessing for Nomic (Matryoshka / v1.5)
        # Requires "search_query: " or "search_document: " prefix
        if "nomic" in self.model_name.lower():
            prefix = ""
            if task_type == "search_query":
                prefix = "search_query: "
            elif task_type == "search_document":
                prefix = "search_document: "

            # Apply prefix if not already present
            processed_texts = []
            for t in texts:
                if t.startswith("search_query:") or t.startswith("search_document:"):
                    processed_texts.append(t)
                else:
                    processed_texts.append(f"{prefix}{t}")
            texts = processed_texts

        # 1. Tokenize
        encoded = self.tokenizer.encode_batch(texts)

        # Prepare inputs for ONNX
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Check if model needs token_type_ids
        input_names = [i.name for i in self.session.get_inputs()]
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        else:
            # Fallback for models that might not have type_ids in encoded but need it in session
            # (though tokenizers usually handle this)
            pass

        # 2. Run Inference
        outputs = self.session.run(None, inputs)

        # Output is typically last_hidden_state (batch, seq, dim)
        last_hidden_state = outputs[0]

        # 3. Pooling (Mean)
        embeddings = self._mean_pooling(last_hidden_state, attention_mask)

        # 4. Normalization (L2)
        if self.normalize:
            embeddings = self._normalize_l2(embeddings)

        return cast(list[list[float]], embeddings.tolist())
