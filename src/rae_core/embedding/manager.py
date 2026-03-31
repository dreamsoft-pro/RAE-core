from rae_core.interfaces.embedding import IEmbeddingProvider


class EmbeddingManager(IEmbeddingProvider):
    """
    Manages multiple embedding providers.
    Implements IEmbeddingProvider to act as a default provider/proxy.
    """

    def __init__(
        self, default_provider: IEmbeddingProvider, default_model_name: str = "default"
    ) -> None:
        self.providers: dict[str, IEmbeddingProvider] = {
            default_model_name: default_provider
        }
        self.default_model_name = default_model_name
        self._default_provider = default_provider

    def register_provider(self, model_name: str, provider: IEmbeddingProvider) -> None:
        """Register a provider for a specific model/profile name."""
        self.providers[model_name] = provider

    def get_provider(self, model_name: str) -> IEmbeddingProvider | None:
        return self.providers.get(model_name)

    # IEmbeddingProvider implementation (delegates to default)
    async def embed_text(
        self, text: str, task_type: str = "search_document"
    ) -> list[float]:
        return await self._default_provider.embed_text(text, task_type=task_type)

    async def embed_batch(
        self, texts: list[str], task_type: str = "search_document"
    ) -> list[list[float]]:
        return await self._default_provider.embed_batch(texts, task_type=task_type)

    def get_dimension(self) -> int:
        return self._default_provider.get_dimension()

    # Manager methods
    async def generate_all_embeddings(
        self, texts: list[str], task_type: str = "search_document"
    ) -> dict[str, list[list[float]]]:
        """
        Generate embeddings for all registered models.
        Returns: Dict[model_name, embeddings_list]
        """
        results = {}
        for model_name, provider in self.providers.items():
            # TODO: Add parallelism here using asyncio.gather if providers are async/remote
            # For now, sequential to avoid complexity in initial implementation
            try:
                embeddings = await provider.embed_batch(texts, task_type=task_type)
                results[model_name] = embeddings
            except Exception as e:
                # Log error but continue with other models?
                # For now print/log and skip
                print(f"Failed to generate embeddings for {model_name}: {e}")
                results[model_name] = []

        return results
