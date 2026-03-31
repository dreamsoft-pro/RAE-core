"""Abstract database provider interface for RAE-core."""

from typing import Any, AsyncContextManager, Protocol, runtime_checkable


@runtime_checkable
class ITransaction(Protocol):
    """Abstract interface for database transactions."""

    async def __aenter__(self) -> "ITransaction": ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    async def commit(self) -> None:
        """Commit the transaction."""
        ...

    async def rollback(self) -> None:
        """Rollback the transaction."""
        ...


@runtime_checkable
class IDatabaseConnection(Protocol):
    """Abstract interface for database connections."""

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL query."""
        ...

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Fetch multiple rows from a SQL query."""
        ...

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """Fetch a single row from a SQL query."""
        ...

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value from a SQL query."""
        ...

    async def executemany(self, query: str, args: list[Any]) -> str:
        """Execute a SQL query for multiple arguments."""
        ...

    def transaction(self) -> ITransaction:
        """Start a new transaction."""
        ...


@runtime_checkable
class IDatabaseProvider(Protocol):
    """Abstract interface for database providers."""

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL query."""
        ...

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Fetch multiple rows from a SQL query."""
        ...

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """Fetch a single row from a SQL query."""
        ...

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value from a SQL query."""
        ...

    async def executemany(self, query: str, args: list[Any]) -> str:
        """Execute a SQL query for multiple arguments."""
        ...

    async def close(self) -> None:
        """Close the database connection."""
        ...

    def acquire(self) -> AsyncContextManager[IDatabaseConnection]:
        """Acquire a connection from the pool."""
        ...
