"""PostgreSQL implementation of IDatabaseProvider."""

from typing import Any, AsyncContextManager, cast

import asyncpg

from rae_core.interfaces.database import (
    IDatabaseConnection,
    IDatabaseProvider,
    ITransaction,
)


class PostgresTransaction(ITransaction):
    """PostgreSQL transaction implementation."""

    def __init__(self, transaction: asyncpg.transaction.Transaction):
        self._transaction = transaction

    async def __aenter__(self) -> "PostgresTransaction":
        await self._transaction.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._transaction.__aexit__(exc_type, exc_val, exc_tb)

    async def commit(self) -> None:
        """Commit the transaction."""
        await self._transaction.commit()

    async def rollback(self) -> None:
        """Rollback the transaction."""
        await self._transaction.rollback()


class PostgresDatabaseConnection(IDatabaseConnection):
    """PostgreSQL connection implementation."""

    def __init__(self, connection: asyncpg.Connection):
        self._connection = connection

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL query."""
        return cast(str, await self._connection.execute(query, *args))

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Fetch multiple rows from a SQL query."""
        records = await self._connection.fetch(query, *args)
        return [dict(r) for r in records]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """Fetch a single row from a SQL query."""
        record = await self._connection.fetchrow(query, *args)
        return dict(record) if record else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value from a SQL query."""
        return await self._connection.fetchval(query, *args)

    async def executemany(self, query: str, args: list[Any]) -> str:
        """Execute a SQL query for multiple arguments."""
        return cast(str, await self._connection.executemany(query, args))

    def transaction(self) -> ITransaction:
        """Start a new transaction."""
        return PostgresTransaction(self._connection.transaction())


class PostgresAcquireContext:
    """Context manager for acquiring a PostgreSQL connection."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._context = pool.acquire()
        self._connection: PostgresDatabaseConnection | None = None

    async def __aenter__(self) -> IDatabaseConnection:
        conn = await self._context.__aenter__()
        self._connection = PostgresDatabaseConnection(conn)
        return self._connection

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._context.__aexit__(exc_type, exc_val, exc_tb)


class PostgresDatabaseProvider(IDatabaseProvider):
    """Database provider using asyncpg for PostgreSQL."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        record = await self.pool.fetchrow(query, *args)
        return dict(record) if record else None

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        records = await self.pool.fetch(query, *args)
        return [dict(r) for r in records]

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self.pool.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return cast(str, await self.pool.execute(query, *args))

    async def executemany(self, query: str, args: list[Any]) -> str:
        return cast(str, await self.pool.executemany(query, args))

    async def close(self) -> None:
        await self.pool.close()

    def acquire(self) -> AsyncContextManager[IDatabaseConnection]:
        return cast(
            AsyncContextManager[IDatabaseConnection], PostgresAcquireContext(self.pool)
        )
