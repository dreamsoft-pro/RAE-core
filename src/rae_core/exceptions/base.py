"""Base exceptions for RAE-core."""


class RAEError(Exception):
    """Base exception for all RAE-core errors."""

    pass


class StorageError(RAEError):
    """Exception raised for errors in storage adapters."""

    pass


class ValidationError(RAEError):
    """Exception raised for validation failures."""

    pass


class SecurityPolicyViolationError(RAEError):
    """Exception raised when a security policy is violated (ISO 27000)."""

    pass

class ContractViolationError(RAEError):
    """Exception raised when an agentic contract is violated."""
    pass


class InfrastructureError(RAEError):
    """Exception raised for underlying infrastructure failures (Redis, Qdrant, etc)."""
    pass
