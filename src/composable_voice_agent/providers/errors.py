"""Provider error contracts."""


class ProviderError(Exception):
    """A provider failed without exposing its internal response details."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProviderRequestError(ProviderError):
    """A provider request could not be completed."""


class ProviderProtocolError(ProviderError):
    """A provider returned an unsupported response sequence."""
