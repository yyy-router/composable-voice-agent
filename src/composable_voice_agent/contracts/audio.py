"""Audio value objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioFormat:
    encoding: str = "pcm_s16le"
    sample_rate_hz: int = 16000
    channels: int = 1

    def __post_init__(self) -> None:
        if not self.encoding.strip():
            raise ValueError("encoding must be non-empty")
        if self.sample_rate_hz <= 0 or self.channels <= 0:
            raise ValueError("sample rate and channels must be positive")
