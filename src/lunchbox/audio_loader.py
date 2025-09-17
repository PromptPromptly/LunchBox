"""Utilities for loading and saving audio clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class AudioClip:
    """Simple container for audio data.

    The internal representation always uses a two-dimensional float32 array
    shaped as ``(num_samples, num_channels)``.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim != 2:
            raise ValueError(
                "AudioClip samples must be a 1-D or 2-D array; "
                f"received shape {arr.shape}."
            )
        self.samples = arr
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")

    @property
    def num_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def num_channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return float(self.num_samples) / float(self.sample_rate)

    def copy(self) -> "AudioClip":
        return AudioClip(np.copy(self.samples), int(self.sample_rate))

    def to_mono(self) -> np.ndarray:
        return np.mean(self.samples, axis=1)

    def to_mono_clip(self) -> "AudioClip":
        return AudioClip(self.to_mono(), self.sample_rate)

    def channel(self, index: int) -> np.ndarray:
        return self.samples[:, index]

    def with_samples(self, samples: np.ndarray) -> "AudioClip":
        return AudioClip(samples, self.sample_rate)

    def normalized(self) -> "AudioClip":
        maximum = np.max(np.abs(self.samples))
        if maximum <= 0.0:
            return self.copy()
        return AudioClip(self.samples / maximum, self.sample_rate)

    def slice(self, start_sample: int, end_sample: int) -> "AudioClip":
        start = max(0, int(start_sample))
        end = min(int(end_sample), self.num_samples)
        if end <= start:
            raise ValueError("Slice end must be greater than start.")
        return AudioClip(self.samples[start:end], self.sample_rate)

    def pad_to_length(self, num_samples: int) -> "AudioClip":
        if num_samples <= self.num_samples:
            return self.copy()
        pad_amount = num_samples - self.num_samples
        pad = np.zeros((pad_amount, self.num_channels), dtype=np.float32)
        samples = np.concatenate([self.samples, pad], axis=0)
        return AudioClip(samples, self.sample_rate)

    def mix(self, other: "AudioClip") -> "AudioClip":
        if other.sample_rate != self.sample_rate:
            raise ValueError("Sample rates must match to mix audio clips.")
        max_len = max(self.num_samples, other.num_samples)
        a = self.pad_to_length(max_len).samples
        b = other.pad_to_length(max_len).samples
        return AudioClip(a + b, self.sample_rate)

    def resampled(self, target_sample_rate: int) -> "AudioClip":
        if target_sample_rate <= 0:
            raise ValueError("Target sample rate must be positive.")
        if target_sample_rate == self.sample_rate:
            return self.copy()
        samples = _resample_audio(self.samples, self.sample_rate, target_sample_rate)
        return AudioClip(samples, target_sample_rate)


def load_audio(
    path: str | Path,
    target_sample_rate: int | None = None,
    mono: bool = False,
) -> AudioClip:
    """Load an audio file from disk."""

    data, sr = sf.read(path, always_2d=True)
    data = data.astype(np.float32)
    if target_sample_rate and target_sample_rate != sr:
        data = _resample_audio(data, sr, target_sample_rate)
        sr = target_sample_rate
    clip = AudioClip(data, int(sr))
    return clip.to_mono_clip() if mono else clip


def save_audio(path: str | Path, clip: AudioClip) -> None:
    """Save an audio clip to disk."""

    sf.write(path, clip.samples, clip.sample_rate)


def _resample_audio(
    data: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    if original_rate == target_rate:
        return data
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive for resampling.")
    if data.ndim != 2:
        raise ValueError("Audio data must be two-dimensional for resampling.")
    gcd = np.gcd(original_rate, target_rate)
    up = target_rate // gcd
    down = original_rate // gcd
    resampled = [resample_poly(channel, up, down) for channel in data.T]
    min_len = min(len(ch) for ch in resampled)
    clipped = np.vstack([ch[:min_len] for ch in resampled]).T
    return clipped.astype(np.float32)


def concatenate(clips: Iterable[AudioClip]) -> AudioClip:
    """Concatenate multiple clips sequentially."""

    clips = list(clips)
    if not clips:
        raise ValueError("At least one clip is required for concatenation.")
    sample_rate = clips[0].sample_rate
    for clip in clips[1:]:
        if clip.sample_rate != sample_rate:
            raise ValueError("All clips must share the same sample rate.")
    data = np.concatenate([clip.samples for clip in clips], axis=0)
    return AudioClip(data, sample_rate)
