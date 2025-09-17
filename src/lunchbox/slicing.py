"""Audio slicing utilities based on transient detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import librosa
import numpy as np

from .audio_loader import AudioClip


@dataclass
class SliceRegion:
    """Represents an audio slice bounded by sample indices."""

    start: int
    end: int
    label: str = ""

    def duration(self, sample_rate: int) -> float:
        return float(self.end - self.start) / float(sample_rate)


class TransientSlicer:
    """Detect onsets to break audio into manageable slices."""

    def __init__(
        self,
        hop_length: int = 512,
        backtrack: bool = True,
        pre_max: float = 0.03,
        post_max: float = 0.00,
        pre_avg: float = 0.10,
        post_avg: float = 0.10,
        delta: float = 0.2,
        wait: float = 0.03,
        min_duration: float = 0.05,
    ) -> None:
        self.hop_length = hop_length
        self.backtrack = backtrack
        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.delta = delta
        self.wait = wait
        self.min_duration = min_duration

    def slice(self, clip: AudioClip) -> List[SliceRegion]:
        y = clip.to_mono()
        sr = clip.sample_rate
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            pre_max=self._seconds_to_frames(self.pre_max, sr),
            post_max=self._seconds_to_frames(self.post_max, sr),
            pre_avg=self._seconds_to_frames(self.pre_avg, sr),
            post_avg=self._seconds_to_frames(self.post_avg, sr),
            delta=self.delta,
            wait=self._seconds_to_frames(self.wait, sr),
        )
        onset_samples = librosa.frames_to_samples(frames, hop_length=self.hop_length)
        onset_samples = np.unique(np.clip(onset_samples, 0, clip.num_samples))
        min_samples = int(self.min_duration * sr)
        if min_samples <= 0:
            min_samples = 1
        slices: List[SliceRegion] = []
        start = 0
        for onset in onset_samples:
            if onset - start < min_samples:
                continue
            slices.append(SliceRegion(start=start, end=int(onset)))
            start = int(onset)
        if clip.num_samples - start >= min_samples:
            slices.append(SliceRegion(start=start, end=clip.num_samples))
        if not slices:
            slices.append(SliceRegion(start=0, end=clip.num_samples))
        return slices

    def _seconds_to_frames(self, seconds: float, sample_rate: int) -> int:
        frames = int(round(seconds * sample_rate / self.hop_length))
        return max(1, frames)
