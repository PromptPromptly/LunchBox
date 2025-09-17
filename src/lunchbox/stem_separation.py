"""Tools to split an audio clip into musical stems."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .audio_loader import AudioClip


class SeparationError(RuntimeError):
    """Raised when stem separation cannot be performed."""


class StemSeparator:
    """Split an :class:`AudioClip` into component stems.

    Parameters
    ----------
    backend:
        Which separation backend to use. ``"naive"`` performs lightweight
        DSP-driven separation while ``"spleeter"`` and ``"demucs"`` rely on
        optional machine learning packages.
    model:
        Optional model identifier forwarded to the backend implementation.
    """

    DEFAULT_STEMS: tuple[str, ...] = (
        "drums",
        "hats",
        "kick",
        "bass",
        "vocals",
        "other",
    )

    def __init__(self, backend: str = "naive", model: str | None = None) -> None:
        backend = backend.lower()
        self.backend = backend
        self.model = model
        if backend not in {"naive", "spleeter", "demucs"}:
            raise ValueError(f"Unsupported backend '{backend}'.")
        if backend == "spleeter":
            self._backend_impl = _SpleeterBackend(model)
        elif backend == "demucs":
            self._backend_impl = _DemucsBackend(model)
        else:
            self._backend_impl = None

    def separate(
        self,
        clip: AudioClip,
        stems: Iterable[str] | None = None,
    ) -> Dict[str, AudioClip]:
        stems = tuple(stems) if stems is not None else self.DEFAULT_STEMS
        if self.backend == "naive":
            return self._separate_naive(clip, stems)
        if not self._backend_impl:
            raise SeparationError("Backend implementation is missing.")
        available = self._backend_impl.separate(clip)
        if not available:
            raise SeparationError("No stems were produced by the backend.")
        return {stem: available[stem] for stem in stems if stem in available}

    # ------------------------------------------------------------------
    # Naive DSP-based separation

    def _separate_naive(
        self,
        clip: AudioClip,
        stems: Iterable[str],
    ) -> Dict[str, AudioClip]:
        harmonic, percussive = _compute_hpss(clip.samples)
        outputs: Dict[str, AudioClip] = {}
        residual = np.zeros_like(clip.samples)
        for stem in stems:
            if stem.lower() == "other":
                continue
            data = self._render_naive_stem(stem.lower(), clip, harmonic, percussive)
            if data is None:
                continue
            outputs[stem] = AudioClip(data, clip.sample_rate)
            residual += data
        if any(stem.lower() == "other" for stem in stems):
            outputs["other"] = AudioClip(clip.samples - residual, clip.sample_rate)
        return outputs

    def _render_naive_stem(
        self,
        stem: str,
        clip: AudioClip,
        harmonic: np.ndarray,
        percussive: np.ndarray,
    ) -> np.ndarray | None:
        sr = clip.sample_rate
        if stem in {"drums", "percussion"}:
            return percussive
        if stem in {"hats", "hi-hats", "hihat"}:
            return _filter_band(percussive, sr, low=6000.0, high=None)
        if stem in {"kick", "kicks"}:
            return _filter_band(percussive, sr, low=20.0, high=160.0)
        if stem in {"snare", "snares"}:
            return _filter_band(percussive, sr, low=150.0, high=3000.0)
        if stem == "bass":
            return _filter_band(harmonic, sr, low=None, high=180.0)
        if stem in {"vocals", "voice"}:
            return _filter_band(harmonic, sr, low=120.0, high=5000.0)
        if stem in {"melody", "guitar", "keys"}:
            return _filter_band(harmonic, sr, low=400.0, high=None)
        return None


# ----------------------------------------------------------------------
# Backend implementations


class _SpleeterBackend:
    def __init__(self, model: str | None) -> None:
        try:
            from spleeter.separator import Separator  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise SeparationError(
                "Spleeter is not installed. Install lunchbox[ml] to enable this backend."
            ) from exc
        self.model = model or "spleeter:4stems"
        self.separator = Separator(self.model)

    def separate(self, clip: AudioClip) -> Dict[str, AudioClip]:
        target_sr = 44100
        working = clip.resampled(target_sr) if clip.sample_rate != target_sr else clip
        waveform = working.samples
        if waveform.shape[1] == 1:
            waveform = np.repeat(waveform, 2, axis=1)
        prediction = self.separator.separate(waveform)
        outputs: Dict[str, AudioClip] = {}
        for name, data in prediction.items():
            stem_clip = AudioClip(data, working.sample_rate)
            stem_clip = _match_channels(stem_clip, clip.num_channels)
            if stem_clip.sample_rate != clip.sample_rate:
                stem_clip = stem_clip.resampled(clip.sample_rate)
            outputs[name] = stem_clip
        return outputs


class _DemucsBackend:
    def __init__(self, model: str | None) -> None:
        try:
            from demucs.pretrained import get_model  # type: ignore
            from demucs.apply import apply_model  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise SeparationError(
                "Demucs is not installed. Install lunchbox[ml] to enable this backend."
            ) from exc
        self.model_name = model or "htdemucs"
        self._get_model = get_model
        self._apply_model = apply_model

    def separate(self, clip: AudioClip) -> Dict[str, AudioClip]:
        import torch  # type: ignore

        model = self._get_model(name=self.model_name)
        model.to("cpu")
        model.eval()
        waveform = torch.from_numpy(clip.samples.T).unsqueeze(0)
        with torch.no_grad():
            predictions = self._apply_model(model, waveform, split=True, overlap=0.25)
        prediction = predictions.squeeze(0).cpu().numpy()
        # Demucs returns shape (stems, channels, samples)
        outputs: Dict[str, AudioClip] = {}
        for name, data in zip(model.sources, prediction):
            data = data.transpose(1, 0).astype(np.float32)
            data = _ensure_channels(data, clip.num_channels)
            outputs[name] = AudioClip(data, clip.sample_rate)
        return outputs


# ----------------------------------------------------------------------
# Helper utilities


def _compute_hpss(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    harmonic_channels: List[np.ndarray] = []
    percussive_channels: List[np.ndarray] = []
    for channel in samples.T:
        harmonic, percussive = librosa.effects.hpss(channel)
        harmonic_channels.append(harmonic)
        percussive_channels.append(percussive)
    harmonic_arr = np.stack(harmonic_channels, axis=1)
    percussive_arr = np.stack(percussive_channels, axis=1)
    return harmonic_arr.astype(np.float32), percussive_arr.astype(np.float32)


def _filter_band(
    samples: np.ndarray,
    sample_rate: int,
    low: float | None,
    high: float | None,
    order: int = 6,
) -> np.ndarray:
    if low is None and high is None:
        return samples
    nyquist = sample_rate * 0.5
    if high is not None and high >= nyquist:
        high = nyquist * 0.99
    if low is not None and low <= 0.0:
        low = nyquist * 0.0001
    if high is not None and low is not None and low >= high:
        raise ValueError("Low cutoff must be less than high cutoff for band filtering.")
    if low is None:
        wn = high / nyquist
        btype = "lowpass"
    elif high is None:
        wn = low / nyquist
        btype = "highpass"
    else:
        wn = [low / nyquist, high / nyquist]
        btype = "bandpass"
    sos = butter(order, wn, btype=btype, output="sos")
    filtered = [sosfiltfilt(sos, channel) for channel in samples.T]
    return np.stack(filtered, axis=1).astype(np.float32)


def _match_channels(clip: AudioClip, num_channels: int) -> AudioClip:
    if clip.num_channels == num_channels:
        return clip
    samples = _ensure_channels(clip.samples, num_channels)
    return AudioClip(samples, clip.sample_rate)


def _ensure_channels(samples: np.ndarray, num_channels: int) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("Expected a two-dimensional array for channel adjustment.")
    current = samples.shape[1]
    if current == num_channels:
        return samples
    if num_channels == 1:
        return np.mean(samples, axis=1, keepdims=True)
    if current == 1:
        return np.repeat(samples, num_channels, axis=1)
    if current > num_channels:
        return samples[:, :num_channels]
    repeats = int(np.ceil(num_channels / current))
    expanded = np.tile(samples, (1, repeats))
    return expanded[:, :num_channels]
