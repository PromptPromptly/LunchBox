import json
from pathlib import Path

import numpy as np
import pytest

from lunchbox.audio_loader import AudioClip
from lunchbox.project import LunchBoxProject
from lunchbox.slicing import SliceRegion
from lunchbox.stem_separation import (
    StemSeparator,
    _filter_band,
    _ensure_channels,
)



def test_stem_separator_rejects_unknown_backend():
    with pytest.raises(ValueError):
        StemSeparator("invalid")



def test_naive_separator_returns_requested_stems(monkeypatch):
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)

    harmonic = np.sin(2 * np.pi * 110.0 * t).astype(np.float32)[:, np.newaxis]
    percussive = np.zeros_like(harmonic)
    percussive[:: sample_rate // 10] = 1.0
    mixed = harmonic + percussive
    clip = AudioClip(mixed, sample_rate=sample_rate)

    def fake_hpss(_):
        return harmonic, percussive

    monkeypatch.setattr("lunchbox.stem_separation._compute_hpss", fake_hpss)

    separator = StemSeparator("naive")
    stems = separator.separate(clip, stems=["bass", "drums", "other"])

    assert set(stems) == {"bass", "drums", "other"}
    assert stems["bass"].sample_rate == sample_rate
    assert stems["drums"].sample_rate == sample_rate
    assert stems["other"].sample_rate == sample_rate

    drums = stems["drums"].samples[:, 0]
    bass = stems["bass"].samples[:, 0]
    residual = stems["other"].samples[:, 0]

    assert np.allclose(drums, percussive[:, 0], atol=1e-4)
    assert np.max(np.abs(bass)) > 0.1
    assert np.allclose(residual + drums + bass, mixed[:, 0], atol=1e-4)



def test_filter_band_validates_bounds():
    samples = np.ones((32, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        _filter_band(samples, sample_rate=44100, low=5000.0, high=1000.0)



def test_ensure_channels_adjusts_shapes():
    mono = np.ones((8, 1), dtype=np.float32)
    stereo = _ensure_channels(mono, num_channels=2)
    assert stereo.shape == (8, 2)

    wide = np.ones((8, 4), dtype=np.float32)
    downmixed = _ensure_channels(wide, num_channels=1)
    assert downmixed.shape == (8, 1)



def test_project_exports_and_metadata(tmp_path):
    samples = np.linspace(-0.5, 0.5, 64, dtype=np.float32)[:, np.newaxis]
    clip = AudioClip(samples, sample_rate=16000)
    project = LunchBoxProject(Path("song.wav"), clip)

    project.add_stems({"vocals": clip})
    project.add_slices([SliceRegion(start=0, end=16, label="intro")])

    audio_dir = tmp_path / "stems"
    project.export_audio(audio_dir, include_original=True)
    assert (audio_dir / "original.wav").exists()
    assert (audio_dir / "stem_vocals.wav").exists()

    slice_dir = tmp_path / "slices"
    project.export_slices(slice_dir)
    assert any(f.name.startswith("slice_") for f in slice_dir.iterdir())

    metadata_path = tmp_path / "metadata.json"
    project.save_metadata(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source"].endswith("song.wav")
    assert metadata["stems"] == ["vocals"]
    assert metadata["slices"][0]["label"] == "intro"
