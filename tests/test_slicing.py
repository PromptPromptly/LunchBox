import numpy as np
import pytest

from lunchbox.audio_loader import AudioClip
from lunchbox.slicing import SliceRegion, TransientSlicer


def test_slice_region_duration():
    region = SliceRegion(start=0, end=4410)
    assert region.duration(sample_rate=44100) == pytest.approx(0.1)


def test_transient_slicer_respects_minimum_duration(monkeypatch):
    sample_rate = 1000
    clip = AudioClip(np.zeros((sample_rate, 1), dtype=np.float32), sample_rate=sample_rate)
    slicer = TransientSlicer(hop_length=10, min_duration=0.05, backtrack=False)

    def fake_onset_strength(y, sr, hop_length):
        assert y.shape[0] == sample_rate
        assert sr == sample_rate
        assert hop_length == slicer.hop_length
        return np.ones(32)

    def fake_onset_detect(**kwargs):
        return np.array([1, 6, 9])

    def fake_frames_to_samples(frames, hop_length):
        return np.array(frames) * hop_length

    monkeypatch.setattr("librosa.onset.onset_strength", fake_onset_strength)
    monkeypatch.setattr("librosa.onset.onset_detect", fake_onset_detect)
    monkeypatch.setattr("librosa.frames_to_samples", fake_frames_to_samples)

    slices = slicer.slice(clip)

    assert [(s.start, s.end) for s in slices] == [(0, 60), (60, sample_rate)]
