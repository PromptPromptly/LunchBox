import numpy as np
import pytest

from lunchbox.audio_loader import AudioClip, concatenate, load_audio, save_audio


def test_audio_clip_from_mono_array_has_expected_shape():
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    clip = AudioClip(samples, sample_rate=48000)

    assert clip.samples.shape == (4, 1)
    assert clip.num_channels == 1
    assert clip.num_samples == 4
    assert pytest.approx(clip.duration, rel=1e-6) == 4 / 48000


def test_audio_clip_resampled_updates_rate_and_length():
    rng = np.random.default_rng(1234)
    samples = rng.standard_normal((16, 2), dtype=np.float32)
    clip = AudioClip(samples, sample_rate=16000)

    resampled = clip.resampled(8000)

    assert resampled.sample_rate == 8000
    assert resampled.samples.shape == (8, 2)
    assert resampled.samples.dtype == np.float32



def test_audio_clip_mix_requires_matching_rate():
    clip_a = AudioClip(np.zeros((8, 1), dtype=np.float32), sample_rate=44100)
    clip_b = AudioClip(np.zeros((8, 1), dtype=np.float32), sample_rate=48000)

    with pytest.raises(ValueError):
        _ = clip_a.mix(clip_b)



def test_concatenate_requires_consistent_sample_rate():
    clip_a = AudioClip(np.zeros((4, 1), dtype=np.float32), sample_rate=22050)
    clip_b = AudioClip(np.zeros((4, 1), dtype=np.float32), sample_rate=44100)

    with pytest.raises(ValueError):
        concatenate([clip_a, clip_b])



def test_load_and_save_audio_roundtrip(tmp_path):
    samples = np.column_stack(
        [np.linspace(-1.0, 1.0, 32, dtype=np.float32), np.linspace(1.0, -1.0, 32, dtype=np.float32)]
    )
    clip = AudioClip(samples, sample_rate=22050)

    path = tmp_path / "test.wav"
    save_audio(path, clip)

    loaded = load_audio(path)

    assert loaded.sample_rate == clip.sample_rate
    assert loaded.samples.shape == clip.samples.shape
    assert np.allclose(loaded.samples, clip.samples, atol=1e-4)
