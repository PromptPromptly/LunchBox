# LunchBox

LunchBox is a lightweight digital-audio workstation (DAW) toolkit focused on
quickly slicing audio files and extracting rough stems such as vocals, drums,
and bass. The goal is to provide a scriptable foundation similar to tools like
Fruity Slicer or Slicex while remaining completely open source.

The project currently ships with:

- A `lunchbox` command line interface capable of loading most common audio
  files, slicing them on transient onsets, and exporting the slices as
  individual WAV files.
- A DSP-based stem separation backend that uses harmonic/percussive source
  separation combined with adaptive filter bands to approximate stems such as
  vocals, bass, kick, hats, and drums.
- Optional integration with machine-learning powered backends like
  [Spleeter](https://github.com/deezer/spleeter) and
  [Demucs](https://github.com/facebookresearch/demucs) when the corresponding
  Python packages are installed.
- A small Python API (`AudioClip`, `StemSeparator`, `TransientSlicer`, and
  `LunchBoxProject`) that can be embedded into notebooks or other tools.

## Installation

The project is published as a Python package. Install the core functionality
with:

```bash
pip install -e .
```

For higher quality ML-based stem separation install the optional dependencies:

```bash
pip install -e .[ml]
```

> **Note**: Spleeter and Demucs rely on TensorFlow and PyTorch respectively. The
> first time you enable these backends they will download pretrained models,
> which can take a few minutes.

## Command line usage

Run the toolkit against an audio file with:

```bash
lunchbox path/to/song.wav --output exports/
```

This will create:

- `exports/stems/` containing `stem_<name>.wav` files for each separated stem.
- `exports/slices/` containing transient-based slices of the original track.
- `exports/metadata.json` describing the session.

Useful flags:

| Flag | Description |
| --- | --- |
| `--backend` | Select `naive`, `spleeter`, or `demucs` for stem separation. |
| `--stems` | Specify which stems to export (default: drums, hats, kick, bass, vocals, other). |
| `--no-slices` | Skip transient slicing. |
| `--slice-min-duration` | Minimum length of each slice in seconds. |
| `--sample-rate` | Resample audio before processing. |
| `--mono` | Down-mix to mono prior to any processing. |
| `--export-original` | Save a copy of the original audio. |
| `--metadata` | Override where metadata is written. |
| `--list-stems` | Print the default stems and exit. |

## Running tests

The repository ships with an automated test suite covering the audio helpers,
stem separator, slicing logic, and CLI utilities. Install the optional testing
dependencies and execute the suite with:

```bash
pip install -e .[test]
pytest
```

## Python API overview

```python
from lunchbox import load_audio, StemSeparator, TransientSlicer, LunchBoxProject

clip = load_audio("song.wav")
separator = StemSeparator("naive")
stems = separator.separate(clip)

slicer = TransientSlicer(min_duration=0.1)
slices = slicer.slice(clip)

project = LunchBoxProject(Path("song.wav"), clip)
project.add_stems(stems)
project.add_slices(slices)
project.export_audio(Path("exports/stems"))
project.export_slices(Path("exports/slices"))
```

## Roadmap

- [ ] Expand the GUI/UX with an interactive waveform editor.
- [ ] Add unit tests for the DSP pipeline.
- [ ] Improve stem estimation quality by integrating mask-based approaches.

Contributions, bug reports, and ideas are very welcome!
