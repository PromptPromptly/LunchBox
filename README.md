# LunchBox

This project provides a minimal command-line tool for separating an audio file
into musical stems (vocals, drums, bass, and other elements). It uses the
[Spleeter](https://github.com/deezer/spleeter) library under the hood.

## Usage

1. Install dependencies:
   ```bash
   pip install spleeter
   ```

2. Separate an audio file:
   ```bash
   python audio_separator.py input.mp3 -o output_dir --stems 4
   ```

By default the script splits the track into four stems (vocals, drums, bass,
and other). Use `--stems 2` for vocals and accompaniment, or `--stems 5` for
vocals, drums, bass, piano, and other.
