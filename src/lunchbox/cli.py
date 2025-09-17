"""Command line interface for the LunchBox audio toolkit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .audio_loader import load_audio
from .project import LunchBoxProject
from .slicing import TransientSlicer
from .stem_separation import StemSeparator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the source audio file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lunchbox_output"),
        help="Directory where artefacts should be written.",
    )
    parser.add_argument(
        "--backend",
        default="naive",
        choices=["naive", "spleeter", "demucs"],
        help="Which stem separation backend to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model identifier for ML-driven backends.",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=None,
        help="Explicit list of stems to extract (defaults to built-in list).",
    )
    parser.add_argument(
        "--no-slices",
        action="store_true",
        help="Disable transient slicing even if separation is performed.",
    )
    parser.add_argument(
        "--slice-min-duration",
        type=float,
        default=0.05,
        help="Minimum duration (in seconds) of any exported slice.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Resample the audio to this rate before processing.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Force mono processing by downmixing the input.",
    )
    parser.add_argument(
        "--export-original",
        action="store_true",
        help="Store a copy of the original audio alongside the stems.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Override the path where metadata should be written.",
    )
    parser.add_argument(
        "--list-stems",
        action="store_true",
        help="Print the default stems and exit.",
    )
    return parser


def run_cli(args: Iterable[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    if parsed.list_stems:
        print("Default stems:")
        for stem in StemSeparator.DEFAULT_STEMS:
            print(f"  - {stem}")
        return

    clip = load_audio(
        parsed.input,
        target_sample_rate=parsed.sample_rate,
        mono=parsed.mono,
    )

    project = LunchBoxProject(parsed.input, clip)

    stems_to_use = parsed.stems if parsed.stems else StemSeparator.DEFAULT_STEMS
    separator = StemSeparator(parsed.backend, model=parsed.model)
    stem_results = separator.separate(clip, stems=stems_to_use)
    project.add_stems(stem_results)

    output_dir = parsed.output
    project.export_audio(output_dir / "stems", include_original=parsed.export_original)

    if not parsed.no_slices:
        slicer = TransientSlicer(min_duration=parsed.slice_min_duration)
        slices = slicer.slice(clip)
        project.add_slices(slices)
        project.export_slices(output_dir / "slices")

    metadata_path = parsed.metadata or (output_dir / "metadata.json")
    project.save_metadata(metadata_path)
    print(project.describe())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli()
