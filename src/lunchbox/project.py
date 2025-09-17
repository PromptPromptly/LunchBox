"""Project container tying together stems and slices."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from .audio_loader import AudioClip, save_audio
from .slicing import SliceRegion


@dataclass
class LunchBoxProject:
    """Represents a processing session."""

    source_path: Path
    original: AudioClip
    stems: Dict[str, AudioClip] = field(default_factory=dict)
    slices: List[SliceRegion] = field(default_factory=list)

    def add_stems(self, stems: Mapping[str, AudioClip]) -> None:
        for name, clip in stems.items():
            self.stems[name] = clip

    def add_slices(self, slices: Iterable[SliceRegion]) -> None:
        self.slices.extend(list(slices))

    def export_audio(self, directory: Path, include_original: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if include_original:
            save_audio(directory / "original.wav", self.original)
        for name, clip in self.stems.items():
            filename = f"stem_{name.replace(' ', '_')}.wav"
            save_audio(directory / filename, clip)

    def export_slices(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index, region in enumerate(self.slices):
            clip = self.original.slice(region.start, region.end)
            label = f"_{region.label}" if region.label else ""
            filename = f"slice_{index:03d}{label}.wav"
            save_audio(directory / filename, clip)

    def save_metadata(self, path: Path) -> None:
        metadata = {
            "source": str(self.source_path),
            "sample_rate": self.original.sample_rate,
            "num_samples": self.original.num_samples,
            "stems": sorted(self.stems.keys()),
            "slices": [asdict(slice_) for slice_ in self.slices],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2))

    def describe(self) -> str:
        return (
            f"LunchBoxProject(source={self.source_path}, "
            f"stems={list(self.stems)}, slices={len(self.slices)})"
        )
