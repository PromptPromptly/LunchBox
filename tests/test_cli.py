from lunchbox.cli import run_cli
from lunchbox.stem_separation import StemSeparator


def test_cli_list_stems_prints_defaults(capsys):
    run_cli(["dummy.wav", "--list-stems"])
    captured = capsys.readouterr()
    for stem in StemSeparator.DEFAULT_STEMS:
        assert stem in captured.out
