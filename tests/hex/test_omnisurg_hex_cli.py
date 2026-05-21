from __future__ import annotations

import subprocess
import sys

from omnisurg.hex.cli import build_parser


def test_unified_hex_parser_prog():
    parser = build_parser()

    assert parser.prog == "omnisurg hex"


def test_python_module_hex_synthetic_headless_exit_after_init():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnisurg",
            "hex",
            "--dataset",
            "synthetic",
            "--size",
            "2",
            "--texture",
            "off",
            "--viewer",
            "headless",
            "--input-backend",
            "off",
            "--exit-after-init",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "[startup]" in result.stdout


def test_removed_spring_flags_are_rejected():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnisurg",
            "hex",
            "--dataset",
            "synthetic",
            "--size",
            "1",
            "--texture",
            "off",
            "--viewer",
            "headless",
            "--input-backend",
            "off",
            "--exit-after-init",
            "--edge-ke",
            "1.0",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --edge-ke" in result.stderr
