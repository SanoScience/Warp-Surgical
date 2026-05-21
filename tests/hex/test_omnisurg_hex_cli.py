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
