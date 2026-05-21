# SPDX-License-Identifier: Apache-2.0
"""Dataset loader registry."""

from __future__ import annotations

from pathlib import Path

from .types import DatasetLoader

_LOADERS: dict[str, DatasetLoader] = {}
_ALIASES: dict[str, str] = {}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def register_loader(loader: DatasetLoader, *aliases: str) -> DatasetLoader:
    canonical = _norm(loader.name)
    _LOADERS[canonical] = loader
    _ALIASES[canonical] = canonical
    for alias in aliases:
        _ALIASES[_norm(alias)] = canonical
    return loader


def get_loader(name: str) -> DatasetLoader:
    _ensure_builtin_loaders()
    key = _ALIASES.get(_norm(name))
    if key is None:
        choices = ", ".join(list_loaders())
        raise KeyError(f"unknown dataset loader {name!r}; choices: {choices}")
    return _LOADERS[key]


def list_loaders() -> list[str]:
    _ensure_builtin_loaders()
    return sorted(_LOADERS)


def choose_loader(path: str | Path | None) -> DatasetLoader:
    _ensure_builtin_loaders()
    for loader in _LOADERS.values():
        if loader.can_load(path):
            return loader
    raise ValueError(f"no dataset loader can read {path}")


def _ensure_builtin_loaders() -> None:
    if _LOADERS:
        return
    from .loaders.abdomen_atlas import AbdomenAtlasLoader
    from .loaders.digimouse import DigimouseLoader
    from .loaders.ircad_3d import ThreeDircadbLoader
    from .loaders.kits23 import Kits23Loader
    from .loaders.preprocessed_npz import PreprocessedNpzLoader
    from .loaders.synthetic_block import SyntheticBlockLoader
    from .loaders.vhp import VHPLoader

    register_loader(DigimouseLoader(), "digimouse")
    register_loader(VHPLoader(), "vhp")
    register_loader(AbdomenAtlasLoader(), "abdomen-atlas", "abdomen_atlas")
    register_loader(ThreeDircadbLoader(), "3dircadb1", "3dircadb2", "ircad-3d", "3d-ircadb")
    register_loader(Kits23Loader(), "kits-23", "kits2023", "kits-2023")
    register_loader(PreprocessedNpzLoader(), "preprocessed", "preprocessed-npz", "preprocessed_npz")
    register_loader(SyntheticBlockLoader(), "synthetic", "synthetic-block", "synthetic_block")
