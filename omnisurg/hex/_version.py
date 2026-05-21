# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("newton-cutting")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"
