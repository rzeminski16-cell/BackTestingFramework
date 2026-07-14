"""
btf — the BackTestingFramework command-line interface.

Run ``python -m btf --help`` (or just ``btf --help`` after ``pip install -e .``)
for the command overview. Every command is headless and scriptable; the GUIs
remain the interactive path.
"""

from Classes._version import __version__

__all__ = ["__version__"]
