"""Tests that every SECQUOIA module imports on its own.

No fixture data is needed. The test walks the package and imports each
module by name, then repeats the walk back to front in a separate
interpreter. Order matters, because a circular import only raises from one
of its two ends.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import textwrap

import pytest

import SECQUOIA


def _module_names() -> list[str]:
    """Every importable module under ``SECQUOIA``, sorted."""
    return sorted(
        info.name
        for info in pkgutil.walk_packages(SECQUOIA.__path__, "SECQUOIA.")
    )


MODULES = _module_names()


def test_the_package_exposes_modules_to_check():
    """Guard against the discovery itself silently returning nothing."""
    assert len(MODULES) > 50


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


@pytest.mark.smoke
@pytest.mark.slow
def test_modules_import_in_reverse_order():
    """The same modules, imported back-to-front in a clean interpreter.

    A circular import raises only when the cycle is entered from the end that
    is reached first. Running the reverse order in a subprocess covers the
    other end without polluting this interpreter's module cache.
    """
    script = textwrap.dedent("""
        import importlib, pkgutil, SECQUOIA
        names = sorted(
            (info.name for info in
             pkgutil.walk_packages(SECQUOIA.__path__, "SECQUOIA.")),
            reverse=True,
        )
        for name in names:
            importlib.import_module(name)
        print(len(names))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) == len(MODULES)
