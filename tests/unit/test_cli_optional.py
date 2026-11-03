"""The core API must import without optional CLI packages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_public_api_imports_without_cli_dependencies() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH")))
    )
    script = dedent(
        """
        import importlib.abc
        import sys

        class BlockCliDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"prompt_toolkit", "rich"}:
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, BlockCliDependencies())
        from odsbox_seman import ModelSearcher, cli_main

        assert callable(ModelSearcher)
        assert callable(cli_main)
        assert "odsbox_seman.cli.repl" not in sys.modules
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=source_root.parent,
        env=environment,
    )
