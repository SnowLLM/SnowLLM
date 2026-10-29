# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil

from setuptools.command.build_py import build_py


class BundleRecipes(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent.parent / "recipes"
        files = sorted(source.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no recipes to bundle in {source}")
        destination = Path(self.build_lib) / "snowllm/hub/bundle"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for path in files:
            shutil.copyfile(path, destination / path.name)
