import json
import subprocess
from pathlib import Path

import torch.utils
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

TARGET = "pulsar"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = self.root
        subprocess.run(
            [
                "xmake",
                "config",
                "-P",
                root,
                "-y",
                "-m",
                "release",
                "--tests=n",
                f"--torch_cmake_prefix={torch.utils.cmake_prefix_path}",
            ],
            check=True,
        )
        subprocess.run(["xmake", "build", "-P", root, "-y", TARGET], check=True)
        library = Path(root) / self._targetfile(root)

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        force_include = (
            "force_include_editable" if version == "editable" else "force_include"
        )
        build_data[force_include][str(library)] = f"pulsar/{library.name}"

    def _targetfile(self, root):
        shown = subprocess.run(
            ["xmake", "show", "-P", root, "-t", TARGET, "--json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        info, _ = json.JSONDecoder().raw_decode(shown)
        return info["targetfile"]
