"""Loading pulsar_core.so registers torch.ops.pulsar.* kernels and the
TorchBind runtime as torch.classes.pulsar.Engine (see engine.py).
"""

from importlib import resources

import torch

# CMake target `_core_impl`'s OUTPUT_NAME, with PREFIX "" (no "lib" prefix).
# Linux/CUDA only, so the .so suffix is fixed. Installed at the top of the
# pulsar package, not under core/.
_SONAME = "pulsar_core.so"

_loaded = False


def load() -> None:
    """In an editable install, this source lives in the source tree while
    CMake installs the library into the site-packages copy; both are on the
    pulsar package's __path__, so files() finds either.
    """
    global _loaded
    if _loaded:
        return
    lib = resources.files("pulsar").joinpath(_SONAME)
    if not lib.is_file():
        raise RuntimeError(
            f"engine library {_SONAME} not found in the pulsar package; build the "
            "extension (uv sync / uv build) so CMake installs it into pulsar/"
        )
    torch.ops.load_library(str(lib))
    _loaded = True


load()
