"""torch.classes.pulsar.Engine, the TorchBind runtime (pulsar/ext/engine.cpp).

pulsar_core.so is loaded when pulsar.core is imported; see engine.pyi (same
module name, paired stub) for Engine's typed method signatures.
"""

import torch

Engine = torch.classes.pulsar.Engine
