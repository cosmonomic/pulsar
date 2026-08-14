"""FakeTensor/export validation for the pulsar CUDA custom ops.

opcheck runs only the non-autograd subset: these ops have no autograd
registration (inference only), so test_autograd_registration and the
autograd AOT-dispatch subtests do not apply.
"""

import pytest
import torch

import pulsar.core.ops  # noqa: F401  (registers torch.ops.pulsar.* and their fakes)

_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pulsar ops have CUDA-only kernels"
)


def _rmsnorm_case():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(8, device="cuda", dtype=torch.float32)
    return (x, weight, 1e-6), {}


def _rope_case(dtype):
    x = torch.randn(2, 4, 8, device="cuda", dtype=dtype)
    pos = torch.arange(4, device="cuda", dtype=torch.int64)
    return (x, pos, 1000000.0), {}


def _attention_case(dtype):
    q = torch.randn(2, 2, 8, device="cuda", dtype=dtype)
    k = torch.randn(2, 4, 8, device="cuda", dtype=dtype)
    v = torch.randn(2, 4, 8, device="cuda", dtype=dtype)
    attention_mass = torch.zeros(2, 4, device="cuda", dtype=torch.float32)
    return (q, k, v, attention_mass, 0.35, 0), {}


def test_opcheck_rmsnorm():
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4, 8, device="cuda", dtype=dtype)
        weight = torch.randn(8, device="cuda", dtype=dtype)
        torch.library.opcheck(
            torch.ops.pulsar.rmsnorm, (x, weight, 1e-6), test_utils=_OPCHECK_UTILS
        )


def test_opcheck_rope():
    for dtype in (torch.float32, torch.bfloat16):
        args, kwargs = _rope_case(dtype)
        torch.library.opcheck(
            torch.ops.pulsar.rope, args, kwargs, test_utils=_OPCHECK_UTILS
        )


def test_opcheck_attn_causal():
    for dtype in (torch.float32, torch.bfloat16):
        args, kwargs = _attention_case(dtype)
        torch.library.opcheck(
            torch.ops.pulsar.attn_causal,
            args,
            kwargs,
            test_utils=_OPCHECK_UTILS,
        )


class _AllOps(torch.nn.Module):
    def forward(self, x, weight, pos, q, k, v, attention_mass):
        h = torch.ops.pulsar.rmsnorm(x, weight, 1e-6)
        h = torch.ops.pulsar.rope(h, pos, 1000000.0)
        o, mass = torch.ops.pulsar.attn_causal(q, k, v, attention_mass, 0.35, 0)
        return o, mass, h


def _export_inputs():
    x = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(8, device="cuda", dtype=torch.float32)
    pos = torch.arange(4, device="cuda", dtype=torch.int64)
    q = torch.randn(2, 2, 8, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    attention_mass = torch.zeros(2, 4, device="cuda", dtype=torch.float32)
    return (x, weight, pos, q, k, v, attention_mass)


def _exported_pulsar_targets(ep):
    targets = []
    for node in ep.graph.nodes:
        if node.op == "call_function" and "pulsar" in str(node.target):
            targets.append(str(node.target))
    return targets


def test_export_smoke():
    mod = _AllOps().cuda()
    inputs = _export_inputs()
    ep = torch.export.export(mod, inputs)
    targets = _exported_pulsar_targets(ep)
    assert any("rmsnorm" in t for t in targets)
    assert any("rope" in t for t in targets)
    assert any("attn_causal" in t for t in targets)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        raise SystemExit(0)

    results = []
    for name, fn in (
        ("opcheck rmsnorm", test_opcheck_rmsnorm),
        ("opcheck rope", test_opcheck_rope),
        ("opcheck attn_causal", test_opcheck_attn_causal),
    ):
        try:
            fn()
            print(f"PASS: {name} (test_utils={_OPCHECK_UTILS})")
            results.append(True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {name}: {exc}")
            results.append(False)

    try:
        mod = _AllOps().cuda()
        inputs = _export_inputs()
        ep = torch.export.export(mod, inputs)
        targets = _exported_pulsar_targets(ep)
        print(f"PASS: export smoke; surviving pulsar op nodes: {targets}")
        results.append(len(targets) == 3)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: export smoke: {exc}")
        results.append(False)

    raise SystemExit(0 if all(results) else 1)
