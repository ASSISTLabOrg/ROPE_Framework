#!/usr/bin/env python3
"""
Generate synthetic test fixtures for ROPE unit tests.

Creates:
  tests/fixtures/test_models/
    base_model_00.onnx .. base_model_14.onnx  — 15 base temporal models
    meta_model.onnx                            — ensemble meta-model (inner ONNX)
    coae_decoder.onnx                          — COAE decoder (ONNX Runtime path)
    coae_decoder.pt                            — COAE decoder (LibTorch path)
    stats_ts.bin                               — feature normalizer stats (identity)
    stats_cae.bin                              — CAE denormalizer stats (identity)
  tests/fixtures/
    sw_test.csv    — space weather stub (5 hourly rows)
    ic_test.csv    — IC table stub (2×2 grid)

All models return constant zeros of the correct output shape.
Stats use identity (mu=0, sigma=1) so normalization is a no-op.
With identity stats and zero-output models, the pipeline produces density=1.0
everywhere (10^0 = 1.0 from the CAE denormalizer), satisfying the positivity
invariant tested by test_pipeline.cpp.

Run locally:
    pip install onnx
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python tests/gen_fixtures.py
"""

import struct
import sys
from pathlib import Path

ROOT      = Path(__file__).parent
MODELS    = ROOT / "fixtures" / "test_models"
FIXTURES  = ROOT / "fixtures"

MODELS.mkdir(parents=True, exist_ok=True)
FIXTURES.mkdir(parents=True, exist_ok=True)

# Pipeline constants (must match source constants exactly)
K          = 10       # latent_dim  (pipeline.cpp K=10)
S          = 3        # seq_len     (pipeline.cpp S=3)
D          = 16       # total_dim   (K=10 + driver_dim=6, DriverCols::Six)
M          = 15       # n_base_models
GRID_VOXELS = 72 * 36 * 45   # 116,640


# ---------------------------------------------------------------------------
# Binary stats files
# ---------------------------------------------------------------------------

def write_stats_bin(path: Path, ndim: int, shape: list[int], mu: list[float], sigma: list[float]) -> None:
    """Write a stats .bin file in the format expected by rope::io::Stats::load()."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", ndim))
        for s in shape:
            f.write(struct.pack("<I", s))
        for v in mu:
            f.write(struct.pack("<f", v))
        for v in sigma:
            f.write(struct.pack("<f", v))

print("Writing stats_ts.bin …")
write_stats_bin(
    MODELS / "stats_ts.bin",
    ndim=1, shape=[D],
    mu=[0.0] * D,
    sigma=[1.0] * D,
)

print("Writing stats_cae.bin …")
write_stats_bin(
    MODELS / "stats_cae.bin",
    ndim=1, shape=[1],
    mu=[0.0],
    sigma=[1.0],
)


# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------

SW_CSV = """\
datetime,f10,kp
2023-12-31T22:00:00,150.0,2.0
2023-12-31T23:00:00,150.0,2.0
2024-01-01T00:00:00,150.0,2.0
2024-01-01T01:00:00,150.0,2.0
2024-01-01T02:00:00,150.0,2.0
"""

# 2×2 grid bracketing the test query (f10=150, kp=2)
IC_HEADER = "F10,Kp," + ",".join(f"y{i+1}" for i in range(K))
IC_ZEROS  = ",".join(["0.0"] * K)
IC_CSV = IC_HEADER + "\n"
for f10 in [100.0, 200.0]:
    for kp in [1.0, 3.0]:
        IC_CSV += f"{f10},{kp},{IC_ZEROS}\n"

print("Writing sw_test.csv …")
(FIXTURES / "sw_test.csv").write_text(SW_CSV)

print("Writing ic_test.csv …")
(FIXTURES / "ic_test.csv").write_text(IC_CSV)


# ---------------------------------------------------------------------------
# ONNX fixtures
# ---------------------------------------------------------------------------

try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx not installed. Run: pip install onnx numpy", file=sys.stderr)
    sys.exit(1)


def _gemm_zero(name: str, in_shape, out_dim: int) -> onnx.ModelProto:
    """Build a tiny ONNX model: Reshape → Gemm(W=0, b=0).

    in_shape must be [batch, *rest] where product(rest) is the Gemm input width.
    Output shape is [batch, out_dim].
    A value of 0 in the reshape shape means "copy from input".
    """
    in_flat = int(np.prod(in_shape[1:]))
    W  = numpy_helper.from_array(np.zeros((in_flat, out_dim), dtype=np.float32), name="W")
    b  = numpy_helper.from_array(np.zeros(out_dim, dtype=np.float32),            name="b")
    rs = numpy_helper.from_array(
        np.array([0] + [in_flat], dtype=np.int64), name="rs")

    nodes = [
        helper.make_node("Reshape", ["input", "rs"],    ["flat"]),
        helper.make_node("Gemm",    ["flat", "W", "b"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, list(in_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [in_shape[0], out_dim])],
        initializer=[W, b, rs],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


def _meta_inner() -> onnx.ModelProto:
    """Meta inner ONNX: {T, S, D} → {T, M}.

    Output shape is 2-D with inner_out_shape[1]==M so MetaModel uses
    time_level mode (coeff_mode=False, equal weights 1/M → mean of base outputs).
    Dynamic T is handled by using 0 in the Reshape shape vector.
    """
    W  = numpy_helper.from_array(
        np.full((S * D, M), fill_value=1.0 / M, dtype=np.float32), name="W_meta")
    b  = numpy_helper.from_array(np.zeros(M, dtype=np.float32),    name="b_meta")
    rs = numpy_helper.from_array(
        np.array([0, S * D], dtype=np.int64), name="rs_meta")

    nodes = [
        helper.make_node("Reshape", ["input", "rs_meta"],        ["flat"]),
        helper.make_node("Gemm",    ["flat", "W_meta", "b_meta"],["output"]),
    ]
    graph = helper.make_graph(
        nodes, "meta_inner",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, ["T", S, D])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["T", M])],
        initializer=[W, b, rs],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


def _decoder_onnx() -> onnx.ModelProto:
    """COAE decoder ONNX: {batch, K} → {batch, GRID_VOXELS} filled with zeros.

    Uses ConstantOfShape to avoid a ~4 MB weight matrix.
    """
    idx0  = numpy_helper.from_array(np.array(0,            dtype=np.int64), name="idx0")
    axes0 = numpy_helper.from_array(np.array([0],          dtype=np.int64), name="axes0")
    nvox  = numpy_helper.from_array(np.array([GRID_VOXELS], dtype=np.int64), name="nvox")

    zero_val = helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0])

    nodes = [
        helper.make_node("Shape",          ["input"],               ["in_shape"]),
        helper.make_node("Gather",         ["in_shape", "idx0"],    ["batch_scalar"], axis=0),
        helper.make_node("Unsqueeze",      ["batch_scalar", "axes0"], ["batch_1d"]),
        helper.make_node("Concat",         ["batch_1d", "nvox"],    ["out_shape"], axis=0),
        helper.make_node("ConstantOfShape",["out_shape"],           ["output"],
                         value=zero_val),
    ]
    graph = helper.make_graph(
        nodes, "coae_decoder",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, ["batch", K])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", GRID_VOXELS])],
        initializer=[idx0, axes0, nvox],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


print("Writing 15 base ONNX models …")
for i in range(M):
    name = f"base_model_{i:02d}.onnx"
    m = _gemm_zero(f"base_{i:02d}", in_shape=[1, S, D], out_dim=K)
    onnx.save(m, str(MODELS / name))

print("Writing meta_model.onnx …")
onnx.save(_meta_inner(), str(MODELS / "meta_model.onnx"))

print("Writing coae_decoder.onnx …")
onnx.save(_decoder_onnx(), str(MODELS / "coae_decoder.onnx"))


# ---------------------------------------------------------------------------
# TorchScript fixture
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class _FakeDecoder(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 116640)  # GRID_VOXELS = 72*36*45

    print("Writing coae_decoder.pt …")
    scripted = torch.jit.script(_FakeDecoder())
    scripted.save(str(MODELS / "coae_decoder.pt"))

except ImportError:
    print("WARNING: torch not installed — skipping coae_decoder.pt")
    print("         For LibTorch-enabled builds, install with:")
    print("         pip install torch --index-url https://download.pytorch.org/whl/cpu")

print("Done.  Fixtures written to tests/fixtures/")
