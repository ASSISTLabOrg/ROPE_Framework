#!/usr/bin/env python3
"""
One-time model export script for ROPE C++ runtime.

Converts all neural-network weights and normalization statistics to formats
readable by the C++ implementation:
  - Keras base models (15x) and meta model → ONNX
  - PyTorch COAE decoder → ONNX
  - stats_ts.pt  → exported/stats_ts.bin   (1-D z-score stats)
  - stats_cae.pt → exported/stats_cae.bin  (spatial z-score stats for density)

Binary stats format:
  uint32  ndim
  uint32  shape[ndim]          # e.g. [16] for ts, [1,72,36,45] for cae
  float32 mu[product(shape)]
  float32 sigma[product(shape)]

Run from the ROPE_Test directory:
    python export_models.py

Dependencies (pip install if absent):
    tf2onnx torch onnx onnxruntime tensorflow numpy pyyaml
"""

import os, sys, struct
import numpy as np
import torch
import yaml
import tensorflow as tf

ROPE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(ROPE_DIR, "exported")
os.makedirs(EXPORT_DIR, exist_ok=True)

SEQ_LEN   = 3
LATENT_DIM = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_stats_bin(stats: dict, out_path: str) -> None:
    """Write mu/sigma tensors to the binary format expected by normalizer.hpp."""
    mu_key    = next(k for k in ("mu", "mean", "means") if k in stats)
    sigma_key = next(k for k in ("sigma", "std", "stds") if k in stats)
    mu    = stats[mu_key].detach().cpu().numpy().astype(np.float32)
    sigma = stats[sigma_key].detach().cpu().numpy().astype(np.float32)
    shape = list(mu.shape)
    ndim  = len(shape)
    with open(out_path, "wb") as f:
        f.write(struct.pack("I", ndim))
        for s in shape:
            f.write(struct.pack("I", s))
        f.write(mu.ravel().tobytes())
        f.write(sigma.ravel().tobytes())
    print(f"  saved {out_path}  shape={shape}")


# ---------------------------------------------------------------------------
# 1.  Normalization statistics
# ---------------------------------------------------------------------------

def export_stats() -> None:
    print("\n=== Exporting normalization statistics ===")

    # Time-series stats  (shape: [total_dim])
    ts_path = os.path.join(ROPE_DIR, "data", "stats_ts.pt")
    try:
        stats_ts = torch.load(ts_path, map_location="cpu", weights_only=True)
    except TypeError:
        stats_ts = torch.load(ts_path, map_location="cpu")
    _save_stats_bin(stats_ts, os.path.join(EXPORT_DIR, "stats_ts.bin"))

    # CAE stats  (shape: [1, 72, 36, 45]  or scalar)
    cae_path = os.path.join(ROPE_DIR, "data", "stats_cae.pt")
    try:
        stats_cae = torch.load(cae_path, map_location="cpu", weights_only=True)
    except TypeError:
        stats_cae = torch.load(cae_path, map_location="cpu")
    _save_stats_bin(stats_cae, os.path.join(EXPORT_DIR, "stats_cae.bin"))


# ---------------------------------------------------------------------------
# 2.  Keras base models (15) + meta model
# ---------------------------------------------------------------------------

def _keras_to_onnx(model, input_dim: int, seq_len: int, opset: int = 17):
    """
    Convert a Keras model to ONNX via tf2onnx.convert.from_function.

    Why not simpler alternatives:
      - from_keras:        fails on Keras v3 (.keras); KeyError on output names.
      - from_saved_model:  not part of the tf2onnx Python API (CLI-only).
      - tf.saved_model.save(…, signatures=…): crashes with "_DictWrapper"
                           TypeError from Keras's internal attribute tracking.

    The remaining problem with from_function: tf.function captures plain
    tf.Tensor layer attributes (like PositionalEncoding.pos_enc) as *free
    variables*, which tf2onnx emits as extra ONNX input nodes.  ONNX Runtime
    then receives a (batch,3,16) tensor but expects a (3,16) for the pos_enc
    slot → rank-mismatch crash.

    Fix: before tracing, replace any tf.Tensor layer attributes that represent
    static constants with their numpy equivalents.  tf.function treats numpy
    arrays as Python literals and folds them into the graph as constants
    (ONNX initializers), eliminating the spurious input nodes.
    """
    import tf2onnx

    # Warm the model so all weights/shapes are fully materialised.
    _ = model(tf.zeros((1, seq_len, input_dim), dtype=tf.float32), training=False)

    # Freeze any plain-tensor layer attributes so tf.function embeds them as
    # graph constants.  This is a no-op for LSTM/GRU layers (no such attrs).
    for layer in model.layers:
        if hasattr(layer, "pos_enc"):
            try:
                layer.pos_enc = layer.pos_enc.numpy()
            except AttributeError:
                pass  # already numpy or non-tensor

    input_spec = tf.TensorSpec((None, seq_len, input_dim), tf.float32, name="x")

    @tf.function(input_signature=[input_spec])
    def serving_fn(x):
        return model(x, training=False)

    onnx_model, _ = tf2onnx.convert.from_function(
        serving_fn,
        input_signature=[input_spec],
        opset=opset,
    )
    return onnx_model


def export_keras_models() -> None:
    import tensorflow as tf

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    sys.path.insert(0, ROPE_DIR)
    from ts_utils.custom_layers import PositionalEncoding

    print("\n=== Exporting Keras base models (15 total) ===")

    arch_dirs = [
        ("LSTM MODELS",        None),
        ("GRU MODELS",         None),
        ("TRANSFORMER MODELS", {"PositionalEncoding": PositionalEncoding}),
    ]

    # Determine input_dim from the already-exported stats binary.
    ts_path = os.path.join(EXPORT_DIR, "stats_ts.bin")
    with open(ts_path, "rb") as f:
        ndim  = struct.unpack("I", f.read(4))[0]
        shape = [struct.unpack("I", f.read(4))[0] for _ in range(ndim)]
    input_dim = shape[0]   # total_dim = latent_dim + driver_dim

    model_idx = 0
    for arch_name, custom_objects in arch_dirs:
        arch_dir = os.path.join(ROPE_DIR, "Models", "Storms", arch_name)
        for i in range(1, 6):
            path = os.path.join(arch_dir, f"best_model_{i}.keras")
            print(f"  [{model_idx:02d}] {arch_name}/best_model_{i}.keras")
            model = tf.keras.models.load_model(
                path, compile=False, custom_objects=custom_objects
            )
            onnx_model = _keras_to_onnx(model, input_dim, SEQ_LEN)
            out_path = os.path.join(EXPORT_DIR, f"base_model_{model_idx:02d}.onnx")
            with open(out_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            model_idx += 1

    print("\n=== Exporting meta model ===")
    meta_path = os.path.join(ROPE_DIR, "Meta Models", "MetaStormTunedBLa0.keras")
    meta_model = tf.keras.models.load_model(meta_path, compile=False)
    onnx_meta = _keras_to_onnx(meta_model, input_dim, SEQ_LEN)
    out_path = os.path.join(EXPORT_DIR, "meta_model.onnx")
    with open(out_path, "wb") as f:
        f.write(onnx_meta.SerializeToString())
    print(f"  saved {out_path}")

    # Persist meta model output shape for C++ to read
    dummy = meta_model(tf.zeros((1, SEQ_LEN, input_dim), dtype=tf.float32))
    meta_out_shape = list(dummy.shape[1:])   # drop batch dim
    shape_path = os.path.join(EXPORT_DIR, "meta_model_out_shape.bin")
    with open(shape_path, "wb") as f:
        f.write(struct.pack("I", len(meta_out_shape)))
        for s in meta_out_shape:
            f.write(struct.pack("I", int(s)))
    print(f"  meta output shape (no batch): {meta_out_shape}")


# ---------------------------------------------------------------------------
# 3.  PyTorch COAE decoder
# ---------------------------------------------------------------------------

def export_coae_decoder() -> None:
    print("\n=== Exporting COAE decoder ===")
    sys.path.insert(0, ROPE_DIR)
    from ae_utils.attn_models import COAE

    cfg_path = os.path.join(ROPE_DIR, "weights", "finetuned_coae", "config.yaml")
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    model_cfg = full_cfg["model"]

    coae = COAE(config=model_cfg)

    weights_path = os.path.join(
        ROPE_DIR, "weights", "finetuned_coae", "best_weights_1gpu.pth"
    )
    try:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(weights_path, map_location="cpu")
    coae.load_state_dict(sd)
    coae.eval()

    decoder = coae.decoder
    dummy   = torch.zeros(1, LATENT_DIM)

    # --- TorchScript export (primary: used by C++ libtorch backend) ---
    pt_path = os.path.join(EXPORT_DIR, "coae_decoder.pt")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, dummy)
    torch.jit.save(traced, pt_path)
    print(f"  saved {pt_path}")

    # Quick shape sanity-check
    with torch.no_grad():
        test_out = traced(dummy)
    print(f"  decoder output shape: {tuple(test_out.shape)}  (expected (1,1,72,36,45))")

    # --- ONNX export (kept for reference / ORT fallback) ---
    out_path = os.path.join(EXPORT_DIR, "coae_decoder.onnx")
    torch.onnx.export(
        decoder,
        dummy,
        out_path,
        input_names  = ["latent"],
        output_names = ["density"],
        dynamic_axes = {"latent": {0: "batch"}, "density": {0: "batch"}},
        opset_version = 17,
    )

    # Patch the output shape annotation: torch.onnx.export traces with batch=1
    # and leaves a static dim_value=1, overriding dynamic_axes for shape metadata.
    # ORT warns when actual batch ≠ 1.  Fix: mark dim[0] as symbolic.
    try:
        import onnx as _onnx
        _m = _onnx.load(out_path)
        for _out in _m.graph.output:
            _dim = _out.type.tensor_type.shape.dim[0]
            _dim.ClearField("dim_value")
            _dim.dim_param = "batch"
        _onnx.save(_m, out_path)
        print(f"  patched output batch dim to symbolic")
    except ImportError:
        print("  onnx not installed — skipping batch-dim patch (ORT may warn)")

    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    export_stats()
    export_keras_models()
    export_coae_decoder()
    print(f"\nAll exports written to: {EXPORT_DIR}")
    print("You can now build the C++ ROPE library.")
