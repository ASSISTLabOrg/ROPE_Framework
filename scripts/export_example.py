"""
Example: define and export a drivers → predictor → decoder pipeline to ONNX.

Model weights are random — this is purely for validating the export path.
"""

import torch
import torch.nn as nn

import onnx_exporter

# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------
N_DRIVERS = 8       # e.g. F10.7, Kp, Dst, lat, lon, alt, doy_sin, doy_cos
SEQ_LEN = 10        # driver history window
LATENT_DIM = 32
STATE_DIM = 128     # full density profile output


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class Predictor(nn.Module):
    """drivers [batch, seq_len, n_drivers] → latent [batch, latent_dim]"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_DRIVERS, 64, num_layers=2, batch_first=True)
        self.head = nn.Linear(64, LATENT_DIM)

    def forward(self, drivers):
        out, _ = self.lstm(drivers)
        return self.head(out[:, -1, :])


class Decoder(nn.Module):
    """latent [batch, latent_dim] → full state [batch, state_dim]"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, STATE_DIM),
        )

    def forward(self, latent):
        return self.net(latent)


class Encoder(nn.Module):
    """full state [batch, state_dim] → latent [batch, latent_dim]"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
        )

    def forward(self, state):
        return self.net(state)

# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = Predictor()
    decoder = Decoder()
    encoder = Encoder()

    sample_drivers = torch.randn(1, SEQ_LEN, N_DRIVERS)
    sample_latent = torch.randn(1, LATENT_DIM)
    sample_state = torch.randn(1, STATE_DIM)

    batch_ax = {0: "batch"}

    print("--- predictor ---")
    onnx_exporter.save(
        predictor, sample_drivers, "predictor.onnx",
        opset_version=18,
        input_names=["drivers"],
        output_names=["latent"],
        dynamic_axes={"drivers": {0: "batch"}, "latent": batch_ax},
    )

    print("\n--- decoder ---")
    onnx_exporter.save(
        decoder, sample_latent, "decoder.onnx",
        opset_version=18,
        input_names=["latent"],
        output_names=["state"],
        dynamic_axes={"latent": batch_ax, "state": batch_ax},
    )

    print("\n--- encoder ---")
    onnx_exporter.save(
        encoder, sample_state, "encoder.onnx",
        opset_version=18,
        input_names=["state"],
        output_names=["latent"],
        dynamic_axes={"state": batch_ax, "latent": batch_ax},
    )
