
import torch
import torch.nn as nn

from scipy.signal import savgol_coeffs




# =============================================================================
# INFORMED PREPROCESSING MODULE (Component 1)
# Learnable SG-initialised 1D convolution + InstanceNorm1d
# =============================================================================

class InformedPreprocessing(nn.Module):
    """
    Savitzky-Golay-initialised learnable 1D convolution followed by
    per-spectrum instance normalisation.

    The convolutional kernel is initialised with analytic first-derivative
    SG coefficients, providing a physically grounded starting point that
    embeds noise suppression and baseline correction into the optimisation.
    The kernel is kept trainable so that BISN can adapt it jointly through
    the species-classification and entropy-regularised adversarial objectives.
    """

    def __init__(self, n_features: int, window_size: int = 61,
                 poly_order: int = 1, deriv_order: int = 1):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")

        self.n_features = n_features

        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=window_size,
            padding=window_size // 2,
            bias=False
        )

        # Initialise kernel with analytic SG coefficients
        coeffs = savgol_coeffs(
            window_length=window_size,
            polyorder=poly_order,
            deriv=deriv_order
        )
        sg_kernel = torch.from_numpy(coeffs[::-1].copy()).float().view(1, 1, -1)
        self.conv.weight = nn.Parameter(sg_kernel, requires_grad=True)

        # Per-spectrum instance normalisation removes amplitude offsets
        self.norm = nn.InstanceNorm1d(1, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (N, 1, d)
        x = self.conv(x)            # (N, 1, d + padding artefact)
        x = self.norm(x)            # per-spectrum normalisation
        x = x.squeeze(1)            # (N, d')

        # Trim padding artefact symmetrically if convolution extended the sequence
        if x.shape[1] > self.n_features:
            trim = (x.shape[1] - self.n_features) // 2
            x = x[:, trim: trim + self.n_features]

        return x                    # (N, d)

