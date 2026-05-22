
import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNetNoEmbeddings as Encoder
from informed_preproc import InformedPreprocessing
from grl import GRL


# =============================================================================
# BATCH-INVARIANT SPECTRAL NETWORK (BISN)
# =============================================================================

class BISN(nn.Module):
    """
    Batch-Invariant Spectral Network.

    Architecture (four jointly optimised components):
      1. InformedPreprocessing  — SG-initialised learnable 1D conv + InstanceNorm
      2. Sparse attentive encoder 
      3. Linear species classifier
      4. Entropy-regularised batch discriminator connected via GRL

    The GRL ensures that gradients from the batch entropy loss reverse sign
    before updating the preprocessing module, driving the learned representation
    toward batch invariance while preserving species-discriminative structure.
    """

    def __init__(self, input_dim: int, n_species_classes: int,
                 n_batch_classes: int,
                 n_d: int = 8, n_a: int = 8, n_steps: int = 3,
                 lambda_init: float = 0.0):
        super().__init__()

        # Component 1: informed preprocessing
        self.preprocessing = InformedPreprocessing(input_dim)

        # Component 2: sparse attentive encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=64,
            momentum=0.02,
            mask_type="sparsemax"
        )

        # Component 3: species classification head
        self.species_head = nn.Linear(n_d, n_species_classes)

        # Component 4: entropy-regularised batch discriminator
        self.grl = GRL(lambda_init)
        self.batch_discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_batch_classes)
        )

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        # Move group_attention_matrix if present 
        encoder = getattr(self.encoder, "encoder", None)
        if encoder is not None:
            gam = getattr(encoder, "group_attention_matrix", None)
            if gam is not None:
                encoder.group_attention_matrix = gam.to(device)
        return self

    def forward(self, x_raw: torch.Tensor):
        # Component 1: batch-invariant spectral representation
        x_hat = self.preprocessing(x_raw)

        # Component 4 forward: GRL reverses gradients into preprocessing
        batch_logits = self.batch_discriminator(self.grl(x_hat))

        # Components 2 & 3: species prediction
        z, sparsity_loss = self.encoder(x_hat)
        species_logits = self.species_head(z)

        return species_logits, batch_logits, sparsity_loss

    def set_grl_lambda(self, new_lambda: float):
        self.grl.set_lambda(new_lambda)

