import torch
import torch.nn as nn


class SAE(nn.Module):
    """ReLU sparse autoencoder with unit-norm decoder rows.

    encode:  f = ReLU((x - b_dec) @ W_enc + b_enc)
    decode:  x_hat = f @ W_dec + b_dec
    Shapes:  W_enc [d_model, d_sae], W_dec [d_sae, d_model]
    """

    def __init__(self, d_model: int, d_sae: int, seed: int = 0):
        super().__init__()
        self.d_model, self.d_sae = d_model, d_sae

        g = torch.Generator().manual_seed(seed)
        W_dec = torch.randn(d_sae, d_model, generator=g)
        W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)

        # Scale the encoder so the initial reconstruction has gain ~1.
        # With unit-norm random decoder rows, gain ~ s * d_sae / (2 * d_model).
        self.enc_init_scale = min(1.0, 2 * d_model / d_sae)

        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone() * self.enc_init_scale)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Set by loss(): which features fired in the most recent batch
        self.last_active = None

    # ---- forward pieces -------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    # ---- loss -----------------------------------------------------------
    def loss(self, x: torch.Tensor, l1_coeff: float):
        """Return (total_loss, metrics).

        MSE is summed over d_model and averaged over the batch. With
        activations scaled to mean norm sqrt(d_model), E||x||^2 ~ d_model.
        Decoder rows have unit norm, so the L1 penalty on f * ||W_dec||
        is just sum(f).
        """
        x_hat, f = self(x)
        mse = (x_hat - x).pow(2).sum(-1).mean()
        l1 = f.sum(-1).mean()  # f >= 0 after ReLU
        total = mse + l1_coeff * l1

        with torch.no_grad():
            self.last_active = (f > 0).any(0)  # [d_sae] bool: fired in this batch
            l0 = (f > 0).float().sum(-1).mean()
            var = (x - x.mean(0)).pow(2).sum(-1).mean()
            explained_var = 1 - mse / var
        metrics = {
            "loss": total.item(), "mse": mse.item(), "l1": l1.item(),
            "l0": l0.item(), "explained_var": explained_var.item(),
        }
        return total, metrics

    # ---- constraints ----------------------------------------------------
    @torch.no_grad()
    def normalize_decoder(self):
        """Rescale each decoder row to unit norm."""
        self.W_dec /= self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Project out the gradient component along each decoder row, so the
        optimizer step doesn't fight the unit-norm constraint."""
        if self.W_dec.grad is None:
            return
        g = self.W_dec.grad
        g -= (g * self.W_dec).sum(-1, keepdim=True) * self.W_dec

    @torch.no_grad()
    def init_b_dec(self, data_mean: torch.Tensor):
        """Set b_dec to the data mean (use a batch of normalized activations)."""
        self.b_dec.copy_(data_mean)