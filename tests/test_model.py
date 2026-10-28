import torch
from sae_gpt2.model import SAE


def test_shapes_and_nonnegative_features():
    sae = SAE(d_model=16, d_sae=64)
    x = torch.randn(32, 16)
    x_hat, f = sae(x)
    assert x_hat.shape == (32, 16)
    assert f.shape == (32, 64)
    assert (f >= 0).all()


def test_decoder_rows_unit_norm_at_init_and_after_normalize():
    sae = SAE(16, 64)
    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(64), atol=1e-5)

    with torch.no_grad():
        sae.W_dec.mul_(3.0)
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_enc_initialized_as_dec_transpose():
    sae = SAE(16, 64)
    assert torch.equal(sae.W_enc.data, sae.W_dec.data.T)


def test_remove_parallel_grad():
    sae = SAE(16, 64)
    x = torch.randn(32, 16)
    total, _ = sae.loss(x, l1_coeff=1.0)
    total.backward()
    sae.remove_parallel_grad()
    dots = (sae.W_dec.grad * sae.W_dec).sum(-1)
    assert dots.abs().max() < 1e-5


def test_init_b_dec():
    sae = SAE(16, 64)
    mean = torch.randn(16)
    sae.init_b_dec(mean)
    assert torch.equal(sae.b_dec.data, mean)


def test_overfit_on_synthetic_sparse_data():
    """On data built from sparse combinations of known directions,
    reconstruction error should fall substantially and norms stay at 1."""
    torch.manual_seed(0)
    d_model, n_true, d_sae = 32, 64, 128

    true_dirs = torch.randn(n_true, d_model)
    true_dirs /= true_dirs.norm(dim=-1, keepdim=True)

    def sample(n):
        coeffs = torch.rand(n, n_true) * (torch.rand(n, n_true) < 0.05)
        return coeffs @ true_dirs * 3.0

    sae = SAE(d_model, d_sae)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

    x0 = sample(2048)
    _, first = sae.loss(x0, l1_coeff=0.1)

    for _ in range(1500):
        x = sample(512)
        total, metrics = sae.loss(x, l1_coeff=0.1)
        opt.zero_grad()
        total.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()

    _, last = sae.loss(sample(2048), l1_coeff=0.1)
    assert last["mse"] < 0.5 * first["mse"]
    assert torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(d_sae), atol=1e-4)