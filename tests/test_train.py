import torch
from sae_gpt2.model import SAE
from sae_gpt2.train import l1_mult, lr_mult, save_ckpt


def test_lr_schedule_shape():
    total, warm, decay = 100, 10, 20
    mults = [lr_mult(s, total, warm, decay) for s in range(total)]
    assert mults[0] < mults[warm - 1] <= 1.0   # warming up
    assert mults[warm] == 1.0                  # plateau
    assert mults[total - decay - 1] == 1.0     # plateau ends just before decay
    assert mults[-1] < 0.1                     # decayed near zero
    assert all(m >= 0 for m in mults)


def test_l1_warmup():
    assert l1_mult(0, 10) < l1_mult(5, 10) < l1_mult(9, 10) == 1.0
    assert l1_mult(50, 10) == 1.0
    assert l1_mult(0, 0) == 1.0  # no warmup configured


def test_last_active_tracks_firing():
    sae = SAE(8, 32)
    x = torch.randn(16, 8)
    sae.loss(x, l1_coeff=1.0)
    assert sae.last_active.shape == (32,)
    assert sae.last_active.dtype == torch.bool
    _, f = sae(x)
    assert torch.equal(sae.last_active, (f > 0).any(0))

def test_save_ckpt_roundtrip_and_atomic(tmp_path):
    sae = SAE(8, 32)
    p = tmp_path / "latest.pt"
    save_ckpt(p, sae, {"a": 1}, 0.35, 10, 1000, extra={"docs_consumed": 5})
    assert p.exists() and not list(tmp_path.glob("*.tmp"))

    ck = torch.load(p, weights_only=False)
    assert ck["step"] == 10 and ck["docs_consumed"] == 5 and ck["scale"] == 0.35

    sae2 = SAE(8, 32, seed=1)
    sae2.load_state_dict(ck["state_dict"])
    assert torch.equal(sae2.W_dec, sae.W_dec)