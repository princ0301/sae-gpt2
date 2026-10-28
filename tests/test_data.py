import math

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_gpt2.data import ActivationBuffer

CTX = 64


@pytest.fixture(scope="module")
def model():
    m = HookedTransformer.from_pretrained("gpt2", device="cpu")
    m.eval()
    return m


def fake_stream(vocab, bos, n=10_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        body = torch.randint(0, vocab, (CTX - 1,), generator=g)
        yield torch.cat([torch.tensor([bos]), body])


def make_buffer(model, **kw):
    stream = fake_stream(model.cfg.d_vocab, model.tokenizer.bos_token_id)
    return ActivationBuffer(model, stream, buffer_tokens=8192,
                            batch_size=1024, gen_batch=16, **kw)


def test_batch_shape_and_normalization(model):
    buf = make_buffer(model)
    buf.estimate_scale(n_tokens=5000)
    batch = next(buf)
    assert batch.shape == (1024, 768)
    mean_norm = batch.norm(dim=-1).mean().item()
    assert abs(mean_norm - math.sqrt(768)) / math.sqrt(768) < 0.15


def test_no_bos_leakage(model):
    buf = make_buffer(model)
    buf.estimate_scale(n_tokens=5000)
    batch = next(buf)
    # A scaled BOS vector would have norm ~35 * sqrt(768); real tokens stay far below
    assert batch.norm(dim=-1).max().item() < 10 * math.sqrt(768)


def test_token_count_per_context(model):
    buf = make_buffer(model)
    acts = buf._generate(CTX - 1)  # rounds up to a full gen_batch of contexts
    assert acts.shape[0] % (CTX - 1) == 0


def test_refill_shuffles_without_changing_contents(model):
    # Two buffers reading identical streams (same seed)
    unshuffled = make_buffer(model)._generate(8192)

    buf = make_buffer(model)
    buf._refill()
    shuffled = buf.storage

    assert shuffled.shape == unshuffled.shape
    # Order changed...
    assert not torch.allclose(shuffled, unshuffled)
    # ...but the set of activations is identical (compare sorted norms)
    a = unshuffled.norm(dim=-1).sort().values
    b = shuffled.norm(dim=-1).sort().values
    assert torch.allclose(a, b, atol=1e-3)