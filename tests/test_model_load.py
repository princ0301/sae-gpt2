import torch
from transformer_lens import HookedTransformer

def test_activation_shape():
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = model.to_tokens("Hello world")
    _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
    acts = cache["blocks.6.hook_resid_pre"]
    assert acts.shape == (1, tokens.shape[1], 768)

def test_bos_norm_is_outlier():
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = model.to_tokens("The quick brown fox jumps over the lazy dog.")
    _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
    norms = cache["blocks.6.hook_resid_pre"][0].norm(dim=-1)
    assert norms[0] > 3 * norms[1:].mean()