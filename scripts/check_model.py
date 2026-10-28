import torch
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

hf_tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

prompt = "The quick brown fix jumps over the lazy dog."
tokens = model.to_tokens(prompt)

with torch.no_grad():
    tl_logits = model(tokens)
    hf_logits = hf_model(tokens).logits

tl_lp = tl_logits.log_softmax(-1)
hf_lp = hf_logits.log_softmax(-1)
max_diff = (tl_lp - hf_lp).abs().max().item()
print(f"Max log-prob difference: {max_diff:.2e}")
assert max_diff < 1e-3, "TransformerLens and HF outputs disagree"

_, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
acts = cache["blocks.6.hook_resid_pre"]
print("Activation shape:", acts.shape) 
print("Mean norm per position:", acts.norm(dim=-1)[0])