import time
import torch
from transformer_lens import HookedTransformer
from sae_gpt2.data import token_stream, ActivationBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

bos = model.tokenizer.bos_token_id

# Sanity check on the stream itself (held-out split)
ctx = next(token_stream(model.tokenizer, ctx_len=128, heldout=True))
assert ctx.shape == (128,), f"unexpected context shape: {ctx.shape}"
assert ctx[0].item() == bos, "context must start with BOS"
assert ctx[1].item() != bos, "double BOS at start of context"
print("First held-out context:", repr(model.tokenizer.decode(ctx[:40])))

# Training stream + activation buffer
stream = token_stream(model.tokenizer, ctx_len=128, heldout=False)

buf_tokens = 262_144 if device == "cuda" else 32_768
batch_size = 4096 if device == "cuda" else 1024
buf = ActivationBuffer(model, stream, layer=6, buffer_tokens=buf_tokens,
                       batch_size=batch_size, device=device)
print("Scale:", buf.estimate_scale())

# Throughput measurement
n_batches = 50 if device == "cuda" else 10
t0 = time.time()
total = 0
batch = None
for i, batch in zip(range(n_batches), buf):
    total += batch.shape[0]
if device == "cuda":
    torch.cuda.synchronize()  # make sure GPU work is finished before stopping the clock
elapsed = time.time() - t0

print(f"{total} tokens in {elapsed:.1f}s -> {total / elapsed:,.0f} tokens/sec "
      f"(includes model forward + shuffling, device={device})")
print("Batch mean norm:", batch.norm(dim=-1).mean().item())