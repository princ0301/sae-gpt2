import math
import torch
from datasets import load_dataset

N_HELDOUT_DOCS = 20000

def token_stream(tokenizer, ctx_len=128, heldout=False,
                 dataset="Skylion007/openwebtext"):
    """Yield LongTensors of shape [ctx_len]: BOS + (ctx_len - 1) text tokens.

    Documents are concatenated with EOS separators and chunked.
    heldout=True streams only the first N_HELDOUT_DOCS docs; heldout=False
    skips them, so train and eval text never overlap.
    """
    ds = load_dataset(dataset, split="train", streaming=True)
    ds = ds.take(N_HELDOUT_DOCS) if heldout else ds.skip(N_HELDOUT_DOCS)

    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    body = ctx_len - 1
    pending = []
    for ex in ds:
        # TransformerLens's tokenizer can prepend BOS on its own, so disable
        # special tokens and strip a leading BOS if one still sneaks through.
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        if ids and ids[0] == bos:
            ids = ids[1:]
        pending.extend(ids)
        pending.append(eos)  # document separator (same id as BOS in GPT-2)
        while len(pending) >= body:
            chunk, pending = pending[:body], pending[body:]
            yield torch.tensor([bos] + chunk, dtype=torch.long)

class ActivationBuffer:
    """Generates layer activations on the fly, shuffles them, serves minibatches.

    - Position 0 (BOS) is dropped: its norm is ~35x larger than other positions.
    - Activations are scaled so the mean norm is ~sqrt(d_model).
    """

    def __init__(self, model, token_iter, layer=6, buffer_tokens=262144, batch_size=4096, gen_batch=32, device=None, seed=0):
        self.model = model
        self.token_iter = token_iter
        self.layer = layer
        self.hook_name = f"blocks.{layer}.hook_resid_pre"
        self.buffer_tokens = buffer_tokens
        self.batch_size = batch_size
        self.gen_batch = gen_batch
        self.device = device or next(model.parameters()).device
        self.d_model = model.cfg.d_model
        self.scale = 1.0
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.storage = torch.empty(0, self.d_model, device=self.device)
        self.ptr = 0
        self.exhausted = False

    @torch.no_grad()
    def _generate(self, n_tokens):
        """Run the model n fresh contexts until we have >= n_tokens activations."""
        out, count = [], 0

        while count < n_tokens:
            contexts = []

            for _ in range(self.gen_batch):
                try:
                    contexts.append(next(self.token_iter))
                except StopIteration:
                    self.exhausted = True
                    break

            if not contexts:
                break

            tokens = torch.stack(contexts).to(self.device)
            # stop_at_layer = layer + 1 so block `layer` runs and its hook fires
            _, cache = self.model.run_with_cache(
                tokens, names_filter=self.hook_name, stop_at_layer=self.layer + 1,
            )
            acts = cache[self.hook_name][:, 1:, :]  # drop BOS position
            acts = acts.reshape(-1, self.d_model) * self.scale
            out.append(acts)
            count += acts.shape[0]
        if not out:
            return torch.empty(0, self.d_model, device=self.device)
        return torch.cat(out)

    def estimate_scale(self, n_tokens=50000):
        """Set scale so mean activation norm = sqrt(d_model). Consumes a little data."""
        self.scale = 1.0
        acts = self._generate(n_tokens)
        self.scale = math.sqrt(self.d_model) / acts.norm(dim=-1).mean().item()
        return self.scale

    def _refill(self):
        leftover = self.storage[self.ptr:]
        new = self._generate(self.buffer_tokens - leftover.shape[0])
        storage = torch.cat([leftover, new])
        perm = torch.randperm(storage.shape[0], generator=self.gen).to(self.device)
        self.storage, self.ptr = storage[perm], 0

    def __iter__(self):
        return self

    def __next__(self):
        remaining = self.storage.shape[0] - self.ptr

        if remaining < self.buffer_tokens // 2 and not self.exhausted:
            self._refill()
            remaining = self.storage.shape[0]
        if remaining < self.batch_size:
            raise StopIteration
        batch = self.storage[self.ptr:self.ptr + self.batch_size]
        self.ptr += self.batch_size
        return batch
