import argparse
from itertools import islice

import torch
from transformer_lens import HookedTransformer

from sae_gpt2.data import token_stream
from sae_gpt2.model import SAE


def load_sae(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)  # our own file
    sae = SAE(ck["d_model"], ck["d_sae"]).to(device)
    sae.load_state_dict(ck["state_dict"])
    sae.eval()
    return sae, ck["scale"], ck["cfg"]


def make_batches(tokenizer, ctx_len, batch_size, device):
    """Fresh held-out stream each call, so passes never consume each other's data."""
    buf = []
    for ctx in token_stream(tokenizer, ctx_len=ctx_len, heldout=True):
        buf.append(ctx)
        if len(buf) == batch_size:
            yield torch.stack(buf).to(device)
            buf = []


def splice_hook(fn, scale):
    """Replace activations at positions 1: with fn(act * scale) / scale.
    Position 0 (BOS) is left alone, exactly as in training."""
    def hook(act, hook):
        out = act.clone()
        out[:, 1:, :] = fn(act[:, 1:, :] * scale) / scale
        return out
    return hook


@torch.no_grad()
def evaluate(model, sae, scale, layer=6, n_batches=100, batch_size=32,
             ctx_len=128, n_mean_batches=10):
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_pre"

    # Mean of scaled activations (positions 1:), for the mean-ablation baseline.
    # Uses the first n_mean_batches held-out batches, which overlap slightly
    # with the eval batches; the effect on the baseline should be small.
    acc = torch.zeros(sae.d_model, dtype=torch.float64, device=device)
    n_acc = 0
    for tokens in islice(make_batches(model.tokenizer, ctx_len, batch_size, device),
                         n_mean_batches):
        _, cache = model.run_with_cache(tokens, names_filter=hook_name,
                                        stop_at_layer=layer + 1)
        a = cache[hook_name][:, 1:, :].reshape(-1, sae.d_model) * scale
        acc += a.double().sum(0)
        n_acc += a.shape[0]
    mean_vec = (acc / n_acc).float()

    stats = dict(n=0, se=0.0,
                 sum_x=torch.zeros(sae.d_model, dtype=torch.float64, device=device),
                 sum_x2=0.0, l0=0.0, fires=torch.zeros(sae.d_sae, device=device))

    def sae_fn(x):
        x_hat, f = sae(x)
        flat_x, flat_f = x.reshape(-1, x.shape[-1]), f.reshape(-1, f.shape[-1])
        stats["n"] += flat_x.shape[0]
        stats["se"] += (x_hat - x).pow(2).sum().item()
        stats["sum_x"] += flat_x.double().sum(0)
        stats["sum_x2"] += flat_x.double().pow(2).sum().item()
        stats["l0"] += (flat_f > 0).float().sum().item()
        stats["fires"] += (flat_f > 0).float().sum(0)
        return x_hat

    fns = {
        "identity": lambda x: x,
        "zero": lambda x: torch.zeros_like(x),
        "mean": lambda x: mean_vec.expand_as(x),
        "sae": sae_fn,
    }
    ce = {k: 0.0 for k in ["clean", *fns]}
    n_done = 0
    for tokens in islice(make_batches(model.tokenizer, ctx_len, batch_size, device),
                         n_batches):
        ce["clean"] += model(tokens, return_type="loss").item()
        for name, fn in fns.items():
            ce[name] += model.run_with_hooks(
                tokens, return_type="loss",
                fwd_hooks=[(hook_name, splice_hook(fn, scale))],
            ).item()
        n_done += 1
    ce = {k: v / n_done for k, v in ce.items()}

    # Identity splice must reproduce the clean loss, or the hook is buggy
    assert abs(ce["identity"] - ce["clean"]) < 1e-3, \
        f"identity splice changed loss: {ce['identity']:.5f} vs {ce['clean']:.5f}"

    n = stats["n"]
    mean = stats["sum_x"] / n
    var = stats["sum_x2"] / n - mean.pow(2).sum().item()
    mse = stats["se"] / n
    density = stats["fires"] / n
    return {
        "ce_clean": ce["clean"], "ce_zero": ce["zero"], "ce_mean": ce["mean"],
        "ce_sae": ce["sae"],
        "ce_cost": ce["sae"] - ce["clean"],
        "loss_recovered_zero": (ce["zero"] - ce["sae"]) / (ce["zero"] - ce["clean"]),
        "loss_recovered_mean": (ce["mean"] - ce["sae"]) / (ce["mean"] - ce["clean"]),
        "l0": stats["l0"] / n,
        "explained_var": 1 - mse / var,
        "dead_frac": (stats["fires"] == 0).float().mean().item(),
        "dense_frac": (density > 0.1).float().mean().item(),
        "eval_tokens": n,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, scale, cfg = load_sae(args.ckpt, device)
    model = HookedTransformer.from_pretrained(cfg["model_name"], device=device)
    model.eval()

    res = evaluate(model, sae, scale, layer=cfg["layer"], n_batches=args.n_batches,
                   ctx_len=cfg["ctx_len"])
    for k, v in res.items():
        print(f"{k:>20}: {v:.4f}" if isinstance(v, float) else f"{k:>20}: {v}")